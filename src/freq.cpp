#include "keel/control.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"
#include "keel/poly.h"

namespace keel::control {

namespace {

c64 horner_c64(ArrayRef<f64> p, c64 x) noexcept {
    c64 y(0.0, 0.0);
    for (usize i = 0; i < p.size(); ++i) {
        y = y * x + p[i];
    }
    return y;
}

c64 freq_point(f64 w, f64 ts) noexcept {
    if (ts > 0.0) {
        return keel::polar(1.0, w * ts);
    }
    return c64(0.0, w);
}

c64 eval_g_jw(ArrayRef<f64> num, ArrayRef<f64> den, f64 w) noexcept {
    const c64 jw(0.0, w);
    const c64 d = horner_c64(den, jw);
    if (d == c64(0.0, 0.0)) {
        return c64(Limits<f64>::infinity(), 0.0);
    }
    return horner_c64(num, jw) / d;
}

struct CrossHit {
    bool found = false;
    f64 w = 0.0;
};

CrossHit find_first_crossing(ArrayRef<f64> x, ArrayRef<f64> y, f64 lvl) noexcept {
    for (usize i = 1; i < y.size(); ++i) {
        const f64 a = y[i - 1] - lvl;
        const f64 b = y[i] - lvl;
        if (a == 0.0) {
            return {true, x[i - 1]};
        }
        if ((a < 0.0) != (b < 0.0)) {
            const f64 frac = a / (a - b);
            return {true, x[i - 1] + frac * (x[i] - x[i - 1])};
        }
    }
    return {false, 0.0};
}

f64 interp_sorted(ArrayRef<f64> x, ArrayRef<f64> y, f64 x0) noexcept {
    const usize n = x.size();
    if (n == 0) {
        return 0.0;
    }
    if (x0 <= x[0]) {
        return y[0];
    }
    if (x0 >= x[n - 1]) {
        return y[n - 1];
    }
    usize lo = 0;
    usize hi = n;
    while (lo < hi) {
        const usize mid = lo + (hi - lo) / 2;
        if (x[mid] <= x0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const usize j = lo;
    const f64 frac = (x0 - x[j - 1]) / (x[j] - x[j - 1]);
    return y[j - 1] + frac * (y[j] - y[j - 1]);
}

template <typename Fn>
f64 bisect_root(Fn&& fn, f64 a, f64 b) noexcept {
    f64 fa = fn(a);
    for (i32 it = 0; it < 80; ++it) {
        const f64 m = 0.5 * (a + b);
        const f64 fm = fn(m);
        if (fm == 0.0 || (b - a) <= 1e-13 * (1.0 + b)) {
            return m;
        }
        if ((fa < 0.0) != (fm < 0.0)) {
            b = m;
        } else {
            a = m;
            fa = fm;
        }
    }
    return 0.5 * (a + b);
}

ArrayRef<f64> strip_leading(ArrayRef<f64> v) noexcept {
    usize i = 0;
    while (i + 1 < v.size() && v[i] == 0.0) {
        ++i;
    }
    return v.slice(i);
}

} // namespace

ControlStatus damp(ArrayRef<c64> poles, f64 ts, MutArrayRef<f64> wn_out,
                   MutArrayRef<f64> zeta_out) noexcept {
    const usize n = poles.size();
    if (!keel::is_finite(ts) || wn_out.size() < n || zeta_out.size() < n) {
        return ControlStatus::kInvalidArgument;
    }
    const bool discrete = (ts > 0.0);
    for (usize i = 0; i < n; ++i) {
        c64 s = poles[i];
        if (discrete) {
            if (s.real() == 0.0 && s.imag() == 0.0) {
                wn_out[i] = Limits<f64>::infinity();
                zeta_out[i] = 1.0;
                continue;
            }
            s = keel::log(s) / ts;
        }
        const f64 wn_i = keel::abs(s);
        const f64 zeta_i = (wn_i == 0.0) ? 0.0 : (-s.real() / wn_i);
        wn_out[i] = wn_i;
        zeta_out[i] = zeta_i;
    }
    return ControlStatus::kOk;
}

c64 evalfr_tf(ArrayRef<f64> num, ArrayRef<f64> den, f64 w, f64 ts) noexcept {
    const c64 s = freq_point(w, ts);
    const c64 nv = horner_c64(num, s);
    const c64 dv = horner_c64(den, s);
    if (dv == c64(0.0, 0.0)) {
        if (nv == c64(0.0, 0.0)) {
            const f64 nan = Limits<f64>::quiet_nan();
            return c64(nan, nan);
        }
        return c64(Limits<f64>::infinity(), 0.0);
    }
    return nv / dv;
}

ControlStatus freqresp_tf(ArrayRef<f64> num, ArrayRef<f64> den, ArrayRef<f64> w, f64 ts,
                          MutArrayRef<c64> h_out) noexcept {
    if (!keel::is_finite(ts) || h_out.size() < w.size()) {
        return ControlStatus::kInvalidArgument;
    }
    for (usize i = 0; i < w.size(); ++i) {
        h_out[i] = evalfr_tf(num, den, w[i], ts);
    }
    return ControlStatus::kOk;
}

ControlStatus pick_freq_grid(ArrayRef<c64> poles, ArrayRef<c64> zeros, usize n_pts, f64 ts,
                             MutArrayRef<f64> w_out) noexcept {
    if (!keel::is_finite(ts) || w_out.size() < n_pts) {
        return ControlStatus::kInvalidArgument;
    }
    if (n_pts == 0) {
        return ControlStatus::kOk;
    }

    bool has_mag = false;
    f64 min_m = 0.0;
    f64 max_m = 0.0;
    auto scan = [&](ArrayRef<c64> arr) noexcept {
        for (usize i = 0; i < arr.size(); ++i) {
            if (!keel::is_finite(arr[i].real()) || !keel::is_finite(arr[i].imag())) {
                continue;
            }
            const f64 m = keel::abs(arr[i]);
            if (m > 0.0 && keel::is_finite(m)) {
                if (!has_mag) {
                    min_m = m;
                    max_m = m;
                    has_mag = true;
                } else {
                    if (m < min_m) min_m = m;
                    if (m > max_m) max_m = m;
                }
            }
        }
    };
    scan(poles);
    scan(zeros);

    f64 lo = 1e-2;
    f64 hi = 1e2;
    if (has_mag) {
        lo = min_m / 100.0;
        hi = max_m * 100.0;
    }
    if (ts > 0.0) {
        const f64 nyq = kPi / ts;
        if (hi > nyq) hi = nyq;
        if (lo >= hi) lo = hi / 1000.0;
    }
    if (lo <= 0.0) lo = 1e-3;
    if (hi <= lo) hi = lo * 100.0;

    const f64 a = keel::log10(lo);
    const f64 b = keel::log10(hi);
    for (usize i = 0; i < n_pts; ++i) {
        const f64 frac = (n_pts == 1) ? 0.0 : (as<f64>(i) / as<f64>(n_pts - 1));
        w_out[i] = keel::pow(10.0, a + (b - a) * frac);
    }
    return ControlStatus::kOk;
}

ControlStatus bode_tf(ArrayRef<f64> num, ArrayRef<f64> den, ArrayRef<f64> w, f64 ts,
                      MutArrayRef<f64> mag_out, MutArrayRef<f64> phase_deg_out, Allocator* alloc) {
    if (!keel::is_finite(ts) || mag_out.size() < w.size() || phase_deg_out.size() < w.size()) {
        return ControlStatus::kInvalidArgument;
    }
    constexpr f64 kRadToDeg = 180.0 / kPi;
    ArenaTemp scratch = keel::get_scratch(alloc);

    f64 phi0 = 0.0;
    bool has_phi0 = false;
    if (ts <= 0.0 && !w.empty() && keel::is_finite(w[0]) && w[0] >= 0.0) {
        auto first_nz = [](ArrayRef<f64> v, usize& idx) noexcept -> f64 {
            for (usize i = 0; i < v.size(); ++i) {
                if (v[i] != 0.0) {
                    idx = i;
                    return v[i];
                }
            }
            idx = v.size();
            return 0.0;
        };
        usize in_idx = 0;
        usize id_idx = 0;
        const f64 b0 = first_nz(num, in_idx);
        const f64 a0 = first_nz(den, id_idx);
        if (b0 != 0.0 && a0 != 0.0) {
            auto poly_phase_at = [&](ArrayRef<f64> c, usize start, f64 w0) -> f64 {
                usize end = c.size();
                usize q = 0;
                while (end > start + 1 && c[end - 1] == 0.0) {
                    --end;
                    ++q;
                }
                f64 ang = as<f64>(q) * 90.0;
                if (end - start <= 1) {
                    return ang;
                }
                const Array<c64> rts =
                    keel::poly::roots(c.slice(start, end - start), scratch.allocator());
                for (usize k = 0; k < rts.size(); ++k) {
                    const c64 r = rts[k];
                    if (keel::abs(r.imag()) <= 1e-9 * keel::max(1.0, keel::abs(r.real()))) {
                        ang += keel::atan2(w0, -r.real()) * kRadToDeg;
                    } else if (r.imag() > 0.0) {
                        const f64 mag2 = keel::norm(r);
                        ang += keel::atan2(-2.0 * r.real() * w0, mag2 - w0 * w0) * kRadToDeg;
                    }
                }
                return ang;
            };
            const f64 k_gain = b0 / a0;
            phi0 = (k_gain < 0.0 ? 180.0 : 0.0) + poly_phase_at(num, in_idx, w[0]) -
                   poly_phase_at(den, id_idx, w[0]);
            has_phi0 = keel::is_finite(phi0);
        }
    }

    f64 prev = 0.0;
    bool has_prev = false;
    for (usize i = 0; i < w.size(); ++i) {
        const c64 h = evalfr_tf(num, den, w[i], ts);
        mag_out[i] = keel::abs(h);
        f64 ph = (keel::is_finite(h.real()) && keel::is_finite(h.imag()))
                     ? (keel::arg(h) * kRadToDeg)
                     : Limits<f64>::quiet_nan();
        if (keel::is_finite(ph)) {
            if (!has_prev && has_phi0) {
                ph -= 360.0 * keel::round((ph - phi0) / 360.0);
            } else if (has_prev) {
                ph -= 360.0 * keel::round((ph - prev) / 360.0);
            }
            prev = ph;
            has_prev = true;
        }
        phase_deg_out[i] = ph;
    }
    return ControlStatus::kOk;
}

ControlStatus nyquist_tf(ArrayRef<f64> num, ArrayRef<f64> den, ArrayRef<f64> w, f64 ts,
                         MutArrayRef<f64> re_out, MutArrayRef<f64> im_out) noexcept {
    if (!keel::is_finite(ts) || re_out.size() < w.size() || im_out.size() < w.size()) {
        return ControlStatus::kInvalidArgument;
    }
    for (usize i = 0; i < w.size(); ++i) {
        const c64 h = evalfr_tf(num, den, w[i], ts);
        re_out[i] = h.real();
        im_out[i] = h.imag();
    }
    return ControlStatus::kOk;
}

ControlStatus rlocus_tf(ArrayRef<f64> num_in, ArrayRef<f64> den_in, ArrayRef<f64> gains,
                        usize n_poles, MutArrayRef<c64> roots_out, Allocator* alloc) {
    const ArrayRef<f64> num = strip_leading(num_in);
    const ArrayRef<f64> den = strip_leading(den_in);
    if (den.empty() || den[0] == 0.0) {
        return ControlStatus::kInvalidArgument;
    }
    const usize n = den.size() - 1;
    const usize k_cnt = gains.size();
    if (n_poles != n || (n > 0 && roots_out.size() / n < k_cnt)) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0 || k_cnt == 0) {
        return ControlStatus::kOk;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Allocator* sa = scratch.allocator();
    Array<f64> num_aligned(den.size(), 0.0, sa);
    if (num.size() <= den.size()) {
        const usize pad = den.size() - num.size();
        for (usize i = 0; i < num.size(); ++i) {
            num_aligned[pad + i] = num[i];
        }
    } else {
        return ControlStatus::kInvalidArgument;
    }

    Array<f64> cpoly(den.size(), 0.0, sa);
    for (usize row = 0; row < k_cnt; ++row) {
        const f64 k = gains[row];
        for (usize i = 0; i < den.size(); ++i) {
            cpoly[i] = den[i] + k * num_aligned[i];
        }
        const Array<c64> rts = keel::poly::roots(cpoly.as_ref(), sa);
        for (usize j = 0; j < n; ++j) {
            roots_out[j * k_cnt + row] = (j < rts.size()) ? rts[j] : c64(0.0, 0.0);
        }
    }
    return ControlStatus::kOk;
}

ControlStatus margin_from_bode(ArrayRef<f64> w, ArrayRef<f64> mag, ArrayRef<f64> phase_deg,
                               MarginResult& out, Allocator* alloc) {
    const usize n = w.size();
    if (mag.size() < n || phase_deg.size() < n) {
        return ControlStatus::kInvalidArgument;
    }
    const CrossHit pc = find_first_crossing(w, phase_deg, -180.0);

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> mag_db(n, 0.0, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        mag_db[i] = (mag[i] > 0.0) ? (20.0 * keel::log10(mag[i])) : -300.0;
    }
    const CrossHit gc = find_first_crossing(w, mag_db.as_ref(), 0.0);

    const f64 inf = Limits<f64>::infinity();
    out.gm = inf;
    out.pm = inf;
    out.wcg = inf;
    out.wcp = inf;

    if (pc.found) {
        out.wcg = pc.w;
        const f64 mag_at_pc = interp_sorted(w, mag, pc.w);
        out.gm = (mag_at_pc > 0.0) ? (1.0 / mag_at_pc) : inf;
    }
    if (gc.found) {
        out.wcp = gc.w;
        const f64 ph_at_gc = interp_sorted(w, phase_deg, gc.w);
        out.pm = ph_at_gc + 180.0;
    }
    return ControlStatus::kOk;
}

ControlStatus allmargin_tf(ArrayRef<f64> num, ArrayRef<f64> den, AllMarginResult& out,
                           Allocator* alloc) {
    const ArrayRef<f64> den_s = strip_leading(den);
    if (den_s.empty() || den_s[0] == 0.0) {
        return ControlStatus::kInvalidArgument;
    }
    const f64 inf = Limits<f64>::infinity();
    ArenaTemp scratch = keel::get_scratch(alloc);
    Allocator* sa = scratch.allocator();

    f64 wlo = 1e-3;
    f64 whi = 1e3;
    {
        bool has_mag = false;
        f64 min_m = 0.0;
        f64 max_m = 0.0;
        auto add_roots = [&](ArrayRef<f64> c) {
            if (c.size() < 2) {
                return;
            }
            const Array<c64> rts = keel::poly::roots(c, sa);
            for (usize i = 0; i < rts.size(); ++i) {
                const f64 m = keel::abs(rts[i]);
                if (m > 1e-9 && keel::is_finite(m)) {
                    if (!has_mag) {
                        min_m = m;
                        max_m = m;
                        has_mag = true;
                    } else {
                        if (m < min_m) min_m = m;
                        if (m > max_m) max_m = m;
                    }
                }
            }
        };
        add_roots(num);
        add_roots(den_s);
        if (has_mag) {
            wlo = min_m * 1e-3;
            whi = max_m * 1e3;
        }
    }

    out.gain_margin = Array<f64>(alloc);
    out.gm_frequency = Array<f64>(alloc);
    out.phase_margin = Array<f64>(alloc);
    out.pm_frequency = Array<f64>(alloc);
    out.delay_margin = Array<f64>(alloc);
    out.dm_frequency = Array<f64>(alloc);

    constexpr i32 kSteps = 4000;
    const f64 llo = keel::log10(wlo);
    const f64 lhi = keel::log10(whi);
    f64 w_prev = wlo;
    c64 g_prev = eval_g_jw(num, den_s, w_prev);

    auto record_pm = [&](f64 wc) {
        const c64 gc = eval_g_jw(num, den_s, wc);
        f64 pm = keel::arg(gc) * (180.0 / kPi) + 180.0;
        pm -= 360.0 * keel::round(pm / 360.0);
        if (pm <= -180.0) pm += 360.0;
        out.phase_margin.push_back(pm);
        out.pm_frequency.push_back(wc);
        f64 pm_rad = pm * (kPi / 180.0);
        if (pm_rad < 0.0) pm_rad += 2.0 * kPi;
        out.delay_margin.push_back(wc > 0.0 ? (pm_rad / wc) : inf);
        out.dm_frequency.push_back(wc);
    };

    auto record_gm = [&](f64 wc) {
        const c64 gc = eval_g_jw(num, den_s, wc);
        if (gc.real() < 0.0) {
            const f64 m = keel::abs(gc);
            out.gain_margin.push_back(m > 0.0 ? (1.0 / m) : inf);
            out.gm_frequency.push_back(wc);
        }
    };

    for (i32 i = 1; i <= kSteps; ++i) {
        const f64 w = keel::pow(10.0, llo + (lhi - llo) * as<f64>(i) / as<f64>(kSteps));
        const c64 g = eval_g_jw(num, den_s, w);

        // Gain crossover: |G| - 1 changes sign or hits 0 exactly at w.
        const f64 a = keel::abs(g_prev) - 1.0;
        const f64 b = keel::abs(g) - 1.0;
        if (keel::is_finite(a) && keel::is_finite(b)) {
            if (b == 0.0) {
                record_pm(w);
            } else if (a != 0.0 && (a < 0.0) != (b < 0.0)) {
                const f64 wc = bisect_root(
                    [&](f64 x) noexcept { return keel::abs(eval_g_jw(num, den_s, x)) - 1.0; },
                    w_prev, w);
                record_pm(wc);
            }
        }

        // Phase crossover: Im(G) changes sign or hits 0 exactly at w with Re(G) < 0.
        const f64 ia = g_prev.imag();
        const f64 ib = g.imag();
        if (keel::is_finite(ia) && keel::is_finite(ib)) {
            if (ib == 0.0) {
                record_gm(w);
            } else if (ia != 0.0 && (ia < 0.0) != (ib < 0.0)) {
                const f64 wc = bisect_root(
                    [&](f64 x) noexcept { return eval_g_jw(num, den_s, x).imag(); }, w_prev, w);
                record_gm(wc);
            }
        }
        w_prev = w;
        g_prev = g;
    }

    // Closed-loop stability: roots(den + num) all strictly in LHP.
    bool stable = true;
    const usize l = keel::max(num.size(), den_s.size());
    Array<f64> cl(l, 0.0, sa);
    for (usize i = 0; i < den_s.size(); ++i) {
        cl[l - den_s.size() + i] += den_s[i];
    }
    for (usize i = 0; i < num.size(); ++i) {
        cl[l - num.size() + i] += num[i];
    }
    const ArrayRef<f64> cl_trim = strip_leading(cl.as_ref());
    if (cl_trim.size() >= 2) {
        const Array<c64> rts = keel::poly::roots(cl_trim, sa);
        for (usize i = 0; i < rts.size(); ++i) {
            if (rts[i].real() >= -1e-9) {
                stable = false;
                break;
            }
        }
    }
    out.stable = stable;
    return ControlStatus::kOk;
}

} // namespace keel::control
