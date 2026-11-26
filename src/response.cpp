#include "keel/control.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::control {

namespace {

bool is_uniform_grid(ArrayRef<f64> t, f64& dt_out) noexcept {
    const usize n = t.size();
    dt_out = (n >= 2) ? (t[1] - t[0]) : 0.0;
    if (!keel::is_finite(dt_out)) return false;
    for (usize k = 2; k < n; ++k) {
        const f64 dk = t[k] - t[k - 1];
        if (!keel::is_finite(dk) ||
            keel::abs(dk - dt_out) > 1e-12 * keel::max(1.0, keel::abs(dt_out))) {
            return false;
        }
    }
    return true;
}

bool is_strictly_increasing_finite(ArrayRef<f64> t) noexcept {
    for (usize k = 0; k < t.size(); ++k) {
        if (!keel::is_finite(t[k])) return false;
        if (k > 0 && !(t[k] > t[k - 1])) return false;
    }
    return true;
}

ControlStatus foh_mimo_step(ArrayRef<f64> a, ArrayRef<f64> b, usize n, usize nu, f64 dt,
                            MutArrayRef<f64> ad, MutArrayRef<f64> b0, MutArrayRef<f64> b1,
                            Allocator* alloc) {
    if (!keel::is_finite(dt) || !(dt > 0.0)) {
        return ControlStatus::kInvalidArgument;
    }
    if (nu > Limits<usize>::kMax / 2u || n > Limits<usize>::kMax - 2u * nu) {
        return ControlStatus::kInvalidArgument;
    }
    const usize aug = n + 2 * nu;
    if (aug > Limits<usize>::kMax / aug) {
        return ControlStatus::kInvalidArgument;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> m(aug * aug, 0.0, scratch.allocator());
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            m[j * aug + i] = a[j * n + i] * dt;
        }
    }
    for (usize j = 0; j < nu; ++j) {
        for (usize i = 0; i < n; ++i) {
            m[(n + j) * aug + i] = b[j * n + i] * dt;
        }
        m[(n + nu + j) * aug + (n + j)] = 1.0;
    }
    Array<f64> e(aug * aug, 0.0, scratch.allocator());
    const ControlStatus st = expm(m.as_ref(), aug, e.as_mut_ref(), scratch.allocator());
    if (st != ControlStatus::kOk) {
        return st;
    }
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            ad[j * n + i] = e[j * aug + i];
        }
    }
    for (usize j = 0; j < nu; ++j) {
        for (usize i = 0; i < n; ++i) {
            const f64 phi1 = e[(n + j) * aug + i];
            const f64 phi2 = e[(n + nu + j) * aug + i];
            b0[j * n + i] = phi1 - phi2;
            b1[j * n + i] = phi2;
        }
    }
    return ControlStatus::kOk;
}

} // namespace

ControlStatus pick_time_grid(ArrayRef<c64> poles, f64 t_final_override, Array<f64>& t_out,
                             Allocator* alloc) {
    if (!keel::is_finite(t_final_override) || t_final_override < 0.0) {
        return ControlStatus::kInvalidArgument;
    }
    f64 min_abs_real = 1.0;
    f64 max_abs_real = 1.0;
    if (!poles.empty()) {
        bool any_stable = false;
        for (usize i = 0; i < poles.size(); ++i) {
            const c64 p = poles[i];
            if (!keel::is_finite(p.real()) || !keel::is_finite(p.imag())) {
                continue;
            }
            const f64 ar = keel::abs(p.real());
            const f64 mag = keel::abs(p);
            if (p.real() < 0.0 && mag > 0.0) {
                if (!any_stable || ar < min_abs_real) {
                    min_abs_real = ar;
                }
                any_stable = true;
            }
            if (mag > max_abs_real) {
                max_abs_real = mag;
            }
        }
        if (!any_stable) {
            min_abs_real = 1.0;
        }
    }

    const f64 t_final_default = 5.80251 / min_abs_real;
    const f64 dt_raw = keel::min(t_final_default / 126.0, 1.0 / (4.0 * max_abs_real));
    const f64 ratio_def = t_final_default / dt_raw;
    usize n_def = (ratio_def > 1000.0) ? 1000u : (as<usize>(keel::ceil(ratio_def)) + 1u);
    if (n_def < 60) {
        n_def = 60;
    }
    if (n_def > 1000) {
        n_def = 1000;
    }
    const f64 dt_final = t_final_default / as<f64>(n_def - 1);

    if (t_final_override > 0.0) {
        const f64 ratio = t_final_override / dt_final;
        if (!keel::is_finite(ratio) || ratio > 1e7) {
            return ControlStatus::kInvalidArgument;
        }
        const usize n = as<usize>(keel::ceil(ratio)) + 1;
        t_out = Array<f64>(n, 0.0, alloc);
        for (usize i = 0; i < n; ++i) {
            t_out[i] = as<f64>(i) * dt_final;
        }
        return ControlStatus::kOk;
    }

    t_out = Array<f64>(n_def, 0.0, alloc);
    for (usize i = 0; i < n_def; ++i) {
        t_out[i] = as<f64>(i) * dt_final;
    }
    return ControlStatus::kOk;
}

ControlStatus simulate_siso(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<f64> c, f64 d, usize n,
                            f64 ts, ArrayRef<f64> t, ArrayRef<f64> u, ArrayRef<f64> x0,
                            MutArrayRef<f64> y_out, MutArrayRef<f64> x_traj_out, Allocator* alloc) {
    const usize n_steps = t.size();
    if (!keel::is_finite(ts) || !keel::is_finite(d) || (n > 0 && a.size() / n < n) ||
        b.size() < n || c.size() < n || u.size() < n_steps || y_out.size() < n_steps) {
        return ControlStatus::kInvalidArgument;
    }
    if (!x0.empty() && x0.size() < n) {
        return ControlStatus::kInvalidArgument;
    }
    const bool want_traj = !x_traj_out.empty();
    if (want_traj && n > 0 && x_traj_out.size() / n < n_steps) {
        return ControlStatus::kInvalidArgument;
    }
    if (ts <= 0.0 && n_steps >= 2 && !is_strictly_increasing_finite(t)) {
        return ControlStatus::kInvalidArgument;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Allocator* sa = scratch.allocator();
    Array<f64> x(n, 0.0, sa);
    Array<f64> xn(n, 0.0, sa);
    if (!x0.empty()) {
        for (usize i = 0; i < n; ++i) {
            x[i] = x0[i];
        }
    }

    auto record = [&](usize k) noexcept {
        if (want_traj) {
            for (usize i = 0; i < n; ++i) {
                x_traj_out[i * n_steps + k] = x[i];
            }
        }
    };

    if (ts > 0.0) {
        for (usize k = 0; k < n_steps; ++k) {
            record(k);
            f64 yk = d * u[k];
            for (usize i = 0; i < n; ++i) {
                yk += c[i] * x[i];
            }
            y_out[k] = yk;
            for (usize i = 0; i < n; ++i) {
                f64 s = b[i] * u[k];
                for (usize j = 0; j < n; ++j) {
                    s += a[j * n + i] * x[j];
                }
                xn[i] = s;
            }
            keel::swap(x, xn);
        }
        return ControlStatus::kOk;
    }

    f64 dt = 0.0;
    const bool uniform = is_uniform_grid(t, dt);
    Array<f64> ad(n * n, 0.0, sa);
    Array<f64> bd(n, 0.0, sa);

    if (uniform && dt > 0.0) {
        const ControlStatus st = c2d_zoh(a, b, n, 1, dt, ad.as_mut_ref(), bd.as_mut_ref(), sa);
        if (st != ControlStatus::kOk) {
            return st;
        }
        for (usize k = 0; k < n_steps; ++k) {
            record(k);
            f64 yk = d * u[k];
            for (usize i = 0; i < n; ++i) {
                yk += c[i] * x[i];
            }
            y_out[k] = yk;
            if (k + 1 < n_steps) {
                for (usize i = 0; i < n; ++i) {
                    f64 s = bd[i] * u[k];
                    for (usize j = 0; j < n; ++j) {
                        s += ad[j * n + i] * x[j];
                    }
                    xn[i] = s;
                }
                keel::swap(x, xn);
            }
        }
        return ControlStatus::kOk;
    }

    // Non-uniform time grid: per-step ZOH.
    for (usize k = 0; k < n_steps; ++k) {
        record(k);
        f64 yk = d * u[k];
        for (usize i = 0; i < n; ++i) {
            yk += c[i] * x[i];
        }
        y_out[k] = yk;
        if (k + 1 < n_steps) {
            const f64 dtk = t[k + 1] - t[k];
            const ControlStatus st =
                c2d_zoh(a, b, n, 1, dtk, ad.as_mut_ref(), bd.as_mut_ref(), sa);
            if (st != ControlStatus::kOk) {
                return st;
            }
            for (usize i = 0; i < n; ++i) {
                f64 s = bd[i] * u[k];
                for (usize j = 0; j < n; ++j) {
                    s += ad[j * n + i] * x[j];
                }
                xn[i] = s;
            }
            keel::swap(x, xn);
        }
    }
    return ControlStatus::kOk;
}

ControlStatus simulate_mimo(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<f64> c, ArrayRef<f64> d,
                            usize n, usize nu, usize ny, f64 ts, ArrayRef<f64> t,
                            ArrayRef<f64> u_colmajor, ArrayRef<f64> x0, MutArrayRef<f64> y_out,
                            MutArrayRef<f64> x_traj_out, Allocator* alloc) {
    const usize n_steps = t.size();
    if (!keel::is_finite(ts) || (n > 0 && a.size() / n < n) || (n > 0 && nu > 0 && b.size() / nu < n) ||
        (n > 0 && ny > 0 && c.size() / n < ny) || (nu > 0 && u_colmajor.size() / nu < n_steps) ||
        (ny > 0 && y_out.size() / ny < n_steps)) {
        return ControlStatus::kInvalidArgument;
    }
    if (!d.empty() && d.size() != 1 && (nu > 0 && d.size() / nu < ny)) {
        return ControlStatus::kInvalidArgument;
    }
    if (!x0.empty() && x0.size() < n) {
        return ControlStatus::kInvalidArgument;
    }
    const bool want_traj = !x_traj_out.empty();
    if (want_traj && n > 0 && x_traj_out.size() / n < n_steps) {
        return ControlStatus::kInvalidArgument;
    }
    if (ts <= 0.0 && n_steps >= 2 && !is_strictly_increasing_finite(t)) {
        return ControlStatus::kInvalidArgument;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Allocator* sa = scratch.allocator();
    Array<f64> x(n, 0.0, sa);
    Array<f64> xn(n, 0.0, sa);
    if (!x0.empty()) {
        for (usize i = 0; i < n; ++i) {
            x[i] = x0[i];
        }
    }

    f64 dt = 0.0;
    const bool uniform = is_uniform_grid(t, dt);
    Array<f64> ad(n * n, 0.0, sa);
    Array<f64> b0(n * nu, 0.0, sa);
    Array<f64> b1(n * nu, 0.0, sa);
    if (ts <= 0.0 && uniform && dt > 0.0 && n > 0) {
        const ControlStatus st =
            foh_mimo_step(a, b, n, nu, dt, ad.as_mut_ref(), b0.as_mut_ref(), b1.as_mut_ref(), sa);
        if (st != ControlStatus::kOk) {
            return st;
        }
    }

    for (usize k = 0; k < n_steps; ++k) {
        if (want_traj) {
            for (usize i = 0; i < n; ++i) {
                x_traj_out[i * n_steps + k] = x[i];
            }
        }
        for (usize p = 0; p < ny; ++p) {
            f64 ypk = 0.0;
            for (usize i = 0; i < n; ++i) {
                ypk += c[i * ny + p] * x[i];
            }
            if (d.size() >= ny * nu) {
                for (usize j = 0; j < nu; ++j) {
                    ypk += d[j * ny + p] * u_colmajor[j * n_steps + k];
                }
            } else if (d.size() == 1 && d[0] != 0.0) {
                for (usize j = 0; j < nu; ++j) {
                    ypk += d[0] * u_colmajor[j * n_steps + k];
                }
            }
            y_out[p * n_steps + k] = ypk;
        }

        if (k + 1 < n_steps && n > 0) {
            if (ts > 0.0) {
                for (usize i = 0; i < n; ++i) {
                    f64 s = 0.0;
                    for (usize j = 0; j < n; ++j) {
                        s += a[j * n + i] * x[j];
                    }
                    for (usize j = 0; j < nu; ++j) {
                        s += b[j * n + i] * u_colmajor[j * n_steps + k];
                    }
                    xn[i] = s;
                }
            } else {
                if (!uniform) {
                    const ControlStatus st =
                        foh_mimo_step(a, b, n, nu, t[k + 1] - t[k], ad.as_mut_ref(),
                                      b0.as_mut_ref(), b1.as_mut_ref(), sa);
                    if (st != ControlStatus::kOk) {
                        return st;
                    }
                }
                for (usize i = 0; i < n; ++i) {
                    f64 s = 0.0;
                    for (usize j = 0; j < n; ++j) {
                        s += ad[j * n + i] * x[j];
                    }
                    for (usize j = 0; j < nu; ++j) {
                        s += b0[j * n + i] * u_colmajor[j * n_steps + k] +
                             b1[j * n + i] * u_colmajor[j * n_steps + (k + 1)];
                    }
                    xn[i] = s;
                }
            }
            keel::swap(x, xn);
        }
    }
    return ControlStatus::kOk;
}

ControlStatus step_info(ArrayRef<f64> t, ArrayRef<f64> y, StepInfo& out) noexcept {
    const usize n = t.size();
    if (n < 2 || y.size() < n) {
        return ControlStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        if (!keel::is_finite(t[i]) || !keel::is_finite(y[i])) {
            return ControlStatus::kInvalidArgument;
        }
    }

    const f64 yfinal = y[n - 1];
    const f64 abs_yf = keel::abs(yfinal);
    const f64 band = 0.02 * abs_yf;

    // Peak & peak time.
    f64 peak = keel::abs(y[0]);
    f64 peak_time = t[0];
    for (usize i = 0; i < n; ++i) {
        const f64 a = keel::abs(y[i]);
        if (a > peak) {
            peak = a;
            peak_time = t[i];
        }
    }

    // Overshoot & undershoot.
    f64 overshoot = 0.0;
    f64 undershoot = 0.0;
    if (yfinal > 0.0) {
        f64 max_y = y[0];
        f64 min_y = 0.0;
        for (usize i = 0; i < n; ++i) {
            if (y[i] > max_y) max_y = y[i];
            if (y[i] < min_y) min_y = y[i];
        }
        overshoot = keel::max(0.0, 100.0 * (max_y - yfinal) / yfinal);
        undershoot = keel::max(0.0, 100.0 * (-min_y) / yfinal);
    } else if (yfinal < 0.0) {
        f64 min_y = y[0];
        f64 max_y = 0.0;
        for (usize i = 0; i < n; ++i) {
            if (y[i] < min_y) min_y = y[i];
            if (y[i] > max_y) max_y = y[i];
        }
        overshoot = keel::max(0.0, 100.0 * (min_y - yfinal) / yfinal);
        undershoot = keel::max(0.0, 100.0 * max_y / (-yfinal));
    }

    // Rise time: 10% -> 90% of yfinal.
    const f64 nan = Limits<f64>::quiet_nan();
    f64 rise_time = nan;
    if (abs_yf > 0.0) {
        const f64 y10 = 0.1 * yfinal;
        const f64 y90 = 0.9 * yfinal;
        f64 t10 = nan;
        f64 t90 = nan;
        for (usize i = 1; i < n; ++i) {
            const f64 a = y[i - 1] - y10;
            const f64 b = y[i] - y10;
            if (keel::is_nan(t10)) {
                if (a == 0.0) {
                    t10 = t[i - 1];
                } else if (b == 0.0) {
                    t10 = t[i];
                } else if ((a < 0.0) != (b < 0.0)) {
                    const f64 frac = a / (a - b);
                    t10 = t[i - 1] + frac * (t[i] - t[i - 1]);
                }
            }
            const f64 a9 = y[i - 1] - y90;
            const f64 b9 = y[i] - y90;
            if (a9 == 0.0) {
                t90 = t[i - 1];
                if (!keel::is_nan(t10)) break;
            } else if (b9 == 0.0) {
                t90 = t[i];
                if (!keel::is_nan(t10)) break;
            } else if ((a9 < 0.0) != (b9 < 0.0)) {
                const f64 frac = a9 / (a9 - b9);
                t90 = t[i - 1] + frac * (t[i] - t[i - 1]);
                if (!keel::is_nan(t10)) {
                    break;
                }
            }
        }
        if (!keel::is_nan(t10) && !keel::is_nan(t90)) {
            rise_time = t90 - t10;
        }
    }

    // Settling time (2% of |yfinal|).
    f64 settling_time = 0.0;
    for (usize i = 0; i < n; ++i) {
        if (keel::abs(y[i] - yfinal) > band) {
            settling_time = t[i];
        }
    }

    // Transient time (2% of peak deviation max|y - yfinal|).
    f64 peak_dev = 0.0;
    for (usize i = 0; i < n; ++i) {
        peak_dev = keel::max(peak_dev, keel::abs(y[i] - yfinal));
    }
    const f64 trans_band = 0.02 * peak_dev;
    f64 transient_time = 0.0;
    for (usize i = 0; i < n; ++i) {
        if (keel::abs(y[i] - yfinal) > trans_band) {
            transient_time = t[i];
        }
    }

    // Min/max after settling.
    f64 s_min = yfinal;
    f64 s_max = yfinal;
    for (usize i = 0; i < n; ++i) {
        if (t[i] >= settling_time) {
            s_min = keel::min(s_min, y[i]);
            s_max = keel::max(s_max, y[i]);
        }
    }

    out.rise_time = rise_time;
    out.transient_time = transient_time;
    out.settling_time = settling_time;
    out.settling_min = s_min;
    out.settling_max = s_max;
    out.overshoot = overshoot;
    out.undershoot = undershoot;
    out.peak = peak;
    out.peak_time = peak_time;
    return ControlStatus::kOk;
}

} // namespace keel::control
