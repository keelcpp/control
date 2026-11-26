#include "keel/control.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/linalg.h"
#include "keel/math.h"

namespace keel::control {

namespace {

bool all_finite_n(ArrayRef<f64> v, usize cnt) noexcept {
    for (usize i = 0; i < cnt; ++i) {
        if (!keel::is_finite(v[i])) return false;
    }
    return true;
}

} // namespace

ControlStatus c2d_zoh(ArrayRef<f64> a, ArrayRef<f64> b, usize n, usize m, f64 ts,
                      MutArrayRef<f64> ad_out, MutArrayRef<f64> bd_out, Allocator* alloc) {
    if (!keel::is_finite(ts) || !(ts > 0.0) ||
        (n > 0 && (a.size() / n < n || ad_out.size() / n < n)) ||
        (n > 0 && m > 0 && (b.size() / m < n || bd_out.size() / m < n))) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ControlStatus::kOk;
    }
    if (n > Limits<usize>::kMax - m) {
        return ControlStatus::kInvalidArgument;
    }
    const usize aug = n + m;
    if (aug > Limits<usize>::kMax / aug) {
        return ControlStatus::kInvalidArgument;
    }
    if (!all_finite_n(a, n * n) || !all_finite_n(b, n * m)) {
        return ControlStatus::kInvalidArgument;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> mat(aug * aug, 0.0, scratch.allocator());
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            mat[j * aug + i] = a[j * n + i] * ts;
        }
    }
    for (usize j = 0; j < m; ++j) {
        for (usize i = 0; i < n; ++i) {
            mat[(n + j) * aug + i] = b[j * n + i] * ts;
        }
    }
    Array<f64> e(aug * aug, 0.0, scratch.allocator());
    const ControlStatus st = expm(mat.as_ref(), aug, e.as_mut_ref(), scratch.allocator());
    if (st != ControlStatus::kOk) {
        return st;
    }
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            ad_out[j * n + i] = e[j * aug + i];
        }
    }
    for (usize j = 0; j < m; ++j) {
        for (usize i = 0; i < n; ++i) {
            bd_out[j * n + i] = e[(n + j) * aug + i];
        }
    }
    return ControlStatus::kOk;
}

ControlStatus c2d_foh(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<f64> c, ArrayRef<f64> d, usize n,
                      usize m, usize p, f64 ts, MutArrayRef<f64> ad_out, MutArrayRef<f64> bd_out,
                      MutArrayRef<f64> cd_out, MutArrayRef<f64> dd_out, MutArrayRef<f64> g_out,
                      Allocator* alloc) {
    if (!keel::is_finite(ts) || !(ts > 0.0) ||
        (n > 0 && (a.size() / n < n || ad_out.size() / n < n)) ||
        (n > 0 && m > 0 && (b.size() / m < n || bd_out.size() / m < n)) ||
        (n > 0 && p > 0 && (c.size() / n < p || cd_out.size() / n < p)) ||
        (m > 0 && p > 0 && (d.size() / m < p || dd_out.size() / m < p))) {
        return ControlStatus::kInvalidArgument;
    }
    const bool want_g = !g_out.empty();
    if (want_g && n > 0 && (n > Limits<usize>::kMax - m || g_out.size() / n < (n + m))) {
        return ControlStatus::kInvalidArgument;
    }
    if (!all_finite_n(a, n * n) || !all_finite_n(b, n * m) || !all_finite_n(c, p * n) ||
        !all_finite_n(d, p * m)) {
        return ControlStatus::kInvalidArgument;
    }

    for (usize i = 0; i < p * n; ++i) {
        cd_out[i] = c[i];
    }
    for (usize i = 0; i < p * m; ++i) {
        dd_out[i] = d[i];
    }
    if (want_g) {
        for (usize i = 0; i < n * (n + m); ++i) {
            g_out[i] = 0.0;
        }
        for (usize i = 0; i < n; ++i) {
            g_out[i * n + i] = 1.0;
        }
    }
    if (n == 0) {
        return ControlStatus::kOk;
    }
    if (m > Limits<usize>::kMax / 2u || n > Limits<usize>::kMax - 2u * m) {
        return ControlStatus::kInvalidArgument;
    }
    const usize aug = n + 2 * m;
    if (aug > Limits<usize>::kMax / aug) {
        return ControlStatus::kInvalidArgument;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> mat(aug * aug, 0.0, scratch.allocator());
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            mat[j * aug + i] = a[j * n + i] * ts;
        }
    }
    for (usize j = 0; j < m; ++j) {
        for (usize i = 0; i < n; ++i) {
            mat[(n + j) * aug + i] = b[j * n + i] * ts;
        }
        mat[(n + m + j) * aug + (n + j)] = ts;
    }

    Array<f64> e(aug * aug, 0.0, scratch.allocator());
    const ControlStatus st = expm(mat.as_ref(), aug, e.as_mut_ref(), scratch.allocator());
    if (st != ControlStatus::kOk) {
        return st;
    }

    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            ad_out[j * n + i] = e[j * aug + i];
        }
    }

    for (usize j = 0; j < m; ++j) {
        for (usize r = 0; r < n; ++r) {
            const f64 g1 = e[(n + j) * aug + r];
            const f64 g2 = e[(n + m + j) * aug + r];
            f64 phi_g2 = 0.0;
            for (usize k = 0; k < n; ++k) {
                phi_g2 += ad_out[k * n + r] * e[(n + m + j) * aug + k];
            }
            bd_out[j * n + r] = g1 + phi_g2 / ts - g2 / ts;
            if (want_g) {
                g_out[(n + j) * n + r] = -g2 / ts;
            }
        }
        for (usize row = 0; row < p; ++row) {
            f64 cg2 = 0.0;
            for (usize k = 0; k < n; ++k) {
                cg2 += c[k * p + row] * e[(n + m + j) * aug + k];
            }
            dd_out[j * p + row] += cg2 / ts;
        }
    }
    return ControlStatus::kOk;
}

ControlStatus c2d_tustin(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<f64> c, ArrayRef<f64> d,
                         usize n, usize m, usize p, f64 ts, MutArrayRef<f64> ad_out,
                         MutArrayRef<f64> bd_out, MutArrayRef<f64> cd_out, MutArrayRef<f64> dd_out,
                         Allocator* alloc) {
    if (!keel::is_finite(ts) || !(ts > 0.0) ||
        (n > 0 && (a.size() / n < n || ad_out.size() / n < n)) ||
        (n > 0 && m > 0 && (b.size() / m < n || bd_out.size() / m < n)) ||
        (n > 0 && p > 0 && (c.size() / n < p || cd_out.size() / n < p)) ||
        (m > 0 && p > 0 && (d.size() / m < p || dd_out.size() / m < p))) {
        return ControlStatus::kInvalidArgument;
    }
    if (!all_finite_n(a, n * n) || !all_finite_n(b, n * m) || !all_finite_n(c, p * n) ||
        !all_finite_n(d, p * m)) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0) {
        for (usize i = 0; i < p * m; ++i) {
            dd_out[i] = d[i];
        }
        return ControlStatus::kOk;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Allocator* sa = scratch.allocator();
    const f64 half_ts = 0.5 * ts;
    const f64 s_ts = keel::sqrt(ts);

    Array<f64> m_minus(n * n, 0.0, sa);
    Array<f64> m_plus(n * n, 0.0, sa);
    for (usize i = 0; i < n * n; ++i) {
        m_minus[i] = -a[i] * half_ts;
        m_plus[i] = a[i] * half_ts;
    }
    for (usize i = 0; i < n; ++i) {
        m_minus[i * n + i] += 1.0;
        m_plus[i * n + i] += 1.0;
    }

    // Ad = inv(M_minus) * M_plus
    Array<f64> lu = Array<f64>::from(m_minus.as_ref(), sa);
    Array<f64> ad_tmp = Array<f64>::from(m_plus.as_ref(), sa);
    ControlStatus st = solve_in_place(lu.as_mut_ref(), ad_tmp.as_mut_ref(), n, n);
    if (st != ControlStatus::kOk) {
        return st;
    }

    // Y = inv(M_minus) * B  (n x m);  Bd = sqrt(Ts) * Y
    Array<f64> y(n * m, 0.0, sa);
    if (m > 0) {
        lu = Array<f64>::from(m_minus.as_ref(), sa);
        for (usize i = 0; i < n * m; ++i) {
            y[i] = b[i];
        }
        st = solve_in_place(lu.as_mut_ref(), y.as_mut_ref(), n, m);
        if (st != ControlStatus::kOk) {
            return st;
        }
    }

    // Cd = sqrt(Ts) * C * inv(M_minus)
    Array<f64> cd_tmp(p * n, 0.0, sa);
    if (p > 0) {
        Array<f64> mt(n * n, 0.0, sa);
        for (usize j = 0; j < n; ++j) {
            for (usize i = 0; i < n; ++i) {
                mt[i * n + j] = m_minus[j * n + i];
            }
        }
        Array<f64> z(n * p, 0.0, sa);
        for (usize row = 0; row < p; ++row) {
            for (usize col = 0; col < n; ++col) {
                z[row * n + col] = c[col * p + row];
            }
        }
        st = solve_in_place(mt.as_mut_ref(), z.as_mut_ref(), n, p);
        if (st != ControlStatus::kOk) {
            return st;
        }
        for (usize col = 0; col < n; ++col) {
            for (usize row = 0; row < p; ++row) {
                cd_tmp[col * p + row] = s_ts * z[row * n + col];
            }
        }
    }

    // Dd = D + (Ts / 2) * C * Y (read C before writing cd_out in case of aliasing)
    if (p > 0 && m > 0) {
        Array<f64> cy(p * m, 0.0, sa);
        st = matmul(c, p, n, y.as_ref(), m, cy.as_mut_ref());
        if (st != ControlStatus::kOk) {
            return st;
        }
        for (usize i = 0; i < p * m; ++i) {
            dd_out[i] = d[i] + half_ts * cy[i];
        }
    }
    for (usize i = 0; i < n * n; ++i) ad_out[i] = ad_tmp[i];
    for (usize i = 0; i < n * m; ++i) bd_out[i] = s_ts * y[i];
    for (usize i = 0; i < p * n; ++i) cd_out[i] = cd_tmp[i];
    return ControlStatus::kOk;
}

ControlStatus d2c_zoh(ArrayRef<f64> ad, ArrayRef<f64> bd, ArrayRef<f64> cd, f64 dd, usize n, f64 ts,
                      StateSpace& out, Allocator* alloc) {
    if (!keel::is_finite(ts) || !(ts > 0.0) || !keel::is_finite(dd) ||
        (n > 0 && ad.size() / n < n) || bd.size() < n || cd.size() < n) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0) {
        out.n = 0;
        out.a = Array<f64>(alloc);
        out.b = Array<f64>(alloc);
        out.c = Array<f64>(alloc);
        out.d = dd;
        return ControlStatus::kOk;
    }
    if (!all_finite_n(ad, n * n) || !all_finite_n(bd, n) || !all_finite_n(cd, n)) {
        return ControlStatus::kInvalidArgument;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Allocator* sa = scratch.allocator();
    const usize m = n + 1;
    Array<c64> phi(m * m, c64(0.0, 0.0), sa);
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            phi[j * m + i] = c64(ad[j * n + i], 0.0);
        }
    }
    for (usize i = 0; i < n; ++i) {
        phi[n * m + i] = c64(bd[i], 0.0);
    }
    phi[n * m + n] = c64(1.0, 0.0);

    Array<c64> log_phi(m * m, c64(0.0, 0.0), sa);
    if (!keel::linalg::logm(phi.data(), m, log_phi.data(), sa)) {
        return ControlStatus::kSingular;
    }

    bool has_imag = false;
    for (usize j = 0; j <= n; ++j) {
        for (usize i = 0; i < n; ++i) {
            const c64 v = log_phi[j * m + i];
            if (!keel::is_finite(v.real()) || !keel::is_finite(v.imag())) {
                return ControlStatus::kSingular;
            }
            if (keel::abs(v.imag()) > 1e-9) {
                has_imag = true;
            }
        }
    }

    if (!has_imag) {
        out.n = n;
        out.a = Array<f64>(n * n, 0.0, alloc);
        out.b = Array<f64>(n, 0.0, alloc);
        out.c = Array<f64>(n, 0.0, alloc);
        out.d = dd;
        for (usize j = 0; j < n; ++j) {
            for (usize i = 0; i < n; ++i) {
                out.a[j * n + i] = log_phi[j * m + i].real() / ts;
            }
        }
        for (usize i = 0; i < n; ++i) {
            out.b[i] = log_phi[n * m + i].real() / ts;
            out.c[i] = cd[i];
        }
        return ControlStatus::kOk;
    }

    // Negative real discrete eigenvalues produce complex logm entries:
    // expand complex state z = x_R + i*x_I into real 2n-dimensional system.
    const usize n2 = 2 * n;
    out.n = n2;
    out.a = Array<f64>(n2 * n2, 0.0, alloc);
    out.b = Array<f64>(n2, 0.0, alloc);
    out.c = Array<f64>(n2, 0.0, alloc);
    out.d = dd;
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            const f64 ar = log_phi[j * m + i].real() / ts;
            const f64 ai = log_phi[j * m + i].imag() / ts;
            out.a[j * n2 + i] = ar;
            out.a[(j + n) * n2 + i] = -ai;
            out.a[j * n2 + (i + n)] = ai;
            out.a[(j + n) * n2 + (i + n)] = ar;
        }
    }
    for (usize i = 0; i < n; ++i) {
        out.b[i] = log_phi[n * m + i].real() / ts;
        out.b[i + n] = log_phi[n * m + i].imag() / ts;
        out.c[i] = cd[i];
    }
    return ControlStatus::kOk;
}

ControlStatus d2c_tustin(ArrayRef<f64> ad, ArrayRef<f64> bd, ArrayRef<f64> cd, f64 dd, usize n,
                         f64 ts, MutArrayRef<f64> ac_out, MutArrayRef<f64> bc_out,
                         MutArrayRef<f64> cc_out, f64& dc_out, Allocator* alloc) {
    if (!keel::is_finite(ts) || !(ts > 0.0) || !keel::is_finite(dd) ||
        (n > 0 && (ad.size() / n < n || ac_out.size() / n < n)) || bd.size() < n || cd.size() < n ||
        bc_out.size() < n || cc_out.size() < n) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0) {
        dc_out = dd;
        return ControlStatus::kOk;
    }
    if (!all_finite_n(ad, n * n) || !all_finite_n(bd, n) || !all_finite_n(cd, n)) {
        return ControlStatus::kInvalidArgument;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Allocator* sa = scratch.allocator();
    Array<f64> a_plus(n * n, 0.0, sa);
    Array<f64> a_minus(n * n, 0.0, sa);
    for (usize i = 0; i < n * n; ++i) {
        a_plus[i] = ad[i];
        a_minus[i] = ad[i];
    }
    for (usize i = 0; i < n; ++i) {
        a_plus[i * n + i] += 1.0;
        a_minus[i * n + i] -= 1.0;
    }

    // Ac = (2 / Ts) * inv(Ad + I) * (Ad - I)
    Array<f64> lu = Array<f64>::from(a_plus.as_ref(), sa);
    Array<f64> ac_tmp = Array<f64>::from(a_minus.as_ref(), sa);
    ControlStatus st = solve_in_place(lu.as_mut_ref(), ac_tmp.as_mut_ref(), n, n);
    if (st != ControlStatus::kOk) {
        return st;
    }
    const f64 scale_a = 2.0 / ts;
    for (usize i = 0; i < n * n; ++i) {
        ac_tmp[i] *= scale_a;
    }

    // Bc = (2 / sqrt(Ts)) * inv(Ad + I) * Bd
    lu = Array<f64>::from(a_plus.as_ref(), sa);
    Array<f64> bc_tmp(n, 0.0, sa);
    for (usize i = 0; i < n; ++i) {
        bc_tmp[i] = bd[i];
    }
    st = solve_in_place(lu.as_mut_ref(), bc_tmp.as_mut_ref(), n, 1);
    if (st != ControlStatus::kOk) {
        return st;
    }
    const f64 inv_sqrt_ts = 2.0 / keel::sqrt(ts);
    for (usize i = 0; i < n; ++i) {
        bc_tmp[i] *= inv_sqrt_ts;
    }

    // Cc = (2 / sqrt(Ts)) * Cd * inv(Ad + I)
    Array<f64> apt(n * n, 0.0, sa);
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            apt[i * n + j] = a_plus[j * n + i];
        }
    }
    Array<f64> cc_tmp(n, 0.0, sa);
    for (usize i = 0; i < n; ++i) {
        cc_tmp[i] = cd[i];
    }
    st = solve_in_place(apt.as_mut_ref(), cc_tmp.as_mut_ref(), n, 1);
    if (st != ControlStatus::kOk) {
        return st;
    }
    for (usize i = 0; i < n; ++i) {
        cc_tmp[i] *= inv_sqrt_ts;
    }

    // Dc = Dd - (Ts / 4) * Cc * (Ad + I) * Bc
    f64 cab = 0.0;
    for (usize i = 0; i < n; ++i) {
        f64 s = 0.0;
        for (usize j = 0; j < n; ++j) {
            s += a_plus[j * n + i] * bc_tmp[j];
        }
        cab += cc_tmp[i] * s;
    }
    for (usize i = 0; i < n * n; ++i) ac_out[i] = ac_tmp[i];
    for (usize i = 0; i < n; ++i) bc_out[i] = bc_tmp[i];
    for (usize i = 0; i < n; ++i) cc_out[i] = cc_tmp[i];
    dc_out = dd - (0.25 * ts) * cab;
    return ControlStatus::kOk;
}

} // namespace keel::control
