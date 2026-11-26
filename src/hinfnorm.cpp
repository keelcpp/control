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

Array<f64> transpose_mat(ArrayRef<f64> a, usize r, usize c, Allocator* alloc) {
    Array<f64> t(r * c, 0.0, alloc);
    for (usize j = 0; j < c; ++j) {
        for (usize i = 0; i < r; ++i) {
            t[i * c + j] = a[j * r + i];
        }
    }
    return t;
}

Array<f64> mul_mat(ArrayRef<f64> a, usize ra, usize ca, ArrayRef<f64> b, usize cb,
                   Allocator* alloc) {
    Array<f64> c(ra * cb, 0.0, alloc);
    (void)matmul(a, ra, ca, b, cb, c.as_mut_ref());
    return c;
}

bool invert_mat(ArrayRef<f64> a, usize n, Array<f64>& out, Allocator* alloc) {
    Array<f64> lu(n * n, 0.0, alloc);
    for (usize i = 0; i < n * n; ++i) {
        lu[i] = a[i];
    }
    out = Array<f64>(n * n, 0.0, alloc);
    for (usize i = 0; i < n; ++i) {
        out[i * n + i] = 1.0;
    }
    return solve_in_place(lu.as_mut_ref(), out.as_mut_ref(), n, n) == ControlStatus::kOk;
}

// Exact largest singular value via SVD.
f64 sigma_max(ArrayRef<f64> m, usize r, usize c, Allocator* alloc) {
    if (r == 0 || c == 0) {
        return 0.0;
    }
    const usize mn = keel::min(r, c);
    ArenaTemp scratch = keel::get_scratch(alloc);
    Allocator* sa = scratch.allocator();
    Array<f64> u(r * r, 0.0, sa);
    Array<f64> s(mn, 0.0, sa);
    Array<f64> v(c * c, 0.0, sa);
    keel::linalg::svd_full(m.data(), r, c, u.data(), s.data(), v.data(), sa);
    return s[0];
}

// Eigenvalues via real Schur decomposition (accurate for arbitrary order n).
Array<c64> eig_via_schur(ArrayRef<f64> m, usize n, Allocator* alloc) {
    Array<c64> ev(alloc);
    ev.reserve(n);
    if (n == 0) {
        return ev;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    Allocator* sa = scratch.allocator();
    Array<f64> u(n * n, 0.0, sa);
    Array<f64> t(n * n, 0.0, sa);
    keel::linalg::schur_real(m.data(), n, u.data(), t.data(), sa);

    usize i = 0;
    while (i < n) {
        if (i + 1 < n && t[i * n + (i + 1)] != 0.0) {
            const f64 a11 = t[i * n + i];
            const f64 a12 = t[(i + 1) * n + i];
            const f64 a21 = t[i * n + (i + 1)];
            const f64 a22 = t[(i + 1) * n + (i + 1)];
            const f64 tr = 0.5 * (a11 + a22);
            const f64 diff = 0.5 * (a11 - a22);
            const f64 disc = diff * diff + a12 * a21;
            if (disc < 0.0) {
                const f64 im = keel::sqrt(-disc);
                ev.push_back(c64(tr, im));
                ev.push_back(c64(tr, -im));
            } else {
                const f64 rt = keel::sqrt(disc);
                ev.push_back(c64(tr + rt, 0.0));
                ev.push_back(c64(tr - rt, 0.0));
            }
            i += 2;
        } else {
            ev.push_back(c64(t[i * n + i], 0.0));
            ++i;
        }
    }
    return ev;
}

bool has_imag_eig(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<f64> c, ArrayRef<f64> d, usize n,
                  usize m, usize p, f64 gamma, f64 sig_d, Allocator* alloc) {
    if (gamma <= sig_d * (1.0 + 1e-12)) {
        return true;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    Allocator* sa = scratch.allocator();

    const Array<f64> dt = transpose_mat(d, p, m, sa);
    const Array<f64> dtd = mul_mat(dt.as_ref(), m, p, d, m, sa);
    Array<f64> r(m * m, 0.0, sa);
    for (usize i = 0; i < m * m; ++i) {
        r[i] = -dtd[i];
    }
    for (usize i = 0; i < m; ++i) {
        r[i * m + i] += gamma * gamma;
    }
    Array<f64> ri(sa);
    if (!invert_mat(r.as_ref(), m, ri, sa)) {
        return true;
    }

    const Array<f64> bt = transpose_mat(b, n, m, sa);
    const Array<f64> ct = transpose_mat(c, p, n, sa);

    // Abar = A + B * Ri * D^T * C
    const Array<f64> ri_dt = mul_mat(ri.as_ref(), m, m, dt.as_ref(), p, sa);
    const Array<f64> b_ri_dt = mul_mat(b, n, m, ri_dt.as_ref(), p, sa);
    Array<f64> abar = mul_mat(b_ri_dt.as_ref(), n, p, c, n, sa);
    for (usize i = 0; i < n * n; ++i) {
        abar[i] += a[i];
    }

    // B * Ri * B^T
    const Array<f64> ri_bt = mul_mat(ri.as_ref(), m, m, bt.as_ref(), n, sa);
    const Array<f64> b_ri_bt = mul_mat(b, n, m, ri_bt.as_ref(), n, sa);

    // C^T * (I + D * Ri * D^T) * C
    const Array<f64> d_ri = mul_mat(d, p, m, ri.as_ref(), m, sa);
    Array<f64> d_ri_dt = mul_mat(d_ri.as_ref(), p, m, dt.as_ref(), p, sa);
    for (usize i = 0; i < p; ++i) {
        d_ri_dt[i * p + i] += 1.0;
    }
    const Array<f64> mid_c = mul_mat(d_ri_dt.as_ref(), p, p, c, n, sa);
    const Array<f64> cmid = mul_mat(ct.as_ref(), n, p, mid_c.as_ref(), n, sa);

    // Assemble 2n x 2n Hamiltonian M(gamma) = [Abar, B*Ri*B^T; -Cmid, -Abar^T].
    const usize k = 2 * n;
    Array<f64> ham(k * k, 0.0, sa);
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            ham[j * k + i] = abar[j * n + i];
            ham[(n + j) * k + i] = b_ri_bt[j * n + i];
            ham[j * k + (n + i)] = -cmid[j * n + i];
            ham[(n + j) * k + (n + i)] = -abar[i * n + j];
        }
    }

    const Array<c64> ev = eig_via_schur(ham.as_ref(), k, sa);
    f64 scale = 1.0;
    for (usize i = 0; i < ev.size(); ++i) {
        scale = keel::max(scale, keel::abs(ev[i]));
    }
    const f64 tol = 1e-8 * scale;
    for (usize i = 0; i < ev.size(); ++i) {
        if (keel::abs(ev[i].real()) < tol && keel::abs(ev[i].imag()) > tol) {
            return true;
        }
    }
    return false;
}

} // namespace

ControlStatus hinfnorm(ArrayRef<f64> a_in, ArrayRef<f64> b_in, ArrayRef<f64> c_in,
                       ArrayRef<f64> d_in, usize n, usize m, usize p, bool discrete, f64& norm_out,
                       Allocator* alloc) {
    if ((n > 0 && (a_in.size() / n < n || n > Limits<usize>::kMax / 4u ||
                   (2u * n) > Limits<usize>::kMax / (2u * n))) ||
        (n > 0 && m > 0 && b_in.size() / m < n) || (n > 0 && p > 0 && c_in.size() / n < p) ||
        (m > 0 && p > 0 && d_in.size() / m < p)) {
        return ControlStatus::kInvalidArgument;
    }
    if (!all_finite_n(a_in, n * n) || !all_finite_n(b_in, n * m) || !all_finite_n(c_in, p * n) ||
        !all_finite_n(d_in, p * m)) {
        return ControlStatus::kInvalidArgument;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Allocator* sa = scratch.allocator();
    const f64 inf = Limits<f64>::infinity();
    f64 sig_d = sigma_max(d_in, p, m, sa);
    if (n == 0) {
        norm_out = sig_d;
        return ControlStatus::kOk;
    }

    Array<f64> a(n * n, 0.0, sa);
    Array<f64> b(n * m, 0.0, sa);
    Array<f64> c(p * n, 0.0, sa);
    Array<f64> d(p * m, 0.0, sa);
    for (usize i = 0; i < n * n; ++i) a[i] = a_in[i];
    for (usize i = 0; i < n * m; ++i) b[i] = b_in[i];
    for (usize i = 0; i < p * n; ++i) c[i] = c_in[i];
    for (usize i = 0; i < p * m; ++i) d[i] = d_in[i];

    if (discrete) {
        const Array<c64> ev = eig_via_schur(a.as_ref(), n, sa);
        f64 max_abs = 0.0;
        for (usize i = 0; i < ev.size(); ++i) {
            max_abs = keel::max(max_abs, keel::abs(ev[i]));
        }
        if (max_abs >= 1.0 - 1e-9) {
            norm_out = inf;
            return ControlStatus::kOk;
        }

        // Isometric bilinear transform z = (1 + s) / (1 - s).
        Array<f64> mat_m = Array<f64>::from(a.as_ref(), sa);
        Array<f64> a_minus_i = Array<f64>::from(a.as_ref(), sa);
        for (usize i = 0; i < n; ++i) {
            mat_m[i + i * n] += 1.0;
            a_minus_i[i + i * n] -= 1.0;
        }
        Array<f64> mi(sa);
        if (!invert_mat(mat_m.as_ref(), n, mi, sa)) {
            norm_out = inf;
            return ControlStatus::kOk;
        }
        Array<f64> ac = mul_mat(a_minus_i.as_ref(), n, n, mi.as_ref(), n, sa);
        Array<f64> mi_b = mul_mat(mi.as_ref(), n, n, b.as_ref(), m, sa);
        Array<f64> c_mi = mul_mat(c.as_ref(), p, n, mi.as_ref(), n, sa);
        const Array<f64> c_mi_b = mul_mat(c.as_ref(), p, n, mi_b.as_ref(), m, sa);
        for (usize i = 0; i < mi_b.size(); ++i) mi_b[i] *= kSqrt2;
        for (usize i = 0; i < c_mi.size(); ++i) c_mi[i] *= kSqrt2;
        Array<f64> dc(p * m, 0.0, sa);
        for (usize i = 0; i < p * m; ++i) dc[i] = d[i] - c_mi_b[i];

        a = keel::move(ac);
        b = keel::move(mi_b);
        c = keel::move(c_mi);
        d = keel::move(dc);
        sig_d = sigma_max(d.as_ref(), p, m, sa);
    } else {
        const Array<c64> ev = eig_via_schur(a.as_ref(), n, sa);
        f64 max_re = -inf;
        for (usize i = 0; i < ev.size(); ++i) {
            max_re = keel::max(max_re, ev[i].real());
        }
        if (max_re >= -1e-9) {
            norm_out = inf;
            return ControlStatus::kOk;
        }
    }

    // Lower bracket from max(sigma_max(D), sigma_max(G(0))).
    f64 lb = sig_d;
    {
        Array<f64> ai(sa);
        if (invert_mat(a.as_ref(), n, ai, sa)) {
            const Array<f64> ai_b = mul_mat(ai.as_ref(), n, n, b.as_ref(), m, sa);
            const Array<f64> c_ai_b = mul_mat(c.as_ref(), p, n, ai_b.as_ref(), m, sa);
            Array<f64> g0(p * m, 0.0, sa);
            for (usize i = 0; i < p * m; ++i) {
                g0[i] = d[i] - c_ai_b[i];
            }
            lb = keel::max(lb, sigma_max(g0.as_ref(), p, m, sa));
        }
    }

    f64 lo = lb;
    f64 hi = (lb > 0.0 ? lb : 1.0) * (1.0 + 1e-6) + 1e-300;
    i32 grow = 0;
    while (has_imag_eig(a.as_ref(), b.as_ref(), c.as_ref(), d.as_ref(), n, m, p, hi, sig_d, sa)) {
        hi *= 2.0;
        if (++grow > 80) {
            norm_out = inf;
            return ControlStatus::kOk;
        }
    }

    for (i32 it = 0; it < 200; ++it) {
        const f64 mid = 0.5 * (lo + hi);
        if (mid <= lo || mid >= hi) {
            break;
        }
        if (has_imag_eig(a.as_ref(), b.as_ref(), c.as_ref(), d.as_ref(), n, m, p, mid, sig_d, sa)) {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo <= 1e-12 * (1.0 + hi)) {
            break;
        }
    }
    norm_out = 0.5 * (lo + hi);
    return ControlStatus::kOk;
}

} // namespace keel::control
