#include "keel/control.h"

#include "keel/array.h"
#include "keel/math.h"
#include "keel/poly.h"

namespace keel::control {

namespace {

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

// Largest singular value via Rayleigh power iteration on the smaller Gram matrix
// (Rayleigh quotient <= lambda_max, guaranteeing a safe lower bound).
f64 sigma_max(ArrayRef<f64> m, usize r, usize c, Allocator* alloc) {
    if (r == 0 || c == 0) {
        return 0.0;
    }
    const usize g = keel::min(r, c);
    Array<f64> gram_m(g * g, 0.0, alloc);
    if (c <= r) {
        for (usize a = 0; a < c; ++a) {
            for (usize b = 0; b < c; ++b) {
                f64 s = 0.0;
                for (usize k = 0; k < r; ++k) {
                    s += m[a * r + k] * m[b * r + k];
                }
                gram_m[b * c + a] = s;
            }
        }
    } else {
        for (usize a = 0; a < r; ++a) {
            for (usize b = 0; b < r; ++b) {
                f64 s = 0.0;
                for (usize k = 0; k < c; ++k) {
                    s += m[k * r + a] * m[k * r + b];
                }
                gram_m[b * r + a] = s;
            }
        }
    }
    Array<f64> x(g, 1.0, alloc);
    Array<f64> y(g, 0.0, alloc);
    f64 lambda = 0.0;
    for (i32 it = 0; it < 200; ++it) {
        for (usize i = 0; i < g; ++i) {
            f64 s = 0.0;
            for (usize k = 0; k < g; ++k) {
                s += gram_m[k * g + i] * x[k];
            }
            y[i] = s;
        }
        f64 num = 0.0;
        f64 den = 0.0;
        for (usize i = 0; i < g; ++i) {
            num += x[i] * y[i];
            den += x[i] * x[i];
        }
        if (den <= 0.0) {
            break;
        }
        lambda = num / den;
        f64 nrm = 0.0;
        for (usize i = 0; i < g; ++i) {
            nrm += y[i] * y[i];
        }
        nrm = keel::sqrt(nrm);
        if (nrm <= 0.0) {
            break;
        }
        for (usize i = 0; i < g; ++i) {
            x[i] = y[i] / nrm;
        }
    }
    return keel::sqrt(keel::max(lambda, 0.0));
}

Array<c64> eig_via_char_poly(ArrayRef<f64> m, usize n, Allocator* alloc) {
    Array<f64> cp(n + 1, 0.0, alloc);
    (void)char_poly(m, n, cp.as_mut_ref(), alloc);
    return keel::poly::roots(cp.as_ref(), alloc);
}

bool has_imag_eig(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<f64> c, ArrayRef<f64> d, usize n,
                  usize m, usize p, f64 gamma, f64 sig_d, Allocator* alloc) {
    if (gamma <= sig_d * (1.0 + 1e-12)) {
        return true;
    }

    const Array<f64> dt = transpose_mat(d, p, m, alloc);
    const Array<f64> dtd = mul_mat(dt.as_ref(), m, p, d, m, alloc);
    Array<f64> r(m * m, 0.0, alloc);
    for (usize i = 0; i < m * m; ++i) {
        r[i] = -dtd[i];
    }
    for (usize i = 0; i < m; ++i) {
        r[i * m + i] += gamma * gamma;
    }
    Array<f64> ri(alloc);
    if (!invert_mat(r.as_ref(), m, ri, alloc)) {
        return true;
    }

    const Array<f64> bt = transpose_mat(b, n, m, alloc);
    const Array<f64> ct = transpose_mat(c, p, n, alloc);

    // Abar = A + B * Ri * D^T * C
    const Array<f64> ri_dt = mul_mat(ri.as_ref(), m, m, dt.as_ref(), p, alloc);
    const Array<f64> b_ri_dt = mul_mat(b, n, m, ri_dt.as_ref(), p, alloc);
    Array<f64> abar = mul_mat(b_ri_dt.as_ref(), n, p, c, n, alloc);
    for (usize i = 0; i < n * n; ++i) {
        abar[i] += a[i];
    }

    // B * Ri * B^T
    const Array<f64> ri_bt = mul_mat(ri.as_ref(), m, m, bt.as_ref(), n, alloc);
    const Array<f64> b_ri_bt = mul_mat(b, n, m, ri_bt.as_ref(), n, alloc);

    // C^T * (I + D * Ri * D^T) * C
    const Array<f64> d_ri = mul_mat(d, p, m, ri.as_ref(), m, alloc);
    Array<f64> d_ri_dt = mul_mat(d_ri.as_ref(), p, m, dt.as_ref(), p, alloc);
    for (usize i = 0; i < p; ++i) {
        d_ri_dt[i * p + i] += 1.0;
    }
    const Array<f64> mid_c = mul_mat(d_ri_dt.as_ref(), p, p, c, n, alloc);
    const Array<f64> cmid = mul_mat(ct.as_ref(), n, p, mid_c.as_ref(), n, alloc);

    // Assemble 2n x 2n Hamiltonian M(gamma) = [Abar, B*Ri*B^T; -Cmid, -Abar^T].
    const usize k = 2 * n;
    Array<f64> ham(k * k, 0.0, alloc);
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            ham[j * k + i] = abar[j * n + i];
            ham[(n + j) * k + i] = b_ri_bt[j * n + i];
            ham[j * k + (n + i)] = -cmid[j * n + i];
            ham[(n + j) * k + (n + i)] = -abar[i * n + j];
        }
    }

    const Array<c64> ev = eig_via_char_poly(ham.as_ref(), k, alloc);
    f64 scale = 1.0;
    for (usize i = 0; i < ev.size(); ++i) {
        scale = keel::max(scale, keel::abs(ev[i]));
    }
    const f64 tol = 1e-7 * scale;
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
    if (a_in.size() < n * n || b_in.size() < n * m || c_in.size() < p * n || d_in.size() < p * m) {
        return ControlStatus::kInvalidArgument;
    }

    const f64 inf = Limits<f64>::infinity();
    f64 sig_d = sigma_max(d_in, p, m, alloc);
    if (n == 0) {
        norm_out = sig_d;
        return ControlStatus::kOk;
    }

    Array<f64> a(n * n, 0.0, alloc);
    Array<f64> b(n * m, 0.0, alloc);
    Array<f64> c(p * n, 0.0, alloc);
    Array<f64> d(p * m, 0.0, alloc);
    for (usize i = 0; i < n * n; ++i) a[i] = a_in[i];
    for (usize i = 0; i < n * m; ++i) b[i] = b_in[i];
    for (usize i = 0; i < p * n; ++i) c[i] = c_in[i];
    for (usize i = 0; i < p * m; ++i) d[i] = d_in[i];

    if (discrete) {
        const Array<c64> ev = eig_via_char_poly(a.as_ref(), n, alloc);
        f64 max_abs = 0.0;
        for (usize i = 0; i < ev.size(); ++i) {
            max_abs = keel::max(max_abs, keel::abs(ev[i]));
        }
        if (max_abs >= 1.0 - 1e-9) {
            norm_out = inf;
            return ControlStatus::kOk;
        }

        // Isometric bilinear transform z = (1 + s) / (1 - s).
        Array<f64> mat_m = a.clone();
        Array<f64> a_minus_i = a.clone();
        for (usize i = 0; i < n; ++i) {
            mat_m[i + i * n] += 1.0;
            a_minus_i[i + i * n] -= 1.0;
        }
        Array<f64> mi(alloc);
        if (!invert_mat(mat_m.as_ref(), n, mi, alloc)) {
            norm_out = inf;
            return ControlStatus::kOk;
        }
        Array<f64> ac = mul_mat(a_minus_i.as_ref(), n, n, mi.as_ref(), n, alloc);
        Array<f64> mi_b = mul_mat(mi.as_ref(), n, n, b.as_ref(), m, alloc);
        Array<f64> c_mi = mul_mat(c.as_ref(), p, n, mi.as_ref(), n, alloc);
        const Array<f64> c_mi_b = mul_mat(c.as_ref(), p, n, mi_b.as_ref(), m, alloc);
        for (usize i = 0; i < mi_b.size(); ++i) mi_b[i] *= kSqrt2;
        for (usize i = 0; i < c_mi.size(); ++i) c_mi[i] *= kSqrt2;
        Array<f64> dc(p * m, 0.0, alloc);
        for (usize i = 0; i < p * m; ++i) dc[i] = d[i] - c_mi_b[i];

        a = keel::move(ac);
        b = keel::move(mi_b);
        c = keel::move(c_mi);
        d = keel::move(dc);
        sig_d = sigma_max(d.as_ref(), p, m, alloc);
    } else {
        const Array<c64> ev = eig_via_char_poly(a.as_ref(), n, alloc);
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
        Array<f64> ai(alloc);
        if (invert_mat(a.as_ref(), n, ai, alloc)) {
            const Array<f64> ai_b = mul_mat(ai.as_ref(), n, n, b.as_ref(), m, alloc);
            const Array<f64> c_ai_b = mul_mat(c.as_ref(), p, n, ai_b.as_ref(), m, alloc);
            Array<f64> g0(p * m, 0.0, alloc);
            for (usize i = 0; i < p * m; ++i) {
                g0[i] = d[i] - c_ai_b[i];
            }
            lb = keel::max(lb, sigma_max(g0.as_ref(), p, m, alloc));
        }
    }

    f64 lo = lb;
    f64 hi = (lb > 0.0 ? lb : 1.0) * (1.0 + 1e-6) + 1e-300;
    i32 grow = 0;
    while (has_imag_eig(a.as_ref(), b.as_ref(), c.as_ref(), d.as_ref(), n, m, p, hi, sig_d,
                        alloc)) {
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
        if (has_imag_eig(a.as_ref(), b.as_ref(), c.as_ref(), d.as_ref(), n, m, p, mid, sig_d,
                         alloc)) {
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
