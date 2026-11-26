#include "keel/control.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"
#include "keel/poly.h"

namespace keel::control {

namespace {

ArrayRef<f64> strip_leading_zeros(ArrayRef<f64> v) noexcept {
    usize i = 0;
    while (i + 1 < v.size() && v[i] == 0.0) {
        ++i;
    }
    return v.slice(i);
}

bool all_finite(ArrayRef<f64> v) noexcept {
    for (usize i = 0; i < v.size(); ++i) {
        if (!keel::is_finite(v[i])) return false;
    }
    return true;
}

} // namespace

ControlStatus tf2ss(ArrayRef<f64> num, ArrayRef<f64> den, StateSpace& out, Allocator* alloc) {
    if (!all_finite(num) || !all_finite(den)) {
        return ControlStatus::kInvalidArgument;
    }
    const ArrayRef<f64> num_s = strip_leading_zeros(num);
    const ArrayRef<f64> den_s = strip_leading_zeros(den);
    if (den_s.empty() || den_s[0] == 0.0) {
        return ControlStatus::kInvalidArgument;
    }
    if (num_s.size() > den_s.size()) {
        return ControlStatus::kInvalidArgument;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    const f64 a0 = den_s[0];
    const usize m = den_s.size();
    Array<f64> a(m, 0.0, scratch.allocator());
    for (usize i = 0; i < m; ++i) {
        a[i] = den_s[i] / a0;
    }

    Array<f64> b(m, 0.0, scratch.allocator());
    const usize pad = m - num_s.size();
    for (usize i = 0; i < num_s.size(); ++i) {
        b[pad + i] = num_s[i] / a0;
    }

    const usize n = m - 1;
    out.n = n;
    out.a = Array<f64>(n * n, 0.0, alloc);
    out.b = Array<f64>(n, 0.0, alloc);
    out.c = Array<f64>(n, 0.0, alloc);
    out.d = b.empty() ? 0.0 : b[0];

    if (n == 0) {
        return ControlStatus::kOk;
    }

    // Controllable canonical form (column-major).
    for (usize i = 0; i + 1 < n; ++i) {
        out.a[(i + 1) * n + i] = 1.0;
    }
    for (usize j = 0; j < n; ++j) {
        out.a[j * n + (n - 1)] = -a[n - j];
    }
    out.b[n - 1] = 1.0;
    for (usize k = 0; k < n; ++k) {
        out.c[k] = b[n - k] - b[0] * a[n - k];
    }
    return ControlStatus::kOk;
}

ControlStatus ss2tf(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<f64> c, f64 d_scalar, usize n,
                    usize nu, usize iu, Array<f64>& num_out, Array<f64>& den_out,
                    Allocator* alloc) {
    const usize eff_nu = (nu == 0) ? 1 : nu;
    if (iu >= eff_nu || !keel::is_finite(d_scalar) || (n > 0 && a.size() / n < n) || c.size() < n) {
        return ControlStatus::kInvalidArgument;
    }
    if (n > 0 && b.size() / n < iu + 1) {
        return ControlStatus::kInvalidArgument;
    }

    if (n == 0) {
        den_out = Array<f64>(1, 1.0, alloc);
        if (d_scalar != 0.0) {
            num_out = Array<f64>(1, d_scalar, alloc);
        } else {
            num_out = Array<f64>(alloc);
        }
        return ControlStatus::kOk;
    }

    for (usize i = 0; i < n * n; ++i) {
        if (!keel::is_finite(a[i])) return ControlStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        if (!keel::is_finite(b[iu * n + i]) || !keel::is_finite(c[i])) {
            return ControlStatus::kInvalidArgument;
        }
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> bv(n, 0.0, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        bv[i] = b[iu * n + i];
    }

    // Faddeev-LeVerrier: M_0 = I, c_k = -(1/k) * tr(A * M_{k-1}), M_k = A * M_{k-1} + c_k * I.
    Array<f64> m_cur(n * n, 0.0, scratch.allocator());
    Array<f64> am(n * n, 0.0, scratch.allocator());
    Array<f64> mb(n, 0.0, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        m_cur[i * n + i] = 1.0;
    }

    Array<f64> coeff(n + 1, 0.0, alloc);
    coeff[0] = 1.0;
    Array<f64> num_acc(n, 0.0, alloc);

    // k = 0: C * I * B
    {
        f64 cb = 0.0;
        for (usize i = 0; i < n; ++i) {
            cb += c[i] * bv[i];
        }
        num_acc[0] = cb;
    }

    for (usize k = 1; k <= n; ++k) {
        const ControlStatus st = matmul(a, n, n, m_cur.as_ref(), n, am.as_mut_ref());
        if (st != ControlStatus::kOk) {
            return st;
        }
        f64 tr = 0.0;
        for (usize i = 0; i < n; ++i) {
            tr += am[i * n + i];
        }
        const f64 ck = -tr / as<f64>(k);
        coeff[k] = ck;

        for (usize i = 0; i < n * n; ++i) {
            m_cur[i] = am[i];
        }
        for (usize i = 0; i < n; ++i) {
            m_cur[i * n + i] += ck;
        }

        if (k < n) {
            for (usize i = 0; i < n; ++i) {
                f64 s = 0.0;
                for (usize j = 0; j < n; ++j) {
                    s += m_cur[j * n + i] * bv[j];
                }
                mb[i] = s;
            }
            f64 cmb = 0.0;
            for (usize i = 0; i < n; ++i) {
                cmb += c[i] * mb[i];
            }
            num_acc[k] = cmb;
        }
    }

    if (d_scalar != 0.0) {
        num_out = Array<f64>(n + 1, 0.0, alloc);
        for (usize i = 0; i <= n; ++i) {
            num_out[i] = d_scalar * coeff[i];
        }
        for (usize k = 0; k < n; ++k) {
            num_out[k + 1] += num_acc[k];
        }
    } else {
        num_out = keel::move(num_acc);
    }
    den_out = keel::move(coeff);
    return ControlStatus::kOk;
}

ControlStatus minreal_tf(ArrayRef<f64> num, ArrayRef<f64> den, f64 tol, Array<f64>& num_out,
                         Array<f64>& den_out, Allocator* alloc) {
    if (keel::is_nan(tol) || !all_finite(num) || !all_finite(den)) {
        return ControlStatus::kInvalidArgument;
    }
    const ArrayRef<f64> num_s = strip_leading_zeros(num);
    const ArrayRef<f64> den_s = strip_leading_zeros(den);
    if (den_s.empty() || den_s[0] == 0.0) {
        return ControlStatus::kInvalidArgument;
    }
    if (tol <= 0.0) {
        tol = keel::sqrt(Limits<f64>::kEpsilon);
    }

    const f64 num_lead = num_s.empty() ? 0.0 : num_s[0];
    const f64 den_lead = den_s[0];

    ArenaTemp scratch = keel::get_scratch(alloc);
    const Array<c64> z = keel::poly::roots(num_s, scratch.allocator());
    const Array<c64> p = keel::poly::roots(den_s, scratch.allocator());

    Array<u8> zused(z.size(), u8{0}, scratch.allocator());
    Array<c64> rem_p(scratch.allocator());
    Array<c64> rem_z(scratch.allocator());
    rem_p.reserve(p.size());
    rem_z.reserve(z.size());

    for (usize j = 0; j < p.size(); ++j) {
        const c64 pj = p[j];
        usize best = z.size();
        f64 best_dist = Limits<f64>::kMax;
        for (usize i = 0; i < z.size(); ++i) {
            if (zused[i] == 0) {
                const f64 d = keel::abs(z[i] - pj);
                if (d < best_dist) {
                    best_dist = d;
                    best = i;
                }
            }
        }
        if (best < z.size() && best_dist <= tol * keel::max(1.0, keel::abs(pj))) {
            zused[best] = 1;
        } else {
            rem_p.push_back(pj);
        }
    }
    for (usize i = 0; i < z.size(); ++i) {
        if (zused[i] == 0) {
            rem_z.push_back(z[i]);
        }
    }

    num_out = keel::poly::from_roots(rem_z.as_ref(), alloc);
    for (usize i = 0; i < num_out.size(); ++i) {
        num_out[i] *= num_lead;
    }
    den_out = keel::poly::from_roots(rem_p.as_ref(), alloc);
    for (usize i = 0; i < den_out.size(); ++i) {
        den_out[i] *= den_lead;
    }
    return ControlStatus::kOk;
}

} // namespace keel::control
