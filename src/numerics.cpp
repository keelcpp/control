#include "keel/control.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/linalg.h"
#include "keel/math.h"

namespace keel::control {

namespace {

inline bool is_finite_val(f64 v) noexcept { return keel::is_finite(v); }
inline bool is_finite_val(c64 v) noexcept {
    return keel::is_finite(v.real()) && keel::is_finite(v.imag());
}

template <typename T>
ControlStatus solve_in_place_tpl(MutArrayRef<T> a, MutArrayRef<T> b, usize n, usize nrhs) noexcept {
    if (n == 0) {
        return ControlStatus::kOk;
    }
    if (a.size() / n < n || (nrhs > 0 && b.size() / n < nrhs)) {
        return ControlStatus::kInvalidArgument;
    }
    const usize n2 = n * n;
    f64 max_abs = 0.0;
    for (usize i = 0; i < n2; ++i) {
        if (!is_finite_val(a[i])) {
            return ControlStatus::kSingular;
        }
        const f64 v = keel::abs(a[i]);
        if (v > max_abs) max_abs = v;
    }
    if (max_abs == 0.0) {
        return ControlStatus::kSingular;
    }
    const f64 piv_tol = 1e-14 * max_abs;

    for (usize k = 0; k < n; ++k) {
        usize pk = k;
        f64 best_abs = keel::abs(a[k * n + k]);
        for (usize i = k + 1; i < n; ++i) {
            const f64 v = keel::abs(a[k * n + i]);
            if (v > best_abs) {
                best_abs = v;
                pk = i;
            }
        }
        if (!keel::is_finite(best_abs) || best_abs <= piv_tol) {
            return ControlStatus::kSingular;
        }
        if (pk != k) {
            for (usize j = 0; j < n; ++j) {
                keel::swap(a[j * n + k], a[j * n + pk]);
            }
            for (usize j = 0; j < nrhs; ++j) {
                keel::swap(b[j * n + k], b[j * n + pk]);
            }
        }
        const T diag = a[k * n + k];
        for (usize i = k + 1; i < n; ++i) {
            const T f = a[k * n + i] / diag;
            a[k * n + i] = f;
            for (usize j = k + 1; j < n; ++j) {
                a[j * n + i] -= f * a[j * n + k];
            }
            for (usize j = 0; j < nrhs; ++j) {
                b[j * n + i] -= f * b[j * n + k];
            }
        }
    }
    for (usize j = 0; j < nrhs; ++j) {
        for (usize i = n; i-- > 0;) {
            T s = b[j * n + i];
            for (usize k = i + 1; k < n; ++k) {
                s -= a[k * n + i] * b[j * n + k];
            }
            b[j * n + i] = s / a[i * n + i];
        }
    }
    return ControlStatus::kOk;
}

template <typename T>
ControlStatus matmul_tpl(ArrayRef<T> a, usize ra, usize ca, ArrayRef<T> b, usize cb,
                         MutArrayRef<T> c) noexcept {
    if ((ca > 0 && a.size() / ca < ra) || (cb > 0 && (b.size() / cb < ca || c.size() / cb < ra))) {
        return ControlStatus::kInvalidArgument;
    }
    const usize total = ra * cb;
    for (usize i = 0; i < total; ++i) {
        c[i] = T(0);
    }
    for (usize j = 0; j < cb; ++j) {
        for (usize k = 0; k < ca; ++k) {
            const T bkj = b[j * ca + k];
            for (usize i = 0; i < ra; ++i) {
                c[j * ra + i] += a[k * ra + i] * bkj;
            }
        }
    }
    return ControlStatus::kOk;
}

} // namespace

ControlStatus solve_in_place(MutArrayRef<f64> a, MutArrayRef<f64> b, usize n, usize nrhs) noexcept {
    return solve_in_place_tpl(a, b, n, nrhs);
}

ControlStatus solve_in_place_c64(MutArrayRef<c64> a, MutArrayRef<c64> b, usize n,
                                 usize nrhs) noexcept {
    return solve_in_place_tpl(a, b, n, nrhs);
}

ControlStatus matmul(ArrayRef<f64> a, usize ra, usize ca, ArrayRef<f64> b, usize cb,
                     MutArrayRef<f64> c) noexcept {
    return matmul_tpl(a, ra, ca, b, cb, c);
}

ControlStatus matmul_c64(ArrayRef<c64> a, usize ra, usize ca, ArrayRef<c64> b, usize cb,
                         MutArrayRef<c64> c) noexcept {
    return matmul_tpl(a, ra, ca, b, cb, c);
}

ControlStatus expm(ArrayRef<f64> a_in, usize n, MutArrayRef<f64> out, Allocator* alloc) {
    if (n == 0) {
        return ControlStatus::kOk;
    }
    if (a_in.size() / n < n || out.size() / n < n) {
        return ControlStatus::kInvalidArgument;
    }
    const usize n2 = n * n;
    for (usize i = 0; i < n2; ++i) {
        if (!keel::is_finite(a_in[i])) {
            return ControlStatus::kInvalidArgument;
        }
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> res(n2, 0.0, scratch.allocator());
    if (!keel::linalg::expm(a_in.data(), n, res.data(), scratch.allocator())) {
        return ControlStatus::kSingular;
    }
    for (usize i = 0; i < n2; ++i) {
        out[i] = res[i];
    }
    return ControlStatus::kOk;
}

ControlStatus char_poly(ArrayRef<f64> a, usize n, MutArrayRef<f64> out, Allocator* alloc) {
    if (out.size() < n + 1 || (n > 0 && a.size() / n < n)) {
        return ControlStatus::kInvalidArgument;
    }
    out[0] = 1.0;
    if (n == 0) {
        return ControlStatus::kOk;
    }
    const usize n2 = n * n;
    for (usize i = 0; i < n2; ++i) {
        if (!keel::is_finite(a[i])) {
            return ControlStatus::kInvalidArgument;
        }
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> m(n2, 0.0, scratch.allocator());
    Array<f64> am(n2, 0.0, scratch.allocator());
    for (usize i = 0; i < n; ++i) m[i * n + i] = 1.0;

    for (usize k = 1; k <= n; ++k) {
        (void)matmul(a, n, n, m, n, am);
        f64 tr = 0.0;
        for (usize i = 0; i < n; ++i) tr += am[i * n + i];
        const f64 ck = -tr / as<f64>(k);
        out[k] = ck;
        for (usize i = 0; i < n; ++i) am[i * n + i] += ck;
        keel::swap(m, am);
    }
    return ControlStatus::kOk;
}

ControlStatus ctrb(ArrayRef<f64> a, usize n, ArrayRef<f64> b, usize m, MutArrayRef<f64> out,
                   Allocator* alloc) {
    if (n == 0 || m == 0) {
        return ControlStatus::kOk;
    }
    if (a.size() / n < n || b.size() / m < n || out.size() / n < n || (out.size() / n) / n < m) {
        return ControlStatus::kInvalidArgument;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> cur(n * m, scratch.allocator());
    Array<f64> nxt(n * m, scratch.allocator());
    for (usize i = 0; i < n * m; ++i) cur[i] = b[i];

    for (usize k = 0; k < n; ++k) {
        const usize col_off = k * m * n;
        for (usize i = 0; i < n * m; ++i) {
            out[col_off + i] = cur[i];
        }
        if (k + 1 < n) {
            (void)matmul(a, n, n, cur, m, nxt);
            keel::swap(cur, nxt);
        }
    }
    return ControlStatus::kOk;
}

ControlStatus obsv(ArrayRef<f64> a, usize n, ArrayRef<f64> c, usize p, MutArrayRef<f64> out,
                   Allocator* alloc) {
    if (n == 0 || p == 0) {
        return ControlStatus::kOk;
    }
    if (a.size() / n < n || c.size() / n < p || out.size() / n < n || (out.size() / n) / n < p) {
        return ControlStatus::kInvalidArgument;
    }
    const usize out_rows = n * p;
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> cur(p * n, scratch.allocator());
    Array<f64> nxt(p * n, scratch.allocator());
    for (usize i = 0; i < p * n; ++i) cur[i] = c[i];

    for (usize k = 0; k < n; ++k) {
        const usize row_off = k * p;
        for (usize col = 0; col < n; ++col) {
            for (usize r = 0; r < p; ++r) {
                out[col * out_rows + row_off + r] = cur[col * p + r];
            }
        }
        if (k + 1 < n) {
            (void)matmul(cur, p, n, a, n, nxt);
            keel::swap(cur, nxt);
        }
    }
    return ControlStatus::kOk;
}

} // namespace keel::control
