#include "keel/control.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::control {

namespace {

template <typename T>
ControlStatus solve_in_place_tpl(MutArrayRef<T> a, MutArrayRef<T> b, usize n, usize nrhs) noexcept {
    if (a.size() < n * n || b.size() < n * nrhs) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ControlStatus::kOk;
    }
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
        if (best_abs < 1e-14) {
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
    if (a.size() < ra * ca || b.size() < ca * cb || c.size() < ra * cb) {
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

[[nodiscard]] f64 mat_inf_norm(ArrayRef<f64> a, usize n) noexcept {
    f64 m = 0.0;
    for (usize i = 0; i < n; ++i) {
        f64 s = 0.0;
        for (usize j = 0; j < n; ++j) {
            s += keel::abs(a[j * n + i]);
        }
        if (s > m) m = s;
    }
    return m;
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
    const usize n2 = n * n;
    if (a_in.size() < n2 || out.size() < n2) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ControlStatus::kOk;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> a(n2, scratch.allocator());
    for (usize i = 0; i < n2; ++i) a[i] = a_in[i];

    const f64 norm_a = mat_inf_norm(a, n);
    i32 s = 0;
    if (norm_a > 0.5) {
        const f64 l2 = keel::log2(norm_a / 0.5);
        s = static_cast<i32>(keel::ceil(keel::max(l2, 0.0)));
    }
    if (s > 0) {
        const f64 scale = keel::pow(0.5, static_cast<f64>(s));
        for (usize i = 0; i < n2; ++i) a[i] *= scale;
    }

    constexpr f64 c[7] = {
        1.0,
        1.0 / 2.0,
        5.0 / 44.0,
        1.0 / 66.0,
        1.0 / 792.0,
        1.0 / 15840.0,
        1.0 / 665280.0,
    };

    Array<f64> eye(n2, 0.0, scratch.allocator());
    for (usize i = 0; i < n; ++i) eye[i * n + i] = 1.0;

    Array<f64> a2(n2, scratch.allocator());
    Array<f64> a3(n2, scratch.allocator());
    Array<f64> a4(n2, scratch.allocator());
    Array<f64> a5(n2, scratch.allocator());
    Array<f64> a6(n2, scratch.allocator());

    (void)matmul(a, n, n, a, n, a2);
    (void)matmul(a2, n, n, a, n, a3);
    (void)matmul(a2, n, n, a2, n, a4);
    (void)matmul(a4, n, n, a, n, a5);
    (void)matmul(a4, n, n, a2, n, a6);

    Array<f64> num(n2, 0.0, scratch.allocator());
    Array<f64> den(n2, 0.0, scratch.allocator());
    auto axpy = [&](MutArrayRef<f64> dst, ArrayRef<f64> src, f64 alpha) noexcept {
        for (usize i = 0; i < n2; ++i) dst[i] += alpha * src[i];
    };
    axpy(num, eye, c[0]); axpy(den, eye, c[0]);
    axpy(num, a, c[1]);   axpy(den, a, -c[1]);
    axpy(num, a2, c[2]);  axpy(den, a2, c[2]);
    axpy(num, a3, c[3]);  axpy(den, a3, -c[3]);
    axpy(num, a4, c[4]);  axpy(den, a4, c[4]);
    axpy(num, a5, c[5]);  axpy(den, a5, -c[5]);
    axpy(num, a6, c[6]);  axpy(den, a6, c[6]);

    Array<f64> x = keel::move(num);
    if (solve_in_place(den, x, n, n) != ControlStatus::kOk) {
        Array<f64> e(n2, scratch.allocator());
        Array<f64> term(n2, scratch.allocator());
        Array<f64> next_term(n2, scratch.allocator());
        for (usize i = 0; i < n2; ++i) {
            e[i] = eye[i];
            term[i] = eye[i];
        }
        for (i32 k = 1; k < 30; ++k) {
            (void)matmul(term, n, n, a, n, next_term);
            const f64 inv = 1.0 / static_cast<f64>(k);
            for (usize i = 0; i < n2; ++i) {
                term[i] = next_term[i] * inv;
                e[i] += term[i];
            }
        }
        x = keel::move(e);
    }

    Array<f64> sq(n2, scratch.allocator());
    for (i32 k = 0; k < s; ++k) {
        (void)matmul(x, n, n, x, n, sq);
        keel::swap(x, sq);
    }
    for (usize i = 0; i < n2; ++i) out[i] = x[i];
    return ControlStatus::kOk;
}

ControlStatus char_poly(ArrayRef<f64> a, usize n, MutArrayRef<f64> out, Allocator* alloc) {
    const usize n2 = n * n;
    if (a.size() < n2 || out.size() < n + 1) {
        return ControlStatus::kInvalidArgument;
    }
    out[0] = 1.0;
    if (n == 0) {
        return ControlStatus::kOk;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> m(n2, 0.0, scratch.allocator());
    Array<f64> am(n2, 0.0, scratch.allocator());
    for (usize i = 0; i < n; ++i) m[i * n + i] = 1.0;

    for (usize k = 1; k <= n; ++k) {
        (void)matmul(a, n, n, m, n, am);
        f64 tr = 0.0;
        for (usize i = 0; i < n; ++i) tr += am[i * n + i];
        const f64 ck = -tr / static_cast<f64>(k);
        out[k] = ck;
        for (usize i = 0; i < n; ++i) am[i * n + i] += ck;
        keel::swap(m, am);
    }
    return ControlStatus::kOk;
}

ControlStatus ctrb(ArrayRef<f64> a, usize n, ArrayRef<f64> b, usize m, MutArrayRef<f64> out,
                   Allocator* alloc) {
    if (a.size() < n * n || b.size() < n * m || out.size() < n * n * m) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0 || m == 0) {
        return ControlStatus::kOk;
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
    if (a.size() < n * n || c.size() < p * n || out.size() < (n * p) * n) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0 || p == 0) {
        return ControlStatus::kOk;
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
