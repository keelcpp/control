#include "keel/control.h"

#include "keel/arena.h"
#include "keel/array.h"

namespace keel::control {

ControlStatus acker_from_poly(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<f64> char_poly_coeffs,
                              usize n, MutArrayRef<f64> k_out, Allocator* alloc) {
    if (a.size() < n * n || b.size() < n || char_poly_coeffs.size() < n + 1 || k_out.size() < n) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0) return ControlStatus::kOk;

    ArenaTemp scratch = keel::get_scratch(alloc);
    Allocator* sa = scratch.allocator();

    // Horner evaluation of phi(A) = c[0]*A^n + ... + c[n]*I
    Array<f64> phi(n * n, 0.0, sa);
    Array<f64> tmp(n * n, 0.0, sa);
    for (usize i = 0; i < n; ++i) phi[i * n + i] = char_poly_coeffs[0];
    for (usize k = 1; k <= n; ++k) {
        (void)matmul(phi, n, n, a, n, tmp);
        for (usize i = 0; i < n; ++i) tmp[i * n + i] += char_poly_coeffs[k];
        keel::swap(phi, tmp);
    }

    Array<f64> co(n * n, sa);
    (void)ctrb(a, n, b, 1, co, sa);

    Array<f64> co_t(n * n, sa);
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            co_t[j * n + i] = co[i * n + j];
        }
    }

    Array<f64> e_n(n, 0.0, sa);
    e_n[n - 1] = 1.0;
    const ControlStatus st = solve_in_place(co_t, e_n, n, 1);
    if (st != ControlStatus::kOk) return st;

    for (usize k = 0; k < n; ++k) {
        f64 s = 0.0;
        for (usize j = 0; j < n; ++j) {
            s += e_n[j] * phi[k * n + j];
        }
        k_out[k] = s;
    }
    return ControlStatus::kOk;
}

ControlStatus acker_from_poly_c64(ArrayRef<c64> a, ArrayRef<c64> b, ArrayRef<f64> char_poly_coeffs,
                                  usize n, MutArrayRef<c64> k_out, Allocator* alloc) {
    if (a.size() < n * n || b.size() < n || char_poly_coeffs.size() < n + 1 || k_out.size() < n) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0) return ControlStatus::kOk;

    ArenaTemp scratch = keel::get_scratch(alloc);
    Allocator* sa = scratch.allocator();

    const c64 zero(0.0, 0.0);
    Array<c64> phi(n * n, zero, sa);
    Array<c64> tmp(n * n, zero, sa);
    for (usize i = 0; i < n; ++i) phi[i * n + i] = c64(char_poly_coeffs[0], 0.0);
    for (usize k = 1; k <= n; ++k) {
        (void)matmul_c64(phi, n, n, a, n, tmp);
        for (usize i = 0; i < n; ++i) tmp[i * n + i] += c64(char_poly_coeffs[k], 0.0);
        keel::swap(phi, tmp);
    }

    // Build complex controllability matrix Co = [b, A*b, ..., A^{n-1}*b]
    Array<c64> co(n * n, zero, sa);
    Array<c64> cur(n, sa);
    Array<c64> nxt(n, sa);
    for (usize i = 0; i < n; ++i) cur[i] = b[i];
    for (usize k = 0; k < n; ++k) {
        for (usize i = 0; i < n; ++i) co[k * n + i] = cur[i];
        if (k + 1 < n) {
            (void)matmul_c64(a, n, n, cur, 1, nxt);
            keel::swap(cur, nxt);
        }
    }

    Array<c64> co_t(n * n, sa);
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            co_t[j * n + i] = co[i * n + j];
        }
    }

    Array<c64> e_n(n, zero, sa);
    e_n[n - 1] = c64(1.0, 0.0);
    const ControlStatus st = solve_in_place_c64(co_t, e_n, n, 1);
    if (st != ControlStatus::kOk) return st;

    for (usize k = 0; k < n; ++k) {
        c64 s(0.0, 0.0);
        for (usize j = 0; j < n; ++j) {
            s += e_n[j] * phi[k * n + j];
        }
        k_out[k] = s;
    }
    return ControlStatus::kOk;
}

ControlStatus acker(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<c64> poles, usize n,
                    MutArrayRef<f64> k_out, Allocator* alloc) {
    if (poles.size() < n) {
        return ControlStatus::kInvalidArgument;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    Allocator* sa = scratch.allocator();

    Array<c64> cp(n + 1, c64(0.0, 0.0), sa);
    cp[0] = c64(1.0, 0.0);
    for (usize k = 0; k < n; ++k) {
        const c64 p = poles[k];
        for (usize j = k + 1; j >= 1; --j) {
            cp[j] = cp[j] - p * cp[j - 1];
        }
    }
    Array<f64> cr(n + 1, sa);
    for (usize i = 0; i <= n; ++i) cr[i] = cp[i].real();
    return acker_from_poly(a, b, cr, n, k_out, sa);
}

} // namespace keel::control
