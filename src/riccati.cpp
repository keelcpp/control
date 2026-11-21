#include "keel/control.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::control {

namespace {

void transpose_mat(ArrayRef<f64> a, usize r, usize c, MutArrayRef<f64> t) noexcept {
    for (usize j = 0; j < c; ++j) {
        for (usize i = 0; i < r; ++i) {
            t[i * c + j] = a[j * r + i];
        }
    }
}

bool invert_mat(ArrayRef<f64> a, usize n, MutArrayRef<f64> out, Allocator* scratch_alloc) {
    const usize n2 = n * n;
    Array<f64> lu(n2, scratch_alloc);
    for (usize i = 0; i < n2; ++i) {
        lu[i] = a[i];
        out[i] = 0.0;
    }
    for (usize i = 0; i < n; ++i) {
        out[i * n + i] = 1.0;
    }
    return solve_in_place(lu, out, n, n) == ControlStatus::kOk;
}

[[nodiscard]] f64 frob_norm(ArrayRef<f64> a) noexcept {
    f64 s = 0.0;
    for (usize i = 0; i < a.size(); ++i) {
        s += a[i] * a[i];
    }
    return keel::sqrt(s);
}

bool matsign(MutArrayRef<f64> z, usize k, Allocator* scratch_alloc) {
    const usize k2 = k * k;
    constexpr f64 tol = 1e-14;
    Array<f64> zi(k2, scratch_alloc);
    Array<f64> zn(k2, scratch_alloc);

    for (i32 iter = 0; iter < 200; ++iter) {
        if (!invert_mat(z, k, zi, scratch_alloc)) {
            return false;
        }
        const f64 c = keel::sqrt(frob_norm(zi) / keel::max(frob_norm(z), 1e-300));
        f64 diff = 0.0;
        f64 scale = 0.0;
        for (usize i = 0; i < k2; ++i) {
            const f64 v = 0.5 * (c * z[i] + zi[i] / c);
            zn[i] = v;
            const f64 d = v - z[i];
            diff += d * d;
            scale += v * v;
        }
        for (usize i = 0; i < k2; ++i) z[i] = zn[i];
        if (keel::sqrt(diff) <= tol * keel::sqrt(keel::max(scale, 1e-300))) {
            break;
        }
    }
    return true;
}

bool subspace_to_x(ArrayRef<f64> s, usize n, MutArrayRef<f64> x_out, Allocator* scratch_alloc) {
    const usize k = 2 * n;
    Array<f64> w1(k * n, 0.0, scratch_alloc);
    Array<f64> w2(k * n, 0.0, scratch_alloc);

    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            const f64 z11 = s[j * k + i];
            const f64 z12 = s[(n + j) * k + i];
            const f64 z21 = s[j * k + (n + i)];
            const f64 z22 = s[(n + j) * k + (n + i)];
            w1[j * k + i] = z12;
            w1[j * k + (n + i)] = z22 + (i == j ? 1.0 : 0.0);
            w2[j * k + i] = -(z11 + (i == j ? 1.0 : 0.0));
            w2[j * k + (n + i)] = -z21;
        }
    }

    Array<f64> w1t(n * k, scratch_alloc);
    transpose_mat(w1, k, n, w1t);

    Array<f64> lhs(n * n, scratch_alloc);
    Array<f64> rhs(n * n, scratch_alloc);
    (void)matmul(w1t, n, k, w1, n, lhs);
    (void)matmul(w1t, n, k, w2, n, rhs);
    if (solve_in_place(lhs, rhs, n, n) != ControlStatus::kOk) {
        return false;
    }

    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            x_out[j * n + i] = 0.5 * (rhs[j * n + i] + rhs[i * n + j]);
        }
    }
    return true;
}

} // namespace

ControlStatus care(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<f64> q, ArrayRef<f64> r, usize n,
                   usize m, MutArrayRef<f64> x_out, MutArrayRef<f64> g_out, Allocator* alloc) {
    if (a.size() < n * n || b.size() < n * m || q.size() < n * n || r.size() < m * m ||
        x_out.size() < n * n || (!g_out.empty() && g_out.size() < m * n)) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0) return ControlStatus::kOk;

    ArenaTemp scratch = keel::get_scratch(alloc);
    Allocator* sa = scratch.allocator();

    Array<f64> ri(m * m, sa);
    if (!invert_mat(r, m, ri, sa)) {
        return ControlStatus::kSingularR;
    }

    Array<f64> bt(m * n, sa);
    transpose_mat(b, n, m, bt);

    Array<f64> ri_bt(m * n, sa);
    Array<f64> gbr(n * n, sa);
    (void)matmul(ri, m, m, bt, n, ri_bt);
    (void)matmul(b, n, m, ri_bt, n, gbr);

    const usize k = 2 * n;
    Array<f64> h(k * k, 0.0, sa);
    Array<f64> at(n * n, sa);
    transpose_mat(a, n, n, at);

    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            h[j * k + i] = a[j * n + i];
            h[(n + j) * k + i] = -gbr[j * n + i];
            h[j * k + (n + i)] = -q[j * n + i];
            h[(n + j) * k + (n + i)] = -at[j * n + i];
        }
    }

    if (!matsign(h, k, sa)) {
        return ControlStatus::kSignFailed;
    }
    if (!subspace_to_x(h, n, x_out, sa)) {
        return ControlStatus::kSingularSubspace;
    }

    if (!g_out.empty()) {
        Array<f64> bt_x(m * n, sa);
        (void)matmul(bt, m, n, x_out, n, bt_x);
        (void)matmul(ri, m, m, bt_x, n, g_out);
    }
    return ControlStatus::kOk;
}

ControlStatus dare(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<f64> q, ArrayRef<f64> r, usize n,
                   usize m, MutArrayRef<f64> x_out, MutArrayRef<f64> g_out, Allocator* alloc) {
    if (a.size() < n * n || b.size() < n * m || q.size() < n * n || r.size() < m * m ||
        x_out.size() < n * n || (!g_out.empty() && g_out.size() < m * n)) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0) return ControlStatus::kOk;

    ArenaTemp scratch = keel::get_scratch(alloc);
    Allocator* sa = scratch.allocator();

    Array<f64> ri(m * m, sa);
    if (!invert_mat(r, m, ri, sa)) {
        return ControlStatus::kSingularR;
    }
    Array<f64> ai(n * n, sa);
    if (!invert_mat(a, n, ai, sa)) {
        return ControlStatus::kSingularA;
    }

    Array<f64> bt(m * n, sa);
    transpose_mat(b, n, m, bt);

    Array<f64> ri_bt(m * n, sa);
    Array<f64> gbr(n * n, sa);
    (void)matmul(ri, m, m, bt, n, ri_bt);
    (void)matmul(b, n, m, ri_bt, n, gbr);

    Array<f64> ait(n * n, sa);
    transpose_mat(ai, n, n, ait);

    Array<f64> gbr_ait(n * n, sa);
    Array<f64> ait_q(n * n, sa);
    Array<f64> top(n * n, sa);
    (void)matmul(gbr, n, n, ait, n, gbr_ait);
    (void)matmul(ait, n, n, q, n, ait_q);
    (void)matmul(gbr_ait, n, n, q, n, top);

    const usize k = 2 * n;
    Array<f64> sp(k * k, 0.0, sa);
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            sp[j * k + i] = a[j * n + i] + top[j * n + i];
            sp[(n + j) * k + i] = -gbr_ait[j * n + i];
            sp[j * k + (n + i)] = -ait_q[j * n + i];
            sp[(n + j) * k + (n + i)] = ait[j * n + i];
        }
    }

    Array<f64> smi(k * k, sa);
    Array<f64> spi(k * k, sa);
    for (usize i = 0; i < k * k; ++i) {
        smi[i] = sp[i];
        spi[i] = sp[i];
    }
    for (usize i = 0; i < k; ++i) {
        smi[i * k + i] -= 1.0;
        spi[i * k + i] += 1.0;
    }

    Array<f64> spi_inv(k * k, sa);
    if (!invert_mat(spi, k, spi_inv, sa)) {
        return ControlStatus::kSingularCayley;
    }

    Array<f64> cayley(k * k, sa);
    (void)matmul(smi, k, k, spi_inv, k, cayley);

    if (!matsign(cayley, k, sa)) {
        return ControlStatus::kSignFailed;
    }
    if (!subspace_to_x(cayley, n, x_out, sa)) {
        return ControlStatus::kSingularSubspace;
    }

    if (!g_out.empty()) {
        Array<f64> xb(n * m, sa);
        Array<f64> bt_xb(m * m, sa);
        (void)matmul(x_out, n, n, b, m, xb);
        (void)matmul(bt, m, n, xb, m, bt_xb);

        Array<f64> lhs(m * m, sa);
        for (usize i = 0; i < m * m; ++i) lhs[i] = r[i] + bt_xb[i];

        Array<f64> bt_x(m * n, sa);
        Array<f64> bt_xa(m * n, sa);
        (void)matmul(bt, m, n, x_out, n, bt_x);
        (void)matmul(bt_x, m, n, a, n, bt_xa);

        if (solve_in_place(lhs, bt_xa, m, n) != ControlStatus::kOk) {
            return ControlStatus::kSingularGain;
        }
        for (usize i = 0; i < m * n; ++i) g_out[i] = bt_xa[i];
    }
    return ControlStatus::kOk;
}

} // namespace keel::control
