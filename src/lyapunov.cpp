#include "keel/control.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::control {

namespace {

inline bool check_lyap_dims(usize n, usize a_sz, usize q_sz, usize x_sz) noexcept {
    if (n == 0) return true;
    if (a_sz / n < n || q_sz / n < n || x_sz / n < n) return false;
    const usize n2 = n * n;
    return n2 <= Limits<usize>::kMax / n2;
}

} // namespace

ControlStatus lyap(ArrayRef<f64> a, ArrayRef<f64> q, usize n, MutArrayRef<f64> x,
                   Allocator* alloc) {
    if (!check_lyap_dims(n, a.size(), q.size(), x.size())) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0) return ControlStatus::kOk;
    const usize n2 = n * n;
    for (usize i = 0; i < n2; ++i) {
        if (!keel::is_finite(a[i]) || !keel::is_finite(q[i])) {
            return ControlStatus::kInvalidArgument;
        }
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> m(n2 * n2, 0.0, scratch.allocator());
    for (usize p = 0; p < n; ++p) {
        for (usize i = 0; i < n; ++i) {
            for (usize j = 0; j < n; ++j) {
                const usize row = i + n * p;
                const usize col = j + n * p;
                m[col * n2 + row] += a[j * n + i];
            }
        }
    }
    for (usize p = 0; p < n; ++p) {
        for (usize q_idx = 0; q_idx < n; ++q_idx) {
            for (usize i = 0; i < n; ++i) {
                const usize row = i + n * p;
                const usize col = i + n * q_idx;
                m[col * n2 + row] += a[q_idx * n + p];
            }
        }
    }

    Array<f64> rhs(n2, scratch.allocator());
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            rhs[i + n * j] = -q[j * n + i];
        }
    }

    const ControlStatus st = solve_in_place(m, rhs, n2, 1);
    if (st != ControlStatus::kOk) return st;

    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            x[j * n + i] = rhs[i + n * j];
        }
    }
    return ControlStatus::kOk;
}

ControlStatus lyap_c64(ArrayRef<c64> a, ArrayRef<c64> q, usize n, MutArrayRef<c64> x,
                       Allocator* alloc) {
    if (!check_lyap_dims(n, a.size(), q.size(), x.size())) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0) return ControlStatus::kOk;
    const usize n2 = n * n;
    for (usize i = 0; i < n2; ++i) {
        if (!keel::is_finite(a[i].real()) || !keel::is_finite(a[i].imag()) ||
            !keel::is_finite(q[i].real()) || !keel::is_finite(q[i].imag())) {
            return ControlStatus::kInvalidArgument;
        }
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<c64> m(n2 * n2, c64(0.0, 0.0), scratch.allocator());
    for (usize p = 0; p < n; ++p) {
        for (usize i = 0; i < n; ++i) {
            for (usize j = 0; j < n; ++j) {
                const usize row = i + n * p;
                const usize col = j + n * p;
                m[col * n2 + row] += a[j * n + i];
            }
        }
    }
    for (usize p = 0; p < n; ++p) {
        for (usize q_idx = 0; q_idx < n; ++q_idx) {
            for (usize i = 0; i < n; ++i) {
                const usize row = i + n * p;
                const usize col = i + n * q_idx;
                m[col * n2 + row] += keel::conj(a[q_idx * n + p]);
            }
        }
    }

    Array<c64> rhs(n2, scratch.allocator());
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            rhs[i + n * j] = -q[j * n + i];
        }
    }

    const ControlStatus st = solve_in_place_c64(m, rhs, n2, 1);
    if (st != ControlStatus::kOk) return st;

    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            x[j * n + i] = rhs[i + n * j];
        }
    }
    return ControlStatus::kOk;
}

ControlStatus dlyap(ArrayRef<f64> a, ArrayRef<f64> q, usize n, MutArrayRef<f64> x,
                    Allocator* alloc) {
    if (!check_lyap_dims(n, a.size(), q.size(), x.size())) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0) return ControlStatus::kOk;
    const usize n2 = n * n;
    for (usize i = 0; i < n2; ++i) {
        if (!keel::is_finite(a[i]) || !keel::is_finite(q[i])) {
            return ControlStatus::kInvalidArgument;
        }
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> m(n2 * n2, 0.0, scratch.allocator());
    for (usize p = 0; p < n; ++p) {
        for (usize q_idx = 0; q_idx < n; ++q_idx) {
            for (usize i = 0; i < n; ++i) {
                for (usize j = 0; j < n; ++j) {
                    const usize row = i + n * p;
                    const usize col = j + n * q_idx;
                    m[col * n2 + row] += a[q_idx * n + p] * a[j * n + i];
                }
            }
        }
    }
    for (usize k = 0; k < n2; ++k) {
        m[k * n2 + k] -= 1.0;
    }

    Array<f64> rhs(n2, scratch.allocator());
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            rhs[i + n * j] = -q[j * n + i];
        }
    }

    const ControlStatus st = solve_in_place(m, rhs, n2, 1);
    if (st != ControlStatus::kOk) return st;

    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            x[j * n + i] = rhs[i + n * j];
        }
    }
    return ControlStatus::kOk;
}

ControlStatus dlyap_c64(ArrayRef<c64> a, ArrayRef<c64> q, usize n, MutArrayRef<c64> x,
                        Allocator* alloc) {
    if (!check_lyap_dims(n, a.size(), q.size(), x.size())) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0) return ControlStatus::kOk;
    const usize n2 = n * n;
    for (usize i = 0; i < n2; ++i) {
        if (!keel::is_finite(a[i].real()) || !keel::is_finite(a[i].imag()) ||
            !keel::is_finite(q[i].real()) || !keel::is_finite(q[i].imag())) {
            return ControlStatus::kInvalidArgument;
        }
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<c64> m(n2 * n2, c64(0.0, 0.0), scratch.allocator());
    for (usize p = 0; p < n; ++p) {
        for (usize q_idx = 0; q_idx < n; ++q_idx) {
            for (usize i = 0; i < n; ++i) {
                for (usize j = 0; j < n; ++j) {
                    const usize row = i + n * p;
                    const usize col = j + n * q_idx;
                    m[col * n2 + row] += keel::conj(a[q_idx * n + p]) * a[j * n + i];
                }
            }
        }
    }
    for (usize k = 0; k < n2; ++k) {
        m[k * n2 + k] -= c64(1.0, 0.0);
    }

    Array<c64> rhs(n2, scratch.allocator());
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            rhs[i + n * j] = -q[j * n + i];
        }
    }

    const ControlStatus st = solve_in_place_c64(m, rhs, n2, 1);
    if (st != ControlStatus::kOk) return st;

    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            x[j * n + i] = rhs[i + n * j];
        }
    }
    return ControlStatus::kOk;
}

} // namespace keel::control
