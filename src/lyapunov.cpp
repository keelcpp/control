#include "keel/control.h"

#include "keel/arena.h"
#include "keel/array.h"

namespace keel::control {

ControlStatus lyap(ArrayRef<f64> a, ArrayRef<f64> q, usize n, MutArrayRef<f64> x,
                   Allocator* alloc) {
    const usize n2 = n * n;
    if (a.size() < n2 || q.size() < n2 || x.size() < n2) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0) return ControlStatus::kOk;

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
    const usize n2 = n * n;
    if (a.size() < n2 || q.size() < n2 || x.size() < n2) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0) return ControlStatus::kOk;

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
    const usize n2 = n * n;
    if (a.size() < n2 || q.size() < n2 || x.size() < n2) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0) return ControlStatus::kOk;

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
    const usize n2 = n * n;
    if (a.size() < n2 || q.size() < n2 || x.size() < n2) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0) return ControlStatus::kOk;

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
