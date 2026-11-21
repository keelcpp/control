#include "keel/control.h"

#include "keel/array.h"
#include "keel/math.h"

namespace keel::control {

ControlStatus ctrb_c64(ArrayRef<c64> a, usize n, ArrayRef<c64> b, usize m, MutArrayRef<c64> out,
                       Allocator* alloc) {
    if (a.size() < n * n || b.size() < n * m || out.size() < n * n * m) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0 || m == 0) {
        return ControlStatus::kOk;
    }
    for (usize j = 0; j < m; ++j) {
        for (usize i = 0; i < n; ++i) {
            out[j * n + i] = b[j * n + i];
        }
    }
    Array<c64> prev(n * m, c64(0.0, 0.0), alloc);
    Array<c64> next(n * m, c64(0.0, 0.0), alloc);
    for (usize i = 0; i < n * m; ++i) {
        prev[i] = b[i];
    }
    for (usize k = 1; k < n; ++k) {
        const ControlStatus st = matmul_c64(a, n, n, prev.as_ref(), m, next.as_mut_ref());
        if (st != ControlStatus::kOk) {
            return st;
        }
        const usize col_off = k * m;
        for (usize j = 0; j < m; ++j) {
            for (usize i = 0; i < n; ++i) {
                out[(col_off + j) * n + i] = next[j * n + i];
            }
        }
        keel::swap(prev, next);
    }
    return ControlStatus::kOk;
}

ControlStatus obsv_c64(ArrayRef<c64> a, usize n, ArrayRef<c64> c, usize p, MutArrayRef<c64> out,
                       Allocator* alloc) {
    if (a.size() < n * n || c.size() < p * n || out.size() < n * p * n) {
        return ControlStatus::kInvalidArgument;
    }
    if (n == 0 || p == 0) {
        return ControlStatus::kOk;
    }
    const usize total_rows = n * p;
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < p; ++i) {
            out[j * total_rows + i] = c[j * p + i];
        }
    }
    Array<c64> prev(p * n, c64(0.0, 0.0), alloc);
    Array<c64> next(p * n, c64(0.0, 0.0), alloc);
    for (usize i = 0; i < p * n; ++i) {
        prev[i] = c[i];
    }
    for (usize k = 1; k < n; ++k) {
        const ControlStatus st = matmul_c64(prev.as_ref(), p, n, a, n, next.as_mut_ref());
        if (st != ControlStatus::kOk) {
            return st;
        }
        const usize row_off = k * p;
        for (usize j = 0; j < n; ++j) {
            for (usize i = 0; i < p; ++i) {
                out[j * total_rows + (row_off + i)] = next[j * p + i];
            }
        }
        keel::swap(prev, next);
    }
    return ControlStatus::kOk;
}

ControlStatus gram(ArrayRef<f64> a, ArrayRef<f64> bc, usize n, usize k, GramType type,
                   bool discrete, MutArrayRef<f64> w_out, Allocator* alloc) {
    if (a.size() < n * n || w_out.size() < n * n) {
        return ControlStatus::kInvalidArgument;
    }
    if (type == GramType::kControllability) {
        if (bc.size() < n * k) {
            return ControlStatus::kInvalidArgument;
        }
    } else {
        if (bc.size() < k * n) {
            return ControlStatus::kInvalidArgument;
        }
    }
    if (n == 0) {
        return ControlStatus::kOk;
    }

    Array<f64> q(n * n, 0.0, alloc);
    if (type == GramType::kControllability) {
        for (usize j = 0; j < n; ++j) {
            for (usize i = 0; i < n; ++i) {
                f64 s = 0.0;
                for (usize l = 0; l < k; ++l) {
                    s += bc[l * n + i] * bc[l * n + j];
                }
                q[j * n + i] = s;
            }
        }
        return discrete ? dlyap(a, q.as_ref(), n, w_out, alloc)
                        : lyap(a, q.as_ref(), n, w_out, alloc);
    }

    // Observability: A^T * Wo + Wo * A + C^T * C = 0 (or discrete analog).
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            f64 s = 0.0;
            for (usize l = 0; l < k; ++l) {
                s += bc[i * k + l] * bc[j * k + l];
            }
            q[j * n + i] = s;
        }
    }
    Array<f64> at(n * n, 0.0, alloc);
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < n; ++i) {
            at[j * n + i] = a[i * n + j];
        }
    }
    return discrete ? dlyap(at.as_ref(), q.as_ref(), n, w_out, alloc)
                    : lyap(at.as_ref(), q.as_ref(), n, w_out, alloc);
}

ControlStatus covar(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<f64> c, ArrayRef<f64> d, usize n,
                    usize m, usize p, ArrayRef<f64> w, bool w_is_scalar, bool discrete,
                    MutArrayRef<f64> p_out, MutArrayRef<f64> q_out, Allocator* alloc) {
    if (a.size() < n * n || b.size() < n * m || c.size() < p * n || d.size() < p * m ||
        p_out.size() < p * p || q_out.size() < n * n) {
        return ControlStatus::kInvalidArgument;
    }
    if (w_is_scalar ? (w.size() < 1) : (w.size() < m * m)) {
        return ControlStatus::kInvalidArgument;
    }

    const f64 wval = w_is_scalar ? w[0] : 0.0;
    auto wij = [&](usize i, usize j) noexcept -> f64 {
        return w_is_scalar ? (i == j ? wval : 0.0) : w[j * m + i];
    };

    if (n > 0) {
        Array<f64> bwbt(n * n, 0.0, alloc);
        for (usize col = 0; col < n; ++col) {
            for (usize row = 0; row < n; ++row) {
                f64 s = 0.0;
                for (usize i = 0; i < m; ++i) {
                    for (usize j = 0; j < m; ++j) {
                        s += b[i * n + row] * wij(i, j) * b[j * n + col];
                    }
                }
                bwbt[col * n + row] = s;
            }
        }
        const ControlStatus st = discrete ? dlyap(a, bwbt.as_ref(), n, q_out, alloc)
                                          : lyap(a, bwbt.as_ref(), n, q_out, alloc);
        if (st != ControlStatus::kOk) {
            return st;
        }
    }

    // P = C * Q * C^T
    for (usize col = 0; col < p; ++col) {
        for (usize row = 0; row < p; ++row) {
            f64 s = 0.0;
            for (usize i = 0; i < n; ++i) {
                for (usize j = 0; j < n; ++j) {
                    s += c[i * p + row] * q_out[j * n + i] * c[j * p + col];
                }
            }
            p_out[col * p + row] = s;
        }
    }

    bool d_nonzero = false;
    for (usize i = 0; i < p * m; ++i) {
        if (d[i] != 0.0) {
            d_nonzero = true;
            break;
        }
    }
    if (discrete) {
        for (usize col = 0; col < p; ++col) {
            for (usize row = 0; row < p; ++row) {
                f64 s = 0.0;
                for (usize i = 0; i < m; ++i) {
                    for (usize j = 0; j < m; ++j) {
                        s += d[i * p + row] * wij(i, j) * d[j * p + col];
                    }
                }
                p_out[col * p + row] += s;
            }
        }
    } else if (d_nonzero) {
        const f64 inf = Limits<f64>::infinity();
        for (usize i = 0; i < p * p; ++i) {
            p_out[i] = inf;
        }
    }

    return ControlStatus::kOk;
}

} // namespace keel::control
