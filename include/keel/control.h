#pragma once

#include "keel/complex.h"
#include "keel/core.h"
#include "keel/memory.h"
#include "keel/ref.h"

namespace keel::control {

enum class ControlStatus : u8 {
    kOk = 0,
    kInvalidArgument,
    kSingular,
    kSingularR,
    kSingularA,
    kSingularCayley,
    kSignFailed,
    kSingularSubspace,
    kSingularGain,
};

// All matrices below are stored in column-major order: entry (r, c) of an
// R x C matrix sits at index `c * R + r`.

// ============================================================================
// Matrix & State-Space Primitives
// ============================================================================

// Partial-pivot LU solve: overwrites `a` (n x n) and `b` (n x nrhs) with X
// satisfying A * X = B.
[[nodiscard]] ControlStatus solve_in_place(MutArrayRef<f64> a, MutArrayRef<f64> b, usize n,
                                           usize nrhs) noexcept;
[[nodiscard]] ControlStatus solve_in_place_c64(MutArrayRef<c64> a, MutArrayRef<c64> b, usize n,
                                               usize nrhs) noexcept;

// Column-major matrix product C = A * B (A: ra x ca, B: ca x cb, C: ra x cb).
[[nodiscard]] ControlStatus matmul(ArrayRef<f64> a, usize ra, usize ca, ArrayRef<f64> b, usize cb,
                                   MutArrayRef<f64> c) noexcept;
[[nodiscard]] ControlStatus matmul_c64(ArrayRef<c64> a, usize ra, usize ca, ArrayRef<c64> b,
                                       usize cb, MutArrayRef<c64> c) noexcept;

// Matrix exponential exp(A) for n x n matrix via scaling-and-squaring [6/6] Pade.
[[nodiscard]] ControlStatus expm(ArrayRef<f64> a, usize n, MutArrayRef<f64> out,
                                 Allocator* alloc = nullptr);

// Characteristic polynomial det(s*I - A) = s^n + c[1]*s^{n-1} + ... + c[n]
// via Faddeev-LeVerrier (`out.size() >= n + 1`, `out[0] == 1`).
[[nodiscard]] ControlStatus char_poly(ArrayRef<f64> a, usize n, MutArrayRef<f64> out,
                                      Allocator* alloc = nullptr);

// Controllability matrix Co = [B, A*B, ..., A^{n-1}*B] (n x (n*m), column-major).
[[nodiscard]] ControlStatus ctrb(ArrayRef<f64> a, usize n, ArrayRef<f64> b, usize m,
                                 MutArrayRef<f64> out, Allocator* alloc = nullptr);

// Observability matrix Ob = [C; C*A; ...; C*A^{n-1}] ((n*p) x n, column-major).
[[nodiscard]] ControlStatus obsv(ArrayRef<f64> a, usize n, ArrayRef<f64> c, usize p,
                                 MutArrayRef<f64> out, Allocator* alloc = nullptr);

// ============================================================================
// Lyapunov Equations
// ============================================================================

// Continuous Lyapunov: A * X + X * A^T + Q = 0 (real) or A * X + X * A^H + Q = 0 (complex).
[[nodiscard]] ControlStatus lyap(ArrayRef<f64> a, ArrayRef<f64> q, usize n, MutArrayRef<f64> x,
                                 Allocator* alloc = nullptr);
[[nodiscard]] ControlStatus lyap_c64(ArrayRef<c64> a, ArrayRef<c64> q, usize n, MutArrayRef<c64> x,
                                     Allocator* alloc = nullptr);

// Discrete Lyapunov: A * X * A^T - X + Q = 0 (real) or A * X * A^H - X + Q = 0 (complex).
[[nodiscard]] ControlStatus dlyap(ArrayRef<f64> a, ArrayRef<f64> q, usize n, MutArrayRef<f64> x,
                                  Allocator* alloc = nullptr);
[[nodiscard]] ControlStatus dlyap_c64(ArrayRef<c64> a, ArrayRef<c64> q, usize n, MutArrayRef<c64> x,
                                      Allocator* alloc = nullptr);

// ============================================================================
// Algebraic Riccati Equations (CARE / DARE)
// ============================================================================

// Continuous Algebraic Riccati Equation:
//   A^T * X + X * A - X * B * R^{-1} * B^T * X + Q = 0.
// Writes stabilizing X (n x n) into `x_out` and optional state-feedback gain
// G = R^{-1} * B^T * X (m x n) into `g_out` (when non-empty).
[[nodiscard]] ControlStatus care(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<f64> q, ArrayRef<f64> r,
                                 usize n, usize m, MutArrayRef<f64> x_out,
                                 MutArrayRef<f64> g_out = {}, Allocator* alloc = nullptr);

// Discrete Algebraic Riccati Equation:
//   A^T * X * A - X - A^T * X * B * (R + B^T * X * B)^{-1} * B^T * X * A + Q = 0.
// Writes stabilizing X (n x n) into `x_out` and optional gain
// G = (R + B^T * X * B)^{-1} * (B^T * X * A) (m x n) into `g_out` (when non-empty).
[[nodiscard]] ControlStatus dare(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<f64> q, ArrayRef<f64> r,
                                 usize n, usize m, MutArrayRef<f64> x_out,
                                 MutArrayRef<f64> g_out = {}, Allocator* alloc = nullptr);

// ============================================================================
// Pole Placement (Ackermann SISO)
// ============================================================================

// Ackermann gain K (length n) placing poles of A - B*K at the roots of
// `char_poly_coeffs` (length n + 1, descending powers).
[[nodiscard]] ControlStatus acker_from_poly(ArrayRef<f64> a, ArrayRef<f64> b,
                                            ArrayRef<f64> char_poly_coeffs, usize n,
                                            MutArrayRef<f64> k_out, Allocator* alloc = nullptr);
[[nodiscard]] ControlStatus acker_from_poly_c64(ArrayRef<c64> a, ArrayRef<c64> b,
                                                ArrayRef<f64> char_poly_coeffs, usize n,
                                                MutArrayRef<c64> k_out, Allocator* alloc = nullptr);

// Ackermann gain K (length n) from desired complex poles (length n).
[[nodiscard]] ControlStatus acker(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<c64> poles, usize n,
                                  MutArrayRef<f64> k_out, Allocator* alloc = nullptr);

} // namespace keel::control
