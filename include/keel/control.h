#pragma once

#include "keel/array.h"
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

enum class GramType : u8 {
    kControllability = 0,
    kObservability,
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
[[nodiscard]] ControlStatus ctrb_c64(ArrayRef<c64> a, usize n, ArrayRef<c64> b, usize m,
                                     MutArrayRef<c64> out, Allocator* alloc = nullptr);

// Observability matrix Ob = [C; C*A; ...; C*A^{n-1}] ((n*p) x n, column-major).
[[nodiscard]] ControlStatus obsv(ArrayRef<f64> a, usize n, ArrayRef<f64> c, usize p,
                                 MutArrayRef<f64> out, Allocator* alloc = nullptr);
[[nodiscard]] ControlStatus obsv_c64(ArrayRef<c64> a, usize n, ArrayRef<c64> c, usize p,
                                     MutArrayRef<c64> out, Allocator* alloc = nullptr);

// ============================================================================
// Lyapunov Equations, Gramians & Noise Covariance
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

// Controllability ('c') or observability ('o') Gramian W (n x n, column-major).
// For kControllability, `bc` is B (n x k); for kObservability, `bc` is C (k x n).
[[nodiscard]] ControlStatus gram(ArrayRef<f64> a, ArrayRef<f64> bc, usize n, usize k,
                                 GramType type, bool discrete, MutArrayRef<f64> w_out,
                                 Allocator* alloc = nullptr);

// Output & state covariance (P: p x p, Q: n x n) for white-noise input with intensity W.
// If `w_is_scalar` is true, W = w[0] * I_m; otherwise `w` is m x m column-major.
[[nodiscard]] ControlStatus covar(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<f64> c,
                                  ArrayRef<f64> d, usize n, usize m, usize p, ArrayRef<f64> w,
                                  bool w_is_scalar, bool discrete, MutArrayRef<f64> p_out,
                                  MutArrayRef<f64> q_out, Allocator* alloc = nullptr);

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

// ============================================================================
// Inter-Form Conversions & Minimal Realization
// ============================================================================

// SISO state-space realization in column-major order:
// A: n x n, B: n x 1, C: 1 x n, D: scalar.
struct StateSpace {
    usize n = 0;
    Array<f64> a;
    Array<f64> b;
    Array<f64> c;
    f64 d = 0.0;
};

// Transfer function (num, den, descending powers) -> controllable canonical SISO state-space.
[[nodiscard]] ControlStatus tf2ss(ArrayRef<f64> num, ArrayRef<f64> den, StateSpace& out,
                                  Allocator* alloc = nullptr);

// Single-output state-space (A: n x n, B: n x nu, C: 1 x n, D: 1 x nu) -> transfer function
// for 0-based input index `iu` via Faddeev-LeVerrier.
[[nodiscard]] ControlStatus ss2tf(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<f64> c, f64 d_scalar,
                                  usize n, usize nu, usize iu, Array<f64>& num_out,
                                  Array<f64>& den_out, Allocator* alloc = nullptr);

// Pole/zero cancellation of a SISO transfer function (descending powers) within relative
// tolerance `tol` (if `tol <= 0`, defaults to sqrt(eps)).
[[nodiscard]] ControlStatus minreal_tf(ArrayRef<f64> num, ArrayRef<f64> den, f64 tol,
                                       Array<f64>& num_out, Array<f64>& den_out,
                                       Allocator* alloc = nullptr);

// ============================================================================
// Discretization (c2d / d2c)
// ============================================================================

// Zero-Order Hold discretization of continuous (A: n x n, B: n x m) at sample time `ts > 0`
// via Van Loan augmented matrix exponential.
[[nodiscard]] ControlStatus c2d_zoh(ArrayRef<f64> a, ArrayRef<f64> b, usize n, usize m, f64 ts,
                                    MutArrayRef<f64> ad_out, MutArrayRef<f64> bd_out,
                                    Allocator* alloc = nullptr);

// First-Order Hold (triangle approximation) discretization of continuous (A, B, C, D)
// (A: n x n, B: n x m, C: p x n, D: p x m) at `ts > 0`.
// Optional `g_out` (size n * (n + m), when non-empty) receives the state map G = [I_n, -G2/Ts].
[[nodiscard]] ControlStatus c2d_foh(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<f64> c,
                                    ArrayRef<f64> d, usize n, usize m, usize p, f64 ts,
                                    MutArrayRef<f64> ad_out, MutArrayRef<f64> bd_out,
                                    MutArrayRef<f64> cd_out, MutArrayRef<f64> dd_out,
                                    MutArrayRef<f64> g_out = {}, Allocator* alloc = nullptr);

// Bilinear (Tustin) discretization of continuous (A, B, C, D) (n x n, n x m, p x n, p x m).
[[nodiscard]] ControlStatus c2d_tustin(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<f64> c,
                                       ArrayRef<f64> d, usize n, usize m, usize p, f64 ts,
                                       MutArrayRef<f64> ad_out, MutArrayRef<f64> bd_out,
                                       MutArrayRef<f64> cd_out, MutArrayRef<f64> dd_out,
                                       Allocator* alloc = nullptr);

// Discrete-to-continuous ZOH conversion of SISO discrete (Ad, Bd, Cd, Dd) via matrix logarithm.
// Automatically expands to a 2n real realization when negative real poles yield complex logm.
[[nodiscard]] ControlStatus d2c_zoh(ArrayRef<f64> ad, ArrayRef<f64> bd, ArrayRef<f64> cd, f64 dd,
                                    usize n, f64 ts, StateSpace& out, Allocator* alloc = nullptr);

// Inverse Tustin conversion of SISO discrete (Ad, Bd, Cd, Dd) at `ts > 0`.
[[nodiscard]] ControlStatus d2c_tustin(ArrayRef<f64> ad, ArrayRef<f64> bd, ArrayRef<f64> cd, f64 dd,
                                       usize n, f64 ts, MutArrayRef<f64> ac_out,
                                       MutArrayRef<f64> bc_out, MutArrayRef<f64> cc_out,
                                       f64& dc_out, Allocator* alloc = nullptr);

// ============================================================================
// Time-Domain Simulation & Step Metrics
// ============================================================================

// Pick default simulation time grid from system poles (continuous-time).
// If `t_final_override > 0`, tiles at the default `dt` up to `t_final_override`.
[[nodiscard]] ControlStatus pick_time_grid(ArrayRef<c64> poles, f64 t_final_override,
                                           Array<f64>& t_out, Allocator* alloc = nullptr);

// Simulate SISO state-space (continuous if ts <= 0 via ZOH, discrete if ts > 0).
// `x0` may be empty (defaults to zeros). Optional `x_traj_out` (N x n column-major).
[[nodiscard]] ControlStatus simulate_siso(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<f64> c, f64 d,
                                          usize n, f64 ts, ArrayRef<f64> t, ArrayRef<f64> u,
                                          ArrayRef<f64> x0, MutArrayRef<f64> y_out,
                                          MutArrayRef<f64> x_traj_out = {},
                                          Allocator* alloc = nullptr);

// Simulate MIMO state-space (A: n x n, B: n x nu, C: ny x n, D: ny x nu).
// Continuous (ts <= 0) uses First-Order Hold; discrete (ts > 0) iterates directly.
// `u_colmajor` is N x nu column-major (`u[j * N + k]` = input j at sample k).
// `y_out` is N x ny column-major; optional `x_traj_out` is N x n column-major.
[[nodiscard]] ControlStatus simulate_mimo(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<f64> c,
                                          ArrayRef<f64> d, usize n, usize nu, usize ny, f64 ts,
                                          ArrayRef<f64> t, ArrayRef<f64> u_colmajor,
                                          ArrayRef<f64> x0, MutArrayRef<f64> y_out,
                                          MutArrayRef<f64> x_traj_out = {},
                                          Allocator* alloc = nullptr);

struct StepInfo {
    f64 rise_time = 0.0;
    f64 transient_time = 0.0;
    f64 settling_time = 0.0;
    f64 settling_min = 0.0;
    f64 settling_max = 0.0;
    f64 overshoot = 0.0;
    f64 undershoot = 0.0;
    f64 peak = 0.0;
    f64 peak_time = 0.0;
};

// Compute rise/transient/settling/overshoot/undershoot/peak metrics from step response `(t, y)`.
[[nodiscard]] ControlStatus step_info(ArrayRef<f64> t, ArrayRef<f64> y, StepInfo& out) noexcept;

// ============================================================================
// Frequency-Domain, Damping, Margins & H-Infinity Norm
// ============================================================================

// Natural frequency `wn` and damping ratio `zeta` of poles (continuous if ts <= 0, discrete if ts > 0).
[[nodiscard]] ControlStatus damp(ArrayRef<c64> poles, f64 ts, MutArrayRef<f64> wn_out,
                                 MutArrayRef<f64> zeta_out) noexcept;

// Evaluate SISO transfer function H(s) at s = j*w (ts <= 0) or z = exp(j*w*ts) (ts > 0).
[[nodiscard]] c64 evalfr_tf(ArrayRef<f64> num, ArrayRef<f64> den, f64 w, f64 ts) noexcept;

// Batch frequency response H(w[i]) for a SISO transfer function.
[[nodiscard]] ControlStatus freqresp_tf(ArrayRef<f64> num, ArrayRef<f64> den, ArrayRef<f64> w,
                                        f64 ts, MutArrayRef<c64> h_out) noexcept;

// Default log-spaced frequency grid spanning ~2 decades around pole/zero magnitudes
// (clamped to Nyquist pi/ts when ts > 0).
[[nodiscard]] ControlStatus pick_freq_grid(ArrayRef<c64> poles, ArrayRef<c64> zeros, usize n_pts,
                                           f64 ts, MutArrayRef<f64> w_out) noexcept;

// Bode magnitude (linear) and unwrapped phase (degrees) for a SISO transfer function.
[[nodiscard]] ControlStatus bode_tf(ArrayRef<f64> num, ArrayRef<f64> den, ArrayRef<f64> w, f64 ts,
                                    MutArrayRef<f64> mag_out, MutArrayRef<f64> phase_deg_out,
                                    Allocator* alloc = nullptr);

// Nyquist real and imaginary parts for a SISO transfer function.
[[nodiscard]] ControlStatus nyquist_tf(ArrayRef<f64> num, ArrayRef<f64> den, ArrayRef<f64> w,
                                       f64 ts, MutArrayRef<f64> re_out,
                                       MutArrayRef<f64> im_out) noexcept;

// Evans root locus of `den(s) + k * num(s) = 0` for gains `k` (length K).
// Writes `K x n` column-major roots into `roots_out` where `n = stripped_den.size() - 1`.
[[nodiscard]] ControlStatus rlocus_tf(ArrayRef<f64> num, ArrayRef<f64> den, ArrayRef<f64> gains,
                                      usize n_poles, MutArrayRef<c64> roots_out,
                                      Allocator* alloc = nullptr);

struct MarginResult {
    f64 gm = 0.0;
    f64 pm = 0.0;
    f64 wcg = 0.0;
    f64 wcp = 0.0;
};

// Gain and phase margins from a Bode magnitude/phase-degrees grid.
[[nodiscard]] ControlStatus margin_from_bode(ArrayRef<f64> w, ArrayRef<f64> mag,
                                             ArrayRef<f64> phase_deg, MarginResult& out,
                                             Allocator* alloc = nullptr);

struct AllMarginResult {
    Array<f64> gain_margin;
    Array<f64> gm_frequency;
    Array<f64> phase_margin;
    Array<f64> pm_frequency;
    Array<f64> delay_margin;
    Array<f64> dm_frequency;
    bool stable = true;
};

// Exact bracket-and-bisect gain, phase, and delay margins + closed-loop stability of continuous SISO TF.
[[nodiscard]] ControlStatus allmargin_tf(ArrayRef<f64> num, ArrayRef<f64> den, AllMarginResult& out,
                                         Allocator* alloc = nullptr);

// H-infinity norm ||G||_inf via Bruinsma-Steinbuch Hamiltonian bisection (continuous or discrete).
// A: n x n, B: n x m, C: p x n, D: p x m (column-major).
[[nodiscard]] ControlStatus hinfnorm(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<f64> c,
                                     ArrayRef<f64> d, usize n, usize m, usize p, bool discrete,
                                     f64& norm_out, Allocator* alloc = nullptr);

} // namespace keel::control
