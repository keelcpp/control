# keel::control

Zero-STL C++20 control-systems numerical kernels built on [`keel`](https://github.com/keelcpp/keel).

## Features

- **Matrix primitives**: column-major partial-pivot LU solve (`solve_in_place`, `solve_in_place_c64`), matrix multiplication (`matmul`, `matmul_c64`), scaling-and-squaring `[6/6]` Padé matrix exponential (`expm`), Faddeev–LeVerrier characteristic polynomial (`char_poly`), controllability (`ctrb`, `ctrb_c64`) and observability (`obsv`, `obsv_c64`) matrices.
- **Lyapunov, Gramians & Covariance**: continuous (`lyap`, `lyap_c64`) and discrete (`dlyap`, `dlyap_c64`), controllability/observability Gramians (`gram`), and white-noise state/output covariance (`covar`).
- **Algebraic Riccati solvers**: continuous (`care`) and discrete (`dare`) via Higham scaled Newton matrix sign-function iteration.
- **Pole placement**: SISO Ackermann formula (`acker`, `acker_from_poly`, `acker_from_poly_c64`).
- **Conversions & Minimal Realization**: `tf2ss`, `ss2tf`, and pole/zero cancellation (`minreal_tf`).
- **Discretization**: ZOH (`c2d_zoh`, `d2c_zoh` with complex `logm` 2n real expansion), FOH (`c2d_foh`), and bilinear/Tustin (`c2d_tustin`, `d2c_tustin`).
- **Time-Domain Simulation**: `pick_time_grid`, `simulate_siso`, `simulate_mimo` (FOH continuous / discrete), and `step_info`.
- **Frequency-Domain, Margins & Robust Norms**: `damp`, `evalfr_tf`, `freqresp_tf`, `pick_freq_grid`, `bode_tf`, `nyquist_tf`, `rlocus_tf`, `margin_from_bode`, `allmargin_tf`, and Bruinsma–Steinbuch `hinfnorm`.

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```
