# keel::control

Zero-STL C++20 control-systems numerical kernels built on [`keel`](https://github.com/keelcpp/keel).

## Features

- **Matrix primitives**: column-major partial-pivot LU solve (`solve_in_place`, `solve_in_place_c64`), matrix multiplication (`matmul`, `matmul_c64`), scaling-and-squaring `[6/6]` Padé matrix exponential (`expm`), Faddeev–LeVerrier characteristic polynomial (`char_poly`), controllability (`ctrb`) and observability (`obsv`) matrices.
- **Lyapunov solvers**: continuous (`lyap`, `lyap_c64`) and discrete (`dlyap`, `dlyap_c64`).
- **Algebraic Riccati solvers**: continuous (`care`) and discrete (`dare`) via Higham scaled Newton matrix sign-function iteration.
- **Pole placement**: SISO Ackermann formula (`acker`, `acker_from_poly`, `acker_from_poly_c64`).

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```
