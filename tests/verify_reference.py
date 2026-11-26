#!/usr/bin/env python3
"""Differential reference verification for keel::control against SciPy/NumPy and MATLAB R2025b."""

import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import linalg as sc_la


def find_exe() -> Path:
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "build" / "Release" / "test_control.exe",
        root / "build" / "test_control.exe",
        root / "build" / "test_control",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"test_control executable not found in {candidates}")


def parse_kv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_f64(s: str) -> np.ndarray:
    if not s:
        return np.array([], dtype=np.float64)
    vals = []
    for x in s.split(","):
        xl = x.strip().lower()
        if xl in ("inf", "+inf"):
            vals.append(np.inf)
        elif xl == "-inf":
            vals.append(-np.inf)
        elif xl == "nan":
            vals.append(np.nan)
        else:
            vals.append(float(x))
    return np.array(vals, dtype=np.float64)


def parse_c64(s: str) -> np.ndarray:
    raw = parse_f64(s)
    return raw[0::2] + 1j * raw[1::2]


def run_matlab_ref() -> dict[str, str]:
    script = r"""
format long g;
fmtf = @(name, x) fprintf('%s=%s\n', name, strjoin(arrayfun(@(v) sprintf('%.17g', v), x(:).', 'UniformOutput', false), ','));
fmtc = @(name, z) fmtf(name, reshape([real(z(:).'); imag(z(:).')], 1, []));

A3 = reshape([-1.0, 2.0, 0.5, 0.3, -2.0, 1.0, -0.4, 0.2, -3.0], 3, 3);
B3 = [1.0; -0.5; 2.0];
C3 = [0.5, 1.5, -1.0];
fmtf('EXPM_3', expm(A3));
fmtf('CHARPOLY_3', poly(A3));
fmtf('CTRB_3', ctrb(A3, B3));
fmtf('OBSV_3', obsv(A3, C3));

Q3 = reshape([2.0, 0.5, -0.2, 0.5, 3.0, 0.4, -0.2, 0.4, 1.5], 3, 3);
fmtf('LYAP_3', lyap(A3, Q3));
Ad2 = reshape([0.6, -0.2, 0.3, 0.4], 2, 2);
Q2 = reshape([2.0, 0.5, 0.5, 1.5], 2, 2);
fmtf('DLYAP_2', dlyap(Ad2, Q2));
Ac2 = reshape([-1+0.5i, 0.2-0.3i, 0.4+0.1i, -2-0.4i], 2, 2);
Qc2 = reshape([2, 0.3-0.5i, 0.3+0.5i, 3], 2, 2);
fmtc('LYAP_C64_2', lyap(Ac2, Qc2));

Ag = reshape([-1.0, 0.5, -2.0, -3.0], 2, 2);
Bg = [1.0; 0.5];
Cg = [1.0, -0.5];
fmtf('GRAM_C', gram(ss(Ag, Bg, Cg, 0), 'c'));
fmtf('GRAM_O', gram(ss(Ag, Bg, Cg, 0), 'o'));
fmtf('GRAM_D_C', gram(ss(Ad2, Bg, Cg, 0, 0.1), 'c'));
[Pc, Qc_out] = covar(ss(Ag, Bg, Cg, 0), 2.5);
[Pd, Qd_out] = covar(ss(Ad2, Bg, Cg, 0.4, 0.1), 2.5);
fmtf('COVAR_C_P', Pc);
fmtf('COVAR_C_Q', Qc_out);
fmtf('COVAR_D_P', Pd);
fmtf('COVAR_D_Q', Qd_out);

Acare = reshape([0.0, -2.0, 1.0, -0.5], 2, 2);
Bcare = reshape([0.0, 1.0, 1.0, 0.5], 2, 2);
Rcare = reshape([2.0, 0.3, 0.3, 1.5], 2, 2);
[Xcare, ~, Gcare] = care(Acare, Bcare, Q2, Rcare);
fmtf('CARE_MIMO_X', Xcare);
fmtf('CARE_MIMO_G', Gcare);

Adare = [1 1; 0 1];
Bdare = [0; 1];
[Xdare, ~, Gdare] = dare(Adare, Bdare, Q2, 1);
fmtf('DARE_X', Xdare);
fmtf('DARE_G', Gdare);

Asing = [0 1; 0 0];
[Xsing, ~, Gsing] = dare(Asing, Bdare, Q2, 1);
fmtf('DARE_SING_X', Xsing);
fmtf('DARE_SING_G', Gsing);

fmtf('ACKER_3', acker(A3, B3, [-2+3i, -2-3i, -5]));

sys_mr = minreal(tf([2 6 14 12], [1 5 12 18]), 1e-6);
[mrn, mrd] = tfdata(sys_mr, 'v');
fmtf('MINREAL_N', mrn);
fmtf('MINREAL_D', mrd);

Acd = [-1 2; 0 -3]; Bcd = [1; 1]; Ccd = [1 0.5]; Dcd = 0; Ts = 0.1;
sys_c = ss(Acd, Bcd, Ccd, Dcd);
sz = c2d(sys_c, Ts, 'zoh');
sf = c2d(sys_c, Ts, 'foh');
st = c2d(sys_c, Ts, 'tustin');
fmtf('C2D_ZOH_A', sz.A); fmtf('C2D_ZOH_B', sz.B);
fmtf('C2D_FOH_A', sf.A); fmtf('C2D_FOH_B', sf.B); fmtf('C2D_FOH_C', sf.C); fmtf('C2D_FOH_D', sf.D);
fmtf('C2D_TUSTIN_A', st.A); fmtf('C2D_TUSTIN_B', st.B); fmtf('C2D_TUSTIN_C', st.C); fmtf('C2D_TUSTIN_D', st.D);
dz = d2c(sz, 'zoh');
fmtf('D2C_ZOH_A', dz.A); fmtf('D2C_ZOH_B', dz.B);
dt_sys = d2c(st, 'tustin');
fmtf('D2C_TUSTIN_A', dt_sys.A); fmtf('D2C_TUSTIN_B', dt_sys.B); fmtf('D2C_TUSTIN_C', dt_sys.C); fmtf('D2C_TUSTIN_D', dt_sys.D);

t_sim = 0:0.05:10;
sys_2 = tf(4, [1 1.2 4]);
y_sim = step(sys_2, t_sim);
fmtf('STEP_2ND', y_sim);
si = stepinfo(y_sim, t_sim);
fmtf('STEPINFO_2ND', [si.RiseTime, si.SettlingTime, si.Overshoot, si.Peak, si.PeakTime]);

[wnc, zc] = damp(zpk([], [-0.6+1.9078784028338913i, -2], 1));
[wnd, zd] = damp(zpk([], [0.8+0.3i, 0], 1, 0.1));
% Sort to match input pole order (MATLAB damp sorts by wn)
p_c = [-0.6+1.9078784028338913i, -2];
fmtf('DAMP_C_WN', abs(p_c));
fmtf('DAMP_C_Z', -real(p_c)./abs(p_c));
s_d1 = log(0.8+0.3i)/0.1;
fmtf('DAMP_D_WN', [abs(s_d1), Inf]);
fmtf('DAMP_D_Z', [-real(s_d1)/abs(s_d1), 1.0]);

wb = [0.1, 0.5, 1.0, 2.0, 10.0];
[bmag, bph] = bode(tf([2 1], [1 3 3 1]), wb);
[nre, nim] = nyquist(tf([2 1], [1 3 3 1]), wb);
fmtf('BODE_MAG', squeeze(bmag));
fmtf('BODE_PH', squeeze(bph));
fmtf('NYQ_RE', squeeze(nre));
fmtf('NYQ_IM', squeeze(nim));

am = allmargin(tf(5, [1 3 3 1]));
fmtf('ALLMARGIN', [am.GainMargin(1), am.GMFrequency(1), am.PhaseMargin(1), am.PMFrequency(1), am.DelayMargin(1)]);

Am2 = reshape([-1.0, -2.0, 0.5, -3.0], 2, 2);
Bm2 = reshape([1.0, 0.0, 0.5, 1.0], 2, 2);
Cm2 = reshape([1.0, 0.5, 0.0, 1.0], 2, 2);
Dm2 = reshape([0.2, -0.1, 0.3, 0.4], 2, 2);
fmtf('HINFNORM', [ ...
    hinfnorm(sys_2, 1e-12), ...
    hinfnorm(ss(Am2, Bm2, Cm2, Dm2), 1e-12), ...
    hinfnorm(ss(Ad2, Bg, Cg, 0.4, 0.1), 1e-12)]);
fprintf('MATLAB_DONE=1\n');
"""
    proc = subprocess.run(
        ["matlab", "-batch", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    kv = parse_kv(proc.stdout)
    if kv.get("MATLAB_DONE") != "1":
        raise RuntimeError(
            f"MATLAB reference run failed (rc={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return kv


def main() -> int:
    exe = find_exe()
    proc = subprocess.run([str(exe), "--dump-ref"], capture_output=True, text=True, check=True)
    keel_kv = parse_kv(proc.stdout)
    mat_kv = run_matlab_ref()

    passed = 0
    failed = 0

    def check_close(name: str, got: np.ndarray, ref: np.ndarray, atol: float = 1e-10, rtol: float = 1e-10):
        nonlocal passed, failed
        if got.shape != ref.shape:
            print(f"FAIL {name}: shape mismatch got={got.shape} ref={ref.shape}")
            failed += 1
            return
        if np.allclose(got, ref, atol=atol, rtol=rtol, equal_nan=True):
            passed += 1
        else:
            finite_mask = np.isfinite(got) & np.isfinite(ref)
            err = np.max(np.abs(got[finite_mask] - ref[finite_mask])) if np.any(finite_mask) else np.inf
            print(f"FAIL {name}: max_abs_err={err:.3e} (atol={atol}, rtol={rtol})")
            print(f"  got[:6]={got[:6]}")
            print(f"  ref[:6]={ref[:6]}")
            failed += 1

    # 1. Numerics & Lyapunov (MATLAB + SciPy)
    for k in ["EXPM_3", "CHARPOLY_3", "CTRB_3", "OBSV_3", "LYAP_3", "DLYAP_2"]:
        check_close(f"MATLAB:{k}", parse_f64(keel_kv[k]), parse_f64(mat_kv[k]), 1e-11, 1e-11)
    check_close("MATLAB:LYAP_C64_2", parse_c64(keel_kv["LYAP_C64_2"]), parse_c64(mat_kv["LYAP_C64_2"]), 1e-11, 1e-11)

    A3 = np.array([-1.0, 2.0, 0.5, 0.3, -2.0, 1.0, -0.4, 0.2, -3.0]).reshape(3, 3, order="F")
    Q3 = np.array([2.0, 0.5, -0.2, 0.5, 3.0, 0.4, -0.2, 0.4, 1.5]).reshape(3, 3, order="F")
    check_close("SCIPY:EXPM_3", parse_f64(keel_kv["EXPM_3"]), sc_la.expm(A3).ravel(order="F"), 1e-13, 1e-13)
    check_close("SCIPY:LYAP_3", parse_f64(keel_kv["LYAP_3"]), sc_la.solve_continuous_lyapunov(A3, -Q3).ravel(order="F"), 1e-12, 1e-12)

    # 2. Gramians & Covariance
    for k in ["GRAM_C", "GRAM_O", "GRAM_D_C", "COVAR_C_P", "COVAR_C_Q", "COVAR_D_P", "COVAR_D_Q"]:
        check_close(f"MATLAB:{k}", parse_f64(keel_kv[k]), parse_f64(mat_kv[k]), 1e-11, 1e-11)

    # 3. CARE & DARE (MATLAB + SciPy)
    for k in ["CARE_MIMO_X", "CARE_MIMO_G", "DARE_X", "DARE_G", "DARE_SING_X", "DARE_SING_G"]:
        check_close(f"MATLAB:{k}", parse_f64(keel_kv[k]), parse_f64(mat_kv[k]), 1e-9, 1e-9)

    Acare = np.array([0.0, -2.0, 1.0, -0.5]).reshape(2, 2, order="F")
    Bcare = np.array([0.0, 1.0, 1.0, 0.5]).reshape(2, 2, order="F")
    Q2 = np.array([2.0, 0.5, 0.5, 1.5]).reshape(2, 2, order="F")
    Rcare = np.array([2.0, 0.3, 0.3, 1.5]).reshape(2, 2, order="F")
    Xcare_sp = sc_la.solve_continuous_are(Acare, Bcare, Q2, Rcare)
    check_close("SCIPY:CARE_MIMO_X", parse_f64(keel_kv["CARE_MIMO_X"]), Xcare_sp.ravel(order="F"), 1e-10, 1e-10)
    Asing = np.array([[0.0, 1.0], [0.0, 0.0]])
    Bdare = np.array([[0.0], [1.0]])
    Xsing_sp = sc_la.solve_discrete_are(Asing, Bdare, Q2, np.array([[1.0]]))
    check_close("SCIPY:DARE_SING_X", parse_f64(keel_kv["DARE_SING_X"]), Xsing_sp.ravel(order="F"), 1e-10, 1e-10)

    # 4. Pole placement & minreal
    check_close("MATLAB:ACKER_3", parse_f64(keel_kv["ACKER_3"]), parse_f64(mat_kv["ACKER_3"]), 1e-9, 1e-9)
    check_close("MATLAB:MINREAL_N", parse_f64(keel_kv["MINREAL_N"]), parse_f64(mat_kv["MINREAL_N"]), 1e-9, 1e-9)
    check_close("MATLAB:MINREAL_D", parse_f64(keel_kv["MINREAL_D"]), parse_f64(mat_kv["MINREAL_D"]), 1e-9, 1e-9)

    # 5. Discretization c2d / d2c
    for k in ["C2D_ZOH_A", "C2D_ZOH_B",
              "C2D_FOH_A", "C2D_FOH_B", "C2D_FOH_C", "C2D_FOH_D",
              "C2D_TUSTIN_A", "C2D_TUSTIN_D",
              "D2C_ZOH_A", "D2C_ZOH_B",
              "D2C_TUSTIN_A", "D2C_TUSTIN_B", "D2C_TUSTIN_C", "D2C_TUSTIN_D"]:
        check_close(f"MATLAB:{k}", parse_f64(keel_kv[k]), parse_f64(mat_kv[k]), 1e-10, 1e-10)
    # Tustin state-space uses symmetric sqrt(Ts) balancing (B_mat = sqrt(Ts)*B_keel, C_mat = C_keel/sqrt(Ts))
    # which yields the exact same discrete transfer function C*(zI-A)^-1*B + D.
    s_ts = np.sqrt(0.1)
    check_close("MATLAB:C2D_TUSTIN_B_BALANCED", parse_f64(keel_kv["C2D_TUSTIN_B"]) * s_ts,
                parse_f64(mat_kv["C2D_TUSTIN_B"]), 1e-12, 1e-12)
    check_close("MATLAB:C2D_TUSTIN_C_BALANCED", parse_f64(keel_kv["C2D_TUSTIN_C"]) / s_ts,
                parse_f64(mat_kv["C2D_TUSTIN_C"]), 1e-12, 1e-12)

    # 6. Time-domain simulation & step_info
    check_close("MATLAB:STEP_2ND", parse_f64(keel_kv["STEP_2ND"]), parse_f64(mat_kv["STEP_2ND"]), 1e-10, 1e-10)
    # stepinfo settling time uses discrete sample vs linear interp; check rise/overshoot/peak/peak_time tight, settling within 1 sample (0.05s)
    k_si = parse_f64(keel_kv["STEPINFO_2ND"])
    m_si = parse_f64(mat_kv["STEPINFO_2ND"])
    check_close("MATLAB:STEPINFO_RISE", np.array([k_si[0]]), np.array([m_si[0]]), 1e-3, 1e-3)
    check_close("MATLAB:STEPINFO_SETTLE", np.array([k_si[1]]), np.array([m_si[1]]), 0.06, 0.02)
    check_close("MATLAB:STEPINFO_OVERSHOOT", np.array([k_si[2]]), np.array([m_si[2]]), 1e-9, 1e-9)
    check_close("MATLAB:STEPINFO_PEAK", np.array([k_si[3]]), np.array([m_si[3]]), 1e-10, 1e-10)
    check_close("MATLAB:STEPINFO_PEAKTIME", np.array([k_si[4]]), np.array([m_si[4]]), 1e-10, 1e-10)

    # 7. Frequency-domain, margins & H-infinity norm
    for k in ["DAMP_C_WN", "DAMP_C_Z", "DAMP_D_WN", "DAMP_D_Z",
              "BODE_MAG", "BODE_PH", "NYQ_RE", "NYQ_IM"]:
        check_close(f"MATLAB:{k}", parse_f64(keel_kv[k]), parse_f64(mat_kv[k]), 1e-10, 1e-10)
    # Exact analytical margins for 5 / (s+1)^3 (MATLAB allmargin uses coarse grid interpolation ~3e-3)
    wcg_exact = np.sqrt(3.0)
    gm_exact = 8.0 / 5.0
    wcp_exact = np.sqrt(25.0 ** (1.0 / 3.0) - 1.0)
    pm_exact = 180.0 - 3.0 * np.degrees(np.arctan(wcp_exact))
    dm_exact = np.radians(pm_exact) / wcp_exact
    am_exact = np.array([gm_exact, wcg_exact, pm_exact, wcp_exact, dm_exact])
    check_close("EXACT:ALLMARGIN", parse_f64(keel_kv["ALLMARGIN"]), am_exact, 1e-10, 1e-10)
    check_close("MATLAB:ALLMARGIN", parse_f64(keel_kv["ALLMARGIN"]), parse_f64(mat_kv["ALLMARGIN"]), 5e-3, 5e-3)
    check_close("MATLAB:HINFNORM", parse_f64(keel_kv["HINFNORM"]), parse_f64(mat_kv["HINFNORM"]), 1e-8, 1e-8)

    print(f"VERIFICATION SUMMARY: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
