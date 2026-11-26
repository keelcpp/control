#include "keel/control.h"

#include "keel/format.h"
#include "keel/math.h"

namespace {

using namespace keel;
namespace ctl = keel::control;

i32 g_failures = 0;

void check_true(bool cond, const char* expr, i32 line) {
    if (!cond) {
        keel::eprintln("FAIL line %d: %s", line, expr);
        ++g_failures;
    }
}

void check_near(f64 actual, f64 expected, f64 tol, const char* expr, i32 line) {
    if (!(keel::abs(actual - expected) <= tol)) {
        keel::eprintln("FAIL line %d: %s (actual=%.15g, expected=%.15g, tol=%.15g)",
                       line, expr, actual, expected, tol);
        ++g_failures;
    }
}

#define CHECK(cond) check_true((cond), #cond, __LINE__)
#define CHECK_NEAR(a, e, t) check_near((a), (e), (t), #a " ~= " #e, __LINE__)

void test_numerics() {
    // expm on rotation matrix [[0, -theta], [theta, 0]] (col-major: [0, theta, -theta, 0])
    const f64 th = 0.7;
    const f64 A[4] = {0.0, th, -th, 0.0};
    f64 E[4]{};
    CHECK(ctl::expm(A, 2, E) == ctl::ControlStatus::kOk);
    CHECK_NEAR(E[0], keel::cos(th), 1e-13);
    CHECK_NEAR(E[1], keel::sin(th), 1e-13);
    CHECK_NEAR(E[2], -keel::sin(th), 1e-13);
    CHECK_NEAR(E[3], keel::cos(th), 1e-13);

    // char_poly of [[1, 2], [3, 4]] -> s^2 - 5s - 2
    const f64 M[4] = {1.0, 3.0, 2.0, 4.0};
    f64 cp[3]{};
    CHECK(ctl::char_poly(M, 2, cp) == ctl::ControlStatus::kOk);
    CHECK_NEAR(cp[0], 1.0, 1e-14);
    CHECK_NEAR(cp[1], -5.0, 1e-14);
    CHECK_NEAR(cp[2], -2.0, 1e-14);

    // ctrb & obsv
    const f64 Ad[4] = {0.0, 0.0, 1.0, 0.0}; // [[0, 1], [0, 0]]
    const f64 Bd[2] = {0.0, 1.0};
    const f64 Cd[2] = {1.0, 0.0};
    f64 Co[4]{}, Ob[4]{};
    CHECK(ctl::ctrb(Ad, 2, Bd, 1, Co) == ctl::ControlStatus::kOk);
    CHECK_NEAR(Co[0], 0.0, 1e-15);
    CHECK_NEAR(Co[1], 1.0, 1e-15);
    CHECK_NEAR(Co[2], 1.0, 1e-15);
    CHECK_NEAR(Co[3], 0.0, 1e-15);

    CHECK(ctl::obsv(Ad, 2, Cd, 1, Ob) == ctl::ControlStatus::kOk);
    CHECK_NEAR(Ob[0], 1.0, 1e-15);
    CHECK_NEAR(Ob[1], 0.0, 1e-15);
    CHECK_NEAR(Ob[2], 0.0, 1e-15);
    CHECK_NEAR(Ob[3], 1.0, 1e-15);
}

void test_lyapunov() {
    // A = [[-1, 2], [0, -3]] (col-major: [-1, 0, 2, -3]), Q = I
    const f64 A[4] = {-1.0, 0.0, 2.0, -3.0};
    const f64 Q[4] = {1.0, 0.0, 0.0, 1.0};
    f64 X[4]{};
    CHECK(ctl::lyap(A, Q, 2, X) == ctl::ControlStatus::kOk);

    // Verify residual A*X + X*A^T + Q == 0
    const f64 At[4] = {-1.0, 2.0, 0.0, -3.0};
    f64 AX[4]{}, XAt[4]{};
    CHECK(ctl::matmul(A, 2, 2, X, 2, AX) == ctl::ControlStatus::kOk);
    CHECK(ctl::matmul(X, 2, 2, At, 2, XAt) == ctl::ControlStatus::kOk);
    for (usize i = 0; i < 4; ++i) {
        CHECK_NEAR(AX[i] + XAt[i] + Q[i], 0.0, 1e-12);
    }

    // Discrete Lyapunov: A = [[0.5, 0.2], [0, 0.3]], Q = I
    const f64 Ad[4] = {0.5, 0.0, 0.2, 0.3};
    const f64 Adt[4] = {0.5, 0.2, 0.0, 0.3};
    f64 Xd[4]{};
    CHECK(ctl::dlyap(Ad, Q, 2, Xd) == ctl::ControlStatus::kOk);
    f64 tmp[4]{}, AdXdAdt[4]{};
    CHECK(ctl::matmul(Ad, 2, 2, Xd, 2, tmp) == ctl::ControlStatus::kOk);
    CHECK(ctl::matmul(tmp, 2, 2, Adt, 2, AdXdAdt) == ctl::ControlStatus::kOk);
    for (usize i = 0; i < 4; ++i) {
        CHECK_NEAR(AdXdAdt[i] - Xd[i] + Q[i], 0.0, 1e-12);
    }
}

void test_riccati() {
    // Double integrator: A = [[0, 1], [0, 0]], B = [0; 1], Q = I, R = 1
    // Analytical CARE solution: X = [[sqrt(3), 1], [1, sqrt(3)]], G = [1, sqrt(3)]
    const f64 A[4] = {0.0, 0.0, 1.0, 0.0};
    const f64 B[2] = {0.0, 1.0};
    const f64 Q[4] = {1.0, 0.0, 0.0, 1.0};
    const f64 R[1] = {1.0};
    f64 X[4]{}, G[2]{};
    CHECK(ctl::care(A, B, Q, R, 2, 1, X, G) == ctl::ControlStatus::kOk);
    const f64 s3 = keel::sqrt(3.0);
    CHECK_NEAR(X[0], s3, 1e-11);
    CHECK_NEAR(X[1], 1.0, 1e-11);
    CHECK_NEAR(X[2], 1.0, 1e-11);
    CHECK_NEAR(X[3], s3, 1e-11);
    CHECK_NEAR(G[0], 1.0, 1e-11);
    CHECK_NEAR(G[1], s3, 1e-11);

    // DARE on scalar system: a=1, b=1, q=1, r=1 -> x^2 - x - 1 = 0 -> x = (1 + sqrt(5))/2
    const f64 ad[1] = {1.0}, bd[1] = {1.0}, qd[1] = {1.0}, rd[1] = {1.0};
    f64 xd[1]{}, gd[1]{};
    CHECK(ctl::dare(ad, bd, qd, rd, 1, 1, xd, gd) == ctl::ControlStatus::kOk);
    const f64 phi = 0.5 * (1.0 + keel::sqrt(5.0));
    CHECK_NEAR(xd[0], phi, 1e-11);
    CHECK_NEAR(gd[0], phi / (1.0 + phi), 1e-11);
}

void test_acker() {
    // Double integrator A = [[0, 1], [0, 0]], B = [0; 1], poles = {-2, -3}
    // Desired char poly: (s+2)(s+3) = s^2 + 5s + 6 -> K = [6, 5]
    const f64 A[4] = {0.0, 0.0, 1.0, 0.0};
    const f64 B[2] = {0.0, 1.0};
    const c64 poles[2] = {c64(-2.0, 0.0), c64(-3.0, 0.0)};
    f64 K[2]{};
    CHECK(ctl::acker(A, B, poles, 2, K) == ctl::ControlStatus::kOk);
    CHECK_NEAR(K[0], 6.0, 1e-12);
    CHECK_NEAR(K[1], 5.0, 1e-12);
}

void test_state_gram_covar() {
    // Complex ctrb & obsv
    const c64 Ac[4] = {c64(0.0, 1.0), c64(0.0, 0.0), c64(1.0, 0.0), c64(0.0, -1.0)};
    const c64 Bc[2] = {c64(0.0, 0.0), c64(1.0, 0.0)};
    c64 Co[4]{};
    CHECK(ctl::ctrb_c64(Ac, 2, Bc, 1, Co) == ctl::ControlStatus::kOk);
    CHECK_NEAR(Co[1].real(), 1.0, 1e-12);
    CHECK_NEAR(Co[2].real(), 1.0, 1e-12);
    CHECK_NEAR(Co[3].imag(), -1.0, 1e-12);

    const c64 Cc[2] = {c64(1.0, 0.0), c64(0.0, 0.0)};
    c64 Ob[4]{};
    CHECK(ctl::obsv_c64(Ac, 2, Cc, 1, Ob) == ctl::ControlStatus::kOk);
    CHECK_NEAR(Ob[0].real(), 1.0, 1e-12);
    CHECK_NEAR(Ob[1].imag(), 1.0, 1e-12);
    CHECK_NEAR(Ob[3].real(), 1.0, 1e-12);

    // 1st-order continuous Gramian: dx/dt = -2x + u, y = 3x => Wc = 1/4, Wo = 9/4
    const f64 A1[1] = {-2.0}, B1[1] = {1.0}, C1[1] = {3.0}, D1[1] = {0.0}, W1[1] = {2.0};
    f64 Wc[1]{}, Wo[1]{};
    CHECK(ctl::gram(A1, B1, 1, 1, ctl::GramType::kControllability, false, Wc) == ctl::ControlStatus::kOk);
    CHECK(ctl::gram(A1, C1, 1, 1, ctl::GramType::kObservability, false, Wo) == ctl::ControlStatus::kOk);
    CHECK_NEAR(Wc[0], 0.25, 1e-12);
    CHECK_NEAR(Wo[0], 2.25, 1e-12);

    // Covariance: Q = W * Wc = 0.5, P = C * Q * C^T = 4.5
    f64 P[1]{}, Q[1]{};
    CHECK(ctl::covar(A1, B1, C1, D1, 1, 1, 1, W1, true, false, P, Q) == ctl::ControlStatus::kOk);
    CHECK_NEAR(Q[0], 0.5, 1e-12);
    CHECK_NEAR(P[0], 4.5, 1e-12);
}

void test_conversion_and_minreal() {
    // G(s) = (2s + 5) / (s^2 + 3s + 2) -> tf2ss -> ss2tf roundtrip
    const f64 num[2] = {2.0, 5.0};
    const f64 den[3] = {1.0, 3.0, 2.0};
    ctl::StateSpace ss;
    CHECK(ctl::tf2ss(num, den, ss) == ctl::ControlStatus::kOk);
    CHECK(ss.n == 2);
    CHECK_NEAR(ss.d, 0.0, 1e-12);

    keel::Array<f64> num2, den2;
    CHECK(ctl::ss2tf(ss.a.as_ref(), ss.b.as_ref(), ss.c.as_ref(), ss.d, ss.n, 1, 0, num2, den2) ==
          ctl::ControlStatus::kOk);
    CHECK(num2.size() == 2 && den2.size() == 3);
    CHECK_NEAR(num2[0], 2.0, 1e-12);
    CHECK_NEAR(num2[1], 5.0, 1e-12);
    CHECK_NEAR(den2[0], 1.0, 1e-12);
    CHECK_NEAR(den2[1], 3.0, 1e-12);
    CHECK_NEAR(den2[2], 2.0, 1e-12);

    // minreal_tf: (s + 1)(s + 2) / ((s + 1)(s + 3)) -> (s + 2) / (s + 3)
    const f64 mn[3] = {1.0, 3.0, 2.0};
    const f64 md[3] = {1.0, 4.0, 3.0};
    keel::Array<f64> rn, rd;
    CHECK(ctl::minreal_tf(mn, md, 0.0, rn, rd) == ctl::ControlStatus::kOk);
    CHECK(rn.size() == 2 && rd.size() == 2);
    CHECK_NEAR(rn[0], 1.0, 1e-10);
    CHECK_NEAR(rn[1], 2.0, 1e-10);
    CHECK_NEAR(rd[0], 1.0, 1e-10);
    CHECK_NEAR(rd[1], 3.0, 1e-10);
}

void test_discretize() {
    // 1st-order G(s) = 1 / (s + 2): ZOH & d2c_zoh roundtrip
    const f64 A[1] = {-2.0}, B[1] = {1.0}, C[1] = {1.0}, D[1] = {0.0};
    const f64 ts = 0.1;
    f64 Ad[1]{}, Bd[1]{}, Cd[1]{}, Dd[1]{};
    CHECK(ctl::c2d_zoh(A, B, 1, 1, ts, Ad, Bd) == ctl::ControlStatus::kOk);
    CHECK_NEAR(Ad[0], keel::exp(-0.2), 1e-12);
    CHECK_NEAR(Bd[0], (1.0 - keel::exp(-0.2)) / 2.0, 1e-12);

    ctl::StateSpace back;
    CHECK(ctl::d2c_zoh(Ad, Bd, C, 0.0, 1, ts, back) == ctl::ControlStatus::kOk);
    CHECK(back.n == 1);
    CHECK_NEAR(back.a[0], -2.0, 1e-11);
    CHECK_NEAR(back.b[0], 1.0, 1e-11);

    // Negative real discrete pole triggers 2n real expansion in d2c_zoh
    const f64 Ad_neg[1] = {-0.5}, Bd_neg[1] = {1.0};
    ctl::StateSpace back2;
    CHECK(ctl::d2c_zoh(Ad_neg, Bd_neg, C, 0.0, 1, ts, back2) == ctl::ControlStatus::kOk);
    CHECK(back2.n == 2);

    // Tustin roundtrip
    CHECK(ctl::c2d_tustin(A, B, C, D, 1, 1, 1, ts, Ad, Bd, Cd, Dd) == ctl::ControlStatus::kOk);
    f64 Ac[1]{}, Bc[1]{}, Cc[1]{}, Dc = 0.0;
    CHECK(ctl::d2c_tustin(Ad, Bd, Cd, Dd[0], 1, ts, Ac, Bc, Cc, Dc) == ctl::ControlStatus::kOk);
    CHECK_NEAR(Ac[0], -2.0, 1e-12);
    CHECK_NEAR(Bc[0], 1.0, 1e-12);
    CHECK_NEAR(Cc[0], 1.0, 1e-12);
    CHECK_NEAR(Dc, 0.0, 1e-12);

    // FOH preserves DC gain = 0.5
    f64 Gmap[2]{};
    CHECK(ctl::c2d_foh(A, B, C, D, 1, 1, 1, ts, Ad, Bd, Cd, Dd, Gmap) == ctl::ControlStatus::kOk);
    const f64 dc_foh = Dd[0] + Cd[0] * Bd[0] / (1.0 - Ad[0]);
    CHECK_NEAR(dc_foh, 0.5, 1e-11);
}

void test_response_and_stepinfo() {
    // 1st-order G(s) = 1 / (s + 1): y(t) = 1 - exp(-t)
    const f64 A[1] = {-1.0}, B[1] = {1.0}, C[1] = {1.0};
    const c64 poles[1] = {c64(-1.0, 0.0)};
    keel::Array<f64> t;
    CHECK(ctl::pick_time_grid(poles, 10.0, t) == ctl::ControlStatus::kOk);
    CHECK(t.size() > 50);

    keel::Array<f64> u(t.size(), 1.0);
    keel::Array<f64> y(t.size(), 0.0);
    CHECK(ctl::simulate_siso(A, B, C, 0.0, 1, 0.0, t.as_ref(), u.as_ref(), {}, y.as_mut_ref()) ==
          ctl::ControlStatus::kOk);
    CHECK_NEAR(y[0], 0.0, 1e-12);
    CHECK_NEAR(y[t.size() - 1], 1.0 - keel::exp(-t[t.size() - 1]), 1e-10);

    ctl::StepInfo info;
    CHECK(ctl::step_info(t.as_ref(), y.as_ref(), info) == ctl::ControlStatus::kOk);
    CHECK_NEAR(info.rise_time, keel::log(9.0), 0.02);
    CHECK_NEAR(info.overshoot, 0.0, 1e-9);
    CHECK(info.settling_time > 3.5 && info.settling_time < 4.2);

    // MIMO FOH ramp input u(t) = t on dx/dt = -x + u: exact y(t) = t - 1 + exp(-t)
    keel::Array<f64> u_ramp(t.size(), 0.0);
    for (usize i = 0; i < t.size(); ++i) u_ramp[i] = t[i];
    const f64 D0[1] = {0.0};
    CHECK(ctl::simulate_mimo(A, B, C, D0, 1, 1, 1, 0.0, t.as_ref(), u_ramp.as_ref(), {},
                             y.as_mut_ref()) == ctl::ControlStatus::kOk);
    const f64 tf = t[t.size() - 1];
    CHECK_NEAR(y[t.size() - 1], tf - 1.0 + keel::exp(-tf), 1e-9);
}

void test_freq_margins_and_hinfnorm() {
    // G(s) = 1 / (s + 1)^3 = 1 / (s^3 + 3s^2 + 3s + 1):
    // Phase crosses -180 deg at w = sqrt(3), where |G(j*sqrt(3))| = 1/8 => Gm = 8.
    const f64 num[1] = {1.0};
    const f64 den[4] = {1.0, 3.0, 3.0, 1.0};

    const c64 h0 = ctl::evalfr_tf(num, den, 0.0, 0.0);
    CHECK_NEAR(h0.real(), 1.0, 1e-12);
    CHECK_NEAR(h0.imag(), 0.0, 1e-12);

    ctl::AllMarginResult am;
    CHECK(ctl::allmargin_tf(num, den, am) == ctl::ControlStatus::kOk);
    CHECK(am.stable);
    CHECK(am.gain_margin.size() == 1);
    CHECK_NEAR(am.gain_margin[0], 8.0, 1e-8);
    CHECK_NEAR(am.gm_frequency[0], keel::sqrt(3.0), 1e-8);

    // Bode + margin_from_bode
    f64 w[200]{}, mag[200]{}, ph[200]{};
    const c64 poles[3] = {c64(-1.0, 0.0), c64(-1.0, 0.0), c64(-1.0, 0.0)};
    CHECK(ctl::pick_freq_grid(poles, {}, 200, 0.0, w) == ctl::ControlStatus::kOk);
    CHECK(ctl::bode_tf(num, den, w, 0.0, mag, ph) == ctl::ControlStatus::kOk);
    ctl::MarginResult mr;
    CHECK(ctl::margin_from_bode(w, mag, ph, mr) == ctl::ControlStatus::kOk);
    CHECK_NEAR(mr.gm, 8.0, 0.15);

    // Damp
    const c64 dp[2] = {c64(-1.0, 2.0), c64(-1.0, -2.0)};
    f64 wn[2]{}, zeta[2]{};
    CHECK(ctl::damp(dp, 0.0, wn, zeta) == ctl::ControlStatus::kOk);
    CHECK_NEAR(wn[0], keel::sqrt(5.0), 1e-12);
    CHECK_NEAR(zeta[0], 1.0 / keel::sqrt(5.0), 1e-12);

    // Root locus for 1/(s+1) with k = {0, 1, 2} -> poles at {-1, -2, -3}
    const f64 n1[1] = {1.0}, d1[2] = {1.0, 1.0}, ks[3] = {0.0, 1.0, 2.0};
    c64 rts[3]{};
    CHECK(ctl::rlocus_tf(n1, d1, ks, 1, rts) == ctl::ControlStatus::kOk);
    CHECK_NEAR(rts[0].real(), -1.0, 1e-10);
    CHECK_NEAR(rts[1].real(), -2.0, 1e-10);
    CHECK_NEAR(rts[2].real(), -3.0, 1e-10);

    // H-infinity norm of resonant 2nd-order system: G(s) = 1 / (s^2 + 2*z*s + 1), z = 0.25
    // Exact ||G||_inf = 1 / (2 * z * sqrt(1 - z^2))
    const f64 z = 0.25;
    const f64 n2[1] = {1.0};
    const f64 d2[3] = {1.0, 2.0 * z, 1.0};
    ctl::StateSpace ss2;
    CHECK(ctl::tf2ss(n2, d2, ss2) == ctl::ControlStatus::kOk);
    const f64 d2_arr[1] = {ss2.d};
    f64 hinf = 0.0;
    CHECK(ctl::hinfnorm(ss2.a.as_ref(), ss2.b.as_ref(), ss2.c.as_ref(), d2_arr, 2, 1, 1, false,
                        hinf) == ctl::ControlStatus::kOk);
    const f64 exact = 1.0 / (2.0 * z * keel::sqrt(1.0 - z * z));
    CHECK_NEAR(hinf, exact, 1e-6);

    // Unstable system -> Inf
    const f64 Au[1] = {1.0}, Bu[1] = {1.0}, Cu[1] = {1.0}, Du[1] = {0.0};
    f64 hinf_u = 0.0;
    CHECK(ctl::hinfnorm(Au, Bu, Cu, Du, 1, 1, 1, false, hinf_u) == ctl::ControlStatus::kOk);
    CHECK(!keel::is_finite(hinf_u));
}

void test_hardening_regressions() {
    const f64 nan_v = Limits<f64>::quiet_nan();
    const f64 inf_v = Limits<f64>::infinity();

    // 1. solve_in_place relative pivot threshold (small well-conditioned matrix & NaN)
    {
        f64 a_small[4] = {1e-16, 0.0, 0.0, 2e-16};
        f64 b_small[2] = {3e-16, 8e-16};
        CHECK(ctl::solve_in_place(a_small, b_small, 2, 1) == ctl::ControlStatus::kOk);
        CHECK_NEAR(b_small[0], 3.0, 1e-12);
        CHECK_NEAR(b_small[1], 4.0, 1e-12);

        f64 a_nan[4] = {1.0, 0.0, 0.0, nan_v};
        f64 b_nan[2] = {1.0, 1.0};
        CHECK(ctl::solve_in_place(a_nan, b_nan, 2, 1) == ctl::ControlStatus::kSingular);
    }

    // 2. expm in-place & NaN guard
    {
        f64 a_ip[4] = {0.0, 0.5, -0.5, 0.0};
        CHECK(ctl::expm(a_ip, 2, a_ip) == ctl::ControlStatus::kOk);
        CHECK_NEAR(a_ip[0], keel::cos(0.5), 1e-14);
        CHECK_NEAR(a_ip[1], keel::sin(0.5), 1e-14);

        const f64 a_nan[4] = {0.0, nan_v, 0.0, 0.0};
        f64 out[4]{};
        CHECK(ctl::expm(a_nan, 2, out) == ctl::ControlStatus::kInvalidArgument);
    }

    // 3. DARE on singular A (shift register & zero A) + CARE non-convergent sign guard
    {
        const f64 a_shift[4] = {0.0, 0.0, 1.0, 0.0}; // [[0, 1], [0, 0]]
        const f64 b_shift[2] = {0.0, 1.0};
        const f64 q_eye[4] = {1.0, 0.0, 0.0, 1.0};
        const f64 r_one[1] = {1.0};
        f64 x_out[4]{}, g_out[2]{};
        CHECK(ctl::dare(a_shift, b_shift, q_eye, r_one, 2, 1, x_out, g_out) ==
              ctl::ControlStatus::kOk);
        CHECK_NEAR(x_out[0], 1.0, 1e-11);
        CHECK_NEAR(x_out[1], 0.0, 1e-11);
        CHECK_NEAR(x_out[2], 0.0, 1e-11);
        CHECK_NEAR(x_out[3], 2.0, 1e-11);
        CHECK_NEAR(g_out[0], 0.0, 1e-11);
        CHECK_NEAR(g_out[1], 0.0, 1e-11);

        const f64 a_zero[4] = {0.0, 0.0, 0.0, 0.0};
        CHECK(ctl::dare(a_zero, b_shift, q_eye, r_one, 2, 1, x_out, g_out) ==
              ctl::ControlStatus::kOk);
        CHECK_NEAR(x_out[0], 1.0, 1e-11);
        CHECK_NEAR(x_out[3], 1.0, 1e-11);

        // Imaginary-axis Hamiltonian eigenvalues (oscillator, B=0, Q=0) must fail matsign
        const f64 a_osc[4] = {0.0, -1.0, 1.0, 0.0};
        const f64 b_zero[2] = {0.0, 0.0};
        const f64 q_zero[4] = {0.0, 0.0, 0.0, 0.0};
        CHECK(ctl::care(a_osc, b_zero, q_zero, r_one, 2, 1, x_out) != ctl::ControlStatus::kOk);
    }

    // 4. acker rejects non-conjugate complex poles
    {
        const f64 a[4] = {0.0, 0.0, 1.0, 0.0};
        const f64 b[2] = {0.0, 1.0};
        const c64 bad_poles[2] = {c64(-1.0, 2.0), c64(-3.0, 0.0)};
        f64 k[2]{};
        CHECK(ctl::acker(a, b, bad_poles, 2, k) == ctl::ControlStatus::kInvalidArgument);
    }

    // 5. covar continuous with D != 0 and W == 0 gives finite 0
    {
        const f64 a[1] = {-2.0}, b[1] = {1.0}, c[1] = {1.0}, d[1] = {3.0}, w0[1] = {0.0};
        f64 p[1]{}, q[1]{};
        CHECK(ctl::covar(a, b, c, d, 1, 1, 1, w0, true, false, p, q) == ctl::ControlStatus::kOk);
        CHECK_NEAR(p[0], 0.0, 1e-14);
        CHECK_NEAR(q[0], 0.0, 1e-14);
    }

    // 6. tf2ss leading zeros in numerator & minreal NaN tol
    {
        const f64 num[3] = {0.0, 0.0, 5.0};
        const f64 den[2] = {1.0, 2.0};
        ctl::StateSpace ss;
        CHECK(ctl::tf2ss(num, den, ss) == ctl::ControlStatus::kOk);
        CHECK(ss.n == 1);
        CHECK_NEAR(ss.a[0], -2.0, 1e-14);
        CHECK_NEAR(ss.c[0], 5.0, 1e-14);

        keel::Array<f64> rn, rd;
        CHECK(ctl::minreal_tf(num, den, nan_v, rn, rd) == ctl::ControlStatus::kInvalidArgument);
    }

    // 7. discretize NaN ts, in-place Tustin, singular d2c_zoh
    {
        f64 a[1] = {-2.0}, b[1] = {1.0}, c[1] = {1.0}, d[1] = {0.0};
        CHECK(ctl::c2d_zoh(a, b, 1, 1, nan_v, a, b) == ctl::ControlStatus::kInvalidArgument);
        CHECK(ctl::c2d_foh(a, b, c, d, 1, 1, 1, inf_v, a, b, c, d) ==
              ctl::ControlStatus::kInvalidArgument);
        CHECK(ctl::c2d_tustin(a, b, c, d, 1, 1, 1, 0.1, a, b, c, d) == ctl::ControlStatus::kOk);
        f64 dc = 0.0;
        CHECK(ctl::d2c_tustin(a, b, c, d[0], 1, 0.1, a, b, c, dc) == ctl::ControlStatus::kOk);
        CHECK_NEAR(a[0], -2.0, 1e-12);
        CHECK_NEAR(b[0], 1.0, 1e-12);
        CHECK_NEAR(c[0], 1.0, 1e-12);
        CHECK_NEAR(dc, 0.0, 1e-12);

        const f64 ad0[1] = {0.0}, bd0[1] = {1.0}, cd0[1] = {1.0};
        ctl::StateSpace ss_bad;
        CHECK(ctl::d2c_zoh(ad0, bd0, cd0, 0.0, 1, 0.1, ss_bad) == ctl::ControlStatus::kSingular);
    }

    // 8. simulate_mimo rejects dt == 0 uniform grid; step_info negative yfinal undershoot
    {
        const f64 a[1] = {-1.0}, b[1] = {1.0}, c[1] = {1.0}, d[1] = {0.0};
        const f64 t_bad[2] = {0.0, 0.0};
        const f64 u[2] = {1.0, 1.0};
        f64 y[2]{};
        CHECK(ctl::simulate_mimo(a, b, c, d, 1, 1, 1, 0.0, t_bad, u, {}, y) ==
              ctl::ControlStatus::kInvalidArgument);

        const f64 tn[6] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
        const f64 yn[6] = {0.0, 0.25, -0.5, -1.2, -1.01, -1.0};
        ctl::StepInfo si;
        CHECK(ctl::step_info(tn, yn, si) == ctl::ControlStatus::kOk);
        CHECK_NEAR(si.overshoot, 20.0, 1e-12);
        CHECK_NEAR(si.undershoot, 25.0, 1e-12);
    }

    // 9. damp discrete z == 0; bode_tf pole on jw axis
    {
        const c64 p0[1] = {c64(0.0, 0.0)};
        f64 wn[1]{}, zeta[1]{};
        CHECK(ctl::damp(p0, 0.1, wn, zeta) == ctl::ControlStatus::kOk);
        CHECK(keel::is_inf(wn[0]) && wn[0] > 0.0);
        CHECK_NEAR(zeta[0], 1.0, 1e-14);

        const f64 num[1] = {1.0};
        const f64 den[3] = {1.0, 0.0, 1.0}; // poles at +-j
        const f64 w[3] = {0.5, 1.0, 2.0};
        f64 mag[3]{}, ph[3]{};
        CHECK(ctl::bode_tf(num, den, w, -1.0, mag, ph) == ctl::ControlStatus::kOk);
        CHECK(keel::is_finite(ph[0]) && keel::is_nan(ph[1]) && keel::is_finite(ph[2]));
    }

    // 10. hinfnorm static D = [[1, -1], [-1, 1]] (old power-iter returned 0!) & 6th-order system
    {
        const f64 d_sym[4] = {1.0, -1.0, -1.0, 1.0};
        f64 h = 0.0;
        CHECK(ctl::hinfnorm({}, {}, {}, d_sym, 0, 2, 2, false, h) == ctl::ControlStatus::kOk);
        CHECK_NEAR(h, 2.0, 1e-13);

        // 6th-order cascade (1 / (s^2 + 0.4*s + 1))^3: peak at w = sqrt(1 - 0.08) = sqrt(0.92),
        // |G(j*w)| = (1 / (0.4 * sqrt(0.96)))^3
        const f64 num6[1] = {1.0};
        const f64 den6[7] = {1.0, 1.2, 3.48, 2.464, 3.48, 1.2, 1.0};
        ctl::StateSpace ss6;
        CHECK(ctl::tf2ss(num6, den6, ss6) == ctl::ControlStatus::kOk);
        const f64 d6[1] = {ss6.d};
        f64 h6 = 0.0;
        CHECK(ctl::hinfnorm(ss6.a.as_ref(), ss6.b.as_ref(), ss6.c.as_ref(), d6, 6, 1, 1, false,
                            h6) == ctl::ControlStatus::kOk);
        const f64 peak1 = 1.0 / (0.4 * keel::sqrt(1.0 - 0.04));
        CHECK_NEAR(h6, peak1 * peak1 * peak1, 1e-5);
    }
}

void print_f64_vec(const char* key, ArrayRef<f64> v) {
    keel::print("%s=", key);
    for (usize i = 0; i < v.size(); ++i) {
        if (i > 0) keel::print(",");
        keel::print("%.17g", v[i]);
    }
    keel::print("\n");
}

void print_c64_vec(const char* key, ArrayRef<c64> v) {
    keel::print("%s=", key);
    for (usize i = 0; i < v.size(); ++i) {
        if (i > 0) keel::print(",");
        keel::print("%.17g,%.17g", v[i].real(), v[i].imag());
    }
    keel::print("\n");
}

void dump_reference() {
    // 1. expm, char_poly, ctrb, obsv
    const f64 a3[9] = {-1.0, 2.0, 0.5, 0.3, -2.0, 1.0, -0.4, 0.2, -3.0}; // col-major
    f64 e3[9]{}, cp3[4]{};
    (void)ctl::expm(a3, 3, e3);
    (void)ctl::char_poly(a3, 3, cp3);
    print_f64_vec("EXPM_3", e3);
    print_f64_vec("CHARPOLY_3", cp3);

    const f64 b3[3] = {1.0, -0.5, 2.0};
    const f64 c3[3] = {0.5, 1.5, -1.0};
    f64 co3[9]{}, ob3[9]{};
    (void)ctl::ctrb(a3, 3, b3, 1, co3);
    (void)ctl::obsv(a3, 3, c3, 1, ob3);
    print_f64_vec("CTRB_3", co3);
    print_f64_vec("OBSV_3", ob3);

    // 2. lyap, lyap_c64, dlyap, dlyap_c64
    const f64 q3[9] = {2.0, 0.5, -0.2, 0.5, 3.0, 0.4, -0.2, 0.4, 1.5};
    f64 x_lyap[9]{};
    (void)ctl::lyap(a3, q3, 3, x_lyap);
    print_f64_vec("LYAP_3", x_lyap);

    const f64 ad2[4] = {0.6, -0.2, 0.3, 0.4};
    const f64 q2[4] = {2.0, 0.5, 0.5, 1.5};
    f64 x_dlyap[4]{};
    (void)ctl::dlyap(ad2, q2, 2, x_dlyap);
    print_f64_vec("DLYAP_2", x_dlyap);

    const c64 ac2[4] = {c64(-1.0, 0.5), c64(0.2, -0.3), c64(0.4, 0.1), c64(-2.0, -0.4)};
    const c64 qc2[4] = {c64(2.0, 0.0), c64(0.3, -0.5), c64(0.3, 0.5), c64(3.0, 0.0)};
    c64 xc_lyap[4]{};
    (void)ctl::lyap_c64(ac2, qc2, 2, xc_lyap);
    print_c64_vec("LYAP_C64_2", xc_lyap);

    // 3. gram & covar
    const f64 ag[4] = {-1.0, 0.5, -2.0, -3.0};
    const f64 bg[2] = {1.0, 0.5};
    const f64 cg[2] = {1.0, -0.5};
    const f64 dg[1] = {0.0};
    f64 wc[4]{}, wo[4]{}, wd_c[4]{}, pc[1]{}, qc_out[4]{}, pd[1]{}, qd_out[4]{};
    (void)ctl::gram(ag, bg, 2, 1, ctl::GramType::kControllability, false, wc);
    (void)ctl::gram(ag, cg, 2, 1, ctl::GramType::kObservability, false, wo);
    (void)ctl::gram(ad2, bg, 2, 1, ctl::GramType::kControllability, true, wd_c);
    const f64 w_int[1] = {2.5};
    const f64 dd_nz[1] = {0.4};
    (void)ctl::covar(ag, bg, cg, dg, 2, 1, 1, w_int, true, false, pc, qc_out);
    (void)ctl::covar(ad2, bg, cg, dd_nz, 2, 1, 1, w_int, true, true, pd, qd_out);
    print_f64_vec("GRAM_C", wc);
    print_f64_vec("GRAM_O", wo);
    print_f64_vec("GRAM_D_C", wd_c);
    print_f64_vec("COVAR_C_P", pc);
    print_f64_vec("COVAR_C_Q", qc_out);
    print_f64_vec("COVAR_D_P", pd);
    print_f64_vec("COVAR_D_Q", qd_out);

    // 4. CARE & DARE (including singular A and MIMO)
    const f64 acare[4] = {0.0, -2.0, 1.0, -0.5};
    const f64 bcare[4] = {0.0, 1.0, 1.0, 0.5}; // 2x2
    const f64 rcare[4] = {2.0, 0.3, 0.3, 1.5};
    f64 xcare[4]{}, gcare[4]{};
    (void)ctl::care(acare, bcare, q2, rcare, 2, 2, xcare, gcare);
    print_f64_vec("CARE_MIMO_X", xcare);
    print_f64_vec("CARE_MIMO_G", gcare);

    const f64 adare[4] = {1.0, 0.0, 1.0, 1.0}; // [[1,1],[0,1]]
    const f64 bdare[2] = {0.0, 1.0};
    const f64 rdare[1] = {1.0};
    f64 xdare[4]{}, gdare[2]{};
    (void)ctl::dare(adare, bdare, q2, rdare, 2, 1, xdare, gdare);
    print_f64_vec("DARE_X", xdare);
    print_f64_vec("DARE_G", gdare);

    const f64 asing[4] = {0.0, 0.0, 1.0, 0.0};
    f64 xsing[4]{}, gsing[2]{};
    (void)ctl::dare(asing, bdare, q2, rdare, 2, 1, xsing, gsing);
    print_f64_vec("DARE_SING_X", xsing);
    print_f64_vec("DARE_SING_G", gsing);

    // 5. acker (complex conjugate poles)
    const c64 apoles[3] = {c64(-2.0, 3.0), c64(-2.0, -3.0), c64(-5.0, 0.0)};
    f64 kacker[3]{};
    (void)ctl::acker(a3, b3, apoles, 3, kacker);
    print_f64_vec("ACKER_3", kacker);

    // 6. tf2ss / ss2tf / minreal_tf
    const f64 num_mr[4] = {2.0, 6.0, 14.0, 12.0}; // 2*(s+1)*(s^2+2s+6)
    const f64 den_mr[4] = {1.0, 5.0, 12.0, 18.0}; // (s+3)*(s^2+2s+6) -> cancels complex pair!
    keel::Array<f64> mr_n, mr_d;
    (void)ctl::minreal_tf(num_mr, den_mr, 1e-6, mr_n, mr_d);
    print_f64_vec("MINREAL_N", mr_n.as_ref());
    print_f64_vec("MINREAL_D", mr_d.as_ref());

    // 7. c2d (ZOH, FOH, Tustin) & d2c (ZOH, Tustin)
    const f64 ac_d[4] = {-1.0, 0.0, 2.0, -3.0}; // [[-1,2],[0,-3]]
    const f64 bc_d[2] = {1.0, 1.0};
    const f64 cc_d[2] = {1.0, 0.5};
    const f64 dc_d[1] = {0.0};
    const f64 ts = 0.1;
    f64 az[4]{}, bz[2]{}, af[4]{}, bf[2]{}, cf[2]{}, df[1]{};
    f64 at[4]{}, bt[2]{}, ct[2]{}, dt[1]{};
    (void)ctl::c2d_zoh(ac_d, bc_d, 2, 1, ts, az, bz);
    (void)ctl::c2d_foh(ac_d, bc_d, cc_d, dc_d, 2, 1, 1, ts, af, bf, cf, df);
    (void)ctl::c2d_tustin(ac_d, bc_d, cc_d, dc_d, 2, 1, 1, ts, at, bt, ct, dt);
    print_f64_vec("C2D_ZOH_A", az);
    print_f64_vec("C2D_ZOH_B", bz);
    print_f64_vec("C2D_FOH_A", af);
    print_f64_vec("C2D_FOH_B", bf);
    print_f64_vec("C2D_FOH_C", cf);
    print_f64_vec("C2D_FOH_D", df);
    print_f64_vec("C2D_TUSTIN_A", at);
    print_f64_vec("C2D_TUSTIN_B", bt);
    print_f64_vec("C2D_TUSTIN_C", ct);
    print_f64_vec("C2D_TUSTIN_D", dt);

    ctl::StateSpace d2cz;
    (void)ctl::d2c_zoh(az, bz, cc_d, 0.0, 2, ts, d2cz);
    print_f64_vec("D2C_ZOH_A", d2cz.a.as_ref());
    print_f64_vec("D2C_ZOH_B", d2cz.b.as_ref());

    f64 act[4]{}, bct[2]{}, cct[2]{}, dct = 0.0;
    (void)ctl::d2c_tustin(at, bt, ct, dt[0], 2, ts, act, bct, cct, dct);
    print_f64_vec("D2C_TUSTIN_A", act);
    print_f64_vec("D2C_TUSTIN_B", bct);
    print_f64_vec("D2C_TUSTIN_C", cct);
    const f64 dct_v[1] = {dct};
    print_f64_vec("D2C_TUSTIN_D", dct_v);

    // 8. simulate_siso & step_info (underdamped 2nd-order: 4 / (s^2 + 1.2s + 4))
    const f64 num_s[1] = {4.0};
    const f64 den_s[3] = {1.0, 1.2, 4.0};
    ctl::StateSpace ssu;
    (void)ctl::tf2ss(num_s, den_s, ssu);
    f64 t_sim[201]{}, u_sim[201]{}, y_sim[201]{};
    for (usize i = 0; i < 201; ++i) {
        t_sim[i] = 0.05 * as<f64>(i);
        u_sim[i] = 1.0;
    }
    (void)ctl::simulate_siso(ssu.a.as_ref(), ssu.b.as_ref(), ssu.c.as_ref(), ssu.d, ssu.n, 0.0,
                             t_sim, u_sim, {}, y_sim);
    print_f64_vec("STEP_2ND", y_sim);

    ctl::StepInfo si;
    (void)ctl::step_info(t_sim, y_sim, si);
    const f64 si_v[5] = {si.rise_time, si.settling_time, si.overshoot, si.peak, si.peak_time};
    print_f64_vec("STEPINFO_2ND", si_v);

    // 9. damp, bode_tf, nyquist_tf, rlocus_tf, allmargin_tf
    const c64 dpoles_c[2] = {c64(-0.6, 1.9078784028338913), c64(-2.0, 0.0)};
    const c64 dpoles_d[2] = {c64(0.8, 0.3), c64(0.0, 0.0)};
    f64 wnc[2]{}, zc[2]{}, wnd[2]{}, zd[2]{};
    (void)ctl::damp(dpoles_c, 0.0, wnc, zc);
    (void)ctl::damp(dpoles_d, 0.1, wnd, zd);
    print_f64_vec("DAMP_C_WN", wnc);
    print_f64_vec("DAMP_C_Z", zc);
    print_f64_vec("DAMP_D_WN", wnd);
    print_f64_vec("DAMP_D_Z", zd);

    const f64 nb[2] = {2.0, 1.0};
    const f64 db[4] = {1.0, 3.0, 3.0, 1.0};
    const f64 wb[5] = {0.1, 0.5, 1.0, 2.0, 10.0};
    f64 bmag[5]{}, bph[5]{}, nre[5]{}, nim[5]{};
    (void)ctl::bode_tf(nb, db, wb, 0.0, bmag, bph);
    (void)ctl::nyquist_tf(nb, db, wb, 0.0, nre, nim);
    print_f64_vec("BODE_MAG", bmag);
    print_f64_vec("BODE_PH", bph);
    print_f64_vec("NYQ_RE", nre);
    print_f64_vec("NYQ_IM", nim);

    const f64 nam[1] = {5.0};
    const f64 dam[4] = {1.0, 3.0, 3.0, 1.0};
    ctl::AllMarginResult am;
    (void)ctl::allmargin_tf(nam, dam, am);
    const f64 am_v[5] = {
        am.gain_margin.empty() ? 0.0 : am.gain_margin[0],
        am.gm_frequency.empty() ? 0.0 : am.gm_frequency[0],
        am.phase_margin.empty() ? 0.0 : am.phase_margin[0],
        am.pm_frequency.empty() ? 0.0 : am.pm_frequency[0],
        am.delay_margin.empty() ? 0.0 : am.delay_margin[0],
    };
    print_f64_vec("ALLMARGIN", am_v);

    // 10. hinfnorm (continuous SISO, MIMO 2x2 with D, discrete)
    f64 h_siso = 0.0, h_mimo = 0.0, h_disc = 0.0;
    const f64 d_s0[1] = {ssu.d};
    (void)ctl::hinfnorm(ssu.a.as_ref(), ssu.b.as_ref(), ssu.c.as_ref(), d_s0, ssu.n, 1, 1, false,
                        h_siso);
    const f64 am2[4] = {-1.0, -2.0, 0.5, -3.0};
    const f64 bm2[4] = {1.0, 0.0, 0.5, 1.0};
    const f64 cm2[4] = {1.0, 0.5, 0.0, 1.0};
    const f64 dm2[4] = {0.2, -0.1, 0.3, 0.4};
    (void)ctl::hinfnorm(am2, bm2, cm2, dm2, 2, 2, 2, false, h_mimo);
    (void)ctl::hinfnorm(ad2, bg, cg, dd_nz, 2, 1, 1, true, h_disc);
    const f64 hinf_v[3] = {h_siso, h_mimo, h_disc};
    print_f64_vec("HINFNORM", hinf_v);
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && keel::StringRef(argv[1]) == "--dump-ref") {
        dump_reference();
        return 0;
    }

    test_numerics();
    test_lyapunov();
    test_riccati();
    test_acker();
    test_state_gram_covar();
    test_conversion_and_minreal();
    test_discretize();
    test_response_and_stepinfo();
    test_freq_margins_and_hinfnorm();
    test_hardening_regressions();

    if (g_failures == 0) {
        keel::println("ALL keel::control TESTS PASSED");
        return 0;
    }
    keel::eprintln("TOTAL FAILURES: %d", g_failures);
    return 1;
}
