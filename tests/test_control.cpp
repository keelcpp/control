#include "keel/control.h"

#include "keel/math.h"

#include <cstdio>

namespace {

using namespace keel;
namespace ctl = keel::control;

i32 g_failures = 0;

void check_true(bool cond, const char* expr, i32 line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, expr);
        ++g_failures;
    }
}

void check_near(f64 actual, f64 expected, f64 tol, const char* expr, i32 line) {
    if (!(keel::abs(actual - expected) <= tol)) {
        std::fprintf(stderr, "FAIL line %d: %s (actual=%.15g, expected=%.15g, tol=%.3g)\n",
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

} // namespace

int main() {
    test_numerics();
    test_lyapunov();
    test_riccati();
    test_acker();
    test_state_gram_covar();
    test_conversion_and_minreal();
    test_discretize();
    test_response_and_stepinfo();
    test_freq_margins_and_hinfnorm();

    if (g_failures == 0) {
        std::printf("ALL keel::control TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "TOTAL FAILURES: %d\n", g_failures);
    return 1;
}
