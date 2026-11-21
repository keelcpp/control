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

} // namespace

int main() {
    test_numerics();
    test_lyapunov();
    test_riccati();
    test_acker();

    if (g_failures == 0) {
        std::printf("ALL keel::control TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "TOTAL FAILURES: %d\n", g_failures);
    return 1;
}
