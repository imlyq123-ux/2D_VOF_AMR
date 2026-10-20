// test_mac.cpp - space-time convergence with and without the MAC predictor.
//
// Translating inviscid Taylor-Green vortex on a periodic [0, 2pi]^2 uniform mesh:
//   u = U0 + sin(x - U0 t) cos(y - V0 t),  v = V0 - cos(x - U0 t) sin(y - V0 t),
// refined at fixed CFL (dt proportional to h), L2 error against the exact solution at T.
// BCG-type schemes have O(dt h) cross terms, so a fixed-mesh dt study is not meaningful;
// at fixed CFL the lagged advection velocity (uf from the previous projection) leaves an
// O(dt) error, while prm.macPredictor (time-centred, MAC-projected advection velocity)
// should converge at second order. With the production slope limiter (clipping at the
// extrema of the vortex) the observed order is ~1.6-1.7; unlimited slopes give 2.0.
// Default meshes 32^2..128^2 (~7 s); --full adds 256^2.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "cases.hpp"
#include "simulation.hpp"

namespace {

constexpr double U0 = 1.0, V0 = 0.5, T = 2.0;

struct Result {
    double err = 0.0, wall = 0.0;
};

Result run(bool predictor, int level, double dt) {
    State s;
    Params& p = s.prm;
    p.nx0 = p.ny0 = 1;
    p.x0 = p.y0 = 0.0;
    p.L0 = 2.0 * M_PI;
    p.minLevel = p.maxLevel = level;
    p.rho1 = p.rho2 = 1.0;
    p.mu1 = p.mu2 = 0.0;
    p.cfl = 10.0;            // the time step is set by dtMax
    p.dtMax = dt;
    p.adaptEvery = 0;
    p.poissonTol = 1e-11;
    p.macPredictor = predictor;
    for (int k = 0; k < 4; ++k) p.bc[k].type = BCType::Periodic;
    cases::CaseSetup cs;
    cs.phi = [](double, double) { return 1.0; };
    cs.u0 = [](double x, double y) { return Vec2(U0 + std::sin(x) * std::cos(y), V0 - std::cos(x) * std::sin(y)); };

    const auto t0 = std::chrono::steady_clock::now();
    sim::initialize(s, cs);
    while (s.t < T - 1e-12) sim::step(s, cs, std::min(dt, T - s.t));
    Result r;
    r.wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const Mesh& m = s.m();
    double e = 0.0, vol = 0.0;
    for (int c : m.leaves()) {
        const Vec2 x = m.center(c);
        const double X = x.x - U0 * T, Y = x.y - V0 * T;
        e += (sq(s.F(s.id.u)[c] - (U0 + std::sin(X) * std::cos(Y))) +
              sq(s.F(s.id.v)[c] - (V0 - std::cos(X) * std::sin(Y)))) * m.volume(c);
        vol += m.volume(c);
    }
    r.err = std::sqrt(e / vol);
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> levels = {5, 6, 7};   // 32^2 .. 128^2
    if (argc > 1 && std::string(argv[1]) == "--full") levels.push_back(8);
    const double cfl = 0.4;              // max |u| ~ 2
    int fails = 0;
    double order[2] = {0, 0}, finest[2] = {0, 0}, wall[2] = {0, 0};
    for (int pred = 0; pred < 2; ++pred) {
        std::printf("%s, CFL %.1f, T = %g\n", pred ? "MAC predictor" : "lagged advection velocity", cfl, T);
        std::printf("    mesh     dt         L2 error     order   wall [s]\n");
        double prev = 0.0;
        for (int L : levels) {
            const int n = 1 << L;
            const double dt = cfl * (2.0 * M_PI / n) / 2.0;
            const Result r = run(pred, L, dt);
            const double o = prev > 0.0 ? std::log2(prev / r.err) : 0.0;
            if (prev > 0.0) order[pred] = o;
            std::printf("    %4d^2   %-10.3e %-12.3e %-7s %.2f\n", n, dt, r.err,
                        prev > 0.0 ? std::to_string(o).substr(0, 4).c_str() : "-", r.wall);
            prev = r.err;
            finest[pred] = r.err;
            wall[pred] = r.wall;
        }
    }
    std::printf("finest mesh: error ratio lagged/predictor %.1f, cost ratio predictor/lagged %.2f\n",
                finest[0] / finest[1], wall[1] / wall[0]);
    if (order[1] < 1.5) { std::printf("FAIL: predictor order %.2f < 1.5\n", order[1]); ++fails; }
    if (order[0] > 1.3) { std::printf("FAIL: lagged order %.2f unexpectedly high (test not discriminating)\n", order[0]); ++fails; }
    if (finest[1] >= finest[0]) { std::printf("FAIL: predictor not more accurate on the finest mesh\n"); ++fails; }
    std::printf(fails ? "MAC TEST FAILED\n" : "MAC TEST PASSED\n");
    return fails != 0;
}
