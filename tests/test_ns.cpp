// test_ns.cpp - Navier-Stokes module: hydrostatics (balanced gravity on AMR), Taylor-Green
// convergence (uniform + AMR), lid-driven cavity Re = 100 (Ghia et al. 1982), static drop
// with exact curvature (balanced surface tension).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "../src/curvature.hpp"
#include "../src/navier_stokes.hpp"

namespace {

int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("  FAIL: %s\n", what);
        ++fails;
    }
}

State makeState(const Params& prm, bool periodic = false) {
    State s;
    s.prm = prm;
    Mesh::Config cfg;
    cfg.nx0 = prm.nx0; cfg.ny0 = prm.ny0;
    cfg.x0 = prm.x0; cfg.y0 = prm.y0; cfg.L0 = prm.L0;
    cfg.minLevel = prm.minLevel; cfg.maxLevel = prm.maxLevel;
    cfg.periodicX = cfg.periodicY = periodic;
    s.mesh = std::make_unique<Mesh>(cfg);
    Mesh& m = s.m();
    s.id.f = m.addField("f");
    s.id.u = m.addField("u");
    s.id.v = m.addField("v");
    s.id.p = m.addField("p");
    s.id.rho = m.addField("rho", &Mesh::prolongInject);
    s.id.mu = m.addField("mu", &Mesh::prolongInject);
    s.id.kappa = m.addField("kappa", &Mesh::prolongInject);
    s.uf.assign(m.nFaces(), 0.0);
    return s;
}

// Area fraction of {phi > 0} in [x0,x0+h]x[y0,y0+h] (phi ~ signed distance), recursive.
double fraction(const std::function<double(double, double)>& phi, double x0, double y0, double h,
                int depth) {
    const double pc = phi(x0 + 0.5 * h, y0 + 0.5 * h);
    if (std::fabs(pc) > 0.75 * h) return pc > 0 ? 1.0 : 0.0;
    if (depth == 0) return std::min(1.0, std::max(0.0, 0.5 + pc / h));
    const double hh = 0.5 * h;
    return 0.25 * (fraction(phi, x0, y0, hh, depth - 1) + fraction(phi, x0 + hh, y0, hh, depth - 1) +
                   fraction(phi, x0, y0 + hh, hh, depth - 1) + fraction(phi, x0 + hh, y0 + hh, hh, depth - 1));
}

void setFraction(State& s, const std::function<double(double, double)>& phi) {
    Mesh& m = s.m();
    for (int c : m.leaves()) {
        const double h = m.size(c);
        const Vec2 x = m.center(c);
        s.F(s.id.f)[c] = fraction(phi, x.x - 0.5 * h, x.y - 0.5 * h, h, 8);
    }
}

void stepNS(State& s, double dt) {
    ns::updateProperties(s);
    ns::advanceMomentum(s, dt);
    const poisson::Result r = ns::project(s, dt);
    if (!r.converged) std::printf("  warning: pressure not converged (%d its, %.2e)\n", r.iterations, r.residual);
    s.t += dt;
    s.dt = dt;
    s.step++;
}

double maxCellVel(const State& s) {
    double a = 0;
    for (int c : s.m().leaves()) a = std::max({a, std::fabs(s.F(s.id.u)[c]), std::fabs(s.F(s.id.v)[c])});
    return a;
}
double maxFaceVel(const State& s) {
    double a = 0;
    for (double w : s.uf) a = std::max(a, std::fabs(w));
    return a;
}
double maxDiv(const State& s) {
    const Mesh& m = s.m();
    double md = 0;
    for (int c : m.leaves()) {
        double d = 0;
        for (const int* p = m.cellFacesBegin(c); p != m.cellFacesEnd(c); ++p) {
            const auto& F = m.faces()[*p];
            d += (F.L == c ? 1 : -1) * s.uf[*p] * F.area;
        }
        md = std::max(md, std::fabs(d) / m.volume(c));
    }
    return md;
}

// ---------------------------------------------------------------- 1. hydrostatics
void testHydrostatic() {
    std::printf("[1] hydrostatics, rho 1000/1, AMR, walls\n");
    Params prm;
    prm.minLevel = 3; prm.maxLevel = 6;
    prm.rho1 = 1000; prm.rho2 = 1; prm.mu1 = 1e-3; prm.mu2 = 1.8e-5;
    prm.gy = -9.81;
    prm.dtMax = 1e-2;
    prm.poissonTol = 1e-10; prm.poissonMaxIter = 5000;
    for (int k = 0; k < 4; ++k) prm.bc[k].type = BCType::Wall;
    State s = makeState(prm);
    const double yI = 0.4321;
    Mesh& m = s.m();
    // interface band + a refined patch in the heavy fluid (coarse/fine faces normal to x)
    m.refineWhere([&](const Mesh& mm, int c) {
        const Vec2 x = mm.center(c);
        return std::fabs(x.y - yI) < 4 * mm.size(c) || std::hypot(x.x - 0.3, x.y - 0.2) < 0.08;
    });
    s.uf.assign(m.nFaces(), 0.0);
    setFraction(s, [&](double, double y) { return yI - y; });
    ns::updateProperties(s);
    double umax = 0, ufmax = 0;
    int its = 0;
    for (int n = 0; n < 20; ++n) {
        stepNS(s, ns::stableDt(s));
        its = std::max(its, s.lastPoissonIters);
        if (n == 9) {  // regrid in the pure fluid + reproject must keep the rest state
            std::vector<int> fl(m.nLeaves(), 0);
            for (int k = 0; k < m.nLeaves(); ++k) {
                const Vec2 x = m.center(m.leaves()[k]);
                if (std::hypot(x.x - 0.7, x.y - 0.8) < 0.1) fl[k] = 1;
            }
            m.adapt(fl);
            ns::updateProperties(s);
            ns::reproject(s);
        }
        umax = std::max(umax, maxCellVel(s));
        ufmax = std::max(ufmax, maxFaceVel(s));
    }
    // hydrostatic pressure difference bottom/top row centre vs analytic
    double pb = 0, pt = 0;
    const int cb = m.leafAt({0.5, 1e-6}), ct = m.leafAt({0.5, 1 - 1e-6});
    pb = s.F(s.id.p)[cb]; pt = s.F(s.id.p)[ct];
    const double yb = m.center(cb).y, yt = m.center(ct).y;
    const double dpExact = 9.81 * (prm.rho1 * (yI - yb) + prm.rho2 * (yt - yI));
    std::printf("  leaves %d, dt %.1e, 20 steps (regrid+reproject at 10): max|u| = %.2e, max|uf| = %.2e, "
                "max Poisson its %d\n", m.nLeaves(), s.dt, umax, ufmax, its);
    std::printf("  p_bottom - p_top = %.6f (analytic %.6f)\n", pb - pt, dpExact);
    check(umax < 1e-10 && ufmax < 1e-10, "hydrostatic state not at rest");
    check(std::fabs(pb - pt - dpExact) < 1e-3 * dpExact, "hydrostatic pressure");
}

// ---------------------------------------------------------------- 2. Taylor-Green
struct TGResult { double errL2, keRatioErr, div; int leaves; };

TGResult runTG(int base, int patchLevel, double dt, double T, double nu = 0.01) {
    const double k = 2 * M_PI;
    Params prm;
    prm.minLevel = base; prm.maxLevel = std::max(base, patchLevel);
    prm.rho1 = prm.rho2 = 1; prm.mu1 = prm.mu2 = nu;
    prm.poissonTol = 1e-10; prm.poissonMaxIter = 5000;
    for (int q = 0; q < 4; ++q) prm.bc[q].type = BCType::Periodic;
    State s = makeState(prm, true);
    Mesh& m = s.m();
    if (patchLevel > base)
        m.refineWhere([](const Mesh& mm, int c) {
            const Vec2 x = mm.center(c);
            return std::hypot(x.x - 0.45, x.y - 0.55) < 0.25;
        });
    s.uf.assign(m.nFaces(), 0.0);
    auto exact = [&](Vec2 x, double t) {
        const double e = std::exp(-2 * nu * k * k * t);
        return Vec2(std::sin(k * x.x) * std::cos(k * x.y) * e, -std::cos(k * x.x) * std::sin(k * x.y) * e);
    };
    for (int c : m.leaves()) {
        s.F(s.id.f)[c] = 1.0;
        const Vec2 u = exact(m.center(c), 0);
        s.F(s.id.u)[c] = u.x;
        s.F(s.id.v)[c] = u.y;
    }
    ns::updateProperties(s);
    ns::reproject(s);
    const int n = (int)std::lround(T / dt);
    for (int i = 0; i < n; ++i) stepNS(s, dt);
    double e2 = 0, n2 = 0, ke = 0, ke0 = 0;
    for (int c : m.leaves()) {
        const Vec2 ue = exact(m.center(c), s.t), u0 = exact(m.center(c), 0);
        const double V = m.volume(c), u = s.F(s.id.u)[c], v = s.F(s.id.v)[c];
        e2 += V * (sq(u - ue.x) + sq(v - ue.y));
        n2 += V * (sq(ue.x) + sq(ue.y));
        ke += V * (u * u + v * v);
        ke0 += V * (sq(u0.x) + sq(u0.y));
    }
    TGResult r;
    r.errL2 = std::sqrt(e2 / n2);
    r.keRatioErr = std::fabs(ke / ke0 - std::exp(-4 * nu * k * k * s.t)) / std::exp(-4 * nu * k * k * s.t);
    r.leaves = m.nLeaves();
    r.div = maxDiv(s);
    return r;
}

void testTaylorGreen() {
    // nu = 0.002 keeps the (first-order, backward Euler) viscous time error below the
    // spatial error at 128^2 with a common dt; KE decays by 8% over T.
    const double dt = 1.0 / 1024, T = 0.25, nu = 0.002;
    std::printf("[2] Taylor-Green vortex, periodic, nu = %g, T = %g, fixed dt = 1/1024\n", nu, T);
    double prev = 0, prevA = 0;
    for (int L = 5; L <= 7; ++L) {
        const TGResult r = runTG(L, L, dt, T, nu);
        std::printf("  uniform %4d^2       : L2 rel err %.3e  KE-decay rel err %.3e  max|div uf| %.1e", 1 << L, r.errL2,
                    r.keRatioErr, r.div);
        if (L > 5) std::printf("  order %.2f", std::log2(prev / r.errL2));
        std::printf("\n");
        if (L > 5) check(std::log2(prev / r.errL2) > 1.8, "TG uniform convergence order");
        if (L == 7) check(r.keRatioErr < 2e-4, "TG kinetic energy decay at 128^2");
        prev = r.errL2;
    }
    for (int L = 5; L <= 7; ++L) {
        const TGResult r = runTG(L, L + 1, dt, T, nu);
        std::printf("  AMR %3d^2+patch %3d : L2 rel err %.3e  KE-decay rel err %.3e  (%d leaves)", 1 << L,
                    2 << L, r.errL2, r.keRatioErr, r.leaves);
        if (L > 5) std::printf("  order %.2f", std::log2(prevA / r.errL2));
        std::printf("\n");
        if (L > 5) check(std::log2(prevA / r.errL2) > 1.8, "TG AMR convergence order");
        prevA = r.errL2;
    }
}

// ---------------------------------------------------------------- 3. lid-driven cavity
void testCavity() {
    std::printf("[3] lid-driven cavity Re = 100, 32^2 + AMR band 64 under the lid\n");
    Params prm;
    prm.minLevel = 5; prm.maxLevel = 6;
    prm.rho1 = prm.rho2 = 1; prm.mu1 = prm.mu2 = 0.01;
    prm.cfl = 0.5; prm.dtMax = 0.01;
    prm.poissonTol = 1e-8;
    for (int k = 0; k < 4; ++k) prm.bc[k].type = BCType::Wall;
    prm.bc[TOP].u = 1.0;
    State s = makeState(prm);
    Mesh& m = s.m();
    m.refineWhere([](const Mesh& mm, int c) { return mm.center(c).y > 0.75; });  // lid region
    s.uf.assign(m.nFaces(), 0.0);
    for (int c : m.leaves()) s.F(s.id.f)[c] = 1.0;
    ns::updateProperties(s);
    std::vector<double> uold;
    double change = 1;
    while (s.t < 30 && change > 1e-4) {
        uold = s.F(s.id.u);
        stepNS(s, ns::stableDt(s));
        change = 0;
        for (int c : m.leaves()) change = std::max(change, std::fabs(s.F(s.id.u)[c] - uold[c]) / s.dt);
    }
    // u on the vertical centreline: average of the two columns adjacent to x = 0.5, linear in y
    auto uAt = [&](double y) {
        double acc = 0;
        for (double x : {0.5 - 1e-9, 0.5 + 1e-9}) {
            const int c = m.leafAt({x, y});
            const double h = m.size(c), yc = m.center(c).y;
            const int c2 = m.leafAt({x, std::min(1 - 1e-9, std::max(1e-9, y + (y > yc ? h : -h)))});
            const double y2 = m.center(c2).y;
            const double u1 = s.F(s.id.u)[c], u2 = s.F(s.id.u)[c2];
            acc += (c2 == c || y2 == yc) ? u1 : u1 + (u2 - u1) * (y - yc) / (y2 - yc);
        }
        return 0.5 * acc;
    };
    const double ghia[][2] = {{0.9766, 0.84123}, {0.9531, 0.68717}, {0.8516, 0.23151}, {0.7344, 0.00332},
                              {0.6172, -0.13641}, {0.5000, -0.20581}, {0.4531, -0.21090},
                              {0.2813, -0.15662}, {0.1719, -0.10150}, {0.1016, -0.06434}, {0.0625, -0.04192}};
    double emax = 0;
    std::printf("  t = %.2f (%d steps, change rate %.1e), leaves %d\n   y      u        Ghia\n", s.t, s.step,
                change, m.nLeaves());
    for (const auto& g : ghia) {
        const double u = uAt(g[0]);
        std::printf("  %.4f %+.5f %+.5f\n", g[0], u, g[1]);
        emax = std::max(emax, std::fabs(u - g[1]));
    }
    std::printf("  max |u - u_Ghia| = %.4f (lid velocity 1)\n", emax);
    check(emax < 0.02, "cavity vs Ghia");
}

// ---------------------------------------------------------------- 4. static drop
void testStaticDrop(double rho1, double rho2, double mu1, double mu2, int steps, bool computed = false) {
    std::printf("[4] static drop, %s curvature, rho %g/%g, mu %g/%g\n", computed ? "computed (height function)" : "exact",
                rho1, rho2, mu1, mu2);
    const double R = 0.2, xc = 0.5, yc = 0.5;
    Params prm;
    prm.minLevel = 3; prm.maxLevel = 6;
    prm.rho1 = rho1; prm.rho2 = rho2; prm.mu1 = mu1; prm.mu2 = mu2;
    prm.sigma = 1.0;
    prm.poissonTol = 1e-10; prm.poissonMaxIter = 5000;
    for (int k = 0; k < 4; ++k) prm.bc[k].type = BCType::Wall;
    State s = makeState(prm);
    Mesh& m = s.m();
    m.refineWhere([&](const Mesh& mm, int c) {
        const Vec2 x = mm.center(c);
        return std::fabs(std::hypot(x.x - xc, x.y - yc) - R) < 4 * mm.size(c);
    });
    s.uf.assign(m.nFaces(), 0.0);
    setFraction(s, [&](double x, double y) { return R - std::hypot(x - xc, y - yc); });
    const double hmin = m.h(m.maxLevel());
    if (computed) {
        curvature::compute(m, s.F(s.id.f), s.F(s.id.kappa));
    } else {
        for (int c : m.leaves()) {
            const Vec2 x = m.center(c);
            s.F(s.id.kappa)[c] = std::fabs(std::hypot(x.x - xc, x.y - yc) - R) < 3 * hmin ? 1.0 / R : 0.0;
        }
    }
    ns::updateProperties(s);
    double umax = 0;
    for (int n = 0; n < steps; ++n) {
        stepNS(s, ns::stableDt(s));
        umax = std::max({umax, maxCellVel(s), maxFaceVel(s)});
    }
    // pressure jump: mean p inside (f = 1) minus outside (f = 0)
    double pin = 0, vin = 0, pout = 0, vout = 0;
    for (int c : m.leaves()) {
        const double f = s.F(s.id.f)[c], V = m.volume(c);
        if (f == 1) { pin += V * s.F(s.id.p)[c]; vin += V; }
        if (f == 0) { pout += V * s.F(s.id.p)[c]; vout += V; }
    }
    const double dp = pin / vin - pout / vout;
    const double muMax = std::max(mu1, mu2);
    std::printf("  leaves %d, %d steps (dt %.2e, t %.3f): max|u| = %.2e (Ca = %.2e), dp = %.6f (sigma/R = %.6f)\n",
                m.nLeaves(), steps, s.dt, s.t, umax, umax * muMax / prm.sigma, dp, 1.0 / R);
    if (computed) {
        check(umax * muMax / prm.sigma < 1e-3, "static drop spurious currents with computed curvature");
        check(std::fabs(dp - 1.0 / R) < 1e-2 / R, "Laplace pressure jump (computed curvature)");
    } else {
        check(umax < 1e-9, "static drop spurious currents with exact curvature");
        check(std::fabs(dp - 1.0 / R) < 1e-9 / R, "Laplace pressure jump");
    }
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string only = argc > 1 ? argv[1] : "";
    auto want = [&](const char* k) { return only.empty() || only == k; };
    if (only == "tgone" && argc >= 6) {  // tgone base patchLevel dt T
        const TGResult r = runTG(std::atoi(argv[2]), std::atoi(argv[3]), std::atof(argv[4]), std::atof(argv[5]),
                                 argc > 6 ? std::atof(argv[6]) : 0.01);
        std::printf("L2 %.4e KE %.4e div %.1e leaves %d\n", r.errL2, r.keRatioErr, r.div, r.leaves);
        return 0;
    }
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    auto lap = [&](const char* n) {
        std::printf("  (%s: %.1f s)\n", n, std::chrono::duration<double>(clk::now() - t0).count());
    };
    if (want("hydro")) { testHydrostatic(); lap("hydrostatic"); }
    if (want("drop")) {
        testStaticDrop(1, 1, 1e-2, 1e-2, 100);
        testStaticDrop(1000, 1, 1e-1, 1e-3, 100);
        // La = rho sigma D / mu^2 = 12000 (Popinet 2009), D/h = 25.6, f fixed
        const double muLa = std::sqrt(0.4 / 12000.0);
        testStaticDrop(1, 1, muLa, muLa, 400, true);
        lap("static drop");
    }
    if (want("tg")) { testTaylorGreen(); lap("Taylor-Green"); }
    if (want("cavity")) { testCavity(); lap("cavity"); }
    std::printf(fails ? "NS TEST FAILED (%d)\n" : "NS TEST PASSED\n", fails);
    return fails != 0;
}
