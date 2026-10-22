// test_poisson.cpp - elliptic solver: manufactured solutions (uniform / AMR, Dirichlet /
// Neumann), jump coefficients (density ratio 1000), symmetry, Helmholtz, fallback, and
// (CUDA builds with a device) the GPU solver against the CPU one.
// Usage: test_poisson [--bench [--gpu]]   (exit 0 on success)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "../src/mesh.hpp"
#include "../src/poisson.hpp"

namespace {

int fails = 0;
void expect(bool ok, const char* what) {
    if (!ok) { std::printf("  FAIL: %s\n", what); ++fails; }
}
double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

const double PI = 3.14159265358979323846;
const Vec2 C0(0.5, 0.5);
const double R0 = 0.25;

std::unique_ptr<Mesh> uniformMesh(int L, bool perX = false) {
    Mesh::Config cfg;
    cfg.minLevel = cfg.maxLevel = L;
    cfg.periodicX = perX;
    return std::make_unique<Mesh>(cfg);
}

// Refined around the circle |x - C0| = R0 up to maxLevel, `depth` coarser levels elsewhere.
std::unique_ptr<Mesh> amrMesh(int maxLevel, int depth, double band = 2.0, bool perX = false) {
    Mesh::Config cfg;
    cfg.maxLevel = maxLevel;
    cfg.minLevel = std::max(1, maxLevel - depth);
    cfg.periodicX = perX;
    auto m = std::make_unique<Mesh>(cfg);
    m->refineWhere([band](const Mesh& mm, int c) {
        const double r = (mm.center(c) - C0).norm();
        return std::fabs(r - R0) < band * mm.size(c);
    });
    return m;
}

// Fraction of face f lying inside the circle (sampled).
double faceInside(const Mesh::Face& F) {
    const int ns = 16;
    int in = 0;
    for (int s = 0; s < ns; ++s) {
        const double t = ((s + 0.5) / ns - 0.5) * F.area;
        const Vec2 p = F.dir == 0 ? Vec2(F.x.x, F.x.y + t) : Vec2(F.x.x + t, F.x.y);
        if ((p - C0).norm() < R0) ++in;
    }
    return double(in) / ns;
}

// beta_f = 1 / rho_f, rho_f from the face-averaged indicator.
std::vector<double> jumpBeta(const Mesh& m, double rhoIn, double rhoOut) {
    std::vector<double> b(m.nFaces());
    for (int f = 0; f < m.nFaces(); ++f) {
        const double fi = faceInside(m.faces()[f]);
        b[f] = 1.0 / (fi * rhoIn + (1.0 - fi) * rhoOut);
    }
    return b;
}

// Independent operator application (header formula): returns the volume-weighted relative
// residual || b - (A x)/V ||_V / || b_eff ||_V, with the Dirichlet data moved to the rhs.
double checkResidual(const Mesh& m, const poisson::Problem& p, const std::vector<double>& x, bool singular) {
    std::vector<double> bb(m.nCells(), 0.0);
    double vol = 0.0, mean = 0.0;
    for (int c : m.leaves()) { vol += m.volume(c); mean += p.rhs[c] * m.volume(c); }
    mean = singular ? mean / vol : 0.0;
    double rn = 0.0, bn = 0.0;
    for (int c : m.leaves()) {
        const double V = m.volume(c);
        double Ax = p.lambda.empty() ? 0.0 : p.lambda[c] * V * x[c];
        double b = (p.rhs[c] - mean) * V;
        for (const int* fp = m.cellFacesBegin(c); fp != m.cellFacesEnd(c); ++fp) {
            const Mesh::Face& F = m.faces()[*fp];
            const int nb = F.L == c ? F.R : F.L;
            const double a = p.beta[*fp] * F.area / F.dist;
            if (nb < 0) {
                if (!p.dirichlet.empty() && p.dirichlet[*fp]) {
                    Ax += a * x[c];
                    b += a * (p.bvalue.empty() ? 0.0 : p.bvalue[*fp]);
                }
            } else {
                Ax += a * (x[c] - x[nb]);
            }
        }
        rn += (b - Ax) * (b - Ax) / V;
        bn += b * b / V;
    }
    return std::sqrt(rn / bn);
}

double wmean(const Mesh& m, const std::vector<double>& x) {
    double s = 0.0, v = 0.0;
    for (int c : m.leaves()) { s += x[c] * m.volume(c); v += m.volume(c); }
    return s / v;
}

struct Err { double l2, linf; };
Err error(const Mesh& m, const std::vector<double>& x, const std::function<double(Vec2)>& ue, bool zeroMean) {
    double shift = 0.0;
    if (zeroMean) {
        std::vector<double> e(m.nCells(), 0.0);
        for (int c : m.leaves()) e[c] = ue(m.center(c));
        shift = wmean(m, e);
    }
    double l2 = 0.0, li = 0.0, v = 0.0;
    for (int c : m.leaves()) {
        const double d = x[c] - (ue(m.center(c)) - shift);
        l2 += d * d * m.volume(c);
        v += m.volume(c);
        li = std::max(li, std::fabs(d));
    }
    return {std::sqrt(l2 / v), li};
}

// Problem for -div(beta grad u) + lambda u = f with constant beta, lambda and exact ue.
poisson::Problem manufactured(const Mesh& m, bool dirichlet, double lambda,
                              const std::function<double(Vec2)>& ue, const std::function<double(Vec2)>& f) {
    poisson::Problem p;
    p.beta.assign(m.nFaces(), 1.0);
    p.rhs.assign(m.nCells(), 0.0);
    for (int c : m.leaves()) p.rhs[c] = f(m.center(c));
    if (lambda != 0.0) p.lambda.assign(m.nCells(), lambda);
    if (dirichlet) {
        p.dirichlet.assign(m.nFaces(), 0);
        p.bvalue.assign(m.nFaces(), 0.0);
        for (int k = 0; k < m.nFaces(); ++k) {
            const Mesh::Face& F = m.faces()[k];
            if (F.bside >= 0) { p.dirichlet[k] = 1; p.bvalue[k] = ue(F.x); }
        }
    }
    return p;
}

// ---------------------------------------------------------------- tests

void testConvergence(bool amr, bool dirichlet) {
    // Dirichlet: u = sin(pi x + 0.4) cos(1.3 pi y);  Neumann: u = cos(pi x) cos(2 pi y)
    const double k2 = dirichlet ? PI * PI * (1.0 + 1.69) : 5.0 * PI * PI;
    auto ue = [dirichlet](Vec2 p) {
        return dirichlet ? std::sin(PI * p.x + 0.4) * std::cos(1.3 * PI * p.y)
                         : std::cos(PI * p.x) * std::cos(2.0 * PI * p.y);
    };
    auto f = [&](Vec2 p) { return k2 * ue(p); };
    std::printf("%s mesh, %s BCs:\n", amr ? "AMR (circle, 3 coarser levels)" : "uniform",
                dirichlet ? "Dirichlet" : "Neumann (singular)");
    std::printf("  %8s %8s %12s %6s %12s %6s %5s %10s\n", "maxLev", "leaves", "L2", "order", "Linf", "order",
                "iter", "resid");
    std::vector<double> l2s, lis;
    for (int L = 5; L <= 9; ++L) {
        auto m = amr ? amrMesh(L, 3) : uniformMesh(L);
        poisson::Problem p = manufactured(*m, dirichlet, 0.0, ue, f);
        std::vector<double> x(m->nCells(), 0.0);
        poisson::Options o;
        o.tol = 1e-10;
        poisson::Result r = poisson::solve(*m, p, x, o);
        Err e = error(*m, x, ue, !dirichlet);
        const double chk = checkResidual(*m, p, x, !dirichlet);
        if (L == 5) std::printf("  %8d %8d %12.4e %6s %12.4e %6s %5d %10.2e\n", L, m->nLeaves(), e.l2, "-", e.linf,
                                "-", r.iterations, r.residual);
        else std::printf("  %8d %8d %12.4e %6.2f %12.4e %6.2f %5d %10.2e\n", L, m->nLeaves(), e.l2,
                         std::log2(l2s.back() / e.l2), e.linf, std::log2(lis.back() / e.linf), r.iterations,
                         r.residual);
        expect(r.converged && r.residual <= 1e-10 * 1.01, "converged");
        expect(chk < 1e-9, "independent residual check");
        if (!dirichlet) expect(std::fabs(wmean(*m, x)) < 1e-12, "zero mean");
        if (L > 5 && !amr) {
            expect(std::log2(l2s.back() / e.l2) > 1.8, "L2 order (uniform)");
            expect(std::log2(lis.back() / e.linf) > 1.8, "Linf order (uniform)");
        }
        expect(r.iterations <= 25, "iteration count");
        l2s.push_back(e.l2);
        lis.push_back(e.linf);
    }
    if (amr) {
        // The two-point flux of poisson.hpp at coarse/fine faces (d = 1.5 h_fine, centres
        // not aligned with the normal) is inconsistent: first order expected on AMR meshes.
        const double o2 = std::log2(l2s[1] / l2s[4]) / 3.0, oi = std::log2(lis[1] / lis[4]) / 3.0;
        std::printf("  mean order levels 6->9: L2 %.2f, Linf %.2f (first order expected: two-point coarse/fine flux)\n",
                    o2, oi);
        expect(o2 > 0.75 && oi > 0.4, "AMR order");
    }
}

void testJump() {
    std::printf("Jump coefficient beta = 1/rho, rho ratio 1000 across the circle:\n");
    std::printf("  %-40s %8s %5s %10s %10s\n", "case", "leaves", "iter", "resid", "check");
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (int amr = 0; amr < 2; ++amr)
        for (int light = 0; light < 2; ++light)
            for (int dir = 0; dir < 2; ++dir) {
                auto m = amr ? amrMesh(9, 4) : uniformMesh(8);
                poisson::Problem p;
                p.beta = light ? jumpBeta(*m, 1.0, 1000.0) : jumpBeta(*m, 1000.0, 1.0);
                p.rhs.assign(m->nCells(), 0.0);
                for (int c : m->leaves()) {
                    const Vec2 x = m->center(c);
                    p.rhs[c] = std::sin(3 * PI * x.x) * std::cos(2 * PI * x.y) + 0.3 * U(rng);
                }
                if (dir) {  // Dirichlet p = 0 on the top side (outflow)
                    p.dirichlet.assign(m->nFaces(), 0);
                    for (int k = 0; k < m->nFaces(); ++k) p.dirichlet[k] = m->faces()[k].bside == TOP;
                }
                std::vector<double> x(m->nCells(), 0.0);
                poisson::Options o;
                poisson::Result r = poisson::solve(*m, p, x, o);
                const double chk = checkResidual(*m, p, x, !dir);
                char name[96];
                std::snprintf(name, sizeof name, "%s, %s bubble, %s", amr ? "AMR" : "uniform 256^2",
                              light ? "light" : "heavy", dir ? "Dirichlet top" : "all Neumann");
                std::printf("  %-40s %8d %5d %10.2e %10.2e\n", name, m->nLeaves(), r.iterations, r.residual, chk);
                expect(r.converged && chk < 2e-9, "jump case converged");
                expect(r.iterations <= 30, "jump case iterations");
                // warm start with the solution: must converge immediately
                poisson::Result r2 = poisson::solve(*m, p, x, o);
                expect(r2.converged && r2.iterations <= 1, "warm start");
            }
}

// A symmetric <=> A^{-1} symmetric: (V b2) . A^{-1}(V b1) == (V b1) . A^{-1}(V b2).
void testSymmetry() {
    std::printf("Symmetry of the assembled operator (via A^-1, AMR + jump + Helmholtz):\n");
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (int sing = 0; sing < 2; ++sing) {
        auto m = amrMesh(8, 4, 2.0, sing == 1);
        poisson::Problem p;
        p.beta = jumpBeta(*m, 1.0, 1000.0);
        if (!sing) {
            p.lambda.assign(m->nCells(), 0.0);
            for (int c : m->leaves()) p.lambda[c] = (m->center(c) - C0).norm() < R0 ? 1.0 : 30.0;
            p.dirichlet.assign(m->nFaces(), 0);
            for (int k = 0; k < m->nFaces(); ++k) p.dirichlet[k] = m->faces()[k].bside == LEFT;
        }
        std::vector<double> b1(m->nCells(), 0.0), b2(m->nCells(), 0.0);
        for (int c : m->leaves()) { b1[c] = U(rng); b2[c] = U(rng); }
        if (sing) {  // make them compatible so that the mean removal does not interfere
            const double m1 = wmean(*m, b1), m2 = wmean(*m, b2);
            for (int c : m->leaves()) { b1[c] -= m1; b2[c] -= m2; }
        }
        poisson::Options o;
        o.tol = 1e-12;   // ~ attainable accuracy for the singular jump case
        o.absTol = 0.0;
        o.maxIter = 300;
        std::vector<double> x1(m->nCells(), 0.0), x2(m->nCells(), 0.0);
        p.rhs = b1;
        poisson::Result r1 = poisson::solve(*m, p, x1, o);
        p.rhs = b2;
        poisson::Result r2 = poisson::solve(*m, p, x2, o);
        double s12 = 0, s21 = 0, n12 = 0;
        for (int c : m->leaves()) {
            const double V = m->volume(c);
            s12 += V * b2[c] * x1[c];
            s21 += V * b1[c] * x2[c];
            n12 += V * std::fabs(b2[c] * x1[c]);
        }
        const double asym = std::fabs(s12 - s21) / n12;
        std::printf("  %-26s leaves=%d  iters %d/%d  |b2.A^-1 b1 - b1.A^-1 b2| / sum = %.2e\n",
                    sing ? "periodic-x, all Neumann" : "Dirichlet left, lambda>0", m->nLeaves(), r1.iterations,
                    r2.iterations, asym);
        expect(r1.residual < 1e-11 && r2.residual < 1e-11, "tight solves");
        expect(asym < 1e-10, "symmetry");
    }
}

void testHelmholtz() {
    std::printf("Helmholtz (implicit viscosity) lambda u - div(mu grad u) = f:\n");
    // (a) manufactured, lambda = 1/dt constant, Dirichlet, uniform (order) and AMR (iterations)
    auto ue = [](Vec2 p) { return std::sin(PI * p.x + 0.4) * std::cos(1.3 * PI * p.y); };
    for (double lam : {1e2, 1e4}) {
        auto f = [&](Vec2 p) { return (PI * PI * 2.69 + lam) * ue(p); };
        double pl2 = 0;
        for (int L = 6; L <= 8; ++L) {
            auto m = uniformMesh(L);
            poisson::Problem p = manufactured(*m, true, lam, ue, f);
            std::vector<double> x(m->nCells(), 0.0);
            poisson::Result r = poisson::solve(*m, p, x, poisson::Options());
            Err e = error(*m, x, ue, false);
            std::printf("  lambda=%.0e uniform %4d^2 L2=%.3e", lam, 1 << L, e.l2);
            if (L > 6) std::printf(" order=%.2f", std::log2(pl2 / e.l2));
            std::printf(" iter=%d res=%.1e\n", r.iterations, r.residual);
            expect(r.converged, "Helmholtz converged");
            if (L > 6) expect(std::log2(pl2 / e.l2) > (lam < 1e3 ? 1.8 : 1.5), "Helmholtz order");
            expect(r.iterations <= 20, "Helmholtz iterations");
            pl2 = e.l2;
        }
    }
    for (double lam : {1e2, 1e6}) {
        auto f = [&](Vec2 p) { return (PI * PI * 2.69 + lam) * ue(p); };
        auto m = amrMesh(9, 4);
        poisson::Problem p = manufactured(*m, true, lam, ue, f);
        std::vector<double> x(m->nCells(), 0.0);
        poisson::Result r = poisson::solve(*m, p, x, poisson::Options());
        std::printf("  lambda=%.0e AMR leaves=%d iter=%d res=%.1e\n", lam, m->nLeaves(), r.iterations, r.residual);
        expect(r.converged && r.iterations <= 20, "Helmholtz AMR");
    }
    // (b) viscous system with density/viscosity jump: lambda = rho/dt, beta = mu_f, Neumann
    for (double dt : {1e-2, 1e-5}) {
        auto m = amrMesh(9, 4);
        poisson::Problem p;
        p.beta.resize(m->nFaces());
        for (int k = 0; k < m->nFaces(); ++k) {
            const double fi = faceInside(m->faces()[k]);
            p.beta[k] = fi * 1e-3 + (1 - fi) * 1e-1;
        }
        p.lambda.assign(m->nCells(), 0.0);
        p.rhs.assign(m->nCells(), 0.0);
        for (int c : m->leaves()) {
            const double rho = (m->center(c) - C0).norm() < R0 ? 1.0 : 1000.0;
            p.lambda[c] = rho / dt;
            p.rhs[c] = p.lambda[c] * std::sin(2 * PI * m->center(c).x);
        }
        std::vector<double> x(m->nCells(), 0.0);
        poisson::Result r = poisson::solve(*m, p, x, poisson::Options());
        const double chk = checkResidual(*m, p, x, false);
        std::printf("  rho/dt jump 1000, mu jump 100, dt=%.0e: leaves=%d iter=%d res=%.1e check=%.1e\n", dt,
                    m->nLeaves(), r.iterations, r.residual, chk);
        expect(r.converged && chk < 1e-8, "viscous jump converged");
        expect(r.iterations <= 20, "viscous jump iterations");
    }
}

void testMisc() {
    std::printf("Misc:\n");
    // Jacobi fallback gives the same answer
    auto m = amrMesh(7, 3);
    poisson::Problem p;
    p.beta = jumpBeta(*m, 1.0, 1000.0);
    p.rhs.assign(m->nCells(), 0.0);
    for (int c : m->leaves()) p.rhs[c] = std::cos(PI * m->center(c).x);
    std::vector<double> xa(m->nCells(), 0.0), xj(m->nCells(), 0.0);
    poisson::Options o;
    o.tol = 1e-11;
    poisson::Result ra = poisson::solve(*m, p, xa, o);
    o.precond = 0;
    o.maxIter = 20000;
    poisson::Result rj = poisson::solve(*m, p, xj, o);
    double d = 0, s = 0;
    for (int c : m->leaves()) { d = std::max(d, std::fabs(xa[c] - xj[c])); s = std::max(s, std::fabs(xa[c])); }
    std::printf("  AMG vs Jacobi (AMR, jump, Neumann): iters %d vs %d, max diff / max = %.1e\n", ra.iterations,
                rj.iterations, d / s);
    expect(ra.converged && rj.converged && d / s < 1e-6, "Jacobi fallback agrees");
    // maxIter exhaustion is reported, not thrown
    o.precond = 1;
    o.maxIter = 2;
    o.tol = 1e-14;
    std::vector<double> x0(m->nCells(), 0.0);
    poisson::Result rf = poisson::solve(*m, p, x0, o);
    expect(!rf.converged && rf.iterations == 2 && rf.residual > 0, "non-convergence reported");
    // zero rhs, Dirichlet zero: x -> 0 immediately
    poisson::Problem pz = p;
    std::fill(pz.rhs.begin(), pz.rhs.end(), 0.0);
    pz.dirichlet.assign(m->nFaces(), 1);
    std::vector<double> xz(m->nCells(), 0.0);
    poisson::Result rz = poisson::solve(*m, pz, xz, poisson::Options());
    expect(rz.converged && rz.iterations == 0, "zero rhs");
    // tiny mesh (single level, dense coarse solve only), periodic in x
    auto mt = uniformMesh(2, true);
    poisson::Problem pt;
    pt.beta.assign(mt->nFaces(), 1.0);
    pt.rhs.assign(mt->nCells(), 0.0);
    for (int c : mt->leaves()) pt.rhs[c] = std::sin(2 * PI * mt->center(c).x) + 1.0;
    std::vector<double> xt(mt->nCells(), 0.0);
    poisson::Result rt = poisson::solve(*mt, pt, xt, poisson::Options());
    expect(rt.converged && checkResidual(*mt, pt, xt, true) < 1e-9, "tiny periodic mesh");
    // non-leaf entries untouched
    std::vector<double> xn(m->nCells(), 42.0);
    poisson::solve(*m, p, xn, poisson::Options());
    bool untouched = true;
    for (int c = 0; c < m->nCells(); ++c) if (!m->isLeaf(c) && xn[c] != 42.0) untouched = false;
    expect(untouched, "non-leaf entries untouched");
    std::printf("  failure reporting, zero rhs, tiny periodic mesh, non-leaf entries: checked\n");
}

// CUDA backend: same answers as the CPU solver (Options::gpu, forced to the device with
// Options::gpuMin = 0). Skipped without a usable device.
void testGpu() {
    std::printf("GPU solver:\n");
    if (poisson::gpuDevice().empty()) {
        std::printf("  skipped (CPU-only build or no CUDA device)\n");
        return;
    }
    std::printf("  device: %s\n", poisson::gpuDevice().c_str());
    auto compare = [](const char* what, const Mesh& m, const poisson::Problem& p, poisson::Options o,
                      bool singular) {
        std::vector<double> xc(m.nCells(), 0.0), xg(m.nCells(), 0.0);
        o.gpu = false;
        const poisson::Result rc = poisson::solve(m, p, xc, o);
        o.gpu = true;
        o.gpuMin = 0;
        const poisson::Result rg = poisson::solve(m, p, xg, o);
        double d = 0, s = 0;
        for (int c : m.leaves()) { d = std::max(d, std::fabs(xc[c] - xg[c])); s = std::max(s, std::fabs(xc[c])); }
        const double chk = checkResidual(m, p, xg, singular);
        std::printf("  %-38s leaves=%6d iter cpu %3d gpu %3d  check=%.1e  max diff / max = %.1e\n", what,
                    m.nLeaves(), rc.iterations, rg.iterations, chk, d / s);
        expect(rg.converged && chk < 1e-8, "GPU solve converged");
        expect(d / s < 1e-6, "GPU solution agrees with the CPU one");
        expect(rg.iterations <= rc.iterations + rc.iterations / 2 + 2, "GPU iteration count");
        return xg;
    };
    auto rhsOf = [](const Mesh& m) {
        std::vector<double> b(m.nCells(), 0.0);
        for (int c : m.leaves()) {
            const Vec2 x = m.center(c);
            b[c] = std::sin(3 * PI * x.x) * std::cos(2 * PI * x.y) + ((c * 7919) % 13) * 0.01;
        }
        return b;
    };
    auto um = uniformMesh(8);
    auto am = amrMesh(11, 5, 5.0);
    poisson::Problem p;
    // pure Neumann (singular), density ratio 1000
    p.beta = jumpBeta(*um, 1.0, 1000.0);
    p.rhs = rhsOf(*um);
    compare("uniform, jump 1000, Neumann", *um, p, poisson::Options(), true);
    p.beta = jumpBeta(*am, 1.0, 1000.0);
    p.rhs = rhsOf(*am);
    const std::vector<double> xa = compare("AMR, jump 1000, Neumann", *am, p, poisson::Options(), true);
    // a converged initial guess is accepted as is; non-leaf entries are untouched
    std::vector<double> x0 = xa;
    for (int c = 0; c < am->nCells(); ++c) if (!am->isLeaf(c)) x0[c] = 42.0;
    poisson::Options og;
    og.gpu = true;
    og.gpuMin = 0;
    const poisson::Result r0 = poisson::solve(*am, p, x0, og);
    bool untouched = true;
    for (int c = 0; c < am->nCells(); ++c) if (!am->isLeaf(c) && x0[c] != 42.0) untouched = false;
    expect(r0.converged && r0.iterations == 0 && untouched, "GPU: initial guess, non-leaf entries");
    // Dirichlet data + Helmholtz term (implicit viscosity), AMG and Jacobi preconditioners
    auto ue = [](Vec2 q) { return std::sin(PI * q.x + 0.4) * std::cos(1.3 * PI * q.y); };
    for (double lam : {1e2, 1e5}) {
        auto f = [&](Vec2 q) { return (PI * PI * 2.69 + lam) * ue(q); };
        poisson::Problem ph = manufactured(*am, true, lam, ue, f);
        poisson::Options o;
        o.precond = lam > 1e3 ? 0 : 1;
        compare(lam > 1e3 ? "AMR, Helmholtz 1e5, Dirichlet, Jacobi" : "AMR, Helmholtz 1e2, Dirichlet, AMG", *am, ph,
                o, false);
    }
    // periodic in x
    auto pm = uniformMesh(8, true);
    poisson::Problem pp;
    pp.beta = jumpBeta(*pm, 1.0, 1000.0);
    pp.rhs = rhsOf(*pm);
    compare("uniform periodic-x, jump 1000", *pm, pp, poisson::Options(), true);
    // maxIter exhaustion is reported
    poisson::Options of;
    of.gpu = true;
    of.gpuMin = 0;
    of.maxIter = 2;
    of.tol = 1e-14;
    std::vector<double> xf(am->nCells(), 0.0);
    const poisson::Result rf = poisson::solve(*am, p, xf, of);
    expect(!rf.converged && rf.iterations == 2 && rf.residual > 0, "GPU: non-convergence reported");
    std::printf("  initial guess, non-leaf entries, failure reporting: checked\n");
}

void bench(bool gpu) {
    std::printf("Benchmark on the %s (tol 1e-9, zero initial guess; times include setup):\n",
                gpu ? ("GPU: " + poisson::gpuDevice()).c_str() : "CPU");
    std::printf("  %-22s %9s %-10s %5s %9s %9s %9s\n", "mesh", "leaves", "beta", "iter", "setup[s]", "total[s]",
                "resid");
    std::vector<std::pair<std::string, std::function<std::unique_ptr<Mesh>()>>> meshes = {
        {"uniform 256^2", [] { return uniformMesh(8); }},
        {"uniform 512^2", [] { return uniformMesh(9); }},
        {"uniform 1024^2", [] { return uniformMesh(10); }},
        {"AMR maxLev 11 (6..11)", [] { return amrMesh(11, 5, 5.0); }},
        {"AMR maxLev 12 (6..12)", [] { return amrMesh(12, 6, 8.0); }},
    };
    for (auto& mk : meshes) {
        auto m = mk.second();
        for (int jump = 0; jump < 2; ++jump) {
            poisson::Problem p;
            p.beta = jump ? jumpBeta(*m, 1.0, 1000.0) : std::vector<double>(m->nFaces(), 1.0);
            p.rhs.assign(m->nCells(), 0.0);
            for (int c : m->leaves()) {
                const Vec2 x = m->center(c);
                p.rhs[c] = std::sin(3 * PI * x.x) * std::cos(2 * PI * x.y) + ((c * 7919) % 13) * 0.01;
            }
            std::vector<double> x(m->nCells(), 0.0);
            poisson::Options o;
            o.gpu = gpu;
            o.gpuMin = 0;
            const double t0 = now();
            poisson::Result r = poisson::solve(*m, p, x, o);
            const double t1 = now();
            // setup-only time: a solve with maxIter = 0
            std::vector<double> x0(m->nCells(), 0.0);
            o.maxIter = 0;
            const double t2 = now();
            poisson::solve(*m, p, x0, o);
            const double t3 = now();
            std::printf("  %-22s %9d %-10s %5d %9.3f %9.3f %9.1e\n", mk.first.c_str(), m->nLeaves(),
                        jump ? "1/rho 1000" : "const", r.iterations, t3 - t2, t1 - t0, r.residual);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool doBench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    const double t0 = now();
    if (doBench) {
        const bool gpu = argc > 2 && std::strcmp(argv[2], "--gpu") == 0;
        if (gpu && poisson::gpuDevice().empty()) {
            std::printf("no GPU solver (CPU-only build or no CUDA device)\n");
            return 1;
        }
        bench(gpu);
        return 0;
    }
    testConvergence(false, true);
    testConvergence(false, false);
    testConvergence(true, true);
    testConvergence(true, false);
    testJump();
    testSymmetry();
    testHelmholtz();
    testMisc();
    testGpu();
    std::printf("%s (%d failures, %.1f s)\n", fails ? "POISSON TEST FAILED" : "POISSON TEST PASSED", fails,
                now() - t0);
    return fails != 0;
}
