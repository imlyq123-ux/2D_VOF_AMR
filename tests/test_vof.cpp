// test_vof.cpp - PLIC geometry, geometric prolongation, and Weymouth-Yue advection on AMR
// meshes (translation on a periodic domain, Rider-Kothe single vortex with walls).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../src/mesh.hpp"
#include "../src/plic.hpp"
#include "../src/vof.hpp"

static int fails = 0;
#define CHECK(cond, ...)                                       \
    do {                                                       \
        if (!(cond)) {                                         \
            std::printf("  FAIL (%s:%d): ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                          \
            std::printf("\n");                                 \
            ++fails;                                           \
        }                                                      \
    } while (0)

using Clock = std::chrono::steady_clock;
static double since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ------------------------------------------------------------------ helpers

// Fraction of [xa,xb]x[ya,yb] inside the disk (xc,yc,R): midpoint rule in x with the
// exact chord in y (error ~1e-7 of the cell area).
static double diskFraction(double xa, double xb, double ya, double yb, double xc, double yc, double R) {
    const double dx = std::max({xa - xc, 0.0, xc - xb}), dy = std::max({ya - yc, 0.0, yc - yb});
    if (dx * dx + dy * dy >= R * R) return 0.0;
    const double fx = std::max(std::fabs(xa - xc), std::fabs(xb - xc));
    const double fy = std::max(std::fabs(ya - yc), std::fabs(yb - yc));
    if (fx * fx + fy * fy <= R * R) return 1.0;
    const int N = 256;
    double s = 0.0;
    for (int k = 0; k < N; ++k) {
        const double x = xa + (k + 0.5) * (xb - xa) / N, r2 = R * R - sq(x - xc);
        if (r2 <= 0.0) continue;
        const double q = std::sqrt(r2);
        s += std::max(0.0, std::min(yb, yc + q) - std::max(ya, yc - q));
    }
    return s / N / (yb - ya);
}

struct Disk {
    double xc, yc, R;
    bool periodic;  // sum the images shifted by +-1 (unit periodic domain)
    double operator()(const Mesh& m, int c) const {
        const double h = m.size(c);
        const Vec2 x = m.center(c);
        double s = 0.0;
        const int k = periodic ? 1 : 0;
        for (int a = -k; a <= k; ++a)
            for (int b = -k; b <= k; ++b)
                s += diskFraction(x.x - 0.5 * h, x.x + 0.5 * h, x.y - 0.5 * h, x.y + 0.5 * h, xc + a, yc + b, R);
        return std::min(s, 1.0);
    }
};

static bool isMixed(double v) { return v > vof::EPS && v < 1.0 - vof::EPS; }

// AMR criterion: leaves within `buf` max-level cells of an interfacial leaf (mixed, or
// pure next to the opposite pure value) go to maxLevel, all others are coarsened.
static std::vector<int> amrFlags(const Mesh& m, const std::vector<double>& f, int buf) {
    const int Lm = m.maxLevel(), nx = m.nx(Lm), ny = m.ny(Lm);
    std::vector<uint8_t> itf(m.nCells(), 0), near((size_t)nx * ny, 0);
    for (int c : m.leaves()) itf[c] = isMixed(f[c]);
    for (const auto& F : m.faces())
        if (F.L >= 0 && F.R >= 0 && std::fabs(f[F.L] - f[F.R]) > 0.5) itf[F.L] = itf[F.R] = 1;
    auto wrap = [](int i, int n, bool per) { return per ? (i + n) % n : std::clamp(i, 0, n - 1); };
    for (int c : m.leaves()) {
        if (!itf[c]) continue;
        const int s = 1 << (Lm - m.level(c));
        for (int j = m.iy(c) * s - buf; j < (m.iy(c) + 1) * s + buf; ++j)
            for (int i = m.ix(c) * s - buf; i < (m.ix(c) + 1) * s + buf; ++i)
                near[(size_t)wrap(j, ny, m.periodicY()) * nx + wrap(i, nx, m.periodicX())] = 1;
    }
    std::vector<int> flags(m.nLeaves(), -1);
    for (int k = 0; k < m.nLeaves(); ++k) {
        const int c = m.leaves()[k], s = 1 << (Lm - m.level(c));
        bool any = false;
        for (int j = m.iy(c) * s; j < (m.iy(c) + 1) * s && !any; ++j)
            for (int i = m.ix(c) * s; i < (m.ix(c) + 1) * s && !any; ++i) any = near[(size_t)j * nx + i];
        if (any) flags[k] = m.level(c) < Lm ? 1 : 0;
    }
    return flags;
}

static bool regrid(Mesh& m, int fid, int buf) {
    bool changed = false;
    for (int it = 0; it < 4; ++it) {
        if (!m.adapt(amrFlags(m, m.field(fid), buf))) break;
        changed = true;
    }
    return changed;
}

// Initial AMR mesh around the disk with exact fractions on every pass.
static void buildInitial(Mesh& m, int fid, const Disk& disk, int buf) {
    auto init = [&] { for (int c : m.leaves()) m.field(fid)[c] = disk(m, c); };
    init();
    for (int it = 0; it < 30; ++it) {
        if (!m.adapt(amrFlags(m, m.field(fid), buf))) break;
        init();
    }
}

static double volume(const Mesh& m, const std::vector<double>& f) {
    double s = 0.0;
    for (int c : m.leaves()) s += f[c] * m.volume(c);
    return s;
}

static void bounds(const Mesh& m, const std::vector<double>& f, double& lo, double& hi) {
    lo = 1e30; hi = -1e30;
    for (int c : m.leaves()) { lo = std::min(lo, f[c]); hi = std::max(hi, f[c]); }
}

static double shapeError(const Mesh& m, const std::vector<double>& f, const Disk& disk) {
    double e = 0.0;
    for (int c : m.leaves()) e += std::fabs(f[c] - disk(m, c)) * m.volume(c);
    return e;
}

// Face velocities from a stream function at face end points (exactly divergence free).
static void faceVelocity(const Mesh& m, const std::function<double(double, double)>& psi,
                         std::vector<double>& uf) {
    uf.resize(m.nFaces());
    for (int p = 0; p < m.nFaces(); ++p) {
        const auto& F = m.faces()[p];
        const double a = 0.5 * F.area;
        uf[p] = F.dir == 0 ? (psi(F.x.x, F.x.y + a) - psi(F.x.x, F.x.y - a)) / F.area
                           : -(psi(F.x.x + a, F.x.y) - psi(F.x.x - a, F.x.y)) / F.area;
    }
}

// ------------------------------------------------------------------ 1. PLIC geometry

static void testPlic() {
    std::printf("[plic] analytic area / alpha / sub-rectangles / MYC\n");
    std::vector<double> angles;
    for (int k = 0; k < 720; ++k) angles.push_back(k * M_PI / 360.0);   // includes 0, pi/2, ...
    for (double e : {1e-15, 1e-12, 1e-8}) for (int q = 0; q < 4; ++q) {
        angles.push_back(q * M_PI / 2 + e);
        angles.push_back(q * M_PI / 2 - e);
    }
    std::vector<double> fs = {0.0, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 0.1, 0.25, 0.5, 0.75, 0.9,
                              1 - 1e-3, 1 - 1e-6, 1 - 1e-9, 1 - 1e-12, 1.0};
    double errRT = 0.0, errQ = 0.0, errH = 0.0;
    for (double th : angles) {
        double mx = std::cos(th), my = std::sin(th);
        const double n = std::fabs(mx) + std::fabs(my);
        mx /= n; my /= n;
        for (double f : fs) {
            const double a = plic::alpha(mx, my, f);
            errRT = std::max(errRT, std::fabs(plic::area(mx, my, a) - f));
            double q = 0.0;
            for (int k = 0; k < 4; ++k) {
                const double X0 = 0.5 * (k & 1), Y0 = 0.5 * (k >> 1);
                q += plic::rectArea(mx, my, a, X0, X0 + 0.5, Y0, Y0 + 0.5);
            }
            errQ = std::max(errQ, std::fabs(q - f));
            // two arbitrary strips that tile the cell
            const double h = plic::rectArea(mx, my, a, 0.0, 0.37, 0.0, 1.0) +
                             plic::rectArea(mx, my, a, 0.37, 1.0, 0.0, 1.0);
            errH = std::max(errH, std::fabs(h - f));
        }
    }
    std::printf("  max |area(alpha(f)) - f| = %.2e, quarters %.2e, strips %.2e\n", errRT, errQ, errH);
    CHECK(errRT < 1e-14, "round trip %.3e", errRT);
    CHECK(errQ < 1e-14, "quarters %.3e", errQ);
    CHECK(errH < 1e-14, "strips %.3e", errH);
    // exactly degenerate normals and pure limits
    CHECK(std::fabs(plic::area(0.0, 1.0, 0.3) - 0.3) < 1e-15, "mx=0");
    CHECK(std::fabs(plic::area(-1.0, 0.0, -0.3) - 0.7) < 1e-15, "mx=-1");  // X >= 0.3
    CHECK(plic::area(0.5, 0.5, -0.1) == 0.0 && plic::area(0.5, 0.5, 1.1) == 1.0, "pure limits");
    CHECK(std::fabs(plic::rectArea(0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.5) - 0.25) < 1e-15, "zero normal");
    // segment end points lie on the line and inside the cell; diagonal case
    double P[4];
    CHECK(plic::segment(0.5, 0.5, 0.5, P) && std::fabs(std::hypot(P[2] - P[0], P[3] - P[1]) - std::sqrt(2.0)) < 1e-14,
          "diagonal segment");
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double errS = 0.0;
    for (int it = 0; it < 10000; ++it) {
        const double th = 2 * M_PI * U(rng);
        double mx = std::cos(th), my = std::sin(th);
        const double n = std::fabs(mx) + std::fabs(my);
        mx /= n; my /= n;
        const double a = plic::alpha(mx, my, 0.001 + 0.998 * U(rng));
        if (!plic::segment(mx, my, a, P)) { errS = 1; continue; }
        for (int e = 0; e < 2; ++e) {
            errS = std::max(errS, std::fabs(mx * P[2 * e] + my * P[2 * e + 1] - a));
            for (int q = 0; q < 2; ++q) errS = std::max({errS, -P[2 * e + q], P[2 * e + q] - 1.0});
        }
    }
    CHECK(errS < 1e-14, "segment end points %.3e", errS);
    // MYC on straight lines: exact fractions of a half plane on the 3x3 block
    double eMax = 0.0, eMean = 0.0;
    int cnt = 0;
    for (int ia = 0; ia < 720; ++ia) {
        const double th = ia * M_PI / 360 + 1e-3;
        double mx = std::cos(th), my = std::sin(th);
        const double n = std::fabs(mx) + std::fabs(my);
        mx /= n; my /= n;
        for (int ib = 1; ib < 40; ++ib) {
            const double a = plic::alpha(mx, my, ib / 40.0);
            double s[3][3];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) s[i][j] = plic::area(mx, my, a - mx * (i - 1) - my * (j - 1));
            double nx, ny;
            plic::mycNormal(s, nx, ny);
            double e = std::fabs(std::atan2(ny, nx) - std::atan2(my, mx));
            if (e > M_PI) e = 2 * M_PI - e;
            eMax = std::max(eMax, e);
            eMean += e;
            ++cnt;
        }
    }
    std::printf("  MYC normal on straight lines: max angle error %.2e rad, mean %.2e\n", eMax, eMean / cnt);
    CHECK(eMax < 0.03 && eMean / cnt < 2e-3, "MYC accuracy");
}

// ------------------------------------------------------------------ 2. prolongation

// Mesh with the disk's interfacial region at level 6 (exact fractions), prolongator p.
static void prolongMesh(Mesh& m, int fid, const Disk& disk) {
    auto init = [&] { for (int c : m.leaves()) m.field(fid)[c] = disk(m, c); };
    init();
    for (int L = 3; L < 6; ++L) {
        m.refineWhere([&](const Mesh& mm, int c) { return mm.level(c) < 6 && isMixed(mm.field(fid)[c]); });
        init();
    }
}

static void testProlong() {
    std::printf("[prolong] geometric refinement of an interfacial region (level 6 -> 7)\n");
    Mesh::Config cfg;
    cfg.minLevel = 3; cfg.maxLevel = 7;
    const Disk disk{0.47, 0.53, 0.23, false};
    double err[2];
    for (int geo = 1; geo >= 0; --geo) {
        Mesh m(cfg);
        const int fid = m.addField("f", geo ? Mesh::Prolongator(vof::prolong) : Mesh::Prolongator());
        prolongMesh(m, fid, disk);
        double eMean = 0.0;   // direct check: mean of the children == parent
        if (geo)
            for (int c : m.leaves()) {
                double out[4];
                vof::prolong(m, m.field(fid), c, out);
                eMean = std::max(eMean, std::fabs(0.25 * (out[0] + out[1] + out[2] + out[3]) - m.field(fid)[c]));
            }
        const double V0 = volume(m, m.field(fid));
        m.refineWhere([&](const Mesh& mm, int c) { return mm.level(c) < 7 && isMixed(mm.field(fid)[c]); });
        const double V1 = volume(m, m.field(fid));
        double lo, hi;
        bounds(m, m.field(fid), lo, hi);
        err[geo] = shapeError(m, m.field(fid), disk);
        if (!geo) break;
        std::printf("  max |mean(children) - parent| = %.2e, volume change %.2e, f in [%g, %g]\n", eMean,
                    std::fabs(V1 - V0) / V0, lo, hi);
        CHECK(eMean < 1e-15, "prolongation mean %.3e", eMean);
        CHECK(std::fabs(V1 - V0) < 1e-14 * V0, "prolongation volume %.3e", std::fabs(V1 - V0) / V0);
        CHECK(lo >= 0.0 && hi <= 1.0, "prolongation bounds");
    }
    std::printf("  L1 error of the level-7 children vs exact fractions: geometric %.3e, minmod-linear %.3e\n",
                err[1], err[0]);
    CHECK(err[1] < 0.2 * err[0], "prolongation accuracy");
}

// ------------------------------------------------------------------ 3. translation

static void testTranslation() {
    std::printf("[translation] disk R=0.15, u=(1,0.5), periodic, AMR levels 3..7, one period T=2\n");
    Mesh::Config cfg;
    cfg.minLevel = 3; cfg.maxLevel = 7;
    cfg.periodicX = cfg.periodicY = true;
    Mesh m(cfg);
    const int fid = m.addField("f", vof::prolong);
    Params prm;
    for (auto& b : prm.bc) b.type = BCType::Periodic;
    const Disk disk{0.35, 0.4, 0.15, true};
    const int buf = 4;
    buildInitial(m, fid, disk, buf);
    {   // PLIC segments: perimeter and distance of the end points from the circle
        double len = 0.0, dmax = 0.0;
        for (const auto& sg : vof::interfaceSegments(m, m.field(fid))) {
            len += (sg.second - sg.first).norm();
            for (const Vec2& q : {sg.first, sg.second})
                dmax = std::max(dmax, std::fabs((q - Vec2(disk.xc, disk.yc)).norm() - disk.R));
        }
        const double h = m.h(cfg.maxLevel);
        std::printf("  interfaceSegments: perimeter error %.2e, max end-point distance %.3f h\n",
                    len / (2 * M_PI * disk.R) - 1, dmax / h);
        CHECK(std::fabs(len / (2 * M_PI * disk.R) - 1) < 1.5e-2 && dmax < 0.05 * h, "interface segments");
    }
    const double V0 = volume(m, m.field(fid));
    const double U = 1.0, Vv = 0.5, T = 2.0, cfl = 0.4;
    const int nsteps = (int)std::ceil(T / (cfl * m.h(cfg.maxLevel) / U));
    const double dt = T / nsteps;
    std::vector<double> uf;
    auto build = [&] {
        uf.resize(m.nFaces());
        for (int p = 0; p < m.nFaces(); ++p) uf[p] = m.faces()[p].dir == 0 ? U : Vv;
    };
    build();
    double lo = 0, hi = 1, maxDrift = 0;
    for (int n = 0; n < nsteps; ++n) {
        if (n > 0 && n % 3 == 0 && regrid(m, fid, buf)) build();
        vof::advect(m, m.field(fid), uf, dt, n, prm);
        double a, b;
        bounds(m, m.field(fid), a, b);
        lo = std::min(lo, a); hi = std::max(hi, b);
        maxDrift = std::max(maxDrift, std::fabs(volume(m, m.field(fid)) - V0) / V0);
    }
    const double E = shapeError(m, m.field(fid), disk);
    std::printf("  steps %d, leaves %d, max rel. volume drift %.2e, f in [%g, %g], L1 shape error %.3e\n",
                nsteps, m.nLeaves(), maxDrift, lo, hi, E);
    CHECK(maxDrift < 1e-13, "translation mass %.3e", maxDrift);
    CHECK(lo >= 0.0 && hi <= 1.0, "translation bounds");
    CHECK(E < 1e-3, "translation shape error %.3e", E);
    CHECK(m.checkConsistency().empty(), "mesh consistency");
}

// Interface crossing coarse/fine jumps: static random mesh, levels 4..6 (exercises
// half-side donor regions of coarse donors and fine donors feeding coarse cells).
static void testRandomMesh() {
    std::printf("[random AMR] disk R=0.2 translated by u=(1,0.5) for T=2 on a static random mesh, levels 4..6\n");
    Mesh::Config cfg;
    cfg.minLevel = 4; cfg.maxLevel = 6;
    cfg.periodicX = cfg.periodicY = true;
    Mesh m(cfg);
    const int fid = m.addField("f", vof::prolong);
    std::mt19937 rng(3);
    for (int it = 0; it < 2; ++it) {
        std::vector<int> fl(m.nLeaves());
        for (auto& v : fl) v = rng() % 3 == 0;
        m.adapt(fl);
    }
    Params prm;
    for (auto& b : prm.bc) b.type = BCType::Periodic;
    const Disk disk{0.5, 0.5, 0.2, true};
    for (int c : m.leaves()) m.field(fid)[c] = disk(m, c);
    const double V0 = volume(m, m.field(fid));
    std::vector<double> uf(m.nFaces());
    for (int p = 0; p < m.nFaces(); ++p) uf[p] = m.faces()[p].dir == 0 ? 1.0 : 0.5;
    const int nsteps = 320;   // CFL 0.4 on level 6
    const double dt = 2.0 / nsteps;
    double lo = 0, hi = 1;
    for (int n = 0; n < nsteps; ++n) {
        vof::advect(m, m.field(fid), uf, dt, n, prm);
        double a, b;
        bounds(m, m.field(fid), a, b);
        lo = std::min(lo, a); hi = std::max(hi, b);
    }
    const double drift = std::fabs(volume(m, m.field(fid)) - V0) / V0, E = shapeError(m, m.field(fid), disk);
    std::printf("  leaves %d, rel. volume drift %.2e, f in [%g, %g], L1 shape error %.3e\n", m.nLeaves(), drift, lo,
                hi, E);
    CHECK(drift < 1e-13, "random mesh mass %.3e", drift);
    CHECK(lo >= 0.0 && hi <= 1.0, "random mesh bounds");
    CHECK(E < 5e-3, "random mesh shape error %.3e", E);
}

// ------------------------------------------------------------------ 3b. inflow / outflow

static void testInflow() {
    std::printf("[inflow] channel u=1, left Inflow (f=1 for t<0.5, then 0), right Outflow, slip walls\n");
    Mesh::Config cfg;
    cfg.minLevel = cfg.maxLevel = 5;
    Mesh m(cfg);
    const int fid = m.addField("f", vof::prolong);
    Params prm;
    prm.bc[LEFT].type = BCType::Inflow; prm.bc[LEFT].u = 1.0; prm.bc[LEFT].f = 1.0;
    prm.bc[RIGHT].type = BCType::Outflow;
    prm.bc[BOTTOM].type = prm.bc[TOP].type = BCType::Slip;
    std::vector<double> uf(m.nFaces());
    for (int p = 0; p < m.nFaces(); ++p) uf[p] = m.faces()[p].dir == 0 ? 1.0 : 0.0;
    const double dt = 0.4 / 32;   // CFL 0.4, 80 steps per unit time
    // slab of phase 1 between x = a and x = b
    auto check = [&](double a, double b) {
        const double V = volume(m, m.field(fid));
        double E = 0.0;
        for (int c : m.leaves()) {
            const double x = m.center(c).x;
            E += std::fabs(m.field(fid)[c] - (x > a && x < b ? 1.0 : 0.0)) * m.volume(c);
        }
        std::printf("  slab [%.2f, %.2f]: volume %.15f, L1 vs exact %.1e\n", a, b, V, E);
        CHECK(std::fabs(V - (b - a)) < 1e-13 && E < 1e-13, "inflow/outflow slab");
    };
    int n = 0;
    for (; n < 40; ++n) vof::advect(m, m.field(fid), uf, dt, n, prm);
    check(0.0, 0.5);
    prm.bc[LEFT].f = 0.0;
    for (; n < 100; ++n) vof::advect(m, m.field(fid), uf, dt, n, prm);
    check(0.75, 1.0);
}

// ------------------------------------------------------------------ 4. single vortex

struct VortexResult {
    double E, drift, lo, hi, msPerStep, avgLeaves;
    int steps;
};

static VortexResult singleVortex(int Lmax, double T) {
    Mesh::Config cfg;
    cfg.minLevel = 3; cfg.maxLevel = Lmax;
    Mesh m(cfg);
    const int fid = m.addField("f", vof::prolong);
    Params prm;
    for (auto& b : prm.bc) b.type = BCType::Wall;
    const Disk disk{0.5, 0.75, 0.15, false};
    const int buf = 4;
    buildInitial(m, fid, disk, buf);
    const double V0 = volume(m, m.field(fid));
    const double cfl = 0.4;   // max |u| = 1
    const int nsteps = (int)std::ceil(T / (cfl * m.h(Lmax)));
    const double dt = T / nsteps;
    auto psi = [](double x, double y) { return sq(std::sin(M_PI * x)) * sq(std::sin(M_PI * y)) / M_PI; };
    std::vector<double> base, uf;
    faceVelocity(m, psi, base);
    VortexResult r{0, 0, 0, 1, 0, 0, nsteps};
    double tAdv = 0.0, leaves = 0.0;
    for (int n = 0; n < nsteps; ++n) {
        if (n > 0 && n % 2 == 0 && regrid(m, fid, buf)) faceVelocity(m, psi, base);
        const double s = std::cos(M_PI * (n + 0.5) * dt / T);
        uf.resize(base.size());
        for (size_t p = 0; p < base.size(); ++p) uf[p] = s * base[p];
        const auto t0 = Clock::now();
        vof::advect(m, m.field(fid), uf, dt, n, prm);
        tAdv += since(t0);
        leaves += m.nLeaves();
        double a, b;
        bounds(m, m.field(fid), a, b);
        r.lo = std::min(r.lo, a); r.hi = std::max(r.hi, b);
        r.drift = std::max(r.drift, std::fabs(volume(m, m.field(fid)) - V0) / V0);
    }
    r.E = shapeError(m, m.field(fid), disk);
    r.msPerStep = 1e3 * tAdv / nsteps;
    r.avgLeaves = leaves / nsteps;
    if (!m.checkConsistency().empty()) { std::printf("  FAIL mesh consistency\n"); ++fails; }
    return r;
}

static void testVortex() {
    const double T = 8.0;
    std::printf("[single vortex] R=0.15 at (0.5,0.75), walls, T=%g, AMR minLevel 3, buffer 4\n", T);
    double Eprev = 0.0;
    for (int L = 6; L <= 8; ++L) {
        const VortexResult r = singleVortex(L, T);
        std::printf("  level %d (%4d^2): L1 %.3e", L, 1 << L, r.E);
        if (Eprev > 0) std::printf("  order %.2f", std::log2(Eprev / r.E));
        else std::printf("            ");
        std::printf("  mass %.1e  f in [%g, %g]  steps %d  leaves %.0f  advect %.2f ms/step\n", r.drift, r.lo,
                    r.hi, r.steps, r.avgLeaves, r.msPerStep);
        CHECK(r.drift < 1e-13, "vortex mass %.3e", r.drift);
        CHECK(r.lo >= 0.0 && r.hi <= 1.0, "vortex bounds");
        if (L == 7) CHECK(r.E < 3e-3, "vortex L1 at level 7: %.3e", r.E);
        if (Eprev > 0) CHECK(r.E < 0.6 * Eprev, "vortex convergence");
        Eprev = r.E;
    }
}

// ------------------------------------------------------------------ 5. timing

static void testTiming() {
    Mesh::Config cfg;
    cfg.nx0 = 4; cfg.ny0 = 3; cfg.L0 = 0.25; cfg.minLevel = cfg.maxLevel = 6;  // 256 x 192
    cfg.periodicX = cfg.periodicY = true;
    Mesh m(cfg);
    const int fid = m.addField("f", vof::prolong);
    Params prm;
    for (auto& b : prm.bc) b.type = BCType::Periodic;
    // many interfaces: a lattice of disks
    for (int c : m.leaves()) {
        const Vec2 x = m.center(c);
        const double h = m.size(c);
        double v = 0;
        for (int a = 0; a < 8; ++a)
            for (int b = 0; b < 6; ++b)
                v += diskFraction(x.x - h / 2, x.x + h / 2, x.y - h / 2, x.y + h / 2, 0.0625 + a * 0.125,
                                  0.0625 + b * 0.125, 0.04);
        m.field(fid)[c] = v;
    }
    std::vector<double> uf(m.nFaces());
    for (int p = 0; p < m.nFaces(); ++p) uf[p] = m.faces()[p].dir == 0 ? 1.0 : 0.7;
    const double dt = 0.4 * m.h(6);
    vof::advect(m, m.field(fid), uf, dt, 0, prm);   // warm-up
    const int n = 40;
    const auto t0 = Clock::now();
    for (int k = 1; k <= n; ++k) vof::advect(m, m.field(fid), uf, dt, k, prm);
    int nm = 0;
    for (int c : m.leaves()) nm += isMixed(m.field(fid)[c]);
    std::printf("[timing] %d leaves (%d interfacial): advect %.2f ms/step\n", m.nLeaves(), nm,
                1e3 * since(t0) / n);
}

int main() {
#ifdef _OPENMP
    // the meshes here are small: cap the threads unless the user chose (robust timing on busy hosts)
    if (!std::getenv("OMP_NUM_THREADS")) omp_set_num_threads(std::min(2, omp_get_max_threads()));
    std::printf("OpenMP threads: %d\n", omp_get_max_threads());
#endif
    const auto t0 = Clock::now();
    testPlic();
    testProlong();
    testTranslation();
    testRandomMesh();
    testInflow();
    testVortex();
    testTiming();
    std::printf("total %.1f s\n", since(t0));
    std::printf(fails ? "VOF TEST FAILED (%d)\n" : "VOF TEST PASSED\n", fails);
    return fails != 0;
}
