// test_curvature.cpp - height-function curvature: convergence on circles, bubbles, an
// ellipse, a sine wave, drops on walls; NaN/Inf, bounds and the extension rule.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>
#include "../src/curvature.hpp"
#include "../src/mesh.hpp"

namespace curvature {
void lastStats(long out[8]);  // diagnostic hook defined in curvature.cpp
}

namespace {

constexpr double EPS = 1e-10;  // == vof::EPS
int gFails = 0;

void require(bool ok, const char* what) {
    if (!ok) { std::printf("  FAIL: %s\n", what); ++gFails; }
}

// ------------------------------------------------------------ exact area fractions
// Gauss-Legendre 8 points on [a, b].
double gauss(const std::function<double(double)>& g, double a, double b) {
    static const double x[4] = {0.1834346424956498, 0.5255324099163290, 0.7966664774136267,
                                0.9602898564975363};
    static const double w[4] = {0.3626837833783620, 0.3137066458778873, 0.2223810344533745,
                                0.1012285362903763};
    const double c = 0.5 * (a + b), r = 0.5 * (b - a);
    double s = 0.0;
    for (int k = 0; k < 4; ++k) s += w[k] * (g(c - r * x[k]) + g(c + r * x[k]));
    return s * r;
}
double piecewise(const std::function<double(double)>& g, double a, double b, std::vector<double> br) {
    br.push_back(a);
    br.push_back(b);
    std::sort(br.begin(), br.end());
    double s = 0.0;
    for (size_t k = 0; k + 1 < br.size(); ++k) {
        const double lo = std::max(a, br[k]), hi = std::min(b, br[k + 1]);
        if (hi > lo) s += gauss(g, lo, hi);
    }
    return s;
}

struct Shape {
    virtual ~Shape() = default;
    virtual double area(double x0, double y0, double x1, double y1) const = 0;  // phase 1
    virtual double dist(Vec2 p) const = 0;     // (approximate) distance to the interface
    virtual double kappa(Vec2 p) const = 0;    // exact curvature "at" cell centre p
};

// Ellipse (a = b: circle), phase 1 inside (or outside if inverted).
struct Ellipse : Shape {
    double xc, yc, a, b;
    bool inv;
    Ellipse(double xc_, double yc_, double a_, double b_, bool inv_ = false)
        : xc(xc_), yc(yc_), a(a_), b(b_), inv(inv_) {}
    double area(double x0, double y0, double x1, double y1) const override {
        const double t0 = std::asin(std::clamp((x0 - xc) / a, -1.0, 1.0));
        const double t1 = std::asin(std::clamp((x1 - xc) / a, -1.0, 1.0));
        double A = 0.0;
        if (t1 > t0) {
            auto g = [&](double t) {
                const double c = std::cos(t);
                return (std::clamp(yc + b * c, y0, y1) - std::clamp(yc - b * c, y0, y1)) * a * c;
            };
            std::vector<double> br;
            for (double v : {(y0 - yc) / b, (y1 - yc) / b, (yc - y0) / b, (yc - y1) / b})
                if (v >= 0.0 && v <= 1.0) { br.push_back(std::acos(v)); br.push_back(-std::acos(v)); }
            A = piecewise(g, t0, t1, br);
        }
        const double frac = A / ((x1 - x0) * (y1 - y0));
        return inv ? 1.0 - frac : frac;
    }
    double dist(Vec2 p) const override {
        const double dx = p.x - xc, dy = p.y - yc;
        const double g = sq(dx / a) + sq(dy / b) - 1.0;
        const double gg = 2.0 * std::sqrt(sq(dx / (a * a)) + sq(dy / (b * b))) + 1e-300;
        return std::fabs(g) / gg;
    }
    // curvature at the interface point closest to p
    double kappa(Vec2 p) const override {
        double best = 1e300, tb = 0.0;
        const int N = 720;
        auto d2 = [&](double t) { return sq(xc + a * std::cos(t) - p.x) + sq(yc + b * std::sin(t) - p.y); };
        for (int k = 0; k < N; ++k) {
            const double t = 2.0 * M_PI * k / N, d = d2(t);
            if (d < best) { best = d; tb = t; }
        }
        double lo = tb - 2.0 * M_PI / N, hi = tb + 2.0 * M_PI / N;
        for (int it = 0; it < 100; ++it) {
            const double m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3;
            if (d2(m1) < d2(m2)) hi = m2; else lo = m1;
        }
        const double t = 0.5 * (lo + hi);
        // the height function evaluates kappa on the cell's column (x = p.x) if the
        // interface is mostly horizontal, on its row (y = p.y) otherwise
        double ct = std::cos(t), st = std::sin(t);
        if (std::fabs(st / b) >= std::fabs(ct / a)) {
            ct = std::clamp((p.x - xc) / a, -1.0, 1.0);
            st = std::copysign(std::sqrt(1.0 - ct * ct), st);
        } else {
            st = std::clamp((p.y - yc) / b, -1.0, 1.0);
            ct = std::copysign(std::sqrt(1.0 - st * st), ct);
        }
        const double k = a * b / std::pow(sq(a * st) + sq(b * ct), 1.5);
        return inv ? -k : k;
    }
};

// Phase 1 below y = y0 + A cos(2 pi x) (periodic in x).
struct Wave : Shape {
    double y0, A;
    Wave(double y0_, double A_) : y0(y0_), A(A_) {}
    double Y(double x) const { return y0 + A * std::cos(2 * M_PI * x); }
    double area(double x0, double ya, double x1, double yb) const override {
        auto g = [&](double x) { return std::clamp(Y(x) - ya, 0.0, yb - ya); };
        std::vector<double> br;
        for (double yv : {ya, yb}) {
            const double v = (yv - y0) / A;
            if (v < -1.0 || v > 1.0) continue;
            const double r = std::acos(v) / (2 * M_PI);
            for (int n = (int)std::floor(x0) - 1; n <= (int)std::ceil(x1) + 1; ++n) {
                br.push_back(n + r);
                br.push_back(n - r);
            }
        }
        return piecewise(g, x0, x1, br) / ((x1 - x0) * (yb - ya));
    }
    double dist(Vec2 p) const override {
        const double s = -2 * M_PI * A * std::sin(2 * M_PI * p.x);
        return std::fabs(p.y - Y(p.x)) / std::sqrt(1 + s * s);
    }
    // curvature at the interface point of the column x = p.x (where y-heights evaluate it)
    double kappa(Vec2 p) const override {
        const double yp = -2 * M_PI * A * std::sin(2 * M_PI * p.x);
        const double ypp = -4 * M_PI * M_PI * A * std::cos(2 * M_PI * p.x);
        return -ypp / std::pow(1 + yp * yp, 1.5);
    }
};

// ------------------------------------------------------------ setup / measurement
struct Result {
    double l2 = 0, linf = 0, linfWall = 0, tcall = 0;
    long st[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int nLeaves = 0, nIfc = 0;
    bool finite = true, bounded = true;
    long jumpFaces = 0, jumpOneSide = 0, jumpBoth = 0;
};

// kref: normalization of the error (|kappa - kex| / kref).
Result run(const Shape& S, int level, double kref, bool periodicX, bool uniform = false) {
    Mesh::Config cfg;
    cfg.minLevel = uniform ? level : std::min(3, level);
    cfg.maxLevel = level;
    cfg.periodicX = periodicX;
    Mesh m(cfg);
    const double hmax = m.h(level);
    if (!uniform)
        m.refineWhere([&](const Mesh& mm, int c) {
            return S.dist(mm.center(c)) < 4.0 * hmax + 0.75 * mm.size(c) + 1e-12;
        });
    const int fid = m.addField("f");
    std::vector<double>& f = m.field(fid);
    for (int c : m.leaves()) {
        const Vec2 x = m.center(c);
        const double hh = 0.5 * m.size(c);
        double v = S.area(x.x - hh, x.y - hh, x.x + hh, x.y + hh);
        if (v < EPS) v = 0.0;
        if (v > 1.0 - EPS) v = 1.0;
        f[c] = v;
    }
    std::vector<double> kappa;
    Result R;
    curvature::compute(m, f, kappa);  // warm-up (allocations)
    const int reps = 5;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) curvature::compute(m, f, kappa);
    R.tcall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
    curvature::lastStats(R.st);
    R.nLeaves = m.nLeaves();
    double s2 = 0;
    for (int c : m.leaves()) {
        const double k = kappa[c];
        if (!std::isfinite(k)) R.finite = false;
        if (std::fabs(k) > 1.0 / m.size(c) * (1 + 1e-12)) R.bounded = false;
        if (f[c] <= EPS || f[c] >= 1 - EPS) continue;
        ++R.nIfc;
        if (m.level(c) != level) continue;
        const Vec2 x = m.center(c);
        const double e = std::fabs(k - S.kappa(x)) / kref;
        s2 += e * e;
        R.linf = std::max(R.linf, e);
        if (x.y < m.ymin() + 3 * hmax || x.x < m.xmin() + 3 * hmax) R.linfWall = std::max(R.linfWall, e);
    }
    R.l2 = std::sqrt(s2 / std::max(R.nIfc, 1));
    for (const Mesh::Face& F : m.faces()) {
        if (F.L < 0 || F.R < 0 || f[F.L] == f[F.R]) continue;
        ++R.jumpFaces;
        const bool a = kappa[F.L] != 0.0, b = kappa[F.R] != 0.0;
        if (a || b) ++R.jumpOneSide;
        if (a && b) ++R.jumpBoth;
    }
    return R;
}

void printRow(int level, const Result& R, double prevL2, double prevLinf) {
    std::printf("  L=%d leaves=%7d ifc=%5d  L2=%.3e Linf=%.3e", level, R.nLeaves, R.nIfc, R.l2, R.linf);
    if (prevL2 > 0) std::printf(" ord(L2)=%5.2f ord(Linf)=%5.2f", std::log2(prevL2 / R.l2), std::log2(prevLinf / R.linf));
    else std::printf("                              ");
    std::printf("  HF=%ld/%ld mix=%ld cen=%ld nb=%ld none=%ld  %.2f ms\n", R.st[1], R.st[2], R.st[3], R.st[4],
                R.st[5], R.st[6], 1e3 * R.tcall);
}

void checkCommon(const Result& R, const char* name) {
    char buf[256];
    std::snprintf(buf, sizeof buf, "%s: NaN/Inf in kappa", name);
    require(R.finite, buf);
    std::snprintf(buf, sizeof buf, "%s: |kappa| > 1/h", name);
    require(R.bounded, buf);
    std::snprintf(buf, sizeof buf, "%s: face with f jump and kappa = 0 on both sides", name);
    require(R.jumpOneSide == R.jumpFaces, buf);
}

// Convergence study; returns the results per level.
std::vector<Result> study(const char* name, const Shape& S, double kref, int l0, int l1, bool perX = false) {
    std::printf("%s\n", name);
    std::vector<Result> res;
    for (int L = l0; L <= l1; ++L) {
        Result R = run(S, L, kref, perX);
        printRow(L, R, res.empty() ? 0 : res.back().l2, res.empty() ? 0 : res.back().linf);
        checkCommon(R, name);
        res.push_back(R);
    }
    const Result& B = res.back();
    std::printf("  jump faces: %ld, kappa != 0 on both sides: %ld (%.1f%%)\n", B.jumpFaces, B.jumpBoth,
                100.0 * B.jumpBoth / std::max(B.jumpFaces, 1L));
    return res;
}

double order(const std::vector<Result>& r, int k0) {  // mean L2 order from index k0 to the end
    return std::log2(r[k0].l2 / r.back().l2) / (double)(r.size() - 1 - k0);
}

}  // namespace

int main() {
    auto T0 = std::chrono::steady_clock::now();

    // ---- drops (phase 1 inside), several radii / off-grid centres
    {
        const double cs[3][3] = {{0.5123, 0.4871, 0.25}, {0.4637, 0.5291, 0.3}, {0.531, 0.4583, 0.2}};
        for (auto& c : cs) {
            char name[128];
            std::snprintf(name, sizeof name, "circle R=%.2f at (%.4f, %.4f): |kappa R - 1|", c[2], c[0], c[1]);
            Ellipse E(c[0], c[1], c[2], c[2]);
            auto r = study(name, E, 1.0 / c[2], 5, 9);
            const double o = order(r, 2);
            std::printf("  mean L2 order (levels 7-9) = %.2f\n", o);
            require(r.back().linf < 2e-3, "circle: Linf at level 9 >= 2e-3");
            require(o > 1.6, "circle: L2 order < 1.6");
            require(r.back().st[1] + r.back().st[2] >= 0.97 * r.back().st[0], "circle: < 97% HF at level 9");
        }
    }
    // ---- bubble (phase 2 inside): kappa = -1/R
    {
        Ellipse E(0.4713, 0.5319, 0.22, 0.22, true);
        auto r = study("bubble R=0.22 at (0.4713, 0.5319): |kappa R + 1|", E, 1.0 / 0.22, 5, 9);
        const double o = order(r, 2);
        std::printf("  mean L2 order (levels 7-9) = %.2f\n", o);
        require(r.back().linf < 2e-3, "bubble: Linf at level 9 >= 2e-3");
        require(o > 1.6, "bubble: L2 order < 1.6");
    }
    // ---- ellipse (exact kappa on the column/row of the cell centre)
    {
        Ellipse E(0.5071, 0.4932, 0.32, 0.18);
        const double kmax = 0.32 / (0.18 * 0.18);
        auto r = study("ellipse a=0.32 b=0.18: |kappa - kex| / kmax", E, kmax, 5, 9);
        std::printf("  mean L2 order (levels 7-9) = %.2f\n", order(r, 2));
        require(r.back().linf < 2e-3, "ellipse: Linf at level 9 >= 2e-3");
        require(order(r, 2) > 1.6, "ellipse: L2 order < 1.6");
    }
    // ---- sine wave (periodic in x), exact kappa at the column of the cell centre
    {
        Wave Wv(0.5037, 0.05);
        const double kmax = 4 * M_PI * M_PI * 0.05;
        auto r = study("sine wave y = 0.5037 + 0.05 cos(2 pi x): |kappa - kex| / kmax", Wv, kmax, 5, 9, true);
        const double o = order(r, 2);
        std::printf("  mean L2 order (levels 7-9) = %.2f\n", o);
        require(r.back().linf < 2e-3, "wave: Linf at level 9 >= 2e-3");
        require(o > 1.6, "wave: L2 order < 1.6");
    }
    // ---- drops on walls (mirroring by Mesh::locate = 90 degree contact angle)
    {
        Ellipse E(0.5137, 0.0, 0.25, 0.25);
        auto r = study("half drop on the bottom wall, R=0.25: |kappa R - 1|", E, 4.0, 5, 9);
        std::printf("  Linf within 3 cells of the wall (level 9) = %.3e, mean L2 order = %.2f\n", r.back().linfWall,
                    order(r, 2));
        require(r.back().linf < 2e-3, "wall drop: Linf at level 9 >= 2e-3");
        require(r.back().linfWall < 2e-3, "wall drop: Linf near wall >= 2e-3");
        Ellipse Q(0.0, 0.0, 0.4, 0.4);
        auto q = study("quarter drop in the corner, R=0.4: |kappa R - 1|", Q, 2.5, 5, 9);
        std::printf("  Linf within 3 cells of the walls (level 9) = %.3e\n", q.back().linfWall);
        require(q.back().linf < 2e-3, "corner drop: Linf at level 9 >= 2e-3");
    }
    // ---- under-resolved drops (R = 1.5 .. 4 cells): must stay finite and bounded
    {
        std::printf("under-resolved drops (level 6, h = 1/64)\n");
        for (double Rc : {0.7, 1.5, 2.5, 4.0}) {
            Ellipse E(0.5123, 0.4871, Rc / 64, Rc / 64);
            Result R = run(E, 6, 64.0 / Rc, false);
            std::printf("  R=%.1fh: ifc=%d Linf(kappa R - 1)=%.3f  HF=%ld/%ld mix=%ld cen=%ld nb=%ld none=%ld\n", Rc,
                        R.nIfc, R.linf, R.st[1], R.st[2], R.st[3], R.st[4], R.st[5], R.st[6]);
            require(R.finite && R.bounded, "small drop: NaN/Inf or unbounded");
        }
    }
    // ---- robustness: random fractions on an adapted mesh
    {
        Mesh::Config cfg;
        cfg.minLevel = 3;
        cfg.maxLevel = 7;
        Mesh m(cfg);
        m.refineWhere([](const Mesh& mm, int c) {
            const Vec2 x = mm.center(c);
            return std::fabs(x.x - 0.5) < 0.3 && std::fabs(x.y - 0.5) < 0.3;
        });
        std::vector<double> f(m.nCells(), 0.0), kappa;
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        bool okAll = true;
        for (int it = 0; it < 20; ++it) {
            for (int c : m.leaves()) {
                const double r = U(rng);
                f[c] = r < 0.4 ? 0.0 : (r < 0.7 ? 1.0 : U(rng));
                if (it % 2) f[c] = (m.center(c).x < 0.5) ? f[c] : std::round(f[c]);
            }
            curvature::compute(m, f, kappa);
            for (int c = 0; c < m.nCells(); ++c)
                if (!std::isfinite(kappa[c]) || (m.isLeaf(c) && std::fabs(kappa[c]) > 1.0 / m.size(c) * (1 + 1e-12)))
                    okAll = false;
        }
        long st[8];
        curvature::lastStats(st);
        std::printf("random fields: ifc=%ld HF=%ld/%ld mix=%ld cen=%ld nb=%ld none=%ld -> %s\n", st[0], st[1], st[2],
                    st[3], st[4], st[5], st[6], okAll ? "finite, bounded" : "BAD");
        require(okAll, "random fields: NaN/Inf or |kappa| > 1/h");
    }
    // ---- rejectUnresolved: a sub-cell droplet gets no curvature (by default the methods
    //      clamp to |kappa h| = 1, with mixed signs); resolved interfaces are unchanged
    {
        Mesh::Config cfg;
        cfg.minLevel = 7;
        cfg.maxLevel = 7;
        Mesh m(cfg);
        const double h = m.size(m.leaves()[0]);
        auto disk = [&](double xc, double yc, double r) {
            std::vector<double> f(m.nCells(), 0.0);
            constexpr int N = 64;  // sub-samples per direction
            for (int c : m.leaves()) {
                const Vec2 x = m.center(c);
                if (std::hypot(x.x - xc, x.y - yc) > r + h) continue;
                int in = 0;
                for (int b = 0; b < N; ++b)
                    for (int a = 0; a < N; ++a)
                        in += std::hypot(x.x + ((a + 0.5) / N - 0.5) * h - xc, x.y + ((b + 0.5) / N - 0.5) * h - yc) < r;
                f[c] = double(in) / (N * N);
            }
            return f;
        };
        // (a position where the default methods hit the clamp; at others they give smaller
        //  wrong values, which rejectUnresolved does not remove)
        const std::vector<double> small = disk(0.5 + 0.3 * h, 0.5 + 0.3 * h, 0.4 * h);
        std::vector<double> k0, k1;
        curvature::compute(m, small, k0);
        curvature::compute(m, small, k1, true);
        double max0 = 0.0, max1 = 0.0;
        for (int c : m.leaves()) {
            max0 = std::max(max0, std::fabs(k0[c]) * h);
            max1 = std::max(max1, std::fabs(k1[c]) * h);
        }
        const std::vector<double> big = disk(0.5013, 0.4987, 8 * h);
        curvature::compute(m, big, k0);
        curvature::compute(m, big, k1, true);
        bool same = true;
        for (int c : m.leaves()) same = same && k0[c] == k1[c];
        std::printf("droplet r = 0.4 h: max |kappa h| %.2f (default), %.2f (rejectUnresolved); r = 8 h: %s\n", max0,
                    max1, same ? "identical" : "DIFFERENT");
        require(max0 > 0.99, "default: sub-cell droplet hits the curvature clamp");
        require(max1 == 0.0, "rejectUnresolved: sub-cell droplet gets no curvature");
        require(same, "rejectUnresolved: resolved droplet unchanged");
    }
    // ---- timing on a large uniform mesh
    {
        Ellipse E(0.5123, 0.4871, 0.25, 0.25);
        Result R = run(E, 10, 4.0, false, true);
        std::printf("uniform level 10 (%d leaves, %d interfacial): %.2f ms per call, Linf=%.2e\n", R.nLeaves, R.nIfc,
                    1e3 * R.tcall, R.linf);
        checkCommon(R, "uniform level 10");
    }

    std::printf("total %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count());
    std::printf(gFails ? "CURVATURE TEST FAILED (%d)\n" : "CURVATURE TEST PASSED\n", gFails);
    return gFails != 0;
}
