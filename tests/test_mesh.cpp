// test_mesh.cpp - consistency of the quadtree mesh under random refine/coarsen.
#include <cstdio>
#include <random>
#include "../src/mesh.hpp"

static int check(const Mesh& m, const char* what) {
    std::string e = m.checkConsistency();
    if (!e.empty()) { std::printf("FAIL %s:\n%s", what, e.substr(0, 2000).c_str()); return 1; }
    return 0;
}

int main() {
    int fails = 0;
    for (int per = 0; per < 2; ++per) {
        Mesh::Config cfg; cfg.nx0 = 1; cfg.ny0 = 2; cfg.minLevel = 2; cfg.maxLevel = 8;
        cfg.periodicX = per; 
        Mesh m(cfg);
        int fid = m.addField("q");
        auto fn = [](Vec2 p) { return std::sin(3 * p.x) + p.y * p.y; };
        for (int c : m.leaves()) m.field(fid)[c] = fn(m.center(c));
        // refine around a circle
        m.refineWhere([](const Mesh& mm, int c) {
            Vec2 x = mm.center(c); double r = std::hypot(x.x - 0.5, x.y - 0.5);
            return std::fabs(r - 0.25) < 1.5 * mm.size(c);
        });
        fails += check(m, "circle refine");
        std::printf("periodic=%d leaves=%d cells=%d faces=%d\n", per, m.nLeaves(), m.nCells(), m.nFaces());
        // conservation of integral through random adapt cycles
        auto integral = [&]() { double s = 0; for (int c : m.leaves()) s += m.field(fid)[c] * m.volume(c); return s; };
        double I0 = integral();
        std::mt19937 rng(42 + per);
        for (int it = 0; it < 30; ++it) {
            std::vector<int> fl(m.nLeaves());
            for (auto& v : fl) { int r = rng() % 10; v = r < 2 ? 1 : (r < 8 ? -1 : 0); }
            m.adapt(fl);
            fails += check(m, "random adapt");
        }
        double I1 = integral();
        std::printf("  after random adapt: leaves=%d integral drift=%.3e\n", m.nLeaves(), I1 - I0);
        if (std::fabs(I1 - I0) > 1e-10) { std::printf("FAIL conservation\n"); fails++; }
        // divergence of a discretely div-free face field from a stream function
        auto psi = [](double x, double y) { return std::sin(M_PI * x) * std::sin(M_PI * y); };
        std::vector<double> uf(m.nFaces());
        for (int f = 0; f < m.nFaces(); ++f) {
            const auto& F = m.faces()[f]; double a = 0.5 * F.area;
            uf[f] = F.dir == 0 ? (psi(F.x.x, F.x.y + a) - psi(F.x.x, F.x.y - a)) / F.area
                               : -(psi(F.x.x + a, F.x.y) - psi(F.x.x - a, F.x.y)) / F.area;
        }
        double maxdiv = 0;
        for (int c : m.leaves()) {
            double d = 0;
            for (const int* f = m.cellFacesBegin(c); f != m.cellFacesEnd(c); ++f) {
                const auto& F = m.faces()[*f]; d += (F.L == c ? 1 : -1) * uf[*f] * F.area;
            }
            maxdiv = std::max(maxdiv, std::fabs(d) / m.volume(c));
        }
        std::printf("  max |div| of stream-function face field = %.3e\n", maxdiv);
        if (!per && maxdiv > 1e-9) { std::printf("FAIL divergence\n"); fails++; }
    }
    // deep mesh: levels above 2^20 positions use the sparse (tiled) index maps
    {
        Mesh::Config cfg; cfg.nx0 = 1; cfg.ny0 = 2; cfg.minLevel = 3; cfg.maxLevel = 12;
        Mesh m(cfg);
        int fid = m.addField("q");
        for (int c : m.leaves()) m.field(fid)[c] = m.center(c).x;
        m.refineWhere([](const Mesh& mm, int c) {
            Vec2 x = mm.center(c); double r = std::hypot(x.x - 0.5, x.y - 0.7);
            return std::fabs(r - 0.25) < 2.0 * mm.size(c);
        });
        fails += check(m, "deep refine");
        auto integral = [&]() { double s = 0; for (int c : m.leaves()) s += m.field(fid)[c] * m.volume(c); return s; };
        const double I0 = integral();
        int deep = 0;
        for (int c : m.leaves()) deep += m.level(c) == 12;
        // every finest leaf is found by (L, i, j), and locate() of a finer index returns it
        for (int c : m.leaves())
            if (m.find(m.level(c), m.ix(c), m.iy(c)) != c || m.locate(12, m.ix(c) << (12 - m.level(c)),
                                                                       m.iy(c) << (12 - m.level(c))) != c) {
                std::printf("FAIL deep lookup\n");
                fails++;
                break;
            }
        std::mt19937 rng(7);
        for (int it = 0; it < 6; ++it) {
            std::vector<int> fl(m.nLeaves());
            for (auto& v : fl) { int r = rng() % 10; v = r < 2 ? 1 : (r < 8 ? -1 : 0); }
            m.adapt(fl);
            fails += check(m, "deep random adapt");
        }
        const double drift = integral() - I0;
        std::printf("deep mesh: %d finest leaves, after random adapt leaves=%d integral drift=%.3e\n", deep,
                    m.nLeaves(), drift);
        if (deep == 0 || std::fabs(drift) > 1e-10) { std::printf("FAIL deep mesh\n"); fails++; }
    }
    std::printf(fails ? "MESH TEST FAILED\n" : "MESH TEST PASSED\n");
    return fails != 0;
}
