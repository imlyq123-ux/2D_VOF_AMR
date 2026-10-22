// test_io.cpp - configuration parsing, test-case definitions (exact initial areas),
// VTK output files and integral diagnostics.
//
//   test_io [output_dir]     (default: <system temp dir>/vof2d_test_io)
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "../src/cases.hpp"
#include "../src/io.hpp"

namespace fs = std::filesystem;

static int fails = 0;
#define CHECK(cond, ...)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            std::printf("FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); \
            std::printf(__VA_ARGS__);                        \
            std::printf("\n");                               \
            ++fails;                                         \
        }                                                    \
    } while (0)

static std::string dump(const Params& p) {
    std::ostringstream os;
    io::printParams(p, os);
    return os.str();
}

static bool throws(const io::ConfigMap& map, std::string* msg = nullptr) {
    Params p;
    try {
        io::applyConfig(map, p);
    } catch (const std::runtime_error& e) {
        if (msg) *msg = e.what();
        return true;
    }
    return false;
}

// Mesh for prm's domain refined to maxLevel around the interface of phi (plus a band of
// `buffer` max-level cells where |phi| is small, as the AMR criterion would do).
static std::unique_ptr<Mesh> caseMesh(const Params& prm, const std::function<double(double, double)>& phi,
                                      double buffer = 0.0) {
    Mesh::Config cfg;
    cfg.nx0 = prm.nx0; cfg.ny0 = prm.ny0; cfg.x0 = prm.x0; cfg.y0 = prm.y0; cfg.L0 = prm.L0;
    cfg.minLevel = prm.minLevel; cfg.maxLevel = prm.maxLevel;
    auto m = std::make_unique<Mesh>(cfg);
    m->refineWhere([&](const Mesh& mm, int c) {
        const Vec2 x = mm.center(c);
        if (std::fabs(phi(x.x, x.y)) < buffer * mm.h(mm.maxLevel()) + mm.size(c)) return true;
        const double fr = cases::cellFraction(mm, c, phi);
        return fr > 0.0 && fr < 1.0;
    });
    return m;
}

static void testConfig(const fs::path& dir) {
    // ---- file parsing: comments, inline comments, blank lines, BC keys
    const fs::path cfgFile = dir / "test.cfg";
    {
        std::ofstream os(cfgFile);
        os << "# a comment line\n\n"
              "caseName = static_drop   # inline comment\n"
              "  maxLevel=9\n"
              "rho1 = 2.5e3\n"
              "sigma = 0.07\n"
              "implicitViscosity = false\n"
              "vofOnly = 1\n"
              "outDir = \"out dir\"\n"
              "bc_left = periodic\n"
              "bc_right = Periodic\n"
              "bc_top = inflow\n"
              "bc_top_u = 0.5\n"
              "bc_top_v = -1.25\n"
              "bc_top_f = 1\n"
              "bc_bottom = outflow\n";
    }
    io::ConfigMap map = io::readConfigFile(cfgFile.string());
    const char* argv[] = {"vof2d", "x.cfg", "maxLevel=10", "--gy=-9.81", "bc_bottom=slip"};
    io::parseOverrides(5, const_cast<char**>(argv), 2, map);
    Params p;
    io::applyConfig(map, p);
    CHECK(p.caseName == "static_drop", "%s", p.caseName.c_str());
    CHECK(p.outDir == "out dir", "%s", p.outDir.c_str());
    CHECK(p.maxLevel == 10, "%d", p.maxLevel);
    CHECK(p.rho1 == 2500.0 && p.sigma == 0.07 && p.gy == -9.81, "%g %g %g", p.rho1, p.sigma, p.gy);
    CHECK(!p.implicitViscosity && p.vofOnly, "bools");
    CHECK(p.bc[LEFT].type == BCType::Periodic && p.bc[RIGHT].type == BCType::Periodic, "periodic");
    CHECK(p.bc[TOP].type == BCType::Inflow && p.bc[TOP].u == 0.5 && p.bc[TOP].v == -1.25 && p.bc[TOP].f == 1.0,
          "inflow");
    CHECK(p.bc[BOTTOM].type == BCType::Slip, "override of bc_bottom");

    // ---- errors
    std::string msg;
    CHECK(throws({{"maxlevel", "3"}}, &msg) && msg.find("maxLevel") != std::string::npos, "%s", msg.c_str());
    CHECK(throws({{"noSuchKey", "3"}}, &msg) && msg.find("noSuchKey") != std::string::npos, "%s", msg.c_str());
    CHECK(throws({{"rho1", "abc"}}), "bad double");
    CHECK(throws({{"rho1", "1.0x"}}), "trailing garbage");
    CHECK(throws({{"maxLevel", "7.5"}}), "bad int");
    CHECK(throws({{"vofOnly", "maybe"}}), "bad bool");
    CHECK(throws({{"bc_left", "sticky"}}), "bad bc");
    {
        std::ofstream os(dir / "bad.cfg");
        os << "rho1 = 1\nthis line has no equals sign\n";
    }
    bool threw = false;
    try { io::readConfigFile((dir / "bad.cfg").string()); } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find(":2:") != std::string::npos;
    }
    CHECK(threw, "line without '=' must throw with the line number");
    threw = false;
    try {
        const char* bad[] = {"vof2d", "maxLevel"};
        io::ConfigMap mm;
        io::parseOverrides(2, const_cast<char**>(bad), 1, mm);
    } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw, "override without '='");

    // ---- round trip: printParams output is itself a valid config file
    Params q = p;
    q.dtMin = 1.0 / 3.0;
    q.uTol = 0.1;
    q.bc[RIGHT].u = 1e-300;
    q.period = M_PI;
    {
        std::ofstream os(dir / "roundtrip.cfg");
        io::printParams(q, os);
    }
    Params r;
    io::applyConfig(io::readConfigFile((dir / "roundtrip.cfg").string()), r);
    CHECK(dump(q) == dump(r), "printParams/readConfigFile/applyConfig round trip");
    CHECK(r.dtMin == 1.0 / 3.0 && r.period == M_PI && r.bc[RIGHT].u == 1e-300, "exact doubles");
    std::printf("config: ok (%zu keys)\n", io::readConfigFile((dir / "roundtrip.cfg").string()).size());
}

static void testCases() {
    const double a = 0.05715, w = 0.025, R = 0.15;
    const double slot = 0.1 * 2 * w + w * std::sqrt(R * R - w * w) + R * R * std::asin(w / R);
    const std::map<std::string, double> area1 = {
        {"rising_bubble", 2.0 - M_PI / 16},  {"rising_bubble2", 2.0 - M_PI / 16},
        {"static_drop", M_PI * 0.04},        {"oscillating_drop", M_PI * 0.04 * 1.005},
        {"rayleigh_taylor", 2.0},
        {"dam_break", 2 * a * a},            {"breaking_wave", 2 * M_PI * M_PI},
        {"zalesak", M_PI * R * R - slot},
        {"single_vortex", M_PI * R * R}};
    CHECK(cases::caseNames().size() == area1.size(), "number of cases");
    {
        Params p;
        cases::CaseSetup cs;
        CHECK(!cases::setupCase("no_such_case", p, cs), "unknown case must return false");
    }
    for (const std::string& name : cases::caseNames()) {
        const auto t0 = std::chrono::steady_clock::now();
        Params prm;
        prm.sigma = 123.0;   // must be overwritten by every case
        cases::CaseSetup cs;
        CHECK(cases::setupCase(name, prm, cs), "%s", name.c_str());
        CHECK(prm.caseName == name, "%s", name.c_str());
        CHECK(prm.sigma != 123.0, "%s: sigma not set", name.c_str());
        CHECK((bool)cs.phi, "%s: phi", name.c_str());
        CHECK(!prm.vofOnly || (bool)cs.psi, "%s: vofOnly needs psi", name.c_str());
        const auto m = caseMesh(prm, cs.phi);
        CHECK(m->checkConsistency().empty(), "%s mesh", name.c_str());
        double A = 0.0;
        int nmixed = 0;
        for (int c : m->leaves()) {
            const double fr = cases::cellFraction(*m, c, cs.phi);
            A += fr * m->volume(c);
            nmixed += fr > 0 && fr < 1;
        }
        const double ex = area1.at(name), err = std::fabs(A - ex) / ex;
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("case %-16s leaves %7d mixed %5d area1 %.10f exact %.10f rel.err %.2e (%.0f ms)\n",
                    name.c_str(), m->nLeaves(), nmixed, A, ex, err, ms);
        CHECK(err < 2e-7, "%s: relative area error %.3e", name.c_str(), err);

        // cases/<name>.cfg (if found) must reproduce the defaults exactly
        for (const char* d : {"cases", "../cases", "../../cases"}) {
            const fs::path f = fs::path(d) / (name + ".cfg");
            if (!fs::exists(f)) continue;
            Params q = prm;
            io::applyConfig(io::readConfigFile(f.string()), q);
            CHECK(dump(q) == dump(prm), "%s does not match the case defaults", f.string().c_str());
            break;
        }
    }
    // prescribed flows: u = dpsi/dy, v = -dpsi/dx
    {
        Params prm;
        cases::CaseSetup cs;
        cases::setupCase("zalesak", prm, cs);
        const double e = 1e-6, x = 0.3, y = 0.8;
        const double u = (cs.psi(x, y + e, 0) - cs.psi(x, y - e, 0)) / (2 * e);
        const double v = -(cs.psi(x + e, y, 0) - cs.psi(x - e, y, 0)) / (2 * e);
        CHECK(std::fabs(u + 2 * M_PI * (y - 0.5)) < 1e-6 && std::fabs(v - 2 * M_PI * (x - 0.5)) < 1e-6,
              "zalesak rotation u=%g v=%g", u, v);
        cases::setupCase("single_vortex", prm, cs);
        CHECK(std::fabs(cs.psi(0.5, 0.5, 4.0)) < 1e-15 && cs.psi(0.5, 0.5, 0.0) > 0.3, "vortex reversal");
    }
    // a pure cell and a half cell
    {
        Mesh::Config cfg;
        cfg.minLevel = 2; cfg.maxLevel = 2;
        Mesh m(cfg);
        const int c = m.find(2, 1, 1);   // [0.25,0.5]^2
        CHECK(cases::cellFraction(m, c, [](double, double) { return 1.0; }) == 1.0, "pure");
        const double fr = cases::cellFraction(m, c, [](double x, double y) { return 0.75 - x - y; });
        CHECK(std::fabs(fr - 0.5) < 1e-12, "diagonal half cell %.15f", fr);
        const double fr2 = cases::cellFraction(m, c, [](double x, double) { return x - 0.3; });
        CHECK(std::fabs(fr2 - 0.8) < 1e-12, "vertical line %.15f", fr2);
    }
}

// State of the rising bubble case (bubble R = 0.25 centred at xb) with a uniform or
// rotating velocity.
static void buildState(State& s, Vec2 U, bool rotation, int maxLevel = 6, Vec2 xb = Vec2(0.5, 0.5)) {
    cases::CaseSetup cs;
    cases::setupCase("rising_bubble", s.prm, cs);
    cs.phi = [xb](double x, double y) { return std::hypot(x - xb.x, y - xb.y) - 0.25; };
    s.prm.maxLevel = maxLevel;
    s.mesh = caseMesh(s.prm, cs.phi, 4.0);
    Mesh& m = s.m();
    s.id.f = m.addField("f");
    s.id.u = m.addField("u");
    s.id.v = m.addField("v");
    s.id.p = m.addField("p");
    s.id.rho = m.addField("rho");
    s.id.mu = m.addField("mu");
    s.id.kappa = m.addField("kappa");
    for (int c : m.leaves()) {
        const double f = cases::cellFraction(m, c, cs.phi);
        const Vec2 x = m.center(c);
        s.F(s.id.f)[c] = f;
        s.F(s.id.u)[c] = rotation ? -(x.y - 1.0) : U.x;
        s.F(s.id.v)[c] = rotation ? (x.x - 0.5) : U.y;
        s.F(s.id.p)[c] = x.y;
        s.F(s.id.rho)[c] = f * s.prm.rho1 + (1 - f) * s.prm.rho2;
        s.F(s.id.mu)[c] = f * s.prm.mu1 + (1 - f) * s.prm.mu2;
        s.F(s.id.kappa)[c] = (f > 0 && f < 1) ? -4.0 : 0.0;
    }
    s.t = 0.125;
    s.dt = 1e-3;
    s.step = 7;
}

static void testOutput(const fs::path& dir) {
    State s;
    buildState(s, Vec2(), true);   // solid-body rotation: vorticity = 2
    const std::string vtu = (dir / "fields_00000.vtu").string(), vtp = (dir / "interface_00000.vtp").string();
    const auto t0 = std::chrono::steady_clock::now();
    io::writeVTU(s, vtu);
    io::writeInterfaceVTP(s, vtp);
    io::writePVD((dir / "fields.pvd").string(), {{0.0, "fields_00000.vtu"}, {0.125, "fields_00001.vtu"}});
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (const std::string& f : {vtu, vtp, (dir / "fields.pvd").string()}) {
        CHECK(fs::exists(f) && fs::file_size(f) > 200, "%s", f.c_str());
        std::ifstream in(f, std::ios::binary);
        std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        CHECK(all.rfind("<?xml", 0) == 0 && all.find("</VTKFile>") != std::string::npos, "%s", f.c_str());
    }
    std::printf("output: %d leaves, vtu %ju bytes, vtp %ju bytes (%.0f ms)\n", s.m().nLeaves(),
                (uintmax_t)fs::file_size(vtu), (uintmax_t)fs::file_size(vtp), ms);
}

static void testDiagnostics(const fs::path& dir) {
    // bubble (phase 2) R = 0.25 at a generic position xb in [0,1]x[0,2], uniform velocity
    State s;
    const Vec2 U(0.3, -0.2), xb(0.5123, 0.5371);
    buildState(s, U, false, 6, xb);
    const io::Diagnostics d = io::computeDiagnostics(s);
    // Note: a circle centred on grid nodes and tangent to grid lines (the exact Hysing
    // initial condition) gives circularity ~1 + 0.23 h/R: at the 4 tangent points the
    // two PLIC segments leave a gap of ~0.37 h each. Printed for information only.
    for (int L = 6; L <= 7; ++L) {
        State a, b;
        buildState(a, U, false, L, xb);
        buildState(b, U, false, L);
        std::printf("  circularity R/h = %d: generic position %.6f, grid-aligned %.6f\n", (1 << L) / 4,
                    io::computeDiagnostics(a).circularity, io::computeDiagnostics(b).circularity);
    }
    const double A2 = M_PI / 16, A1 = 2.0 - A2;
    const double xc1 = (0.5 * 2.0 - A2 * xb.x) / A1, yc1 = (2.0 * 1.0 - A2 * xb.y) / A1;
    const double ek = 0.5 * (s.prm.rho1 * A1 + s.prm.rho2 * A2) * (U.x * U.x + U.y * U.y);
    std::printf("diagnostics: vol2 %.10f (%.10f) xc2 %.10f yc2 %.10f yc1 %.10f (%.10f) circ %.6f\n", d.vol2, A2,
                d.xc2, d.yc2, d.yc1, yc1, d.circularity);
    CHECK(std::fabs(d.vol1 - A1) < 1e-7 && std::fabs(d.vol2 - A2) < 1e-7, "volumes %g %g", d.vol1, d.vol2);
    CHECK(std::fabs(d.xc2 - xb.x) < 2e-5 && std::fabs(d.yc2 - xb.y) < 2e-5, "bubble centroid");  // O(h^2): cell-centre moments
    CHECK(std::fabs(d.xc1 - xc1) < 2e-5 && std::fabs(d.yc1 - yc1) < 2e-5, "liquid centroid");
    CHECK(std::fabs(d.uc1 - U.x) < 1e-12 && std::fabs(d.vc1 - U.y) < 1e-12, "phase 1 velocity");
    CHECK(std::fabs(d.uc2 - U.x) < 1e-12 && std::fabs(d.vc2 - U.y) < 1e-12, "phase 2 velocity");
    CHECK(std::fabs(d.ekin - ek) < 1e-6 * ek, "ekin %g %g", d.ekin, ek);
    CHECK(std::fabs(d.umax - U.norm()) < 1e-12, "umax");
    CHECK(std::fabs(d.circularity - 1.0) < 2e-3, "circularity of a circle %g", d.circularity);
    CHECK(d.step == 7 && d.nLeaves == s.m().nLeaves() && d.t == 0.125, "counters");

    // header and line have the same number of columns
    std::ostringstream os;
    io::writeDiagnosticsHeader(os);
    io::writeDiagnosticsLine(os, d);
    std::istringstream is(os.str());
    std::string h, l, tok;
    std::getline(is, h);
    std::getline(is, l);
    CHECK(!h.empty() && h[0] == '#', "header starts with #");
    int nh = -1, nl = 0;   // '#' is the first header token
    for (std::istringstream hs(h); hs >> tok;) ++nh;
    for (std::istringstream ls(l); ls >> tok;) ++nl;
    CHECK(nh == nl && nh == 18, "columns header %d line %d", nh, nl);
    std::ofstream(dir / "diagnostics.dat") << os.str();
}

int main(int argc, char** argv) {
    const fs::path dir = argc > 1 ? fs::path(argv[1]) : fs::temp_directory_path() / "vof2d_test_io";
    fs::create_directories(dir);
    try {
        testConfig(dir);
        testCases();
        testOutput(dir);
        testDiagnostics(dir);
    } catch (const std::exception& e) {
        std::printf("FAIL: exception %s\n", e.what());
        ++fails;
    }
    std::printf("output files in %s\n", dir.string().c_str());
    std::printf(fails ? "IO TEST FAILED (%d)\n" : "IO TEST PASSED\n", fails);
    return fails != 0;
}
