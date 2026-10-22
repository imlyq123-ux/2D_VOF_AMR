// io.cpp - configuration, VTK output and integral diagnostics (see io.hpp).
#include "io.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include "vof.hpp"

namespace io {

namespace {

// ------------------------------------------------------------------ parameter table

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

std::string lower(std::string s) {
    for (char& ch : s) ch = (char)std::tolower((unsigned char)ch);
    return s;
}

const char* bcName(BCType t) {
    switch (t) {
        case BCType::Wall: return "wall";
        case BCType::Slip: return "slip";
        case BCType::Inflow: return "inflow";
        case BCType::Outflow: return "outflow";
        case BCType::Periodic: return "periodic";
    }
    return "wall";
}

// value parsers (throw std::invalid_argument with a short reason)
void parseValue(const std::string& v, double& out) {
    size_t pos = 0;
    double d = std::stod(v, &pos);
    if (pos != v.size()) throw std::invalid_argument("not a number");
    out = d;
}
void parseValue(const std::string& v, int& out) {
    size_t pos = 0;
    long d = std::stol(v, &pos);
    if (pos != v.size()) throw std::invalid_argument("not an integer");
    out = (int)d;
}
void parseValue(const std::string& v, bool& out) {
    const std::string s = lower(v);
    if (s == "1" || s == "true" || s == "yes" || s == "on") out = true;
    else if (s == "0" || s == "false" || s == "no" || s == "off") out = false;
    else throw std::invalid_argument("not a boolean (1/0/true/false)");
}
void parseValue(const std::string& v, std::string& out) {
    if (v.size() >= 2 && (v.front() == '"' || v.front() == '\'') && v.back() == v.front())
        out = v.substr(1, v.size() - 2);
    else
        out = v;
}
void parseValue(const std::string& v, BCType& out) {
    const std::string s = lower(v);
    if (s == "wall") out = BCType::Wall;
    else if (s == "slip") out = BCType::Slip;
    else if (s == "inflow") out = BCType::Inflow;
    else if (s == "outflow") out = BCType::Outflow;
    else if (s == "periodic") out = BCType::Periodic;
    else throw std::invalid_argument("not a BC type (wall|slip|inflow|outflow|periodic)");
}

// value formatters: shortest representation that reads back exactly
std::string formatValue(double d) {
    char buf[64];
    for (int prec = 6; prec <= 17; ++prec) {
        std::snprintf(buf, sizeof buf, "%.*g", prec, d);
        if (std::strtod(buf, nullptr) == d) break;
    }
    return buf;
}
std::string formatValue(int i) { return std::to_string(i); }
std::string formatValue(bool b) { return b ? "true" : "false"; }
std::string formatValue(const std::string& s) { return s; }
std::string formatValue(BCType t) { return bcName(t); }

// Visits every configurable member of Params: fn(section, key, member reference, help).
// `section` is non-null for the first entry of a group (used for printing).
template <class P, class Fn>
void forEachParam(P& p, Fn&& fn) {
    fn("case / output", "caseName", p.caseName, "test case (see cases.hpp / vof2d --help)");
    fn(nullptr, "outDir", p.outDir, "output directory");
    fn(nullptr, "outputEvery", p.outputEvery, "simulated time between VTK dumps (<= 0: never)");
    fn(nullptr, "diagEvery", p.diagEvery, "steps between diagnostics lines");

    fn("domain and mesh: nx0 x ny0 root cells of size L0, lower-left corner (x0, y0)", "nx0", p.nx0,
       "root cells in x");
    fn(nullptr, "ny0", p.ny0, "root cells in y");
    fn(nullptr, "x0", p.x0, "domain lower-left x");
    fn(nullptr, "y0", p.y0, "domain lower-left y");
    fn(nullptr, "L0", p.L0, "root cell size");
    fn(nullptr, "minLevel", p.minLevel, "coarsest leaf level (h = L0 / 2^level)");
    fn(nullptr, "maxLevel", p.maxLevel, "finest leaf level (interface region)");

    fn("fluids: phase 1 is where f = 1, phase 2 where f = 0", "rho1", p.rho1, "density of phase 1");
    fn(nullptr, "rho2", p.rho2, "density of phase 2");
    fn(nullptr, "mu1", p.mu1, "dynamic viscosity of phase 1");
    fn(nullptr, "mu2", p.mu2, "dynamic viscosity of phase 2");
    fn(nullptr, "sigma", p.sigma, "surface tension coefficient");
    fn(nullptr, "rejectUnresolvedCurvature", p.rejectUnresolvedCurvature,
       "no surface tension on sub-cell structures (true/false)");
    fn(nullptr, "gx", p.gx, "gravity x");
    fn(nullptr, "gy", p.gy, "gravity y");

    fn("time integration", "tEnd", p.tEnd, "final time");
    fn(nullptr, "cfl", p.cfl, "advective CFL number (<= 0.5)");
    fn(nullptr, "dtMax", p.dtMax, "maximum time step");
    fn(nullptr, "dtMin", p.dtMin, "abort if the time step falls below this");
    fn(nullptr, "maxSteps", p.maxSteps, "maximum number of time steps");
    fn(nullptr, "implicitViscosity", p.implicitViscosity, "implicit viscous term (true/false)");
    fn(nullptr, "conservingMomentum", p.conservingMomentum, "momentum advected with the VOF fluxes (true/false)");
    fn(nullptr, "macPredictor", p.macPredictor, "MAC-projected advection velocity at t+dt/2 (true/false)");

    static const char* sideKey[4] = {"bc_left", "bc_right", "bc_bottom", "bc_top"};
    for (int s = 0; s < 4; ++s) {
        const std::string k = sideKey[s];
        fn(s == 0 ? "boundary conditions: wall|slip|inflow|outflow|periodic; _u, _v: wall (lid) or "
                    "inflow velocity; _f: inflow volume fraction"
                  : nullptr,
           k, p.bc[s].type, "type");
        fn(nullptr, k + "_u", p.bc[s].u, "x velocity");
        fn(nullptr, k + "_v", p.bc[s].v, "y velocity");
        fn(nullptr, k + "_f", p.bc[s].f, "inflow volume fraction");
    }

    fn("adaptivity", "adaptEvery", p.adaptEvery, "steps between regrids");
    fn(nullptr, "interfaceBuffer", p.interfaceBuffer, "max-level cells kept around the interface");
    fn(nullptr, "uTol", p.uTol, "velocity wavelet threshold (absolute)");
    fn(nullptr, "adaptVelocity", p.adaptVelocity, "use the velocity criterion away from the interface");

    fn("linear solver", "poissonTol", p.poissonTol, "relative residual tolerance");
    fn(nullptr, "poissonMaxIter", p.poissonMaxIter, "maximum iterations");
    fn(nullptr, "gpu", p.gpu, "linear solves on the CUDA device (CUDA builds, true/false)");

    fn("kinematic runs (prescribed stream function, no Navier-Stokes)", "vofOnly", p.vofOnly,
       "advect f with the case's stream function only");
    fn(nullptr, "period", p.period, "period of time-reversing prescribed flows");
}

// ------------------------------------------------------------------ binary VTK writer

bool littleEndian() {
    const uint16_t one = 1;
    unsigned char b;
    std::memcpy(&b, &one, 1);
    return b == 1;
}

// Collects DataArrays for the "appended raw" VTK XML encoding: the XML part references
// each array by its byte offset into the appended section; each block is a UInt64 byte
// count followed by the raw little/big-endian values.
class Appended {
public:
    template <class T>
    std::string array(const char* vtkType, const std::string& name, int ncomp, const std::vector<T>& v) {
        std::ostringstream os;
        os << "<DataArray type=\"" << vtkType << "\" Name=\"" << name << "\"";
        if (ncomp > 1) os << " NumberOfComponents=\"" << ncomp << "\"";
        os << " format=\"appended\" offset=\"" << data_.size() << "\"/>\n";
        const uint64_t nb = (uint64_t)v.size() * sizeof(T);
        const char* p = reinterpret_cast<const char*>(&nb);
        data_.insert(data_.end(), p, p + sizeof nb);
        const char* q = reinterpret_cast<const char*>(v.data());
        data_.insert(data_.end(), q, q + nb);
        return os.str();
    }
    void write(std::ostream& os) const {
        os << "<AppendedData encoding=\"raw\">\n_";
        os.write(data_.data(), (std::streamsize)data_.size());
        os << "\n</AppendedData>\n";
    }

private:
    std::vector<char> data_;
};

std::string fileHeader(const char* type) {
    std::ostringstream os;
    os << "<?xml version=\"1.0\"?>\n<VTKFile type=\"" << type << "\" version=\"1.0\" byte_order=\""
       << (littleEndian() ? "LittleEndian" : "BigEndian") << "\" header_type=\"UInt64\">\n";
    return os.str();
}

std::string timeFieldData(double t) {
    std::ostringstream os;
    os << "<FieldData>\n<DataArray type=\"Float64\" Name=\"TimeValue\" NumberOfTuples=\"1\" "
          "format=\"ascii\">"
       << formatValue(t) << "</DataArray>\n</FieldData>\n";
    return os.str();
}

std::ofstream openOut(const std::string& filename, bool binary) {
    std::ofstream os(filename, binary ? std::ios::binary : std::ios::out);
    if (!os) throw std::runtime_error("cannot open '" + filename + "' for writing");
    return os;
}

// Leaf field values (zeros if the id is invalid).
std::vector<double> leafValues(const State& s, int id) {
    const Mesh& m = s.m();
    std::vector<double> out(m.nLeaves(), 0.0);
    if (id < 0 || id >= m.nFields()) return out;
    const auto& d = m.field(id);
    for (int k = 0; k < m.nLeaves(); ++k) out[k] = d[m.leaves()[k]];
    return out;
}

// Restricted copy of a field (parents valid), or zeros for an invalid id.
std::vector<double> restricted(const State& s, int id) {
    const Mesh& m = s.m();
    if (id < 0 || id >= m.nFields()) return std::vector<double>(m.nCells(), 0.0);
    std::vector<double> d = m.field(id);
    m.restrict(d);
    return d;
}

// Vorticity dv/dx - du/dy at the leaves, from neighbours at the leaf's own level
// (Mesh::locate; a coarser neighbour is used at its own centre). Neighbours across a
// non-periodic boundary are ignored (one-sided difference).
std::vector<double> vorticity(const State& s) {
    const Mesh& m = s.m();
    std::vector<double> w(m.nLeaves(), 0.0);
    if (s.id.u < 0 || s.id.v < 0) return w;
    const std::vector<double> u = restricted(s, s.id.u), v = restricted(s, s.id.v);
    const double Lx = m.xmax() - m.xmin(), Ly = m.ymax() - m.ymin();
#pragma omp parallel for schedule(static)
    for (int k = 0; k < m.nLeaves(); ++k) {
        const int c = m.leaves()[k];
        const int L = m.level(c), i = m.ix(c), j = m.iy(c);
        const Vec2 xc = m.center(c);
        // derivative of q along direction d (0: x, 1: y)
        auto deriv = [&](const std::vector<double>& q, int d) {
            int cm = c, cp = c, mir = 0;
            double xm = 0.0, xp = 0.0;  // coordinates relative to xc
            int n = m.locate(L, i - (d == 0), j - (d == 1), &mir);
            if (!(mir & (1 << d))) {
                cm = n;
                xm = d == 0 ? m.center(n).x - xc.x : m.center(n).y - xc.y;
                if (xm > 0) xm -= d == 0 ? Lx : Ly;   // periodic wrap
            }
            n = m.locate(L, i + (d == 0), j + (d == 1), &mir);
            if (!(mir & (1 << d))) {
                cp = n;
                xp = d == 0 ? m.center(n).x - xc.x : m.center(n).y - xc.y;
                if (xp < 0) xp += d == 0 ? Lx : Ly;
            }
            return xp > xm ? (q[cp] - q[cm]) / (xp - xm) : 0.0;
        };
        w[k] = deriv(v, 0) - deriv(u, 1);
    }
    return w;
}

}  // namespace

// ------------------------------------------------------------------ configuration

ConfigMap readConfigFile(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open config file '" + path + "'");
    ConfigMap map;
    std::string line;
    int lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        line = trim(line);
        if (line.empty()) continue;
        const size_t eq = line.find('=');
        const std::string key = eq == std::string::npos ? "" : trim(line.substr(0, eq));
        if (key.empty())
            throw std::runtime_error(path + ":" + std::to_string(lineNo) + ": expected 'key = value', got '" +
                                     line + "'");
        map[key] = trim(line.substr(eq + 1));
    }
    return map;
}

void parseOverrides(int argc, char** argv, int first, ConfigMap& map) {
    for (int a = first; a < argc; ++a) {
        std::string tok = argv[a];
        while (!tok.empty() && tok[0] == '-') tok.erase(0, 1);   // tolerate --key=value
        const size_t eq = tok.find('=');
        const std::string key = eq == std::string::npos ? "" : trim(tok.substr(0, eq));
        if (key.empty())
            throw std::runtime_error(std::string("command line: expected key=value, got '") + argv[a] + "'");
        map[key] = trim(tok.substr(eq + 1));
    }
}

void applyConfig(const ConfigMap& map, Params& prm) {
    std::map<std::string, std::function<void(const std::string&)>> setters;
    forEachParam(prm, [&](const char*, const std::string& key, auto& ref, const char*) {
        auto* p = &ref;
        setters[key] = [p](const std::string& v) { parseValue(v, *p); };
    });
    for (const auto& kv : map) {
        auto it = setters.find(kv.first);
        if (it == setters.end()) {
            // suggest keys that match case-insensitively
            std::string hint;
            for (const auto& s : setters)
                if (lower(s.first) == lower(kv.first)) hint = " (did you mean '" + s.first + "'?)";
            throw std::runtime_error("unknown parameter '" + kv.first + "'" + hint +
                                     "; see outDir/params.log or cases/*.cfg for the valid keys");
        }
        try {
            it->second(trim(kv.second));
        } catch (const std::exception& e) {
            throw std::runtime_error("bad value '" + kv.second + "' for parameter '" + kv.first + "': " +
                                     e.what());
        }
    }
}

void printParams(const Params& prm, std::ostream& os) {
    forEachParam(prm, [&](const char* section, const std::string& key, const auto& ref, const char* help) {
        if (section) os << "\n# ---- " << section << "\n";
        std::string kv = key + " = " + formatValue(ref);
        if (kv.size() < 32) kv.resize(32, ' ');
        os << kv << " # " << help << "\n";
    });
}

// ------------------------------------------------------------------ VTK output

void writeVTU(const State& s, const std::string& filename) {
    const Mesh& m = s.m();
    const int nl = m.nLeaves();
    const int Lmax = m.maxLevel();
    const double hmax = m.h(Lmax);
    const uint64_t stride = (uint64_t)m.nx(Lmax) + 1;

    // unique corner points on the max-level lattice
    std::unordered_map<uint64_t, int> pointId;
    pointId.reserve((size_t)nl * 2);
    std::vector<double> pts;
    pts.reserve((size_t)nl * 3 * 2);
    std::vector<int32_t> conn((size_t)4 * nl), offs(nl);
    std::vector<uint8_t> types(nl, 9);   // VTK_QUAD
    for (int k = 0; k < nl; ++k) {
        const int c = m.leaves()[k];
        const int sh = Lmax - m.level(c);
        const uint64_t I = (uint64_t)m.ix(c) << sh, J = (uint64_t)m.iy(c) << sh, n = (uint64_t)1 << sh;
        const uint64_t ci[4] = {I, I + n, I + n, I}, cj[4] = {J, J, J + n, J + n};
        for (int q = 0; q < 4; ++q) {
            const uint64_t key = cj[q] * stride + ci[q];
            auto ins = pointId.emplace(key, (int)(pts.size() / 3));
            if (ins.second) {
                pts.push_back(m.xmin() + ci[q] * hmax);
                pts.push_back(m.ymin() + cj[q] * hmax);
                pts.push_back(0.0);
            }
            conn[4 * (size_t)k + q] = ins.first->second;
        }
        offs[k] = 4 * (k + 1);
    }
    const int np = (int)(pts.size() / 3);

    auto toFloat = [](const std::vector<double>& d) { return std::vector<float>(d.begin(), d.end()); };
    const std::vector<double> f = leafValues(s, s.id.f);
    std::vector<double> rho = leafValues(s, s.id.rho), mu = leafValues(s, s.id.mu);
    if (s.id.rho < 0 || s.id.mu < 0)
        for (int k = 0; k < nl; ++k) {
            const double ff = std::clamp(f[k], 0.0, 1.0);
            if (s.id.rho < 0) rho[k] = ff * s.prm.rho1 + (1 - ff) * s.prm.rho2;
            if (s.id.mu < 0) mu[k] = ff * s.prm.mu1 + (1 - ff) * s.prm.mu2;
        }
    std::vector<int32_t> level(nl);
    for (int k = 0; k < nl; ++k) level[k] = m.level(m.leaves()[k]);

    Appended app;
    std::ostringstream xml;
    xml << fileHeader("UnstructuredGrid") << "<UnstructuredGrid>\n" << timeFieldData(s.t)
        << "<Piece NumberOfPoints=\"" << np << "\" NumberOfCells=\"" << nl << "\">\n";
    xml << "<Points>\n" << app.array("Float64", "Points", 3, pts) << "</Points>\n";
    xml << "<Cells>\n"
        << app.array("Int32", "connectivity", 1, conn) << app.array("Int32", "offsets", 1, offs)
        << app.array("UInt8", "types", 1, types) << "</Cells>\n";
    xml << "<CellData Scalars=\"f\">\n";
    xml << app.array("Float64", "f", 1, f);
    xml << app.array("Float32", "u", 1, toFloat(leafValues(s, s.id.u)));
    xml << app.array("Float32", "v", 1, toFloat(leafValues(s, s.id.v)));
    xml << app.array("Float32", "p", 1, toFloat(leafValues(s, s.id.p)));
    xml << app.array("Float32", "rho", 1, toFloat(rho));
    xml << app.array("Float32", "mu", 1, toFloat(mu));
    xml << app.array("Float32", "kappa", 1, toFloat(leafValues(s, s.id.kappa)));
    xml << app.array("Int32", "level", 1, level);
    xml << app.array("Float32", "vorticity", 1, toFloat(vorticity(s)));
    xml << "</CellData>\n</Piece>\n</UnstructuredGrid>\n";

    std::ofstream os = openOut(filename, true);
    os << xml.str();
    app.write(os);
    os << "</VTKFile>\n";
    if (!os) throw std::runtime_error("error writing '" + filename + "'");
}

void writeInterfaceVTP(const State& s, const std::string& filename) {
    const Mesh& m = s.m();
    std::vector<double> f = restricted(s, s.id.f);
    const auto segs = vof::interfaceSegments(m, f);
    const int ns = (int)segs.size();
    std::vector<double> pts;
    pts.reserve((size_t)6 * ns);
    std::vector<int32_t> conn(2 * (size_t)ns), offs(ns);
    for (int k = 0; k < ns; ++k) {
        for (const Vec2& p : {segs[k].first, segs[k].second}) {
            pts.push_back(p.x);
            pts.push_back(p.y);
            pts.push_back(0.0);
        }
        conn[2 * (size_t)k] = 2 * k;
        conn[2 * (size_t)k + 1] = 2 * k + 1;
        offs[k] = 2 * (k + 1);
    }
    Appended app;
    std::ostringstream xml;
    xml << fileHeader("PolyData") << "<PolyData>\n" << timeFieldData(s.t) << "<Piece NumberOfPoints=\""
        << 2 * ns << "\" NumberOfVerts=\"0\" NumberOfLines=\"" << ns
        << "\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";
    xml << "<Points>\n" << app.array("Float64", "Points", 3, pts) << "</Points>\n";
    xml << "<Lines>\n"
        << app.array("Int32", "connectivity", 1, conn) << app.array("Int32", "offsets", 1, offs)
        << "</Lines>\n";
    xml << "</Piece>\n</PolyData>\n";
    std::ofstream os = openOut(filename, true);
    os << xml.str();
    app.write(os);
    os << "</VTKFile>\n";
    if (!os) throw std::runtime_error("error writing '" + filename + "'");
}

void writePVD(const std::string& filename, const std::vector<std::pair<double, std::string>>& entries) {
    std::ofstream os = openOut(filename, false);
    os << "<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\""
       << (littleEndian() ? "LittleEndian" : "BigEndian") << "\">\n<Collection>\n";
    for (const auto& e : entries)
        os << "  <DataSet timestep=\"" << formatValue(e.first) << "\" group=\"\" part=\"0\" file=\"" << e.second
           << "\"/>\n";
    os << "</Collection>\n</VTKFile>\n";
    if (!os) throw std::runtime_error("error writing '" + filename + "'");
}

// ------------------------------------------------------------------ diagnostics

Diagnostics computeDiagnostics(const State& s) {
    const Mesh& m = s.m();
    const Params& prm = s.prm;
    Diagnostics d;
    d.t = s.t;
    d.dt = s.dt;
    d.step = s.step;
    d.nLeaves = m.nLeaves();
    d.poissonIters = s.lastPoissonIters;
    if (s.id.f < 0) return d;
    const auto& f = m.field(s.id.f);
    const std::vector<double>* u = s.id.u >= 0 ? &m.field(s.id.u) : nullptr;
    const std::vector<double>* v = s.id.v >= 0 ? &m.field(s.id.v) : nullptr;

    double v1 = 0, v2 = 0, x1 = 0, y1 = 0, x2 = 0, y2 = 0, u1 = 0, w1 = 0, u2 = 0, w2 = 0, ek = 0, um2 = 0;
#pragma omp parallel for reduction(+ : v1, v2, x1, y1, x2, y2, u1, w1, u2, w2, ek) reduction(max : um2)
    for (int k = 0; k < m.nLeaves(); ++k) {
        const int c = m.leaves()[k];
        const double V = m.volume(c);
        const double a = std::clamp(f[c], 0.0, 1.0), b = 1.0 - a;
        const Vec2 x = m.center(c);
        const double uc = u ? (*u)[c] : 0.0, vc = v ? (*v)[c] : 0.0;
        v1 += a * V;
        v2 += b * V;
        x1 += a * V * x.x;
        y1 += a * V * x.y;
        x2 += b * V * x.x;
        y2 += b * V * x.y;
        u1 += a * V * uc;
        w1 += a * V * vc;
        u2 += b * V * uc;
        w2 += b * V * vc;
        const double q2 = uc * uc + vc * vc;
        ek += 0.5 * (a * prm.rho1 + b * prm.rho2) * q2 * V;
        um2 = std::max(um2, q2);
    }
    d.vol1 = v1;
    d.vol2 = v2;
    if (v1 > 0) { d.xc1 = x1 / v1; d.yc1 = y1 / v1; d.uc1 = u1 / v1; d.vc1 = w1 / v1; }
    if (v2 > 0) { d.xc2 = x2 / v2; d.yc2 = y2 / v2; d.uc2 = u2 / v2; d.vc2 = w2 / v2; }
    d.ekin = ek;
    d.umax = std::sqrt(um2);

    std::vector<double> fr = restricted(s, s.id.f);
    double len = 0.0;   // total PLIC length (see the note on grid-aligned circles in the report)
    for (const auto& sg : vof::interfaceSegments(m, fr)) len += (sg.second - sg.first).norm();
    const double Amin = std::min(v1, v2);
    d.circularity = len > 0.0 ? 2.0 * std::sqrt(M_PI * Amin) / len : 0.0;
    return d;
}

void writeDiagnosticsHeader(std::ostream& os) {
    os << "# t dt step nLeaves poissonIters vol1 vol2 xc1 yc1 uc1 vc1 xc2 yc2 uc2 vc2 ekin umax "
          "circularity\n";
}

void writeDiagnosticsLine(std::ostream& os, const Diagnostics& d) {
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "%.10g %.6e %d %d %d %.12e %.12e %.10e %.10e %.8e %.8e %.10e %.10e %.8e %.8e %.8e %.8e "
                  "%.8e\n",
                  d.t, d.dt, d.step, d.nLeaves, d.poissonIters, d.vol1, d.vol2, d.xc1, d.yc1, d.uc1, d.vc1,
                  d.xc2, d.yc2, d.uc2, d.vc2, d.ekin, d.umax, d.circularity);
    os << buf;
}

}  // namespace io
