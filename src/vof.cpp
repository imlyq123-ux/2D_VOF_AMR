// vof.cpp - PLIC reconstruction, Weymouth-Yue split advection, geometric prolongation.
#include "vof.hpp"

#include <algorithm>
#include <cmath>

#include "plic.hpp"

namespace vof {

namespace {

inline bool mixed(double v) { return v > EPS && v < 1.0 - EPS; }

// PLIC line of cell c (any cell, leaf or not) from the 3x3 block at c's level.
// f must hold valid (restricted) values for every cell locate() can return.
Line lineAt(const Mesh& m, const std::vector<double>& f, int c) {
    const int L = m.level(c), i = m.ix(c), j = m.iy(c);
    double s[3][3];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            s[a][b] = (a == 1 && b == 1) ? f[c] : f[m.locate(L, i + a - 1, j + b - 1)];
    Line l;
    plic::mycNormal(s, l.mx, l.my);
    l.alpha = plic::alpha(l.mx, l.my, f[c]);
    return l;
}

// Clip to [0,1] and remove values within EPS of 0 / 1.
inline double clean(double v) {
    if (v < EPS) return 0.0;
    if (v > 1.0 - EPS) return 1.0;
    return v;
}

inline double minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return std::fabs(a) < std::fabs(b) ? a : b;
}

// One directional sweep of a phase tracer, using the phase-1 fluxes `flux` and u*A `uA`
// of the same sweep and f at the start of the sweep (see vof::Tracer).
void sweepTracer(const Mesh& m, const std::vector<double>& f, const Tracer& tr, const std::vector<double>& flux,
                 const std::vector<double>& uA, const std::vector<double>& cc, int d, double dt,
                 const Params& prm, bool par) {
    constexpr double kTiny = 1e-6;  // below this phase fraction phi = T/g is meaningless
    std::vector<double>& T = *tr.data;
    const std::vector<Mesh::Face>& faces = m.faces();
    const std::vector<int>& leaves = m.leaves();
    const int nc = m.nCells(), nf = m.nFaces(), nl = m.nLeaves();
    m.restrict(T);

    static std::vector<double> phi, tflux;
    static std::vector<char> ok;
    phi.resize(nc);
    ok.resize(nc);
    tflux.assign(nf, 0.0);
#pragma omp parallel for schedule(static) if (par)
    for (int c = 0; c < nc; ++c) {
        const double g = tr.phase2 ? 1.0 - f[c] : f[c];
        ok[c] = g > kTiny;
        phi[c] = ok[c] ? T[c] / g : (tr.fallback && m.isLeaf(c) ? (*tr.fallback)[c] : 0.0);
    }

#pragma omp parallel for schedule(static) if (par)
    for (int p = 0; p < nf; ++p) {
        const Mesh::Face& F = faces[p];
        if (F.dir != d || uA[p] == 0.0) continue;
        const double u = uA[p] / F.area;
        const double phaseFlux = tr.phase2 ? u * dt * F.area - flux[p] : flux[p];
        const int donor = u > 0.0 ? F.L : F.R;
        double ph;
        if (donor < 0) {
            const int in = u > 0.0 ? F.R : F.L;
            ph = prm.bc[F.bside].type == BCType::Inflow ? tr.inflow[F.bside] : phi[in];
        } else {
            // limited linear concentration, averaged over the donor strip next to the face
            const int L = m.level(donor), i = m.ix(donor), j = m.iy(donor);
            const int nm = d == 0 ? m.locate(L, i - 1, j) : m.locate(L, i, j - 1);
            const int np = d == 0 ? m.locate(L, i + 1, j) : m.locate(L, i, j + 1);
            const double pc = phi[donor];
            const double pm = ok[nm] ? phi[nm] : pc, pp = ok[np] ? phi[np] : pc;
            const double w = std::min(std::fabs(u) * dt / m.size(donor), 1.0);
            ph = pc + (u > 0.0 ? 0.5 : -0.5) * (1.0 - w) * minmod(pc - pm, pp - pc);
        }
        tflux[p] = phaseFlux * ph;
    }

#pragma omp parallel for schedule(static) if (par)
    for (int k = 0; k < nl; ++k) {
        const int c = leaves[k];
        double sT = 0.0, sU = 0.0;
        for (const int* p = m.cellFacesBegin(c); p != m.cellFacesEnd(c); ++p) {
            const Mesh::Face& F = faces[*p];
            if (F.dir != d) continue;
            const double s = F.L == c ? 1.0 : -1.0;
            sT += s * tflux[*p];
            sU += s * uA[*p];
        }
        const double ccp = tr.phase2 ? 1.0 - cc[k] : cc[k];
        T[c] -= (sT - ccp * phi[c] * dt * sU) / m.volume(c);
    }
}

}  // namespace

Line reconstruct(const Mesh& m, const std::vector<double>& f, int c) { return lineAt(m, f, c); }

void advect(Mesh& m, std::vector<double>& f, const std::vector<double>& uf, double dt, int step,
            const Params& prm, const std::vector<Tracer>& tracers) {
    const std::vector<int>& leaves = m.leaves();
    const std::vector<Mesh::Face>& faces = m.faces();
    const int nl = m.nLeaves(), nf = m.nFaces();
    const double x0 = m.xmin(), y0 = m.ymin();
    // reused work buffers (advect is called from one thread)
    static std::vector<Line> lines;       // per leaf ordinal, valid for mixed leaves
    static std::vector<double> cc;        // WY compression coefficient H(f^n - 1/2)
    static std::vector<double> flux, uA;  // per face: phase-1 volume flux, u*A (sweep dir)
    lines.resize(nl);
    cc.resize(nl);
    flux.resize(nf);
    uA.resize(nf);

    const bool par = nl > 4000;  // threading overhead dominates on small meshes
    double lost = 0.0;           // volume removed by clean()
#pragma omp parallel for schedule(static) if (par)
    for (int k = 0; k < nl; ++k) cc[k] = f[leaves[k]] > 0.5 ? 1.0 : 0.0;

    for (int sweep = 0; sweep < 2; ++sweep) {
        const int d = (step % 2 == 0) ? sweep : 1 - sweep;
        m.restrict(f);  // parents valid for the stencils (f's parent values are scratch)

#pragma omp parallel for schedule(dynamic, 256) if (par)
        for (int k = 0; k < nl; ++k) {
            const int c = leaves[k];
            if (mixed(f[c])) lines[k] = lineAt(m, f, c);
        }

#pragma omp parallel for schedule(static) if (par)
        for (int p = 0; p < nf; ++p) {
            const Mesh::Face& F = faces[p];
            if (F.dir != d) continue;
            double u = uf[p];
            if (F.bside >= 0) {
                const BCType t = prm.bc[F.bside].type;
                if (t == BCType::Wall || t == BCType::Slip) u = 0.0;
            }
            uA[p] = u * F.area;
            if (u == 0.0) { flux[p] = 0.0; continue; }
            const int donor = u > 0.0 ? F.L : F.R;
            if (donor < 0) {  // inflow through a boundary face
                const SideBC& bc = prm.bc[F.bside];
                const double fin = bc.type == BCType::Inflow ? bc.f : f[u > 0.0 ? F.R : F.L];
                flux[p] = u * dt * F.area * fin;
                continue;
            }
            const double fd = f[donor];
            if (!mixed(fd)) { flux[p] = u * dt * F.area * fd; continue; }
            // donor region: strip of width |u| dt along the face, in donor-local coordinates
            const double h = m.size(donor);
            const double w = std::min(std::fabs(u) * dt / h, 1.0);
            const double n0 = u > 0.0 ? 1.0 - w : 0.0, n1 = u > 0.0 ? 1.0 : w;
            double t0 = 0.0, t1 = 1.0;  // tangential extent: whole side or half of it
            if (F.area < 0.75 * h) {
                const double tc = d == 0 ? (F.x.y - y0) / h - m.iy(donor) : (F.x.x - x0) / h - m.ix(donor);
                if (tc < 0.5) t1 = 0.5; else t0 = 0.5;
            }
            const Line& l = lines[m.leafIndex(donor)];
            const double vol = d == 0 ? plic::rectArea(l.mx, l.my, l.alpha, n0, n1, t0, t1)
                                      : plic::rectArea(l.mx, l.my, l.alpha, t0, t1, n0, n1);
            flux[p] = (u > 0.0 ? h * h : -h * h) * vol;
        }

        // tracers use f at the start of the sweep, so they go before the f update
        for (const Tracer& tr : tracers) sweepTracer(m, f, tr, flux, uA, cc, d, dt, prm, par);

        const double* fl = flux.data();
        const double* ua = uA.data();
#pragma omp parallel for schedule(static) reduction(+ : lost) if (par)
        for (int k = 0; k < nl; ++k) {
            const int c = leaves[k];
            double sF = 0.0, sU = 0.0;
            for (const int* p = m.cellFacesBegin(c); p != m.cellFacesEnd(c); ++p) {
                const Mesh::Face& F = faces[*p];
                if (F.dir != d) continue;
                if (F.L == c) { sF += fl[*p]; sU += ua[*p]; }
                else { sF -= fl[*p]; sU -= ua[*p]; }
            }
            const double v = f[c] - (sF - cc[k] * dt * sU) / m.volume(c);
            f[c] = clean(v);
            lost += (v - f[c]) * m.volume(c);
        }
    }

    // Give the volume removed by clean() (values within EPS of 0/1, round-off outside
    // [0,1]) back to the interfacial cells, weighted by f(1-f): keeps the total volume
    // exact to round-off. The change per cell is ~1e-16, far below the bounds margin.
    if (lost != 0.0) {
        double W = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : W) if (par)
        for (int k = 0; k < nl; ++k) {
            const double v = f[leaves[k]];
            W += v * (1.0 - v) * m.volume(leaves[k]);
        }
        if (W > 0.0) {
            const double s = lost / W;
#pragma omp parallel for schedule(static) if (par)
            for (int k = 0; k < nl; ++k) {
                double& v = f[leaves[k]];
                v = std::clamp(v + s * v * (1.0 - v), 0.0, 1.0);
            }
        }
    }
}

void prolong(const Mesh& m, const std::vector<double>& f, int parent, double out[4]) {
    const double fp = f[parent];
    if (!mixed(fp)) { out[0] = out[1] = out[2] = out[3] = fp; return; }
    const Line l = lineAt(m, f, parent);
    double sum = 0.0;
    int best = 0;
    for (int k = 0; k < 4; ++k) {
        const double X0 = 0.5 * (k & 1), Y0 = 0.5 * (k >> 1);
        out[k] = clean(4.0 * plic::rectArea(l.mx, l.my, l.alpha, X0, X0 + 0.5, Y0, Y0 + 0.5));
        sum += out[k];
        if (std::min(out[k], 1.0 - out[k]) > std::min(out[best], 1.0 - out[best])) best = k;
    }
    // put the round-off / cleaning residual on the most mixed child: mean == fp exactly
    out[best] = std::clamp(out[best] + (4.0 * fp - sum), 0.0, 1.0);
}

std::vector<std::pair<Vec2, Vec2>> interfaceSegments(const Mesh& m, const std::vector<double>& f) {
    std::vector<double> g(f);
    m.restrict(g);
    std::vector<std::pair<Vec2, Vec2>> segs;
    for (int c : m.leaves()) {
        if (!mixed(g[c])) continue;
        const Line l = lineAt(m, g, c);
        double P[4];
        if (!plic::segment(l.mx, l.my, l.alpha, P)) continue;
        const double h = m.size(c), xc = m.xmin() + m.ix(c) * h, yc = m.ymin() + m.iy(c) * h;
        segs.emplace_back(Vec2(xc + P[0] * h, yc + P[1] * h), Vec2(xc + P[2] * h, yc + P[3] * h));
    }
    return segs;
}

}  // namespace vof
