// navier_stokes.cpp - incompressible two-phase Navier-Stokes (see navier_stokes.hpp).
//
// Discretization summary
//   * Face properties from the face-averaged volume fraction f_f = (f_L + f_R)/2
//     (interior value on boundary faces): rho_f, alpha_f = 1/rho_f, mu_f arithmetic.
//   * Advection: (1/V) sum_f s_f uf_f A_f (q_f - q_c) with the divergence-free uf and a
//     BCG-like time-centred upwind face value
//       q_f = q_U + (+-h_U - uf dt)/2 slope_n + offset_t slope_t
//             - dt/2 u_t,U (upwind one-sided transverse difference) + dt/4 (g_L + g_R),
//     slope_n/slope_t limited with a generalized minmod (theta = THETA), g = the cell
//     acceleration of the previous projection (mesh fields "ns_gx", "ns_gy", owned
//     by this module). Ghost values from the BCs. Second order in time and space.
//   * Viscosity: div(mu_f grad u) implicit (backward Euler, poisson::solve, applied to
//     u* + dt g as in Basilisk) or explicit, plus the explicit transpose term
//     div(mu (grad u)^T) when mu1 != mu2 (it vanishes identically for a constant
//     viscosity and a divergence-free field). At 2:1 faces the two-point flux is
//     corrected explicitly for the tangential offset of the coarse centre.
//   * Projection: uf* = face(u*) + dt a_f on interior and outflow faces; the boundary
//     faces with prescribed normal velocity keep their value and have homogeneous
//     Neumann pressure. Gravity term a_f = g.(x_R - x_L)/d_f: equals g.n_f on faces
//     between same-size cells and is the exact discrete gradient of g.x on coarse/fine
//     faces, so that hydrostatic states are discretely balanced on AMR meshes.
//   * Cell correction: u_c += dt * mean over the -x/+x sides of the area-weighted face
//     values g_f = a_f - alpha_f grad_f p; g_f = 0 on wall/slip/inflow faces (their
//     normal acceleration is balanced by the wall, cf. Basilisk's neumann_pressure()).
#include "navier_stokes.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ns {

namespace {

using Face = Mesh::Face;

constexpr double THETA = 1.5;   // generalized minmod: 1 = minmod, 2 = MC
constexpr int kParMin = 16384;  // smaller loops run serially (OpenMP start-up cost)

inline double clamp01(double a) { return std::min(1.0, std::max(0.0, a)); }

inline double fracAt(const State& s, int c) {
    return s.id.f >= 0 ? clamp01(s.F(s.id.f)[c]) : 1.0;
}

// Face-averaged volume fraction (interior value on boundary faces).
inline double faceFraction(const State& s, const Face& F) {
    if (F.L < 0) return fracAt(s, F.R);
    if (F.R < 0) return fracAt(s, F.L);
    return 0.5 * (fracAt(s, F.L) + fracAt(s, F.R));
}

inline int interiorCell(const Face& F) { return F.L >= 0 ? F.L : F.R; }

inline bool isPressureDirichlet(const Params& prm, const Face& F) {
    return F.bside >= 0 && prm.bc[F.bside].type == BCType::Outflow;
}

// Value of velocity component comp (0: u, 1: v) ON boundary side `side`, given the
// interior value qc. *dirichlet tells whether the value is imposed.
inline double bcValue(const Params& prm, int side, int comp, double qc, bool* dirichlet) {
    const SideBC& b = prm.bc[side];
    const bool normal = comp == (side <= RIGHT ? 0 : 1);
    switch (b.type) {
    case BCType::Wall:
        *dirichlet = true;
        return normal ? 0.0 : (comp == 0 ? b.u : b.v);
    case BCType::Slip:
        *dirichlet = normal;
        return normal ? 0.0 : qc;
    case BCType::Inflow:
        *dirichlet = true;
        return comp == 0 ? b.u : b.v;
    default:  // Outflow (Periodic sides have no boundary faces)
        *dirichlet = false;
        return qc;
    }
}

// Normal derivative (along +x/+y) of component comp on face F; boundary faces use the
// BC value at distance d_f = h/2 (zero for non-Dirichlet faces).
inline double normalDerivative(const Params& prm, const std::vector<double>& q, int comp,
                               const Face& F) {
    if (F.L >= 0 && F.R >= 0) return (q[F.R] - q[F.L]) / F.dist;
    const int c = interiorCell(F);
    bool dir;
    const double qb = bcValue(prm, F.bside, comp, q[c], &dir);
    if (!dir) return 0.0;
    return F.L >= 0 ? (qb - q[c]) / F.dist : (q[c] - qb) / F.dist;
}

// Area-weighted neighbour values and centre distances on the 4 sides (Side order) of c.
inline void sideMeans(const Mesh& m, const Params& prm, const std::vector<double>& q, int comp,
                      int c, double val[4], double dist[4]) {
    double sa[4] = {0, 0, 0, 0}, sq[4] = {0, 0, 0, 0}, sd[4] = {0, 0, 0, 0};
    const auto& faces = m.faces();
    for (const int* p = m.cellFacesBegin(c); p != m.cellFacesEnd(c); ++p) {
        const Face& F = faces[*p];
        const bool plus = F.L == c;
        const int sd_ = 2 * F.dir + (plus ? 1 : 0);
        const int nb = plus ? F.R : F.L;
        double qv;
        if (nb >= 0) {
            qv = q[nb];
        } else {
            bool d;
            qv = bcValue(prm, F.bside, comp, q[c], &d);
        }
        sa[sd_] += F.area;
        sq[sd_] += F.area * qv;
        sd[sd_] += F.area * F.dist;
    }
    for (int k = 0; k < 4; ++k) {
        val[k] = sq[k] / sa[k];
        dist[k] = sd[k] / sa[k];
    }
}

// One-sided differences gm, gp, weighted central cen and limited lim slopes.
inline void slopes(double q, double qm, double dm, double qp, double dp, double& gm, double& gp,
                   double& lim, double& cen) {
    gm = (q - qm) / dm;
    gp = (qp - q) / dp;
    cen = (gm * dp + gp * dm) / (dm + dp);
    if (gm * gp <= 0.0) {
        lim = 0.0;
    } else {
        const double a = std::min(THETA * std::min(std::fabs(gm), std::fabs(gp)), std::fabs(cen));
        lim = cen > 0.0 ? a : -a;
    }
}

// At a 2:1 face, the coarse centre is offset tangentially by +-h_fine/2 from the face
// midpoint, which makes the two-point difference q_R - q_L inconsistent. Returns delta
// such that (q_R - q_L + delta)/d_f is the consistent normal derivative (coarse value
// moved to the face midpoint with its central tangential slope); 0 on other faces.
inline double coarseFineDelta(const Mesh& m, const Params& prm, const std::vector<double>& q, int comp,
                              const Face& F) {
    if (F.L < 0 || F.R < 0) return 0.0;
    const int lL = m.level(F.L), lR = m.level(F.R);
    if (lL == lR) return 0.0;
    const int C = lL < lR ? F.L : F.R;
    double val[4], dist[4], gm, gp, lim, cen;
    sideMeans(m, prm, q, comp, C, val, dist);
    const int t = 1 - F.dir;
    slopes(q[C], val[2 * t], dist[2 * t], val[2 * t + 1], dist[2 * t + 1], gm, gp, lim, cen);
    const Vec2 xc = m.center(C);
    const double dq = cen * (F.dir == 0 ? F.x.y - xc.y : F.x.x - xc.x);
    return C == F.R ? dq : -dq;
}

// Mesh fields owned by this module: cell-centred acceleration g = cellAverage(a_f -
// alpha_f grad_f p) of the last projection (BCG source term, viscous splitting).
const char* const kGName[2] = {"ns_gx", "ns_gy"};

// Per-face density reciprocal alpha_f and viscosity mu_f.
void faceProperties(const State& s, std::vector<double>* alpha, std::vector<double>* mu) {
    const Mesh& m = s.m();
    const Params& prm = s.prm;
    const int nf = m.nFaces();
    if (alpha) alpha->resize(nf);
    if (mu) mu->resize(nf);
    const auto& faces = m.faces();
#pragma omp parallel for schedule(static) if (nf > kParMin)
    for (int k = 0; k < nf; ++k) {
        const double ff = faceFraction(s, faces[k]);
        if (alpha) (*alpha)[k] = 1.0 / (ff * prm.rho1 + (1.0 - ff) * prm.rho2);
        if (mu) (*mu)[k] = ff * prm.mu1 + (1.0 - ff) * prm.mu2;
    }
}

// Face acceleration a_f (gravity + balanced surface tension) on interior and
// Dirichlet-pressure faces; 0 elsewhere.
double faceAcceleration(const State& s, const Face& F, double alphaF) {
    const Mesh& m = s.m();
    const Params& prm = s.prm;
    double a;
    if (F.L >= 0 && F.R >= 0) {
        const Vec2 xL = m.center(F.L), xR = m.center(F.R);
        // g.(x_R - x_L)/d_f with the normal offset d_f (periodic-safe) + tangential offset
        a = F.dir == 0 ? prm.gx + prm.gy * (xR.y - xL.y) / F.dist
                       : prm.gy + prm.gx * (xR.x - xL.x) / F.dist;
        if (prm.sigma != 0.0 && s.id.f >= 0 && s.id.kappa >= 0) {
            const auto& f = s.F(s.id.f);
            const double df = f[F.R] - f[F.L];
            if (df != 0.0) {
                const auto& kap = s.F(s.id.kappa);
                const double kL = kap[F.L], kR = kap[F.R];
                const double kf = (kL != 0.0 && kR != 0.0) ? 0.5 * (kL + kR) : (kL != 0.0 ? kL : kR);
                a += prm.sigma * kf * df / F.dist * alphaF;
            }
        }
    } else {
        a = F.dir == 0 ? prm.gx : prm.gy;
    }
    return a;
}

// uf <- face(u) + dt a_f, pressure solve, uf <- uf - dt alpha grad p. If g != nullptr it
// receives the face values a_f - alpha_f grad_f p used by the cell correction. With
// fromCells == false the current s.uf is projected instead of face(u).
poisson::Result projectFaces(State& s, double dt, bool withAccel, std::vector<double>& p,
                             std::vector<double>* g, bool fromCells = true) {
    const Mesh& m = s.m();
    const Params& prm = s.prm;
    const auto& faces = m.faces();
    const auto& leaves = m.leaves();
    const int nf = m.nFaces(), nl = m.nLeaves();

    std::vector<double> alpha;
    faceProperties(s, &alpha, nullptr);
    if (fromCells) cellToFace(s, s.uf);
    std::vector<double> acc(withAccel ? nf : 0, 0.0);
    std::vector<char> dirichlet(nf, 0);
    bool anyDir = false;
#pragma omp parallel for schedule(static) reduction(|| : anyDir) if (nf > kParMin)
    for (int k = 0; k < nf; ++k) {
        const Face& F = faces[k];
        const bool interior = F.L >= 0 && F.R >= 0;
        const bool dir = !interior && isPressureDirichlet(prm, F);
        dirichlet[k] = dir;
        anyDir = anyDir || dir;
        if (withAccel && (interior || dir)) {
            acc[k] = faceAcceleration(s, F, alpha[k]);
            s.uf[k] += dt * acc[k];
        }
    }

    poisson::Problem P;
    P.beta = alpha;
    P.rhs.assign(m.nCells(), 0.0);
    if (anyDir) P.dirichlet = dirichlet;
    const double idt = 1.0 / dt;
#pragma omp parallel for schedule(static) if (nl > kParMin)
    for (int k = 0; k < nl; ++k) {
        const int c = leaves[k];
        double div = 0.0;
        for (const int* q = m.cellFacesBegin(c); q != m.cellFacesEnd(c); ++q) {
            const Face& F = faces[*q];
            div += (F.L == c ? 1.0 : -1.0) * s.uf[*q] * F.area;
        }
        P.rhs[c] = -div / m.volume(c) * idt;
    }
    poisson::Options opt;
    opt.tol = prm.poissonTol;
    opt.maxIter = prm.poissonMaxIter;
    opt.gpu = prm.gpu;
    if ((int)p.size() != m.nCells()) p.assign(m.nCells(), 0.0);
    const poisson::Result r = poisson::solve(m, P, p, opt);

    if (g) g->assign(nf, 0.0);
#pragma omp parallel for schedule(static) if (nf > kParMin)
    for (int k = 0; k < nf; ++k) {
        const Face& F = faces[k];
        double gp;
        if (F.L >= 0 && F.R >= 0) gp = (p[F.R] - p[F.L]) / F.dist;
        else if (dirichlet[k]) gp = F.L >= 0 ? -p[F.L] / F.dist : p[F.R] / F.dist;
        else continue;  // prescribed normal velocity: uf unchanged, g_f = 0
        s.uf[k] -= dt * alpha[k] * gp;
        if (g) (*g)[k] = (withAccel ? acc[k] : 0.0) - alpha[k] * gp;
    }
    return r;
}

// BCG-type time-centred (t + dt/2) face values of u and v, upwinded with the face
// velocity uf (steps 1-2 of the momentum predictor, also used by predictFaceVelocity):
//   q_f = q_U + (+-h_U - uf dt)/2 dq/dn + (tangential offset) dq/dt
//         - dt/2 u_t,U (dq/dt)_upwind + dt/4 (g_L + g_R)
// If faceValues is false only the central slopes (cen, for the transpose viscous term)
// are computed; cen is filled only if wantCentral.
struct FaceStates {
    std::vector<double> qf[2];        // [comp] per face
    std::vector<double> cen[2][2];    // [comp][dir] per cell, central slopes
};

void bcgFaceValues(const State& s, const std::vector<double>& uf, double dt, bool faceValues,
                   bool wantCentral, FaceStates& out) {
    const Mesh& m = s.m();
    const Params& prm = s.prm;
    const auto& faces = m.faces();
    const auto& leaves = m.leaves();
    const int nf = m.nFaces(), nl = m.nLeaves(), nc = m.nCells();
    const std::vector<double>* q[2] = {&s.F(s.id.u), &s.F(s.id.v)};
    const bool transpose = wantCentral;
    auto& cen = out.cen;
    // cell acceleration of the previous projection (zero before the first one)
    const std::vector<double>* gc[2] = {nullptr, nullptr};
    for (int a = 0; a < 2; ++a) {
        const int id = m.fieldId(kGName[a]);
        if (id >= 0) gc[a] = &m.field(id);
    }

    // ---- 1. per cell: limited slopes (normal extrapolation), upwind one-sided slopes
    //      (transverse term), central slopes (transpose term), all [comp][dir]; and the
    //      cell-averaged advecting velocity ufc[dir]
    std::vector<double> lim[2][2], tr[2][2], ufc[2];
    for (int a = 0; a < 2; ++a) {
        ufc[a].assign(nc, 0.0);
        for (int d = 0; d < 2; ++d) {
            lim[a][d].assign(nc, 0.0);
            tr[a][d].assign(nc, 0.0);
            if (transpose) cen[a][d].assign(nc, 0.0);
        }
    }
#pragma omp parallel for schedule(static) if (nl > kParMin)
    for (int k = 0; k < nl; ++k) {
        const int c = leaves[k];
        double su[4] = {0, 0, 0, 0}, sa[4] = {0, 0, 0, 0};
        for (const int* p = m.cellFacesBegin(c); p != m.cellFacesEnd(c); ++p) {
            const Face& F = faces[*p];
            const int sd = 2 * F.dir + (F.L == c ? 1 : 0);
            su[sd] += F.area * uf[*p];
            sa[sd] += F.area;
        }
        for (int d = 0; d < 2; ++d) ufc[d][c] = 0.5 * (su[2 * d] / sa[2 * d] + su[2 * d + 1] / sa[2 * d + 1]);
        for (int a = 0; a < 2; ++a) {
            double val[4], dist[4];
            sideMeans(m, prm, *q[a], a, c, val, dist);
            for (int d = 0; d < 2; ++d) {
                double gm, gp, l, ce;
                slopes((*q[a])[c], val[2 * d], dist[2 * d], val[2 * d + 1], dist[2 * d + 1], gm, gp, l, ce);
                lim[a][d][c] = l;
                tr[a][d][c] = ufc[d][c] < 0.0 ? gp : gm;
                if (transpose) cen[a][d][c] = ce;
            }
        }
    }

    // ---- 2. upwind face values at t + dt/2 (BCG-like predictor):
    //      q_f = q_U + (+-h_U - uf dt)/2 dq/dn + (tangential offset) dq/dt
    //            - dt/2 u_t,U (dq/dt)_upwind + dt/4 (g_L + g_R)
    auto& qf = out.qf;
    for (int a = 0; a < 2; ++a) qf[a].assign(faceValues ? nf : 0, 0.0);
#pragma omp parallel for schedule(static) if (nf > kParMin)
    for (int k = 0; k < (faceValues ? nf : 0); ++k) {
        const Face& F = faces[k];
        if (F.L < 0 || F.R < 0) {
            const int c = interiorCell(F);
            for (int a = 0; a < 2; ++a) {
                bool d;
                qf[a][k] = bcValue(prm, F.bside, a, (*q[a])[c], &d);
            }
            continue;
        }
        const double w = uf[k];
        const int U = w >= 0.0 ? F.L : F.R;
        const double sg = w >= 0.0 ? 1.0 : -1.0;
        const double hU = m.size(U);
        const double dn = sg * 0.5 * std::max(0.0, hU - std::fabs(w) * dt);
        const Vec2 xU = m.center(U);
        const double dtan = F.dir == 0 ? F.x.y - xU.y : F.x.x - xU.x;
        const int nd = F.dir, td = 1 - F.dir;
        const double vt = 0.5 * dt * ufc[td][U];
        for (int a = 0; a < 2; ++a) {
            double val = (*q[a])[U] + dn * lim[a][nd][U] + dtan * lim[a][td][U] - vt * tr[a][td][U];
            if (gc[a]) val += 0.25 * dt * ((*gc[a])[F.L] + (*gc[a])[F.R]);
            qf[a][k] = val;
        }
    }

}

}  // namespace

// ------------------------------------------------------------------ properties

void updateProperties(State& s) {
    const Mesh& m = s.m();
    const Params& prm = s.prm;
    auto& rho = s.F(s.id.rho);
    auto& mu = s.F(s.id.mu);
    const auto& leaves = m.leaves();
#pragma omp parallel for schedule(static) if (m.nLeaves() > kParMin)
    for (int k = 0; k < m.nLeaves(); ++k) {
        const int c = leaves[k];
        const double f = fracAt(s, c);
        rho[c] = f * prm.rho1 + (1.0 - f) * prm.rho2;
        mu[c] = f * prm.mu1 + (1.0 - f) * prm.mu2;
    }
}

// ------------------------------------------------------------------ face velocity

void cellToFace(const State& s, std::vector<double>& uf) {
    const Mesh& m = s.m();
    const Params& prm = s.prm;
    const auto& faces = m.faces();
    const auto& u = s.F(s.id.u);
    const auto& v = s.F(s.id.v);
    const int nf = m.nFaces();
    uf.resize(nf);
#pragma omp parallel for schedule(static) if (nf > kParMin)
    for (int k = 0; k < nf; ++k) {
        const Face& F = faces[k];
        const std::vector<double>& q = F.dir == 0 ? u : v;
        if (F.L >= 0 && F.R >= 0) {
            // linear interpolation along the normal (2/3 fine, 1/3 coarse at a 2:1 face);
            // at a 2:1 face the coarse value is shifted tangentially to the face midpoint
            const double hL = m.size(F.L), hR = m.size(F.R);
            double qL = q[F.L], qR = q[F.R];
            if (hL != hR) {
                const double delta = coarseFineDelta(m, prm, q, F.dir, F);
                (hL > hR ? qL : qR) += hL > hR ? -delta : delta;
            }
            uf[k] = (qL * hR + qR * hL) / (hL + hR);
            continue;
        }
        const SideBC& b = prm.bc[F.bside];
        switch (b.type) {
        case BCType::Wall:
        case BCType::Slip: uf[k] = 0.0; break;
        case BCType::Inflow: uf[k] = F.dir == 0 ? b.u : b.v; break;
        default: uf[k] = q[interiorCell(F)]; break;
        }
    }
}

// ------------------------------------------------------------------ time step

double stableDt(const State& s) {
    const Mesh& m = s.m();
    const Params& prm = s.prm;
    const auto& faces = m.faces();
    const auto& leaves = m.leaves();
    const auto& u = s.F(s.id.u);
    const auto& v = s.F(s.id.v);
    const int nf = m.nFaces(), nl = m.nLeaves();
    const bool haveUf = (int)s.uf.size() == nf;
    const bool explicitVisc = !prm.implicitViscosity && (prm.mu1 > 0.0 || prm.mu2 > 0.0);
    const std::vector<double>* rho = s.id.rho >= 0 ? &s.F(s.id.rho) : nullptr;

    double rate = 0.0;      // max |velocity| / h
    double viscRate = 0.0;  // max mu / (rho h^2)
#pragma omp parallel for schedule(static) reduction(max : rate, viscRate) if (nf > kParMin)
    for (int k = 0; k < nf; ++k) {
        const Face& F = faces[k];
        if (haveUf) rate = std::max(rate, std::fabs(s.uf[k]) / F.area);
        if (explicitVisc) {
            const double ff = faceFraction(s, F);
            const double muf = ff * prm.mu1 + (1.0 - ff) * prm.mu2;
            double r = std::numeric_limits<double>::max();
            for (int c : {F.L, F.R})
                if (c >= 0)
                    r = std::min(r, rho ? (*rho)[c] : fracAt(s, c) * prm.rho1 + (1 - fracAt(s, c)) * prm.rho2);
            viscRate = std::max(viscRate, muf / (r * F.area * F.area));
        }
    }
    double hMinAll = std::numeric_limits<double>::max(), hMinItf = hMinAll;
#pragma omp parallel for schedule(static) reduction(max : rate) reduction(min : hMinAll, hMinItf) if (nl > kParMin)
    for (int k = 0; k < nl; ++k) {
        const int c = leaves[k];
        const double h = m.size(c);
        rate = std::max(rate, std::max(std::fabs(u[c]), std::fabs(v[c])) / h);
        hMinAll = std::min(hMinAll, h);
        if (s.id.f >= 0) {
            const double f = s.F(s.id.f)[c];
            if (f > 1e-10 && f < 1.0 - 1e-10) hMinItf = std::min(hMinItf, h);
        }
    }
    double dt = prm.dtMax > 0.0 ? prm.dtMax : std::numeric_limits<double>::max();
    if (rate > 0.0) dt = std::min(dt, prm.cfl / rate);
    if (prm.sigma > 0.0) {
        const double h = hMinItf < std::numeric_limits<double>::max() ? hMinItf : hMinAll;
        dt = std::min(dt, std::sqrt(0.5 * (prm.rho1 + prm.rho2) * h * h * h / (M_PI * prm.sigma)));
    }
    // explicit 2D diffusion with the full stress (normal coefficient 2 mu): dt <= h^2/(6 nu)
    if (viscRate > 0.0) dt = std::min(dt, 1.0 / (6.0 * viscRate));
    if (!(dt > 0.0) || !std::isfinite(dt)) dt = prm.dtMin > 0.0 ? prm.dtMin : 1e-12;
    return dt;
}

// ------------------------------------------------------------------ momentum predictor

void advanceMomentum(State& s, double dt) {
    if (!(dt > 0.0)) return;
    const Mesh& m = s.m();
    const Params& prm = s.prm;
    const auto& faces = m.faces();
    const auto& leaves = m.leaves();
    const int nf = m.nFaces(), nl = m.nLeaves(), nc = m.nCells();
    std::vector<double>* q[2] = {&s.F(s.id.u), &s.F(s.id.v)};

    std::vector<double> ufLocal;
    const std::vector<double>* ufp = &s.uf;
    if ((int)s.uf.size() != nf) {
        cellToFace(s, ufLocal);
        ufp = &ufLocal;
    }
    const std::vector<double>& uf = *ufp;

    const bool viscous = prm.mu1 != 0.0 || prm.mu2 != 0.0;
    const bool transpose = viscous && prm.mu1 != prm.mu2;

    // ---- 1.-2. limited slopes and time-centred face values (skipped with
    //      prm.conservingMomentum: momentum was advected with the VOF fluxes, see sim::step)
    const bool advect = !prm.conservingMomentum;
    FaceStates fs;
    bcgFaceValues(s, uf, dt, advect, transpose, fs);
    const std::vector<double>(&qf)[2] = fs.qf;
    const std::vector<double>(&cen)[2][2] = fs.cen;
    const std::vector<double>* gc[2] = {nullptr, nullptr};   // previous cell acceleration
    for (int a = 0; a < 2; ++a) {
        const int id = m.fieldId(kGName[a]);
        if (id >= 0) gc[a] = &m.field(id);
    }

    // ---- 3. advected velocity
    std::vector<double> qa[2] = {*q[0], *q[1]};
#pragma omp parallel for schedule(static) if (nl > kParMin)
    for (int k = 0; k < (advect ? nl : 0); ++k) {
        const int c = leaves[k];
        double adv[2] = {0.0, 0.0};
        for (const int* p = m.cellFacesBegin(c); p != m.cellFacesEnd(c); ++p) {
            const Face& F = faces[*p];
            const double flux = (F.L == c ? 1.0 : -1.0) * uf[*p] * F.area;
            for (int a = 0; a < 2; ++a) adv[a] += flux * (qf[a][*p] - (*q[a])[c]);
        }
        const double iv = dt / m.volume(c);
        for (int a = 0; a < 2; ++a) qa[a][c] = (*q[a])[c] - iv * adv[a];
    }

    if (!viscous) {
        *q[0] = std::move(qa[0]);
        *q[1] = std::move(qa[1]);
        return;
    }

    // ---- 4. viscosity
    std::vector<double> muf;
    faceProperties(s, nullptr, &muf);
    const std::vector<double>& rho = s.F(s.id.rho);

    // explicit transpose term T_a = div(mu (d u_dir / d x_a)) from u^n
    std::vector<double> T[2];
    if (transpose) {
        std::vector<double> tf[2] = {std::vector<double>(nf), std::vector<double>(nf)};
#pragma omp parallel for schedule(static) if (nf > kParMin)
        for (int k = 0; k < nf; ++k) {
            const Face& F = faces[k];
            const bool interior = F.L >= 0 && F.R >= 0;
            for (int a = 0; a < 2; ++a) {
                double D;
                if (F.dir == a) {
                    D = normalDerivative(prm, *q[a], a, F) + coarseFineDelta(m, prm, *q[a], a, F) / F.dist;
                } else if (interior) {
                    D = 0.5 * (cen[F.dir][a][F.L] + cen[F.dir][a][F.R]);
                } else {
                    // tangential derivative of the normal velocity along the boundary:
                    // zero where it is prescribed, interior estimate on outflow sides
                    D = prm.bc[F.bside].type == BCType::Outflow ? cen[F.dir][a][interiorCell(F)] : 0.0;
                }
                tf[a][k] = F.area * muf[k] * D;
            }
        }
        for (int a = 0; a < 2; ++a) T[a].assign(nc, 0.0);
#pragma omp parallel for schedule(static) if (nl > kParMin)
        for (int k = 0; k < nl; ++k) {
            const int c = leaves[k];
            double t[2] = {0.0, 0.0};
            for (const int* p = m.cellFacesBegin(c); p != m.cellFacesEnd(c); ++p) {
                const double sg = faces[*p].L == c ? 1.0 : -1.0;
                t[0] += sg * tf[0][*p];
                t[1] += sg * tf[1][*p];
            }
            const double iv = 1.0 / m.volume(c);
            T[0][c] = t[0] * iv;
            T[1][c] = t[1] * iv;
        }
    }

    if (prm.implicitViscosity) {
        // (rho/dt) u - div(mu_f grad u) = (rho/dt) u_adv + T, per component
        double muMax = std::max(prm.mu1, prm.mu2);
        poisson::Problem P;
        P.beta.resize(nf);
        for (int k = 0; k < nf; ++k) P.beta[k] = std::max(muf[k], 1e-30 * muMax);
        P.lambda.assign(nc, 0.0);
        P.rhs.assign(nc, 0.0);
        P.dirichlet.assign(nf, 0);
        P.bvalue.assign(nf, 0.0);
        for (int c : leaves) P.lambda[c] = rho[c] / dt;
        poisson::Options opt;
        opt.tol = prm.poissonTol;
        opt.maxIter = prm.poissonMaxIter;
        opt.gpu = prm.gpu;
        // Mass-dominated (h^2 rho / (mu dt) not small): Jacobi-PCG converges in a few tens
        // of iterations and avoids the AMG setup cost; otherwise use the default (AMG).
        double ratio = std::numeric_limits<double>::max();
#pragma omp parallel for schedule(static) reduction(min : ratio) if (nl > kParMin)
        for (int k = 0; k < nl; ++k) {
            const int c = leaves[k];
            double off = 0.0;
            for (const int* p = m.cellFacesBegin(c); p != m.cellFacesEnd(c); ++p)
                off += P.beta[*p] * faces[*p].area / faces[*p].dist;
            if (off > 0.0) ratio = std::min(ratio, P.lambda[c] * m.volume(c) / off);
        }
        if (ratio > 0.05) opt.precond = 0;
        for (int a = 0; a < 2; ++a) {
            // solve for u* + dt g (the velocity after the coming projection, to first order)
            // so that the viscous operator acts on the corrected velocity (as in Basilisk)
            if (gc[a])
                for (int c : leaves) qa[a][c] += dt * (*gc[a])[c];
#pragma omp parallel for schedule(static) if (nf > kParMin)
            for (int k = 0; k < nf; ++k) {
                const Face& F = faces[k];
                if (F.L >= 0 && F.R >= 0) continue;
                bool d;
                P.bvalue[k] = bcValue(prm, F.bside, a, qa[a][interiorCell(F)], &d);
                P.dirichlet[k] = d;
            }
            // explicit (lagged) tangential correction of the two-point flux at 2:1 faces
            std::vector<double> cf(nf, 0.0);
#pragma omp parallel for schedule(static) if (nf > kParMin)
            for (int k = 0; k < nf; ++k) {
                const Face& F = faces[k];
                if (F.L >= 0 && F.R >= 0 && m.level(F.L) != m.level(F.R))
                    cf[k] = F.area * muf[k] * coarseFineDelta(m, prm, qa[a], a, F) / F.dist;
            }
#pragma omp parallel for schedule(static) if (nl > kParMin)
            for (int k = 0; k < nl; ++k) {
                const int c = leaves[k];
                double corr = 0.0;
                for (const int* p = m.cellFacesBegin(c); p != m.cellFacesEnd(c); ++p)
                    corr += (faces[*p].L == c ? 1.0 : -1.0) * cf[*p];
                P.rhs[c] = P.lambda[c] * qa[a][c] + (transpose ? T[a][c] : 0.0) + corr / m.volume(c);
            }
            poisson::solve(m, P, qa[a], opt);
            if (gc[a])
                for (int c : leaves) qa[a][c] -= dt * (*gc[a])[c];
            *q[a] = qa[a];
        }
    } else {
        std::vector<double> lf[2] = {std::vector<double>(nf), std::vector<double>(nf)};
#pragma omp parallel for schedule(static) if (nf > kParMin)
        for (int k = 0; k < nf; ++k) {
            const Face& F = faces[k];
            for (int a = 0; a < 2; ++a)
                lf[a][k] = F.area * muf[k] *
                           (normalDerivative(prm, *q[a], a, F) + coarseFineDelta(m, prm, *q[a], a, F) / F.dist);
        }
#pragma omp parallel for schedule(static) if (nl > kParMin)
        for (int k = 0; k < nl; ++k) {
            const int c = leaves[k];
            double l[2] = {0.0, 0.0};
            for (const int* p = m.cellFacesBegin(c); p != m.cellFacesEnd(c); ++p) {
                const double sg = faces[*p].L == c ? 1.0 : -1.0;
                l[0] += sg * lf[0][*p];
                l[1] += sg * lf[1][*p];
            }
            const double iv = 1.0 / m.volume(c);
            for (int a = 0; a < 2; ++a)
                qa[a][c] += dt / rho[c] * (l[a] * iv + (transpose ? T[a][c] : 0.0));
        }
        *q[0] = std::move(qa[0]);
        *q[1] = std::move(qa[1]);
    }
}

// ------------------------------------------------------------------ MAC predictor

poisson::Result predictFaceVelocity(State& s, double dt) {
    Mesh& m = s.m();
    const auto& faces = m.faces();
    const int nf = m.nFaces();
    std::vector<double> ufOld;
    if ((int)s.uf.size() == nf) ufOld = s.uf;
    else cellToFace(s, ufOld);
    // normal component of the time-centred BCG face values (includes dt/2 of the
    // previous cell acceleration g, i.e. the lagged pressure gradient and body forces);
    // boundary faces get the BC value (0 on walls, prescribed on inflow)
    FaceStates fs;
    bcgFaceValues(s, ufOld, dt, true, false, fs);
    s.uf.resize(nf);
#pragma omp parallel for schedule(static) if (nf > kParMin)
    for (int k = 0; k < nf; ++k) s.uf[k] = fs.qf[faces[k].dir][k];
    // MAC projection; its pressure increment is kept as the next initial guess
    int pid = m.fieldId("ns_pf");
    if (pid < 0) pid = m.addField("ns_pf");
    return projectFaces(s, dt, false, m.field(pid), nullptr, false);
}

// ------------------------------------------------------------------ projection

poisson::Result project(State& s, double dt, bool withAccel) {
    if (!(dt > 0.0)) {
        poisson::Result r;
        r.converged = true;
        return r;
    }
    const Mesh& m = s.m();
    const auto& faces = m.faces();
    const auto& leaves = m.leaves();
    std::vector<double> g;
    const poisson::Result r = projectFaces(s, dt, withAccel, s.F(s.id.p), &g);
    s.lastPoissonIters = r.iterations;

    Mesh& mm = s.m();
    int gid[2];
    for (int a = 0; a < 2; ++a) {
        gid[a] = mm.fieldId(kGName[a]);
        if (gid[a] < 0) gid[a] = mm.addField(kGName[a]);
    }
    auto& u = s.F(s.id.u);
    auto& v = s.F(s.id.v);
    auto& gx = s.F(gid[0]);
    auto& gy = s.F(gid[1]);
#pragma omp parallel for schedule(static) if (m.nLeaves() > kParMin)
    for (int k = 0; k < m.nLeaves(); ++k) {
        const int c = leaves[k];
        double sg[4] = {0, 0, 0, 0}, sa[4] = {0, 0, 0, 0};
        for (const int* p = m.cellFacesBegin(c); p != m.cellFacesEnd(c); ++p) {
            const Face& F = faces[*p];
            const int sd = 2 * F.dir + (F.L == c ? 1 : 0);
            sg[sd] += F.area * g[*p];
            sa[sd] += F.area;
        }
        gx[c] = 0.5 * (sg[LEFT] / sa[LEFT] + sg[RIGHT] / sa[RIGHT]);
        gy[c] = 0.5 * (sg[BOTTOM] / sa[BOTTOM] + sg[TOP] / sa[TOP]);
        u[c] += dt * gx[c];
        v[c] += dt * gy[c];
    }
    return r;
}

poisson::Result reproject(State& s) {
    std::vector<double> phi(s.m().nCells(), 0.0);
    return projectFaces(s, 1.0, false, phi, nullptr);
}

}  // namespace ns
