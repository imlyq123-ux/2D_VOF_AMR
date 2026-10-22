// curvature.cpp - interface curvature by generalized height functions (Popinet 2009).
//
// For every interfacial leaf c = (L, i, j) (vof::EPS < f < 1 - vof::EPS), working on its
// own level L with Mesh::locate on a restricted copy of f (logical indices: walls are
// mirrored by locate, periodic directions wrapped):
//   1. Youngs normal n = -grad f (3x3). Dominant direction d = argmax |n_d|.
//   2. Heights along d in the 3 columns through c and its two neighbours across d. A column
//      is walked from the row of c up to NMAX cells towards phase 1 until a full cell and
//      up to NMAX cells towards phase 2 until an empty cell ("closed" column of at most
//      2 NMAX + 1 = 11 cells, generalized height = sum of f between them). A pure cell of
//      the wrong kind after an interfacial one invalidates the column (second interface).
//      kappa = -s H'' / (h (1 + H'^2)^1.5), s = +1 if phase 2 is on the +d side.
//   3. Same in the other direction.
//   4. Mixed heights: interface points given by every valid height (both directions) that
//      falls inside the 3x3 block; >= 3 independent points -> least-squares parabola.
//   5. The points of 4 plus the PLIC segment centroids of the interfacial cells of the 3x3
//      (then 5x5) block -> parabola.
//   6. Average of the curvatures of the neighbouring (3x3) interfacial leaves that
//      succeeded, else 0.
// |kappa| is clamped to KMAX / h, or rejected beyond it with rejectUnresolved (see KMAX).
// Non-interfacial leaves having a face neighbour with a different f get the mean
// curvature of the interfacial leaves of their 3x3 (else 5x5) block at their level; all
// other leaves (and non-leaf cells) get 0.
#include "curvature.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include "vof.hpp"

namespace curvature {

// Diagnostic counters of the last compute() call made by the calling thread. Not part of
// the public header (declared by tests/test_curvature.cpp):
//   out[0] interfacial leaves      out[1] HF, dominant direction   out[2] HF, other direction
//   out[3] mixed-height fit        out[4] centroid fit             out[5] neighbour average
//   out[6] no curvature (0)        out[7] extended (non-interfacial) leaves with kappa != 0
void lastStats(long out[8]);

namespace {

constexpr double EPS = vof::EPS;
constexpr int NMAX = 5;        // max cells walked from the row of c in each direction
// |kappa| <= KMAX / h(level).
//
// Unresolved structures (Params::rejectUnresolvedCurvature). For a fragment smaller than
// a cell (droplet or bubble of radius < h) the height functions and fits do not see a
// closed interface: they return values of either sign that hit the clamp, e.g.
// kappa h in [-1, +1] on neighbouring cells of a droplet of radius 0.4 h whose exact
// kappa h is +2.5 everywhere. The balanced-force term sigma kappa_f grad_f f then no
// longer integrates to zero around the fragment and propels it: in the breaking wave at
// levels 4-10 a fragment of 0.4 h radius received net surface-tension kicks of 4-11 g,
// rose against gravity and stuck to the top wall, and an isolated droplet of 0.4 h
// launched upward in still air levitated instead of falling (it falls with sigma = 0,
// and droplets of >= 3 h follow the ballistic path either way). With rejectUnresolved
// a method whose |kappa| exceeds the limit counts as failed (next fallback, else
// kappa = 0): the clamped values disappear, and with them the fragment at the top wall
// (the 0.4 h droplet then falls); resolved interfaces are unchanged (identical results
// for droplets of >= 3 h). It is a partial remedy: depending on the position on the
// grid, the fits can also return wrong values below the limit for fragments of
// 0.4-1.5 h, which are kept (a 1.2 h droplet in the levels 4-10 wave still hovered
// for ~1 time unit before falling). Off by default: it changes results only on meshes
// that produce sub-cell fragments.
constexpr double KMAX = 1.0;
constexpr double HMAX = 1.5;   // mixed fit: heights must fall inside the 3x3 block

enum Method { HF_MAIN = 0, HF_OTHER, MIXED, CENTROID, NEIGHBOUR, NONE, NMETHODS };

thread_local long gStats[8] = {0, 0, 0, 0, 0, 0, 0, 0};

struct Workspace {
    std::vector<double> fr;        // restricted copy of f
    std::vector<uint8_t> has;      // 1: interfacial leaf with a valid curvature
    std::vector<int> ifc;          // interfacial leaves
    std::vector<int8_t> method;    // per ifc entry
    std::vector<int> failed;       // indices into ifc
    std::vector<double> tmp;       // neighbour averages of failed cells
};
thread_local Workspace tWork;

// f at logical index (i, j) of level L.
struct Stencil {
    const Mesh& m;
    const double* f;
    int L;
    double operator()(int i, int j) const { return f[m.locate(L, i, j)]; }
};

// Outward normal of phase 1 (-grad f, Youngs 3x3), not normalized.
Vec2 youngs(const Stencil& S, int i, int j) {
    double v[3][3];
    for (int b = -1; b <= 1; ++b)
        for (int a = -1; a <= 1; ++a) v[a + 1][b + 1] = S(i + a, j + b);
    const double gx = (v[2][0] + 2.0 * v[2][1] + v[2][2]) - (v[0][0] + 2.0 * v[0][1] + v[0][2]);
    const double gy = (v[0][2] + 2.0 * v[1][2] + v[2][2]) - (v[0][0] + 2.0 * v[1][0] + v[2][0]);
    return {-gx, -gy};
}

// Height of the interface in the column through (i, j) along direction d (0: x, 1: y).
// s = +1 if phase 2 lies on the +d side. H = interface coordinate along d relative to the
// centre of (i, j), in cell units. Returns false if the column is not closed / inconsistent.
bool height(const Stencil& S, int i, int j, int d, int s, double& H) {
    auto at = [&](int k) { return d == 0 ? S(i + k, j) : S(i, j + k); };
    const double f0 = at(0);
    double sum = f0;
    int A = 0;  // number of cells walked towards phase 1 up to the full cell
    if (f0 < 1.0 - EPS) {
        bool seen = f0 > EPS, found = false;
        for (int k = 1; k <= NMAX; ++k) {
            const double fk = at(-s * k);
            sum += fk;
            if (fk >= 1.0 - EPS) { A = k; found = true; break; }
            if (fk <= EPS) {
                if (seen) return false;
            } else {
                seen = true;
            }
        }
        if (!found) return false;
    }
    if (f0 > EPS) {
        bool seen = f0 < 1.0 - EPS, found = false;
        for (int k = 1; k <= NMAX; ++k) {
            const double fk = at(s * k);
            sum += fk;
            if (fk <= EPS) { found = true; break; }
            if (fk >= 1.0 - EPS) {
                if (seen) return false;
            } else {
                seen = true;
            }
        }
        if (!found) return false;
    }
    H = s * (sum - A - 0.5);
    return true;
}

// PLIC segment of cell (i, j) of the stencil level: centroid (cell units, relative to the
// cell centre) and length (cell units). Returns false if no segment.
bool plicCentroid(const Stencil& S, int i, int j, double fc, double& cx, double& cy, double& len) {
    Vec2 n = youngs(S, i, j);
    double nn = std::fabs(n.x) + std::fabs(n.y);
    if (!(nn > 1e-12)) { n = Vec2(0.0, 1.0); nn = 1.0; }
    const double mx = n.x / nn, my = n.y / nn;
    const double ax = std::fabs(mx), ay = std::fabs(my);
    const double m1 = std::min(ax, ay), m2 = std::max(ax, ay);
    double alpha;
    if (m1 < 1e-14) {
        alpha = fc;
    } else {
        const double v1 = 0.5 * m1 / m2;
        if (fc <= v1) alpha = std::sqrt(2.0 * m1 * m2 * fc);
        else if (fc <= 1.0 - v1) alpha = fc * m2 + 0.5 * m1;
        else alpha = 1.0 - std::sqrt(2.0 * m1 * m2 * (1.0 - fc));
    }
    // intersections of ax X + ay Y = alpha with the unit square (canonical frame)
    double px[4], py[4];
    int np = 0;
    const double tol = 1e-12;
    auto add = [&](double x, double y) {
        if (x >= -tol && x <= 1.0 + tol && y >= -tol && y <= 1.0 + tol) {
            px[np] = std::clamp(x, 0.0, 1.0);
            py[np] = std::clamp(y, 0.0, 1.0);
            ++np;
        }
    };
    if (ay > 1e-14) { add(0.0, alpha / ay); add(1.0, (alpha - ax) / ay); }
    if (ax > 1e-14) { add(alpha / ax, 0.0); add((alpha - ay) / ax, 1.0); }
    if (np < 2) return false;
    int ba = 0, bb = 1;
    double best = -1.0;
    for (int a = 0; a < np; ++a)
        for (int b = a + 1; b < np; ++b) {
            const double dd = sq(px[a] - px[b]) + sq(py[a] - py[b]);
            if (dd > best) { best = dd; ba = a; bb = b; }
        }
    double X = 0.5 * (px[ba] + px[bb]), Y = 0.5 * (py[ba] + py[bb]);
    if (mx < 0.0) X = 1.0 - X;
    if (my < 0.0) Y = 1.0 - Y;
    cx = X - 0.5;
    cy = Y - 0.5;
    len = std::sqrt(best);
    return true;
}

// Interface points (cell units, relative to the centre of the stencil cell).
struct Points {
    static constexpr int MAXP = 32;
    double x[MAXP], y[MAXP], w[MAXP];
    int n = 0;
    // keeps only points more than 0.5 cells away from the ones already stored
    void add(double px, double py, double pw) {
        if (n >= MAXP || !std::isfinite(px) || !std::isfinite(py)) return;
        for (int k = 0; k < n; ++k)
            if (sq(px - x[k]) + sq(py - y[k]) < 0.25) return;
        x[n] = px; y[n] = py; w[n] = pw; ++n;
    }
};

// Weighted least-squares parabola z = a + b s + c s^2 in the frame (t, n) centred on the
// cell centre (n = unit outward normal). kappa (cell units) = -2c / (1 + b^2)^1.5.
bool fitParabola(const Points& P, Vec2 n, double& kap) {
    if (P.n < 3) return false;
    const double tx = -n.y, ty = n.x;
    double M[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, r[3] = {0, 0, 0}, W = 0.0;
    for (int k = 0; k < P.n; ++k) {
        const double s = P.x[k] * tx + P.y[k] * ty, z = P.x[k] * n.x + P.y[k] * n.y;
        const double phi[3] = {1.0, s, s * s}, w = P.w[k];
        W += w;
        for (int a = 0; a < 3; ++a) {
            r[a] += w * phi[a] * z;
            for (int b = 0; b < 3; ++b) M[a][b] += w * phi[a] * phi[b];
        }
    }
    auto det3 = [](const double A[3][3]) {
        return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
               A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
               A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    };
    const double D = det3(M);
    if (!(std::fabs(D) > 1e-5 * W * W * W)) return false;
    double Mb[3][3], Mc[3][3];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) Mb[a][b] = Mc[a][b] = M[a][b];
    for (int a = 0; a < 3; ++a) { Mb[a][1] = r[a]; Mc[a][2] = r[a]; }
    const double b = det3(Mb) / D, c = det3(Mc) / D;
    kap = -2.0 * c / std::pow(1.0 + b * b, 1.5);
    return std::isfinite(kap);
}

// Curvature of interfacial leaf c (physical units, |kap| <= KMAX / h). Returns the Method
// used; with reject, a method whose value exceeds the limit fails instead of clamping.
int cellKappa(const Mesh& m, const double* fr, int c, bool reject, double& kap) {
    const int L = m.level(c), i = m.ix(c), j = m.iy(c);
    const double h = m.h(L), kmax = KMAX / h;
    auto accept = [&](double k) { return !reject || (std::isfinite(k) && std::fabs(k) <= kmax); };
    const Stencil S{m, fr, L};
    const Vec2 n = youngs(S, i, j);
    const int d0 = std::fabs(n.x) > std::fabs(n.y) ? 0 : 1;
    double H[2][3] = {{0, 0, 0}, {0, 0, 0}};
    bool ok[2][3] = {{false, false, false}, {false, false, false}};

    // ---- height functions, dominant direction first
    for (int pass = 0; pass < 2; ++pass) {
        const int d = pass == 0 ? d0 : 1 - d0;
        const double comp = d == 0 ? n.x : n.y;
        if (comp == 0.0) continue;
        const int s = comp > 0.0 ? 1 : -1;
        for (int k = -1; k <= 1; ++k)
            ok[d][k + 1] = d == 0 ? height(S, i, j + k, 0, s, H[d][k + 1])
                                  : height(S, i + k, j, 1, s, H[d][k + 1]);
        if (ok[d][0] && ok[d][1] && ok[d][2]) {
            const double hp = 0.5 * (H[d][2] - H[d][0]);
            const double hpp = H[d][2] - 2.0 * H[d][1] + H[d][0];
            const double k = -s * hpp / (h * std::pow(1.0 + hp * hp, 1.5));
            if (std::isfinite(k) && accept(k)) {
                kap = std::clamp(k, -kmax, kmax);
                return pass == 0 ? HF_MAIN : HF_OTHER;
            }
        }
    }

    double nn = n.norm();
    const Vec2 nu = nn > 1e-12 ? Vec2(n.x / nn, n.y / nn) : Vec2(0.0, 1.0);
    double k = 0.0;

    // ---- mixed heights
    Points P;
    for (int d = 0; d < 2; ++d)
        for (int q = 0; q < 3; ++q) {
            if (!ok[d][q] || std::fabs(H[d][q]) > HMAX) continue;
            if (d == 0) P.add(H[d][q], q - 1.0, 1.0);
            else P.add(q - 1.0, H[d][q], 1.0);
        }
    if (fitParabola(P, nu, k) && accept(k / h)) {
        kap = std::clamp(k / h, -kmax, kmax);
        return MIXED;
    }

    // ---- PLIC centroids of the 3x3 (then 5x5) block, added to the valid height points
    for (int r = 1; r <= 2; ++r) {
        Points C = P;
        for (int b = -r; b <= r; ++b)
            for (int a = -r; a <= r; ++a) {
                const double fa = S(i + a, j + b);
                if (fa <= EPS || fa >= 1.0 - EPS) continue;
                double cx, cy, len;
                if (plicCentroid(S, i + a, j + b, fa, cx, cy, len))
                    C.add(a + cx, b + cy, std::max(len, 1e-3));
            }
        if (fitParabola(C, nu, k) && accept(k / h)) {
            kap = std::clamp(k / h, -kmax, kmax);
            return CENTROID;
        }
    }
    kap = 0.0;
    return NONE;
}

// Adds the curvature of the leaves represented by cell n (n itself if it is a leaf, its
// leaf children otherwise) to (sum, cnt) if they are interfacial with a valid value.
inline void gather(const Mesh& m, const std::vector<uint8_t>& has, const std::vector<double>& kappa,
                   int n, double& sum, int& cnt) {
    if (m.isLeaf(n)) {
        if (has[n]) { sum += kappa[n]; ++cnt; }
        return;
    }
    for (int q = 0; q < 4; ++q) {
        const int ch = m.child(n, q);
        if (m.isLeaf(ch) && has[ch]) { sum += kappa[ch]; ++cnt; }
    }
}

// Mean curvature of the valid interfacial leaves of the (2r+1)^2 block around c at its
// level (c itself excluded). Returns false if there is none.
bool neighbourMean(const Mesh& m, const std::vector<uint8_t>& has, const std::vector<double>& kappa,
                   int c, int r, double& out) {
    const int L = m.level(c), i = m.ix(c), j = m.iy(c);
    double sum = 0.0;
    int cnt = 0;
    for (int b = -r; b <= r; ++b)
        for (int a = -r; a <= r; ++a) {
            if (!a && !b) continue;
            const int n = m.locate(L, i + a, j + b);
            if (n != c) gather(m, has, kappa, n, sum, cnt);
        }
    if (!cnt) return false;
    out = sum / cnt;
    return std::isfinite(out);
}

}  // namespace

void lastStats(long out[8]) {
    for (int k = 0; k < 8; ++k) out[k] = gStats[k];
}

void compute(const Mesh& m, const std::vector<double>& f, std::vector<double>& kappa, bool rejectUnresolved) {
    Workspace& W = tWork;  // bind before parallel regions (thread_local!)
    const int nc = m.nCells();
    if ((int)kappa.size() != nc) kappa.assign(nc, 0.0);

    std::vector<double>& fr = W.fr;
    fr.assign(f.begin(), f.end());
    m.restrict(fr);
    std::vector<uint8_t>& has = W.has;
    has.assign(nc, 0);

    // interfacial leaves
    std::vector<int>& ifc = W.ifc;
    ifc.clear();
    for (int c : m.leaves())
        if (f[c] > EPS && f[c] < 1.0 - EPS) ifc.push_back(c);
    const int ni = (int)ifc.size();
    std::vector<int8_t>& method = W.method;
    method.assign(ni, (int8_t)NONE);

    // ---- pass 1: height functions and fits
    const double* frp = fr.data();
#pragma omp parallel for schedule(dynamic, 64) if (ni > 4096)
    for (int q = 0; q < ni; ++q) {
        const int c = ifc[q];
        double k = 0.0;
        const int meth = cellKappa(m, frp, c, rejectUnresolved, k);
        method[q] = (int8_t)meth;
        kappa[c] = std::isfinite(k) ? k : 0.0;
        if (meth != NONE) has[c] = 1;
    }

    // ---- pass 2: failed cells get the mean of their successful neighbours
    std::vector<int>& failed = W.failed;
    failed.clear();
    for (int q = 0; q < ni; ++q)
        if (method[q] == NONE) failed.push_back(q);
    const int nf = (int)failed.size();
    std::vector<double>& tmp = W.tmp;
    tmp.assign(nf, 0.0);
#pragma omp parallel for schedule(dynamic, 16) if (nf > 4096)
    for (int q = 0; q < nf; ++q) {
        const int c = ifc[failed[q]];
        double v = 0.0;
        if (neighbourMean(m, has, kappa, c, 1, v)) {
            const double kmax = KMAX / m.size(c);
            tmp[q] = std::clamp(v, -kmax, kmax);
            method[failed[q]] = (int8_t)NEIGHBOUR;
        }
    }
    for (int q = 0; q < nf; ++q) {
        const int c = ifc[failed[q]];
        kappa[c] = tmp[q];
        if (method[failed[q]] == NEIGHBOUR) has[c] = 1;
    }

    // ---- pass 3: extension to non-interfacial leaves next to a jump of f; zero elsewhere
    const std::vector<Mesh::Face>& faces = m.faces();
    long nExt = 0;
#pragma omp parallel for schedule(static) reduction(+ : nExt) if (nc > 200000)
    for (int c = 0; c < nc; ++c) {
        if (!m.isLeaf(c)) { kappa[c] = 0.0; continue; }
        const double fc = f[c];
        if (fc > EPS && fc < 1.0 - EPS) continue;  // interfacial: done
        bool jump = false;
        for (const int* p = m.cellFacesBegin(c); p != m.cellFacesEnd(c) && !jump; ++p) {
            const Mesh::Face& F = faces[*p];
            const int o = F.L == c ? F.R : F.L;
            if (o >= 0 && f[o] != fc) jump = true;
        }
        double v = 0.0;
        if (jump && (neighbourMean(m, has, kappa, c, 1, v) || neighbourMean(m, has, kappa, c, 2, v))) {
            const double kmax = KMAX / m.size(c);
            v = std::clamp(v, -kmax, kmax);
            if (v != 0.0) ++nExt;
        } else {
            v = 0.0;
        }
        kappa[c] = v;
    }

    // ---- statistics
    long cnt[NMETHODS] = {0, 0, 0, 0, 0, 0};
    for (int q = 0; q < ni; ++q) cnt[method[q]]++;
    gStats[0] = ni;
    for (int k = 0; k < NMETHODS; ++k) gStats[1 + k] = cnt[k];
    gStats[7] = nExt;
}

}  // namespace curvature
