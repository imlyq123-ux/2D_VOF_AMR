// poisson.cpp - finite-volume elliptic solver on the AMR leaves (see poisson.hpp).
//
// Algorithm
//   * The operator is assembled as a CSR matrix over the leaves, rows numbered in quadtree
//     Z-order (see zOrder); diagonal kept apart from the off-diagonal entries. The rows are
//     the integrated (volume-multiplied) equations, so the matrix is symmetric, an
//     M-matrix (off-diagonals < 0, weakly diagonally dominant).
//   * Outer Krylov solver: flexible CG ("FCG(1)", Notay 2000) - identical to PCG for a
//     fixed SPD preconditioner, and still robust with the (slightly nonlinear) K-cycle.
//   * Preconditioner (Options::precond = 1): unsmoothed aggregation AMG in the spirit of
//     Notay's AGMG:
//       - coarsening by double pairwise aggregation: two passes of greedy matching, each
//         pairing a node with its strongest negative coupling (strength threshold 0.25
//         of the row's largest negative coupling), giving aggregates of <= 4 nodes; rows
//         with a_ii >= 5 sum|a_ij| are left out of the coarse space (smoother handles
//         them; important for Helmholtz systems with large lambda);
//       - piecewise-constant prolongation P, Galerkin coarse matrices P^T A P (exact
//         sums, the coarse matrices stay M-matrices and keep the constant null space);
//       - K-cycle: on each level the coarse correction is two iterations of FCG
//         preconditioned by the next level's cycle (one if the first already reduces the
//         residual by 4) - this is what makes unsmoothed aggregation scalable;
//       - smoother: one forward Gauss-Seidel sweep before and one backward sweep after
//         the coarse correction. For OpenMP the sweep is "hybrid": GS inside contiguous
//         blocks of kBlock rows, block-Jacobi between blocks (block size fixed, so the
//         result does not depend on the thread count);
//       - coarsest level (<= kCoarseMax unknowns): dense Cholesky. In the singular
//         (pure Neumann) case the coarsest matrix is regularized with s*1*1^T, which
//         gives the exact minimum-norm solve for compatible right-hand sides.
//   * Options::precond = 0: Jacobi preconditioner (fallback / reference).
//   * Singular case: b's volume-weighted mean is removed, the preconditioned residual is
//     projected onto the range (Euclidean mean removed), the final solution is shifted
//     to zero volume-weighted mean.
//   * Everything is rebuilt on every call (setup ~ a few SpMV-equivalents per level).
//   * Options::gpu (CUDA builds): the system and hierarchy are handed to the device solver
//     (poisson_gpu.cu), which runs the same FCG/K-cycle iteration with a multicolour
//     Gauss-Seidel smoother. Below level kGpuAggressive the hierarchy coarsens by triple
//     pairwise aggregation (aggregates of <= 8): the K-cycle visits level l up to 2^l
//     times, and on the GPU every visit of a small level costs a few microseconds of
//     barrier latency whatever its size, so fewer levels pay off; the convergence is
//     practically unchanged.
#include "poisson.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#ifdef VOF2D_CUDA
#include "poisson_gpu.hpp"
#endif
#ifdef _OPENMP
#include <omp.h>
#else
#pragma GCC diagnostic ignored "-Wunknown-pragmas"  // serial build: omp pragmas are inert
#endif

namespace poisson {
namespace {

constexpr int kPar = 8192;          // minimum vector length for OpenMP loops
constexpr int kPerThread = 32768;   // rows per thread (few threads on small levels)
constexpr int kBlock = 4096;        // hybrid Gauss-Seidel block size
constexpr int kCoarseMax = 100;     // stop coarsening at this size
constexpr int kDenseMax = 800;      // dense Cholesky on the coarsest level up to this size
constexpr int kMaxLevels = 40;
constexpr double kStrong = 0.25;    // strength of connection threshold
constexpr double kDiagDom = 5.0;    // rows with a_ii >= kDiagDom sum|a_ij| are not coarsened
constexpr double kKTol = 0.25;      // K-cycle: skip the 2nd inner iteration below this ratio
constexpr int kGpuMin = 30000;      // Options::gpu: smaller systems are solved on the CPU
constexpr int kGpuAggressive = 2;   // Options::gpu: levels >= this coarsen by aggregates of 8

// Number of threads for a loop of length n: small levels use few threads (barrier cost,
// robustness when the machine is oversubscribed).
inline int nthr(long n) {
#ifdef _OPENMP
    const long t = std::min<long>(omp_get_max_threads(), std::max<long>(1, n / kPerThread));
    return (int)t;
#else
    (void)n;
    return 1;
#endif
}

// Uninitialized scratch array (avoids zero-filling large temporaries).
template <class T>
std::unique_ptr<T[]> scratch(size_t n) {
    return std::unique_ptr<T[]>(new T[n]);
}

double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Symmetric sparse matrix: diagonal + off-diagonal CSR (columns sorted).
struct Csr {
    int n = 0;
    std::vector<int> ptr, col;
    std::vector<double> val;
    std::vector<double> diag;
    size_t nnz() const { return val.size() + (size_t)n; }
};

void spmv(const Csr& A, const double* x, double* y) {
    const int n = A.n;
    const int* ptr = A.ptr.data();
    const int* col = A.col.data();
    const double* val = A.val.data();
    const double* dg = A.diag.data();
#pragma omp parallel for schedule(static) num_threads(nthr(n)) if (n > kPar)
    for (int i = 0; i < n; ++i) {
        double s = dg[i] * x[i];
        for (int k = ptr[i]; k < ptr[i + 1]; ++k) s += val[k] * x[col[k]];
        y[i] = s;
    }
}

// r = b - A x
void residual(const Csr& A, const double* b, const double* x, double* r) {
    const int n = A.n;
    const int* ptr = A.ptr.data();
    const int* col = A.col.data();
    const double* val = A.val.data();
    const double* dg = A.diag.data();
#pragma omp parallel for schedule(static) num_threads(nthr(n)) if (n > kPar)
    for (int i = 0; i < n; ++i) {
        double s = b[i] - dg[i] * x[i];
        for (int k = ptr[i]; k < ptr[i + 1]; ++k) s -= val[k] * x[col[k]];
        r[i] = s;
    }
}

double dot(const double* a, const double* b, int n) {
    double s = 0.0;
#pragma omp parallel for reduction(+ : s) schedule(static) num_threads(nthr(n)) if (n > kPar)
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// z = forward hybrid GS sweep for A z = r starting from z = 0.
void gsForward(const Csr& A, const double* invd, const double* r, double* z) {
    const int n = A.n, nb = (n + kBlock - 1) / kBlock;
    const int* ptr = A.ptr.data();
    const int* col = A.col.data();
    const double* val = A.val.data();
#pragma omp parallel for schedule(static) num_threads(std::min(nb, nthr((long)n))) if (nb > 1)
    for (int b = 0; b < nb; ++b) {
        const int lo = b * kBlock, hi = std::min(n, lo + kBlock);
        for (int i = lo; i < hi; ++i) {
            double s = r[i];
            for (int k = ptr[i]; k < ptr[i + 1]; ++k) {
                const int j = col[k];
                if (j >= lo && j < i) s -= val[k] * z[j];
            }
            z[i] = s * invd[i];
        }
    }
}

// z += backward hybrid GS sweep for A e = r starting from e = 0 (e: work array).
void gsBackwardAdd(const Csr& A, const double* invd, const double* r, double* e, double* z) {
    const int n = A.n, nb = (n + kBlock - 1) / kBlock;
    const int* ptr = A.ptr.data();
    const int* col = A.col.data();
    const double* val = A.val.data();
#pragma omp parallel for schedule(static) num_threads(std::min(nb, nthr((long)n))) if (nb > 1)
    for (int b = 0; b < nb; ++b) {
        const int lo = b * kBlock, hi = std::min(n, lo + kBlock);
        for (int i = hi - 1; i >= lo; --i) {
            double s = r[i];
            for (int k = ptr[i]; k < ptr[i + 1]; ++k) {
                const int j = col[k];
                if (j > i && j < hi) s -= val[k] * e[j];
            }
            e[i] = s * invd[i];
        }
        for (int i = lo; i < hi; ++i) z[i] += e[i];
    }
}

// rc[I] = sum over the members i of aggregate I of (r - A z)_i.
void restrictResidual(const Csr& A, const std::vector<int>& mptr, const std::vector<int>& mem, const double* r,
                      const double* z, double* rc) {
    const int nc = (int)mptr.size() - 1;
    const int* ptr = A.ptr.data();
    const int* col = A.col.data();
    const double* val = A.val.data();
    const double* dg = A.diag.data();
    const int* mp = mptr.data();
    const int* mm = mem.data();
#pragma omp parallel for schedule(static) num_threads(nthr(4L * nc)) if (nc > kPar / 4)
    for (int I = 0; I < nc; ++I) {
        double s = 0.0;
        for (int q = mp[I]; q < mp[I + 1]; ++q) {
            const int i = mm[q];
            s += r[i] - dg[i] * z[i];
            for (int k = ptr[i]; k < ptr[i + 1]; ++k) s -= val[k] * z[col[k]];
        }
        rc[I] = s;
    }
}

// One pass of greedy pairwise matching. agg[i] = pair index, -1 = left out of the
// coarse space (only if excludeDD). Returns the number of pairs/singletons.
int pairwise(const Csr& A, std::vector<int>& agg, bool excludeDD) {
    const int n = A.n;
    agg.assign(n, -2);
    std::vector<double> thr(n);
#pragma omp parallel for schedule(static) num_threads(nthr(n)) if (n > kPar)
    for (int i = 0; i < n; ++i) {
        double mx = 0.0, off = 0.0;
        for (int k = A.ptr[i]; k < A.ptr[i + 1]; ++k) {
            off += std::fabs(A.val[k]);
            mx = std::max(mx, -A.val[k]);
        }
        thr[i] = -kStrong * mx;
        if (excludeDD && A.diag[i] >= kDiagDom * off) agg[i] = -1;
    }
    int nc = 0;
    for (int i = 0; i < n; ++i) {
        if (agg[i] != -2) continue;
        int best = -1;
        double bv = thr[i];
        for (int k = A.ptr[i]; k < A.ptr[i + 1]; ++k) {
            const int j = A.col[k];
            if (agg[j] == -2 && A.val[k] < bv) { bv = A.val[k]; best = j; }
        }
        agg[i] = nc;
        if (best >= 0) agg[best] = nc;
        ++nc;
    }
    return nc;
}

// Members of each aggregate (CSR: aggregate -> fine rows, ascending).
void members(const std::vector<int>& agg, int nc, std::vector<int>& mptr, std::vector<int>& mem) {
    mptr.assign(nc + 1, 0);
    for (int a : agg) if (a >= 0) mptr[a + 1]++;
    for (int I = 0; I < nc; ++I) mptr[I + 1] += mptr[I];
    mem.resize(mptr[nc]);
    std::vector<int> pos(mptr.begin(), mptr.end() - 1);
    for (int i = 0; i < (int)agg.size(); ++i) if (agg[i] >= 0) mem[pos[agg[i]]++] = i;
}

// Ac = P^T A P for the piecewise-constant prolongation defined by agg.
void galerkin(const Csr& A, const std::vector<int>& agg, int nc, const std::vector<int>& mptr,
              const std::vector<int>& mem, Csr& Ac) {
    std::vector<int> ub(nc + 1, 0);
#pragma omp parallel for schedule(static) num_threads(nthr(nc)) if (nc > kPar)
    for (int I = 0; I < nc; ++I) {
        int s = 0;
        for (int q = mptr[I]; q < mptr[I + 1]; ++q) s += A.ptr[mem[q] + 1] - A.ptr[mem[q]];
        ub[I + 1] = s;
    }
    for (int I = 0; I < nc; ++I) ub[I + 1] += ub[I];
    std::vector<int> cnt(nc);
    auto tcolBuf = scratch<int>(ub[nc]);
    auto tvalBuf = scratch<double>(ub[nc]);
    int* tcol = tcolBuf.get();
    double* tval = tvalBuf.get();
    Ac.n = nc;
    Ac.diag.assign(nc, 0.0);
#pragma omp parallel for schedule(static) num_threads(nthr(nc)) if (nc > kPar / 4)
    for (int I = 0; I < nc; ++I) {
        const int base = ub[I];
        int c = 0;
        double d = 0.0;
        for (int q = mptr[I]; q < mptr[I + 1]; ++q) {
            const int i = mem[q];
            d += A.diag[i];
            for (int k = A.ptr[i]; k < A.ptr[i + 1]; ++k) {
                const int J = agg[A.col[k]];
                if (J < 0) continue;
                const double v = A.val[k];
                if (J == I) { d += v; continue; }
                int t = base;
                while (t < base + c && tcol[t] != J) ++t;
                if (t == base + c) { tcol[t] = J; tval[t] = v; ++c; } else tval[t] += v;
            }
        }
        // insertion sort by column (rows are short)
        for (int a = base + 1; a < base + c; ++a) {
            const int cj = tcol[a];
            const double vj = tval[a];
            int b = a - 1;
            while (b >= base && tcol[b] > cj) { tcol[b + 1] = tcol[b]; tval[b + 1] = tval[b]; --b; }
            tcol[b + 1] = cj;
            tval[b + 1] = vj;
        }
        Ac.diag[I] = d;
        cnt[I] = c;
    }
    Ac.ptr.assign(nc + 1, 0);
    for (int I = 0; I < nc; ++I) Ac.ptr[I + 1] = Ac.ptr[I] + cnt[I];
    Ac.col.resize(Ac.ptr[nc]);
    Ac.val.resize(Ac.ptr[nc]);
#pragma omp parallel for schedule(static) num_threads(nthr(nc)) if (nc > kPar)
    for (int I = 0; I < nc; ++I) {
        std::copy(tcol + ub[I], tcol + ub[I] + cnt[I], Ac.col.begin() + Ac.ptr[I]);
        std::copy(tval + ub[I], tval + ub[I] + cnt[I], Ac.val.begin() + Ac.ptr[I]);
    }
}

std::vector<double> invDiag(const Csr& A) {
    std::vector<double> d(A.n);
    for (int i = 0; i < A.n; ++i) d[i] = A.diag[i] > 0.0 ? 1.0 / A.diag[i] : 0.0;
    return d;
}

// ------------------------------------------------------------------ AMG hierarchy

struct Level {
    Csr A;
    std::vector<double> invd;
    std::vector<int> agg;        // row -> next-level aggregate (-1: not in coarse space)
    std::vector<int> mptr, mem;  // next-level aggregate -> rows
    std::vector<double> res, e;  // size n
    // coarse-correction buffers, size of the next level
    std::vector<double> rc, ec, v1, w1, v2, w2, tc;
};

class Amg {
public:
    // Levels l >= aggressiveFrom are coarsened by three pairwise passes instead of two.
    void setup(Csr&& A0, bool singular, int aggressiveFrom = kMaxLevels);
    void apply(const double* r, double* z) { cycle(0, r, z); }
    int nLevels() const { return (int)lv_.size(); }
    const Csr& matrix(int l) const { return lv_[l].A; }
    const Level& level(int l) const { return lv_[l]; }
    bool denseCoarse() const { return dense_; }
    // lower Cholesky factor of the coarsest matrix (row-major), null if not dense
    const double* coarseFactor() const { return dense_ ? chol_.data() : nullptr; }
    bool regularized() const { return regularized_; }

private:
    std::vector<Level> lv_;
    bool singular_ = false;
    bool dense_ = false;
    bool regularized_ = false;
    std::vector<double> chol_;   // dense lower Cholesky factor of the coarsest matrix

    void cycle(int l, const double* r, double* z);
    void coarseSolve(const double* r, double* z);
    void factorCoarse();
};

void Amg::setup(Csr&& A0, bool singular, int aggressiveFrom) {
    singular_ = singular;
    lv_.clear();
    lv_.emplace_back();
    lv_[0].A = std::move(A0);
    for (;;) {
        Level& L = lv_.back();
        L.invd = invDiag(L.A);
        L.res.assign(L.A.n, 0.0);
        L.e.assign(L.A.n, 0.0);
        if (L.A.n <= kCoarseMax || (int)lv_.size() >= kMaxLevels) break;
        // double (triple) pairwise aggregation
        std::vector<int> agg1, agg2, mp1, mm1, mp2, mm2;
        const int n1 = pairwise(L.A, agg1, true);
        if (n1 == 0) break;  // everything strongly diagonally dominant: smoother suffices
        members(agg1, n1, mp1, mm1);
        Csr A1;
        galerkin(L.A, agg1, n1, mp1, mm1, A1);
        int n2 = pairwise(A1, agg2, false);
        members(agg2, n2, mp2, mm2);
        Csr A2;
        galerkin(A1, agg2, n2, mp2, mm2, A2);
        if ((int)lv_.size() - 1 >= aggressiveFrom) {
            std::vector<int> agg3, mp3, mm3;
            const int n3 = pairwise(A2, agg3, false);
            members(agg3, n3, mp3, mm3);
            Csr A3;
            galerkin(A2, agg3, n3, mp3, mm3, A3);
            for (int& a : agg2) a = agg3[a];
            A2 = std::move(A3);
            n2 = n3;
        }
        if (n2 > 0.85 * L.A.n) break;  // coarsening stalled
        L.agg.resize(L.A.n);
        for (int i = 0; i < L.A.n; ++i) L.agg[i] = agg1[i] >= 0 ? agg2[agg1[i]] : -1;
        members(L.agg, n2, L.mptr, L.mem);
        for (auto* v : {&L.rc, &L.ec, &L.v1, &L.w1, &L.v2, &L.w2, &L.tc}) v->assign(n2, 0.0);
        lv_.emplace_back();
        lv_.back().A = std::move(A2);
    }
    // the last level must not keep a coarse map
    lv_.back().agg.clear();
    factorCoarse();
}

void Amg::factorCoarse() {
    const Csr& A = lv_.back().A;
    const int n = A.n;
    dense_ = n <= kDenseMax;
    regularized_ = false;
    if (!dense_) return;
    for (int attempt = 0; attempt < 2; ++attempt) {
        const bool reg = singular_ || attempt == 1;
        chol_.assign((size_t)n * n, 0.0);
        double dmean = 0.0;
        for (int i = 0; i < n; ++i) dmean += A.diag[i];
        dmean = n > 0 ? dmean / n : 1.0;
        if (!(dmean > 0.0)) dmean = 1.0;
        const double s = reg ? dmean / n : 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) chol_[(size_t)i * n + j] = s;
            chol_[(size_t)i * n + i] += A.diag[i];
            for (int k = A.ptr[i]; k < A.ptr[i + 1]; ++k) chol_[(size_t)i * n + A.col[k]] += A.val[k];
        }
        bool ok = true;
        for (int j = 0; j < n && ok; ++j) {
            double* Lj = &chol_[(size_t)j * n];
            double d = Lj[j];
            for (int k = 0; k < j; ++k) d -= Lj[k] * Lj[k];
            if (!(d > 1e-13 * std::max(A.diag[j], 1e-300))) { ok = false; break; }
            d = std::sqrt(d);
            Lj[j] = d;
            for (int i = j + 1; i < n; ++i) {
                double* Li = &chol_[(size_t)i * n];
                double v = Li[j];
                for (int k = 0; k < j; ++k) v -= Li[k] * Lj[k];
                Li[j] = v / d;
            }
        }
        if (ok) { regularized_ = reg; return; }
        if (reg) break;
    }
    dense_ = false;  // fall back to smoothing on the coarsest level
}

void Amg::coarseSolve(const double* r, double* z) {
    Level& L = lv_.back();
    const int n = L.A.n;
    if (dense_) {
        double mean = 0.0;
        if (regularized_) {
            for (int i = 0; i < n; ++i) mean += r[i];
            mean /= n;
        }
        for (int i = 0; i < n; ++i) {
            const double* Li = &chol_[(size_t)i * n];
            double v = r[i] - mean;
            for (int k = 0; k < i; ++k) v -= Li[k] * z[k];
            z[i] = v / Li[i];
        }
        for (int i = n - 1; i >= 0; --i) {
            double v = z[i];
            for (int k = i + 1; k < n; ++k) v -= chol_[(size_t)k * n + i] * z[k];
            z[i] = v / chol_[(size_t)i * n + i];
        }
        return;
    }
    // large coarsest level (strongly diagonally dominant or stalled coarsening):
    // a few symmetric hybrid GS sweeps
    gsForward(L.A, L.invd.data(), r, z);
    for (int s = 0; s < 3; ++s) {
        residual(L.A, r, z, L.res.data());
        gsBackwardAdd(L.A, L.invd.data(), L.res.data(), L.e.data(), z);
        residual(L.A, r, z, L.res.data());
        gsBackwardAdd(L.A, L.invd.data(), L.res.data(), L.e.data(), z);
    }
}

void Amg::cycle(int l, const double* r, double* z) {
    if (l + 1 == (int)lv_.size()) { coarseSolve(r, z); return; }
    Level& L = lv_[l];
    const Csr& A = L.A;
    const int n = A.n;
    const Csr& Ac = lv_[l + 1].A;
    const int nc = Ac.n;
    double* res = L.res.data();
    gsForward(A, L.invd.data(), r, z);
    double* rc = L.rc.data();
    double* ec = L.ec.data();
    restrictResidual(A, L.mptr, L.mem, r, z, rc);
    if (l + 2 == (int)lv_.size()) {
        coarseSolve(rc, ec);
    } else {
        // K-cycle: two FCG iterations on the next level
        double* v1 = L.v1.data();
        double* w1 = L.w1.data();
        cycle(l + 1, rc, v1);
        spmv(Ac, v1, w1);
        double rho1 = 0.0, a1 = 0.0;
#pragma omp parallel for reduction(+ : rho1, a1) schedule(static) num_threads(nthr(nc)) if (nc > kPar)
        for (int I = 0; I < nc; ++I) { rho1 += v1[I] * w1[I]; a1 += v1[I] * rc[I]; }
        if (!(rho1 > 0.0)) {
            std::fill(ec, ec + nc, 0.0);
        } else {
            const double al = a1 / rho1;
            double* tc = L.tc.data();
            double rn = 0.0, tn = 0.0;
#pragma omp parallel for reduction(+ : rn, tn) schedule(static) num_threads(nthr(nc)) if (nc > kPar)
            for (int I = 0; I < nc; ++I) {
                tc[I] = rc[I] - al * w1[I];
                rn += rc[I] * rc[I];
                tn += tc[I] * tc[I];
            }
            if (tn <= kKTol * kKTol * rn) {
#pragma omp parallel for schedule(static) num_threads(nthr(nc)) if (nc > kPar)
                for (int I = 0; I < nc; ++I) ec[I] = al * v1[I];
            } else {
                double* v2 = L.v2.data();
                double* w2 = L.w2.data();
                cycle(l + 1, tc, v2);
                spmv(Ac, v2, w2);
                double gam = 0.0, bet = 0.0, a2 = 0.0;
#pragma omp parallel for reduction(+ : gam, bet, a2) schedule(static) num_threads(nthr(nc)) if (nc > kPar)
                for (int I = 0; I < nc; ++I) {
                    gam += v2[I] * w1[I];
                    bet += v2[I] * w2[I];
                    a2 += v2[I] * tc[I];
                }
                const double rho2 = bet - gam * gam / rho1;
                const double c1 = rho2 > 0.0 ? al - gam * a2 / (rho1 * rho2) : al;
                const double c2 = rho2 > 0.0 ? a2 / rho2 : 0.0;
#pragma omp parallel for schedule(static) num_threads(nthr(nc)) if (nc > kPar)
                for (int I = 0; I < nc; ++I) ec[I] = c1 * v1[I] + c2 * v2[I];
            }
        }
    }
    {
        const int* agg = L.agg.data();
#pragma omp parallel for schedule(static) num_threads(nthr(n)) if (n > kPar)
        for (int i = 0; i < n; ++i)
            if (agg[i] >= 0) z[i] += ec[agg[i]];
    }
    residual(A, r, z, res);
    gsBackwardAdd(A, L.invd.data(), res, L.e.data(), z);
}

// ------------------------------------------------------------------ assembly

// Leaves in quadtree Z-order (depth-first, children 0..3, roots row-major). With this
// row numbering the 4 siblings of a refined cell are consecutive, so the greedy matching
// (ties broken by ascending column) forms the sibling 2x2 boxes wherever the couplings
// are uniform - aggregates of geometric-multigrid quality on AMR meshes - and follows
// the strength of connection elsewhere. It also improves locality.
void zOrder(const Mesh& m, std::vector<int>& order, std::vector<int>& rowOfLeaf) {
    const int n = m.nLeaves();
    order.clear();
    order.reserve(n);
    rowOfLeaf.assign(n, -1);
    std::vector<int> stack;
    for (int jr = 0; jr < m.ny(0); ++jr)
        for (int ir = 0; ir < m.nx(0); ++ir) {
            stack.push_back(m.find(0, ir, jr));
            while (!stack.empty()) {
                const int c = stack.back();
                stack.pop_back();
                if (m.isLeaf(c)) {
                    rowOfLeaf[m.leafIndex(c)] = (int)order.size();
                    order.push_back(c);
                } else {
                    for (int k = 3; k >= 0; --k) stack.push_back(m.child(c, k));
                }
            }
        }
}

// Topology-only part of the operator, cached between solves on the same mesh (the
// driver solves three systems per time step and regrids only every few steps): Z-order
// rows, the sorted off-diagonal CSR pattern and, per row, its faces with the CSR
// position they contribute to. Assembling the values is then a streaming pass.
struct Pattern {
    uint64_t revision = 0;       // Mesh::revision() it was built for
    std::vector<int> order;      // row -> cell index (Z-order)
    std::vector<int> rowOfLeaf;  // leaf ordinal -> row
    std::vector<double> vol;     // V per row
    std::vector<int> ptr, col;   // off-diagonal CSR (columns sorted)
    std::vector<int> fptr;       // row -> its faces [fptr[k], fptr[k+1])
    std::vector<int> face;       // Mesh face index
    std::vector<int> slot;       // CSR position; -1: boundary face, -2: periodic self-face
    std::vector<double> area, dist;
};

// Pattern of m, rebuilt when the mesh changed (poisson::solve is not reentrant).
const Pattern& pattern(const Mesh& m) {
    static Pattern P;
    if (P.revision == m.revision()) return P;
    P.revision = m.revision();
    const int n = m.nLeaves();
    zOrder(m, P.order, P.rowOfLeaf);
    const auto& faces = m.faces();
    P.vol.resize(n);
    P.fptr.assign(n + 1, 0);
    for (int k = 0; k < n; ++k) {
        const int c = P.order[k];
        P.vol[k] = m.volume(c);
        P.fptr[k + 1] = P.fptr[k] + (int)(m.cellFacesEnd(c) - m.cellFacesBegin(c));
    }
    const int ne = P.fptr[n];
    P.face.resize(ne);
    P.slot.resize(ne);
    P.area.resize(ne);
    P.dist.resize(ne);
    constexpr int W = 8;  // max off-diagonal entries per row (<= 8 faces by 2:1 balance)
    std::vector<int> tc((size_t)W * n), cnt(n);
#pragma omp parallel for schedule(static) num_threads(nthr(n)) if (n > kPar)
    for (int k = 0; k < n; ++k) {
        const int c = P.order[k];
        int* rc = &tc[(size_t)W * k];
        int q = 0, e = P.fptr[k];
        for (const int* f = m.cellFacesBegin(c); f != m.cellFacesEnd(c); ++f, ++e) {
            const Mesh::Face& F = faces[*f];
            const int nb = (F.L == c) ? F.R : F.L;
            P.face[e] = *f;
            P.area[e] = F.area;
            P.dist[e] = F.dist;
            if (nb < 0) { P.slot[e] = -1; continue; }
            if (nb == c) { P.slot[e] = -2; continue; }  // single cell across a periodic domain
            const int j = P.rowOfLeaf[m.leafIndex(nb)];
            P.slot[e] = j;  // column for now
            int t = 0;
            while (t < q && rc[t] != j) ++t;
            if (t == q) rc[q++] = j;
        }
        std::sort(rc, rc + q);
        cnt[k] = q;
    }
    P.ptr.assign(n + 1, 0);
    for (int k = 0; k < n; ++k) P.ptr[k + 1] = P.ptr[k] + cnt[k];
    P.col.resize(P.ptr[n]);
#pragma omp parallel for schedule(static) num_threads(nthr(n)) if (n > kPar)
    for (int k = 0; k < n; ++k) {
        const int* rc = &tc[(size_t)W * k];
        std::copy(rc, rc + cnt[k], P.col.begin() + P.ptr[k]);
        for (int e = P.fptr[k]; e < P.fptr[k + 1]; ++e)
            if (P.slot[e] >= 0) P.slot[e] = P.ptr[k] + (int)(std::find(rc, rc + cnt[k], P.slot[e]) - rc);
    }
    return P;
}

struct System {
    const Pattern* pat = nullptr;
    Csr A;
    std::vector<double> b;      // integrated rhs (incl. Dirichlet contributions)
    std::vector<double> invV;   // 1 / V per row (norm weights)
    bool singular = false;
};

void assemble(const Mesh& m, const Problem& p, System& S) {
    const Pattern& P = pattern(m);
    S.pat = &P;
    const int n = m.nLeaves();
    Csr& A = S.A;
    A.n = n;
    A.ptr = P.ptr;
    A.col = P.col;
    A.val.assign(P.col.size(), 0.0);
    A.diag.resize(n);
    S.b.resize(n);
    S.invV.resize(n);
    const bool hasLam = !p.lambda.empty();
    const bool hasDirArr = !p.dirichlet.empty();
    const bool hasBv = !p.bvalue.empty();
    int nDir = 0, nLam = 0;
#pragma omp parallel for schedule(static) reduction(+ : nDir, nLam) num_threads(nthr(n)) if (n > kPar)
    for (int k = 0; k < n; ++k) {
        const int c = P.order[k];
        const double V = P.vol[k];
        S.invV[k] = 1.0 / V;
        double d = 0.0;
        if (hasLam && p.lambda[c] != 0.0) { d = p.lambda[c] * V; ++nLam; }
        double b = p.rhs[c] * V;
        for (int e = P.fptr[k]; e < P.fptr[k + 1]; ++e) {
            const int f = P.face[e], sl = P.slot[e];
            const double a = p.beta[f] * P.area[e] / P.dist[e];
            if (sl == -1) {
                if (hasDirArr && p.dirichlet[f]) {
                    ++nDir;
                    d += a;
                    if (hasBv) b += a * p.bvalue[f];
                }
                continue;
            }
            if (sl == -2) continue;
            d += a;
            A.val[sl] -= a;
        }
        A.diag[k] = d;
        S.b[k] = b;
    }
    S.singular = (nDir == 0 && nLam == 0);
}

}  // namespace

Result solve(const Mesh& m, const Problem& p, std::vector<double>& x, const Options& opt) {
    const double t0 = now();
    const int n = m.nLeaves();
    if ((int)x.size() != m.nCells()) x.resize(m.nCells(), 0.0);

    System S;
    assemble(m, p, S);
    const std::vector<int>& leaves = S.pat->order;  // row -> cell
    const std::vector<double>& w = S.invV;
    std::vector<double>& B = S.b;

    double vol = 0.0;
    for (int k = 0; k < n; ++k) vol += 1.0 / w[k];
    if (S.singular) {  // compatibility: remove the volume-weighted mean of b
        double s = 0.0;
        for (int k = 0; k < n; ++k) s += B[k];
        const double mean = s / vol;
        for (int k = 0; k < n; ++k) B[k] -= mean / w[k];
    }

    auto wnorm = [&](const std::vector<double>& r) {
        double s = 0.0;
#pragma omp parallel for reduction(+ : s) schedule(static) num_threads(nthr(n)) if (n > kPar)
        for (int k = 0; k < n; ++k) s += r[k] * r[k] * w[k];
        return std::sqrt(s);
    };
    auto projectMean = [&](std::vector<double>& v) {  // Euclidean mean (range of A)
        double s = 0.0;
#pragma omp parallel for reduction(+ : s) schedule(static) num_threads(nthr(n)) if (n > kPar)
        for (int k = 0; k < n; ++k) s += v[k];
        s /= n;
#pragma omp parallel for schedule(static) num_threads(nthr(n)) if (n > kPar)
        for (int k = 0; k < n; ++k) v[k] -= s;
    };

    std::vector<double> xs(n), r, z, d, q;  // CPU work vectors: sized below if used
    for (int k = 0; k < n; ++k) xs[k] = x[leaves[k]];

    // ---- preconditioner
    Amg amg;
    std::vector<double> jac;
    const bool useAmg = opt.precond != 0;
    bool onGpu = false;
#ifdef VOF2D_CUDA
    onGpu = opt.gpu && n >= kGpuMin && gpu::available();
#endif
    if (useAmg)  // the hierarchy owns the finest matrix
        amg.setup(std::move(S.A), S.singular, onGpu ? kGpuAggressive : kMaxLevels);
    else
        jac = invDiag(S.A);
    const Csr& A = useAmg ? amg.matrix(0) : S.A;
    auto precondition = [&](const std::vector<double>& rr, std::vector<double>& zz) {
        if (useAmg) {
            amg.apply(rr.data(), zz.data());
        } else {
#pragma omp parallel for schedule(static) num_threads(nthr(n)) if (n > kPar)
            for (int k = 0; k < n; ++k) zz[k] = jac[k] * rr[k];
        }
        if (S.singular) projectMean(zz);
    };
    const double t1 = now();

    // ---- flexible PCG
    // In the singular case the residual's component along the null space (Euclidean mean)
    // is rounding noise that no x can reduce: it is projected out before measuring.
    const double bn = wnorm(B);
    auto done = [&](double rr) { return rr <= opt.tol * bn || rr <= opt.absTol; };
    auto trueResidual = [&]() {
        residual(A, B.data(), xs.data(), r.data());
        if (S.singular) projectMean(r);
        return wnorm(r);
    };
    Result res;
    int it = 0;
    bool conv = false;
    double rn = 0.0;
#ifdef VOF2D_CUDA
    if (onGpu) {
        const int nl = useAmg ? amg.nLevels() : 1;
        std::vector<gpu::Level> gl(nl);
        for (int l = 0; l < nl; ++l) {
            const Csr& Al = useAmg ? amg.matrix(l) : A;
            gpu::Level& g = gl[l];
            g.n = Al.n;
            g.ptr = Al.ptr.data();
            g.col = Al.col.data();
            g.val = Al.val.data();
            g.diag = Al.diag.data();
            if (l + 1 < nl) {
                const Level& L = amg.level(l);
                g.nNext = amg.matrix(l + 1).n;
                g.agg = L.agg.data();
                g.mptr = L.mptr.data();
                g.mem = L.mem.data();
            }
        }
        gpu::Problem gp;
        gp.levels = gl.data();
        gp.nLevels = nl;
        gp.amg = useAmg;
        gp.chol = useAmg ? amg.coarseFactor() : nullptr;
        gp.regularized = useAmg && amg.regularized();
        gp.singular = S.singular;
        gp.b = B.data();
        gp.w = w.data();
        gp.x = xs.data();
        gp.tol = opt.tol;
        gp.absTol = opt.absTol;
        gp.maxIter = opt.maxIter;
        const gpu::Outcome o = gpu::solve(gp);
        onGpu = o.ran;
        it = o.iterations;
        rn = o.rnorm;
        conv = o.converged;
    }
#endif
    if (!onGpu)
        for (auto* v : {&r, &z, &d, &q}) v->resize(n);
    // FCG; restarted from the true residual (at most 3 times) if the recursively updated
    // residual met the tolerance but the true one does not.
    for (int pass = 0; !onGpu; ++pass) {
        rn = trueResidual();
        if (done(rn)) { conv = true; break; }
        if (pass == 3 || it >= opt.maxIter) break;
        bool rconv = false;
        precondition(r, z);
        d = z;
        while (it < opt.maxIter) {
            ++it;
            spmv(A, d.data(), q.data());
            const double rho = dot(d.data(), q.data(), n);
            if (!(rho > 0.0)) break;
            const double alpha = dot(d.data(), r.data(), n) / rho;
#pragma omp parallel for schedule(static) num_threads(nthr(n)) if (n > kPar)
            for (int k = 0; k < n; ++k) {
                xs[k] += alpha * d[k];
                r[k] -= alpha * q[k];
            }
            if (S.singular && it % 8 == 0) projectMean(r);
            rn = wnorm(r);
            if (opt.verbose) std::printf("  poisson it %3d  rel.res %.3e\n", it, bn > 0 ? rn / bn : rn);
            if (done(rn)) { rconv = true; break; }
            precondition(r, z);
            const double beta = dot(z.data(), q.data(), n) / rho;
#pragma omp parallel for schedule(static) num_threads(nthr(n)) if (n > kPar)
            for (int k = 0; k < n; ++k) d[k] = z[k] - beta * d[k];
        }
        if (!rconv) {  // maxIter reached or breakdown
            rn = trueResidual();
            conv = done(rn);
            break;
        }
    }

    if (S.singular) {  // zero volume-weighted mean
        double s = 0.0;
        for (int k = 0; k < n; ++k) s += xs[k] / w[k];
        s /= vol;
        for (int k = 0; k < n; ++k) xs[k] -= s;
    }
    for (int k = 0; k < n; ++k) x[leaves[k]] = xs[k];

    res.iterations = it;
    res.residual = bn > 0.0 ? rn / bn : rn;
    res.converged = conv;
    if (opt.verbose) {
        std::printf("poisson: n=%d %s%s levels=%d singular=%d setup %.3fs solve %.3fs it=%d res=%.3e %s\n", n,
                    useAmg ? "AMG" : "Jacobi", onGpu ? " (GPU)" : "", useAmg ? amg.nLevels() : 1, (int)S.singular, t1 - t0,
                    now() - t1, it, res.residual, res.converged ? "" : "NOT CONVERGED");
        if (useAmg) {
            size_t tot = 0;
            for (int l = 0; l < amg.nLevels(); ++l) {
                tot += amg.matrix(l).nnz();
                std::printf("   level %d: n=%d nnz=%zu\n", l, amg.matrix(l).n, amg.matrix(l).nnz());
            }
            std::printf("   operator complexity %.3f, coarsest %s\n", (double)tot / amg.matrix(0).nnz(),
                        amg.denseCoarse() ? "dense Cholesky" : "GS sweeps");
        }
    }
    return res;
}

std::string gpuDevice() {
#ifdef VOF2D_CUDA
    return gpu::deviceName();
#else
    return "";
#endif
}

}  // namespace poisson
