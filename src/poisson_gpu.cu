// poisson_gpu.cu - CUDA backend of the elliptic solver (see poisson_gpu.hpp).
//
// Execution model
//   * The whole solve is ONE cooperative kernel: persistent threads, grid-wide barriers
//     (cooperative_groups grid sync, ~1 us) between dependent phases. A K-cycle has
//     hundreds of small phases per iteration; with one kernel launch or host round trip
//     per phase (tens of microseconds each, more under WSL) latency would dominate.
//   * Large levels are processed by the whole grid ("GridTeam"); levels with at most
//     kBlockRows rows run on block 0 alone ("BlockTeam", __syncthreads barriers) while
//     the other blocks wait at the next grid barrier.
//   * Every thread follows the same control flow: all scalars (dot products, K-cycle
//     coefficients, convergence tests) come from deterministic reductions (fixed-order
//     warp/block trees, per-block partials summed in block order), so every block sees
//     bit-identical values.
//   * Smoother: multicolour Gauss-Seidel. Rows of one colour are not coupled, so each
//     colour is one parallel phase; the forward sweep runs the colours ascending, the
//     backward sweep descending, which keeps the preconditioner symmetric. The colouring
//     is greedy in row (Z-)order, and the unknowns of every level are renumbered colour
//     by colour (Z-order kept inside a colour) so that each phase streams a contiguous
//     range of rows.
//   * Coarsest level: dense inverse, computed in the kernel from the host's Cholesky
//     factor and applied as a matrix-vector product; Gauss-Seidel sweeps when the CPU
//     solver has no factor either (large coarsest level). A large product requested by
//     block 0 in single-block mode is handed to the whole grid (fork-join through gTask:
//     the other blocks wait in a service loop instead of a plain barrier).
//   * Level table and solve parameters live in constant memory (cached, and never
//     aliased by the stores to the work vectors). Device and pinned host buffers are
//     grow-only and reused between solves; the host data of one solve (permuted
//     matrices, rhs, guess) go up in a single copy.
#include "poisson_gpu.hpp"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cg = cooperative_groups;

namespace poisson {
namespace gpu {
namespace {

constexpr int kThreads = 512;     // threads per block (one block per SM: cheap grid barriers)
constexpr int kBlockRows = 1024;  // levels up to this size run on a single block
constexpr int kForkDense = 128;   // larger dense coarse solves are spread over the grid
constexpr int kMaxColors = 32;    // more colours: the hierarchy is solved on the CPU
constexpr double kKTol = 0.25;    // K-cycle: skip the 2nd inner iteration below this ratio
constexpr int kRed = 5;           // max number of values reduced at once

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(e));
}
#define CUDA_CHECK(call) check((call), #call)

// ------------------------------------------------------------------ device data

struct DLevel {
    int n, nNext, ncolor, block;
    int cptr[kMaxColors + 1];        // colour k = rows [cptr[k], cptr[k+1])
    const int *ptr, *col;            // off-diagonal CSR
    const double *val, *diag, *invd;
    const int *agg, *mptr, *mem;     // map to the next level
    double *rc, *v1, *w1, *v2, *tc;  // coarse-correction vectors (size nNext)
};

struct DCtx {
    int nl, n, amg, singular, regularized, maxIter, nCoarse;
    const double *b, *w;
    const double* chol;   // Cholesky factor of the coarsest matrix (row-major), or null
    double* ainv;         // its inverse, computed by the kernel
    double *x, *r, *z, *d, *q;
    double tol, absTol;
    double* part;         // per-block partial sums, 2 x kRed x gridDim
    double* out;          // iterations, rnorm, bnorm, converged
};

__constant__ DCtx gC;
__constant__ DLevel gL[kMaxLevels];

// Work posted by block 0 (single-block mode) for the whole grid.
struct DTask {
    int op;  // 0: none (leave the service loop), 1: dense coarse solve
    const double* r;
    double* z;
    double mean;
};
__device__ DTask gTask;

struct GridTeam {
    static constexpr bool kBlock = false;
    __device__ int rank() const { return blockIdx.x * blockDim.x + threadIdx.x; }
    __device__ int size() const { return gridDim.x * blockDim.x; }
    __device__ void sync() const { cg::this_grid().sync(); }
};

struct BlockTeam {
    static constexpr bool kBlock = true;
    __device__ int rank() const { return threadIdx.x; }
    __device__ int size() const { return blockDim.x; }
    __device__ void sync() const { __syncthreads(); }
};

__device__ inline double warpSum(double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ double* redScratch() {
    __shared__ double s[kRed * 32 + kRed];
    return s;
}

// Sum over the block; every thread gets the result.
template <int K>
__device__ void blockSum(double (&v)[K]) {
    double* s = redScratch();
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int k = 0; k < K; ++k) v[k] = warpSum(v[k]);
    __syncthreads();  // readers of the previous result are done
    if (lane == 0)
        for (int k = 0; k < K; ++k) s[k * 32 + wid] = v[k];
    __syncthreads();
    if (wid == 0)
        for (int k = 0; k < K; ++k) {
            const double t = warpSum(lane < nw ? s[k * 32 + lane] : 0.0);
            if (lane == 0) s[kRed * 32 + k] = t;
        }
    __syncthreads();
    for (int k = 0; k < K; ++k) v[k] = s[kRed * 32 + k];
}

// Team-wide sums; also a barrier (data written before are visible after). `parity`
// alternates the halves of gC.part: the other half may still be read by slow blocks.
template <int K>
__device__ void teamSum(const GridTeam& t, int& parity, double (&v)[K]) {
    blockSum(v);
    double* P = gC.part + parity * kRed * gridDim.x;
    parity ^= 1;
    if (threadIdx.x == 0)
        for (int k = 0; k < K; ++k) P[k * gridDim.x + blockIdx.x] = v[k];
    t.sync();
    for (int k = 0; k < K; ++k) {
        double s = 0.0;
        for (int b = threadIdx.x; b < gridDim.x; b += blockDim.x) s += P[k * gridDim.x + b];
        v[k] = s;
    }
    blockSum(v);
}

template <int K>
__device__ void teamSum(const BlockTeam&, int&, double (&v)[K]) {
    blockSum(v);
}

// sum_j a_ij x_j over the off-diagonal entries of row i
__device__ inline double rowOff(const DLevel& L, int i, const double* x) {
    double s = 0.0;
    const int e = __ldg(L.ptr + i + 1);
    for (int p = __ldg(L.ptr + i); p < e; ++p) s += __ldg(L.val + p) * x[__ldg(L.col + p)];
    return s;
}

// In-place multicolour Gauss-Seidel sweep for A z = r (fromZero: z = 0 on entry).
template <class T>
__device__ void gsSweep(const T& t, int l, const double* r, double* z, bool forward, bool fromZero) {
    const DLevel& L = gL[l];
    for (int s = 0; s < L.ncolor; ++s) {
        const int k = forward ? s : L.ncolor - 1 - s;
        const int lo = L.cptr[k], hi = L.cptr[k + 1];
        if (fromZero && s == 0) {
            // the first colour sees only zeros; every other row starts at 0
            for (int i = t.rank(); i < L.n; i += t.size())
                z[i] = i >= lo && i < hi ? r[i] * __ldg(L.invd + i) : 0.0;
        } else {
            for (int i = lo + t.rank(); i < hi; i += t.size()) {
                const double id = __ldg(L.invd + i);
                if (id != 0.0) z[i] = (r[i] - rowOff(L, i, z)) * id;
            }
        }
        t.sync();
    }
}

// rc[I] = sum over the members i of aggregate I of (r - A z)_i
template <class T>
__device__ void restrictResidual(const T& t, int l, const double* r, const double* z) {
    const DLevel& L = gL[l];
    for (int I = t.rank(); I < L.nNext; I += t.size()) {
        double s = 0.0;
        const int e = __ldg(L.mptr + I + 1);
        for (int q = __ldg(L.mptr + I); q < e; ++q) {
            const int i = __ldg(L.mem + q);
            s += r[i] - __ldg(L.diag + i) * z[i] - rowOff(L, i, z);
        }
        L.rc[I] = s;
    }
    t.sync();
}

// z = A^{-1} (r - mean), one warp per row
template <class T>
__device__ void denseApply(const T& t, const double* r, double* z, double mean) {
    const int n = gC.nCoarse;
    const int lane = t.rank() & 31;
    for (int i = t.rank() >> 5; i < n; i += t.size() >> 5) {
        const double* Ai = gC.ainv + (size_t)i * n;
        double s = 0.0;
        for (int j = lane; j < n; j += 32) s += Ai[j] * (r[j] - mean);
        s = warpSum(s);
        if (lane == 0) z[i] = s;
    }
}

template <class T>
__device__ void coarseSolve(const T& t, int& parity, const double* r, double* z) {
    const int l = gC.nl - 1;
    const int n = gL[l].n;
    if (!gC.ainv) {  // large coarsest level: a few symmetric sweeps (as on the CPU)
        gsSweep(t, l, r, z, true, true);
        for (int s = 0; s < 6; ++s) gsSweep(t, l, r, z, false, false);
        return;
    }
    double mean[1] = {0.0};
    if (gC.regularized) {  // (A + s 1 1^T) z = r - mean(r)
        for (int i = t.rank(); i < n; i += t.size()) mean[0] += r[i];
        teamSum(t, parity, mean);
        mean[0] /= n;
    }
    if (T::kBlock && n > kForkDense) {  // fork: the grid's service loop (subCycle) helps
        if (threadIdx.x == 0) gTask = DTask{1, r, z, mean[0]};
        const GridTeam g;
        g.sync();
        denseApply(g, r, z, mean[0]);
        g.sync();
        return;
    }
    denseApply(t, r, z, mean[0]);
    t.sync();
}

// gC.ainv <- (L L^T)^{-1} from the lower Cholesky factor gC.chol: one warp per column
// (forward then backward substitution; the inverse is symmetric, so column j is
// stored as row j).
__device__ void invertCoarse(const GridTeam& t) {
    const int n = gC.nCoarse;
    const double* F = gC.chol;
    const int lane = threadIdx.x & 31;
    for (int j = t.rank() >> 5; j < n; j += t.size() >> 5) {
        double* y = gC.ainv + (size_t)j * n;
        for (int i = lane; i < j; i += 32) y[i] = 0.0;
        for (int i = j; i < n; ++i) {  // L y = e_j
            const double* Fi = F + (size_t)i * n;
            double s = 0.0;
            for (int k = j + lane; k < i; k += 32) s += Fi[k] * y[k];
            s = warpSum(s);
            if (lane == 0) y[i] = ((i == j ? 1.0 : 0.0) - s) / Fi[i];
            __syncwarp();
        }
        for (int i = n - 1; i >= 0; --i) {  // L^T x = y, in place
            double s = 0.0;
            for (int k = i + 1 + lane; k < n; k += 32) s += F[(size_t)k * n + i] * y[k];
            s = warpSum(s);
            if (lane == 0) y[i] = (y[i] - s) / F[(size_t)i * n + i];
            __syncwarp();
        }
    }
    t.sync();
}

template <class T>
__device__ void cycle(const T& t, int& parity, int l, const double* r, double* z);

// Cycle on level l called from the grid: small levels run on block 0 only, while the
// other blocks serve the tasks it posts (see coarseSolve).
__device__ void subCycle(const GridTeam& t, int& parity, int l, const double* r, double* z) {
    if (!gL[l].block) {
        cycle(t, parity, l, r, z);
        return;
    }
    if (blockIdx.x == 0) {
        cycle(BlockTeam(), parity, l, r, z);
        if (threadIdx.x == 0) gTask.op = 0;
        t.sync();
        return;
    }
    for (;;) {
        t.sync();
        const DTask task = gTask;
        if (task.op == 0) break;
        denseApply(t, task.r, task.z, task.mean);
        t.sync();
    }
}

__device__ void subCycle(const BlockTeam& t, int& parity, int l, const double* r, double* z) {
    cycle(t, parity, l, r, z);
}

// K-cycle (see Amg::cycle in poisson.cpp): z = B_l r.
template <class T>
__device__ void cycle(const T& t, int& parity, int l, const double* r, double* z) {
    if (l + 1 == gC.nl) {
        coarseSolve(t, parity, r, z);
        return;
    }
    const DLevel& L = gL[l];
    const DLevel& Lc = gL[l + 1];
    gsSweep(t, l, r, z, true, true);
    restrictResidual(t, l, r, z);
    // coarse correction ec = c1 v1 + c2 v2
    double c1 = 1.0, c2 = 0.0;
    subCycle(t, parity, l + 1, L.rc, L.v1);
    if (l + 2 < gC.nl) {
        // K-cycle: two FCG iterations on the next level
        double a[5] = {0.0, 0.0, 0.0, 0.0, 0.0};  // v1.w1, v1.rc, rc.rc, rc.w1, w1.w1
        for (int I = t.rank(); I < Lc.n; I += t.size()) {
            const double v = L.v1[I], rr = L.rc[I];
            const double w = __ldg(Lc.diag + I) * v + rowOff(Lc, I, L.v1);
            L.w1[I] = w;
            a[0] += v * w;
            a[1] += v * rr;
            a[2] += rr * rr;
            a[3] += rr * w;
            a[4] += w * w;
        }
        teamSum(t, parity, a);
        const double rho1 = a[0];
        if (!(rho1 > 0.0)) {
            c1 = 0.0;
        } else {
            const double al = a[1] / rho1;
            const double tn = a[2] - 2.0 * al * a[3] + al * al * a[4];  // |rc - al w1|^2
            c1 = al;
            if (tn > kKTol * kKTol * a[2]) {
                for (int I = t.rank(); I < Lc.n; I += t.size()) L.tc[I] = L.rc[I] - al * L.w1[I];
                t.sync();
                subCycle(t, parity, l + 1, L.tc, L.v2);
                double g[3] = {0.0, 0.0, 0.0};  // v2.w1, v2.(A v2), v2.tc
                for (int I = t.rank(); I < Lc.n; I += t.size()) {
                    const double v = L.v2[I];
                    g[0] += v * L.w1[I];
                    g[1] += v * (__ldg(Lc.diag + I) * v + rowOff(Lc, I, L.v2));
                    g[2] += v * L.tc[I];
                }
                teamSum(t, parity, g);
                const double rho2 = g[1] - g[0] * g[0] / rho1;
                c1 = rho2 > 0.0 ? al - g[0] * g[2] / (rho1 * rho2) : al;
                c2 = rho2 > 0.0 ? g[2] / rho2 : 0.0;
            }
        }
    }
    for (int i = t.rank(); i < L.n; i += t.size()) {
        const int I = __ldg(L.agg + i);
        if (I >= 0) z[i] += c1 * L.v1[I] + (c2 != 0.0 ? c2 * L.v2[I] : 0.0);
    }
    t.sync();
    gsSweep(t, l, r, z, false, false);
}

// Flexible CG (FCG(1)) with the K-cycle or Jacobi preconditioner; mirrors poisson::solve.
__global__ void __launch_bounds__(kThreads) solveKernel() {
    const GridTeam t;
    const DLevel& A = gL[0];
    const int n = gC.n, i0 = t.rank(), di = t.size();
    double* const x = gC.x;
    double* const r = gC.r;
    double* const z = gC.z;
    double* const d = gC.d;
    double* const q = gC.q;
    const double* const b = gC.b;
    const double* const w = gC.w;
    int parity = 0;

    if (gC.ainv) invertCoarse(t);

    auto projectMean = [&](double* v) {  // Euclidean mean (range of A)
        double s[1] = {0.0};
        for (int i = i0; i < n; i += di) s[0] += v[i];
        teamSum(t, parity, s);
        const double m = s[0] / n;
        for (int i = i0; i < n; i += di) v[i] -= m;
        t.sync();
    };
    auto trueResidual = [&]() {
        double s[2] = {0.0, 0.0};  // sum r^2 w, sum r
        for (int i = i0; i < n; i += di) {
            const double ri = b[i] - __ldg(A.diag + i) * x[i] - rowOff(A, i, x);
            r[i] = ri;
            s[0] += ri * ri * w[i];
            s[1] += ri;
        }
        teamSum(t, parity, s);
        if (!gC.singular) return sqrt(s[0]);
        const double m = s[1] / n;
        double u[1] = {0.0};
        for (int i = i0; i < n; i += di) {
            const double ri = r[i] - m;
            r[i] = ri;
            u[0] += ri * ri * w[i];
        }
        teamSum(t, parity, u);
        return sqrt(u[0]);
    };
    auto precondition = [&]() {
        if (gC.amg) {
            cycle(t, parity, 0, r, z);
        } else {
            for (int i = i0; i < n; i += di) z[i] = __ldg(A.invd + i) * r[i];
            t.sync();
        }
        if (gC.singular) projectMean(z);
    };

    double bn[1] = {0.0};
    for (int i = i0; i < n; i += di) bn[0] += b[i] * b[i] * w[i];
    teamSum(t, parity, bn);
    bn[0] = sqrt(bn[0]);
    auto done = [&](double rr) { return rr <= gC.tol * bn[0] || rr <= gC.absTol; };

    int it = 0;
    bool conv = false;
    double rn = 0.0;
    for (int pass = 0;; ++pass) {
        rn = trueResidual();
        if (done(rn)) { conv = true; break; }
        if (pass == 3 || it >= gC.maxIter) break;
        bool rconv = false;
        precondition();
        for (int i = i0; i < n; i += di) d[i] = z[i];
        t.sync();
        while (it < gC.maxIter) {
            ++it;
            double s[2] = {0.0, 0.0};  // d.q, d.r
            for (int i = i0; i < n; i += di) {
                const double dd = d[i];
                const double qi = __ldg(A.diag + i) * dd + rowOff(A, i, d);
                q[i] = qi;
                s[0] += dd * qi;
                s[1] += dd * r[i];
            }
            teamSum(t, parity, s);
            const double rho = s[0];
            if (!(rho > 0.0)) break;
            const double alpha = s[1] / rho;
            double u[2] = {0.0, 0.0};  // sum r^2 w, sum r
            for (int i = i0; i < n; i += di) {
                x[i] += alpha * d[i];
                const double ri = r[i] - alpha * q[i];
                r[i] = ri;
                u[0] += ri * ri * w[i];
                u[1] += ri;
            }
            teamSum(t, parity, u);
            if (gC.singular && it % 8 == 0) {
                const double m = u[1] / n;
                double v[1] = {0.0};
                for (int i = i0; i < n; i += di) {
                    const double ri = r[i] - m;
                    r[i] = ri;
                    v[0] += ri * ri * w[i];
                }
                teamSum(t, parity, v);
                rn = sqrt(v[0]);
            } else {
                rn = sqrt(u[0]);
            }
            if (done(rn)) { rconv = true; break; }
            precondition();
            double g[1] = {0.0};
            for (int i = i0; i < n; i += di) g[0] += z[i] * q[i];
            teamSum(t, parity, g);
            const double beta = g[0] / rho;
            for (int i = i0; i < n; i += di) d[i] = z[i] - beta * d[i];
            t.sync();
        }
        if (!rconv) {  // maxIter reached or breakdown
            rn = trueResidual();
            conv = done(rn);
            break;
        }
    }
    if (i0 == 0) {
        gC.out[0] = it;
        gC.out[1] = rn;
        gC.out[2] = bn[0];
        gC.out[3] = conv ? 1.0 : 0.0;
    }
}

// ------------------------------------------------------------------ host side

struct Runtime {
    bool initialized = false, ok = false;
    std::string name;
    int grid = 0;
    cudaStream_t stream = nullptr;
    char* dev = nullptr;   // device buffer
    size_t devCap = 0;
    char* host = nullptr;  // pinned staging buffer
    size_t hostCap = 0;
};

Runtime& runtime() {
    static Runtime r;
    return r;
}

void init(Runtime& R) {
    R.initialized = true;
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd < 1) {
        cudaGetLastError();  // clear the sticky "no device" error
        return;
    }
    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess ||
        !prop.cooperativeLaunch)
        return;
    // device stack for the recursive K-cycle (< 512 bytes of frames per level)
    if (cudaDeviceSetLimit(cudaLimitStackSize, 1024 + 512 * kMaxLevels) != cudaSuccess) return;
    int perSM = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, solveKernel, kThreads, 0) != cudaSuccess || perSM < 1)
        return;
    if (cudaStreamCreateWithFlags(&R.stream, cudaStreamNonBlocking) != cudaSuccess) return;
    R.grid = perSM * prop.multiProcessorCount;
    R.name = prop.name;
    R.ok = true;
}

template <class T>
void grow(T*& p, size_t& cap, size_t need, bool pinned) {
    if (need <= cap) return;
    const size_t c = std::max(need, cap + cap / 2);
    if (p) CUDA_CHECK(pinned ? cudaFreeHost(p) : cudaFree(p));
    p = nullptr;
    cap = 0;
    void* q = nullptr;
    CUDA_CHECK(pinned ? cudaMallocHost(&q, c) : cudaMalloc(&q, c));
    p = static_cast<T*>(q);
    cap = c;
}

// Byte layout of the device buffer: [uploaded data | work arrays].
class Layout {
public:
    template <class T>
    size_t add(size_t count) {
        size_ = (size_ + 255) & ~size_t(255);
        const size_t off = size_;
        size_ += count * sizeof(T);
        return off;
    }
    size_t size() const { return size_; }

private:
    size_t size_ = 0;
};

// Greedy colouring in row order. perm: new -> old row numbering (colour by colour,
// ascending inside a colour), cptr: colour -> new rows. Returns the number of colours,
// or 0 if more than kMaxColors would be needed.
int colour(const Level& L, std::vector<int>& perm, int cptr[kMaxColors + 1]) {
    const int n = L.n;
    std::vector<uint8_t> color(n, 0xff);
    int mark[kMaxColors + 1];
    std::fill(mark, mark + kMaxColors + 1, -1);
    int nc = 0;
    for (int i = 0; i < n; ++i) {
        for (int p = L.ptr[i]; p < L.ptr[i + 1]; ++p) {
            const uint8_t cj = color[L.col[p]];
            if (cj != 0xff) mark[cj] = i;
        }
        int k = 0;
        while (mark[k] == i) ++k;
        if (k >= kMaxColors) return 0;
        color[i] = (uint8_t)k;
        nc = std::max(nc, k + 1);
    }
    std::fill(cptr, cptr + kMaxColors + 1, 0);
    for (int i = 0; i < n; ++i) cptr[color[i] + 1]++;
    for (int k = 0; k < nc; ++k) cptr[k + 1] += cptr[k];
    for (int k = nc + 1; k <= kMaxColors; ++k) cptr[k] = n;
    perm.resize(n);
    int pos[kMaxColors];
    std::copy(cptr, cptr + nc, pos);
    for (int i = 0; i < n; ++i) perm[pos[color[i]]++] = i;
    return nc;
}

}  // namespace

bool available() {
    Runtime& R = runtime();
    if (!R.initialized) init(R);
    return R.ok;
}

const char* deviceName() {
    return available() ? runtime().name.c_str() : "";
}

Outcome solve(const Problem& p) {
    Outcome res;
    if (!available() || p.nLevels < 1 || p.nLevels > kMaxLevels) return res;
    Runtime& R = runtime();
    const int nl = p.amg ? p.nLevels : 1, n = p.levels[0].n;
    const Level* lv = p.levels;

    // ---- colourings -> renumbering of every level (identity for Jacobi, and for a
    //      coarsest level solved with the dense factor, which is in the host numbering)
    const int nCoarse = lv[nl - 1].n;
    const bool dense = p.amg && p.chol != nullptr;
    std::vector<DLevel> dl(nl);
    std::vector<std::vector<int>> perm(nl), inv(nl);
    for (int l = 0; l < nl; ++l) {
        std::memset(&dl[l], 0, sizeof(DLevel));
        if (p.amg && !(dense && l == nl - 1)) {
            dl[l].ncolor = colour(lv[l], perm[l], dl[l].cptr);
            if (dl[l].ncolor == 0) return res;  // not supported: solve on the CPU
        } else {
            perm[l].resize(lv[l].n);
            for (int i = 0; i < lv[l].n; ++i) perm[l][i] = i;
            dl[l].ncolor = 1;  // unused
            dl[l].cptr[1] = lv[l].n;
        }
        inv[l].resize(lv[l].n);
        for (int i = 0; i < lv[l].n; ++i) inv[l][perm[l][i]] = i;
    }

    // ---- layout: uploaded part first, then work arrays
    struct Off {
        size_t ptr, col, val, diag, invd, agg, mptr, mem, rc, v1, w1, v2, tc;
    };
    std::vector<Off> o(nl);
    Layout lay;
    for (int l = 0; l < nl; ++l) {
        const Level& L = lv[l];
        const size_t nnz = L.ptr[L.n];
        o[l].ptr = lay.add<int>(L.n + 1);
        o[l].col = lay.add<int>(nnz);
        o[l].val = lay.add<double>(nnz);
        o[l].diag = lay.add<double>(L.n);
        o[l].invd = lay.add<double>(L.n);
        if (l + 1 < nl) {
            o[l].agg = lay.add<int>(L.n);
            o[l].mptr = lay.add<int>(L.nNext + 1);
            o[l].mem = lay.add<int>(L.mptr[L.nNext]);
        }
    }
    const size_t oB = lay.add<double>(n), oW = lay.add<double>(n), oX = lay.add<double>(n);
    const size_t oChol = dense ? lay.add<double>((size_t)nCoarse * nCoarse) : 0;
    const size_t upBytes = lay.size();
    const size_t oAinv = dense ? lay.add<double>((size_t)nCoarse * nCoarse) : 0;
    const size_t oR = lay.add<double>(n), oZ = lay.add<double>(n), oD = lay.add<double>(n), oQ = lay.add<double>(n);
    for (int l = 0; l + 1 < nl; ++l) {
        const int m = lv[l].nNext;
        o[l].rc = lay.add<double>(m);
        o[l].v1 = lay.add<double>(m);
        o[l].w1 = lay.add<double>(m);
        o[l].v2 = lay.add<double>(m);
        o[l].tc = lay.add<double>(m);
    }
    const size_t oPart = lay.add<double>(2 * kRed * (size_t)R.grid);
    const size_t oOut = lay.add<double>(4);
    const size_t total = lay.size();

    grow(R.dev, R.devCap, total, false);
    grow(R.host, R.hostCap, std::max(upBytes, n * sizeof(double) + 64) + sizeof(DCtx) + 64 + 4 * sizeof(double),
         true);
    char* const D = R.dev;
    char* const H = R.host;
    auto hp = [&](size_t off) { return H + off; };
    auto dp = [&](size_t off) { return D + off; };

    // ---- stage the renumbered hierarchy
    for (int l = 0; l < nl; ++l) {
        const Level& L = lv[l];
        const std::vector<int>& P = perm[l];
        const std::vector<int>& Q = inv[l];
        int* ptr = reinterpret_cast<int*>(hp(o[l].ptr));
        int* col = reinterpret_cast<int*>(hp(o[l].col));
        double* val = reinterpret_cast<double*>(hp(o[l].val));
        double* diag = reinterpret_cast<double*>(hp(o[l].diag));
        double* invd = reinterpret_cast<double*>(hp(o[l].invd));
        ptr[0] = 0;
        for (int i = 0; i < L.n; ++i) ptr[i + 1] = ptr[i] + (L.ptr[P[i] + 1] - L.ptr[P[i]]);
#pragma omp parallel for schedule(static) if (L.n > 16384)
        for (int i = 0; i < L.n; ++i) {
            const int io = P[i];
            int k = ptr[i];
            for (int q = L.ptr[io]; q < L.ptr[io + 1]; ++q, ++k) {
                col[k] = Q[L.col[q]];
                val[k] = L.val[q];
            }
            diag[i] = L.diag[io];
            invd[i] = L.diag[io] > 0.0 ? 1.0 / L.diag[io] : 0.0;
        }
        if (l + 1 < nl) {
            const std::vector<int>& Pn = perm[l + 1];
            const std::vector<int>& Qn = inv[l + 1];
            int* agg = reinterpret_cast<int*>(hp(o[l].agg));
            int* mptr = reinterpret_cast<int*>(hp(o[l].mptr));
            int* mem = reinterpret_cast<int*>(hp(o[l].mem));
#pragma omp parallel for schedule(static) if (L.n > 16384)
            for (int i = 0; i < L.n; ++i) {
                const int a = L.agg[P[i]];
                agg[i] = a >= 0 ? Qn[a] : -1;
            }
            mptr[0] = 0;
            for (int I = 0; I < L.nNext; ++I) {
                const int Io = Pn[I];
                int k = mptr[I];
                for (int q = L.mptr[Io]; q < L.mptr[Io + 1]; ++q) mem[k++] = Q[L.mem[q]];
                mptr[I + 1] = k;
            }
        }
        DLevel& d = dl[l];
        d.n = L.n;
        d.nNext = l + 1 < nl ? L.nNext : 0;
        d.block = L.n <= kBlockRows;
        d.ptr = reinterpret_cast<const int*>(dp(o[l].ptr));
        d.col = reinterpret_cast<const int*>(dp(o[l].col));
        d.val = reinterpret_cast<const double*>(dp(o[l].val));
        d.diag = reinterpret_cast<const double*>(dp(o[l].diag));
        d.invd = reinterpret_cast<const double*>(dp(o[l].invd));
        if (l + 1 < nl) {
            d.agg = reinterpret_cast<const int*>(dp(o[l].agg));
            d.mptr = reinterpret_cast<const int*>(dp(o[l].mptr));
            d.mem = reinterpret_cast<const int*>(dp(o[l].mem));
            d.rc = reinterpret_cast<double*>(dp(o[l].rc));
            d.v1 = reinterpret_cast<double*>(dp(o[l].v1));
            d.w1 = reinterpret_cast<double*>(dp(o[l].w1));
            d.v2 = reinterpret_cast<double*>(dp(o[l].v2));
            d.tc = reinterpret_cast<double*>(dp(o[l].tc));
        }
    }
    {
        double* hb = reinterpret_cast<double*>(hp(oB));
        double* hw = reinterpret_cast<double*>(hp(oW));
        double* hx = reinterpret_cast<double*>(hp(oX));
        const std::vector<int>& P = perm[0];
        for (int i = 0; i < n; ++i) {
            hb[i] = p.b[P[i]];
            hw[i] = p.w[P[i]];
            hx[i] = p.x[P[i]];
        }
        if (dense) std::memcpy(hp(oChol), p.chol, (size_t)nCoarse * nCoarse * sizeof(double));
    }
    DCtx& c = *reinterpret_cast<DCtx*>(H + ((upBytes + 63) & ~size_t(63)));
    std::memset(&c, 0, sizeof c);
    c.nl = nl;
    c.n = n;
    c.amg = p.amg;
    c.singular = p.singular;
    c.regularized = p.regularized;
    c.maxIter = p.maxIter;
    c.nCoarse = nCoarse;
    c.b = reinterpret_cast<const double*>(dp(oB));
    c.w = reinterpret_cast<const double*>(dp(oW));
    c.chol = dense ? reinterpret_cast<const double*>(dp(oChol)) : nullptr;
    c.ainv = dense ? reinterpret_cast<double*>(dp(oAinv)) : nullptr;
    c.x = reinterpret_cast<double*>(dp(oX));
    c.r = reinterpret_cast<double*>(dp(oR));
    c.z = reinterpret_cast<double*>(dp(oZ));
    c.d = reinterpret_cast<double*>(dp(oD));
    c.q = reinterpret_cast<double*>(dp(oQ));
    c.tol = p.tol;
    c.absTol = p.absTol;
    c.part = reinterpret_cast<double*>(dp(oPart));
    c.out = reinterpret_cast<double*>(dp(oOut));

    // ---- run (the level table is small: pageable copy)
    CUDA_CHECK(cudaMemcpyAsync(D, H, upBytes, cudaMemcpyHostToDevice, R.stream));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(gC, &c, sizeof c, 0, cudaMemcpyHostToDevice, R.stream));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(gL, dl.data(), nl * sizeof(DLevel), 0, cudaMemcpyHostToDevice, R.stream));
    CUDA_CHECK(cudaLaunchCooperativeKernel((const void*)solveKernel, R.grid, kThreads, nullptr, 0, R.stream));
    double* hx = reinterpret_cast<double*>(H);
    double* hout = reinterpret_cast<double*>(H + ((n * sizeof(double) + 63) & ~size_t(63)));
    CUDA_CHECK(cudaMemcpyAsync(hx, dp(oX), n * sizeof(double), cudaMemcpyDeviceToHost, R.stream));
    CUDA_CHECK(cudaMemcpyAsync(hout, dp(oOut), 4 * sizeof(double), cudaMemcpyDeviceToHost, R.stream));
    CUDA_CHECK(cudaStreamSynchronize(R.stream));
    const std::vector<int>& P = perm[0];
    for (int i = 0; i < n; ++i) p.x[P[i]] = hx[i];

    res.ran = true;
    res.iterations = (int)hout[0];
    res.rnorm = hout[1];
    res.bnorm = hout[2];
    res.converged = hout[3] != 0.0;
    return res;
}

}  // namespace gpu
}  // namespace poisson
