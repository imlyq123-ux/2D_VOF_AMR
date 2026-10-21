// poisson_gpu.hpp - CUDA backend of the elliptic solver (used by poisson.cpp).
//
// Owner: linear solver module (src/poisson_gpu.cu). Only compiled with `make CUDA=1`
// (VOF2D_CUDA defined). poisson.cpp assembles the system and builds the AMG hierarchy on
// the host (with larger aggregates on the coarse levels, see poisson.cpp); this backend
// uploads it and runs the whole flexible-CG iteration (K-cycle preconditioner included)
// on the device. The smoother is multicolour Gauss-Seidel (greedy colouring of each
// level's graph) instead of the CPU's hybrid block Gauss-Seidel.
#pragma once

namespace poisson {
namespace gpu {

// Deepest AMG hierarchy the device solver accepts (bounds the stack of the recursive
// K-cycle; 16 levels is far beyond any practical mesh).
constexpr int kMaxLevels = 16;

// One level of the AMG hierarchy (host arrays, borrowed for the duration of solve()).
struct Level {
    int n = 0;
    const int* ptr = nullptr;      // off-diagonal CSR, size n + 1
    const int* col = nullptr;
    const double* val = nullptr;
    const double* diag = nullptr;  // size n
    // map to the next level (null on the coarsest level)
    int nNext = 0;
    const int* agg = nullptr;      // row -> next-level aggregate, -1: not in the coarse space
    const int* mptr = nullptr;     // aggregate -> rows (CSR), size nNext + 1
    const int* mem = nullptr;
};

struct Problem {
    const Level* levels = nullptr;  // levels[0] is the system matrix
    int nLevels = 0;
    bool amg = true;                // false: Jacobi preconditioner (levels[0] only)
    // coarsest-level direct solve: lower Cholesky factor (row-major, n x n), or null for
    // Gauss-Seidel sweeps; `regularized`: the factor is of A + s 1 1^T (singular case)
    const double* chol = nullptr;
    bool regularized = false;
    bool singular = false;          // pure Neumann: project out the constant
    const double* b = nullptr;      // integrated rhs (compatible if singular), size n
    const double* w = nullptr;      // norm weights 1 / V, size n
    double* x = nullptr;            // initial guess in, solution out, size n
    double tol = 1e-9, absTol = 1e-14;
    int maxIter = 1000;
};

struct Outcome {
    bool ran = false;     // false: no device, or hierarchy not supported -> use the CPU
    int iterations = 0;
    double rnorm = 0.0;   // final weighted residual norm
    double bnorm = 0.0;   // weighted norm of b
    bool converged = false;
};

// True if a CUDA device that supports cooperative launches is present (checked once).
bool available();

// Name of the device (for log messages); "" if none.
const char* deviceName();

// Same stopping rules and restarts as the CPU FCG in poisson.cpp. Returns ran = false
// (nothing done) if there is no device or the hierarchy is not supported (too many levels
// or colours); throws std::runtime_error on CUDA errors.
Outcome solve(const Problem& p);

}  // namespace gpu
}  // namespace poisson
