// poisson.hpp - SPD finite-volume elliptic solver on the leaves of the AMR mesh.
//
// Owner: linear solver module (src/poisson.cpp).
//
// Solves for x (one value per leaf):
//
//   sum_{faces f of c} beta_f * A_f / d_f * (x_c - x_nb(f))  +  lambda_c * V_c * x_c  =  b_c * V_c
//
// i.e. the conservative discretization of  -div(beta grad x) + lambda x = b, with the
// two-point flux beta_f (x_R - x_L) / d_f on every face of Mesh::faces() (A_f = area,
// d_f = dist, V_c = volume). At coarse/fine faces this is the same flux seen from both
// sides, so the operator is symmetric positive (semi-)definite.
//
// Boundary faces: if dirichlet[f] != 0, x_nb = bvalue[f] (the value ON the boundary,
// at distance d_f = h/2 from the cell centre); otherwise the flux is zero (homogeneous
// Neumann).
//
// If there is no Dirichlet face and lambda == 0 everywhere, the system is singular:
// the solver removes the volume-weighted mean of b (compatibility) and returns the
// solution with zero volume-weighted mean.
#pragma once
#include <string>
#include <vector>
#include "mesh.hpp"

namespace poisson {

struct Problem {
    std::vector<double> beta;       // per face (size nFaces), > 0
    std::vector<double> lambda;     // per cell index (size nCells) >= 0, or empty (= 0)
    std::vector<double> rhs;        // b, per cell index (size nCells); only leaves used
    std::vector<char> dirichlet;    // per face (size nFaces), or empty (= all Neumann)
    std::vector<double> bvalue;     // per face Dirichlet value, or empty (= 0)
};

struct Options {
    double tol = 1e-9;     // stop when ||r||_2 <= tol * ||b||_2 (volume-weighted norms)
    double absTol = 1e-14; // or ||r||_2 <= absTol
    int maxIter = 1000;
    bool verbose = false;
    // Preconditioner (added by the linear-solver module):
    //   1 = aggregation AMG, K-cycle (default);  0 = Jacobi (simple fallback).
    int precond = 1;
    // Run the iteration on the CUDA device (CUDA builds only; ignored otherwise) if the
    // system has at least gpuMin unknowns. Smaller systems are faster on the CPU (measured
    // crossover ~1.5e5 unknowns in full time steps: RTX 4060 Ti vs 8 Zen 4 cores).
    bool gpu = false;
    int gpuMin = 120000;
};

struct Result {
    int iterations = 0;
    double residual = 0.0;  // final relative residual
    bool converged = false;
};

// x: per cell index (size nCells); on input the initial guess (leaf entries), on output
// the solution. Non-leaf entries are left untouched. Not reentrant (the matrix pattern of
// the last mesh and the GPU buffers are kept between calls).
Result solve(const Mesh& m, const Problem& prob, std::vector<double>& x, const Options& opt);

// Wall time spent in solve() since the start of the process, split into assembly +
// preconditioner setup and the Krylov iteration.
struct Timing {
    int solves = 0;
    double setup = 0.0, iterate = 0.0;  // seconds
};
Timing timing();

// Name of the CUDA device used for Options::gpu, or "" if there is none (CPU-only build or
// no usable device; the solver then runs on the CPU).
std::string gpuDevice();

}  // namespace poisson
