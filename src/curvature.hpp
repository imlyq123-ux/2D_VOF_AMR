// curvature.hpp - interface curvature from the volume fraction field.
//
// Owner: curvature module (src/curvature.cpp).
#pragma once
#include <vector>
#include "mesh.hpp"

namespace curvature {

// Computes kappa (a Mesh-sized vector, only leaf entries are used/written) from f.
//
//  * Sign convention: kappa = -div(n), n = grad f / |grad f|, i.e. kappa = +1/R for a
//    disk of phase 1 (f = 1 inside) of radius R; -1/R for a bubble of phase 2.
//  * Interfacial leaves (vof::EPS < f < 1 - vof::EPS): generalized height-function
//    curvature (Popinet 2009) on the leaf's level, with fallbacks (mixed heights,
//    parabolic fit of interface positions) when heights are inconsistent.
//  * Every leaf that is not interfacial but has a face neighbour (Mesh::faces) with a
//    different f gets the average curvature of nearby interfacial leaves, so that
//    the surface-tension force sigma * kappa_f * grad_f f is defined on every face
//    where grad_f f != 0.
//  * All other leaves get 0. The result never contains NaN/Inf.
//  * The interface region is assumed to be at the maximum level with a buffer of at
//    least 3-4 cells (guaranteed by the AMR criterion); at walls neighbour data are
//    mirrored by Mesh::locate (90 degree contact angle).
//  * |kappa| is limited to 1/h (radius of curvature at least one cell). By default a
//    larger value is clamped to that limit. With rejectUnresolved, a method that gives
//    a larger value counts as failed instead (next fallback, else 0): see the comment
//    at KMAX in curvature.cpp for why (Params::rejectUnresolvedCurvature).
// Calls m.restrict on an internal copy of f when it needs parent values.
void compute(const Mesh& m, const std::vector<double>& f, std::vector<double>& kappa,
             bool rejectUnresolved = false);

// Counters of the last compute() call (cells per method / fallback), see curvature.cpp.
void lastStats(long out[8]);

}  // namespace curvature
