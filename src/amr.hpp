// amr.hpp - refinement criterion for the two-phase solver.
#pragma once
#include <vector>
#include "state.hpp"

namespace amr {

// Flags for Mesh::adapt (one per leaf ordinal):
//  * +1 for leaves within prm.interfaceBuffer max-level cells of an interfacial leaf
//    (0 < f < 1, or f differs across a face) that are not yet at maxLevel; such leaves
//    are never coarsened;
//  * elsewhere (if prm.adaptVelocity) a wavelet estimate of u and v - the difference
//    between the leaf value and the bilinear prediction from its parent - gives +1
//    above prm.uTol and -1 below prm.uTol / 1.5;
//  * otherwise -1 (coarsen as far as allowed).
std::vector<int> computeFlags(const State& s);

// Leaves that are interfacial in the sense above (marker per cell index).
std::vector<char> interfacialCells(const Mesh& m, const std::vector<double>& f);

}  // namespace amr
