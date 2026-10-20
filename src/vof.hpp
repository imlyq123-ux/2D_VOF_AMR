// vof.hpp - geometric VOF: PLIC reconstruction and conservative split advection.
//
// Owner: VOF module (src/plic.hpp, src/plic.cpp, src/vof.cpp).
//
// f is the volume fraction of phase 1, stored as a Mesh field (leaf values are the
// solution). Functions that read neighbour data call m.restrict(f) themselves when
// they need parent values (Mesh::locate may return a non-leaf cell).
#pragma once
#include <utility>
#include <vector>
#include "mesh.hpp"
#include "params.hpp"

namespace vof {

// Interface line in a cell, in cell-local coordinates X, Y in [0,1] (X = (x - xcell_min)/h):
//   phase 1 occupies { (X,Y) in [0,1]^2 : mx*X + my*Y <= alpha }.
// (mx, my) is therefore the outward normal of phase 1 (points from phase 1 to phase 2),
// normalized so that |mx| + |my| = 1.
struct Line {
    double mx = 0.0, my = 0.0, alpha = 0.0;
};

// Interface normal for leaf c (Mixed-Youngs-Centered on the 3x3 stencil at c's level),
// normalized |mx|+|my|=1. Requires restricted parent values of f (call m.restrict).
Line reconstruct(const Mesh& m, const std::vector<double>& f, int c);

// A quantity carried by one phase: data = (phase volume fraction) * phi. It is transported
// with exactly the geometric phase fluxes used for f (phase 2 flux = total volume flux -
// phase-1 flux) and the matching Weymouth-Yue compression term, using a minmod-limited
// linear reconstruction of the concentration phi in the donor cell. With phi = a velocity
// component this gives momentum-conserving advection consistent with the VOF transport.
struct Tracer {
    std::vector<double>* data = nullptr;              // per cell index (leaf values used)
    bool phase2 = false;                              // carried by phase 2 (fraction 1 - f)
    const std::vector<double>* fallback = nullptr;    // phi where the phase fraction vanishes
    double inflow[4] = {0.0, 0.0, 0.0, 0.0};          // phi entering through Inflow sides
};

// One time step of conservative, bounded, directionally split advection
// (Weymouth & Yue 2010) with face-normal velocities uf (Mesh::faces() order).
// The sweep order alternates with `step` parity (x-y, then y-x).
// Boundary faces: Inflow sides bring in prm.bc[side].f; other sides use the
// upwind interior cell. Leaf values of f are clipped to [0,1] (tiny values removed).
// Optional tracers are advected consistently (see Tracer). Their parent values are
// overwritten (restricted) like those of f.
// Requires |uf| dt / h <= 0.5 on every face.
void advect(Mesh& m, std::vector<double>& f, const std::vector<double>& uf, double dt, int step,
            const Params& prm, const std::vector<Tracer>& tracers = {});

// Geometric prolongation (Mesh::Prolongator): children fractions from the parent's PLIC
// line, so that refining an interfacial cell keeps the interface shape and the volume.
void prolong(const Mesh& m, const std::vector<double>& f, int parent, double out[4]);

// PLIC segments of all interfacial leaves (0 < f < 1), in physical coordinates.
std::vector<std::pair<Vec2, Vec2>> interfaceSegments(const Mesh& m, const std::vector<double>& f);

// Threshold below which f is considered pure (0 or 1).
constexpr double EPS = 1e-10;

}  // namespace vof
