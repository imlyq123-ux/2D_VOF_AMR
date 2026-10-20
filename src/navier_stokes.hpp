// navier_stokes.hpp - incompressible two-phase Navier-Stokes on the AMR mesh.
//
// Owner: Navier-Stokes module (src/navier_stokes.cpp).
//
// Discretization (collocated cell-centred u, v, p + face-normal velocity uf, in the
// spirit of Basilisk's centered solver, balanced-force surface tension):
//
//   one time step, called by sim::step() in this order after the VOF advection:
//     updateProperties(s)            rho, mu from the new f
//     curvature::compute(...)        kappa from the new f
//     advanceMomentum(s, dt)         u* = u^n + dt [ -(uf.grad)u + (1/rho) div(2 mu D) ]
//                                     (advection uses the old, divergence-free uf;
//                                      viscous part implicit when prm.implicitViscosity)
//     project(s, dt)                 uf* = face(u*) + dt a_f,
//                                     a_f = g.n_f + sigma kappa_f (f_R - f_L)/d_f / rho_f
//                                     solve  div(1/rho_f grad p) = div(uf*)/dt
//                                     uf  = uf* - dt/rho_f grad_f p
//                                     u   = u* + dt * cellAverage(a_f - 1/rho_f grad_f p)
//
// Boundary conditions from prm.bc (see params.hpp). Pressure: Dirichlet p = 0 on Outflow
// sides, homogeneous Neumann elsewhere.
#pragma once
#include <vector>
#include "poisson.hpp"
#include "state.hpp"

namespace ns {

// rho = f rho1 + (1-f) rho2, mu = f mu1 + (1-f) mu2 on leaves (f clamped to [0,1]).
void updateProperties(State& s);

// Largest stable time step: advective CFL (prm.cfl, on uf and cell velocities),
// capillary constraint sqrt(rho_mean h^3 / (pi sigma)), explicit viscous limit when
// !prm.implicitViscosity, capped by prm.dtMax.
double stableDt(const State& s);

// Momentum predictor: u, v (leaf values) <- u* (see above). Uses s.uf as advecting
// velocity. Updates s.F(s.id.u), s.F(s.id.v).
void advanceMomentum(State& s, double dt);

// Projection step (see above). Updates s.uf, s.F(p), s.F(u), s.F(v).
// If withAccel == false, a_f = 0 (used by reproject).
poisson::Result project(State& s, double dt, bool withAccel = true);

// MAC predictor (prm.macPredictor): s.uf <- time-centred (t + dt/2) BCG face values of
// the normal velocity from u^n, the previous uf (upwinding) and the previous cell
// acceleration, made divergence free with a MAC projection (coefficient 1/rho). Called
// after the VOF advection and before advanceMomentum, so that only the momentum advection
// uses it (as in Basilisk): the VOF step keeps the exactly balanced face velocity of the
// last projection, which preserves the balanced-force equilibrium of static interfaces.
// Makes the advecting velocity of the momentum second order in time (+1 Poisson solve).
poisson::Result predictFaceVelocity(State& s, double dt);

// Face-normal velocity interpolated from the cell-centred u, v, honouring the BCs
// (walls: 0, inflow: prescribed, outflow: interior value). Result size nFaces.
void cellToFace(const State& s, std::vector<double>& uf);

// After an adapt(): rebuild s.uf = cellToFace(u), then make it divergence free with a
// projection that does NOT modify s.F(p), s.F(u), s.F(v).
poisson::Result reproject(State& s);

}  // namespace ns
