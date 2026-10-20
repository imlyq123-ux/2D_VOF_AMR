// simulation.hpp - initialization, time stepping and the run loop.
#pragma once
#include "cases.hpp"
#include "state.hpp"

namespace sim {

// Builds the mesh and fields from s.prm, refines the initial mesh around the interface
// given by cs.phi, sets f (exact area fractions), u0, properties and a divergence-free uf.
void initialize(State& s, const cases::CaseSetup& cs);

// Advances one time step of at most dtLimit (the step may be shorter: CFL etc.),
// then adapts the mesh every prm.adaptEvery steps.
void step(State& s, const cases::CaseSetup& cs, double dtLimit);

// Regrid now (flags from amr::computeFlags). Returns true if the mesh changed.
bool regrid(State& s, const cases::CaseSetup& cs);

// Exactly divergence-free face velocities of a prescribed stream function at time t.
void faceVelocityFromStreamFunction(const Mesh& m, const cases::CaseSetup& cs, double t,
                                    std::vector<double>& uf);

// Time loop until prm.tEnd with VTK output, diagnostics and progress messages.
void run(State& s, const cases::CaseSetup& cs);

}  // namespace sim
