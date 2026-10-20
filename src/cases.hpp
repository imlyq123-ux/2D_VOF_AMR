// cases.hpp - predefined test cases (initial conditions, default parameters).
//
// Owner: I/O module (src/cases.cpp).
#pragma once
#include <functional>
#include <string>
#include <vector>
#include "mesh.hpp"
#include "params.hpp"

namespace cases {

struct CaseSetup {
    // Level set of the initial interface: phase 1 (f = 1) where phi(x, y) > 0. Required.
    std::function<double(double, double)> phi;
    // Initial velocity (optional; zero if empty).
    std::function<Vec2(double, double)> u0;
    // Stream function psi(x, y, t) of a prescribed flow (required when prm.vofOnly):
    // u = d psi / dy, v = - d psi / dx. The driver builds exactly divergence-free face
    // velocities from psi at face end points.
    std::function<double(double, double, double)> psi;
};

// Writes the case's default parameters into prm (domain, levels, fluids, BCs, tEnd, ...)
// and fills cs. The driver applies the user's config file / command-line overrides
// AFTER this call. Returns false for an unknown name.
//
// Available cases:
//   rising_bubble   Hysing et al. 2009 test case 1 (rho 1000/100, mu 10/1, sigma 24.5)
//   rising_bubble2  Hysing et al. 2009 test case 2 (rho 1000/1, mu 10/0.1, sigma 1.96)
//   static_drop     Laplace equilibrium, spurious currents
//   oscillating_drop  capillary oscillation of a mode-2 perturbed drop (Lamb frequency)
//   rayleigh_taylor Rayleigh-Taylor instability (Atwood 0.5)
//   dam_break       collapse of a water column (Martin & Moyce)
//   breaking_wave   plunging breaker, Deike, Popinet & Melville JFM 769 (2015), periodic
//   zalesak         rigid rotation of Zalesak's slotted disk (vofOnly)
//   single_vortex   reversible single vortex (Bell-Colella-Glaz / Rider-Kothe, vofOnly)
bool setupCase(const std::string& name, Params& prm, CaseSetup& cs);

std::vector<std::string> caseNames();

// Volume fraction of {phi > 0} in leaf c, accurate to ~1e-6 for smooth interfaces
// (adaptive sub-sampling of the cell).
double cellFraction(const Mesh& m, int c, const std::function<double(double, double)>& phi);

}  // namespace cases
