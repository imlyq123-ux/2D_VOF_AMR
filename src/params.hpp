// params.hpp - run-time parameters of a simulation (filled by io::loadParams and cases).
#pragma once
#include <string>
#include "common.hpp"

// Boundary condition type of one domain side.
//  Wall     : no-slip, u = v = 0 (or the given tangential velocity u/v for a moving lid)
//  Slip     : free-slip, normal velocity 0, zero tangential stress
//  Inflow   : prescribed velocity (u, v) and volume fraction f
//  Outflow  : p = 0 (Dirichlet), zero normal gradient for u, v, f
//  Periodic : must be set on both opposite sides
enum class BCType { Wall, Slip, Inflow, Outflow, Periodic };

struct SideBC {
    BCType type = BCType::Wall;
    double u = 0.0, v = 0.0;  // velocity for Wall (tangential lid velocity) / Inflow
    double f = 0.0;           // volume fraction for Inflow
};

struct Params {
    // ---- case / output
    std::string caseName = "rising_bubble";
    std::string outDir = "output";
    double outputEvery = 0.1;   // simulated time between VTK dumps (<=0: never)
    int diagEvery = 1;          // steps between diagnostics lines

    // ---- domain and mesh: nx0 x ny0 root cells of size L0, lower-left corner (x0,y0)
    int nx0 = 1, ny0 = 1;
    double x0 = 0.0, y0 = 0.0, L0 = 1.0;
    int minLevel = 4, maxLevel = 7;

    // ---- fluids: phase 1 is where f = 1, phase 2 where f = 0
    double rho1 = 1.0, rho2 = 1.0;
    double mu1 = 0.0, mu2 = 0.0;
    double sigma = 0.0;
    double gx = 0.0, gy = 0.0;

    // ---- time integration
    double tEnd = 1.0;
    double cfl = 0.25;          // advective CFL (VOF split advection needs <= 0.5)
    double dtMax = 1e-2;
    double dtMin = 1e-12;
    int maxSteps = 100000000;
    bool implicitViscosity = true;
    bool conservingMomentum = false;  // advect momentum with the VOF fluxes (large density ratios)
    bool macPredictor = false;        // MAC-projected time-centred momentum advection velocity (+1 solve/step)

    // ---- boundary conditions, indexed by Side
    SideBC bc[4];

    // ---- adaptivity
    int adaptEvery = 2;          // steps between regrids
    int interfaceBuffer = 4;     // max-level cells around the interface kept at maxLevel
    double uTol = 1e-2;          // wavelet error threshold on velocity (absolute)
    bool adaptVelocity = true;   // use the velocity wavelet criterion away from the interface

    // ---- linear solver
    double poissonTol = 1e-9;   // relative residual
    int poissonMaxIter = 1000;

    // ---- kinematic-only runs (prescribed velocity from a stream function, no NS solve)
    bool vofOnly = false;
    double period = 0.0;         // used by time-reversing prescribed flows (single vortex)
};
