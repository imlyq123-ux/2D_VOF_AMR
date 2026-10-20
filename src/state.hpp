// state.hpp - the complete simulation state shared by all modules.
#pragma once
#include <memory>
#include <vector>
#include "mesh.hpp"
#include "params.hpp"

// Mesh field ids of the cell-centred fields (created by sim::initialize).
struct FieldIds {
    int f = -1;      // volume fraction of phase 1           (prolongation: vof::prolong)
    int u = -1;      // cell-centred velocity x              (linear)
    int v = -1;      // cell-centred velocity y              (linear)
    int p = -1;      // pressure                              (linear)
    int rho = -1;    // density, from f                       (recomputed after adapt)
    int mu = -1;     // dynamic viscosity, from f             (recomputed after adapt)
    int kappa = -1;  // interface curvature (see curvature.hpp)
};

struct State {
    Params prm;
    std::unique_ptr<Mesh> mesh;
    FieldIds id;
    // Face-normal velocity, one value per mesh->faces() entry, positive along +x/+y.
    // Discretely divergence free after every projection. Rebuilt after adapt().
    std::vector<double> uf;
    double t = 0.0, dt = 0.0;
    int step = 0;
    int lastPoissonIters = 0;

    Mesh& m() { return *mesh; }
    const Mesh& m() const { return *mesh; }
    std::vector<double>& F(int fid) { return mesh->field(fid); }
    const std::vector<double>& F(int fid) const { return mesh->field(fid); }
};
