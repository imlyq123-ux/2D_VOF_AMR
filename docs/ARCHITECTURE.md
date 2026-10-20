# vof2d — architecture

2D incompressible two-phase flow, geometric VOF (PLIC), quadtree AMR, C++17, no
external libraries (OpenMP optional).

## Modules and ownership

| Module            | Files                                              | Public header        |
|-------------------|----------------------------------------------------|----------------------|
| Mesh / AMR core   | `src/mesh.cpp`                                     | `src/mesh.hpp`       |
| VOF (PLIC + advection) | `src/plic.hpp`, `src/plic.cpp`, `src/vof.cpp` | `src/vof.hpp`        |
| Curvature         | `src/curvature.cpp`                                | `src/curvature.hpp`  |
| Linear solver     | `src/poisson.cpp`                                  | `src/poisson.hpp`    |
| Navier–Stokes     | `src/navier_stokes.cpp`                            | `src/navier_stokes.hpp` |
| I/O, config, cases| `src/io.cpp`, `src/cases.cpp`, `scripts/`, `cases/`| `src/io.hpp`, `src/cases.hpp` |
| AMR criteria, driver | `src/amr.cpp`, `src/simulation.cpp`, `src/main.cpp` | `src/amr.hpp`, `src/simulation.hpp` |

Shared types: `src/common.hpp` (Vec2, Side), `src/params.hpp` (Params, BCs),
`src/state.hpp` (State: Params + Mesh + field ids + face velocity).

## Mesh in one paragraph

Cell-based quadtree with per-level dense index maps → O(1) neighbour lookup
(`Mesh::locate(L,i,j)` returns the cell (L,i,j) or the coarser leaf covering it,
mirroring out-of-domain indices at non-periodic walls). Fields are
`std::vector<double>` over all cells (leaves + ancestors); leaf values are the solution,
parents are averages after `restrict()`. The tree is 2:1 balanced including corners.
`faces()` is the list of leaf faces (coarse/fine faces have the fine size, so a coarse
side may have two faces); every finite-volume operator is a loop over faces, which makes
fluxes conservative across resolution jumps automatically. Face convention: normal
from `L` to `R` (+x or +y); boundary faces have `L = -1` (left/bottom) or `R = -1`
(right/top) and `bside` set.

Typical leaf loop and divergence:

```cpp
for (int c : m.leaves()) {
  double div = 0;
  for (const int* p = m.cellFacesBegin(c); p != m.cellFacesEnd(c); ++p) {
    const Mesh::Face& F = m.faces()[*p];
    div += (F.L == c ? 1.0 : -1.0) * uf[*p] * F.area;
  }
  div /= m.volume(c);
}
```

Stencil access at a leaf's own level (e.g. 3x3 block for normals):

```cpp
m.restrict(fcopy);                       // parents valid
int L = m.level(c), i = m.ix(c), j = m.iy(c);
double fij = fcopy[m.locate(L, i + di, j + dj)];
```

## Time step (sim::step)

```
dt = min(ns::stableDt, output time)                 (vofOnly: CFL on prescribed uf)
vof::advect(f, uf, dt)                              Weymouth–Yue split, uf divergence-free
   [conservingMomentum: phase momenta f u, (1-f) u advected as VOF tracers with the
    same geometric fluxes, then u = (rho1 q1 + rho2 q2) / rho]
ns::updateProperties                                rho, mu from f
curvature::compute                                  height functions
[macPredictor: ns::predictFaceVelocity(dt)          BCG face velocity at t+dt/2, MAC-projected;
                                                    used only by the momentum advection]
ns::advanceMomentum(dt)                             u* (advection unless conservingMomentum, viscosity)
ns::project(dt)                                     balanced-force projection -> uf^{n+1}, u^{n+1}, p
every adaptEvery steps: amr::adapt -> mesh.adapt -> ns::updateProperties -> ns::reproject
```

The VOF step always uses the face velocity of the last projection: it is exactly balanced
(a_f - alpha_f grad_f p = 0 at equilibrium), which keeps static interfaces at rest. Using
the predicted velocity there was tried and makes static-drop spurious currents grow.

## AMR criterion (src/amr.cpp)

* every leaf within `interfaceBuffer` max-level cells of an interfacial cell is refined
  to `maxLevel` (so VOF/curvature stencils never see a resolution jump);
* elsewhere, a wavelet estimate of u, v (difference between the leaf values and the
  prolongation from the parent) refines above `uTol` and coarsens below `uTol/1.5`.
