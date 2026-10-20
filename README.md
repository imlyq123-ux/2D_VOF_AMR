# vof2d — 2D two-phase flow with VOF and quadtree AMR

Incompressible two-phase Navier–Stokes solver in C++17 (no external libraries, optional
OpenMP):

* **Interface**: geometric VOF — PLIC reconstruction with Mixed-Youngs-Centered normals,
  Weymouth & Yue (2010) conservative, bounded, directionally split advection.
  Volume is conserved to round-off (~1e-14) and 0 ≤ f ≤ 1 exactly.
* **Surface tension**: balanced-force CSF with height-function curvature
  (2nd-order, with parabola-fit fallbacks).
* **Flow**: collocated cell-centred velocity + face fluxes, BCG-type 2nd-order momentum
  advection, implicit viscosity, approximate projection (Basilisk-style), density ratios
  up to 1000+. Optional momentum-conserving advection (`conservingMomentum = true`):
  each phase's momentum is carried by the same geometric fluxes as the volume fraction,
  which prevents spurious kinetic-energy growth in violent high-density-ratio flows
  (on by default for `breaking_wave`). Optional MAC predictor (`macPredictor = true`):
  the momentum is advected with a time-centred, MAC-projected face velocity
  (Bell–Colella–Glaz / Basilisk style) instead of the face velocity of the previous
  projection — second order in time instead of first, for one extra Poisson solve
  per step (~1.9× cost). The VOF step always uses the exactly balanced face velocity of
  the last projection (keeps the balanced-force equilibrium of static interfaces).
* **Pressure**: flexible CG preconditioned by aggregation AMG (K-cycle); ~15–25
  iterations independent of mesh size and density ratio.
* **AMR**: cell-based quadtree, 2:1 balanced (incl. corners), O(1) neighbour lookup.
  The interface is always at the finest level with a buffer; elsewhere the mesh follows
  a wavelet estimate of the velocity. All operators are face-flux based, so they are
  conservative across resolution jumps.

## Build and test

```bash
make -j            # bin/vof2d
make check         # builds and runs all unit tests (~35 s)
```

## Run

```bash
./bin/vof2d cases/rising_bubble.cfg                     # Hysing TC1 at h = 1/128
./bin/vof2d cases/rising_bubble.cfg maxLevel=8 outDir=out/l8   # command-line overrides
OMP_NUM_THREADS=4 ./bin/vof2d cases/dam_break.cfg
```

Case defaults are applied first, then the config file, then `key=value` overrides.
Every key is documented in the `cases/*.cfg` files; `outDir/params.log` records the full
parameter set of a run and can itself be used as a config file.

Available cases: `rising_bubble`, `rising_bubble2` (Hysing et al. 2009 TC1/TC2),
`static_drop`, `oscillating_drop`, `rayleigh_taylor`, `dam_break`, `breaking_wave`,
`zalesak`, `single_vortex`.

Output in `outDir`:

| file | content |
|------|---------|
| `fields_*.vtu`, `fields.pvd` | leaf cells with f, u, v, p, rho, mu, kappa, level, vorticity (ParaView) |
| `interface_*.vtp`, `interface.pvd` | PLIC interface segments |
| `diagnostics.dat` | per-step volumes, centroids, phase velocities, kinetic energy, circularity |

Post-processing (numpy + matplotlib only):

```bash
python3 scripts/plot_field.py output/rising_bubble/fields_00030.vtu --field vorticity --edges
python3 scripts/plot_diagnostics.py output/rising_bubble/diagnostics.dat   # overlays Hysing reference
python3 scripts/make_video.py output/rising_bubble                          # MP4: grid + f + vorticity (needs ffmpeg)
python3 scripts/make_video.py output/static_drop --field3 speed --log --diag umax --diag-log --normalize-arrows
```

## Validation (results from this code)

Hysing et al. (2009) rising bubble, test case 1 (reference: TP2D):

| quantity | level 7, uTol=0.005 | level 8, uTol=0.002 (default) | reference |
|---|---|---|---|
| max rise velocity | 0.2414 @ t=0.934 | 0.2420 @ t=0.939 | 0.2417 @ t=0.921 |
| min circularity | 0.8995 @ t=2.01 | 0.9023 @ t=1.936 | 0.9013 @ t=1.904 |
| centroid y(t=3) | 1.0837 | 1.0829 | 1.0813 |
| leaves / wall time | 4.3k / 18 s | 10k / 85 s | |

AMR vs uniform at level 7: with `uTol=0.001` the AMR run gives y(3) = 1.0822 vs 1.0825 on a
uniform 128² mesh, with 4× fewer cells and 2.2× less wall time. Volume drift ~1e-12.

Test case 2 (level 8): rise-velocity maxima 0.2506 @ 0.735 and 0.2378 @ 2.00
(reference 0.2502 @ 0.732, 0.2393 @ 2.06), centroid y(3) = 1.130 (ref. ≈ 1.138).

Unit tests (`make check`):

* VOF single vortex (T = 8): L1 shape error 2.1e-3 (128²), 5.4e-4 (256²), order ≈ 2;
  mass error ≤ 5e-15; AMR result identical to uniform mesh.
* Height-function curvature: 2nd order for circles, ellipse, sine wave, wall-attached drops.
* Static drop with VOF + computed curvature: capillary number of spurious currents ~4e-5;
  with exact curvature the drop stays at rest to ~1e-11 (balanced force).
* Oscillating drop (mode 2, ε = 0.1, R/h = 25.6): period 0.2350 vs 0.2306 from linear
  theory, damping rate 0.209 vs ≈ 0.20.
* Breaking wave (Deike, Popinet & Melville, JFM 769, 2015: eps = 0.55, Re = 40000,
  Bo = 1000; level 9, 11 min): 84% of the wave energy dissipated after 4 periods vs ≈70%
  in the paper (Chen et al. 1999 ≈80%, Iafrati 2011 ≈60%); breaking-stage decay rate
  0.27 vs 0.22; breaking starts at t/T ≈ 0.6 vs ≈ 1. The paper used level 10 and a vortex
  sheet in the air at t = 0 (here: air at rest). `make_video.py --wave-energy` plots it.
* Hydrostatics, density ratio 1000 on AMR with regridding: |u| ≤ 1e-11.
* Taylor–Green vortex: 2nd order (uniform and AMR); lid-driven cavity Re=100 within
  0.005 of Ghia et al.
* Poisson/AMG: 2nd order on uniform meshes; iterations constant from 65k to 1M unknowns.
* MAC predictor (translating Taylor–Green, refined at fixed CFL): L2 order 1.6 vs 1.1
  with the lagged advection velocity, 3× (128²) to 5× (256²) more accurate; with the
  slope limiter switched off the orders are exactly 2.0 vs 1.0.

## Layout

```
src/mesh.*          quadtree AMR mesh, fields, faces, refine/coarsen, transfer
src/amr.*           refinement criterion (interface band + velocity wavelets)
src/plic.*, vof.*   PLIC geometry, reconstruction, advection, VOF prolongation
src/curvature.*     height-function curvature
src/poisson.*       AMG-preconditioned CG for -div(beta grad x) + lambda x = b
src/navier_stokes.* momentum predictor, balanced-force projection, time-step control
src/io.*, cases.*   config, VTK output, diagnostics, test cases
src/simulation.*    initialization, time step sequence, run loop
docs/ARCHITECTURE.md  design notes and conventions
```

## Known limitations

* At coarse/fine faces the elliptic operator uses a symmetric two-point flux that
  ignores the tangential offset of the coarse cell centre; this is locally first order
  (the solver stays SPD and the interface is never at a resolution jump, so two-phase
  results are unaffected, but single-phase AMR errors are 2–4× those of the uniform
  mesh at the coarse resolution).
* Time integration of the viscous term is backward Euler (first order in time). Without
  `macPredictor` the momentum advection velocity is lagged (first order in time).
* Static drop, long times: once the spurious currents have decayed to ~1e-8, solver
  round-off seeds a slow drift of the whole drop (surface-tension force errors of the
  height-function curvature push a drop that is off its grid-symmetric position further),
  growing ~4× per unit time. With `poissonTol = 1e-13` the currents keep decaying to
  ~5e-10 by t = 7 before the drift starts. Irrelevant for flows that are not at rest.
* Contact lines: walls act as a 90° contact angle.
* For small meshes (< ~10^4 leaves) OpenMP brings little; use `OMP_NUM_THREADS=1..4`.
* Memory: per-level dense index maps (4 bytes × (nx0·ny0·4^maxLevel) at the finest
  level), fine up to maxLevel ≈ 11–12 in 2D.

## Figures

* `docs/images/rising_bubble_tc1.png` — TC1 diagnostics at h = 1/128 and 1/256 vs reference
* `docs/images/rising_bubble_v_t3.png` — vertical velocity and AMR mesh at t = 3
* `docs/images/rt_dambreak_zalesak.png` — Rayleigh–Taylor (t = 3.5), dam break (t = 0.25 s),
  Zalesak disk after one revolution
* `docs/images/breaking_wave_energy.png` — breaking wave after 4 periods: AMR grid, fields
  and the energy budget against Deike et al. (2015)

## License

MIT — see [LICENSE](LICENSE).
