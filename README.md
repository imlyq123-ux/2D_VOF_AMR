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

## Gallery

Each animation shows, left to right: the AMR grid coloured by refinement level (red: PLIC
interface), the volume fraction with velocity arrows, a flow field, and time series.
Click an animation for the full-resolution MP4. The raw data (VTK files, diagnostics)
are not in the repository; the commands below regenerate them.

### Rising bubble (Hysing et al. 2009, test case 1)

[![rising bubble](docs/media/rising_bubble.gif)](docs/media/rising_bubble.mp4)

A light bubble (ρ 1000/100, μ 10/1, σ 24.5) rises and deforms into a skirted cap; the
interface band stays at level 7 while the wake and the side gaps refine with the flow
(3,400 → 6,200 leaves). Max rise velocity 0.2421 (ref. 0.2417), min circularity 0.9007
(ref. 0.9013), centroid y(3) = 1.0823 (ref. 1.0813).

```bash
./bin/vof2d cases/rising_bubble.cfg maxLevel=7 outputEvery=0.02 outDir=output/video_rising_bubble
python3 scripts/make_video.py output/video_rising_bubble
```

### Static drop (spurious currents)

[![static drop](docs/media/static_drop.gif)](docs/media/static_drop.mp4)

A drop at rest (La = 12000): the parasitic currents of the start-up transient decay from
8e-4 to ~2e-8 (log-scale speed in the third panel, arrow directions only). The grid does not
change because nothing moves.

```bash
./bin/vof2d cases/static_drop.cfg outputEvery=0.05 outDir=output/video_static_drop
python3 scripts/make_video.py output/video_static_drop --field3 speed --log --diag umax --diag-log --normalize-arrows
```

### Oscillating drop (capillary mode 2)

[![oscillating drop](docs/media/oscillating_drop.gif)](docs/media/oscillating_drop.mp4)

An elliptic drop (10 % deformation, density ratio 100) oscillates under surface tension; the
far-field grid refines and coarsens with every swing. Period 0.2350 vs 0.2306 (linear theory),
damping rate 0.209 vs ≈ 0.20.

```bash
./bin/vof2d cases/oscillating_drop.cfg outputEvery=0.01 outDir=output/video_oscillating_drop
python3 scripts/make_video.py output/video_oscillating_drop --field3 speed --diag ekin --fps 20
```

### Rayleigh–Taylor instability (At = 0.5, Re = 3000)

[![Rayleigh-Taylor](docs/media/rayleigh_taylor.gif)](docs/media/rayleigh_taylor.mp4)

Heavy fluid falls into light fluid: mushroom, Kelvin–Helmholtz roll-ups, impact on the
bottom wall and mixing (t = 6). The finest-level band follows the stretching interface
(2,300 → 41,000 leaves) while the undisturbed fluid stays coarse.

```bash
./bin/vof2d cases/rayleigh_taylor.cfg tEnd=6 outputEvery=0.025 outDir=output/video_rayleigh_taylor
python3 scripts/make_video.py output/video_rayleigh_taylor --height 10 --dpi 100 --arrows 10 --diag ekin
```

### Breaking wave (Deike, Popinet & Melville 2015)

[![breaking wave](docs/media/breaking_wave.gif)](docs/media/breaking_wave.mp4)

A steep Stokes wave (ε = 0.55, Re = 40000, Bo = 1000, density ratio 850) overturns, plunges,
traps air and breaks up over 4 wave periods (levels 4–9, momentum-conserving advection).
The top-right panel compares the energy budget with the literature: 84 % of the wave energy
dissipated after 4 periods vs ≈ 70 % in Deike et al. (Chen et al. 1999: ≈ 80 %, Iafrati
2011: ≈ 60 %).

```bash
./bin/vof2d cases/breaking_wave.cfg outDir=output/breaking_wave_deike2015
python3 scripts/make_video.py output/breaking_wave_deike2015 --wave-energy --fps 20 --dpi 100 --arrows 16 --arrow-umax 1.0
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

## Acknowledgements and references

This is an independent C++ implementation (no third-party source code), but its design
follows closely the methods of [Basilisk](http://basilisk.fr) and Gerris by Stéphane
Popinet: cell-based quadtree AMR with composite operators, collocated approximate
projection with balanced-force surface tension, height-function curvature, conservative
split VOF, momentum-conserving two-phase transport and wavelet-based adaptation.

**Adaptive solver framework**
- S. Popinet, Gerris: a tree-based adaptive solver for the incompressible Euler equations
  in complex geometries, *J. Comput. Phys.* 190 (2003) 572–600.
- S. Popinet, An accurate adaptive solver for surface-tension-driven interfacial flows,
  *J. Comput. Phys.* 228 (2009) 5838–5866 (balanced force, height functions).
- S. Popinet, A quadtree-adaptive multigrid solver for the Serre–Green–Naghdi equations,
  *J. Comput. Phys.* 302 (2015) 336–358 (Basilisk).

**Volume of fluid**
- R. Scardovelli, S. Zaleski, Analytical relations connecting linear interfaces and volume
  fractions in rectangular grids, *J. Comput. Phys.* 164 (2000) 228–237.
- E. Aulisa, S. Manservisi, R. Scardovelli, S. Zaleski, Interface reconstruction with
  least-squares fit and split advection in three-dimensional Cartesian geometry,
  *J. Comput. Phys.* 225 (2007) 2301–2319 (Mixed-Youngs-Centered normals).
- G. D. Weymouth, D. K.-P. Yue, Conservative Volume-of-Fluid method for free-surface
  simulations on Cartesian-grids, *J. Comput. Phys.* 229 (2010) 2853–2865.
- M. Rudman, A volume-tracking method for incompressible multifluid flows with large
  density variations, *Int. J. Numer. Meth. Fluids* 28 (1998) 357–378 (consistent momentum
  transport).

**Surface tension and Navier–Stokes**
- J. U. Brackbill, D. B. Kothe, C. Zemach, A continuum method for modeling surface
  tension, *J. Comput. Phys.* 100 (1992) 335–354.
- M. M. Francois et al., A balanced-force algorithm for continuous and sharp interfacial
  surface tension models within a volume tracking framework, *J. Comput. Phys.* 213 (2006)
  141–173.
- J. B. Bell, P. Colella, H. M. Glaz, A second-order projection method for the
  incompressible Navier–Stokes equations, *J. Comput. Phys.* 85 (1989) 257–283.

**Linear solver**
- Y. Notay, Flexible conjugate gradients, *SIAM J. Sci. Comput.* 22 (2000) 1444–1460.
- Y. Notay, P. S. Vassilevski, Recursive Krylov-based multigrid cycles, *Numer. Linear
  Algebra Appl.* 15 (2008) 473–487 (K-cycle).
- Y. Notay, An aggregation-based algebraic multigrid method, *Electron. Trans. Numer.
  Anal.* 37 (2010) 123–146.

**Test cases and reference data**
- S. Hysing et al., Quantitative benchmark computations of two-dimensional bubble
  dynamics, *Int. J. Numer. Meth. Fluids* 60 (2009) 1259–1288 (`rising_bubble`, `rising_bubble2`).
- L. Deike, S. Popinet, W. K. Melville, Capillary effects on wave breaking, *J. Fluid
  Mech.* 769 (2015) 541–569 (`breaking_wave`; energy dissipation data); H. Lamb,
  *Hydrodynamics*, 6th ed., Cambridge University Press (1932) (third-order Stokes wave,
  drop oscillations).
- G. Chen, C. Kharif, S. Zaleski, J. Li, Two-dimensional Navier–Stokes simulation of
  breaking waves, *Phys. Fluids* 11 (1999) 121–133; A. Iafrati, Energy dissipation
  mechanisms in wave breaking processes: spilling and highly aerated plunging breaking
  events, *J. Geophys. Res.* 116 (2011) C07024 (energy loss values quoted by Deike et al.).
- G. Tryggvason, Numerical simulations of the Rayleigh–Taylor instability, *J. Comput.
  Phys.* 75 (1988) 253–282 (`rayleigh_taylor`).
- J. C. Martin, W. J. Moyce, An experimental study of the collapse of liquid columns on a
  rigid horizontal plane, *Phil. Trans. R. Soc. Lond. A* 244 (1952) 312–324 (`dam_break`).
- S. T. Zalesak, Fully multidimensional flux-corrected transport algorithms for fluids,
  *J. Comput. Phys.* 31 (1979) 335–362 (`zalesak`); W. J. Rider, D. B. Kothe,
  Reconstructing volume tracking, *J. Comput. Phys.* 141 (1998) 112–152 (`single_vortex`).
- U. Ghia, K. N. Ghia, C. T. Shin, High-Re solutions for incompressible flow using the
  Navier–Stokes equations and a multigrid method, *J. Comput. Phys.* 48 (1982) 387–411
  (lid-driven cavity test).

## License

MIT — see [LICENSE](LICENSE).
