# Diffusion in the gap between cells

Status: first benchmark of the extracellular-space (ECS) work. Flow and electrical coupling are not started.
Results: `results/diffusion_benchmark.md` and `.png`; code: `zmorph/bd.py`, `zmorph/diffusion_ref.py`,
`examples/run_diffusion_benchmark.py`, `examples/report_diffusion.py`; tests: `tests/test_bd.py`.

## What was built

* **Implicit gap.** For a label partition the competition margin `m = best - second-best` of the per-label signed
  distances equals twice the distance to the nearest cell-cell interface, so `ECS = { m < w }` is a gap of
  thickness `w` centred on every interface. Sheets on faces, tunnels along triple lines and pools at four-cell
  corners appear automatically; no surface is offset, so nothing self-intersects. `w` can later be a field
  (and a function of time, for the sleep expansion).
* **Brownian dynamics (BD).** Point tracers do a Gaussian random walk; a step that leaves `{m < w}` is bisected to the
  wall and reflected specularly about the field gradient. numba, multi-core, about 2e7 steps/s on 4 cores.
  No mesh. Geometry from labels (`margin_field`) or from any level set.
* **Reference solvers** for comparison: voxel finite volumes (periodic cell problem for tortuosity; graph-Laplacian
  decay rate), a triangle-surface cotangent Laplacian (the thin-film limit on a surface mesh), and the exact solution
  for a spherical shell.

## Findings (D = 1, voxel 8 nm, gap 12-32 nm)

1. **Reflecting walls are unbiased**: uniform density across the gap and in-plane D equal to the free D, for steps up
   to w/5 (`results/diffusion_benchmark.md`, section 1).
2. **Lattice tortuosity** (cubic cells of side 200 nm): BD agrees with the voxel solver converged at 2 nm
   (1.386 +- 0.027 vs 1.392 at w = 20 nm), from labels as well as from the analytic geometry.
3. **Voxel solvers mis-size the gap.** At 8 nm voxels a 12 nm and a 20 nm gap give the same 16 nm mask. The ECS volume
   fraction is off by 30% (too high at 12 nm) or 18% (too low at 20 nm, 4 nm voxels). The ratio lambda^2 hides this;
   a flow or osmotic calculation would not.
4. **Ball in socket (maximum bend)**: BD is within about 1.5% of the exact shell solution on every geometry source
   (analytic, labels, smoothed labels). The voxel solver is 19%, 9% and 7% low at 8, 4 and 3 nm (first-order
   convergence, so about 1 nm voxels would be needed). A surface mesh with the cotangent Laplacian converges fast
   on an analytic sphere (-2.7%, -0.6%, -0.1% at 8, 4, 2 nm).
5. **The mesh needs smoothing.** A mesh marched out of the raw voxelised labels is 7-8% low at both 8 and 4 nm: the
   staircase roughness adds surface area and does not shrink with resolution. A Gaussian-smoothed distance field
   (as sdf_align uses) or Taubin smoothing removes it (within +-0.6%).

So at the data's resolution a mesh-free, level-set geometry with BD (or a smoothed surface with a thin-film
equation) is accurate where a voxel grid is not, and the bend error the earlier discussion worried about is a
roughness and resolution error, not a dihedral-angle error.

## Limits and what this does not show

* The geometry is a model. The gap width is a prior, not a measurement (8 x 40 nm voxels vs a 20 nm gap), the cells
  are idealised (cubes, a sphere) and the gap uniform. These tests show the numerics agree with exact answers, not
  that a real ECS looks like this.
* BD errors are about 1-2% (1 sem); differences below that are not resolved. Particle counts: 4-6 thousand
  (lattice) and 20 thousand (sphere).
* Point tracers in a free fluid: no matrix, no charge, no channels, no flow. Those are the next steps.
* Tissue volumes were not run: only phantoms. `margin_field` on a real tile has not been timed.

## Moving walls and membrane patches (second benchmark)

Results: `results/bd_dynamic.md`; code: `simulate_dynamic`, `wall_force` in `zmorph/bd.py`; script
`examples/run_bd_dynamic.py`; tests `tests/test_bd_dynamic.py`.

* **Moving gap.** `simulate_dynamic(m, h, x0, w_of_t, ...)` moves the wall as the level set `m - w(t) = 0`; a wall that
  advances onto a particle pushes it back inside. `w` is a scalar function of time here; a spatial field `w(x, t)` is a
  one-line extension.
* **Wall force from the reflections.** In overdamped dynamics the wall force is (kT/D) x (displacement the wall imposes) / time.
  For a planar gap the total force on both walls equals 2 N kT / w to within 0.5% for every gap (12-40 nm) and step
  (0.5-2 nm). This is the osmotic pressure of the tracer, measured mechanically, with no solvent.
* **Piston.** Compressing a 40 nm gap to 20 nm slowly costs 1.00-1.05 x the isothermal work N kT ln 2; compressing fast costs
  1.48 x. Expanding slowly returns 0.95-0.99 x, fast 0.66 x. Particles stay inside the gap throughout.
* **Patches.** An integer patch map on the grid is looked up at the wall-hit point; absorption probability per hit p.
  One absorbing wall: the decay rate of survivors matches the exact Robin result to 0.5-1.6% (a 4.6% outlier at the
  coarsest step and Biot number 2), with p = kappa sqrt(pi dt / D). A checkerboard of patches over half of one wall behaves like
  a uniform wall with kappa/2 (-0.7% and -0.3%); assuming full coverage would be 73-85% wrong. Absorptions on a spherical-shell
  cap land inside the cap (largest polar angle 21, 41, 61 degrees for 20, 40, 60 degree caps).

Limits: absorbed particles disappear (no compartment on the other side yet), patches are absorbing or reflecting (no gating),
and the moving wall is a prescribed `w(t)` (no coupling to water flux).

## Next

1. Compartments on the far side of the patches (re-injection, gating), reactive species, and a discreteness check against the
   continuum limit. `w` as a field, and `w(t)` coupled to osmotic water flux.
2. Flow: Stokes/lubrication on the interface network with a thickness field; compare against BD-derived permeability
   in the same geometry.
3. Electrical coupling: Poisson in the gap with membrane current sources.
4. `w(t)` driven by osmotic water flux (the sleep expansion) as a moving level set.
5. Apply to the real tile (`data/tile_*.npz`): margin field after sdf_align, ECS fraction and tortuosity versus `w`.
