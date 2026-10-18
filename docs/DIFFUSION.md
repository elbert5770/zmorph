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
* The first two benchmarks use phantoms. A real tile crop is treated in the last two sections; `margin_field` on the 71 x 256 x 256 crop takes
  about 10-100 s depending on the number of labels in the crop.

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

## Synaptic sources and release on a real tile

Code: `zmorph/synapse.py`; scripts `examples/run_synapse_source.py`, `run_synapse_sweep.py`, `make_spread_figure.py`;
results `results/synapse_source/`; tests `tests/test_synapse.py`. Crop: tile `170398_60741`, rows and columns 0-256,
sections 0-40 (201 slices at 8 nm).

* **Candidate clefts.** `contact_patches` finds connected contact patches between touching labels and describes each by centre,
  normal, disc-equivalent area and radius, roundness and flatness; `score_patches` gives a synapse-likeness from 0 to 1 (area window about
  0.02-0.3 um^2, round, flat). This is geometry only: labels carry no PSD or vesicle information, so non-synaptic contacts score the same.
  `snap_annotations` replaces the guess with real synapse positions when they exist. On the crop there are 759 patches; the top ten
  score 0.85-0.89 with areas 0.04-0.10 um^2.
* **Release.** `release_positions` returns a burst of tracers in the ECS voxels near the chosen patch. Units: it returns nm;
  `simulate_dynamic` takes `x0` in voxel units (an early run passed nm and the tracers never moved).
* **Static gap and breathing gap.** From the top patch, 2000 tracers, tmax 2e5 (about 1.3 ms for D = 1.5e-10 m^2/s, an assumed Abeta value):
  rms displacement is 698-704 nm for w = 12-30 nm (no change within seed scatter), 723 nm at 40 nm and 741 nm at 60 nm. A ramp
  from 20 to 32 nm gives 719 nm; oscillations of +-30% about 20 nm give 707-712 nm, indistinguishable from the static mean. Diffusion in a thin
  sheet is two-dimensional and independent of its thickness, so only tunnels and pools should make `w` matter, which they do slowly here.
  These runs used a short window; apparent D from them is biased high (see below).
* Release is a single burst at t = 0; absorbing sinks and a release time course are not built.

## The real-tile pipeline: interpolation, meshes and ECS tortuosity (15 sections)

Script `examples/run_tile15.py` (results `results/tile15/summary_*.json`, `volume.npz`; `meshes.npz` is regenerated, not committed),
`run_tile15_bd_periodic.py`, `run_tile15_fv_refined.py`. Same crop, sections 0-14, sdf_align with factor 5 (71 slices).

**Interpolation and meshes.** The input sections are reproduced at 99.95% of pixels (not exactly; cause not traced). 60 of the 61 labels
in the sections appear in the volume and one label gains an extra connected component. The 59 competition-field meshes are all watertight
(1.95 million triangles); 93% of interface vertices are shared exactly between neighbouring cells (95th percentile of the rest 0.05 voxel); the
5th-percentile minimum angle is 5.7 degrees.

**ECS fraction and tortuosity** (`m < w`, periodic cell problem for FV; D and 1/lambda^2 are in units of the free D; axes z, y, x):

| w (nm) | ECS fraction | FV f=1 | FV f=2 | BD periodic, tmax 2e6 |
|---|---|---|---|---|
| 20 | 0.158 (f=1), 0.143 (f=2) | 0.305, 0.397, 0.400 | 0.336, 0.437, 0.435 | 0.368, 0.495, 0.478 |
| 40 | 0.261 (f=1), 0.272 (f=2) | 0.382, 0.473, 0.480 | 0.416, 0.507, 0.509 | 0.462, 0.542, 0.554 |

Cell problem on the original grid (f=1) for the other widths: w = 10, 16, 30, 60 nm give ECS fractions 0.089, 0.137, 0.218, 0.389 and
1/lambda^2 in x of 0.31, 0.39, 0.45, 0.55.

**How the FV and BD estimates were reconciled.** The first comparison (BD run to about 220 nm rms displacement, non-periodic) had BD 20-85% above FV.
Three separate causes, in order of size:

1. *BD time window.* In a structured medium the apparent diffusivity falls with time towards its long-time value. Extending tmax from 6e4 to
   6e5 to 2e6 reduced D at w = 20 nm in x from 0.62 to 0.52 to 0.48 and at 40 nm from 0.67 to 0.57 to 0.55; the last step is within error. Short-window
   D from the sweeps above is therefore biased high.
2. *The voxel staircase in FV.* Refining the grid by resampling the margin field (trilinear, periodic) raises FV: f = 1 to 2 by 8-13% at 20 nm and
   7-9% at 40 nm; f = 2 to 3 (128-pixel sub-box) by 3-4% at 20 nm and about 1% at 40 nm. The full crop was run at f = 2 only; f = 3 was not run.
3. *Crop boundaries.* Making BD periodic like the FV solver changed almost nothing.

What remains is BD about 7-13% above FV at f = 2 (1.09-1.13 at 20 nm; 1.07-1.11 at 40 nm). The cause is not identified: FV may not be fully
converged at 20 nm, and the BD wall treatment or timestep may bias it slightly. Quote the tortuosity as 1/lambda^2 of roughly 0.4-0.5 in x and y
at w = 20-40 nm, with the method spread as the uncertainty. Both methods wrap the crop, so seams are shared artifacts, and the gap width is a prior.
An earlier extrapolation of the sub-box correction to the full crop overshot (predicted about 0.36, 0.50, 0.48; direct f = 2 gave 0.34, 0.44, 0.44 at 20 nm).

## Small patch sinks (disc-shaped absorbers)

Script `examples/run_synapse_smallsink.py`; results `results/synapse_source/smallsink_170398_60741.{json,md}`. Same crop. Two sites in the gap
about 500 nm from the top cleft; the sink is the part of the cell under the site within R of it (R = 100, 200, 300 nm), p = 1 per wall hit.
800 tracers x 2 seeds per condition, tmax 1e6 (about 6.7 ms for D = 1.5e-10 m^2/s, an assumed value). Static w = 12, 20, 30, 40 nm; breathing
w = 20 nm +-30% (period 1e5), burst released at 3 phases. No other sink, so the uncaptured fraction is still wandering at tmax.

| R (nm) | site 0 captured by tmax | site 1 captured by tmax | spread over w (site 0 / 1) |
|---|---|---|---|
| 100 | 0.32-0.36 | 0.32-0.35 | 0.036 / 0.033 |
| 200 | 0.63-0.65 | 0.55-0.57 | 0.018 / 0.023 |
| 300 | 0.74 | 0.67-0.70 | 0.008 / 0.024 |

* **w does not matter here either.** Across a 3-fold change in w the captured fraction varies by at most 3.6 points, with no monotonic trend, and
  breathing falls inside the static range (phase-to-phase range 0.6-5.4 points). I did not measure seed scatter separately (two seeds were averaged),
  so effects below about 3 points are not resolved.
* **The patch size and the site do matter.** Capture rises steeply from R = 100 to 200 nm and saturates towards 300 nm; at R = 200 nm the two
  sites differ by about 10 points. The local sheet network (how many routes lead to the patch) matters more than the gap width.
* Half of the final capture is reached at 1.3-2.8e5, earlier for larger patches. A release spread uniformly over 2e5 lowers capture by tmax by
  1-2 points (the table in the .md).
* Not done: a comparison with an analytic rate (the sheet network is not a plane, so no closed form applies), partial absorption (p < 1) on
  small patches, and more sites. The large sink earlier and these small ones agree that, in a thin-sheet network, geometry of connectivity
  and target size set the clearance and the gap width does not.

## Next

1. Compartments on the far side of the patches (re-injection, gating), reactive species, and a discreteness check against the
   continuum limit. `w` as a field, and `w(t)` coupled to osmotic water flux.
2. Flow: Stokes/lubrication on the interface network with a thickness field; compare against BD-derived permeability
   in the same geometry.
3. Electrical coupling: Poisson in the gap with membrane current sources.
4. `w(t)` driven by osmotic water flux (the sleep expansion) as a moving level set.
5. Done for one 15-section crop (above). Still open: more tiles and crops without periodic wrapping, full-crop FV at f = 3, a BD timestep
   and wall-treatment check to close the last 10%, absorbing sinks and a release time course for the Abeta runs.
