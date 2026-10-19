# zmorph: where the Aβ / moving-wall work stands

Status at the end of this working session. Details and numbers live in `docs/DIFFUSION.md`, `README.md` and the `results/` folders;
this note is the map.

## The question

Can wall motion during sleep (the extracellular space, ECS, widens and narrows) help clear Aβ from the brain? The route we took:
build a geometry of the ECS from real EM label sections, put diffusing tracers in it, and ask which geometric and dynamic
factors change release and clearance. The method for the geometry is `sdf_align` interpolation onto an isotropic grid.

## What has been built

**Geometry**
- **Interpolation and meshes** (`zmorph/interp.py`, `zmorph/mesh.py`). Label sections become an isotropic volume with conforming,
  watertight cell meshes. A minimal smoothing step (σ = 1 px, two passes) is now the default for the `sdf` family; it removes
  one-voxel spikes without rounding real structure (README, "Regularisation").
- **An implicit gap.** The ECS is the set where the competition margin `m` (twice the distance to the nearest cell–cell interface) is below
  a width `w`. No surface is offset, so nothing self-intersects, and `w` can depend on time.
- **Export for viewing** (`zmorph/export.py`): VTK ImageData (`.vti`) for the fields and PLY for coloured cell meshes, checked
  against VTK's own reader. Figures: `examples/make_ecs_figure.py`, `make_figure2.py`. OpenVDB was considered and not adopted (same samples, extra dependencies).

**Transport**
- **Brownian dynamics (BD)** on the level-set geometry (`zmorph/bd.py`): reflecting walls, moving walls (`w(t)`), absorbing membrane patches,
  periodic boundaries. Validated against exact or reference solutions (shell geometry, planar gap, osmotic wall force, piston work, Robin absorption).
- **GPU port** (`zmorph/bd_gpu.py`, float32). Tested on the A40: kernel tests pass, results agree with the CPU code within
  sampling error, and it is much faster at 10⁴–10⁵ tracers (`examples/bench_bd_gpu.py`, `slurm/*.sbatch`).
- **Synaptic sources and sinks** (`zmorph/synapse.py`, `zmorph/sinks.py`). Candidate clefts from contact geometry (an approximation
  to be replaced by annotations), burst release, whole-cell and small-patch sinks, a fixed-area disc patch.
- **Reference solvers** for cross-checks: voxel finite volumes, a cotangent surface Laplacian, analytic cases (`zmorph/diffusion_ref.py`).
- **A 15-section real tile** run end to end (`examples/run_tile15*.py`): ECS fraction and tortuosity versus `w`, BD against finite volumes.

## What we learned so far (qualitative)

1. **At 8 nm voxels the geometry method matters more than the solver.** Voxel solvers mis-size a 12–20 nm gap; the level-set BD
   and a smoothed surface mesh do not. A raw voxel mesh carries staircase area that does not shrink with resolution.
2. **Reflecting walls are unbiased; the wall force equals the ideal-gas value.** The moving-wall machinery behaves as it should.
3. **Spreading in the thin sheets is effectively two-dimensional.** Spread from a cleft barely depends on `w`, or on whether the gap breathes.
   A ±30% oscillation about 20 nm changes capture by about a point or less (GPU runs with many tracers).
4. **Clearance to a small absorber is set mainly by the target and its surroundings.** Patch radius and site matter far more than the gap; how the
   patch is defined also changes the capture a lot (a 3-D ball of membrane around a site absorbs far more than a disc of the same radius).
5. **A small, systematic dependence on `w` remains**, and a cleaner fixed-area patch makes it larger, not smaller. Its cause is not identified; simple
   dilution into the extra ECS volume is too strong an explanation. It is the main open puzzle at the pore scale.
6. **Tortuosity and ECS fraction from the 15-section tile** are within a spread between methods (BD above finite volumes by roughly 10%); the
   cause of the last 10% is not resolved.

## What this work cannot show, and why

- **BD in a 2 µm volume cannot test "sleep pumps Aβ out".** At the assumed D (1.5×10⁻¹⁰ m²/s, an assumption) one simulated second costs
  about 15 GPU-hours at 10⁵ tracers; sleep acts over minutes to hours, so even many GPUs are short by 3–5 orders of magnitude.
- **It does not need to be real-time:** wall speeds (≈0.02 nm/s) are about 10⁹ times slower than diffusion across a gap, so the gap is always quasi-static.
- **Pumping needs solvent flow and non-reciprocal motion**, neither of which is in the current BD. The Péclet number from a volume change scales as
  r L²/D: negligible at 2 µm and about 1 near a millimetre (with the contested 60% expansion over an hour). A travelling wave of expansion across
  centimetres is a tissue-scale idea, not a pore-scale one.
- **Assumptions that are priors, not measurements:** the gap width `w`, D for Aβ, the "synapse" identification from geometry, and the crop's boundary effects.

## Next steps (in the order I would do them)

1. **Quick tissue-scale estimate.** Advection–diffusion with literature ranges for volume change rate, K and D_eff, to see whether Pe is ever large
   enough for clearance to change. This decides whether the next two steps are worth the effort.
2. **Close the pore-scale puzzle (item 5).** Record where tracers are in the geometry (sheets, tunnels, pools) per `w` and find which part carries
   the drop; check against finite volumes. Also close the last ~10% BD–finite-volume gap (BD timestep and wall treatment, finite volumes at
   finer grids).
3. **Closure parameters across crops.** ECS fraction, D_eff(w), exit time from a source to the crop boundary, on several tiles/crops without
   periodic wrapping.
4. **Pore-scale flow.** A lubrication or Stokes solver on the sheet network (permeability scales as w³, so it is very sensitive to `w`), giving K(w)
   and the flow driven by dw/dt.
5. **Tissue-scale continuum model** using steps 3–4: sources at synapses, a perivascular/CSF sink, sleep and wake `w(t)` schedules, with and
   without a phase-lagged wave. This is the model that can address the main question.
6. **Smaller items:** real synapse annotations in place of the geometric guess, small sinks with partial absorption, gating, more sites, and
   coupling `w` to osmotic water flux.

## What we hope to learn

- Whether the sleep-time change in ECS width can plausibly change Aβ clearance, by how much, and through which route
  (diffusion only, volume change, or flow).
- Which pore-scale parameters the tissue model is sensitive to, so that measurements and further EM analysis can target them.
- How much the answer depends on assumptions we cannot yet measure (the true gap width, D for Aβ, and the synapse identification).
- An honest bound: if the plausible flow is too small to matter, a clear negative result is as useful as a positive one.

## Where things are

| Topic | Location |
|---|---|
| Method write-up, benchmarks, sink and tile results | `docs/DIFFUSION.md` |
| Novelty, merit, risks and what would make it publishable | `docs/NOVELTY.md` |
| Interpolation, regularisation, ParaView export | `README.md` |
| BD, GPU port, sinks, synapses | `zmorph/bd.py`, `bd_gpu.py`, `sinks.py`, `synapse.py` |
| Scripts to rerun the studies | `examples/run_*.py`, `examples/bench_bd_gpu.py` |
| Cluster jobs (A40 benchmark, sweeps, CPU reference) | `slurm/*.sbatch` |
| Result files and figures | `results/` |
