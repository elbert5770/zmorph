# zmorph

Smooth interpolation of z-stacked multi-label images (for example EM segmentations at 8×8×40 nm voxels)
onto an isotropic grid, plus cell surface meshes whose shared interfaces coincide.

Prototype. In this repository it is evaluated on synthetic ground truth only (see *Results* and *Limitations*).

## Idea

1. **One field per label, one competition.** Each label gets a sub-voxel signed-distance field on every
   section, smoothed in-plane (σ ≈ 1 px, which also removes small label noise), interpolated linearly in z.
   Every voxel takes the label with the highest value. The output is therefore a partition: no gaps, no
   overlaps, and no per-cell bookkeeping. Branching, fusion and cells that begin or end between sections
   follow from the interpolation instead of being tracked.
2. **Optional translation** for obliquely running neurites: an FFT cross-correlation shift per label is
   applied (both fields move towards each other) only if it clearly raises overlap.
3. **Conforming meshes.** Each cell is meshed from the zero set of `m_k = f_k − max_{j≠k} f_j`. Because
   `m_k = −m_j` on an interface, neighbouring cells get identical vertices. An optional joint Taubin pass on
   the welded mesh keeps them identical and fixes sliver triangles.

Only 2-D distance transforms, Gaussian filters and marching cubes are used: seconds on the test volumes.

## Install and use

    pip install -e ".[test]"

```python
import numpy as np
from zmorph import reconstruct
from zmorph.mesh import competition_meshes, taubin_joint

# sections: list of 2-D integer label images, one per section (any integer IDs, int64 is kept)
vol = reconstruct(sections, factor=5, method="sdf")         # shape ((S-1)*5+1, ny, nx)
vol = reconstruct(sections, factor=5, method="sdf", align=True)   # optional translation
labels, meshes = competition_meshes(sections, factor=5)     # {label: (vertices, faces)}, (z, y, x) voxel units
meshes = taubin_joint(meshes)                               # optional joint smoothing
```

Other methods for comparison: `nearest`, `sdf0` (classic binary SDF), `harmonic` (steady-state heat field),
`heat` (a port of the algorithm of Elbert, *A heat method to interpolate z-stacked cell images that preserves
interfaces without explicit identification of object boundaries*, bioRxiv 2025.04.11.648268).

## Reproduce the synthetic benchmarks

    python run_benchmark.py --seeds 5   # results/benchmark.md
    python run_sweeps.py                # results/sweeps.md   (shift, sigma/absent, noise level)
    python make_figures.py              # results/fig6_*.png  (Figure-6-style branching arch)
    python run_mesh_demo.py             # results/mesh.md, results/mesh_arch.png
    python -m pytest tests

The phantom (`zmorph/phantom.py`) is a space-filling partition of tubes with a branching/fusing arch
(two components at one section, one at the next), a tip that ends between sections, a Y-branch, a cell that
appears mid-gap, and a variant with thin neurites that move farther than their diameter between sections.
Noise on the observed sections: jagged boundaries, isolated wrong pixels, one badly imputed section.

## Results (synthetic, mean of 5 noise seeds; Dice over intermediate slices / mean interface distance in px)

| scene | nearest | port + majority filter | sdf | sdf + translation |
|---|---|---|---|---|
| arch, clean | 0.962 / 0.40 | 0.968 / 0.30 | **0.983 / 0.18** | 0.983 / 0.18 |
| arch, noisy | 0.938 / 0.60 | 0.962 / 0.38 | **0.970 / 0.30** | 0.970 / 0.30 |
| tilted thin neurites, clean | 0.880 / 0.81 | 0.922 / 0.52 | 0.883 / 0.71 | **0.938 / 0.39** |
| tilted thin neurites, noisy | 0.850 / 0.95 | 0.912 / 0.58 | 0.869 / 0.80 | **0.921 / 0.50** |

* Plain SDF is best on the branching arch but falls behind the heat-path port on strongly tilted thin neurites;
  the translation step fixes that case.
* Across lateral shifts of 0–24 px per section gap, SDF + translation is best throughout (`results/sweeps.md`).
* σ = 1–1.5 px is a plateau; the absent-label depth parameter is insensitive.
* Meshes (`results/mesh.md`): 96% of interface vertices are shared exactly, 95th-percentile interface mismatch
  0.000 voxel (independently meshed smoothed indicators: 0.23 voxel). Raw competition meshes have poor
  triangles (5th-percentile minimum angle 7.4°); joint Taubin raises it to 34.5° and keeps the interfaces shared.

## Limitations

* **Synthetic validation only here.** The phantom is tubes and arches with my noise model. Differences between
  methods are a few percent Dice; real segmentations will differ.
* The `heat` method is a port of the published algorithm with a closed-form (DCT) heat solve, a plain 3×3×3
  majority filter in place of its rule-based mode filter, and gradients taken everywhere as the paper describes.
  It is a baseline, not the original implementation.
* The z-aware cleaning step (`clean=True`) showed no measurable benefit on this noise; in-plane SDF smoothing
  already removes small islands.
* σ smoothing rounds sharp corners and will thin features narrower than about 2σ.
* The translation is one rigid shift per label: it does not handle a label whose parts move in different
  directions.
* Known artefact: thin, shifting structures can bulge in the middle frames (linear SDF interpolation).
* Not implemented: tiling with halos / bounding-box cropping for very large volumes, use of raw EM intensities
  as an edge prior.

## License

MIT (see `LICENSE`).
