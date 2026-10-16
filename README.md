# zmorph

Smooth interpolation of z-stacked multi-label images (for example EM segmentations at 8×8×40 nm voxels)
onto an isotropic grid, plus cell surface meshes whose shared interfaces coincide.

Prototype. Evaluated on synthetic ground truth and on one real triplet of MICrONS sections (see *Results*, *Real-data example* and *Limitations*).

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

A fuller description (equations, parameters, baselines, evaluation protocol, failure modes) is in
[`docs/METHOD.md`](docs/METHOD.md).

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
    python run_published_comparison.py  # results/published_sdf.md (published SDF methods, ablations)
    python run_split_experiment.py      # results/split3.md (neurite splitting into three branches)
    python run_split_sweep.py           # results/split3_sweep.md (tube-link robustness)
    python make_scene_figures.py        # results/tilted_scene.png, split3_scene.png, split3_methods.png
    python make_figures.py              # results/fig6_*.png  (Figure-6-style branching arch)
    python run_mesh_demo.py             # results/mesh.md, results/mesh_arch.png
    python examples/micron_17414_17416.py   # real-data example, results/micron_17414_17416.*
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
* Across lateral shifts of 0–24 px per section gap, SDF + translation is best among the two-section methods compared in
  that sweep (`results/sweeps.md`, which does not include the higher-order methods below).
* σ = 1–1.5 px is a plateau; the absent-label depth parameter is insensitive.
* Meshes (`results/mesh.md`): 96% of interface vertices are shared exactly, 95th-percentile interface mismatch
  0.000 voxel (independently meshed smoothed indicators: 0.23 voxel). Raw competition meshes have poor
  triangles (5th-percentile minimum angle 7.4°); joint Taubin raises it to 34.5° and keeps the interfaces shared.

### Against published SDF interpolation methods (`docs/METHOD.md` §7, `results/published_sdf.md`)

Raya & Udupa / Herman et al. (linear), higher-order (cubic-spline) SDF interpolation and 4-point weights on SDFs
(a stand-in for Kels & Dyn, whose set average is not an SDF average) were implemented as baselines
(`run_published_comparison.py`). Summary: the proposed method is the better **two-section** method under noise, but
it is **not uniformly better**. Higher-order interpolation from four or more sections is better on clean data and on
thin tilted neurites (Dice 0.947 vs 0.883) and worse under noise. On the real sections the proposed field construction
changes accuracy by less than 0.3 points compared with the textbook linear SDF; the translation step adds 0.24 points.

### Stress test: a thin neurite splitting into three non-overlapping branches (`docs/METHOD.md` §8)

`phantom.scene_split3`, images in `results/split3_scene.png` and `results/split3_methods.png` (tilted neurites:
`results/tilted_scene.png`). No daughter overlaps the parent's section footprint, and no post-filter is used. **All
SDF-based methods delete the neurite in the gap** (the proposed method scores gap Dice 0.000, classic linear SDF 0.085);
only the raw heat-path port beats repeating the nearest section (0.628 vs 0.590), with 6-9 spurious fragments.
**Tube links** (`tube=True`) infer each footprint's tilt from its elongation, test that the axes close on the partner
footprint, and render the intermediate slices as the tube's cross-section: gap Dice **0.927** (0.888 noisy), the three
branches and the junction are reproduced, and the output is bit-identical to the baseline where no link is accepted. They
work when the daughters are resolved ellipses at the upper section and fall back to a persistence guard (`persist=True`,
about nearest-section level) when the junction is mid-gap. **Caveats:** the phantom is built from the geometry the method
assumes, and on the real MICrONS tile no link is accepted (the situation is rare there at the 40 nm gap), so there is no
real-data evidence yet (`examples/scan_hard_cases.py` measures prevalence on more data).

## Real-data example: MICrONS sections 17414–17416

Held-out test on three consecutive 40 nm sections (`data/micron_17414_17416/`, see its README for provenance and
formats): the middle section is predicted from its two neighbours, an 80 nm gap (twice the usual 40 nm), and scored
against the real middle section. `python examples/micron_17414_17416.py` (about 3 minutes) writes
`results/micron_17414_17416.{md,json,png}`.

| method | pixel accuracy | accuracy where the inputs differ | interface distance (px) | mean Dice |
|---|---|---|---|---|
| nearest section (lower / upper) | 0.837 / 0.845 | 0.46 / 0.49 | 3.5 / 3.1 | 0.761 / 0.765 |
| heat-paths port + majority filter | 0.921 | 0.742 | 1.65 | 0.888 |
| harmonic | 0.933 | 0.782 | 1.44 | 0.901 |
| sdf0 (classic binary SDF) | 0.940 | 0.806 | 1.35 | 0.914 |
| sdf (σ=1) | 0.940 | 0.804 | 1.30 | 0.914 |
| sdf (σ=1) + translation | **0.942** | **0.812** | **1.27** | **0.922** |

Smoothed inputs, smoothed truth. With **raw** inputs and raw truth (older 500×500 export of 17415) the ranking is the same:
heat-paths + majority 0.921, harmonic 0.934, classic SDF 0.944, sdf σ=1 0.943, sdf σ=1 + translation **0.946**
(nearest 0.837 / 0.846). `results/micron_17414_17416.png` shows the region of largest change: wrong pixels cover
34% of that crop for nearest, 16% for the heat-paths port + majority filter and 11% for sdf + translation.

* The original mode filter changes only 0.37% / 0.47% of the pixels of 17414 / 17416 (median changed blob 1–2 px), and
  SDF smoothing agrees with it equally well for σ between 0.5 and 2: on these sections z-interpolation, not
  denoising, drives accuracy.
* One triplet in one tile is indicative only. The translation step adds about +0.2 points here; on the synthetic
  data it matters mainly for strongly tilted neurites, so treat it as optional.

## Limitations

* **Validation is thin.** The synthetic phantom is tubes and arches with my noise model, and the real example is a
  single triplet from one tile. Differences between methods are a few percent Dice.
* The `heat` method is a port of the published algorithm with a closed-form (DCT) heat solve, a plain 3×3×3
  majority filter in place of its rule-based mode filter, and gradients taken everywhere as the paper describes.
  It is a baseline, not the original implementation.
* The z-aware cleaning step (`clean=True`) showed no measurable benefit on this noise; in-plane SDF smoothing
  already removes small islands.
* σ smoothing rounds sharp corners and will thin features narrower than about 2σ.
* The translation is one rigid shift per label: it does not handle a label whose parts move in different
  directions.
* Possible artefact: thin, shifting structures may bulge in the middle frames under linear SDF interpolation (seen in
  an exploratory check not included here; see `docs/METHOD.md`, section 9).
* Not implemented: tiling with halos / bounding-box cropping for very large volumes, use of raw EM intensities
  as an edge prior.

## License

MIT (see `LICENSE`).
