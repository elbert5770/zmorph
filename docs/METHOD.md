# zmorph: method description

This note describes what the code in `zmorph/` does and why, how it is evaluated, and where it is known to fail.
Numbers quoted here are measured in this repository; the tables they come from are in `results/`.

## 1. Problem

Input: a stack of 2-D integer label images (one segment ID per pixel), spaced farther apart in z than in x and y.
For MICrONS the voxels are 8 × 8 × 40 nm, so a single z-step covers 5 pixels. Output: a label volume on an isotropic
grid (`factor` new z-slices per section gap, 5 for MICrONS, so each z-step is 8 nm), and per-cell surface meshes.

Requirements that drove the design:

1. **Partition.** Every output voxel carries exactly one label: no gaps, no overlaps between cells.
2. **Topology change.** Branching (one region in one section, two in the next), fusion, and cells that begin or end
   between sections must be handled by the method itself, without correspondence bookkeeping.
3. **Noise.** Segmentation boundaries are jagged and contain small wrong islands.
4. **Conforming meshes.** Neighbouring cells should share their interface exactly, so the extracellular space and
   cell–cell contacts stay consistent in downstream simulations.

## 2. Interpolation by field competition

For each pair of adjacent sections `(s0, s1)` and every label `k` present in either section:

1. **Signed distance.** `f(x)` is positive inside `k`, negative outside, and its zero level lies half a pixel outside
   the last inside pixel (`sdf_subvoxel`: `edt(inside) − 0.5` inside, `−(edt(outside) − 0.5)` outside, with exact
   Euclidean distance transforms). A label absent from one section gets the constant field `−absent`
   (default 4 voxels) there.
2. **Smoothing.** The *field* (not the mask) is Gaussian-smoothed in plane (`sigma`, default 1 pixel) and clipped to
   ±`cap` (default 12). Smoothing a distance field moves its zero level as a curvature-driven flow: small islands and
   one-pixel jaggedness disappear, large smooth boundaries barely move. This is the only denoising step by default.
3. **Interpolation.** For `t = j / factor`, `f_k(t) = (1 − t) · f_k(s0) + t · f_k(s1)`.
4. **Competition.** Each voxel takes `argmax_k f_k(t)`. The labels at `t = 0` and `t = 1` are the (slightly denoised)
   observed sections themselves.

Because every voxel is assigned to exactly one label, requirement 1 holds by construction.

**Why topology changes work.** Take a pixel that carries label A in the lower section and B in the upper one, with
`a` = depth inside A in the lower section and `b` = distance to A in the upper section. In the two-label case
`f_A(t) = (1 − t)·a − t·b = −f_B(t)`, so the pixel switches from A to B at `t = a / (a + b)`: pixels near the
moving interface switch first and the interface sweeps through the gap. This relative position between the two interfaces
plays a role similar to the normalised path-length fraction of the heat-path method of the manuscript, but it needs no
overlap region, no paths and no per-cell solve (the equivalence is not claimed to be exact). Branching and fusion need no special case because linear interpolation of distance fields changes topology
smoothly where the zero level set merges or splits. A cell present in only one section is eroded away over the gap
at a rate set by `absent`; a cell that appears grows from its deepest point outward.

**Cost.** One pair of 2-D distance transforms and Gaussian filters per label and section pair, plus an argmax update
per output slice. The test volumes run in seconds. The current implementation works on full frames; bounding-box
cropping and tiling with halos are not implemented.

## 3. Optional translation (`align=True`)

Linear interpolation of distance fields has no notion of motion: a neurite that shifts sideways by more than its width
between sections fades out and a copy fades in. For each label present in both sections, `_estimate_shift` finds the
integer translation `d` (within ±24 px) that maximises the FFT cross-correlation of the two Gaussian-smoothed (σ = 2)
masks, and accepts it only if shifting raises the intersection-over-union by more than 0.2. If accepted, both fields
move toward each other: `f_k(t) = (1 − t)·shift(f_k(s0), t·d) + t·shift(f_k(s1), −(1 − t)·d)` (linear
resampling). The competition is unchanged, so the result is still a partition.

This is one rigid shift per label. It does not handle a label whose components move in different directions, and the
gain is not reliable: large on synthetic strongly tilted neurites, small (+0.2 points of accuracy) on the real
example, so it is off by default.

## 4. Conforming meshes

For cell `k` define the **competition field** `m_k = f_k − max_{j≠k} f_j`. It is positive where `k` wins, negative
elsewhere, and on an interface between cells `j` and `k` (where each is the other's runner-up) `m_k = −m_j`.
`competition_meshes` runs marching cubes at level 0 on `m_k` per cell, using the winner and runner-up values tracked
during the competition. Marching cubes places a vertex on a grid edge at the position determined by the two end
values, so the two neighbours compute the same vertex from opposite signs: identical vertices on shared interfaces,
at sub-voxel positions, without any mesh smoothing. Only near triple junctions, where the runner-up changes along an
edge, can the two sides differ.

The raw meshes inherit marching-cubes sliver triangles (5th-percentile minimum angle 7.4° on the phantom). `taubin_joint`
welds identical vertices across cells (rounding to 1e-4), applies Taubin smoothing (λ = 0.5, μ = −0.53, uniform Laplacian,
15 iterations) on the union, and returns per-cell meshes. Shared vertices are one vertex, so they move together and
conformity is preserved by construction. Measured effect is in `results/mesh.md`.

## 5. Methods compared

| name (`reconstruct(..., method=)`) | what it is |
|---|---|
| `nearest` | repeat the closer section; what stair-stepped meshes come from |
| `sdf0` | classic shape-based interpolation: binary signed distance (no half-pixel offset, no smoothing), linear in z, argmax across labels |
| `sdf` | the proposed method of section 2 (`align=True` adds section 3) |
| `harmonic` | per label, the steady-state heat (harmonic) field between the two sections, with lateral no-flux boundaries, solved in closed form with a 2-D DCT (continuous in z over a gap `L`, default `factor`), then argmax. This is the same kind of field the manuscript uses, without path tracing. Note it cannot reproduce two identical sections exactly: lateral modes decay toward the mid-plane, so thin features blur |
| `heat` | port of the published algorithm (Elbert, bioRxiv 2025.04.11.648268): per-cell 3-D heat solve with 25 layers and observation layer 7, gradient ascent from pixels unique to the upper section until the nearest pixel is in the overlap, path lengths normalised by the longest path per terminal pixel, assignment in frame `j` if `ceil(5·fraction) ≤ j` |

Differences between `heat` and the published implementation: the 3-D solve is done in closed form (DCT) instead of
ILU-conjugate-gradient; gradients are taken everywhere, as the paper describes, rather than only on the morphing
region; a cell with no overlap uses its temperature maximum as the sink; and `majority` (a plain 3 × 3 × 3 majority
filter) replaces the manuscript's rule-based mode filter. It is a baseline, not the original code.

`clean_sections` (`clean=True`) removes 2-D fragments smaller than `min_area` (12 px) that are not supported by the
same label in an adjacent section, filling them from the nearest surviving pixel. It showed no measurable benefit
here, because the field smoothing already removes small islands.

## 6. Evaluation

### Synthetic phantom (`zmorph/phantom.py`, `run_benchmark.py`, `run_sweeps.py`)

A space-filling partition of capsule-shaped cells in a 31 × 96 × 128 voxel volume (isotropic ground truth,
sections every 5 slices). A voxel takes the label of the cell with the smallest signed capsule distance, limited to
a few voxels beyond the cell surface, so cells touch along shared interfaces including triple junctions.
`scene_branch` contains the Figure-6 situation (an arch whose two legs form two components at one section and one
at the next, with a tip cell rising into the saddle), a Y-branch, a cell that appears mid-gap, and tilted
filler cells. `scene_tilted` contains thin neurites that move farther than their diameter between sections.
`corrupt` adds noise to the observed sections only: boundary pixels flipped to a neighbour's label (p = 0.35, two
rounds), isolated wrong pixels, and optionally one badly imputed section with larger wrong blobs.

Metrics (`zmorph/metrics.py`), over intermediate slices only unless stated: pixel accuracy; mean per-label Dice
(label 0 excluded); mean symmetric distance between in-plane label interfaces (`bdist`); the Dice of the
branching cell in the gap that contains the junction (`dice_branch_gap`); a count of connected-component
differences from ground truth; accuracy on the observed sections against the clean truth (denoising); and the ratio
of interface face counts to ground truth (roughness).

Sweeps (`results/sweeps.md`): lateral shift per gap (0–24 px), σ and `absent`, and boundary-noise level.

### Real sections (`examples/micron_17414_17416.py`)

Held-out prediction: the middle of three consecutive sections is predicted from its two neighbours (an 80 nm gap, twice
the usual 40 nm; `factor = 2` and `t = 0.5`) and scored against the real middle section, using the helpers in
`examples/real_io.py`: pixel accuracy (also restricted to pixels where the two inputs differ), `bdist`, mean Dice
over labels with at least 300 px, and component-count difference. Run on smoothed sections with smoothed truth and
on raw sections with raw truth. The raw 17415 comes from an older export of 500 × 500 pixels, verified to be the
top-left crop of the 501 × 501 grid, so that test crops everything to 500 × 500. See
`data/micron_17414_17416/README.md`.

## 7. Known limitations and open questions

* **Thin, shifting structures can bulge in the middle frames** under linear interpolation of distance fields: in an
  exploratory frame-by-frame area check (not included in this repository), a thin diagonal band grew about 3% above its
  end value at an intermediate frame. The cause has not been diagnosed and the check has not been repeated on reference
  frames of known provenance.
* **Translation is one rigid shift per label**, and its benefit is inconsistent on real data.
* **Smoothing** rounds sharp corners (about 0.02% of pixels on clean phantom sections) and will thin features narrower
  than about 2σ. σ between 0.5 and 2 px made little difference on the real example.
* **Absent-label depth** (`absent`) is a free parameter; the phantom sweep was insensitive to it, but it controls
  how fast cells that exist in only one section vanish or appear.
* **Raw meshes have sliver triangles** until `taubin_joint` is applied; joint smoothing slightly moves interface
  positions (by design it moves shared vertices together).
* **No ground truth at the 40 nm scale.** Real-data evaluation here is an 80 nm held-out section. It tests the
  midpoint only, not the four intermediate frames.
* **Validation is thin:** one synthetic phantom family and one real triplet from a single tile. The raw EM
  intensities are not used.
* **Not implemented:** tiling with halos and bounding-box cropping for very large volumes; use of raw EM as an edge
  prior; a per-component rather than per-label translation.
