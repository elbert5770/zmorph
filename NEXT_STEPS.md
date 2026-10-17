# Next steps

State at the end of this session: `main` has the SDF method, tube links, `heat2`, the fusion modes, overlap-tile processing, the
synthetic benchmarks and one real MICrONS triplet (17414-17416). Everything is described in `docs/METHOD.md`; all 37 tests pass.
Evidence is thin (my phantoms, one real tile), so most items below are about getting more of it.

## Highest value
1. **Measure prevalence on more real data.** Run `examples/scan_hard_cases.py` over many section pairs and tiles (with spines and
   branching). If labels with no overlap across the gap are about 1% as in the first tile, tube links and the heat guard are
   marginal; if not, they matter. This decides whether the thin-neurite work is a publication or a footnote.
2. **More real held-out triplets from other tiles** (the second 17424-17426 triplet was not added to the repo), ideally with the list
   of sections the segmentation network imputed (the manuscript blames them for the worst artefacts).
3. **Re-run the original pipeline for all cells.** The supplied `zframe_87071-74` morphed only three cells; confirm the version and
   settings, then compare its full output with `heat2` and the SDF method. Pin the source frames in the mode filter (restore every
   5th frame after each pass; tested in Python, the Julia version is untested).

## Method
4. **Per-label method choice** from overlap statistics (high overlap: SDF; none: tube links, else `heat2`/guard); not built.
5. **Unresolved forks:** decompose a union-of-ellipses footprint into several tube sections; test curved and tapering tubes and
   noisy footprints; use three or more sections to fix the sign of the tilt.
6. **Higher-order SDF interpolation with noise-robust fields and translation** (the published cubic/4-point methods win on clean
   data, lose under noise); untested combination.
7. **Global normalisation for the rank-based methods** so that neighbouring tiles agree (they differ in 0.6-1.5% of pixels).
8. **Edges:** test the pipeline's real 1-pixel tile overlap and the volume border; use a halo of at least max(24 px, translation window).

## Engineering
9. Bounding-box cropping and a tiled, parallel driver for full volumes; the Julia pipeline could call `reconstruct_tiled`.
10. A first learned baseline: a small supervised U-Net on triplets (no diffusion), conditioned on the heat/SDF fields and, if available,
    the EM; only then ask whether a diffusion model is needed (`docs/METHOD.md` section 9.4).
11. Housekeeping: confirm the data terms for `data/micron_17414_17416`, add a CI workflow (needs the GitHub token to allow workflows),
    trim the `docs/METHOD.md` tables into a short paper-style summary.

## Cautions
- Phantom results (tube links, `sdf_rank`) are model-matched; do not quote them as validation.
- On the real tile the differences among SDF, `field`, `rank_avg` and translation are a few tenths of a point on one triplet.
- The `heat` baseline is my port of the published algorithm, not the Julia code.


## Added 2026-10: extracellular-space (gap) work

See `docs/DIFFUSION.md`. Diffusion benchmark done on phantoms. Moving walls, wall force and absorbing patches are done (`results/bd_dynamic.md`). Next: compartments behind the patches and gating, reactive species, flow (Stokes/lubrication
on the interface network), electrical coupling (Poisson in the gap), `w(t)` from osmotic water flux, and a run on the real tiles
(ECS fraction and tortuosity versus `w`). Open modelling choices: gap width prior, uniform vs variable `w`, matrix (Brinkman) vs free fluid.
