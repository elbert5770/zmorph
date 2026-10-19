# Novelty and scientific merit

An assessment of the question and the approach, written to be useful to a sceptical reader. Status of the work: `docs/SUMMARY.md`;
methods and results: `docs/DIFFUSION.md`. I did only a light literature check (a handful of searches, abstracts and titles, not full texts), so
the "novel" items below are candidates to be confirmed by a proper review, not established claims. Statements from memory are marked.

## The question

Does the widening and narrowing of the extracellular space (ECS) during sleep change how fast Aβ released at synapses leaves the tissue?

**Why it matters.** Aβ clearance and sleep are linked to Alzheimer's risk, and the physical route for clearance is disputed. Xie et al. (2013) reported
an ECS increase of about 60% in sleep and faster clearance, and proposed convective exchange
([PubMed abstract](https://pubmed.ncbi.nlm.nih.gov/24199995/), [commentary](https://pmc.ncbi.nlm.nih.gov/articles/PMC3867735)).
Later work argued for diffusion rather than bulk flow in the parenchyma, e.g. Holter et al. (2017), who computed permeability in reconstructed
neuropil and found bulk flow negligible at physiological pressures
([PNAS record](https://pmc.ncbi.nlm.nih.gov/articles/5604020)), and Smith et al. (2017, eLife; I saw the title only). A quantitative, geometry-based
answer to "how much could wall motion matter?" is a sensible thing to want, whichever way it comes out.

**How well-posed is it?** As stated, only partly. The pore-scale geometry can say a lot about diffusion and, with a flow solver, permeability. Whether the
walls "pump" depends on solvent flow and on non-reciprocal wall motion over centimetres, which a 2 µm tissue block does not contain (`SUMMARY.md`, "What this work cannot show").
The well-posed pore-scale question is: *how do ECS fraction, effective diffusivity, permeability and source-to-boundary transport depend on the gap width and its change, in real cortical geometry?*
The answer then feeds a tissue-scale model, which is where the pumping question gets decided.

## What is already known (so we do not claim it)

- ECS in EM reconstructions is organised as thin sheets (10–40 nm) joined by tunnels (40–80 nm), and Monte Carlo diffusion in a reconstructed volume was already done
  by Kinney et al. (2013) on about 180 µm³ of rat hippocampal neuropil
  ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3540825)).
- Permeability from reconstructed ECS and the weakness of bulk flow: Holter et al. (2017), above.
- Diffusion in ECS geometries built from EM depends strongly on how the images are processed (noisy, frayed surfaces hinder diffusion much more than smooth ones):
  Hiroi et al. (2012, found in the search results; not read in full).
- Continuum electrodiffusion on dense EM-based cerebral reconstructions exists as a recent preprint
  ([arXiv:2512.03224](https://arxiv.org/pdf/2512.03224), title and abstract only).
- In vivo ECS fraction near 0.2 and tortuosity near 1.45 from small-molecule probes (quoted in the Kinney introduction, via the search result).

## What may be new (to be confirmed)

1. **An implicit, time-dependent gap built from label partitions.** The ECS is `{m < w(t)}` where `m` is the competition margin from the
   interpolation. There is no surface offset (so no self-intersection), the width is a tunable field, and the same object serves BD, finite volumes and visualisation.
   I did not find this formulation in the search results; a reader from the level-set or mesh-based diffusion literature may know prior art.
2. **Brownian dynamics on that geometry with moving walls and patches, validated** against exact or reference solutions (shell, planar gap, osmotic wall force, piston work,
   Robin absorption). Cross-checked on the CPU and GPU. This is engineering with care, which is useful but not a scientific result by itself.
3. **Scale: dense cortical segmentation with many cells.** The data are MICrONS-type sections (hundreds of labels in a 2 µm crop, a mm³ volume available), not one hippocampal block.
   That allows statistics across sites, cell pairs and, later, cell types and annotated synapses. I did not find a published particle-based study of ECS transport on MICrONS meshes
   (one search; absence of a hit is weak evidence).
4. **A methods result on resolution.** At 8 × 8 × 40 nm sampling the gap is below the voxel size; voxel solvers mis-size a 12–20 nm gap and raw voxel meshes carry
   staircase area that does not shrink with resolution, while the level-set approach and smoothed meshes do not. This is consistent with Hiroi et al. and is useful for anyone
   using dense EM segmentations for transport, but it is incremental.
5. **Bounds, including negative ones.** A clean statement that reciprocal breathing without flow does nothing measurable to capture, and a scaling argument (Pe ≈ r L²/D) for where advection could matter,
   would be a contribution if it is made carefully and with sourced inputs.
6. **A small dependence of clearance on gap width that we cannot yet explain.** It survives a patch-geometry control. If it holds up, it is a real finding about transport in
   sheet-and-tunnel networks, not just a check.

## Weaknesses and risks (what a reviewer will say)

- **The gap width is not in the data.** Labels give contact (zero-width) interfaces; `w` is a prior. Chemically fixed tissue is known to lose much of its ECS (from memory, not checked for this dataset),
  so the EM-derived ECS may not match the living one. Every ECS-dependent quantity is conditional on `w`. The only handle is calibration to an in vivo fraction.
- **Weak consistency check only.** Our ECS fraction at `w` = 20–40 nm brackets 0.2, and 1/λ² of about 0.4–0.5 implies λ of about 1.4–1.6, close to the in vivo values quoted above.
  But `w` is a free parameter, so this is consistency, not validation.
- **Aβ is assumed to behave as a free tracer.** D is an assumption, with no binding, aggregation, matrix interaction or charge. Synapses are identified from contact geometry, not annotation.
  Release is a single burst (a time course exists only as a convolution of it).
- **The question's key mechanism is outside the model.** No solvent flow, no non-reciprocal motion, a 2 µm domain. On our own estimate, advection at the pore scale is negligible
  (Pe ~ 10⁻⁵ at 2 µm), so the likely finding at this scale is "not by diffusion alone". That is informative, but it does not support or refute the sleep-pumping hypothesis on its own.
- **Contested premises.** The 60% sleep expansion and the glymphatic picture are debated, and I quote the first from memory; the sources I found do not settle either way.
- **Numerical open items.** About 10% of BD–finite-volume difference unexplained; the unexplained `w` dependence; one tile; periodic wrapping in some runs.
- **Incremental feel unless it feeds a model.** Without a flow solver and a tissue-scale model, the pore-scale part risks reading as a careful re-derivation of known qualitative facts
  (diffusion-dominated, sheet-and-tunnel structure) on a new dataset.

## What would make it a publishable contribution

1. A **flow solver** giving permeability K(w) in the same geometry and a **Pe estimate** with sourced inputs, so the diffusion-versus-flow statement is quantitative.
2. A **tissue-scale model** that takes α(w), D_eff(w) and K(w) from the pore scale and answers the sleep question with sensitivity ranges, including a phase-lagged wall wave.
3. **Calibration and validation**: set `w` from an in vivo ECS fraction, then compare tortuosity and, if available, measured Aβ or probe spread in tissue.
4. **Robustness**: several tiles/crops without periodic wrapping, sensitivity to the interpolation and regularisation choices, an explanation of the `w` dependence, and an uncertainty budget for `w`, D, and the synapse proxy.
5. **Real synapse annotations** and, if feasible, cell-type-resolved source and sink statistics, which is where a MICrONS-scale dataset offers something earlier studies lacked.

## Honest summary

- **Merit of the question:** high and current, but it is a multiscale question that pore-scale simulation can inform, not settle.
- **Merit of the approach:** sound and carefully validated tooling; a small number of potentially new methods points (implicit gap, level-set BD, scale of data); the scientific claims to date are modest and mostly confirm that
  transport at this scale is diffusion-dominated and weakly dependent on gap width.
- **Biggest lever:** turning pore-scale closure into a tissue-scale answer, with an explicit treatment of the unmeasured gap width. A well-supported negative result or a bound
  would be a legitimate outcome.

## References found in this session (abstracts or titles only unless noted)

- Xie et al. 2013, sleep and clearance: [PubMed](https://pubmed.ncbi.nlm.nih.gov/24199995/); commentary: [PMC3867735](https://pmc.ncbi.nlm.nih.gov/articles/PMC3867735).
- Holter et al. 2017, diffusion rather than bulk flow: [PMC5604020](https://pmc.ncbi.nlm.nih.gov/articles/5604020).
- Kinney et al. 2013, ECS sheets and tunnels: [PMC3540825](https://pmc.ncbi.nlm.nih.gov/articles/PMC3540825).
- Smith et al. 2017 (eLife), test of the glymphatic hypothesis: title only, not read.
- Hiroi et al. 2012, diffusion in TEM-reconstructed spaces: reported via search snippet, not read.
- Electrodiffusion in dense EM reconstructions: [arXiv:2512.03224](https://arxiv.org/pdf/2512.03224).
