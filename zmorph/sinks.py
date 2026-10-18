"""Absorbing sink membranes for the Brownian dynamics (bd.simulate_dynamic).

A sink is (part of) the membrane of one or more cells. The ECS { m < w } is centred on each cell-cell interface, so
its wall on a cell's side lies inside that cell, at depth w/2. The patch map is therefore the label volume itself:
patch id 1 on the voxels of the sink cell(s), 0 elsewhere. The BD kernel looks the map up at the wall-hit point, so
only the sink cell's side of the gap absorbs; the opposite wall reflects.

  sink_patch_map(vol, labels)        int32 map (1 on the sink cells)
  absorption_probability(kappa, dt)  per-hit probability p = kappa sqrt(pi dt / D) for a reaction-rate constant kappa
  pick_sink(vol, h, source, ...)     a sink cell far from the source, with large surface in contact with other cells
  first_passage(res)                 absorbed fraction versus time and the first-passage-time summary of a run
  disc_patch_map(vol, m, h, site, R) a disc of the membrane (radius R in the interface plane) on the cell under `site`

Units: nm, D = 1, time nm^2/D.
"""
import math
import numpy as np


def sink_patch_map(vol, labels):
    return np.isin(vol, np.atleast_1d(labels)).astype(np.int32)


def absorption_probability(kappa, dt, D=1.0):
    """Per-hit absorption probability of a boundary with reaction rate kappa (nm per unit time, flux = kappa c)."""
    p = kappa * math.sqrt(math.pi * dt / D)
    if p > 1.0:
        raise ValueError(f"kappa too large for dt: p = {p:.2f} > 1 (use perfect absorption, p = 1, or a smaller dt)")
    return p


def pick_sink(vol, h, source, min_dist=600.0, exclude=(), top=1):
    """Cells whose nearest point is at least `min_dist` nm from `source` (a point in nm), ranked by the area (nm^2) of
    their contacts with other cells (face count x h^2). Returns a list of (label, contact area, distance) tuples."""
    faces = {}
    for ax in range(3):
        lo = vol[tuple(slice(None, -1) if k == ax else slice(None) for k in range(3))]
        hi = vol[tuple(slice(1, None) if k == ax else slice(None) for k in range(3))]
        sel = (lo != hi) & (lo != 0) & (hi != 0)
        for lab in np.unique(np.concatenate([lo[sel], hi[sel]])):
            n = int(((lo == lab) & sel).sum() + ((hi == lab) & sel).sum())
            faces[int(lab)] = faces.get(int(lab), 0) + n
    out = []
    src = np.asarray(source, float)
    for lab, n in faces.items():
        if lab in exclude:
            continue
        ii = np.argwhere(vol == lab) * h
        d = float(np.linalg.norm(ii - src, axis=1).min())
        if d >= min_dist:
            out.append((lab, n * h * h, d))
    return sorted(out, key=lambda r: -r[1])[:top]


def first_passage(res, n_grid=60):
    """Summary of a simulate_dynamic run with absorbing patches: `absorbed_frac` (cumulative fraction absorbed at each
    record time), `frac_absorbed` (final), and the median, mean and 10% quantile of the absorption time over the tracers
    that were absorbed (None if none). The time statistics are censored: read them together with `frac_absorbed`."""
    t = res["t"]; got = res["absorbed"]
    at = res["abs_time"][got]
    frac = np.array([(at <= tt).mean() * got.mean() if got.any() else 0.0 for tt in t])
    return dict(t=t, absorbed_frac=frac, frac_absorbed=float(got.mean()),
                t_median=float(np.median(at)) if got.any() else None,
                t_mean=float(at.mean()) if got.any() else None,
                t_q10=float(np.quantile(at, 0.1)) if got.any() else None)


def disc_patch_map(vol, m, h, site, radius, depth=40.0, smooth=1.5):
    """Patch map (int32, 1 on the patch) for a disc of radius `radius` nm in the interface plane under `site` (nm, a point of the gap).

    Unlike a ball around the site, the disc does not shrink as the gap widens: the wall lies w/2 inside the cell, and
    a ball of fixed radius cuts a smaller disc there. The patch is the voxels of the cell at the site whose lateral
    distance (in the interface plane) from the site's projection onto the interface is at most `radius` and whose
    distance from the interface plane is at most `depth` (must exceed w/2 of the widest gap used). The interface plane
    and its normal come from the smoothed gradient of the margin m (m = 2 x distance to the interface).
    Returns (patch, label, normal, centre) with centre the projection of `site` onto the interface (nm)."""
    from scipy.ndimage import gaussian_filter
    site = np.asarray(site, float)
    idx = np.round(site / h).astype(int)
    r = int(np.ceil((radius + depth) / h)) + 1
    lo = np.maximum(idx - r, 0); hi = np.minimum(idx + r + 1, m.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    mloc = gaussian_filter(np.asarray(m[sl], float), smooth)
    g = np.array([np.gradient(mloc, h, axis=k)[tuple(idx - lo)] for k in range(3)])
    n = g / np.linalg.norm(g)
    msite = float(np.asarray(m[tuple(idx)], float))
    centre = site - 0.5 * msite * n                                   # m = 2 d: d = m / 2 along the gradient
    lab = vol[tuple(idx)]
    coords = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), -1) * h - centre
    along = coords @ n
    lateral = np.linalg.norm(coords - along[..., None] * n, axis=-1)
    patch = np.zeros(m.shape, np.int32)
    patch[sl] = ((vol[sl] == lab) & (lateral <= radius) & (np.abs(along) <= depth)).astype(np.int32)
    return patch, int(lab), n, centre
