"""Candidate synaptic clefts from a label volume, and release sources for Brownian dynamics (bd.py).

A cleft is taken to be a compact, flat contact patch between two different cells. This is a geometric guess: the
labels carry no PSD or vesicle information, so axon-dendrite contacts that are not synapses will also score. Real
annotations (positions from a synapse table) override the guess through `snap_annotations`. The detector is meant
to be refined later; the `Patch` it returns (centre, normal, radius, pair, face points) is the interface the rest of
the code uses.

  contact_patches(labels, h)                 connected contact patches per label pair, with shape features
  score_patches(patches, ...)                synapse-likeness in [0, 1] from area, roundness and flatness
  snap_annotations(patches, xyz_nm, ...)     patches nearest to given synapse positions
  release_positions(patch, m, h, w, n, ...)  start positions for a burst of n particles in the cleft

Units: nm; `labels` is a 3-D array on a grid with voxel size h (isotropic); axes (x, y, z) = array axes (0, 1, 2).
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class Patch:
    pair: tuple            # (label a, label b) with a < b
    points: np.ndarray     # (n, 3) face midpoints in nm
    center: np.ndarray     # centroid, nm
    normal: np.ndarray     # unit normal (smallest principal axis)
    radius: float          # disc-equivalent radius from the in-plane spread, nm
    area: float            # disc-equivalent area, nm^2 (robust to voxel staircasing)
    face_area: float       # raw count of faces x h^2, nm^2
    aspect: float          # in-plane minor/major axis ratio (1 = round)
    flatness: float        # out-of-plane spread / in-plane radius (0 = flat)
    score: float = 0.0
    meta: dict = field(default_factory=dict)


def _faces(labels, h):
    """Face midpoints (nm) and label pairs for every 6-neighbour label change with both labels nonzero.
    Grid node (i, j, k) sits at (i, j, k) * h, the convention of bd.py."""
    pts, keys = [], []
    for ax in range(3):
        lo = labels[tuple(slice(None, -1) if k == ax else slice(None) for k in range(3))]
        hi = labels[tuple(slice(1, None) if k == ax else slice(None) for k in range(3))]
        ii = np.nonzero((lo != hi) & (lo != 0) & (hi != 0))
        if len(ii[0]) == 0:
            continue
        p = np.stack(ii, 1).astype(float); p[:, ax] += 0.5
        la, lb = lo[ii], hi[ii]
        pts.append(p * h); keys.append(np.stack([np.minimum(la, lb), np.maximum(la, lb)], 1))
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 2), labels.dtype)
    return np.concatenate(pts), np.concatenate(keys)


def _cluster(pts, link):
    tree = cKDTree(pts)
    pr = tree.query_pairs(link, output_type="ndarray")
    n = len(pts)
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)[1]


def contact_patches(labels, h=8.0, link=None, min_faces=8):
    """Connected contact patches between every pair of touching labels. Faces closer than `link` (default 1.8 h)
    belong to the same patch. Patches below `min_faces` faces are dropped."""
    link = 1.8 * h if link is None else link
    P, K = _faces(labels, h)
    if len(P) == 0:
        return []
    _, inv = np.unique(K, axis=0, return_inverse=True)
    inv = inv.ravel(); out = []
    for g in np.unique(inv):
        sel = np.nonzero(inv == g)[0]
        if len(sel) < min_faces:
            continue
        pts = P[sel]; comp = _cluster(pts, link)
        for c in np.unique(comp):
            q = pts[comp == c]
            if len(q) < min_faces:
                continue
            out.append(_describe(q, (int(K[sel[0], 0]), int(K[sel[0], 1])), h))
    return out


def _describe(q, pair, h):
    c = q.mean(0)
    ev, evec = np.linalg.eigh(np.cov((q - c).T) + 1e-9 * np.eye(3))      # ascending
    s3, s2, s1 = np.sqrt(np.maximum(ev, 0))
    R = 2.0 * np.sqrt(s1 * s2) if s2 > 0 else 2.0 * s1                   # uniform disc: sigma = R/2
    n = evec[:, 0] / np.linalg.norm(evec[:, 0])
    return Patch(pair=pair, points=q, center=c, normal=n, radius=float(R), area=float(np.pi * 4 * s1 * s2),
                 face_area=float(len(q) * h * h), aspect=float(s2 / s1) if s1 > 0 else 0.0,
                 flatness=float(s3 / max(R, 1e-9)))


def score_patches(patches, area_range=(2e4, 3e5), min_aspect=0.4, max_flatness=0.35):
    """Synapse-likeness in [0, 1]: product of a window on disc area (nm^2; default 0.02-0.3 um^2, roughly one PSD),
    a roundness term and a flatness term. Sets `.score` on each patch and returns the list sorted high to low."""
    lo, hi = area_range
    for p in patches:
        a = np.log(max(p.area, 1.0))
        s_area = float(np.exp(-0.5 * ((a - 0.5 * (np.log(lo) + np.log(hi))) / (0.5 * (np.log(hi) - np.log(lo)))) ** 2))
        s_round = float(np.clip(p.aspect / min_aspect, 0, 1))
        s_flat = float(np.clip(1.0 - p.flatness / max_flatness, 0, 1))
        p.score = s_area * s_round * s_flat
    return sorted(patches, key=lambda p: -p.score)


def snap_annotations(patches, xyz_nm, max_dist=300.0):
    """For each annotated synapse position, the nearest patch (by distance to its face points) within max_dist,
    else None. Returns a list aligned with xyz_nm."""
    trees = [cKDTree(p.points) for p in patches]
    out = []
    for x in np.atleast_2d(xyz_nm):
        d = [t.query(x)[0] for t in trees]
        i = int(np.argmin(d)) if d else -1
        out.append(patches[i] if i >= 0 and d[i] <= max_dist else None)
    return out


def release_positions(patch, m, h, w, n, reach=None, seed=0):
    """n start positions in nm (divide by h for bd.simulate_dynamic, whose x0 is in voxel units), uniform over ECS voxels { m < w } within `reach` (default max(w, 2h)) of the patch
    faces, jittered by up to half a voxel. A burst release; a release time course is a matter of when these start."""
    rng = np.random.default_rng(seed)
    reach = max(w, 2 * h) if reach is None else reach
    lo = np.maximum(np.floor((patch.points.min(0) - reach) / h).astype(int), 0)
    hi = np.minimum(np.ceil((patch.points.max(0) + reach) / h).astype(int) + 1, m.shape)
    g = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
    g = g[m[tuple(g.T)] < w]
    d = cKDTree(patch.points).query(g * h)[0]
    g = g[d <= reach]
    if len(g) == 0:
        raise ValueError("no ECS voxels near this patch; increase w or reach")
    pick = g[rng.integers(0, len(g), n)]
    x = (pick + rng.random((n, 3)) - 0.5) * h
    return np.clip(x, 0, (np.array(m.shape) - 1) * h)
