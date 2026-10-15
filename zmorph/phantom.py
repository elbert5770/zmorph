"""Synthetic ground-truth tissue: space-filling partition of tubes, with branching,
fusing, tips that end/begin between sections, and (optionally) thin tilted neurites.

A cell is a list of capsule segments (a, b, ra, rb) in (z, y, x) voxel units. Every
voxel is labelled with the cell of smallest signed capsule distance, so cells tile
space exactly and touch along shared interfaces (including triple junctions).
"""
import numpy as np
import scipy.ndimage as ndi


def _seg_dist(P, a, b, ra, rb):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    closest = a + t[:, None] * ab
    r = ra + (rb - ra) * t
    return np.linalg.norm(P - closest, axis=1) - r


def build(cells, shape, bg=5.0):
    nz, ny, nx = shape
    Z, Y, X = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    P = np.stack([Z.ravel(), Y.ravel(), X.ravel()], 1).astype(float)
    best = np.full(P.shape[0], bg)
    lab = np.zeros(P.shape[0], np.int32)
    for i, segs in enumerate(cells, start=1):
        d = np.min([_seg_dist(P, *s) for s in segs], axis=0)
        upd = d < best
        best[upd] = d[upd]
        lab[upd] = i
    return lab.reshape(shape)


NAMES = {1: "A arch (Fig.6 branching)", 2: "B tip under arch", 3: "Y branch",
         4: "C", 5: "D", 6: "F appears", 7: "G"}


def scene_branch(nz=31):
    """Cell 1 is an arch: two legs joined by a bridge. Section z=15 has two components,
    z=20 one (the Figure-6 situation). Cell 2 rises into the saddle and ends below the bridge.
    Cell 3 is a Y whose daughters separate below the junction. Cell 6 appears mid-gap."""
    ny, nx = 96, 128
    cells = [
        # A: legs + bridge (z, y, x)
        [((0, 48, 24), (19.5, 48, 24), 6.5, 6.5), ((0, 48, 56), (19.5, 48, 56), 6.5, 6.5),
         ((19.5, 48, 24), (19.5, 48, 56), 3.5, 3.5)],
        # B: tip in the saddle
        [((0, 48, 40), (15.5, 48, 40), 5.0, 4.0)],
        # Y: trunk + two daughters
        [((31, 48, 100), (19.5, 48, 100), 3.8, 3.8), ((19.5, 48, 100), (3, 48, 76), 3.8, 3.8),
         ((19.5, 48, 100), (3, 48, 124), 3.8, 3.8)],
        # C, D: moderately tilted fat tubes
        [((0, 15, 8), (31, 22, 24), 9.0, 9.0)],
        [((0, 80, 70), (31, 70, 82), 8.0, 8.0)],
        # F: appears between sections 10 and 15
        [((12, 20, 60), (31, 20, 60), 1.0, 6.5)],
        # G: filler on the right-hand side
        [((0, 85, 112), (31, 80, 108), 8.0, 8.0)],
    ]
    return build(cells, (nz, ny, nx))


def scene_tilted(nz=31):
    """Thin neurites that move farther than their diameter between sections (no overlap)."""
    ny, nx = 96, 128
    cells = [
        [((0, 25, 10), (30, 25, 100), 3.5, 3.5)],    # 3 vox/z -> 15 px per gap, diameter ~10
        [((0, 45, 20), (30, 45, 110), 3.5, 3.5)],
        [((0, 65, 110), (30, 65, 20), 3.5, 3.5)],    # opposite direction
        [((0, 85, 64), (30, 85, 64), 8.0, 8.0)],
    ]
    return build(cells, (nz, ny, nx), bg=1.5)


def corrupt(sections, rng, p_flip=0.35, rounds=2, n_iso=12, bad=None):
    """Segmentation-like noise on the observed sections: jagged boundaries, isolated wrong
    pixels (the 'spikes' of the manuscript) and, optionally, one badly imputed section."""
    out = []
    for si, s in enumerate(sections):
        s = s.copy()
        strong = bad is not None and si == bad
        pf = 0.6 if strong else p_flip
        for _ in range(rounds):
            nb = [np.roll(s, sh, ax) for ax in (0, 1) for sh in (1, -1)]
            diff = np.zeros(s.shape, bool)
            for n in nb:
                diff |= n != s
            flip = diff & (rng.random(s.shape) < pf)
            pick = np.stack(nb)[rng.integers(0, 4, s.shape), np.arange(s.shape[0])[:, None],
                                np.arange(s.shape[1])[None, :]]
            s = np.where(flip, pick, s)
        for _ in range(n_iso * (3 if strong else 1)):
            y, x = rng.integers(3, s.shape[0] - 3), rng.integers(3, s.shape[1] - 3)
            win = s[max(0, y - 6):y + 7, max(0, x - 6):x + 7]
            vals = np.unique(win)
            v = rng.choice(vals)
            w = 1 if rng.random() < 0.6 else 2
            s[y:y + w, x:x + w] = v
        if strong:
            for _ in range(5):
                y, x = rng.integers(5, s.shape[0] - 8), rng.integers(5, s.shape[1] - 8)
                s[y:y + 5, x:x + 5] = rng.choice(np.unique(s[max(0, y - 8):y + 9, max(0, x - 8):x + 9]))
        out.append(s)
    return out
