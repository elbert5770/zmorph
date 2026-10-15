"""Z-interpolation of multi-label sections onto an isotropic grid.

All methods return a label volume of shape ((S-1)*factor+1, ny, nx) whose z index matches
the ground truth (section s sits at z = s*factor). Every method except 'nearest' picks, at
each voxel, the label with the highest interpolated field, so the result is a partition:
no gaps and no overlaps by construction.

Methods
  nearest   repeat the closer section (what stair-stepped meshes come from)
  sdf0      classic shape-based interpolation: per-label binary signed distance, linear in z
  sdf       proposed: cleaned sections, sub-voxel signed distance, in-plane smoothing,
            optional per-label translation for non-overlapping (tilted) neurites
  harmonic  per-label harmonic (steady-state heat / random-walker) field between the sections,
            solved in closed form with a 2-D DCT -- the manuscript's field, without path tracing
  heat      port of the manuscript's isotherm-path morphing (see heat_paths.py)
"""
import numpy as np
import scipy.ndimage as ndi
from scipy.fft import dctn, idctn


# ----------------------------------------------------------------------------- cleaning
def clean_sections(sections, min_area=12):
    """Remove small 2-D fragments that are not supported by the same label in an adjacent
    section (z-aware version of the manuscript's 11x11 rule). Removed pixels are given the
    label of the nearest surviving pixel."""
    out = [s.copy() for s in sections]
    for i, s in enumerate(sections):
        keep = np.ones(s.shape, bool)
        for k in np.unique(s):
            comp, n = ndi.label(s == k)
            if n == 0:
                continue
            areas = ndi.sum(np.ones_like(comp), comp, index=np.arange(1, n + 1))
            for c in np.nonzero(areas < min_area)[0] + 1:
                m = comp == c
                supported = any(
                    j >= 0 and j < len(sections) and np.any(sections[j][m] == k)
                    for j in (i - 1, i + 1))
                if not supported:
                    keep[m] = False
        if not keep.all():
            idx = ndi.distance_transform_edt(~keep, return_distances=False, return_indices=True)
            out[i] = s[tuple(idx)]
    return out


# ----------------------------------------------------------------------------- SDF helpers
def sdf_subvoxel(mask):
    """Signed distance with the zero level half a pixel outside the last inside pixel."""
    inside = ndi.distance_transform_edt(mask)
    outside = ndi.distance_transform_edt(~mask)
    return np.where(mask, inside - 0.5, -(outside - 0.5))


def sdf_binary(mask):
    """Textbook version: distance to the nearest opposite pixel, no half-pixel offset."""
    return ndi.distance_transform_edt(mask) - ndi.distance_transform_edt(~mask)


def _estimate_shift(m0, m1, maxshift=24, min_gain=0.2, iou_gate=0.3):
    """Integer translation of label mask m0 onto m1 by FFT cross-correlation of smoothed
    masks. Used only when the masks barely overlap and the shift clearly helps."""
    iou0 = (m0 & m1).sum() / max((m0 | m1).sum(), 1)
    if iou0 >= iou_gate:
        return np.zeros(2)
    g0 = ndi.gaussian_filter(m0.astype(float), 2.0)
    g1 = ndi.gaussian_filter(m1.astype(float), 2.0)
    F = np.fft.rfft2(g1) * np.conj(np.fft.rfft2(g0))
    c = np.fft.irfft2(F, s=g0.shape)
    ny, nx = c.shape
    yy = (np.arange(ny) + ny // 2) % ny - ny // 2
    xx = (np.arange(nx) + nx // 2) % nx - nx // 2
    ok = (np.abs(yy)[:, None] <= maxshift) & (np.abs(xx)[None, :] <= maxshift)
    c = np.where(ok, c, -np.inf)
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    d = np.array([yy[iy], xx[ix]], float)
    sh = ndi.shift(m0.astype(float), d, order=0) > 0.5
    iou1 = (sh & m1).sum() / max((sh | m1).sum(), 1)
    return d if iou1 > iou0 + min_gain else np.zeros(2)


# ----------------------------------------------------------------------------- pair kernels
def _argmax_update(best, second, lab, val, k, ti):
    b = best[ti]
    gt = val > b
    second[ti] = np.where(gt, b, np.maximum(second[ti], val))
    lab[ti] = np.where(gt, k, lab[ti])
    best[ti] = np.where(gt, val, b)


def _pair_fields(s0, s1, ts, kind, p):
    """Yield (label, [field_t for t in ts]) for every label present in either section."""
    labels = np.union1d(np.unique(s0), np.unique(s1))
    ny, nx = s0.shape
    if kind == "harmonic":
        lam = (2 - 2 * np.cos(np.pi * np.arange(ny) / ny))[:, None] + \
              (2 - 2 * np.cos(np.pi * np.arange(nx) / nx))[None, :]
        kap = np.sqrt(lam)
        L = p["L"]
        d = 1 - np.exp(-2 * kap * L)
        with np.errstate(divide="ignore", invalid="ignore"):
            W = []
            for t in ts:
                z = t * L
                w0 = (np.exp(-kap * z) - np.exp(-kap * (2 * L - z))) / d
                w1 = (np.exp(-kap * (L - z)) - np.exp(-kap * (L + z))) / d
                W.append((np.where(kap < 1e-12, 1 - t, w0), np.where(kap < 1e-12, t, w1)))
        for k in labels:
            A = dctn((s0 == k).astype(float), type=2, norm="ortho")
            B = dctn((s1 == k).astype(float), type=2, norm="ortho")
            yield k, [idctn(w0 * A + w1 * B, type=2, norm="ortho") for w0, w1 in W]
        return

    sigma, absent, cap = p["sigma"], p["absent"], p["cap"]
    sdf = sdf_subvoxel if p["subvoxel"] else sdf_binary
    for k in labels:
        m0, m1 = s0 == k, s1 == k
        f0 = sdf(m0) if m0.any() else np.full(m0.shape, -absent)
        f1 = sdf(m1) if m1.any() else np.full(m1.shape, -absent)
        if sigma > 0:
            f0 = ndi.gaussian_filter(f0, sigma, mode="nearest") if m0.any() else f0
            f1 = ndi.gaussian_filter(f1, sigma, mode="nearest") if m1.any() else f1
        f0, f1 = np.clip(f0, -cap, cap), np.clip(f1, -cap, cap)
        d = np.zeros(2)
        if p["align"] and m0.any() and m1.any():
            d = _estimate_shift(m0, m1, iou_gate=p["iou_gate"])
        out = []
        for t in ts:
            if d.any():
                a = ndi.shift(f0, t * d, order=1, mode="nearest")
                b = ndi.shift(f1, -(1 - t) * d, order=1, mode="nearest")
            else:
                a, b = f0, f1
            out.append((1 - t) * a + t * b)
        yield k, out


# ----------------------------------------------------------------------------- driver
def competition(sections, factor=5, kind="sdf", return_fields=False, **kw):
    """Run the field interpolation and return label, best and second-best field values
    (the latter two are what the conforming mesher needs)."""
    p = dict(sigma=1.0, absent=4.0, cap=12.0, subvoxel=True, align=False, iou_gate=1.01, L=float(factor))
    p.update(kw)
    S = len(sections)
    ny, nx = sections[0].shape
    nz = (S - 1) * factor + 1
    lab = np.zeros((nz, ny, nx), np.int64)  # MICrONS segment IDs are 18-digit
    best = np.full((nz, ny, nx), -np.inf)
    second = np.full((nz, ny, nx), -np.inf)
    fields = {} if return_fields else None
    for s in range(S - 1):
        last = s == S - 2
        ts = [j / factor for j in range(factor + (1 if last else 0))]
        zs = [s * factor + j for j in range(len(ts))]
        b, sec, l = best[zs], second[zs], lab[zs]
        for k, fl in _pair_fields(sections[s], sections[s + 1], ts, kind, p):
            for ti, f in enumerate(fl):
                _argmax_update(b, sec, l, f, k, ti)
                if return_fields:
                    fields.setdefault(k, np.full((nz, ny, nx), -p["absent"], np.float32))[zs[ti]] = f
        best[zs], second[zs], lab[zs] = b, sec, l
    return (lab, best, second, fields) if return_fields else (lab, best, second)


def reconstruct(sections, factor=5, method="sdf", clean=False, min_area=12, **kw):
    sections = [np.asarray(s) for s in sections]
    if clean:
        sections = clean_sections(sections, min_area)
    S = len(sections)
    ny, nx = sections[0].shape
    nz = (S - 1) * factor + 1
    if method == "nearest":
        vol = np.zeros((nz, ny, nx), sections[0].dtype)
        for z in range(nz):
            vol[z] = sections[min(int(round(z / factor)), S - 1)]
        return vol
    if method == "heat":
        from .heat_paths import heat_path_morph
        return heat_path_morph(sections, factor, **kw)
    if method == "sdf0":
        kw = dict(sigma=0.0, subvoxel=False, absent=kw.get("absent", 12.0), **{k: v for k, v in kw.items() if k != "absent"})
        return competition(sections, factor, "sdf", **kw)[0]
    return competition(sections, factor, "harmonic" if method == "harmonic" else "sdf", **kw)[0]


def majority(vol, iters=1):
    """Plain 3x3x3 majority filter (used as optional post-process for every method)."""
    labels = np.unique(vol)
    for _ in range(iters):
        best = np.full(vol.shape, -1.0)
        out = vol.copy()
        for k in labels:
            c = ndi.uniform_filter((vol == k).astype(np.float32), 3, mode="nearest")
            u = c > best
            best[u], out[u] = c[u], k
        vol = out
    return vol
