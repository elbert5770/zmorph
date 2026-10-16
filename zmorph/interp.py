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
  heat2     improved heat method: temperature-rank arrival, time-symmetric, observation layer z=2
  sdf0_cubic / sdf0_4pt   higher-order SDF interpolation from several sections with textbook fields
  sdf_cubic  / sdf_4pt    the same interpolation orders with the proposed field construction
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



def _components(mask):
    cc, n = ndi.label(mask)
    return [cc == i for i in range(1, n + 1)]



def _descriptor(mask, min_area=12, min_slope=1.0):
    """Centroid c (y, x), unit major axis u (y, x) and tilt slope v (pixels per z voxel) of a footprint.
    A thin straight tube of radius r with in-plane slope v cuts a z-plane in an ellipse with semi-axes
    a = r*sqrt(1 + v^2) and b = r, so v = sqrt((a/b)^2 - 1) = sqrt(lambda1/lambda2 - 1) from the second moments.
    Footprints below `min_area` or with v < `min_slope` are treated as vertical (v = 0)."""
    ys, xs = np.nonzero(mask)
    c = np.array([ys.mean(), xs.mean()])
    if len(ys) < min_area:
        return c, np.array([0.0, 1.0]), 0.0, 0.0
    cov = np.cov(np.stack([ys, xs]))
    w, V = np.linalg.eigh(cov)
    u = V[:, 1]
    v = float(np.sqrt(max(w[1] / max(w[0], 1e-6) - 1.0, 0.0)))
    # how ellipse-like is the footprint? IoU with the uniform ellipse that has the same second moments
    a, b = 2.0 * np.sqrt(max(w[1], 1e-6)), 2.0 * np.sqrt(max(w[0], 1e-6))
    y0, y1, x0, x1 = ys.min() - 2, ys.max() + 3, xs.min() - 2, xs.max() + 3
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.stack([yy - c[0], xx - c[1]], -1)
    e = ((d @ V[:, 1]) / a) ** 2 + ((d @ V[:, 0]) / b) ** 2 <= 1.0
    inb = (yy >= 0) & (yy < mask.shape[0]) & (xx >= 0) & (xx < mask.shape[1])
    mm = np.zeros(e.shape, bool)
    mm[inb] = mask[yy[inb], xx[inb]]
    ell = float((mm & e).sum() / max((mm | e).sum(), 1))
    return c, u, (v if v >= min_slope else 0.0), ell


def _tube_link(m0, m1, L, p):
    """Can the non-overlapping footprints m0 (lower section) and m1 (upper section) be two cross-sections of
    one tube that branches or ends between them? Tests both directions: the axis of m1, extended back
    along its tilt, must enter m0's footprint (or m0's axis extended forward must enter m1's), within `tube_tol`
    pixels, at some height in the gap. Returns None or dict(mode, shift, zJ): `mode` 'up' = m1's tube is
    extended back (m0 holds until zJ, then m1 shifted back), 'down' = m0's tube is extended forward
    (m0 shifted forward until zJ, then m1 holds)."""
    zs = np.linspace(0.0, L, 41)
    best = None
    c0, u0, v0, e0 = _descriptor(m0, p["tube_min_area"], p["tube_min_slope"])
    c1, u1, v1, e1 = _descriptor(m1, p["tube_min_area"], p["tube_min_slope"])
    ny, nx = m0.shape
    for mode, (src, tgt, c, u, v, ell) in (("up", (m1, m0, c1, u1, v1, e1)), ("down", (m0, m1, c0, u0, v0, e0))):
        if v == 0.0 or ell < p["tube_min_ellipse"]:
            continue
        dist = ndi.distance_transform_edt(~tgt)
        for sg in (1.0, -1.0):
            if mode == "up":      # axis of m1 at height z (0 = lower section): c1 - sg*v*u*(L - z)
                pts = c[None, :] - sg * v * u[None, :] * (L - zs)[:, None]
            else:                 # axis of m0 at height z: c0 + sg*v*u*z
                pts = c[None, :] + sg * v * u[None, :] * zs[:, None]
            ij = np.rint(pts).astype(int)
            ok = (ij[:, 0] >= 0) & (ij[:, 0] < ny) & (ij[:, 1] >= 0) & (ij[:, 1] < nx)
            d = np.full(len(zs), np.inf)
            d[ok] = dist[ij[ok, 0], ij[ok, 1]]
            hit = np.nonzero(d <= p["tube_tol"])[0]
            if hit.size == 0:
                continue
            # up: the tube of m1 exists above the last height at which its axis is inside m0's neighbourhood;
            # down: the tube of m0 exists below the first height at which its axis reaches m1.
            ct = np.array([np.nonzero(tgt)[0].mean(), np.nonzero(tgt)[1].mean()])
            k = hit[np.argmin(np.linalg.norm(pts[hit] - ct[None, :], axis=1))]   # axis passes closest to the partner's centre
            score = float(np.linalg.norm(pts[k] - ct))
            if mode == "down":    # the same tube must continue: the translated footprint has to match the partner
                moved = ndi.shift(src.astype(float), sg * v * u * L, order=0) > 0.5
                if (moved & tgt).sum() / max((moved | tgt).sum(), 1) < p["tube_min_continue"]:
                    continue
            if best is None or score < best[0] - 1e-9:
                best = (score, dict(mode=mode, shift=sg * v * u, zJ=float(zs[k])))
    if v0 > 0.0 and v1 > 0.0 and min(e0, e1) >= p["tube_min_ellipse"]:     # both axes extended: the junction is where the two lines meet
        for s0_ in (1.0, -1.0):
            for s1_ in (1.0, -1.0):
                a = c0[None, :] + s0_ * v0 * u0[None, :] * zs[:, None]
                b = c1[None, :] - s1_ * v1 * u1[None, :] * (L - zs)[:, None]
                d = np.linalg.norm(a - b, axis=1)
                k = int(np.argmin(d))
                if d[k] <= p["tube_tol"] and (best is None or d[k] < best[0] - 1e-9):
                    best = (float(d[k]), dict(mode="both", shift=s0_ * v0 * u0, shift1=s1_ * v1 * u1, zJ=float(zs[k])))
    return None if best is None else best[1]

def _component_fields(m0, m1, fld, ts, p, translate=True):
    """Component-wise translation for splits and fusions. Every connected component of the label in one
    section is paired with its best-overlapping component in the other section (after the best
    translation, `_estimate_shift`); a component may be paired with several (split) or be the partner of
    several (fusion). Unpaired components pair with 'nothing' (they appear or vanish). The label's field
    is the maximum over pairs of the translated, interpolated pair field, so the competition and the
    partition guarantee are unchanged. Returns None if there are too many pairs to evaluate."""
    c0, c1 = _components(m0), _components(m1)
    if len(c0) * len(c1) > p["max_pairs"]:
        return None
    iou = np.zeros((len(c0), len(c1)))
    shift = {}
    for i, a in enumerate(c0):
        for j, b in enumerate(c1):
            d = _estimate_shift(a, b, maxshift=p["maxshift"], iou_gate=1.01) if translate else np.zeros(2)
            sh = ndi.shift(a.astype(float), d, order=0) > 0.5 if d.any() else a
            iou[i, j] = (sh & b).sum() / max((sh | b).sum(), 1)
            shift[i, j] = d
    tubes = {}
    if p.get("tube"):
        for i in range(len(c0)):
            for j in range(len(c1)):
                if (c0[i] & c1[j]).any():
                    continue
                lk = _tube_link(c0[i], c1[j], p["L"], p)
                if lk is not None:
                    tubes[i, j] = lk
                    iou[i, j] = 0.0          # a tube link replaces a translation match for this pair
    if not translate and not tubes:
        return None                     # tube mode found nothing: leave this label to the ordinary label-level field
    pairs = set()
    for j in range(len(c1)):
        i = int(np.argmax(iou[:, j]))
        if iou[i, j] > p["min_iou"]:
            pairs.add((i, j))
    for i in range(len(c0)):
        j = int(np.argmax(iou[i]))
        if iou[i, j] > p["min_iou"]:
            pairs.add((i, j))
    pairs |= {(i, None) for i in range(len(c0)) if not any(q[0] == i for q in pairs) and not any(k[0] == i for k in tubes)}
    pairs |= {(None, j) for j in range(len(c1)) if not any(q[1] == j for q in pairs) and not any(k[1] == j for k in tubes)}
    f0 = [fld(a) for a in c0]
    f1 = [fld(b) for b in c1]
    gone = np.full(m0.shape, -min(p["absent"], p["neg_cap"] if p["neg_cap"] is not None else p["cap"]))
    out = []
    for t in ts:
        best = None
        for i, j in pairs:
            a0 = f0[i] if i is not None else gone
            a1 = f1[j] if j is not None else gone
            d = shift.get((i, j), np.zeros(2))
            if d.any():
                a0 = ndi.shift(a0, t * d, order=1, mode="nearest")
                a1 = ndi.shift(a1, -(1 - t) * d, order=1, mode="nearest")
            v = (1 - t) * a0 + t * a1
            best = v if best is None else np.maximum(best, v)
        for (i, j), lk in tubes.items():     # tube-linked pair: hard switch at the junction height, shifted along the axis
            z = t * p["L"]
            if lk["mode"] == "both":
                f = ndi.shift(f0[i], lk["shift"] * z, order=1, mode="nearest") if z <= lk["zJ"] else \
                    ndi.shift(f1[j], -lk["shift1"] * (p["L"] - z), order=1, mode="nearest")
            elif lk["mode"] == "up":
                f = f0[i] if z <= lk["zJ"] else ndi.shift(f1[j], -lk["shift"] * (p["L"] - z), order=1, mode="nearest")
            else:
                f = ndi.shift(f0[i], lk["shift"] * z, order=1, mode="nearest") if z <= lk["zJ"] else f1[j]
            v = np.where(f > 0, f + 24.0, f)      # the tube asserts presence: it overrides neighbours inside its footprint
            best = v if best is None else np.maximum(best, v)
        out.append(best)
    return out

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
    ncap = p["neg_cap"] if p["neg_cap"] is not None else cap

    def fld(m):
        f = sdf(m)
        if sigma > 0:
            f = ndi.gaussian_filter(f, sigma, mode="nearest")
        return np.clip(f, -ncap, cap)

    for k in labels:
        m0, m1 = s0 == k, s1 == k
        if (p["align"] == "component" or p["tube"]) and m0.any() and m1.any():
            res = _component_fields(m0, m1, fld, ts, p, translate=(p["align"] == "component"))
            if res is not None:
                yield k, res
                continue
        f0 = sdf(m0) if m0.any() else np.full(m0.shape, -absent)
        f1 = sdf(m1) if m1.any() else np.full(m1.shape, -absent)
        if sigma > 0:
            f0 = ndi.gaussian_filter(f0, sigma, mode="nearest") if m0.any() else f0
            f1 = ndi.gaussian_filter(f1, sigma, mode="nearest") if m1.any() else f1
        f0, f1 = np.clip(f0, -ncap, cap), np.clip(f1, -ncap, cap)
        d = np.zeros(2)
        if p["align"] and m0.any() and m1.any():     # True or "component" with too many pairs: label-level shift
            d = _estimate_shift(m0, m1, maxshift=p["maxshift"], iou_gate=p["iou_gate"])
        out = []
        for t in ts:
            if d.any():
                a = ndi.shift(f0, t * d, order=1, mode="nearest")
                b = ndi.shift(f1, -(1 - t) * d, order=1, mode="nearest")
            else:
                a, b = f0, f1
            out.append((1 - t) * a + t * b)
        yield k, out


def _persist(s0, s1, ts, zs, best, second, lab, fields, p):
    """Persistence guard. A label present in both sections must not vanish between them: where its area in
    an intermediate slice falls below `persist_frac` times the smaller of its two section areas, its
    footprint from the nearer section is restored (it overrides the other labels there). Without this, a
    thin structure that moves farther than its width is deleted in the gap (the signed-distance blend is
    negative everywhere). It limits the damage to 'repeat the nearest section' for that label."""
    for k in np.intersect1d(np.unique(s0), np.unique(s1)):
        m0, m1 = s0 == k, s1 == k
        need = p["persist_frac"] * min(m0.sum(), m1.sum())
        for ti, t in enumerate(ts):
            if 0 < t < 1 and (lab[ti] == k).sum() < need:
                src = m0 if t < 0.5 else m1
                f = _field(src, p)
                inside = f > 0
                top = np.maximum(best[ti], 0) + 24.0
                second[ti] = np.where(inside & (lab[ti] != k), best[ti], np.where(inside, second[ti], second[ti]))
                best[ti] = np.where(inside, top + f, best[ti])
                lab[ti] = np.where(inside, k, lab[ti])
                if fields is not None and k in fields:
                    fields[k][zs[ti]] = np.where(inside, top + f, fields[k][zs[ti]])


# ----------------------------------------------------------------------------- driver
def competition(sections, factor=5, kind="sdf", return_fields=False, **kw):
    """Run the field interpolation and return label, best and second-best field values
    (the latter two are what the conforming mesher needs)."""
    p = dict(sigma=1.0, absent=4.0, cap=12.0, subvoxel=True, align=False, iou_gate=1.01, L=float(factor),
             maxshift=24, min_iou=0.05, max_pairs=64, neg_cap=None,
             persist=False, persist_frac=0.5,
             tube=False, tube_tol=3.0, tube_min_area=12, tube_min_slope=1.0,
             tube_min_ellipse=0.6, tube_min_continue=0.3)
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
        if p.get("persist") and kind == "sdf":
            _persist(sections[s], sections[s + 1], ts, zs, b, sec, l, fields, p)
        best[zs], second[zs], lab[zs] = b, sec, l
    return (lab, best, second, fields) if return_fields else (lab, best, second)


# ----------------------------------------------------------------------------- higher-order (multi-section)
def _field(mask, p):
    """One label's field on one section, built exactly as in _pair_fields."""
    if not mask.any():
        return np.full(mask.shape, -p["absent"])
    f = (sdf_subvoxel if p["subvoxel"] else sdf_binary)(mask)
    if p["sigma"] > 0:
        f = ndi.gaussian_filter(f, p["sigma"], mode="nearest")
    return np.clip(f, -p["cap"], p["cap"])


def _lagrange_weights(x, nodes):
    w = np.ones(len(nodes))
    for j, xj in enumerate(nodes):
        for m, xm in enumerate(nodes):
            if m != j:
                w[j] *= (x - xm) / (xj - xm)
    return w


def competition_multi(sections, factor=5, interp="cubic", **kw):
    """Like `competition` for kind='sdf', but each label's field is interpolated in z from several
    sections instead of the two neighbouring ones.
      interp='cubic'  natural cubic spline through all sections (scipy CubicSpline)
      interp='4pt'    4-point (cubic Lagrange) interpolation from the four nearest sections; at the
                      midpoint the weights are (-1/16, 9/16, 9/16, -1/16) of the 4-point scheme.
                      With fewer than four sections the highest available degree is used.
    Both pass through every observed section's field exactly."""
    from scipy.interpolate import CubicSpline
    p = dict(sigma=1.0, absent=4.0, cap=12.0, subvoxel=True)
    p.update(kw)
    S = len(sections)
    ny, nx = sections[0].shape
    nz = (S - 1) * factor + 1
    lab = np.zeros((nz, ny, nx), np.int64)
    best = np.full((nz, ny, nx), -np.inf)
    second = np.full((nz, ny, nx), -np.inf)
    labels = np.unique(np.concatenate([np.unique(s) for s in sections]))
    zpos = np.arange(nz) / factor
    for k in labels:
        F = np.stack([_field(s == k, p) for s in sections])
        if interp == "cubic":
            V = CubicSpline(np.arange(S), F, axis=0, bc_type="natural")(zpos) if S > 2 else \
                (1 - zpos)[:, None, None] * F[0] + zpos[:, None, None] * F[-1]   # two sections: linear, zpos in [0, 1]
        else:
            n = min(4, S)
            V = np.empty((nz, ny, nx))
            for z in range(nz):
                seg = min(int(zpos[z]), S - 2)
                i0 = int(np.clip(seg - (n // 2 - 1), 0, S - n))
                w = _lagrange_weights(zpos[z], np.arange(i0, i0 + n))
                V[z] = np.tensordot(w, F[i0:i0 + n], axes=1)
        for z in range(nz):
            _argmax_update(best, second, lab, V[z], k, z)
    return lab, best, second


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
    if method == "heat2":     # improved heat method: temperature-rank arrival, time-symmetric, observation layer near the face
        from .heat_paths import heat_path_morph
        return heat_path_morph(sections, factor, **{**dict(arrival="Trank", symmetric=True, layer_z=2.0), **kw})
    if method in ("sdf_cubic", "sdf_4pt", "sdf0_cubic", "sdf0_4pt"):
        classic = method.startswith("sdf0")
        kw = dict(kw)
        if classic:   # textbook fields: binary distance, no smoothing, absent labels at the cap
            kw = dict(sigma=0.0, subvoxel=False, absent=kw.pop("absent", 12.0), **kw)
        return competition_multi(sections, factor, "cubic" if method.endswith("cubic") else "4pt", **kw)[0]
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
