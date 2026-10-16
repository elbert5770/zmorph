"""Combinations of the heat method (heat2) and the signed-distance (SDF) method.

modes (all keep the source sections exact; all but 'field' and 'guard' switch each differing pixel between its
lower and its upper label, so they are partitions by construction)
  heat2       temperature-rank arrival, time-symmetric (the improved heat method), for reference
  sdf_rank    the same switching rule, with the pixels of each label ordered by the SDF two-label crossing time
  rank_avg    mean of the heat2 switch time and the sdf_rank switch time
  time_avg    mean of the heat2 switch time and the raw SDF crossing time (weight `lam` on the SDF)
  refine      heat2 frames smoothed with the SDF label smoothing (sources pinned)
  guard       SDF competition; a label that would vanish between two sections that both contain it is
              restored from the heat2 frame (instead of the nearer section)
  field       SDF competition on fields fused with the SDF of the heat2 segmentation of the same slice
"""
import numpy as np
import scipy.ndimage as ndi

from .heat_paths import heat_tstar, render_tstar
from .interp import _field, competition, sdf_subvoxel

HEAT2 = dict(arrival="Trank", symmetric=True, layer_z=2.0)


def sdf_crossing(s0, s1, sigma=1.0, absent=4.0, cap=12.0):
    """Time t in (0, 1) at which the lower label A and the upper label B of a differing pixel have equal
    interpolated signed distance: f_A(t) = f_B(t) (NaN where the sections agree)."""
    p = dict(sigma=sigma, absent=absent, cap=cap, subvoxel=True)
    chg = s0 != s1
    a0 = np.full(s0.shape, np.nan); a1 = a0.copy(); b0 = a0.copy(); b1 = a0.copy()
    for k in np.union1d(np.unique(s0), np.unique(s1)):
        m0, m1 = s0 == k, s1 == k
        f0, f1 = _field(m0, p), _field(m1, p)
        lo, hi = m0 & chg, m1 & chg
        a0[lo], a1[lo] = f0[lo], f1[lo]
        b0[hi], b1[hi] = f0[hi], f1[hi]
    num = a0 - b0
    den = num + (b1 - a1)
    return np.clip(num / np.maximum(den, 1e-9), 0.0, 1.0)


def _rank01(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x)) / max(len(x) - 1, 1)
    return r


def sdf_rank_tstar(s0, s1, t_sdf):
    """Equal-area, time-symmetric switch times ordered by the SDF crossing time: per upper label the rank of the
    crossing time (forward), per lower label the reverse rank, averaged."""
    chg = s0 != s1
    tf = np.full(s0.shape, np.nan); tr = tf.copy()
    for k in np.unique(s1[chg]):
        m = chg & (s1 == k)
        tf[m] = _rank01(t_sdf[m])
    for k in np.unique(s0[chg]):
        m = chg & (s0 == k)
        tr[m] = 1.0 - _rank01(1.0 - t_sdf[m])
    return 0.5 * (tf + tr)


def refine_with_sdf(vol, factor, sigma=1.0):
    """SDF label smoothing of every interpolated slice (argmax of smoothed per-label signed distances);
    the source slices are left untouched."""
    out = vol.copy()
    p = dict(sigma=sigma, absent=4.0, cap=12.0, subvoxel=True)
    for z in range(vol.shape[0]):
        if z % factor == 0:
            continue
        sl = vol[z]
        best = np.full(sl.shape, -np.inf)
        lab = sl.copy()
        for k in np.unique(sl):
            f = _field(sl == k, p)
            u = f > best
            best[u], lab[u] = f[u], k
        out[z] = lab
    return out


def heat_sdf_morph(sections, factor=5, mode="rank_avg", lam=0.5, sigma=1.0):
    sections = [np.asarray(s) for s in sections]
    if mode == "heat2":
        ts = [heat_tstar(sections[s], sections[s + 1], factor, **HEAT2) for s in range(len(sections) - 1)]
        return render_tstar(sections, ts, factor)
    if mode in ("sdf_rank", "rank_avg", "time_avg"):
        ts = []
        for s in range(len(sections) - 1):
            s0, s1 = sections[s], sections[s + 1]
            tsdf = sdf_crossing(s0, s1, sigma)
            if mode == "sdf_rank":
                ts.append(sdf_rank_tstar(s0, s1, tsdf))
                continue
            th = heat_tstar(s0, s1, factor, **HEAT2)
            other = sdf_rank_tstar(s0, s1, tsdf) if mode == "rank_avg" else tsdf
            w = 0.5 if mode == "rank_avg" else lam
            ts.append(np.where(np.isnan(th), other, np.where(np.isnan(other), th, (1 - w) * th + w * other)))
        return render_tstar(sections, ts, factor)
    heat = heat_sdf_morph(sections, factor, "heat2")
    if mode == "refine":
        return refine_with_sdf(heat, factor, sigma)
    if mode == "guard":
        return competition(sections, factor, "sdf", persist=True, persist_vol=heat, sigma=sigma)[0]
    if mode == "field":
        return competition(sections, factor, "sdf", prior_vol=heat, prior_w=lam, sigma=sigma)[0]
    raise ValueError(mode)
