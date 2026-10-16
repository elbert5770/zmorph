"""Port of the manuscript's method (Elbert 2025) for use as a baseline.

Per cell and section pair: steady-state heat between the two masks (Dirichlet T=10 where the
cell is present on a face, 0 elsewhere, no-flux laterally, 25 layers, observation layer 7),
then gradient ascent from pixels unique to the upper section until the nearest pixel is in the
overlap, path lengths normalised by the longest path per terminal pixel, and pixels handed to
the cell in frame j if ceil(5*fraction) <= j.

Differences from the original code: the 3-D solve is done in closed form with a DCT instead of
ILU-CG; the gradient is taken everywhere (as described in the paper) rather than only on the
morphing region; cells with no overlap use the temperature maximum as the sink.
"""
import numpy as np
import scipy.ndimage as ndi
from scipy.fft import dctn, idctn


def _layer_temperature(m0, m1, L=25.0, z=6.5, temp=10.0):
    ny, nx = m0.shape
    lam = (2 - 2 * np.cos(np.pi * np.arange(ny) / ny))[:, None] + \
          (2 - 2 * np.cos(np.pi * np.arange(nx) / nx))[None, :]
    kap = np.sqrt(lam)
    d = 1 - np.exp(-2 * kap * L)
    with np.errstate(divide="ignore", invalid="ignore"):
        w0 = (np.exp(-kap * z) - np.exp(-kap * (2 * L - z))) / d
        w1 = (np.exp(-kap * (L - z)) - np.exp(-kap * (L + z))) / d
    w0 = np.where(kap < 1e-12, 1 - z / L, w0)
    w1 = np.where(kap < 1e-12, z / L, w1)
    A = dctn(temp * m0.astype(float), type=2, norm="ortho")
    B = dctn(temp * m1.astype(float), type=2, norm="ortho")
    return idctn(w0 * A + w1 * B, type=2, norm="ortho")


def _grad(T, stencil, mask):
    if stencil == 4:   # the Julia code's 4th-order central stencil (nearest-edge padding)
        w = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0
        gy = ndi.correlate1d(T, w, axis=0, mode="nearest")
        gx = ndi.correlate1d(T, w, axis=1, mode="nearest")
    else:
        gy, gx = np.gradient(T)
    if mask is not None:
        gy, gx = np.where(mask, gy, 0.0), np.where(mask, gx, 0.0)
    return gy, gx


def _trace(T, start, sink, max_iter=2000, stencil=2, grad_mask=None):
    gy, gx = _grad(T, stencil, grad_mask)
    ny, nx = T.shape
    pos = start.astype(float).copy()
    length = np.zeros(len(pos))
    done = np.zeros(len(pos), bool)
    term = np.zeros((len(pos), 2), int)
    for _ in range(max_iter):
        act = np.nonzero(~done)[0]
        if act.size == 0:
            break
        p = pos[act]
        near = np.clip(np.rint(p).astype(int), 0, [ny - 1, nx - 1])
        hit = sink[near[:, 0], near[:, 1]] | (p[:, 0] < 0) | (p[:, 1] < 0) | \
              (p[:, 0] > ny - 1) | (p[:, 1] > nx - 1)
        term[act[hit]] = near[hit]
        done[act[hit]] = True
        act, p = act[~hit], p[~hit]
        if act.size == 0:
            break
        step = np.stack([ndi.map_coordinates(gy, p.T, order=1, mode="nearest"),
                         ndi.map_coordinates(gx, p.T, order=1, mode="nearest")], 1)
        pos[act] = p + step
        length[act] += np.linalg.norm(step, axis=1)
    rest = ~done
    term[rest] = np.clip(np.rint(pos[rest]).astype(int), 0, [ny - 1, nx - 1])
    return length, term


def _arrivals(m0, m1, mode, layer_z, L, max_iter, grad, sink, stencil):
    """Arrival fraction in [0, 1] of the label at each pixel unique to the upper mask m1 (not in m0):
    0 = it takes the pixel immediately, 1 = last. Returns (pixel coordinates (n, 2), arrival (n,)).
    layer_z may be a sequence of layers; the arrival is then averaged over them.
    mode 'paths'        the paper: path length normalised by the longest path ending at the same terminal pixel
         'paths_global' path length normalised by the longest path of the cell
         'rank'         area rank of the path length among the cell's unique pixels (equal-area progress)
         'T'            1 - T/Tmax of the layer temperature over the unique pixels (no path tracing)
         'Trank'        area rank of the layer temperature, hottest first (no path tracing)"""
    if not np.isscalar(layer_z):               # scale-space ensemble: average the arrival over several layers
        res = [_arrivals(m0, m1, mode, float(z), L, max_iter, grad, sink, stencil) for z in layer_z]
        return res[0][0], np.mean([r[1] for r in res], axis=0)
    ny, nx = m0.shape
    uniq = m1 & ~m0
    start = np.argwhere(uniq)
    if len(start) == 0:
        return start, np.zeros(0)
    T = _layer_temperature(m0, m1, L=L, z=layer_z)
    if mode in ("T", "Trank"):
        t = T[start[:, 0], start[:, 1]]
        if mode == "T":
            return start, np.clip(1.0 - (t - t.min()) / max(t.max() - t.min(), 1e-12), 0.0, 1.0)
        order = np.argsort(-t, kind="stable")
        r = np.empty(len(t)); r[order] = np.arange(len(t)) / max(len(t) - 1, 1)
        return start, r
    sk = (m0 & m1) if sink == "own" else None
    if sk is None or not sk.any():
        sk = (T >= 0.98 * T.max()) if not (m0 & m1).any() else (m0 & m1)
    length, term = _trace(T, start, sk, max_iter, stencil, uniq if grad == "unique" else None)
    if mode == "paths":
        gmax = np.zeros((ny, nx))
        np.maximum.at(gmax, (term[:, 0], term[:, 1]), length)
        denom = gmax[term[:, 0], term[:, 1]]
        return start, np.where(denom > 0, length / np.maximum(denom, 1e-12), 0.0)
    if mode == "paths_global":
        return start, length / max(length.max(), 1e-12)
    order = np.argsort(length, kind="stable")                                  # 'rank'
    r = np.empty(len(length)); r[order] = np.arange(len(length)) / max(len(length) - 1, 1)
    return start, r


def heat_tstar(s0, s1, factor=5, grad="all", sink="own", stencil=2, max_iter=2000, layer_z=6.5, L=25.0,
               arrival="paths", symmetric=False):
    """Switch time t* in [0, 1] of every pixel at which the two sections differ (NaN elsewhere): the pixel
    carries the lower label for t < t* and the upper label for t >= t*."""
    ny, nx = s0.shape
    tf = np.full((ny, nx), np.nan)            # forward arrival of the upper label
    tr = np.full((ny, nx), np.nan)            # reverse estimate: time at which the lower label leaves
    for k in np.union1d(np.unique(s0), np.unique(s1)):
        st, arr = _arrivals(s0 == k, s1 == k, arrival, layer_z, L, max_iter, grad, sink, stencil)
        if len(st):
            tf[st[:, 0], st[:, 1]] = arr
        if symmetric:
            st, arr = _arrivals(s1 == k, s0 == k, arrival, layer_z, L, max_iter, grad, sink, stencil)
            if len(st):
                tr[st[:, 0], st[:, 1]] = 1.0 - arr
    return np.where(np.isnan(tf), tr, np.where(np.isnan(tr), tf, 0.5 * (tf + tr))) if symmetric else tf


def render_tstar(sections, tstars, factor):
    """Build the label volume from per-pair switch-time maps."""
    S = len(sections)
    ny, nx = sections[0].shape
    vol = np.zeros(((S - 1) * factor + 1, ny, nx), sections[0].dtype)
    for s in range(S):
        vol[s * factor] = sections[s]
    for s in range(S - 1):
        s0, s1, ts_ = sections[s], sections[s + 1], tstars[s]
        for j in range(1, factor):
            t = j / factor
            sw = (s0 != s1) & ~np.isnan(ts_) & (t >= ts_ - 1e-9)
            vol[s * factor + j] = np.where(sw, s1, s0)
    return vol


def heat_path_morph(sections, factor=5, grad="all", sink="own", stencil=2, max_iter=2000, layer_z=6.5, L=25.0,
                    arrival="paths", symmetric=False):
    """Defaults reproduce the published algorithm. `arrival` selects how the arrival fraction is computed
    (see `_arrivals`). With `symmetric=True` a reverse pass (upper section to lower) is run as well: the
    lower label leaves a pixel at time 1 - (its reverse arrival), and the pixel switches at the mean of the
    two estimates, which makes the result independent of the direction of z."""
    ts = [heat_tstar(sections[s], sections[s + 1], factor, grad, sink, stencil, max_iter, layer_z, L, arrival, symmetric)
          for s in range(len(sections) - 1)]
    return render_tstar(sections, ts, factor)
