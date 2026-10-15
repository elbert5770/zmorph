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


def heat_path_morph(sections, factor=5, grad="all", sink="own", stencil=2, max_iter=2000):
    """grad: 'all' (as the paper describes) or 'unique' (as the Julia code does: the gradient is
    only evaluated on pixels unique to the upper frame and is zero elsewhere).
    sink: 'own' (the cell's own overlap, as the paper) or 'any' (any pixel where the two sections
    carry the same label, an approximation of the code's shared overlap_keys)."""
    S = len(sections)
    ny, nx = sections[0].shape
    nz = (S - 1) * factor + 1
    vol = np.zeros((nz, ny, nx), sections[0].dtype)
    for s in range(S):
        vol[s * factor] = sections[s]
    for s in range(S - 1):
        s0, s1 = sections[s], sections[s + 1]
        frames = np.repeat(s0[None], factor - 1, axis=0)
        for k in np.union1d(np.unique(s0), np.unique(s1)):
            m0, m1 = s0 == k, s1 == k
            uniq = m1 & ~m0
            if not uniq.any():
                continue
            T = _layer_temperature(m0, m1)
            sk = (m0 & m1) if sink == "own" else (s0 == s1)
            if not (m0 & m1).any():
                sk = (T >= 0.98 * T.max()) | (sk if sink == "any" else False)
            start = np.argwhere(uniq)
            length, term = _trace(T, start, sk, max_iter, stencil, uniq if grad == "unique" else None)
            gmax = np.zeros((ny, nx))
            np.maximum.at(gmax, (term[:, 0], term[:, 1]), length)
            denom = gmax[term[:, 0], term[:, 1]]
            frac = np.where(denom > 0, length / np.maximum(denom, 1e-12), 0.0)
            jmin = np.maximum(1, np.ceil(factor * frac).astype(int))
            for j in range(1, factor):
                sel = jmin <= j
                frames[j - 1][start[sel, 0], start[sel, 1]] = k
        vol[s * factor + 1:s * factor + factor] = frames
    return vol
