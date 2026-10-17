"""Brownian dynamics of point tracers in the gap between cells (the extracellular space, ECS).

Geometry is implicit. For a label partition the competition margin  m = (best field) - (second-best field)  is
twice the distance to the nearest cell-cell interface, so  ECS = { m < w }  is a gap of thickness w centred on every
interface (sheets on faces, tunnels along triple lines, pools at four-cell corners; no surface is offset, so there
are no self-intersections). The wall is the level set  phi = m - w = 0.  No mesh is built.

  margin_field(labels, h, ...)   3-D label volume -> margin (nm), via per-label sub-voxel signed distances
  simulate(phi, h, x0, ...)      reflecting Brownian motion with specular reflection at phi = 0 (numba)
  sample_ecs(phi, h, n)          uniform start positions in {phi < 0}

Units: lengths in nm, D in nm^2 per time unit (use D = 1 and read time in nm^2/D). Axes are (x, y, z) = array axes
(0, 1, 2); `periodic` is a 3-tuple of bool. numba is required (pip install numba).
"""
import math
import numpy as np
import scipy.ndimage as ndi
import numba as nb


# ----------------------------------------------------------------------------- geometry
def _sdf_subvoxel(mask):
    inside = ndi.distance_transform_edt(mask)
    outside = ndi.distance_transform_edt(~mask)
    return np.where(mask, inside - 0.5, -(outside - 0.5))


def margin_field(labels, h=1.0, cap=None, sigma=0.0, periodic=(False, False, False)):
    """Margin m = best - second-best of the per-label signed distances (nm) on a 3-D label volume with voxel size h.
    With a distance cap (default 64 nm) the margin is exact where it is below ~2*cap. `sigma` (voxels) optionally
    smooths each field like sdf_align does. `periodic` axes are wrapped (each cell then needs a different label from
    its periodic neighbour, i.e. at least 2 cells per periodic axis)."""
    cap = 64.0 if cap is None else cap
    capv = cap / h
    pad = int(math.ceil(capv)) + 3
    per = [bool(p) for p in periodic]
    L = np.pad(labels, [(pad, pad) if p else (0, 0) for p in per], mode="wrap")
    best = np.full(L.shape, -capv); second = np.full(L.shape, -capv)
    objs = ndi.find_objects(np.unique(L, return_inverse=True)[1].reshape(L.shape) + 1)
    for k, sl in zip(np.unique(L), objs):
        box = tuple(slice(max(s.start - pad, 0), min(s.stop + pad, n)) for s, n in zip(sl, L.shape))
        f = _sdf_subvoxel(L[box] == k)
        if sigma > 0:
            f = ndi.gaussian_filter(f, sigma, mode="nearest")
        f = np.clip(f, -capv, capv)
        full = np.full(L.shape, -capv); full[box] = f
        second = np.maximum(second, np.minimum(best, full)); best = np.maximum(best, full)
    m = (best - second) * h
    return m[tuple(slice(pad, -pad) if p else slice(None) for p in per)]


# ----------------------------------------------------------------------------- numba kernels
@nb.njit(cache=True, fastmath=True)
def _val(phi, x, y, z, px, py, pz):
    nx, ny, nz = phi.shape
    ix, iy, iz = math.floor(x), math.floor(y), math.floor(z)
    fx, fy, fz = x - ix, y - iy, z - iz
    if px:
        i0 = ix % nx; i1 = (i0 + 1) % nx
    else:
        if ix < 0 or ix + 1 > nx - 1:
            return 1e9
        i0 = ix; i1 = ix + 1
    if py:
        j0 = iy % ny; j1 = (j0 + 1) % ny
    else:
        if iy < 0 or iy + 1 > ny - 1:
            return 1e9
        j0 = iy; j1 = iy + 1
    if pz:
        k0 = iz % nz; k1 = (k0 + 1) % nz
    else:
        if iz < 0 or iz + 1 > nz - 1:
            return 1e9
        k0 = iz; k1 = iz + 1
    c00 = phi[i0, j0, k0] * (1 - fx) + phi[i1, j0, k0] * fx
    c10 = phi[i0, j1, k0] * (1 - fx) + phi[i1, j1, k0] * fx
    c01 = phi[i0, j0, k1] * (1 - fx) + phi[i1, j0, k1] * fx
    c11 = phi[i0, j1, k1] * (1 - fx) + phi[i1, j1, k1] * fx
    return (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz


@nb.njit(cache=True, fastmath=True)
def _grad(phi, x, y, z, px, py, pz):
    e = 0.5
    gx = _val(phi, x + e, y, z, px, py, pz) - _val(phi, x - e, y, z, px, py, pz)
    gy = _val(phi, x, y + e, z, px, py, pz) - _val(phi, x, y - e, z, px, py, pz)
    gz = _val(phi, x, y, z + e, px, py, pz) - _val(phi, x, y, z - e, px, py, pz)
    n = math.sqrt(gx * gx + gy * gy + gz * gz)
    if n < 1e-12 or n > 1e6:
        return 0.0, 0.0, 0.0
    return gx / n, gy / n, gz / n


@nb.njit(cache=True, fastmath=True, parallel=True)
def _advance(phi, pos, nsteps, sig, px, py, pz):
    """Advance every particle by `nsteps` steps (positions in voxel units, modified in place)."""
    for p in nb.prange(pos.shape[0]):
        x, y, z = pos[p, 0], pos[p, 1], pos[p, 2]
        for _ in range(nsteps):
            dx, dy, dz = sig * np.random.randn(), sig * np.random.randn(), sig * np.random.randn()
            xn, yn, zn = x + dx, y + dy, z + dz
            if _val(phi, xn, yn, zn, px, py, pz) <= 0.0:
                x, y, z = xn, yn, zn
                continue
            lo, hi = 0.0, 1.0                                   # bisect the crossing along the step
            for _it in range(12):
                mid = 0.5 * (lo + hi)
                if _val(phi, x + mid * dx, y + mid * dy, z + mid * dz, px, py, pz) <= 0.0:
                    lo = mid
                else:
                    hi = mid
            xc, yc, zc = x + lo * dx, y + lo * dy, z + lo * dz
            nx_, ny_, nz_ = _grad(phi, xc, yc, zc, px, py, pz)
            rx, ry, rz = (1 - lo) * dx, (1 - lo) * dy, (1 - lo) * dz
            dot = rx * nx_ + ry * ny_ + rz * nz_
            xr, yr, zr = xc + rx - 2 * dot * nx_, yc + ry - 2 * dot * ny_, zc + rz - 2 * dot * nz_
            if _val(phi, xr, yr, zr, px, py, pz) <= 0.0:
                x, y, z = xr, yr, zr
            else:
                x, y, z = xc, yc, zc
        pos[p, 0], pos[p, 1], pos[p, 2] = x, y, z


def sample_ecs(phi, h, n, periodic=(False, False, False), seed=0):
    """n positions (voxel units) uniformly distributed in {phi < 0} by rejection."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = phi.shape
    hi = np.array([nx - 1 + (1 if p else 0) for p in periodic] and [nx - (0 if periodic[0] else 1), ny - (0 if periodic[1] else 1), nz - (0 if periodic[2] else 1)], float)
    out = []
    while sum(len(o) for o in out) < n:
        c = rng.random((4 * n, 3)) * hi
        v = np.array([_val(phi, *r, *map(bool, periodic)) for r in c]) if len(c) < 2000 else _vals(phi, c, periodic)
        out.append(c[v <= 0])
    return np.concatenate(out)[:n]


@nb.njit(cache=True, parallel=True)
def _vals_nb(phi, c, px, py, pz):
    v = np.empty(c.shape[0])
    for i in nb.prange(c.shape[0]):
        v[i] = _val(phi, c[i, 0], c[i, 1], c[i, 2], px, py, pz)
    return v


def _vals(phi, c, periodic):
    return _vals_nb(phi, c, *map(bool, periodic))


def simulate(phi, h, x0, D=1.0, dt=0.25, nrec=100, steps_per_rec=1000, periodic=(False, False, False), seed=0):
    """Reflecting Brownian motion in {phi < 0}. phi: (nx, ny, nz) level-set samples at voxel centres (index = voxel
    coordinate, voxel size h nm); x0: (n, 3) starts in voxel units. Returns times (nrec+1,) and unwrapped positions
    in nm, shape (nrec+1, n, 3). Step sigma = sqrt(2 D dt) must be well below the gap width."""
    if seed is not None:
        _seed(seed)
    pos = np.array(x0, float)
    sig = math.sqrt(2.0 * D * dt) / h
    out = np.empty((nrec + 1,) + pos.shape); out[0] = pos * h
    for r in range(nrec):
        _advance(phi.astype(np.float64), pos, steps_per_rec, sig, *map(bool, periodic))
        out[r + 1] = pos * h
    return np.arange(nrec + 1) * steps_per_rec * dt, out


@nb.njit(cache=True)
def _seed_nb(s):
    np.random.seed(s)


def _seed(s):
    _seed_nb(s)


def _lag_msd(pos, lags):
    """Mean-square displacement per axis at the given record lags, averaged over particles AND time origins."""
    T = pos.shape[0]
    return np.array([((pos[l:] - pos[:T - l]) ** 2).mean(axis=(0, 1)) for l in lags])


def effective_diffusivity(t, pos, window=(0.25, 0.5), groups=10):
    """D_eff per axis and its standard error, from the unwrapped mean-square displacement averaged over time
    origins, linear fit MSD = 2 D t + c over lags window[0]..window[1] of the record. The error is the scatter of
    the estimate over `groups` independent particle groups. Returns (D, sem), each of shape (3,)."""
    T = len(t) - 1
    lags = np.arange(max(int(window[0] * T), 2), int(window[1] * T) + 1)
    est = []
    for g in np.array_split(np.arange(pos.shape[1]), groups):
        m = _lag_msd(pos[:, g], lags)
        est.append([np.polyfit(t[lags], m[:, a], 1)[0] / 2.0 for a in range(3)])
    est = np.array(est)
    return est.mean(0), est.std(0, ddof=1) / math.sqrt(groups)
