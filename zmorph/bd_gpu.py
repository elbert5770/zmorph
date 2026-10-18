"""GPU (numba.cuda) version of bd.simulate_dynamic: reflecting Brownian motion in the moving gap { m < w(t) } with
absorbing patches. One thread per tracer, float32 throughout (consumer and workstation GPUs run float64 far slower),
Philox-free xoroshiro128+ streams from numba.cuda.random. The logic is a line-by-line port of bd._advance2, so the
CPU code remains the reference; results agree statistically, not bit for bit (different random streams, float32).

  simulate_dynamic_gpu(m, h, x0, w_of_t, ...)   same arguments and return dict as bd.simulate_dynamic, plus
                                                 `record_pos=False` to keep only the first and last positions
                                                 (a million tracers x 40 records would otherwise sit in host memory)

Needs a CUDA GPU and numba with CUDA support. With NUMBA_ENABLE_CUDASIM=1 the kernel runs in the pure-Python CUDA
simulator (very slow, for logic tests only). Positions and lengths are in voxel units inside the kernel, as in bd.py.
"""
import math
import numpy as np
import numba
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32, xoroshiro128p_uniform_float32

F = numba.float32


@cuda.jit(device=True, fastmath=True)
def _val(phi, x, y, z, px, py, pz):
    nx, ny, nz = phi.shape
    ix, iy, iz = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
    fx, fy, fz = x - F(ix), y - F(iy), z - F(iz)
    if px:
        i0 = ix % nx; i1 = (i0 + 1) % nx
    else:
        if ix < 0 or ix + 1 > nx - 1:
            return F(1e9)
        i0 = ix; i1 = ix + 1
    if py:
        j0 = iy % ny; j1 = (j0 + 1) % ny
    else:
        if iy < 0 or iy + 1 > ny - 1:
            return F(1e9)
        j0 = iy; j1 = iy + 1
    if pz:
        k0 = iz % nz; k1 = (k0 + 1) % nz
    else:
        if iz < 0 or iz + 1 > nz - 1:
            return F(1e9)
        k0 = iz; k1 = iz + 1
    one = F(1.0)
    c00 = phi[i0, j0, k0] * (one - fx) + phi[i1, j0, k0] * fx
    c10 = phi[i0, j1, k0] * (one - fx) + phi[i1, j1, k0] * fx
    c01 = phi[i0, j0, k1] * (one - fx) + phi[i1, j0, k1] * fx
    c11 = phi[i0, j1, k1] * (one - fx) + phi[i1, j1, k1] * fx
    return (c00 * (one - fy) + c10 * fy) * (one - fz) + (c01 * (one - fy) + c11 * fy) * fz


@cuda.jit(device=True, fastmath=True)
def _gradfull(phi, x, y, z, px, py, pz):
    e = F(0.5)
    gx = _val(phi, x + e, y, z, px, py, pz) - _val(phi, x - e, y, z, px, py, pz)
    gy = _val(phi, x, y + e, z, px, py, pz) - _val(phi, x, y - e, z, px, py, pz)
    gz = _val(phi, x, y, z + e, px, py, pz) - _val(phi, x, y, z - e, px, py, pz)
    n = math.sqrt(gx * gx + gy * gy + gz * gz)
    if n < F(1e-6) or n > F(1e6):
        return F(0.0), F(0.0), F(0.0), F(0.0)
    return gx / n, gy / n, gz / n, n


@cuda.jit(device=True)
def _patch_at(patch, x, y, z, px, py, pz):
    nx, ny, nz = patch.shape
    i, j, k = int(math.floor(x + F(0.5))), int(math.floor(y + F(0.5))), int(math.floor(z + F(0.5)))
    if px:
        i = i % nx
    elif i < 0 or i >= nx:
        return 0
    if py:
        j = j % ny
    elif j < 0 or j >= ny:
        return 0
    if pz:
        k = k % nz
    elif k < 0 or k >= nz:
        return 0
    return patch[i, j, k]


@cuda.jit(fastmath=True)
def _advance2_cu(m, pos, alive, nsteps, step0, sig, w0, w1, px, py, pz, patch, pabs, imp_s, imp_v,
                 a_step, a_patch, a_pos, rng):
    p = cuda.grid(1)
    if p >= pos.shape[0]:
        return
    if alive[p] == 0:
        return
    zero, one, two = F(0.0), F(1.0), F(2.0)
    x, y, z = pos[p, 0], pos[p, 1], pos[p, 2]
    for s in range(nsteps):
        w = w0 + (w1 - w0) * F(s + 1) / F(nsteps)
        v = _val(m, x, y, z, px, py, pz) - w
        tries = 0
        while v > zero and tries < 6:
            gx, gy, gz, gn = _gradfull(m, x, y, z, px, py, pz)
            if gn == zero:
                break
            d = F(1.02) * v / gn
            x, y, z = x - d * gx, y - d * gy, z - d * gz
            imp_s[p] += d; imp_v[p, 0] -= d * gx; imp_v[p, 1] -= d * gy; imp_v[p, 2] -= d * gz
            v = _val(m, x, y, z, px, py, pz) - w
            tries += 1
        dx = sig * xoroshiro128p_normal_float32(rng, p)
        dy = sig * xoroshiro128p_normal_float32(rng, p)
        dz = sig * xoroshiro128p_normal_float32(rng, p)
        xn, yn, zn = x + dx, y + dy, z + dz
        if _val(m, xn, yn, zn, px, py, pz) - w <= zero:
            x, y, z = xn, yn, zn
            continue
        lo, hi = zero, one
        for _it in range(12):
            mid = F(0.5) * (lo + hi)
            if _val(m, x + mid * dx, y + mid * dy, z + mid * dz, px, py, pz) - w <= zero:
                lo = mid
            else:
                hi = mid
        xc, yc, zc = x + lo * dx, y + lo * dy, z + lo * dz
        h2 = F(0.5) * (hi - lo)
        q = _patch_at(patch, xc + h2 * dx, yc + h2 * dy, zc + h2 * dz, px, py, pz)
        if q > 0 and xoroshiro128p_uniform_float32(rng, p) < pabs[q]:
            alive[p] = 0
            a_step[p] = step0 + s; a_patch[p] = q
            a_pos[p, 0] = xc; a_pos[p, 1] = yc; a_pos[p, 2] = zc
            x, y, z = xc, yc, zc
            break
        nx_, ny_, nz_, gn = _gradfull(m, xc, yc, zc, px, py, pz)
        rx, ry, rz = (one - lo) * dx, (one - lo) * dy, (one - lo) * dz
        dot = rx * nx_ + ry * ny_ + rz * nz_
        xr, yr, zr = xc + rx - two * dot * nx_, yc + ry - two * dot * ny_, zc + rz - two * dot * nz_
        if _val(m, xr, yr, zr, px, py, pz) - w <= zero:
            x, y, z = xr, yr, zr
            imp_s[p] += two * abs(dot); imp_v[p, 0] -= two * dot * nx_; imp_v[p, 1] -= two * dot * ny_; imp_v[p, 2] -= two * dot * nz_
        else:
            x, y, z = xc, yc, zc
    pos[p, 0] = x; pos[p, 1] = y; pos[p, 2] = z


def simulate_dynamic_gpu(m, h, x0, w_of_t, D=1.0, dt=0.25, nrec=100, steps_per_rec=1000, periodic=(False, False, False),
                         patch=None, p_abs=None, seed=0, threads=128, record_pos=True):
    """GPU counterpart of bd.simulate_dynamic (same arguments, same return dict). `x0` in voxel units, `m` in nm."""
    n = len(x0)
    sig = F(math.sqrt(2.0 * D * dt) / h)
    patch = np.zeros(m.shape, np.int32) if patch is None else np.ascontiguousarray(patch, np.int32)
    npatch = int(patch.max()) + 1
    pabs = np.zeros(max(npatch, 2), np.float32) if p_abs is None else np.asarray(p_abs, np.float32)
    pabs = np.concatenate([pabs, np.zeros(max(0, npatch - len(pabs)), np.float32)])
    t = np.arange(nrec + 1) * steps_per_rec * dt
    wv = np.atleast_1d(np.array([w_of_t(tt) for tt in t], float)) / h
    d_m = cuda.to_device(np.ascontiguousarray(m / h, np.float32))
    d_patch = cuda.to_device(patch); d_pabs = cuda.to_device(pabs)
    d_pos = cuda.to_device(np.ascontiguousarray(x0, np.float32))
    d_alive = cuda.to_device(np.ones(n, np.uint8))
    d_a_step = cuda.to_device(-np.ones(n, np.int64)); d_a_patch = cuda.to_device(np.zeros(n, np.int32))
    d_a_pos = cuda.to_device(np.zeros((n, 3), np.float32))
    rng = create_xoroshiro128p_states(n, seed=0 if seed is None else int(seed))
    pos0 = d_pos.copy_to_host() * h
    out = [pos0] if record_pos else None
    alive_hist = [np.ones(n, bool)]
    push = np.zeros((nrec, n)); vec = np.zeros((nrec, 3))
    blocks = (n + threads - 1) // threads
    for r in range(nrec):
        d_imp_s = cuda.to_device(np.zeros(n, np.float32)); d_imp_v = cuda.to_device(np.zeros((n, 3), np.float32))
        _advance2_cu[blocks, threads](d_m, d_pos, d_alive, steps_per_rec, r * steps_per_rec, sig, F(wv[r]), F(wv[r + 1]),
                                      bool(periodic[0]), bool(periodic[1]), bool(periodic[2]), d_patch, d_pabs,
                                      d_imp_s, d_imp_v, d_a_step, d_a_patch, d_a_pos, rng)
        cuda.synchronize()
        if record_pos:
            out.append(d_pos.copy_to_host().astype(np.float64) * h)
        alive_hist.append(d_alive.copy_to_host().astype(bool))
        push[r] = d_imp_s.copy_to_host() * h; vec[r] = d_imp_v.copy_to_host().sum(0) * h
    pos_last = d_pos.copy_to_host().astype(np.float64) * h
    pos = np.stack(out) if record_pos else np.stack([pos0, pos_last])
    a_step = d_a_step.copy_to_host(); got = a_step >= 0
    res = dict(t=t, pos=pos, alive=np.stack(alive_hist), wall_push=push, wall_vec=vec, w=wv * h)
    res.update(abs_time=np.where(got, (a_step + 1) * dt, np.nan), abs_patch=d_a_patch.copy_to_host(),
               abs_pos=d_a_pos.copy_to_host().astype(np.float64) * h, absorbed=got)
    return res
