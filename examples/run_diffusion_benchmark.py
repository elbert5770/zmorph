"""Diffusion benchmark for the gap between cells: Brownian dynamics (BD) on the implicit signed-distance geometry vs
continuum solvers on voxels and on a surface mesh, against analytic answers. D = 1, lengths in nm.

  slab     reflecting-wall check: uniform density across a gap, in-plane D = D_free, versus the time step
  lattice  periodic 2x2x2 box of cubic cells, gap w on every interface: tortuosity lambda^2 = D / D_ECS
           (BD, analytic and from labels; voxel finite volumes at several voxel sizes; thin-sheet limit 1.5)
  sphere   ball in socket: gap shell of thickness w around a sphere of radius R, maximal out-of-plane bend. Slowest
           decay rate of diffusion in the shell: BD, voxel finite volumes, triangle-surface cotangent Laplacian
           (thin-film limit), against the exact spherical-shell solution

usage: python examples/run_diffusion_benchmark.py [slab|lattice|sphere|all]   -> results/diffusion_benchmark.json (+ .md, .png via --report)
"""
import os, sys, json, time, math
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
from skimage.measure import marching_cubes
from zmorph.bd import margin_field, simulate, sample_ecs, effective_diffusivity, _sdf_subvoxel
from zmorph.diffusion_ref import tortuosity_fv, fv_decay_rate, cotan_decay_rate, shell_exact_rate

OUT = os.path.join(ROOT, "results", "diffusion_benchmark.json")
RES = json.load(open(OUT)) if os.path.exists(OUT) else {}
save = lambda: json.dump(RES, open(OUT, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
log = lambda *a: print(*a, flush=True)
H = 8.0                                              # voxel size (nm) of the data


# ----------------------------------------------------------------------------- slab
def slab():
    w, nz = 20.0, 13
    z = (np.arange(nz) - nz // 2) * H
    phi = np.broadcast_to((2 * np.abs(z) - w)[None, None, :], (8, 8, nz)).copy()
    out = {}
    for sig in (0.5, 1.0, 2.0, 4.0):                 # step length in nm
        dt = sig ** 2 / 2
        x0 = sample_ecs(phi, H, 20000, periodic=(True, True, False), seed=1)
        t, pos = simulate(phi, H, x0, 1.0, dt, nrec=40, steps_per_rec=int(4000 / dt / 40), periodic=(True, True, False), seed=2)
        zn = pos[-1, :, 2] - (nz // 2) * H
        cnt, _ = np.histogram(zn, bins=10, range=(-w / 2, w / 2))
        D, sem = effective_diffusivity(t, pos)
        near = np.mean(np.abs(zn) > 0.8 * w / 2) / 0.2          # density in the outer 20% of the gap relative to uniform
        out[f"sigma={sig}"] = dict(counts=cnt.tolist(), outside=int((np.abs(zn) > w / 2).sum()), wall_density_ratio=float(near),
                                   D_par=float(D[:2].mean()), D_par_sem=float(sem[:2].mean()))
        log("slab", sig, out[f"sigma={sig}"]["wall_density_ratio"], out[f"sigma={sig}"]["D_par"])
    RES["slab"] = out; save()


# ----------------------------------------------------------------------------- lattice
def lattice_phi(a, w, h, n=2):
    """phi = 2*dist-to-nearest-cell-face - w on a periodic n x n x n box of cubic cells of side a (voxel centres)."""
    m = int(round(a * n / h)); ax = (np.arange(m) + 0.5) * h
    d = np.abs(((ax + a / 2) % a) - a / 2)           # distance to the nearest plane x = k a  (voxel centres at (i+.5)h)
    dist = np.minimum.reduce(np.meshgrid(d, d, d, indexing="ij"))
    return 2 * dist - w


def lattice_labels(a, h, n=2):
    m = int(round(a * n / h)); c = ((np.arange(m) + 0.5) * h // a).astype(int)
    return (c[:, None, None] * n + c[None, :, None]) * n + c[None, None, :]


def lattice():
    a, N = 200.0, 6000
    out = {}
    for w in (12.0, 20.0, 32.0):
        r = dict(w=w, alpha=float(1 - ((a - w) / a) ** 3), thin_limit=1.5)
        sig = w / 16; dt = sig ** 2 / 2
        t_a = a * a / 2; T = 12 * t_a; nrec = 240
        for tag, phi in (("bd_analytic", lattice_phi(a, w, H)),) + ((("bd_labels", margin_field(lattice_labels(a, H), H, periodic=(True,) * 3) - w),) if w == 20.0 else ()):
            t0 = time.time()
            x0 = sample_ecs(phi, H, N, periodic=(True,) * 3, seed=3)
            t, pos = simulate(phi, H, x0, 1.0, dt, nrec=nrec, steps_per_rec=int(T / dt / nrec), periodic=(True,) * 3, seed=4)
            D, sem = effective_diffusivity(t, pos, window=(0.15, 0.4))
            r[tag] = dict(lam2=float(1 / D.mean()), lam2_sem=float(sem.mean() / D.mean() ** 2), D_axes=D.tolist(), seconds=round(time.time() - t0))
            log("lattice w", w, tag, r[tag]["lam2"], "+-", r[tag]["lam2_sem"], r[tag]["seconds"], "s")
        for hv in (8.0, 4.0) + ((2.0,) if w == 20.0 else ()):
            t0 = time.time(); f = tortuosity_fv(lattice_phi(a, w, hv) < 0, hv)
            r[f"fv_h{hv:g}"] = dict(lam2=float(f["lam2"]), alpha=float(f["alpha"]), cg=f["cg_info"], seconds=round(time.time() - t0))
            log("lattice w", w, f"fv h={hv}", f["lam2"], "alpha", f["alpha"], round(time.time() - t0), "s")
        out[f"w={w:g}"] = r
        RES["lattice"] = out; save()


# ----------------------------------------------------------------------------- sphere
R, W = 200.0, 20.0


def sphere_grid(h, pad=24.0):
    n = int(math.ceil(2 * (R + W / 2 + pad) / h)) | 1
    c = n // 2
    i = np.arange(n) - c
    X, Y, Z = np.meshgrid(i, i, i, indexing="ij")
    return np.sqrt(X ** 2 + Y ** 2 + Z ** 2) * h, c


def bd_decay(phi, h, c, N=20000, seed=5):
    gam = shell_exact_rate(R - W / 2, R + W / 2)
    sig = W / 16; dt = sig ** 2 / 2
    nrec = 60; steps = int(1.5 / gam / dt / nrec)
    x0 = sample_ecs(phi, h, N, seed=seed)
    t, pos = simulate(phi, h, x0, 1.0, dt, nrec=nrec, steps_per_rec=steps, seed=seed + 1)
    p = pos - c * h
    n = p / np.linalg.norm(p, axis=2, keepdims=True)
    C = (n * n[0]).sum(2)                                     # (T, N) cos of the angle to the start
    est = []
    for g in np.array_split(np.arange(N), 10):
        cm = C[:, g].mean(1); s = (t <= 0.9 / gam) & (cm > 0.05)
        est.append(-np.polyfit(t[s], np.log(cm[s]), 1)[0])
    est = np.array(est)
    return float(est.mean()), float(est.std(ddof=1) / math.sqrt(10)), t, C.mean(1)


def sphere():
    out = {"exact_rate": shell_exact_rate(R - W / 2, R + W / 2), "thin_rate": 2 / R ** 2}
    gam = out["exact_rate"]
    for tag, hh, kind in (("bd_analytic_h8", 8.0, "analytic"), ("bd_analytic_h2", 2.0, "analytic"), ("bd_labels_h8", 8.0, "labels"), ("bd_labels_h8_sigma1", 8.0, "labels1")):
        t0 = time.time()
        r, c = sphere_grid(hh)
        if kind == "analytic":
            phi = 2 * np.abs(r - R) - W
        else:
            ball = r < R
            phi = margin_field(ball.astype(int), hh, sigma=1.0 if kind == "labels1" else 0.0) - W
        k, sem, t, C = bd_decay(phi, hh, c)
        out[tag] = dict(rate=k, sem=sem, rel_err=k / gam - 1, seconds=round(time.time() - t0))
        log("sphere", tag, k / gam - 1, "+-", sem / gam, round(time.time() - t0), "s")
        RES["sphere"] = out; save()
    for hv in (8.0, 4.0, 3.0):
        t0 = time.time(); r, c = sphere_grid(hv)
        mask = np.abs(r - R) < W / 2
        k, _ = fv_decay_rate(mask, hv)
        out[f"fv_h{hv:g}"] = dict(rate=float(k), rel_err=float(k / gam - 1), nodes=int(mask.sum()), seconds=round(time.time() - t0))
        log("sphere fv", hv, k / gam - 1, int(mask.sum()), round(time.time() - t0), "s")
        RES["sphere"] = out; save()
    for tag, hv, kind in (("mesh_analytic_h8", 8.0, "analytic"), ("mesh_analytic_h4", 4.0, "analytic"), ("mesh_analytic_h2", 2.0, "analytic"),
                          ("mesh_labels_h8", 8.0, "labels"), ("mesh_labels_h4", 4.0, "labels")):
        t0 = time.time(); r, c = sphere_grid(hv, pad=8.0)
        f = (R - r) if kind == "analytic" else _sdf_subvoxel(r < R) * hv
        v, fc, _, _ = marching_cubes(f, 0.0, spacing=(hv,) * 3)
        k, _ = cotan_decay_rate(v, fc)
        out[tag] = dict(rate=float(k), rel_err=float(k / gam - 1), verts=len(v), seconds=round(time.time() - t0))
        log("sphere mesh", tag, k / gam - 1, len(v), round(time.time() - t0), "s")
        RES["sphere"] = out; save()


def sphere_mesh_smooth():
    """Surface meshes from voxelised labels, with the two smoothing remedies: a Gaussian-smoothed distance field
    (as sdf_align uses) and Taubin smoothing of the mesh."""
    import scipy.ndimage as ndi
    from zmorph.mesh import taubin_joint
    out = RES["sphere"]; gam = out["exact_rate"]
    for hv in (8.0, 4.0):
        r, c = sphere_grid(hv, pad=8.0)
        f0 = _sdf_subvoxel(r < R)
        for tag, f, tb in ((f"mesh_labels_sigma1_h{hv:g}", ndi.gaussian_filter(f0, 1.0) * hv, False),
                           (f"mesh_labels_taubin_h{hv:g}", f0 * hv, True),
                           (f"mesh_labels_sigma1_taubin_h{hv:g}", ndi.gaussian_filter(f0, 1.0) * hv, True)):
            v, fc, _, _ = marching_cubes(f, 0.0, spacing=(hv,) * 3)
            if tb:
                v, fc = taubin_joint({1: (v, fc)})[1]
            k, _ = cotan_decay_rate(v, fc)
            out[tag] = dict(rate=float(k), rel_err=float(k / gam - 1), verts=len(v))
            log("sphere mesh", tag, k / gam - 1, len(v))
    save()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("slab", slab), ("lattice", lattice), ("sphere", sphere), ("meshsmooth", sphere_mesh_smooth)):
        if which == name or (which == "all" and name != "meshsmooth"): fn()
