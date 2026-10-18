"""Time and cross-check the GPU Brownian dynamics against the CPU (numba) reference.

1. Throughput: tracer-steps per second on the CPU and the GPU for several tracer counts, in a planar slab (cheap
   geometry) and, if results/synapse_source/geom_<tile>.npz exists, on the real tile crop.
2. Agreement: the same absorbing-sink problem on both backends (tile crop, sink label from vol_<tile>.npz) with
   --check-n tracers; prints the absorbed fraction and the median absorption time with binomial errors. The two
   agree when the difference is within about two standard errors.

usage: python examples/bench_bd_gpu.py [--ns 1000 10000 100000 1000000] [--steps 2000] [--check-n 50000] [--no-cpu]
"""
import os, sys, argparse, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
from zmorph.bd import simulate_dynamic, sample_ecs
from zmorph.bd_gpu import simulate_dynamic_gpu
from zmorph.sinks import sink_patch_map

ap = argparse.ArgumentParser()
ap.add_argument("--ns", type=int, nargs="+", default=[1000, 10000, 100000, 1000000])
ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--w", type=float, default=20.0)
ap.add_argument("--check-n", type=int, default=50000); ap.add_argument("--no-cpu", action="store_true")
ap.add_argument("--tile", default="170398_60741"); ap.add_argument("--sink", type=int, default=653)
a = ap.parse_args()
H = 8.0; dt = 0.5; nrec = 4; spr = a.steps // nrec
from numba import cuda
print("GPU:", cuda.get_current_device().name if cuda.is_available() else "none (simulator)")

geoms = {}
z = (np.arange(13) - 6) * H
geoms["slab (periodic xy)"] = (np.broadcast_to((2 * np.abs(z))[None, None, :], (8, 8, 13)).copy(), (True, True, False), None)
gp = os.path.join(ROOT, "results", "synapse_source", f"geom_{a.tile}.npz")
if os.path.exists(gp):
    geoms["tile crop 201x256x256"] = (np.load(gp)["m"].astype(np.float64), (False, False, False), None)

print(f"{'geometry':26s} {'N':>9s} {'CPU Msteps/s':>13s} {'GPU Msteps/s':>13s} {'speed-up':>9s}")
for name, (m, per, _) in geoms.items():
    for n in a.ns:
        x0 = sample_ecs(m - a.w, H, n, periodic=per, seed=1)
        t = time.time(); simulate_dynamic_gpu(m, H, x0, lambda t: a.w, 1.0, dt, nrec=nrec, steps_per_rec=spr, periodic=per,
                                              record_pos=False, seed=1); tg = time.time() - t
        if n == a.ns[0]:            # first call compiles; repeat for the timing
            t = time.time(); simulate_dynamic_gpu(m, H, x0, lambda t: a.w, 1.0, dt, nrec=nrec, steps_per_rec=spr, periodic=per,
                                                  record_pos=False, seed=1); tg = time.time() - t
        rg = n * a.steps / tg / 1e6
        if a.no_cpu or n > 200000:
            print(f"{name:26s} {n:9d} {'-':>13s} {rg:13.1f} {'-':>9s}", flush=True); continue
        simulate_dynamic(m, H, x0[:100], lambda t: a.w, 1.0, dt, nrec=1, steps_per_rec=10, periodic=per)     # compile
        t = time.time(); simulate_dynamic(m, H, x0, lambda t: a.w, 1.0, dt, nrec=nrec, steps_per_rec=spr, periodic=per, seed=1)
        rc = n * a.steps / (time.time() - t) / 1e6
        print(f"{name:26s} {n:9d} {rc:13.1f} {rg:13.1f} {rg / rc:9.1f}", flush=True)

vp = os.path.join(ROOT, "results", "synapse_source", f"vol_{a.tile}.npz")
if os.path.exists(gp) and os.path.exists(vp):
    from zmorph.synapse import release_positions
    g = np.load(gp); m = g["m"].astype(np.float64); pts = g["points"]
    patch = sink_patch_map(np.load(vp)["vol"].astype(np.int64), a.sink)

    class _P:
        points = pts

    n = a.check_n; tmax = 4e5; spr = int(tmax / dt / 20)
    x0 = release_positions(_P, m, H, a.w, n, seed=1) / H
    kw = dict(D=1.0, dt=dt, nrec=20, steps_per_rec=spr, patch=patch, p_abs=[0.0, 1.0])
    res = {}
    if not a.no_cpu:
        res["cpu"] = simulate_dynamic(m, H, x0.copy(), lambda t: a.w, seed=3, **kw)
    res["gpu"] = simulate_dynamic_gpu(m, H, x0.copy(), lambda t: a.w, seed=3, record_pos=False, **kw)
    print(f"\nagreement, sink label {a.sink}, w = {a.w:g}, N = {n}, tmax = {tmax:g}")
    for k, r in res.items():
        f = r["absorbed"].mean(); se = np.sqrt(f * (1 - f) / n)
        at = r["abs_time"][r["absorbed"]]
        print(f"  {k}: absorbed {f:.4f} +- {se:.4f}   median time {np.median(at) if len(at) else float('nan'):.0f}")
