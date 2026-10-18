"""Clearance of tracers released from a candidate cleft into an absorbing sink cell (real tile crop).

Same crop as run_synapse_source.py (tile 170398_60741, rows/cols 0-256, sections 0-40, 8 nm voxels). The sink is the
membrane of one cell, chosen as the cell with the most contact area at least --min-dist nm from the source (or pass
--sink LABEL). Cases: perfect absorption (p = 1 per wall hit) for several gap widths w, partial absorption
(kappa, p = kappa sqrt(pi dt / D)) and a breathing ramp. Metrics: fraction absorbed versus time, median/mean
absorption time among the absorbed, and where on the sink the tracers land.
Caches the label volume in results/synapse_source/vol_<tile>.npz and reads the margin from geom_<tile>.npz
(written by run_synapse_sweep.py).

usage: python examples/run_synapse_sink.py [--sink LABEL] [--min-dist 600] [--tmax 2e6] [--n 1500] [--ws 16 20 30 40]
"""
import os, sys, argparse, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
from zmorph.synapse import release_positions
from zmorph.sinks import sink_patch_map, absorption_probability, pick_sink, first_passage
from zmorph.bd import simulate_dynamic

ap = argparse.ArgumentParser()
ap.add_argument("--tile", default="170398_60741"); ap.add_argument("--sections", type=int, nargs=2, default=[0, 41])
ap.add_argument("--crop", type=int, nargs=4, default=[0, 256, 0, 256])
ap.add_argument("--sink", type=int, default=None); ap.add_argument("--min-dist", type=float, default=600.0)
ap.add_argument("--w0", type=float, default=20.0); ap.add_argument("--ws", type=float, nargs="+", default=[16, 20, 30, 40])
ap.add_argument("--kappa", type=float, default=0.08); ap.add_argument("--n", type=int, default=1500)
ap.add_argument("--tmax", type=float, default=2e6); ap.add_argument("--list", action="store_true")
a = ap.parse_args()
H = 8.0; dt = 0.5; nrec = 40; spr = int(a.tmax / dt / nrec)
out = os.path.join(ROOT, "results", "synapse_source")
g = np.load(os.path.join(out, f"geom_{a.tile}.npz")); m = g["m"].astype(np.float64); center = g["center"]; pts = g["points"]
vp = os.path.join(out, f"vol_{a.tile}.npz")
if os.path.exists(vp):
    vol = np.load(vp)["vol"].astype(np.int64)
else:
    from zmorph.interp import reconstruct
    d = np.load(os.path.join(ROOT, "data", f"tile_{a.tile}.npz")); y0, y1, x0, x1 = a.crop
    vol = reconstruct([d["vol"][i, y0:y1, x0:x1].astype(np.int64) for i in range(*a.sections)], 5, "sdf", align=True)
    np.savez_compressed(vp, vol=vol.astype(np.uint32))
assert vol.shape == m.shape, (vol.shape, m.shape)

cands = pick_sink(vol, H, center, a.min_dist, top=8)
print("sink candidates (label, contact area um^2, distance nm):", [(l, round(ar * 1e-6, 2), round(dd)) for l, ar, dd in cands], flush=True)
if a.list:
    sys.exit()
sink = a.sink if a.sink is not None else cands[0][0]
patch = sink_patch_map(vol, sink)
sink_pts = np.argwhere(patch > 0) * H
print(f"sink label {sink}: {int(patch.sum())} voxels, centroid (nm) {np.round(sink_pts.mean(0))}, source centre {np.round(center)}", flush=True)


class _P:
    points = pts


x0 = release_positions(_P, m, H, a.w0, a.n, seed=1) / H
p_kappa = absorption_probability(a.kappa, dt)
cases = {f"perfect w={w:g}": (lambda t, w=w: w, 1.0) for w in a.ws}
cases[f"kappa={a.kappa:g} (p={p_kappa:.2f}) w={a.w0:g}"] = (lambda t: a.w0, p_kappa)
cases["perfect ramp w0->1.6 w0"] = (lambda t: a.w0 * (1 + 0.6 * t / a.tmax), 1.0)
rows = []
for name, (wf, p) in cases.items():
    t0 = time.time()
    r = simulate_dynamic(m, H, x0.copy(), wf, 1.0, dt, nrec=nrec, steps_per_rec=spr, patch=patch, p_abs=[0.0, p], seed=3)
    fp = first_passage(r)
    got = r["absorbed"]
    dsrc = float(np.median(np.linalg.norm(r["abs_pos"][got] - center, axis=1))) if got.any() else None
    row = dict(case=name, p=p, frac_absorbed=fp["frac_absorbed"], t_median=fp["t_median"], t_mean=fp["t_mean"], t_q10=fp["t_q10"],
               median_dist_from_source_nm=dsrc, absorbed_frac_curve=fp["absorbed_frac"].tolist(), t=fp["t"].tolist())
    rows.append(row)
    print(f"{name:34s} absorbed {row['frac_absorbed']:.3f}  t_q10 {row['t_q10']}  t_median {row['t_median']}  "
          f"({time.time() - t0:.0f}s)", flush=True)
    json.dump(dict(tile=a.tile, sink=sink, sink_candidates=[(int(l), float(ar), float(dd)) for l, ar, dd in cands],
                   source_center=center.tolist(), n=a.n, tmax=a.tmax, rows=rows),
              open(os.path.join(out, f"sink_{a.tile}.json"), "w"), indent=1)
with open(os.path.join(out, f"sink_{a.tile}.md"), "w") as f:
    f.write(f"# sink clearance, tile {a.tile}, sink label {sink}, N={a.n}, tmax={a.tmax:g} (time units nm^2/D)\n\n"
            "| case | absorbed by tmax | 10% quantile of t | median t | mean t |\n|---|---|---|---|---|\n")
    fmt = lambda v: "n/a" if v is None else f"{v:.3g}"
    for r in rows:
        f.write(f"| {r['case']} | {r['frac_absorbed']:.3f} | {fmt(r['t_q10'])} | {fmt(r['t_median'])} | {fmt(r['t_mean'])} |\n")
