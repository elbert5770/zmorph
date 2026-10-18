"""Tile-15 ECS: BD with periodic boundaries on the same mask as the FV cell problem.

Loads results/tile15/volume.npz (from run_tile15.py) and, for each w, runs BD on { m < w } with all three axes
periodic (the same wrapped crop the finite-volume solver sees) and compares D_BD per axis with 1/lam2 from the FV
cell problem stored in results/tile15/summary_<tile>.json. Writes results/tile15/bd_periodic_<tile>.{json,md}.

usage: python examples/run_tile15_bd_periodic.py [--tile 170398_60741] [--np 3000] [--tmax 6e4]
"""
import os, sys, argparse, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
from zmorph.bd import margin_field, simulate_dynamic, sample_ecs, effective_diffusivity

ap = argparse.ArgumentParser()
ap.add_argument("--tile", default="170398_60741"); ap.add_argument("--np", type=int, default=3000, dest="npart")
ap.add_argument("--tmax", type=float, default=6e4)
a = ap.parse_args()
H = 8.0; PER = (True, True, True)
out = os.path.join(ROOT, "results", "tile15")
vol = np.load(os.path.join(out, "volume.npz"))["vol"].astype(np.int64)
S = json.load(open(os.path.join(out, f"summary_{a.tile}.json")))
fv = {r["w"]: r for r in S["ecs"]}
m = margin_field(vol, H, cap=64.0)                       # same non-periodic margin the FV mask came from
rows = []
for w in sorted(fv):
    x0 = sample_ecs(m - w, H, a.npart, periodic=PER, seed=1)
    dt = 0.5; nrec = 40; spr = int(a.tmax / dt / nrec)
    r = simulate_dynamic(m, H, x0, lambda t: w, 1.0, dt, nrec=nrec, steps_per_rec=spr, periodic=PER, seed=2)
    D, sem = effective_diffusivity(r["t"], r["pos"], window=(0.1, 0.5), groups=10)
    f = fv[w]["inv_lam2_fv"]
    rows.append(dict(w=w, alpha=fv[w]["ecs_fraction"], fv=f, bd=D.tolist(), bd_sem=sem.tolist(),
                     ratio=[float(D[i] / f[i]) for i in range(3)]))
    print(f"w={w:3g}  FV (z,y,x)={np.round(f, 3)}  BD periodic={np.round(D, 3)} +-{np.round(sem, 3)}  "
          f"ratio={np.round(rows[-1]['ratio'], 2)}", flush=True)
json.dump(dict(tile=a.tile, rows=rows), open(os.path.join(out, f"bd_periodic_{a.tile}.json"), "w"), indent=1)
with open(os.path.join(out, f"bd_periodic_{a.tile}.md"), "w") as fh:
    fh.write("| w (nm) | ECS fraction | FV 1/lam2 (z, y, x) | BD periodic D (z, y, x) | BD / FV (z, y, x) |\n|---|---|---|---|---|\n")
    for r in rows:
        j = lambda v: ", ".join(f"{x:.3f}" for x in v)
        fh.write(f"| {r['w']:g} | {r['alpha']:.3f} | {j(r['fv'])} | {j(r['bd'])} | {', '.join(f'{x:.2f}' for x in r['ratio'])} |\n")
