"""Grid-refinement check of the finite-volume cell problem (tile 15).

The ECS mask is { m < w } with m the margin field. Refinement by factor f samples m by trilinear interpolation onto an
f-times finer periodic grid (scipy zoom, grid_mode, wrap), so the mask follows the same level set BD uses, with a
smaller staircase. Tortuosity is the periodic cell problem (zmorph.diffusion_ref.tortuosity_fv). Runs
  - the full crop at f = 1, 2 for w in --ws
  - a sub-box (rows and columns 0..--box) at f = 1, 2, 3 for convergence
and writes results/tile15/fv_refined_<tile>.json and .md. Output is 1/lam2 per axis (z, y, x).

usage: python examples/run_tile15_fv_refined.py [--ws 20 40] [--box 128] [--tol 1e-7]
"""
import os, sys, argparse, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
import scipy.ndimage as ndi
from zmorph.bd import margin_field
from zmorph.diffusion_ref import tortuosity_fv

ap = argparse.ArgumentParser()
ap.add_argument("--tile", default="170398_60741"); ap.add_argument("--ws", type=float, nargs="+", default=[20, 40])
ap.add_argument("--box", type=int, default=128); ap.add_argument("--tol", type=float, default=1e-7)
ap.add_argument("--full-factors", type=int, nargs="+", default=[1, 2]); ap.add_argument("--box-factors", type=int, nargs="+", default=[1, 2, 3])
a = ap.parse_args()
H = 8.0
out = os.path.join(ROOT, "results", "tile15")
vol = np.load(os.path.join(out, "volume.npz"))["vol"].astype(np.int64)
m = margin_field(vol, H, cap=64.0)
rows = []


def run(region, mm, f, w):
    t = time.time()
    mf = mm if f == 1 else ndi.zoom(mm, f, order=1, mode="grid-wrap", grid_mode=True)
    mask = mf < w; del mf
    inv = [1.0 / tortuosity_fv(mask, H / f, axis=ax, tol=a.tol)["lam2"] for ax in range(3)]
    rows.append(dict(region=region, f=f, w=w, ecs_fraction=float(mask.mean()), inv_lam2=inv, shape=list(mask.shape),
                     seconds=round(time.time() - t)))
    print(f"{region:5s} f={f} w={w:3g} alpha={mask.mean():.3f} 1/lam2 (z,y,x)={np.round(inv, 4)}  ({rows[-1]['seconds']}s)", flush=True)
    json.dump(dict(tile=a.tile, rows=rows), open(os.path.join(out, f"fv_refined_{a.tile}.json"), "w"), indent=1)


for w in a.ws:
    for f in a.box_factors:
        run("box", m[:, :a.box, :a.box], f, w)
for w in a.ws:
    for f in a.full_factors:
        run("full", m, f, w)
with open(os.path.join(out, f"fv_refined_{a.tile}.md"), "w") as fh:
    fh.write("| region | w (nm) | f | ECS fraction | 1/lam2 (z, y, x) |\n|---|---|---|---|---|\n")
    for r in rows:
        fh.write(f"| {r['region']} | {r['w']:g} | {r['f']} | {r['ecs_fraction']:.3f} | {', '.join(f'{x:.4f}' for x in r['inv_lam2'])} |\n")
