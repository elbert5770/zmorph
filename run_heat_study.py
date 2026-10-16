"""Improving the heat method: observation layer, arrival definition, time symmetry.
usage: python run_heat_study.py [--seeds N]    writes results/heat_study.md (about 15 minutes)"""
import argparse, itertools, os, sys
import numpy as np
sys.path.insert(0, "examples")
from zmorph import phantom
from zmorph.interp import reconstruct
from zmorph.metrics import evaluate
from real_io import DATA, load_labels, score

T = "170398_61241"
a = load_labels(f"{DATA}/sframe_17414_{T}_4.csv.gz"); b = load_labels(f"{DATA}/sframe_17416_{T}_4.csv.gz"); t = load_labels(f"{DATA}/sframe_17415_{T}_4.csv.gz")
ra = load_labels(f"{DATA}/frame_17414_{T}_4.csv.gz")[:500, :500]; rb = load_labels(f"{DATA}/frame_17416_{T}_4.csv.gz")[:500, :500]; rt = load_labels(f"{DATA}/frame_17415_{T}.csv.gz")
SC = {}
for nm, fn in (("arch", phantom.scene_branch), ("tilted", phantom.scene_tilted), ("split", phantom.scene_split3)):
    g = fn(); SC[nm] = (g, [g[z] for z in range(0, g.shape[0], 5)])
COLS = "| variant | arch Dice / bd | tilted Dice / bd | arch noisy Dice / bd (seeds) | split3 gap Dice | real smoothed acc / bd / Dice | real raw acc / bd / Dice |"

def run(m, kw, seeds):
    out = []
    for nm in ("arch", "tilted"):
        g, s = SC[nm]; r = evaluate(reconstruct(s, 5, m, **kw), g); out.append(f"{r['dice_mid']:.3f} / {r['bdist_mid']:.2f}")
    rs = [evaluate(reconstruct(phantom.corrupt(SC["arch"][1], np.random.default_rng(sd), bad=3), 5, m, **kw), SC["arch"][0]) for sd in range(seeds)]
    out.append(f"{np.mean([r['dice_mid'] for r in rs]):.3f} / {np.mean([r['bdist_mid'] for r in rs]):.2f}")
    g, s = SC["split"]; out.append(f"{evaluate(reconstruct(s, 5, m, **kw), g)['dice_branch_gap']:.3f}")
    s1 = score(reconstruct([a, b], 2, m, **kw)[1], t, a != b); s2 = score(reconstruct([ra, rb], 2, m, **kw)[1], rt, ra != rb)
    out += [f"{s1['acc']:.4f} / {s1['bdist']:.2f} / {s1['dice']:.3f}", f"{s2['acc']:.4f} / {s2['bdist']:.2f} / {s2['dice']:.3f}"]
    return out

def table(title, rows, seeds, md):
    md += [f"\n### {title}\n", COLS, "|---|" + "---|" * 6]
    for lab, m, kw in rows:
        md.append(f"| {lab} | " + " | ".join(run(m, kw, seeds)) + " |"); print(title[:20], lab, flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); n = ap.parse_args().seeds
    md = ["Heat method study (no post-filter anywhere). Observation layer z is in pixels from the source face (25-layer stack, L=25);\nthe paper uses z=6.5.\n"]
    table("1. Observation layer, published algorithm (path arrival, one-directional)", [(f"z={z}", "heat", dict(layer_z=z)) for z in (0.5, 1.5, 2.5, 4.5, 6.5, 9.5, 12.5, 16.5, 24.5)], n, md)
    table("2. Arrival definition x time symmetry at z=6.5", [(f"{arr}{' + symmetric' if sym else ''}", "heat", dict(arrival=arr, symmetric=sym)) for arr, sym in itertools.product(("paths", "paths_global", "rank", "T", "Trank"), (False, True))], n, md)
    for arr in ("rank", "Trank"):
        table(f"3. Observation layer, {arr} + symmetric", [(f"z={z}", "heat", dict(arrival=arr, symmetric=True, layer_z=z)) for z in (1.5, 2.5, 4.5, 6.5, 9.5, 12.5, 16.5, 20.5)], n, md)
    table("4. Final comparison", [("nearest (repeat)", "nearest", {}), ("heat, published (paths, z=6.5)", "heat", {}), ("heat2 (Trank + symmetric, z=2)", "heat2", {}),
                                   ("sdf (proposed)", "sdf", {}), ("sdf + translation", "sdf", dict(align=True))], n, md)
    # time-reversal invariance
    g, s = SC["arch"]
    md.append("\n### 5. Time-reversal check (arch phantom): is the result independent of the direction of z?\n")
    for lab, kw in (("published (paths, one-directional)", {}), ("heat2", dict(arrival="Trank", symmetric=True, layer_z=2.0))):
        v = reconstruct(s, 5, "heat", **kw); w = reconstruct(s[::-1], 5, "heat", **kw)[::-1]
        md.append(f"* {lab}: {int((v != w).sum())} of {v.size} voxels differ when the stack is reversed")
    open("results/heat_study.md", "w").write("\n".join(md) + "\n")
    open("/tmp/claude-0/-home-user-zmorph/37828f76-e0d7-565c-8881-7b12bd3db680/scratchpad/heat_study.done", "w").write("done")
