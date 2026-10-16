"""Robustness of tube links on the split phantom: junction height, daughter speed, radius, noise.
usage: python run_split_sweep.py [--seeds N]   writes results/split3_sweep.md"""
import argparse
import numpy as np
from zmorph import phantom
from zmorph.interp import reconstruct
from zmorph.metrics import evaluate

M = {"nearest": ("nearest", {}), "proposed": ("sdf", {}), "persist": ("sdf", dict(persist=True)),
     "tube": ("sdf", dict(tube=True)), "tube + persist": ("sdf", dict(tube=True, persist=True))}
V = [("baseline (junction z=15.5, 6 px/z, r=2.0)", {}), ("junction z=16.5", dict(zJ=16.5)), ("junction z=17.5 (mid-gap)", dict(zJ=17.5)),
     ("junction z=18.5", dict(zJ=18.5)), ("junction z=19.5", dict(zJ=19.5)), ("daughter speed 4.5 px/z", dict(speed=4.5)),
     ("daughter speed 7.5 px/z", dict(speed=7.5)), ("radius 2.4 (footprints overlap)", dict(r=2.4, bg=0.6))]

def gap(sc, kw, noisy, seeds):
    gt = phantom.scene_split3(**sc); clean = [gt[z] for z in range(0, gt.shape[0], 5)]
    vals = []
    for sd in range(seeds if noisy else 1):
        secs = phantom.corrupt(clean, np.random.default_rng(sd), bad=None) if noisy else clean
        m, k = kw; vals.append(evaluate(reconstruct(secs, 5, m, **k), gt)["dice_branch_gap"])
    return float(np.mean(vals))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    md = ["Neurite Dice in the gap slices 16-19 (clean sections / mean of seeds with boundary noise).\n",
          "| variant | " + " | ".join(M) + " |", "|---|" + "---|" * len(M)]
    for lab, sc in V:
        row = [f"{gap(sc, kw, False, 1):.3f} / {gap(sc, kw, True, a.seeds):.3f}" for kw in M.values()]
        md.append(f"| {lab} | " + " | ".join(row) + " |"); print(lab, row, flush=True)
    open("results/split3_sweep.md", "w").write("\n".join(md) + "\n")
