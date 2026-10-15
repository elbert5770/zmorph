"""Benchmark z-interpolation methods on synthetic tissue with known ground truth.

Usage: python run_benchmark.py [--seeds N]
Writes results/benchmark.json and results/benchmark.md.
"""
import argparse, json, time
import numpy as np
from zmorph import phantom
from zmorph.interp import reconstruct, majority
from zmorph.metrics import evaluate

# (method, kwargs, majority post-filter). Every interpolator gets the same z-aware cleaning
# front-end unless the row says "raw"; "heat-paths + majority" is the manuscript's pipeline shape.
METHODS = {
    "nearest, raw":            ("nearest", {}, False),
    "nearest":                 ("nearest", dict(clean=True), False),
    "sdf0 (classic)":          ("sdf0", dict(clean=True), False),
    "harmonic":                ("harmonic", dict(clean=True), False),
    "heat-paths (manuscript)": ("heat", dict(clean=True), False),
    "heat-paths + majority":   ("heat", dict(clean=True), True),
    "sdf, raw (no cleaning)":  ("sdf", {}, False),
    "sdf (proposed)":          ("sdf", dict(clean=True), False),
    "sdf + majority":          ("sdf", dict(clean=True), True),
    "sdf + align":             ("sdf", dict(clean=True, align=True), False),
}
SCENES = {"branch": phantom.scene_branch, "tilted": phantom.scene_tilted}
COLS = ["acc_mid", "dice_mid", "dice_branch_gap", "bdist_mid", "component_err", "acc_sec", "area_ratio"]


def run(seeds):
    table = {}
    for scene, fn in SCENES.items():
        gt = fn()
        clean = [gt[z] for z in range(0, gt.shape[0], 5)]
        for noise in (False, True):
            for name, (m, kw, post) in METHODS.items():
                rows = []
                for sd in range(seeds if noise else 1):
                    secs = phantom.corrupt(clean, np.random.default_rng(sd), bad=3) if noise else clean
                    t = time.time()
                    v = reconstruct(secs, 5, m, **kw)
                    if post:
                        v = majority(v, 2)
                    r = evaluate(v, gt)
                    r["sec"] = time.time() - t
                    rows.append(r)
                table[(scene, "noisy" if noise else "clean", name)] = {
                    c: float(np.mean([r[c] for r in rows])) for c in COLS + ["sec"]}
    return table


def to_md(table):
    out = []
    for scene in SCENES:
        for noise in ("clean", "noisy"):
            out.append(f"\n### {scene} / {noise} sections\n")
            out.append("| method | " + " | ".join(COLS) + " | s |")
            out.append("|---|" + "---|" * (len(COLS) + 1))
            for name in METHODS:
                r = table[(scene, noise, name)]
                out.append(f"| {name} | " + " | ".join(f"{r[c]:.3f}" if c != "component_err" else f"{r[c]:.1f}" for c in COLS) + f" | {r['sec']:.1f} |")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    t = run(a.seeds)
    md = to_md(t)
    print(md)
    open("results/benchmark.md", "w").write(md)
    json.dump({"|".join(k): v for k, v in t.items()}, open("results/benchmark.json", "w"), indent=1)
