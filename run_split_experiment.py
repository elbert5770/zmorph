"""Thin neurite that splits into three non-overlapping daughters (phantom.scene_split3), and the
tilted-neurite scene, with every method. usage: python run_split_experiment.py [--seeds N]
Writes results/split3.md and results/split3.json."""
import argparse, json
import numpy as np, scipy.ndimage as ndi
from zmorph import phantom
from zmorph.interp import reconstruct, majority
from zmorph.metrics import evaluate

SH = dict(maxshift=40)
METHODS = {
    "nearest (repeat)":                       ("nearest", {}, False),
    "Raya-Udupa linear SDF":                  ("sdf0", {}, False),
    "cubic-spline SDF (binary fields)":       ("sdf0_cubic", {}, False),
    "4-point SDF (binary fields)":            ("sdf0_4pt", {}, False),
    "harmonic":                               ("harmonic", {}, False),
    "heat-paths port (raw, no filter)":       ("heat", {}, False),
    "proposed (linear)":                      ("sdf", {}, False),
    "proposed, negative side truncated at 2": ("sdf", dict(neg_cap=2), False),
    "proposed + label translation":           ("sdf", dict(align=True, **SH), False),
    "proposed + component translation":       ("sdf", dict(align="component", **SH), False),
    "proposed + persistence guard":           ("sdf", dict(persist=True), False),
    "proposed + tube links":                  ("sdf", dict(tube=True), False),
    "proposed + tube links + persistence guard": ("sdf", dict(tube=True, persist=True), False),
    "component translation + persistence guard": ("sdf", dict(align="component", persist=True, **SH), False),
}


def comps(v, z):
    return int(ndi.label(v[z] == 1)[1])


def area(v, z):
    return int((v[z] == 1).sum())


def run(seeds):
    gt = phantom.scene_split3()
    clean = [gt[z] for z in range(0, gt.shape[0], 5)]
    out, md, prop = {}, [], {}
    md.append(f"Ground-truth components of the neurite in slices 16-19: {[comps(gt, z) for z in range(16, 20)]} "
              f"(sections 15 / 20 / 25: {[comps(gt, z) for z in (15, 20, 25)]}).\n")
    for noisy in (False, True):
        md += [f"\n### {'noisy' if noisy else 'clean'} sections\n",
               "| method | neurite Dice in gap 16-19 | Dice all intermediate slices (all cells) | interface dist (px) | neurite components in slices 16-19 (clean only) |",
               "|---|---|---|---|---|"]
        for name, (m, kw, post) in METHODS.items():
            rs, cc = [], None
            for sd in range(seeds if noisy else 1):
                secs = phantom.corrupt(clean, np.random.default_rng(sd), bad=None) if noisy else clean
                v = reconstruct(secs, 5, m, **kw)
                if post:
                    v = majority(v, 2)
                rs.append(evaluate(v, gt))
                if not noisy:
                    cc = [comps(v, z) for z in range(16, 20)]
                    prop[name] = [(area(v, z), comps(v, z)) for z in range(15, 21)]
            g = float(np.mean([r["dice_branch_gap"] for r in rs])); d = float(np.mean([r["dice_mid"] for r in rs])); b = float(np.mean([r["bdist_mid"] for r in rs]))
            out[f"{'noisy' if noisy else 'clean'}|{name}"] = dict(gap_dice=g, dice_mid=d, bdist=b, comps=cc)
            md.append(f"| {name} | {g:.3f} | {d:.3f} | {b:.2f} | {cc if cc is not None else ''} |")
            print("noisy" if noisy else "clean", f"{name:36s} gap dice {g:.3f} dice_mid {d:.3f} bd {b:.2f}", cc if cc else "", flush=True)
    md += ["\n### How each method carries the neurite through the gap (clean sections): neurite pixels / components per slice\n",
           "| method | z=15 (section) | 16 | 17 | 18 | 19 | z=20 (section) |", "|---|---|---|---|---|---|---|"]
    md.append("| **ground truth** | " + " | ".join(f"{area(gt, z)} / {comps(gt, z)}" for z in range(15, 21)) + " |")
    for name in METHODS:
        md.append(f"| {name} | " + " | ".join(f"{a} / {c}" for a, c in prop[name]) + " |")
    out["per_slice"] = {k: v for k, v in prop.items()}
    return out, md


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    out, md = run(a.seeds)
    open("results/split3.md", "w").write("\n".join(md) + "\n"); json.dump(out, open("results/split3.json", "w"), indent=1)
