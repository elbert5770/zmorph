"""Combining heat2 and the SDF method. usage: python run_fusion_study.py [--seeds N]
Writes results/fusion_study.md (about 10-15 minutes)."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, "examples")
from zmorph import phantom
from zmorph.interp import reconstruct
from zmorph.fusion import heat_sdf_morph
from zmorph.metrics import evaluate
from real_io import DATA, load_labels, score

T = "170398_61241"
a = load_labels(f"{DATA}/sframe_17414_{T}_4.csv.gz"); b = load_labels(f"{DATA}/sframe_17416_{T}_4.csv.gz"); t = load_labels(f"{DATA}/sframe_17415_{T}_4.csv.gz")
ra = load_labels(f"{DATA}/frame_17414_{T}_4.csv.gz")[:500, :500]; rb = load_labels(f"{DATA}/frame_17416_{T}_4.csv.gz")[:500, :500]; rt = load_labels(f"{DATA}/frame_17415_{T}.csv.gz")
SC = {}
for nm, fn in (("arch", phantom.scene_branch), ("tilted", phantom.scene_tilted), ("split", phantom.scene_split3)):
    g = fn(); SC[nm] = (g, [g[z] for z in range(0, g.shape[0], 5)])

def runner(spec):
    kind, arg = spec
    if kind == "rec":
        m, kw = arg
        return lambda secs, f: reconstruct(secs, f, m, **kw)
    return lambda secs, f: heat_sdf_morph(secs, f, arg[0], **arg[1])

ROWS = [("nearest (repeat)", ("rec", ("nearest", {}))),
        ("heat, published", ("rec", ("heat", {}))),
        ("heat2", ("fus", ("heat2", {}))),
        ("sdf (proposed)", ("rec", ("sdf", {}))),
        ("sdf + translation", ("rec", ("sdf", dict(align=True)))),
        ("sdf + persistence guard", ("rec", ("sdf", dict(persist=True)))),
        ("sdf + tube links + guard", ("rec", ("sdf", dict(tube=True, persist=True)))),
        ("FUSION sdf_rank (heat switching, SDF order)", ("fus", ("sdf_rank", {}))),
        ("FUSION rank_avg (mean of heat2 and sdf_rank times)", ("fus", ("rank_avg", {}))),
        ("FUSION time_avg (heat2 and raw SDF crossing, w=0.5)", ("fus", ("time_avg", dict(lam=0.5)))),
        ("FUSION refine (heat2 frames, SDF smoothing)", ("fus", ("refine", {}))),
        ("FUSION guard (SDF, vanishing labels restored from heat2)", ("fus", ("guard", {}))),
        ("FUSION field (SDF blend fused with SDF of heat2 slice, w=0.5)", ("fus", ("field", dict(lam=0.5))))]
HEAD = "| method | arch Dice / bd | tilted Dice / bd | arch noisy Dice / bd | split3 gap Dice | real smoothed acc / bd / Dice | real raw acc / bd / Dice |"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); n = ap.parse_args().seeds
    md = ["Combining heat2 and the SDF method (no post-filter). bd = mean interface distance in px; noisy = mean of seeds.\n", HEAD, "|---|" + "---|" * 6]
    for lab, spec in ROWS:
        f = runner(spec); out = []
        for nm in ("arch", "tilted"):
            g, s = SC[nm]; r = evaluate(f(s, 5), g); out.append(f"{r['dice_mid']:.3f} / {r['bdist_mid']:.2f}")
        rs = [evaluate(f(phantom.corrupt(SC["arch"][1], np.random.default_rng(sd), bad=3), 5), SC["arch"][0]) for sd in range(n)]
        out.append(f"{np.mean([r['dice_mid'] for r in rs]):.3f} / {np.mean([r['bdist_mid'] for r in rs]):.2f}")
        g, s = SC["split"]; out.append(f"{evaluate(f(s, 5), g)['dice_branch_gap']:.3f}")
        s1 = score(f([a, b], 2)[1], t, a != b); s2 = score(f([ra, rb], 2)[1], rt, ra != rb)
        out += [f"{s1['acc']:.4f} / {s1['bdist']:.2f} / {s1['dice']:.3f}", f"{s2['acc']:.4f} / {s2['bdist']:.2f} / {s2['dice']:.3f}"]
        md.append(f"| {lab} | " + " | ".join(out) + " |"); print(lab, out, flush=True)
    open("results/fusion_study.md", "w").write("\n".join(md) + "\n")
    open("/tmp/claude-0/-home-user-zmorph/37828f76-e0d7-565c-8881-7b12bd3db680/scratchpad/fusion.done", "w").write("done")
