"""Held-out test on three consecutive MICrONS sections (17414, 17415, 17416; 40 nm apart).

The middle section is predicted from its two neighbours (an 80 nm gap, twice the 40 nm gap the pipeline
normally bridges, so factor=2 here) and scored against the real middle section.

  A. smoothed inputs  -> smoothed 17415   (sframe_*, already mode-filtered by the original pipeline)
  B. raw inputs       -> raw 17415        (frame_*; 17415 is an older 500x500 export = top-left crop of the
                                           501x501 grid, so everything is cropped to 500x500)
  C. what the pipeline's mode filter changed, and how SDF smoothing compares

usage: python examples/micron_17414_17416.py [DATA_DIR]      (writes results/micron_17414_17416.{md,json,png})
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np, scipy.ndimage as ndi, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from real_io import DATA, load_labels, score
from zmorph.interp import reconstruct, majority, competition

D = sys.argv[1] if len(sys.argv) > 1 else DATA
tile = "170398_61241"
sm = {z: load_labels(os.path.join(D, f"sframe_{z}_{tile}_4.csv.gz")) for z in (17414, 17415, 17416)}
raw = {z: load_labels(os.path.join(D, f"frame_{z}_{tile}_4.csv.gz")) for z in (17414, 17416)}
raw15 = load_labels(os.path.join(D, f"frame_17415_{tile}.csv.gz"))
c500 = lambda a: a[:500, :500]
out, md = {}, []

def run(methods, inputs, truth, tag, changed):
    rows = []
    md.append("| method | pixel acc. | acc. where inputs differ | boundary dist (px) | mean Dice | component-count error |")
    md.append("|---|---|---|---|---|---|")
    preds = {}
    for name, f in methods.items():
        p = f(*inputs); preds[name] = p
        s = score(p, truth, changed); out[f"{tag}|{name}"] = s
        md.append(f"| {name} | {s['acc']:.4f} | {s['acc_changed']:.4f} | {s['bdist']:.3f} | {s['dice']:.4f} | {s['comp_err']} |")
        print(f"{tag} {name:34s} acc {s['acc']:.4f} chg {s['acc_changed']:.4f} bd {s['bdist']:.3f} dice {s['dice']:.4f} comp {s['comp_err']}", flush=True)
    return preds

mid = lambda v: v[1]
def methods(small_sigma_sweep):
    m = {
        "nearest (lower section)": lambda a, b: a,
        "nearest (upper section)": lambda a, b: b,
        "heat-paths port": lambda a, b: mid(reconstruct([a, b], 2, "heat")),
        "heat-paths port + majority": lambda a, b: mid(majority(reconstruct([a, b], 2, "heat"), 2)),
        "harmonic (L=10)": lambda a, b: mid(reconstruct([a, b], 2, "harmonic", L=10.0)),
        "sdf0 (classic binary SDF)": lambda a, b: mid(reconstruct([a, b], 2, "sdf0")),
        "sdf (sigma=1)": lambda a, b: mid(reconstruct([a, b], 2, "sdf")),
        "sdf (sigma=1) + translation": lambda a, b: mid(reconstruct([a, b], 2, "sdf", align=True)),
    }
    if small_sigma_sweep:
        for sg in (0.5, 1.5, 2.0):
            m[f"sdf (sigma={sg}) + translation"] = (lambda s: lambda a, b: mid(reconstruct([a, b], 2, "sdf", align=True, sigma=s)))(sg)
    return m

# ---- A: smoothed inputs
a14, a16, t15 = sm[17414], sm[17416], sm[17415]
chg = a14 != a16
md.append("## A. Smoothed sections: predict 17415 from 17414 and 17416\n")
md.append(f"{len(np.unique(t15))} labels; 17414 and 17416 differ in {chg.mean()*100:.1f}% of pixels.\n")
predsA = run(methods(True), (a14, a16), t15, "A_smoothed", chg)

# ---- B: raw inputs, raw truth (500x500)
r14, r16, tr15 = c500(raw[17414]), c500(raw[17416]), raw15
chgB = r14 != r16
md.append("\n## B. Raw sections: predict raw 17415 from raw 17414 and 17416 (all cropped to 500x500)\n")
md.append(f"{len(np.unique(tr15))} labels; 17414 and 17416 differ in {chgB.mean()*100:.1f}% of pixels.\n")
mB = methods(False)
mB["sdf (sigma=1) + translation, SMOOTHED inputs"] = lambda a, b: mid(reconstruct([c500(sm[17414]), c500(sm[17416])], 2, "sdf", align=True))
mB["heat-paths port + majority, SMOOTHED inputs"] = lambda a, b: mid(majority(reconstruct([c500(sm[17414]), c500(sm[17416])], 2, "heat"), 2))
run(mB, (r14, r16), tr15, "B_raw", chgB)
md.append(f"\nReference: smoothed 17415 agrees with raw 17415 on {(c500(sm[17415]) == tr15).mean()*100:.2f}% of pixels.")

# ---- C: what the original mode filter changed
md.append("\n## C. What the original mode filter changed (raw vs smoothed)\n")
md.append("| section | pixels changed | changed blobs | median blob (px) | max blob (px) | labels raw -> smoothed |")
md.append("|---|---|---|---|---|---|")
for z in (17414, 17416):
    ch = raw[z] != sm[z]; cc, n = ndi.label(ch); sz = np.bincount(cc.ravel())[1:]
    md.append(f"| {z} | {ch.sum()} ({ch.mean()*100:.2f}%) | {n} | {int(np.median(sz))} | {sz.max()} | {len(np.unique(raw[z]))} -> {len(np.unique(sm[z]))} |")
md.append("\nAgreement of each denoiser with the mode-filtered 17414 (not an independent truth), computed on 17414 only:\n")
md.append("| denoiser | pixel agreement | interface distance (px) |")
md.append("|---|---|---|")
md.append(f"| raw, unchanged | {(raw[17414] == sm[17414]).mean():.4f} | {score(raw[17414], sm[17414])['bdist']:.3f} |")
for sg in (0.5, 1.0, 1.5, 2.0):
    lab = competition([raw[17414], raw[17416]], 1, "sdf", sigma=sg)[0][0]
    md.append(f"| SDF smoothing, sigma={sg} | {(lab == sm[17414]).mean():.4f} | {score(lab, sm[17414])['bdist']:.3f} |")

open(os.path.join(ROOT, "results", "micron_17414_17416.md"), "w").write("\n".join(md) + "\n")
json.dump(out, open(os.path.join(ROOT, "results", "micron_17414_17416.json"), "w"), indent=1)

# ---- figure: region of the largest change, smoothed test
chgmap = ndi.uniform_filter(chg.astype(float), 160)
cy, cx = np.unravel_index(np.argmax(chgmap), chgmap.shape)
y0, x0 = int(np.clip(cy - 80, 0, 501 - 160)), int(np.clip(cx - 80, 0, 501 - 160)); sl = (slice(y0, y0 + 160), slice(x0, x0 + 160))
ids = {k: i for i, k in enumerate(np.unique(np.concatenate([a14.ravel(), t15.ravel(), a16.ravel()])))}
cm = plt.get_cmap("tab20"); rgb = lambda im: cm(np.vectorize(ids.get)(im[sl]) % 20)[..., :3]
show = {"nearest (lower section)": predsA["nearest (lower section)"], "heat-paths port + majority": predsA["heat-paths port + majority"],
        "sdf (sigma=1) + translation": predsA["sdf (sigma=1) + translation"]}
fig, ax = plt.subplots(2, 4, figsize=(14, 7))
for a, im, t in zip(ax[0], [a14, t15, a16], ["17414 (input)", "17415 (held-out truth)", "17416 (input)"]):
    a.imshow(rgb(im), interpolation="nearest"); a.set_title(t)
ax[0, 3].axis("off")
for a, (n, p) in zip(ax[1], show.items()):
    err = (p != t15)[sl]
    a.imshow(rgb(p), interpolation="nearest"); a.imshow(np.where(err, 1.0, np.nan), cmap="gray_r", alpha=.85, vmin=0, vmax=1.6, interpolation="nearest")
    a.set_title(f"{n}\nwrong px (black): {err.mean()*100:.1f}% of crop")
ax[1, 3].axis("off")
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "results", "micron_17414_17416.png"), dpi=100)
print("wrote results/micron_17414_17416.{md,json,png}")
