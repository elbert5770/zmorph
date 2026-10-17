"""heat_port (the published algorithm, as ported) vs sdf_align on the manuscript's Figure 9 and Figure 12 gaps.

Figure 9  : zframe 87070 -> 87075  = sections 17414 -> 17415 (frame number x 5), intermediates 87071-87074
Figure 12 : zframe 87075 -> 87080  = sections 17415 -> 17416, intermediates 87076-87079
Inputs are the mode-filtered sframes (what the figures start from); the raw frames are run too (sdf_align needs no
pre-filter). 'heat_port + majority' stands in for the manuscript's post-filtering with a plain 3x3x3 majority.
There is no ground truth inside a gap, so the measures are: sources kept exactly, equal progress per step,
no flicker (a pixel changes label at most once), no tiny fragments, and the held-out score of section 17415.

usage: python examples/compare_figures_9_12.py      (writes results/figures_9_12.{md,json}, fig9.png, fig12.png)
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np, scipy.ndimage as ndi, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from real_io import DATA, load_labels, score
from zmorph.interp import reconstruct, majority

T = "170398_61241"
sm = {z: load_labels(os.path.join(DATA, f"sframe_{z}_{T}_4.csv.gz")) for z in (17414, 17415, 17416)}
tile = np.load(os.path.join(ROOT, "data", f"tile_{T}.npz")); zi = list(tile["z"])
raw = {z: tile["ids"][tile["vol"][zi.index(z)]] for z in (17414, 17415, 17416)}
X0, X1, Y0, Y1 = 250, 450, 100, 300                      # crop used for Figure 9 (rows = x, columns = y)
PURPLE, RED = 864691136697855485, 864691135091912045     # the '485' cell; the red cell that is pushed away
crop = lambda a: a[..., X0:X1, Y0:Y1]

METHODS = {
    "heat_port": lambda a, b: reconstruct([a, b], 5, "heat"),
    "heat_port + majority": lambda a, b: majority(reconstruct([a, b], 5, "heat"), 1),
    "sdf_align": lambda a, b: reconstruct([a, b], 5, "sdf", align=True),
}


def frags(vol, thr=12):
    n = 0
    for f in vol[1:-1]:
        for k in np.unique(f):
            c, m = ndi.label(f == k)
            if m > 1: n += int((np.bincount(c.ravel())[1:] < thr).sum())
    return n


def measures(vol, a, b):
    """vol: (6, H, W) frames from the lower source a to the upper source b."""
    chg = a != b; n = int(chg.sum())
    step = [float((vol[i + 1][chg] != vol[i][chg]).sum()) / n for i in range(5)]
    changes = sum((vol[i + 1] != vol[i]).astype(np.int8) for i in range(5))
    done = (vol[4][chg] == b[chg]).mean()
    area = lambda k: [int((f == k).sum()) for f in vol]
    comps = lambda k: [int(ndi.label(f == k)[1]) for f in vol]
    return dict(sources_kept=bool((vol[0] == a).all() and (vol[-1] == b).all()), changed_px=n,
                step_share=[round(s, 3) for s in step], step_cv=round(float(np.std(step) / np.mean(step)), 3),
                flicker_px=int((changes[chg] > 1).sum()), flicker_share=round(float((changes[chg] > 1).mean()), 4),
                returned_px=int(((vol[1:5] != a)[:, chg].any(0) & (vol[4][chg] == a[chg])).sum()),
                fragments_lt12=frags(vol), done_at_frame4=round(float(done), 3),
                purple_area=area(PURPLE), purple_comps=comps(PURPLE), red_area=area(RED))


def draw(frames, names, ref, path, title, z0):
    ids = {k: i for i, k in enumerate(np.unique(np.concatenate([ref[0].ravel(), ref[1].ravel()])))}
    cm = plt.get_cmap("tab20")
    fig, ax = plt.subplots(len(frames), 6, figsize=(16, 2.7 * len(frames)))
    for r, (vol, nm) in enumerate(zip(frames, names)):
        for k in range(6):
            f = crop(vol[k]); im = np.vectorize(lambda v: ids.get(v, 0))(f)
            a = ax[r, k]; a.imshow(cm(im % 20)[..., :3].transpose(1, 0, 2), origin="lower", extent=[X0, X1, Y0, Y1], interpolation="nearest")
            a.contour(np.arange(X0, X1) + .5, np.arange(Y0, Y1) + .5, (f == PURPLE).T, [.5], colors="k", linewidths=1.0)
            a.set_xticks([]); a.set_yticks([]); a.set_title(f"{nm}  z={z0 + k}" if k == 0 else f"z={z0 + k}", fontsize=8)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


out, md = {}, ["# heat_port vs sdf_align: the Figure 9 and Figure 12 gaps", ""]
gaps = {"Figure 9 (zframe 87070-87075, sections 17414->17415)": (17414, 17415, 87070, "fig9"),
        "Figure 12 (zframe 87075-87080, sections 17415->17416)": (17415, 17416, 87075, "fig12")}
for gname, (za, zb, zf, tag) in gaps.items():
    print(gname, flush=True)
    runs = {}
    for inp, src in (("smoothed", sm), ("raw", raw)):
        for mn, f in METHODS.items():
            if inp == "raw" and mn == "heat_port": continue
            runs[f"{mn} [{inp} in]"] = f(src[za], src[zb])
    a, b = sm[za], sm[zb]
    md += [f"## {gname}", "", "Whole frame (501x501), smoothed inputs unless stated. 'step share' = fraction of the changed pixels converted in each of the 5 steps (ideal 0.2 each).", "",
           "| method | sources kept | changed px | step shares | step CV | flicker px (share) | frags <12 px | changed px done by frame 4 | purple cell comps (6 frames) | purple area |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for name, vol in runs.items():
        ref = (raw if "raw" in name else sm)
        m = measures(vol, ref[za], ref[zb]); out[f"{tag}|{name}"] = m
        md.append(f"| {name} | {m['sources_kept']} | {m['changed_px']} | {m['step_share']} | {m['step_cv']} | {m['flicker_px']} ({100 * m['flicker_share']:.1f}%) | {m['fragments_lt12']} | {m['done_at_frame4']} | {m['purple_comps']} | {m['purple_area']} |")
        print(f"  {name:36s} CV {m['step_cv']:.3f} flicker {m['flicker_px']:6d} frags {m['fragments_lt12']:4d}", flush=True)
    md.append("")
    sel = ["heat_port [smoothed in]", "heat_port + majority [smoothed in]", "sdf_align [smoothed in]"]
    draw([runs[s] for s in sel], ["heat_port", "+majority", "sdf_align"], (a, b), os.path.join(ROOT, "results", f"{tag}.png"),
         f"{gname}: crop rows {X0}-{X1}, cols {Y0}-{Y1}; black outline = the '485' cell", zf)

# held-out score on the real middle section (gap 2), whole frame and the figure crop
md += ["## Held-out check: predict section 17415 from 17414 and 17416 (80 nm gap), scored against the real 17415", "",
       "| method | input | acc (frame) | acc where inputs differ | boundary dist | Dice | acc (crop) |", "|---|---|---|---|---|---|---|"]
for inp, src, truth in (("smoothed", sm, sm[17415]), ("raw", raw, raw[17415])):
    chg = src[17414] != src[17416]
    for mn, kw in (("heat_port", dict(method="heat")), ("heat_port + majority", dict(method="heat", maj=True)), ("sdf_align", dict(method="sdf", align=True))):
        v = reconstruct([src[17414], src[17416]], 2, kw["method"], **({"align": True} if kw.get("align") else {}))
        p = majority(v, 1)[1] if kw.get("maj") else v[1]
        s = score(p, truth, chg); c = float((crop(p) == crop(truth)).mean()); out[f"heldout|{mn}|{inp}"] = dict(s, crop_acc=c)
        md.append(f"| {mn} | {inp} | {s['acc']:.4f} | {s['acc_changed']:.4f} | {s['bdist']:.3f} | {s['dice']:.4f} | {c:.4f} |")
open(os.path.join(ROOT, "results", "figures_9_12.md"), "w").write("\n".join(md) + "\n")
json.dump(out, open(os.path.join(ROOT, "results", "figures_9_12.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("\n".join(md))
