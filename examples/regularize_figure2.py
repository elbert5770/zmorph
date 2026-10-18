"""Remove the thin ledge on cell ...779 (Figure 2 region) with a minimum-feature-width regularisation.

Smoothing each cell's signed distance does not remove it (a distance field is unbounded, so a thin sheet keeps a positive
core: sigma = 1-4 px changed the thin part of the cell from 5.4% to 4.7%). Threshold dynamics on the labels does:
blur each label's indicator and give each voxel to the largest (zmorph.mesh.smooth_labels), twice. This script
compares sigma = 0 (sdf_align as is) with several sigma, reporting the thin part of cell ...779 (voxels not kept by a
3-D opening of radius 24 nm), the volume change, accuracy at the input sections and held-out accuracy (predict frames
from every other frame, factor 2, then regularise the stack), and draws conforming meshes of each.
Writes results/figure2_regularize.{png,json}.

usage: python examples/regularize_figure2.py [--sigmas 2 3 4] [--iters 2]
"""
import os, sys, argparse, json, colorsys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
import scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.morphology import ball
from zmorph.interp import reconstruct
from zmorph.mesh import competition_meshes, smooth_labels, smoothed_competition_meshes

ap = argparse.ArgumentParser()
ap.add_argument("--tile", default="170398_61241"); ap.add_argument("--frames", type=int, nargs=2, default=[17411, 17423])
ap.add_argument("--crop", type=int, nargs=4, default=[100, 190, 160, 290])
ap.add_argument("--sigmas", type=float, nargs="+", default=[2, 3, 4]); ap.add_argument("--iters", type=int, default=2)
a = ap.parse_args()
A, B = 864691136812081779, 864691135516164338
H = 8.0
d = np.load(os.path.join(ROOT, "data", f"tile_{a.tile}.npz")); vol, ids, z0 = d["vol"], d["ids"], int(d["z"][0])
ia, ib = int(np.nonzero(ids == A)[0][0]), int(np.nonzero(ids == B)[0][0])
r0, r1, c0, c1 = a.crop; f0, f1 = a.frames
secs = [vol[f - z0, r0:r1, c0:c1].astype(np.int64) for f in range(f0, f1 + 1)]
col = {ia: np.array(colorsys.hsv_to_rgb(0.08, 0.9, 0.97)), ib: np.array(colorsys.hsv_to_rgb(0.62, 0.8, 0.9))}
ink, sub = "#1f2933", "#5f6b76"
plt.rcParams.update({"text.color": ink})
ball3 = ball(3); a_led = (17413, 17417)
base_lab, base_meshes = competition_meshes(secs, 5, align=True)
even = secs[0::2]; odd_truth = secs[1::2]                                        # predict the odd frames from the even ones
pred_vol = reconstruct(even, 10, "sdf", align=True, regularize=None)                            # isotropic 8 nm slices: frame f0 + 2i at slice 10 i, odd frames at 10 i + 5
rows, meshes = [], {0.0: base_meshes}


def stats(lab, pred=None, sg=0.0):
    mA = lab == ia
    led = slice(5 * (a_led[0] - f0), 5 * (a_led[1] - f0) + 1)                    # slices of frames 17413-17417, where A is only the ledge
    r = dict(sigma=sg, ledge_volume=int(mA[led].sum()), thin_fraction_A=float((mA & ~ndi.binary_opening(mA, ball3)).sum() / max(mA.sum(), 1)), volume_A=int(mA.sum()),
             repro_at_input=float(np.mean([(lab[5 * i] == s).mean() for i, s in enumerate(secs)])))
    if pred is not None:
        r["heldout_acc"] = float(np.mean([(pred[10 * i + 5] == odd_truth[i]).mean() for i in range(len(odd_truth)) if 10 * i + 5 < len(pred)]))
    return r


rows.append(stats(base_lab, pred_vol, 0.0))
for sg in a.sigmas:
    lab, mm = smoothed_competition_meshes(base_lab, sg, a.iters)
    pred = smooth_labels(pred_vol, sg, a.iters)[0]
    rows.append(stats(lab, pred, sg)); meshes[sg] = mm
for r in rows:
    print(r, flush=True)
json.dump(dict(iters=a.iters, rows=rows), open(os.path.join(ROOT, "results", "figure2_regularize.json"), "w"), indent=1)

keys = [0.0] + list(a.sigmas); n = len(keys)
fig = plt.figure(figsize=(5.4 * n, 6.6))
Z = 5 * (len(secs) - 1) * H; X = (c1 - c0) * H; Y = (r1 - r0) * H
for j, sg in enumerate(keys):
    ax = fig.add_subplot(1, n, j + 1, projection="3d")
    for k in (ia, ib):
        if k not in meshes[sg]:
            continue
        v, f = meshes[sg][k]; t = v[f][:, :, [2, 1, 0]] * H
        nn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-12
        L = np.array([0.4, -0.5, 0.75]); L /= np.linalg.norm(L); sh = 0.5 + 0.5 * np.abs(nn @ L)
        ax.add_collection3d(Poly3DCollection(t, facecolors=np.clip(col[k][None] * sh[:, None], 0, 1), linewidths=0))
    ax.set_xlim(0, X); ax.set_ylim(Y, 0); ax.set_zlim(0, Z); ax.set_box_aspect((X, Y, Z)); ax.view_init(22, 118); ax.set_axis_off()
    r = rows[j]
    name = "sdf_align, no regularisation" if sg == 0 else f"sigma {sg:g} px ({sg * H:.0f} nm), {a.iters} passes"
    ax.set_title(f"{name}\nledge volume {100 * r['ledge_volume'] / rows[0]['ledge_volume']:.0f}% of unregularised\n"
                 f"held-out accuracy {100 * r['heldout_acc']:.2f}%\ninput sections reproduced {100 * r['repro_at_input']:.1f}%", fontsize=10)
fig.suptitle("A minimum feature width removes the thin ledge on …779 (orange) over …338 (blue)", fontsize=12.5, fontweight="bold", x=0.01, ha="left")
fig.text(0.01, 0.01, "Regularisation: blur each cell's indicator and give every voxel to the largest (threshold dynamics), repeated. Ledge volume: cell …779 voxels in frames 17413-17417. Held-out: frames predicted from every other frame (80 nm gap), then regularised.", fontsize=8.5, color=sub)
out = os.path.join(ROOT, "results", "figure2_regularize.png"); fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white"); print(out)
