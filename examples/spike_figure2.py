"""The one-voxel spike on cell ...779 in frame 17416 (Figure 2 region), before and after a light regularisation.

The spike is a run of A pixels one voxel wide and about 11-13 voxels long that exists only in frame 17416 (frames 17415 and
17417 have no A at those pixels). Top: the three frames zoomed on it. Bottom (cell B faint so the orange spike shows): stair-step mesh (every frame repeated five
times), sdf_align as it stands, and sdf_align followed by two passes of threshold dynamics with sigma = 1 voxel (8 nm).
Writes results/figure2_spike.png.

usage: python examples/spike_figure2.py [--sigma 1.0] [--iters 2]
"""
import os, sys, argparse, colorsys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from zmorph.mesh import competition_meshes, stair_meshes, smoothed_competition_meshes

ap = argparse.ArgumentParser()
ap.add_argument("--sigma", type=float, default=1.0); ap.add_argument("--iters", type=int, default=2)
ap.add_argument("--crop", type=int, nargs=4, default=[100, 190, 160, 290])
ap.add_argument("--zoom", type=int, nargs=4, default=[22, 72, 18, 70])                # rows and cols inside the crop
a = ap.parse_args()
A, B = 864691136812081779, 864691135516164338
H = 8.0
d = np.load(os.path.join(ROOT, "data", "tile_170398_61241.npz")); vol, ids, z0 = d["vol"], d["ids"], int(d["z"][0])
ia, ib = int(np.nonzero(ids == A)[0][0]), int(np.nonzero(ids == B)[0][0])
r0, r1, c0, c1 = a.crop; zr0, zr1, zc0, zc1 = a.zoom
frames = list(range(17411, 17424)); secs = [vol[f - z0, r0:r1, c0:c1].astype(np.int64) for f in frames]
ink, sub = "#1f2933", "#5f6b76"
col = {ia: np.array(colorsys.hsv_to_rgb(0.08, 0.9, 0.97)), ib: np.array(colorsys.hsv_to_rgb(0.62, 0.8, 0.9))}
plt.rcParams.update({"text.color": ink})
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.5], hspace=0.12, wspace=0.05)
for j, f in enumerate((17415, 17416, 17417)):
    ax = fig.add_subplot(gs[0, j]); s = secs[f - 17411][zr0:zr1, zc0:zc1]
    img = np.ones(s.shape + (3,)) * 0.86; img[s == ia] = col[ia]; img[s == ib] = col[ib]
    ax.imshow(img, interpolation="nearest"); ax.set_title(f"frame {f}" + ("  (spike)" if f == 17416 else ""), fontsize=11); ax.axis("off")
    sb = 75 / H; ax.add_patch(Rectangle((s.shape[1] - sb - 3, s.shape[0] - 5), sb, 1.5, color=ink)); ax.text(s.shape[1] - sb / 2 - 3, s.shape[0] - 7, "75 nm", ha="center", fontsize=8)

base_lab, base_meshes = competition_meshes(secs, 5, align=True)
reg_lab, reg_meshes = smoothed_competition_meshes(base_lab, a.sigma, a.iters)
stair = np.repeat(np.stack(secs), 5, axis=0)[:-4]
st = stair_meshes(stair, min_voxels=50)
lo = np.array([20, zr0, zc0]) * 1.0; hi = np.array([40, zr1, zc1]) * 1.0                # slices 20..40 (frames 17415-17419 around the spike)
Xl, Yl, Zl = (zc1 - zc0) * H, (zr1 - zr0) * H, (hi[0] - lo[0]) * H


def draw(ax, meshes, title):
    for k in (ib, ia):
        if k not in meshes:
            continue
        v, f = meshes[k]; t = v[f]
        keep = np.all((t[:, :, 0] >= lo[0] - 1) & (t[:, :, 0] <= hi[0] + 1) & (t[:, :, 1] >= zr0 - 1) & (t[:, :, 1] <= zr1 + 1)
                      & (t[:, :, 2] >= zc0 - 1) & (t[:, :, 2] <= zc1 + 1), axis=1)
        t = (t[keep] - np.array([lo[0], 0, 0])) * H                                      # (slice, row, col) -> nm
        t = t[:, :, [2, 1, 0]] - np.array([zc0 * H, zr0 * H, 0])
        nn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-12
        Ld = np.array([0.4, -0.5, 0.75]); Ld /= np.linalg.norm(Ld); sh = 0.5 + 0.5 * np.abs(nn @ Ld)
        fc = np.concatenate([np.clip(col[k][None] * sh[:, None], 0, 1), np.full((len(t), 1), 0.10 if k == ib else 1.0)], 1)
        ax.add_collection3d(Poly3DCollection(t, facecolors=fc, linewidths=0))
    ax.set_xlim(0, Xl); ax.set_ylim(Yl, 0); ax.set_zlim(0, Zl); ax.set_box_aspect((Xl, Yl, Zl)); ax.view_init(30, 118)
    ax.set_axis_off(); ax.set_title(title, fontsize=11)


draw(fig.add_subplot(gs[1, 0], projection="3d"), st, "stair-step (each frame repeated 5x)")
draw(fig.add_subplot(gs[1, 1], projection="3d"), base_meshes, "sdf_align as it stands")
draw(fig.add_subplot(gs[1, 2], projection="3d"), reg_meshes, f"sdf_align + regularisation (sigma {a.sigma:g} px = {a.sigma * H:.0f} nm, {a.iters} passes)")
fig.suptitle("One-voxel spike on cell …779 present only in frame 17416, before and after a light regularisation",
             fontsize=13, fontweight="bold", x=0.01, ha="left", y=0.97)
fig.text(0.01, 0.01, "Spike: a run of …779 pixels 1 voxel wide and 11-13 long in frame 17416 only. The regularisation blurs each cell's indicator and "
         "gives each voxel to the largest value (zmorph.mesh.smooth_labels).", fontsize=9, color=sub)
out = os.path.join(ROOT, "results", "figure2_spike.png"); fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white"); print(out)
