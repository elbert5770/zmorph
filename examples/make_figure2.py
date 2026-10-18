"""Reproduce the idea of Figure 2 (thin ledges and spikes at frames 17415-17417) from the raw labels.

Cells 864691136812081779 (A, orange) and 864691135516164338 (B, blue) touch in tile 170398_61241. Top row: consecutive
raw frames with the pixels that belong to features narrower than 3 pixels (24 nm; removed by a 3x3 opening) outlined in
red. Bottom: the same region as a surface mesh when every frame is repeated five times (stair-step, two views), and as
the sdf_align conforming mesh. Scale bars: 75 nm in the frames; 100 nm in the 3-D views.

usage: python examples/make_figure2.py [--tile 170398_61241] [--frames 17411 17423] [--crop 100 190 160 290]
"""
import os, sys, argparse, colorsys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
import scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from zmorph.mesh import competition_meshes, stair_meshes

ap = argparse.ArgumentParser()
ap.add_argument("--tile", default="170398_61241"); ap.add_argument("--frames", type=int, nargs=2, default=[17411, 17423])
ap.add_argument("--show", type=int, nargs="+", default=[17413, 17414, 17415, 17416, 17417, 17418])
ap.add_argument("--crop", type=int, nargs=4, default=[100, 190, 160, 290])
a = ap.parse_args()
A, B = 864691136812081779, 864691135516164338
H = 8.0
d = np.load(os.path.join(ROOT, "data", f"tile_{a.tile}.npz")); vol, ids, z0 = d["vol"], d["ids"], int(d["z"][0])
ia, ib = int(np.nonzero(ids == A)[0][0]), int(np.nonzero(ids == B)[0][0])
r0, r1, c0, c1 = a.crop
f0, f1 = a.frames
secs = [vol[f - z0, r0:r1, c0:c1].astype(np.int64) for f in range(f0, f1 + 1)]
ink, sub = "#1f2933", "#5f6b76"
col = {ia: np.array(colorsys.hsv_to_rgb(0.08, 0.9, 0.97)), ib: np.array(colorsys.hsv_to_rgb(0.62, 0.8, 0.9))}
plt.rcParams.update({"text.color": ink})

fig = plt.figure(figsize=(17, 9.2))
gs = fig.add_gridspec(2, len(a.show), height_ratios=[1, 1.35], hspace=0.12, wspace=0.05)
sb = 75.0 / H                                                                    # scale bar in pixels
for j, f in enumerate(a.show):
    s = secs[f - f0]; ax = fig.add_subplot(gs[0, j])
    img = np.ones(s.shape + (3,)) * 0.86
    img[s == ia] = col[ia]; img[s == ib] = col[ib]
    ax.imshow(img, interpolation="nearest")
    for k in (ia, ib):
        m = s == k; thin = m & ~ndi.binary_opening(m, np.ones((3, 3), bool))
        if thin.any():
            ax.contour(ndi.binary_dilation(thin, np.ones((3, 3), bool)), levels=[0.5], colors=["#d00000"], linewidths=1.3)
    ax.add_patch(Rectangle((s.shape[1] - sb - 6, s.shape[0] - 8), sb, 2.5, color=ink)); ax.text(s.shape[1] - sb / 2 - 6, s.shape[0] - 11, "75 nm", ha="center", fontsize=8)
    ax.set_title(f"frame {f}", fontsize=11); ax.axis("off")

stair = np.repeat(np.stack(secs), 5, axis=0)[:-4]                                # every frame repeated five times (8 nm slices)
sm = competition_meshes(secs, 5, align=True)[1]
st = stair_meshes(stair, min_voxels=50)


def draw(ax, meshes, view, title, bar=True):
    for k in (ia, ib):
        if k not in meshes:
            continue
        v, f = meshes[k]; t = v[f][:, :, [2, 1, 0]] * H                         # (x=col, y=row, z=slice) in nm
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        L = np.array([0.4, -0.5, 0.75]); L /= np.linalg.norm(L)
        sh = 0.5 + 0.5 * np.abs(n @ L)
        ax.add_collection3d(Poly3DCollection(t, facecolors=np.clip(col[k][None] * sh[:, None], 0, 1), linewidths=0))
    X = (c1 - c0) * H; Y = (r1 - r0) * H; Z = (stair.shape[0] - 1) * H
    ax.set_xlim(0, X); ax.set_ylim(Y, 0); ax.set_zlim(0, Z); ax.set_box_aspect((X, Y, Z * 1.0)); ax.view_init(*view)
    ax.set_axis_off(); ax.set_title(title, fontsize=11)


axes = [fig.add_subplot(gs[1, :2], projection="3d"), fig.add_subplot(gs[1, 2:4], projection="3d"),
        fig.add_subplot(gs[1, 4:], projection="3d")]
draw(axes[0], st, (22, 118), "stair-step (each frame repeated 5x): view 1")
draw(axes[1], st, (50, 62), "stair-step: view 2")
draw(axes[2], sm, (22, 118), "sdf_align conforming mesh: view 1")
fig.suptitle("Thin ledge and spikes on cell …779 (orange) over cell …338 (blue), frames 17413-17418 (tile 170398_61241)",
             fontsize=13, fontweight="bold", x=0.01, ha="left", y=0.965)
fig.text(0.01, 0.01, "Red outlines: pixels in features narrower than 3 pixels (24 nm). Raw frames from the supplied tile data; "
         "the cells were identified by ID and by contact in frames 17415-17417.", fontsize=9, color=sub)
out = os.path.join(ROOT, "results", "figure2_reproduction.png"); fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white"); print(out)
