"""Export the signed-distance (competition) fields and meshes of the Figure 2 region, with and without a gap, for ParaView,
and draw them.

Files in results/sdf_export/ (spacing 8 nm, axes x fastest):
  fields.vti        ImageData with arrays  sdf_A, sdf_B (signed competition field of each cell, nm, > 0 inside the cell, 0 at the
                    interface), margin (best minus second best, nm; the gap is margin < w), label (cell index), gap_<w>nm (0/1)
  cells_touching.ply, cells_gap<w>nm.ply   conforming meshes of every cell, without and with the gap
results/sdf_gap.png  a slice through the fields and the meshes with and without the gap.

usage: python examples/export_sdf_gap.py [--w 40]
"""
import os, sys, argparse, colorsys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from zmorph.interp import competition
from zmorph.mesh import competition_meshes
from zmorph.export import write_vti, write_ply

ap = argparse.ArgumentParser()
ap.add_argument("--w", type=float, default=40.0); ap.add_argument("--slice-frame", type=int, default=17415)
ap.add_argument("--crop", type=int, nargs=4, default=[100, 190, 160, 290]); ap.add_argument("--frames", type=int, nargs=2, default=[17411, 17423])
a = ap.parse_args()
A, B = 864691136812081779, 864691135516164338
H = 8.0; w = a.w
d = np.load(os.path.join(ROOT, "data", "tile_170398_61241.npz")); vol, ids, z0 = d["vol"], d["ids"], int(d["z"][0])
ia, ib = int(np.nonzero(ids == A)[0][0]), int(np.nonzero(ids == B)[0][0])
r0, r1, c0, c1 = a.crop; f0, f1 = a.frames
secs = [vol[f - z0, r0:r1, c0:c1].astype(np.int64) for f in range(f0, f1 + 1)]
out = os.path.join(ROOT, "results", "sdf_export"); os.makedirs(out, exist_ok=True)

lab, best, second, fields = competition(secs, 5, "sdf", return_fields=True, align=True)
def comp(k):                                                  # signed competition field of cell k, in nm
    other = np.where(lab == k, second, best)
    return ((fields[k] - np.where(np.isfinite(other), other, -12.0)) * H).astype(np.float32)
margin = np.where(np.isfinite(second), (best - second) * H, 2 * 12.0 * H).astype(np.float32)
sdfA, sdfB = comp(ia), comp(ib)
arrays = {"sdf_A": sdfA, "sdf_B": sdfB, "margin": margin, "label": lab.astype(np.int64), f"gap_{w:g}nm": (margin < w)}
write_vti(os.path.join(out, "fields.vti"), arrays, spacing=(H, H, H))
_, touch = competition_meshes(secs, 5, align=True)
_, gapped = competition_meshes(secs, 5, gap=w / H, align=True)
hues = {int(k): colorsys.hsv_to_rgb(0.08, 0.9, 0.97) if k == ia else colorsys.hsv_to_rgb(0.62, 0.8, 0.9) if k == ib else (0.75, 0.75, 0.78) for k in touch}
write_ply(os.path.join(out, "cells_touching.ply"), touch, spacing=(H, H, H), colors=hues)
write_ply(os.path.join(out, f"cells_gap{w:g}nm.ply"), gapped, spacing=(H, H, H), colors=hues)
print({f: os.path.getsize(os.path.join(out, f)) // 1024 for f in os.listdir(out)}, "KiB")

# ------------------------------------------------------------------ figure
ink, sub = "#1f2933", "#5f6b76"
plt.rcParams.update({"text.color": ink})
k = 5 * (a.slice_frame - f0)
fig = plt.figure(figsize=(17, 9.6), constrained_layout=True); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
ext = [0, (c1 - c0) * H, (r1 - r0) * H, 0]
imgaxes = []
ax = fig.add_subplot(gs[0, 0]); imgaxes.append(ax); im = ax.imshow(sdfB[k], cmap="RdBu", vmin=-120, vmax=120, extent=ext, interpolation="bilinear")
ax.contour(np.linspace(0, ext[1], sdfB.shape[2]), np.linspace(0, ext[2], sdfB.shape[1]), sdfB[k], levels=[0], colors=["k"], linewidths=1.6)
ax.set_title("A  signed field of cell …338: zero level set = its surface\n(blue inside, red outside); no gap", fontsize=10.5)
fig.colorbar(im, ax=ax, location="bottom", fraction=0.05, pad=0.12, label="signed field (nm; twice the distance to the interface)")
ax = fig.add_subplot(gs[0, 1]); imgaxes.append(ax); im = ax.imshow(np.minimum(margin[k], 160), cmap="viridis", extent=ext, interpolation="bilinear")
ax.contour(np.linspace(0, ext[1], margin.shape[2]), np.linspace(0, ext[2], margin.shape[1]), margin[k], levels=[w], colors=["w"], linewidths=1.6)
ax.set_title(f"B  margin = |field| to the nearest interface (nm)\nthe gap is margin < {w:g} nm (white contour)", fontsize=10.5)
fig.colorbar(im, ax=ax, location="bottom", fraction=0.05, pad=0.12, label="margin (nm)")
ax = fig.add_subplot(gs[0, 2]); imgaxes.append(ax)
img = np.ones(lab[k].shape + (3,)) * 0.86; img[lab[k] == ia] = hues[ia]; img[lab[k] == ib] = hues[ib]
img[margin[k] < w] = (0.08, 0.08, 0.1)
ax.imshow(img, extent=ext, interpolation="nearest"); ax.set_title(f"C  cells with the gap opened (black), frame {a.slice_frame}", fontsize=10.5)
for axx in imgaxes:
    axx.set_xlabel("y (nm)"); axx.set_ylabel("x (nm)")

from skimage import measure
fieldc = {kk: comp(kk) for kk in fields}
nslab = 30; kz0 = max(k - nslab, 0)
X = (c1 - c0) * H; Y = (r1 - r0) * H; Zs = (k - kz0) * H


def block(ax, level, title):
    """Cut-away block (the slab of slices below the slice shown above): every cell at the level set  field = level."""
    for kk in fieldc:
        c = fieldc[kk][kz0:k + 1]
        if not (lab[kz0:k + 1] == kk).any():
            continue
        try:
            v, f, _, _ = measure.marching_cubes(np.pad(c, 1, constant_values=-(level + 400.0)), level, step_size=1)
        except ValueError:
            continue
        t = ((v - 1.0)[f])[:, :, [2, 1, 0]] * H
        nn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-12
        Ld = np.array([0.4, -0.5, 0.75]); Ld /= np.linalg.norm(Ld); sh = 0.5 + 0.5 * np.abs(nn @ Ld)
        base = np.array(hues.get(kk, (0.72, 0.72, 0.76)))
        ax.add_collection3d(Poly3DCollection(t, facecolors=np.clip(base[None] * sh[:, None], 0, 1), linewidths=0))
    ax.set_xlim(0, X); ax.set_ylim(Y, 0); ax.set_zlim(0, Zs); ax.set_box_aspect((X, Y, 2.2 * Zs)); ax.view_init(46, -62); ax.set_axis_off(); ax.set_title(title, fontsize=10.5)


block(fig.add_subplot(gs[1, 0], projection="3d"), 0.0, "D  surfaces at level 0: neighbouring cells touch")
block(fig.add_subplot(gs[1, 1], projection="3d"), w, f"E  surfaces at level {w:g} nm: each cell shrunk by {w / 2:g} nm; the grooves are the gap")
ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
ax.text(0, 1, "Open it in ParaView", fontsize=12, fontweight="bold", va="top")
ax.text(0, 0.9, "File > Open  results/sdf_export/fields.vti  (Apply)\n"
        "Slice: colour by sdf_A, sdf_B or margin\n"
        "Contour on sdf_B at 0  (touching surface)\n"
        f"Contour on sdf_B at {w:g}  (gapped surface)\n"
        f"Threshold gap_{w:g}nm  = 1  (the gap as a solid)\n"
        "Volume rendering of margin: opacity high where small\n"
        "Meshes: File > Open  cells_touching.ply, cells_gap*.ply\n"
        "(vertex colours: Coloring > RGB)\n\n"
        "Units are nm; the file's X axis is the last array axis.\n"
        "Fields are on the 8 nm isotropic grid; the gap is where\n"
        "margin < w, i.e. the zero-level shell of thickness w.", fontsize=10, va="top", family="monospace", linespacing=1.5)
fig.suptitle("The gap in the signed-distance description: level 0 (cells touch) versus level w (gap of width w), cells …779 (orange) and …338 (blue)",
             fontsize=12.5, fontweight="bold", x=0.01, ha="left")
fig.text(0.01, 0.005, "The gap is a model (w is a prior). The same fields and meshes are in results/sdf_export/ (fields.vti, *.ply) for ParaView.", fontsize=9, color=sub)
p = os.path.join(ROOT, "results", "sdf_gap.png"); fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white"); print(p)
