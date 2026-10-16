"""Figures of the tilted-neurite and three-way-split phantoms and how the methods reconstruct them.
Writes results/tilted_scene.png, split3_scene.png, split3_methods.png."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from zmorph import phantom
from zmorph.interp import reconstruct, majority

def lab_rgb(sl, focus=(1,), cmap_focus=("#d62728", "#1f77b4", "#2ca02c"), grey=0.86):
    out = np.full(sl.shape + (3,), grey)
    for l in np.unique(sl):
        if l == 0: out[sl == l] = 1.0
        elif l in focus: out[sl == l] = matplotlib.colors.to_rgb(cmap_focus[focus.index(l) % 3])
        else: out[sl == l] = grey - 0.08
    return out

def topdown(gt, labels, ax, title):
    """max projection over z of each neurite, colour = height z (viridis); section planes marked in the legend."""
    nz = gt.shape[0]
    img = np.full(gt.shape[1:] + (3,), 1.0)
    cm = plt.get_cmap("viridis")
    for z in range(nz):  # later z painted on top
        m = np.isin(gt[z], labels)
        img[m] = cm(z / (nz - 1))[:3]
    ax.imshow(img, interpolation="nearest"); ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, nz - 1)); plt.colorbar(sm, ax=ax, fraction=0.04, label="z (voxels; sections every 5)")

# ---------------------------------------------------------------- tilted scene
gt = phantom.scene_tilted()
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
topdown(gt, [1, 2, 3], ax[0, 0], "tilted neurites (cells 1-3) seen from above,\ncolour = height")
for a, z in zip(ax[0, 1:], (10, 15)):
    a.imshow(lab_rgb(gt[z], focus=(1, 2, 3)), interpolation="nearest")
    nxt = gt[z + 5]
    for l, c in ((1, "#d62728"), (2, "#1f77b4"), (3, "#2ca02c")):
        a.contour(nxt == l, levels=[0.5], colors=[c], linewidths=1.2, linestyles="--")
    a.set_title(f"section z={z} (filled) and z={z+5} (dashed outline)", fontsize=9); a.set_xticks([]); a.set_yticks([])
for a, y in zip(ax[1, :2], (25, 45)):
    a.imshow(lab_rgb(gt[:, y, :], focus=(1, 2, 3)), interpolation="nearest", aspect=2, origin="lower")
    for z in range(0, 31, 5): a.axhline(z, color="k", lw=0.4, alpha=0.4)
    a.set_title(f"xz cut at y={y}: the neurite is a diagonal; lines = sections", fontsize=9); a.set_xticks([]); a.set_yticks([])
ax[1, 2].axis("off")
ax[1, 2].text(0, 0.5, "Section footprints of one tilted neurite\n(radius 3.5 px, about 15 px shift per section gap)\n"
              "are stretched along the direction of motion,\nso consecutive footprints still overlap\n(IoU about 0.28). A straight tube of radius r\nonly stops overlapping itself for r < 2.5 px.", fontsize=10, va="center")
plt.tight_layout(); plt.savefig("results/tilted_scene.png", dpi=90); plt.close()

# ---------------------------------------------------------------- split scene
gt = phantom.scene_split3()
secs = [gt[z] for z in range(0, 26, 5)]
sl = (slice(24, 74), slice(35, 150))
fig, ax = plt.subplots(2, 4, figsize=(17, 6.5))
topdown(gt, [1], ax[0, 0], "neurite (label 1) from above,\ncolour = height; splits near z=15.5")
for a, z in zip(ax[0, 1:], (15, 19, 20)):
    a.imshow(lab_rgb(gt[z][sl]), interpolation="nearest")
    if z > 15: a.contour((gt[15][sl] == 1), levels=[0.5], colors=["k"], linewidths=1.2)
    a.set_title(f"z={z}" + (" (parent, section)" if z == 15 else " (3 components)" if z >= 19 else "") + ("   black = footprint at z=15" if z > 15 else ""), fontsize=9)
    a.set_xticks([]); a.set_yticks([])
for a, z in zip(ax[1], (16, 17, 18, 25)):
    a.imshow(lab_rgb(gt[z][sl]), interpolation="nearest"); a.contour((gt[15][sl] == 1), levels=[0.5], colors=["k"], linewidths=1.0)
    a.set_title(f"z={z}" + (" (section)" if z % 5 == 0 else ""), fontsize=9); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig("results/split3_scene.png", dpi=90); plt.close()

# ---------------------------------------------------------------- methods on the split scene
rows = {"ground truth": gt,
        "nearest (repeat)": reconstruct(secs, 5, "nearest"),
        "Raya-Udupa linear SDF": reconstruct(secs, 5, "sdf0"),
        "cubic-spline SDF (binary)": reconstruct(secs, 5, "sdf0_cubic"),
        "heat-paths port (raw, no filter)": reconstruct(secs, 5, "heat"),
        "proposed (linear)": reconstruct(secs, 5, "sdf"),
        "proposed + component translation": reconstruct(secs, 5, "sdf", align="component", maxshift=40),
        "proposed + persistence guard": reconstruct(secs, 5, "sdf", persist=True)}
zs = (15, 16, 17, 18, 19, 20)
fig, ax = plt.subplots(len(rows), len(zs), figsize=(2.9 * len(zs), 1.35 * len(rows)))
for r, (n, v) in enumerate(rows.items()):
    for c, z in enumerate(zs):
        a = ax[r, c]; a.imshow(lab_rgb(v[z][sl]), interpolation="nearest"); a.set_xticks([]); a.set_yticks([])
        a.contour((gt[z][sl] == 1), levels=[0.5], colors=["k"], linewidths=0.7)
        if r == 0: a.set_title(f"z={z}" + (" (section)" if z % 5 == 0 else ""), fontsize=9)
        if c == 0: a.set_ylabel(n, fontsize=7, rotation=0, ha="right", va="center")
plt.suptitle("Neurite (red) in the gap between sections 15 and 20; black outline = ground truth", fontsize=10)
plt.tight_layout(); plt.savefig("results/split3_methods.png", dpi=85); plt.close()
print("ok")
