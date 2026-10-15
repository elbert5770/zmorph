import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from zmorph import phantom
from zmorph.interp import reconstruct, majority

gt = phantom.scene_branch()
clean = [gt[z] for z in range(0, 31, 5)]
secs = phantom.corrupt(clean, np.random.default_rng(1), bad=3)
vols = {
    "ground truth": gt,
    "nearest (repeat x5)": reconstruct(secs, 5, "nearest"),
    "heat-paths + majority (manuscript)": majority(reconstruct(secs, 5, "heat", clean=True), 2),
    "sdf (proposed)": reconstruct(secs, 5, "sdf", clean=True, align=True),
}
cmap = plt.get_cmap("tab10")
def rgb(sl):
    return cmap(sl % 10)[..., :3] * (sl[..., None] > 0) + 0.93 * (sl[..., None] == 0)

# xy slices through the arch gap (sections z=15 and z=20 are observed; 16-19 are interpolated)
zs = [15, 16, 17, 18, 19, 20]
fig, ax = plt.subplots(len(vols), len(zs), figsize=(2.3 * len(zs), 1.9 * len(vols)))
for i, (n, v) in enumerate(vols.items()):
    for j, z in enumerate(zs):
        ax[i, j].imshow(rgb(v[z][25:72, 5:80]), interpolation="nearest")
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        if i == 0: ax[i, j].set_title(f"z={z}" + (" (section)" if z % 5 == 0 else ""), fontsize=9)
        if j == 0: ax[i, j].set_ylabel(n, fontsize=7)
plt.tight_layout(); plt.savefig("results/fig6_xy_slices.png", dpi=110); plt.close()

# xz cut through the legs/bridge (y=48): the reconstructed arch
fig, ax = plt.subplots(1, len(vols), figsize=(3.6 * len(vols), 3))
for a, (n, v) in zip(ax, vols.items()):
    a.imshow(rgb(v[:, 48, :80]), interpolation="nearest", aspect=1.2, origin="lower")
    a.set_title(n, fontsize=8); a.set_xticks([]); a.set_yticks([])
    for z in range(0, 31, 5): a.axhline(z, color="k", lw=0.3, alpha=0.4)
plt.tight_layout(); plt.savefig("results/fig6_xz_cut.png", dpi=110); plt.close()
