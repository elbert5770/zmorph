"""Figure: the extracellular gap as a 3-D object, around a candidate synapse in the real tile crop.

A  a slice through the label volume with the gap { m < w } in black (what the cells look like once the gap is opened)
B  a cut-away block of the same crop: each cell shrunk by w/2 (its own surface), the gap is the groove between neighbours
C  the same gap, translucent, with tracers released from the cleft at three times (orange, light to dark = early to late)

The gap is derived from the labels, not measured: w is a prior (here 40 nm; the simulations used 16-40 nm). Needs
results/synapse_source/{geom,vol}_<tile>.npz (examples/make_geom_cache.py).

usage: python examples/make_ecs_figure.py [--w 40] [--half 640]
"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from zmorph.synapse import release_positions
from zmorph.bd import simulate_dynamic

ap = argparse.ArgumentParser()
ap.add_argument("--tile", default="170398_60741"); ap.add_argument("--w", type=float, default=40.0)
ap.add_argument("--half", type=float, default=640.0); ap.add_argument("--n", type=int, default=1500)
ap.add_argument("--step", type=int, default=2); ap.add_argument("--tmax", type=float, default=2e5)
a = ap.parse_args()
H = 8.0
out = os.path.join(ROOT, "results", "synapse_source")
g = np.load(os.path.join(out, f"geom_{a.tile}.npz")); m = g["m"].astype(np.float64); c = g["center"]; pts = g["points"]
vol = np.load(os.path.join(out, f"vol_{a.tile}.npz"))["vol"].astype(np.int64)

lo = np.maximum(np.round((c - a.half) / H).astype(int), 0); hi = np.minimum(np.round((c + a.half) / H).astype(int) + 1, m.shape)
sl = tuple(slice(l, h) for l, h in zip(lo, hi)); off = lo * H                    # crop origin in nm, axes (z, y, x)
mc, vc = m[sl], vol[sl]

# --- tracers (BD, same gap) ---
class _P:
    points = pts
x0 = release_positions(_P, m, H, a.w, a.n, seed=1) / H
nrec = 20; spr = int(a.tmax / 0.5 / nrec)
r = simulate_dynamic(m, H, x0, lambda t: a.w, 1.0, 0.5, nrec=nrec, steps_per_rec=spr, seed=2)
ms = lambda t: t / 1.5e8 * 1e3                                              # model time -> ms for D = 1.5e-10 m^2/s

# --- gap surface ---
pad = np.pad(mc - a.w, 1, constant_values=1.0)                                # > 0 outside: closes the surface at the crop box
v, f, nrm, _ = measure.marching_cubes(pad, 0.0, step_size=a.step)
v = (v - 1.0) * H + off                                                      # nm, (z, y, x)
print(f"gap surface: {len(f)} triangles; ECS fraction in crop {(mc < a.w).mean():.3f}")


def shade(v, f, base, alpha, light=(0.4, -0.5, 0.75)):
    t = v[f]; n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    L = np.array(light) / np.linalg.norm(light)
    s = 0.45 + 0.55 * np.abs(n @ L)
    col = np.clip(np.array(to_rgb(base))[None, :] * s[:, None], 0, 1)
    return np.concatenate([col, np.full((len(f), 1), alpha)], 1), t


def draw_surface(ax, base, alpha):
    col, t = shade(v, f, base, alpha)
    tri = Poly3DCollection(t[:, :, [2, 1, 0]] / 1e3, facecolors=col, linewidths=0)
    ax.add_collection3d(tri)


def setup3d(ax, title):
    lim = [(off[2], off[2] + (hi[2] - lo[2]) * H), (off[1], off[1] + (hi[1] - lo[1]) * H), (off[0], off[0] + (hi[0] - lo[0]) * H)]
    ax.set_xlim(*[q / 1e3 for q in lim[0]]); ax.set_ylim(*[q / 1e3 for q in lim[1]]); ax.set_zlim(*[q / 1e3 for q in lim[2]])
    ax.set_box_aspect([lim[0][1] - lim[0][0], lim[1][1] - lim[1][0], lim[2][1] - lim[2][0]])
    ax.view_init(24, -58); ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (µm)", fontsize=8); ax.set_ylabel("y (µm)", fontsize=8); ax.set_zlabel("z (µm)", fontsize=8)
    ax.tick_params(labelsize=7); ax.xaxis.pane.set_alpha(0.0); ax.yaxis.pane.set_alpha(0.0); ax.zaxis.pane.set_alpha(0.0)
    ax.grid(False)


PAL = np.array([to_rgb(h) for h in ["#8fb8d8", "#e3b778", "#a6c98f", "#d99aa5", "#b7a6d6", "#9fd1c7", "#e0cf86", "#c4a58a", "#9bb0c2", "#cdb4a3"]])
ink, sub = "#1f2933", "#5f6b76"
plt.rcParams.update({"text.color": ink, "axes.labelcolor": ink, "xtick.color": sub, "ytick.color": sub})
fig = plt.figure(figsize=(15.5, 6.2))


def colour(q):                                   # one colour per cell, shared by panels A and B
    return PAL[(int(q) * 2654435761 % 2**32) % len(PAL)]


# A: slice
ax = fig.add_subplot(1, 3, 1)
k = int(round(c[0] / H)) - lo[0]; k = min(max(k, 0), vc.shape[0] - 1)
lab = vc[k]; ids = np.unique(lab)
img = np.zeros(lab.shape + (3,))
for i, q in enumerate(ids):
    img[lab == q] = colour(q) if q else (1, 1, 1)
gap = mc[k] < a.w
img[gap] = (0.08, 0.08, 0.1)
ext = [off[2] / 1e3, (off[2] + lab.shape[1] * H) / 1e3, (off[1] + lab.shape[0] * H) / 1e3, off[1] / 1e3]
ax.imshow(img, extent=ext, interpolation="nearest")
ax.scatter([c[2] / 1e3], [c[1] / 1e3], s=110, facecolors="none", edgecolors="#d9622b", linewidths=2)
ax.set_title(f"A  slice: cells (colour), gap (black)\nw = {a.w:g} nm", fontsize=11)
ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
# B: cut-away block of shrunk cells
ax = fig.add_subplot(1, 3, 2, projection="3d")
from scipy.ndimage import gaussian_filter
nslab = 30                                                                    # voxels below the cut (240 nm)
k0 = max(k - nslab, 0)
blk_lab, blk_m = vc[k0:k + 1], mc[k0:k + 1]
for q in np.unique(blk_lab):
    if q == 0:
        continue
    mask = (blk_lab == q) & (blk_m >= a.w)
    if mask.sum() < 200:
        continue
    fld = gaussian_filter(np.pad(mask.astype(float), 1), 0.7)
    try:
        bv, bf, _, _ = measure.marching_cubes(fld, 0.5)
    except ValueError:
        continue
    bv = (bv - 1.0) * H + off + np.array([k0 * H, 0, 0])
    col, t = shade(bv, bf, tuple(colour(q)), 1.0)
    ax.add_collection3d(Poly3DCollection(t[:, :, [2, 1, 0]] / 1e3, facecolors=col, linewidths=0))
x_lo, x_hi = off[2], off[2] + (hi[2] - lo[2]) * H; y_lo, y_hi = off[1], off[1] + (hi[1] - lo[1]) * H
z_lo, z_hi = off[0] + k0 * H, off[0] + k * H
ax.set_xlim(x_lo / 1e3, x_hi / 1e3); ax.set_ylim(y_lo / 1e3, y_hi / 1e3); ax.set_zlim(z_lo / 1e3, z_hi / 1e3)
ax.set_box_aspect([x_hi - x_lo, y_hi - y_lo, 2.2 * (z_hi - z_lo)]); ax.view_init(48, -62)
ax.set_title("B  cut-away block: cells shrunk by w/2;\nthe gap is the groove between them", fontsize=11)
ax.set_xlabel("x (µm)", fontsize=8); ax.set_ylabel("y (µm)", fontsize=8); ax.set_zlabel("z (µm)", fontsize=8)
ax.tick_params(labelsize=7); ax.grid(False)
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.set_alpha(0.0)
# C: gap + tracers
ax = fig.add_subplot(1, 3, 3, projection="3d"); draw_surface(ax, "#6f8aa3", 0.05)
cols = ["#f7c9a8", "#ec8a4a", "#a8420f"]
for kk, cc in zip([3, 8, nrec], cols):
    p = r["pos"][kk]; ok = np.all((p >= off) & (p <= off + (hi - lo) * H), axis=1)
    ax.scatter(p[ok, 2] / 1e3, p[ok, 1] / 1e3, p[ok, 0] / 1e3, s=4, c=cc, alpha=0.85, linewidths=0, label=f"{ms(r['t'][kk]):.2g} ms")
ax.scatter([c[2] / 1e3], [c[1] / 1e3], [c[0] / 1e3], s=60, facecolors="none", edgecolors=ink, linewidths=1.5)
setup3d(ax, "C  tracers released from the cleft,\nmoving through the gap (translucent)")
from matplotlib.lines import Line2D
hs = [Line2D([], [], marker="o", ls="", ms=4, color=cc) for cc in cols] + [Line2D([], [], marker="o", ls="", ms=7, mfc="none", mec=ink)]
ax.legend(hs, [f"{ms(r['t'][kk]):.2g} ms" for kk in (3, 8, nrec)] + ["cleft"], loc="upper left", fontsize=8, frameon=False)
fig.suptitle("The extracellular space as a derived 3-D object: the gap is the set { m < w } of the label volume (tile crop, 8 nm voxels)",
             fontsize=12.5, fontweight="bold", x=0.01, ha="left")
fig.text(0.01, 0.005, f"The gap is computed from segmentation labels and the width w ({a.w:g} nm here) is a prior, not a measurement. "
         f"Crop {(hi[2] - lo[2]) * H / 1e3:.1f} x {(hi[1] - lo[1]) * H / 1e3:.1f} x {(hi[0] - lo[0]) * H / 1e3:.1f} µm (x, y, z; the tile edge limits it); times assume D = 1.5×10⁻¹⁰ m²/s.", fontsize=9, color=sub)
fig.subplots_adjust(left=0.05, right=0.99, top=0.80, bottom=0.14, wspace=0.12)
p = os.path.join(out, f"ecs_figure_{a.tile}.png"); fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white"); print(p)
