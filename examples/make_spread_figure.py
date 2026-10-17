"""Figure: spread of tracers (Abeta-sized, D = 1.5e8 nm^2/s assumed) released from a candidate cleft.

Needs the geometry cache written by run_synapse_sweep.py (results/synapse_source/geom_<tile>.npz).
Top row: x-y projection of the ECS (grey) with tracer positions at four times; the cleft is marked.
Bottom: rms displacement against free-diffusion references, and the distance distribution at three times.
usage: python examples/make_spread_figure.py [--tile 170398_60741] [--w 20] [--n 3000]
"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from zmorph.synapse import release_positions
from zmorph.bd import simulate_dynamic

ap = argparse.ArgumentParser()
ap.add_argument("--tile", default="170398_60741"); ap.add_argument("--w", type=float, default=20.0)
ap.add_argument("--n", type=int, default=3000); ap.add_argument("--tmax", type=float, default=2e5)
a = ap.parse_args()
H, D_NM2_S = 8.0, 1.5e8                       # nm; assumed Abeta monomer D, nm^2/s (1.5e-10 m^2/s)
out = os.path.join(ROOT, "results", "synapse_source")
g = np.load(os.path.join(out, f"geom_{a.tile}.npz")); m = g["m"].astype(float); center = g["center"]; pts = g["points"]


class _P:
    points = pts


x0 = release_positions(_P, m, H, a.w, a.n, seed=1) / H
dt, nrec = 0.5, 40; spr = int(a.tmax / dt / nrec)
r = simulate_dynamic(m, H, x0, lambda t: a.w, 1.0, dt, nrec=nrec, steps_per_rec=spr, seed=2)
t, pos = r["t"], r["pos"]
ms = lambda tt: tt / D_NM2_S * 1e3            # model time (nm^2/D) -> ms

ink, sub, grid = "#1f2933", "#5f6b76", "#d5d9de"
blue, orange = "#1d6fb8", "#d9622b"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": ink, "text.color": ink,
                     "xtick.color": sub, "ytick.color": sub, "axes.spines.top": False, "axes.spines.right": False})
ext = [0, m.shape[2] * H / 1e3, m.shape[1] * H / 1e3, 0]
fig = plt.figure(figsize=(13, 7.6)); gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.38, wspace=0.18)
for j, k in enumerate([0, 2, 10, nrec]):
    ax = fig.add_subplot(gs[0, j])
    ax.set_xlim(0, ext[1]); ax.set_ylim(ext[2], 0)
    ax.scatter(pos[k, :, 2] / 1e3, pos[k, :, 1] / 1e3, s=3, c=blue, alpha=0.45, linewidths=0)
    ax.scatter([center[2] / 1e3], [center[1] / 1e3], s=70, marker="o", facecolors="none", edgecolors=orange, linewidths=1.8)
    ax.set_title(f"{ms(t[k]):.2g} ms" if k else "release (t = 0)", fontsize=11, color=ink)
    ax.set_xlabel("x (µm)"); ax.set_aspect("equal")
    if j == 0:
        ax.set_ylabel("y (µm)")
    else:
        ax.set_yticklabels([])
fig.text(0.125, 0.955, "Spread of Aβ-sized tracers from a candidate cleft (circle) through the extracellular gap",
         fontsize=13, fontweight="bold", ha="left")
fig.text(0.125, 0.925, f"Tile {a.tile}, w = {a.w:g} nm, {a.n} tracers, projection along z. The crop edge (2 µm) is a reflecting wall, "
         "which limits spread to the right. Times assume D = 1.5×10⁻¹⁰ m²/s.", fontsize=9.5, color=sub, ha="left")

ax = fig.add_subplot(gs[1, :2])
msd = ((pos - pos[0]) ** 2).sum(2).mean(1)
ax.plot(ms(t), np.sqrt(msd) / 1e3, color=blue, lw=2, label="simulation (rms displacement)")
ax.plot(ms(t), np.sqrt(6 * t) / 1e3, color=sub, lw=1.2, ls="--", label="free 3-D diffusion, √(6Dt)")
ax.plot(ms(t), np.sqrt(4 * t) / 1e3, color=sub, lw=1.2, ls=":", label="flat sheet, √(4Dt)")
ax.set_xlabel("time (ms)"); ax.set_ylabel("rms displacement (µm)"); ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.grid(color=grid, lw=0.5); ax.set_axisbelow(True)

ax = fig.add_subplot(gs[1, 2:])
dist = np.linalg.norm(pos - center, axis=2) / 1e3
bins = np.linspace(0, 2.4, 49)
for k, c, al in [(2, "#9bc3e6", 1), (10, "#4f93cf", 1), (nrec, "#14518a", 1)]:
    ax.hist(dist[k], bins=bins, density=True, histtype="stepfilled", color=c, alpha=0.55, edgecolor=c, lw=1.2,
            label=f"{ms(t[k]):.2g} ms")
ax.set_xlabel("distance from cleft centre (µm)"); ax.set_ylabel("density of tracers (per µm)")
ax.legend(frameon=False, fontsize=9, title="time", title_fontsize=9); ax.grid(color=grid, lw=0.5); ax.set_axisbelow(True)
p = os.path.join(out, f"spread_{a.tile}.png"); fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white"); print(p)
