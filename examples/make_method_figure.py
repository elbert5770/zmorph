"""Schematic of the sdf_align method (the counterpart of the manuscript's Figures 6, 8 and 11 for the heat method).

A single cell (blue) occupies one region in the upper section and two regions, displaced sideways, in the lower
section (the Figure 6 branching case). Everything drawn is computed by zmorph itself (same functions as
reconstruct(..., 'sdf', align=True)), not hand-drawn:

  row 1  the two sections; the signed-distance field of the blue cell in each; the translation estimate
  row 2  that field interpolated linearly in z (t = 0 ... 1); its zero contour is the cell's boundary
  row 3  every label does the same; the label with the highest field wins each pixel (a partition, no gaps/overlaps)
  row 4  one cross-section through the competition, and the recipe

usage: python examples/make_method_figure.py          (writes results/method_sdf_align.png)
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np, scipy.ndimage as ndi, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from zmorph.interp import competition, _field, _estimate_shift

# ---- scene (labels: 1 neurite, 2 left cell, 3 upper-right cell, 4 background) -------------------------------------
N = 80
yy, xx = np.mgrid[:N, :N]


def section(blobs, seeds):
    seeds = np.array(seeds, float)
    d = np.stack([np.hypot(yy - r, xx - c) for r, c in seeds])
    lab = np.array([2, 3, 4])[np.argmin(d, 0)]
    for r, c, ry, rx in blobs:
        lab[((yy - r) / ry) ** 2 + ((xx - c) / rx) ** 2 <= 1] = 1
    return lab


upper = section([(40, 40, 21, 9)], [(38, 6), (8, 66), (72, 66)])
lower = section([(27, 40, 10, 9), (54, 42, 10, 9)], [(40, 9), (10, 68), (70, 64)])
S = [upper, lower]
K = 1
FACTOR = 4
ts = np.arange(FACTOR + 1) / FACTOR

lab, best, second, F = competition(S, FACTOR, "sdf", align=True, return_fields=True)
p = dict(sigma=1.0, subvoxel=True, absent=4.0, cap=12.0)
f0, f1 = _field(upper == K, p), _field(lower == K, p)
# translation example: a thin neurite that moved farther than its own width (a separate, single-region scene)
ma = ((yy - 36) / 5.0) ** 2 + ((xx - 22) / 11.0) ** 2 <= 1
mb = ((yy - 46) / 5.0) ** 2 + ((xx - 56) / 11.0) ** 2 <= 1
d = _estimate_shift(ma, mb)
shifted = ndi.shift(ma.astype(float), d, order=0) > 0.5
iou = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)

# ---- style --------------------------------------------------------------------------------------------------------
COL = {1: "#0072B2", 2: "#E69F00", 3: "#009E73", 4: "#BBBBBB"}          # Okabe-Ito: distinguishable under CVD
cm_lab = ListedColormap([COL[k] for k in (1, 2, 3, 4)])
img = lambda L: L - 1
FIELD = dict(cmap="RdBu", norm=TwoSlopeNorm(0, -20, 20))                  # + inside (blue), - outside (red), grey-white = boundary
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold"})
fig = plt.figure(figsize=(15, 13.6))
gs = fig.add_gridspec(4, 5, height_ratios=[1, 1, 1, 1.05], hspace=0.52, wspace=0.12)


def ax_(r, c, title=None, sub=None):
    a = fig.add_subplot(gs[r, c]); a.set_xticks([]); a.set_yticks([])
    if title: a.set_title(title, loc="left")
    if sub: a.set_xlabel(sub, fontsize=8, color="#444")
    return a


def show_labels(a, L):
    a.imshow(img(L), cmap=cm_lab, vmin=0, vmax=3, interpolation="nearest")
    a.contour(L == K, [.5], colors="k", linewidths=1.6)


def show_field(a, f, zero_color="k"):
    im = a.imshow(f, interpolation="bilinear", **FIELD)
    a.contour(f, [0], colors=zero_color, linewidths=1.8)
    return im


# ---- row 1 ---------------------------------------------------------------------------------------------------------
a = ax_(0, 0, "1  Two sections", "z0: blue cell = one region"); show_labels(a, upper)
a = ax_(0, 1, None, "z1: two regions (branching)"); show_labels(a, lower)
a = a_f0 = ax_(0, 2, "2  Signed distance of ONE label", "f at z0"); im = show_field(a, f0)
a = a_f1 = ax_(0, 3, None, "f at z1"); show_field(a, f1)
A2 = a_f0; fig.canvas.draw(); p2, p3 = A2.get_position(), a_f1.get_position(); cax = fig.add_axes([p2.x0 + 0.25 * (p3.x1 - p2.x0), p2.y0 - 0.036, 0.5 * (p3.x1 - p2.x0), 0.008]); cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_ticks([-12, 0, 12]); cb.set_ticklabels(["-12 outside", "0 boundary", "+12 inside"]); cb.ax.tick_params(labelsize=7)
a = ax_(0, 4, "3  Translation (only if needed)", f"d = ({d[0]:+.0f}, {d[1]:+.0f}) px; overlap IoU {iou(ma, mb):.2f} -> {iou(shifted, mb):.2f}")
a.imshow(np.ones((N, N, 3)) * .97)
a.contour(ma, [.5], colors=COL[1], linewidths=1.8, linestyles="--")
a.contour(mb, [.5], colors="k", linewidths=1.8)
a.contour(shifted, [.5], colors=COL[1], linewidths=1.8)
cy, cx = ndi.center_of_mass(ma)
if d.any(): a.annotate("", xy=(cx + d[1], cy + d[0]), xytext=(cx, cy), arrowprops=dict(arrowstyle="-|>", color="#D55E00", lw=2))
a.text(2, 78, "a thin cell that moved farther than its width\ndashed: z0   black: z1   solid blue: z0 + d", fontsize=6.3, va="bottom")

# ---- row 2: one label's field through z --------------------------------------------------------------------------
for j, t in enumerate(ts):
    a = ax_(1, j, "4  f(t) = (1-t) f(z0) + t f(z1): the zero contour is the boundary" if j == 0 else None, f"t = {t:.2f}" + ("  (= z0)" if t == 0 else "  (= z1)" if t == 1 else ""))
    show_field(a, F[K][j])

# ---- row 3: competition --------------------------------------------------------------------------------------------
for j, t in enumerate(ts):
    a = ax_(2, j, "5  Every label interpolates; the highest field wins each pixel" if j == 0 else None, f"t = {t:.2f}")
    show_labels(a, lab[j])

# ---- row 4: cross-section + recipe --------------------------------------------------------------------------------
tj, row = 2, 40
a = fig.add_subplot(gs[3, 0:3]); a.set_title("6  Why it is a partition: fields along one line at t = 0.5 (row %d)" % row, loc="left")
x = np.arange(N)
for k in (1, 2, 3, 4):
    a.plot(x, F[k][tj][row], color=COL[k], lw=2.2, label={1: "blue cell (neurite)", 2: "orange", 3: "green", 4: "grey"}[k])
a.axhline(0, color="#888", lw=.8)
win = lab[tj][row]
y0 = a.get_ylim()[0]
for xi in range(N): a.add_patch(plt.Rectangle((xi - .5, y0), 1, 1.2, color=COL[win[xi]], lw=0))
a.set_xlim(-.5, N - .5); a.set_xlabel("x (pixels)   bar at the bottom = winning label"); a.set_ylabel("field value")
a.legend(frameon=False, ncol=2, fontsize=8, loc="upper right"); a.spines[["top", "right"]].set_visible(False)

a = fig.add_subplot(gs[3, 3:5]); a.axis("off")
recipe = ("Recipe  (sdf_align)\n\n"
          "1  Per section, per label k: mask  ->  signed distance\n"
          "    (half-pixel offset, Gaussian sigma = 1 px, clipped at +-12)\n"
          "2  If k barely overlaps itself across the gap (panel 3):\n"
          "    estimate a translation d by cross-correlation and\n"
          "    apply it only when it clearly raises the overlap\n"
          "    (not needed in this branching scene)\n"
          "3  f_k(t) = (1-t) f_k(z0) + t f_k(z1)\n"
          "    absent label: constant -4 (so it can still win near\n"
          "    its own section)\n"
          "4  Label of a pixel = argmax_k f_k(t)\n\n"
          "No paths, no boundary tracking, no per-cell solves.\n"
          "Sources are reproduced (the field is >0 only inside k at\n"
          "t = 0, 1). Branching and fusing fall out of step 3:\n"
          "one zero contour splits into two (or the reverse).")
a.text(0, 1, recipe, va="top", ha="left", fontsize=8.6, family="monospace", linespacing=1.35)
fig.suptitle("sdf_align: signed distance per label, linear in z, highest field wins   (Figure 6 branching case, computed by zmorph)", fontsize=11.5, fontweight="bold", y=0.945)
out = os.path.join(ROOT, "results", "method_sdf_align.png")
fig.savefig(out, dpi=110, bbox_inches="tight"); print("wrote", out, "| d =", d, "| iou", iou(upper == K, lower == K), "->", iou(shifted, lower == K))
