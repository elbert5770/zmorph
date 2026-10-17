"""sdf_align on the real '485' cell (segment 864691136697855485) of the Figure 12 gap: sections 17415 -> 17416
(zframe 87075 -> 87080), the manuscript's fusion case (two regions in the lower frame, one in the upper).

Same layout as make_method_figure.py, but every panel is computed from the real mode-filtered sections:
  row 1  the two sections (crop), the signed-distance field of the 485 cell in each, the label-level translation test
  row 2  the 485 field interpolated linearly in z; its zero contour is the cell boundary
  row 3  all labels interpolate, the highest field wins each pixel (the 485 cell is blue, the rest muted)
  row 4  fields of the competing labels along a line through the fusion, and the numbers

usage: python examples/make_method_figure_real.py [--raw]      (writes results/method_sdf_align_real_485.png)
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np, scipy.ndimage as ndi, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from real_io import DATA, load_labels
from zmorph.interp import competition, _field, _estimate_shift

T = "170398_61241"; K = 864691136697855485
ZA, ZB, FACTOR = 17415, 17416, 5
X0, X1, Y0, Y1 = 250, 450, 100, 300            # rows = x, columns = y (as in the manuscript's Figure 9/12 views)
M = 32                                         # context margin so the borders of the crop are not edge effects
use_raw = "--raw" in sys.argv
if use_raw:
    tile = np.load(os.path.join(ROOT, "data", f"tile_{T}.npz")); zi = list(tile["z"])
    sec = [tile["ids"][tile["vol"][zi.index(z)]] for z in (ZA, ZB)]
else:
    sec = [load_labels(os.path.join(DATA, f"sframe_{z}_{T}_4.csv.gz")) for z in (ZA, ZB)]
big = (slice(X0 - M, X1 + M), slice(Y0 - M, Y1 + M)); inner = (slice(M, M + X1 - X0), slice(M, M + Y1 - Y0))
S = [s[big] for s in sec]
ts = np.arange(FACTOR + 1) / FACTOR

lab, best, second, F = competition(S, FACTOR, "sdf", align=True, return_fields=True)
crop = lambda a: a[..., inner[0], inner[1]]
m0, m1 = S[0] == K, S[1] == K
p = dict(sigma=1.0, subvoxel=True, absent=4.0, cap=12.0)
f0, f1 = _field(m0, p), _field(m1, p)
d = _estimate_shift(m0, m1, iou_gate=1.01)
shifted = ndi.shift(m0.astype(float), d, order=0) > 0.5
iou = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)
ncomp = lambda m: ndi.label(m)[1]

# ---- style ---------------------------------------------------------------------------------------------------------
BLUE = "#0072B2"
rng = np.random.default_rng(3); ids = np.unique(np.concatenate([s.ravel() for s in S] + [lab.ravel()]))
MUTED = {k: tuple(np.clip(0.62 + 0.30 * np.array(plt.cm.tab20(rng.integers(0, 20))[:3]), 0, 1)) for k in ids}   # pale, so the 485 cell stands out
FIELD = dict(cmap="RdBu", norm=TwoSlopeNorm(0, -20, 20))
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold"})
fig = plt.figure(figsize=(17, 14.2))
gs = fig.add_gridspec(4, 6, height_ratios=[1, 1, 1, 1.05], hspace=0.52, wspace=0.10)
EXT = [X0, X1, Y0, Y1]


def ax_(r, c, title=None, sub=None):
    a = fig.add_subplot(gs[r, c]); a.set_xticks([]); a.set_yticks([])
    if title: a.set_title(title, loc="left")
    if sub: a.set_xlabel(sub, fontsize=8, color="#444")
    return a


def rgb(L):
    out = np.zeros(L.shape + (3,))
    for k in np.unique(L): out[L == k] = (0.0, 0.447, 0.698) if k == K else MUTED[k]
    return out


def outline(a, mask, color="k", lw=1.6, ls="-"):
    a.contour(np.arange(X0, X1) + .5, np.arange(Y0, Y1) + .5, mask.T, [.5], colors=color, linewidths=lw, linestyles=ls)


def show_labels(a, L):
    a.imshow(rgb(L).transpose(1, 0, 2), origin="lower", extent=EXT, interpolation="nearest"); outline(a, L == K)


def show_field(a, f):
    im = a.imshow(f.T, origin="lower", extent=EXT, interpolation="bilinear", **FIELD)
    a.contour(np.arange(X0, X1) + .5, np.arange(Y0, Y1) + .5, f.T, [0], colors="k", linewidths=1.8)
    return im


# ---- row 1 ---------------------------------------------------------------------------------------------------------
a = ax_(0, 0, "1  Two sections", f"section {ZA}: the 485 cell has {ncomp(crop(m0))} regions"); show_labels(a, crop(S[0]))
a = ax_(0, 1, None, f"section {ZB}: {ncomp(crop(m1))} region (fusion)"); show_labels(a, crop(S[1]))
a = a_f0 = ax_(0, 2, "2  Signed distance of the 485 label", f"f at {ZA}"); im = show_field(a, crop(f0))
a = a_f1 = ax_(0, 3, None, f"f at {ZB}"); show_field(a, crop(f1))
fig.canvas.draw(); p2, p3 = a_f0.get_position(), a_f1.get_position()
cax = fig.add_axes([p2.x0 + 0.25 * (p3.x1 - p2.x0), p2.y0 - 0.036, 0.5 * (p3.x1 - p2.x0), 0.008])
cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_ticks([-12, 0, 12]); cb.set_ticklabels(["-12 outside", "0 boundary", "+12 inside"]); cb.ax.tick_params(labelsize=7)
a = ax_(0, 4, "3  Translation test", f"estimated d = ({d[0]:+.0f}, {d[1]:+.0f}) px; IoU {iou(m0, m1):.2f} -> {iou(shifted, m1):.2f}" + ("" if d.any() else "  (not applied)"))
a.imshow(np.ones((Y1 - Y0, X1 - X0, 3)) * .97, origin="lower", extent=EXT)
outline(a, crop(m0), BLUE, 1.8, "--"); outline(a, crop(m1), "k", 1.8)
if d.any(): outline(a, crop(shifted), BLUE, 1.8)
a.text(X0 + 4, Y0 + 3, "dashed: lower section   black: upper section", fontsize=6.3, va="bottom")
a = ax_(0, 5, None); a.axis("off")
a.text(0, 1, f"{'raw' if use_raw else 'mode-filtered'} sections\n{ZA} -> {ZB}\n(zframe 87075 -> 87080)\n\ncrop rows {X0}-{X1},\ncols {Y0}-{Y1}\n\nblue = the 485 cell,\nother cells muted\n\ntranslation is applied\nonly if it clearly raises\nthe overlap", va="top", fontsize=8.6, family="monospace", linespacing=1.3)

# ---- rows 2 and 3 ---------------------------------------------------------------------------------------------------
for j, t in enumerate(ts):
    z = 87075 + j
    tag = f"t = {t:.1f}  (zframe {z})" + ("  = section" if j in (0, FACTOR) else "")
    a = ax_(1, j, "4  f(t) = (1-t) f(z0) + t f(z1): zero contour = boundary" if j == 0 else None, tag); show_field(a, crop(F[K][j]))
    a = ax_(2, j, "5  All labels interpolate; the highest field wins each pixel" if j == 0 else None, tag); show_labels(a, crop(lab[j]))

# ---- row 4: profile through the fusion + numbers -------------------------------------------------------------------
from scipy.spatial.distance import cdist
lbl0, n0 = ndi.label(crop(m0))
edge = lambda m: np.argwhere(m & ~ndi.binary_erosion(m))
pa, pb = edge(lbl0 == 1), edge(lbl0 == 2)
D = cdist(pa, pb); ia, ib = np.unravel_index(np.argmin(D), D.shape)
(ya, xa), (yb, xb) = pa[ia].astype(float), pb[ib].astype(float)   # closest points of the two regions (array coordinates in the crop)
u = np.array([yb - ya, xb - xa]); u /= np.linalg.norm(u)
ya, xa, yb, xb = ya - 12 * u[0], xa - 12 * u[1], yb + 12 * u[0], xb + 12 * u[1]   # extend a little beyond both regions
tj = 3                                                         # t = 0.6: the regions have just fused
n = 200; rr = np.linspace(ya, yb, n); cc = np.linspace(xa, xb, n)
dist = np.hypot(rr - ya, cc - xa)
prof = {k: ndi.map_coordinates(crop(F[k][tj]), [rr, cc], order=1) for k in F}
top = [K] + [k for k in sorted(prof, key=lambda k: -prof[k].max()) if k != K][:3]
a = fig.add_subplot(gs[3, 0:3]); a.set_title(f"6  Fields along the shortest gap between the two regions at t = {ts[tj]:.1f} (zframe {87075 + tj})", loc="left")
for k in top:
    a.plot(dist, prof[k], lw=2.4 if k == K else 1.6, color=BLUE if k == K else MUTED[k] if False else plt.cm.tab10(top.index(k)), label="485 cell" if k == K else f"...{str(k)[-5:]}")
a.axhline(0, color="#888", lw=.8)
win = np.array([max(prof, key=lambda k: prof[k][i]) for i in range(n)])
y0 = a.get_ylim()[0]
for i in range(n - 1): a.add_patch(plt.Rectangle((dist[i], y0), dist[i + 1] - dist[i], 1.2, color=(0.0, 0.447, 0.698) if win[i] == K else "#BBBBBB", lw=0))
a.set_xlabel("distance along the gap (px)   bar: blue where the 485 cell wins"); a.set_ylabel("field value")
a.legend(frameon=False, ncol=2, fontsize=8, loc="upper right"); a.spines[["top", "right"]].set_visible(False)
a.set_xlim(0, dist.max())

a = fig.add_subplot(gs[3, 3:6]); a.axis("off")
comps = [ncomp(crop(lab[j]) == K) for j in range(FACTOR + 1)]; area = [int((crop(lab[j]) == K).sum()) for j in range(FACTOR + 1)]
a.text(0, 1, "The 485 cell through the gap\n\n"
       f"regions per frame : {comps}\n"
       f"area (px)         : {area}\n\n"
       "Pixels never change by hand: the cell's boundary is the\n"
       "zero contour of one smooth field; the two lobes' fields\n"
       "merge across the gap and the contour fuses. The\n"
       "neighbouring cells are pushed aside because their own\n"
       "fields fall below the 485 field there; no gaps or overlaps.", va="top", ha="left", fontsize=8.6, family="monospace", linespacing=1.35)
fig.suptitle(f"sdf_align on the real 485 cell: sections {ZA} -> {ZB} (Figure 12 fusion case)", fontsize=11.5, fontweight="bold", y=0.945)
out = os.path.join(ROOT, "results", "method_sdf_align_real_485" + ("_raw" if use_raw else "") + ".png")
fig.savefig(out, dpi=105, bbox_inches="tight"); print("wrote", out, "| d", d, "| comps", comps, "| area", area)
