import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from zmorph import phantom
from zmorph.interp import reconstruct
from zmorph.mesh import taubin_joint, competition_meshes, smoothed_indicator_meshes, stair_meshes, interface_conformity, quality

gt = phantom.scene_branch()
clean = [gt[z] for z in range(0, 31, 5)]
secs = phantom.corrupt(clean, np.random.default_rng(1), bad=3)

lab, prop = competition_meshes(secs, 5, clean=False) if False else (None, None)
from zmorph.interp import clean_sections
cs = clean_sections(secs)
lab, prop = competition_meshes(cs, 5, align=True)
rows = {
    "repeat x5 + binary marching cubes (stair-step)": stair_meshes(reconstruct(secs, 5, "nearest")),
    "proposed labels + smoothed-indicator meshes (sigma=1)": smoothed_indicator_meshes(lab, 1.0),
    "proposed competition-field meshes": prop,
    "proposed + joint Taubin (15 it.)": taubin_joint(prop),
}
out = ["| pipeline | triangles | min-angle p5 (deg) | dihedral mean / p95 (deg) | interface vertices shared (<1e-3 vox) | interface median / p95 vertex distance (vox) |", "|---|---|---|---|---|---|"]
for n, m in rows.items():
    q, c = quality(m), interface_conformity(m)
    out.append(f"| {n} | {q['triangles']} | {q['min_angle_p5']:.1f} | {q['dihedral_mean']:.1f} / {q['dihedral_p95']:.1f} | {100*c['frac_shared']:.1f}% | {c['median_dist']:.3f} / {c['p95_dist']:.3f} |")
md = "\n".join(out); print(md); open("results/mesh.md", "w").write(md)

fig = plt.figure(figsize=(11, 4.5))
for i, (n, m) in enumerate([("stair-step", rows[list(rows)[0]]), ("proposed", prop)]):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    for k, col in ((1, "#ff7f0e"), (2, "#2ca02c")):
        v, f = m[k]
        tri = Poly3DCollection(v[f][:, :, [2, 1, 0]], alpha=0.9, linewidths=0.05, edgecolor=(0, 0, 0, .25))
        tri.set_facecolor(col); ax.add_collection3d(tri)
    ax.set_xlim(0, 80); ax.set_ylim(20, 76); ax.set_zlim(0, 30); ax.view_init(22, -62)
    ax.set_box_aspect((80, 56, 30)); ax.set_title(n); ax.set_axis_off()
plt.tight_layout(); plt.savefig("results/mesh_arch.png", dpi=110)
