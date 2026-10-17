"""Tables and figure from results/diffusion_benchmark.json  ->  results/diffusion_benchmark.{md,png}"""
import os, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "results", "diffusion_benchmark.json")))
md = ["# Diffusion in the gap between cells: Brownian dynamics vs continuum solvers", "",
      "D = 1, lengths in nm, voxel size of the data 8 nm. Brownian dynamics (BD) runs on the implicit gap {margin < w} of the "
      "signed-distance fields (`zmorph/bd.py`); the continuum solvers are in `zmorph/diffusion_ref.py`. Errors are relative to the "
      "analytic or fine-grid answer; BD errors are +- one standard error over 10 particle groups.", ""]

# --- slab
md += ["## 1. Reflecting walls (planar gap, w = 20 nm)", "",
       "| step sigma (nm) | density in the outer 20% of the gap / uniform | in-plane D / D_free | particles outside the gap |", "|---|---|---|---|"]
for k, v in R["slab"].items():
    md.append(f"| {k.split('=')[1]} | {v['wall_density_ratio']:.3f} | {v['D_par']:.3f} +- {v['D_par_sem']:.3f} | {v['outside']} |")
md += ["", "A reflecting wall that biased the density would show a ratio away from 1.0. None does, up to steps of w/5.", ""]

# --- lattice
md += ["## 2. Periodic lattice of cubic cells (cell side 200 nm, gap w on every face)", "",
       "Tortuosity lambda^2 = D / D_ECS (thin-sheet limit 1.5). 'ECS fraction' is the share of the volume the solver treats as gap; "
       "the true value is 1 - ((a-w)/a)^3.", "",
       "| w (nm) | true ECS fraction | BD, analytic geometry | BD, from labels | voxel FV 8 nm | voxel FV 4 nm | voxel FV 2 nm |", "|---|---|---|---|---|---|---|"]
for k, v in R["lattice"].items():
    bd = lambda t: f"{v[t]['lam2']:.3f} +- {v[t]['lam2_sem']:.3f}" if t in v else "-"
    fv = lambda t: f"{v[t]['lam2']:.3f} (alpha {v[t]['alpha']:.3f})" if t in v else "-"
    md.append(f"| {v['w']:g} | {v['alpha']:.3f} | {bd('bd_analytic')} | {bd('bd_labels')} | {fv('fv_h8')} | {fv('fv_h4')} | {fv('fv_h2')} |")
md += ["", "Voxel solvers quantise the gap: at 8 nm the 12 nm and 20 nm gaps give the identical 16 nm mask (same answer in both rows). "
       "lambda^2 is forgiving because it is a ratio, but the ECS volume fraction, which any flow or osmotic calculation needs, is not.", ""]

# --- sphere
S = R["sphere"]
md += ["## 3. Ball in socket: gap shell around a sphere of radius 200 nm (w = 20 nm)", "",
       f"Slowest decay rate of diffusion in the shell. Exact (spherical shell, reflecting walls): {S['exact_rate']:.4e} per time unit "
       f"(thin-film limit 2D/R^2 = {S['thin_rate']:.4e}).", "",
       "| method | geometry source | resolution | rate error vs exact |", "|---|---|---|---|"]
rows = [("Brownian dynamics", "analytic", "h = 8 nm", "bd_analytic_h8"), ("Brownian dynamics", "analytic", "h = 2 nm", "bd_analytic_h2"),
        ("Brownian dynamics", "voxelised labels", "h = 8 nm", "bd_labels_h8"), ("Brownian dynamics", "labels, smoothed field", "h = 8 nm", "bd_labels_h8_sigma1"),
        ("voxel finite volumes", "analytic mask", "h = 8 nm", "fv_h8"), ("voxel finite volumes", "analytic mask", "h = 4 nm", "fv_h4"), ("voxel finite volumes", "analytic mask", "h = 3 nm", "fv_h3"),
        ("surface mesh, cotangent Laplacian", "analytic", "h = 8 nm", "mesh_analytic_h8"), ("surface mesh, cotangent Laplacian", "analytic", "h = 4 nm", "mesh_analytic_h4"), ("surface mesh, cotangent Laplacian", "analytic", "h = 2 nm", "mesh_analytic_h2"),
        ("surface mesh, cotangent Laplacian", "voxelised labels, raw", "h = 8 nm", "mesh_labels_h8"), ("surface mesh, cotangent Laplacian", "voxelised labels, raw", "h = 4 nm", "mesh_labels_h4"),
        ("surface mesh, cotangent Laplacian", "labels + smoothed field", "h = 8 nm", "mesh_labels_sigma1_h8"), ("surface mesh, cotangent Laplacian", "labels + smoothed field", "h = 4 nm", "mesh_labels_sigma1_h4"),
        ("surface mesh, cotangent Laplacian", "labels + Taubin", "h = 8 nm", "mesh_labels_taubin_h8"), ("surface mesh, cotangent Laplacian", "labels + Taubin", "h = 4 nm", "mesh_labels_taubin_h4")]
for m, g, h, k in rows:
    if k in S:
        e = f"{100 * S[k]['rel_err']:+.1f}%" + (f" +- {100 * S[k]['sem'] / S['exact_rate']:.1f}%" if "sem" in S[k] else "")
        md.append(f"| {m} | {g} | {h} | {e} |")
open(os.path.join(ROOT, "results", "diffusion_benchmark.md"), "w").write("\n".join(md) + "\n")

# --- figure
OI = dict(bd="#0072B2", fv="#D55E00", mesh="#009E73", ref="#555555")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1, 1.15, 1.35]))
a = ax[0]; sg = [float(k.split("=")[1]) for k in R["slab"]]
a.plot(sg, [v["wall_density_ratio"] for v in R["slab"].values()], "o-", color=OI["bd"], lw=2, label="density in outer 20% of gap")
a.plot(sg, [v["D_par"] for v in R["slab"].values()], "s-", color=OI["mesh"], lw=2, label="in-plane D / D_free")
a.axhline(1, color=OI["ref"], lw=1); a.set_ylim(0.9, 1.1); a.set_xlabel("step length (nm)   [gap w = 20 nm]"); a.set_title("a  Walls: no bias", loc="left", fontweight="bold"); a.legend(frameon=False, fontsize=8, loc="upper left")
a = ax[1]; ws = [v["w"] for v in R["lattice"].values()]
a.errorbar(ws, [v["bd_analytic"]["lam2"] for v in R["lattice"].values()], [v["bd_analytic"]["lam2_sem"] for v in R["lattice"].values()], fmt="o-", color=OI["bd"], lw=2, capsize=3, label="BD (analytic geometry)")
for hv, mk in (("fv_h8", "^"), ("fv_h4", "v")):
    a.plot(ws, [v[hv]["lam2"] for v in R["lattice"].values()], mk + "--", color=OI["fv"], alpha=1.0 if hv == "fv_h8" else 0.6, label=f"voxel FV, {hv[4:]} nm")
if "fv_h2" in R["lattice"]["w=20"]: a.plot([20], [R["lattice"]["w=20"]["fv_h2"]["lam2"]], "D", color=OI["fv"], ms=7, label="voxel FV, 2 nm (reference)")
a.axhline(1.5, color=OI["ref"], lw=1, ls=":"); a.text(ws[-1], 1.505, "thin-sheet limit 1.5", ha="right", fontsize=8, color=OI["ref"])
a.set_xlabel("gap width w (nm)"); a.set_ylabel("tortuosity lambda^2"); a.set_title("b  Cubic lattice: tortuosity", loc="left", fontweight="bold"); a.legend(frameon=False, fontsize=8, loc="lower left"); a.set_ylim(1.25, 1.52)
a = ax[2]
sel = [("BD analytic 8", "bd_analytic_h8", "bd"), ("BD analytic 2", "bd_analytic_h2", "bd"), ("BD labels 8", "bd_labels_h8", "bd"),
       ("voxel 8", "fv_h8", "fv"), ("voxel 4", "fv_h4", "fv"), ("voxel 3", "fv_h3", "fv"),
       ("mesh analytic 4", "mesh_analytic_h4", "mesh"), ("mesh analytic 2", "mesh_analytic_h2", "mesh"),
       ("mesh labels raw 8", "mesh_labels_h8", "mesh"), ("mesh labels raw 4", "mesh_labels_h4", "mesh"),
       ("mesh labels smoothed 8", "mesh_labels_sigma1_h8", "mesh"), ("mesh labels smoothed 4", "mesh_labels_sigma1_h4", "mesh")]
sel = [s for s in sel if s[1] in S]
y = np.arange(len(sel))[::-1]
a.barh(y, [100 * S[k]["rel_err"] for _, k, _ in sel], color=[OI[c] for _, _, c in sel], height=0.65)
a.set_yticks(y); a.set_yticklabels([n for n, _, _ in sel], fontsize=8); a.axvline(0, color="k", lw=0.8)
a.set_xlabel("error of the slowest decay rate vs exact (%)"); a.set_title("c  Ball in socket: curvature", loc="left", fontweight="bold")
for yy, (_, k, _) in zip(y, sel): a.text(100 * S[k]["rel_err"] + (0.6 if S[k]["rel_err"] >= 0 else -0.6), yy, f"{100 * S[k]['rel_err']:+.1f}", va="center", ha="left" if S[k]["rel_err"] >= 0 else "right", fontsize=7.5)
a.set_xlim(-22, 5)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "results", "diffusion_benchmark.png"), dpi=110)
print("\n".join(md))
