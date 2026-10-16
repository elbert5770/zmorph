"""Frame edges: how much does tiling cost each method, and how large a halo of neighbouring data removes it?
The real held-out test (predict section 17415 from 17414 and 17416) is cut into 2x2 tiles of about 250 px; each tile is
interpolated on its own with a halo of real neighbour pixels, and compared with the same method run on the whole
frame. usage: python run_edge_study.py     writes results/edge_study.md"""
import sys
import numpy as np
sys.path.insert(0, "examples")
from zmorph.interp import reconstruct
from zmorph.fusion import heat_sdf_morph
from zmorph.tiling import reconstruct_tiled
from real_io import DATA, load_labels

T = "170398_61241"
D = {"smoothed": (load_labels(f"{DATA}/sframe_17414_{T}_4.csv.gz"), load_labels(f"{DATA}/sframe_17416_{T}_4.csv.gz"), load_labels(f"{DATA}/sframe_17415_{T}_4.csv.gz")),
     "raw (500x500)": (load_labels(f"{DATA}/frame_17414_{T}_4.csv.gz")[:500, :500], load_labels(f"{DATA}/frame_17416_{T}_4.csv.gz")[:500, :500], load_labels(f"{DATA}/frame_17415_{T}.csv.gz"))}
METHODS = {
    "nearest": lambda s, k: reconstruct(s, k, "nearest"),
    "heat, published": lambda s, k: reconstruct(s, k, "heat"),
    "heat2": lambda s, k: reconstruct(s, k, "heat2"),
    "sdf": lambda s, k: reconstruct(s, k, "sdf"),
    "sdf + translation": lambda s, k: reconstruct(s, k, "sdf", align=True),
    "sdf_rank (fusion)": lambda s, k: heat_sdf_morph(s, k, "sdf_rank"),
    "rank_avg (fusion)": lambda s, k: heat_sdf_morph(s, k, "rank_avg"),
}
HALOS = (0, 8, 24, 64)
BANDS = ((0, 4), (4, 12), (12, 24), (24, 1e9))

def band_acc(pred, truth, d):
    return [float((pred == truth)[(d >= lo) & (d < hi)].mean()) for lo, hi in BANDS]

if __name__ == "__main__":
    md = []
    for dn, (a, b, t) in D.items():
        ny, nx = t.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        mid = ny // 2 - 0.5                       # the internal boundary (tiles split at about half)
        d = np.minimum(np.abs(yy - mid), np.abs(xx - mid)) - 0.5
        md += [f"\n## {dn}: held-out accuracy by distance (px) to the internal tile boundary\n",
               "bands: 0-4 | 4-12 | 12-24 | >24 px. 'full' = the same method on the whole frame. Entries: accuracy; in brackets, the loss relative to full.\n",
               "| method | full (bands) | " + " | ".join(f"halo {h}" for h in HALOS) + " |", "|---|---|" + "---|" * len(HALOS)]
        for name, f in METHODS.items():
            full = f([a, b], 2)[1]
            fb = band_acc(full, t, d)
            row = ["/".join(f"{x:.3f}" for x in fb)]
            for h in HALOS:
                tiled = reconstruct_tiled([a, b], 2, f, tile=250, halo=h)[1]
                tb = band_acc(tiled, t, d)
                row.append("/".join(f"{x:.3f}" for x in tb) + f" [{', '.join(f'{(u - v) * 100:+.1f}' for u, v in zip(tb, fb))}]" + f" seam diff {(tiled != full).mean() * 100:.2f}%")
            md.append(f"| {name} | " + " | ".join(row) + " |"); print(dn, name, flush=True)
    open("results/edge_study.md", "w").write("\n".join(md) + "\n")
    open("/tmp/claude-0/-home-user-zmorph/37828f76-e0d7-565c-8881-7b12bd3db680/scratchpad/edge.done", "w").write("done")
