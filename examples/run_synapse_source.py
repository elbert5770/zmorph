"""Find candidate synaptic clefts in a real tile and release a burst of tracers from the best one.

1. crop the tile, rebuild the isotropic volume with sdf_align (8 nm voxels),
2. list contact patches between labels and score them (zmorph.synapse),
3. take the top patch (or --patch N) as the source; release N tracers in the cleft,
4. run Brownian dynamics in the ECS { m < w } and report spreading.

usage: python examples/run_synapse_source.py [--tile 170398_60741] [--sections 0 41] [--crop 0 256 0 256] [--w 20]
"""
import os, sys, argparse, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
from zmorph.interp import reconstruct
from zmorph.synapse import contact_patches, score_patches, release_positions
from zmorph.bd import margin_field, simulate_dynamic

ap = argparse.ArgumentParser()
ap.add_argument("--tile", default="170398_60741"); ap.add_argument("--sections", type=int, nargs=2, default=[0, 41])
ap.add_argument("--crop", type=int, nargs=4, default=[0, 256, 0, 256]); ap.add_argument("--w", type=float, default=20.0)
ap.add_argument("--patch", type=int, default=0); ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--top", type=int, default=10); ap.add_argument("--tmax", type=float, default=2e5)
a = ap.parse_args()
H = 8.0                                           # nm; 8 x 8 in plane, 40 nm / 5 in z
out = os.path.join(ROOT, "results", "synapse_source"); os.makedirs(out, exist_ok=True)

d = np.load(os.path.join(ROOT, "data", f"tile_{a.tile}.npz"))
s0, s1 = a.sections; y0, y1, x0, x1 = a.crop
secs = [d["vol"][i, y0:y1, x0:x1].astype(np.int64) for i in range(s0, s1)]
t = time.time(); vol = reconstruct(secs, 5, "sdf", align=True); print(f"reconstruct {vol.shape} {time.time() - t:.0f}s")

patches = score_patches(contact_patches(vol, H))
print(f"{len(patches)} contact patches")
rows = [dict(rank=i, pair=list(map(str, p.pair)), score=round(p.score, 3), area_um2=round(p.area * 1e-6, 4),
             radius_nm=round(p.radius), aspect=round(p.aspect, 2), flatness=round(p.flatness, 3),
             center_nm=[round(float(c)) for c in p.center]) for i, p in enumerate(patches[:a.top])]
for r in rows:
    print(r)

p = patches[a.patch]
t = time.time(); m = margin_field(vol, H, cap=64.0); print(f"margin {time.time() - t:.0f}s")
x0_ = release_positions(p, m, H, a.w, a.n, seed=1)
dt = 0.5; nrec = 40; spr = int(a.tmax / dt / nrec)
r = simulate_dynamic(m, H, x0_ / H, lambda t: a.w, 1.0, dt, nrec=nrec, steps_per_rec=spr, seed=2)
dist = np.linalg.norm(r["pos"] - p.center, axis=2)
res = dict(tile=a.tile, w=a.w, source=rows[a.patch] if a.patch < len(rows) else None, t=r["t"].tolist(),
           mean_dist_nm=dist.mean(1).tolist(), median_dist_nm=np.median(dist, 1).tolist(),
           frac_within_100nm=(dist < 100).mean(1).tolist(), frac_within_500nm=(dist < 500).mean(1).tolist(),
           rms_nm=np.sqrt((dist ** 2).mean(1)).tolist())
json.dump(dict(patches=rows, release=res), open(os.path.join(out, f"{a.tile}.json"), "w"), indent=1)
k = [0, nrec // 4, nrec // 2, nrec]
print("t (nm^2/D):", [int(r["t"][i]) for i in k])
print("rms distance (nm):", [round(res["rms_nm"][i]) for i in k])
print("fraction within 100 nm:", [round(res["frac_within_100nm"][i], 2) for i in k])
