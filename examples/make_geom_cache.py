"""Build the margin-field cache results/synapse_source/geom_<tile>.npz (and vol_<tile>.npz) used by the synapse,
sink and GPU benchmark scripts, without running any simulation. The cache is git-ignored (4 MB) so a fresh clone
needs this once (about 2-3 minutes).

usage: python examples/make_geom_cache.py [--tile 170398_60741] [--sections 0 41] [--crop 0 256 0 256]
"""
import os, sys, argparse, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
from zmorph.interp import reconstruct
from zmorph.synapse import contact_patches, score_patches
from zmorph.bd import margin_field

ap = argparse.ArgumentParser()
ap.add_argument("--tile", default="170398_60741"); ap.add_argument("--sections", type=int, nargs=2, default=[0, 41])
ap.add_argument("--crop", type=int, nargs=4, default=[0, 256, 0, 256]); ap.add_argument("--patch", type=int, default=0)
a = ap.parse_args()
H = 8.0
out = os.path.join(ROOT, "results", "synapse_source"); os.makedirs(out, exist_ok=True)
d = np.load(os.path.join(ROOT, "data", f"tile_{a.tile}.npz")); y0, y1, x0, x1 = a.crop
t = time.time()
vol = reconstruct([d["vol"][i, y0:y1, x0:x1].astype(np.int64) for i in range(*a.sections)], 5, "sdf", align=True)
P = score_patches(contact_patches(vol, H)); m = margin_field(vol, H, cap=64.0)
np.savez_compressed(os.path.join(out, f"geom_{a.tile}.npz"), m=m.astype(np.float32), center=P[a.patch].center, points=P[a.patch].points)
np.savez_compressed(os.path.join(out, f"vol_{a.tile}.npz"), vol=vol.astype(np.uint32))
print(f"wrote geom and vol for {a.tile}, shape {vol.shape}, {len(P)} patches, {time.time() - t:.0f}s")
