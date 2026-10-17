"""Apply the method to a real tile: 15 sections -> isotropic volume + conforming meshes -> ECS metrics vs gap width.

A. sdf_align interpolation (factor 5, 8 nm voxels), checks that the input sections are reproduced, per-label
   continuity, conforming per-cell meshes (watertight, vertex sharing, quality).
B. ECS = { m < w } from the competition margin: ECS fraction, tortuosity from the voxel finite-volume cell problem
   (periodic box), and the BD effective diffusivity, for several w. In a periodic box the cell problem sees the crop
   wrapped around, so crop-edge cells are joined to the opposite face; BD uses the crop as is (reflecting at its
   edge), and only x and y are compared (the z extent is short).

usage: python examples/run_tile15.py [--tile 170398_60741] [--first 0] [--n 15] [--crop 0 256 0 256]
"""
import os, sys, argparse, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
import scipy.ndimage as ndi
from zmorph.interp import competition
from zmorph.mesh import competition_meshes, quality, interface_conformity
from zmorph.bd import margin_field, simulate_dynamic, sample_ecs, effective_diffusivity
from zmorph.diffusion_ref import tortuosity_fv

ap = argparse.ArgumentParser()
ap.add_argument("--tile", default="170398_60741"); ap.add_argument("--first", type=int, default=0)
ap.add_argument("--n", type=int, default=15); ap.add_argument("--crop", type=int, nargs=4, default=[0, 256, 0, 256])
ap.add_argument("--ws", type=float, nargs="+", default=[10, 16, 20, 30, 40, 60])
ap.add_argument("--np", type=int, default=3000, dest="npart"); ap.add_argument("--tmax", type=float, default=3e4)
a = ap.parse_args()
H = 8.0; F = 5
out = os.path.join(ROOT, "results", "tile15"); os.makedirs(out, exist_ok=True)
d = np.load(os.path.join(ROOT, "data", f"tile_{a.tile}.npz")); y0, y1, x0, x1 = a.crop
secs = [d["vol"][i, y0:y1, x0:x1].astype(np.int64) for i in range(a.first, a.first + a.n)]
R = {"tile": a.tile, "sections": [a.first, a.first + a.n], "crop": a.crop}

# ------------------------------------------------------------------ A. interpolation + meshes
t = time.time()
lab, meshes = competition_meshes(secs, F, align=True)
vol = lab; R["shape"] = list(vol.shape)
R["input_reproduced"] = bool(all((vol[F * i] == s).all() for i, s in enumerate(secs)))
R["input_reproduced_frac"] = float(np.mean([(vol[F * i] == s).mean() for i, s in enumerate(secs)]))
nin = [len(np.unique(s)) for s in secs]; nout = len(np.unique(vol))
R["labels_in_sections"] = [min(nin), max(nin)]; R["labels_in_volume"] = int(nout)
R["labels_in_union_of_sections"] = int(len(np.unique(np.concatenate([s.ravel() for s in secs]))))
frag = []                                              # extra components a label gains versus its sections
for k in np.unique(vol):
    if k == 0:
        continue
    c3 = ndi.label(vol == k, structure=np.ones((3, 3, 3)))[1]
    c2 = max(ndi.label(s == k)[1] for s in secs)
    frag.append(c3 - max(c2, 1))
R["labels_gaining_components"] = int(sum(f > 0 for f in frag)); R["labels_total"] = len(frag)


def watertight(v, f):
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1)
    _, c = np.unique(e, axis=0, return_counts=True)
    return bool((c == 2).all())


R["meshes"] = len(meshes); R["meshes_watertight"] = int(sum(watertight(*m) for m in meshes.values()))
R["mesh_quality"] = quality(meshes); R["mesh_conformity"] = interface_conformity(meshes)
R["time_interp_mesh_s"] = round(time.time() - t)
np.savez_compressed(os.path.join(out, "volume.npz"), vol=vol.astype(np.uint32), ids=np.unique(vol))
np.savez_compressed(os.path.join(out, "meshes.npz"), **{f"{k}_v": v.astype(np.float32) for k, (v, f) in meshes.items()},
                    **{f"{k}_f": f.astype(np.int32) for k, (v, f) in meshes.items()})
print("A:", json.dumps({k: v for k, v in R.items()}), flush=True)

# ------------------------------------------------------------------ B. ECS metrics vs w
t = time.time(); m = margin_field(vol, H, cap=64.0); print(f"margin {time.time() - t:.0f}s", flush=True)
rows = []
for w in a.ws:
    mask = m < w
    tor = [tortuosity_fv(mask, H, axis=ax) for ax in range(3)]
    x0_ = sample_ecs(m - w, H, a.npart, seed=1)
    dt = 0.5; nrec = 30; spr = int(a.tmax / dt / nrec)
    r = simulate_dynamic(m, H, x0_, lambda tt: w, 1.0, dt, nrec=nrec, steps_per_rec=spr, seed=2)
    D, sem = effective_diffusivity(r["t"], r["pos"], window=(0.1, 0.5), groups=10)
    row = dict(w=w, ecs_fraction=float(mask.mean()),
               inv_lam2_fv=[1.0 / x["lam2"] for x in tor], D_tissue_fv=[x["D_tissue"] for x in tor],
               D_bd=D.tolist(), D_bd_sem=sem.tolist())
    rows.append(row)
    print(f"w={w:3g} alpha={row['ecs_fraction']:.3f}  1/lam2 FV (z,y,x)={np.round(row['inv_lam2_fv'], 3)}  "
          f"D BD (z,y,x)={np.round(row['D_bd'], 3)}", flush=True)
R["ecs"] = rows
json.dump(R, open(os.path.join(out, f"summary_{a.tile}.json"), "w"), indent=1)
