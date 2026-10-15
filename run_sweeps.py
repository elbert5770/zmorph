"""Sensitivity experiments: (1) lateral shift between sections, (2) sigma/absent, (3) noise level."""
import numpy as np
from zmorph import phantom
from zmorph.interp import reconstruct
from zmorph.metrics import evaluate

def tube_scene(shift_per_gap, r=5.0, nz=31):
    s = shift_per_gap / 5.0
    L = s * 30
    cells = [[((0, 30, 20), (30, 30, 20 + L), r, r)],
             [((0, 62, 20 + 10), (30, 62, 20 + 10 + L), r, r)],
             [((0, 85, 90), (30, 85, 90), 7.0, 7.0)]]
    return phantom.build(cells, (nz, 96, 128 + int(L)), bg=1.5)

def dice_tubes(v, gt):
    z = np.arange(gt.shape[0]); mid = z[z % 5 != 0]
    ds = []
    for k in (1, 2):
        a, b = v[mid] == k, gt[mid] == k
        ds.append(2 * (a & b).sum() / (a.sum() + b.sum()))
    return np.mean(ds)

METH = {"nearest": ("nearest", {}), "sdf0": ("sdf0", {}), "harmonic": ("harmonic", {}),
        "heat-paths": ("heat", {}), "sdf": ("sdf", {}),
        "sdf+align": ("sdf", dict(align=True))}

def shift_sweep():
    shifts = [0, 3, 6, 9, 12, 15, 18, 24]
    print("\n## Tube Dice vs lateral shift per 5-voxel gap (tube diameter 10 + ~3 dilation)\n")
    print("| shift (px) | " + " | ".join(METH) + " |"); print("|---|" + "---|" * len(METH))
    out = {}
    for sh in shifts:
        gt = tube_scene(sh); secs = [gt[z] for z in range(0, 31, 5)]
        row = []
        for name, (m, kw) in METH.items():
            row.append(dice_tubes(reconstruct(secs, 5, m, **kw), gt))
        out[sh] = row
        print(f"| {sh} | " + " | ".join(f"{x:.3f}" for x in row) + " |")
    return out

def param_sweep(seeds=4):
    gt = phantom.scene_branch(); clean = [gt[z] for z in range(0, 31, 5)]
    print("\n## sdf parameter sensitivity, branch scene, noisy sections (mean of seeds): dice_mid / bdist_mid\n")
    sig = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]; ab = [1.0, 2.0, 4.0, 8.0]
    print("| sigma \\ absent | " + " | ".join(map(str, ab)) + " |"); print("|---|" + "---|" * len(ab))
    for sg in sig:
        row = []
        for a in ab:
            r = [evaluate(reconstruct(phantom.corrupt(clean, np.random.default_rng(s), bad=3), 5, "sdf", clean=True, sigma=sg, absent=a), gt) for s in range(seeds)]
            row.append(f"{np.mean([x['dice_mid'] for x in r]):.3f} / {np.mean([x['bdist_mid'] for x in r]):.2f}")
        print(f"| {sg} | " + " | ".join(row) + " |")

def noise_sweep(seeds=4):
    gt = phantom.scene_branch(); clean = [gt[z] for z in range(0, 31, 5)]
    print("\n## Noise level (p_flip of boundary pixels), branch scene: bdist_mid\n")
    names = {"nearest": ("nearest", dict(clean=True)), "heat+maj": ("heat", dict(clean=True)),
             "harmonic": ("harmonic", dict(clean=True)), "sdf": ("sdf", dict(clean=True))}
    from zmorph.interp import majority
    ps = [0.0, 0.2, 0.35, 0.5, 0.7]
    print("| p_flip | " + " | ".join(names) + " |"); print("|---|" + "---|" * len(names))
    for p in ps:
        row = []
        for n, (m, kw) in names.items():
            vals = []
            for s in range(seeds):
                secs = phantom.corrupt(clean, np.random.default_rng(s), p_flip=p, bad=None)
                v = reconstruct(secs, 5, m, **kw)
                if n == "heat+maj": v = majority(v, 2)
                vals.append(evaluate(v, gt)["bdist_mid"])
            row.append(f"{np.mean(vals):.2f}")
        print(f"| {p} | " + " | ".join(row) + " |")

if __name__ == "__main__":
    shift_sweep(); param_sweep(); noise_sweep()
