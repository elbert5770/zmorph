"""Small patch sinks: capture of tracers released at a cleft by a disc-shaped absorbing patch on a neighbouring membrane.

Same crop as run_synapse_sink.py (geometry from geom_<tile>.npz and vol_<tile>.npz). A site is a point of the gap
(m < w0) at about --dist nm from the source centre; the sink patch is the part of the cell under that point inside a
ball of radius R (100-300 nm), so only a disc of the membrane absorbs (p = 1 per hit). Sweeps R and a static gap w, plus a
breathing gap w(t) = w0 (1 + 0.3 sin(2 pi (t + phase) / period)) released at several phases.
Release schedule: capture is linear in the release rate for a static gap, so a release of duration tau is the burst
response F(t) convolved with the release profile (reported for tau = 0, i.e. the burst, and a uniform release over
--tau). For the breathing gap the burst is repeated at several phases and the phase average is the steady pulse train.

usage: python examples/run_synapse_smallsink.py [--dist 500] [--sites 2] [--rs 100 200 300] [--ws 12 20 30 40]
"""
import os, sys, argparse, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
from zmorph.synapse import release_positions
from zmorph.bd import simulate_dynamic
from zmorph.sinks import first_passage

ap = argparse.ArgumentParser()
ap.add_argument("--tile", default="170398_60741")
ap.add_argument("--dist", type=float, default=500.0); ap.add_argument("--sites", type=int, default=2)
ap.add_argument("--rs", type=float, nargs="+", default=[100, 200, 300])
ap.add_argument("--ws", type=float, nargs="+", default=[12, 20, 30, 40])
ap.add_argument("--w0", type=float, default=20.0); ap.add_argument("--amp", type=float, default=0.3)
ap.add_argument("--period", type=float, default=1e5); ap.add_argument("--phases", type=int, default=4)
ap.add_argument("--n", type=int, default=1500); ap.add_argument("--tmax", type=float, default=2e6)
ap.add_argument("--tau", type=float, default=2e5, help="duration of the uniform release profile")
ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--tag", default="")
a = ap.parse_args()
H = 8.0; dt = 0.5; nrec = 40; spr = int(a.tmax / dt / nrec)
out = os.path.join(ROOT, "results", "synapse_source")
g = np.load(os.path.join(out, f"geom_{a.tile}.npz")); m = g["m"].astype(np.float64); center = g["center"]; pts = g["points"]
vol = np.load(os.path.join(out, f"vol_{a.tile}.npz"))["vol"].astype(np.int64)
assert vol.shape == m.shape


class _P:
    points = pts


# candidate sites: gap voxels at the requested distance, away from the crop boundary
ii = np.argwhere(m < a.w0) * H
ok = (np.abs(np.linalg.norm(ii - center, axis=1) - a.dist) < 40) & np.all(ii > 400, 1) & np.all(ii < (np.array(m.shape) - 1) * H - 400, 1)
cand = ii[ok]
rng = np.random.default_rng(0)
sites = [cand[rng.integers(len(cand))]]
while len(sites) < a.sites:        # next site: farthest from those chosen
    d = np.min([np.linalg.norm(cand - s, axis=1) for s in sites], axis=0)
    sites.append(cand[int(d.argmax())])
grid = np.indices(m.shape).reshape(3, -1).T * H


def patch_map(site, R):
    """1 on the cell under `site` within R nm of it (the membrane there absorbs)."""
    lab = vol[tuple(np.round(site / H).astype(int))]
    near = np.linalg.norm(grid - site, axis=1) <= R
    return ((vol.ravel() == lab) & near).reshape(m.shape).astype(np.int32), int(lab)


def run(patch, wf, wrel, seed):
    x0 = release_positions(_P, m, H, wrel, a.n, seed=seed) / H
    r = simulate_dynamic(m, H, x0, wf, 1.0, dt, nrec=nrec, steps_per_rec=spr, patch=patch, p_abs=[0.0, 1.0], seed=seed + 100)
    return first_passage(r)


def uniform_release(t, F, tau):
    """capture(t) for a release spread uniformly over [0, tau], from the burst response F(t) (linear, static gap)."""
    s = np.linspace(0, tau, 201)[1:]
    return np.array([np.mean(np.interp(tt - s, t, F, left=0.0)) for tt in t])


rows = []
t_start = time.time()
for si, site in enumerate(sites):
    for R in a.rs:
        patch, lab = patch_map(site, R)
        for w in a.ws:
            F = []
            for sd in range(a.seeds):
                fp = run(patch, lambda t, w=w: w, w, 10 * si + sd + 1)
                F.append(fp["absorbed_frac"]); t = fp["t"]
            F = np.mean(F, 0)
            rows.append(dict(site=si, site_nm=site.tolist(), label=lab, R=R, mode="static", w=w, t=t.tolist(), F=F.tolist(),
                             F_final=float(F[-1]), F_uniform=uniform_release(t, F, a.tau).tolist(), patch_voxels=int(patch.sum())))
            print(f"site {si} R {R:4.0f} static w={w:3.0f}: captured {F[-1]:.3f}  ({time.time() - t_start:.0f}s)", flush=True)
        # breathing, phase-averaged
        Fs = []
        for ph in range(a.phases):
            off = ph * a.period / a.phases
            wf = lambda t, off=off: a.w0 * (1 + a.amp * np.sin(2 * np.pi * (t + off) / a.period))
            fp = run(patch, wf, a.w0 * (1 - a.amp), ph + 50)
            Fs.append(fp["absorbed_frac"])
        F = np.mean(Fs, 0)
        rows.append(dict(site=si, site_nm=site.tolist(), label=lab, R=R, mode="breathing", w=a.w0, t=t.tolist(), F=F.tolist(),
                         F_final=float(F[-1]), F_phase_spread=float(np.ptp([f[-1] for f in Fs])), patch_voxels=int(patch.sum())))
        print(f"site {si} R {R:4.0f} breathing w0={a.w0:g}: captured {F[-1]:.3f} (phase range {np.ptp([f[-1] for f in Fs]):.3f})", flush=True)
        json.dump(dict(tile=a.tile, args=vars(a), source_center=center.tolist(), rows=rows),
                  open(os.path.join(out, f"smallsink{a.tag}_{a.tile}.json"), "w"), indent=1)

with open(os.path.join(out, f"smallsink{a.tag}_{a.tile}.md"), "w") as f:
    f.write(f"# small patch sinks, tile {a.tile}, N={a.n} x {a.seeds} seeds, tmax={a.tmax:g} (nm^2/D), sink at ~{a.dist:g} nm from the source\n\n"
            "Captured fraction of the burst by tmax (uniform release over tau = %g in brackets).\n\n" % a.tau +
            "| site | R (nm) | mode | w (nm) | captured | uniform release |\n|---|---|---|---|---|---|\n")
    for r in rows:
        u = f"{r['F_uniform'][-1]:.3f}" if "F_uniform" in r else "-"
        f.write(f"| {r['site']} | {r['R']:g} | {r['mode']} | {r['w']:g} | {r['F_final']:.3f} | {u} |\n")
