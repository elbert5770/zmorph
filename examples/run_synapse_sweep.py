"""Sweep the gap width w (static, then breathing) for tracers released from one candidate cleft.

Geometry (reconstruct + patches + margin) is cached in --cache so reruns only repeat the Brownian dynamics.
Static: w in --ws. Breathing (about w0): linear ramp w0 -> 1.6 w0 (sleep-like expansion), and sinusoids
w0 (1 + 0.3 sin(2 pi t / T)) with T = tmax/8 and tmax/2. Release is the same burst in every run (made at w0).
Metrics: rms displacement, apparent D = MSD / (6 t) from the second half of the run, fractions escaped beyond
100 / 500 nm of the source. D = 1, lengths nm, time nm^2/D (for Abeta, D ~ 1.5e8 nm^2/s, so tmax = 2e5 is ~1.3 ms).

usage: python examples/run_synapse_sweep.py [--patch 0] [--w0 20] [--seeds 2]
"""
import os, sys, argparse, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
from zmorph.synapse import contact_patches, score_patches, release_positions
from zmorph.bd import margin_field, simulate_dynamic

ap = argparse.ArgumentParser()
ap.add_argument("--tile", default="170398_60741"); ap.add_argument("--sections", type=int, nargs=2, default=[0, 41])
ap.add_argument("--crop", type=int, nargs=4, default=[0, 256, 0, 256])
ap.add_argument("--patch", type=int, default=0); ap.add_argument("--w0", type=float, default=20.0)
ap.add_argument("--ws", type=float, nargs="+", default=[12, 16, 20, 30, 40, 60])
ap.add_argument("--n", type=int, default=2000); ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--tmax", type=float, default=2e5); ap.add_argument("--cache", default=None)
a = ap.parse_args()
H = 8.0; dt = 0.5; nrec = 40; spr = int(a.tmax / dt / nrec)
out = os.path.join(ROOT, "results", "synapse_source"); os.makedirs(out, exist_ok=True)
cache = a.cache or os.path.join(out, f"geom_{a.tile}.npz")

if os.path.exists(cache):
    g = np.load(cache); m = g["m"]; center = g["center"]; pts = g["points"]
else:
    from zmorph.interp import reconstruct
    d = np.load(os.path.join(ROOT, "data", f"tile_{a.tile}.npz")); y0, y1, x0, x1 = a.crop
    secs = [d["vol"][i, y0:y1, x0:x1].astype(np.int64) for i in range(*a.sections)]
    vol = reconstruct(secs, 5, "sdf", align=True)
    P = score_patches(contact_patches(vol, H))
    m = margin_field(vol, H, cap=64.0)
    np.savez_compressed(cache, m=m.astype(np.float32), center=P[a.patch].center, points=P[a.patch].points)
    center, pts = P[a.patch].center, P[a.patch].points
m = m.astype(np.float64)


class _P:        # minimal stand-in for a synapse.Patch (release_positions needs only .points)
    points = pts


x0 = release_positions(_P, m, H, a.w0, a.n, seed=1) / H
T1, T2 = a.tmax / 8, a.tmax / 2
cases = {f"static w={w:g}": (lambda t, w=w: w) for w in a.ws}
cases["ramp w0->1.6 w0"] = lambda t: a.w0 * (1 + 0.6 * t / a.tmax)
cases["osc T=tmax/8"] = lambda t: a.w0 * (1 + 0.3 * np.sin(2 * np.pi * t / T1))
cases["osc T=tmax/2"] = lambda t: a.w0 * (1 + 0.3 * np.sin(2 * np.pi * t / T2))


def metrics(r):
    t = r["t"]; disp = r["pos"] - r["pos"][0]; msd = (disp ** 2).sum(2).mean(1)
    k = t >= t[-1] / 2
    slope = np.polyfit(t[k], msd[k], 1)[0]
    dist = np.linalg.norm(r["pos"] - center, axis=2)
    return dict(rms=float(np.sqrt(msd[-1])), D_app=float(slope / 6), esc100=float((dist[-1] > 100).mean()),
                esc500=float((dist[-1] > 500).mean()), alive=float(r["alive"][-1].mean()),
                w_end=float(r["w"][-1]))


rows = []
for name, wf in cases.items():
    t0 = time.time(); ms = []
    for s in range(a.seeds):
        r = simulate_dynamic(m, H, x0.copy(), wf, 1.0, dt, nrec=nrec, steps_per_rec=spr, seed=10 + s)
        ms.append(metrics(r))
    row = dict(case=name, **{k: float(np.mean([q[k] for q in ms])) for k in ms[0]},
               D_app_sd=float(np.std([q["D_app"] for q in ms])), rms_sd=float(np.std([q["rms"] for q in ms])))
    rows.append(row); print(f"{name:18s} rms {row['rms']:6.0f}±{row['rms_sd']:3.0f}  D_app {row['D_app']:.3f}±{row['D_app_sd']:.3f}"
                            f"  esc500 {row['esc500']:.2f}  alive {row['alive']:.2f}  ({time.time() - t0:.0f}s)", flush=True)

json.dump(dict(tile=a.tile, patch=a.patch, w0=a.w0, tmax=a.tmax, n=a.n, rows=rows),
          open(os.path.join(out, f"sweep_{a.tile}.json"), "w"), indent=1)
with open(os.path.join(out, f"sweep_{a.tile}.md"), "w") as f:
    f.write(f"# w sweep, tile {a.tile}, source patch {a.patch}, N={a.n}, tmax={a.tmax:g}\n\n"
            "| case | rms (nm) | D_app | esc>100 | esc>500 |\n|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['case']} | {r['rms']:.0f} | {r['D_app']:.3f} | {r['esc100']:.2f} | {r['esc500']:.2f} |\n")
