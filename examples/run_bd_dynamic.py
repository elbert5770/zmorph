"""Moving walls and membrane patches in Brownian dynamics, against exact results (D = 1, kT = 1, lengths in nm).

  force      wall force from the wall-imposed displacements: total force on both walls of a slab = 2 N kT / w
  piston     compress / expand a slab: isothermal work N kT ln(w0/w1) in the quasi-static limit, more when fast
  robin      slab with one partially absorbing wall: survivor decay rate vs the exact Robin eigenvalue
  mosaic     fine mosaic of patches covering a fraction f of the wall: effective Robin rate f * kappa
  cap        absorbing patch on a spherical shell: events land on the patch

usage: python examples/run_bd_dynamic.py [force|piston|robin|mosaic|cap|all]  -> results/bd_dynamic.json, results/bd_dynamic.md
"""
import os, sys, json, math, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
from scipy.optimize import brentq
from zmorph.bd import simulate_dynamic, sample_ecs, wall_force

OUT = os.path.join(ROOT, "results", "bd_dynamic.json")
RES = json.load(open(OUT)) if os.path.exists(OUT) else {}
save = lambda: json.dump(RES, open(OUT, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
log = lambda *a: print(*a, flush=True)
PER = (True, True, False)


def slab_margin(h=8.0, nz=13, nxy=8):
    z = (np.arange(nz) - nz // 2) * h
    return np.broadcast_to((2 * np.abs(z))[None, None, :], (nxy, nxy, nz)).copy(), nz // 2


def force():
    h = 8.0; m, c = slab_margin(h); N = 20000; out = {}
    for w in (12.0, 20.0, 40.0):
        for sig in (0.5, 1.0, 2.0):
            dt = sig ** 2 / 2
            x0 = sample_ecs(m - w, h, N, periodic=PER, seed=1)
            r = simulate_dynamic(m, h, x0, lambda t: w, 1.0, dt, nrec=20, steps_per_rec=int(2000 / dt / 20), periodic=PER, seed=2)
            _, F = wall_force(r["wall_push"], r["t"])
            out[f"w={w:g},sigma={sig}"] = dict(F=float(F.mean()), F_sem=float(F.std(ddof=1) / math.sqrt(len(F))), expected=2 * N / w, ratio=float(F.mean() / (2 * N / w)))
            log("force", w, sig, out[f"w={w:g},sigma={sig}"]["ratio"])
    RES["force"] = out; save()


def piston():
    h = 8.0; m, c = slab_margin(h, nz=17); N = 20000; out = {}
    for tag, w0, w1 in (("compress 40->20", 40.0, 20.0), ("expand 20->40", 20.0, 40.0)):
        for T in (1e4, 1e3, 1e2):
            sig = 1.0; dt = 0.5; nrec = 50
            wf = lambda t, w0=w0, w1=w1, T=T: w0 + (w1 - w0) * min(t / T, 1.0)
            x0 = sample_ecs(m - w0, h, N, periodic=PER, seed=1)
            r = simulate_dynamic(m, h, x0, wf, 1.0, dt, nrec=nrec, steps_per_rec=int(T / dt / nrec), periodic=PER, seed=2)
            tm, F = wall_force(r["wall_push"], r["t"])
            w = r["w"]; dw = np.abs(np.diff(w))
            W = float((F * dw / 2).sum()) * (1 if w1 < w0 else -1)          # work done ON the gas (positive for compression)
            ideal = N * math.log(w0 / w1)
            zz = r["pos"][-1, :, 2] - c * h
            cnt, _ = np.histogram(zz / (w1 / 2), bins=8, range=(-1, 1))
            out[f"{tag} T={T:g}"] = dict(work=W, ideal=ideal, ratio=W / ideal, Pe=abs(w1 - w0) / T * 0.5 * (w0 + w1) / 1.0,
                                         uniformity_dev=float(np.abs(cnt / cnt.mean() - 1).max()), outside=int((np.abs(zz) > w1 / 2 + 1e-6).sum()))
            log("piston", tag, T, out[f"{tag} T={T:g}"]["ratio"], out[f"{tag} T={T:g}"]["uniformity_dev"])
    RES["piston"] = out; save()


def robin_rate(kappa, L):
    k = brentq(lambda k: k * math.tan(k * L) - kappa, 1e-9, math.pi / (2 * L) - 1e-9)
    return k * k


def survivors_rate(r, lam):
    t = r["t"]; n = r["alive"].sum(1).astype(float)
    s = (t >= 0.4 / lam) & (t <= 2.0 / lam) & (n > 50)
    return -np.polyfit(t[s], np.log(n[s]), 1)[0]


def robin():
    h = 8.0; m, c = slab_margin(h); w = 20.0; N = 40000; out = {}
    patch = np.zeros(m.shape, np.int32); patch[:, :, c + 1:] = 1                     # top wall only
    for Bi in (0.5, 2.0):
        kappa = Bi / w; lam = robin_rate(kappa, w)
        for sig in (1.0, 0.5, 0.25):
            dt = sig ** 2 / 2; pabs = np.array([0.0, min(1.0, kappa * math.sqrt(math.pi * dt))])
            T = 2.2 / lam
            x0 = sample_ecs(m - w, h, N, periodic=PER, seed=1)
            r = simulate_dynamic(m, h, x0, lambda t: w, 1.0, dt, nrec=40, steps_per_rec=int(T / dt / 40), periodic=PER, patch=patch, p_abs=pabs, seed=2)
            k = survivors_rate(r, lam)
            out[f"Bi={Bi:g},sigma={sig}"] = dict(rate=float(k), exact=float(lam), rel_err=float(k / lam - 1), p_abs=float(pabs[1]))
            log("robin", Bi, sig, k / lam - 1)
    RES["robin"] = out; save()


def mosaic():
    h = 2.0; nz = 33; z = (np.arange(nz) - nz // 2) * h
    m = np.broadcast_to((2 * np.abs(z))[None, None, :], (32, 32, nz)).copy(); c = nz // 2
    w = 20.0; N = 40000; out = {}
    xi = (np.arange(32) // 4) % 2                                                    # 8 nm checkerboard of 4-voxel tiles... f = 1/2
    chk = ((xi[:, None] + xi[None, :]) % 2 == 0).astype(np.int32)
    patch = np.zeros(m.shape, np.int32); patch[:, :, c + 1:] = chk[:, :, None]
    for Bi in (0.5, 1.0):
        kappa = Bi / w; lam_full = robin_rate(kappa, w); lam_eff = robin_rate(0.5 * kappa, w)
        sig = 1.0; dt = sig ** 2 / 2; pabs = np.array([0.0, min(1.0, kappa * math.sqrt(math.pi * dt))])
        T = 2.2 / lam_eff
        x0 = sample_ecs(m - w, h, N, periodic=PER, seed=1)
        r = simulate_dynamic(m, h, x0, lambda t: w, 1.0, dt, nrec=40, steps_per_rec=int(T / dt / 40), periodic=PER, patch=patch, p_abs=pabs, seed=2)
        k = survivors_rate(r, lam_eff)
        out[f"Bi={Bi:g}"] = dict(rate=float(k), eff_robin=float(lam_eff), full_cover=float(lam_full), rel_err_vs_eff=float(k / lam_eff - 1))
        log("mosaic", Bi, k / lam_eff - 1, "(full cover would be", lam_full / lam_eff - 1, ")")
    RES["mosaic"] = out; save()


def cap():
    h = 8.0; R, W = 200.0, 20.0; n = int(math.ceil(2 * (R + W / 2 + 24) / h)) | 1; c = n // 2; i = np.arange(n) - c
    X, Y, Z = np.meshgrid(i, i, i, indexing="ij"); r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2) * h
    m = 2 * np.abs(r - R); out = {}
    for deg in (20, 40, 60):
        cosc = math.cos(math.radians(deg))
        patch = (Z * h / np.maximum(r, 1e-9) > cosc).astype(np.int32)
        N = 20000; x0 = sample_ecs(m - W, h, N, seed=1)
        # start particles on the far side of the sphere so the first absorptions are real transport
        rr = simulate_dynamic(m, h, x0, lambda t: W, 1.0, 0.5, nrec=30, steps_per_rec=2000, patch=patch, p_abs=np.array([0.0, 0.3]), seed=2)
        a = rr["absorbed"]; p = rr["abs_pos"][a] - c * h
        ang = np.degrees(np.arccos(np.clip(p[:, 2] / np.linalg.norm(p, axis=1), -1, 1)))
        out[f"cap {deg} deg"] = dict(absorbed=int(a.sum()), max_angle=float(ang.max()) if a.any() else None, area_fraction=(1 - cosc) / 2, absorbed_fraction=float(a.mean()))
        log("cap", deg, out[f"cap {deg} deg"])
    RES["cap"] = out; save()


def report():
    md = ["# Moving walls and membrane patches in Brownian dynamics", "", "D = 1, kT = 1, lengths in nm; errors are single runs unless stated.", ""]
    if "force" in RES:
        md += ["## Wall force from reflections (planar gap)", "", "Measured total force on both walls / (2 N kT / w), 20000 particles.", "", "| w (nm) | step 0.5 nm | step 1 nm | step 2 nm |", "|---|---|---|---|"]
        for w in (12, 20, 40):
            md.append(f"| {w} | " + " | ".join(f"{RES['force'][f'w={w:g},sigma={s}']['ratio']:.4f}" for s in (0.5, 1.0, 2.0)) + " |")
        md.append("")
    if "piston" in RES:
        md += ["## Piston: work against the ideal-gas isotherm", "", "Work done on the gas / (N kT ln(w0/w1)); 1.0 = reversible. Pe = wall speed x gap / D.", "",
               "| run | Pe | work / ideal | max density deviation from uniform | particles outside |", "|---|---|---|---|---|"]
        for k, v in RES["piston"].items(): md.append(f"| {k} | {v['Pe']:.2f} | {v['ratio']:.3f} | {v['uniformity_dev']:.3f} | {v['outside']} |")
        md.append("")
    if "robin" in RES:
        md += ["## Partially absorbing wall (Robin), decay of survivors", "", "Absorption probability per hit p = kappa sqrt(pi dt / D). Exact rate from k tan(kL) = kappa/D, rate = D k^2.", "", "| Bi = kappa w / D | step (nm) | p per hit | measured rate | exact rate | error |", "|---|---|---|---|---|---|"]
        for k, v in RES["robin"].items():
            Bi, s = k.split(","); md.append(f"| {Bi.split('=')[1]} | {s.split('=')[1]} | {v['p_abs']:.3f} | {v['rate']:.3e} | {v['exact']:.3e} | {100 * v['rel_err']:+.1f}% |")
        md.append("")
    if "mosaic" in RES:
        md += ["## Checkerboard of absorbing patches (half of one wall)", "", "Fine mosaic (patch tiles 8 nm vs gap 20 nm) should act like a uniform wall with kappa/2.", "", "| Bi | measured rate | effective-Robin prediction | error | full coverage would be |", "|---|---|---|---|---|"]
        for k, v in RES["mosaic"].items():
            md.append(f"| {k.split('=')[1]} | {v['rate']:.3e} | {v['eff_robin']:.3e} | {100 * v['rel_err_vs_eff']:+.1f}% | {100 * (v['full_cover'] / v['eff_robin'] - 1):+.1f}% |")
        md.append("")
    if "cap" in RES:
        md += ["## Absorbing cap on the spherical shell", "", "| cap half-angle | cap area fraction | absorbed fraction | largest polar angle of an absorption |", "|---|---|---|---|"]
        for k, v in RES["cap"].items(): md.append(f"| {k.split()[1]} deg | {v['area_fraction']:.3f} | {v['absorbed_fraction']:.3f} | {v['max_angle']:.1f} deg |")
    open(os.path.join(ROOT, "results", "bd_dynamic.md"), "w").write("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("force", force), ("piston", piston), ("robin", robin), ("mosaic", mosaic), ("cap", cap)):
        if which in (name, "all"): fn()
    report()
