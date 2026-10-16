"""Compare against published SDF interpolation methods, and ablate what differs.

  Raya & Udupa 1990 / Herman et al. 1992   binary SDF, linear in z, zero level          -> sdf0
  higher-order SDF (Grevera & Udupa 1996; the manuscript's "cubic spline")              -> sdf0_cubic
  4-point scheme on SDFs (weights of Kels & Dyn 2011; their set average is NOT an SDF)  -> sdf0_4pt
Multi-label extension for all: one SDF per label, argmax across labels (classic fields use absent depth 12).

Ablation ladder (linear): textbook -> absent depth 4 -> half-pixel offset -> in-plane smoothing (proposed)
Interpolation order with the proposed fields: sdf_cubic, sdf_4pt.

usage: python run_published_comparison.py [--seeds N] [--no-real]   writes results/published_sdf.{md,json}
"""
import argparse, json, os, sys
import numpy as np
from zmorph import phantom
from zmorph.interp import reconstruct
from zmorph.metrics import evaluate

ROWS = {  # name: (method, kwargs)
    "Raya-Udupa / Herman: linear binary SDF":              ("sdf0", {}),
    "cubic-spline SDF, binary fields (Grevera-Udupa)":     ("sdf0_cubic", {}),
    "4-point SDF, binary fields (Kels-Dyn weights)":       ("sdf0_4pt", {}),
    "linear, absent depth 4":                              ("sdf", dict(sigma=0.0, subvoxel=False)),
    "linear, + half-pixel offset":                         ("sdf", dict(sigma=0.0)),
    "linear, + smoothing = proposed (sdf)":                ("sdf", {}),
    "cubic spline, proposed fields":                       ("sdf_cubic", {}),
    "4-point, proposed fields":                            ("sdf_4pt", {}),
    "4-point, proposed fields, sigma=2":                   ("sdf_4pt", dict(sigma=2.0)),
    "proposed + translation":                              ("sdf", dict(align=True)),
}
COLS = ["acc_mid", "dice_mid", "bdist_mid", "component_err", "acc_sec", "area_ratio"]


def phantom_table(seeds):
    out, md = {}, []
    for scene, fn in (("branch", phantom.scene_branch), ("tilted", phantom.scene_tilted)):
        gt = fn(); clean = [gt[z] for z in range(0, gt.shape[0], 5)]
        for noisy in (False, True):
            md += [f"\n### {scene} / {'noisy' if noisy else 'clean'} sections\n", "| method | " + " | ".join(COLS) + " |", "|---|" + "---|" * len(COLS)]
            for name, (m, kw) in ROWS.items():
                rs = []
                for sd in range(seeds if noisy else 1):
                    secs = phantom.corrupt(clean, np.random.default_rng(sd), bad=3) if noisy else clean
                    rs.append(evaluate(reconstruct(secs, 5, m, **kw), gt))
                r = {c: float(np.mean([x[c] for x in rs])) for c in COLS}
                out[f"{scene}|{'noisy' if noisy else 'clean'}|{name}"] = r
                md.append(f"| {name} | " + " | ".join(f"{r[c]:.3f}" if c != "component_err" else f"{r[c]:.1f}" for c in COLS) + " |")
                print(scene, "noisy" if noisy else "clean", f"{name:50s} dice {r['dice_mid']:.3f} bd {r['bdist_mid']:.3f}", flush=True)
    return out, md


def real_table():
    sys.path.insert(0, "examples")
    from real_io import DATA, load_labels, score
    T = "170398_61241"; f = lambda p: load_labels(os.path.join(DATA, p))
    sm = {z: f(f"sframe_{z}_{T}_4.csv.gz") for z in (17414, 17415, 17416)}
    raw14, raw16, raw15 = f(f"frame_17414_{T}_4.csv.gz")[:500, :500], f(f"frame_17416_{T}_4.csv.gz")[:500, :500], f(f"frame_17415_{T}.csv.gz")
    cases = {"smoothed 17414/17416 -> smoothed 17415": (sm[17414], sm[17416], sm[17415]),
             "raw 17414/17416 -> raw 17415 (500x500)": (raw14, raw16, raw15)}
    rows = {k: v for k, v in ROWS.items() if v[0] in ("sdf0", "sdf") or "cubic" in v[0] and False}
    out, md = {}, []
    md.append("\nOnly two input sections exist for the held-out middle section, so the higher-order variants reduce to linear "
              "interpolation and are not shown; the ladder isolates the field construction.\n")
    for cname, (a, b, t) in cases.items():
        chg = a != b
        md += [f"\n### {cname}\n", "| method | pixel acc. | acc. where inputs differ | boundary dist (px) | mean Dice |", "|---|---|---|---|---|"]
        for name, (m, kw) in rows.items():
            s = score(reconstruct([a, b], 2, m, **kw)[1], t, chg)
            out[f"{cname}|{name}"] = s
            md.append(f"| {name} | {s['acc']:.4f} | {s['acc_changed']:.4f} | {s['bdist']:.3f} | {s['dice']:.4f} |")
            print("real", cname[:8], f"{name:45s} acc {s['acc']:.4f} bd {s['bdist']:.3f} dice {s['dice']:.4f}", flush=True)
    return out, md


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--no-real", action="store_true")
    a = ap.parse_args()
    out, md = phantom_table(a.seeds)
    md = ["## Synthetic phantom (mean of seeds for noisy sections)"] + md
    if not a.no_real and os.path.isdir(os.path.join("data", "micron_17414_17416")):
        o2, m2 = real_table(); out.update(o2); md += ["\n## Real sections (MICrONS 17414-17416, 80 nm held-out)"] + m2
    open("results/published_sdf.md", "w").write("\n".join(md) + "\n")
    json.dump(out, open("results/published_sdf.json", "w"), indent=1)
