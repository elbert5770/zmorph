"""Held-out test + hard-case scan over the 4 tiles x 201 sections (data/tile_*.npz).

For every STRIDE-th section i: predict it from i-1 and i+1 (80 nm gap, factor=2, L=10 voxels) and score against the
real section; scan the pair (i-1, i+1) and the pair (i-1, i) for labels with no overlap and accepted tube links.
Results are appended to results/real_heldout/rows.jsonl (resumable); summarise with `--summary`.

usage: python examples/run_real_heldout.py [--stride 4] [--workers 4] [--summary]
"""
import os, sys, json, glob, argparse, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
from multiprocessing import Pool

OUT = os.path.join(ROOT, "results", "real_heldout"); ROWS = os.path.join(OUT, "rows.jsonl")


def methods():
    from zmorph.interp import reconstruct
    from zmorph.fusion import heat_sdf_morph
    R = lambda *a, **k: (lambda s: reconstruct(list(s), 2, *a, **k)[1])
    return {
        "nearest": lambda s: s[0],
        "sdf": R("sdf"),
        "sdf_align": R("sdf", align=True),
        "sdf_align_tube_persist": R("sdf", align=True, tube=True, persist=True),
        "heat_port": R("heat"),
        "heat2": R("heat2"),
        "rank_avg": lambda s: heat_sdf_morph(list(s), 2, "rank_avg")[1],
        "tube_guard": lambda s: heat_sdf_morph(list(s), 2, "tube_guard")[1],
    }


def scan(a, b, L):
    from zmorph.interp import _components, _tube_link
    P = dict(tube_min_area=12, tube_min_slope=1.0, tube_tol=3.0, tube_min_ellipse=0.6, tube_min_continue=0.3)
    both = np.intersect1d(np.unique(a), np.unique(b))
    nov = [k for k in both if not ((a == k) & (b == k)).any()]
    pairs = links = 0
    for k in both:
        for x in _components(a == k):
            for y in _components(b == k):
                if not (x & y).any() and min(x.sum(), y.sum()) >= 12:
                    pairs += 1
                    links += _tube_link(x, y, L, P) is not None
    return dict(both=len(both), no_overlap=len(nov), pairs=pairs, links=links)


def work(job):
    from real_io import score
    tile, i = job
    d = np.load(os.path.join(ROOT, "data", f"tile_{tile}.npz"))
    lab = lambda k: d["ids"][d["vol"][k]]
    s = [lab(i - 1), lab(i + 1)]; truth = lab(i)
    changed = s[0] != s[1]
    row = dict(tile=tile, z=int(d["z"][i]), differ=float(changed.mean()), methods={}, t={})
    for name, f in methods().items():
        t = time.time()
        try:
            row["methods"][name] = score(f(s), truth, changed)
        except Exception as e:                                   # keep the run going, record the failure
            row["methods"][name] = dict(error=repr(e))
        row["t"][name] = round(time.time() - t, 1)
    row["scan_gap2"] = scan(s[0], s[1], 10.0)
    row["scan_gap1"] = scan(s[0], truth, 5.0)
    return row


def summary():
    rows = [json.loads(l) for l in open(ROWS)]
    names = list(rows[0]["methods"])
    md = [f"{len(rows)} held-out sections, tiles: {sorted(set(r['tile'] for r in rows))}", ""]
    md += ["| method | acc | acc where inputs differ | boundary dist (px) | Dice | comp. error / section | wins vs sdf_align (acc) |", "|---|---|---|---|---|---|---|"]
    base = np.array([r["methods"]["sdf_align"]["acc"] for r in rows])
    for n in names:
        ok = [r["methods"][n] for r in rows if "error" not in r["methods"][n]]
        a = np.array([r["methods"][n].get("acc", np.nan) for r in rows])
        g = lambda k: np.mean([m[k] for m in ok])
        md.append(f"| {n} | {g('acc'):.4f} | {g('acc_changed'):.4f} | {g('bdist'):.3f} | {g('dice'):.4f} | {g('comp_err'):.1f} | {np.nanmean(a > base):.2f} |")
    md += ["", "Per-tile accuracy (mean):", "", "| method | " + " | ".join(sorted(set(r["tile"] for r in rows))) + " |", "|---|" + "---|" * len(set(r["tile"] for r in rows))]
    for n in names:
        md.append(f"| {n} | " + " | ".join(f"{np.mean([r['methods'][n]['acc'] for r in rows if r['tile'] == t and 'acc' in r['methods'][n]]):.4f}" for t in sorted(set(r['tile'] for r in rows))) + " |")
    for g, lab in (("scan_gap2", "pair (i-1, i+1), 80 nm"), ("scan_gap1", "pair (i-1, i), 40 nm")):
        S = {k: sum(r[g][k] for r in rows) for k in rows[0][g]}
        md += ["", f"Scan, {lab}, totals over {len(rows)} sections: labels in both {S['both']}, labels with no overlap {S['no_overlap']} "
               f"({100 * S['no_overlap'] / max(S['both'], 1):.2f}%), non-overlapping component pairs {S['pairs']}, accepted tube links {S['links']}"]
    md += ["", "Run time per method (s, mean): " + ", ".join(f"{n} {np.mean([r['t'][n] for r in rows]):.1f}" for n in names)]
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--summary", action="store_true")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    if a.summary: summary(); sys.exit()
    done = set()
    if os.path.exists(ROWS):
        done = {(r["tile"], r["z"]) for r in map(json.loads, open(ROWS))}
    jobs = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "tile_*.npz"))):
        tile = os.path.basename(f)[5:-4]; z = np.load(f)["z"]
        jobs += [(tile, i) for i in range(1, len(z) - 1, a.stride) if (tile, int(z[i])) not in done]
    print(f"{len(jobs)} jobs", flush=True)
    with Pool(a.workers) as p, open(ROWS, "a") as out:
        for n, row in enumerate(p.imap_unordered(work, jobs)):
            out.write(json.dumps(row) + "\n"); out.flush()
            print(n + 1, "/", len(jobs), row["tile"], row["z"], flush=True)
