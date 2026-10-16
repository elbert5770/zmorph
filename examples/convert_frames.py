"""Convert the per-frame CSVs of a MICrONS crop into one compressed array per tile.

Frame names:  frame_{z}_{x}_{y}_4.csv   (also accepts .csv.gz)
  z: 17391..17591 (step 1), x in {170398, 170898}, y in {60741, 61241} -> 4 tiles x 201 sections.

Per tile the output is  <out>/tile_{x}_{y}.npz  with
  vol  (nz, 501, 501) uint32   index into `ids`
  ids  (n,) int64              segment IDs (~1e18), sorted; ids[vol] restores the original labels
  z    (nz,) int64             section numbers
Reload:  d = np.load(path); labels = d["ids"][d["vol"]]      # int64, same as load_labels()

Usage (run from anywhere; only numpy is required, pandas is used if present):
  python convert_frames.py /path/to/csv_folder /path/to/output_folder [--workers 8]
Edit SRC / OUT below or pass them on the command line.
"""
import argparse
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np

SRC = "/path/to/csv_folder"      # dummy: folder holding the frame_*.csv files
OUT = "/path/to/output_folder"   # dummy: where the .npz files go
Z0, Z1 = 17391, 17591
XS = (170398, 170898)
YS = (60741, 61241)
SUFFIX = "4"                     # the trailing _4 in the file name
SHAPE = (501, 501)

try:
    import pandas as pd
except ImportError:
    pd = None


def find(src, z, x, y):
    base = os.path.join(src, f"frame_{z}_{x}_{y}_{SUFFIX}")
    for ext in (".csv", ".csv.gz"):
        if os.path.exists(base + ext):
            return base + ext
    return None


def load_one(path):
    """Segment-ID image (int64). The first row is a header of column indices."""
    if pd is not None:
        a = pd.read_csv(path, dtype=np.int64).to_numpy()
    else:
        a = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)
    return a


def convert_tile(src, out, x, y, workers):
    zs = list(range(Z0, Z1 + 1))
    paths = [find(src, z, x, y) for z in zs]
    missing = [z for z, p in zip(zs, paths) if p is None]
    if missing:
        print(f"tile {x}_{y}: {len(missing)} frames missing (first {missing[:5]}); skipping tile", file=sys.stderr)
        return False
    with Pool(workers) as pool:
        frames = pool.map(load_one, paths)
    bad = [(z, f.shape) for z, f in zip(zs, frames) if f.shape != SHAPE]
    if bad:
        print(f"tile {x}_{y}: unexpected shapes {bad[:5]}; skipping tile", file=sys.stderr)
        return False
    stack = np.stack(frames)
    ids, inv = np.unique(stack, return_inverse=True)
    vol = inv.reshape(stack.shape).astype(np.uint32)
    assert np.array_equal(ids[vol], stack)          # lossless round trip
    dest = os.path.join(out, f"tile_{x}_{y}.npz")
    np.savez_compressed(dest, vol=vol, ids=ids.astype(np.int64), z=np.array(zs, dtype=np.int64))
    print(f"tile {x}_{y}: {len(zs)} sections, {len(ids)} ids -> {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", default=SRC)
    ap.add_argument("out", nargs="?", default=OUT)
    ap.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    a = ap.parse_args()
    if not os.path.isdir(a.src):
        sys.exit(f"source folder not found: {a.src}")
    os.makedirs(a.out, exist_ok=True)
    ok = [convert_tile(a.src, a.out, x, y, a.workers) for x, y in itertools.product(XS, YS)]
    print(f"{sum(ok)}/{len(ok)} tiles written")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
