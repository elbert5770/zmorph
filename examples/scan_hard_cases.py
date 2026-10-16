"""How often does the 'neurite vanishes in the gap' situation occur, and would tube links fire?

usage: python examples/scan_hard_cases.py LOWER.csv[.gz] UPPER.csv[.gz] [GAP_VOXELS]

For two sections (GAP_VOXELS isotropic voxels apart, default 5 = 40 nm at 8 nm): counts labels present in both
sections with no overlap at all (those a signed-distance blend can delete), same-label component pairs
without overlap, and the tube links the closure and ellipse tests accept. Run it over many section pairs to
estimate prevalence before relying on `tube=True`."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from real_io import load_labels
from zmorph.interp import _components, _tube_link

a, b = load_labels(sys.argv[1]), load_labels(sys.argv[2])
L = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
P = dict(tube_min_area=12, tube_min_slope=1.0, tube_tol=3.0, tube_min_ellipse=0.6, tube_min_continue=0.3)
both = np.intersect1d(np.unique(a), np.unique(b))
no_overlap_labels = [k for k in both if not ((a == k) & (b == k)).any()]
pairs, links = 0, []
for k in both:
    for x in _components(a == k):
        for y in _components(b == k):
            if not (x & y).any() and min(x.sum(), y.sum()) >= 12:
                pairs += 1
                lk = _tube_link(x, y, L, P)
                if lk is not None:
                    links.append((k, lk["mode"], round(lk["zJ"], 1)))
print(f"labels in both sections: {len(both)}; with no overlap at all: {len(no_overlap_labels)}; "
      f"non-overlapping component pairs (>=12 px): {pairs}; accepted tube links: {len(links)}")
for k, m, z in links:
    print(f"  link: label {k}, mode {m}, junction at {z} of {L} voxels")
