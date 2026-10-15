import os, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "examples"))
from real_io import DATA, load_labels  # noqa: E402
from zmorph.interp import reconstruct  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.isdir(DATA), reason="example data not present")
T = "170398_61241"
f = lambda p: load_labels(os.path.join(DATA, p))


def test_formats_and_raw_vs_smoothed():
    s = {z: f(f"sframe_{z}_{T}_4.csv.gz") for z in (17414, 17415, 17416)}
    r14, r16 = f(f"frame_17414_{T}_4.csv.gz"), f(f"frame_17416_{T}_4.csv.gz")
    old15 = f(f"frame_17415_{T}.csv.gz")
    assert all(a.shape == (501, 501) and a.dtype == np.int64 for a in (*s.values(), r14, r16))
    assert old15.shape == (500, 500)
    assert (r14 == s[17414]).mean() > 0.99 and (r16 == s[17416]).mean() > 0.99      # filter changes <1% of pixels
    assert (old15 == s[17415][:500, :500]).mean() > 0.99                            # older export = top-left crop
    assert (old15 == s[17415][1:501, :500]).mean() < 0.98                           # and not a shifted crop


def test_held_out_beats_nearest_on_real_sections():
    a, t, b = (f(f"sframe_{z}_{T}_4.csv.gz") for z in (17414, 17415, 17416))
    pred = reconstruct([a, b], 2, "sdf")[1]
    assert set(np.unique(pred)) <= set(np.unique(a)) | set(np.unique(b))
    acc, acc_nearest = (pred == t).mean(), max((a == t).mean(), (b == t).mean())
    assert acc > acc_nearest + 0.05                                                  # measured: 0.940 vs 0.845
