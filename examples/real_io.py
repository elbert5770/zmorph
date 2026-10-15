"""Loading and scoring helpers for the MICrONS example (examples/micron_17414_17416.py)."""
import os
import numpy as np
import scipy.ndimage as ndi

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "micron_17414_17416")


def load_labels(path):
    """Segment-ID image from CSV (or .csv.gz). The first row is a header of column indices.
    Row index is the first array axis. IDs are ~1e18, so int64 is required."""
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)


def _boundary(sl):
    b = np.zeros(sl.shape, bool)
    b[:-1] |= sl[:-1] != sl[1:]
    b[:, :-1] |= sl[:, :-1] != sl[:, 1:]
    return b


def score(pred, truth, changed=None, min_area=300):
    """Pixel accuracy (optionally restricted to `changed` pixels), mean interface distance in px,
    mean per-label Dice (labels with at least `min_area` px in truth) and the summed difference
    in connected-component counts per label."""
    r = {"acc": float((pred == truth).mean())}
    if changed is not None:
        r["acc_changed"] = float((pred == truth)[changed].mean())
    bp, bt = _boundary(pred), _boundary(truth)
    dt, dp = ndi.distance_transform_edt(~bt), ndi.distance_transform_edt(~bp)
    r["bdist"] = float(0.5 * (dt[bp].mean() + dp[bt].mean()))
    r["dice"] = float(np.mean([2 * ((pred == k) & (truth == k)).sum() / ((pred == k).sum() + (truth == k).sum())
                               for k in np.unique(truth) if (truth == k).sum() >= min_area]))
    nc = lambda im: {int(k): ndi.label(im == k)[1] for k in np.unique(im)}
    a, b = nc(truth), nc(pred)
    r["comp_err"] = int(sum(abs(b.get(k, 0) - a.get(k, 0)) for k in set(a) | set(b)))
    return r
