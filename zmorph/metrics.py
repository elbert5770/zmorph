import numpy as np
import scipy.ndimage as ndi
from skimage import measure


def _boundary(sl):
    b = np.zeros(sl.shape, bool)
    b[:-1] |= sl[:-1] != sl[1:]
    b[:, :-1] |= sl[:, :-1] != sl[:, 1:]
    return b


def boundary_distance(pred, gt, zs):
    """Mean symmetric distance (voxels) between label interfaces, averaged over slices."""
    out = []
    for z in zs:
        bp, bg = _boundary(pred[z]), _boundary(gt[z])
        if not bg.any() or not bp.any():
            continue
        dg, dp = ndi.distance_transform_edt(~bg), ndi.distance_transform_edt(~bp)
        out.append(0.5 * (dg[bp].mean() + dp[bg].mean()))
    return float(np.mean(out))


def face_area(vol):
    return sum(int((np.diff(vol, axis=a) != 0).sum()) for a in range(3))


def evaluate(pred, gt, factor=5, cell_of_interest=1, gap_range=(16, 20)):
    z = np.arange(gt.shape[0])
    mid = z[z % factor != 0]
    sec = z[z % factor == 0]
    res = {}
    res["acc_mid"] = float((pred[mid] == gt[mid]).mean())
    res["acc_sec"] = float((pred[sec] == gt[sec]).mean())
    dices = []
    per = {}
    for k in np.unique(gt):
        if k == 0:
            continue
        a, b = pred[mid] == k, gt[mid] == k
        d = 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)
        per[int(k)] = float(d)
        dices.append(d)
    res["dice_mid"] = float(np.mean(dices))
    zz = np.arange(*gap_range)
    a, b = pred[zz] == cell_of_interest, gt[zz] == cell_of_interest
    res["dice_branch_gap"] = float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))
    res["bdist_mid"] = boundary_distance(pred, gt, mid)
    comp_err, extra = 0, 0
    for k in np.unique(gt):
        if k == 0:
            continue
        n_g = measure.label(gt == k, connectivity=1).max()
        n_p = measure.label(pred == k, connectivity=1).max()
        comp_err += abs(n_p - n_g)
    for k in np.unique(pred):
        if k not in np.unique(gt):
            extra += 1
    res["component_err"] = int(comp_err + extra)
    res["area_ratio"] = face_area(pred) / face_area(gt)
    res["per_cell_dice"] = per
    return res
