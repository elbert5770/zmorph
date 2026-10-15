import numpy as np
import pytest
from zmorph import phantom
from zmorph.interp import reconstruct, clean_sections
from zmorph.metrics import evaluate
from zmorph.mesh import competition_meshes, interface_conformity


@pytest.fixture(scope="module")
def scene():
    gt = phantom.scene_branch()
    return gt, [gt[z] for z in range(0, 31, 5)]


@pytest.mark.parametrize("method", ["nearest", "sdf0", "sdf", "harmonic", "heat"])
def test_output_is_a_partition_with_known_labels(scene, method):
    gt, secs = scene
    v = reconstruct(secs, 5, method)
    assert v.shape == gt.shape
    assert set(np.unique(v)) <= set(np.unique(gt))  # every voxel carries exactly one real label


def test_sdf_reproduces_clean_sections_and_identical_sections(scene):
    gt, secs = scene
    v = reconstruct(secs, 5, "sdf", clean=True)
    assert (v[::5] == gt[::5]).mean() > 0.999        # sigma=1 rounds a few corner pixels (~0.02%)
    same = reconstruct([secs[2]] * 3, 5, "sdf")       # nothing moves -> nothing may change
    assert all((same[z] == same[0]).all() for z in range(same.shape[0]))
    assert (same[0] == secs[2]).mean() > 0.999


def test_branching_cell_is_handled(scene):
    gt, secs = scene
    r = evaluate(reconstruct(secs, 5, "sdf", clean=True), gt)
    assert r["dice_branch_gap"] > 0.98 and r["component_err"] == 0


def test_cleaning_keeps_supported_fragments_and_drops_unsupported(scene):
    gt, secs = scene
    s = [x.copy() for x in secs]
    assert (secs[3][88:90, 4:6] == 0).all()   # background patch away from any cell
    s[3][88:90, 4:6] = 2                      # tiny island of label 2, absent from neighbouring sections
    out = clean_sections(s)
    assert (out[3] == secs[3]).all()
    assert (clean_sections(secs)[3] == secs[3]).all()   # clean data is left untouched


def test_meshes_are_conforming(scene):
    gt, secs = scene
    _, meshes = competition_meshes(secs, 5)
    c = interface_conformity(meshes)
    assert c["frac_shared"] > 0.9 and c["p95_dist"] < 1e-3


def test_large_segment_ids_survive():
    """MICrONS IDs are ~1e18; labels must not be truncated to int32 anywhere."""
    gt = phantom.scene_branch()
    big = (gt.astype(np.int64) + 864691135000000000)
    secs = [big[z] for z in range(0, 31, 5)]
    for m in ("sdf", "harmonic", "heat", "sdf0", "nearest"):
        v = reconstruct(secs, 5, m)
        assert v.dtype == np.int64 and set(np.unique(v)) <= set(np.unique(big)), m
