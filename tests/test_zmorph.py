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


@pytest.mark.parametrize("method", ["sdf0_cubic", "sdf0_4pt", "sdf_cubic", "sdf_4pt"])
def test_higher_order_sdf_methods_are_partitions(scene, method):
    gt, secs = scene
    v = reconstruct(secs, 5, method)
    assert v.shape == gt.shape and v.dtype == np.int64
    assert set(np.unique(v)) <= set(np.unique(gt))
    if method.startswith("sdf0"):
        assert (v[::5] == gt[::5]).all()          # interpolating, binary fields: observed sections reproduced exactly


def test_higher_order_reduces_to_linear_with_two_sections(scene):
    gt, secs = scene
    two = [secs[1], secs[2]]
    lin = reconstruct(two, 5, "sdf0")
    for m in ("sdf0_4pt", "sdf0_cubic"):
        assert (reconstruct(two, 5, m) == lin).all(), m


def test_split3_scene_geometry():
    """Three daughters, none overlapping the parent footprint of the previous section."""
    import scipy.ndimage as ndi
    gt = phantom.scene_split3()
    L = gt == 1
    comps = lambda z: ndi.label(L[z])[1]
    assert [comps(z) for z in (0, 5, 10, 15, 20, 25)] == [1, 1, 1, 1, 3, 3]
    cc, n = ndi.label(L[20])
    assert all((L[15] & (cc == i)).sum() == 0 for i in range(1, n + 1))
    assert [comps(z) for z in (16, 17, 18, 19)] == [1, 1, 1, 3]


def test_persistence_guard_prevents_deletion_and_is_neutral_elsewhere():
    gt = phantom.scene_split3()
    secs = [gt[z] for z in range(0, 26, 5)]
    plain = reconstruct(secs, 5, "sdf")
    guarded = reconstruct(secs, 5, "sdf", persist=True)
    assert min((plain[z] == 1).sum() for z in range(16, 20)) == 0           # the plain method deletes the neurite
    assert min((guarded[z] == 1).sum() for z in range(16, 20)) > 0          # the guard keeps it
    g2 = phantom.scene_branch()
    s2 = [g2[z] for z in range(0, 31, 5)]
    assert (reconstruct(s2, 5, "sdf", persist=True) == reconstruct(s2, 5, "sdf")).all()


def test_component_translation_is_a_partition_and_keeps_sections():
    gt = phantom.scene_split3()
    secs = [gt[z] for z in range(0, 26, 5)]
    v = reconstruct(secs, 5, "sdf", align="component", maxshift=40)
    assert set(np.unique(v)) <= set(np.unique(gt))
    assert (v[::5] == gt[::5]).mean() > 0.999


def test_descriptor_recovers_tilt_from_footprint_elongation():
    """A tube of radius r and slope v cuts a plane in an ellipse with a/b = sqrt(1+v^2)."""
    from zmorph.interp import _descriptor
    yy, xx = np.mgrid[0:80, 0:160]
    r, v = 3.0, 4.0
    a = r * np.sqrt(1 + v * v)
    m = ((xx - 80) / a) ** 2 + ((yy - 40) / r) ** 2 <= 1.0
    c, u, vv, ell = _descriptor(m)
    assert abs(vv - v) / v < 0.15 and abs(u[1]) > 0.99 and ell > 0.9      # pixelisation biases a 3 px radius by about +11%


def test_tube_links_solve_the_split_scene_and_do_no_harm_elsewhere():
    from zmorph.interp import _components, _tube_link
    gt = phantom.scene_split3()
    secs = [gt[z] for z in range(0, 26, 5)]
    p = dict(tube_min_area=12, tube_min_slope=1.0, tube_tol=3.0, tube_min_ellipse=0.6, tube_min_continue=0.3)
    speeds = []
    for d in _components(secs[4] == 1):
        lk = _tube_link(secs[3] == 1, d, 5.0, p)
        assert lk is not None and lk["mode"] == "up" and 0.0 < lk["zJ"] < 2.0   # junction estimated near the true z=15.5 (0.5)
        speeds.append(float(np.linalg.norm(lk["shift"])))
    assert all(abs(s - 6.0) < 0.8 for s in speeds)                              # true daughter speed is 6 px per z
    r = evaluate(reconstruct(secs, 5, "sdf", tube=True, persist=True), gt)
    assert r["dice_branch_gap"] > 0.85
    for fn in (phantom.scene_branch, phantom.scene_tilted):                      # no overlap-free thin branches there: unchanged
        g = fn()
        s2 = [g[z] for z in range(0, g.shape[0], 5)]
        assert (reconstruct(s2, 5, "sdf", tube=True) == reconstruct(s2, 5, "sdf")).all()


def test_tube_with_persistence_is_not_worse_than_persistence_when_the_junction_pair_has_no_valid_link():
    gt = phantom.scene_split3(zJ=17.5)       # daughters unresolved at section 20: that footprint is not an ellipse
    secs = [gt[z] for z in range(0, 26, 5)]
    a = reconstruct(secs, 5, "sdf", persist=True)
    b = reconstruct(secs, 5, "sdf", tube=True, persist=True)
    gap = lambda v: evaluate(v, gt)["dice_branch_gap"]
    assert gap(b) >= gap(a) - 0.01                                           # junction pair: falls back to the guard
    assert evaluate(b, gt)["dice_mid"] >= evaluate(a, gt)["dice_mid"]        # and other pairs (non-overlapping parent) improve


@pytest.mark.parametrize("kw", [dict(), dict(symmetric=True), dict(arrival="rank", symmetric=True),
                                dict(arrival="Trank", symmetric=True, layer_z=2.0), dict(arrival="T", layer_z=[1.5, 4.5])])
def test_heat_variants_keep_sources_exact_and_are_partitions(scene, kw):
    gt, secs = scene
    v = reconstruct(secs, 5, "heat", **kw)
    assert (v[::5] == gt[::5]).all()                       # the source slices are boundary conditions
    assert set(np.unique(v)) <= set(np.unique(gt))


def test_symmetric_heat_is_independent_of_the_direction_of_z_but_the_published_one_is_not(scene):
    gt, secs = scene
    for kw in (dict(symmetric=True), dict(arrival="Trank", symmetric=True, layer_z=2.0)):
        assert (reconstruct(secs, 5, "heat", **kw) == reconstruct(secs[::-1], 5, "heat", **kw)[::-1]).all()
    assert (reconstruct(secs, 5, "heat") != reconstruct(secs[::-1], 5, "heat")[::-1]).any()


def test_heat2_improves_on_the_published_heat_algorithm_on_the_arch_and_tilted_phantoms():
    for fn in (phantom.scene_branch, phantom.scene_tilted):
        g = fn()
        s = [g[z] for z in range(0, g.shape[0], 5)]
        old = evaluate(reconstruct(s, 5, "heat"), g)["bdist_mid"]
        new = evaluate(reconstruct(s, 5, "heat2"), g)["bdist_mid"]
        assert new < old
