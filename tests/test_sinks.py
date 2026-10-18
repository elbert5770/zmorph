"""Sink patch maps and absorption on one side of the gap."""
import numpy as np
import pytest

pytest.importorskip("numba")
from zmorph.bd import simulate_dynamic, sample_ecs
from zmorph.sinks import disc_patch_map, sink_patch_map, absorption_probability, first_passage

H = 8.0


def _two_cells(n=6, nz=17):
    vol = np.ones((n, n, nz), np.int64); vol[:, :, nz // 2:] = 2          # interface at nz//2 - 0.5
    z = (np.arange(nz) - (nz // 2 - 0.5)) * H
    m = np.broadcast_to((2 * np.abs(z))[None, None, :], vol.shape).copy()
    return vol, m


def test_only_the_sink_side_absorbs():
    vol, m = _two_cells(); w = 24.0
    patch = sink_patch_map(vol, 2)
    x0 = sample_ecs(m - w, H, 3000, periodic=(True, True, False), seed=1)
    r = simulate_dynamic(m, H, x0, lambda t: w, 1.0, 0.5, nrec=10, steps_per_rec=400, periodic=(True, True, False),
                         patch=patch, p_abs=[0.0, 1.0], seed=2)
    assert r["absorbed"].mean() > 0.3
    assert (r["abs_pos"][r["absorbed"], 2] > (vol.shape[2] // 2 - 0.5) * H).all()      # on the label-2 side of the interface
    assert (r["abs_patch"][r["absorbed"]] == 1).all()


def test_first_passage_summary_and_probability():
    vol, m = _two_cells(); w = 24.0
    x0 = sample_ecs(m - w, H, 2000, periodic=(True, True, False), seed=1)
    r = simulate_dynamic(m, H, x0, lambda t: w, 1.0, 0.5, nrec=10, steps_per_rec=400, periodic=(True, True, False),
                         patch=sink_patch_map(vol, 2), p_abs=[0.0, 1.0], seed=2)
    fp = first_passage(r)
    assert np.all(np.diff(fp["absorbed_frac"]) >= -1e-12) and fp["absorbed_frac"][-1] == pytest.approx(fp["frac_absorbed"])
    assert absorption_probability(0.05, 0.5) == pytest.approx(0.05 * np.sqrt(np.pi * 0.5))
    with pytest.raises(ValueError):
        absorption_probability(2.0, 0.5)


def test_disc_patch_has_fixed_area_in_the_interface_plane():
    n, nz = 40, 21
    vol = np.ones((n, n, nz), np.int64); vol[:, :, nz // 2:] = 2
    z = (np.arange(nz) - (nz // 2 - 0.5)) * H
    m = np.broadcast_to((2 * np.abs(z))[None, None, :], vol.shape).copy()
    site = np.array([20 * H, 20 * H, (nz // 2) * H])                          # in the gap, on the label-2 side
    R = 100.0
    patch, lab, normal, centre = disc_patch_map(vol, m, H, site, R, depth=40.0)
    assert lab == 2 and abs(abs(normal[2]) - 1) < 1e-6 and abs(centre[2] - (nz // 2 - 0.5) * H) < 1e-6
    layer = patch[:, :, nz // 2 + 1].sum() * H * H                           # one layer 1.5 voxels from the interface
    assert layer == pytest.approx(np.pi * R * R, rel=0.1)
    assert patch[:, :, : nz // 2].sum() == 0                                   # nothing on the other cell
    ball = (np.linalg.norm(np.indices(vol.shape).transpose(1, 2, 3, 0) * H - site, axis=-1) <= R) & (vol == 2)
    assert ball[:, :, nz // 2 + 4].sum() < patch[:, :, nz // 2 + 4].sum()    # the ball is smaller on a deeper layer
