"""Sink patch maps and absorption on one side of the gap."""
import numpy as np
import pytest

pytest.importorskip("numba")
from zmorph.bd import simulate_dynamic, sample_ecs
from zmorph.sinks import sink_patch_map, absorption_probability, first_passage

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
