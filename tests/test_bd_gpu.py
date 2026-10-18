"""GPU kernel logic checks. Run with a CUDA GPU, or on the CPU with  NUMBA_ENABLE_CUDASIM=1 pytest tests/test_bd_gpu.py
(the simulator is very slow, so the sizes here are small)."""
import os
import numpy as np
import pytest

numba = pytest.importorskip("numba")
from numba import cuda
if not (cuda.is_available() or os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"):
    pytest.skip("no CUDA device (set NUMBA_ENABLE_CUDASIM=1 for the simulator)", allow_module_level=True)

from zmorph.bd import sample_ecs, wall_force
from zmorph.bd_gpu import simulate_dynamic_gpu

H = 8.0
PER = (True, True, False)
SMALL = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"


def _slab(nz=13):
    z = (np.arange(nz) - nz // 2) * H
    return np.broadcast_to((2 * np.abs(z))[None, None, :], (6, 6, nz)).copy(), nz // 2


def test_particles_stay_inside_and_force_matches_ideal_gas():
    m, c = _slab(); w = 20.0; N = 150 if SMALL else 20000
    x0 = sample_ecs(m - w, H, N, periodic=PER, seed=1)
    r = simulate_dynamic_gpu(m, H, x0, lambda t: w, 1.0, 0.5, nrec=4, steps_per_rec=60 if SMALL else 400, periodic=PER, seed=2)
    zz = r["pos"][-1, :, 2] - c * H
    assert np.abs(zz).max() <= w / 2 + 1e-3 and r["alive"][-1].all()
    _, F = wall_force(r["wall_push"], r["t"])
    assert F.mean() == pytest.approx(2 * N / w, rel=0.5 if SMALL else 0.06)


def test_absorbing_wall_absorbs_on_the_patch_side_only():
    m, c = _slab(); w = 24.0; N = 150 if SMALL else 5000
    patch = np.zeros(m.shape, np.int32); patch[:, :, c + 1:] = 1                # upper side only
    x0 = sample_ecs(m - w, H, N, periodic=PER, seed=1)
    r = simulate_dynamic_gpu(m, H, x0, lambda t: w, 1.0, 0.5, nrec=4, steps_per_rec=100 if SMALL else 1000, periodic=PER,
                             patch=patch, p_abs=[0.0, 1.0], seed=2)
    got = r["absorbed"]
    assert got.mean() > 0.2
    assert (r["abs_pos"][got, 2] > c * H).all()
