"""Moving walls, wall force and absorbing patches in the Brownian dynamics (fast checks)."""
import math
import numpy as np
import pytest

pytest.importorskip("numba")
from scipy.optimize import brentq
from zmorph.bd import simulate_dynamic, sample_ecs, wall_force

H = 8.0
PER = (True, True, False)


def _slab(nz=13):
    z = (np.arange(nz) - nz // 2) * H
    return np.broadcast_to((2 * np.abs(z))[None, None, :], (6, 6, nz)).copy(), nz // 2


def test_wall_force_equals_ideal_gas_pressure():
    m, c = _slab(); w, N = 20.0, 6000
    x0 = sample_ecs(m - w, H, N, periodic=PER, seed=1)
    r = simulate_dynamic(m, H, x0, lambda t: w, 1.0, 0.5, nrec=10, steps_per_rec=400, periodic=PER, seed=2)
    _, F = wall_force(r["wall_push"], r["t"])
    assert F.mean() == pytest.approx(2 * N / w, rel=0.04)


def test_moving_walls_keep_particles_inside_and_do_isothermal_work():
    m, c = _slab(17); N = 6000; w0, w1, T = 40.0, 20.0, 4000.0
    wf = lambda t: w0 + (w1 - w0) * min(t / T, 1.0)
    x0 = sample_ecs(m - w0, H, N, periodic=PER, seed=1)
    r = simulate_dynamic(m, H, x0, wf, 1.0, 0.5, nrec=40, steps_per_rec=int(T / 0.5 / 40), periodic=PER, seed=2)
    zz = r["pos"][-1, :, 2] - c * H
    assert np.abs(zz).max() <= w1 / 2 + 1e-6 and r["alive"][-1].all()
    tm, F = wall_force(r["wall_push"], r["t"]); dw = np.abs(np.diff(r["w"]))
    W = (F * dw / 2).sum()
    assert W == pytest.approx(N * math.log(w0 / w1), rel=0.08)             # slow compression: isothermal ideal-gas work


def test_partially_absorbing_wall_matches_robin_decay():
    m, c = _slab(); w, N = 20.0, 12000
    patch = np.zeros(m.shape, np.int32); patch[:, :, c + 1:] = 1
    kappa = 1.0 / w
    lam = brentq(lambda k: k * math.tan(k * w) - kappa, 1e-9, math.pi / (2 * w) - 1e-9) ** 2
    sig = 0.5; dt = sig ** 2 / 2; p = kappa * math.sqrt(math.pi * dt)
    x0 = sample_ecs(m - w, H, N, periodic=PER, seed=1)
    T = 2.2 / lam
    r = simulate_dynamic(m, H, x0, lambda t: w, 1.0, dt, nrec=30, steps_per_rec=int(T / dt / 30), periodic=PER, patch=patch, p_abs=np.array([0.0, p]), seed=2)
    t, n = r["t"], r["alive"].sum(1).astype(float)
    s = (t >= 0.4 / lam) & (t <= 2.0 / lam) & (n > 50)
    assert -np.polyfit(t[s], np.log(n[s]), 1)[0] == pytest.approx(lam, rel=0.12)
    assert (r["abs_pos"][r["absorbed"], 2] - c * H > 0).all()                # absorbed only at the patch wall


def test_patch_id_zero_never_absorbs():
    m, c = _slab(); w = 20.0
    x0 = sample_ecs(m - w, H, 2000, periodic=PER, seed=1)
    r = simulate_dynamic(m, H, x0, lambda t: w, 1.0, 0.5, nrec=5, steps_per_rec=300, periodic=PER, patch=np.zeros(m.shape, np.int32), p_abs=np.array([1.0, 1.0]), seed=2)
    assert r["alive"][-1].all()
