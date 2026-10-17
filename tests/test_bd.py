"""Brownian-dynamics geometry and reference solvers (fast checks)."""
import numpy as np
import pytest

pytest.importorskip("numba")
from zmorph.bd import margin_field, simulate, sample_ecs, effective_diffusivity
from zmorph.diffusion_ref import tortuosity_fv, cotan_decay_rate, shell_exact_rate


def _slab(w=20.0, h=8.0, nz=13):
    z = (np.arange(nz) - nz // 2) * h
    return np.broadcast_to((2 * np.abs(z) - w)[None, None, :], (6, 6, nz)).copy(), z


def test_margin_is_twice_interface_distance():
    lab = np.zeros((40, 12, 12), int); lab[20:] = 1                      # one planar interface at x = 19.5
    m = margin_field(lab, h=1.0)
    x = np.arange(40)
    assert np.allclose(m[:, 6, 6], 2 * np.abs(x - 19.5), atol=1e-6)


def test_walls_confine_and_density_is_uniform():
    phi, z = _slab()
    x0 = sample_ecs(phi, 8.0, 6000, periodic=(True, True, False), seed=1)
    t, pos = simulate(phi, 8.0, x0, 1.0, 0.5, nrec=10, steps_per_rec=300, periodic=(True, True, False), seed=2)
    zz = pos[-1, :, 2] - 6 * 8.0
    assert np.abs(zz).max() <= 10.0 + 1e-6                              # reflected, never outside the gap
    cnt, _ = np.histogram(zz, bins=5, range=(-10, 10))
    assert np.abs(cnt / cnt.mean() - 1).max() < 0.1                     # uniform stationary density (no wall bias)


def test_in_plane_diffusivity_equals_free_value():
    phi, _ = _slab()
    x0 = sample_ecs(phi, 8.0, 8000, periodic=(True, True, False), seed=3)
    t, pos = simulate(phi, 8.0, x0, 1.0, 0.5, nrec=60, steps_per_rec=120, periodic=(True, True, False), seed=4)
    D, sem = effective_diffusivity(t, pos)
    assert abs(D[:2].mean() - 1.0) < 0.06


def test_finite_volume_tortuosity_sheet_and_free():
    assert tortuosity_fv(np.ones((8, 8, 8), bool), 1.0)["lam2"] == pytest.approx(1.0)
    m = np.zeros((16, 16, 16), bool); m[:, :, 6:10] = True               # a sheet: in-plane D_tissue = alpha, lam2 = 1
    assert tortuosity_fv(m, 1.0, axis=0)["lam2"] == pytest.approx(1.0, abs=1e-6)


def test_cotan_sphere_rate_matches_two_over_r_squared():
    from skimage.measure import marching_cubes
    n = 41; i = np.arange(n) - n // 2
    r = np.sqrt(sum(a ** 2 for a in np.meshgrid(i, i, i, indexing="ij")))
    v, f, _, _ = marching_cubes(10.0 - r, 0.0)
    k, _ = cotan_decay_rate(v, f)
    assert k * 100.0 == pytest.approx(2.0, rel=0.05)                    # l = 1 on a sphere of radius 10: 2 D / R^2


def test_shell_exact_rate_close_to_thin_limit():
    assert shell_exact_rate(190.0, 210.0) == pytest.approx(2 / 200.0 ** 2, rel=2e-3)
