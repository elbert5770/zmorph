"""Synaptic-cleft candidates and release sources."""
import numpy as np
import pytest

from zmorph.synapse import contact_patches, score_patches, snap_annotations, release_positions

H = 8.0


def _two_cells(r=200.0):
    """Cell 1 below z=40 and cell 2 above, separated by a gap except over a disc contact of radius r; a long thin
    sheet contact elsewhere (not synapse-like)."""
    n = 80; L = np.zeros((n, n, 24), np.int64)
    L[:, :, :12] = 1; L[:, :, 12:] = 2
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    # break the contact everywhere except a disc: fill a 1-voxel background layer
    disc = np.hypot(xx - 40, yy - 40) * H <= r
    L[:, :, 11] = np.where(disc, 1, 0)           # background layer away from the disc, so the cells
    L[:, :, 12] = np.where(disc, 2, 0)           # touch only where `disc` is True
    return L


def test_disc_contact_area_and_normal():
    L = _two_cells(200.0)
    ps = contact_patches(L, H)
    assert len(ps) == 1
    p = ps[0]
    assert p.area == pytest.approx(np.pi * 200.0 ** 2, rel=0.15)
    assert abs(p.normal[2]) > 0.99 and p.aspect > 0.8 and p.pair == (1, 2)
    assert np.allclose(p.center[:2], np.array([40, 40]) * H, atol=H)


def test_scoring_prefers_synapse_sized_round_flat_patch():
    big = _two_cells(200.0)
    n = 140
    sheet = np.zeros((n, n, 24), np.int64); sheet[:, :, :12] = 3; sheet[:, :, 12:] = 4          # whole-face contact
    s = score_patches(contact_patches(big, H) + contact_patches(sheet, H))
    assert s[0].pair == (1, 2) and s[0].score > 0.5 and s[1].score < 0.3


def test_snap_annotations():
    ps = contact_patches(_two_cells(200.0), H)
    got = snap_annotations(ps, [[40 * H, 40 * H, 12 * H - 4], [10.0, 10.0, 100.0]], max_dist=100.0)
    assert got[0] is ps[0] and got[1] is None


def test_release_positions_lie_in_cleft():
    L = _two_cells(200.0); p = contact_patches(L, H)[0]
    z = np.arange(L.shape[2]) * H
    m = np.broadcast_to(2 * np.abs(z - 11.5 * H)[None, None, :], L.shape).copy()
    w = 24.0
    x = release_positions(p, m, H, w, 500, reach=40.0, seed=1)
    assert x.shape == (500, 3)
    assert np.abs(x[:, 2] - 11.5 * H).max() <= w / 2 + H / 2
    assert np.linalg.norm(x - p.center, axis=1).max() <= p.radius + 40 + H


def test_smooth_labels_removes_thin_sheet_but_keeps_blocks():
    from zmorph.mesh import smooth_labels
    vol = np.ones((40, 40, 40), np.int64)
    vol[:, :, 20:] = 2                                           # two big blocks
    vol[10:30, 10:30, 17:20] = 3                                  # a 3-voxel thick sheet of cell 3 on block 1
    out = smooth_labels(vol, 3.0, iters=2)[0]              # wedged between two blocks it needs a second pass
    assert (out == 3).sum() == 0                                 # the sheet is gone
    assert abs(int((out == 1).sum()) - int((vol != 2).sum())) < 0.02 * vol.size
    big = np.ones((40, 40, 40), np.int64); big[:, :, 20:] = 2; big[5:35, 5:35, 4:20] = 3      # a 16-voxel slab survives, with its edges rounded
    assert (smooth_labels(big, 3.0, iters=2)[0] == 3).sum() > 0.75 * (big == 3).sum()
