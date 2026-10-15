"""Conforming per-cell surface meshes straight from the interpolated fields.

For cell k we mesh the zero level set of the *competition field*
    m_k = f_k - max_{j != k} f_j
which is >0 inside k, <0 outside and, at an interface between cells j and k, satisfies
m_k = -m_j. Marching cubes therefore places identical vertices on the shared interface from
either side, with sub-voxel positions, and no mesh smoothing is needed.
"""
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure

from .interp import competition


def competition_meshes(sections, factor=5, min_voxels=50, **kw):
    lab, best, second, fields = competition(sections, factor, "sdf", return_fields=True, **kw)
    meshes = {}
    for k, f in fields.items():
        if (lab == k).sum() < min_voxels:
            continue
        other = np.where(lab == k, second, best)
        m = np.pad(f - np.where(np.isfinite(other), other, -12.0), 1, constant_values=-1.0)
        v, fc, _, _ = measure.marching_cubes(m, 0.0)
        meshes[int(k)] = (v - 1.0, fc)
    return lab, meshes


def smoothed_indicator_meshes(vol, sigma=1.0, min_voxels=50):
    """Common alternative: smooth each label's indicator and mesh it independently."""
    meshes = {}
    for k in np.unique(vol):
        if (vol == k).sum() < min_voxels:
            continue
        g = np.pad(ndi.gaussian_filter((vol == k).astype(np.float32), sigma), 1)
        v, fc, _, _ = measure.marching_cubes(g, 0.5)
        meshes[int(k)] = (v - 1.0, fc)
    return meshes


def stair_meshes(vol, min_voxels=50):
    meshes = {}
    for k in np.unique(vol):
        if (vol == k).sum() < min_voxels:
            continue
        v, fc, _, _ = measure.marching_cubes(np.pad((vol == k).astype(np.float32), 1), 0.5)
        meshes[int(k)] = (v - 1.0, fc)
    return meshes


def interface_conformity(meshes, near=1.0, tol=1e-3):
    """Vertex-to-vertex statistics over interface vertices (those within `near` voxels of
    another cell's mesh)."""
    d_all = []
    for k, (v, _) in meshes.items():
        others = np.vstack([w for j, (w, _) in meshes.items() if j != k])
        d, _ = cKDTree(others).query(v)
        d_all.append(d[d < near])
    d = np.concatenate(d_all)
    return {"n_interface_vertices": int(d.size), "frac_shared": float((d < tol).mean()),
            "median_dist": float(np.median(d)), "p95_dist": float(np.percentile(d, 95))}


def quality(meshes):
    ang, dih = [], []
    for v, f in meshes.values():
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        def ang_at(p, q, r):
            u, w = q - p, r - p
            return np.degrees(np.arccos(np.clip((u * w).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1) + 1e-12), -1, 1)))
        A, B = ang_at(a, b, c), ang_at(b, c, a)
        ang.append(np.minimum(np.minimum(A, B), 180 - A - B))
        n = np.cross(b - a, c - a); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        fid = np.tile(np.arange(len(f)), 3)
        order = np.lexsort((e[:, 1], e[:, 0])); e, fid = e[order], fid[order]
        same = (e[1:] == e[:-1]).all(1)
        i, j = fid[:-1][same], fid[1:][same]
        dih.append(np.degrees(np.arccos(np.clip((n[i] * n[j]).sum(1), -1, 1))))
    ang, dih = np.concatenate(ang), np.concatenate(dih)
    return {"triangles": int(ang.size), "min_angle_p5": float(np.percentile(ang, 5)),
            "dihedral_mean": float(dih.mean()), "dihedral_p95": float(np.percentile(dih, 95))}


def taubin_joint(meshes, iters=15, lam=0.5, mu=-0.53):
    """Taubin smoothing on the welded union of all cell meshes. Shared interface vertices are
    one vertex, so they move together and conformity is preserved by construction."""
    allv = np.vstack([v for v, _ in meshes.values()])
    key = np.round(allv, 4)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    V = np.zeros((len(uniq), 3)); np.add.at(V, inv, allv); V /= np.bincount(inv, minlength=len(uniq))[:, None]
    off, faces = 0, {}
    edges = []
    for k, (v, f) in meshes.items():
        g = inv[f + off]; off += len(v); faces[k] = g
        edges += [g[:, [0, 1]], g[:, [1, 2]], g[:, [2, 0]]]
    e = np.vstack(edges); e = np.unique(np.sort(e, axis=1), axis=0)
    i = np.concatenate([e[:, 0], e[:, 1]]); j = np.concatenate([e[:, 1], e[:, 0]])
    deg = np.bincount(i, minlength=len(V)).astype(float)
    for _ in range(iters):
        for w in (lam, mu):
            avg = np.zeros_like(V); np.add.at(avg, i, V[j]); avg /= np.maximum(deg, 1)[:, None]
            V = V + w * (avg - V)
    out = {}
    for k, g in faces.items():
        u, loc = np.unique(g, return_inverse=True)
        out[k] = (V[u], loc.reshape(g.shape))
    return out
