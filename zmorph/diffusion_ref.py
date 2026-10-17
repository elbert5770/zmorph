"""Continuum reference solvers for the diffusion benchmark (all with D = 1, lengths in nm).

  tortuosity_fv(mask, h)        periodic homogenisation cell problem on a voxel mask (finite volumes, CG):
                                tissue-averaged D_xx/D and the ECS tortuosity lambda^2 = alpha / (D_xx/D)
  fv_decay_rate(mask, h)        slowest non-zero decay rate of the voxel graph Laplacian (diffusion in a thin shell)
  cotan_decay_rate(verts, faces)  the same for a triangle surface (cotangent Laplacian, lumped mass): the thin-film
                                limit solved on a surface mesh, the route that is sensitive to out-of-plane bend
  shell_exact_rate(r1, r2)      exact slowest (l = 1) Neumann decay rate of a spherical shell r1 < r < r2
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq
from scipy.special import spherical_jn, spherical_yn


# ----------------------------------------------------------------------------- voxel finite volumes
def _graph(mask, periodic):
    """Sparse adjacency of face-neighbouring True voxels; returns (index grid, A, list of (i, j, axis) arrays)."""
    idx = -np.ones(mask.shape, np.int64); n = int(mask.sum()); idx[mask] = np.arange(n)
    I, J, AX = [], [], []
    for ax in range(3):
        nb_ = np.roll(idx, -1, axis=ax)
        ok = mask & (nb_ >= 0)
        if not periodic[ax]:
            last = [slice(None)] * 3; last[ax] = -1; ok[tuple(last)] = False
        I.append(idx[ok]); J.append(nb_[ok]); AX.append(np.full(ok.sum(), ax))
    return idx, n, np.concatenate(I), np.concatenate(J), np.concatenate(AX)


def tortuosity_fv(mask, h, axis=0, tol=1e-9, maxiter=20000):
    """Fully periodic box. Solves  div(grad chi + e_axis) = 0  in the ECS voxels (no flux through walls).
    Returns dict(alpha, D_tissue, lam2): D_tissue = D_xx/D for the tissue-averaged concentration, lam2 = alpha / D_tissue
    (= D / D_ECS, the tortuosity squared)."""
    idx, n, I, J, AX = _graph(mask, (True, True, True))
    A = sp.coo_matrix((np.ones(len(I)), (I, J)), shape=(n, n)); A = A + A.T
    deg = np.asarray(A.sum(1)).ravel()
    L = sp.diags(deg) - A
    b = np.zeros(n)                                         # sum of the +e offsets over the faces of each voxel
    sel = AX == axis
    np.add.at(b, I[sel], 1.0); np.add.at(b, J[sel], -1.0)   # chi_j - chi_i + 1 across face i->j
    chi, info = spla.cg(L + 1e-12 * sp.identity(n), b, rtol=tol, maxiter=maxiter)
    g = chi[J[sel]] - chi[I[sel]] + 1.0                     # face gradient in units of the imposed unit gradient
    N = mask.size
    Dt = g.sum() / N
    alpha = n / N
    return dict(alpha=alpha, D_tissue=Dt, lam2=alpha / Dt, cg_info=int(info))


def fv_decay_rate(mask, h, periodic=(False, False, False), k=6):
    """Slowest non-zero decay rate (1/time, D = 1) of diffusion on a voxel mask: eigenvalue of (graph Laplacian)/h^2."""
    idx, n, I, J, AX = _graph(mask, periodic)
    A = sp.coo_matrix((np.ones(len(I)), (I, J)), shape=(n, n)); A = A + A.T
    L = (sp.diags(np.asarray(A.sum(1)).ravel()) - A) / h ** 2
    w = np.sort(spla.eigsh(L.tocsc(), k=k, sigma=-1e-4 / h ** 2, which="LM", return_eigenvectors=False))
    return w[w > 1e-9 / h ** 2][0], w


# ----------------------------------------------------------------------------- triangle surface
def cotan_decay_rate(verts, faces, k=6):
    """Slowest non-zero eigenvalue of  L u = lambda M u  with the cotangent stiffness matrix and lumped (barycentric)
    mass: the decay rate of pure surface diffusion (a film of uniform thickness) on a closed triangle mesh."""
    v = np.asarray(verts, float); f = np.asarray(faces)
    a_ = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    f = f[a_ > 1e-6 * np.median(a_)]                    # marching cubes emits zero-area slivers when the level hits a node
    used = np.unique(f); remap = -np.ones(len(v), int); remap[used] = np.arange(len(used))
    v, f = v[used], remap[f]
    n = len(v)
    I, J, W = [], [], []
    area = np.zeros(n)
    for a in range(3):
        i, j, kk = f[:, a], f[:, (a + 1) % 3], f[:, (a + 2) % 3]
        e1, e2 = v[j] - v[i], v[kk] - v[i]
        cr = np.linalg.norm(np.cross(e1, e2), axis=1)
        cot = (e1 * e2).sum(1) / np.maximum(cr, 1e-30)            # cot of the angle at vertex i, opposite edge (j, kk)
        I += [j, kk]; J += [kk, j]; W += [0.5 * cot, 0.5 * cot]
        np.add.at(area, i, cr / 6.0)
    I, J, W = np.concatenate(I), np.concatenate(J), np.concatenate(W)
    Wm = sp.coo_matrix((W, (I, J)), shape=(n, n)).tocsr()
    L = sp.diags(np.asarray(Wm.sum(1)).ravel()) - Wm
    M = sp.diags(area)
    w = np.sort(spla.eigsh(L.tocsc(), k=k, M=M.tocsc(), sigma=-1e-6, which="LM", return_eigenvectors=False))
    return w[w > 1e-8][0], w


# ----------------------------------------------------------------------------- analytic
def shell_exact_rate(r1, r2):
    """Exact slowest decay rate (D = 1) of the l = 1 mode in a spherical shell with reflecting walls r1, r2:
    smallest k with f'(r1) = f'(r2) = 0 for f = A j1(kr) + B y1(kr); rate = k^2."""
    dj = lambda k, r: spherical_jn(1, k * r, derivative=True) * 1.0
    dy = lambda k, r: spherical_yn(1, k * r, derivative=True) * 1.0
    det = lambda k: dj(k, r1) * dy(k, r2) - dy(k, r1) * dj(k, r2)
    ks = np.linspace(0.2 / r2, 6.0 / r2, 4000)
    v = np.array([det(k) for k in ks])
    s = np.nonzero(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    return brentq(det, ks[s[0]], ks[s[0] + 1]) ** 2
