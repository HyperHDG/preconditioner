"""Test problems and their subdomain covers. Everything here is built in numpy; no network
file is needed."""
import numpy as np
import scipy.sparse as sp

from .gortz_hellman_malqvist_22 import gortz_hellman_malqvist_22 as GHM22
from .jprecond import Domains


def grid_laplacian(n):
    """n x n points on the unit square, 4-neighbour graph Laplacian with unit weights,
    Dirichlet rows and columns on the whole boundary replaced by the identity (SPD): the
    reference preconditioner assumes Dirichlet data on the box boundary, its subdomains
    leave those nodes out. Returns points (n*n, 2), the matrix (csr) and the mask."""
    xs = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    points = np.c_[X.ravel(), Y.ravel()]
    idx = np.arange(n * n).reshape(n, n)
    e = np.r_[np.c_[idx[:, :-1].ravel(), idx[:, 1:].ravel()], np.c_[idx[:-1, :].ravel(), idx[1:, :].ravel()]]
    W = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n * n, n * n))
    W = (W + W.T).tocsr()
    L = sp.diags(np.asarray(W.sum(axis=1)).ravel()) - W
    dirichlet = (points[:, 0] == 0.0) | (points[:, 0] == 1.0) | (points[:, 1] == 0.0) | (points[:, 1] == 1.0)
    P = sp.diags((~dirichlet).astype(float))
    A = P @ L @ P + sp.diags(dirichlet.astype(float))
    return points, A.tocsr(), dirichlet


def rhs_sin(A, dirichlet):
    """The right-hand side examples/laplacian.cxx uses: sin(1 + i), zero on Dirichlet nodes."""
    b = np.sin(1.0 + np.arange(A.shape[0]))
    b[dirichlet] = 0.0
    return b


def q1_subdomains(points, H, eps=1e-14):
    """The subdomain cover of the reference method: per coarse hat function of the H[0] x H[1]
    coarse grid, the nodes where it is positive, boundary nodes removed. Returned as the
    Domains object jprecond takes, so jprecond q1 and the reference are the same method."""
    cb = GHM22(points, H).coarse_basis.tocsc()
    doms = []
    for k in range(cb.shape[1]):
        nj = np.flatnonzero(cb.getcol(k).toarray().ravel() > eps)
        if len(nj):
            doms.append(nj)
    offsets = np.r_[0, np.cumsum([len(d) for d in doms])]
    return Domains(offsets, np.concatenate(doms))


def network_laplacian(path, weights="unit"):
    """Graph Laplacian of a .geo.h5 network (fiber_network's make_geo format): /domain/points,
    /domain/edges, /domain/types_points. Nodes with a nonzero type are Dirichlet, their rows
    and columns become the identity. Components without a Dirichlet node would make the
    matrix singular and are dropped. weights: "unit" or "inverse_length" (1 / edge length).
    Returns points (n, 3), the matrix (csr), the Dirichlet mask and the kept node indices."""
    import h5py
    from scipy.sparse.csgraph import connected_components

    with h5py.File(path, "r") as f:
        points = f["domain/points"][:]
        edges = f["domain/edges"][:]
        types = f["domain/types_points"][:]
    n = len(points)
    if weights == "unit":
        w = np.ones(len(edges))
    elif weights == "inverse_length":
        w = 1.0 / np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    else:
        raise ValueError(weights)
    W = sp.coo_matrix((w, (edges[:, 0], edges[:, 1])), shape=(n, n))
    W = (W + W.T).tocsr()
    dirichlet = types != 0
    _, comp = connected_components(W, directed=False)
    keep = np.isin(comp, np.unique(comp[dirichlet]))
    if not keep.all():
        W = W[keep][:, keep]
        points, dirichlet = points[keep], dirichlet[keep]
    L = sp.diags(np.asarray(W.sum(axis=1)).ravel()) - W
    P = sp.diags((~dirichlet).astype(float))
    A = P @ L @ P + sp.diags(dirichlet.astype(float))
    return points, A.tocsr(), dirichlet, np.flatnonzero(keep)
