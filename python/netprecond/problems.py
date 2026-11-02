"""Networks as graph Laplacians. Networks come as .geo.h5 files from fiber_network's
make_geo, grids included (make_geo --grid N)."""
import numpy as np
import scipy.sparse as sp


def rhs_sin(A, dirichlet):
    """The right-hand side every driver uses: sin(1 + i), zero on Dirichlet nodes."""
    b = np.sin(1.0 + np.arange(A.shape[0]))
    b[dirichlet] = 0.0
    return b


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
