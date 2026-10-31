"""Small test problems. Everything here is built in numpy; no network files are needed."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


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


def cg_history(A, b, M, rtol=1e-10, maxiter=500):
    """Relative residual after every CG iteration with preconditioner M (a LinearOperator)."""
    hist = []
    nb = np.linalg.norm(b)

    def cb(x):
        hist.append(float(np.linalg.norm(A @ x - b) / nb))

    x, info = spla.cg(A, b, rtol=rtol, maxiter=maxiter, M=M, callback=cb)
    assert info == 0, f"cg did not converge (info={info})"
    return hist
