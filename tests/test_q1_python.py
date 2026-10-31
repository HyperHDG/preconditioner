"""Generation 1 against generation 2: the reference gortz_hellman_malqvist_22 and the
optimized jprecond in q1 mode must produce the same CG iteration history when jprecond
gets the subdomains the reference uses, the supports of the coarse hat functions."""
import numpy as np
import scipy.sparse.linalg as spla

from netprecond.gortz_hellman_malqvist_22 import gortz_hellman_malqvist_22 as GHM22
from netprecond.jprecond import JPrecond, Domains

from conftest import golden
from problems import cg_history, grid_laplacian


def domains_from_supports(coarse_basis, eps=1e-14):
    """What GHM22.precond loops over: per coarse column, the nodes where it is positive."""
    cb = coarse_basis.tocsc()
    doms = []
    for k in range(cb.shape[1]):
        col = cb.getcol(k).toarray().ravel()
        nj = np.flatnonzero(col > eps)
        if len(nj):
            doms.append(nj)
    offsets = np.r_[0, np.cumsum([len(d) for d in doms])]
    return Domains(offsets, np.concatenate(doms))


def test_jprecond_q1_matches_reference():
    n, H = 33, [4, 4]
    points, A, dirichlet = grid_laplacian(n)
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    b[dirichlet] = 0.0

    ref = GHM22(points, H)
    M_ref = spla.LinearOperator(A.shape, matvec=lambda r: ref.precond(A, r))
    jp = JPrecond(A, points, H, domains=domains_from_supports(ref.coarse_basis), coarse_space="q1")
    M_jp = spla.LinearOperator(A.shape, matvec=jp.matmul)

    h_ref = cg_history(A, b, M_ref)
    h_jp = cg_history(A, b, M_jp)
    assert len(h_ref) == len(h_jp), (len(h_ref), len(h_jp))
    np.testing.assert_allclose(h_ref, h_jp, rtol=1e-6, atol=1e-13)

    g = golden("q1_grid33_H4")
    assert len(h_ref) == g["iterations"]
    np.testing.assert_allclose(h_ref, g["residuals"], rtol=1e-6, atol=1e-13)
