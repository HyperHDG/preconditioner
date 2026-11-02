"""Generation 1 against generation 2: the reference gortz_hellman_malqvist_22 and the
optimized jprecond in q1 mode must produce the same CG iteration history when jprecond
gets the subdomains the reference uses, the supports of the coarse hat functions."""
import numpy as np
import scipy.sparse.linalg as spla

from netprecond.gortz_hellman_malqvist_22 import gortz_hellman_malqvist_22 as GHM22
from netprecond.jprecond import JPrecond

from support import golden
from netprecond.problems import grid_laplacian, q1_subdomains
from problems_cg import cg_history


def test_jprecond_q1_matches_reference():
    n, H = 33, [4, 4]
    points, A, dirichlet = grid_laplacian(n)
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    b[dirichlet] = 0.0

    ref = GHM22(points, H)
    M_ref = spla.LinearOperator(A.shape, matvec=lambda r: ref.precond(A, r))
    jp = JPrecond(A, points, H, domains=q1_subdomains(points, H), coarse_space="q1")
    M_jp = spla.LinearOperator(A.shape, matvec=jp.matmul)

    h_ref = cg_history(A, b, M_ref)
    h_jp = cg_history(A, b, M_jp)
    assert len(h_ref) == len(h_jp), (len(h_ref), len(h_jp))
    np.testing.assert_allclose(h_ref, h_jp, rtol=1e-6, atol=1e-13)

    g = golden("q1_grid33_H4")
    assert len(h_ref) == g["iterations"]
    np.testing.assert_allclose(h_ref, g["residuals"], rtol=1e-6, atol=1e-13)
