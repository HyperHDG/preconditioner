"""AdditiveSchwarz on the q1 decomposition against the reference's precond: the same CG
history on the grid Laplacian, pinned by the golden file. Also: a block-expanded problem,
two decoupled copies of the scalar one, has the same history with expand(dec, 2)."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from netprecond import AdditiveSchwarz, expand, q1
from netprecond.problems import network_laplacian
from netprecond.reference.gortz_hellman_malqvist_22 import gortz_hellman_malqvist_22 as Reference

from problems_cg import cg_history
from support import golden


def test_schwarz_q1_matches_reference(grid33):
    points, A, dirichlet, _ = network_laplacian(grid33)
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    b[dirichlet] = 0.0
    ref = Reference(points[:, :2], [4, 4])
    M = AdditiveSchwarz(A, q1(points, (4, 4)))

    h_ref = cg_history(A, b, spla.LinearOperator(A.shape, matvec=lambda r: ref.precond(A, r)))
    h = cg_history(A, b, M.aslinearoperator())
    assert len(h) == len(h_ref)
    np.testing.assert_allclose(h, h_ref, rtol=1e-6, atol=1e-13)
    g = golden("q1_grid33_H4")
    assert len(h) == g["iterations"]
    np.testing.assert_allclose(h, g["residuals"], rtol=1e-6, atol=1e-13)


def test_schwarz_on_expanded_blocks(grid33):
    points, A, dirichlet, _ = network_laplacian(grid33)
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    b[dirichlet] = 0.0
    h = cg_history(A, b, AdditiveSchwarz(A, q1(points, (4, 4))).aslinearoperator())
    A2 = sp.kron(A, sp.identity(2), format="csr")
    b2 = np.kron(b, [1.0, 0.5])
    h2 = cg_history(A2, b2, AdditiveSchwarz(A2, expand(q1(points, (4, 4)), 2)).aslinearoperator())
    assert len(h2) == len(h)
    np.testing.assert_allclose(h2, h, rtol=1e-6, atol=1e-13)
