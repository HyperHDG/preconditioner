"""jprecond's q1 coarse space is the reference's: the coarse_basis_int both store is the
same matrix, on a grid and on scattered 2D and 3D points (where the bilinear hats are
evaluated at arbitrary positions and the empty-support bookkeeping matters), and so is the
coarse operator built from it."""
import numpy as np
import scipy.sparse as sp

from netprecond.gortz_hellman_malqvist_22 import gortz_hellman_malqvist_22 as GHM22
from netprecond.jprecond import Domains, JPrecond

from netprecond.problems import network_laplacian


def jprecond_q1(points, H, A=None):
    """A JPrecond in q1 mode; the constructor factorizes the coarse operator and the
    subdomain blocks, so it gets the identity and one dummy subdomain when only the coarse
    space is under test."""
    A = sp.identity(len(points), format="csr") if A is None else A
    return JPrecond(A, points, H, domains=Domains(np.array([0, 1]), np.array([0])), coarse_space="q1")


def same_sparse(a, b):
    if a.shape != b.shape:
        return False
    d = (a - b).tocsr()
    d.eliminate_zeros()
    return d.nnz == 0


def test_q1_coarse_basis_on_grid(grid33):
    points, A, _, _ = network_laplacian(grid33)
    points = points[:, :2]
    ref, jp = GHM22(points, [4, 4]), jprecond_q1(points, [4, 4])
    assert ref.coarse_basis_int.shape == (33 * 33, 3 * 3)
    assert same_sparse(ref.coarse_basis_int, jp.coarse_basis_int)
    B = ref.coarse_basis_int
    assert same_sparse(B.T @ A @ B, jp.coarse_basis_int.T @ A @ jp.coarse_basis_int)


def test_q1_coarse_basis_on_scattered_points_2d_and_3d():
    rng = np.random.default_rng(1)
    for H in ([5, 3], [3, 4, 2]):
        points = rng.random((2000, len(H)))
        ref, jp = GHM22(points, H), jprecond_q1(points, H)
        assert same_sparse(ref.coarse_basis_int, jp.coarse_basis_int)
        s = np.asarray(ref.coarse_basis_int.sum(axis=1)).ravel()   # interior hats: 0 <= sum <= 1
        assert s.max() <= 1 + 1e-12 and s.min() >= 0
