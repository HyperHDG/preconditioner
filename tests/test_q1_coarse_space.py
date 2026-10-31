"""jprecond's q1 coarse space is the reference's: same interior hat functions, hence the
same coarse operator, on a grid and on scattered points (where the bilinear hats are
evaluated at arbitrary positions and the empty-support bookkeeping matters)."""
import numpy as np
import scipy.sparse as sp

from netprecond import gortz_hellman_malqvist_22 as ref
from netprecond import jprecond as jp

from problems import grid_laplacian


def coarse_bases(points, H):
    r = ref.gortz_hellman_malqvist_22(points, H)
    if len(H) == 2:
        full = jp._coarse_basis_2d(points, H)
        int_nodes = np.concatenate([j * (H[0] + 1) + np.arange(1, H[0]) for j in range(1, H[1])])
    else:
        full = jp._coarse_basis_3d(points, H)
        int_nodes = np.concatenate([(k * (H[1] + 1) + j) * (H[0] + 1) + np.arange(1, H[0])
                                    for j in range(1, H[1]) for k in range(1, H[2])])
    return r.coarse_basis_int, sp.csr_matrix(full[:, int_nodes])


def same_sparse(a, b):
    d = (a - b).tocsr()
    d.eliminate_zeros()
    return d.nnz == 0 and a.shape == b.shape


def test_q1_coarse_basis_on_grid():
    points, A, _ = grid_laplacian(33)
    b_ref, b_jp = coarse_bases(points, [4, 4])
    assert b_ref.shape == (33 * 33, 3 * 3)
    assert same_sparse(b_ref, b_jp)
    assert same_sparse(b_ref.T @ A @ b_ref, b_jp.T @ A @ b_jp)


def test_q1_coarse_basis_on_scattered_points_2d_and_3d():
    rng = np.random.default_rng(1)
    for H in ([5, 3], [3, 4, 2]):
        points = rng.random((2000, len(H)))
        b_ref, b_jp = coarse_bases(points, H)
        assert same_sparse(b_ref, b_jp)
        # interior hats sum to 1 where no boundary hat is active: partition of unity inside
        s = np.asarray(b_ref.sum(axis=1)).ravel()
        assert s.max() <= 1 + 1e-12 and s.min() >= 0


def test_jprecond_q1_uses_that_basis():
    """The matrix JPrecond stores is the interior basis (kron with repeat=1 is the identity)."""
    points, A, _ = grid_laplacian(17)
    r = ref.gortz_hellman_malqvist_22(points, [4, 4])
    doms = jp.Domains(np.array([0, 1]), np.array([0]))   # one dummy subdomain, not under test
    p = jp.JPrecond(A, points, [4, 4], domains=doms, coarse_space="q1")
    assert same_sparse(p.coarse_basis_int, r.coarse_basis_int)
