"""The q1 decomposition is the reference's: the same interior coarse basis and the same
subdomains (the supports of all hats, boundary points removed), on a grid and on scattered
2D and 3D points, where hats are evaluated at arbitrary positions and the cell and support
conventions matter. Values may differ in the last bits, the products are associated
differently; the index sets must be identical."""
import numpy as np
import pytest

from netprecond import q1, expand
from netprecond.problems import network_laplacian
from netprecond.reference.gortz_hellman_malqvist_22 import gortz_hellman_malqvist_22 as Reference


def reference_subdomains(ref, eps=1e-14):
    """What Reference.precond loops over: per hat, the points where it is positive."""
    cb = ref.coarse_basis.tocsc()
    out = []
    for k in range(cb.shape[1]):
        nj = np.flatnonzero(cb.getcol(k).toarray().ravel() > eps)
        if len(nj):
            out.append(nj)
    return out


def assert_same(dec, ref):
    B, R = dec.coarse_basis, ref.coarse_basis_int.tocsr()
    assert B.shape == R.shape
    assert (B != 0).nnz == (R != 0).nnz and ((B != 0) != (R != 0)).nnz == 0, "sparsity pattern"
    np.testing.assert_allclose(B.toarray(), R.toarray(), rtol=0, atol=1e-14)
    subs = reference_subdomains(ref)
    assert len(dec.subdomains) == len(subs)
    for a, b in zip(dec.subdomains, subs):
        np.testing.assert_array_equal(a, b)


def test_q1_on_grid(grid33):
    points, A, _, _ = network_laplacian(grid33)
    dec = q1(points, (4, 4))
    assert dec.coarse_basis.shape == (33 * 33, 3 * 3)
    assert_same(dec, Reference(points[:, :2], [4, 4]))


@pytest.mark.parametrize("cells", [(5, 3), (3, 4, 2)])
def test_q1_on_scattered_points(cells):
    rng = np.random.default_rng(1)
    points = rng.random((2000, len(cells)))
    dec = q1(points, cells)
    assert_same(dec, Reference(points, list(cells)))
    s = np.asarray(dec.coarse_basis.sum(axis=1)).ravel()     # interior hats: 0 <= sum <= 1
    assert s.max() <= 1 + 1e-12 and s.min() >= 0


def test_expand_blocks():
    rng = np.random.default_rng(2)
    dec = q1(rng.random((500, 2)), (3, 3))
    dec2 = expand(dec, 2)
    assert dec2.coarse_basis.shape == (1000, 2 * dec.coarse_basis.shape[1])
    assert dec2.coarse_basis[1::2, 1::2].nnz == dec.coarse_basis.nnz and dec2.coarse_basis[0::2, 1::2].nnz == 0
    for nj, nj2 in zip(dec.subdomains, dec2.subdomains):
        np.testing.assert_array_equal(nj2, np.sort(np.r_[2 * nj, 2 * nj + 1]))
