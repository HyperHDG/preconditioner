"""Subspace decompositions: the data the additive Schwarz preconditioner is built from, and
the constructors that produce it. A decomposition lives on degrees of freedom; node-based
ones are turned into block-structured ones with expand()."""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class Decomposition:
    """coarse_basis: sparse (n_dofs, n_coarse), the coarse functions as columns, only those
    that conform to the boundary conditions. subdomains: index arrays into the dofs, an
    overlapping cover; each gets its own local solve."""
    coarse_basis: sp.csr_matrix
    subdomains: list

    @property
    def n_dofs(self):
        return self.coarse_basis.shape[0]


def expand(dec, block_size):
    """The same decomposition on block_size dofs per node, dof = node * block_size + component."""
    if block_size == 1:
        return dec
    B = sp.kron(dec.coarse_basis, sp.identity(block_size), format="csr")
    subdomains = [(nj[:, None] * block_size + np.arange(block_size)).ravel() for nj in dec.subdomains]
    return Decomposition(B, subdomains)


def q1(points, cells, eps=1e-10, support_eps=1e-14):
    """The decomposition of Görtz, Hellman, Målqvist (2024) as the reference implements it:
    coarse space = the Q1 hats of the interior coarse nodes of a Cartesian grid over the
    bounding box (zero on the box boundary, where the data is Dirichlet); subdomains = the
    supports of all hats, boundary points removed. A boundary point is one where no interior
    hat is positive. Node-based. The hats come from the C++ library (netprecond._decomposition)."""
    from . import _decomposition as nd
    cells = [int(c) for c in cells]
    x = np.ascontiguousarray(np.asarray(points, dtype=float)[:, :len(cells)])
    lo, hi = x.min(axis=0).tolist(), x.max(axis=0).tolist()
    rows, cols, vals, n_hats = nd.q1_hats(x, cells, lo, hi, eps, support_eps)
    interior, n_interior = nd.q1_interior(cells, lo, hi)
    boundary = nd.q1_boundary_points(rows, cols, vals, interior, len(x), eps)
    hats = sp.csc_matrix((vals, (rows, cols)), shape=(len(x), n_hats))
    coarse = hats[:, np.flatnonzero(interior >= 0)]
    keep = ~boundary[rows]                                   # boundary points leave every support
    supports = sp.csc_matrix((vals[keep], (rows[keep], cols[keep])), shape=hats.shape)
    subdomains = [supports.indices[supports.indptr[k]:supports.indptr[k + 1]] for k in range(n_hats)]
    subdomains = [np.sort(nj) for nj in subdomains if len(nj)]
    return Decomposition(coarse.tocsr(), subdomains)


def restrict(dec, keep):
    """The decomposition on a subset of the dofs: subdomains lose the dofs where keep is
    False and the coarse functions are zeroed there. For dofs a system does not carry, such
    as Dirichlet dofs whose rows HyperHDG leaves empty: a subdomain block containing one of
    them would be singular."""
    keep = np.asarray(keep, dtype=bool)
    B = sp.diags(keep.astype(float)) @ dec.coarse_basis
    B.eliminate_zeros()
    subdomains = [nj[keep[nj]] for nj in dec.subdomains]
    return Decomposition(B.tocsr(), [nj for nj in subdomains if len(nj)])
