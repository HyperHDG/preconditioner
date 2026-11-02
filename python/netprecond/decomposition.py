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


def q1_hats(points, cells, eps=1e-10):
    """Every tensor-product Q1 hat function of a Cartesian grid with cells[d] cells per axis
    over the bounding box of the points, evaluated at the points: sparse (n_points, n_hats),
    hats numbered with the first axis fastest. A point belongs to the cell it lies in with
    tolerance eps on the lower side and is strictly inside on the upper, so points on the
    upper box faces belong to no cell (the reference's convention)."""
    d = len(cells)
    x = np.asarray(points, dtype=float)[:, :d]
    lo = x.min(axis=0)
    h = (x.max(axis=0) - lo) / np.asarray(cells, dtype=float)
    nodes_per_axis = np.asarray(cells) + 1
    stride = np.cumprod(np.r_[1, nodes_per_axis[:-1]])          # flat hat index: i + n0 * (j + n1 * k)
    rows, cols, vals = [], [], []
    for cell in np.ndindex(*cells):
        cell = np.asarray(cell)
        inside = np.ones(len(x), dtype=bool)
        for a in range(d):
            s = x[:, a] - lo[a] + eps
            inside &= (s > h[a] * cell[a]) & (s < h[a] * (cell[a] + 1))
        idx = np.flatnonzero(inside)
        if not len(idx):
            continue
        local = (x[idx] - lo) / h - cell                          # in [0, 1]^d
        for corner in np.ndindex(*([2] * d)):
            corner = np.asarray(corner)
            value = np.prod(np.where(corner, local, 1.0 - local), axis=1)
            rows.append(idx)
            cols.append(np.full(len(idx), int(np.dot(cell + corner, stride))))
            vals.append(value)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(len(x), int(np.prod(nodes_per_axis))))


def q1(points, cells, eps=1e-10, support_eps=1e-14):
    """The decomposition of Görtz, Hellman, Målqvist (2024) as the reference implements it:
    coarse space = the Q1 hats of the interior coarse nodes (zero on the box boundary, where
    the data is Dirichlet); subdomains = the supports of all hats, boundary points removed.
    A boundary point is one where no interior hat is positive. Node-based."""
    cells = tuple(cells)
    d = len(cells)
    hats = q1_hats(points, cells, eps)
    nodes_per_axis = np.asarray(cells) + 1
    multi = np.array(list(np.ndindex(*nodes_per_axis[::-1])))[:, ::-1]   # flat index: first axis fastest
    interior = np.all((multi > 0) & (multi < np.asarray(cells)), axis=1)
    coarse = hats[:, np.flatnonzero(interior)]
    boundary = np.asarray(coarse.sum(axis=1)).ravel() < eps
    supports = hats.tolil()
    supports[boundary, :] = 0
    supports = supports.tocsc()
    subdomains = []
    for k in range(supports.shape[1]):
        nj = np.flatnonzero(supports.getcol(k).toarray().ravel() > support_eps)
        if len(nj):
            subdomains.append(nj)
    return Decomposition(coarse.tocsr(), subdomains)
