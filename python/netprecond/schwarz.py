"""The two-level additive Schwarz preconditioner on a decomposition, in scipy:
M^-1 r = B (B^T A B)^-1 B^T r + sum_j R_j^T (R_j A R_j^T)^-1 R_j r."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class AdditiveSchwarz:
    """Factorizes the coarse operator and every subdomain block once; callable on a residual."""

    def __init__(self, A, dec):
        A = sp.csc_matrix(A)
        self.B = dec.coarse_basis.tocsc()
        self.coarse = spla.splu(sp.csc_matrix(self.B.T @ A @ self.B))
        self.subdomains = dec.subdomains
        self.local = [spla.splu(A[nj][:, nj].tocsc()) for nj in self.subdomains]
        self.shape = A.shape

    def __call__(self, r):
        y = self.B @ self.coarse.solve(self.B.T @ r)
        for nj, lu in zip(self.subdomains, self.local):
            y[nj] += lu.solve(r[nj])
        return y

    def aslinearoperator(self):
        return spla.LinearOperator(self.shape, matvec=self, dtype=float)
