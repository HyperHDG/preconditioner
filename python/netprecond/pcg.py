"""Preconditioned conjugate gradients with the Lanczos estimate of the extreme eigenvalues
of the preconditioned operator, as PETSc's KSPComputeExtremeSingularValues does for CG:
the CG coefficients alpha_k, beta_k form the Lanczos tridiagonal T_k whose eigenvalues
approximate those of M^-1 A; the condition number is their ratio."""
import numpy as np
from scipy.linalg import eigvalsh_tridiagonal


def pcg(A, b, M, rtol=1e-10, maxiter=1000, x0=None, callback=None):
    """Solve A x = b with preconditioner M (a callable r -> M^-1 r). Returns x and a dict
    with the relative residual history, the Lanczos eigenvalue bounds and their ratio.
    callback(k, relative residual) is called after every iteration."""
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - A @ x
    nb = np.linalg.norm(b)
    z = M(r)
    p = z.copy()
    rz = r @ z
    alphas, betas, hist = [], [], []
    for _ in range(maxiter):
        Ap = A @ p
        alpha = rz / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        hist.append(float(np.linalg.norm(r) / nb))
        alphas.append(alpha)
        if callback:
            callback(len(hist), hist[-1])
        if hist[-1] < rtol:
            break
        z = M(r)
        rz_new = r @ z
        beta = rz_new / rz
        betas.append(beta)
        p = z + beta * p
        rz = rz_new
    k = len(alphas)
    diag = np.array([1.0 / alphas[i] + (betas[i - 1] / alphas[i - 1] if i > 0 else 0.0) for i in range(k)])
    off = np.array([np.sqrt(betas[i]) / alphas[i] for i in range(k - 1)])
    ev = eigvalsh_tridiagonal(diag, off) if k > 1 else diag
    return x, {"iterations": k, "residuals": hist, "lmin": float(ev.min()), "lmax": float(ev.max()),
               "cond": float(ev.max() / ev.min()), "converged": hist[-1] < rtol}
