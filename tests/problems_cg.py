"""scipy's CG with the relative true residual after every iteration."""
import numpy as np
import scipy.sparse.linalg as spla


def cg_history(A, b, M, rtol=1e-10, maxiter=500):
    hist = []
    nb = np.linalg.norm(b)

    def cb(x):
        hist.append(float(np.linalg.norm(A @ x - b) / nb))

    x, info = spla.cg(A, b, rtol=rtol, maxiter=maxiter, M=M, callback=cb)
    assert info == 0, f"cg did not converge (info={info})"
    return hist
