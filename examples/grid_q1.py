#!/usr/bin/env python3
"""One point of the grid sweep: jprecond in q1 mode (the reference method, see the tests) on
the graph Laplacian of an n x n grid with an H x H coarse grid, solved with PCG. Prints one
JSON record: sizes, iterations, the Lanczos estimates of the extreme eigenvalues of the
preconditioned operator and their ratio, the condition number.

    python examples/grid_q1.py -n 65 -H 8
    examples/sweep_grid_q1.sh                # the whole sweep, plots and table
"""
import argparse
import contextlib
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
from netprecond.jprecond import JPrecond                                  # noqa: E402
from netprecond.pcg import pcg                                            # noqa: E402
from netprecond.problems import grid_laplacian, q1_subdomains, rhs_sin   # noqa: E402

p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("-n", type=int, default=33, help="points per axis")
p.add_argument("-H", type=int, default=4, help="coarse cells per axis; subdomains are 2 x 2 coarse cells")
p.add_argument("--rtol", type=float, default=1e-10)
args = p.parse_args()
if args.H >= args.n - 1:
    p.error(f"H={args.H} coarse cells need at least one fine edge each: H < n - 1 = {args.n - 1}")

points, A, dirichlet = grid_laplacian(args.n)
b = rhs_sin(A, dirichlet)
t0 = time.time()
doms = q1_subdomains(points, [args.H, args.H])
with contextlib.redirect_stdout(sys.stderr):      # jprecond reports its sizes on stdout
    M = JPrecond(A, points, [args.H, args.H], domains=doms, coarse_space="q1")
t_setup = time.time() - t0
t0 = time.time()
x, info = pcg(A, b, M.matmul, rtol=args.rtol)
t_solve = time.time() - t0

rec = {"n": args.n, "nodes": int(A.shape[0]), "H": args.H, "h_over_H": (args.H) / (args.n - 1),
       "subdomains": len(doms), "subdomain_nodes_max": int(max(len(d) for d in doms)),
       "coarse_dofs": int(M.coarse_basis_int.shape[1]),
       "iterations": info["iterations"], "cond": info["cond"], "lmin": info["lmin"], "lmax": info["lmax"],
       "residual": info["residuals"][-1], "converged": info["converged"],
       "t_setup": t_setup, "t_solve": t_solve}
print(json.dumps(rec))
