#!/usr/bin/env python3
"""One point of the network sweep: jprecond in q1 mode on the graph Laplacian of a .geo.h5
network with an H x H coarse grid over its bounding box, solved with PCG. Prints one JSON
record like examples/grid_q1.py.

    python examples/network_q1.py -domain mikado.geo.h5 -H 8
    examples/sweep_mikado_q1.sh              # generates the networks, runs, plots

The network needs Dirichlet nodes on its whole bounding box (make_geo --dirichlet xmin xmax
ymin ymax): the reference method's coarse hats vanish on the box boundary and its subdomains
leave those nodes out, so nodes there must be constrained.
"""
import argparse
import contextlib
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
from netprecond.jprecond import JPrecond                                      # noqa: E402
from netprecond.pcg import pcg                                                # noqa: E402
from netprecond.problems import network_laplacian, q1_subdomains, rhs_sin   # noqa: E402

p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("-domain", required=True, help=".geo.h5 network")
p.add_argument("-H", type=int, default=4, help="coarse cells per axis over the xy bounding box")
p.add_argument("--weights", default="unit", choices=["unit", "inverse_length"])
p.add_argument("--rtol", type=float, default=1e-10)
p.add_argument("--label", default=None, help="free text copied into the record (e.g. the mikado mass)")
args = p.parse_args()

points, A, dirichlet, kept = network_laplacian(args.domain, args.weights)
b = rhs_sin(A, dirichlet)
t0 = time.time()
doms = q1_subdomains(points[:, :2], [args.H, args.H])
with contextlib.redirect_stdout(sys.stderr):
    M = JPrecond(A, points[:, :2], [args.H, args.H], domains=doms, coarse_space="q1")
t_setup = time.time() - t0
t0 = time.time()
x, info = pcg(A, b, M.matmul, rtol=args.rtol)
t_solve = time.time() - t0

rec = {"domain": args.domain, "label": args.label, "nodes": int(A.shape[0]), "dirichlet": int(dirichlet.sum()),
       "H": args.H, "subdomains": len(doms), "subdomain_nodes_max": int(max(len(d) for d in doms)),
       "coarse_dofs": int(M.coarse_basis_int.shape[1]), "weights": args.weights,
       "iterations": info["iterations"], "cond": info["cond"], "lmin": info["lmin"], "lmax": info["lmax"],
       "residual": info["residuals"][-1], "converged": info["converged"], "t_setup": t_setup, "t_solve": t_solve}
print(json.dumps(rec))
