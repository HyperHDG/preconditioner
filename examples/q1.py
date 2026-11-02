#!/usr/bin/env python3
"""The reference method on the graph Laplacian of a network: jprecond in q1 mode (the tests
show it is the reference), PCG, and the Lanczos estimate of the condition number of the
preconditioned operator. Output is YAML on stdout, written as the values become available:

    python examples/q1.py --domain grid.geo.h5 -H 8

The network is a fiber_network .geo.h5, a grid included (make_geo --grid 65). It needs
Dirichlet nodes on its whole bounding box (make_geo --dirichlet xmin xmax ymin ymax): the
method's coarse hats vanish there and its subdomains leave those nodes out. The sweeps
examples/sweep_*.sh run this under GNU parallel and plot with sweepplot.
"""
import argparse
import contextlib
import sys
import time

from netprecond.jprecond import JPrecond
from netprecond.pcg import pcg
from netprecond.problems import network_laplacian, q1_subdomains, rhs_sin


def emit(key, value):
    print(f"{key}: {value}", flush=True)


p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("--domain", required=True, metavar="FILE", help=".geo.h5 network")
p.add_argument("-H", type=int, default=4, help="coarse cells per axis over the xy bounding box")
p.add_argument("--weights", default="unit", choices=["unit", "inverse_length"], help="edge weights")
p.add_argument("--rtol", type=float, default=1e-10)
p.add_argument("--label", default=None, help="free text copied into the output, e.g. the grid size or mikado mass")
args = p.parse_args()

points, A, dirichlet, _ = network_laplacian(args.domain, args.weights)
emit("domain", args.domain)
emit("weights", args.weights)
if args.label is not None:
    emit("label", args.label)
emit("H", args.H)
emit("nodes", A.shape[0])
emit("dirichlet", int(dirichlet.sum()))

t0 = time.time()
doms = q1_subdomains(points[:, :2], [args.H, args.H])
with contextlib.redirect_stdout(sys.stderr):      # jprecond reports its sizes on stdout
    M = JPrecond(A, points[:, :2], [args.H, args.H], domains=doms, coarse_space="q1")
emit("subdomains", len(doms))
emit("subdomain_nodes_max", max(len(d) for d in doms))
emit("coarse_dofs", M.coarse_basis_int.shape[1])
emit("t_setup", round(time.time() - t0, 4))

print("residuals:", flush=True)
t0 = time.time()
x, info = pcg(A, rhs_sin(A, dirichlet), M.matmul, rtol=args.rtol,
              callback=lambda k, r: print(f"  - {r:.6e}", flush=True))
emit("t_solve", round(time.time() - t0, 4))
emit("iterations", info["iterations"])
emit("converged", str(info["converged"]).lower())
emit("lmin", f"{info['lmin']:.6f}")
emit("lmax", f"{info['lmax']:.6f}")
emit("cond", f"{info['cond']:.6f}")
