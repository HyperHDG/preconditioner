#!/usr/bin/env python3
"""The reference method on an HDG system assembled by HyperHDG: diffusion or Timoshenko beams
on a network, the q1 decomposition expanded to the dofs per node, AdditiveSchwarz, PCG and the
Lanczos estimate of the condition number. YAML on stdout as the values become available.

    python examples/hyperhdg_net_q1.py --domain grid.geo.h5 --problem diffusion -H 4
    python examples/hyperhdg_net_q1.py --domain grid.geo.h5 --problem timoshenko --degree 2 -H 4

HyperHDG is the pinned submodule; its Python interface compiles the C++ below the first time a
degree is asked for and caches it under .cache/hyperhdg/. Networks are three-dimensional;
--flatten solves on a copy with z = 0, the network projected to the plane (what the old
diffusion reproducible did with its 2D topology). The network needs Dirichlet nodes on its whole
bounding box (make_geo --dirichlet xmin xmax ymin ymax).
"""
import argparse
import atexit
import pathlib
import shutil
import sys
import tempfile
import time

import h5py
import numpy as np
import scipy.sparse as sp

from netprecond import AdditiveSchwarz, expand, pcg, q1, restrict

ROOT = pathlib.Path(__file__).resolve().parents[1]
HYPERHDG = ROOT / "submodules" / "HyperHDG"
sys.path.insert(0, str(HYPERHDG / "python"))   # hyperhdg.py is a script next to HyperHDG's examples, not a package yet
import hyperhdg  # noqa: E402

MODULE = """\
#include <HyperHDG/bind_python.hxx>

#include <HyperHDG/geometry/file.hxx>
#include <HyperHDG/node_descriptor/file.hxx>
#include <HyperHDG/topology/file.hxx>

#include <HyperHDG/global_loop/elliptic.hxx>
#include <HyperHDG/local_solver/diffusion_ldgh.hxx>
#include <HyperHDG/local_solver/timoshenko_network.hxx>

#include "reproducibles_python/parameters/diffusion.hxx"

template <typename LocalSolverT>
using Problem = GlobalLoop::Elliptic<
  Topology::File<1, 3>, Geometry::File<1, 3>, NodeDescriptor::File<1, 3>, LocalSolverT, std::vector<double> >;

using Diffusion = Problem<LocalSolver::Diffusion<1, {degree}, {quadrature}, TestParametersSinEllipt, double> >;
using Timoshenko = Problem<LocalSolver::TimoshenkoBeam<1, 3, {degree}, {quadrature},
                                                       LocalSolver::TimoschenkoBeamParametersClamped, double> >;

NB_MODULE({name}, m) {{
  HyperHDG::bind_python<Diffusion>(m, "Diffusion");
  HyperHDG::bind_python<Timoshenko>(m, "Timoshenko");
}}
"""


def code(degree):
    """The module source for one degree: Diffusion and Timoshenko on the 3D file topology."""
    return MODULE.format(degree=degree, quadrature=2 * degree, name=f"netprecond_hdg_net_P{degree}")


def flattened(domain):
    """A temporary copy of the network with z = 0, removed at exit."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="netprecond-flatten-"))
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)
    copy = tmp / pathlib.Path(domain).name
    shutil.copyfile(domain, copy)
    with h5py.File(copy, "r+") as f:
        f["domain/points"][:, 2] = 0.0
    return str(copy)


def emit(key, value):
    print(f"{key}: {value}", flush=True)


p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("--domain", required=True, metavar="FILE", help=".geo.h5 network, three coordinates per node")
p.add_argument("--problem", choices=["diffusion", "timoshenko"], default="diffusion")
p.add_argument("--degree", type=int, default=1, help="polynomial degree of the local solver")
p.add_argument("--tau", type=float, default=1.0, help="HDG penalty parameter")
p.add_argument("-H", type=int, default=4, help="coarse cells per axis over the xy bounding box")
p.add_argument("--subdomains", choices=["node", "component"], default="node",
               help="node: patches keep all components of a node (default); component: one patch per component, the reference's repeat variant")
p.add_argument("--flatten", action="store_true", help="solve on the network projected to the plane (z = 0)")
p.add_argument("--rtol", type=float, default=1e-10)
p.add_argument("--label", default=None, help="free text copied into the output")
args = p.parse_args()

with h5py.File(args.domain, "r") as f:
    points = f["domain/points"][:]
if points.shape[1] != 3:
    p.error(f"{args.domain} has {points.shape[1]} coordinates per node, networks need 3 (use --flatten for a planar problem)")
domain = flattened(args.domain) if args.flatten else args.domain
module = hyperhdg.load(code(args.degree), hyperhdg=HYPERHDG, cache_dir=ROOT / ".cache" / "hyperhdg")
hdg = getattr(module, args.problem.capitalize())(domain, args.tau)

t0 = time.time()
size = hdg.size_of_system()
rows, cols, vals = hdg.trace_to_flux_mat()
A = sp.csr_matrix((vals, (rows, cols)), shape=(size, size))
b = -np.asarray(hdg.residual_flux(hdg.zero_vector()))
block_size = size // len(points)
assert block_size * len(points) == size, "dofs are not a multiple of the nodes"
emit("domain", args.domain)
if args.label is not None:
    emit("label", args.label)
emit("problem", args.problem)
emit("degree", args.degree)
emit("H", args.H)
emit("subdomains_per", args.subdomains)
emit("flatten", str(args.flatten).lower())
emit("nodes", len(points))
emit("block_size", block_size)
emit("dofs", size)
emit("t_assemble", round(time.time() - t0, 4))

t0 = time.time()
dec = expand(q1(points, (args.H, args.H)), block_size, coupled=args.subdomains == "node")   # dofs: node * block_size + component
free = np.asarray(abs(A).sum(axis=1)).ravel() != 0           # Dirichlet dofs have empty rows in HyperHDG's system
dec = restrict(dec, free)
emit("dirichlet_dofs", int((~free).sum()))
M = AdditiveSchwarz(A, dec)
emit("subdomains", len(dec.subdomains))
emit("subdomain_dofs_max", max(len(d) for d in dec.subdomains))
emit("coarse_dofs", dec.coarse_basis.shape[1])
emit("t_setup", round(time.time() - t0, 4))

print("residuals:", flush=True)
t0 = time.time()
x, info = pcg(A, b, M, rtol=args.rtol, callback=lambda k, r: print(f"  - {r:.6e}", flush=True))
emit("t_solve", round(time.time() - t0, 4))
emit("iterations", info["iterations"])
emit("converged", str(info["converged"]).lower())
emit("lmin", f"{info['lmin']:.6f}")
emit("lmax", f"{info['lmax']:.6f}")
emit("cond", f"{info['cond']:.6f}")
