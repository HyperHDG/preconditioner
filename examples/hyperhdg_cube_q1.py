#!/usr/bin/env python3
"""The q1 preconditioner on plain HDG diffusion on the unit square (HyperHDG's Cubic<2,2>
topology, n x n cells, the constant-coefficient test problem of its own test suite, Dirichlet on
the whole boundary): the hypernodes are the cell faces, located at their barycenters, each
carrying degree+1 dofs; the q1 decomposition over those points is expanded to the dofs, then
AdditiveSchwarz, PCG and the Lanczos condition estimate. YAML on stdout.

    python examples/hyperhdg_cube_q1.py -n 32 -H 4
    python examples/hyperhdg_cube_q1.py -n 64 --degree 2 -H 8

A sanity check that the method is not tied to networks; HyperHDG compiles the module for a
degree on first use and caches it under .cache/hyperhdg/.
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import scipy.sparse as sp

from netprecond import AdditiveSchwarz, expand, pcg, q1, restrict

ROOT = pathlib.Path(__file__).resolve().parents[1]
HYPERHDG = ROOT / "submodules" / "HyperHDG"
sys.path.insert(0, str(HYPERHDG / "python"))   # hyperhdg.py is a script next to HyperHDG's examples, not a package yet
import hyperhdg  # noqa: E402

MODULE = """\
#include <HyperHDG/bind_python.hxx>

#include <HyperHDG/geometry/unit_cube.hxx>
#include <HyperHDG/global_loop/elliptic.hxx>
#include <HyperHDG/local_solver/diffusion_ldgh.hxx>
#include <HyperHDG/node_descriptor/cubic.hxx>
#include <HyperHDG/topology/cubic.hxx>

#include "reproducibles_python/parameters/diffusion.hxx"

using Topo = Topology::Cubic<2, 2>;
using Geom = Geometry::UnitCube<2, 2, double>;
using Diffusion = GlobalLoop::Elliptic<
  Topo, Geom, NodeDescriptor::Cubic<2, 2>,
  LocalSolver::Diffusion<2, {degree}, {quadrature}, TestParametersSinEllipt, double> >;

// barycenter of every hypernode (= cell face), hypernode-major, flat (x0, y0, x1, y1, ...)
std::vector<double> hypernode_points(const std::vector<unsigned int>& n)
{{
  Topo topo(n);
  Geom geom(topo);
  std::vector<double> pts(2 * topo.n_hyNodes(), 0.);
  for (unsigned int e = 0; e < topo.n_hyEdges(); ++e)
  {{
    const auto& nodes = topo[e].get_hyNode_indices();
    auto edge = geom[e];
    for (unsigned int i = 0; i < nodes.size(); ++i)
    {{
      auto bary = edge.face_barycenter(i);
      pts[2 * nodes[i]] = bary[0];
      pts[2 * nodes[i] + 1] = bary[1];
    }}
  }}
  return pts;
}}

NB_MODULE({name}, m) {{
  HyperHDG::bind_python<Diffusion, std::vector<unsigned int> >(m, "Diffusion");
  m.def("hypernode_points", &hypernode_points);
}}
"""


def code(degree):
    return MODULE.format(degree=degree, quadrature=2 * degree, name=f"netprecond_cube_D2_P{degree}")


def emit(key, value):
    print(f"{key}: {value}", flush=True)


p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("-n", type=int, default=16, help="cells per axis of the unit square")
p.add_argument("--degree", type=int, default=1, help="polynomial degree of the local solver")
p.add_argument("--tau", type=float, default=1.0, help="HDG penalty parameter")
p.add_argument("-H", type=int, default=4, help="coarse cells per axis")
p.add_argument("--rtol", type=float, default=1e-10)
p.add_argument("--label", default=None, help="free text copied into the output")
args = p.parse_args()

module = hyperhdg.load(code(args.degree), hyperhdg=HYPERHDG, cache_dir=ROOT / ".cache" / "hyperhdg")
hdg = module.Diffusion([args.n, args.n], args.tau)
points = np.asarray(module.hypernode_points([args.n, args.n])).reshape(-1, 2)

t0 = time.time()
size = hdg.size_of_system()
rows, cols, vals = hdg.trace_to_flux_mat()
A = sp.csr_matrix((vals, (rows, cols)), shape=(size, size))
b = -np.asarray(hdg.residual_flux(hdg.zero_vector()))
block_size = size // len(points)
assert block_size * len(points) == size, "dofs are not a multiple of the hypernodes"
if args.label is not None:
    emit("label", args.label)
emit("n", args.n)
emit("degree", args.degree)
emit("H", args.H)
emit("hypernodes", len(points))
emit("block_size", block_size)
emit("dofs", size)
emit("t_assemble", round(time.time() - t0, 4))

t0 = time.time()
dec = expand(q1(points, (args.H, args.H)), block_size)       # dofs: hypernode * block_size + local dof
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
emit("error", f"{hdg.errors(list(x))[0]:.3e}")
