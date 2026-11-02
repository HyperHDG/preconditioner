#!/usr/bin/env python3
"""The old reproducibles against examples/hyperhdg_q1.py, on the same networks.

HyperHDG's reproducibles_python/fiber_network_{diffusion,elastic}.py (the cython-era scripts
the example replaces) are run from the pinned submodule, unmodified except for the domain
name, the degree loop and, for the beams, the coarse grid: the reference preconditioner builds
dense patch matrices, so Timoshenko runs on the 1141-node mikado network with 4 x 4 coarse
cells. Their printed CG residual histories are compared with the example's on the identical
network. Run by hand (it compiles both HyperHDG interfaces, minutes):

    python tests/verify_reproducibles.py                 # diffusion
    python tests/verify_reproducibles.py --timoshenko    # also the beams: the old script's
                                                         # preconditioner takes many minutes
"""
import os
import pathlib
import re
import subprocess
import sys

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
HYPERHDG = ROOT / "submodules" / "HyperHDG"
MAKE_GEO = [sys.executable, str(ROOT / "submodules" / "fiber_network" / "fiber_network" / "make_geo.py")]
WORK = ROOT / ".cache" / "verify"
DOMAINS, OLD = WORK / "domains", WORK / "reproducibles_python"
ENV = dict(os.environ, PYTHONPATH=f"{HYPERHDG / 'import'}:{ROOT / 'submodules' / 'fiber_network'}")


def write_geo_2d(h5, geo, points_txt):
    """A 2D .geo (the old diffusion script instantiates a 2D topology) and the points file
    the old scripts read next to it, from a .geo.h5."""
    with h5py.File(h5) as f:
        points, edges, types = f["domain/points"][:, :2], f["domain/edges"][:], f["domain/types_faces"][:]
    with open(geo, "w") as g:
        g.write(f"Space_Dim     = 2;\nHyperEdge_Dim = 1;\n\nN_Points      = {len(points)};\n"
                f"N_HyperNodes  = {len(points)};\nN_HyperEdges  = {len(edges)};\n\nPOINTS:\n")
        np.savetxt(g, points, fmt="%.17g")
        g.write("\nHYPERNODES_OF_HYPEREDGES:\n"); np.savetxt(g, edges, fmt="%d")
        g.write("\nTYPES_OF_HYPERFACES:\n"); np.savetxt(g, types, fmt="%d")
        g.write("\nPOINTS_OF_HYPEREDGES:\n"); np.savetxt(g, edges, fmt="%d")
    np.savetxt(points_txt, points)


def old_script(name, domain, patches):
    src = (HYPERHDG / "reproducibles_python" / name).read_text()
    src = src.replace('domain = "fiber_network_615452"', f'domain = "{domain}"')
    for old, new in patches:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    path = OLD / name
    path.write_text(src)
    return path


def run_old(script):
    out = subprocess.run([sys.executable, str(script)], cwd=WORK, env=ENV, capture_output=True, text=True)
    if out.returncode:
        sys.exit(f"{script.name} failed:\n{out.stderr[-3000:]}")
    # the callback prints "<iteration> \t <relative residual> \t <energy> \t <time>"
    return [float(m.group(1)) for m in re.finditer(r"^\d+\s+\t\s+(\S+)\s+\t", out.stdout, re.M)]


def run_new(domain, problem, degree, H):
    out = subprocess.run([sys.executable, str(ROOT / "examples" / "hyperhdg_q1.py"), "--domain", str(domain),
                          "--problem", problem, "--degree", str(degree), "-H", str(H)],
                         capture_output=True, text=True, env=dict(os.environ))
    if out.returncode:
        sys.exit(f"hyperhdg_q1.py failed:\n{out.stderr[-3000:]}")
    return [float(m.group(1)) for m in re.finditer(r"^  - (\S+)$", out.stdout, re.M)]


def compare(label, old, new):
    same = len(old) == len(new) and np.allclose(old, new, rtol=1e-3, atol=1e-12)
    print(f"{label}:\n  old: {len(old)} iterations, residuals {old[:3]} ... {old[-1]:.3e}\n"
          f"  new: {len(new)} iterations, residuals {new[:3]} ... {new[-1]:.3e}\n  match: {same}", flush=True)
    return same


for d in (DOMAINS, OLD, WORK / "output"):
    d.mkdir(parents=True, exist_ok=True)
ok = True

# diffusion: the 33-grid, 8 x 8 coarse grid as in the old script, degree 1
grid = ROOT / ".cache" / "grid33.geo.h5"
if not grid.exists():
    subprocess.run(MAKE_GEO + ["--grid", "33", "-t", "1e-6", "--dirichlet", "xmin", "xmax", "ymin", "ymax", "-o", str(grid)], check=True, capture_output=True)
write_geo_2d(grid, DOMAINS / "grid33.geo", DOMAINS / "grid33_points.txt")
script = old_script("fiber_network_diffusion.py", "grid33", [("for poly_degree in [1, 6]:", "for poly_degree in [1]:")])
ok &= compare("diffusion, grid33, H=8, degree 1", run_old(script), run_new(grid, "diffusion", 1, 8))

if "--timoshenko" not in sys.argv:
    print("ALL MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)

# Timoshenko beams: mikado 100 (1141 nodes), 4 x 4 coarse grid (the old script's 8 x 8 cannot
# be used, the empty hats make the coarse problem singular), degree 5 as in the old script
mik_h5, mik_geo = DOMAINS / "mikado100.geo.h5", DOMAINS / "mikado100.geo"
for out in (mik_h5, mik_geo):
    if not out.exists():
        subprocess.run(MAKE_GEO + ["--mikado", "100", "--seed", "1", "--dirichlet", "xmin", "xmax", "ymin", "ymax", "-o", str(out)], check=True, capture_output=True)
with h5py.File(mik_h5) as f:
    np.savetxt(DOMAINS / "mikado100_points.txt", f["domain/points"][:])
script = old_script("fiber_network_elastic.py", "mikado100", [("[2**3, 2**3], repeat=6", "[4, 4], repeat=6")])
ok &= compare("timoshenko, mikado100, H=4, degree 5", run_old(script), run_new(mik_h5, "timoshenko", 5, 4))

print("ALL MATCH" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
