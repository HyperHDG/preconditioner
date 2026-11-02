#!/usr/bin/env python3
"""Turn the stdout of HyperHDG's cython-era reproducibles into golden files.

reproducibles_python/fiber_network_{diffusion,elastic}.py are the scripts examples/hyperhdg_q1.py
replaces; they leave HyperHDG (PR-C2), so their CG histories are kept here as
tests/golden/reproducible_*.json with the provenance inside, and tests/test_hyperhdg.py compares
the example against them on the identical file tests/data/fiber_network_14871.geo. The scripts
were run on pde22 from HyperHDG at the h5-io tip a5891e3f, unmodified except for the domain
name (one of the three the script offers) and scipy's tol -> rtol rename; the diffusion script
loops over degrees 1 and 6, the elastic one is degree 5, both 8 x 8 coarse cells, rtol 1e-10.

    python tests/record_reproducibles.py diffusion-14871.log elastic-14871.log
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
NETWORK = "tests/data/fiber_network_14871.geo"
SOURCE = ("HyperHDG reproducibles_python/{script} at a5891e3f (h5-io), cython interface, run on pde22; "
          "patched only in the domain name and scipy's tol -> rtol")


def histories(text):
    """The residual history of every CG run in a log: lines '<it> \\t <residual> \\t <energy> \\t <time>',
    runs separated by the start of a new iteration count 1."""
    runs, current = [], []
    for m in re.finditer(r"^(\d+)\s+\t\s+(\S+)\s+\t", text, re.M):
        it, res = int(m.group(1)), float(m.group(2))
        if it == 1 and current:
            runs.append(current)
            current = []
        current.append(res)
    if current:
        runs.append(current)
    return runs


def write(name, script, settings, residuals):
    doc = {"source": SOURCE.format(script=script), "network": NETWORK, "settings": settings,
           "iterations": len(residuals), "residuals": residuals}
    path = ROOT / "tests" / "golden" / f"{name}.json"
    path.write_text(json.dumps(doc, indent=1))
    print(f"{path.relative_to(ROOT)}: {len(residuals)} iterations, final {residuals[-1]:.3e}")


for log in sys.argv[1:]:
    text = pathlib.Path(log).read_text()
    if "diffusion" in log:
        degrees = [int(d) for d in re.findall(r"Polynomial degree is set to be\s+(\d+)", text)]
        runs = histories(text)
        assert len(runs) == len(degrees), (len(runs), degrees)
        for degree, res in zip(degrees, runs):
            write(f"reproducible_diffusion_14871_H8_P{degree}", "fiber_network_diffusion.py",
                  {"problem": "diffusion", "degree": degree, "H": 8, "tau": 1.0, "rtol": 1e-10}, res)
    else:
        (res,) = histories(text)
        write("reproducible_timoshenko_14871_H8_P5", "fiber_network_elastic.py",
              {"problem": "timoshenko", "degree": 5, "H": 8, "tau": 1.0, "rtol": 1e-10}, res)
