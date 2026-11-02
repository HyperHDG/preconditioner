"""examples/hyperhdg_net_q1.py against the golden histories of HyperHDG's cython-era reproducibles
(tests/record_reproducibles.py), on the network file each golden names. Compiles HyperHDG's module on
first use, minutes, so it only runs with NETPRECOND_HYPERHDG=1."""
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

from support import ROOT, golden

pytestmark = pytest.mark.skipif(os.environ.get("NETPRECOND_HYPERHDG") != "1",
                                reason="set NETPRECOND_HYPERHDG=1 (compiles HyperHDG modules)")


def run_example(network, problem, degree, H, subdomains="node", flatten=False):
    out = subprocess.run([sys.executable, str(ROOT / "examples" / "hyperhdg_net_q1.py"), "--domain", str(ROOT / network),
                          "--problem", problem, "--degree", str(degree), "-H", str(H), "--subdomains", subdomains]
                         + (["--flatten"] if flatten else []), capture_output=True, text=True, check=True).stdout
    return [float(m.group(1)) for m in re.finditer(r"^  - (\S+)$", out, re.M)]


@pytest.mark.parametrize("name", ["reproducible_diffusion_14871_H8_P1", "reproducible_diffusion_14871_H8_P6",
                                  "reproducible_timoshenko_14871_H8_P5"])
def test_against_reproducible(name):
    g = golden(name)
    s = g["settings"]
    h = run_example(g["network"], s["problem"], s["degree"], s["H"], s.get("subdomains", "node"), s.get("flatten", False))
    assert len(h) == g["iterations"], (len(h), g["iterations"])
    # the same CG trajectory up to rounding: six digits at most iterations, but the residual norm
    # of a rounded CG run deviates transiently where the curve wiggles (up to 13 % at iterations
    # 64-70 and 80-81 of the Timoshenko run, re-synchronising to 1e-6 afterwards); a different
    # operator would drift and not come back
    dev = np.abs(np.asarray(h) / np.asarray(g["residuals"]) - 1)
    assert np.median(dev) < 1e-3, np.median(dev)
    assert dev.max() < 0.2, (dev.argmax() + 1, dev.max())
