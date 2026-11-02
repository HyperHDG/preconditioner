"""examples/hyperhdg_q1.py against the golden histories of HyperHDG's cython-era reproducibles
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


def run_example(network, problem, degree, H):
    out = subprocess.run([sys.executable, str(ROOT / "examples" / "hyperhdg_q1.py"), "--domain", str(ROOT / network),
                          "--problem", problem, "--degree", str(degree), "-H", str(H)],
                         capture_output=True, text=True, check=True).stdout
    return [float(m.group(1)) for m in re.finditer(r"^  - (\S+)$", out, re.M)]


@pytest.mark.parametrize("name", ["reproducible_diffusion_14871_H8_P1", "reproducible_diffusion_14871_H8_P6",
                                  "reproducible_timoshenko_14871_H8_P5"])
def test_against_reproducible(name):
    g = golden(name)
    s = g["settings"]
    h = run_example(g["network"], s["problem"], s["degree"], s["H"])
    assert len(h) == g["iterations"], (len(h), g["iterations"])
    np.testing.assert_allclose(h, g["residuals"], rtol=1e-3, atol=1e-12)
