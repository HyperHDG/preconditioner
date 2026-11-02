"""Generation 3 against generation 1: net2as in q1 mode through examples/laplacian.cxx on
the grid .geo.h5 the reference tests use.

With -net2as_cb_trim (coarse DoFs on the box boundary removed) net2as reproduces the
reference's CG history to rounding. Without it the coarse space keeps the boundary hats,
which the reference never has, and takes more iterations; that default is pinned too."""
import json
import shutil
import subprocess

import numpy as np
import pytest

from support import ROOT, golden

LAPLACIAN = ROOT / "build" / "laplacian"
pytestmark = pytest.mark.skipif(not LAPLACIAN.exists(), reason="build/laplacian not built (cmake -B build && cmake --build build)")


def run_laplacian(domain, *opts):
    out = subprocess.run([str(LAPLACIAN), "-domain", str(domain), "-net2as_cb_type", "q1", "-net2as_p", "3", *opts],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(next(l for l in out.splitlines() if l.startswith("{")))


def test_net2as_q1_trimmed_matches_reference(grid33):
    d = run_laplacian(grid33, "-net2as_cb_trim")
    g = golden("q1_grid33_H4_sin")           # recorded from gortz_hellman_malqvist_22
    assert d["iterations"] == g["iterations"]
    np.testing.assert_allclose(d["residuals"], g["residuals"], rtol=1e-5, atol=1e-13)


def test_net2as_q1_default_keeps_boundary_hats(grid33):
    d = run_laplacian(grid33)
    g = golden("q1_grid33_H4_sin_net2as_untrimmed")
    assert d["iterations"] == g["iterations"]
    np.testing.assert_allclose(d["residuals"], g["residuals"], rtol=1e-6, atol=1e-13)
