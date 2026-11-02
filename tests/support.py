import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAKE_GEO = ROOT / "fiber_network" / "fiber_network" / "make_geo.py"   # pinned submodule
GOLDEN = ROOT / "tests" / "golden"


def golden(name):
    """The recorded numbers for a test problem, tests/golden/<name>.json; a missing file
    fails the test with the values it would have to contain."""
    path = GOLDEN / f"{name}.json"
    if not path.exists():
        raise AssertionError(f"no golden file {path.relative_to(ROOT)}; record it from a trusted state")
    return json.loads(path.read_text())


def make_grid(path, n):
    """An n x n grid on the unit square with Dirichlet on the whole boundary, as .geo.h5."""
    subprocess.run([sys.executable, str(MAKE_GEO), "--grid", str(n), "--dirichlet", "xmin", "xmax", "ymin", "ymax",
                    "-o", str(path)], check=True, capture_output=True, text=True)
    return path


@pytest.fixture(scope="session")
def grid33(tmp_path_factory):
    return make_grid(tmp_path_factory.mktemp("grid") / "grid33.geo.h5", 33)
