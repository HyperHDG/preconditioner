import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # the package is installed (pip install -e .), not path-hacked

GOLDEN = ROOT / "tests" / "golden"


def golden(name):
    """The recorded numbers for a test problem, tests/golden/<name>.json; a missing file
    fails the test with the values it would have to contain."""
    path = GOLDEN / f"{name}.json"
    if not path.exists():
        raise AssertionError(f"no golden file {path.relative_to(ROOT)}; record it from a trusted state")
    return json.loads(path.read_text())
