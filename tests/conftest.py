import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

GOLDEN = ROOT / "tests" / "golden"


def golden(name):
    """The recorded numbers for a test problem, tests/golden/<name>.json; a missing file
    fails the test with the values it would have to contain."""
    path = GOLDEN / f"{name}.json"
    if not path.exists():
        raise AssertionError(f"no golden file {path.relative_to(ROOT)}; record it from a trusted state")
    return json.loads(path.read_text())
