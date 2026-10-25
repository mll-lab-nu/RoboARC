"""Tests of the raw simulator interface (docs/API.md, "Tests").

  python -m pytest tests/api -m "not sim and not heavy"   # anywhere: catalogue, spaces, references (numpy + pytest)
  python -m pytest tests/api -m "sim and not heavy"       # inside an image: one task's env end to end (minutes)
  python -m pytest tests/api -m heavy                     # inside an image: the reference bit-identity replays

The sim tests build one environment per pytest process (the simulators are one scene per process): the puzzle image
runs ROBOARC_TEST_PUZZLE (default interlocked-claws-08), an IKEA image ROBOARC_TEST_IKEA (default svallet)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))     # the repository (the package's checkout)

import roboarc  # noqa: E402,F401  (before numpy: the graded numpy settings)
import pytest  # noqa: E402

from roboarc import _runtime  # noqa: E402


def sim_task() -> str | None:
    if _runtime.in_puzzle_image():
        return os.environ.get("ROBOARC_TEST_PUZZLE", "interlocked-claws-08")
    if _runtime.in_ikea_image():
        return os.environ.get("ROBOARC_TEST_IKEA", "svallet")
    return None


def pytest_collection_modifyitems(config, items):
    if sim_task() is None:
        skip = pytest.mark.skip(reason="needs a published RoboARC image (the simulator and the frozen instances)")
        for item in items:
            if "sim" in item.keywords or "heavy" in item.keywords:
                item.add_marker(skip)
