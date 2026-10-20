"""The bit-identity tests (marker heavy): each task's reference trajectory, stepped through roboarc.make() from
reset() in a fresh process (python -m roboarc replay), must end exactly on the state hash the release recorded for it
(the puzzles: the freeze manifest's `expect`; IKEA: the end and settled hashes of ikea_reference_hashes.json).

ROBOARC_BITID_TASKS (comma-separated) overrides the default selection: one base per puzzle family plus two IKEA scenes.
Only the tasks of the image this runs in are run. The blanket families need a GPU (their cloth runs on CUDA)."""
import json
import os
import subprocess
import sys

import pytest

from roboarc import _runtime, registry

pytestmark = pytest.mark.heavy

DEFAULT = ("blanket-flatten-08", "blanket-fold-01", "buzz-wire-3d-06", "cast-coil-01", "cast-keyhole-01",
           "fiber-bundling-01", "hedgehog-in-cage-01", "horseshoe-ring-05", "interlocked-claws-03", "knot-untying-03",
           "lock-picking-01", "orbits-01", "rope-threading-02", "slotted-post-ring-01", "lack", "svallet")
TASKS = [t for t in os.environ.get("ROBOARC_BITID_TASKS", ",".join(DEFAULT)).split(",") if t]


def _replay(info, *args):
    here = "puzzles" if _runtime.in_puzzle_image() else "ikea" if _runtime.in_ikea_image() else None
    if info.suite != here:
        pytest.skip(f"{info.id} runs in {info.image}")
    p = subprocess.run([sys.executable, "-m", "roboarc", "replay", info.id, "--renderer", "none", *args],
                       capture_output=True, text=True)
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    assert lines, p.stderr[-2000:]
    return json.loads(lines[-1])


@pytest.mark.parametrize("task", TASKS + ["fiber-bundling-07"])
def test_reference_end_hash(task):
    """The privileged reference with the privileged rules (what its verifier replays) ends on its recorded hash.
    fiber-bundling-07's reference moves faster than the standard joint-speed rule in 4 rows."""
    info = registry.resolve(task)
    out = _replay(info, "--mode", "privileged")
    assert out["match"] is True, out
    if info.suite == "ikea":
        assert out["settled_match"] is True and out["final_check_success"], out
    else:
        assert out["episode_end"] == "success", out


@pytest.mark.parametrize("task", ["fiber-bundling-07", "slotted-post-ring-01"])
def test_standard_reference_within_the_rule(task):
    """The standard mode's reference under the standard rules: no step is clamped (it was re-timed to the rule, or is
    the privileged file itself), the task succeeds; where it is the privileged file, it ends on the recorded hash."""
    info = registry.resolve(task)
    out = _replay(info, "--mode", "standard")
    assert out["speed_limited_steps"] == 0 and out["success"] and out["episode_end"] == "success", out
    assert out["match"] in (True, None), out
