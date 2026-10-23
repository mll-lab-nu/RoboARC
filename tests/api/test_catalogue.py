"""The task catalogue and the reference records: no simulator needed."""
import json
from pathlib import Path

import numpy as np
import pytest

import roboarc
from roboarc.references import hash_matches, load_reference, reference_record

FAMILIES = {"blanket_flatten": 9, "blanket_fold": 5, "buzz_wire_3d": 10, "cast_coil": 8, "cast_keyhole": 6,
            "fiber_bundling": 10, "hedgehog_in_cage": 8, "horseshoe_ring": 5, "interlocked_claws": 12, "knot_untying": 8,
            "lock_picking": 5, "orbits": 2, "rope_threading": 8, "slotted_post_ring": 5}
IKEA = {"burvik", "eket", "gladom", "ikea365", "kyrre", "lack", "stegesund", "stenerik", "svallet", "uppsta", "variera",
        "vesken"}


def test_counts():
    puzzles = roboarc.list_tasks("puzzles")
    assert len(puzzles) == 101
    fams = {}
    for t in puzzles:
        fams[t.family] = fams.get(t.family, 0) + 1
    assert fams == FAMILIES
    assert {t.name for t in roboarc.list_tasks("ikea")} == IKEA
    assert len(roboarc.list_tasks()) == 113


@pytest.mark.parametrize("alias,canonical", [
    ("roboarc-orbits-02-i01", "roboarc-orbits-02-i01"), ("orbits-02-i01", "roboarc-orbits-02-i01"),
    ("orbits-02", "roboarc-orbits-02-i01"), ("roboarc-orbits-02-i01-standard", "roboarc-orbits-02-i01"),
    ("roboarc/roboarc-orbits-02-i01-privileged", "roboarc-orbits-02-i01"),
    ("robot-coding-bench/roboarc-orbits-02-i01-privileged", "roboarc-orbits-02-i01"),
    ("roboarc-ikea-lack-i00", "roboarc-ikea-lack-i00"), ("lack", "roboarc-ikea-lack-i00"),
    ("ikea-lack", "roboarc-ikea-lack-i00"), ("roboarc-ikea-svallet-i00-standard", "roboarc-ikea-svallet-i00")])
def test_aliases(alias, canonical):
    assert roboarc.resolve(alias).id == canonical


def test_unknown_task():
    with pytest.raises(KeyError):
        roboarc.resolve("no-such-task")


# every image by its manifest digest (images/pins.json)
PINS = {p["name"]: p["ref"] for p in json.loads((Path(__file__).resolve().parents[2] / "images/pins.json").read_text())["images"]}
assert PINS["roboarc"] == "ghcr.io/mll-lab-nu/roboarc:1.3.0@" + PINS["roboarc"].split("@")[1]


def test_metadata():
    for t in roboarc.list_tasks():
        assert t.max_episode_steps > 0 and t.agent_time_limit_s == 10800.0
        assert set(t.task_dirs) == {"privileged", "standard"}
        assert t.image.startswith("ghcr.io/mll-lab-nu/roboarc")
        json.dumps(t.to_dict())
        if t.suite == "puzzles":
            assert t.difficulty in ("very_easy", "easier", "easy", "medium", "hard")
            assert t.image == PINS["roboarc"]
        else:
            assert t.image == PINS["roboarc-ikea"] and t.render_image == PINS["roboarc-ikea-sim"]


def test_puzzle_references():
    """99 of the 101 bases ship a reference; each is the file the freeze manifest pins, with its recorded end hash."""
    with_ref = [t for t in roboarc.list_tasks("puzzles") if t.reference]
    assert len(with_ref) == 99
    for t in with_ref:
        rec = reference_record(t.id)
        assert rec["sha256_ok"], t.id
        assert t.reference_end_hash and len(t.reference_end_hash) == 16
    acts = load_reference("orbits-02")
    assert acts.dtype == np.float32 and acts.shape == (7918, 14)


def test_ikea_references():
    for t in roboarc.list_tasks("ikea"):
        assert t.reference is not None
    lack = load_reference("lack")
    assert lack.dtype == np.float64 and lack.shape == (67697, 16)
    assert roboarc.resolve("lack").reference_end_hash.startswith("684fe3c488e0")


def test_hash_prefixes():
    assert hash_matches("684fe3c488e01d1aabc", "684fe3c488e0")
    assert not hash_matches("684fe3c488e01d1a", "684fe3c488e1")
    assert not hash_matches("684fe3c488e01d1a", "684f")      # too short to mean anything
    assert hash_matches("abc", None) is None


def test_standard_references_obey_the_joint_speed_rule():
    """The standard references move an arm joint target at most 0.04 rad per control step (the standard rule; checked
    between consecutive rows). 12 privileged references are faster in places: their standard reference is the re-timed
    copy (blanket-flatten-06's: the same route re-recorded within the rule)."""
    over_std, over_priv = [], []
    for t in roboarc.list_tasks("puzzles"):
        if not t.reference:
            continue
        for mode, over in (("standard", over_std), ("privileged", over_priv)):
            rows = load_reference(t.id, mode).astype(np.float64)
            arm = [i for i in range(rows.shape[1]) if i not in (6, 13)] if rows.shape[1] == 14 else list(range(7))
            if np.abs(np.diff(rows[:, arm], axis=0)).max() > 0.04 + 1e-6:
                over.append(t.base)
    assert over_std == []
    assert len(over_priv) == 12
