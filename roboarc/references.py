"""The tasks' reference trajectories (the privileged mode's solutions) and the state hashes they are recorded to end on."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ._runtime import repo_root
from .registry import resolve


def load_reference(task_id, mode: str = "privileged") -> np.ndarray | None:
    """(N, action_dim) rows: float32 for the puzzles (scripts/roboarc/oracles/<base>/<mode>.npz, `actions`), float64 for
    IKEA (solution/trajectory.npz, `ctrl`; one reference for both modes). None when the task has no reference.
    mode="privileged": the privileged mode's solution (its verifier replays it with the tooling's limits only; the
    release records its end hash). mode="standard": the standard mode's reference, within the standard joint-speed rule
    (for 12 bases re-timed from the privileged one, otherwise the same file)."""
    if mode not in ("privileged", "standard"):
        raise ValueError("mode must be 'privileged' or 'standard'")
    t = resolve(task_id)
    if t.reference is None:
        return None
    path = t.reference.replace("privileged.npz", f"{mode}.npz") if t.suite == "puzzles" else t.reference
    with np.load(path, allow_pickle=False) as z:
        if t.suite == "puzzles":
            return np.asarray(z["actions"], dtype=np.float32)
        return np.ascontiguousarray(z["ctrl"], dtype=np.float64)


def reference_record(task_id) -> dict:
    """What the release records about the reference: path, sha256 (puzzles: checked against the freeze manifest),
    expected end hash (and IKEA's settled hash)."""
    t = resolve(task_id)
    out = {"task": t.id, "reference": t.reference, "end_hash": t.reference_end_hash, "settled_hash": t.reference_settled_hash}
    if t.reference and t.suite == "puzzles":
        m = json.loads((repo_root() / "records" / "freeze_manifests" / "freeze_manifest.v3.json").read_text())
        rec = m["bases"].get(t.base) or {}
        sha = hashlib.sha256(Path(t.reference).read_bytes()).hexdigest()
        out.update(sha256=sha, manifest_sha256=rec.get("sha256"), sha256_ok=sha == rec.get("sha256"))
    return out


def hash_matches(actual: str, expected: str | None) -> bool | None:
    """Recorded hashes are prefixes (16 hex digits for the puzzles' state_sha, 12-16 for IKEA's sha256)."""
    if expected is None:
        return None
    return len(expected) >= 12 and actual.startswith(expected)
