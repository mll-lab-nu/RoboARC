"""The task catalogue: every RoboARC puzzle base and every IKEA MuJoCo task, read from the repository's task dirs (no
simulator needed). Standard library only.

Canonical ids: `roboarc-<family>-<NN>-i01` (puzzles, e.g. roboarc-orbits-02-i01) and `roboarc-ikea-<name>-i00` (IKEA,
e.g. roboarc-ikea-lack-i00). resolve() also accepts the Harbor names (`...-privileged`, `...-standard`, with or without
the `roboarc/` prefix, or 1.0.0's `robot-coding-bench/`), the bases without `roboarc-` (orbits-02-i01, orbits-02) and the IKEA names (ikea-lack,
lack).
"""
from __future__ import annotations

import json
import re
import tomllib
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from ._runtime import repo_root

SUITES = ("puzzles", "ikea")


@dataclass(frozen=True)
class TaskInfo:
    """Static facts of one task (base): everything that needs no simulator."""
    id: str                         # canonical id
    suite: str                      # "puzzles" | "ikea"
    family: str                     # puzzles: the family (orbits, blanket_flatten, ...); IKEA: ikea_<name>
    name: str                       # the tooling's task key: puzzles: the Session family (orbits); IKEA: the scene (lack)
    base: str                       # puzzles: <family>-<NN>-i01; IKEA: <name>
    difficulty: str | None          # puzzles: very_easy .. hard; IKEA: None
    robot: str                      # aloha-2 | franka-robotiq-2f85 | franka-panda-bimanual
    description: str                # the task's one-line description (task.toml)
    image: str                      # the image the physics runs in
    render_image: str               # the image the standard mode renders in (IKEA: the -sim image with Nyx)
    max_episode_steps: int          # the trajectory row cap the verifier enforces (control steps)
    agent_time_limit_s: float       # the Harbor agent budget (wall clock) of both modes
    task_dirs: dict = field(default_factory=dict)   # mode -> task directory (privileged, standard)
    reference: str | None = None    # the reference trajectory (.npz) of the privileged mode, if any
    reference_end_hash: str | None = None   # the state hash the reference ends on (recorded; prefix for IKEA)
    reference_settled_hash: str | None = None   # IKEA: after the grader's 1 s hold of the last row (recorded)
    seeds: tuple = (0,)             # the instance seeds the task allows ([metadata.seeds] allowed; 0 = the default)

    def to_dict(self) -> dict:
        return asdict(self)


def _toml(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def _image_of(dockerfile: Path) -> str:
    for line in dockerfile.read_text().splitlines():
        if line.startswith("FROM "):
            return line.split()[1]
    raise ValueError(f"no FROM line in {dockerfile}")


def _env_value(dockerfile: Path, key: str) -> str | None:
    for line in dockerfile.read_text().splitlines():
        if line.startswith("ENV "):
            for kv in line.split()[1:]:
                if kv.startswith(key + "="):
                    return kv.split("=", 1)[1]
    return None


def _seeds(md: dict) -> tuple:
    """[metadata.seeds] allowed as a tuple of ints (docs/PROTOCOL.md §2.13; RoboARC lists them), else (default,)."""
    given = md.get("seeds") if isinstance(md.get("seeds"), dict) else {}
    default = int(given.get("default", md.get("instance_seed", 0)))
    allowed = given.get("allowed", [default])
    return tuple(int(x) for x in allowed) if isinstance(allowed, list) else (default,)


def reference_pins(root: Path) -> dict[str, dict]:
    """Per puzzle base with a pinned reference: {"sha256", "expect"} (the privileged reference file's sha256 and its
    recorded end hash). The V3 freeze manifest pins the bases frozen in V3; a base released later, or one whose reference
    a later release re-stamped (2.1.0: the blanket bases' new terminal revision), is pinned by its release record's
    privileged evidence (scripts/roboarc/release.json: trajectory_sha256 and end_state, both given). Only the bases of
    the release are pinned: the manifest is a record and keeps the bases removed since (2.0.0)."""
    path = root / "records" / "freeze_manifests" / "freeze_manifest.v3.json"
    rel = root / "scripts" / "roboarc" / "release.json"
    released = json.loads(rel.read_text())["bases"] if rel.is_file() else {}
    pins = {b: {"sha256": r.get("sha256"), "expect": r.get("expect")}
            for b, r in (json.loads(path.read_text())["bases"] if path.is_file() else {}).items()
            if r.get("reference") and (b in released or not rel.is_file())}
    for b, r in released.items():
        ev = (r.get("evidence") or {}).get("privileged") or {}
        if ev.get("trajectory_sha256") and ev.get("end_state"):
            pins[b] = {"sha256": ev["trajectory_sha256"], "expect": ev["end_state"]}
    return pins


def _puzzle_tasks(root: Path) -> dict[str, TaskInfo]:
    pins = reference_pins(root)
    out = {}
    for d in sorted((root / "tasks").glob("roboarc-*-privileged")):
        if d.name.startswith("roboarc-ikea-"):
            continue
        base = d.name[len("roboarc-"):-len("-privileged")]
        meta = _toml(d / "task.toml")
        md = meta.get("metadata", {})
        test = (d / "tests" / "test.sh").read_text()
        name = re.search(r"--family (\S+)", test).group(1)
        cap = _env_value(d / "tests" / "Dockerfile", "RCB_MAX_STEPS")
        std = d.parent / d.name.replace("-privileged", "-standard")
        ref = root / "scripts" / "roboarc" / "oracles" / base / "privileged.npz"
        expect = (pins.get(base) or {}).get("expect")
        image = _image_of(d / "environment" / "Dockerfile")
        out[f"roboarc-{base}"] = TaskInfo(
            id=f"roboarc-{base}", suite="puzzles", family=md.get("family", name), name=name, base=base,
            difficulty=md.get("difficulty"), robot=md.get("robot", ""), description=meta["task"]["description"],
            image=image, render_image=image, max_episode_steps=int(cap) if cap else 200_000,
            agent_time_limit_s=float(meta.get("agent", {}).get("timeout_sec", 11400.0)),
            task_dirs={"privileged": str(d), **({"standard": str(std)} if std.is_dir() else {})},
            reference=str(ref) if ref.is_file() else None, reference_end_hash=expect, seeds=_seeds(md))
    return out


@lru_cache(maxsize=1)
def _ikea_hashes() -> dict:
    return json.loads((Path(__file__).resolve().parent / "data" / "ikea_reference_hashes.json").read_text())["tasks"]


def _ikea_tasks(root: Path) -> dict[str, TaskInfo]:
    out = {}
    hashes = _ikea_hashes()
    folder = root / "tasks"
    dirs = sorted(folder.glob("roboarc-ikea-*-i00-privileged")) or sorted(folder.glob("ikea-*-i00-privileged"))
    for d in dirs:
        stem = d.name[:-len("-privileged")]
        name = re.search(r"ikea-(.+)-i00$", stem).group(1)
        meta = _toml(d / "task.toml")
        md = meta.get("metadata", {})
        test = (d / "tests" / "test.sh").read_text()
        rows = re.search(r"--max-rows (\d+)", test)
        std = d.parent / d.name.replace("-privileged", "-standard")
        ref = d / "solution" / "trajectory.npz"
        sim_image = _image_of(std / "environment" / "Dockerfile") if (std / "environment" / "Dockerfile").is_file() else None
        compose = std / "environment" / "docker-compose.yaml"
        if compose.is_file():
            m = re.search(r"image:\s*(\S*roboarc-ikea-sim\S*|\S*ikea-mujoco-sim\S*)", compose.read_text())
            sim_image = m.group(1) if m else sim_image
        h = hashes.get(name) or {}
        image = _image_of(d / "environment" / "Dockerfile")
        out[f"roboarc-ikea-{name}-i00"] = TaskInfo(
            id=f"roboarc-ikea-{name}-i00", suite="ikea", family=md.get("family", f"ikea_{name}"), name=name, base=name,
            difficulty=md.get("difficulty"), robot=md.get("robot", ""), description=meta["task"]["description"],
            image=image, render_image=sim_image or image, max_episode_steps=int(rows.group(1)) if rows else 50_000_000,
            agent_time_limit_s=float(meta.get("agent", {}).get("timeout_sec", 11400.0)),
            task_dirs={"privileged": str(d), **({"standard": str(std)} if std.is_dir() else {})},
            reference=str(ref) if ref.is_file() else None, reference_end_hash=h.get("end"),
            reference_settled_hash=h.get("settled"), seeds=_seeds(md))
    return out


@lru_cache(maxsize=4)
def _catalogue(root: str) -> dict[str, TaskInfo]:
    r = Path(root)
    return {**_puzzle_tasks(r), **_ikea_tasks(r)}


def catalogue() -> dict[str, TaskInfo]:
    return _catalogue(str(repo_root()))


def list_tasks(suite: str | None = None, family: str | None = None) -> list[TaskInfo]:
    """Every task (base) of the release, optionally of one suite ("puzzles", "ikea") or family."""
    if suite is not None and suite not in SUITES:
        raise ValueError(f"suite must be one of {SUITES}")
    return [t for t in catalogue().values()
            if (suite is None or t.suite == suite) and (family is None or family in (t.family, t.family.replace("_", "-")))]


def resolve(task_id: str) -> TaskInfo:
    """The TaskInfo of a task id or alias (see the module docstring); KeyError with suggestions otherwise."""
    cat = catalogue()
    s = str(task_id).strip()
    s = s.split("/", 1)[1] if s.startswith(("roboarc/", "robot-coding-bench/")) else s   # Harbor names, 1.0.1 and 1.0.0
    for suffix in ("-privileged", "-standard"):
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    if s.startswith("roboarc-"):
        s = s[len("roboarc-"):]
    if s.startswith("ikea-"):
        s = s[len("ikea-"):]
        s = s[:-len("-i00")] if s.endswith("-i00") else s
        key = f"roboarc-ikea-{s}-i00"
    elif f"roboarc-ikea-{s}-i00" in cat:
        key = f"roboarc-ikea-{s}-i00"
    else:
        key = f"roboarc-{s}" if s.endswith("-i01") else f"roboarc-{s}-i01"
    if key in cat:
        return cat[key]
    near = [k for k in cat if s.split("-")[0] in k][:8]
    raise KeyError(f"unknown RoboARC task {task_id!r}" + (f"; similar: {near}" if near else "") +
                   " (roboarc.list_tasks() lists them all)")


def task_info(task_id: str) -> TaskInfo:
    return resolve(task_id)
