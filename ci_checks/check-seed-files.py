#!/usr/bin/env python3
"""A task that allows more than one seed and builds or stages a seed's instance itself (`[metadata.seeds] build =
"runtime"`, a list of seeds; docs/RUNNING.md "Seeds") has every file that seed needs, so that no run of an allowed seed
falls back to another seed's files (a Dockerfile COPY of a pattern that matches nothing does not fail a build):
  - every place the task stages from (scripts/task_seed.py places(): privileged `environment/` and `tests/`, standard
    `environment/sim/` and `tests/`) holds a file `seed` with the default seed, which scripts/task_seed.py materialize
    overwrites with the trial's seed;
  - every place with a `runtime_binding.json` has `runtime_binding.s<S>.json` for every allowed seed S other than the
    default;
  - a task with goal images (`environment/goal/`) has, for every allowed seed other than the default, that seed's
    images with the same file names under `tests/seeds/<S>/goal/` (its own, or its privileged twin's);
  - a task whose reference solution is a trajectory with a birth stamp (`solution/trajectory.npz` holding
    instance_sha.npy) has `solution/trajectory.s<S>.npz` for every seed of `[metadata.seeds] oracle` other than the
    default (a reference without a stamp serves every seed as it is);
  - its images stage the seed: privileged `environment/Dockerfile` and `tests/Dockerfile` copy every
    `runtime_binding.s<S>.json` and the `seed` file by name and run stage_instance.py with --seed-file; the standard sim
    service's entrypoint passes `--seed-file /srv/seed`, and the standard grader's `tests/Dockerfile` copies the
    bindings and picks the seed's; no COPY source is a pattern (a pattern that matches nothing does not fail a build).
The IKEA MuJoCo tasks (roboarc-ikea-*) have one frozen scene for every seed, and a seed changes only the standard mode's
Nyx lighting: they have the `seed` files but no per-seed bindings or staging; instead the standard sim service's
`environment/sim/rig.json` has the lighting of every allowed seed other than the default (`seeds`), which server.py
picks by its `seed` file.
Tasks that allow one seed only (every task until RoboARC 1.2.0), and frozen-instance seeds (ci_checks/check-seeds.py),
are not concerned. Run by the CI job `unit-tests` (a failure fails it) and by run_all.sh.

    python3 ci_checks/check-seed-files.py [task dirs]      # default: every task
Standard library only.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IKEA = "roboarc-ikea-"   # one frozen scene for every seed; the seed is the standard mode's lighting (module docstring)
sys.path.insert(0, str(ROOT / "scripts"))
import task_seed  # noqa: E402


def check(task: Path) -> list[str]:
    try:
        sd = task_seed.seeds(task)
    except Exception as e:          # check-seeds.py reports a malformed table
        return [f"[metadata.seeds] unreadable: {e}"]
    allowed = sd["allowed"]
    if sd["build"] != "runtime" or not isinstance(allowed, list) or len(allowed) < 2:
        return []
    errs = []
    others = [s for s in allowed if s != sd["default"]]
    ikea = task.name.startswith(IKEA)
    for place in task_seed.places(task):
        d = task / place
        if not d.is_dir():
            continue
        f = d / "seed"
        if not f.is_file():
            errs.append(f"{place}/seed missing (the default seed, overwritten by scripts/task_seed.py materialize)")
        elif f.read_text().strip() != str(sd["default"]):
            errs.append(f"{place}/seed is {f.read_text().strip()!r}, not the default seed {sd['default']}")
        if (d / "runtime_binding.json").is_file() and not ikea:
            for s in others:
                if not (d / f"runtime_binding.s{s}.json").is_file():
                    errs.append(f"{place}/runtime_binding.s{s}.json missing (allowed seed {s})")
    goal = task / task_seed.GOAL
    if goal.is_dir():
        names = sorted(p.name for p in goal.iterdir())
        for s in others:
            g = task_seed.goal_source(task, s)
            if g is None:
                errs.append(f"no goal images of seed {s} (tests/seeds/{s}/goal/ of the task or its twin)")
            elif sorted(p.name for p in g.iterdir()) != names:
                errs.append(f"{g.relative_to(task.parent)}: not the files of {task_seed.GOAL}/ ({names})")
    ref = task / "solution" / "trajectory.npz"
    if ref.is_file() and "instance_sha.npy" in zipfile.ZipFile(ref).namelist():   # a reference without a birth stamp serves every seed
        for s in sd["oracle"]:
            if s != sd["default"] and not (task / "solution" / f"trajectory.s{s}.npz").is_file():
                errs.append(f"solution/trajectory.s{s}.npz missing (seed {s} is in seeds.oracle)")
    return errs + (ikea_lighting(task, others) if ikea else staging(task, others))


def ikea_lighting(task: Path, others: list[int]) -> list[str]:
    """An IKEA standard task's rig.json has the lighting of every other allowed seed (module docstring)."""
    rig = task / "environment" / "sim" / "rig.json"
    if not rig.is_file():                 # the privileged twin: no images
        return []
    seeds = json.loads(rig.read_text()).get("seeds", {})
    return [f"environment/sim/rig.json: no lighting of seed {s} (seeds)" for s in others if str(s) not in seeds]


def read(p: Path) -> str:
    return p.read_text() if p.is_file() else ""


def staging(task: Path, others: list[int]) -> list[str]:
    """The images stage the trial's seed (module docstring)."""
    errs = []
    names = [f"runtime_binding.s{s}.json" for s in others] + ["seed"]
    sidecar = task_seed.places(task)[0] != "environment"
    dfs = ["tests/Dockerfile"] if sidecar else ["environment/Dockerfile", "tests/Dockerfile"]
    for rel in dfs:
        text = read(task / rel)
        copies = [l for l in text.splitlines() if l.lstrip().upper().startswith("COPY")]
        for l in copies:
            if any(c in l for c in "*?["):
                errs.append(f"{rel}: COPY of a pattern ({l.strip()}): name the files")
        named = " ".join(copies).split()
        for n in names:
            if n not in named:
                errs.append(f"{rel}: does not copy {n}")
        if not sidecar and "--seed-file" not in text:
            errs.append(f"{rel}: does not stage with --seed-file (stage_instance.py)")
    if sidecar:
        if "b=runtime_binding.s$s.json" not in read(task / "tests/Dockerfile"):
            errs.append("tests/Dockerfile: does not pick the seed's runtime_binding.s<S>.json")
        if "--seed-file /srv/seed" not in read(task / "environment/docker-compose.yaml"):
            errs.append("environment/docker-compose.yaml: the sim service does not stage with --seed-file /srv/seed")
    return errs


def main(argv: list[str]) -> int:
    dirs = [Path(a) for a in argv] or sorted(p for p in (ROOT / "tasks").iterdir() if (p / "task.toml").is_file())
    bad = n = 0
    for t in dirs:
        if not (t / "task.toml").is_file():
            continue
        errs = check(t)
        n += 1
        for e in errs:
            print(f"FAIL {t}: {e}")
        bad += len(errs)
    if bad:
        return 1
    print(f"check-seed-files: {n} task(s); every allowed seed has its seed file, bindings, goal images and oracle")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
