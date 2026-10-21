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
  - a task whose reference solution is a trajectory (`solution/trajectory.npz`) has `solution/trajectory.s<S>.npz`
    for every seed of `[metadata.seeds] oracle` other than the default.
Tasks that allow one seed only (every task until RoboARC 1.2.0), and frozen-instance seeds (ci_checks/check-seeds.py),
are not concerned. Run by the CI job `unit-tests` (a failure fails it) and by run_all.sh.

    python3 ci_checks/check-seed-files.py [task dirs]      # default: every task
Standard library only.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
    for place in task_seed.places(task):
        d = task / place
        if not d.is_dir():
            continue
        f = d / "seed"
        if not f.is_file():
            errs.append(f"{place}/seed missing (the default seed, overwritten by scripts/task_seed.py materialize)")
        elif f.read_text().strip() != str(sd["default"]):
            errs.append(f"{place}/seed is {f.read_text().strip()!r}, not the default seed {sd['default']}")
        if (d / "runtime_binding.json").is_file():
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
    if (task / "solution" / "trajectory.npz").is_file():
        for s in sd["oracle"]:
            if s != sd["default"] and not (task / "solution" / f"trajectory.s{s}.npz").is_file():
                errs.append(f"solution/trajectory.s{s}.npz missing (seed {s} is in seeds.oracle)")
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
