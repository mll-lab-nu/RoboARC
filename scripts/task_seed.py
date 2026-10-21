#!/usr/bin/env python3
"""Run a task from one of its allowed seeds (docs/PROTOCOL.md §2, docs/RUNNING.md "Seeds"). The runner chooses the seed,
never the agent:

    python3 scripts/task_seed.py list tasks/<task>                                   # default, allowed, oracle, build
    python3 scripts/task_seed.py assign tasks/<task> --batch B --trials K            # the seeds of trials 1..K of a batch
    python3 scripts/task_seed.py pick tasks/<task> [--rng N]                         # one seed drawn uniformly from `allowed`
    python3 scripts/task_seed.py materialize tasks/<task> --seed S --out DIR         # -> DIR/<task>, run that with Harbor

`assign` is how an official batch gives its trials their seeds: the K trials of one task get distinct seeds, without
replacement, in an order fixed by the batch name and the task's base name (the name without its mode suffix), so the
privileged and standard twins of a base get the same seeds, and a rerun of a trial (the infrastructure retry) gets its
seed again. The order: the allowed seeds sorted by sha256("<batch>\\n<base>\\n<seed>") (ties by the seed); trial k
(1-based) runs the k-th of them; past the last one the order starts over (K larger than the number of seeds: every seed
runs floor(K/n) or ceil(K/n) times). A range `{from, to}` or "any" draws trial k's seed from sha256("<batch>\\n<base>\\n
#<k>.<j>") (j = 0, 1, ... until it differs from the earlier trials' seeds, while the range has unused seeds). Each
trial's seed alone is uniform over `allowed`; a task with one allowed seed runs it in every trial.

`materialize` copies the task directory and puts the seed in every place the task runs and grades from (`places()`:
privileged `environment/` and `tests/`; standard and standard-realtime `environment/sim/` and `tests/`):
- a seed with a committed frozen instance (`tests/seeds/<S>/instance.{json,npz}` of the task that holds the pair's
  instances: the privileged twin of a standard task, else the task itself) replaces the instance files there;
- a task with `build = "runtime"` gets a file `seed` (the integer) in each of those places, for every seed (the default
  too), from which it stages or builds that seed's instance itself; a committed `seed` file (the default seed) is
  overwritten;
- goal images: a task with `environment/goal/` gets, for a seed other than its default, that seed's images from
  `tests/seeds/<S>/goal/` (the task's own, else the holder's), which replace the default seed's images there;
  a non-default seed without them is refused.
The copy's `[metadata.seeds] default` (and `instance_seed`, where present) say the seed it runs from; its `tests/seeds/`
is removed, and every file gets its content-derived mtime (scripts/stamp_mtimes.py). A seed that is not allowed (a list,
a range `{ from, to }` or "any") is refused. `--seed draw` draws one uniformly (with `--rng`; "any" draws from 0 to
2**31 - 1). Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stamp_mtimes  # noqa: E402
import task_conventions as tc  # noqa: E402


def seeds(task: Path) -> dict:
    return tc.seeds_of(tomllib.loads((task / "task.toml").read_text()).get("metadata", {}))


def holder(task: Path) -> Path:
    twin = tc.twin_of(task)          # the mode registry's twin mode (PROTOCOL §3)
    return twin if twin is not None and twin.is_dir() else task


def places(task: Path) -> list[str]:
    """Where a task's instance, binding and seed copies are (the same mapping as ci_checks/check-instance-copies.sh):
    a standard task's simulator service and its verifier, a privileged task's environment and its verifier."""
    if tc.mode_of(task) in tc.SIDECAR_MODES:
        return ["environment/sim", "tests"]
    return ["environment", "tests"]


GOAL = "environment/goal"                        # the agent's goal images (/app/goal/), per seed where they show it


def goal_source(task: Path, seed: int) -> Path | None:
    """The committed goal images of `seed`: `tests/seeds/<S>/goal/` of the task, else of the holder; None if neither."""
    for t in (task, holder(task)):
        d = t / "tests" / "seeds" / str(seed) / "goal"
        if d.is_dir():
            return d
    return None


ANY = 2 ** 31                                   # "any": a seed is drawn from 0 .. ANY - 1


def _h(*parts) -> int:
    return int.from_bytes(hashlib.sha256("\n".join(str(p) for p in parts).encode()).digest(), "big")


def assign(task: Path, batch: str, trials: int) -> list[int]:
    """The seeds of trials 1..`trials` of `task` in batch `batch` (module docstring): distinct while the allowed seeds
    last, in an order fixed by the batch name and the task's base name (the twins share it)."""
    if trials < 1:
        raise ValueError("trials must be at least 1")
    base, a = tc.base_of(task), seeds(task)["allowed"]
    if isinstance(a, list):
        order = sorted(a, key=lambda s: (_h(batch, base, s), s))
        return [order[k % len(order)] for k in range(trials)]
    lo, size = (0, ANY) if a == "any" else (a["from"], a["to"] - a["from"] + 1)
    out: list[int] = []
    for k in range(1, trials + 1):
        cycle = out[len(out) - len(out) % size:]       # the seeds used in this pass over the range
        j = 0
        while True:
            s = lo + _h(batch, base, f"#{k}.{j}") % size
            if s not in cycle:
                break
            j += 1
        out.append(s)
    return out


def pick(task: Path, rng: int | None) -> int:
    a, r = seeds(task)["allowed"], random.Random(rng)
    if a == "any":
        return r.randrange(ANY)
    if isinstance(a, dict):
        return r.randint(a["from"], a["to"])
    return r.choice(a)


def materialize(task: Path, seed: int, out: Path) -> Path:
    sd = seeds(task)
    if not tc.seed_allowed(sd, seed):
        raise SystemExit(f"{task.name}: seed {seed} is not allowed ({sd['allowed']})")
    dst = out / task.name
    if dst.exists():
        raise SystemExit(f"{dst} exists: materialize into a new directory")
    goal = None
    if seed != sd["default"] and (task / GOAL).is_dir():
        goal = goal_source(task, seed)
        if goal is None:
            raise SystemExit(f"{task.name}: seed {seed} has no goal images (tests/seeds/{seed}/goal/ of the task or of "
                             f"{holder(task).name}); its {GOAL}/ shows the default seed's")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(task, dst, symlinks=True)
    src = holder(task) / "tests" / "seeds" / str(seed)
    files = [f for f in ("instance.json", "instance.npz") if (src / f).is_file()]
    if seed != sd["default"] and files:            # a committed instance of that seed
        for place in places(task):
            for f in ("instance.json", "instance.npz"):
                if (dst / place / f).is_file():
                    if f not in files:
                        raise SystemExit(f"{task.name}: seed {seed} has no {f} (the default instance has one)")
                    shutil.copyfile(src / f, dst / place / f)
    elif seed != sd["default"] and sd["build"] != "runtime":
        raise SystemExit(f"{task.name}: seed {seed} has no frozen instance under {src}")
    if sd["build"] == "runtime":                   # the task stages or builds that seed's instance itself from this file
        for place in places(task):
            if (dst / place).is_dir():
                (dst / place / "seed").write_text(f"{seed}\n")
    if goal is not None:
        g = dst / GOAL
        shutil.rmtree(g)
        shutil.copytree(goal, g, symlinks=True)
    shutil.rmtree(dst / "tests" / "seeds", ignore_errors=True)
    p = dst / "task.toml"
    s = p.read_text()
    s = re.sub(r"(?m)^(\[metadata\.seeds\](?:\n#.*)*\ndefault = )-?\d+", rf"\g<1>{seed}", s, count=1)
    s = re.sub(r"(?m)^instance_seed = -?\d+", f"instance_seed = {seed}", s, count=1)
    p.write_text(s)
    if stamp_mtimes.stamp_tree(str(dst), full=True)["err"]:     # Harbor builds from this copy: content-derived mtimes
        raise SystemExit(f"{dst}: could not stamp its files (scripts/stamp_mtimes.py)")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_list = sub.add_parser("list")
    a_list.add_argument("task", type=Path)
    a_asg = sub.add_parser("assign", help="the seeds of trials 1..K of a batch, one per line (--json: a JSON list)")
    a_asg.add_argument("task", type=Path)
    a_asg.add_argument("--batch", required=True)
    a_asg.add_argument("--trials", type=int, required=True)
    a_asg.add_argument("--json", action="store_true")
    a_pick = sub.add_parser("pick")
    a_pick.add_argument("task", type=Path)
    a_pick.add_argument("--rng", type=int)
    a_mat = sub.add_parser("materialize")
    a_mat.add_argument("task", type=Path)
    a_mat.add_argument("--seed", required=True, help="an allowed seed, or `draw`")
    a_mat.add_argument("--rng", type=int)
    a_mat.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    task = a.task.resolve()
    if a.cmd == "list":
        print(seeds(task))
    elif a.cmd == "assign":
        got = assign(task, a.batch, a.trials)
        print(json.dumps(got) if a.json else "\n".join(map(str, got)))
    elif a.cmd == "pick":
        print(pick(task, a.rng))
    else:
        seed = pick(task, a.rng) if a.seed == "draw" else int(a.seed)
        print(materialize(task, seed, a.out.resolve()), seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
