#!/usr/bin/env python3
"""Render the IKEA MuJoCo tasks onto a checkout of main (PROTOCOL v2.0) and run main's checks on them: the dry run of the
v2 wave's IKEA part. Nothing here writes to this repository.

usage: python3 scripts/ikea_mujoco/port_main.py <main checkout> [--no-checks]
  <main checkout>  a working tree of origin/main (a sparse one with scripts/, ci_checks/, docs/, .github/ and tasks/ is
                   enough); it is modified in place, so use a throw-away worktree

Steps:
  1. main-side registry entries, from scripts/ikea_mujoco/main_side/ (idempotent): the benchmark label in
     scripts/task_conventions.py BENCHMARKS (prefix `ikea-`), the resource / limit / cap rules appended to
     scripts/task_resources.toml, docs/benchmarks/ikea_mujoco.md;
  2. tasks/ikea-*: copied from this repository (the dev render of mk_tasks.py / mk_standard_tasks.py);
  3. main's scripts/set_agent_egress.py allowlist (H.6), then main's scripts/task_conventions.py normalize on each task
     (seeds, the External material / Time limit sections, resources, the monitor, DIGEST: H.8);
  4. checks (unless --no-checks): ci_checks/run_invariants.sh on the IKEA tasks (Appendix H), then the advisory
     ci_checks/check-task.sh; their output is printed and the exit code is the invariants' one.
"""
from __future__ import annotations

import argparse, pathlib, shutil, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SIDE = HERE / "main_side"
LABEL = '    ("ikea-", "IKEA MuJoCo"),\n'
MARK = "# IKEA MuJoCo (scripts/ikea_mujoco/main_side/task_resources.toml)"


def registry(main: pathlib.Path) -> None:
    tc = main / "scripts/task_conventions.py"
    s = tc.read_text()
    if LABEL not in s:
        i = s.index("BENCHMARKS = [")
        j = s.index("\n]\n", i)
        s = s[:j + 1] + LABEL + s[j + 1:]
        tc.write_text(s)
    tr = main / "scripts/task_resources.toml"
    s = tr.read_text()
    if MARK not in s:
        tr.write_text(s.rstrip("\n") + "\n\n" + MARK + "\n" + (SIDE / "task_resources.toml").read_text())
    shutil.copyfile(ROOT / "docs/benchmarks/ikea_mujoco.md", main / "docs/benchmarks/ikea_mujoco.md")


def copy_tasks(main: pathlib.Path) -> list[pathlib.Path]:
    out = []
    for src in sorted((ROOT / "tasks").glob("ikea-*")):
        dst = main / "tasks" / src.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=True)
        out.append(dst)
    return out


def run(cmd, cwd) -> int:
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=cwd).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("main", type=pathlib.Path)
    ap.add_argument("--no-checks", action="store_true")
    a = ap.parse_args()
    m = a.main.resolve()
    if not (m / "scripts/task_conventions.py").is_file() or not (m / "ci_checks/run_invariants.sh").is_file():
        sys.exit(f"{m}: not a checkout of main (scripts/task_conventions.py, ci_checks/run_invariants.sh)")
    registry(m)
    tasks = copy_tasks(m)
    rel = [str(t.relative_to(m)) for t in tasks]
    if run([sys.executable, "scripts/set_agent_egress.py", "allowlist", *rel], m):
        return 1
    for t in rel:      # privileged first: normalize reads the twin for the README's key table
        if run([sys.executable, "scripts/task_conventions.py", t], m):
            return 1
    if run([sys.executable, "scripts/task_conventions.py", "--check", *rel], m):
        return 1
    if a.no_checks:
        return 0
    rc = run(["bash", "ci_checks/run_invariants.sh", *rel], m)
    run(["bash", "ci_checks/check-task.sh", *rel], m)
    return rc


if __name__ == "__main__":
    sys.exit(main())
