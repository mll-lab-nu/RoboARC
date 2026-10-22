#!/usr/bin/env python3
"""INVARIANT (docs/PROTOCOL.md Appendix H.7, comparable twins): part of the required CI job `invariants`
(ci_checks/run_invariants.sh); no task value can turn it off. Where a task's privileged and standard versions both
exist (the mode registry's `same_cutoff_as` and `twin`, scripts/task_conventions.py MODE_REGISTRY):
  - they have the same cutoff, `[agent] timeout_sec` (owner 2026-10-12);
  - a standard or standard-realtime task has its privileged twin's seeds: `[metadata.seeds]` default, allowed and build.
The shared frozen instance (byte-identical copies) is ci_checks/check-instance-copies.sh, the other half of H.7.

    python3 ci_checks/check-twins.py [task dirs]      # default: every task; a twin outside the given dirs is read too
Standard library only.
"""
from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import task_conventions as tc  # noqa: E402


def cutoff_pairs(task: Path) -> list[tuple[Path, Path]]:
    """(task, its twin) for every mode pair the registry ties together (`same_cutoff_as`), in either direction, where the
    twin exists; sorted so that a pair reached from both sides is the same tuple."""
    mode, base = tc.mode_of(task), tc.base_of(task)
    out = []
    for m, d in tc.MODE_REGISTRY.items():
        o = d["same_cutoff_as"]
        if not o or mode not in (m, o):
            continue
        other = task.with_name(f"{base}-{o if mode == m else m}")
        if (other / "task.toml").is_file():
            out.append(tuple(sorted((task, other))))
    return out


def seeds_key(task: Path):
    sd = tc.seeds_of(tc.metadata_of(task))
    return sd["default"], sd["allowed"], sd["build"]


def errors(tasks: list[Path]) -> list[str]:
    errs, seen = [], set()
    for t in tasks:
        try:
            tc.mode_of(t)
        except ValueError as e:
            errs.append(f"{t.name}: {e}")
            continue
        for a, b in cutoff_pairs(t):
            if (a, b) in seen:
                continue
            seen.add((a, b))
            sa, sb = tc.cutoff_for(a), tc.cutoff_for(b)
            if sa != sb:
                errs.append(f"{a.name} has a cutoff of {sa} s and {b.name} {sb} s; the twins have the same cutoff "
                            "(PROTOCOL Appendix H.7; mode registry `same_cutoff_as`)")
        twin = tc.twin_of(t)
        if twin is not None and (twin / "task.toml").is_file():
            try:
                mine, theirs = seeds_key(t), seeds_key(twin)
            except (KeyError, ValueError, TypeError, tomllib.TOMLDecodeError) as e:
                errs.append(f"{t.name}: [metadata.seeds] unreadable ({e})")
                continue
            if mine != theirs:
                errs.append(f"{t.name}: seeds (default, allowed, build) {mine} differ from its privileged twin's {theirs} "
                            "(PROTOCOL Appendix H.7: twins share instance, success rule and seeds)")
            ma, mb = seed_bindings(t), seed_bindings(twin)
            for task, m in ((t, ma), (twin, mb)):
                for seed, shas in sorted(m.items()):
                    if len(shas) != 1:
                        errs.append(f"{task.name}: its copies of runtime_binding.s{seed}.json name {len(shas)} instances "
                                    "(PROTOCOL Appendix H.7: one frozen instance per seed)")
            if {k: sorted(v) for k, v in ma.items()} != {k: sorted(v) for k, v in mb.items()}:
                errs.append(f"{t.name}: its seed bindings (runtime_binding.s<S>.json: {fmt(ma)}) are not its privileged "
                            f"twin's ({fmt(mb)}) (PROTOCOL Appendix H.7: twins share each seed's frozen instance)")
            if seed_files(t) != seed_files(twin):
                errs.append(f"{t.name}: its `seed` files {seed_files(t)} are not its privileged twin's {seed_files(twin)}")
    return errs


def seed_bindings(task: Path) -> dict:
    """{seed: {instance_sha256 of every runtime_binding.s<seed>.json of the task}} (docs/RUNNING.md "Seeds")."""
    out: dict = {}
    for p in sorted(task.rglob("runtime_binding.s*.json")):
        m = re.fullmatch(r"runtime_binding\.s(\d+)\.json", p.name)
        if not m:
            continue
        try:
            sha = json.loads(p.read_text()).get("instance_sha256")
        except (OSError, ValueError, AttributeError):
            sha = f"unreadable {p.relative_to(task)}"
        out.setdefault(int(m.group(1)), set()).add(sha)
    return out


def seed_files(task: Path) -> set:
    return {p.read_text().strip() for p in task.rglob("seed") if p.is_file()}


def fmt(m: dict) -> str:
    return ", ".join(f"{k}: {','.join(str(x)[:12] for x in sorted(v))}" for k, v in sorted(m.items())) or "none"


def main(argv: list[str]) -> int:
    tasks = [Path(a).resolve() for a in argv if (Path(a) / "task.toml").is_file()] if argv else \
        sorted(p for p in (ROOT / "tasks").iterdir() if (p / "task.toml").is_file())
    errs = errors(tasks)
    for e in errs:
        print(f"FAIL {e}")
    if errs:
        return 1
    print(f"check-twins: {len(tasks)} task(s); twins have equal cutoffs and seeds")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
