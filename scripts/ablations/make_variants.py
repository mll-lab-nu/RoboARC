#!/usr/bin/env python3
"""Build the task variants of the ablation experiments (docs/ABLATIONS.md) next to the released tasks.

    python3 scripts/ablations/make_variants.py build --experiment E1            # every variant E1 runs
    python3 scripts/ablations/make_variants.py build --condition q1c10 <base> ...  # one condition, some bases
    python3 scripts/ablations/make_variants.py list --experiment E1             # the variant dirs, one per line
    python3 scripts/ablations/make_variants.py verify --experiment E1           # the dirs exist, unchanged, as locked
    python3 scripts/ablations/make_variants.py conditions                       # the registry

A variant of the released task `tasks/<base>-<mode>` is `tasks/<base>-<token>-<mode>`: a copy of the task with only its
condition's edits (a condition module's `make(src, dst, **params)`, below), then, as the 2026-11 ablation's tool did:
task.toml's `name` set to the variant, its `description` prefixed with the condition's note, `[metadata] ablation =
"<token>"`, `DIGEST` recomputed (scripts/task_conventions.py) and every file's mtime derived from its content
(scripts/stamp_mtimes.py). Physics, grader, frozen instance and seeds are the base task's unless the condition says
otherwise (E4 changes the clearance). The batch runner finds the variants there: their base names are `<base>-<token>`.

Every build is deterministic: the same checkout gives byte-identical variant dirs, and `scripts/ablations/digests/
<token>.tsv` holds the DIGEST each variant must have (a build or `verify` that gives another one fails). `--record`
writes the DIGESTs of the dirs built into those files instead (for a maintainer, after a deliberate change).

Condition modules (scripts/ablations/cond_*.py) export `make(src, dst, **params)`: dst is a full copy of src
(shutil.copytree, symlinks kept) to edit in place; src is never written. They raise on anything unexpected: the
half-made dir is then removed. An optional `leak_check(dst) -> list[str]` is run after `make`; any problem it returns
fails the variant. Standard library only (Python >= 3.11)."""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import pathlib
import re
import shutil
import sys
import tomllib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "scripts")]
import stamp_mtimes  # noqa: E402
import task_conventions as tc  # noqa: E402

DIGESTS = HERE / "digests"


@dataclasses.dataclass(frozen=True)
class Condition:
    token: str                  # in the variant's dir name: <base>-<token>-<mode>
    label: str                  # the experiments' name for it (C00, C10, ...)
    modes: tuple[str, ...]      # the modes it is built in, in this order
    module: str                 # scripts/ablations/<module>.py, its make(src, dst, **params)
    params: tuple               # (name, value) pairs passed to make
    note: str                   # task.toml: description prefix and [metadata] ablation comment


# The C00..C1G notes are the 2026-11 ablation's, character for character: the variants stay byte-identical to the ones
# it ran (their DIGESTs are in digests/), so their results pool with it by task name.
REGISTRY: dict[str, Condition] = {c.token: c for c in (
    Condition("q1c00", "C00", ("standard",), "cond_info", (("condition", "q1c00"),),
              "probing v2 Q1 C00r: the standard task with the per-call log patch only (the concurrent control)"),
    Condition("q1c10", "C10", ("standard",), "cond_info", (("condition", "q1c10"),),
              "probing v2 Q1 C10: standard plus the task objects' poses in meta['objects'], robot frame"),
    Condition("q1c0g", "C0G", ("standard",), "cond_info", (("condition", "q1c0g"),),
              "probing v2 Q1 C0g (11-15): standard plus the part shapes (GET /geometry), each in its own frame, no poses, "
              "no internal state"),
    Condition("q1c1g", "C1G", ("standard",), "cond_info", (("condition", "q1c1g"),),
              "probing v2 Q1 C1g: standard plus the task objects' poses (C10) and the part shapes (GET /geometry), "
              "no internal state, robot frame"),
    Condition("cstate", "CS", ("standard",), "cond_state_deform", (),
              "ablation E3 CS: standard plus the deformable state and the other task objects' poses and shapes in every "
              "response, robot frame"),
    Condition("clr2x", "clearance x2", ("privileged", "standard"), "cond_clearance", (("factor", 2),),
              "ablation E4: the puzzle's clearance relaxed x2; scene, start state, robot and grader logic otherwise unchanged"),
    Condition("clr4x", "clearance x4", ("privileged", "standard"), "cond_clearance", (("factor", 4),),
              "ablation E4: the puzzle's clearance relaxed x4; scene, start state, robot and grader logic otherwise unchanged"),
    Condition("obsv1", "V1", ("standard",), "cond_obs", (("level", 1),),
              "ablation E5 V1: the standard task (with the C00 log patch) without the contact-safety readings (no meta['safety'])"),
    Condition("obsv2", "V2", ("standard",), "cond_obs", (("level", 2),),
              "ablation E5 V2: V1 plus RealSense-like depth noise and a fixed camera-calibration error (observation side only)"),
)}


def sub1(text: str, old: str, new: str, what: str, regex: bool = False) -> str:
    """text with old replaced by new; old must occur exactly once."""
    n = len(re.findall(old, text)) if regex else text.count(old)
    if n != 1:
        raise SystemExit(f"{what}: expected 1 match, found {n}")
    return re.sub(old, lambda m: new, text) if regex else text.replace(old, new)


def module(cond: Condition):
    try:
        return importlib.import_module(cond.module)
    except ModuleNotFoundError as exc:
        if exc.name == cond.module:
            raise SystemExit(f"condition {cond.token}: scripts/ablations/{cond.module}.py is not on this branch yet") from None
        raise


def variant_name(base: str, token: str, mode: str) -> str:
    return f"{base}-{token}-{mode}"


def load_lock(token: str) -> dict[str, str]:
    p = DIGESTS / f"{token}.tsv"
    if not p.is_file():
        return {}
    rows = [l.split("\t") for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return {r[0]: r[1].strip() for r in rows}


def record_lock(token: str, got: dict[str, str]) -> None:
    lock = {**load_lock(token), **got}
    DIGESTS.mkdir(exist_ok=True)
    head = f"# variant dir\tDIGEST (scripts/ablations/make_variants.py build --record; condition {token})\n"
    (DIGESTS / f"{token}.tsv").write_text(head + "".join(f"{k}\t{lock[k]}\n" for k in sorted(lock)))


def make_one(cond: Condition, base: str, mode: str, tasks: pathlib.Path, out: pathlib.Path) -> tuple[pathlib.Path, str]:
    """Build out/<base>-<token>-<mode> from tasks/<base>-<mode>; returns the dir and its DIGEST."""
    src = tasks / f"{base}-{mode}"
    dst = out / variant_name(base, cond.token, mode)
    if not src.is_dir():
        raise SystemExit(f"no task {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    try:
        mod = module(cond)
        mod.make(src, dst, **dict(cond.params))
        problems = mod.leak_check(dst) if hasattr(mod, "leak_check") else []
        if problems:
            raise SystemExit(f"{dst.name}: leak check: " + "; ".join(problems))
        t = (dst / "task.toml").read_text()
        t = sub1(t, f'name = "roboarc/{src.name}"', f'name = "roboarc/{dst.name}"', f"{src.name} task.toml name")
        t = sub1(t, r'(?m)^description = "', f'description = "[{cond.note}] ', f"{src.name} task.toml description", regex=True)
        t = sub1(t, "[metadata]\n", f'[metadata]\nablation = "{cond.token}"   # {cond.note}; base task {src.name}\n',
                 f"{src.name} task.toml [metadata]")
        tomllib.loads(t)                      # a variant whose task.toml does not parse is not a task for Harbor
        (dst / "task.toml").write_text(t)
        digest = tc.digest(dst)
        (dst / "DIGEST").write_text(digest + "\n")
        if stamp_mtimes.stamp_tree(str(dst), full=True)["err"]:
            raise SystemExit(f"{dst}: stamp_mtimes failed")
    except BaseException:
        shutil.rmtree(dst, ignore_errors=True)
        raise
    return dst, digest


def wanted(args) -> list[tuple[Condition, str, str]]:
    """(condition, base, mode) for every variant the command line names, in a fixed order."""
    import experiments
    if args.experiment:
        return [(REGISTRY[v.token], v.base, v.mode) for v in experiments.variants(args.experiment)]
    if args.condition not in REGISTRY:
        raise SystemExit(f"unknown condition {args.condition!r}: one of {', '.join(REGISTRY)}")
    if not args.bases:
        raise SystemExit("--condition needs the bases to build, e.g. roboarc-bed-bolt-01-i01")
    c = REGISTRY[args.condition]
    return [(c, b, m) for b in args.bases for m in c.modes]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("build", "list", "verify", "conditions"))
    ap.add_argument("--experiment", choices=("E1", "E2", "E3", "E4", "E5"))
    ap.add_argument("--condition")
    ap.add_argument("bases", nargs="*")
    ap.add_argument("--tasks", type=pathlib.Path, default=REPO / "tasks", help="where the released tasks are (tasks/)")
    ap.add_argument("--out", type=pathlib.Path, help="where the variants go (default: --tasks)")
    ap.add_argument("--record", action="store_true", help="write the DIGESTs built into digests/<token>.tsv")
    a = ap.parse_args(argv)
    if a.cmd == "conditions":
        for c in REGISTRY.values():
            print(f"{c.token}\t{c.label}\t{','.join(c.modes)}\t{c.module}\t{c.note}")
        return 0
    if not (a.experiment or a.condition):
        ap.error("name --experiment or --condition")
    tasks = a.tasks.resolve()
    out = (a.out or a.tasks).resolve()
    todo = wanted(a)
    if a.cmd == "list":
        for c, b, m in todo:
            print(variant_name(b, c.token, m))
        return 0
    bad = 0
    if a.cmd == "verify":
        for c, b, m in todo:
            d = out / variant_name(b, c.token, m)
            lock = load_lock(c.token).get(d.name)
            if not d.is_dir():
                print(f"MISSING {d.name}"); bad += 1; continue
            now = tc.digest(d)
            stated = (d / "DIGEST").read_text().strip() if (d / "DIGEST").is_file() else None
            ok = now == stated and (lock is None or now == lock)
            print(f"{'ok' if ok else 'CHANGED'} {d.name} {now}" + ("" if lock else " (no locked DIGEST)"))
            bad += not ok
        return 1 if bad else 0
    built: dict[str, dict[str, str]] = {}
    for c, b, m in todo:
        d, digest = make_one(c, b, m, tasks, out)
        lock = load_lock(c.token).get(d.name)
        state = "new" if lock is None else ("ok" if lock == digest else "DIFFERS FROM LOCK")
        if lock is not None and lock != digest and not a.record:
            bad += 1
        built.setdefault(c.token, {})[d.name] = digest
        print(f"{d.name}  {digest}  {state}")
    if a.record:
        for token, got in built.items():
            record_lock(token, got)
    if bad:
        print(f"{bad} variant(s) differ from scripts/ablations/digests/: this checkout does not build the variants the "
              "experiment was validated with (see docs/ABLATIONS.md, 'If a check fails')", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
