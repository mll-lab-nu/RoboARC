"""python -m roboarc <command>

  list [--suite puzzles|ikea] [--json]     the tasks (no simulator needed)
  info <task>                              one task's static facts (and, with --sim, its spaces and cameras)
  check                                    is this the graded runtime? (image, libmujoco EPA patch, numpy masking)
  replay <task> [--traj F] [--mode privileged|standard] [--rules standard|privileged] [--renderer none]
                                           step a trajectory (default: the task's privileged reference, with the
                                           privileged rules) through roboarc.make() from reset() and compare the end
                                           state hash with the recorded one
  patch-mujoco                             apply the EPA byte patch to the installed mujoco 3.14.0 wheel (outside the
                                           images; the images ship it)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import roboarc  # noqa: F401  (sets the graded numpy settings before numpy is imported)


def _list(a) -> int:
    from .registry import list_tasks
    tasks = list_tasks(a.suite)
    if a.json:
        print(json.dumps([t.to_dict() for t in tasks], indent=1))
        return 0
    for t in tasks:
        ref = "ref" if t.reference else "   "
        print(f"{t.id:42s} {t.suite:8s} {t.family:22s} {(t.difficulty or '-'):10s} {t.robot:24s} {ref} rows<={t.max_episode_steps}")
    print(f"{len(tasks)} tasks", file=sys.stderr)
    return 0


def _info(a) -> int:
    from .registry import resolve
    t = resolve(a.task)
    out = {"task": t.to_dict()}
    if a.sim:
        env = roboarc.make(t.id, renderer=a.renderer, strict=not a.no_strict)
        out["env"] = env.describe()
        env.close()
    print(json.dumps(out, indent=1, default=str))
    return 0


def _check(a) -> int:
    from . import _runtime as R
    out = {"libmujoco_sha256": R.libmujoco_sha256(), "libmujoco_epa_patched": R.libmujoco_sha256() == R.LIBMUJOCO_EPA_SHA256,
           "numpy_masking": R.numpy_masking_active(), "puzzle_image": R.in_puzzle_image(), "ikea_image": R.in_ikea_image(),
           "nyx_python (ikea -sim image)": R.IMAGE_NYX_PYTHON.is_file(), "repo": str(R.repo_root()),
           "problems": R.check_graded_runtime(strict=False)}
    print(json.dumps(out, indent=1))
    return 0 if not out["problems"] else 1


def _replay(a) -> int:
    import numpy as np
    from .references import hash_matches, load_reference
    from .registry import resolve
    t = resolve(a.task)
    if a.traj:
        with np.load(a.traj, allow_pickle=False) as z:
            rows = np.asarray(z["actions"] if "actions" in z.files else z["ctrl"])
        expected = None
    else:
        rows = load_reference(t.id, a.mode)
        if rows is None:
            print(json.dumps({"task": t.id, "error": "no reference trajectory"}))
            return 2
        # the recorded end hash belongs to the privileged reference (and to the standard one where it is the same file)
        same = t.suite == "ikea" or a.mode == "privileged" or np.array_equal(rows, load_reference(t.id, "privileged"))
        expected = t.reference_end_hash if same else None
    t0 = time.time()
    rules = a.rules or ("privileged" if a.mode == "privileged" else "standard")
    kw = {"rules": rules} if t.suite == "puzzles" else {}
    env = roboarc.make(t.id, renderer=a.renderer, strict=not a.no_strict, **kw)
    obs, info = env.reset()
    birth = info["birth_hash"]
    t1 = time.time()
    steps, end, success = 0, None, False
    limited = 0
    for row in rows:
        obs, rew, term, trunc, info = env.step(row)
        steps += 1
        limited += int(bool(info.get("speed_limited")))
        success = success or bool(info.get("success"))
        if (term and t.suite == "puzzles") or trunc:      # a puzzle's scene stops at its episode end; IKEA steps on
            end = info.get("episode_end")
            break
    out = {"task": t.id, "mode": a.mode, "rules": kw.get("rules", "standard"), "rows": int(len(rows)), "steps": steps, "episode_end": end or info.get("episode_end"),
           "success": success, "speed_limited_steps": limited, "birth_hash": birth, "end_hash": env.state_sha(),
           "expected_end_hash": expected, "match": hash_matches(env.state_sha(), expected),
           "build_s": round(t1 - t0, 1), "step_s": round(time.time() - t1, 1)}
    if t.suite == "ikea":
        fc = env.unwrapped._settled()
        out.update(final_check_success=bool(fc["success"]), settled_hash=fc["settled_sha"],
                   expected_settled_hash=t.reference_settled_hash if not a.traj else None,
                   settled_match=hash_matches(fc["settled_sha"], t.reference_settled_hash) if not a.traj else None)
    env.close()
    print(json.dumps(out), flush=True)
    return 0 if out["match"] in (True, None) and out.get("settled_match") in (True, None) else 1


def _patch(a) -> int:
    import importlib.util
    from pathlib import Path
    from . import _runtime as R
    spec = importlib.util.find_spec("mujoco")
    if spec is None:
        print("mujoco is not installed", file=sys.stderr)
        return 2
    lib = Path(list(spec.submodule_search_locations)[0]) / "libmujoco.so.3.14.0"
    script = R.repo_root() / "source" / "images" / "robotangle" / "mujoco-3.14.0-epa" / "bytepatch_libmujoco.py"
    return subprocess.call([sys.executable, str(script), str(lib)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m roboarc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("--suite", choices=("puzzles", "ikea")); p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_list)
    p = sub.add_parser("info"); p.add_argument("task"); p.add_argument("--sim", action="store_true")
    p.add_argument("--renderer", default="none"); p.add_argument("--no-strict", action="store_true"); p.set_defaults(fn=_info)
    p = sub.add_parser("check"); p.set_defaults(fn=_check)
    p = sub.add_parser("replay"); p.add_argument("task"); p.add_argument("--traj")
    p.add_argument("--renderer", default="none", choices=("none", "mujoco", "nyx"))
    p.add_argument("--mode", default="privileged", choices=("privileged", "standard"),
                   help="which reference (default: the privileged mode's, whose end hash the release records)")
    p.add_argument("--rules", default=None, choices=("standard", "privileged"),
                   help="puzzles: default = the reference's mode (its verifier's / service's rules)")
    p.add_argument("--no-strict", action="store_true"); p.set_defaults(fn=_replay)
    p = sub.add_parser("patch-mujoco"); p.set_defaults(fn=_patch)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
