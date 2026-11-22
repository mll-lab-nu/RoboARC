"""roboarc list / info / replay / patch-mujoco, and `check` (the graded-runtime report `doctor` shows inside an image).
`list` and `info` need no simulator; `info --sim` and `replay` build the environment and run inside an image."""
from __future__ import annotations

import json
import sys
import time

from . import _common as C

RENDERERS = ("none", "mujoco", "nyx")


def _list(a) -> int:
    from ..registry import list_tasks
    tasks = list_tasks(a.suite, a.family)
    if a.family and not tasks:
        known = sorted({t.family for t in list_tasks(a.suite)})
        raise C.CliError(f"no family {a.family!r}" + (f" in the suite {a.suite}" if a.suite else "")
                         + f"; the families: {', '.join(known)}")
    if a.json:
        print(json.dumps([t.to_dict() for t in tasks], indent=1))
        return 0
    for t in tasks:
        ref = "ref" if t.reference else "   "
        print(f"{t.id:42s} {t.suite:8s} {t.family:22s} {(t.difficulty or '-'):10s} {t.robot:24s} {ref} rows<={t.max_episode_steps}")
    print(f"{len(tasks)} tasks", file=sys.stderr)
    return 0


def _info(a) -> int:
    import roboarc
    from ..registry import resolve
    t = resolve(a.task)
    out = {"task": t.to_dict()}
    if a.sim:
        env = roboarc.make(t.id, renderer=a.renderer, strict=not a.no_strict)
        out["env"] = env.describe()
        env.close()
    print(json.dumps(out, indent=1, default=str))
    return 0


def runtime_report() -> dict:
    """Is this process the graded runtime? What `roboarc check` prints; `problems` is empty when it is."""
    from .. import _runtime as R
    return {"libmujoco_sha256": R.libmujoco_sha256(), "libmujoco_epa_patched": R.libmujoco_sha256() == R.LIBMUJOCO_EPA_SHA256,
            "numpy_masking": R.numpy_masking_active(), "puzzle_image": R.in_puzzle_image(), "ikea_image": R.in_ikea_image(),
            "nyx_python (ikea -sim image)": R.IMAGE_NYX_PYTHON.is_file(), "repo": str(R.repo_root()),
            "problems": R.check_graded_runtime(strict=False)}


def _check(a) -> int:
    out = runtime_report()
    print(json.dumps(out, indent=1))
    return 0 if not out["problems"] else 1


def _replay(a) -> int:
    import roboarc
    import numpy as np
    from ..references import hash_matches, load_reference
    from ..registry import resolve
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
    spec = importlib.util.find_spec("mujoco")
    if spec is None:
        print("mujoco is not installed", file=sys.stderr)
        return 2
    lib = Path(list(spec.submodule_search_locations)[0]) / "libmujoco.so.3.14.0"
    return C.run(C.command("source/images/roboarc/mujoco-3.14.0-epa/bytepatch_libmujoco.py", lib))


def add_parsers(sub) -> None:
    p = sub.add_parser("list", formatter_class=C.Formatter, help="the tasks", description="""\
The tasks of this checkout, one line each: id, suite, family, difficulty, robot, `ref` when the task has a reference
solution, and its row cap (the most control steps a trajectory may have). The number of tasks goes to stderr.
Reads the task directories under tasks/: no simulator, no image, no numpy.""", epilog="""\
examples:
  roboarc list                             # the 90 tasks
  roboarc list --family interlocked-claws  # one family
  roboarc list --suite ikea --json         # the 12 IKEA tasks with every field""")
    p.add_argument("--suite", choices=("puzzles", "ikea"), help="only this suite (default: both)")
    p.add_argument("--family", metavar="FAMILY", help="only this family, named as the list prints it, e.g. orbits, "
                   "cast_keyhole, ikea_lack; `-` may stand for `_` (default: every family)")
    p.add_argument("--json", action="store_true", help="a JSON list with every field of each task (roboarc.TaskInfo) "
                   "instead of the table")
    p.set_defaults(fn=_list)

    p = sub.add_parser("info", formatter_class=C.Formatter, help="one task's facts", description="""\
One task's facts as JSON: its ids, family, difficulty, robot, description, images, row cap, the agent's time limit,
the two task directories, the reference solution with its recorded end state hash, and the seeds.
With --sim it also builds the environment and adds its action and observation spaces, layouts and cameras: that part
runs inside the task's image (`roboarc doctor` there says whether it is the graded runtime).""", epilog="""\
examples:
  roboarc info orbits-02
  roboarc info roboarc-ikea-lack-i00 --sim --renderer none      # inside the image""")
    p.add_argument("task", metavar="TASK", help="a task id or alias: roboarc-orbits-02-i01, orbits-02, ikea-lack, lack")
    p.add_argument("--sim", action="store_true", help="also build the environment and print its spaces, layouts and "
                   "cameras (inside an image)")
    p.add_argument("--renderer", default="none", choices=RENDERERS, help="with --sim: the renderer the environment is "
                   "built with (default: none)")
    p.add_argument("--no-strict", action="store_true", help="with --sim: build the environment outside the graded "
                   "runtime too (a warning instead of an error; its results are then not the graded ones)")
    p.set_defaults(fn=_info)

    p = sub.add_parser("replay", formatter_class=C.Formatter,
                       help="step a trajectory through the simulator and compare the end state hash", description="""\
Step a trajectory through roboarc.make() from reset() and compare the end state hash with the recorded one. Without
--traj it is the task's reference solution, which must end exactly on its recorded hash on every machine. Prints one
JSON line: rows, steps, episode_end, success, end_hash, expected_end_hash, match (IKEA: also the settled state).
Runs inside the task's image. Exit status: 0 when the hashes match (or none is recorded for that trajectory), 1 when
they differ, 2 when the task has no reference.""", epilog="""\
example (inside the image, with the repository on PYTHONPATH):
  roboarc replay interlocked-claws-08""")
    p.add_argument("task", metavar="TASK", help="a task id or alias")
    p.add_argument("--traj", metavar="NPZ", help="a trajectory file to step instead of the reference: an .npz with "
                   "`actions` (puzzles) or `ctrl` (IKEA), one row per control step (default: the reference)")
    p.add_argument("--mode", default="privileged", choices=C.MODES, help="which reference: the privileged mode's, "
                   "whose end hash the release records, or the standard mode's (default: privileged)")
    p.add_argument("--rules", default=None, choices=("standard", "privileged"), help="puzzles: the joint-speed rules "
                   "the steps run under (default: those of --mode)")
    p.add_argument("--renderer", default="none", choices=RENDERERS, help="the renderer the environment is built with "
                   "(default: none: physics only)")
    p.add_argument("--no-strict", action="store_true", help="run outside the graded runtime too (a warning instead of "
                   "an error; the hashes may then differ)")
    p.set_defaults(fn=_replay)


def add_advanced_parsers(sub) -> None:
    p = sub.add_parser("patch-mujoco", formatter_class=C.Formatter,
                       help="outside the images: apply the EPA byte patch to the installed MuJoCo 3.14.0", description="""\
Apply the EPA byte patch to the libmujoco.so.3.14.0 of the installed `mujoco` 3.14.0 wheel (14 bytes at 5 sites, input
and output sha256 asserted): source/images/roboarc/mujoco-3.14.0-epa/bytepatch_libmujoco.py. The published images ship
the patched library, so this is only for a Python environment of your own. No arguments.""", epilog="""\
example:
  pip install mujoco==3.14.0 && roboarc patch-mujoco""")
    p.set_defaults(fn=_patch)

    # the name `doctor` had before the command line grew: the graded-runtime report alone, as JSON
    p = sub.add_parser("check", formatter_class=C.Formatter, description="""\
Is this process the graded runtime? Prints one JSON object (the libmujoco sha256 and whether it is the EPA-patched
one, numpy's SIMD masking, which image this is, the problems found) and exits 1 when there are problems.
`roboarc doctor` prints the same inside an image, and checks the machine on a host.""", epilog="""\
example (inside the image):
  roboarc check""")
    p.set_defaults(fn=_check)
