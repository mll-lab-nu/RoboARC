"""Placement lab (knot-robot, 10-12): placement tries from one untied, settled state, each from the same checkpoint.

  python place_lab.py <instance.json> <tier> <out> --resume <rows.npz> --resume-rows N [--settle-s 2] [--hold-s 6]
  python place_lab.py <instance.json> <tier> <out> --restore <another lab>/checkpoint.pkl   (no replay)

One process replays a run's rows to an untied state once (the long part), waits --settle-s, checkpoints the episode
(the env's privileged rewind, the mechanism Session.reset uses, plus the Session's own counters, as branching.py does),
then serves placement tries in a loop: every <out>/queue/<name>.json ({"func": "lay", "file": <optional strategy .py>,
"params": {...}, "region": <optional polygon override, task-local>}) is run once, in name order, from the checkpoint:
rewind, (re)load the strategy module (robot_place by default, or the given file: edit it and queue another try, no
restart), fresh RobotLab / RobotUntie on the Session, the strategy, release + park, then hold rows until the Session ends
the episode or --hold-s runs out. Each try writes <out>/tries/<name>/result.json (Session success, goal metrics with the
regions, crossings, rows, wall time, error) and privileged.npz (every row from the frozen start: the prefix, the settle
and the try; a successful try is a complete recording, to be checked by a fresh-process replay). <out>/STOP ends it.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

SESSION_FIELDS = ("n_steps", "physics_steps", "sim_time_s", "last_step_ticks", "_reward", "episode_end", "episode_done")


def save_rows(s, path, meta):
    acts = np.asarray(np.stack(s._recording) if s._recording else np.zeros((0, s.action_dim)), np.float32)
    np.savez(path, actions=acts, task=s.task, topogym_task=s.config["task_id"], variation_id=s.config["variation_id"],
             eval_index=-1 if s.eval_index is None else s.eval_index, scene_seed=int(s.config["seed"]),
             instance_sha=s.instance_sha, action_dim=s.action_dim, action_layout=np.array(s.action_layout),
             gripper_space=s.gripper_space, meta=json.dumps(meta or {}), trajectory_version=2,
             physics_steps_per_action=np.asarray(s._recorded_ticks, np.int64))
    return int(len(acts))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("instance")
    ap.add_argument("tier")
    ap.add_argument("out")
    ap.add_argument("--resume", default=None, help="the rows to replay to the start state (or --restore)")
    ap.add_argument("--restore", default=None, help="a checkpoint.pkl (ckpt.py) of the start state: restored instead of "
                    "replaying the rows (a lab started from the same rows wrote it)")
    ap.add_argument("--no-save-checkpoint", action="store_true", help="do not write <out>/checkpoint.pkl after the replay")
    ap.add_argument("--resume-rows", type=int, default=None)
    ap.add_argument("--settle-s", type=float, default=2.)
    ap.add_argument("--hold-s", type=float, default=6.)
    ap.add_argument("--limit", type=float, default=8.)
    ap.add_argument("--pinch-clearance", type=float, default=.010)
    ap.add_argument("--poll-s", type=float, default=20.)
    ap.add_argument("--once", action="store_true", help="end when the queue is empty (a chain's placement stage: the queue holds its one try)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "queue").mkdir(parents=True, exist_ok=True)
    (out / "tries").mkdir(exist_ok=True)

    def log(*x):
        print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)
    import run_robot_oracle as R
    spec = dict(instance=str(Path(a.instance).resolve()))
    s = R.session_from_spec(spec)
    (out / "session_spec.json").write_text(json.dumps(spec))
    from topogym.planning.knot_untying.redesign import robot_lab as RL, robot_untie as RU
    import topogym.planning.knot_untying.artifacts  # noqa: F401
    import topogym.tasks.knot_untying.redesign.goal  # noqa: F401
    ns = argparse.Namespace(scene=None)
    scene = R.scene_for(ns, s, a.instance)
    s.record_start()
    t0 = time.time()
    import ckpt
    if a.restore:
        meta = ckpt.restore_checkpoint(s, a.restore)
        log("restored", a.restore, json.dumps(meta, default=str)[:400], f"in {time.time() - t0:.1f} s; rows {s.n_steps}")
        lab = RL.RobotLab(s, scene, hand_limit_n=a.limit, budget_s=None, pinch_clearance_m=a.pinch_clearance, log=log)
        if a.settle_s > 0:              # an oracle's checkpoint (taken at a decision); another lab's is settled already
            lab.wait(a.settle_s, label="settle")
    else:
        if not a.resume:
            raise SystemExit("place_lab: --resume <rows> or --restore <checkpoint.pkl>")
        R.replay_prefix(s, a.resume, log, a.resume_rows)
        lab = RL.RobotLab(s, scene, hand_limit_n=a.limit, budget_s=None, pinch_clearance_m=a.pinch_clearance, log=log)
        lab.wait(a.settle_s, label="settle")
        if not a.no_save_checkpoint:
            n = ckpt.save_checkpoint(s, out / "checkpoint.pkl", dict(resume=a.resume, resume_rows=a.resume_rows,
                                                                     settle_s=a.settle_s, rows=int(s.n_steps)))
            log("checkpoint saved", out / "checkpoint.pkl", n, "bytes")
    regions = scene.get("regions") or None
    base = dict(row=int(s.n_steps), t=round(float(s.sim_time_s), 3), crossings=len(lab.crossings()),
                task=dict(s.env._task.result.metrics), lab=lab.goal_metrics(region=regions),
                replay_s=round(time.time() - t0, 1))
    log("checkpoint", json.dumps(base, default=float)[:1500])
    (out / "base.json").write_text(json.dumps(base, default=float, indent=1))
    s.env._checkpoint_current()
    mark = dict(ref=dict(s.env._info()["state_ref"]), session={f: copy.deepcopy(getattr(s, f)) for f in SESSION_FIELDS},
                recording=len(s._recording), ticks=len(s._recorded_ticks), targets=np.array(s._targets, copy=True))
    save_rows(s, out / "base_rows.npz", dict(source="place_lab base (prefix + settle)", resume=a.resume,
                                             resume_rows=a.resume_rows, settle_s=a.settle_s))

    def rewind():
        s.env.privileged.rewind_to(mark["ref"])
        for f in SESSION_FIELDS:
            setattr(s, f, copy.deepcopy(mark["session"][f]))
        del s._recording[mark["recording"]:]
        del s._recorded_ticks[mark["ticks"]:]
        s._targets = mark["targets"].copy()
        s.instability = None
        s.last_step_meta = {}

    done = set(p.name for p in (out / "tries").iterdir())
    while not (out / "STOP").exists():
        todo = [p for p in sorted((out / "queue").glob("*.json")) if p.stem not in done]
        if not todo:
            if a.once:
                break
            time.sleep(a.poll_s)
            continue
        q = todo[0]
        done.add(q.stem)
        tdir = out / "tries" / q.stem
        tdir.mkdir(exist_ok=True)
        try:
            spec_t = json.loads(q.read_text())
        except Exception as e:  # noqa: BLE001
            (tdir / "result.json").write_text(json.dumps(dict(error=f"bad try file: {e!r}")))
            continue
        rewind()
        t1 = time.time()
        err, solved = None, False
        lab = RL.RobotLab(s, scene, hand_limit_n=a.limit, budget_s=None, pinch_clearance_m=a.pinch_clearance, log=log,
                          trace_path=tdir / "trace.jsonl")
        U = RU.RobotUntie(lab, scene, limit=a.limit)
        for k, v in (spec_t.get("untie") or {}).items():
            setattr(U, k, v)
        for k, v in (spec_t.get("lab") or {}).items():   # (10-14, opt-in) e.g. the --turbo lab timings: move_min_time,
            setattr(lab, k, v)                            # rot_speed, close_s
        reg = spec_t.get("region") or regions
        rec = {}
        try:
            if spec_t.get("file"):
                sp = importlib.util.spec_from_file_location(f"place_try_{q.stem}", spec_t["file"])
                mod = importlib.util.module_from_spec(sp)
                sp.loader.exec_module(mod)
            else:
                mod = importlib.reload(importlib.import_module("topogym.planning.knot_untying.redesign.robot_place"))
            rec = getattr(mod, spec_t.get("func", "lay"))(lab, U, scene, log=log, regions=reg,
                                                          **(spec_t.get("params") or {})) or {}
            U.release_all()
            U.park()
            t_end = lab.t + a.hold_s
            while lab.t < t_end:
                lab._step("final_hold")
        except RL.Solved:
            solved = True
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            try:
                U.release_all()
                U.park()
                t_end = lab.t + a.hold_s
                while lab.t < t_end:
                    lab._step("final_hold")
            except RL.Solved:
                solved = True
            except Exception:  # noqa: BLE001
                pass
        st = s.status()
        res = dict(name=q.stem, spec=spec_t, solved=bool(solved or st.get("success")), error=err,
                   episode_end=st.get("episode_end"), state_sha=st.get("state_sha"), rows=int(s.n_steps),
                   try_rows=int(s.n_steps) - base["row"], sim_s=round(float(s.sim_time_s), 3),
                   wall_s=round(time.time() - t1, 1), crossings=len(lab.crossings()),
                   task=dict(s.env._task.result.metrics), lab=lab.goal_metrics(region=reg),
                   max_force={h: round(float(v), 3) for h, v in lab.max_force.items()},
                   events=lab.events[-40:], record=rec)
        n = save_rows(s, tdir / "privileged.npz", dict(source="place_lab try", name=q.stem, spec=spec_t,
                                                        resume=a.resume, resume_rows=a.resume_rows, settle_s=a.settle_s,
                                                        solved=res["solved"]))
        res["saved_rows"] = n
        (tdir / "result.json").write_text(json.dumps(res, default=float, indent=1))
        log("RESULT", q.stem, json.dumps({k: res[k] for k in ("solved", "error", "episode_end", "try_rows", "wall_s",
                                                               "crossings")}, default=float),
            json.dumps(res["lab"], default=float)[:900])
    log("STOP")
    s.close()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
