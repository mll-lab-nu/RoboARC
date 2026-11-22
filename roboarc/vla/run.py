"""Graded RoboARC episodes driven by an openpi policy server (docs/VLA.md).

    python -m roboarc.vla.run --task roboarc-interlocked-claws-08-i01 --policy ws://HOST:8000 --seed 0 --episodes 1 --out DIR
    python -m roboarc.vla.run --task <task> --export-reference ref.npz     # the task's reference in the convention's units

Each episode runs in a fresh process: the task's frozen instance (`--seed`: its instance seed, 0-2 for the puzzles),
the policy queried every `--horizon` actions (images rendered only then), every row through the environment's step
(the standard mode's rules: clamp, never fail), until the task's own episode end (success, its failure predicate, a
simulator breakdown) or the step limit. Then, as the standard mode's grader does, one fresh-process replay of the executed
rows from the frozen instance (a grading replay after a live success, plain after a failure; `--video` renders the
replay); success = live success AND replay success, without an infrastructure error, and a success with deep task
contact is held for review instead of credited (`success_pending_review`). IKEA tasks grade the live episode, as their
standard mode does.

Results (never mixed with Harbor's agent results): <out>/<task>/<run id>/s<seed>/e<k>/ with result.json, trajectory.npz
(the executed rows, the privileged deliverable's format), chunks.npz (every chunk the policy returned), replay.json,
video.mp4 (with --video); <out>/<task>/<run id>/config.json and results.json."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import roboarc  # noqa: F401  (before numpy: the graded numpy settings)
import numpy as np

from ..registry import resolve
from .conventions import CONVENTIONS, make_convention, task_prompt

INTERFACE = "roboarc.vla/openpi-websocket"
DEFAULT_HORIZON = {"aloha": 25, "droid": 8, "droid-velocity": 8}      # openpi's examples: aloha_real 25, droid 8
STEP_FACTOR, NO_REF_STEPS = 2.0, 12000                                  # default step limit (see default_max_steps)
# the settings that make episodes comparable (the evaluation's, not the policy's: the run id names the policy); one run
# id holds episodes of one configuration only
COMPARABLE = ("interface", "convention", "horizon", "policy_hz", "image_size", "rules", "renderer",
              "max_steps", "time_limit_s", "prompt", "roboarc_version", "camera_poses")


def config_key(config: dict) -> str:
    """The comparable configuration of VLA episodes: "vla:" + 12 hex digits of the COMPARABLE settings (not the task,
    not the seed). Episodes compare only within one key; it never equals a Harbor configuration key."""
    import hashlib
    blob = json.dumps({k: config.get(k) for k in COMPARABLE}, sort_keys=True, default=str)
    return "vla:" + hashlib.sha256(blob.encode()).hexdigest()[:12]


def default_max_steps(task) -> int:
    """2 x the task's standard reference (the expert's length, rounded up to 100 control steps); 12000 (5 min of
    simulated time at 40 Hz) for a task without a reference."""
    from ..references import load_reference
    rows = load_reference(task.id, "standard") if task.suite == "puzzles" else load_reference(task.id)
    if rows is None:
        return NO_REF_STEPS
    return int(100 * np.ceil(STEP_FACTOR * len(rows) / 100))


def _cameras_arg(text: str | None) -> dict | None:
    if not text:
        return None
    return dict(kv.split("=", 1) for kv in text.split(","))


def _write_json(path: Path, obj) -> None:
    """Whole or not at all: parallel runners of one run id read each other's result files."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}")   # not the pid: 1 in every container
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def _make_env(a, renderer: str, cameras=None, max_steps=None):
    if a.camera_poses:
        from . import droid_cameras
        droid_cameras.install(a.camera_poses)
    kw = {"rules": a.rules} if resolve(a.task).suite == "puzzles" else {}
    return roboarc.make(a.task, renderer=renderer, cameras=cameras, instance_seed=a.instance_seed,
                        max_episode_steps=max_steps, **kw)


# ------------------------------------------------------------------------------------------------- one live episode
def episode_worker(a) -> int:
    from .wire import WebsocketClientPolicy
    out = Path(a.episode_dir)
    t0 = time.time()
    task = resolve(a.task)
    env = _make_env(a, a.renderer, max_steps=a.max_steps)
    if a.time_limit_s:
        env.max_episode_steps = int(round(a.time_limit_s * env.control_hz))
    conv = make_convention(env, a.convention, image_size=a.image_size, policy_hz=a.policy_hz,
                           cameras=_cameras_arg(a.cameras))
    env.step_images = False                               # images only when the policy is asked
    prompt = a.prompt or task_prompt(task, full=a.full_prompt)
    policy = WebsocketClientPolicy(a.policy, api_key=os.environ.get("OPENPI_API_KEY"), wait_s=a.wait_s)
    meta = policy.get_server_metadata()
    obs, info = env.reset()
    birth = info["birth_hash"]
    build_s = time.time() - t0
    steps, terminated, truncated = 0, False, False
    chunks, chunk_steps, executed = [], [], []
    t_obs = t_infer = t_step = 0.0
    t1 = time.time()
    while not (terminated or truncated):
        ta = time.perf_counter()
        full = env.observe()
        o = conv.observation(full, prompt)
        if not a.no_roboarc_obs:
            o["roboarc"] = {"control_step": int(steps), "task": task.id, "instance_seed": int(a.instance_seed),
                            "episode": int(a.episode)}
        tb = time.perf_counter()
        result = policy.infer(o)
        tc = time.perf_counter()
        t_obs += tb - ta
        t_infer += tc - tb
        if "actions" not in result:
            raise RuntimeError(f"the policy's reply has no 'actions' (keys {sorted(result)})")
        chunk = conv.check_chunk(result["actions"])
        chunks.append(chunk.astype(np.float32))
        chunk_steps.append(steps)
        conv.begin_chunk(chunk, a.horizon)
        n = 0
        proprio = full["proprio"]
        ts = time.perf_counter()
        while True:
            row = conv.next_row(proprio)
            if row is None:
                break
            obs, _, terminated, truncated, info = env.step(row)
            proprio = obs["proprio"]
            steps += 1
            n += 1
            if terminated or truncated:
                break
        t_step += time.perf_counter() - ts
        executed.append(n)
        if n == 0:
            raise RuntimeError("the chunk executed no control step")
    wall = time.time() - t1
    live_success = bool(env.unwrapped._success_seen)
    live = {"task": task.id, "instance_seed": int(a.instance_seed), "episode": int(a.episode), "birth_hash": birth,
            "live_success": live_success, "episode_end": info.get("episode_end"), "truncated": bool(truncated),
            "infra_error": bool(info.get("infra_error")), "steps": int(steps), "sim_time_s": float(env.sim_time_s),
            "live_sha": env.state_sha(), "queries": len(chunks), "prompt": prompt, "server_metadata": _jsonable(meta),
            "build_s": round(build_s, 2), "wall_s": round(wall, 2),
            "ms_per_step_physics": round(1000 * t_step / max(1, steps), 2),
            "ms_per_query_observe": round(1000 * t_obs / max(1, len(chunks)), 2),
            "ms_per_query_infer": round(1000 * t_infer / max(1, len(chunks)), 2),
            "ms_per_step_total": round(1000 * wall / max(1, steps), 2), "cameras": conv.cameras,
            "convention": conv.name, "policy_hz": conv.policy_hz, "control_hz": conv.control_hz, "horizon": a.horizon}
    if task.suite == "puzzles":
        S = env.unwrapped.session
        st = S.status()
        live.update(review=st.get("review"), forcing=st.get("forcing"), instability=st.get("instability"))
        rows = S.record_stop()
        S.save_trajectory(rows, str(out / "trajectory.npz"), meta={"live_rows": int(len(rows)), "source": INTERFACE})
    else:
        env.save_trajectory(out / "trajectory.npz")
    np.savez(out / "chunks.npz", control_step=np.asarray(chunk_steps, dtype=np.int64),
             executed=np.asarray(executed, dtype=np.int64), length=np.asarray([len(c) for c in chunks], dtype=np.int64),
             actions=np.concatenate(chunks) if chunks else np.zeros((0, conv.action_dim), np.float32))
    (out / "live.json").write_text(json.dumps(live, indent=1, default=str))
    policy.close()
    env.close()
    print(json.dumps({k: live[k] for k in ("live_success", "episode_end", "steps", "live_sha", "wall_s")}), flush=True)
    return 0


def _jsonable(x):
    try:
        json.dumps(x)
        return x
    except TypeError:
        return json.loads(json.dumps(x, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))


# ------------------------------------------------------------------------------- the grader's fresh-process replay
def replay_worker(a) -> int:
    """environment/sim/replay.py of the standard tasks, through the API: the executed rows (with their tick ledger)
    replayed once from the frozen instance in this new process; grading=True after a live success."""
    out = Path(a.episode_dir)
    live = json.loads((out / "live.json").read_text())
    t0 = time.time()
    env = _make_env(a, "nyx" if a.video else "none")
    S, M = env.unwrapped.session, env.unwrapped.M
    acts, ticks = M.load_trajectory(str(out / "trajectory.npz"), S.task, S.action_dim, trusted=True, layout=S.action_layout)
    grading = bool(live["live_success"])
    video_error = None
    if a.video:
        try:
            S.start_video(str(out / "video.mp4"), fps=15)
        except Exception as exc:  # noqa: BLE001
            video_error = repr(exc)
    st = S.replay(acts, physics_steps=ticks, grading=grading)
    if a.video and video_error is None:
        try:
            S.stop_video()
        except Exception as exc:  # noqa: BLE001
            video_error = repr(exc)[:300]
    rec = {"replay_success": bool(st["success"]), "replay_sha": st["state_sha"], "replay_episode_end": st["episode_end"],
           "grading_replay": grading, "grading": st.get("grading"), "review": st.get("review"),
           "forcing": st.get("forcing"), "instability": st.get("instability"), "rows": int(len(acts)),
           "instance_exact": bool(S.instance_exact), "instance_matches": bool(S.instance_matches),
           "wall_s": round(time.time() - t0, 2), "video": bool(a.video and video_error is None
                                                                 and (out / "video.mp4").is_file()),
           "video_error": video_error}
    (out / "replay.json").write_text(json.dumps(rec, indent=1, default=str))
    env.close()
    print(json.dumps({k: rec[k] for k in ("replay_success", "replay_sha", "wall_s")}), flush=True)
    return 0


def verdict(live: dict, rep: dict | None, suite: str) -> dict:
    """The standard mode's rule (tests/grade.py of the standard tasks): live success AND replay success (puzzles), no
    infrastructure error (a simulator breakdown, live or replayed; a missing replay after a live success), and a success
    with deep task contact held for review. IKEA: the live episode's settled success."""
    infra = bool(live.get("infra_error"))
    if suite == "puzzles":
        replay_ok = bool(rep and rep.get("replay_success"))
        infra = infra or (rep or {}).get("replay_episode_end") == "simulator_instability"
        grader_error = bool(live.get("live_success")) and rep is None
        ok = bool(live.get("live_success")) and replay_ok and not infra and not grader_error
        review = bool(live.get("review")) or bool((rep or {}).get("review"))
    else:
        replay_ok, grader_error, review = None, False, False
        ok = bool(live.get("live_success")) and not infra
    pending = ok and review
    return {"success": int(ok and not pending), "success_pending_review": int(pending),
            "live_success": int(bool(live.get("live_success"))),
            "replay_success": None if replay_ok is None else int(replay_ok),
            "deterministic": None if rep is None else int(rep.get("replay_sha") == live.get("live_sha")),
            "infra_error": int(infra or grader_error), "grader_error": int(grader_error)}


# ------------------------------------------------------------------------------------------------------- the runner
def _seeds(text: str, task) -> list[int]:
    if text == "all":
        return list(task.seeds)
    seeds = [int(s) for s in text.split(",")]
    bad = [s for s in seeds if s not in task.seeds]
    if bad:
        raise SystemExit(f"{task.id} allows the instance seeds {list(task.seeds)}, not {bad}")
    return seeds


def _worker_cmd(a, worker: str, seed: int, k: int, edir: Path) -> list[str]:
    cmd = [sys.executable, "-m", "roboarc.vla.run", "--worker", worker, "--task", a.task, "--instance-seed", str(seed),
           "--episode", str(k), "--episode-dir", str(edir), "--rules", a.rules, "--renderer", a.renderer,
           "--image-size", str(a.image_size), "--horizon", str(a.horizon), "--wait-s", str(a.wait_s)]
    for flag, v in (("--policy", a.policy), ("--convention", a.convention), ("--policy-hz", a.policy_hz),
                    ("--cameras", a.cameras), ("--prompt", a.prompt), ("--max-steps", a.max_steps),
                    ("--time-limit-s", a.time_limit_s), ("--camera-poses", a.camera_poses)):
        if v is not None:
            cmd += [flag, str(v)]
    for flag, v in (("--full-prompt", a.full_prompt), ("--no-roboarc-obs", a.no_roboarc_obs), ("--video", a.video)):
        if v:
            cmd.append(flag)
    return cmd


def run(a) -> int:
    task = resolve(a.task)
    a.task = task.id
    conv_name = a.convention or ("aloha" if task.robot != "franka-robotiq-2f85" else "droid")
    a.horizon = a.horizon or DEFAULT_HORIZON[conv_name]
    if a.max_steps is None and not a.time_limit_s:
        a.max_steps = default_max_steps(task)
    root = Path(a.out) / task.id / a.run_id
    root.mkdir(parents=True, exist_ok=True)
    config = {"interface": INTERFACE, "kind": "vla", "task": task.id, "suite": task.suite, "robot": task.robot,
              "run_id": a.run_id, "policy": a.policy, "convention": conv_name, "horizon": a.horizon,
              "policy_hz": a.policy_hz, "image_size": a.image_size, "rules": a.rules, "renderer": a.renderer,
              "max_steps": a.max_steps, "time_limit_s": a.time_limit_s, "prompt": a.prompt or task_prompt(task, full=a.full_prompt),
              "roboarc_version": roboarc.__version__, "camera_poses": a.camera_poses, "image": task.image,
              "note": "a VLA evaluation through roboarc.vla: not a Harbor trial; never ranked with agent results"}
    config["config_key"] = config_key(config)
    old = json.loads((root / "config.json").read_text()) if (root / "config.json").is_file() else None
    if old is not None and any(old.get(k) != config[k] for k in COMPARABLE):
        raise SystemExit(f"{root} holds episodes of another configuration ("
                         + ", ".join(f"{k}: {old.get(k)!r} -> {config[k]!r}" for k in COMPARABLE if old.get(k) != config[k])
                         + "): use another --run-id")
    _write_json(root / "config.json", config)
    results = []
    for seed in _seeds(a.seed, task):
        for k in range(1, a.episodes + 1):
            edir = root / f"s{seed}" / f"e{k}"
            edir.mkdir(parents=True, exist_ok=True)
            res = {"task": task.id, "instance_seed": seed, "episode": k, "config_key": config["config_key"],
                   "run_id": a.run_id, "dir": str(edir)}
            with open(edir / "episode.log", "w") as log:
                rc = subprocess.call(_worker_cmd(a, "episode", seed, k, edir), stdout=log, stderr=subprocess.STDOUT)
            live = json.loads((edir / "live.json").read_text()) if (edir / "live.json").is_file() else None
            rep = None
            if live is None:
                res.update(success=0, infra_error=1, error=f"the episode process failed (rc {rc}): see episode.log")
            else:
                if task.suite == "puzzles":
                    with open(edir / "replay.log", "w") as log:
                        subprocess.call(_worker_cmd(a, "replay", seed, k, edir), stdout=log, stderr=subprocess.STDOUT)
                    rep = json.loads((edir / "replay.json").read_text()) if (edir / "replay.json").is_file() else None
                res.update(verdict(live, rep, task.suite))
                res.update({k2: live.get(k2) for k2 in ("episode_end", "steps", "sim_time_s", "queries", "birth_hash",
                                                        "live_sha", "wall_s", "ms_per_step_total", "ms_per_step_physics",
                                                        "ms_per_query_observe", "ms_per_query_infer",
                                                        "server_metadata")})
                if rep:
                    res.update(replay_sha=rep.get("replay_sha"), replay_wall_s=rep.get("wall_s"), video=rep.get("video"))
            _write_json(edir / "result.json", res)
            results.append(res)
            print(json.dumps({k2: res.get(k2) for k2 in ("instance_seed", "episode", "success", "live_success",
                                                          "replay_success", "steps", "episode_end", "wall_s")}), flush=True)
    every = [json.loads(p.read_text()) for p in sorted(root.glob("s*/e*/result.json"))]   # this and earlier runs
    summary = {"task": task.id, "run_id": a.run_id, "config_key": config["config_key"], "episodes": len(every),
               "successes": sum(int(r.get("success") or 0) for r in every),
               "pending_review": sum(int(r.get("success_pending_review") or 0) for r in every),
               "infra_errors": sum(int(r.get("infra_error") or 0) for r in every), "results": every}
    _write_json(root / "results.json", summary)
    return 0


def export_reference(a) -> int:
    """The task's reference (the standard mode's, or --reference-mode privileged) in the convention's action units:
    {"actions": (T, D) float64}, what `stub_server --policy replay` serves."""
    from ..references import load_reference
    task = resolve(a.task)
    a.task = task.id
    rows = load_reference(task.id, a.reference_mode) if task.suite == "puzzles" else load_reference(task.id)
    if rows is None:
        raise SystemExit(f"{task.id} has no reference")
    env = _make_env(a, "none")
    conv = make_convention(env, a.convention, policy_hz=a.policy_hz)
    acts = conv.from_rows(np.asarray(rows, dtype=np.float64))
    back = conv.to_rows(acts)
    exact = bool(np.array_equal(back.astype(rows.dtype), rows))
    np.savez(a.export_reference, actions=acts, task=task.id, convention=conv.name, rows_exact=exact)
    env.close()
    print(json.dumps({"task": task.id, "convention": conv.name, "rows": int(len(acts)), "dim": int(acts.shape[1]),
                      "round_trip_exact": exact, "out": a.export_reference}))
    return 0 if exact else 1


def main(argv=None, prog: str = "python -m roboarc.vla.run") -> int:
    """`prog`: the command's name in its help (`roboarc vla run` is this function under another name)."""
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.replace("python -m roboarc.vla.run", prog),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="a task id or alias: roboarc-orbits-02-i01, orbits-02, lack")
    ap.add_argument("--policy", help="the openpi policy server: ws://host:port or host:port (API key: OPENPI_API_KEY)")
    ap.add_argument("--seed", default="0", help="instance seeds: 0, 0,1,2 or all (the task's allowed seeds)")
    ap.add_argument("--episodes", type=int, default=1, help="episodes per seed")
    ap.add_argument("--out", help="results root (outside the repository; never Harbor's results directory)")
    ap.add_argument("--run-id", default="vla-openpi", help="the policy's name in the results tree")
    ap.add_argument("--convention", choices=CONVENTIONS, help="default: aloha for two arms, droid for one")
    ap.add_argument("--horizon", type=int, help="actions executed per chunk (default: aloha 25, droid 8)")
    ap.add_argument("--policy-hz", type=float, help="the chunk's action rate (default: the control rate, 40 Hz: one "
                                                    "action per control step; droid-velocity: 15)")
    ap.add_argument("--image-size", type=int, default=224, help="the square image size the policy gets, in pixels: "
                    "each camera image is resized with padding (default: 224)")
    ap.add_argument("--cameras", help="openpi key=camera,... (default: the convention's)")
    ap.add_argument("--camera-poses", choices=("droid", "molmoact2"), help="droid: DROID's camera poses on the one-arm "
                    "puzzles; molmoact2: the same with agentview aimed as in MolmoAct2's simulation (droid_cameras.py; "
                    "default: the task's own cameras)")
    ap.add_argument("--max-steps", type=int, help="control steps per episode (default: 2 x the reference's)")
    ap.add_argument("--time-limit-s", type=float, help="simulated seconds per episode instead of --max-steps")
    ap.add_argument("--rules", default="standard", choices=("standard", "privileged"),
                    help="puzzles: the joint-speed rules the steps run under (default: standard)")
    ap.add_argument("--renderer", default="nyx", choices=("nyx", "mujoco", "none"),
                    help="nyx: the benchmark's images; mujoco: a raster for debugging; none: black images (testing "
                         "the loop only)")
    ap.add_argument("--prompt", help="override the prompt (default: the first sentence of the task text)")
    ap.add_argument("--full-prompt", action="store_true", help="the whole task paragraph as the prompt")
    ap.add_argument("--no-roboarc-obs", action="store_true", help="do not add obs['roboarc'] (control step, task, seed)")
    ap.add_argument("--video", action="store_true", help="render video.mp4 during the grading replay (GPU)")
    ap.add_argument("--wait-s", type=float, default=300.0, help="how long to wait for the policy server")
    ap.add_argument("--export-reference", metavar="NPZ", help="write the task's reference in the convention's units")
    ap.add_argument("--reference-mode", default="standard", choices=("standard", "privileged"),
                    help="with --export-reference: which mode's reference (default: standard)")
    ap.add_argument("--worker", choices=("episode", "replay"), help=argparse.SUPPRESS)
    ap.add_argument("--instance-seed", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--episode", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--episode-dir", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.worker == "episode":
        return episode_worker(a)
    if a.worker == "replay":
        return replay_worker(a)
    if a.export_reference:
        return export_reference(a)
    if not a.policy or not a.out:
        ap.error("--policy and --out are needed (or --export-reference)")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
