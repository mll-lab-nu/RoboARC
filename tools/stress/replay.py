"""Replay one action stream on a frozen MuJoCo instance through the production Session, privileged mode: Session.step,
with its joint-limit clip, joint-speed clamp and table guard, which is the agent's own command path. The stress
observer (observer.py) watches. Writes one result JSON, and optionally a per-tick trace.

usage: replay.py --instance <instance.json> --actions <npz | nop:N> --out <result.json> [--hold 40] [--trace <trace.json.gz>]
                 [--id <case id>] [--no-observer] [--experiment-opt "ccd_iterations=500,ccd_tolerance=1e-10"]
--hold N         N control steps holding the birth targets after the stream (a breakdown can come after the input ends)
--experiment-opt forensics only, never a verdict: MuJoCo options set on the built model before the first step; recorded
The stream stops where the episode ends (the task's success or failure predicate, or the Session's breakdown rule), as
it would for an agent: later rows never reach the physics.
A recorded trajectory that carries an executed tick ledger (physics_steps_per_action: a reference or any recording) is
loaded and replayed exactly as Session.replay / the verifier do: rcb_topogym.load_trajectory(trusted=True) (its live_rows
cut, its checks) and each row stepped with its own ticks; streams without a ledger step every row with the default ticks.
Before this (validate, 2026-10-13) the ledger was ignored: reference checks run earlier on a reference whose ledger has a
non-default tick count on a row other than the last executed one are void (knot routes: mixed 3/4/5 ticks)."""
import _env  # noqa: F401  (before numpy: SIMD masking, threads, sys.path)
import argparse, gzip, hashlib, json, os, sys, time, traceback
from pathlib import Path

import numpy as np


def load_actions(path):
    with np.load(path, allow_pickle=False) as z:
        key = "actions" if "actions" in z.files else z.files[0]
        return np.asarray(z[key], dtype=np.float32)


def ledger(path):
    """True when the npz carries an executed tick ledger (then the verifier's loader reads it)."""
    with np.load(path, allow_pickle=False) as z:
        return "physics_steps_per_action" in z.files


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def apply_options(session, experiment):
    if not experiment:
        return
    m = session.env._world.model
    for item in experiment.split(","):
        k, v = item.split("=")
        setattr(m.opt, k.strip(), type(getattr(m.opt, k.strip()))(float(v)))


def run(instance, actions, *, hold=40, trace=None, observe=True, case_id=None, experiment=None):
    from rcb_topogym import Session
    inst = json.loads(Path(instance).read_text())
    t0 = time.time()
    s = Session(inst["task"], instance=str(instance), strict=True)
    build_s = time.time() - t0
    apply_options(s, experiment)
    obs = None
    if observe:
        from observer import Observer
        obs = Observer(s, trace=trace is not None)
    actions_sha, ticks = None, None
    if isinstance(actions, str) and actions.startswith("nop:"):       # nop: hold the birth targets N control steps
        rows = np.repeat(s.hold_action()[None].astype(np.float32), int(actions[4:]), axis=0)
    elif isinstance(actions, (str, Path)) and ledger(actions):        # a recording: the verifier's loader and ledger
        from rcb_topogym import load_trajectory
        actions_sha = sha256(actions)
        rows, ticks = load_trajectory(str(actions), s.task, s.action_dim, trusted=True, layout=s.action_layout)
        rows = np.asarray(rows, np.float32)
    else:
        actions_sha = sha256(actions) if isinstance(actions, (str, Path)) else None   # as loaded: provenance
        rows = load_actions(actions) if isinstance(actions, (str, Path)) else np.asarray(actions, np.float32)
    if rows.ndim != 2 or rows.shape[1] != s.action_dim:
        raise ValueError(f"actions {rows.shape} do not fit action_dim {s.action_dim}")
    fault, ended_at, executed, held = None, None, 0, 0
    t1 = time.time()
    try:
        for i, row in enumerate(rows):
            executed += 1
            if s.step(row, _physics_steps=None if ticks is None else ticks[i]):
                ended_at = i
                break
        if ended_at is None:
            for _ in range(hold):
                held += 1
                if s.step(s.hold_action()):
                    break
    except Exception as exc:
        fault = dict(error=repr(exc)[:400], trace=traceback.format_exc()[-1500:],
                     cause=repr(exc.__cause__)[:400] if exc.__cause__ else None, row=executed - 1 if not held else None,
                     hold_row=held - 1 if held else None)
    wall = time.time() - t1
    try:
        st = s.status()
    except Exception as exc:
        st = dict(status_error=repr(exc)[:300])
    try:
        safety = s.safety(episode=True)
    except Exception as exc:
        safety = dict(error=repr(exc)[:300])
    rec = dict(id=case_id, instance=str(instance), instance_sha=s.instance_sha,
               actions=str(actions) if not isinstance(actions, np.ndarray) else "array", actions_sha256=actions_sha,
               rows=int(len(rows)), tick_ledger=ticks is not None, executed=executed, held=held, ended_at_row=ended_at,
               control_steps=int(s.n_steps), physics_ticks=int(getattr(s, "physics_steps", 0)),
               build_s=round(build_s, 2), wall_s=round(wall, 2), ms_per_control=round(1000 * wall / max(1, s.n_steps), 2),
               slew_limited_steps=int(getattr(s, "slew_count", 0)),
               fault=fault, episode_end=st.get("episode_end"), success=st.get("success"), state_sha=st.get("state_sha"),
               instability=st.get("instability"), review=st.get("review"),
               forcing=(st.get("forcing") or None) and {k: st["forcing"].get(k) for k in ("count", "longest_s") if k in st["forcing"]},
               safety=safety, experiment_opt=experiment, provenance=_env.provenance(),
               joint_speed_limits={k: list(v) for k, v in __import__("topogym.world.mujoco.robot", fromlist=["x"]).JOINT_SPEED_LIMITS_RAD_S.items()},
               observer=obs.summary() if obs else None)
    if obs is not None and trace is not None:
        Path(trace).parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(trace, "wt") as f:
            json.dump(obs.trace, f)
    if obs is not None:
        obs.uninstall()
    s.close()
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--instance", required=True)
    ap.add_argument("--actions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hold", type=int, default=40)
    ap.add_argument("--trace")
    ap.add_argument("--id")
    ap.add_argument("--no-observer", action="store_true")
    ap.add_argument("--experiment-opt", default=os.environ.get("STRESS_EXPERIMENT_OPT"))
    a = ap.parse_args()
    rec = run(a.instance, a.actions, hold=a.hold, trace=a.trace, observe=not a.no_observer, case_id=a.id,
              experiment=a.experiment_opt)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(a.out + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    os.replace(tmp, a.out)
    o = rec["observer"] or {}
    pk = {k: round(v["value"], 4) for k, v in (o.get("peaks") or {}).items()}
    print(json.dumps(dict(id=rec["id"], end=rec["episode_end"], fault=bool(rec["fault"]), steps=rec["control_steps"],
                          ms=rec["ms_per_control"], counts=o.get("counts"), peaks=pk)), flush=True)


if __name__ == "__main__":
    main()
