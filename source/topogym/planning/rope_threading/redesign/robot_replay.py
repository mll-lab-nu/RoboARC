"""Robot-level MuJoCo replays on a redesigned tier's frozen instance (the cloud host): rows open loop in a Session, as the graders
replay them, with the task's own metrics sampled along the way.

  old relay   the strict reference's recorded rows (the family's feedback relay run live on MuJoCo) as-is on the new
              scene: does the old oracle solve the new task? closest approach = the longest correct prefix of the
              required crossing sequence reached at any sample
  nop         the home pose held (the task must not be solved by doing nothing)

usage: python -m topogym.planning.rope_threading.redesign.robot_replay <tier> <instance.json> <rows.npz|nop:N> <out.json>
       [--every 25] [--hold 100]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _prefix(seq, required):
    n = 0
    for a, b in zip(seq, required):
        if a != b:
            break
        n += 1
    return n


def replay(tier, instance, rows_spec, out, every=25, hold=100):
    import rcb_topogym as rt
    from .session_build import register
    register(tier)
    s = rt.Session("rope_threading", instance=instance, strict=False)
    inst = json.loads(Path(instance).read_text())
    if rows_spec.startswith("nop:"):
        rows = np.repeat(np.asarray(inst["reset_joint_targets"], np.float32)[None], int(rows_spec[4:]), axis=0)
        source = "nop (home targets held)"
    else:
        data = np.load(rows_spec)
        rows = data["actions"].astype(np.float32)
        if "action_layout" in data.files and list(data["action_layout"]) != list(s.action_layout):
            raise ValueError("row layout differs from the instance's action layout")
        source = rows_spec
    rows = np.concatenate([rows, np.repeat(rows[-1:], hold, axis=0)])
    required = [f"{n}{'+' if sg > 0 else '-'}" for n, sg in s.env._task.layout.required_crossings()]
    log, best, t0 = [], dict(prefix=0, sequence=[], row=None), time.time()
    done_row = None
    for i, row in enumerate(rows):
        done = s.step(row)
        if i % every == 0 or done:
            m = dict(s.env._task.result.metrics)
            seq = [f"{a}{'+' if b > 0 else '-'}" for a, b in m.get("crossing_sequence", [])]
            k = _prefix(seq, required)
            log.append(dict(row=i, sequence=seq, prefix=k, quiet=m.get("quiet"), thickness_clear=m.get("thickness_clear")))
            if k > best["prefix"] or (k == best["prefix"] and len(seq) > len(best["sequence"])):
                best = dict(prefix=k, sequence=seq, row=i)
        if done:
            done_row = i
            break
    st = s.status()
    rep = dict(tier=tier, instance=instance, rows=source, n_rows=int(len(rows)), hold_rows=hold,
               success=bool(st["success"]), episode_end=st["episode_end"], done_row=done_row, required=required,
               state_sha=st.get("state_sha"), sim_steps=st.get("sim_steps"),
               closest_approach=best, final=log[-1] if log else None, review=st.get("review"), forcing=st.get("forcing"),
               wall_s=round(time.time() - t0, 1), ms_per_control=round(1000 * (time.time() - t0) / max(1, len(log) and (log[-1]["row"] + 1)), 1),
               log=log)
    s.close()
    Path(out).write_text(json.dumps(rep, indent=1, default=str))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tier")
    ap.add_argument("instance")
    ap.add_argument("rows")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--hold", type=int, default=100)
    a = ap.parse_args()
    rep = replay(a.tier, a.instance, a.rows, a.out, a.every, a.hold)
    print(json.dumps({k: v for k, v in rep.items() if k != "log"}, default=str))
