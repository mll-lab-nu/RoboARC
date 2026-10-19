"""Replay action rows open-loop in a fresh MuJoCo Session of an instance and record, per row: the measured arm joints,
the TCP pose (hand + 0.1029 m), the wand pose, the grasp state, success / episode end, and the task's metrics every N
rows. usage: record.py <instance.json> <rows.npz | nop:N> <out.npz> [--every 10] [--hold 0]
Writes <out>.npz and <out>.json (success, first success row, end, state hashes, grasp events)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import rdreg

ap = argparse.ArgumentParser()
ap.add_argument("instance"); ap.add_argument("rows"); ap.add_argument("out")
ap.add_argument("--every", type=int, default=10); ap.add_argument("--hold", type=int, default=0)
ap.add_argument("--max-rows", type=int, default=None)
a = ap.parse_args()
rdreg.register()
from rcb_topogym import Session
from scipy.spatial.transform import Rotation as R
t0 = time.time()
s = Session("buzz_wire_3d", instance=a.instance, strict=True)
import os
from topogym.planning.buzz_wire_3d.redesign.io import set_springref
SPRINGREF = os.environ.get("BUZZ_SPRINGREF")
if SPRINGREF not in (None, "", "none"):
    set_springref(s.env._world.model, float(SPRINGREF))
w = s.env._world
m, d = w.model, w.data
qadr = np.array([m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 8)])
hand, wand = m.body("hand").id, m.body("wand::wand").id
if a.rows.startswith("nop:"):
    rows = np.repeat(s.hold_action()[None], int(a.rows[4:]), axis=0).astype(np.float32)
else:
    rows = np.load(a.rows)["actions"].astype(np.float32)
if a.max_rows:
    rows = rows[:a.max_rows]
rec = {k: [] for k in ("q", "tcp_p", "tcp_q", "wand_p", "wand_q", "held", "grip_cmd")}
metrics, first_success, end_row = [], None, None


def sample():
    rec["q"].append(np.array(d.qpos[qadr]))
    Rh = d.xmat[hand].reshape(3, 3)
    rec["tcp_p"].append(d.xpos[hand] + Rh @ [0, 0, .1029])
    rec["tcp_q"].append(R.from_matrix(Rh).as_quat()[[3, 0, 1, 2]])
    rec["wand_p"].append(np.array(d.xpos[wand])); rec["wand_q"].append(np.array(d.xquat[wand]))
    st = w.read_state().get("grasp") or {}
    rec["held"].append(bool(st.get("active")))


sample(); rec["grip_cmd"].append(0.0)
n = 0
all_rows = list(rows) + [s.hold_action()] * a.hold
for i, row in enumerate(all_rows):
    done = s.step(row)
    n += 1
    sample(); rec["grip_cmd"].append(float(row[7]))
    if s.success() and first_success is None:
        first_success = i
    if i % a.every == 0 or done:
        mm = s.env._task.result.metrics
        metrics.append(dict(row=i, **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in mm.items()
                                    if k in ("route_cursor_m", "traversal_tracked", "traversal_complete", "distance_to_finish_m",
                                             "finish_angle_deg", "supported", "released", "quiet", "stable_time_s")}))
    if done:
        end_row = i
        break
st = s.status()
np.savez_compressed(a.out, **{k: np.asarray(v) for k, v in rec.items()}, rows=np.asarray(all_rows[:n], np.float32))
res = dict(instance=a.instance, springref=SPRINGREF, rows=a.rows, n_rows=n, success=st["success"], first_success_row=first_success,
           episode_end=st["episode_end"], end_row=end_row, state_sha=st["state_sha"], wall_s=round(time.time() - t0, 1),
           metrics=metrics[-50:])
Path(a.out).with_suffix(".json").write_text(json.dumps(res, default=str, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "metrics"}, default=str))
s.close()
