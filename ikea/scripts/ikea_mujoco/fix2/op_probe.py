#!/usr/bin/env python3
"""Live operation metrics of a reference replay, every --every rows in [--from, --to] (diagnosis of a reference that
fails with the prism rule). eket: the operation tracker's status (current operation, gate metrics); ikea365: the screw
metrics of the current operation (gap / lateral / tilt in member_a's frame, turns, windowed contact forces); any
other scene: the monitor's failures / joints and the parts' positions.

usage: python op_probe.py --task eket|ikea365 --scene-dir D --traj F --from R0 --to R1 [--every 250] > out.jsonl
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
import rcb_ikea  # noqa: E402

ap = argparse.ArgumentParser()
for k in ("--task", "--scene-dir", "--traj"):
    ap.add_argument(k, required=True)
ap.add_argument("--from", dest="r0", type=int, required=True)
ap.add_argument("--to", dest="r1", type=int, required=True)
ap.add_argument("--every", type=int, default=250)
a = ap.parse_args()
s = rcb_ikea.Session(a.task, a.scene_dir)
ctrl = rcb_ikea.load_trajectory(a.traj, s.nu, s.timestep)
M = s.S.mech


def probe():
    if a.task == "eket":
        st = M.tracker.status()
        return dict(current=st.get("current"), completed=st.get("completed"),
                    metrics={k: v for k, v in (st.get("current_metrics") or {}).items()
                             if k in ("position_error_m", "angle_error_deg", "relative_speed_m_s", "dwell_updates",
                                      "contact_hits", "progress_m")})
    if a.task == "ikea365":
        from ikea_runtime import rotate, axis_angle_deg
        p, q, speed = M.rel("same_screw")
        delta = p - M.screw_rel
        return dict(op=M.ops[M.next] if not M.complete else "complete", gap_mm=1e3 * float(delta[2]),
                    lat_mm=1e3 * float(np.linalg.norm(delta[:2])),
                    tilt_deg=axis_angle_deg(rotate(q, [0, 0, 1]), [0, 0, 1]), speed=speed,
                    turns=M.op.get("turns"), fa=M.contact_force("same_screw", "member_a"),
                    fb=M.contact_force("same_screw", "member_b"), dwell=M.op.get("dwell"))
    rep = s.success()                                  # other scenes: the monitor's verdict and the parts' poses
    parts = list(s.S.task.get("parts", []))
    return dict(failures=rep.get("failures"), joints=rep.get("joints"),
                pos={p: [round(float(x), 4) for x in s.data.xpos[s.model.body(p).id]] for p in parts},
                top_tilt_deg=rep.get("top_tilt_deg"))


for i, row in enumerate(ctrl[:a.r1], 1):
    s.step(row)
    if i >= a.r0 and (i - a.r0) % a.every == 0:
        print(json.dumps(dict(row=i, **probe()), default=float), flush=True)
