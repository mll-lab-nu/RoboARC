"""Where a fiber stream loses its passage: replay it and, every N rows from a start row, print each strand's passage
stages (stations before / in / through the ring plane), its tip relative to the ring (along the axis, off the axis,
mm), the front, and the redesign terms. Run it through run_registered.py for a redesign instance.

usage: passtrace.py <instance.json> <actions.npz> <row0> [--every N]
"""
import argparse
import json

import numpy as np
from rcb_topogym import Session
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument("instance"); ap.add_argument("actions"); ap.add_argument("row0", type=int)
ap.add_argument("--every", type=int, default=20)
a = ap.parse_args()
s = Session("fiber_bundling", instance=a.instance, strict=True)
task, w = s.env._task, s.env._world
rows = np.asarray(np.load(a.actions)["actions"], np.float32)
for n, row in enumerate(rows):
    ended = s.step(row)
    if n >= a.row0 and (n % a.every == 0 or ended or n == len(rows) - 1):
        st = w.read_state()
        ring = st["bodies"]["bundling_ring"]
        c = np.asarray(ring["position_m"]); q = np.asarray(ring["quaternion_wxyz"])
        ax = Rotation.from_quat(q[[1, 2, 3, 0]]).apply([0., 0., 1.])
        ts = task.task_state or {}
        out = []
        for path, hist in zip(task._strand_points(st), ts.get("strands", [])):
            tip = path[0] - c
            out.append(dict(stages=np.bincount(np.asarray(hist["stages"]), minlength=3).tolist(),
                            tip_ax=round(1e3 * float(tip @ ax), 1), tip_off=round(1e3 * float(np.linalg.norm(tip - (tip @ ax) * ax)), 1)))
        f = task._front_point(st) - c
        m = getattr(task.result, "metrics", None) or {}
        print(json.dumps(dict(row=n, end=s.episode_end, front_ax=round(1e3 * float(f @ ax), 1),
                              front_stage=np.asarray(ts.get("front", {}).get("stages", [])).tolist(), strands=out,
                              complete=m.get("strands_complete"), lay=(m.get("lay") or {}).get("angle_deg"))), flush=True)
    if ended:
        break
