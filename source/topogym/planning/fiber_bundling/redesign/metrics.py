"""Trajectory complexity of a fiber action stream (an old -i01 oracle or a new oracle's rows) with the redesign
tooling's metric library (antihack.trajectory.metrics_from_paths): the Panda's commanded TCP path by forward
kinematics of each row's arm targets (the planner's own TCP: hand link + tool offset), as the tooling's "right" arm
with the left arm parked; closedness from the native gripper column (0 open .. 0.8 closed), holding from the
commanded closure. No dynamics. Run it through run_registered.py for a redesign instance.

usage: metrics.py <instance.json> <actions.npz> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np
from rcb_topogym import Session
from antihack.trajectory import metrics_from_paths, summary
from topogym.planning.fiber_bundling.kinematics import FiberIK

GRIP_CLOSED = 0.8          # the native Robotiq column: 0 open, 0.8 closed (the references' grip)

inst, actions, out = sys.argv[1:4]
s = Session("fiber_bundling", instance=inst, strict=True)
kin = FiberIK(s.env)
rows = np.asarray(np.load(actions)["actions"], float)
q = kin.seed.copy()
P = np.zeros((len(rows), 2, 3))
for i, row in enumerate(rows):
    q[kin.arm_qindices] = row[:7]
    P[i, 1] = kin.tcp_pose(q)[0]
P[:, 0] = P[0, 1] + np.array([0., 0.5, 0.])             # the parked "left" arm of the two-arm metric
closed = np.zeros((len(rows), 2))
closed[:, 1] = np.clip(rows[:, 7] / GRIP_CLOSED, 0., 1.)
dt = float(s.env._world.physics_dt_s) * 10               # one control step (25 ms)
m = metrics_from_paths(P, dt, closedness=closed)
res = dict(instance=inst, actions=actions, rows=len(rows), control_dt_s=dt, trajectory=summary(m))
Path(out).write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res["trajectory"], default=float))
