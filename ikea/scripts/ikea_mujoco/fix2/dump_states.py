#!/usr/bin/env python3
"""Replay a reference like the grader and save the integration state at chosen rows (for render.py on a GPU host).

usage: python dump_states.py --task T --scene-dir D --traj F --rows 0,9000,12000 --out S.npz [--stop-after-last]
"""
import argparse, os, sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
import rcb_ikea  # noqa: E402

ap = argparse.ArgumentParser()
for k in ("--task", "--scene-dir", "--traj", "--rows", "--out"):
    ap.add_argument(k, required=True)
a = ap.parse_args()
want = sorted({int(x) for x in a.rows.split(",")})
s = rcb_ikea.Session(a.task, a.scene_dir)
ctrl = rcb_ikea.load_trajectory(a.traj, s.nu, s.timestep)
spec = mujoco.mjtState.mjSTATE_INTEGRATION
got, states = [], []


def grab(i):
    st = np.zeros(mujoco.mj_stateSize(s.model, spec))
    mujoco.mj_getState(s.model, s.data, st, spec)
    got.append(i)
    states.append(st)


if 0 in want:
    grab(0)
for i, row in enumerate(ctrl[:max(want)], 1):
    s.step(row)
    if i in want:
        grab(i)
np.savez_compressed(a.out, rows=np.array(got), state=np.array(states), task=a.task)
print("saved", got)
