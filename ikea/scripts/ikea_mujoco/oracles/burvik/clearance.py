"""Kinematic clearance of the left hand (hand, fingers, link5-7) to the world fixtures over one turn of j7, at a builder
snapshot's pose (a copy of the state; nothing is stepped).
  python clearance.py <code root> <scenes root> <snap.pkl> [exclude_geom ...]"""
import sys, os, pickle
import numpy as np
import mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orc import Orc, setup

code, scenes, snap = sys.argv[1:4]
excl = set(sys.argv[4:])
o = Orc(setup(code, scenes)); m, d = o.m, o.d
o.restore(pickle.load(open(snap, "rb"))["o"]); mujoco.mj_forward(m, d)
j7 = int(m.actuator_trnid[o.arm["left"][6], 0]); qa = int(m.jnt_qposadr[j7])
hand = [g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name in
        ("left_hand", "left_left_finger", "left_right_finger", "left_link7", "left_link6", "left_link5") and (m.geom_contype[g] | m.geom_conaffinity[g])]
world = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == 0 and (m.geom_contype[g] | m.geom_conaffinity[g])
         and m.geom(g).name not in excl | {"floor"}]
print("tcp", np.round(d.site_xpos[o.site["left"]], 3).tolist(), "q7", round(float(d.qpos[qa]), 3))
d2 = mujoco.MjData(m); d2.qpos[:] = d.qpos; fr = np.zeros(6); q0 = float(d2.qpos[qa])
for deg in range(0, 360, 10):
    d2.qpos[qa] = q0 + np.radians(deg); mujoco.mj_kinematics(m, d2)
    best = (9., None, None)
    for g in hand:
        for w in world:
            dist = mujoco.mj_geomDistance(m, d2, g, w, 0.05, fr)
            if dist < best[0]:
                best = (dist, m.body(int(m.geom_bodyid[g])).name, m.geom(w).name)
    print(f"{deg:4d} deg  {1000 * best[0]:7.1f} mm  {best[1]} - {best[2]}")
