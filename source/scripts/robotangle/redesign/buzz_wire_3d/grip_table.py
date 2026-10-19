"""The Robotiq's settled four-bar per pad gap in the live Session physics (springref as the Session loads it, the head's
2.62): the arm held at HOME, the gripper commanded to each value closing on nothing, settled 2 s; prints a GRIP_TABLE
(pad gap, FOUR_BAR joint values) for kin.py. usage: grip_table.py <tier>"""
import json, sys
import numpy as np
import mujoco
import rdreg
vids = rdreg.register()
from rcb_topogym import Session
from topogym.planning.buzz_wire_3d.redesign.kin import FOUR_BAR, HOME
s = Session("buzz_wire_3d", variation_id=vids[sys.argv[1]], seed=0, gripper_space="native", strict=False)
m, d = s.env._world.model, s.env._world.data
adr = [m.jnt_qposadr[m.joint(n).id] for n in FOUR_BAR]
gl = [g for g in range(m.ngeom) if m.geom_contype[g] and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "") == "left_finger"]
gr = [g for g in range(m.ngeom) if m.geom_contype[g] and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "") == "right_finger"]
spring = [float(m.qpos_spring[m.jnt_qposadr[j]]) for j in range(m.njnt) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").endswith("spring_link_joint")]
q0 = np.array(s.observe()["robot"]["arm"]["joint_position"])[:7] if False else None
arm = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 8)]
qa = np.array(d.qpos[arm])
rows = []
fromto = np.zeros(6)
for c in np.round(np.arange(0.0, 0.801, 0.05), 3):
    for _ in range(80):
        s.step(np.r_[qa, c].astype(np.float32))
    gap = min(mujoco.mj_geomDistance(m, d, a, b, 0.2, fromto) for a in gl for b in gr)
    rows.append((round(float(gap), 5), [round(float(d.qpos[i]), 5) for i in adr], float(c)))
    print(json.dumps(rows[-1]), flush=True)
print("SPRING", spring)
print("GRIP_TABLE_SR262 = " + json.dumps([(g, v) for g, v, c in sorted(rows, key=lambda r: -r[0])]))
s.close()
