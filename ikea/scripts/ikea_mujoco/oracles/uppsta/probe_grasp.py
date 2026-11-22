#!/usr/bin/env python3
"""Study: a grasp in detail. Replay to --from (clamp on), then log every --every rows: the part's world pose, the hand's
fingertip (pad) positions in the part frame, the gripper ctrl / tendon width and the finger contacts (normal force).
  python probe_grasp.py --traj X.npz --part dowel_3 --hand right --from 12600 --to 13700 --every 20 [--noprism]"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True); ap.add_argument("--part", required=True); ap.add_argument("--hand", required=True)
ap.add_argument("--from", dest="r0", type=int, required=True); ap.add_argument("--to", dest="r1", type=int, required=True)
ap.add_argument("--every", type=int, default=20); ap.add_argument("--noprism", action="store_true")
ap.add_argument("--rcb-ikea", default=os.path.join(HERE, "..", "..", "shared"))
a = ap.parse_args()
sys.path.insert(0, a.rcb_ikea)
import rcb_ikea as R, mujoco
if a.noprism:
    sys.path.insert(0, os.path.join(R.SCENES_ROOT, "uppsta")); import ikea_runtime; ikea_runtime.cylinder_prisms = lambda spec: []
s = R.Session("uppsta"); m, d = s.model, s.data
c = np.load(a.traj)["ctrl"]
bp = m.body(a.part).id
fl, fr = m.body(f"{a.hand}/left_finger").id, m.body(f"{a.hand}/right_finger").id
tcp = m.site(f"{a.hand}/tcp").id
gi = [i for i, n in enumerate(s.actuators) if n == f"{a.hand}/actuator8"][0]
s.run(c[:a.r0])
f6 = np.zeros(6)
for i in range(a.r0, a.r1):
    s.step(c[i])
    if (i + 1) % a.every:
        continue
    Rp = d.xmat[bp].reshape(3, 3)
    rel = lambda x: [round(1000 * v, 2) for v in Rp.T @ (x - d.xpos[bp])]
    con = []
    for k in range(d.ncon):
        g1, g2 = d.contact.geom1[k], d.contact.geom2[k]
        b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
        if {b1, b2} & {fl, fr} and (m.body_rootid[b1] == bp or m.body_rootid[b2] == bp or b1 == bp or b2 == bp
                                    or m.body_parentid[b1] == bp or m.body_parentid[b2] == bp):
            mujoco.mj_contactForce(m, d, k, f6)
            n = d.contact.frame[k][:3]
            con.append(f"{m.body(b1).name[-6:]}-{m.body(b2).name[-6:]} d{-1000*d.contact.dist[k]:.2f} F{f6[0]:.1f} n{[round(float(x),2) for x in Rp.T @ n]}")
    print(json.dumps(dict(row=i + 1, part_xyz_mm=[round(1000 * v, 2) for v in d.xpos[bp]], part_quat=[round(float(v), 4) for v in d.xquat[bp]],
                          tcp=rel(d.site_xpos[tcp]), lf=rel(d.xpos[fl]), rf=rel(d.xpos[fr]), g=round(float(d.ctrl[gi]), 1),
                          con=con[:10])), flush=True)
