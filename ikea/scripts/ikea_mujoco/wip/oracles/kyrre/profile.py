#!/usr/bin/env python3
"""Per-block profile of a reference replay: wall time, contacts, robot-part contact, part speed, welds, arm speed use,
plus the full qpos / qvel / eq_active at every block end (the demo's track for a time-warp search).
usage: profile.py TASK REF.npz OUT.npz [--block 400]"""
import os, sys, time, json
sys.path.insert(0, os.environ["RCB_SHARED"])
import numpy as np, mujoco
import rcb_ikea as R
task, ref, out = sys.argv[1:4]
B = int(sys.argv[sys.argv.index("--block") + 1]) if "--block" in sys.argv else 400
s = R.Session(task)
ctrl = R.load_trajectory(ref, s.nu, s.timestep)
v = s.validity
recs, QP, QV, EQ = [], [], [], []
eq = s.data.eq_active
lim = s.step_limit; fin = np.isfinite(lim)
prev = s.data.ctrl.copy()
T0 = time.time()
for b0 in range(0, len(ctrl), B):
    t0 = time.perf_counter(); ncon = 0; rp = 0; ratio = 0.
    for row in ctrl[b0:b0 + B]:
        ratio = max(ratio, float((np.abs(row - prev)[fin] / lim[fin]).max())); prev = row
        s.step(row); ncon += int(s.data.ncon)
        rp += bool(v.robot_part_contacts())
    w = time.perf_counter() - t0
    lin, ang, worst = v.part_speeds()
    e = min(b0 + B, len(ctrl))
    recs.append((e, w, ncon / B, rp / B, lin, ang, int(eq.sum()), ratio, float(ctrl[e - 1, 7]), float(ctrl[e - 1, 15]),
                 float(bool(s.success().get("success")))))
    QP.append(s.data.qpos.copy()); QV.append(s.data.qvel.copy()); EQ.append(s.data.eq_active.copy())
    if (b0 // B) % 50 == 0:
        print(json.dumps(dict(row=e, wall=round(time.time() - T0, 1), wall_blk=round(w, 3), ncon=round(ncon / B, 1), rp=round(rp / B, 2), welds=int(eq.sum()))), flush=True)
    if (b0 // B) % 200 == 0:
        np.savez(out + ".part.npz", rec=np.array(recs), qpos=np.array(QP), qvel=np.array(QV), eq=np.array(EQ))
fc = s.final_check()
np.savez(out, rec=np.array(recs), cols=np.array("row wall ncon robot_part lin ang welds ratio gl gr goal".split()),
         qpos=np.array(QP), qvel=np.array(QV), eq=np.array(EQ), block=B)
print("done", round(time.time() - T0, 1), json.dumps(dict(success=fc["success"], sha=fc["state_sha"], settled=fc["settled_sha"])))
