import os, sys, json
sys.path.insert(0, os.environ["RCB_SHARED"])
import numpy as np, mujoco, rcb_ikea as R
task, track, part, r0, r1 = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
step = int(sys.argv[6]) if len(sys.argv) > 6 else 2000
s = R.Session(task); m = s.model
T = np.load(track); B = 400; Q = T["qpos"]; rec = T["rec"]
j = [j for j in range(m.njnt) if m.jnt_type[j] == 0 and m.body(m.jnt_bodyid[j]).name == part][0]
a = m.jnt_qposadr[j]
hk = [j for j in range(m.njnt) if m.jnt_type[j] == 0 and m.body(m.jnt_bodyid[j]).name == "hex_key"][0]; ah = m.jnt_qposadr[hk]
for r in range(r0, min(r1, len(Q) * B), step):
    k = r // B - 1
    print(r, np.round(Q[k][a:a + 3] * 1000, 1).tolist(), np.round(Q[k][a + 3:a + 7], 3).tolist(), "hex", np.round(Q[k][ah:ah + 3] * 1000, 0).tolist(), "rp %.2f welds %d" % (rec[k][3], rec[k][6]))
