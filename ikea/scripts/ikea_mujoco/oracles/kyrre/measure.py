"""From a checkpoint, step a trajectory to given rows; print body poses and TCP sites there.
usage: measure.py TRAJ CK.pkl BODY ROW [ROW ...]"""
import json, pickle, sys
import numpy as np
import rcb_ikea as R
traj, ck, body = sys.argv[1:4]; rows = sorted(int(x) for x in sys.argv[4:])
s = R.Session('kyrre'); c = R.load_trajectory(traj, s.nu, s.timestep)
s.set_state(pickle.load(open(ck, 'rb'))); i = len(s.rows)
m, d = s.model, s.data; b = m.body(body).id
for r in rows:
    while i < r:
        s.step(c[i]); i += 1
    print(json.dumps(dict(row=r, pos=(d.xpos[b] * 1e3).round(2).tolist(), quat=d.xquat[b].round(4).tolist(),
                          rtcp=(d.site('right_tcp').xpos * 1e3).round(2).tolist(), ltcp=(d.site('left_tcp').xpos * 1e3).round(2).tolist(),
                          gR=round(float(d.ctrl[15]), 1), welds=int(d.eq_active.sum()))))
