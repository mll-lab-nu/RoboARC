"""DEVELOPMENT ONLY (never part of a reference): a builder checkpoint at demo row ROW built from the no-prism demo track's
state there (qpos, qvel 0), the welds of the operations the demo had latched by then engaged at the current poses, the
demo row as ctrl; held HOLD ticks to settle on the final physics. Lets later cycles be worked on before earlier ones pass.
usage: teleport.py TRACK.npz ROW CYCLE OUT.pkl"""
import sys, json, pickle
import numpy as np, mujoco
from ky_lib import Rig, demo_rows, Follower, DemoTrack
tr, row, cyc, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rig = Rig(); D = demo_rows(); T = DemoTrack(tr, rig); m, d = rig.m, rig.d; M = rig.M
k = int(np.clip(np.searchsorted(T.rows, row), 0, len(T.rows) - 1)); row = int(T.rows[k])
recs = [json.loads(l) for l in open('runs/scan_np/scan.jsonl') if '"r"' in l]
nw = max(r['w'] for r in recs if r['r'] <= row)
d.qpos[:] = T.Q[k]; d.qvel[:] = 0; d.ctrl[:] = D[row - 1]; mujoco.mj_forward(m, d)
for o in M.ops[:nw]:
    M.engage(o)
mujoco.mj_forward(m, d)
for t in range(400): rig.s.step(D[row - 1])
fol = Follower(rig, D)
st = rig.get(); st.update(demo_i=row, cycle=cyc, fol=fol.state(), log=dict(cycles=[], teleport=row))
pickle.dump(st, open(out, 'wb'))
print('teleport', row, 'welds', nw, [o['key'] for o in M.ops if rig.welded(o['part'])], 'seat', (rig.pose('seat')[0] * 1e3).round(2).tolist())
