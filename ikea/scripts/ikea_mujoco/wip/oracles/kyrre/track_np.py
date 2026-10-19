"""The demo on the author's geometry (prism rule off; a guide only): qpos, the grippers' contacts, every EVERY rows.
usage: track_np.py REF.npz OUT.npz [--every 50]"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.environ.get("IKEA_SCENES", "/opt/ikea/scenes"), "kyrre"))
import ikea_runtime
ikea_runtime.cylinder_prisms = lambda spec: []
import rcb_ikea as R
ref, out = sys.argv[1:3]; E = int(sys.argv[sys.argv.index('--every') + 1]) if '--every' in sys.argv else 50
s = R.Session("kyrre"); c = R.load_trajectory(ref, s.nu, s.timestep)
Q, rows = [], []; t0 = time.time()
for i, row in enumerate(c):
    s.step(row)
    if (i + 1) % E == 0:
        Q.append(s.data.qpos.copy()); rows.append(i + 1)
    if (i + 1) % 50000 == 0:
        np.savez(out + '.part.npz', qpos=np.array(Q), rows=np.array(rows)); print(i + 1, round(time.time() - t0), flush=True)
np.savez(out, qpos=np.array(Q), rows=np.array(rows), every=E); print('done', round(time.time() - t0))
