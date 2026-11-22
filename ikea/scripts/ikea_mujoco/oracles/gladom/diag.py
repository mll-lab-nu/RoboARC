#!/usr/bin/env python3
"""Diagnostic run of the demo: execute its rows (with the in-loop hand wait, or --noslew = the old unlimited hand) and log
the driver / screw / hand state every --every demo rows; checkpoint every --ck demo rows.
  diag.py --out DIR [--noslew] [--from CK.pkl] [--to ROW]"""
import argparse, json, os, pickle, time
import numpy as np
from gl_lib import Rig, GRIP, ARM, demo_rows

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('--noslew', action='store_true')
ap.add_argument('--from', dest='frm'); ap.add_argument('--to', type=int)
ap.add_argument('--every', type=int, default=50); ap.add_argument('--ck', type=int, default=10000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
g = Rig(noslew=a.noslew); demo = demo_rows(); i = 0
if a.frm:
    st = pickle.load(open(a.frm, 'rb')); g.set(st['g']); i = st['i']
end = a.to or len(demo)
log = open(os.path.join(a.out, 'log.jsonl'), 'a'); t0 = time.time()
gl = [GRIP[x] for x in ARM]
while i < end:
    g.step(demo[i]); w = 0
    while not a.noslew and np.any(np.abs(demo[i][gl] - g.d.ctrl[gl]) > 1e-12):
        g.step(demo[i]); w += 1
    i += 1
    if i % a.every == 0:
        r = g.summary(); r.update(i=i, tick=len(g.ex), wall=round(time.time() - t0, 1)); log.write(json.dumps(r) + '\n'); log.flush()
    if i % a.ck == 0:
        pickle.dump(dict(g=g.get(), i=i), open(os.path.join(a.out, f'ck_{i}.pkl'), 'wb'))
fc = g.s.final_check()
json.dump(dict(success=fc['success'], end=fc['end'], settled_sha=fc['settled_sha'], rows=len(g.ex), wall=time.time() - t0),
          open(os.path.join(a.out, 'final.json'), 'w'), default=str)
g.s.save_trajectory(np.array(g.ex), os.path.join(a.out, 'ex.npz'))
