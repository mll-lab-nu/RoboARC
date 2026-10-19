#!/usr/bin/env python3
"""Probe one part around a demo window: from a builder checkpoint (build.py's offsets replayed) or a diag checkpoint,
print the part's world pose, tilt, the left TCP and the part's contacts every N rows.
  probe_part.py CK.pkl PART R0 R1 [--noslew] [--every N]"""
import argparse, sys, os
import numpy as np
sys.argv, argv = sys.argv[:1], sys.argv[1:]
import build as B
ap = argparse.ArgumentParser(); ap.add_argument('ck'); ap.add_argument('part'); ap.add_argument('r0', type=int); ap.add_argument('r1', type=int)
ap.add_argument('--every', type=int, default=20); ap.add_argument('--noslew', action='store_true')
a = ap.parse_args(argv)
b = B.Builder('/tmp/probe', dict(B.CFG))
if a.noslew:
    b.g = B.Rig(noslew=True); b.g.lfing = []
b.load(a.ck); g = b.g
gl = [B.GRIP[x] for x in B.ARM]; pb = g.m.body(a.part).id; tcp = g.site['left']
while b.i < a.r1:
    b.tick()
    while not a.noslew and np.any(np.abs(b.demo[b.i][gl] - g.d.ctrl[gl]) > 1e-12):
        b.tick()
    b.i += 1
    if b.i >= a.r0 and b.i % a.every == 0:
        n = g.d.ncon; con = []
        for j in range(n):
            c = g.d.contact[j]; b1, b2 = g.m.geom_bodyid[c.geom1], g.m.geom_bodyid[c.geom2]
            if pb in (b1, b2):
                o = c.geom2 if b1 == pb else c.geom1
                con.append(g.m.geom(o).name)
        z = g.d.xmat[pb].reshape(3, 3)[:, 2]
        fz = [round(float(g.d.qpos[j]), 5) for j in g.fj['left']]
        print(b.i, len(g.ex), 'g %.1f w %.4f' % (g.d.ctrl[7], g.width('left')), 'fj', fz, 'p', np.round(g.d.xpos[pb], 5).tolist(),
              'tilt %.2f' % np.degrees(np.arccos(min(1, abs(z[2])))), 'tcp', np.round(g.d.site_xpos[tcp], 5).tolist(),
              'con', sorted(set(con)), flush=True)
