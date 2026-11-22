#!/usr/bin/env python3
"""Probe the driver pick: from a diag checkpoint, run demo rows R0..R1 (wait or noslew) and print the left TCP, the
driver and the in-hand offset every N rows.   probe_grasp.py CK.pkl R1 [--noslew] [--every N]"""
import argparse, pickle
import numpy as np
from gl_lib import Rig, GRIP, ARM, demo_rows

ap = argparse.ArgumentParser(); ap.add_argument('ck'); ap.add_argument('r1', type=int)
ap.add_argument('--noslew', action='store_true'); ap.add_argument('--every', type=int, default=50)
ap.add_argument('--r0', type=int, default=0)
a = ap.parse_args()
g = Rig(noslew=a.noslew); demo = demo_rows(); st = pickle.load(open(a.ck, 'rb')); g.set(st['g']); i = st['i']
gl = [GRIP[x] for x in ARM]; hd = g.bid['hex_driver']; tcp = g.site['left']
fb = [g.m.body('left_left_finger').id, g.m.body('left_right_finger').id]
while i < a.r1:
    g.step(demo[i])
    while not a.noslew and np.any(np.abs(demo[i][gl] - g.d.ctrl[gl]) > 1e-12):
        g.step(demo[i])
    i += 1
    if i % a.every == 0 and i >= a.r0:
        R = g.d.site_xmat[tcp].reshape(3, 3); pt = g.d.site_xpos[tcp]
        rel = R.T @ (g.d.xpos[hd] - pt); zr = R.T @ g.axis('hex_driver')
        f = [np.round(R.T @ (g.d.xpos[b] - pt), 4).tolist() for b in fb]
        print(i, len(g.ex), 'g %.1f w %.4f' % (g.d.ctrl[7], g.width('left')), 'tcp', np.round(pt, 5).tolist(),
              'drv', np.round(g.d.xpos[hd], 5).tolist(), 'rel', np.round(rel, 5).tolist(), 'zr', np.round(zr, 4).tolist(),
              'fing', f, flush=True)
