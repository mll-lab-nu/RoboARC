#!/usr/bin/env python3
"""The tempo profile for uppsta_oracle.py (ref/tempo.npz, S per demo row, an integer >= 1: the controller takes demo
rows i, i+S, ... where it may move faster than the demo). From the guide run's contact track (diag.py --contacts):
S > 1 only where every part a hand touches touches nothing else (free motion or a free carry), away from every grasp
and release (gripper command changes), and as fast as the joint-speed limits allow (SPEED_FRAC of the limit over a
window); 1 elsewhere. The search gates every segment, and a candidate with tempo_max 1 is the demo's own pace.
  python tempo.py --demo D.npz --guide DIR --out REFDIR [--smax 4]"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SPEED_FRAC = 0.9
ap = argparse.ArgumentParser(); ap.add_argument('--demo', required=True); ap.add_argument('--guide', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--smax', type=int, default=4)
ap.add_argument('--grip-pad', type=int, nargs=2, default=(1600, 800), help='rows kept at S=1 before / after a gripper change')
ap.add_argument('--contact-pad', type=int, default=400)
ap.add_argument('--still', type=float, default=0.02, help='a hand slower than this (in limits per row) is holding still')
ap.add_argument('--rcb-ikea', default=os.path.join(HERE, '..', '..', 'shared'))
a = ap.parse_args(); sys.path.insert(0, a.rcb_ikea)
import rcb_ikea as R
c = np.load(a.demo)['ctrl']; N = len(c)
acts = [f'{x}/actuator{i}' for x in ('left', 'right') for i in range(1, 9)]
lim = R.joint_speed_limits(acts) * 4 * 0.0003125
arm = np.isfinite(lim)
step = np.abs(np.diff(np.vstack([c[:1], c]), axis=0))[:, arm] / lim[arm]         # per row, in limits
v = step.max(1)
z = np.load(os.path.join(a.guide, 'track.npz'))
rows, cm = z['rows'], z['contact']
other = (cm & 4) > 0
cols = {1: [i for i in range(8) if arm[i]], 2: [i for i in range(8, 16) if arm[i]]}
vk = {h: np.abs(np.diff(np.vstack([c[:1], c]), axis=0))[:, cols[h]] / lim[cols[h]] for h in cols}
vk = {h: v_.max(1) for h, v_ in vk.items()}
busy_k = np.zeros(len(rows), bool)
for h in (1, 2):                                     # a MOVING hand touches a part that touches something else
    touch = ((cm & h) > 0) & other
    mv = np.array([vk[h][max(0, r - 64):r + 64].max() > a.still for r in rows])
    busy_k |= touch.any(1) & mv
busy = np.zeros(N, bool)
for k, r in enumerate(rows):
    if busy_k[k]:
        busy[max(0, r - 8 - a.contact_pad):min(N, r + a.contact_pad)] = True
for col in (7, 15):                                  # grasps and releases: the gripper command crossing half open
    g = c[:, col]; cl = g < 100.
    for r in np.flatnonzero(cl[1:] != cl[:-1]) + 1:
        r0 = r
        while r0 > 0 and abs(g[r0 - 1] - g[r0]) > 0:   # back to where the ramp starts
            r0 -= 1
        busy[max(0, r0 - a.grip_pad[0]):min(N, r + a.grip_pad[1])] = True
W = 400
from numpy.lib.stride_tricks import sliding_window_view
vpad = np.r_[np.full(W, v[0]), v, np.full(W * 2, v[-1])]
vwin = sliding_window_view(vpad, 3 * W + 1).max(1)[:N]           # the next S rows' speed (and a margin) must fit
S = np.clip(np.floor(SPEED_FRAC / np.maximum(vwin, 1e-9)), 1, a.smax).astype(np.int64)
S[busy] = 1
rows_out = 0; i = 0
while i < N:
    rows_out += 1; i += int(S[i])
os.makedirs(a.out, exist_ok=True)
np.savez(os.path.join(a.out, 'tempo.npz'), S=S)
print(json.dumps(dict(demo_rows=N, rows_out=rows_out, ratio=round(rows_out / N, 3), busy_frac=round(busy.mean(), 3),
                      hist={int(k): int((S == k).sum()) for k in range(1, a.smax + 1)})))
