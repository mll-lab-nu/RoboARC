#!/usr/bin/env python3
"""uppsta_oracle's tempo profile (tempo.npz, S per demo row): S = K from demo row START on, except S = 1 within PAD rows
before / after every gripper command change (grasps, releases, re-grips); S = 1 before START (the rows already built).
Runner.emit splits any step over the joint-speed limit, so S only shortens rows where the demo moves slower than the
limits; each segment is gated by search_v3.py, with the demo's pace (tempo_max 1) as the fallback.
  python mk_tempo.py DEMO.npz OUTDIR [--start 0] [--k 4] [--pad 400 600]"""
import argparse, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('demo'); ap.add_argument('out')
ap.add_argument('--start', type=int, default=0); ap.add_argument('--k', type=int, default=4)
ap.add_argument('--pad', type=int, nargs=2, default=(400, 600))
a = ap.parse_args()
c = np.load(a.demo)['ctrl']; N = len(c)
busy = np.zeros(N, bool)
for col in (7, 15):
    g = c[:, col]
    for r in np.flatnonzero(g[1:] != g[:-1]) + 1:
        busy[max(0, r - a.pad[0]):min(N, r + a.pad[1])] = True
S = np.where(busy, 1, a.k).astype(np.int64); S[:a.start] = 1
os.makedirs(a.out, exist_ok=True)
np.savez_compressed(os.path.join(a.out, 'tempo.npz'), S=S.astype(np.int8))
i = n = 0
while i < N:
    n += 1; i += int(S[i])
print(json.dumps(dict(demo_rows=N, rows_at_most=n, start=a.start, k=a.k, pad=list(a.pad), busy=round(float(busy.mean()), 3))))
