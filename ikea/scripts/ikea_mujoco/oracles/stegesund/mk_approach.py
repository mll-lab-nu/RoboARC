#!/usr/bin/env python3
"""mk_approach.py GUIDE.npz DEMO.npz ROW0 BASE_TRACKS.json OUT.json [lead]: add a TCP approach window before every grasp
after ROW0: for each guide contact interval (arm touches part, >= 150 rows) whose demo grip is closing into it, the
window [c0 - lead, c0) tracks the arm's TCP in the part's frame onto the guide's (c0 = the start of that closing)."""
import json, sys
import numpy as np
z = np.load(sys.argv[1]); c = np.load(sys.argv[2])['ctrl']; row0 = int(sys.argv[3])
base = json.load(open(sys.argv[4])); lead = int(sys.argv[6]) if len(sys.argv) > 6 else 1200
P = list(z['parts']); T = z['touch']; out = list(base); added = []
for a, arm in enumerate(('left', 'right')):
    g = c[:, 7 if a == 0 else 15]
    for k, p in enumerate(P):
        on = (T[:, a] >> k) & 1
        d = np.diff(np.r_[0, on, 0]); st = np.flatnonzero(d == 1); en = np.flatnonzero(d == -1)
        iv = []
        for s, e in zip(st, en):
            if iv and s - iv[-1][1] < 200: iv[-1][1] = e
            else: iv.append([s, e])
        for s, e in iv:
            if e - s < 150 or s < row0: continue
            # the closing segment: walk back from s while the grip command is falling (or flat right before contact)
            j = s
            while j > 0 and g[j - 1] <= g[j] + 1e-9 and not (g[j - 1] < g[j] - 1e-9):
                j -= 1
                if s - j > 400: break
            if not (g[j - 1] > g[j] + 1e-9 or g[s] < g[max(0, s - 300)] - 1):
                continue                                   # not a grasp (a push with a closed or open hand)
            c0 = s
            while c0 > 0 and g[c0 - 1] > g[c0] + 1e-9:
                c0 -= 1
            c0 = min(c0, s)
            w0 = max(row0, c0 - lead)
            if c0 - w0 < 200: continue
            out.append([int(w0), int(c0), arm, 'tcp', p]); added.append((arm, p, int(w0), int(c0), int(s)))
json.dump(out, open(sys.argv[5], 'w'))
for x in sorted(added, key=lambda x: x[2]): print(x)
