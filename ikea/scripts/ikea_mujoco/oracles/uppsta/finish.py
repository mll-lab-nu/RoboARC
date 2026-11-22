#!/usr/bin/env python3
"""The reference from the last segment checkpoint of search.py: its recorded rows (the executed targets), optionally a
hold tail of the last row (--hold S seconds), the grader's verdict in this session (final_check) and the step rule
(0 rows over the limit + 1e-9 from the keyframe ctrl). Writes the rcb_ikea trajectory file.
  python finish.py --ck ck/seg030.pkl --out uppsta.npz [--hold 0]"""
import argparse, json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument('--ck', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--hold', type=float, default=0.0); ap.add_argument('--scene-dir', default=None); ap.add_argument('--rcb-ikea', default=os.path.join(HERE, '..', '..', 'shared'))
a = ap.parse_args(); sys.path.insert(0, a.rcb_ikea)
import rcb_ikea as R
ck = pickle.load(open(a.ck, 'rb'))
s = R.Session('uppsta', a.scene_dir)
ctrl0 = s.data.ctrl.copy(); lim = s.step_limit
st = dict(ck['st']); st['rows'] = list(ck['rows']); s.set_state(st)
for _ in range(int(round(a.hold / s.control_dt))):
    s.step(s.rows[-1])
rows = np.array(s.rows)
d = np.abs(np.diff(np.vstack([ctrl0[None], rows]), axis=0))
over = int((d > lim + 1e-9).any(1).sum())
inr = bool(((rows >= s._clo) & (rows <= s._chi)).all())
fc = s.final_check()
s.save_trajectory(rows, a.out)
print(json.dumps(dict(rows=len(rows), over_limit=over, in_ctrlrange=inr, success=fc['success'], state_sha=fc['state_sha'],
                      settled_sha=fc['settled_sha'], end_failed=fc['end']['report'].get('failed'),
                      released=fc['end']['released'], at_rest=fc['end']['at_rest'], breakdown=fc['end']['breakdown'],
                      settled_ok=fc['settled']['success'], terminal=fc['end']['terminal'], sha256=R.file_sha(a.out)), default=str))
