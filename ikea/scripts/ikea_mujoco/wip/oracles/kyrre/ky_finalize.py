#!/usr/bin/env python3
"""The reference from ky_search.py's last segment checkpoint: the demo's remaining rows (if any), then the last row held
(at most HOLD_MAX s, checked every 0.25 s) until everything rests and no hand touches a part; final_check; writes
OUT.npz (the executed rows: Session.rows, every one within the joint-speed limits) and OUT.json.
  python ky_finalize.py --demo kyrre_demo.npz --ref REFDIR --ck SEG.pkl --out OUT"""
import argparse, json, os, pickle, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ky_guide as U
import rcb_ikea as R
ap = argparse.ArgumentParser(); ap.add_argument('--demo', required=True); ap.add_argument('--ref', required=True)
ap.add_argument('--ck', required=True); ap.add_argument('--out', required=True); ap.add_argument('--hold-max', type=float, default=8.0)
a = ap.parse_args(); t0 = time.time()
run = U.Runner(a.demo, a.ref); run.restore(pickle.load(open(a.ck, 'rb'))); run.p = {}
for i in range(run.tau, len(run.demo)):
    run.step_row(i)
s = run.s; row = s.data.ctrl.copy(); held = 0
for k in range(int(a.hold_max / 0.25)):
    term = s.validity.terminal()
    if term['at_rest'] and term['released']: break
    for _ in range(int(round(0.25 / s.control_dt))): s.step(row)
    held += 1
fc = s.final_check()
s.save_trajectory(path=a.out + '.npz')
c = R.load_trajectory(a.out + '.npz', s.nu, s.timestep)
s2 = R.Session('kyrre'); prev = np.vstack([s2.data.ctrl.copy()[None], c[:-1]]); lim = s2.step_limit; fin = np.isfinite(lim)
d = np.abs(np.clip(c, s2._clo, s2._chi) - prev)
res = dict(rows=len(c), held_s=held * 0.25, rows_over_limit=int((d[:, fin] > lim[fin] + R.SPEED_TOL).any(1).sum()),
           max_ratio=float((d[:, fin] / lim[fin]).max()), success=fc['success'], state_sha=fc['state_sha'], settled_sha=fc['settled_sha'],
           end={k: fc['end'][k] for k in ('goal', 'released', 'at_rest', 'breakdown')}, settled={k: fc['settled'][k] for k in ('goal', 'released', 'at_rest', 'breakdown')},
           report=fc['end']['report'], wall=round(time.time() - t0, 1))
json.dump(res, open(a.out + '.json', 'w'), default=R._plain, indent=1)
print(json.dumps({k: res[k] for k in ('rows', 'held_s', 'rows_over_limit', 'max_ratio', 'success', 'state_sha', 'settled_sha', 'end', 'wall')}, default=R._plain))
