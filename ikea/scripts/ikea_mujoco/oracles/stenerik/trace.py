"""Replay a ctrl file under the final physics (rcb_ikea Session, clamp in) and trace the mechanism: op acceptance
times, weld switches, slewed ticks, breakdown; optional checkpoints (pickled get_state) every --ckpt rows."""
import argparse, json, os, pickle, sys, time
import numpy as np
import rcb_ikea as R

ap = argparse.ArgumentParser()
ap.add_argument('--traj', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--ckpt', type=int, default=0); ap.add_argument('--ckdir', default=None)
ap.add_argument('--every', type=int, default=2000); ap.add_argument('--final', type=int, default=1)
a = ap.parse_args()
if os.environ.get('NOCLAMP'):   # a study only: the author's physics without the joint-speed clamp
    R.JOINT_SPEED_LIMITS_RAD_S = {}
s = R.Session('stenerik')
init = s.state_sha()
c = R.load_trajectory(a.traj, s.nu, s.timestep)
mc = s.S.mech
log = dict(initial_state_sha=init, rows=len(c), events=[], samples=[])
prev_eq = {n: mc.active(n) for n in mc.eq}
t0 = time.time()
for i, row in enumerate(c):
    s.step(row)
    for n in mc.eq:
        v = mc.active(n)
        if v != prev_eq[n]:
            log['events'].append(dict(row=i, eq=n, on=v)); prev_eq[n] = v
            print(i, n, v, round(time.time() - t0, 1), flush=True)
    if a.ckpt and (i + 1) % a.ckpt == 0:
        with open(os.path.join(a.ckdir, 'ck_%07d.pkl' % (i + 1)), 'wb') as f:
            pickle.dump(s.get_state(), f)
    if (i + 1) % a.every == 0:
        bd = s.validity.breakdown
        log['samples'].append(dict(row=i + 1, wall=round(time.time() - t0, 1), slewed=s.slewed_ticks, bd=bd,
                                   pos={k: s.S.mech.pose(k)[0].round(4).tolist() for k in ('support_1', 'support_2', 'crossbar', 'backing')}))
        if bd is not None and not any(x.get('bd') for x in log['samples'][:-1]):
            print('BREAKDOWN', i, bd, flush=True)
log['ops'] = dict(mc.ops); log['slewed'] = s.slewed_ticks; log['wall'] = round(time.time() - t0, 1)
if a.final:
    fc = s.final_check()
    log['final'] = dict(success=fc['success'], state_sha=fc['state_sha'], settled_sha=fc['settled_sha'],
                        end={k: fc['end'][k] for k in ('success', 'goal', 'released', 'at_rest', 'breakdown')},
                        settled={k: fc['settled'][k] for k in ('success', 'goal', 'released', 'at_rest', 'breakdown')},
                        end_terminal=fc['end']['terminal'], end_report=fc['end']['report'], settled_report=fc['settled']['report'])
json.dump(log, open(a.out, 'w'), default=R._plain)
print('DONE', log['wall'], log.get('final', {}).get('success'), flush=True)
