#!/usr/bin/env python3
"""The guide run's schedule for uppsta_oracle.py: per arm the demo's grasp intervals (gripper command below OPEN_ABOVE,
from the row its closing ramp starts, `cs`, to the row it is open again, `oe`) and the part each one carries in the guide
run (the part that moves most while staying fixed relative to the TCP), plus the mechanism's events by row.
  python mksched.py --demo D.npz --guide DIR (diag.py --track output: track.npz, log.jsonl) --out REFDIR"""
import argparse, json, os, sys
import numpy as np
import mujoco
HERE = os.path.dirname(os.path.abspath(__file__))
OPEN_ABOVE = 100.


def build(model, ctrl, track, log_path, control_dt):
    m = model; d = mujoco.MjData(m)
    rows, aq, pose, names = track['rows'], track['arm_qpos'], track['body_pose'], [str(x) for x in track['bodies']]
    arm_adr = [m.jnt_qposadr[j] for j in range(m.njnt) if m.body(m.jnt_bodyid[j]).name.startswith(('left/', 'right/'))]
    site = {x: m.site(f'{x}/tcp').id for x in ('left', 'right')}
    step = int(rows[1] - rows[0])

    def tcp(k, arm):
        d.qpos[arm_adr] = aq[k]; mujoco.mj_kinematics(m, d)
        return d.site_xpos[site[arm]].copy(), d.site_xmat[site[arm]].reshape(3, 3).copy()

    out = {}
    for arm, col in (('left', 7), ('right', 15)):
        g = ctrl[:, col]; closed = g < OPEN_ABOVE; ivs = []; i = 0; n = len(g)
        while i < n:
            if closed[i]:
                j = i
                while j < n and closed[j]: j += 1
                cs = i
                while cs > 0 and g[cs - 1] > g[cs]: cs -= 1          # back to where the closing ramp starts
                ivs.append([cs, j]); i = j
            else:
                i += 1
        res = []
        for cs, oe in ivs:
            k0 = (cs + 600) // step; k1 = min(len(rows) - 1, (oe - 100) // step)
            if (oe - 100) // step > len(rows) - 1:                  # beyond a partial guide track
                res.append(dict(cs=int(cs), oe=int(oe), part=None, beyond_track=True)); continue
            best = None
            # the whole interval; failing that its first two thirds (a peg that slides in the grasp as it is pushed home)
            for kk1 in (k1, k0 + max(1, (2 * (k1 - k0)) // 3)):
                ks = np.linspace(k0, kk1, 12).astype(int)
                T = [tcp(k, arm) for k in ks]
                for b, nm in enumerate(names):
                    rel = np.array([Rm.T @ (pose[k, b, :3] - p) for k, (p, Rm) in zip(ks, T)])
                    path = float(np.sum(np.linalg.norm(np.diff(pose[ks, b, :3], axis=0), axis=1)))
                    spread = float(np.max(np.linalg.norm(rel - rel.mean(0), axis=1)))
                    dist = float(np.linalg.norm(rel.mean(0)))
                    if path > 0.01 and spread < 0.006 and dist < 0.15 and (best is None or path > best[1]):
                        best = (nm, path, spread, dist)
                if best is not None:
                    break
            res.append(dict(cs=int(cs), oe=int(oe), part=best[0] if best else None,
                            path_mm=round(1000 * best[1], 1) if best else None,
                            spread_mm=round(1000 * best[2], 2) if best else None))
        out[arm] = res
    ev = []
    for line in open(log_path):
        for e in json.loads(line).get('events', []):
            ev.append([int(round(e['time'] / control_dt)), e])
    return out, ev


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--demo', required=True); ap.add_argument('--guide', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--rcb-ikea', default=os.path.join(HERE, '..', '..', 'shared'))
    a = ap.parse_args(); sys.path.insert(0, a.rcb_ikea)
    import rcb_ikea as R
    s = R.Session('uppsta')
    sched, ev = build(s.model, np.load(a.demo)['ctrl'], np.load(os.path.join(a.guide, 'track.npz')),
                      os.path.join(a.guide, 'log.jsonl'), s.control_dt)
    for arm in sched:
        for it in sched[arm]:
            print(arm, it)
    os.makedirs(a.out, exist_ok=True)
    json.dump(sched, open(os.path.join(a.out, 'sched.json'), 'w'), indent=1)
    json.dump(ev, open(os.path.join(a.out, 'events.json'), 'w'), indent=1)
    import shutil
    shutil.copyfile(os.path.join(a.guide, 'track.npz'), os.path.join(a.out, 'track.npz'))
    print('events', ev)
