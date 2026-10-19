#!/usr/bin/env python3
"""The guide for ky_guide.py from the demo's run on the author's geometry (track_np.py: qpos every N rows, the prism rule
off): every free body's pose (track.npz: rows, body_pose (N, B, 7), bodies, arm_qpos), the weld events by row
(events.json, from scan.py's log of that run) and per arm the demo's grasp intervals with the part each carries
(sched.json; the method of oracles/uppsta's mksched.py).
  python mkguide.py --track TRACK.npz --scan SCAN.jsonl --out REFDIR"""
import argparse, json, os
import numpy as np, mujoco
from ky_lib import Rig, demo_rows, ARM

OPEN_ABOVE = 100.
ap = argparse.ArgumentParser(); ap.add_argument('--track', required=True); ap.add_argument('--scan', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
rig = Rig(); m = rig.m; D = demo_rows()
z = np.load(a.track); Q, rows = z['qpos'], z['rows']
bodies = list(rig.s.S.task['parts']) + ['hex_key']
adr = {}
for b in bodies:
    j = [j for j in range(m.njnt) if m.jnt_bodyid[j] == rig.bid[b] and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
    adr[b] = int(m.jnt_qposadr[j])
body_pose = np.stack([Q[:, adr[b]:adr[b] + 7] for b in bodies], 1)
arm_adr = np.r_[rig.qadr['left'], rig.qadr['right']]
np.savez_compressed(os.path.join(a.out, 'track.npz'), rows=rows, body_pose=body_pose, bodies=np.array(bodies), arm_qpos=Q[:, arm_adr])
# weld events
recs = [json.loads(l) for l in open(a.scan) if '"r"' in l]
ops = rig.ops; ev = []; pw = 0
for r in recs:
    if r['w'] > pw:
        for k in range(pw, r['w']): ev.append([r['r'], dict(fastener=ops[k]['part'], op=ops[k]['key'])])
        pw = r['w']
json.dump(ev, open(os.path.join(a.out, 'events.json'), 'w'), indent=1)
# grasp intervals
d = mujoco.MjData(m); step = int(rows[1] - rows[0])


def tcp(k, arm):
    d.qpos[:] = Q[k]; mujoco.mj_kinematics(m, d)
    return d.site_xpos[rig.site[arm]].copy(), d.site_xmat[rig.site[arm]].reshape(3, 3).copy()


out = {}
for arm in ('left', 'right'):
    g = D[:, 7 if arm == 'left' else 15]; closed = g < OPEN_ABOVE; ivs = []; i = 0; n = len(g)
    while i < n:
        if closed[i]:
            j = i
            while j < n and closed[j]: j += 1
            cs = i
            while cs > 0 and g[cs - 1] > g[cs]: cs -= 1
            ivs.append([cs, j]); i = j
        else:
            i += 1
    res = []
    for cs, oe in ivs:
        k0 = min(len(rows) - 1, (cs + 600) // step); k1 = min(len(rows) - 1, max(k0 + 1, (oe - 100) // step))
        best = None
        for kk1 in (k1, min(len(rows) - 1, k0 + max(1, (2 * (k1 - k0)) // 3))):
            ks = np.linspace(k0, kk1, 12).astype(int); T = [tcp(k, arm) for k in ks]
            for b, nm in enumerate(bodies):
                rel = np.array([Rm.T @ (body_pose[k, b, :3] - p) for k, (p, Rm) in zip(ks, T)])
                path = float(np.sum(np.linalg.norm(np.diff(body_pose[ks, b, :3], axis=0), axis=1)))
                spread = float(np.max(np.linalg.norm(rel - rel.mean(0), axis=1)))
                dist = float(np.linalg.norm(rel.mean(0)))
                if path > 0.01 and spread < 0.006 and dist < 0.45 and (best is None or path > best[1]):
                    best = (nm, path, spread, dist)
            if best is not None: break
        rec = dict(cs=int(cs), oe=int(oe), part=best[0] if best else None, path_mm=round(1000 * best[1], 1) if best else None)
        if best is None:
            # a grasp that carries nothing (a finger twirl of a screw in its thread): the part the hand closes on, the
            # nearest one to the TCP when the fingers have closed, if within 3 cm; the approach is offset onto it
            kc = min(len(rows) - 1, (cs + 400) // step); p, Rm = tcp(kc, arm)
            dd = {nm: float(np.linalg.norm(body_pose[kc, b, :3] - p)) for b, nm in enumerate(bodies)}
            nm = min(dd, key=dd.get)
            if dd[nm] < 0.03:
                rec.update(part=nm, twirl=True)
        res.append(rec)
    out[arm] = res
    for it in res: print(arm, it)
json.dump(out, open(os.path.join(a.out, 'sched.json'), 'w'), indent=1)
print('events', ev)
