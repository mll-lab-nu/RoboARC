"""Trace a variera trajectory: --mode noslew (old rule: no hand slew) | slew | wait (grip_wait re-time inside, with the
source-row map). Logs every --every rows (jsonl), per-row world poses of the parts + screwdriver (guide.npz), and
checkpoints (pickle of Session.get_state()) every --ck rows. usage: trace.py --traj T --mode M --out DIR"""
import argparse, json, os, pickle, sys, time
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--traj'); ap.add_argument('--mode', default='slew'); ap.add_argument('--out')
ap.add_argument('--every', type=int, default=200); ap.add_argument('--ck', type=int, default=10000)
ap.add_argument('--upto', type=int, default=0); ap.add_argument('--guide', type=int, default=1)
a = ap.parse_args()
import rcb_ikea as R
if a.mode == 'noslew':
    R.GRIPPER_WIDTH_SPEED_M_S = {}
s = R.Session('variera'); m, d = s.model, s.data
c = R.load_trajectory(a.traj, s.nu, s.timestep)
src = np.arange(len(c))
if a.mode == 'wait':
    lim = s.step_limit; g = [7, 15]; y = d.ctrl[g].copy(); out = []; srcl = []
    for i, row in enumerate(c):
        first = True
        while True:
            dd = row[g] - y; y = np.where(np.abs(dd) > lim[g] + 1e-9, y + np.sign(dd) * lim[g], row[g])
            r = row.copy(); r[g] = y; out.append(r); srcl.append(i)
            if np.all(np.abs(row[g] - y) <= 1e-12): break
    c = np.array(out); src = np.array(srcl)
if a.upto: c = c[:a.upto]; src = src[:a.upto]
os.makedirs(a.out, exist_ok=True)
np.save(os.path.join(a.out, 'src.npy'), src)
names = ['shelf', 'frame_1', 'frame_2', 'screw_1', 'screw_2', 'screw_3', 'screw_4', 'screwdriver']
ids = [m.body(n).id for n in names]
G = np.zeros((len(c), len(ids), 7)) if a.guide else None
sites = [m.site('left_tcp').id, m.site('right_tcp').id]; T = np.zeros((len(c), 2, 12)) if a.guide else None
mon = s.S.monitor
fj = {arm: [m.jnt_qposadr[m.joint(f'{arm}_finger_joint{k}').id] for k in (1, 2)] for arm in ('left', 'right')}
lg = open(os.path.join(a.out, 'log.jsonl'), 'w'); t0 = time.time()
def rel(b, ref=0):
    R0 = d.xmat[ids[ref]].reshape(3, 3); return R0.T @ (d.xpos[b] - d.xpos[ids[ref]]), R0.T @ d.xmat[b].reshape(3, 3)[:, 2]
for i, row in enumerate(c):
    s.step(row)
    if G is not None:
        G[i, :, :3] = d.xpos[ids]; G[i, :, 3:] = d.xquat[ids]; T[i, :, :3] = d.site_xpos[sites]; T[i, :, 3:] = d.site_xmat[sites]
    if (i + 1) % a.every == 0 or i + 1 == len(c):
        e = dict(row=i + 1, src=int(src[i]), t=round(float(d.time), 4), wall=round(time.time() - t0, 1),
                 gc=[round(float(d.ctrl[7]), 2), round(float(d.ctrl[15]), 2)],
                 gw=[round(1000 * float(d.qpos[fj[k]].sum()), 2) for k in ('left', 'right')], slewed=s.slewed_ticks)
        for n, b in zip(names[1:], ids[1:]):
            p, z = rel(b); e[n] = [round(1000 * float(x), 2) for x in p] + [round(float(x), 4) for x in z]
        e['acc'] = {'%s|%d' % k: [round(v['turns'], 3), round(1000 * v['descent'], 3)] for k, v in mon.acc.items()}
        e['touch'] = s.validity.robot_part_contacts()
        e['ncon'] = int(d.ncon)
        lg.write(json.dumps(e) + '\n'); lg.flush()
    if a.ck and (i + 1) % a.ck == 0:
        with open(os.path.join(a.out, 'ck_%07d.pkl' % (i + 1)), 'wb') as f: pickle.dump(s.get_state(), f)
fc = s.final_check()
json.dump(fc, open(os.path.join(a.out, 'final.json'), 'w'), default=R._plain)
if G is not None: np.savez(os.path.join(a.out, 'guide.npz'), pose=G, tcp=T, names=np.array(names), src=src)
with open(os.path.join(a.out, 'ck_end.pkl'), 'wb') as f: pickle.dump(s.get_state(), f)
print(json.dumps(dict(success=fc['success'], rows=len(c), wall=round(time.time() - t0, 1), sha=fc['state_sha'][:16])))
