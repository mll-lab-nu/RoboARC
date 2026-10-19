#!/usr/bin/env python3
"""The phase guide: the shipped reference (the author's speed-limited demo) replayed with the Franka Hand slew OFF (the
pre-rule clamp; it passes there, end sha f82c5e47...). Per row: every part's world pose (pos, wxyz quat), both TCP site
poses, finger joints, and per arm a bitmask of the parts that arm touches.  Never executed: a target only.
  rec_guide.py DEMO.npz OUT.npz"""
import sys, numpy as np
import mujoco
import rcb_ikea as R

PARTS = ['half_A', 'half_B', 'shelf', 'rod', 'bolt_1', 'bolt_2', 'bolt_3', 'bolt_4', 'hex_key']
s = R.Session('stegesund'); m, d = s.model, s.data
s.step_limit[[7, 15]] = np.inf; s._limited = np.isfinite(s.step_limit)
c = np.load(sys.argv[1])['ctrl']
pid = np.array([m.body(p).id for p in PARTS])
part_of_body = -np.ones(m.nbody, int)
for k, b in enumerate(pid):
    part_of_body[b] = k
arm_of_body = -np.ones(m.nbody, int)
for b in range(m.nbody):
    n = m.body(b).name
    if n.startswith('left/'): arm_of_body[b] = 0
    if n.startswith('right/'): arm_of_body[b] = 1
gb = m.geom_bodyid
site = [m.site('left/tcp').id, m.site('right/tcp').id]
fq = [m.joint(j).qposadr[0] for j in ('left/finger_joint1', 'left/finger_joint2', 'right/finger_joint1', 'right/finger_joint2')]
N = len(c)
pos = np.zeros((N, 9, 3)); quat = np.zeros((N, 9, 4)); tcp = np.zeros((N, 2, 7)); fing = np.zeros((N, 4))
touch = np.zeros((N, 2), np.int32)
q = np.zeros(4)
for i, row in enumerate(c):
    s.step(row)
    pos[i] = d.xpos[pid]; quat[i] = d.xquat[pid]
    for a in range(2):
        tcp[i, a, :3] = d.site_xpos[site[a]]
        mujoco.mju_mat2Quat(q, d.site_xmat[site[a]]); tcp[i, a, 3:] = q
    fing[i] = d.qpos[fq]
    nc = d.ncon
    if nc:
        b1 = gb[d.contact.geom1[:nc]]; b2 = gb[d.contact.geom2[:nc]]
        for x, y in ((b1, b2), (b2, b1)):
            ar = arm_of_body[x]; pt = part_of_body[y]
            sel = (ar >= 0) & (pt >= 0)
            for a, p in zip(ar[sel], pt[sel]):
                touch[i, a] |= 1 << p
    if (i + 1) % 20000 == 0:
        print(i + 1, flush=True)
fc = s.final_check()
np.savez_compressed(sys.argv[2], pos=pos, quat=quat, tcp=tcp, fing=fing, touch=touch, parts=np.array(PARTS))
print('success', fc['success'], fc['state_sha'])
