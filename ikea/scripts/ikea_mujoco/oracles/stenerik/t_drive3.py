"""From a builder s0 checkpoint: st_lib.drive on the screw nearest the tip, counting right-left robot contacts and
right-hand contacts with parts other than the screwdriver, then the supports' alignment."""
import json, sys, collections
import numpy as np
from st_lib import Rig, drive, align_state, untorque
rig = Rig(); rig.load(sys.argv[1]); m = rig.m
s = rig.nearest_screw_to_tip(); print('screw', s, rig.screw_state(s), rig.tip(s)[1], align_state(rig), flush=True)
L = np.array([m.body(i).name.startswith('left_') for i in range(m.nbody)]); Rb = np.array([m.body(i).name.startswith('right_') for i in range(m.nbody)])
cnt = collections.Counter(); orig = rig.step
def step(row):
    orig(row); d = rig.d; n = d.ncon
    if n:
        b1 = m.geom_bodyid[d.contact.geom1[:n]]; b2 = m.geom_bodyid[d.contact.geom2[:n]]
        for x, y in zip(b1, b2):
            if (Rb[x] and L[y]) or (Rb[y] and L[x]): cnt[(m.body(int(x)).name, m.body(int(y)).name)] += 1
            elif Rb[x] and not Rb[y] and m.body(int(y)).name not in ('screwdriver',): cnt[('R', m.body(int(y)).name)] += 1
            elif Rb[y] and not Rb[x] and m.body(int(x)).name not in ('screwdriver',): cnt[('R', m.body(int(x)).name)] += 1
rig.step = step
log = []; out = drive(rig, s, log, every=int(sys.argv[2]) if len(sys.argv) > 2 else 2000, max_ticks=int(sys.argv[3]) if len(sys.argv) > 3 else 40000)
print("drive done", flush=True)
for l in log: l.pop('met', None); print(json.dumps(l, default=float))
print(json.dumps(out, default=float)); untorque(rig); print('align', align_state(rig)); print('contacts', dict(cnt))
