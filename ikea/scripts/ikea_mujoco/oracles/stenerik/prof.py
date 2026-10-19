"""Per-row cost and contact census at builder checkpoints: ncon, contacts by body pair, ms per tick (holding the row)."""
import pickle, sys, time, collections
import numpy as np
from st_lib import Rig
rig = Rig(); m = rig.m
for ck in sys.argv[1:]:
    st = pickle.load(open(ck, 'rb')); rig.set(st)
    t0 = time.time(); n = 200
    for _ in range(n): rig.s.step(rig.d.ctrl.copy())
    dt = (time.time() - t0) / n * 1e3
    d = rig.d; c = collections.Counter()
    for i in range(d.ncon):
        a, b = m.body(m.geom_bodyid[d.contact.geom1[i]]).name, m.body(m.geom_bodyid[d.contact.geom2[i]]).name
        c[tuple(sorted((a, b)))] += 1
    print(ck, 'rows', len(st['ex']), 'ms/tick %.1f' % dt, 'ncon', d.ncon, 'nefc', d.nefc, c.most_common(14), flush=True)
