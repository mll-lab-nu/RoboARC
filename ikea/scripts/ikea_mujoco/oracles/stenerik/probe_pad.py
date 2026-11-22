"""From a pads.py checkpoint, run pads.py's loop (no new press) and print pad P's view every N rows from row A."""
import json, os, pickle, sys
import numpy as np
from st_lib import Rig, Follower, demo_rows, pad_hook, pad_view
import pads as PP
ck, pad, a0, every, end = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
rig = Rig(); D = demo_rows(); fol = Follower(rig, D); st = pickle.load(open(ck, 'rb')); rig.set(st); PP.fol_set(fol, st['fol'])
i = int(st['demo_i']); ph = pad_hook(rig, fol, 0, len(D))
while i < end:
    rig.move(ph(i, fol.row(i))); i += 1
    if i >= a0 and i % every == 0:
        v = pad_view(rig, pad); met = {k: {x: round(y, 4) for x, y in m.items() if isinstance(y, float)} for k, m in rig.M.metrics.items() if pad in k}
        print(i, 'L', rig.touching('left'), 'R', rig.touching('right'), 'g', D[i, [7, 15]].round(0).tolist(), json.dumps(v), json.dumps(met), flush=True)
