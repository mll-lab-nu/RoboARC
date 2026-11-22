import json, pickle, sys
import numpy as np
from st_lib import Rig, Follower, demo_rows, pad_view, pad_hook
rig = Rig(); D = demo_rows(); st = pickle.load(open(sys.argv[1], 'rb')); rig.set(st)
fol = Follower(rig, D); fol.off = np.array(st.get('fol_off', np.zeros(7))); a, b, p0, every = [int(x) for x in sys.argv[2:6]]
pads = sys.argv[6].split(',')
ph = pad_hook(rig, fol, a, b)
for i in range(a, b):
    rig.move(ph(i, fol.row(i)))
    if i + 1 >= p0 and (i + 1) % every == 0:
        print(i + 1, 'L', rig.touching('left'), 'R', rig.touching('right'), 'g', D[i, [7, 15]].round(0).tolist(), 'offL', np.abs(fol.offL).max().round(4),
              json.dumps({p: pad_view(rig, p) for p in pads}),
              json.dumps({k: {x: (round(y, 4) if isinstance(y, float) else y) for x, y in v.items()} for k, v in rig.M.metrics.items() if k.startswith('pad')}), flush=True)
