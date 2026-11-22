"""Follow the demo rows dilated to the joint-speed limits (Rig.move per row) and record a phase scan every --every demo
rows: hands' contacts, grippers, screw states, the screwdriver tip vs its nearest screw; checkpoints every --ck rows."""
import argparse, json, os, time
import numpy as np
from st_lib import Rig, demo_rows

ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--every', type=int, default=20)
ap.add_argument('--ck', type=int, default=5000); ap.add_argument('--a', type=int, default=0); ap.add_argument('--b', type=int, default=0)
ap.add_argument('--from-ck', default=None)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
rig = Rig(); D = demo_rows(); b = a.b or len(D)
if a.from_ck: rig.load(a.from_ck)
rec = []; t0 = time.time()
for i in range(a.a, b):
    rig.move(D[i])
    if (i + 1) % a.every == 0:
        sn = rig.nearest_screw_to_tip(); lat, h = rig.tip(sn)
        rec.append(dict(i=i + 1, ex=len(rig.ex), L=rig.touching('left'), R=rig.touching('right'),
                        g=[float(D[i, 7]), float(D[i, 15])], w7=float(rig.q('right')[6]),
                        sn=sn, tlat=float(np.linalg.norm(lat) * 1e3), th=h * 1e3, fsd=rig.fbody('screwdriver', sn),
                        sc={s: [round(float(x), 5) if not isinstance(x, (bool, np.bool_)) else bool(x) for x in rig.screw_state(s)] for s in rig.screws},
                        bd=rig.s.validity.breakdown is not None,
                        tcp={a: rig.d.site_xpos[rig.site[a]].round(5).tolist() for a in ('left', 'right')},
                        sd=rig.d.xpos[rig.bid['screwdriver']].round(5).tolist(), ops=len(rig.M.ops)))
    if (i + 1) % a.ck == 0:
        rig.save(rig.get(), os.path.join(a.out, 'ck_%07d.pkl' % (i + 1)))
        json.dump(rec, open(os.path.join(a.out, 'scan.json'), 'w'))
        print(i + 1, len(rig.ex), round(time.time() - t0), flush=True)
json.dump(rec, open(os.path.join(a.out, 'scan.json'), 'w'))
print('DONE', flush=True)
