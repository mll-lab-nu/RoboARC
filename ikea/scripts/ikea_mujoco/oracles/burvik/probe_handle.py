"""From a snapshot, run base rows [a, b) and print every k rows the handle's error to its target in the top frame,
the rod-end hole errors, the handle-top contact force and the fingers on the handle; save the end snapshot.
python probe_handle.py <code> <scenes> <base.npz> <snap.pkl> <a> <b> <k> <out snap>"""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from orc import Orc, setup
code, scenes, base, snap, a, b, k, out = sys.argv[1:9]
a, b, k = int(a), int(b), int(k)
R = setup(code, scenes); o = Orc(R); o.restore(pickle.load(open(snap, "rb")))
rows = np.load(base)["ctrl"]
mech = o.mech; op = next(x for x in mech.ops if x["part"] == "handle"); m, d = o.m, o.d
bh, bt = m.body("handle").id, m.body("top").id
for i in range(a, b, k):
    o.run(rows[i:i + k])
    gap, lat, ang, Rr = mech.measure_lean(op); rel, _ = mech._local(op); holes = mech._two_holes(op, Rr, rel)
    ft = sum(abs(d.efc_force[c.efc_address]) for c in d.contact[:d.ncon] if c.efc_address >= 0 and {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} == {bh, bt})
    print(f"row {i+k}: rel {np.round(rel*1000,1)} mm ang {ang:.1f} holes {holes*1000:.1f} mm F(h-top) {ft:.1f} N active {o.active()} top {np.round(o.body('top')[0],3)} z {np.round(o.body('top')[1][:,2],2)}", flush=True)
pickle.dump(o.save(), open(out, "wb"), protocol=4)
