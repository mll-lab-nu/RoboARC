"""Phase B search: lower the held top deeper into the side rack before the release (both TCPs -d in z), then the
retreat as in the base rows. python exp_rack.py <code root> <scenes> <base.npz> <state.pkl> <d_mm> [z_from z_to]"""
import sys, pickle, json, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
from orc import Orc, setup
code, scenes, base, st, dmm = sys.argv[1:6]
R = setup(code, scenes)
o = Orc(R); o.s.set_state(pickle.load(open(st, "rb")))
r0 = int(st.split("st_")[1][:7])
rows = np.load(base)["ctrl"][r0:70000]
d = float(dmm) / 1000
k1, k2, k3, k4 = 400, 2800, 3600, 4800          # ramp down 64400-66800, hold through the release (67200), ramp back
def off(k):
    if k < k1: z = 0.
    elif k < k2: z = -d * (k - k1) / (k2 - k1)
    elif k < k3: z = -d
    elif k < k4: z = -d * (1 - (k - k3) / (k4 - k3))
    else: z = 0.
    return {"left": np.array([0, 0, z]), "right": np.array([0, 0, z])}
log = []
for a in range(0, len(rows), 400):
    o.run(rows[a:a + 400], offset=(lambda k, a=a: off(k + a)))
    p, Rm = o.body("top")
    log.append((r0 + a + 400, np.round(p, 3).tolist(), np.round(Rm[:, 2], 2).tolist()))
pickle.dump(o.save(), open(f"snap_rack_d{int(float(dmm))}.pkl", "wb"), protocol=4)
p, Rm = o.body("top")
print(json.dumps(dict(d_mm=float(dmm), end_top=np.round(p, 4).tolist(), end_z=np.round(Rm[:, 2], 3).tolist(), slewed=o.s.slewed_ticks,
                      trace=log[::2])))
