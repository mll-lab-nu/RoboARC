"""Handle phase: from a snapshot at row r0 (the handle held by the right arm, near the top), servo the handle to its
target in the top's frame: first a pre-pose `above` mm out along the insertion axis, then down in steps of `step` mm
per tick to `depth` mm (negative = past the seat, pressing), the left arm on the base rows. Logs the gates.
python exp_handle.py <code> <scenes> <base.npz> <snap.pkl> <r0> <above_mm> <speed_mm_s> <depth_mm> <ticks> <out snap>"""
import sys, os, pickle, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from orc import Orc, setup
from servo import PoseServo
code, scenes, base, snap, r0, above, speed, depth, ticks, out = sys.argv[1:11]
r0, above, speed, depth, ticks = int(r0), float(above) / 1000, float(speed) / 1000, float(depth) / 1000, int(ticks)
R = setup(code, scenes); o = Orc(R); o.restore(pickle.load(open(snap, "rb")))
rows = np.load(base)["ctrl"]
mech = o.mech; op = next(x for x in mech.ops if x["part"] == "handle")
m, d = o.m, o.d
bt = m.body("top").id
tp = np.asarray(op["target_pos"], float); tq = np.asarray(op["target_quat"], float)
ax = np.asarray(op["axis"], float)
Rt = np.zeros(9); mujoco_q = __import__("mujoco"); mujoco_q.mju_quat2Mat(Rt, tq); Rt = Rt.reshape(3, 3)
sv = PoseServo(o, "right", "handle")
g = above
log = []
for k in range(ticks):
    Rh = d.xmat[bt].reshape(3, 3); ph = d.xpos[bt]
    e_now = None
    p_star = ph + Rh @ (tp + g * ax); R_star = Rh @ Rt
    ep, ew = sv.tick(rows[r0 + k], p_star, R_star)
    if ep < 0.001 and ew < 0.01 or g < above:
        g = max(depth, g - speed * o.s.control_dt)
    if k % 1000 == 0 or k == ticks - 1:
        gap, lat, ang, Rr = mech.measure_lean(op); rel, _ = mech._local(op); holes = mech._two_holes(op, Rr, rel)
        mt = mech.metrics.get("s2-handle-two-holes", {})
        l2 = mech.metrics.get("s2-leg-handle-left", {})
        log.append(dict(row=r0 + k, leg2=dict(near=l2.get("near"), gap=round(1000 * l2.get("gap_m", 0), 1), lat=round(1000 * l2.get("lateral_m", 0), 1),
                                             ang=round(l2.get("angle_deg", 0), 1), turns=round(l2.get("turns", 0), 2)),
                        g_mm=round(g * 1000, 2), e_p_mm=round(ep * 1000, 2), e_w_deg=round(np.degrees(ew), 2),
                        gap_mm=round(gap * 1000, 2), lat_mm=round(lat * 1000, 2), ang=round(ang, 2), holes_mm=round(holes * 1000, 2),
                        contact=mt.get("contact_N"), win=mt.get("contact_window_ok"), active=o.active()))
        print(json.dumps(log[-1]), flush=True)
pickle.dump(o.save(), open(out, "wb"), protocol=4)
