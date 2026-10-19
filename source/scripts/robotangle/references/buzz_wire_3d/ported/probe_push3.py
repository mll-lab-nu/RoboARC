"""Push options at a resting wand over a wider grid (contact fraction along the handle incl. the stem / beyond the grip,
lean, pad height, turn) with rd plan_push; prints the plannable ones. usage: probe_push2.py <instance> <actions.npy> <rows>"""
import sys, json, itertools
import numpy as np
from rcb_topogym import Session
from topogym.planning.buzz_wire_3d.redesign.io import SessionIO
from topogym.planning.buzz_wire_3d.redesign.oracle import TierOracle
from topogym.assets.geometry.buzz_wire_3d import endpoint_pose
inst, acts, nrows = sys.argv[1], sys.argv[2], int(sys.argv[3]); K, KK = int(sys.argv[4]), int(sys.argv[5])
s = Session("buzz_wire_3d", instance=inst, strict=True)
io = SessionIO(s)
for r in np.load(acts)[:nrows]:
    s.step(r)


class PD:
    sleeves = (); sleeve_radius_m = 0.0; robot_mount = None
    def __init__(self, t): self.t = t
    def layout(self): return self.t.layout
    def wire(self): return self.t.route
    def origin(self): return np.asarray(self.t.origin, float)


orc = TierOracle(io, PD(s.env._task), [], log=lambda **kv: None, rng=0, seeds=60)
orc._last_q = io.q()
pw, Rw = io.wand_pose()
tp, tr = endpoint_pose(orc.route, 1)
target = tp + orc.origin
post = np.array([target[0], target[1], pw[2]])
hy_t = orc._yaw(tr.apply([1, 0, 0]))
d = (hy_t - orc._yaw(Rw.apply([1, 0, 0])) + np.pi) % (2 * np.pi) - np.pi
print(json.dumps(dict(dyaw=round(float(np.degrees(d)), 1))), flush=True)
sgn = np.sign(d)
import collections
agg = collections.Counter()
grid = list(itertools.product((8, 15, 25), (-0.3, 0.0, 0.3, 0.6, 0.85, 1.0, 1.15), (0.0, 0.35, 0.7, 1.05),
                              (0.0015, 0.006), (0.004, 0.002), (0.008, 0.012, 0.02)))
for i, (beta, frac, tilt, pad, clear, half) in enumerate(grid):
    if i % KK != K:
        continue
    plan, why = orc.plan_push(pw, Rw, post, sgn * np.deg2rad(beta), frac=frac, tilt=tilt, pad_above_deck=pad, clear=clear, half=half)
    if plan is not None:
        print(json.dumps(dict(ok=1, beta=beta, frac=frac, tilt=tilt, pad=pad, clear=clear, half=half)), flush=True)
    else:
        top = max(why, key=why.get) if why else "none"
        agg[top] += 1
        print(json.dumps(dict(beta=beta, frac=frac, tilt=tilt, why=why)), flush=True)
print(json.dumps(dict(agg=agg)), flush=True)
print("DONE", flush=True)
