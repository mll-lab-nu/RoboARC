"""At a landed wand (replayed actions), which headings about the finish post would be graspable from above? The wand
pose is rotated (kinematically) about the post by each offset; rd grasp_candidates over a spec grid; prints counts.
usage: probe_headings.py <instance> <actions.npy> <rows>"""
import sys, json
import numpy as np
from scipy.spatial.transform import Rotation as R
from rcb_topogym import Session
from topogym.planning.buzz_wire_3d.redesign.io import SessionIO
from topogym.planning.buzz_wire_3d.redesign.oracle import TierOracle
from topogym.assets.geometry.buzz_wire_3d import endpoint_pose
inst, acts, nrows = sys.argv[1], sys.argv[2], int(sys.argv[3])
s = Session("buzz_wire_3d", instance=inst, strict=True)
io = SessionIO(s)
rows = np.load(acts)[:nrows]
for r in rows:
    io.step(r[:7], r[7])


class PD:
    sleeves = (); sleeve_radius_m = 0.0; robot_mount = None
    def __init__(self, t): self.t = t
    def layout(self): return self.t.layout
    def wire(self): return self.t.route
    def origin(self): return np.asarray(self.t.origin, float)


orc = TierOracle(io, PD(s.env._task), [], log=lambda **kv: None, rng=0)
orc._last_q = np.asarray(rows[-1][:7], float)
pw, Rw = io.wand_pose()
tp, tr = endpoint_pose(orc.route, 1)
target = tp + orc.origin
post = np.array([target[0], target[1], pw[2]])
hy_t = orc._yaw(tr.apply([1, 0, 0]))
flip = Rw.apply([0, 0, 1])[2] < 0
kinds = ["topb", "top"] if flip else ["top", "topb"]
specs = [f"{k}:{r}:{f}:{p}:{dd}" for k in kinds[:1] for r in (0, 180, 90, 270) for f in (0.55, 0.7, 0.8, 0.9, 1.0)
         for p in (-40, -20, 0, 20, 40) for dd in (2,)]
d0 = (hy_t - orc._yaw(Rw.apply([1, 0, 0])) + np.pi) % (2 * np.pi) - np.pi
print(json.dumps(dict(dyaw_now=round(float(np.degrees(d0)), 1), flip=bool(flip))), flush=True)
for off in range(-180, 180, 15):
    Rz = R.from_euler("z", np.deg2rad(off))
    p2 = post + Rz.apply(pw - post); R2 = Rz * Rw
    ok = [sp for sp in specs if orc.grasp_candidates(sp, p2, R2, require_sweep=False, deck_ok=True, n_keep=1)]
    d = (hy_t - orc._yaw(R2.apply([1, 0, 0])) + np.pi) % (2 * np.pi) - np.pi
    print(json.dumps(dict(offset=off, dyaw_after=round(float(np.degrees(d)), 1), graspable=len(ok), first=ok[:3])), flush=True)
