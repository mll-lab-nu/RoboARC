"""Ported-tier variant of the rd probe_place.py (missing-refs agent): usage probe_place_p.py <instance> <actions.npy> <rows> <out.json>"""

import json, sys, itertools
from pathlib import Path
import numpy as np
from rcb_topogym import Session
from topogym.planning.buzz_wire_3d.redesign.io import SessionIO
from topogym.planning.buzz_wire_3d.redesign.oracle import TierOracle
from topogym.assets.geometry.buzz_wire_3d import endpoint_pose
from scipy.spatial.transform import Rotation as R
inst, acts, nrows, outf = sys.argv[1:5]
s = Session("buzz_wire_3d", instance=inst, strict=True)
io = SessionIO(s)
rows = np.load(acts)[:int(nrows)]
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
target_p = tp + orc.origin
hy_t = orc._yaw(tr.apply([1, 0, 0]))
d = (hy_t - orc._yaw(Rw.apply([1, 0, 0])) + np.pi) % (2 * np.pi) - np.pi
post = np.array([target_p[0], target_p[1], pw[2]])
flip = Rw.apply([0, 0, 1])[2] < 0
out = dict(dyaw_deg=float(np.degrees(d)), flip=bool(flip), wand_p=pw.tolist(), grasps={}, rot={}, push={})
kinds = ["topb", "top"] if flip else ["top", "topb"]
for kind, roll, fr, pitch, dd in itertools.product(kinds, (0, 180, 90, 270), (0.55, 0.7, 0.8, 0.9, 1.0),
                                                   (-40, -30, -20, -10, 0, 10, 20, 30, 40), (2, 0)):
    spec = f"{kind}:{roll}:{fr}:{pitch}:{dd}"
    c = orc.grasp_candidates(spec, pw, Rw, require_sweep=False, deck_ok=True, n_keep=1)
    if c:
        out["grasps"][spec] = round(float(c[0][0]), 4)
print("grasps", len(out["grasps"]), list(out["grasps"])[:20], flush=True)
for spec in list(out["grasps"])[:40]:
    for turn in ("final", 10, 20, 40, -10, -20, -40):
        step = d if turn == "final" else np.deg2rad(turn)
        p_end = post + R.from_euler("z", step).apply(pw - post); p_end[2] = pw[2]
        if turn == "final":
            p_end = target_p.copy()
        for sweep in (False,):
            pl = orc._rot_plan(spec, pw, Rw, post, step, p_end, sweep=sweep)
            if pl is not None:
                out["rot"].setdefault(spec, []).append(turn)
print("rot", out["rot"], flush=True)
for beta in (15, 25, 40, 60, -15, -25, -40, -60):
    for frac, tilt, pad in itertools.product((0.4, 0.6, 0.85, 1.0), (0.0, 0.35, 0.7, 1.05, 1.3, 1.5), (0.0015, 0.006, 0.012)):
        plan_, why = orc.plan_push(pw, Rw, post, np.deg2rad(beta), frac=frac, tilt=tilt, pad_above_deck=pad)
        if plan_ is not None:
            out["push"].setdefault(str(beta), []).append([frac, tilt, pad])
print("push", {k: len(v) for k, v in out["push"].items()}, flush=True)
Path(outf).write_text(json.dumps(out, indent=1))
print("RESULT", json.dumps(dict(dyaw=out["dyaw_deg"], grasps=len(out["grasps"]), rot=out["rot"], push={k: v[:3] for k, v in out["push"].items()})))
