"""probe_place.py <tier> <instance.json> <plan.json> <actions.npy> <rows> <out.json>: replay the rows (to the 'dropped'
row of a failed run), then probe the final place at the landed wand: grasp configurations over a wide spec grid
(top/topb, roll, fraction, pitch, depth), rotation plans for turns of +-20..+-140 deg, and closed-finger pushes over
fraction / tilt / pad height. Planning only (no motion after the replay)."""
import json, sys, itertools
from pathlib import Path
import numpy as np
import topogym
sys.path.insert(0, str(Path(topogym.__file__).resolve().parents[2] / "tools/redesign/buzz_wire_3d"))
import rdreg
vids = rdreg.register()
from rcb_topogym import Session
from topogym.tasks.buzz_wire_3d.redesign import tiers
from topogym.planning.buzz_wire_3d.redesign.io import SessionIO
from topogym.planning.buzz_wire_3d.redesign.oracle import TierOracle
from topogym.assets.geometry.buzz_wire_3d import endpoint_pose
from scipy.spatial.transform import Rotation as R
tier, inst, planf, acts, nrows, outf = sys.argv[1:7]
s = Session("buzz_wire_3d", instance=inst, strict=True)
io = SessionIO(s)
plan = json.loads(Path(planf).read_text())
rows = np.load(acts)[:int(nrows)]
for r in rows:
    io.step(r[:7], r[7])
orc = TierOracle(io, tiers.design(tier), [], log=lambda **kv: None, rng=0)
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
    for turn in (20, 40, 70, 100, 140, -20, -40, -70):
        step = np.deg2rad(turn)
        p_end = post + R.from_euler("z", step).apply(pw - post); p_end[2] = pw[2]
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
