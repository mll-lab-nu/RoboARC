"""Wide grasp search at a resting wand (replayed actions): rd grasp_candidates over a wide spec grid (kinds, roll, grip
fraction, pitch, depth), and for each feasible grasp the turn plans (rd _rot_plan) towards the finish heading.
usage: probe_wide.py <instance> <actions.npy> <rows> <k> <K> [--turns final,10,20]"""
import sys, json, itertools
import numpy as np
from scipy.spatial.transform import Rotation as R
from rcb_topogym import Session
from topogym.planning.buzz_wire_3d.redesign.io import SessionIO
from topogym.planning.buzz_wire_3d.redesign.oracle import TierOracle
from topogym.assets.geometry.buzz_wire_3d import endpoint_pose
inst, acts, nrows, k, K = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
s = Session("buzz_wire_3d", instance=inst, strict=True)
io = SessionIO(s)
rows = np.load(acts)[:nrows]
for r in rows:
    s.step(r)


class PD:
    sleeves = (); sleeve_radius_m = 0.0; robot_mount = None
    def __init__(self, t): self.t = t
    def layout(self): return self.t.layout
    def wire(self): return self.t.route
    def origin(self): return np.asarray(self.t.origin, float)


orc = TierOracle(io, PD(s.env._task), [], log=lambda **kv: None, rng=0, seeds=60)
orc._last_q = np.asarray(rows[-1][:7], float)
pw, Rw = io.wand_pose()
tp, tr = endpoint_pose(orc.route, 1)
target = tp + orc.origin
hy_t = orc._yaw(tr.apply([1, 0, 0]))
d = (hy_t - orc._yaw(Rw.apply([1, 0, 0])) + np.pi) % (2 * np.pi) - np.pi
post = np.array([target[0], target[1], pw[2]])
specs = [f"{kd}:{r}:{f}:{p}:{dd}" for kd in ("top", "topb", "side", "sideb", "end") for r in (0, 90, 180, 270)
         for f in (0.55, 0.7, 0.85, 1.0) for p in (-60, -45, -30, -15, 0, 15, 30, 45, 60) for dd in (0, 2, 5)]
specs = specs[k::K]
print(json.dumps(dict(dyaw_now=round(float(np.degrees(d)), 1), n_specs=len(specs),
                      metrics={kk: io.s.env._task.result.metrics.get(kk) for kk in ("distance_to_finish_m", "finish_angle_deg")})), flush=True)
for sp in specs:
    c = orc.grasp_candidates(sp, pw, Rw, require_sweep=False, deck_ok=True, n_keep=4)
    if not c:
        continue
    # physically: the closed pads must not go into the deck (deck_ok lets the plan touch it)
    from topogym.planning.buzz_wire_3d.redesign.oracle import gap_for
    orc.kin.set_gap(gap_for(sp))
    deckmin = []
    for mg, qg, qpre in c:
        dm = min([dist for dist, kind, g1, g2 in orc.kin.contacts(qg, wand=(pw, Rw)) if kind == "robot-static" and orc.is_deck(g1, g2)] or [1.0])
        deckmin.append(dm)
    if max(deckmin) < -0.001:
        print(json.dumps(dict(spec=sp, rejected="pads into the deck", deck_mm=round(1e3 * max(deckmin), 2))), flush=True)
        continue
    turns = {}
    for name, st in (("final", d), ("half", d / 2), ("m10", np.sign(d) * np.deg2rad(10)), ("m5", np.sign(d) * np.deg2rad(5))):
        pe = target.copy() if name == "final" else post + R.from_euler("z", st).apply(pw - post)
        pe[2] = pw[2]
        turns[name] = orc._rot_plan(sp, pw, Rw, post, st, pe, sweep=False) is not None
    print(json.dumps(dict(spec=sp, margin_mm=round(1e3 * float(c[0][0]), 2), turns=turns)), flush=True)
print("DONE", flush=True)
