"""Can the traversal grip itself set the wand down on the finish pose? IK + clearance of the wand at the finish pose
(both faces, heading offsets, a few mm above the deck) with the prep's grasp. usage: held_place_probe.py <prep.npz>"""
import sys, json
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import nplan
from topogym.planning.buzz_wire_3d.redesign.wandpath import tcp_from_wand
from topogym.assets.geometry.buzz_wire_3d import endpoint_pose
from topogym.planning.buzz_wire_3d.redesign.kin import HOME
P = np.load(sys.argv[1])
s, kin, cls, deck = nplan._fast_model(P)
task = s.env._task
g = (P["g_p"], R.from_quat(P["g_q"]))
tp, tr = endpoint_pose(task.route, 1)
target = tp + task.origin
rng = np.random.default_rng(0)
seeds = [np.asarray(P["qpos"])[kin.qadr], HOME] + [rng.uniform(kin.lo, kin.hi) for _ in range(60)]
for flip in (False, True):
    for dy in (0, -10, 10, -20, 20, -40, 40):
        for dz in (0.0005, 0.004, 0.012):
            Rw = R.from_euler("z", np.deg2rad(dy)) * tr * (R.from_euler("x", 180, degrees=True) if flip else R.identity())
            pw = target + np.array([0, 0, dz])
            tcp, trr = tcp_from_wand(pw, Rw, g)
            best = None
            for sd in seeds:
                q, ok, pe, re = kin.ik(tcp, trr, sd, iters=300)
                if not ok:
                    continue
                rs, dk, ww, sp = nplan._check(kin, cls, deck, q, (pw, Rw))
                score = min(rs, dk + 0.0005, ww)
                if best is None or score > best[0]:
                    best = (score, rs, dk, ww, sp)
            print(json.dumps(dict(flip=flip, dyaw=dy, dz_mm=dz * 1e3, ik=best is not None,
                                  clear=None if best is None else [round(1e3 * x, 2) for x in best[1:4]] + [best[4]])), flush=True)
