"""Replay a run's actions, then map which (psi, lag) wand poses at the bottom of the finish post are reachable with
the held grasp, collision-free. usage: end_probe.py <instance> <actions.npy> [--rows N] [--s-back 0.008,0.03,0.06]"""
import sys, json, argparse
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
ap = argparse.ArgumentParser(); ap.add_argument("inst"); ap.add_argument("acts"); ap.add_argument("--rows", type=int)
ap.add_argument("--s-back", default="0.008,0.03,0.06"); ap.add_argument("--save"); ap.add_argument("--dx", type=float, default=0.0)
ap.add_argument("--dz", type=float, default=0.0)
a = ap.parse_args()
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from nudge_lib import Dist
from rcb_topogym import Session
from topogym.planning.buzz_wire_3d.redesign.kin import PandaKin, HOME
from topogym.planning.buzz_wire_3d.redesign.wandpath import WirePath, tcp_from_wand
from topogym.planning.buzz_wire_3d.redesign.feasibility import classify
s = Session("buzz_wire_3d", instance=a.inst, strict=True)
w = s.env._world; m, d = w.model, w.data
acts = np.load(a.acts)[:a.rows]
for r in acts:
    s.step(r)
hand, wand = m.body("hand").id, m.body("wand::wand").id
Rh = R.from_matrix(d.xmat[hand].reshape(3, 3)); tcp = d.xpos[hand] + Rh.apply([0, 0, .1029])
Rw = R.from_quat(np.array(d.xquat[wand])[[1, 2, 3, 0]]); pw = np.array(d.xpos[wand])
g = (Rw.inv().apply(tcp - pw) + np.array([a.dx, 0, a.dz]), Rw.inv() * Rh)
kin = PandaKin(m); kin.d.qpos[:] = d.qpos[:]
D = Dist(kin, classify(kin))
path = WirePath(s.env._task.rail)
qadr = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 8)]
q0 = np.array(d.qpos[qadr])
print("held", (w.read_state().get("grasp") or {}).get("active"), "cursor", s.env._task.result.metrics.get("route_cursor_m"), "g", np.round(g[0], 4).tolist())
rng = np.random.default_rng(0)
seeds = [q0, HOME] + [rng.uniform(kin.lo, kin.hi) for _ in range(30)]
out = {}
for sb in [float(x) for x in a.s_back.split(",")]:
    sv = path.length - sb
    print(f"--- s {sv:.3f} (back {sb})")
    for lag in range(-75, 76, 15):
        line = ""
        for j in range(24):
            psi = j * 2 * np.pi / 24
            p, Rr = path.frame(sv, psi, np.deg2rad(lag))
            tp, tr = tcp_from_wand(p, Rr, g)
            best = None
            for sd in seeds:
                q, ok, _, _ = kin.ik(tp, tr, sd, iters=200)
                if not ok:
                    continue
                kin.set_q(q); kin.set_wand(p, Rr); mujoco.mj_kinematics(m, kin.d)
                rs, pair, deck = D.robot_static()
                ww = D.wand_wire()
                if rs < 0.002 or deck < 0.002 or ww < 0.0005:
                    continue
                c = np.abs(q - q0).max()
                if best is None or c < best[0]:
                    best = (c, q)
            line += "." if best is None else ("O" if best[0] < 1.0 else "o")
            if best is not None:
                out[(round(sb, 3), lag, j)] = best[1].tolist()
        print(f"lag {lag:4d} {line}", flush=True)
if a.save:
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in out.items()}, open(a.save, "w"))
