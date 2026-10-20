"""Top view of a tier's robot-mount search: table, deck, wire, the default and the chosen Panda mount (link0 footprint,
the arm at home), and every screened mount coloured by its stage-A verdict (base screen JSONL).
usage: topview.py <tier> <screen.jsonl> <out.png> [--chosen x,y,yaw] [--frozen <dir> --assets <dir>]"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from topogym.tasks.buzz_wire_3d.redesign import tiers
from topogym.tasks.buzz_wire_3d.redesign.design import footprint_corners, TABLE_XY, DEFAULT_MOUNT

ap = argparse.ArgumentParser()
ap.add_argument("tier"); ap.add_argument("screen"); ap.add_argument("out")
ap.add_argument("--chosen", default=None); ap.add_argument("--frozen"); ap.add_argument("--assets")
a = ap.parse_args()
d = tiers.TIERS[a.tier][1]()
o = d.origin()
fig, ax = plt.subplots(figsize=(8, 7.5))
(tx0, ty0), (tx1, ty1) = TABLE_XY
ax.add_patch(Rectangle((tx0, ty0), tx1 - tx0, ty1 - ty0, fc="#eeeeee", ec="#888888", lw=1))
bx, by = d.base_size_m[0], d.base_size_m[1]
ax.add_patch(Rectangle((o[0] - bx / 2, o[1] - by / 2), bx, by, fc="#e8d3b0", ec="#9a7b4f", lw=1, label="deck"))
rail = d.wire().rail + o
ax.plot(rail[:, 0], rail[:, 1], color="#b5651d", lw=1.6, label="wire")
for s0, s1 in d.sleeves:
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(rail, axis=0), axis=1))]
    sel = (arc >= s0) & (arc <= s1)
    ax.plot(rail[sel, 0], rail[sel, 1], color="k", lw=4)
col = dict(legal="#bbbbbb", reset="#999999", chain="#d62728", regrasp="#ff7f0e", done="#2ca02c", nopick="#9467bd")
seen = set()
for line in open(a.screen):
    r = json.loads(line)
    x, y, yaw = r["mount"]
    st = r["stage"]
    if st == "done" and r.get("pick", {}).get("n", 1) == 0:
        st = "nopick"
    size = 30 if st != "done" else 20 + 0.5 * r.get("finish_cover_deg", 0)
    lab = {"legal": "illegal", "reset": "home pose collides", "chain": "a held segment infeasible",
           "regrasp": "a rest regrasp infeasible", "nopick": "no deck pick",
           "done": "all pass (size = finish grasp coverage)"}[st]
    ax.scatter([x], [y], s=size, c=col[st], alpha=.7, edgecolors="none", label=None if lab in seen else lab)
    seen.add(lab)
    t = np.deg2rad(yaw)
    ax.plot([x, x + .02 * np.cos(t)], [y, y + .02 * np.sin(t)], color=col[st], lw=.8)


def foot(m, **kw):
    ax.add_patch(Polygon(footprint_corners(m), closed=True, fill=False, **kw))


foot(DEFAULT_MOUNT, ec="#1f77b4", ls="--", lw=1.4, label="default mount")
if a.chosen:
    m = tuple(float(v) for v in a.chosen.split(","))
    foot(m, ec="#1f77b4", lw=2.2, label=f"chosen mount ({m[0]:.2f}, {m[1]:.2f}, {m[2]:.0f} deg)")
    if a.frozen:
        from topogym.planning.buzz_wire_3d.redesign.scene import planning_model
        from topogym.planning.buzz_wire_3d.redesign.kin import PandaKin, HOME
        import mujoco
        k = PandaKin(planning_model(a.frozen, a.assets, d, mount=m))
        k.set_q(HOME); mujoco.mj_kinematics(k.m, k.d)
        names = ["link0", "link2", "link4", "link6", "hand"]
        P = np.array([k.d.xpos[k.m.body(n).id] for n in names])
        ax.plot(P[:, 0], P[:, 1], "-o", color="#1f77b4", ms=3, lw=1.2, label="arm at home (chosen)")
ax.set_aspect("equal"); ax.set_xlim(tx0 - .02, tx1 + .02); ax.set_ylim(ty0 - .02, ty1 + .02)
ax.set_xlabel("world x (m)"); ax.set_ylabel("world y (m)")
ax.set_title(f"buzz_wire_3d {a.tier} ({d.name}): robot mount search, top view")
ax.legend(loc="upper right", fontsize=7)
fig.tight_layout(); fig.savefig(a.out, dpi=130)
print("wrote", a.out)
