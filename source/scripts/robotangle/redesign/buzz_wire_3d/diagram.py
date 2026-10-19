"""A tier's diagram (matplotlib, no physics): the wire in 3D and from above, the rest sleeves (numbered), the insulated
finish post, the wooden obstacles, the robot base, the start and finish wand poses, and the oracle's held segments
coloured by grasp (from a chain JSON: [{"s0", "s1", "grasp", "tcp": [[x, y, z], ...] (optional)}]).
usage: diagram.py <tier> <out.png> [--chain chain.json] [--title text]"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from topogym.tasks.buzz_wire_3d.redesign import tiers
from topogym.tasks.buzz_wire_3d.redesign.design import rail_arc
from topogym.assets.geometry.buzz_wire_3d import endpoint_pose

COL = {"top": "#1f77b4", "topb": "#17becf", "side": "#d62728", "sideb": "#ff7f0e", "end": "#9467bd"}

ap = argparse.ArgumentParser()
ap.add_argument("tier"); ap.add_argument("out"); ap.add_argument("--chain", default=None); ap.add_argument("--title", default="")
a = ap.parse_args()
d = tiers.design(a.tier)
w = d.wire(); rail = w.rail; s = rail_arc(rail)
chain = json.load(open(a.chain)) if a.chain else []
if isinstance(chain, dict):
    chain = chain.get("segments", [])
fig = plt.figure(figsize=(16, 7))
ax3 = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(1, 2, 2)
base = tiers.ROBOT_XY - np.asarray(d.center_xy_m)


def seg_mask(s0, s1):
    return (s >= s0) & (s <= s1)


ax3.plot(*rail.T, color="#b87333", lw=1.2, alpha=0.6)
ax2.plot(rail[:, 0], rail[:, 1], color="#b87333", lw=1.2, alpha=0.6)
for seg in chain:
    m = seg_mask(seg["s0"], seg["s1"])
    c = COL.get(seg["grasp"].split(":")[0], "k")
    ax3.plot(*rail[m].T, color=c, lw=3.5)
    ax2.plot(rail[m, 0], rail[m, 1], color=c, lw=3.5, label=f'{seg["grasp"]}  s {seg["s0"]:.2f}-{seg["s1"]:.2f}')
    if seg.get("tcp"):
        t = np.asarray(seg["tcp"]) - d.origin()
        ax3.plot(*t.T, color=c, lw=0.8, ls="--")
for k, (s0, s1) in enumerate(d.sleeves):
    m = seg_mask(s0, s1)
    last = k == len(d.sleeves) - 1
    ax3.plot(*rail[m].T, color="k", lw=6 if not last else 4)
    ax2.plot(rail[m, 0], rail[m, 1], color="k", lw=6 if not last else 4)
    p = rail[m][len(rail[m]) // 2]
    lab = "finish (insulated)" if last else f"S{k + 1}"
    ax3.text(*p + [0, 0, 0.015], lab, fontsize=9)
    ax2.annotate(lab, p[:2], xytext=(4, 4), textcoords="offset points", fontsize=9)
for ob in d.obstacles:
    if ob["kind"] == "cylinder":
        th = np.linspace(0, 2 * np.pi, 40)
        x = ob["pos"][0] + ob["radius"] * np.cos(th); y = ob["pos"][1] + ob["radius"] * np.sin(th)
        ax2.fill(x, y, color="#c8a165", alpha=0.8)
        z0 = ob["pos"][2] - ob["height"] / 2; z1 = ob["pos"][2] + ob["height"] / 2
        for zz in (z0, z1):
            ax3.plot(x, y, zz, color="#c8a165")
        ax3.plot([ob["pos"][0]] * 2, [ob["pos"][1]] * 2, [z0, z1], color="#c8a165", lw=4)
    else:
        from scipy.spatial.transform import Rotation as R
        hx, hy, hz = np.asarray(ob["size"]) / 2
        Rz = R.from_euler("xyz", ob.get("euler_deg", (0, 0, 0)), degrees=True)
        cs = np.array([[sx * hx, sy * hy, 0] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))])
        cs = Rz.apply(cs) + ob["pos"]
        ax2.fill(cs[:, 0], cs[:, 1], color="#c8a165", alpha=0.8)
for end, col in ((0, "g"), (1, "r")):
    p, r = endpoint_pose(w, end)
    h = r.apply([1, 0, 0])
    ax3.quiver(*p, *(0.06 * h), color=col, lw=2)
    ax2.arrow(p[0], p[1], 0.06 * h[0], 0.06 * h[1], color=col, width=0.003)
    ax2.plot(*p[:2], "o", color=col)
bx, by = np.asarray(d.base_size_m[:2]) / 2
ax2.plot([-bx, bx, bx, -bx, -bx], [-by, -by, by, by, -by], "k--", lw=0.8)
ax2.plot(*base, "ks", ms=12); ax2.annotate("robot base", base, xytext=(6, -12), textcoords="offset points")
ax2.set_aspect("equal"); ax2.set_xlim(min(-bx, base[0]) - 0.05, bx + 0.05); ax2.set_ylim(-by - 0.05, max(by, base[1]) + 0.05)
ax2.set_title("from above (deck frame, m): rests black, held segments coloured by grasp")
if chain:
    ax2.legend(fontsize=8, loc="lower right")
ax3.set_title("3D (deck frame, m)")
lo, hi = rail.min(0), rail.max(0)
c = (lo + hi) / 2; rr = (hi - lo).max() / 2
ax3.set_xlim(c[0] - rr, c[0] + rr); ax3.set_ylim(c[1] - rr, c[1] + rr); ax3.set_zlim(0.1425, 0.1425 + 2 * rr)
ax3.view_init(elev=28, azim=-60)
summ = d.summary()
fig.suptitle(a.title or f"{d.tier} ({d.name}): wire {summ['rail_length_m']} m, ring radial clearance "
             f"{1e3 * summ['radial_clearance_m']:.1f} mm, {len(d.sleeves) - 1} rest sleeves, {len(d.obstacles)} obstacles")
fig.tight_layout()
fig.savefig(a.out, dpi=110)
print("wrote", a.out)
