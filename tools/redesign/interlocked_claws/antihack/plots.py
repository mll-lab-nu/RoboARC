"""Trajectory-complexity figures (matplotlib, CPU): what the metrics counted, drawn over the TCP paths.

  python -m antihack.plots --metrics metrics_F_T.json --rows privileged.npz --instance instance.json --out fig.png
  (or --plan plan.json for a planned oracle; --title "...")
Left: top view of both TCP paths (left arm blue, right arm orange) with the counted waypoints (dots), grasps (green
up-triangles) and releases (red down-triangles). Right: timeline of the motion phases (coloured by label) and the
holdings per arm, with the summary numbers. <= 800 px wide.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COLORS = dict(left="#2a6fdb", right="#e8710a")
PHASE = dict(approach="#9ecae1", transport="#31a354", bimanual_transport="#006d2c", retreat="#fdae6b",
             reposition="#bdbdbd")


def plot(P, dt, traj: dict, detail: dict, out_png, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.6), dpi=100, gridspec_kw=dict(width_ratios=[1, 1.25]))
    arms = ("left", "right")
    wp = detail.get("waypoint_rows") or {}
    for k, arm in enumerate(arms):
        a1.plot(P[:, k, 0], P[:, k, 1], color=COLORS[arm], lw=1.0, label=f"{arm} TCP")
        rows = [r for r in wp.get(arm, []) if r < len(P)]
        a1.scatter(P[rows, k, 0], P[rows, k, 1], s=9, color=COLORS[arm], zorder=3)
    for e in detail.get("grasp_events") or []:
        k = arms.index(e["arm"]); r = min(int(e["row"]), len(P) - 1)
        if e["event"] == "grasp":
            a1.scatter(P[r, k, 0], P[r, k, 1], marker="^", s=40, color="#2ca02c", edgecolor="k", lw=0.4, zorder=4)
        else:
            a1.scatter(P[r, k, 0], P[r, k, 1], marker="v", s=40, color="#d62728", edgecolor="k", lw=0.4, zorder=4)
    a1.set_aspect("equal", adjustable="datalim")
    a1.set_xlabel("x (m)", fontsize=7); a1.set_ylabel("y (m)", fontsize=7)
    a1.tick_params(labelsize=6)
    a1.legend(fontsize=6, loc="best")
    a1.set_title("top view: TCP paths, waypoints, grasps / releases", fontsize=7)
    # timeline
    T = len(P) * dt
    for ph in detail.get("phases") or []:
        for arm in ph.get("arms", []):
            y = 1.0 if arm == "left" else 0.0
            a2.barh(y + 0.25, ph["t1"] - ph["t0"], left=ph["t0"], height=0.35, color=PHASE.get(ph["label"], "#999"),
                    edgecolor="white", lw=0.3)
    for e in detail.get("grasp_events") or []:
        if e["event"] != "grasp":
            continue
        y = 1.0 if e["arm"] == "left" else 0.0
        t0 = e["row"] * dt; t1 = e.get("end", len(P)) * dt
        a2.barh(y - 0.15, t1 - t0, left=t0, height=0.2, color="#555", alpha=0.8)
        if e.get("kind") in ("handover", "regrasp"):
            a2.text(t0, y - 0.38, e["kind"][:2].upper(), fontsize=5)
    a2.set_yticks([0.05, 1.05]); a2.set_yticklabels(["right", "left"], fontsize=7)
    a2.set_xlim(0, T); a2.set_ylim(-0.6, 1.6)
    a2.set_xlabel("time (s)", fontsize=7); a2.tick_params(labelsize=6)
    from matplotlib.patches import Patch
    a2.legend(handles=[Patch(color=c, label=l) for l, c in PHASE.items()] + [Patch(color="#555", label="holding")],
              fontsize=5, ncol=3, loc="upper right")
    s = (f"phases {traj.get('phases_total')} (motion {traj.get('motion_phases')}), waypoints {traj.get('waypoints_total')}, "
         f"TCP {traj.get('tcp_path_m_total')} m, grasps {traj.get('grasp_count')} (hand-overs {traj.get('handover_count')}, "
         f"regrasps {traj.get('regrasp_count')}), {traj.get('duration_s')} s")
    a2.set_title(s, fontsize=6)
    if title:
        fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return str(out_png)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=None); ap.add_argument("--rows", default=None)
    ap.add_argument("--instance", default=None); ap.add_argument("--plan", default=None)
    ap.add_argument("--family", default="rope_threading"); ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from .aloha import CONTROL_DT_S
    if a.plan:
        from .trajectory import densify_plan, metrics_from_plan
        plan = json.loads(Path(a.plan).read_text())
        fk = None
        if a.instance:
            from .aloha import AlohaFK
            fk = AlohaFK.from_instance(a.instance)
        P, *_rest = densify_plan(plan, fk)
        dt = float(plan.get("control_dt_s", 0.04))
        m = metrics_from_plan(plan, fk)
        traj = m
        detail = dict(waypoint_rows={k: v["waypoint_rows"] for k, v in m["per_arm"].items()},
                      grasp_events=m["grasp_events"], phases=m["phases"])
    else:
        from .aloha import AlohaFK
        mj = json.loads(Path(a.metrics).read_text())
        traj, detail = mj["trajectory"], mj["trajectory_detail"]
        z = np.load(a.rows, allow_pickle=False)
        fk = AlohaFK.from_instance(a.instance)
        P, _ = fk.tcp_path(z["actions"][:traj["rows"]], [str(x) for x in z["action_layout"]])
        dt = CONTROL_DT_S.get(mj.get("family", a.family), 0.04)
    print(plot(P, dt, traj, detail, a.out, a.title))


if __name__ == "__main__":
    main()
