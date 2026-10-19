"""Matplotlib previews of redesigned knot scenes (task-local frame): top view with the ALOHA bases and reach, fixtures,
regions, crossings (over strand marked), and a side (x-z) profile. CPU only; <= 800 px wide PNGs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BASES = {"left": (-.469, -.019), "right": (.469, -.019)}
REACH_M = .59


def _fixture_patch(ax, f, **kw):
    import matplotlib.patches as mp
    from matplotlib.transforms import Affine2D
    if f["kind"] == "cylinder":
        ax.add_patch(mp.Circle(f["pos"][:2], f["radius"], **kw))
    elif f["kind"] == "box":
        sx, sy = f["size"][0], f["size"][1]
        r = mp.Rectangle((-sx / 2, -sy / 2), sx, sy, **kw)
        r.set_transform(Affine2D().rotate_deg(f.get("yaw_deg", 0.)).translate(*f["pos"][:2]) + ax.transData)
        ax.add_patch(r)
    elif f["kind"] == "capsule":
        a, b = np.asarray(f["fromto"][:3]), np.asarray(f["fromto"][3:])
        ax.plot([a[0], b[0]], [a[1], b[1]], lw=2, color=kw.get("facecolor", "k"))


def top_view(scene, out_png, title=None, centerlines=None, show_reach=True, xlim=(-.55, .55), ylim=(-.38, .38),
             crossings=True, hands=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from topogym.tasks.knot_untying.redesign.geometry import projected_crossings
    ropes = centerlines if centerlines is not None else [np.asarray(r["centerline"]) for r in scene["ropes"]]
    fig, ax = plt.subplots(figsize=(8, 5.6), dpi=100)
    ax.set_facecolor("#eef0f2")
    if show_reach:
        for side, b in BASES.items():
            ax.add_patch(plt.Circle(b, REACH_M, fill=False, ls=":", lw=1, color="#7a8a99"))
            ax.plot(*b, marker="s", ms=9, color="#55606b")
            ax.text(b[0], b[1] - .035, f"{side} base", ha="center", fontsize=7, color="#55606b")
    for name, poly in (scene.get("regions") or {}).items():
        P = np.asarray(poly + [poly[0]])
        ax.fill(P[:, 0], P[:, 1], color="#bfe3c4", alpha=.5, lw=0)
        ax.plot(P[:, 0], P[:, 1], ls="--", lw=1.2, color="#2f7d3b")
        ax.text(P[:, 0].mean(), P[:, 1].max() + .008, f"region ({name})", fontsize=7, color="#2f7d3b", ha="center")
    for f in scene.get("fixtures", []):
        col = {"clamp": "#3c4656", "post": "#7b6f5f", "block": "#8a8070", "curb": "#8a8070"}.get(f.get("role"), "#888")
        _fixture_patch(ax, f, facecolor=col, edgecolor="k", lw=.6, alpha=.9)
    cmap = plt.get_cmap("viridis")
    colors = ["#d9822b", "#2b7bd9", "#9b3dc4"]
    for k, p in enumerate(ropes):
        p = np.asarray(p)
        ax.plot(p[:, 0], p[:, 1], lw=3.2, color=colors[k % 3], solid_capstyle="round", zorder=3)
        sc = ax.scatter(p[::6, 0], p[::6, 1], c=p[::6, 2] * 1000, cmap=cmap, s=2, vmin=4, vmax=16, zorder=4)
        ax.plot(*p[0, :2], "o", ms=6, color="#2c73c2", zorder=5)
        ax.plot(*p[-1, :2], "o", ms=6, color="#c23c3c", zorder=5)
    if crossings:
        cr = projected_crossings([np.asarray(p) for p in ropes])
        for c in cr:
            ax.plot(*c["xy"], "x", color="k", ms=5, zorder=6)
        ax.text(xlim[0] + .01, ylim[1] - .03, f"projected crossings: {len(cr)}", fontsize=9)
    if hands:
        for h, path in hands.items():
            P = np.asarray(path)
            if len(P):
                ax.plot(P[:, 0], P[:, 1], lw=.8, ls="-", color={"left": "#1f9e3a", "right": "#b8329a"}[h], alpha=.8)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, task-local; robot midpoint at 0)")
    ax.set_ylabel("y (m)")
    if title:
        ax.set_title(title, fontsize=10)
    fig.colorbar(sc, ax=ax, shrink=.6, label="centre-line height above table (mm)")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def load_scene(path):
    return json.loads(Path(path).read_text())
