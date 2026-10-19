"""Review previews of a redesigned claws tier (CPU only):
  - <tier>_storyboard.png: one panel per designed move (matplotlib 3D, A's frame): the lock (teal), the key (brown) at
    the move's start, ghosts along it, the claw's path in the move's colour, the move's label and why;
  - <tier>_start_<view>.png: the birth (both hands on their grips), MuJoCo render of the tier's own scene;
  - <tier>_path_<view>.png: the birth with the designed path drawn (the claw's path, coloured per move, and ghost keys
    at every move's start);
  - <tier>_goal_<view>.png: the key in the goal area held by the hand that ends with it (IK), the lock where it started;
  - <tier>_sheet.png: start / path / goal contact sheet (aloha view + top view).

  python previews_claws.py <tier> <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from previews.render import setup_cpu_gl          # noqa: E402

setup_cpu_gl()

import json                                        # noqa: E402
import numpy as np                                 # noqa: E402
from scipy.spatial.transform import Rotation       # noqa: E402

MOVE_COLORS = [(0.90, 0.30, 0.10, 1.), (0.10, 0.55, 0.95, 1.), (0.15, 0.75, 0.30, 1.), (0.85, 0.15, 0.60, 1.),
               (0.95, 0.75, 0.10, 1.), (0.55, 0.30, 0.85, 1.), (0.30, 0.80, 0.80, 1.), (0.6, 0.6, 0.6, 1.)]
VIEWS = {
    "aloha": dict(pos=(-0.1356786, -0.42, 1.68), lookat=(-0.1, 0.0, 0.97), fovy=54.0),
    "front": dict(azimuth=90.0, elevation=-25.0, distance=.85),
    "top": dict(azimuth=90.0, elevation=-89.0, distance=1.05),
    "side": dict(azimuth=0.0, elevation=-30.0, distance=.9),
    "close": dict(azimuth=60.0, elevation=-32.0, distance=.55),
}


def box_corners(b, T=np.eye(4)):
    P = np.asarray(b["pose"], float)
    s = np.asarray(b["size"], float) / 2
    c = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float) * s
    w = c @ P[:3, :3].T + P[:3, 3]
    return w @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]


FACES = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]


def draw_piece(ax, boxes, T, color, alpha):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    polys = []
    for b in boxes:
        c = box_corners(b, T)
        polys += [c[f] for f in FACES]
    ax.add_collection3d(Poly3DCollection(polys, facecolor=color, edgecolor=(0, 0, 0, .15 * alpha), alpha=alpha,
                                         linewidths=.2))


def claw_point(T):
    return (np.asarray(T) @ np.array([0., 0., 0., 1.]))[:3]


def storyboard(d, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from topogym.planning.interlocked_claws.redesign import paths
    A, B = d["A"], d["B"]
    moves = d["moves"]
    n = len(moves) + 1
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(4.4 * cols, 4.2 * rows))
    poses, starts, which = paths.dense(d["engaged"], moves, step_m=.004, step_deg=3.)
    allpts = np.concatenate([A.corners(), B.corners() @ np.asarray(d["engaged"])[:3, :3].T
                             + np.asarray(d["engaged"])[:3, 3], np.array([claw_point(P) for P in poses])])
    lo, hi = allpts.min(0), allpts.max(0)
    ctr, rad = (lo + hi) / 2, (hi - lo).max() / 2 * .85
    for k in range(n):
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        draw_piece(ax, A.boxes, np.eye(4), (.2, .53, .61), .18)
        i0 = starts[k] if k < len(moves) else len(poses) - 1
        draw_piece(ax, B.boxes, poses[i0], (.74, .40, .24), .9)
        if k < len(moves):
            idx = [j for j in range(len(poses)) if which[j] == k]
            for j in idx[len(idx) // 3::max(1, len(idx) // 3)][:2]:
                draw_piece(ax, B.boxes, poses[j], (.74, .40, .24), .18)
            pts = np.array([claw_point(poses[j]) for j in [i0] + idx])
            ax.plot(*pts.T, color=MOVE_COLORS[k % len(MOVE_COLORS)][:3], lw=2.5)
            ax.scatter(*pts[-1], color=MOVE_COLORS[k % len(MOVE_COLORS)][:3], s=25)
            mv = moves[k]
            title = mv.get("label", mv["kind"])
            sub = mv.get("why", "")
            if mv.get("hold"):
                sub += f"\n[hand change: {mv['hold']}]"
        else:
            title, sub = f"{n}. free (separated)", "then into the goal area"
        ax.set_title(title, fontsize=10)
        ax.text2D(.02, .02, sub, transform=ax.transAxes, fontsize=7, wrap=True)
        for f, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), ctr):
            f(c - rad, c + rad)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=28, azim=-60)
        ax.set_xlabel("x_A"); ax.set_ylabel("y_A"); ax.set_zlabel("z_A")
        ax.tick_params(labelsize=6)
    fig.suptitle(f"{d['tier']} ({d['name']}): {d['summary']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)
    plt.close(fig)


def _add_geom(scn, gtype, size, pos, mat, rgba):
    import mujoco
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).reshape(9),
                        np.asarray(rgba, np.float32))
    scn.ngeom += 1


def _add_segment(scn, a, b, radius, rgba):
    import mujoco
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).reshape(9),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1


def colour_pieces(model):
    """Brown target, teal lock, as the family's materials (the frozen scene's collision geoms are plain grey)."""
    import mujoco
    for g in range(model.ngeom):
        b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
        if b.startswith("piece_0::"):
            model.geom_rgba[g] = (.74, .40, .24, 1.)
        elif b.startswith("piece_1::"):
            model.geom_rgba[g] = (.20, .53, .61, 1.)


def render(model, data, overlay=None, views=("aloha", "front", "top", "close"), lookat=(-0.12, 0.0, 1.1), w=800,
           h=500, close_at=None):
    import mujoco
    from previews.render import _renderer, _camera
    mujoco.mj_forward(model, data)
    out = {}
    r = _renderer(model, w, h)
    try:
        for name in views:
            view = VIEWS[name]
            fovy0 = float(model.vis.global_.fovy)
            if "fovy" in view:
                model.vis.global_.fovy = float(view["fovy"])
            cam = _camera(model, view, close_at if (name == "close" and close_at is not None) else lookat)
            r.update_scene(data, camera=cam)
            if overlay:
                overlay(r.scene)
            out[name] = r.render().copy()
            model.vis.global_.fovy = fovy0
    finally:
        r.close()
    return out


def main(argv=None):
    import mujoco
    from PIL import Image, ImageDraw
    argv = sys.argv[1:] if argv is None else argv
    tier, out = argv[0], Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    from topogym.planning.interlocked_claws.redesign import paths
    from topogym.tasks.interlocked_claws.redesign.task import make_session
    from topogym.planning.interlocked_claws.redesign.robot_plan import Kin
    s = make_session(tier)
    from topogym.planning.interlocked_claws.redesign import tiers as _tiers
    d = dict(_tiers.make(tier), **{k: v for k, v in s.env._task.design_record.items() if k == "settle"})
    storyboard(d, out / f"{tier}_storyboard.png")
    m, data = s.env._world.model, s.env._world.data
    w = s.env._world
    st = w.read_state()

    com = {k: np.asarray(v, float) for k, v in getattr(s.env._task, "piece_com", {}).items()}

    def T_of(name):                        # the design frame (the body frame sits at the composite COM)
        b = st["bodies"][name]
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(np.asarray(b["quaternion_wxyz"], float)[[1, 2, 3, 0]]).as_matrix()
        T[:3, 3] = b["position_m"]
        if name in com:
            T[:3, 3] = T[:3, 3] - T[:3, :3] @ com[name]
        return T
    T_A, T_B = T_of("piece_1"), T_of("piece_0")
    rel0 = np.linalg.inv(T_A) @ T_B
    poses, starts, which = paths.dense(rel0, d["moves"], step_m=.004, step_deg=2.)
    world = [T_A @ P for P in poses]
    claw = [claw_point(T) for T in world]
    B_boxes = d["B"].boxes

    def overlay_path(scn):
        for j in range(1, len(claw)):
            _add_segment(scn, claw[j - 1], claw[j], .004, MOVE_COLORS[which[j] % len(MOVE_COLORS)])
        for k, i0 in enumerate(starts + [len(world) - 1]):
            T = world[i0]
            for b in B_boxes:
                P = T @ np.asarray(b["pose"], float)
                _add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, np.asarray(b["size"]) / 2, P[:3, 3], P[:3, :3],
                          (.74, .40, .24, .30))
    colour_pieces(m)
    close_at = T_A[:3, 3]
    sheet = {}
    imgs = render(m, data, close_at=close_at)
    for v, im in imgs.items():
        Image.fromarray(im).save(out / f"{tier}_start_{v}.png", optimize=True)
    sheet["start"] = imgs
    imgs = render(m, data, overlay_path, close_at=close_at)
    for v, im in imgs.items():
        Image.fromarray(im).save(out / f"{tier}_path_{v}.png", optimize=True)
    sheet["path"] = imgs
    # goal: B at the end of the path, carried into the goal area by the hand that ends with it
    kin = Kin(m)
    holds = dict(d["hands"])
    for mv in d["moves"]:
        holds.update(mv.get("hold") or {})
    arm = [a for a, g in holds.items() if g and g.startswith("b_")][0]
    region = s.env._task.goal_region
    TB = world[-1].copy()
    wc = d["B"].corners() @ TB[:3, :3].T + TB[:3, 3]
    lo, hi = wc.min(0), wc.max(0)
    centre = (lo + hi) / 2
    goal = np.array([region.center_xy_m[0], region.center_xy_m[1], region.table_height_m + .04 + (centre[2] - lo[2])])
    TB[:3, 3] += goal - centre
    q0 = {a: data.qpos[kin.qadr[a]].copy() for a in ("left", "right")}
    T_tcp = TB @ np.asarray(d["B"].grips[holds[arm]], float)
    seeds = [q0[arm], np.array([0., -.5, .8, 0., .6, 0.]), np.array([.5, -.3, .6, 0., .8, 0.]),
             np.array([-.5, -.3, .6, 0., .8, 0.])]
    q, ep, er = kin.ik_multi(arm, T_tcp, seeds)
    qa = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "piece_0::root")]
    data.qpos[kin.qadr[arm]] = q
    data.qpos[qa:qa + 3] = TB[:3, 3] + TB[:3, :3] @ com.get("piece_0", np.zeros(3))
    data.qpos[qa + 3:qa + 7] = Rotation.from_matrix(TB[:3, :3]).as_quat()[[3, 0, 1, 2]]
    imgs = render(m, data, close_at=close_at)
    for v, im in imgs.items():
        Image.fromarray(im).save(out / f"{tier}_goal_{v}.png", optimize=True)
    sheet["goal"] = imgs
    # contact sheet: rows start / path / goal, columns aloha / top
    tiles = []
    for row in ("start", "path", "goal"):
        for v in ("aloha", "close"):
            im = Image.fromarray(sheet[row][v]).resize((400, 250))
            ImageDraw.Draw(im).text((6, 4), f"{row} ({v})", fill=(255, 255, 255))
            tiles.append(im)
    S = Image.new("RGB", (800, 750), (255, 255, 255))
    for i, t in enumerate(tiles):
        S.paste(t, ((i % 2) * 400, (i // 2) * 250))
    S.save(out / f"{tier}_sheet.png", optimize=True)
    info = dict(tier=tier, birth_sha=s.state_sha(), goal_ik_error_m=ep, goal_hand=arm,
                goal_region=dict(center=list(region.center_xy_m), half=list(region.half_size_m)),
                moves=[mv.get("label") for mv in d["moves"]])
    (out / f"{tier}_preview.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
