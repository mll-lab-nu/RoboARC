"""Heuristic breakers: cheap policies that try to solve a new scene the easy way. A breaker is a TCP program (straight
segments between key points, gripper actions at key points) built from family-supplied (or scene-derived) key points;
it is converted to action rows by IK and checked kinematically (reach, joint speed, robot vs static obstacles) and, where
the scene builds as a Session, by running the rows (success, closest approach).

Breakers (constructors return a list of key points; `program_rows` turns them into rows):
  straight_line(keypoints)                         the family's own key points joined by straight TCP moves
  shortest_route(arm, grasp, goal, ...)            grasp at A, carry straight to B (ignores obstacles), release
  drag_through_rings(arm, tip, rings, ...)         grasp the rope tip, pull it straight through the ring centres in order
  pull_both_ends(end_a, end_b, ...)                both arms grasp the two rope ends and pull apart (knots)
  push_sweep(arm, start, end, ...)                 a closed gripper swept across the cloth just above the table
  two_corner_lift_lay(corner_a, corner_b, ...)     both arms lift two cloth corners, draw back and lay the cloth down
  corner_to_rest(points_now, points_rest, tilt)    one arm at a time (tilted 45 deg) puts each displaced corner / edge
                                                   midpoint back at its flat-rest position (the unfold heuristic)
Key point: {"arm": "left"|"right", "pos": [x, y, z], "yaw": rad (closing axis about z; 0 = the arm's natural top-down),
            "tilt": rad, "grip": "open"|"close"|"keep", "speed": m/s, "dwell_s": s,
            "with": [{"arm", "pos", "yaw", "grip"}]   (optional: the other arm moves simultaneously, bimanual)}

CLI (the cloud host, env sourced):
  python -m antihack.breakers kinematic --instance NEW.json --family F (--spec breaker.json | --auto) --out result.json
  python -m antihack.breakers session (--instance NEW.json | --builder mod:fn [--builder-args JSON]) --family F
         (--spec breaker.json | --auto) --out DIR      # through launch.sh: replays the rows in a Session
spec: {"name": "drag", "breaker": "drag_through_rings", "args": {...}} or {"name": "...", "keypoints": [...]}
(a list of such specs runs them all). --auto derives specs from the scene: rope_threading: drag the rope's two ends
through the rings (both ring orders); knot_untying: pull both ends apart (two directions); blanket_flatten: two-corner
lift-and-lay (two corner pairs) and a push sweep.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .aloha import (ARMS, CONTROL_DT_S, GRIPPER_CLOSED_NATIVE, GRIPPER_OPEN_NATIVE, HOME, JOINT_SPEED_LIMIT_RAD_S,
                    AlohaFK, layout_columns)

PINCH_NATIVE = 0.002          # the rope agent's relay pinch opening (a closed ALOHA gripper on a rope)
HOVER_M = 0.06


def top_down_R(yaw: float = 0.0, arm: str = "left", tilt: float = 0.0) -> np.ndarray:
    """TCP rotation for a top-down grasp: the hand's x axis (approach) points down, its y axis (the closing axis
    between the pads) is horizontal at `yaw` from the arm's natural direction (left +y, right -y); `tilt` > 0 turns
    the approach about the closing axis so the fingertips point AWAY from the arm's base (forward: a longer reach)."""
    s = 1.0 if arm == "left" else -1.0
    y = np.array([-s * np.sin(yaw), s * np.cos(yaw), 0.0])
    x = np.array([0.0, 0.0, -1.0])
    if tilt:
        c, sn = np.cos(-tilt), np.sin(-tilt)
        k = y
        x = x * c + np.cross(k, x) * sn + k * (k @ x) * (1 - c)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def approach_R(approach, arm: str = "left", roll: float = 0.0) -> np.ndarray:
    """TCP rotation whose approach axis (the hand's x) points along `approach` (e.g. the direction of travel through a
    ring); the closing axis horizontal (rolled by `roll` about the approach)."""
    x = np.asarray(approach, float); x = x / np.linalg.norm(x)
    ref = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    y = np.cross(ref, x); y /= np.linalg.norm(y)
    if arm == "right":
        y = -y
    if roll:
        c, sn = np.cos(roll), np.sin(roll)
        y = y * c + np.cross(x, y) * sn
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


# ------------------------------------------------------------------------------------------------ constructors
def kp(arm, pos, grip="keep", yaw=0.0, speed=0.08, dwell_s=0.2, tilt=0.0, **extra):
    return dict(arm=arm, pos=[float(x) for x in pos], grip=grip, yaw=float(yaw), speed=float(speed),
                dwell_s=float(dwell_s), tilt=float(tilt), **extra)


def straight_line(keypoints):
    return list(keypoints)


def grasp_at(arm, point, yaw=0.0, hover=HOVER_M, speed=0.08):
    p = np.asarray(point, float)
    return [kp(arm, p + [0, 0, hover], "open", yaw, speed), kp(arm, p, "open", yaw, 0.04), kp(arm, p, "close", yaw, 0.04, 0.4)]


def release_at(arm, point, yaw=0.0, hover=HOVER_M):
    p = np.asarray(point, float)
    return [kp(arm, p, "open", yaw, 0.04, 0.4), kp(arm, p + [0, 0, hover], "open", yaw, 0.06)]


def shortest_route(arm, grasp, goal, *, yaw=0.0, carry_height=0.05):
    """Grasp at `grasp`, lift, carry in a straight line to above `goal` (ignoring every obstacle), lower, release."""
    g, e = np.asarray(grasp, float), np.asarray(goal, float)
    return (grasp_at(arm, g, yaw) + [kp(arm, g + [0, 0, carry_height], "keep", yaw, 0.06),
                                      kp(arm, e + [0, 0, carry_height], "keep", yaw, 0.08),
                                      kp(arm, e, "keep", yaw, 0.04)] + release_at(arm, e, yaw))


def drag_through_rings(arm, tip, rings, *, lead_m=0.07, exit_m=0.10, yaw=0.0, speed=0.06, orient="travel"):
    """Grasp the rope tip, bring it in front of the first ring (lead_m against its normal, sign chosen toward the tip),
    then pull straight through each ring centre in order and exit exit_m past the last one; release there. orient
    "travel": while dragging, the gripper points along the direction of travel (fingers first through the ring);
    "top_down": it stays top-down."""
    t = np.asarray(tip, float)
    pts = []
    for k, r in enumerate(rings):
        c, n = np.asarray(r["center"], float), np.asarray(r["normal"], float)
        n = n / np.linalg.norm(n)
        prev = t if k == 0 else np.asarray(rings[k - 1]["center"], float)
        if (c - prev) @ n < 0:
            n = -n                           # pass the ring in the direction of travel
        if k == 0:
            pts.append(c - lead_m * n)
        pts.append(c)
        if k == len(rings) - 1:
            pts.append(c + exit_m * n)
    seq = grasp_at(arm, t, yaw)
    seq += [kp(arm, t + [0, 0, 0.05], "keep", yaw, speed)]
    last = t + [0, 0, 0.05]
    for p in pts:
        k = kp(arm, p, "keep", yaw, speed)
        if orient == "travel" and np.linalg.norm(p - last) > 1e-6:
            k["approach"] = list((p - last) / np.linalg.norm(p - last))
        seq.append(k)
        last = p
    seq += release_at(arm, pts[-1], yaw)
    return seq


def pull_both_ends(end_a, end_b, *, pull_m=0.15, lift_m=0.03, yaw_a=0.0, yaw_b=0.0, arms=("left", "right")):
    """Both arms grasp the rope's ends (arm 0 the end nearer to it in x), lift, pull apart along the end-to-end line,
    lay down, release."""
    a, b = np.asarray(end_a, float), np.asarray(end_b, float)
    if a[0] > b[0]:
        a, b = b, a                          # the left arm takes the end with the smaller x
    u = (b - a); u[2] = 0; u /= max(np.linalg.norm(u), 1e-9)
    la, lb = arms
    seq = grasp_at(la, a, yaw_a) + grasp_at(lb, b, yaw_b)
    seq += [kp(la, a + [0, 0, lift_m], "keep", yaw_a, 0.05,
               **{"with": [dict(arm=lb, pos=list(b + [0, 0, lift_m]), yaw=yaw_b, grip="keep")]})]
    seq += [kp(la, a - pull_m * u + [0, 0, lift_m], "keep", yaw_a, 0.05,
               **{"with": [dict(arm=lb, pos=list(b + pull_m * u + [0, 0, lift_m]), yaw=yaw_b, grip="keep")]})]
    seq += [kp(la, a - pull_m * u, "keep", yaw_a, 0.03,
               **{"with": [dict(arm=lb, pos=list(b + pull_m * u), yaw=yaw_b, grip="keep")]})]
    seq += release_at(la, a - pull_m * u, yaw_a) + release_at(lb, b + pull_m * u, yaw_b)
    return seq


def push_sweep(arm, start, end, *, height=0.012, yaw=0.0, passes=1, spacing=0.06, speed=0.06):
    """A closed gripper lowered to `height` above the start point and swept straight to the end (passes shifted
    sideways by `spacing`): pushing instead of unfolding."""
    s, e = np.asarray(start, float), np.asarray(end, float)
    d = e - s; d[2] = 0
    side = np.array([-d[1], d[0], 0.0]); side /= max(np.linalg.norm(side), 1e-9)
    seq = []
    for k in range(passes):
        off = (k - (passes - 1) / 2) * spacing * side
        a, b = s + off, e + off
        seq += [kp(arm, a + [0, 0, HOVER_M], "close", yaw, speed), kp(arm, a + [0, 0, height], "keep", yaw, 0.04),
                kp(arm, b + [0, 0, height], "keep", yaw, speed), kp(arm, b + [0, 0, HOVER_M], "keep", yaw, 0.05)]
    return seq


def two_corner_lift_lay(corner_a, corner_b, *, lift_m=0.25, draw_m=0.15, lay_dir=None, yaw_a=0.0, yaw_b=0.0,
                        arms=("left", "right")):
    """Both arms grasp two cloth corners, lift them, draw back (away from the cloth centre, or along lay_dir) and lay
    the cloth down: the classic flatten-by-lifting heuristic."""
    a, b = np.asarray(corner_a, float), np.asarray(corner_b, float)
    if a[0] > b[0]:
        a, b = b, a
    if lay_dir is None:
        m = 0.5 * (a + b)
        lay = np.array([-(b - a)[1], (b - a)[0], 0.0])
        lay /= max(np.linalg.norm(lay), 1e-9)
    else:
        lay = np.asarray(lay_dir, float); lay /= np.linalg.norm(lay)
    la, lb = arms
    up = np.array([0, 0, lift_m])

    def both(pa, pb, speed, grip="keep"):
        return kp(la, pa, grip, yaw_a, speed, **{"with": [dict(arm=lb, pos=list(pb), yaw=yaw_b, grip=grip)]})
    seq = grasp_at(la, a, yaw_a) + grasp_at(lb, b, yaw_b)
    seq += [both(a + up, b + up, 0.08), both(a + up + draw_m * lay, b + up + draw_m * lay, 0.08),
            both(a + draw_m * lay * 1.6 + [0, 0, 0.02], b + draw_m * lay * 1.6 + [0, 0, 0.02], 0.06)]
    seq += [both(a + draw_m * lay * 1.6 + [0, 0, 0.02], b + draw_m * lay * 1.6 + [0, 0, 0.02], 0.04, "open")]
    seq += [both(a + draw_m * lay * 1.6 + [0, 0, 0.1], b + draw_m * lay * 1.6 + [0, 0, 0.1], 0.06)]
    return seq


def corner_to_rest(points_now, points_rest, *, tilt=np.pi / 4, lift_m=0.06, min_move_m=0.03, arms=None,
                   x_split=-0.136, speed=0.07):
    """Unfold by putting each displaced cloth point (e.g. corner / flap tip) back at its flat-rest position, one at a
    time, with ONE arm per move (the arm on that point's side of x_split unless `arms` names one), the gripper tilted
    by `tilt` (a 45 deg tilted ALOHA gripper reaches ~0.25 m past the centre line: review R1). Points that moved less
    than min_move_m are skipped."""
    seq = []
    for k, (a, b) in enumerate(zip(points_now, points_rest)):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if np.linalg.norm(a[:2] - b[:2]) < min_move_m:
            continue
        arm = (arms[k] if isinstance(arms, (list, tuple)) else arms) or ("left" if a[0] < x_split else "right")
        b = np.array([b[0], b[1], a[2]])          # the rest xy at the grasp height (a cloth's rest may be authored above the table)
        pa, pb = a + [0, 0, 0.002], b + [0, 0, 0.002]
        seq += [kp(arm, pa + [0, 0, HOVER_M], "open", 0.0, speed, tilt=tilt), kp(arm, pa, "open", 0.0, 0.04, tilt=tilt),
                kp(arm, pa, "close", 0.0, 0.04, 0.4, tilt=tilt), kp(arm, pa + [0, 0, lift_m], "keep", 0.0, 0.05, tilt=tilt),
                kp(arm, pb + [0, 0, lift_m], "keep", 0.0, speed, tilt=tilt), kp(arm, pb, "keep", 0.0, 0.04, tilt=tilt),
                kp(arm, pb, "open", 0.0, 0.04, 0.4, tilt=tilt), kp(arm, pb + [0, 0, HOVER_M], "open", 0.0, 0.06, tilt=tilt)]
    return seq


BREAKERS = dict(straight_line=straight_line, corner_to_rest=corner_to_rest, shortest_route=shortest_route, drag_through_rings=drag_through_rings,
                pull_both_ends=pull_both_ends, push_sweep=push_sweep, two_corner_lift_lay=two_corner_lift_lay)


# ------------------------------------------------------------------------------------------------ program -> rows
def program_rows(keypoints, fk: AlohaFK, layout, dt: float, *, start_row=None, close_s=0.4,
                 closed_native=PINCH_NATIVE) -> tuple:
    """Key points -> (rows (T, 14) float32, info). Each key point moves its arm (and its `with` arms, simultaneously)
    in a straight TCP line from the arm's last pose, at the key point's speed (clamped to the ALOHA joint speed limit by
    slowing down), then sets its gripper (over close_s) and dwells. Orientation: top-down at the key point's yaw."""
    from scipy.spatial.transform import Rotation, Slerp
    cols = layout_columns(layout)
    row = np.zeros(len(layout), np.float64) if start_row is None else np.asarray(start_row, float).copy()
    if start_row is None:
        for arm in ARMS:
            row[cols[arm][0]] = HOME
            row[cols[arm][1]] = GRIPPER_OPEN_NATIVE if cols[arm][2] == "native" else 0.07
    state = {}
    for arm in ARMS:
        p, R = fk.tcp({arm: row[cols[arm][0]]})[arm]
        state[arm] = dict(p=p, R=R, q=row[cols[arm][0]].copy(), g=row[cols[arm][1]])
    rows, errs, unreach = [], [], 0

    def gval(arm, grip):
        sp = cols[arm][2]
        if grip == "close":
            return closed_native if sp == "native" else 0.0
        if grip == "open":
            return GRIPPER_OPEN_NATIVE if sp == "native" else 2 * (GRIPPER_OPEN_NATIVE - GRIPPER_CLOSED_NATIVE)
        return None

    for k in keypoints:
        movers = [dict(arm=k["arm"], pos=k["pos"], yaw=k.get("yaw", 0.0), tilt=k.get("tilt", 0.0),
                       grip=k.get("grip", "keep"), approach=k.get("approach"))] + list(k.get("with", []) or [])
        targets = {}
        for m in movers:
            R = (approach_R(m["approach"], m["arm"]) if m.get("approach") is not None
                 else top_down_R(m.get("yaw", 0.0), m["arm"], m.get("tilt", 0.0)))
            targets[m["arm"]] = (np.asarray(m["pos"], float), R)
        dist = max(np.linalg.norm(targets[a][0] - state[a]["p"]) for a in targets)
        n = max(1, int(np.ceil(dist / max(1e-6, k.get("speed", 0.08)) / dt)))
        # slerp rotations
        slerps = {a: Slerp([0, 1], Rotation.from_matrix(np.stack([state[a]["R"], targets[a][1]]))) for a in targets}
        seg = []
        for i in range(1, n + 1):
            f = i / n
            r = row.copy()
            for a in targets:
                p = state[a]["p"] + f * (targets[a][0] - state[a]["p"])
                Rg = slerps[a]([f]).as_matrix()[0]
                # continuity first (seed and weak prior = the arm's last joints); HOME as a fallback seed
                best = None
                for seed in (state[a]["q"], np.asarray(HOME, float)):
                    q, e, er = fk.ik(a, p, Rg, seed=seed, iters=300, prior=state[a]["q"])
                    score = e + 0.05 * er + 0.02 * float(np.abs(q - state[a]["q"]).max())
                    if best is None or score < best[0]:
                        best = (score, q, e, er)
                    if e < 0.002 and er < 0.02 and float(np.abs(q - state[a]["q"]).max()) < 0.3:
                        break
                _, q, e, er = best
                errs.append(e)
                unreach += int(e > 0.005)
                r[cols[a][0]] = q
                state[a]["q"] = q
            seg.append(r); row = r
        # joint-speed check: slow down (repeat interpolation) where a step exceeds the limit
        for r in seg:
            rows.append(r.copy())
        for a in targets:
            state[a]["p"], state[a]["R"] = targets[a]
        # gripper actions
        gtarget = {m["arm"]: gval(m["arm"], m.get("grip", "keep")) for m in movers}
        nclose = max(1, int(round(close_s / dt))) if any(v is not None for v in gtarget.values()) else 0
        g0 = {a: row[cols[a][1]] for a in gtarget}
        for i in range(1, nclose + 1):
            r = row.copy()
            for a, v in gtarget.items():
                if v is not None:
                    r[cols[a][1]] = g0[a] + (v - g0[a]) * i / nclose
            rows.append(r); row = r
        for _ in range(int(round(k.get("dwell_s", 0.2) / dt))):
            rows.append(row.copy())
    R_ = np.asarray(rows, np.float64)
    dq = np.abs(np.diff(np.stack([R_[:, cols[a][0]] for a in ARMS], axis=1), axis=0)) if len(R_) > 1 else np.zeros((0, 2, 6))
    cap = JOINT_SPEED_LIMIT_RAD_S * dt
    info = dict(rows=int(len(R_)), duration_s=round(len(R_) * dt, 3), ik_err_max_m=float(max(errs) if errs else 0),
                unreachable_rows=int(unreach), joint_step_max_rad=float(dq.max()) if len(dq) else 0.0,
                rows_over_speed_limit=int((dq > cap + 1e-9).any(axis=(1, 2)).sum()) if len(dq) else 0)
    return R_.astype(np.float32), info


# ------------------------------------------------------------------------------------------------ auto specs
def scene_keypoints(model, data, manifest, scene_dir, family) -> dict:
    """Key geometry of a scene at its birth: rope ends, rings, cloth corners (flat-rest grid corners at birth)."""
    import mujoco
    out = {}
    for name, rec in manifest["entities"].items():
        if rec.get("kind") == "rope_chain":
            bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, l["body"]) for l in rec["links"]]
            P = np.asarray(data.xpos)[bodies]
            out.setdefault("ropes", {})[name] = dict(ends=[P[0].tolist(), P[-1].tolist()], n=len(P))
        if rec.get("kind") == "pbd_cloth":
            from .scene_geom import cloth_mesh
            cm = cloth_mesh(manifest, scene_dir, name)
            rest, birth = cm["rest"], cm["birth"]
            c = rest[:, :2].mean(axis=0)
            corners, corners_rest, idx = [], [], []
            for sx in (-1, 1):
                for sy in (-1, 1):
                    k = int(np.argmax(sx * (rest[:, 0] - c[0]) + sy * (rest[:, 1] - c[1])))
                    corners.append(birth[k].tolist()); corners_rest.append(rest[k].tolist()); idx.append(k)
            # edge midpoints too (flap tips of half folds)
            mids, mids_rest = [], []
            for ax, sgn in ((0, -1), (0, 1), (1, -1), (1, 1)):
                score = sgn * (rest[:, ax] - c[ax]) - 0.3 * np.abs(rest[:, 1 - ax] - c[1 - ax])
                k = int(np.argmax(score))
                mids.append(birth[k].tolist()); mids_rest.append(rest[k].tolist())
            out.setdefault("cloths", {})[name] = dict(corners=corners, corners_rest=corners_rest, edge_mids=mids,
                                                      edge_mids_rest=mids_rest,
                                                      bbox=[birth.min(0).tolist(), birth.max(0).tolist()],
                                                      center=birth.mean(0).tolist())
    if family == "rope_threading":
        from .scene_geom import rings_from_model
        out["rings"] = rings_from_model(model, data, manifest)
    return out


def auto_specs(geom: dict, family: str) -> list:
    specs = []
    if family == "rope_threading" and geom.get("ropes") and geom.get("rings"):
        rope = next(iter(geom["ropes"].values()))
        rings = geom["rings"]
        for e, tip in enumerate(rope["ends"]):
            for order_name, order in (("given", rings), ("reversed", rings[::-1])):
                arm = "right" if tip[0] > -0.136 else "left"
                specs.append(dict(name=f"drag_end{e}_{order_name}_{arm}", breaker="drag_through_rings",
                                  args=dict(arm=arm, tip=[tip[0], tip[1], tip[2] + 0.0], rings=order)))
    if family == "knot_untying" and geom.get("ropes"):
        for name, rope in geom["ropes"].items():
            a, b = rope["ends"]
            specs.append(dict(name=f"pull_both_ends_{name}", breaker="pull_both_ends", args=dict(end_a=a, end_b=b)))
            specs.append(dict(name=f"pull_both_ends_{name}_far", breaker="pull_both_ends",
                              args=dict(end_a=a, end_b=b, pull_m=0.25, lift_m=0.06)))
    if family == "blanket_flatten" and geom.get("cloths"):
        cl = next(iter(geom["cloths"].values()))
        c = cl["corners"]           # (-,-), (-,+), (+,-), (+,+) in the rest frame
        specs.append(dict(name="lift_lay_near_pair", breaker="two_corner_lift_lay", args=dict(corner_a=c[0], corner_b=c[2])))
        specs.append(dict(name="lift_lay_far_pair", breaker="two_corner_lift_lay", args=dict(corner_a=c[1], corner_b=c[3])))
        lo, hi = np.asarray(cl["bbox"][0]), np.asarray(cl["bbox"][1])
        mid = 0.5 * (lo + hi)
        specs.append(dict(name="corner_to_rest_tilted", breaker="corner_to_rest",
                          args=dict(points_now=c + cl["edge_mids"], points_rest=cl["corners_rest"] + cl["edge_mids_rest"])))
        specs.append(dict(name="push_sweep_right", breaker="push_sweep",
                          args=dict(arm="right", start=[mid[0], lo[1], lo[2]], end=[mid[0], hi[1], lo[2]], passes=3)))
    return specs


def build(spec: dict) -> list:
    if "keypoints" in spec:
        return straight_line(spec["keypoints"])
    return BREAKERS[spec["breaker"]](**spec.get("args", {}))


# ------------------------------------------------------------------------------------------------ CLI
def _load_specs(a, geom):
    if a.auto:
        return auto_specs(geom, a.family)
    s = json.loads(Path(a.spec).read_text())
    return s if isinstance(s, list) else [s]


def cmd_kinematic(a):
    import mujoco
    from .aloha import default_layout, load_model, scene_dir_of_instance
    from .harness import sweep_robot
    from previews.render import set_birth
    sd = scene_dir_of_instance(a.instance)
    model, manifest = load_model(sd)
    data = mujoco.MjData(model)
    set_birth(model, data, manifest, sd)
    geom = scene_keypoints(model, data, manifest, sd, a.family)
    fk = AlohaFK(model)
    layout = default_layout("native")
    dt = CONTROL_DT_S.get(a.family, 0.04)
    robot = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, l["body"]) for l in manifest["entities"]["robot"]["links"]}
    obst = set()
    for name, rec in manifest["entities"].items():
        if name in ("robot", "table", "ground") or name.endswith("_base_riser") or rec.get("kind") in ("rope_chain", "pbd_cloth") \
                or rec.get("free_joint"):
            continue
        obst |= {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, l["body"]) for l in rec.get("links", [])}
    results = []
    for spec in _load_specs(a, geom):
        keys = build(spec)
        rows, info = program_rows(keys, fk, layout, dt)
        sweep = sweep_robot(model, rows.astype(np.float64), layout, obst, robot, every=2)
        results.append(dict(name=spec.get("name"), breaker=spec.get("breaker", "straight_line"), method="kinematic",
                            program=info, sweep=sweep,
                            feasible=bool(info["unreachable_rows"] == 0 and sweep["rows_in_collision"] == 0),
                            note="kinematic only: reach + joint speed + robot-vs-static-obstacle sweep; whether the "
                                 "object follows needs the Session run (breakers session)"))
        np.savez_compressed(Path(a.out).with_suffix("").as_posix() + f"_{spec.get('name')}.npz", actions=rows,
                            action_layout=np.array(layout))
    out = dict(instance=a.instance, family=a.family, geometry=geom, breakers=results)
    Path(a.out).write_text(json.dumps(out, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    for r in results:
        print(r["name"], "feasible", r["feasible"], "unreach", r["program"]["unreachable_rows"],
              "collide_rows", r["sweep"]["rows_in_collision"], r["sweep"]["obstacles_hit"])


def cmd_session(a):
    from . import record
    from .harness import closest_approach, make_session, run_summary
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    s = make_session(a)
    w = s.env._world
    from .aloha import layout_columns as _lc  # noqa: F401
    geom = scene_keypoints(w.model, w.data, w.manifest, w.scene_dir, a.family)
    if a.family == "rope_threading":
        from .scene_geom import task_rings
        tr = task_rings(s)
        if tr:
            geom["rings"] = tr                 # the task's own rings (any names), in the required order
    fk = AlohaFK(w.model)
    layout = list(s.action_layout)
    dt = 1.0 / s.control_hz
    rings = geom.get("rings")
    res = []
    for spec in _load_specs(a, geom):
        keys = build(spec)
        s.reset()
        rows, info = program_rows(keys, fk, layout, dt, start_row=s.hold_action())
        name = spec.get("name", "breaker")
        js = record.run(s, rows, out / f"{name}.npz", every=a.every, hold=a.hold, label=name, extra=dict(breaker=spec))
        res.append(dict(name=name, breaker=spec.get("breaker", "straight_line"), method="mujoco_session",
                        success=bool(js["success"]), program=info, run=run_summary(js),
                        closest_approach=closest_approach(out / f"{name}.npz", a.family, rings)))
        (out / "breakers.json").write_text(json.dumps(dict(family=a.family, breakers=res), indent=1, default=str))
    print(json.dumps([(r["name"], r["success"]) for r in res]))
    s.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="heuristic breakers")
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("kinematic")
    k.add_argument("--instance", required=True); k.add_argument("--family", required=True)
    k.add_argument("--spec", default=None); k.add_argument("--auto", action="store_true"); k.add_argument("--out", required=True)
    s = sub.add_parser("session")
    s.add_argument("--new-instance", dest="new_instance", default=None); s.add_argument("--instance", dest="new_instance")
    s.add_argument("--builder", default=None); s.add_argument("--builder-args", default=None)
    s.add_argument("--no-strict", action="store_true"); s.add_argument("--family", required=True)
    s.add_argument("--spec", default=None); s.add_argument("--auto", action="store_true")
    s.add_argument("--every", type=int, default=10); s.add_argument("--hold", type=int, default=50)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"kinematic": cmd_kinematic, "session": cmd_session}[a.cmd](a)


if __name__ == "__main__":
    main()
