"""Scene generators of the redesigned interlocked_claws tiers (inverse generation).

Every tier is generated the same way (docs/simulator/redesign/interlocked_claws.md):
  1. the key B (brown, the target) is a compound of boxes with grip blocks;
  2. the separated goal is B free of A; the designed insertion is a list of exact moves from that free pose
     (paths.py); the start (engaged) pose is where the insertion ends, and the disengagement is its exact reverse;
  3. the lock A (teal) is generated around that motion: keyhole gates carved out of B's swept volume (carve.py),
     joined by struts outside it, with A's own grips;
  4. checks (topogym/planning/interlocked_claws/redesign): the designed path is free with the clearance, the
     separation predicate holds at its end, and the breakers (straight pulls, skipped / reordered moves, the old
     oracle) are blocked.

A design is a plain dict: pieces (boxes, grips), the engaged pose, the moves, the placement of A in the assembly frame
(the released tiers' robot-mount source frame: x from the left arm to the right arm, table top at z = 0.02), the hands
holding what at birth, and per-move labels for the diagrams.
"""
from __future__ import annotations

import numpy as np

from . import carve, paths
from topogym.tasks.interlocked_claws.redesign.shapes import Piece, COLORS, bar, box, polyline, pose, rot, tcp_frame

T_BAR = .012          # bar section of keys and struts
PLATE = .008          # gate plate thickness


# ------------------------------------------------------------------------------------------------- keys (piece B)
def claw_key(shaft=.24, cross=.070, prong=.026, grip="top", head_grip=False, prong_dir="x"):
    """The claw key in its own frame: a crossbar along y at the origin (the claw), a prong from its +y end pointing
    -x (an asymmetric claw: one keyhole orientation), a shaft along +x, a grip block at its end ("top": the hand
    comes from above, pads across the shaft; "axial": the hand comes along the shaft). head_grip: a second grip on a
    stub ahead of the claw, approached along -y of the key (from above once the key is rolled +90 deg about its
    shaft), pads closing along z."""
    t = T_BAR
    tip = (-prong, cross / 2 - t / 2, 0.) if prong_dir == "x" else (0., cross / 2 - t / 2, prong)
    body = [box((0., 0., 0.), (t, cross, t)),
            bar((0., cross / 2 - t / 2, 0.), tip, t),
            bar((0., 0., 0.), (shaft, 0., 0.), t)]
    g = shaft + .015
    if grip == "axial":
        grips = {"b_grip": tcp_frame((g, 0., 0.), (-1., 0., 0.), (0., 1., 0.))}
    elif str(grip).startswith("side:"):        # across the shaft, the approach turned by <deg> about the shaft
        th = np.deg2rad(float(grip[5:]))
        a = (0., np.sin(th), -np.cos(th))
        c = (0., np.cos(th), np.sin(th))
        grips = {"b_grip": tcp_frame((g - .003, 0., 0.), a, c, up_hint=(1., 0., 0.))}
    else:
        grips = {"b_grip": tcp_frame((g - .003, 0., 0.), (0., 0., -1.), (0., 1., 0.), up_hint=(1., 0., 0.))}
    if head_grip:
        body.append(bar((0., 0., 0.), (-.040, 0., 0.), t))
        grips["b_head"] = tcp_frame((-.045, 0., 0.), (0., -1., 0.), (0., 0., 1.), up_hint=(-1., 0., 0.))
    return Piece("piece_0", body, grips, COLORS[0])


# ------------------------------------------------------------------------------------------------- the lock builder
def plate_corners(info, inset=.006):
    c, n, w, u = carve.plate_frame(info["center"], info["normal"], info["up"])
    hw, hh = info["width"] / 2 - inset, info["height"] / 2 - inset
    return {k: c + sw * hw * w + su * hh * u for k, (sw, su) in
            dict(lo_a=(-1, -1), lo_b=(1, -1), hi_a=(-1, 1), hi_b=(1, 1)).items()}


def join_plates(info_a, info_b):
    """Four struts joining two plates' corners (each corner of a to the nearest corner of b on the same height)."""
    ca, cb = plate_corners(info_a), plate_corners(info_b)
    out = []
    for k in ca:
        same = [j for j in cb if j[:2] == k[:2]]
        j = min(same, key=lambda j: np.linalg.norm(cb[j] - ca[k]))
        out.append(bar(ca[k], cb[j], T_BAR))
    return out


def leg_bars(infos, z_floor, names=None):
    """Legs from every gate plate's two lower corners straight down to the floor (A's frame), each with a foot pad:
    the lock stands on the table at its designed pose (the hard tier's table start)."""
    out = []
    for name, info in infos.items():
        if names and name not in names:
            continue
        cs = plate_corners(info)
        for k in ("lo_a", "lo_b"):
            p = cs[k]
            if p[2] - z_floor < .02:
                continue
            foot = np.array([p[0], p[1], z_floor + .004])
            out.append(bar(p, foot + [0., 0., .004], T_BAR))
            out.append(box(tuple(foot), (.03, .03, .008)))
    return out


def in_A(T_A_asm, p_asm, approach_asm, closing_asm, up_hint=(0., 0., 1.)):
    """A grip frame given in the assembly frame (where the arm is) -> A's frame."""
    T = tcp_frame(p_asm, approach_asm, closing_asm, up_hint=up_hint)
    return np.linalg.inv(T_A_asm) @ T


SHOULDER = {"left": np.array([-.469, -.019, .099]), "right": np.array([.469, -.019, .099])}


def natural_grip_in_A(T_A_asm, p_A, arm, tilt_deg=15.):
    """A grip frame on A at p_A (A's frame) whose approach, in the assembly, points from the arm's shoulder toward the
    grip (horizontal part) and tilts down: the arm reaches it without twisting its wrist."""
    p = (np.asarray(T_A_asm, float) @ np.r_[p_A, 1.])[:3]
    d = p[:2] - SHOULDER[arm][:2]
    yaw = np.arctan2(d[1], d[0])
    t = np.deg2rad(tilt_deg)
    a = np.array([np.cos(yaw) * np.cos(t), np.sin(yaw) * np.cos(t), -np.sin(t)])
    return in_A(T_A_asm, p, a, (-np.sin(yaw), np.cos(yaw), 0.))


def handle_arm(T_A_asm, start_A, arm, length=.10, tilt_deg=10.):
    """A's grip on a handle arm sticking out of A toward the holding arm: the arm's direction (in the assembly) points
    from A toward that arm's shoulder (horizontal), the grip block at its end; the hand approaches back along it,
    tilted down, so the gripper stays outside the lock. -> (grip TCP frame in A, the bar's start in A)."""
    T = np.asarray(T_A_asm, float)
    p0 = (T @ np.r_[start_A, 1.])[:3]
    d = SHOULDER[arm][:2] - p0[:2]
    d = d / np.linalg.norm(d)
    t = np.deg2rad(tilt_deg)
    a = np.array([-d[0] * np.cos(t), -d[1] * np.cos(t), -np.sin(t)])           # approach: toward A, tilted down
    tcp = p0 - (length + .015) * a
    G = in_A(T, tcp, a, (-a[1], a[0], 0.))
    return G, np.asarray(start_A, float)


def hand_box(T_tcp):
    """A box around the ALOHA gripper holding a grip at T_tcp: 0.15 m back along the approach, 0.11 m across the pads,
    0.08 m high (the gripper body, fingers and camera mount, generously)."""
    T = np.asarray(T_tcp, float).copy()
    T[:3, 3] = T[:3, 3] - .06 * T[:3, 0]
    return dict(size=[.17, .11, .08], pose=T.tolist())


def box_points(b, n=5):
    """A grid of n^3 points filling the box b (its surface and inside)."""
    T = np.asarray(b["pose"], float)
    h = np.asarray(b["size"], float) / 2
    g = np.linspace(-1., 1., n)
    loc = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3) * h
    return loc @ T[:3, :3].T + T[:3, 3]


def boxes_apart(b1, b2, margin):
    """Conservative separation test of two boxes by points: every grid point of each box at least `margin` from the
    other box."""
    return strut_clear(b1, box_points(b2, 7), margin) and strut_clear(b2, box_points(b1, 7), margin)


def auto_handle(T_asm, infos, arm, swept, clearance, length, taken=(), hands=(), other_hands=()):
    """The handle arm for `arm` from the first candidate root (plate corners and edge midpoints of every gate, nearest
    the arm's shoulder first) whose bar, grip block and the gripper around it clear the key's swept volume."""
    from topogym.tasks.interlocked_claws.redesign.shapes import grip as grip_box
    cands = []
    for info in infos.values():
        c, n, w, u = carve.plate_frame(info["center"], info["normal"], info["up"])
        hw, hh = info["width"] / 2 - .006, info["height"] / 2 - .006
        for sw, su in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)):
            cands.append(c + sw * hw * w + su * hh * u)
    sh = np.linalg.inv(np.asarray(T_asm, float)) @ np.r_[SHOULDER[arm], 1.]
    # comfort first: the grip (length + 1.5 cm in front of its root) about 0.33 m from the shoulder
    cands.sort(key=lambda p: abs(np.linalg.norm(p - sh[:3]) - length - .015 - .33))
    for root in cands:
        if any(np.linalg.norm(root - t) < .08 for t in taken):
            continue
        G, start = handle_arm(T_asm, root, arm, length=length)
        b = bar(start, G[:3, 3] + .016 * G[:3, 0], T_BAR)
        hb = hand_box(G)
        if (strut_clear(b, swept, clearance + .002) and strut_clear(grip_box(G), swept, clearance + .002)
                and strut_clear(hand_box(G), swept, .015)
                and all(strut_clear(b, hp, .012) and strut_clear(grip_box(G), hp, .012) for hp in hands)
                and all(strut_clear(hb, hp, .01) for hp in other_hands)):
            return G, start
    raise ValueError(f"no clear handle root for the {arm} arm")


def strut_clear(b, swept, margin):
    """True when the box b stays at least `margin` from every swept point (point-to-box distance)."""
    T = np.asarray(b["pose"], float)
    h = np.asarray(b["size"], float) / 2
    local = (swept - T[:3, 3]) @ T[:3, :3]
    d = np.linalg.norm(np.maximum(np.abs(local) - h, 0.), axis=1)
    return bool(d.min() >= margin)


def build_lock(B, T_free, insertion, gates, clearance, links, a_grips, **meta):
    """Carve the gates from B's swept volume along the designed path; A = the gates' plates + the given links (strut
    boxes, or callables of the gate infos) + its grips (name -> (TCP frame in A, attach point in A))."""
    engaged, moves = paths.insertion_to_disengagement(T_free, insertion)
    poses, starts, which = paths.dense(engaged, moves)
    pts = carve.key_points(B.boxes)
    plates, infos = [], {}
    for g in gates:
        boxes, info, _ = carve.carve_gate(pts, poses, g["center"], g["normal"], g.get("up", (0., 0., 1.)),
                                          g["size"][0], g["size"][1], thickness=g.get("thickness", PLATE),
                                          clearance=clearance, cell=.0015, spacing=.001,
                                          open_sides=g.get("open_sides", ()))
        info["name"] = g["name"]
        plates += boxes
        infos[g["name"]] = info
    body = list(plates)
    swept = np.concatenate([pts[::7] @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3] for T in poses[::2]])
    rejected = []
    for item in links:
        for b in (item(infos) if callable(item) else [item]):
            if strut_clear(b, swept, clearance + .002):
                body.append(b)
            else:
                rejected.append(b)
    meta["rejected_struts"] = len(rejected)
    grips = {}
    T_asm = meta.get("A_in_assembly")
    # the hand on B's birth grip, swept along the path until a hand change takes it off B: handles must clear it
    hands0 = meta.get("hands") or {}
    b_hand_pts = []
    for k in range(0, len(poses), 3):
        if any("hold" in m for m in moves[:which[k] + 1]):
            break
        for g in hands0.values():
            if g in B.grips:
                b_hand_pts.append(box_points(hand_box(np.asarray(poses[k]) @ np.asarray(B.grips[g], float)), 4))
    b_hand_pts = [np.concatenate(b_hand_pts)] if b_hand_pts else []
    for name, spec in a_grips.items():
        if isinstance(spec, dict):              # {"arm", "length"}: the first plate corner / edge whose handle clears
            T, attach = auto_handle(T_asm, {k: v for k, v in infos.items() if k not in spec.get("exclude", ())},
                                    spec["arm"], swept, clearance, spec.get("length", .09),
                                    taken=[g[:3, 3] for g in grips.values()],
                                    hands=b_hand_pts + [box_points(hand_box(g), 5) for g in grips.values()],
                                    other_hands=[box_points(hand_box(g), 5) for g in grips.values()])
        else:
            T, attach = spec
        T = np.asarray(T, float)
        approach = T[:3, 0]
        body.append(bar(np.asarray(attach, float), T[:3, 3] + .016 * approach, T_BAR))
        grips[name] = T
    A = Piece("piece_1", body, grips, COLORS[1])
    return dict(A=A, B=B, engaged=engaged, moves=moves, insertion=insertion, T_free=T_free,
                gates=list(infos.values()), clearance=clearance, **meta)


# ------------------------------------------------------------------------------------------------- very_easy
def very_easy(clearance=.004, grip="axial", T_A=None):
    """Swivel key: B's claw sits behind gate G2 (a side pocket); out = pull back along the shaft, swing 50 deg about
    z (the shaft pivots in G1), roll the claw 90 deg about the shaft, pull out of G1. Both pieces start in hand."""
    B = claw_key(grip=grip)
    yaw = -50.
    d_head = np.array([-np.cos(np.deg2rad(yaw)), -np.sin(np.deg2rad(yaw)), 0.])
    T_free = pose((.08, 0., 0.), rot("x", 90))
    insertion = [
        dict(kind="slide", vector=[-.155, 0., 0.], label="through G1", why="the claw passes G1 only rolled upright",
             label_out="4. pull out of G1 (+x)", why_out="the claw passes G1 only upright"),
        dict(kind="turn_body", axis_body=[1., 0., 0.], deg=-90., label="roll 90 (x)",
             why="the claw must lie flat to pass G2", label_out="3. roll the claw 90 (shaft axis x)",
             why_out="G1 lets the claw through only upright"),
        dict(kind="turn", axis=[0., 0., 1.], deg=yaw, pivot=[0., 0., 0.], label="swing 50 (z)",
             why="the shaft pivots in G1 to point the claw at G2", label_out="2. swing 50 about z (pivot G1)",
             why_out="the shaft pivots in G1 until it lines up with G1's slot"),
        dict(kind="slide", vector=(.065 * d_head).tolist(), label="through G2", why="the claw goes behind G2",
             label_out="1. pull back out of G2", why_out="along the shaft: the claw leaves the side pocket"),
    ]
    T_A = pose((-.12, -.06, .20), rot("z", 15)) if T_A is None else T_A   # placement search 22:00 (axial grip)
    gates = [dict(name="G1", center=(0., 0., 0.), normal=(1., 0., 0.), size=(.13, .13)),
             dict(name="G2", center=(.11 * d_head).tolist(), normal=(-d_head).tolist(), size=(.12, .12)),
             dict(name="W0", center=(-.135, -.01, 0.), normal=(1., 0., 0.), size=(.12, .12))]
    a_grips = {"a_grip": dict(arm="left", length=.09)}
    links = [lambda i: join_plates(i["G1"], i["G2"]), lambda i: join_plates(i["G1"], i["W0"])]
    return build_lock(B, T_free, insertion, gates, clearance, links, a_grips,
                      tier="very_easy", name="swivel_key", A_in_assembly=T_A,
                      hands=dict(left="a_grip", right="b_grip"),
                      summary="pull back along the shaft, swing 50 deg about z (shaft pivots in G1), roll the claw "
                              "90 deg about the shaft, pull out of G1")


# ------------------------------------------------------------------------------------------------- the through-key
HEAD_X = [-.045]          # the head grip's TCP along the key (B's x); set per tier by through_key(head_x=...)


def head_grip_frame(roll, pitch):
    """The head grip's TCP frame in B's frame such that, once the key is pitched and rolled for the exit gate, the
    hand comes from straight above; pads closing across the stub."""
    from scipy.spatial.transform import Rotation as R
    a = (R.from_euler("x", -roll, degrees=True) * R.from_euler("y", -pitch, degrees=True)).apply([0., 0., -1.])
    c = np.cross(a, [1., 0., 0.])
    c = c / np.linalg.norm(c) if np.linalg.norm(c) > 1e-6 else np.array([0., 1., 0.])
    up = (R.from_euler("x", -roll, degrees=True) * R.from_euler("y", -pitch, degrees=True)).apply([-1., 0., 0.])
    return tcp_frame((HEAD_X[0], 0., 0.), a, c, up_hint=up)


def head_grip_toward(T_A, R_free, p_B, spec):
    """The head grip's TCP frame in B's frame such that, at the hand change (B at p_B in A's frame, rotation R_free),
    the hand approaches the stub from the given arm's shoulder, tilted down (spec "arm:left:45"); the approach is made
    perpendicular to the stub, the pads close across it."""
    parts = spec.split(":")
    arm, tilt = parts[1], float(parts[2]) if len(parts) > 2 else 45.
    h_A = p_B + R_free @ np.array([HEAD_X[0], 0., 0.])
    h_W = (T_A @ np.r_[h_A, 1.])[:3]
    d = h_W[:2] - SHOULDER[arm][:2]
    d = d / np.linalg.norm(d)
    t = np.deg2rad(tilt)
    a_W = np.array([d[0] * np.cos(t), d[1] * np.cos(t), -np.sin(t)])
    a_A = T_A[:3, :3].T @ a_W
    stub = R_free[:, 0]
    a_A = a_A - (a_A @ stub) * stub
    a_A /= np.linalg.norm(a_A)
    a_B = R_free.T @ a_A
    c_B = np.cross(a_B, [1., 0., 0.])
    c_B /= np.linalg.norm(c_B)
    return tcp_frame((HEAD_X[0], 0., 0.), a_B, c_B, up_hint=np.cross(a_B, c_B))


def through_key(tier, name, *, yaw=50., roll=90., pitch=0., x_gx=-.15, clearance=.004, shaft=.22, T_A=None,
                a_handle=((-.075, -.064, -.064), .09), a_handle2=None, hand_change=None, roll_share=None,
                floor=True, summary="", left_exclude=(), free_extra=0., pre_lift=0., b_grip="top", legs=False,
                head="above", head_x=-.045):
    """A key whose claw sits in a side pocket G2 behind the entry gate G1, and which can only leave forward through
    the exit gate GX (pitched by `pitch` about A's y): G1 never lets the claw back. Disengagement: pull back out of G2,
    swing `yaw` about z and tilt `pitch` about y (the shaft pivots in G1), roll the claw `roll` about the shaft, push
    forward through GX until the hand on the handle reaches G1, change hands (`hand_change`: arm -> grip, in order),
    pull the key out by its head grip."""
    from scipy.spatial.transform import Rotation as Rot
    B = claw_key(shaft=shaft, grip=b_grip, head_grip=False, prong_dir="z")
    HEAD_X[0] = head_x
    hg = head_grip_frame(roll, pitch)
    B.body.append(bar((0., 0., 0.), (head_x + .005, 0., 0.), T_BAR))
    B.grips["b_head"] = hg
    e = Rot.from_euler("y", pitch, degrees=True).apply([1., 0., 0.])        # the exit axis (GX -> G1)
    d_head = np.array([-np.cos(np.deg2rad(yaw)), -np.sin(np.deg2rad(yaw)), 0.])
    t_handle = shaft + .012                                                  # the handle's TCP along the key
    s_swap = .055 - t_handle                                                 # the hand stops 5.5 cm before G1
    s_free = x_gx - .043 - shaft - free_extra
    s_cham = -.075
    R_free = (Rot.from_euler("y", pitch, degrees=True) * Rot.from_euler("x", roll, degrees=True)).as_matrix()
    T_free = pose(s_free * e, R_free)
    hc = hand_change or dict(right="b_head")
    insertion = [
        dict(kind="slide", vector=((s_swap - s_free) * e).tolist(), label="pull in",
             label_out=f"pull out by the head grip", hold=hc,
             why_out="hand change first: the handle cannot follow the hand through G1"),
        dict(kind="slide", vector=((s_cham - s_swap) * e).tolist(), label="push in",
             label_out="push forward through GX", why_out="the hand on the handle stops 5.5 cm before G1"),
        dict(kind="turn_body", axis_body=[1., 0., 0.], deg=-roll, label="unroll",
             label_out=f"roll the claw {roll:.0f} (shaft axis)", why_out="GX lets the claw through only in this roll",
             **({"share": roll_share} if roll_share is not None else {})),
    ]
    if pitch:
        insertion.append(dict(kind="turn", axis=[0., 1., 0.], deg=-pitch, pivot=[0., 0., 0.], label="unpitch",
                              label_out=f"tilt {abs(pitch):.0f} about y (pivot G1)",
                              why_out="the shaft lines up with the pitched exit gate GX"))
    insertion += [
        dict(kind="turn", axis=[0., 0., 1.], deg=yaw, pivot=[0., 0., 0.], label="swing",
             label_out=f"swing {abs(yaw):.0f} about z (pivot G1)", why_out="the shaft leaves the pocket's line"),
        dict(kind="slide", vector=(.065 * d_head).tolist(), label="into G2",
             label_out="pull back out of G2", why_out="along the shaft: the claw leaves the side pocket"),
    ]
    if pre_lift:
        insertion.append(dict(kind="slide", vector=[0., 0., -pre_lift], label="drop into the notch",
                              label_out=f"lift {pre_lift * 1000:.0f} mm out of the notch",
                              why_out="the claw sits low behind G2: only lifted can it come back through G2"))
    T_A = pose((.0, -.20, .23), rot("z", -90)) if T_A is None else T_A
    if head != "above":                        # "arm:<side>[:tilt]": the head grip faces that arm's shoulder
        B.grips["b_head"] = head_grip_toward(np.asarray(T_A, float), R_free, s_swap * e, head)
    gx_c = x_gx * e
    gates = [dict(name="G1", center=(0., 0., 0.), normal=(1., 0., 0.), size=(.14, .14)),
             dict(name="G2", center=(.11 * d_head).tolist(), normal=(-d_head).tolist(), size=(.12, .12)),
             dict(name="GX", center=gx_c.tolist(), normal=e.tolist(), size=(.14, .14),
                  up=Rot.from_euler("y", pitch, degrees=True).apply([0., 0., 1.]).tolist())]
    a_grips = {"a_grip": dict(arm="left", length=.09, exclude=tuple(left_exclude))}
    if a_handle2 is not None:
        a_grips["a_grip2"] = dict(arm="right", length=.08)
    links = [lambda i: join_plates(i["G1"], i["G2"]), lambda i: join_plates(i["G1"], i["GX"])]
    if pitch:
        # a back wall under the tilted exit: nothing may leave the chamber horizontally below GX
        gates.append(dict(name="W0", center=(x_gx + .01, 0., -.025 * np.sign(pitch)), normal=(1., 0., 0.),
                          size=(.14, .09)))
        links.append(lambda i: join_plates(i["G1"], i["W0"]))
    if legs:                                   # A stands on the table: legs down to the table top (z = .02)
        z_floor = .02 - float(np.asarray(T_A)[2, 3])
        links.append(lambda i: leg_bars(i, z_floor))
    if floor:
        links.append(bar(a_handle[0], (a_handle[0][0], -a_handle[0][1], a_handle[0][2]), T_BAR))
    for m_ in insertion:
        m_.setdefault("why", "")
    labels = []
    d = build_lock(B, T_free, insertion, gates, clearance, links, a_grips, tier=tier, name=name, A_in_assembly=T_A,
                   hands=dict(left="a_grip", right="b_grip"), summary=summary)
    for k, mv in enumerate(d["moves"]):
        mv["label"] = f"{k + 1}. {mv['label']}"
    return d


# ------------------------------------------------------------------------------------------------- easier
def easier(head="arm:left:45", clearance=.004, T_A=None, b_grip="axial"):
    """Through-wall key: the claw leaves forward through GX, handle last; the hand on the handle cannot follow it
    through G1. The lock stands on the table on legs; the left hand steadies it while the right hand works the key;
    at the hand change the right hand lets go of the handle, the left hand lets go of the lock (it stays standing) and
    takes the key's head grip behind GX, and pulls the key out."""
    T_A = pose((-.12, -.06, .20), rot("z", 15)) if T_A is None else T_A
    d = through_key("easier", "through_wall_key", yaw=-50., roll=90., T_A=T_A, clearance=clearance, b_grip=b_grip,
                    head=head, hand_change=dict(right=None, left="b_head"), legs=True, floor=False, free_extra=.06,
                    summary="pull back out of G2, swing 50 about z, roll 90, push forward through GX; the right hand "
                            "lets go of the handle, the left hand lets go of the standing lock and takes the head grip "
                            "behind GX, pulls the key out")
    d["a_on_table"] = True
    return d


def easy(head="arm:left:45", clearance=.004, T_A=None, b_grip="axial"):
    """Flip key: GX takes the claw upside down, a 180 deg roll about the shaft, more than one
    wrist turn can give with the grip in place on most layouts; the lock is held up (no legs) and the key leaves on the left: hand-over as medium
    (the right hand takes the lock's second grip, the left hand leaves the lock for the key's head grip)."""
    T_A = pose((-.12, -.06, .22), rot("z", 15)) if T_A is None else T_A
    return through_key("easy", "flip_key", yaw=-50., roll=180., T_A=T_A, clearance=clearance,
                       b_grip=b_grip, head=head, a_handle2=((0., .062, .05), .08),
                       hand_change=dict(right="a_grip2", left="b_head"), left_exclude=("GX",), free_extra=.06,
                       summary="pull back out of G2, swing 50 about z, flip the claw 180 about the shaft, push "
                               "through GX; right hand: key -> lock's second grip; left "
                               "hand: lock -> key's head grip; pull out")


def medium(head="arm:left:45", clearance=.0035, T_A=None, b_grip="axial", pitch=-25.):
    """Hand-over key: the exit gate GX is tilted 35 deg and faces the left arm: swing about z, tilt about y, roll about
    the shaft (three axes); the key comes out on the left, where only the left hand reaches it, but the left hand holds
    the lock: the right hand lets go of the key, takes the lock's second grip, the left hand lets go of the lock and
    takes the key's head grip, and pulls it out."""
    T_A = pose((-.12, -.06, .22), rot("z", 15)) if T_A is None else T_A
    return through_key("medium", "hand_over_key", yaw=-50., roll=90., pitch=pitch, T_A=T_A, clearance=clearance,
                       b_grip=b_grip, head=head,
                       a_handle=((-.075, -.064, -.064), .09), a_handle2=((0., .062, .05), .08),
                       hand_change=dict(right="a_grip2", left="b_head"), left_exclude=("GX",), free_extra=.06,
                       summary="pull back out of G2, swing 50 about z, tilt 35 about y, roll 90, push through GX; "
                               "right hand: key -> lock's second grip; left hand: lock -> key's head grip; pull out")


def hard(head="arm:left:45", clearance=.003, T_A=None, b_grip="axial", pitch=-40.):
    """Table key: medium's three-axis hand-over key, tighter (3 mm design clearance), with the claw dropped into a notch
    behind G2 (lift first), a steeper exit (tilt 40); the lock stands on legs on the table
    and both hands start empty at home: pick up and hold the lock, take the key, then medium's sequence."""
    T_A = pose((-.12, -.06, .22), rot("z", 15)) if T_A is None else T_A
    d = through_key("hard", "table_key", yaw=-50., roll=90., pitch=pitch, T_A=T_A, clearance=clearance, b_grip=b_grip,
                    head=head,
                    a_handle2=((0., .062, .05), .08), hand_change=dict(right="a_grip2", left="b_head"),
                    left_exclude=("GX",), free_extra=.10, pre_lift=.015, legs=True,
                    summary="lift the claw out of its notch, pull back out of G2, swing 50 about z, tilt 40 about y, "
                            "roll 90, push through GX; right hand: key -> lock's second grip; left hand: lock -> key's "
                            "head grip; pull out. Starts resting on the table, hands empty")
    d["start"] = "table"
    return d


TIERS = {"very_easy": very_easy, "easier": easier, "easy": easy, "medium": medium, "hard": hard}

LAYOUTS = {
    "easier": dict(T_A=(25., -.12, -.06, .18), b_grip="axial", head="above"),
    "easy": dict(T_A=(25., -.12, -.06, .20), b_grip="axial", head="above",
                 birth_q=dict(left=[-0.1896, -1.8398, 1.1747, 0.0, 0.8396, -0.0],
                              right=[0.4751, 0.3125, 0.4685, 1.0713, -1.1207, -0.6731])),
    "medium": dict(T_A=(25., -.08, -.02, .20), b_grip="axial", head="above", kw=dict(pitch=25.),
                   birth_q=dict(left=[0.156, -1.5013, 1.2224, 0.0, 0.4534, -0.0],
                                right=[0.4191, 0.1846, 0.6788, 0.9871, -1.1305, -0.5731]),
                   move_shares=[0., 0., .5, 0., 0., .5]),
    "hard": dict(T_A=(25., -.08, -.02, .10), b_grip="axial", head="above", kw=dict(pitch=25.),
                 birth_q=dict(left=[0.156, -1.6772, 1.5587, 0.0, 0.2931, -0.0],
                              right=[0.4191, 0.5541, 0.6766, 0.8845, -1.3502, -0.261]),
                 move_shares=[0., 0., 0., .5, 0., 0., .5]),
}


def make(tier, layout=None):
    """The tier's design with its robot layout applied."""
    from scipy.spatial.transform import Rotation as Rot
    lay = dict(LAYOUTS.get(tier, {}) if layout is None else layout)
    kw = {}
    if "T_A" in lay:
        yaw, x, y, z = lay["T_A"]
        T = np.eye(4)
        T[:3, :3] = Rot.from_euler("z", yaw, degrees=True).as_matrix()
        T[:3, 3] = [x, y, z]
        kw["T_A"] = T
    if "b_grip" in lay and tier != "very_easy":
        kw["b_grip"] = lay["b_grip"]
    if "head" in lay and tier != "very_easy":
        kw["head"] = lay["head"]
    kw.update(lay.get("kw") or {})
    d = TIERS[tier](**kw)
    d["birth_flip"] = dict(lay.get("birth_flip") or {})
    if lay.get("birth_q"):
        d["birth_q"] = lay["birth_q"]
    for mv, sh in zip(d["moves"], lay.get("move_shares") or []):
        if sh is None:
            continue
        sh, fx = (sh if isinstance(sh, (list, tuple)) else (sh, None))
        mv["share"] = float(sh)
        if fx:
            mv["fixed"] = fx
    d["layout"] = lay
    return d
