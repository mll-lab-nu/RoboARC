"""Motion programs for the blanket_flatten redesign (inverse generation of the start states), executed by FlexSim.

A program is a list of stages. A stage grasps material points (hand -> material uv, metres, the task's flat material
coordinates: u along the 0.60 m length, v along the 0.48 m width, centre 0) on the live cloth, moves the hands along
segments, then releases (unless `keep`), optionally settling. Segment targets per hand:
  to=(x, y, h)        world x, y and height h above the table top
  by=(dx, dy, h)      relative to the grasped particle's position at grasp time (h above the table top)
  uv=(u, v, h)        the flat layout's world position of material point (u, v) (centre_xy + (u, v)), height h
  path=[...]          a list of such targets (each a dict with one of the keys above), followed in sequence
  hold                (no target) the hand stays
Generation programs (inverse generation: flat sheet -> start state) may use any number of "tool" hands and bar
grasps (bar=dict(u=(lo, hi), v=(lo, hi)): every flex vertex in that material box pinned to the hand), because they only
produce the start state.
"""
from __future__ import annotations

import math

import numpy as np


# ----------------------------------------------------------------------------------------------- geometry helpers
def mirror(p, a, b):
    """Reflect 2D point p across the line through a and b."""
    p, a, b = (np.asarray(x, float) for x in (p, a, b))
    d = (b - a) / np.linalg.norm(b - a)
    q = a + d * float((p - a) @ d)
    return 2 * q - p


def foot(p, a, b):
    p, a, b = (np.asarray(x, float) for x in (p, a, b))
    d = (b - a) / np.linalg.norm(b - a)
    return a + d * float((p - a) @ d)


def page_turn_path(p_uv, crease_a, crease_b, *, h_max=.12, lift=.05, land=.02, n=6, overshoot=0., start_h=None):
    """Waypoints (uv targets with heights) that turn material point p over the crease line a-b like a page: a lift,
    a semicircle about the crease (radius = distance to the crease, height capped at h_max), landing on the mirror
    image (moved `overshoot` further from the crease) at height `land`. Returns a list of dict(uv=(u, v, h))."""
    p = np.asarray(p_uv, float)
    c = foot(p, crease_a, crease_b)
    r = float(np.linalg.norm(p - c))
    out = [dict(uv=(*p, lift if start_h is None else start_h))]
    for k in range(1, n):
        t = k / n
        horiz = c + math.cos(math.pi * t) * (p - c)
        z = max(lift, min(h_max, r) * math.sin(math.pi * t) + lift * (1 - t))
        out.append(dict(uv=(*horiz, z)))
    m = mirror(p, crease_a, crease_b)
    if overshoot:
        m = m + overshoot * (m - c) / max(np.linalg.norm(m - c), 1e-9)
    out.append(dict(uv=(*m, land + .03)))
    out.append(dict(uv=(*m, land)))
    return out


# ----------------------------------------------------------------------------------------------- execution
def pick_material(sim, uv, search_m=0., tool=False):
    """The particle at material point uv; with search_m > 0, if its patch is occluded, the exposed particle nearest to
    uv in material coordinates within search_m."""
    c = sim.nearest_particle(uv)
    if tool or search_m <= 0:
        return c
    P = sim.particles()
    if sim.exposed(sim.patch(c, P), P):
        return c
    d = np.linalg.norm(sim.uv - np.asarray(uv, float), axis=1)
    for k in np.argsort(d):
        if d[k] > search_m:
            break
        if sim.exposed(sim.patch(int(k), P), P):
            return int(k)
    return c


def pick_top(sim, xy, radius=.02):
    """The topmost particle within `radius` (horizontal) of world point xy (grasp the top layer there)."""
    P = sim.particles()
    near = np.flatnonzero(np.linalg.norm(P[:, :2] - np.asarray(xy, float), axis=1) <= radius)
    if not len(near):
        near = np.array([int(np.argmin(np.linalg.norm(P[:, :2] - np.asarray(xy, float), axis=1)))])
    return int(near[np.argmax(P[near, 2])])


def pick_top_band(sim, band):
    """The topmost exposed particle with |x - band['x']| <= band.get('dx', .02) and band['y'][0] <= y <= band['y'][1]
    (world), e.g. the crest of a roll at one end."""
    P = sim.particles()
    m = (np.abs(P[:, 0] - band["x"]) <= band.get("dx", .02)) & (P[:, 1] >= band["y"][0]) & (P[:, 1] <= band["y"][1])
    idx = np.flatnonzero(m)
    if not len(idx):
        return int(np.argmin(np.abs(P[:, 0] - band["x"]) + np.abs(P[:, 1] - np.mean(band["y"]))))
    for k in idx[np.argsort(-P[idx, 2])][:40]:
        if sim.exposed(sim.patch(int(k), P), P):
            return int(k)
    return int(idx[np.argmax(P[idx, 2])])


def _target(sim, hand, spec, origin):
    if "to" in spec:
        x, y, h = spec["to"]
        return np.array([x, y, sim.table_top + h])
    if "by" in spec:
        dx, dy, h = spec["by"]
        return np.array([origin[hand][0] + dx, origin[hand][1] + dy, sim.table_top + h])
    if "uv" in spec:
        u, v, h = spec["uv"]
        return np.array([sim.centre_xy[0] + u, sim.centre_xy[1] + v, sim.table_top + h])
    raise ValueError(f"bad target {spec}")


def bar_vertices(sim, box):
    (u0, u1), (v0, v1) = box["u"], box["v"]
    m = (sim.vuv[:, 0] >= u0) & (sim.vuv[:, 0] <= u1) & (sim.vuv[:, 1] >= v0) & (sim.vuv[:, 1] <= v1)
    return np.flatnonzero(m)


def grasp_bar(sim, hand, verts, label=None, origin=None):
    """Generation tool grasp: pin the given flex vertices to the hand (placed at their centroid, or at `origin`, a
    world point, e.g. on a fold axis for a rigid page turn). Not the assist."""
    if hasattr(sim, "grasp_bar_vertices"):        # the VBD lab (vbdlab.py): pins through its cloth backend
        return sim.grasp_bar_vertices(hand, verts, origin=origin, label=label)
    if sim.held[hand] is not None:
        sim.release(hand)
    X = sim.vertices()
    p = X[verts].mean(0) if origin is None else np.asarray(origin, float)
    m, d = sim.model, sim.data
    d.mocap_pos[sim.mocap[hand]] = p
    d.mocap_quat[sim.mocap[hand]] = [1., 0., 0., 0.]
    if len(verts) > len(sim.pool[hand]):
        raise ValueError(f"bar of {len(verts)} vertices exceeds the {len(sim.pool[hand])} pins of {hand}")
    pins = []
    for e, v in zip(sim.pool[hand], verts):
        m.eq_obj2id[e] = int(sim.bodies[v])
        m.eq_data[e, 0:3] = X[v] - p
        m.eq_data[e, 3:6] = 0.
        d.eq_active[e] = 1
        pins.append((int(e), int(v)))
    sim.held[hand] = dict(centre=None, patch=None, pins=pins, pos=p.copy(), quat=np.array([1., 0, 0, 0]), bar=True)
    w = sim.window[hand]
    for _ in range(w.maxlen):
        w.append(0.)
    ev = dict(t=round(sim.time, 4), hand=hand, event="grasp_bar", vertices=int(len(verts)), label=label, ok=True)
    sim.events.append(ev)
    return ev


def run_stage(sim, stage, *, tool=False, log=None):
    """Execute one stage; returns its record. A failed acquisition (occluded, strained) stops the stage (ok False)."""
    if hasattr(sim, "robot_stage"):           # the robot routes' proxy (blanket_flatten/redesign_routes.py)
        rec = sim.robot_stage(stage)
        if log is not None:
            log.append(rec)
        return rec
    rec = dict(name=stage.get("name"), grasps={}, ok=True, t0=round(sim.time, 3))
    origin = {}
    picks = {}
    for hand, uv in (stage.get("grasp") or {}).items():
        picks[hand] = pick_material(sim, uv, stage.get("search_m", 0.), tool)
    for hand, xy in (stage.get("grasp_xy") or {}).items():
        picks[hand] = pick_top(sim, xy, stage.get("search_m", .02))
    for hand, band in (stage.get("grasp_top") or {}).items():
        picks[hand] = pick_top_band(sim, band)
    for hand, c in picks.items():
        ev = sim.grasp(hand, c, require_exposed=not tool, label=stage.get("name"))
        rec["grasps"][hand] = ev
        if not ev.get("ok"):
            rec.update(ok=False, reason=f"{hand}: {ev.get('reason')}")
            for h in origin:
                if sim.held[h] is not None:
                    sim.release(h, reason="stage aborted")
            if log is not None:
                log.append(rec)
            return rec
        origin[hand] = sim.particles()[c].copy()
    for hand, box in (stage.get("bar") or {}).items():
        verts = bar_vertices(sim, box)
        axis_origin = None
        if "axis" in box:        # rigid page turn: the hand frame sits on the fold axis (height h above the sheet)
            (a, b), h = box["axis"], box.get("axis_h", .005)
            mid = (np.asarray(a, float) + np.asarray(b, float)) / 2
            axis_origin = [sim.centre_xy[0] + mid[0], sim.centre_xy[1] + mid[1], sim.table_top + sim.sphere_r + h]
        ev = grasp_bar(sim, hand, verts, label=stage.get("name"), origin=axis_origin)
        rec["grasps"][hand] = ev
        origin[hand] = sim.hand_pos(hand)
    for seg in stage.get("segments", ()):
        for h in seg.get("release", ()):           # a hand-over: the other hand lets go before this segment
            if sim.held.get(h) is not None:
                sim.release(h, reason="hand-over")
        held_before = {h: sim.held[h] is not None for h in origin}
        tracks = {}
        for hand, spec in seg.get("hands", {}).items():
            if spec is None or spec == "hold":
                continue
            if "path" in spec:
                tracks[hand] = [_target(sim, hand, s, origin) for s in spec["path"]]
            else:
                tracks[hand] = [_target(sim, hand, spec, origin)]
        if seg.get("rot"):
            rotate_about_axis(sim, seg["rot"], seg["duration_s"], name=f"{stage.get('name')}:{seg.get('name')}")
        elif tracks:
            sim.path(tracks, seg["duration_s"], name=f"{stage.get('name')}:{seg.get('name')}")
        else:
            sim.step(int(round(seg["duration_s"] / sim.dt)))
        lost = [h for h in origin if held_before[h] and sim.held[h] is None]
        if lost:
            rec.update(ok=False, reason=f"lost {lost} in {seg.get('name')}: "
                       + "; ".join(str(e.get("reason")) for e in sim.events[-4:] if e.get("event") == "release"))
            break
    if not stage.get("keep"):
        for hand in origin:
            if sim.held[hand] is not None:
                sim.release(hand, reason="stage end")
    if stage.get("settle_s"):
        rec["settled_s"] = sim.settle(stage["settle_s"])
    rec["t1"] = round(sim.time, 3)
    if log is not None:
        log.append(rec)
    return rec


def run_stages(sim, stages, *, log=None):
    """Several stages fixed by one measurement (a same-side relay's two corners, one round of corner corrections), each
    a stage dict or a callable that builds it on the cloth when it starts. The lab runs them one after the other, each
    built just before it runs (the same as run_stage per stage). The robot routes' proxy builds them all on the cloth
    measured now and hands them to the robot as ONE planner call: no settle-and-replan between them."""
    if hasattr(sim, "robot_stages"):
        recs = sim.robot_stages(stages)
    else:
        recs = [run_stage(sim, st() if callable(st) else st) for st in stages]
    if log is not None:
        log.extend(recs)
    return recs


def run_program(sim, program, *, tool=False, stop_on_fail=True):
    records = []
    for stage in program:
        if "fold" in stage:          # generation tool: rigid page turn of everything on one side of a crease
            f = dict(stage["fold"])
            rec = rigid_fold(sim, f.pop("a"), f.pop("b"), f.pop("side"), name=stage.get("name", "fold"), **f)
            records.append(rec)
            continue
        rec = run_stage(sim, stage, tool=tool, log=records)
        if not rec["ok"] and stop_on_fail:
            break
    return records


# ----------------------------------------------------------------------------------------------- roll (generation)
def roll_positions(sim, rolled_m, *, r0=.015, spacing=.0105, axis="x", from_edge="far"):
    """Isometric rolled sheet: the material within `rolled_m` of the far edge (v = +0.24) wound into an Archimedean
    spiral (inner radius r0, layer spacing `spacing`) that sits on the table where the flat part ends; the far edge is
    the innermost layer (a carpet rolled from the far edge toward the robot). Returns flex vertex positions."""
    X = sim.flat_vertices()
    if rolled_m <= 0:
        return X
    W = .48
    z0 = sim.table_top + sim.sphere_r
    v = sim.vuv[:, 1]
    a = (W / 2 - v) if from_edge == "far" else (v + W / 2)      # material distance from the rolled edge
    # turns N with 2 pi N (r0 + s N / 2) = rolled length
    s = spacing
    A, B, C = math.pi * s, 2 * math.pi * r0, -rolled_m
    N = (-B + math.sqrt(B * B - 4 * A * C)) / (2 * A)
    R = r0 + s * N
    v_entry = (W / 2 - rolled_m) if from_edge == "far" else (-W / 2 + rolled_m)
    yc = sim.centre_xy[1] + v_entry
    zc = z0 + R
    sign = 1. if from_edge == "far" else -1.
    inroll = a <= rolled_m
    # arc length from the entry point inward: a_e = rolled_m - a; solve R phi - s phi^2 / (4 pi) = a_e
    a_e = rolled_m - a[inroll]
    k = s / (4 * math.pi)
    disc = np.maximum(R * R - 4 * k * a_e, 0.)
    phi = (R - np.sqrt(disc)) / (2 * k)
    rho = R - s * phi / (2 * math.pi)
    X[inroll, 1] = yc + sign * rho * np.sin(phi)
    X[inroll, 2] = zc - rho * np.cos(phi)
    return X


def drive_to(sim, targets_fn, duration_s, *, steps_per_update=1):
    """Kinematic generation drive: every free vertex follows targets_fn(s) (s 0 -> 1, smoothstep) with zero velocity,
    the physics skipped (an isometric analytic motion, as the family's ClothPreparation drives its target), then the
    caller releases it to the physics."""
    n = max(1, int(round(duration_s / sim.dt)))
    for i in range(n):
        s = (i + 1) / n
        s = s * s * (3 - 2 * s)
        sim.set_vertices(targets_fn(s))
        sim.tick += 1


def rotate_about_axis(sim, rot, duration_s, name="rotate"):
    """Turn hand rot["hand"] (its frame on the axis) by rot["angle_deg"] about the material axis a -> b (a horizontal
    line through the hand origin), smoothstep in time: a rigid page turn of everything it pins."""
    from scipy.spatial.transform import Rotation
    hand = rot["hand"]
    a, b = (np.asarray(x, float) for x in (rot["a"], rot["b"]))
    axis = np.array([*(b - a), 0.])
    axis /= np.linalg.norm(axis)
    n = max(1, int(round(duration_s / sim.dt)))
    q0 = Rotation.from_quat(np.roll(sim.hand_quat(hand) if hasattr(sim, "hand_quat")
                                    else sim.data.mocap_quat[sim.mocap[hand]], -1))
    p = sim.hand_pos(hand)
    for i in range(n):
        u = (i + 1) / n
        u = u * u * (3 - 2 * u)
        q = Rotation.from_rotvec(axis * math.radians(rot["angle_deg"]) * u) * q0
        sim.set_hand(hand, p + np.array([0., 0., rot.get("lift_m", 0.) * math.sin(math.pi * u)]), np.roll(q.as_quat(), 1))
        sim.step(1)
    sim.events.append(dict(t=round(sim.time, 4), event="segment", name=name, held={hand: sim.held[hand] is not None}))


def rigid_fold(sim, a_xy, b_xy, side_xy, *, angle_deg=180., d0=.015, duration_s=3., extra_h=.006, settle_s=1.5,
               name="fold", select="world", material_region=None):
    """Generation tool: fold everything on the `side_xy` side of the world crease line a-b (beyond d0 from it) over
    the line like a page, as a rigid kinematic drive (the selected vertices follow the rotation, the rest of the sheet
    is simulated), about an axis `extra_h` above the highest material in the flap and its landing zone; then release
    and settle. select="world" folds every layer on that side (a real fold of a stack); select="material" folds only
    the vertices in material_region (a callable uv -> bool mask) on that side."""
    a, b, side = (np.asarray(x, float) for x in (a_xy, b_xy, side_xy))
    d = (b - a) / np.linalg.norm(b - a)
    nrm = np.array([-d[1], d[0]])
    if (side - a) @ nrm < 0:
        nrm = -nrm
    X0 = sim.vertices()
    s = (X0[:, :2] - a) @ nrm
    sel = s >= d0
    if select == "material" and material_region is not None:
        sel &= material_region(sim.vuv)
    reach = float(s[sel].max()) if sel.any() else 0.
    zone = np.abs(s) <= reach + .02
    z_axis = float(X0[zone, 2].max()) + extra_h
    origin = np.array([*a, z_axis])
    axis = np.array([*d, 0.])
    # rotation sense: the flap first moves up
    from scipy.spatial.transform import Rotation
    probe = Rotation.from_rotvec(axis * .1).apply(np.array([*nrm, 0.]))
    sign = 1. if probe[2] > 0 else -1.
    rel = X0[sel] - origin
    n = max(1, int(round(duration_s / sim.dt)))
    idx = np.flatnonzero(sel)
    if hasattr(sim, "drive_begin"):                # the VBD lab: the flap pinned to a mocap frame on the axis
        sim.drive_begin(idx, origin)
        for i in range(n):
            u = (i + 1) / n
            u = u * u * (3 - 2 * u)
            q = Rotation.from_rotvec(axis * sign * math.radians(angle_deg) * u).as_quat()
            sim.drive_set([q[3], q[0], q[1], q[2]])
            sim.step(1)
        sim.drive_end()
        n = 0
    qadr, dadr = sim.qadr[idx].reshape(-1), sim.dadr[idx].reshape(-1)
    prev = X0[sel].copy()
    for i in range(n):
        u = (i + 1) / n
        u = u * u * (3 - 2 * u)
        R = Rotation.from_rotvec(axis * sign * math.radians(angle_deg) * u)
        Xs = origin + R.apply(rel)
        sim.data.qpos[qadr] = (Xs - sim.rest[idx]).reshape(-1)
        sim.data.qvel[dadr] = ((Xs - prev) / sim.dt).reshape(-1)
        prev = Xs
        sim.step(1)
    if not hasattr(sim, "drive_begin"):
        sim.data.qvel[dadr] = 0.
    sim.events.append(dict(t=round(sim.time, 4), event="rigid_fold", name=name, vertices=int(sel.sum()),
                           axis_z_above_table=round(z_axis - sim.table_top, 4)))
    st = sim.settle(settle_s, min_s=min(.5, settle_s)) if settle_s else 0.
    return dict(name=name, ok=True, vertices=int(sel.sum()), settled_s=st)


def tuck_positions(sim, crease_a_uv, crease_b_uv, flap_side_uv, X=None, *, cover_margin=.012, ramp=.03, lift_extra=0.):
    """A corner (or strip) folded UNDER the sheet: the material beyond the crease a-b (material coordinates; the side
    containing flap_side_uv) is mirrored across the crease onto the table, face up under the sheet, and the sheet over
    it is raised by one layer (2 sphere radii), ramping down to the table over `ramp` around the flap's footprint.
    Built on X (default: the flat sheet); returns vertex positions (an analytic target, then settled by physics)."""
    from scipy.spatial import cKDTree
    X = (sim.flat_vertices() if X is None else np.asarray(X, float)).copy()
    a, b, side = (np.asarray(v, float) for v in (crease_a_uv, crease_b_uv, flap_side_uv))
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([-d[1], d[0]])
    if (side - a) @ n < 0:
        n = -n
    s = (sim.vuv - a) @ n
    flap = s > 0
    z0 = sim.table_top + sim.sphere_r
    # flap: mirrored across the crease (in the sheet's current layout, assumed flat here), on the table
    Xf = X[flap].copy()
    off = 2 * s[flap]
    Xf[:, 0] -= off * n[0]
    Xf[:, 1] -= off * n[1]
    Xf[:, 2] = z0
    X[flap] = Xf
    main = np.flatnonzero(~flap)
    dist, _ = cKDTree(Xf[:, :2]).query(X[main, :2])
    w = np.clip(1 - (dist - cover_margin) / ramp, 0., 1.)
    w = w * w * (3 - 2 * w)
    # max with the current heights: several tucks compose
    X[main, 2] = np.maximum(X[main, 2], z0 + (2 * sim.sphere_r + .001 + lift_extra) * w)
    return X
