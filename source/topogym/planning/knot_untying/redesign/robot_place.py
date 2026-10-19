"""Placement stage (knot-robot, 10-12): an untied, settled rope laid into its goal region on the robot.

The untying moves leave the rope untied but mostly outside its region (inst10 hard_nostop: a 91-94 % out, b 24 %),
and the region layout in the order of the scene's layout path re-tangled it: its first leg runs along the region's
near side (next to the curb, where the outside part of the rope lies), so the rest of the rope had to be carried over
that placed leg to reach the later ones. Here the path is planned from the live rope:

  - a serpentine from the clamp along the region's long axis whose FIRST leg runs along the FAR side (away from where
    the rope lies outside) and every further leg one gap nearer, so the rope still outside always hangs off the
    nearest placed leg and never has to cross a placed one;
  - legs `gap` apart (>= 4 cm, the legs are nonlocal strands that need one diameter), U-turns of radius gap / 2, legs
    stop short of the clamp so a return leg never passes it, the near side kept `curb_standoff` from a curb;
  - the path at least as long as the rope (Untie.layout puts stations past a path's end on its last point);

then laid with RobotUntie.layout (hand-over-hand from the clamp outward, the previous station pinned, checked carries
with the hand-over through the bimanual lens), the leftover outside runs carried in by layout_fixup and a strand left
on a fixture flattened. No untying move is used.

Loaded by place_lab.py for tries from an untied checkpoint (lay(lab, U, scene, log=..., regions=..., **params)).
"""
from __future__ import annotations

import json

import numpy as np

from topogym.planning.knot_untying.redesign.mjlab import to_local
from topogym.planning.knot_untying.redesign.untie import BASES_LOCAL, Stop  # noqa: F401


def box_of(poly):
    p = np.asarray(poly, float)
    return p.min(axis=0), p.max(axis=0)


def curb_boxes(scene):
    out = []
    for f in scene.get("fixtures", []):
        if f.get("role") == "curb" and f.get("kind", "box") == "box":
            c = np.asarray(f["pos"][:2], float)
            h = np.asarray(f["size"][:2], float)
            out.append((c - h, c + h))
    return out


def rope_local(lab, rope):
    rod = lab.rods[rope]
    return np.asarray(to_local(rod.nodes(lab.d)), float), np.asarray(rod.s_nodes, float)


def inside_box(q, lo, hi):
    return (q[:, 0] >= lo[0]) & (q[:, 0] <= hi[0]) & (q[:, 1] >= lo[1]) & (q[:, 1] <= hi[1])


def plan_path(lab, U, rope, poly, scene, *, margin=.03, curb_standoff=.045, gap=.05, clamp_keep=.08, exit_len=.03,
              extra=.06, near_side=None):
    """The serpentine (task-local xy, from the clamp) and how it was chosen."""
    lo, hi = box_of(poly)
    P_now, S_now = rope_local(lab, rope)
    L = float(lab.length(rope))
    C = P_now[0, :2]
    ex = P_now[np.searchsorted(S_now, min(exit_len, S_now[-1]))][:2] - C
    ex = ex / max(np.linalg.norm(ex), 1e-9)
    long_x = (hi[0] - lo[0]) >= (hi[1] - lo[1])
    a = 0 if long_x else 1                       # the long axis
    b = 1 - a                                    # the across axis
    out = ~inside_box(P_now, lo, hi)
    if near_side is None:
        if out.any():
            near_side = 1 if P_now[out, b].mean() > .5 * (lo[b] + hi[b]) else -1
        else:
            near_side = 1 if C[b] < .5 * (lo[b] + hi[b]) else -1     # all inside: lay toward the clamp's far side
    # the across levels: from the far side toward the near side
    lim_lo, lim_hi = lo[b] + margin, hi[b] - margin
    for c_lo, c_hi in curb_boxes(scene):          # a curb along a long side: keep the standoff from its face
        if c_hi[a] < lo[a] or c_lo[a] > hi[a]:
            continue
        if c_hi[b] <= lo[b] + .03:
            lim_lo = max(lim_lo, c_hi[b] + curb_standoff)
        if c_lo[b] >= hi[b] - .03:
            lim_hi = min(lim_hi, c_lo[b] - curb_standoff)
    far = lim_lo if near_side > 0 else lim_hi
    near = lim_hi if near_side > 0 else lim_lo
    width = abs(near - far)
    # the long-axis extent: from the clamp's end to the far end; return legs stop clamp_keep short of the clamp
    d_long = 1. if C[a] < .5 * (lo[a] + hi[a]) else -1.
    end_far = hi[a] - margin if d_long > 0 else lo[a] + margin
    end_back = C[a] + d_long * clamp_keep
    leg = abs(end_far - end_back)
    # how many legs: the first runs from the clamp (diagonal to the far level) to the far end, every further one back
    # and forth between the far end and clamp_keep short of the clamp, joined by U-turns of radius g / 2
    first = abs(end_far - C[a]) + abs(far - C[b])
    n, g = 1, 0.
    while first + sum(leg + np.pi * min(gap * 1.6, width / max(n - 1, 1)) / 2 for _ in range(n - 1)) < L + extra:
        n += 1
        if n > 8:
            break
    g = min(gap * 1.6, width / (n - 1)) if n > 1 else 0.
    if n > 1 and g < gap * .8 - 1e-9:
        raise Stop(f"{rope}: {n} legs do not fit {width:.3f} m across (gap {g:.3f} < {gap * .8:.3f})")
    lvl = [far + float(np.sign(near - far)) * k * g for k in range(n)]
    r = g / 2.

    def pt(along, across):
        q = np.zeros(2)
        q[a], q[b] = along, across
        return q
    pts = [C.copy(), C + exit_len * ex]
    pts.append(pt(pts[-1][a] + d_long * .05, lvl[0]))               # to the far level
    for k in range(n):
        out_end = end_far if k % 2 == 0 else end_back              # where leg k ends (the U-turn's outer extreme)
        sgn = d_long if k % 2 == 0 else -d_long                    # leg k runs in this direction along the axis
        if k == n - 1:
            pts.append(pt(out_end, lvl[k]))
            break
        pts.append(pt(out_end - sgn * r, lvl[k]))                  # the leg, then a half circle to level k + 1
        cen = pt(out_end - sgn * r, .5 * (lvl[k] + lvl[k + 1]))
        for th in np.linspace(0, np.pi, 9)[1:-1]:
            off = np.zeros(2)
            off[a] = sgn * r * np.sin(th)
            off[b] = (lvl[k] - cen[b]) * np.cos(th)
            pts.append(cen + off)
        pts.append(pt(out_end - sgn * r, lvl[k + 1]))
    P = np.asarray(pts, float)
    seg = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if seg[-1] < L:                                                 # stretch the last leg (clipped to the box)
        d = P[-1] - P[-2]
        d = d / max(np.linalg.norm(d), 1e-9)
        P = np.vstack([P, P[-1] + d * (L - seg[-1] + .02)])
        P[-1] = np.clip(P[-1], lo + margin, hi - margin)
    info = dict(rope=rope, L=round(L, 3), legs=n, gap=round(float(g), 3), near_side=int(near_side),
                far=round(float(far), 3), near=round(float(near), 3), long_axis="xy"[a],
                path_len=round(float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))), 3),
                outside_now=round(float(out.mean()), 3))
    return P, info


def plan_path_split(lab, U, rope, poly, scene, *, margin=.03, curb_standoff=.045, gap=.05, clamp_keep=.08, exit_len=.03,
                    far_cap=.05, extra=.06, near_side=None):
    """A serpentine for a strip whose far corner away from the clamp is out of the arms' wrist reach (easier_v8 e9: the
    right arm cannot hold a rope-along-x pinch at (0.23, 0.30)): leg 1 on the far level from the clamp to `far_cap`
    (along the long axis, measured from the strip centre toward the far end), leg 2 back on the middle level, leg 3 on
    the near level all the way to the far end, leg 4 back on the middle level toward far_cap with the free end. The
    rope outside still hangs off the last placed leg until leg 4, which holds only the rope's end."""
    lo, hi = box_of(poly)
    P_now, S_now = rope_local(lab, rope)
    L = float(lab.length(rope))
    C = P_now[0, :2]
    ex = P_now[np.searchsorted(S_now, min(exit_len, S_now[-1]))][:2] - C
    ex = ex / max(np.linalg.norm(ex), 1e-9)
    a = 0 if (hi[0] - lo[0]) >= (hi[1] - lo[1]) else 1
    b = 1 - a
    out = ~inside_box(P_now, lo, hi)
    if near_side is None:
        near_side = (1 if P_now[out, b].mean() > .5 * (lo[b] + hi[b]) else -1) if out.any() else \
            (1 if C[b] < .5 * (lo[b] + hi[b]) else -1)
    lim_lo, lim_hi = lo[b] + margin, hi[b] - margin
    for c_lo, c_hi in curb_boxes(scene):
        if c_hi[a] < lo[a] or c_lo[a] > hi[a]:
            continue
        if c_hi[b] <= lo[b] + .03:
            lim_lo = max(lim_lo, c_hi[b] + curb_standoff)
        if c_lo[b] >= hi[b] - .03:
            lim_hi = min(lim_hi, c_lo[b] - curb_standoff)
    far = lim_lo if near_side > 0 else lim_hi
    near = lim_hi if near_side > 0 else lim_lo
    g = abs(near - far) / 2.
    if g < gap * .8:
        raise Stop(f"{rope}: 3 levels do not fit {abs(near - far):.3f} m across")
    sg = float(np.sign(near - far))
    lv = [far, far + sg * g, near]
    d_long = 1. if C[a] < .5 * (lo[a] + hi[a]) else -1.
    mid_a = .5 * (lo[a] + hi[a])
    cap = mid_a + d_long * far_cap
    end_far = hi[a] - margin if d_long > 0 else lo[a] + margin
    end_back = C[a] + d_long * clamp_keep
    r = g / 2.

    def pt(along, across):
        q = np.zeros(2)
        q[a], q[b] = along, across
        return q

    def uturn(at_along, from_lvl, to_lvl, sgn):
        cen = pt(at_along - sgn * r, .5 * (from_lvl + to_lvl))
        out_ = []
        for th in np.linspace(0, np.pi, 9)[1:-1]:
            off = np.zeros(2)
            off[a] = sgn * r * np.sin(th)
            off[b] = (from_lvl - cen[b]) * np.cos(th)
            out_.append(cen + off)
        return out_
    pts = [C.copy(), C + exit_len * ex]
    pts.append(pt(pts[-1][a] + d_long * .05, lv[0]))
    pts.append(pt(cap - d_long * r, lv[0]))                       # leg 1: far level to the cap
    pts += uturn(cap, lv[0], lv[1], d_long)
    pts.append(pt(cap - d_long * r, lv[1]))
    pts.append(pt(end_back + d_long * r, lv[1]))                  # leg 2: middle level back toward the clamp
    pts += uturn(end_back, lv[1], lv[2], -d_long)
    pts.append(pt(end_back + d_long * r, lv[2]))
    pts.append(pt(end_far - d_long * r, lv[2]))                   # leg 3: near level to the far end
    pts += uturn(end_far, lv[2], lv[1], d_long)
    pts.append(pt(end_far - d_long * r, lv[1]))
    pts.append(pt(cap + d_long * .03, lv[1]))                     # leg 4: middle level back toward the cap
    P = np.asarray(pts, float)
    seg = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    info = dict(rope=rope, L=round(L, 3), legs=4, gap=round(float(g), 3), near_side=int(near_side), far=round(float(far), 3),
                near=round(float(near), 3), cap=round(float(cap), 3), path_len=round(float(seg[-1]), 3),
                outside_now=round(float(out.mean()), 3), split=True)
    if seg[-1] < L - .05:
        info["short_m"] = round(float(L - seg[-1]), 3)
    return P, info


def lay(lab, U, scene, *, log=print, regions=None, order=None, spacing=.14, lift=.05, carry_speed=.03, margin=.03,
        gap=.05, curb_standoff=.045, clamp_keep=.08, fixup=True, flatten=True, max_placements=None, near_side=None,
        **_):
    """v1: the far-side-first serpentine, laid by RobotUntie.layout, then layout_fixup and flatten_high."""
    regions = regions or scene.get("regions") or {}
    names = [r["name"] for r in scene.get("ropes", [])] or list(lab.rods)
    order = order or sorted([n for n in names if n in regions],
                            key=lambda n: lab.goal_metrics(region=regions)["ropes"][n].get("outside_fraction", 0.))
    recs = []
    for rope in order:
        gm = lab.goal_metrics(region=regions)["ropes"][rope]
        if gm.get("in_region") and gm.get("on_table"):
            recs.append(dict(rope=rope, skipped="in region and on the table"))
            continue
        P, info = plan_path(lab, U, rope, regions[rope], scene, margin=margin, curb_standoff=curb_standoff, gap=gap,
                            clamp_keep=clamp_keep, near_side=near_side)
        info["path"] = np.round(P, 3).tolist()
        log("place path", json.dumps(info)[:1500])
        rope_start = np.round(rope_local(lab, rope)[0][::3, :2], 3).tolist()
        r = U.layout(rope, P, spacing=spacing, lift=lift, carry_speed=carry_speed, max_placements=max_placements)
        rec = dict(rope=rope, plan={k: v for k, v in info.items() if k != "path"}, crossings_after_layout=U.count(),
                   placements=len(r.get("placements", [])),
                   errors=[p.get("error", "")[:120] for p in r.get("placements", []) if p.get("error")])
        log("place layout", json.dumps(rec)[:1200])
        if fixup and hasattr(U, "layout_fixup"):
            g2 = lab.goal_metrics(region=regions)["ropes"][rope]
            if g2.get("in_region") is False:
                f = U.layout_fixup(rope, regions[rope])
                rec["fixup"] = {k: f.get(k) for k in ("outside_fraction", "crossings_after")} if isinstance(f, dict) else f
        if flatten and hasattr(U, "flatten_high"):
            g3 = lab.goal_metrics(region=regions)["ropes"][rope]
            if not lab.crossings() and g3.get("planar_clear") and not g3.get("on_table"):
                rec["flatten"] = U.flatten_high(rope)
        rec["goal"] = lab.goal_metrics(region=regions)["ropes"][rope]
        rec["crossings"] = U.count()
        log("place rope", json.dumps(rec, default=str)[:1500])
        rec.update(path=info["path"], rope_start=rope_start,
                   rope_end=np.round(rope_local(lab, rope)[0][::3, :2], 3).tolist())
        recs.append(rec)
    return dict(kind="place_v1", ropes=recs)


def plan_only(lab, U, scene, *, log=print, regions=None, **kw):
    """No motion: the paths v1 would lay (for a look at the plan)."""
    regions = regions or scene.get("regions") or {}
    out = {}
    for rope in regions:
        if rope not in lab.rods:
            continue
        try:
            P, info = plan_path(lab, U, rope, regions[rope], scene, **{k: v for k, v in kw.items() if k in (
                "margin", "curb_standoff", "gap", "clamp_keep", "near_side")})
            info["path"] = np.round(P, 3).tolist()
        except Stop as e:
            info = dict(error=str(e))
        P_now, _ = rope_local(lab, rope)
        info["rope_now"] = np.round(P_now[::6, :2], 3).tolist()
        out[rope] = info
        log("plan", json.dumps(info)[:3000])
    return out


# ------------------------------------------------------------------------------------------- v2: two-hand lays
def path_at(P, s):
    seg = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    s = float(np.clip(s, 0., seg[-1]))
    return np.r_[np.interp(s, seg, P[:, 0]), np.interp(s, seg, P[:, 1])]


def path_legs(P, min_len=.10):
    """The straight portions of the serpentine: [(s_start, s_end)] (the U-turns are short chords)."""
    seg = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    out = []
    for i in range(len(P) - 1):
        if seg[i + 1] - seg[i] >= min_len:
            out.append((float(seg[i]), float(seg[i + 1])))
    return out


def crosses_curb(scene, a_xy, b_xy, pad=.025):
    """Whether the plan segment a -> b passes over a curb's footprint grown by pad."""
    for lo, hi in curb_boxes(scene):
        lo, hi = lo - pad, hi + pad
        for u in np.linspace(0., 1., 41):
            q = a_xy + (b_xy - a_xy) * u
            if lo[0] <= q[0] <= hi[0] and lo[1] <= q[1] <= hi[1]:
                return True
    return False


def past_curb(scene, a_xy, b_xy, pad=.025, beyond=.04):
    """On the plan segment a -> b: the point `beyond` past the last point over a curb footprint grown by pad (None if
    the segment never meets a curb)."""
    last = None
    for u in np.linspace(0., 1., 81):
        q = a_xy + (b_xy - a_xy) * u
        for lo, hi in curb_boxes(scene):
            if lo[0] - pad <= q[0] <= hi[0] + pad and lo[1] - pad <= q[1] <= hi[1] + pad:
                last = u
    if last is None:
        return None
    L = float(np.linalg.norm(b_xy - a_xy))
    u = min(1., last + beyond / max(L, 1e-9))
    return a_xy + (b_xy - a_xy) * u


def lay2(lab, U, scene, *, log=print, regions=None, order=None, margin=.03, gap=.05, curb_standoff=.045,
         clamp_keep=.08, delta=.02, low=.03, high=.09, over_curb=.09, speed=.03, lower_speed=.03,
         fixup=True, flatten=True, settle=.3, max_pairs=None, near_side=None, first_s=.24, end_keep=.06,
         fixup_below=.35, touch_rounds=2, carry_rot=.2, **_):
    """v2 (the coordinator's 10-12 recipe): per straight leg of the far-side-first serpentine one two-hand lay: the
    leg's two end stations pinched (left hand on the western one), lifted low (`low`; `over_curb` when either hand's
    way crosses a curb), carried together in straight lines to their places on the leg, lowered, released at the
    approach width; legs from the clamp outward. Then layout_fixup / flatten_high for what is left."""
    from topogym.planning.knot_untying.redesign.mjlab import to_world
    from topogym.planning.knot_untying.redesign.robot_lab import OPEN_W
    regions = regions or scene.get("regions") or {}
    names = [r["name"] for r in scene.get("ropes", [])] or list(lab.rods)
    order = order or sorted([n for n in names if n in regions],
                            key=lambda n: lab.goal_metrics(region=regions)["ropes"][n].get("outside_fraction", 0.))
    z_rope = None
    recs = []
    for rope in order:
        gm = lab.goal_metrics(region=regions)["ropes"][rope]
        if gm.get("in_region") and gm.get("on_table"):
            recs.append(dict(rope=rope, skipped="in region and on the table"))
            continue
        if gm.get("outside_fraction", 1.) < fixup_below and not lab.crossings() and hasattr(U, "layout_fixup"):
            # mostly inside already (hard_nostop b, 24 % out): the fix-up carried its outside runs in without a crossing
            # (the cloud host hnnp_fx / hnc0_fx); a full re-lay would move the placed part too
            f = U.layout_fixup(rope, regions[rope])
            rec = dict(rope=rope, mode="fixup", fixup={k: f.get(k) for k in ("outside_fraction", "crossings_after")}
                       if isinstance(f, dict) else f)
            if flatten and hasattr(U, "flatten_high"):
                g3 = lab.goal_metrics(region=regions)["ropes"][rope]
                if not lab.crossings() and g3.get("planar_clear") and not g3.get("on_table"):
                    rec["flatten"] = U.flatten_high(rope)
            rec["goal"] = lab.goal_metrics(region=regions)["ropes"][rope]
            rec["crossings"] = U.count()
            rec["rope_end"] = np.round(rope_local(lab, rope)[0][::3, :2], 3).tolist()
            log("lay2 rope", json.dumps({k: v for k, v in rec.items() if k != "rope_end"}, default=str)[:1500])
            recs.append(rec)
            continue
        z_rope = lab.rods[rope].radius + .001
        P, info = plan_path(lab, U, rope, regions[rope], scene, margin=margin, curb_standoff=curb_standoff, gap=gap,
                            clamp_keep=clamp_keep, near_side=near_side)
        L = float(lab.length(rope))
        legs = path_legs(P)
        pairs = []
        for s0, s1 in legs:
            a_, b_ = max(s0 + delta, first_s), min(s1 - delta, L - end_keep)
            if b_ - a_ < .08:
                continue
            pairs.append((a_, b_))
            if s1 >= L - .03:
                break
        info["pairs"] = [(round(a_, 3), round(b_, 3)) for a_, b_ in pairs]
        info["rope_now"] = np.round(rope_local(lab, rope)[0][::3, :2], 3).tolist()
        info["path"] = np.round(P, 3).tolist()
        log("lay2 path", json.dumps({k: v for k, v in info.items() if k != "rope_now"})[:1800])
        moves = []

        def carry_height(h, here_xy, tgt_xy):
            return over_curb if crosses_curb(scene, here_xy, tgt_xy) else low

        def lower_release(st, tgt):
            down = {h: to_world(np.r_[tgt[h], z_rope]) for h in st}
            lab.move(down, lower_speed, label="lay2_lower", tol=.006, tol_rot=carry_rot)
            for h in st:
                lab.release(h, reason="lay2_release", width=OPEN_W)
            lab.wait(.1)
            lab.move({h: lab.hand_pos(h) + [0, 0, .05] for h in st}, .1, label="lay2_retreat", tol=.008, tol_rot=.25)

        def ik_ok(h, R, pts, er_max=.12):
            for c in pts:
                q, ep, er = lab.solve_contact(h, c, R)
                if ep > .002 or er > er_max or lab.ik.wrist_unsafe(q):
                    return False
            return True

        def pick_station(h, s_want, window=.10, busy=()):
            """A station within `window` of s_want (nearest first) that h can pinch now and whose pinch yaw the arm
            holds at the station's place on the path, at the carry height and at the table (a yaw turned while holding
            lost the weld: e9_dev try 3, relative_pose_error after a 0.9 rad turn). -> (s, flip) or None."""
            for d in sorted(np.arange(-window, window + 1e-9, .01), key=abs):
                s_ = float(s_want + d)
                if s_ < .06 or s_ > L - end_keep or any(abs(s_ - b_) < .08 for b_ in busy):
                    continue
                now = U.local(rope, s_)
                if not U.reachable(h, now) or lab.pinch_clearance_at(rope, s_) < lab.pinch_clearance:
                    continue
                t_ = path_at(P, s_)
                w_ = past_curb(scene, now[:2], t_)
                if not U.reachable(h, np.r_[t_, low]) or (w_ is not None and not U.reachable(h, np.r_[w_, over_curb])):
                    continue
                try:
                    R, c, q_pre, flip, gap = U.pinch_plan(h, rope, s_, need_gap=U.station_gap(h))
                except Stop:
                    continue
                pts = [to_world(np.r_[t_, z_rope + low]), to_world(np.r_[t_, z_rope])]
                if w_ is not None:
                    pts = [to_world(np.r_[w_, z_rope + over_curb]), to_world(np.r_[w_, z_rope + low])] + pts
                start = w_ if w_ is not None else now[:2]
                mids = [to_world(np.r_[start + (t_ - start) * u, z_rope + low]) for u in (1 / 3, 2 / 3)]
                if ik_ok(h, R, pts) and ik_ok(h, R, mids, er_max=max(.12, carry_rot - .05)):
                    return s_, flip
            return None

        def carry_all(st, tgt, mv):
            """The held stations: a hand whose way crosses a curb lifts to `over_curb`, goes to just past the curb and
            comes down to `low`; the others lift to `low`; then every hand goes to its place at `low`, lowers and lets go
            at the approach width (dev: at 9 cm the right arm lost its wrist room on the way to the far side)."""
            here = {h: to_local(lab.hand_pos(h)) for h in st}
            wp = {h: past_curb(scene, here[h][:2], tgt[h]) for h in st}
            lab.move({h: lab.hand_pos(h) + [0, 0, over_curb if wp[h] is not None else low] for h in st}, speed,
                     label="lay2_lift", tol=.004, tol_rot=.25)
            cross = [h for h in st if wp[h] is not None]
            if cross:
                lab.move({h: to_world(np.r_[wp[h], z_rope + over_curb]) for h in cross}, speed, label="lay2_over_curb",
                         tol=.006, tol_rot=carry_rot)
                lab.move({h: to_world(np.r_[wp[h], z_rope + low]) for h in cross}, speed, label="lay2_down_low",
                         tol=.006, tol_rot=carry_rot)
            above = {h: to_world(np.r_[tgt[h], z_rope + low]) for h in st}
            whys = {h: w for h in st if (w := U.legs_check(h, [(lab.hand_pos(h), above[h], False)], lab.R[h],
                                                           tol_rot=.45)) is not None}
            mv.setdefault("over_curb", []).append({h: (None if wp[h] is None else np.round(wp[h], 3).tolist()) for h in st})
            if whys:
                mv.setdefault("checks", []).append({h: str(w)[:160] for h, w in whys.items()})
            lab.move(above, speed, label="lay2_carry", tol=.006, tol_rot=carry_rot)
            stopped = {h: lab.reach_stopped.get(h) for h in st if lab.reach_stopped.get(h) is not None}
            if stopped:
                mv.setdefault("reach_stopped", []).append({h: str(v)[:200] for h, v in stopped.items()})
            lower_release(st, tgt)

        for k, (sa, sb) in enumerate(pairs):
            if max_pairs is not None and k >= max_pairs:
                break
            ta, tb = path_at(P, sa), path_at(P, sb)
            west_a = ta[0] <= tb[0]
            hand_of = {"left": sa if west_a else sb, "right": sb if west_a else sa}
            mv = dict(k=k, rope=rope, s={h: round(v, 3) for h, v in hand_of.items()},
                      target={h: np.round(path_at(P, v), 3).tolist() for h, v in hand_of.items()})
            try:
                U.release_all()
                # both arms reach their own station now and its place (at the carry height): one two-hand lay; else the
                # stations one at a time, each by an arm that reaches it and its place (the far one along the rope first)
                ok2 = True
                for h, s_ in hand_of.items():
                    now = U.local(rope, s_)
                    t_ = path_at(P, s_)
                    if not (U.reachable(h, now) and U.reachable(h, np.r_[t_, low])):
                        ok2 = False
                xl, xr = U.local(rope, hand_of["left"])[0], U.local(rope, hand_of["right"])[0]
                if xl > xr + .05:
                    ok2 = False                     # the stations lie the other way round: the arms would cross
                mv["mode"] = "two_hand" if ok2 else "one_by_one"
                picks = {}
                if ok2:
                    for h in ("left", "right"):
                        pk = pick_station(h, hand_of[h], busy=[v[0] for v in picks.values()])
                        if pk is None:
                            ok2 = False
                            break
                        picks[h] = pk
                    mv["mode"] = "two_hand" if ok2 else "one_by_one"
                if ok2:
                    st = {}
                    for h in ("left", "right"):
                        s_st, flip = picks[h]
                        U.take(h, rope, s_st, label=f"lay2_{h}", flip=flip)
                        g_ = lab.hands[h]["grasp"] or {}
                        st[h] = float(g_.get("s", s_st))          # take() may have slid the station
                    mv["stations"] = {h: round(float(v), 3) for h, v in st.items()}
                    carry_all(st, {h: path_at(P, st[h]) for h in st}, mv)
                else:
                    done_ = []
                    for s_ in sorted(hand_of.values(), reverse=True):
                        t_ = path_at(P, s_)
                        pk, h = None, None
                        for h_ in sorted(("left", "right"), key=lambda h_: float(np.linalg.norm(t_ - BASES_LOCAL[h_]))):
                            pk = pick_station(h_, s_)
                            if pk is not None:
                                h = h_
                                break
                        if pk is None:
                            done_.append(dict(s=round(s_, 3), error="no station both reachable and holding its yaw"))
                            continue
                        s_st, flip = pk
                        U.take(h, rope, s_st, label=f"lay2_one_{h}", flip=flip)
                        s_st = float((lab.hands[h]["grasp"] or {}).get("s", s_st))
                        carry_all({h: s_st}, {h: path_at(P, s_st)}, mv)
                        done_.append(dict(s=round(float(s_st), 3), hand=h))
                    mv["one_by_one"] = done_
                lab.wait(settle)
                mv["placed_err_mm"] = {str(round(float(s_), 3)): round(float(np.linalg.norm(U.local(rope, s_)[:2]
                                                                                             - path_at(P, s_))) * 1000, 1)
                                       for s_ in hand_of.values()}
            except Stop as e:
                mv["error"] = str(e)[:300]
                U.release_all()
            mv["crossings"] = U.count()
            g = lab.goal_metrics(region=regions)["ropes"][rope]
            mv["outside"] = round(float(g.get("outside_fraction", 0.)), 3)
            log("lay2 move", json.dumps(mv, default=str)[:1500])
            moves.append(mv)
        rec = dict(rope=rope, plan={k: v for k, v in info.items() if k not in ("path", "rope_now")}, moves=moves,
                   path=info["path"], rope_start=info["rope_now"])
        if fixup and hasattr(U, "layout_fixup"):
            g2 = lab.goal_metrics(region=regions)["ropes"][rope]
            if g2.get("in_region") is False and not lab.crossings():
                f = U.layout_fixup(rope, regions[rope])
                rec["fixup"] = {k: f.get(k) for k in ("outside_fraction", "crossings_after")} if isinstance(f, dict) else f
        if flatten and hasattr(U, "flatten_high"):
            g3 = lab.goal_metrics(region=regions)["ropes"][rope]
            if not lab.crossings() and g3.get("planar_clear") and not g3.get("on_table"):
                rec["flatten"] = U.flatten_high(rope)
        rec["goal"] = lab.goal_metrics(region=regions)["ropes"][rope]
        rec["crossings"] = U.count()
        rec["rope_end"] = np.round(rope_local(lab, rope)[0][::3, :2], 3).tolist()
        log("lay2 rope", json.dumps({k: v for k, v in rec.items() if k not in ("moves", "path", "rope_start", "rope_end")},
                                    default=str)[:1500])
        recs.append(rec)
    # touch-up: every rope still partly outside (and no crossing anywhere) gets the fix-up again, at most `touch_rounds`
    # times (hn try 2: b's first fix-up left 4.9 % outside)
    for _ in range(touch_rounds):
        todo = [r for r in order if lab.goal_metrics(region=regions)["ropes"][r].get("in_region") is False]
        if not todo or lab.crossings() or not hasattr(U, "layout_fixup"):
            break
        for rope in todo:
            f = U.layout_fixup(rope, regions[rope])
            log("lay2 touch-up", json.dumps({"rope": rope, **({k: f.get(k) for k in ("outside_fraction", "crossings_after")}
                                                             if isinstance(f, dict) else {})}, default=str))
            if lab.crossings():
                break
    if flatten and hasattr(U, "flatten_high") and not lab.crossings():
        for rope in order:
            g3 = lab.goal_metrics(region=regions)["ropes"][rope]
            if g3.get("planar_clear") and not g3.get("on_table"):
                U.flatten_high(rope)
    return dict(kind="place_v2", ropes=recs, final=lab.goal_metrics(region=regions))


def snapshot(lab, U, scene, *, log=print, regions=None, **_):
    """No motion: the state for the design packet (every rope's full centre line, task-local xyz; the hands; the
    goal reading with the regions)."""
    regions = regions or scene.get("regions") or {}
    out = dict(kind="snapshot", t=round(float(lab.t), 4), row=int(lab.s.n_steps), crossings=U.count(),
               goal=lab.goal_metrics(region=regions), ropes={}, hands={})
    for rope in lab.rods:
        P_now, S_now = rope_local(lab, rope)
        out["ropes"][rope] = dict(xyz=np.round(P_now, 5).tolist(), s=np.round(S_now, 5).tolist(),
                                  radius=float(lab.rods[rope].radius), length=float(lab.length(rope)))
    for h in ("left", "right"):
        out["hands"][h] = np.round(to_local(lab.hand_actual(h)), 4).tolist()
    log("snapshot", json.dumps(dict(row=out["row"], crossings=out["crossings"])))
    return out


def waypoints(lab, U, scene, *, log=print, regions=None, ops=(), **_):
    """A hand-written placement (the design packet's answer format), task frame (metres, z above the table):
      {"op": "grasp", "hand": "left", "rope": "a", "s": 0.62}      pinch rope a at 0.62 m from its clamped end
      {"op": "move", "to": {"left": [x, y, z], "right": [x, y, z]}, "speed": 0.05}   straight lines, hands together
      {"op": "lift", "hands": ["left"], "dz": 0.03}                 straight up
      {"op": "release", "hands": ["left", "right"], "width": 0.03}  open (0.03 = the approach width)
      {"op": "wait", "s": 0.5}
    A move whose IK fails stops that hand where it is (recorded); a grasp the World does not bind is recorded and the
    list goes on."""
    from topogym.planning.knot_untying.redesign.mjlab import to_world
    from topogym.planning.knot_untying.redesign.robot_lab import OPEN_W
    regions = regions or scene.get("regions") or {}
    out = []
    for i, op in enumerate(ops):
        kind = op.get("op")
        rec = dict(i=i, op=kind)
        try:
            if kind == "grasp":
                U.take(op["hand"], op["rope"], float(op["s"]), label=f"wp_{i}")
                rec["held"] = (lab.hands[op["hand"]]["grasp"] or {}).get("s")
            elif kind == "move":
                tgt = {h: to_world(np.asarray(p, float)) for h, p in op["to"].items()}
                lab.move(tgt, float(op.get("speed", .05)), label=f"wp_{i}", tol=.006, tol_rot=.35)
                rec["stopped"] = {h: str(lab.reach_stopped.get(h))[:160] for h in tgt if lab.reach_stopped.get(h)}
            elif kind == "lift":
                lab.move({h: lab.hand_pos(h) + [0, 0, float(op.get("dz", .03))] for h in op["hands"]}, .05,
                         label=f"wp_{i}", tol=.006, tol_rot=.35)
            elif kind == "release":
                for h in op["hands"]:
                    lab.release(h, reason=f"wp_{i}", width=float(op.get("width", OPEN_W)))
                lab.wait(.1)
            elif kind == "wait":
                lab.wait(float(op.get("s", .5)))
            else:
                rec["error"] = "unknown op"
        except Stop as e:
            rec["error"] = str(e)[:240]
        rec["crossings"] = U.count()
        log("waypoint", json.dumps(rec, default=str)[:600])
        out.append(rec)
    g = lab.goal_metrics(region=regions)
    ends = {r: np.round(rope_local(lab, r)[0][::3, :2], 3).tolist() for r in lab.rods}
    return dict(kind="waypoints", steps=out, goal=g, ropes=[dict(rope=r, rope_end=ends[r]) for r in lab.rods])


# ------------------------------------------------------------------------------- v3: pinned, alternating lays
def lay3(lab, U, scene, *, log=print, regions=None, order=None, margin=.03, gap=.05, curb_standoff=.045,
         clamp_keep=.08, delta=.02, low=.03, over_curb=.09, speed=.03, lower_speed=.03, first_s=.24, end_keep=.06,
         fixup_below=.35, touch_rounds=2, carry_rot=.2, settle=.3, flatten=True, near_side=None, split=False,
         far_cap=.05, low_after_curb=True, plan_regions=None, **_):
    """v3 (hn try 2: a leg end released on the table slid ~9-13 cm back toward the curb under the rope still hanging
    over it, and the next leg then crossed it): the first leg by a two-hand lay as in v2, but its end station stays
    pinned; then, leg by leg, the free hand pinches the NEXT leg's end station, carries it to its place (over a curb
    high, then low), lowers it, and only then the old pin lets go, so the hand that just placed becomes the pin. Legs
    alternate east / west, so the hands alternate too; a leg end only the pinning arm reaches is carried by it after the
    pin is let go. Ropes under `fixup_below` outside get the fix-up only; a touch-up fix-up and flatten finish."""
    from topogym.planning.knot_untying.redesign.mjlab import to_world
    from topogym.planning.knot_untying.redesign.robot_lab import OPEN_W
    regions = regions or scene.get("regions") or {}
    names = [r["name"] for r in scene.get("ropes", [])] or list(lab.rods)
    order = order or sorted([n for n in names if n in regions],
                            key=lambda n: lab.goal_metrics(region=regions)["ropes"][n].get("outside_fraction", 0.))
    recs = []

    def ik_ok(h, R, pts, er_max=.12):
        for c in pts:
            q, ep, er = lab.solve_contact(h, c, R)
            if ep > .002 or er > er_max or lab.ik.wrist_unsafe(q):
                return False
        return True

    for rope in order:
        gm = lab.goal_metrics(region=regions)["ropes"][rope]
        if gm.get("in_region") and gm.get("on_table"):
            recs.append(dict(rope=rope, skipped="in region and on the table"))
            continue
        if gm.get("outside_fraction", 1.) < fixup_below and not lab.crossings() and hasattr(U, "layout_fixup"):
            f = U.layout_fixup(rope, regions[rope])
            recs.append(dict(rope=rope, mode="fixup", fixup={k: f.get(k) for k in ("outside_fraction", "crossings_after")}
                             if isinstance(f, dict) else f))
            log("lay3 rope", json.dumps(recs[-1], default=str)[:800])
            continue
        z_rope = lab.rods[rope].radius + .001
        plan_poly = (plan_regions or {}).get(rope, regions[rope])
        if split:
            P, info = plan_path_split(lab, U, rope, plan_poly, scene, margin=margin, curb_standoff=curb_standoff,
                                      gap=gap, clamp_keep=clamp_keep, near_side=near_side, far_cap=far_cap)
        else:
            P, info = plan_path(lab, U, rope, plan_poly, scene, margin=margin, curb_standoff=curb_standoff, gap=gap,
                                clamp_keep=clamp_keep, near_side=near_side)
        L = float(lab.length(rope))
        legs = path_legs(P)
        ends = []
        for s0, s1 in legs:
            e = min(s1 - delta, L - end_keep)
            if e > (ends[-1] if ends else first_s) + .08:
                ends.append(e)
            if s1 >= L - end_keep:
                break
        info["leg_ends"] = [round(e, 3) for e in ends]
        rope_start = np.round(rope_local(lab, rope)[0][::3, :2], 3).tolist()
        log("lay3 path", json.dumps({k: v for k, v in info.items() if k != "path"})[:1200])

        def pick(h, s_want, window=.10, busy=(), place=None):
            for d in sorted(np.arange(-window, window + 1e-9, .01), key=abs):
                s_ = float(s_want + d)
                if s_ < .06 or s_ > L - end_keep or any(abs(s_ - b_) < .08 for b_ in busy):
                    continue
                now = U.local(rope, s_)
                if not U.reachable(h, now) or lab.pinch_clearance_at(rope, s_) < lab.pinch_clearance:
                    continue
                t_ = path_at(P, s_) if place is None else np.asarray(place, float)
                w_ = past_curb(scene, now[:2], t_)
                if not U.reachable(h, np.r_[t_, low if (w_ is None or low_after_curb) else over_curb]) or \
                        (w_ is not None and not U.reachable(h, np.r_[w_, over_curb])):
                    continue
                hc = low if (w_ is None or low_after_curb) else over_curb
                pts = [to_world(np.r_[t_, z_rope + hc]), to_world(np.r_[t_, z_rope])]
                if w_ is not None:
                    pts = [to_world(np.r_[w_, z_rope + over_curb]), to_world(np.r_[w_, z_rope + hc])] + pts
                start = w_ if w_ is not None else now[:2]
                mids = [to_world(np.r_[start + (t_ - start) * u, z_rope + hc]) for u in (1 / 3, 2 / 3)]
                # both jaw flips (the pinch turned 180 deg about the vertical): the place may suit the other one
                for fl in (None, 1, -1):
                    try:
                        R, c, q_pre, flip, gap_ = U.pinch_plan(h, rope, s_, need_gap=U.station_gap(h), flip=fl)
                    except Stop:
                        continue
                    if ik_ok(h, R, pts) and ik_ok(h, R, mids, er_max=max(.12, carry_rot - .05)):
                        return s_, flip
            return None

        def carry(hs, mv, places=None):
            """hs: {hand: station held}; lift (over a curb high), past the curb, down low, to the places, lower."""
            here = {h: to_local(lab.hand_pos(h)) for h in hs}
            tgt = {h: (path_at(P, hs[h]) if not places or h not in places else np.asarray(places[h], float)) for h in hs}
            wp = {h: past_curb(scene, here[h][:2], tgt[h]) for h in hs}
            lab.move({h: lab.hand_pos(h) + [0, 0, over_curb if wp[h] is not None else low] for h in hs}, speed,
                     label="lay3_lift", tol=.004, tol_rot=.25)
            cross = [h for h in hs if wp[h] is not None]
            if cross and low_after_curb:
                lab.move({h: to_world(np.r_[wp[h], z_rope + over_curb]) for h in cross}, speed, label="lay3_over_curb",
                         tol=.006, tol_rot=carry_rot)
                lab.move({h: to_world(np.r_[wp[h], z_rope + low]) for h in cross}, speed, label="lay3_down_low",
                         tol=.006, tol_rot=carry_rot)
            # without low_after_curb a hand that crossed a curb stays at the curb height to its place (hn 002: the
            # straight high carry laid leg 1 with no crossing; the high-then-low one made a crossing in hn 003 / hn2)
            lab.move({h: to_world(np.r_[tgt[h], z_rope + (low if (wp[h] is None or low_after_curb) else over_curb)])
                      for h in hs}, speed, label="lay3_carry", tol=.006, tol_rot=carry_rot)
            stopped = {h: str(lab.reach_stopped.get(h))[:160] for h in hs if lab.reach_stopped.get(h) is not None}
            if stopped:
                mv.setdefault("reach_stopped", []).append(stopped)
            lab.move({h: to_world(np.r_[tgt[h], z_rope]) for h in hs}, lower_speed, label="lay3_lower", tol=.006,
                     tol_rot=carry_rot)

        def relay(s_want, to_hand, mv, busy=()):
            """Move the stretch near s_want with the other arm to a relay point on to_hand's side of the middle (x +-0.05,
            y clipped to +-0.22 toward the place), so to_hand can pick it there. -> True if moved."""
            other = "left" if to_hand == "right" else "right"
            t_ = path_at(P, s_want)
            q = np.array([.05 if to_hand == "right" else -.05, float(np.clip(t_[1], -.22, .22))])
            pk = pick(other, s_want, busy=busy, place=q)
            if pk is None:
                return False
            U.take(other, rope, pk[0], label=f"lay3_relay_{other}", flip=pk[1])
            carry({other: held(other)}, mv, places={other: q})
            let_go(other)
            mv.setdefault("relay", []).append(dict(s=round(pk[0], 3), by=other, to=np.round(q, 3).tolist()))
            return True

        def let_go(h):
            if U.holding(h):
                lab.release(h, reason="lay3_release", width=OPEN_W)
                lab.wait(.1)
                lab.move({h: lab.hand_pos(h) + [0, 0, .05]}, .1, label="lay3_retreat", tol=.008, tol_rot=.25)

        def held(h):
            return float((lab.hands[h]["grasp"] or {}).get("s", np.nan))

        moves = []
        U.release_all()
        pin = None
        # step 0: the first leg, two hands (its start station at first_s, its end station); the end hand stays
        if ends:
            sa, sb = first_s, ends[0]
            ta, tb = path_at(P, sa), path_at(P, sb)
            hand_of = {"left": sa if ta[0] <= tb[0] else sb, "right": sb if ta[0] <= tb[0] else sa}
            end_hand = "right" if hand_of["right"] == sb else "left"
            mv = dict(step=0, rope=rope, want={h: round(v, 3) for h, v in hand_of.items()})
            try:
                picks = {}
                for h in ("left", "right"):
                    pk = pick(h, hand_of[h], busy=[v[0] for v in picks.values()])
                    if pk is None and relay(hand_of[h], h, mv):
                        pk = pick(h, hand_of[h], busy=[v[0] for v in picks.values()])
                    if pk is None:
                        raise Stop(f"{h}: no station near {hand_of[h]:.3f} both reachable and holding its yaw")
                    picks[h] = pk
                for h in ("left", "right"):
                    U.take(h, rope, picks[h][0], label=f"lay3_{h}", flip=picks[h][1])
                hs = {h: held(h) for h in ("left", "right")}
                mv["held"] = {h: round(v, 3) for h, v in hs.items()}
                carry(hs, mv)
                let_go("left" if end_hand == "right" else "right")
                pin = end_hand
                lab.wait(settle)
            except Stop as e:
                mv["error"] = str(e)[:300]
                U.release_all()
                pin = None
            mv["crossings"] = U.count()
            mv["outside"] = round(float(lab.goal_metrics(region=regions)["ropes"][rope].get("outside_fraction", 0.)), 3)
            log("lay3 move", json.dumps(mv, default=str)[:1200])
            moves.append(mv)
        # the next legs: the free hand places the next leg end while the pin holds the last one
        for k, e in enumerate(ends[1:], start=1):
            t_ = path_at(P, e)
            mv = dict(step=k, rope=rope, want=round(e, 3), target=np.round(t_, 3).tolist(), pin=pin)
            try:
                free = [h for h in ("left", "right") if h != pin]
                order_h = sorted(free, key=lambda h: float(np.linalg.norm(t_ - BASES_LOCAL[h])))
                pk, h = None, None
                for h_ in order_h:
                    pk = pick(h_, e, busy=[held(pin)] if pin else ())
                    if pk is not None:
                        h = h_
                        break
                if pk is None and pin is not None and pick(pin, e) is not None:
                    # the pinning arm itself reaches this leg end (a short last leg on its own side): let the pin go and
                    # place with it, before trying a relay (hn2 002z: a relay to the other side made 3 crossings)
                    let_go(pin)
                    mv["pin_let_go_first"] = pin
                    pk = pick(pin, e)
                    h = pin if pk is not None else None
                    pin = None
                if pk is None:
                    for h_ in order_h:
                        if relay(e, h_, mv, busy=[held(pin)] if pin else ()):
                            pk = pick(h_, e, busy=[held(pin)] if pin else ())
                            if pk is not None:
                                h = h_
                                break
                if pk is None and pin is not None:
                    # only the pinning arm reaches this leg end: let the pin go and carry with it
                    let_go(pin)
                    mv["pin_let_go_first"] = pin
                    pk = pick(pin, e)
                    h = pin if pk is not None else None
                    pin = None
                if pk is None:
                    raise Stop(f"no arm reaches a station near {e:.3f} and its place holding its yaw")
                U.take(h, rope, pk[0], label=f"lay3_{h}", flip=pk[1])
                mv["hand"], mv["held"] = h, round(held(h), 3)
                carry({h: held(h)}, mv)
                if pin is not None and pin != h:
                    let_go(pin)
                pin = h
                lab.wait(settle)
            except Stop as e_:
                mv["error"] = str(e_)[:300]
                U.release_all()
                pin = None
            mv["crossings"] = U.count()
            mv["outside"] = round(float(lab.goal_metrics(region=regions)["ropes"][rope].get("outside_fraction", 0.)), 3)
            mv["placed_err_mm"] = round(float(np.linalg.norm(U.local(rope, e)[:2] - t_)) * 1000, 1)
            log("lay3 move", json.dumps(mv, default=str)[:1200])
            moves.append(mv)
        U.release_all()
        rec = dict(rope=rope, plan={k: v for k, v in info.items() if k != "path"}, moves=moves, path=info["path"]
                   if "path" in info else np.round(P, 3).tolist(), rope_start=rope_start)
        rec["path"] = np.round(P, 3).tolist()
        rec["goal"] = lab.goal_metrics(region=regions)["ropes"][rope]
        rec["crossings"] = U.count()
        rec["rope_end"] = np.round(rope_local(lab, rope)[0][::3, :2], 3).tolist()
        log("lay3 rope", json.dumps({k: v for k, v in rec.items() if k not in ("moves", "path", "rope_start", "rope_end")},
                                    default=str)[:1200])
        recs.append(rec)
    for _ in range(touch_rounds):
        todo = [r for r in order if lab.goal_metrics(region=regions)["ropes"][r].get("in_region") is False]
        if not todo or lab.crossings() or not hasattr(U, "layout_fixup"):
            break
        for rope in todo:
            f = U.layout_fixup(rope, regions[rope])
            log("lay3 touch-up", json.dumps({"rope": rope, **({k: f.get(k) for k in ("outside_fraction", "crossings_after")}
                                                             if isinstance(f, dict) else {})}, default=str))
            if lab.crossings():
                break
    if flatten and hasattr(U, "flatten_high") and not lab.crossings():
        for rope in order:
            g3 = lab.goal_metrics(region=regions)["ropes"][rope]
            if g3.get("planar_clear") and not g3.get("on_table"):
                U.flatten_high(rope)
    return dict(kind="place_v3", ropes=recs, final=lab.goal_metrics(region=regions))
