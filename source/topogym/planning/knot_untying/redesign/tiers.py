"""The redesigned knot_untying ladder (scene generators). Each builder returns a JSON-able scene spec:

  ropes     [dict(name, radius, centerline (authored, lifted; task-local m), anchor=dict(end, links) | None)]
  fixtures  [dict(name, role (clamp | post | block | curb), kind (box | cylinder | capsule), pos, size | radius+height,
            yaw_deg)]  task-local, pos = the solid's centre, z above the table top
  regions   {rope: polygon [[x, y], ...]}: the required final layout (task goal), or {} (the family's goal only)
  birth     the start-state recipe run on the MuJoCo chain (planning/knot_untying/redesign/birth.py): settle, tighten
            each knot by pulling its two sides apart with the virtual hands up to `force_n`, anchor, release, settle
  oracle    the untying plan the scene was derived from (material stations, order, hands; planning/.../oracle.py)

Task-local frame: (0, 0) = midpoint between the ALOHA bases (world x -0.1357, y 0); left base at (-0.469, -0.019), right
base at (+0.469, -0.019); each arm's vertical pinch reaches ~0.59 m from its base, both arms only the lens
x in [-0.11, 0.09] (|y| <= 0.1), narrowing to x ~ 0 at |y| = 0.35 (robot_ik reach map).
"""
from __future__ import annotations

import numpy as np

from topogym.tasks.knot_untying.redesign import geometry as G
from . import compose as C

R = G.R_ROPE
POST_H = .16          # clutter posts: tall enough that no rope / knot can be lifted over them by the anchored rope


def staple(name, end_point, end_dir, *, cover=.032, width=.034, top=.0095, thick=.006, leg=.008):
    """The clamp over an anchored rope end: a bridge plate over the last `cover` m of rope (its underside 1.5 mm above
    the rope's top) on two legs beside it. It hides the anchored end from the fingers (no pinch fits under / beside it)
    and marks where the rope is clamped. Returns fixtures (task-local)."""
    p = np.asarray(end_point, float)[:2]
    d = np.asarray(end_dir, float)[:2]
    d = d / np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    centre = p + d * (cover / 2 - .004)
    yaw = float(np.degrees(np.arctan2(d[1], d[0])))
    out = [dict(name=f"{name}_plate", role="clamp", kind="box", pos=[*centre, top + thick / 2],
                size=[cover, width, thick], yaw_deg=yaw, rgba=".25 .3 .38 1")]
    off = R + .0015 + leg / 2
    for k, sgn in enumerate((1, -1)):
        c = centre + sgn * off * n
        out.append(dict(name=f"{name}_leg{k}", role="clamp", kind="box", pos=[*c, top / 2], size=[cover, leg, top],
                        yaw_deg=yaw, rgba=".25 .3 .38 1"))
    return out


def post(name, xy, radius=.012, height=POST_H):
    return dict(name=name, role="post", kind="cylinder", pos=[float(xy[0]), float(xy[1]), height / 2], radius=radius,
                height=height, rgba=".45 .42 .38 1")


def block(name, xy, size, yaw_deg=0., role="block"):
    return dict(name=name, role=role, kind="box", pos=[float(xy[0]), float(xy[1]), size[2] / 2], size=list(size),
                yaw_deg=yaw_deg, rgba=".5 .47 .42 1")


# ALOHA base_link footprints (task-local AABB of the port's base meshes, z 0-79 mm) and the 5 cm keep-out around them:
# a rope resting there lies on the robot (the family's on_table check fails), a fixture there collides with the base.
BASE_FOOTPRINTS = {"left": ((-.666, -.367), (-.121, .083)), "right": ((.367, .666), (-.121, .083))}
KEEP_OUT_M = .05


def keep_out_violations(points_xy):
    """Indices of planar points inside a base footprint grown by KEEP_OUT_M."""
    P = np.asarray(points_xy, float)[:, :2]
    bad = np.zeros(len(P), bool)
    for (x0, x1), (y0, y1) in BASE_FOOTPRINTS.values():
        bad |= ((P[:, 0] > x0 - KEEP_OUT_M) & (P[:, 0] < x1 + KEEP_OUT_M) &
                (P[:, 1] > y0 - KEEP_OUT_M) & (P[:, 1] < y1 + KEEP_OUT_M))
    return np.flatnonzero(bad)


def rect(x0, x1, y0, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def serpentine(start, legs, turn_r=None):
    """A layout path (task-local xy polyline): from `start`, straight legs [(x_end, y), ...] joined by half-circle turns
    (the turn goes from one leg's y to the next leg's y, radius = half the lane gap). v9 (knot agent, 10-11): the turn
    bulges outward and its outermost point is the previous leg's x_end (the straight part stops one radius short), so the
    path never doubles back; v8's turns bulged inward (x0 - r cos a) and folded the path onto itself (hairpins 3-5 mm
    apart, below the goal's 2 r planar clearance). `turn_r` is ignored (kept for old call sites). Returns [[x, y], ...]."""
    pts = [list(map(float, start))]
    x_prev = float(start[0])
    for i, (x_end, y) in enumerate(legs):
        x_end, y = float(x_end), float(y)
        if i == 0:
            if abs(start[1] - y) > 1e-9:
                pts.append([float(start[0]), y])
            pts.append([x_end, y])
            x_prev = float(start[0])
            continue
        x0, y0 = pts[-1]
        s = 1. if x0 > (pts[-2][0] if len(pts) > 1 else x_prev) else -1.   # the previous leg's direction along x
        r = abs(y - y0) / 2
        cy = (y + y0) / 2
        cx = x0 - s * r                    # the straight part ends one radius before x_end
        pts[-1] = [cx, y0]
        for a in np.linspace(-np.pi / 2, np.pi / 2, 13)[1:-1]:
            pts.append([cx + s * r * np.cos(a), cy + np.sign(y - y0) * r * np.sin(a)])
        pts.append([cx, y])
        pts.append([x_end, y])
    return [[round(a, 4), round(b, 4)] for a, b in pts]


def _end_dir(rope, which):
    p, d = G.end_frame(rope, which)
    return p, d


def _finish(tier, ropes_planar, *, anchors, fixtures, regions, birth, oracle, notes, over=None):
    lifted, cr = G.lift(ropes_planar, over=over)
    rep = G.report(lifted)
    ropes = []
    for k, (name, p) in enumerate(zip([a["name"] for a in anchors], lifted)):
        ropes.append(dict(name=name, radius=R, centerline=p.tolist(), anchor=anchors[k].get("anchor")))
    # clamp fixtures at each anchored end
    for k, a in enumerate(anchors):
        if a.get("anchor"):
            which = a["anchor"]["end"]
            pt, d = _end_dir(lifted[k], which)
            fixtures = staple(f"clamp_{a['name']}", pt, -d) + fixtures
    rep["base_keep_out_violations"] = {r["name"]: int(len(keep_out_violations(np.asarray(r["centerline"]))))
                                       for r in ropes}
    rep["base_keep_out_violations"]["regions"] = {k: int(len(keep_out_violations(np.asarray(v)))) for k, v in regions.items()}
    return dict(tier=tier, design="knot_untying_redesign_v0", frame=dict(origin_world_xy=[-0.1356786, 0.],
                table_top_z=0.9196429, units="m"), ropes=ropes, fixtures=fixtures, regions=regions, birth=birth,
                oracle=oracle, authored_report=rep, notes=notes)


# ------------------------------------------------------------------------------------------------ very_easy
def very_easy():
    """Anchored snug figure-eight with a long free tail. 4 crossings (as the old hard), but: the standing end is
    clamped (no end spread, no far-end pin), the knot is tightened by a 3 N pull (snug: the free-end pull cinches it),
    and the free tail is 0.55 m (the withdrawal needs several strokes inside the arms' reach)."""
    core = G.place(G.figure_eight_core(.016), (-.06, 0.))       # v4: the pull drags the knot ~6 cm along the tail
    rope = C.knot_with_paths(core,
                             left_points=[(-.12, -.085), (-.22, -.085), (-.31, -.06)],
                             right_points=[(.08, -.09), (.20, -.12), (.31, -.07), (.34, .05), (.27, .15)])
    birth = dict(steps=[dict(op="settle", max_s=1.5), dict(op="anchor", rope="rope", end="start", links=2),
                        dict(op="tighten", rope="rope", force_n=1.5),
                        dict(op="release_all"), dict(op="settle", max_s=3.)])
    oracle = dict(kind="anchored_withdrawal", rope="rope", pin="over_strand_next_to_crossing",
                  stroke_max_m=.18, loosen=True, straighten="free_end_against_clamp")
    return _finish("very_easy", [rope], anchors=[dict(name="rope", anchor=dict(end="start", links=2))], fixtures=[],
                   regions={}, birth=birth, oracle=oracle,
                   notes="figure-eight, standing end clamped at (-0.31, -0.06), 0.55 m free tail into the right arm's zone")


# ------------------------------------------------------------------------------------------------ easier
def easier():
    """Anchored figure-eight, tighter (4 N), with clutter on the shortcut lines and a layout region behind a curb."""
    core = G.place(G.figure_eight_core(.016), (-.05, .05))      # v4: drift compensation (v3 drift +0.10, -0.07)
    rope = C.knot_with_paths(core,
                             left_points=[(-.12, -.02), (-.20, .03), (-.27, .11), (-.33, .18)],
                             right_points=[(.08, -.07), (.19, -.13), (.30, -.10), (.33, .00), (.30, .08)])
    fixtures = [post("post_away", (.10, -.035)), post("post_front", (-.005, -.115)),
                block("curb", (-.06, .135), (.34, .02, .03), role="curb")]
    regions = {"rope": rect(-.42, .30, .155, .33)}
    birth = dict(steps=[dict(op="settle", max_s=1.5), dict(op="anchor", rope="rope", end="start", links=2),
                        dict(op="tighten", rope="rope", force_n=2.),
                        dict(op="release_all"), dict(op="settle", max_s=3.)])
    oracle = dict(kind="anchored_withdrawal_then_layout", rope="rope", pin="over_strand_next_to_crossing",
                  stroke_max_m=.16, loosen=True, avoid=["post_away", "post_front"], layout_region=True,
                  layout_lift_over=["curb"],
                  layout_path={"rope": serpentine((-.33, .18), [(.28, .18), (-.415, .29)], .055)})
    return _finish("easier", [rope], anchors=[dict(name="rope", anchor=dict(end="start", links=2))],
                   fixtures=fixtures, regions=regions, birth=birth, oracle=oracle,
                   notes="clamp behind the curb (inside the region); posts on the clamp->knot line and in front")


# ------------------------------------------------------------------------------------------------ easy
def easy():
    """Stacked overhands (two same-handed trefoils in series, 6 crossings) on an anchored rope, clutter, region. The
    knot near the free end must go first; then the whole free part (~1 m) is drawn back through the first knot."""
    k1 = G.place(G.trefoil_core(.015)[::-1], (-.075, .05))
    k2 = G.place(G.trefoil_core(.015)[::-1], (.075, .05))
    rope = C.rope_through(start_points=[(-.38, .215), (-.31, .13), (-.20, .01)], cores=[k1, k2],
                          links=[[(0., -.035)]],
                          end_points=[(.16, -.075), (.27, -.11), (.345, -.03), (.33, .09), (.25, .17)])
    fixtures = [post("post_away", (.175, .03)), post("post_front", (-.075, -.085)),
                block("curb", (-.05, .135), (.40, .02, .03), role="curb")]
    regions = {"rope": rect(-.43, .32, .155, .31)}
    birth = dict(steps=[dict(op="settle", max_s=1.5), dict(op="anchor", rope="rope", end="start", links=2),
                        dict(op="tighten", rope="rope", force_n=2.),
                        dict(op="release_all"), dict(op="settle", max_s=3.)])
    oracle = dict(kind="anchored_withdrawal_two_knots_then_layout", rope="rope", order=["k2 (free-end side)", "k1"],
                  pin="over_strand_next_to_crossing", stroke_max_m=.16, layout_region=True, layout_lift_over=["curb"],
                  layout_path={"rope": serpentine((-.38, .215), [(.27, .18), (-.40, .24), (.30, .295)], .03)})
    return _finish("easy", [rope], anchors=[dict(name="rope", anchor=dict(end="start", links=2))],
                   fixtures=fixtures, regions=regions, birth=birth, oracle=oracle,
                   notes="two overhands 15 cm apart in the bimanual lens; clamp in the back region behind the curb")


# ------------------------------------------------------------------------------------------------ medium
def medium():
    """A figure-eight tied around a post (the post stands in the knot's central face) with an overhand stopper on the
    free tail; short standing part (the knot cannot be lifted over the 16 cm post); clutter; region in front-left."""
    f8 = G.place(G.figure_eight_core(.020)[::-1], (0., .04))
    ks = G.place(G.trefoil_core(.012)[::-1], (.165, -.005))
    rope = C.rope_through(start_points=[(-.15, -.075), (-.10, -.035)], cores=[f8, ks],
                          links=[[(.09, -.045)]],
                          end_points=[(.235, -.085), (.33, -.06), (.35, .06), (.27, .16), (.15, .20)])
    fixtures = [post("post_eye", (0., .04), radius=.011), post("post_back", (.095, .105)),
                block("curb", (-.20, .0), (.02, .20, .03), role="curb")]
    regions = {"rope": rect(-.44, .02, -.32, -.03)}
    birth = dict(steps=[dict(op="settle", max_s=1.5), dict(op="anchor", rope="rope", end="start", links=2),
                        dict(op="tighten", rope="rope", force_n=2.),
                        dict(op="release_all"), dict(op="settle", max_s=3.)])
    oracle = dict(kind="stopper_then_post_knot_then_unwrap_and_layout", rope="rope",
                  order=["stopper overhand", "figure-eight around post_eye", "unwrap the post", "layout front-left"],
                  pin="over_strand_next_to_crossing", stroke_max_m=.15, layout_region=True,
                  layout_path={"rope": serpentine((-.15, -.075), [(-.41, -.13), (0., -.18), (-.41, -.23), (0., -.28)],
                                                  .025)})
    return _finish("medium", [rope], anchors=[dict(name="rope", anchor=dict(end="start", links=2))],
                   fixtures=fixtures, regions=regions, birth=birth, oracle=oracle,
                   notes="post_eye through the figure-eight's central face; standing part ~0.1 m from the clamp")


# ------------------------------------------------------------------------------------------------ hard
def hard():
    """Two ropes entangled: A (clamped front-left) carries an overhand whose central face rope B passes through,
    hooked under one of A's strands; B (clamped back-right) carries an overhand stopper near its free end. Clutter;
    each rope has its own region (A: front strip, B: back strip)."""
    # A is authored loose (a = 24 mm) so B can pass its central face with clean crossings; the birth's pull makes it snug
    centre_a = np.array([-.02, .045])
    ka = G.place(G.trefoil_core(.024)[::-1], centre_a)
    rope_a = C.rope_through(start_points=[(-.40, -.235), (-.28, -.18), (-.14, -.09)], cores=[ka], links=[],
                            end_points=[(.10, -.08), (.21, -.12), (.31, -.08), (.34, .02)])
    kb = G.place(G.trefoil_core(.013), (-.27, .06))
    rope_b = C.rope_through(start_points=[(.37, .255), (.24, .20), (.09, .135), (-.012, .05), (-.035, -.005),
                                          (-.10, -.035), (-.17, -.02)],
                            cores=[kb], links=[], end_points=[(-.32, -.04), (-.40, .02), (-.39, .13)])
    # B goes UNDER the last A strand it crosses before passing A's knot centre, OVER every other A strand
    planar = [np.column_stack([rope_a[:, :2], 1e-9 * rope_a[:, 2]]), np.column_stack([rope_b[:, :2], 1e-9 * rope_b[:, 2]])]
    planar = [G.resample(q) for q in planar]
    sb = G.arclength(planar[1])
    s_centre = float(sb[int(np.argmin(np.linalg.norm(planar[1][:, :2] - centre_a, axis=1)))])
    mutual = [c for c in G.projected_crossings(planar) if c["a"][0] != c["b"][0]]
    before = [c["b"][1] for c in mutual if c["b"][1] < s_centre]
    hook_s = max(before) if before else None

    def over(c):
        if c["a"][0] == c["b"][0]:
            return "a" if c["za"] > c["zb"] else "b"
        s_b = c["b"][1] if c["b"][0] == 1 else c["a"][1]
        b_is_b = c["b"][0] == 1
        b_over = hook_s is None or abs(s_b - hook_s) > 1e-6
        return ("b" if b_is_b else "a") if b_over else ("a" if b_is_b else "b")

    fixtures = [post("post_mid", (.13, .015)), post("post_left", (-.16, .14)),
                block("curb", (.02, -.155), (.30, .02, .03), role="curb")]
    regions = {"a": rect(-.44, .30, -.33, -.175), "b": rect(-.30, .42, .17, .32)}
    birth = dict(steps=[dict(op="settle", max_s=1.5), dict(op="anchor", rope="a", end="start", links=2),
                        dict(op="anchor", rope="b", end="start", links=2),
                        dict(op="tighten", rope="a", force_n=2.), dict(op="tighten", rope="b", force_n=1.5),
                        dict(op="release_all"), dict(op="settle", max_s=3.)])
    oracle = dict(kind="two_rope_separation", order=["B stopper", "withdraw B from A's knot", "A overhand",
                                                     "layout A front", "layout B back"],
                  pin="over_strand_next_to_crossing", stroke_max_m=.15, layout_region=True, hook_s_b=hook_s,
                  layout_path={"a": serpentine((-.40, -.235), [(.27, -.20), (-.42, -.30)], .05),
                               "b": serpentine((.37, .255), [(-.27, .20), (.40, .30)], .05)})
    return _finish("hard", [rope_a, rope_b], anchors=[dict(name="a", anchor=dict(end="start", links=2)),
                                                      dict(name="b", anchor=dict(end="start", links=2))],
                   fixtures=fixtures, regions=regions, birth=birth, oracle=oracle, over=over,
                   notes="B threads A's overhand (under one strand at A's central face); each rope: one end clamped, "
                         "one free")


TIERS = dict(very_easy=very_easy, easier=easier, easy=easy, medium=medium, hard=hard)


def build(tier):
    return TIERS[tier]()
