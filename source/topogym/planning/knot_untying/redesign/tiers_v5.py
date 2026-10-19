"""knot_untying redesign ladder v5 (coordinator, 2026-10-10): the v0-v4 ladder with LOOSE knots and the ropes kept
off the ALOHA base footprints.

Why (lab evidence, ITERLOG_<tier> v3 / v4): on the MuJoCo chain (E 5e4 Pa, EI ~1e-5 N m2, friction 0.3) any pull that
tightens a knot cinches it into a ball of ~3 cm (v4 very_easy at 1.6 N: the figure-eight's 4 crossings inside a 3 cm
blob). No station near the crossings then has room for a parallel gripper, and every v4 oracle jammed at the 5 N lab
slip. A "snug" knot on this chain is a jammed knot, which a parallel gripper cannot work; so v5 keeps the knots as
authored (loose, the loops open) and makes the tiers hard by topology (more crossings, nested and stacked knots, two
ropes), an anchored standing end (no end spread, no far-end pin), long tails (withdrawn in several strokes, each with a
pin), clutter on the shortcut lines and a required final layout. The tightness the user asked for is reported as tried
and not supported by the port (a jammed ball is not untieable by the robot's grippers).

Base footprints (tooling 05:03, tiers.BASE_FOOTPRINTS + 5 cm): every rope point stays outside them (the v3 / v4 free
tails ran over the right base at task-local x 0.40-0.54).
"""
from __future__ import annotations

import numpy as np

from . import compose as C
from topogym.tasks.knot_untying.redesign import geometry as G
from .tiers import _finish, block, keep_out_violations, post, rect, serpentine

LOOSE_BIRTH = dict(steps=[dict(op="settle", max_s=1.5), dict(op="anchor", rope="rope", end="start", links=2),
                          dict(op="release_all"), dict(op="settle", max_s=3.)])


def _birth(ropes=("rope",), tighten=None):
    steps = [dict(op="settle", max_s=1.5)]
    steps += [dict(op="anchor", rope=r, end="start", links=2) for r in ropes]
    for r, f in (tighten or {}).items():
        steps.append(dict(op="tighten", rope=r, force_n=f))
    steps += [dict(op="release_all"), dict(op="settle", max_s=3.)]
    return dict(steps=steps)


def very_easy(tighten=None, a=.024):
    """Anchored loose figure-eight (4 crossings) with a 0.55 m free tail hooking through the front-right."""
    core = G.place(G.figure_eight_core(a), (0., .02))
    rope = C.knot_with_paths(core,
                             left_points=[(-.12, -.09), (-.22, -.09), (-.30, -.07)],
                             right_points=[(.10, -.10), (.20, -.17), (.27, -.25), (.20, -.32), (.08, -.31)])
    oracle = dict(kind="anchored_withdrawal", rope="rope", pin="over_strand_next_to_crossing", stroke_max_m=.18,
                  loosen=False, straighten="free_end_against_clamp")
    return _finish("very_easy", [rope], anchors=[dict(name="rope", anchor=dict(end="start", links=2))], fixtures=[],
                   regions={}, birth=_birth(tighten={"rope": tighten} if tighten else None), oracle=oracle,
                   notes=f"v5: loose figure-eight (a {a * 1e3:.0f} mm), standing end clamped at (-0.30, -0.07), 0.55 m "
                         "free tail hooking through the front-right (off the right base)")


def easier(tighten=None, a=.024):
    """Anchored loose figure-eight, clutter on the shortcut lines, a layout region behind a curb."""
    core = G.place(G.figure_eight_core(a), (-.02, .03))
    rope = C.knot_with_paths(core,
                             left_points=[(-.12, -.02), (-.20, .03), (-.27, .11), (-.33, .18)],
                             right_points=[(.09, -.08), (.19, -.15), (.27, -.24), (.18, -.31), (.06, -.29)])
    fixtures = [post("post_away", (.115, -.02)), post("post_front", (-.02, -.13)),
                block("curb", (-.06, .135), (.34, .02, .03), role="curb")]
    regions = {"rope": rect(-.42, .30, .155, .33)}
    oracle = dict(kind="anchored_withdrawal_then_layout", rope="rope", pin="over_strand_next_to_crossing",
                  stroke_max_m=.16, loosen=False, avoid=["post_away", "post_front"], layout_region=True,
                  layout_lift_over=["curb"],
                  layout_path={"rope": serpentine((-.33, .18), [(.28, .18), (-.415, .29)], .055)})
    return _finish("easier", [rope], anchors=[dict(name="rope", anchor=dict(end="start", links=2))],
                   fixtures=fixtures, regions=regions, birth=_birth(tighten={"rope": tighten} if tighten else None),
                   oracle=oracle, notes="v5: loose; clamp behind the curb (inside the region); posts beside the knot "
                                        "on the clamp->knot line and in front")


def easy(tighten=None, a=.020):
    """Stacked overhands (two trefoils in series, 6 crossings), anchored, clutter, region."""
    k1 = G.place(G.trefoil_core(a)[::-1], (-.08, .05))
    k2 = G.place(G.trefoil_core(a)[::-1], (.08, .05))
    rope = C.rope_through(start_points=[(-.38, .215), (-.31, .13), (-.21, .01)], cores=[k1, k2],
                          links=[[(0., -.04)]],
                          end_points=[(.17, -.08), (.25, -.17), (.27, -.27), (.16, -.32), (.04, -.30)])
    fixtures = [post("post_away", (.19, .04)), post("post_front", (-.08, -.09)),
                block("curb", (-.05, .135), (.40, .02, .03), role="curb")]
    regions = {"rope": rect(-.43, .32, .155, .31)}
    oracle = dict(kind="anchored_withdrawal_two_knots_then_layout", rope="rope", order=["k2 (free-end side)", "k1"],
                  pin="over_strand_next_to_crossing", stroke_max_m=.16, layout_region=True, layout_lift_over=["curb"],
                  layout_path={"rope": serpentine((-.38, .215), [(.27, .18), (-.40, .24), (.30, .295)], .03)})
    return _finish("easy", [rope], anchors=[dict(name="rope", anchor=dict(end="start", links=2))],
                   fixtures=fixtures, regions=regions, birth=_birth(tighten={"rope": tighten} if tighten else None),
                   oracle=oracle, notes="v5: two loose overhands 16 cm apart in the bimanual lens; clamp in the back "
                                        "region behind the curb")


def medium(tighten=None):
    """A loose figure-eight tied around a post (the post in its central face) + an overhand stopper on the tail."""
    f8 = G.place(G.figure_eight_core(.022)[::-1], (0., .04))
    ks = G.place(G.trefoil_core(.015)[::-1], (.17, -.02))
    rope = C.rope_through(start_points=[(-.15, -.075), (-.10, -.035)], cores=[f8, ks],
                          links=[[(.085, -.055)]],
                          end_points=[(.25, -.10), (.29, -.19), (.25, -.28), (.13, -.31)])
    fixtures = [post("post_eye", (0., .04), radius=.011), post("post_back", (.10, .12)),
                block("curb", (-.20, .0), (.02, .20, .03), role="curb")]
    regions = {"rope": rect(-.44, .02, -.32, -.03)}
    oracle = dict(kind="stopper_then_post_knot_then_unwrap_and_layout", rope="rope",
                  order=["stopper overhand", "figure-eight around post_eye", "unwrap the post", "layout front-left"],
                  pin="over_strand_next_to_crossing", stroke_max_m=.15, layout_region=True,
                  layout_path={"rope": serpentine((-.15, -.075), [(-.41, -.13), (0., -.18), (-.41, -.23), (0., -.28)],
                                                  .025)})
    return _finish("medium", [rope], anchors=[dict(name="rope", anchor=dict(end="start", links=2))],
                   fixtures=fixtures, regions=regions, birth=_birth(tighten={"rope": tighten} if tighten else None),
                   oracle=oracle, notes="v5: loose; post_eye through the figure-eight's central face; standing part "
                                        "~0.1 m from the clamp")


def hard():
    """Two loose ropes entangled (the v0 hard with both free tails rerouted off the base footprints): A (clamped
    front-left) carries an overhand whose central face rope B passes through, hooked under one of A's strands; B
    (clamped back-right) carries an overhand stopper near its free end. Clutter; each rope has its own region."""
    centre_a = np.array([-.02, .045])
    ka = G.place(G.trefoil_core(.024)[::-1], centre_a)
    rope_a = C.rope_through(start_points=[(-.40, -.235), (-.28, -.18), (-.14, -.09)], cores=[ka], links=[],
                            end_points=[(.10, -.08), (.20, -.13), (.27, -.21), (.21, -.28), (.10, -.29)])
    kb = G.place(G.trefoil_core(.013), (-.24, .07))
    rope_b = C.rope_through(start_points=[(.37, .255), (.24, .20), (.09, .135), (-.012, .05), (-.035, -.005),
                                          (-.10, -.035), (-.16, -.015)],
                            cores=[kb], links=[], end_points=[(-.29, .02), (-.31, .10), (-.26, .19)])
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

    fixtures = [post("post_mid", (.13, .015)), post("post_left", (-.16, .15)),
                block("curb", (.02, -.155), (.30, .02, .03), role="curb")]
    regions = {"a": rect(-.44, .30, -.33, -.175), "b": rect(-.30, .42, .17, .32)}
    oracle = dict(kind="two_rope_separation", order=["B stopper", "withdraw B from A's knot", "A overhand",
                                                     "layout A front", "layout B back"],
                  pin="over_strand_next_to_crossing", stroke_max_m=.15, layout_region=True, hook_s_b=hook_s,
                  layout_path={"a": serpentine((-.40, -.235), [(.27, -.20), (-.42, -.30)], .05),
                               "b": serpentine((.37, .255), [(-.27, .20), (.40, .30)], .05)})
    return _finish("hard", [rope_a, rope_b], anchors=[dict(name="a", anchor=dict(end="start", links=2)),
                                                      dict(name="b", anchor=dict(end="start", links=2))],
                   fixtures=fixtures, regions=regions, birth=_birth(ropes=("a", "b")), oracle=oracle, over=over,
                   notes="v5: loose; B threads A's overhand (under one strand at A's central face); each rope: one end "
                         "clamped, one free; both tails off the base footprints")


TIERS = dict(very_easy=very_easy, easier=easier, easy=easy, medium=medium, hard=hard)


def build(tier, **kw):
    return TIERS[tier](**kw)


def check(scene):
    """Geometry gates of an authored v5 scene: base keep-out, the authored report."""
    bad = {r["name"]: int(len(keep_out_violations(np.asarray(r["centerline"])))) for r in scene["ropes"]}
    rep = scene["authored_report"]
    return dict(base_keep_out=bad, crossings=rep["crossings"], gap_mm=round(rep["min_surface_gap_m"] * 1e3, 2),
                bend_mm=round(rep["min_bend_radius_m"] * 1e3, 1), max_link_angle=round(rep["max_link_angle_rad"], 3),
                lengths=[round(x, 3) for x in rep["lengths_m"]],
                x_range=[[round(float(np.min(np.asarray(r["centerline"])[:, 0])), 3),
                          round(float(np.max(np.asarray(r["centerline"])[:, 0])), 3)] for r in scene["ropes"]])
