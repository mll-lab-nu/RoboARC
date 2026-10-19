"""Reverse oracles (object level) of the redesigned blanket_flatten tiers: the inverse of each tier's generation fold
program, re-planned for the robot's two hands with the assist's patch rule (an exposed 18 mm patch around a material
point, at most 24 pinned vertices per hand, slip at 60 N, follow 10 mm, the tier's strain guard), every grasp point and
hand waypoint inside the arm's top-down reach (reach.py: left x <= -0.03, right x >= -0.24 near the sheet's middle
rows; less toward the far edge). Stages run in order; within a stage both hands move at once.

Executed by topogym.planning.blanket_flatten.redesign.programs.run_program on a FlexSim loaded with the tier's start
state (lanes/blanket_flatten/work/oracle.py). Grasps name material points (re-planned on the live cloth: when the
planned patch is occluded, the nearest exposed particle within `search_m`); targets name the flat layout's position of a
material point (uv), or a displacement from the grasp (by), with heights above the table.
"""
from __future__ import annotations

import math

import numpy as np

from topogym.planning.blanket_flatten.redesign.programs import mirror, foot
from topogym.tasks.blanket_flatten.redesign.tiers import corner


def unfold_path(tip_uv, crease_a, crease_b, *, lift=.05, h_max=.10, land=.025, overshoot=.004, n=5, end_uv=None,
                now_uv=None, spread_u=0.):
    """Turn a folded flap back: the hand holds the flap's tip (material point tip_uv) lying at now_uv (default: the
    mirror image of tip_uv across the crease a-b, i.e. a flap of a sheet otherwise lying flat), lifts it, carries it
    over the crease along a low arc and lays it at end_uv (default: tip_uv moved `overshoot` away from the crease), then
    lowers to `land` above the table. All positions in the flat layout's coordinates (world = centre + uv)."""
    tip = np.asarray(tip_uv, float)
    m = np.asarray(now_uv, float) if now_uv is not None else mirror(tip, crease_a, crease_b)
    c = foot(m, crease_a, crease_b)
    r = float(np.linalg.norm(m - c))
    if end_uv is not None:
        end = np.asarray(end_uv, float)
    else:
        goal = mirror(m, crease_a, crease_b)
        cg = foot(goal, crease_a, crease_b)
        end = goal + overshoot * (goal - cg) / max(np.linalg.norm(goal - cg), 1e-9)
    ce = foot(end, crease_a, crease_b)
    inward = (c - m) / max(r, 1e-9)                 # lift while moving toward the crease: no pull on the flap
    pts = [dict(by=(*(.015 * inward), lift))]
    for k in range(1, n):
        t = k / n
        # rotate about the crease from the current side to the end side (radii interpolated)
        rad = (1 - t) * r + t * float(np.linalg.norm(end - ce))
        base = (1 - t) * c + t * ce
        horiz = base + math.cos(math.pi * t) * rad * (m - c) / max(r, 1e-9)
        z = lift + min(h_max, rad) * math.sin(math.pi * t)
        if spread_u:                  # VBD (10-11): carry the tip outward along u (away from u = 0) to keep the edge taut
            horiz = horiz + np.array([math.copysign(spread_u * t, tip[0]), 0.])
        pts.append(dict(uv=(*horiz, z)))
    if spread_u:
        end = end + np.array([math.copysign(spread_u, tip[0]), 0.])
    pts.append(dict(uv=(*end, land + .03)))
    pts.append(dict(uv=(*end, land)))
    return pts


def unfold(name, hands, *, duration_s=4., search_m=.03, settle_s=0., **kw):
    """One stage: each hand (hand -> (tip_uv, crease_a, crease_b[, options])) turns its flap back."""
    def path(v):
        opts = dict(kw)
        if len(v) > 3:
            opts.update(v[3])
        return unfold_path(v[0], v[1], v[2], **opts)
    return dict(name=name, grasp={h: v[0] for h, v in hands.items()}, search_m=search_m,
                segments=[dict(name="unfold", duration_s=duration_s, hands={h: dict(path=path(v)) for h, v in hands.items()})],
                settle_s=settle_s)


def spread(name, grasps, targets, *, lift=.04, duration_s=3., land=.02, settle_s=0., search_m=.03):
    """Grasp material points and pull them to their flat positions (plus an outward offset) at a low height, then lower:
    the family's spreading move (as the easy route's drag)."""
    segs = [dict(name="lift", duration_s=1., hands={h: dict(by=(0., 0., lift)) for h in grasps}),
            dict(name="pull", duration_s=duration_s, hands={h: dict(uv=(*targets[h], lift)) for h in grasps}),
            dict(name="lower", duration_s=1., hands={h: dict(uv=(*targets[h], land)) for h in grasps})]
    return dict(name=name, grasp=grasps, search_m=search_m, segments=segs, settle_s=settle_s)


def final_spread(prefix="spread", *, out=.004, near=True, far=True, lift=.02, sides=True):
    """The finishing corner pulls (both hands at once): each hand grasps a corner 12 mm in (no hem: a grasp further
    in leaves the strip beyond it folded under, released easier note), lifts 2 cm, pulls it to its flat place moved
    `out` outward along the diagonal, lowers to 1 cm: straightens the corners and edges the page turns leave."""
    st = []
    pairs = []
    if near:
        pairs.append(("near", ((-.3, -.24), (.3, -.24))))
    if far:
        pairs.append(("far", ((-.3, .24), (.3, .24))))
    if sides:
        pairs.append(("sides", ((-.3, 0.), (.3, 0.))))
    for tag, (cl, cr) in pairs:
        gl, gr = (corner(*cl), corner(*cr)) if tag != "sides" else ((-.288, 0.), (.288, 0.))
        tl = (gl[0] + out * np.sign(cl[0]), gl[1] + out * np.sign(cl[1]))
        tr = (gr[0] + out * np.sign(cr[0]), gr[1] + out * np.sign(cr[1]))
        st.append(dict(name=f"{prefix}_{tag}", grasp=dict(left=gl, right=gr), search_m=.03, segments=[
            dict(name="lift", duration_s=1., hands=dict(left=dict(by=(0., 0., lift)), right=dict(by=(0., 0., lift)))),
            dict(name="pull", duration_s=2.5, hands=dict(left=dict(uv=(*tl, lift)), right=dict(uv=(*tr, lift)))),
            dict(name="lower", duration_s=1., hands=dict(left=dict(uv=(*tl, .01)), right=dict(uv=(*tr, .01))))],
            settle_s=.5))
    return st


VE_NL = (corner(-.3, -.24), (0., -.24), (-.3, .06))
VE_FL = (corner(-.3, .24), (-.06, .24), (-.3, 0.))

# ----------------------------------------------------------------------------------------------- easier
# 1. the top layer's turned corner back (right arm); 2. both hands turn the far half back over the 0.60 m crease
# (each hand on its own side of the free edge); 3. the tucked near-left corner: the left hand lifts the crest of the
# tuck 10 cm, carries it outward past its flat place so the tucked flap hangs below, then lowers while moving back
# (the family's medium route move); 4. a spreading pull of the near-left region.
_HALF = lambda uv: (uv[0], -uv[1])                       # material -> its place in the half-folded layer


# ----------------------------------------------------------------------------------------------- measured fix-ups
def fold_layouts(sim, *, max_stages=3, settle_s=3., fold=None, arms=("left", "right"), strict_reach=False):
    """The family's own measured fold layout (robot_oracle.fold_stage, oraclefix 11: the rigid 2-D fit of the rest
    layout onto the single-layer core; the corner with the most particles farther than 3 cm from that fit is laid out
    by the nearer hand: a page turn if it lies over, a lift-drag-lower if it is tucked), repeated up to max_stages
    times while the goal is not met. Converted to FlexSim stages; the hand is re-chosen by reach when the family's
    nearest-hand choice cannot reach the grasp."""
    from types import SimpleNamespace
    from topogym.planning.blanket_flatten.robot_oracle import fold_stage
    from topogym.planning.blanket_flatten.redesign.programs import run_stage
    fold = fold or dict(settle_s=settle_s, deviation_m=.03, corner_radius_m=.15, min_particles=25)
    layout = SimpleNamespace(particle_size_m=.008, length_m=.60, width_m=.48)
    recs = []
    for k in range(max_stages):
        sim.settle(settle_s, min_s=.5)
        ev = sim.evaluate()
        if ev["goal"]:
            break
        tcp = {h: sim.hand_pos(h) for h in ("left", "right")}
        tcp = {h: (p if p[2] < 1.5 else np.array([-.323 if h == "left" else .052, -.019, 1.225])) for h, p in tcp.items()}
        stage, diag = fold_stage(sim.particles(), sim.uv, sim.table_top, layout, fold, tcp)
        if stage is None:
            recs.append(dict(name=f"fold_layout_{k + 1}", ok=False, reason=diag.get("reason"), diagnosis=diag))
            break
        (side, uv), = stage["grasp_uv"].items()
        if side not in arms:                       # a one-arm policy: that arm does every layout
            other = arms[0]
            stage = dict(stage, grasp_uv={other: uv}, segments=tuple(
                dict(seg, to={other: list(seg["to"].values())[0]}) for seg in stage["segments"]))
            side = other
        try:
            from topogym.planning.blanket_flatten.redesign.breakers import Reach
            reach = Reach()
            g = sim.particles()[sim.nearest_particle(uv)]
            if not reach.ok(side, g, .02):
                other = "right" if side == "left" else "left"
                if other in arms and reach.ok(other, g, .02):
                    side = other
                    stage = dict(stage, grasp_uv={side: uv}, segments=tuple(
                        dict(seg, to={side: list(seg["to"].values())[0]}) for seg in stage["segments"]))
                elif strict_reach:          # a reach-enforced breaker: the arm cannot do this layout
                    recs.append(dict(name=f"fold_layout_{k + 1}", ok=False, reason="out of the arm's reach: skipped"))
                    break
            if strict_reach and not all(reach.ok(side, np.asarray(seg["to"][side], float), float(seg["to"][side][2]))
                                        for seg in stage["segments"]):          # "to" heights are above the table
                recs.append(dict(name=f"fold_layout_{k + 1}", ok=False, reason="a waypoint out of the arm's reach: skipped"))
                break
        except (FileNotFoundError, ImportError):     # the lab's reach map (breakers.Reach) is lab-only, not shipped
            pass
        st = dict(name=f"fold_layout_{k + 1}", grasp={side: uv}, search_m=.03,
                  segments=[dict(name=seg["name"], duration_s=seg["duration_s"], hands={side: dict(to=seg["to"][side])})
                            for seg in stage["segments"]], settle_s=.5)
        rec = run_stage(sim, st)
        rec["diagnosis"] = {k2: diag.get(k2) for k2 in ("corner_uv", "corner_folded_particles", "over", "side", "grasp_uv")}
        recs.append(rec)
    return recs


def with_fixups(program, max_stages=3):
    """A reverse program followed by the measured fold layouts (a callable for oracle.py)."""
    def run(sim, tier):
        from topogym.planning.blanket_flatten.redesign.programs import run_stage
        recs = [run_stage(sim, st) for st in program]
        return recs + fold_layouts(sim, max_stages=max_stages)
    run.program = program
    return run


_TOP_R = (corner(.3, .24), (.14, -.24), (.3, -.08))
_TOP_L = (corner(-.3, .24), (-.14, -.24), (-.3, -.08))


def _top_corner(hand, spec, end_uv):
    tip, a, b = spec
    return (tip, a, b, dict(now_uv=mirror(_HALF(tip), a, b), end_uv=end_uv))


def far_half_back(u=.22, search_m=.03):
    """Both hands turn the far half back over the 0.60 m crease, grasping its free edge at u = -u / +u."""
    st = unfold("far_half_back", dict(
        left=((-u, .228), (-.3, 0.), (.3, 0.), dict(now_uv=(-u, -.228), end_uv=(-u, .238))),
        right=((u, .228), (-.3, 0.), (.3, 0.), dict(now_uv=(u, -.228), end_uv=(u, .238)))), duration_s=5., h_max=.12,
        settle_s=1.)
    st["search_m"] = search_m
    return st


def undo_double_corner(tag, hand, cu, cv, small=.12, big=.24):
    """Two page turns: the big flap by its free edge (the small fold's crease, midpoint), then the small fold by its
    corner. Positions in the flat layout's coordinates."""
    su, sv = np.sign(cu), np.sign(cv)
    big_a, big_b = (cu - su * big, cv), (cu, cv - sv * big)
    sm_a, sm_b = (cu - su * small, cv), (cu, cv - sv * small)
    edge = ((sm_a[0] + sm_b[0]) / 2 - su * .012, (sm_a[1] + sm_b[1]) / 2 - sv * .012)   # just inside the small crease
    return [
        unfold(f"big_{tag}", {hand: (edge, big_a, big_b)}, duration_s=4., settle_s=.5, overshoot=-.01),
        unfold(f"small_{tag}", {hand: (corner(cu, cv), sm_a, sm_b)}, duration_s=3.5, settle_s=.5, overshoot=-.01),
    ]


VERY_EASY4 = [
    unfold("flap_FR", dict(right=(corner(.3, .24), (.06, .24), (.3, 0.))), duration_s=4., settle_s=.5, overshoot=-.015),
    unfold("top_flap_NL", dict(left=VE_NL), duration_s=4.5, settle_s=.5, overshoot=-.015),
    unfold("under_flap_FL", dict(left=VE_FL), duration_s=4., settle_s=.5, overshoot=-.015),
    *undo_double_corner("NR", "right", .3, -.24),
    *final_spread(),
]


EASIER4 = [
    unfold("top_corner", dict(right=_top_corner("right", _TOP_R, (.296, -.236))), settle_s=.5),
    far_half_back(.22, .04),
    *undo_double_corner("NL", "left", -.3, -.24, small=.10, big=.20),
    *final_spread(),
]
EASY4 = [
    unfold("top_corners", dict(right=_top_corner("right", _TOP_R, (.296, -.236)),
                               left=_top_corner("left", _TOP_L, (-.296, -.236))), settle_s=.5),
    far_half_back(.12, .04),
    *undo_double_corner("NL", "left", -.3, -.24, small=.10, big=.20),
    *undo_double_corner("NR", "right", .3, -.24, small=.10, big=.20),
    *final_spread(),
]
MEDIUM4 = [
    far_half_back(.10, .04),
    unfold("far_corners_back", dict(left=(corner(-.3, .24), (-.14, .24), (-.3, .08)),
                                    right=(corner(.3, .24), (.14, .24), (.3, .08))), duration_s=4., settle_s=.5,
           overshoot=-.01),
    *undo_double_corner("NL", "left", -.3, -.24, small=.10, big=.20),
    *undo_double_corner("NR", "right", .3, -.24, small=.10, big=.20),
    *final_spread(),
]
HARD4 = [
    far_half_back(.10, .04),
    *undo_double_corner("FL", "left", -.3, .24, small=.08, big=.16),
    *undo_double_corner("FR", "right", .3, .24, small=.08, big=.16),
    *undo_double_corner("NL", "left", -.3, -.24, small=.10, big=.20),
    *undo_double_corner("NR", "right", .3, -.24, small=.10, big=.20),
    *final_spread(),
]
# the v4 ladder with the family's measured fold layouts after the program (as the v1 very_easy fixup that succeeded):
# very_easy4's plain reverse left the far-left under flap occluded behind the returned near-left flap (0.731)
VERY_EASY4_B = [dict(st, search_m=.05) if st.get("name") == "under_flap_FL" else st for st in VERY_EASY4]
# easy4 / medium4 r1 (DIAGEXACT): the half came back with its far corners folded under (0.79 / 0.78). b: easy4 turns
# the half by its edge at +-0.20 (the top corners are back by then) with a 5 cm patch search; medium4 pulls the far
# edge out flat (both hands, 3 cm outward at 1.5 cm) before the far flaps, which are then searched within 5 cm
EASY4_B = [st if st.get("name") != "far_half_back" else far_half_back(.20, .05) for st in EASY4]
_FAR_EDGE_SPREAD = spread("far_edge_spread", dict(left=(-.10, .228), right=(.10, .228)),
                          dict(left=(-.10, .258), right=(.10, .258)), lift=.015, duration_s=2., settle_s=.5)
MEDIUM4_B = [MEDIUM4[0], _FAR_EDGE_SPREAD, dict(MEDIUM4[1], search_m=.05)] + MEDIUM4[2:]
HARD4_B = [HARD4[0], _FAR_EDGE_SPREAD] + [dict(st, search_m=.05) if st.get("name", "").startswith(("big_F", "small_F")) else st
                                         for st in HARD4[1:]]
# very_easy4 fixup r1: 0.935 (the sheet flat, its near-left edge curled in). C: a second round of corner / edge spreads
# pulling 1 cm outward; D: the spreads pull 1.2 cm outward instead of 0.4 cm (one round)
VERY_EASY4_C = VERY_EASY4_B + final_spread(prefix="spread2", out=.010)
VERY_EASY4_D = [st for st in VERY_EASY4_B if not st.get("name", "").startswith("spread")] + final_spread(out=.012)
# easy4c (laptop, 12:30): 0.824 (the half came back with a ~4 cm Z-fold at its crease, far edge at y 0.25 instead of
# 0.29). E / F: the half turned by its edge at u = +-0.22 / +-0.26 (easier4's +-0.22 worked there), a 5 cm patch
# search, then the double folds and two rounds of finishing spreads
EASY4_E = [EASY4[0], far_half_back(.22, .05)] + EASY4[2:] + final_spread(prefix="spread2", out=.010)
EASY4_F = [EASY4[0], far_half_back(.26, .05)] + EASY4[2:] + final_spread(prefix="spread2", out=.010)
# easy4e (laptop, 12:58): 0.944, the half back in place (far edge at y 0.288) but small_NL missed its patch (occluded):
# H = E with a 5 cm patch search at the near double folds (hard4b's fix at the far ones)
EASY4_H = [dict(st, search_m=.05) if st.get("name", "").startswith(("big_N", "small_N")) else st for st in EASY4_E]
# medium4c (laptop, 13:15): 0.929, the sheet dragged 6 cm towards +y (the 3 cm far pull ran, the near pull missed an
# occluded patch). D: the second round pulls 1.5 cm outward, near and far, each with a 5 cm patch search
MEDIUM4_D = MEDIUM4_B + [dict(st, search_m=.05) for st in final_spread(prefix="spread2", out=.015, sides=False)]
# medium4d (laptop, 13:55): 0.920, the sheet flat but skewed (~5 deg) and 4-10 cm too far towards +y after the
# finishing pulls. A measured re-centring: the four corners' displacement from their flat places; while the mean
# exceeds 1.2 cm, the two corners on the side the sheet must come back from are dragged to their flat places (a low
# two-hand drag, 5 cm patch search), up to two rounds, then the family's measured fold layout
def recenter(sim, rounds=2, tol=.012):
    from topogym.planning.blanket_flatten.redesign.programs import run_stage
    from topogym.tasks.blanket_flatten.redesign.tiers import W
    recs = []
    for k in range(rounds):
        sim.settle(2., min_s=.5)
        P = sim.particles()
        disp = {}
        for cu in (-.3, .3):
            for cv in (-.24, .24):
                uv = corner(cu, cv)
                disp[(cu, cv)] = P[sim.nearest_particle(uv), :2] - np.asarray(W(*uv))
        mean = np.mean(list(disp.values()), axis=0)
        if float(np.linalg.norm(mean)) < tol:
            recs.append(dict(name=f"recenter_{k + 1}", ok=True, reason=f"centred (mean shift {np.round(mean, 3).tolist()})"))
            break
        cv = -.24 if mean[1] > 0 else .24          # too far +y: drag the near pair back (and the far pair otherwise)
        g = dict(left=corner(-.3, cv), right=corner(.3, cv))
        st = spread(f"recenter_{k + 1}", g, dict(g), lift=.02, duration_s=2.5, settle_s=.5, search_m=.05)
        rec = run_stage(sim, st)
        rec["shift_before"] = np.round(mean, 3).tolist()
        recs.append(rec)
    return recs


def with_recenter_fixups(program, max_stages=3):
    """A reverse program, the measured re-centring, then the family's measured fold layouts (a callable)."""
    def run(sim, tier):
        from topogym.planning.blanket_flatten.redesign.programs import run_stage
        recs = [run_stage(sim, st) for st in program]
        recs += recenter(sim)
        return recs + fold_layouts(sim, max_stages=max_stages)
    run.program = program
    return run


# finish v2 (16:15): the crease stretches did not lower the remaining ridges (17-20 mm stayed through 4 rounds: they
# are thin double layers along a crease, not single-layer bumps, and a 1.5 cm pull at 1.2 cm height cannot open them).
# The re-lay: the two corners of the sheet's side across the ridge are lifted 8 cm (the side hangs from them and the
# tuck at the crease lets go under gravity), carried to their flat places 8 mm outward at 3 cm, lowered and released.
def finish_relay(sim, rounds=3, ridge_mm=6., min_particles=8, lift=.08, out=.008, diag="far"):
    from topogym.planning.blanket_flatten.redesign.programs import run_stages
    from topogym.tasks.blanket_flatten.redesign.tiers import W
    fast = bool(getattr(sim, "speedup_1013", False))     # the route's 2026-10-13 speed-up (redesign_routes)
    recs = []
    for k in range(rounds):
        sim.settle(2., min_s=.5)
        r = fold_layouts(sim, max_stages=1, fold=dict(settle_s=1., deviation_m=.02, corner_radius_m=.15,
                                                       min_particles=min_particles))
        if r and r[-1].get("ok") and r[-1].get("grasps"):
            for x in r:
                x["name"] = f"relay_{k + 1}_{x.get('name')}"
            recs += r
            continue
        P = sim.particles()
        h = P[:, 2] - sim.table_top - sim.sphere_r
        ridge = h > ridge_mm / 1e3
        if ridge.sum() < 10:
            recs.append(dict(name=f"relay_{k + 1}", ok=True, reason="flat (no ridge, no flipped corner)"))
            break
        Q = P[ridge, :2]
        c = Q.mean(0)
        _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
        n = np.array([-vt[0][1], vt[0][0]])
        # the side across the ridge with the least material, by the corners' flat places
        cs = [(cu, cv) for cu in (-.3, .3) for cv in (-.24, .24)]
        side = {cc: float((np.asarray(W(*corner(*cc))) - c) @ n) for cc in cs}
        pos = [cc for cc in cs if side[cc] > 0]
        neg = [cc for cc in cs if side[cc] <= 0]
        pick = pos if (len(pos) == 2 and (len(neg) != 2 or sum(abs(side[x]) for x in pos) <= sum(abs(side[x]) for x in neg))) else neg
        if len(pick) != 2 and diag == "far":
            # the recorded _fin2 runs' fallback (kept so they reproduce): the two corners farthest from the ridge, one
            # on each side of a diagonal ridge. hard4b_fin2's relay_2 took it and ended well (0.993); very_easy4c_fin2
            # and medium4e_fin2 were folded along the diagonal by it (0.96 -> 0.75, 0.97 -> 0.89)
            pick = sorted(cs, key=lambda cc: -abs(side[cc]))[:2]
        elif len(pick) != 2:
            # diag="small" (18:10, the _fin3 variants): the side with fewer corners, one corner by its own arm
            small = pos if len(pos) < len(neg) else neg
            if len(small) == 1:
                pick = small
            elif not small:
                pick = sorted(cs, key=lambda cc: abs(side[cc]))[:2]
            else:
                pick = sorted(small, key=lambda cc: -abs(side[cc]))[:2]
        pick = sorted(pick, key=lambda cc: cc[0])            # left hand on the left corner
        if len(pick) == 1:
            arm = "left" if pick[0][0] < 0 else "right"
            groups = [{arm: corner(*pick[0])}]
            corners_of = [pick]
        elif np.sign(pick[0][0]) == np.sign(pick[1][0]):
            # a left or right side (both corners at the same u): that side's arm re-lays them one after the other
            # (easy4f_fin2: the right hand was sent to the near-left corner, out of its reach)
            arm = "left" if pick[0][0] < 0 else "right"
            groups = [{arm: corner(*cc)} for cc in pick]
            corners_of = [[cc] for cc in pick]
        else:
            groups = [dict(left=corner(*pick[0]), right=corner(*pick[1]))]
            corners_of = [pick]
        sts = []
        for gi, (g, ccs) in enumerate(zip(groups, corners_of)):
            t = {hh: (uv[0] + out * np.sign(cc[0]), uv[1] + out * np.sign(cc[1])) for (hh, uv), cc in zip(g.items(), ccs)}
            sts.append(dict(name=f"relay_{k + 1}_side{'' if len(groups) == 1 else '_' + str(gi + 1)}", grasp=g, search_m=.03, segments=[
                # 2026-10-13 speed-up (user-approved, the routes recorded after it): the relay legs 30 %
                # faster (1.5 / 2.5 / 1.2 s otherwise)
                dict(name="lift", duration_s=1.05 if fast else 1.5, hands={hh: dict(by=(0., 0., lift)) for hh in g}),
                dict(name="carry", duration_s=1.75 if fast else 2.5, hands={hh: dict(uv=(*t[hh], .03)) for hh in g}),
                dict(name="lower", duration_s=.85 if fast else 1.2, hands={hh: dict(uv=(*t[hh], .008)) for hh in g})], settle_s=.5))
        # a same-side pair is fixed by this one measurement: one robot planner call (run_stages), the lab unchanged
        side_recs = run_stages(sim, sts)
        recs += side_recs[:-1]
        rec = side_recs[-1]
        rec["ridge"] = dict(particles=int(ridge.sum()), max_mm=round(1e3 * float(h.max()), 1), normal=np.round(n, 3).tolist(),
                            corners=[list(cc) for cc in pick])
        recs.append(rec)
    return recs


def with_finish2(run_or_program, diag="far"):
    def run(sim, tier):
        from topogym.planning.blanket_flatten.redesign.programs import run_stage
        recs = run_or_program(sim, tier) if callable(run_or_program) else [run_stage(sim, st) for st in run_or_program]
        return recs + finish_relay(sim, diag=diag)
    run.program = getattr(run_or_program, "program", run_or_program)
    return run


# far hem (2026-10-13, hard4 on d38cdf4bfb, coordinator-approved route change): hard4b_rc_fin2 ended at coverage 0.94 of
# the tier's 0.95 with the far edge 32 mm high; its retreat states held the far hem (the material row v = v_max, the
# whole 60 cm) folded back on top of the sheet, 3-6 cm wide (110-130 cm^2 doubled), which the finishing relays never
# opened. The hem is measured on the live cloth and laid back out before the relays and once after them.
def hem_unfold(sim, name="hem", rounds=2, min_w=.010, min_frac=.15, extra=.008):
    """The far hem folded back on top of the sheet: per arm the column (|u| .06-.27 on its side) with the widest flap
    (crease y = the farthest material of the column's far 12 cm, flap width = crease y - edge y); the edge row grasped
    there (one grid spacing in), lifted, carried to its unfolded place (mirrored over the crease, `extra` outward) and
    lowered. Skipped when < min_frac of the edge row lies on top of material or no flap is >= min_w wide."""
    from scipy.spatial import cKDTree
    from topogym.planning.blanket_flatten.redesign.programs import run_stage
    from topogym.tasks.blanket_flatten.redesign.tiers import GRIP_IN
    recs = []
    for k in range(rounds):
        sim.settle(2., min_s=.5)
        P = sim.particles()
        uv = np.asarray(sim.uv)
        vmax = float(uv[:, 1].max())
        edge = np.flatnonzero(uv[:, 1] >= vmax - 1e-3)
        tree = cKDTree(P[:, :2])
        on_top = np.array([np.any(P[nb, 2] < P[i, 2] - .004) for i, nb in zip(edge, tree.query_ball_point(P[edge, :2], .005))])
        picks = {}
        for hand, sgn in (("left", -1.), ("right", 1.)):
            best = None
            for u0 in sgn * np.arange(.06, .2701, .03):
                col = np.abs(uv[:, 0] - u0) <= .015
                e = col & (uv[:, 1] >= vmax - 1e-3)
                body = col & (uv[:, 1] >= vmax - .12)
                if not e.any():
                    continue
                y_edge, y_crease = float(P[e, 1].mean()), float(P[body, 1].max())
                if best is None or y_crease - y_edge > best[1]:
                    best = (float(u0), y_crease - y_edge, y_edge, y_crease)
            if best is not None and best[1] >= min_w:
                picks[hand] = best
        meas = dict(edge_on_top=round(float(on_top.mean()), 3), flaps={h: [round(x, 4) for x in b] for h, b in picks.items()})
        if on_top.mean() < min_frac or not picks:
            recs.append(dict(name=f"{name}_{k + 1}", ok=True, reason="no far hem flap", hem=meas))
            break
        g = {h: (b[0], vmax - GRIP_IN) for h, b in picks.items()}
        tgt = {}
        for h, b in picks.items():
            p = P[sim.nearest_particle(g[h])]
            tgt[h] = (float(p[0]), float(b[3] + (b[3] - p[1]) + extra))
        lift = float(np.clip(1.2 * max(b[1] for b in picks.values()), .03, .06))
        st = dict(name=f"{name}_{k + 1}", grasp=g, search_m=.02, segments=[
            dict(name="lift", duration_s=.8, hands={h: dict(by=(0., 0., lift)) for h in g}),
            dict(name="carry", duration_s=1.5, hands={h: dict(to=(*tgt[h], lift)) for h in g}),
            dict(name="lower", duration_s=.85, hands={h: dict(to=(*tgt[h], .008)) for h in g})], settle_s=.5)
        r = run_stage(sim, st)
        r["hem"] = dict(meas, targets={h: [round(x, 4) for x in t] for h, t in tgt.items()}, lift=round(lift, 3))
        recs.append(r)
    return recs


def with_hem_finish(run_or_program, diag="far"):
    """The program, the far-hem unfold (up to 2 rounds), the finishing relays, a final far-hem check (1 round)."""
    def run(sim, tier):
        from topogym.planning.blanket_flatten.redesign.programs import run_stage
        recs = run_or_program(sim, tier) if callable(run_or_program) else [run_stage(sim, st) for st in run_or_program]
        recs += hem_unfold(sim, "hem", rounds=2)
        recs += finish_relay(sim, diag=diag)
        return recs + hem_unfold(sim, "hem_end", rounds=1)
    run.program = getattr(run_or_program, "program", run_or_program)
    return run


# the entries the product uses (lane A 2026-10-11: the lab's other variants stay in the redesign sandbox)
ROUTES_FIXUP = {}
ROUTES_FIXUP['easier4'] = with_fixups(EASIER4)
ROUTES_FIXUP['hard4b'] = with_fixups(HARD4_B)
ROUTES_FIXUP['very_easy4c'] = with_fixups(VERY_EASY4_C)
ROUTES_FIXUP['easy4f'] = with_fixups(EASY4_F)
ROUTES_FIXUP['medium4e'] = with_recenter_fixups(MEDIUM4_D)
ROUTES_FIXUP['easier4_fin2'] = with_finish2(EASIER4)
ROUTES_FIXUP['hard4b_fin2'] = with_finish2(ROUTES_FIXUP["hard4b"])
ROUTES_FIXUP['hard4b_hem'] = with_hem_finish(with_recenter_fixups(HARD4_B))
