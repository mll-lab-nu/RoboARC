"""blanket_fold oracles (coordinator 2026-10-10): the robot's two hands (virtual hands under the assist's patch rule,
as in blanket_flatten's FlexSim lab) fold toward the tier's target with page turns; measured corrections of the
exposed named points against the target (a camera-guided robot would do the same); the flatten tiers' programs for
the flatten-then-fold starts. Every program is a callable run(sim, tier, target_particles) -> records.
Evidence for the motions: review/blanket_fold/FEASIBILITY.md (fold lab 10-10)."""
from __future__ import annotations

import numpy as np

from topogym.planning.blanket_flatten.redesign.reverse import unfold
from topogym.planning.blanket_flatten.redesign.programs import run_stage, run_stages
from topogym.tasks.blanket_flatten.redesign.tiers import CX, corner

CF_L, CF_R, CN_L, CN_R = corner(-.3, .24), corner(.3, .24), corner(-.3, -.24), corner(.3, -.24)


def edge_turn(name, tips, crease_a, crease_b, *, duration_s=5., h_max=.12, settle_s=1., search_m=.03, overshoot=.004,
              land=None):
    """A forward fold: each hand holds its material tip where it lies now (flat-layout coords) and turns it over the
    crease a-b onto its mirror image. land: the release height above the table (default unfold_path's 25 mm); a dict
    hand -> height sets it per hand."""
    hands = {h: (tip, crease_a, crease_b, dict(now_uv=now, **({} if land is None else
                                                             dict(land=land[h] if isinstance(land, dict) else land))))
             for h, (tip, now) in tips.items()}
    return unfold(name, hands, duration_s=duration_s, h_max=h_max, settle_s=settle_s, search_m=search_m,
                  overshoot=overshoot)


def _land(sim, T, uv, default=.025):
    """VBD (10-11 15:20): release a turned flap 4 mm above its target height (the flex's 25 mm let a dog-ear drop and
    set standing: b_easier's flap 20-28 mm over the target)."""
    if not _vbd(sim):
        return default
    return float(T[sim.nearest_particle(uv), 2] - sim.table_top) + .004


def _vbd(sim):
    """VBD lab (10-11 08:50): the measured drags run 2x slower (layer-on-layer drags over-stretch next to the hand on
    the 20-iteration solve, lane S issue (a)) and the folded-edge grips search 4 cm (the bend patch can test occluded on
    CPU, issue (b)). The flex paths keep their recorded timings."""
    return getattr(sim, "backend", "flex") == "vbd"


def _h(sim, uv, h):
    """A carry height h above the table (the flex oracles' convention) becomes, on VBD, h above the grasped material
    point's current height (10-11 10:15: layers stand 7 mm apart and free edges curl to 15-25 mm, so an absolute 12 mm
    "lift" pushed the pinned patch down into the layers below: 54 N, strain 1.5, the drags' guard releases)."""
    if not _vbd(sim):
        return h
    i = sim.nearest_particle(uv)
    return max(h, float(sim.particles()[i, 2] - sim.table_top) + h)


def _low(sim, uv, h_flex, z_target=None):
    """Release height on VBD: 4 mm above the higher of the grasped point's starting height and its target height (an
    absolute low release pressed the patch into a stack standing higher); the flex value elsewhere."""
    if not _vbd(sim):
        return h_flex
    z0 = float(sim.particles()[sim.nearest_particle(uv), 2] - sim.table_top)
    return max(z0, (z_target - sim.table_top) if z_target is not None else 0.) + .004


def _drag_s(sim):
    import os
    # 2026-10-13 speed-up (user-approved): 1.5 s on VBD as on the flex (the 3 s was a lab-solver workaround), for the
    # routes recorded after it (redesign_routes.FAST_FOLD: the sim's speedup_1013); the others keep 3 s
    return float(os.environ.get("RD_FOLD_DRAG_S", 1.5 if getattr(sim, "speedup_1013", False) else 3.)) if _vbd(sim) else 1.5


def correct_named(sim, T, rounds=2, tol=.012, names=((-.3, -.24), (-.3, .24), (.3, -.24), (.3, .24))):
    """Each exposed named corner whose xy error exceeds tol is dragged to its target (lift 1.2 cm, drag, lower)."""
    recs = []
    for k in range(rounds):
        sim.settle(2., min_s=.5)
        P = sim.particles()
        todo = []
        for cu, cv in names:
            uv = corner(cu, cv)
            i = sim.nearest_particle(uv)
            err = float(np.linalg.norm(P[i, :2] - T[i, :2]))
            if err > tol and sim.exposed(sim.patch(i, P), P):
                todo.append((err, uv, i))
        if not todo:
            recs.append(dict(name=f"correct_{k + 1}", ok=True, reason="exposed named points within tolerance"))
            break
        todo = sorted(todo, reverse=True)

        def build(uv, i, k=k):
            P = sim.particles()
            hand = "left" if P[i, 0] < CX else "right"
            tx, ty = T[i, 0], T[i, 1]
            return dict(name=f"correct_{k + 1}_{uv[0]:+.2f}_{uv[1]:+.2f}", grasp={hand: uv}, search_m=0., segments=[
                dict(name="lift", duration_s=.6, hands={hand: dict(by=(0., 0., _h(sim, uv, .012)))}),
                dict(name="drag", duration_s=_drag_s(sim), hands={hand: dict(to=(tx, ty, _h(sim, uv, .012)))}),
                dict(name="lower", duration_s=.6, hands={hand: dict(to=(tx, ty, _low(sim, uv, T[i, 2] - sim.table_top + .002, T[i, 2])))})],
                        settle_s=.3)
        # the round's corners are fixed by this one measurement: one robot planner call (run_stages), the lab unchanged
        for (err, _, _), rec in zip(todo, run_stages(sim, [lambda uv=uv, i=i: build(uv, i) for _, uv, i in todo])):
            rec["err_before_mm"] = round(1e3 * err, 1)
            recs.append(rec)
    return recs


def recentre_stack(sim, T, grips=((-.25, 0.), (.25, 0.)), tol=.005, lift=.010, name="recentre"):
    """VBD (10-10 23:40): the realistic sheet weighs 0.5 kg/m^2 (the flex 4.0), so a page turn drags the light base
    layer toward the landing side (very_easy on VBD: the whole stack 2 cm toward the robot). Measured re-centring: the
    offset of the stack's mean from the target's; both hands hold the folded edge (the crease's material, which pins
    both layers of the bend) near its ends and pull the stack by that offset (tension when the stack lies short of
    its crease side), then lower. No-op within tol."""
    sim.settle(2., min_s=.5)
    P = sim.particles()
    off = T[:, :2].mean(0) - P[:, :2].mean(0)
    if float(np.linalg.norm(off)) <= tol:
        return [dict(name=name, ok=True, reason=f"offset {1e3 * float(np.linalg.norm(off)):.1f} mm within tolerance")]
    g, t = {}, {}
    for hand, uv in zip(("left", "right"), grips):
        i = sim.nearest_particle(uv)
        g[hand] = uv
        t[hand] = (float(P[i, 0] + off[0]), float(P[i, 1] + off[1]))
    st = dict(name=name, grasp=g, search_m=.04 if _vbd(sim) else .015, segments=[
        dict(name="lift", duration_s=.6, hands={h: dict(by=(0., 0., _h(sim, g[h], lift))) for h in g}),
        dict(name="pull", duration_s=_drag_s(sim), hands={h: dict(to=(*t[h], _h(sim, g[h], lift) + .004)) for h in g}),
        dict(name="lower", duration_s=.6, hands={h: dict(to=(*t[h], _low(sim, g[h], .006))) for h in g})], settle_s=.5)
    rec = run_stage(sim, st)
    rec["offset_mm"] = [round(1e3 * float(x), 1) for x in off]
    return [rec]


def half_by_corners(name="half", low=True, overshoot=.004, land=None):
    """The far half onto the near half, both hands at the far corners; a flat 6 cm arc over 7 s (the lab's best:
    placement 16 mm, stacking 0.93) or the standard 12 cm arc over 5 s. land: release height (callable allowed)."""
    return edge_turn(name, dict(left=(CF_L, CF_L), right=(CF_R, CF_R)), (-.3, 0.), (.3, 0.),
                     h_max=.06 if low else .12, duration_s=7. if low else 5., overshoot=overshoot,
                     land=land() if callable(land) else land)


def oracle_very_easy_rc(sim, tier, T):
    """very_easy with the measured re-centring of the stack (VBD's light sheet) before the corner corrections; on VBD
    (15:20) the half fold releases 4 mm above the target and a final re-centring puts the whole stack on the target's
    mean (the record sat 4 cm off after the corrections)."""
    import os
    v2 = _vbd(sim) and os.environ.get("RD_FOLD_V2", "1") == "1"
    recs = [run_stage(sim, half_by_corners(land=(lambda: _land(sim, T, CF_L)) if v2 else None))] + recentre_stack(sim, T)
    recs += correct_named(sim, T, rounds=2)
    if v2:
        recs += recentre_stack(sim, T, name="recentre_final") + correct_named(sim, T, rounds=1)
    return recs


def place_flat(sim, rounds=2, tol=.008, names=((-.3, .24), (.3, .24), (-.3, -.24), (.3, -.24))):
    """Measured placement of a flattened sheet before a fold (18:10, easy: the flattened sheet lay 2 cm off and rotated
    1 deg; the bottom layer's corners are covered after the fold, so they must be placed first): each exposed corner
    more than `tol` from its flat place is dragged there at 1.2 cm and lowered."""
    from topogym.tasks.blanket_flatten.redesign.tiers import W
    recs = []
    for k in range(rounds):
        sim.settle(2., min_s=.5)
        P = sim.particles()
        todo = []
        for cu, cv in names:
            uv = corner(cu, cv)
            i = sim.nearest_particle(uv)
            t = np.asarray(W(*uv), float)
            err = float(np.linalg.norm(P[i, :2] - t))
            if err > tol and sim.exposed(sim.patch(i, P), P):
                todo.append((err, uv, t))
        if not todo:
            recs.append(dict(name=f"place_flat_{k + 1}", ok=True, reason="corners within tolerance"))
            break
        todo = sorted(todo, reverse=True)

        def build(uv, t, k=k):
            hand = "left" if t[0] < CX else "right"
            return dict(name=f"place_flat_{k + 1}_{uv[0]:+.2f}_{uv[1]:+.2f}", grasp={hand: uv}, search_m=0., segments=[
                dict(name="lift", duration_s=.6, hands={hand: dict(by=(0., 0., _h(sim, uv, .012)))}),
                dict(name="drag", duration_s=_drag_s(sim), hands={hand: dict(to=(t[0], t[1], _h(sim, uv, .012)))}),
                dict(name="lower", duration_s=.6, hands={hand: dict(to=(t[0], t[1], _low(sim, uv, .004)))})], settle_s=.3)
        # one measurement fixes the round's corners: one robot planner call (run_stages), the lab unchanged
        for (err, _, _), rec in zip(todo, run_stages(sim, [lambda uv=uv, t=t: build(uv, t) for _, uv, t in todo])):
            rec["err_before_mm"] = round(1e3 * err, 1)
            recs.append(rec)
    return recs


# ------------------------------------------------------------------------------------------ ladder B oracles (03:45)
def _layer_frame(sim, T, P, tip, flap_r=.20):
    """The top layer's live frame: R, t (2-D rigid, live ~ R @ target + t) fitted to its particles' live and target xy,
    leaving out the material within flap_r of the turned corner `tip` (material uv)."""
    from topogym.tasks.blanket_fold.redesign.tiers import TOP_LAYER
    m = TOP_LAYER(sim.uv) & (np.linalg.norm(sim.uv - np.asarray(tip, float), axis=1) > flap_r)
    A, B = np.asarray(T)[m, :2], np.asarray(P)[m, :2]
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1., d]) @ U.T
    return R, cb - R @ ca


def lay_bumps(sim, T, rounds=2, min_dz=.012):
    """10-12 (robot, two-way f1): after a dog-ear the sheet's free edge next to the crease's end stood 33-52 mm high (the
    corner fold pulls it up; dz failed in 6 of 8 neighbouring releases). Up to `rounds` measured stages: the exposed
    particle highest above its target is re-grasped, placed on its target xy and lowered to 3 mm above its target
    height. No-op below min_dz."""
    recs = []
    for k in range(rounds):
        P = sim.particles()
        dz = P[:, 2] - T[:, 2]
        pick = None
        for i in np.argsort(-dz)[:400]:
            if dz[i] < min_dz:
                break
            if sim.exposed(sim.patch(int(i), P), P):
                pick = int(i)
                break
        if pick is None:
            recs.append(dict(name=f"bump_{k + 1}", ok=True, reason="no exposed bump above min_dz"))
            break
        g = tuple(float(x) for x in sim.uv[pick])
        hand = "left" if P[pick, 0] < CX else "right"
        tx, ty, h0 = float(T[pick, 0]), float(T[pick, 1]), _h(sim, g, .008)
        rec = run_stage(sim, dict(name=f"bump_{k + 1}", grasp={hand: g}, search_m=0., segments=[
            dict(name="lift", duration_s=.4, hands={hand: dict(by=(0., 0., h0))}),
            dict(name="place", duration_s=1.2, hands={hand: dict(to=(tx, ty, h0))}),
            dict(name="lower", duration_s=.6, hands={hand: dict(to=(tx, ty, float(T[pick, 2] - sim.table_top) + .003))})],
            settle_s=.3))
        rec["dz_before_mm"] = round(1e3 * float(dz[pick]), 1)
        recs.append(rec)
    return recs


def half_then_dog_ears(sim, T, sides=("left",)):
    """The half fold, the measured stack re-centring (VBD's light sheet slides), the exposed corners to their places,
    then each named top-layer corner dog-eared onto its named point, then the named-point corrections."""
    from topogym.tasks.blanket_fold.redesign.tiers import C_TOP_NOW, C_TOP_NOW_R, DOG2_A, DOG2_B, DOG2R_A, DOG2R_B
    import os
    v2 = _vbd(sim) and os.environ.get("RD_FOLD_V2", "1") == "1"      # 15:20 fixes (VBD only; RD_FOLD_V2=0: the 10:30 oracle)
    recs = [run_stage(sim, half_by_corners())] + recentre_stack(sim, T)
    recs += correct_named(sim, T, rounds=1, names=((-.3, -.24), (.3, -.24)))
    from topogym.tasks.blanket_fold.redesign.tiers import DOG2_T, DOG2R_T

    frames = {}

    def dog_ear(name, hand, tip, now, a, b, goal):
        if not v2:
            return edge_turn(name, {hand: (tip, now)}, a, b, h_max=.06, duration_s=4.)
        # 10-11 20:00 (robot run, easier): the top layer's corner lay 3.7 cm off its nominal place after the half fold,
        # and the dog-ear's path, planned from the nominal place, dragged it along a wrong crease (edge ratio 1.40).
        # Start from the corner's measured place and land on the named point (the target's corner place)
        P = sim.particles()
        now_m = tuple(float(x) for x in (P[sim.nearest_particle(tip), :2] - sim.centre_xy))
        # 10-12 00:50 (robot, easier / medium / hard): after the half fold and the stack re-centring the top layer lay
        # 6 cm off its target place (RF 59 mm) and the named point, an absolute place, lay beyond the flap's reach: the
        # flap stretched to 1.30 and sprang back on release. The crease and the named point are places on the top
        # layer, so they follow the top layer's live frame (a rigid 2-D fit of its particles, the flap excluded, to the
        # target); the final re-centring places the stack
        R, t = _layer_frame(sim, T, P, tip)
        live = [tuple(R @ (np.asarray(x, float) + sim.centre_xy) + t - sim.centre_xy) for x in (a, b, goal)]
        frames[name] = dict(goal_shift_mm=[round(1e3 * float(x), 1) for x in np.subtract(live[2], goal)],
                            angle_deg=round(float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), 2))
        a, b, goal = live
        return unfold(name, {hand: (tip, a, b, dict(now_uv=now_m, end_uv=tuple(goal)))}, duration_s=4., h_max=.06,
                      search_m=.03, land=_land(sim, T, tip))
    def turned(name, hand, tip, now, a, b, goal, retries=2, tol=.040):
        """10-13 (d38cdf4bfb, adaptive VBD iterations): the dog-ear's release is a bifurcation; the flap sometimes
        springs back (easier: LF 159 mm). Measured retry: while the corner lies more than tol from its named point
        (in the top layer's live frame), turn it again from where it lies (up to `retries` times)."""
        out = [dict(run_stage(sim, dog_ear(name, hand, tip, now, a, b, goal)), top_layer_frame=frames.get(name))]
        for k in range(retries if v2 else 0):
            P = sim.particles()
            R, t = _layer_frame(sim, T, P, tip)
            err = float(np.linalg.norm(P[sim.nearest_particle(tip), :2] - (R @ (np.asarray(goal, float) + sim.centre_xy) + t)))
            if err <= tol:
                break
            rec = dict(run_stage(sim, dog_ear(f"{name}_retry{k + 1}", hand, tip, now, a, b, goal)),
                       top_layer_frame=frames.get(f"{name}_retry{k + 1}"), err_before_mm=round(1e3 * err, 1))
            out.append(rec)
        return out
    if "left" in sides:
        recs += turned("dog_ear_top", "left", CF_L, C_TOP_NOW, DOG2_A, DOG2_B, DOG2_T)
    if "right" in sides:
        recs += turned("dog_ear_top_r", "right", CF_R, C_TOP_NOW_R, DOG2R_A, DOG2R_B, DOG2R_T)
    if v2 and _vbd(sim):
        recs += lay_bumps(sim, T)
    recs += correct_named(sim, T, rounds=1)
    if v2:      # the whole stack onto the target's mean, last (b_very_easy / b_easier: 3-4 cm off after the corrections)
        recs += recentre_stack(sim, T, name="recentre_final") + correct_named(sim, T, rounds=1)
    return recs


def _flatten_placed(sim, tier):
    from topogym.planning.blanket_flatten.redesign.reverse import ROUTES_FIXUP, recenter
    from topogym.planning.blanket_flatten.redesign.tiers import TIERS as FT
    flat_name = tier["start"]["flatten"]
    recs = ROUTES_FIXUP[flat_name](sim, FT[flat_name])
    recs += recenter(sim, rounds=3, tol=.008)
    return recs + place_flat(sim)


def oracle_b_easier(sim, tier, T):
    return half_then_dog_ears(sim, T)


def oracle_b_medium(sim, tier, T):
    return _flatten_placed(sim, tier) + half_then_dog_ears(sim, T)


def roll_crease(sim, T, cols=((-.15, "left"), (.15, "right")), v_grasp=.02, tol=.008, lift=.008, name="roll_crease"):
    """The half fold's crease lies off the material's middle line when the top layer's near edge overhangs the bottom
    layer's (per column: d = y_top_edge - y_bottom_edge against the target's d_T). A top layer that must go far-ward
    (d_T - d > tol) is pulled from its crease side: the top layer alone grasped v_grasp beyond the middle line (2 cm
    from the crease) and drawn far-ward by d_T - d, so it rolls over the crease under tension (pulled, not pushed: a
    slide from its near edge buckles it, 10-13 tail A: stacking 0.83). A top layer that must come near-ward is pulled by
    its near edge. Measured on the live cloth."""
    from topogym.tasks.blanket_flatten.redesign.tiers import GRIP_IN
    P = sim.particles()
    uv = np.asarray(sim.uv)
    vmax, vmin = float(uv[:, 1].max()), float(uv[:, 1].min())
    grasp, to, meas = {}, {}, {}
    for u0, hand in cols:
        col = np.abs(uv[:, 0] - u0) <= .015
        top, bot = col & (uv[:, 1] >= vmax - 1e-3), col & (uv[:, 1] <= vmin + 1e-3)
        d_a = float(P[top, 1].mean() - P[bot, 1].mean())
        d_t = float(T[top, 1].mean() - T[bot, 1].mean())
        move = d_t - d_a
        meas[hand] = dict(u=u0, d_achieved_mm=round(1e3 * d_a, 1), d_target_mm=round(1e3 * d_t, 1), move_mm=round(1e3 * move, 1))
        if abs(move) > tol:
            g = (u0, v_grasp) if move > 0 else (u0, vmax - GRIP_IN)
            i = sim.nearest_particle(g)
            grasp[hand] = g
            to[hand] = (float(P[i, 0]), float(P[i, 1] + move), float(P[i, 2] - sim.table_top))
    if not grasp:
        return [dict(name=name, ok=True, reason="aligned", roll=meas)]
    st = dict(name=name, grasp=grasp, search_m=.02, segments=[
        dict(name="lift", duration_s=.4, hands={h: dict(by=(0., 0., _h(sim, grasp[h], lift))) for h in grasp}),
        dict(name="draw", duration_s=1.8, hands={h: dict(to=(to[h][0], to[h][1], _h(sim, grasp[h], lift))) for h in grasp}),
        dict(name="lower", duration_s=.5, hands={h: dict(to=(to[h][0], to[h][1], max(to[h][2], .003))) for h in grasp})],
        settle_s=.3)
    r = run_stage(sim, st)
    r["roll"] = meas
    return [r]


def crease_fix(sim, T, rounds=2):
    """Up to `rounds` measured crease rolls, then the stack re-centred and one round of named-corner corrections."""
    recs = []
    for k in range(rounds):
        r = roll_crease(sim, T, name=f"roll_crease_{k + 1}")
        recs += r
        if r[-1].get("reason") == "aligned":
            break
    return recs + recentre_stack(sim, T, name="recentre_final") + correct_named(sim, T, rounds=1)


def oracle_b_hard(sim, tier, T):
    """04:55: one dog-ear (see tiers.py: the two-dog-ear target leaves a standing roll); b_hard_two_dog_ears keeps both.
    10-13 (d38cdf4bfb physics): the route ended one term short, stacking 0.879 of 0.88: the half fold's crease lay ~2 cm
    off the middle line, the top layer overhanging the bottom layer's near edge by 38-47 mm. crease_fix rolls the top
    layer back over the crease (0.879 -> 0.975 after two rolls) and re-centres; hard only (the other tiers' staged
    references were recorded without it)."""
    sides = ("left", "right") if tier.get("name") == "b_hard_two_dog_ears" else ("left",)
    return _flatten_placed(sim, tier) + half_then_dog_ears(sim, T, sides=sides) + crease_fix(sim, T)


def oracle_b_easy(sim, tier, T):
    return _flatten_placed(sim, tier) + [run_stage(sim, half_by_corners())] + recentre_stack(sim, T) + \
        correct_named(sim, T, rounds=2)


# the entries the product uses (lane A 2026-10-11: the lab's other variants stay in the redesign sandbox)
ORACLES = {}
ORACLES['b_very_easy'] = oracle_very_easy_rc
ORACLES['b_easier'] = oracle_b_easier
ORACLES['b_easy'] = oracle_b_easy
ORACLES['b_medium'] = oracle_b_medium
ORACLES['b_hard'] = oracle_b_hard
