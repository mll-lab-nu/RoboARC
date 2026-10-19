"""Whole-program placement search of a redesigned claws tier (kinematic, no physics).

Unlike placement.score_design (moves up to the first hand change), this follows the whole designed program: every move
with the hands that hold the pieces at that point, every hand change (the new grip must be reachable, with its
pre-grasp pose 7 cm back along the approach, by either of the two symmetric pad orientations), and the hand boxes
(the gripper body around a TCP, tiers.hand_box) must stay clear of the piece the hand does not hold (FCL boxes), at
every sampled pose. Score: infeasible stages first, then joint travel, wrist bend and the smallest hand clearance.

  python -m topogym.planning.interlocked_claws.redesign.placement2 <tier> <out.json> --yaw ... --x ... --y ... --z ...
         [--share 0 .5]
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
from scipy.spatial.transform import Rotation

from topogym.planning.interlocked_claws.redesign import paths, tiers
from .placement import SEEDS, T_world, se3_power
from .robot_plan import Kin, solve_path

FLIP = np.diag([1., -1., -1., 1.])          # 180 deg about the approach: the same parallel grasp, pads swapped


def piece_of(grip):
    return "B" if grip.startswith("b_") else "A"


class HandClearance:
    """Smallest distance between the hand box at a TCP frame and a piece's boxes (FCL), in the piece's frame."""

    def __init__(self, piece):
        import fcl
        self.fcl = fcl
        self.objs = []
        for b in piece.boxes:
            T = np.asarray(b["pose"], float)
            self.objs.append(fcl.CollisionObject(fcl.Box(*b["size"]), fcl.Transform(T[:3, :3], T[:3, 3])))

    def distance(self, T_tcp_in_piece):
        hb = tiers.hand_box(T_tcp_in_piece)
        T = np.asarray(hb["pose"], float)
        h = self.fcl.CollisionObject(self.fcl.Box(*hb["size"]), self.fcl.Transform(T[:3, :3], T[:3, 3]))
        best = np.inf
        for o in self.objs:
            r = self.fcl.DistanceResult()
            d = self.fcl.distance(h, o, self.fcl.DistanceRequest(), r)
            best = min(best, d)
        return float(best)


def score_full(kin, d, share=0., seeds=SEEDS, clear_min=.004, step_m=.004, step_deg=2.):
    """Best over the birth pad orientations of both hands (each grip or its 180 deg twin): -> the best score_one."""
    best = None
    for fl in ((False, False), (True, False), (False, True), (True, True)):
        r = score_one(kin, d, share, seeds, clear_min, step_m, step_deg, birth_flip=dict(left=fl[0], right=fl[1]))
        r["birth_flip"] = dict(left=fl[0], right=fl[1])
        if best is None or r["cost"] < best["cost"]:
            best = r
        if r["ok"]:
            break
    return best


SHARES = (0., .5, 1.)
LIFT_M = .10
OVERS = (0., .06, .03, .10)    # the hop's end over the pre-grasp: 0 / .10 = safe height (+ .10); .06 / .03 = above pre


def over_z(safe_z, pre_z, extra):
    return max(safe_z, pre_z) + extra if extra in (0., .10) else pre_z + extra


LIFTS = ((.05, None), (.03, None), (0., None), (.05, .04), (.08, 0.))   # (back off m, rise: None = to the safe height)
TABLE_Z = .9196429           # the table top in the world (source z .02 + the mount offset)


def robot_self_hits(kin, q_arms, margin=.002):
    """Robot-robot contacts (one arm on itself or on the other arm) of a configuration, fingers open."""
    c = kin.contacts(q_arms, {a: .037 for a in q_arms}, {}, margin=margin)
    return [x for x in c if x[0].split("/")[0] in ("left", "right") and x[1].split("/")[0] in ("left", "right")
            and not (x[0].endswith("finger_link") and x[1].endswith("finger_link") and x[0][:4] == x[1][:4])]


def regrasp_path(kin, arm, q0, Tg, safe_z, q_other, clears, target_piece):
    """The oracle's regrasp reach, kinematically (oracle.regrasp): back 5 cm along the approach and up to safe_z
    (cartesian), a joint-space hop to the point over the pre-grasp, down to the pre-grasp and in (cartesian). Checked:
    IK along the cartesian parts, robot-robot contacts everywhere, the gripper-body box clear of the piece not being
    grasped along the hop. -> dict(ok, why, extra)."""
    from .robot_plan import solve_path
    from .geom import interp
    def line(T0, T1, step=.004):
        n = max(2, int(np.ceil(np.linalg.norm(T1[:3, 3] - T0[:3, 3]) / step)))
        return [interp(T0, T1, u) for u in np.linspace(0, 1, n + 1)[1:]]
    T0 = kin.tcp_frame(arm, q0)
    pre = Tg.copy(); pre[:3, 3] -= .07 * Tg[:3, 0]
    q1 = None
    for b_, u_ in LIFTS:                      # back off along the approach, then rise (the oracle tries the same list)
        back = T0.copy(); back[:3, 3] -= b_ * T0[:3, 0]
        up = back.copy(); up[2, 3] = max(safe_z, back[2, 3]) if u_ is None else back[2, 3] + u_
        q1, r1 = solve_path(kin, arm, line(T0, back) + line(back, up), q0, anchor_every=5, n_random=6)
        if q1 is not None:
            break
    if q1 is None:
        return dict(ok=False, why="lift_ik")
    other = "left" if arm == "right" else "right"
    def self_hit(q):
        qa = {arm: q}
        if q_other is not None:
            qa[other] = q_other
        return bool(robot_self_hits(kin, qa))
    why = "no_over_ik"
    for extra in OVERS:
        ov = pre.copy(); ov[2, 3] = over_z(safe_z, pre[2, 3], extra)
        q_ov, ep, er = kin.ik_multi(arm, ov, [q1[-1], q0, np.array([0., -.5, .8, 0., .6, 0.]),
                                               np.array([0., -.3, .5, 0., 1.2, 0.])])
        if ep > 1.5e-3 or er > .02:
            continue
        n = max(2, int(np.ceil(np.abs(q_ov - q1[-1]).max() / .03)))
        hop = [q1[-1] + u * (q_ov - q1[-1]) for u in np.linspace(0, 1, n + 1)[1:]]
        bad = False
        for k in range(0, len(hop), 3):
            if self_hit(hop[k]):
                bad, why = True, "hop_self"; break
            T_ = kin.tcp_frame(arm, hop[k])
            for name, (hc, Tp) in clears.items():
                if hc.distance(np.linalg.inv(Tp) @ T_) < .003:
                    bad, why = True, f"hop_hits_{name}"; break
            if bad:
                break
        if bad:
            continue
        q3, r3 = solve_path(kin, arm, line(ov, pre) + line(pre, Tg, .003), q_ov, anchor_every=5, n_random=6)
        if q3 is None:
            why = "approach_ik"; continue
        if any(self_hit(q) for q in q3[::4]) or any(self_hit(q) for q in q1[::4]):
            why = "approach_self"; continue
        return dict(ok=True, why=None, extra=extra)
    return dict(ok=False, why=why)


def pieces_low(d, TAs, poses, every=4):
    """The lowest corner of the pieces along a move (world z); the lock is skipped when it stands on its legs."""
    lo = np.inf
    cA, cB = d["A"].corners(), d["B"].corners()
    for k in list(range(0, len(poses), every)) + [len(poses) - 1]:
        TB = TAs[k] @ poses[k]
        lo = min(lo, float((cB @ TB[:3, :3].T + TB[:3, 3])[:, 2].min()))
        if not d.get("a_on_table"):
            lo = min(lo, float((cA @ TAs[k][:3, :3].T + TAs[k][:3, 3])[:, 2].min()))
    return lo


def hand_low(T_tcp):
    """The lowest point of the gripper-body box at a TCP frame (world)."""
    from itertools import product
    hb = tiers.hand_box(T_tcp)
    Tb = np.asarray(hb["pose"], float)
    h = np.asarray(hb["size"], float) / 2
    pts = np.array(list(product((-1, 1), repeat=3))) * h
    return float((pts @ Tb[:3, :3].T + Tb[:3, 3])[:, 2].min())


def score_one(kin, d, share=0., seeds=SEEDS, clear_min=.004, step_m=.004, step_deg=2., birth_flip=None):
    TA = T_world(d["A_in_assembly"])
    P = np.asarray(d["engaged"], float)
    pieces = {"A": d["A"], "B": d["B"]}
    clear = {"A": HandClearance(d["A"]), "B": HandClearance(d["B"])}
    holds = dict(d["hands"])                                  # arm -> grip
    grip_T = {arm: np.asarray(pieces[piece_of(g)].grips[g], float) @ (FLIP if (birth_flip or {}).get(arm) else np.eye(4))
              for arm, g in holds.items()}
    shares_used = []
    cur, travel, wr, worst_clear, notes = {}, 0., {"left": [9, -9], "right": [9, -9]}, np.inf, []

    def world(piece, T_A, P_):
        return T_A if piece == "A" else T_A @ P_

    for arm, g in holds.items():
        T = world(piece_of(g), TA, P) @ grip_T[arm]
        sd = list(seeds[arm])
        if (birth_flip or {}).get(arm):          # the twin grasp: the unflipped solution with wrist_rotate +- pi
            T0 = world(piece_of(g), TA, P) @ np.asarray(pieces[piece_of(g)].grips[g], float)
            q0, _, _ = kin.ik_multi(arm, T0, seeds[arm])
            for dq in (np.pi, -np.pi):
                q1 = q0.copy(); q1[5] += dq
                if abs(q1[5]) <= 2.8:
                    sd.insert(0, q1)
        q, ep, er = kin.ik_multi(arm, T, sd)
        if ep > 1e-3 or er > .01:
            return dict(ok=False, cost=100., stage=f"birth:{arm}", err=[ep, er])
        cur[arm] = q
    birth_q = {a: np.round(q, 4).tolist() for a, q in cur.items()}

    def hand_clear(arm, T_A, P_):
        """The hand of `arm` vs the piece it does not hold (both pieces when it holds none)."""
        g = holds.get(arm)
        T_tcp = world(piece_of(g), T_A, P_) @ grip_T[arm]
        other = "A" if piece_of(g) == "B" else "B"
        T_other = world(other, T_A, P_)
        return clear[other].distance(np.linalg.inv(T_other) @ T_tcp)

    if d.get("start") == "table":            # the oracle lifts the grasped assembly 6 cm off its legs first
        TA = TA.copy(); TA[2, 3] += LIFT_M
        for arm, g in holds.items():
            q, ep, er = kin.ik(arm, world(piece_of(g), TA, P) @ grip_T[arm], cur[arm], prior=cur[arm])
            if ep > 1.5e-3 or er > .02:
                return dict(ok=False, cost=90., stage=f"lift:{arm}")
            cur[arm] = q
    n_moves = len(d["moves"])
    seg = None
    for i, mv in enumerate(d["moves"]):
        for arm, g in (mv.get("hold") or {}).items():         # hand change: arm lets go and takes grip g
            if g is None:                                     # only lets go
                holds.pop(arm, None)
                grip_T.pop(arm, None)
                continue
            piece = piece_of(g)
            Tp = world(piece, TA, P)
            best = None
            for flip in (False, True):
                G = np.asarray(pieces[piece].grips[g], float) @ (FLIP if flip else np.eye(4))
                T = Tp @ G
                pre = T.copy(); pre[:3, 3] -= .07 * T[:3, 0]
                q, ep, er = kin.ik_multi(arm, T, [cur[arm]] + list(seeds[arm]))
                q2, ep2, er2 = kin.ik(arm, pre, q, prior=q)
                if max(ep, ep2) > 1.5e-3 or max(er, er2) > .02:
                    continue
                if min(hand_low(T), hand_low(pre)) < TABLE_Z + .01:     # the gripper body would reach the table
                    continue
                other_arm = "left" if arm == "right" else "right"
                if robot_self_hits(kin, {arm: q, other_arm: cur[other_arm]}) or \
                        robot_self_hits(kin, {arm: q2, other_arm: cur[other_arm]}):
                    continue                                                # the arm on itself / the other arm
                # the open hand at the grip and at the pre-grasp vs the other piece
                other = "A" if piece == "B" else "B"
                To = world(other, TA, P)
                c = min(clear[other].distance(np.linalg.inv(To) @ T), clear[other].distance(np.linalg.inv(To) @ pre))
                cost = np.abs(q - cur[arm]).max() + (5. if c < clear_min else 0.)
                if best is None or cost < best[0]:
                    best = (cost, q, flip, c)
            if best is None:
                return dict(ok=False, cost=40. + 10 * (n_moves - i), stage=f"regrasp{i}:{arm}:{g}")
            _, q, flip, c = best
            Tg_ = Tp @ np.asarray(pieces[piece].grips[g], float) @ (FLIP if flip else np.eye(4))
            corners = np.concatenate([d["A"].corners() @ TA[:3, :3].T + TA[:3, 3],
                                      d["B"].corners() @ (TA @ P)[:3, :3].T + (TA @ P)[:3, 3]])
            other_arm = "left" if arm == "right" else "right"
            pv = regrasp_path(kin, arm, cur[arm], Tg_,
                              corners[:, 2].max() + .08, cur[other_arm] if other_arm in cur else None,
                              {"A": (clear["A"], TA), "B": (clear["B"], TA @ P)}, piece)
            if not pv["ok"]:
                return dict(ok=False, cost=35. + 10 * (n_moves - i), stage=f"regrasp_path{i}:{arm}:{g}:{pv['why']}",
                            notes=notes)
            notes.append(dict(regrasp=g, arm=arm, flip=flip, clearance=round(c, 4), path=pv))
            worst_clear = min(worst_clear, c)
            holds[arm] = g
            grip_T[arm] = np.asarray(pieces[piece].grips[g], float) @ (FLIP if flip else np.eye(4))
            cur[arm] = q
        if i == 0 or mv.get("hold"):              # a segment starts: plan all its moves at once (global IK branches)
            seg = segment_plan(kin, d, i, P, TA, holds, grip_T, cur, world, step_m, step_deg)
        poses, _, _ = paths.dense(P, [mv], step_m=step_m, step_deg=step_deg)
        a_held = any(piece_of(g) == "A" for g in holds.values()) and not d.get("a_on_table")
        if seg is not None and i in seg:
            sols, TAs, poses = seg[i]
            shares_used.append([seg["share"][i], "segment"])
            for arm, sol in sols.items():
                travel += float(np.abs(np.diff(np.asarray(sol), axis=0)).sum())
                for q in sol:
                    wr[arm] = [min(wr[arm][0], q[4]), max(wr[arm][1], q[4])]
                cur[arm] = sol[-1]
            for k in range(0, len(poses), 3):
                for arm in holds:
                    worst_clear = min(worst_clear, hand_clear(arm, TAs[k], poses[k]))
            P, TA = poses[-1], TAs[-1]
            continue
        cands = [float(mv["share"])] if "share" in mv else [share] + [x for x in SHARES if x != share]
        if not a_held:
            cands = [0.]
        P0i = np.linalg.inv(P)
        fail = None
        tcp0 = {arm: (world(piece_of(g), TA, P) @ grip_T[arm])[:3, 3] for arm, g in holds.items()}
        opts = []
        for sh in cands:
            opts.append((sh, None))
            if sh and mv["kind"] != "slide":
                opts += [(sh, "mid")]
        for sh, fx in opts:
            TAs = [TA @ (se3_power(np.linalg.inv(Pk @ P0i), sh) if sh else np.eye(4)) for Pk in poses]
            if fx is not None:
                o = (tcp0["left"] + tcp0["right"]) / 2 if fx == "mid" and len(tcp0) == 2 else tcp0.get(fx, list(tcp0.values())[0])
                fixedA = []
                for A_ in TAs:
                    G = A_ @ np.linalg.inv(TA)
                    A2 = A_.copy(); A2[:3, 3] += o - (G[:3, :3] @ o + G[:3, 3]); fixedA.append(A2)
                TAs = fixedA
            sols, fail = {}, None
            for arm, g in holds.items():
                Ts = [world(piece_of(g), A_, Pk) @ grip_T[arm] for A_, Pk in zip(TAs, poses)]
                sol, rep = solve_path(kin, arm, Ts, cur[arm], anchor_every=5, n_random=8)
                if sol is None:
                    fail = (arm, rep)
                    break
                if min(hand_low(T_) for T_ in Ts[::4] + [Ts[-1]]) < TABLE_Z + .01:
                    fail = (arm, dict(why="table"))
                    break
                sols[arm] = sol
            if fail is None and pieces_low(d, TAs, poses) < TABLE_Z + .005:
                fail = ("pieces", dict(why="table"))
            if fail is None:
                break
        if fail is not None:
            return dict(ok=False, cost=20. + 10 * (n_moves - i), stage=f"move{i}:{fail[0]}", rep=fail[1], notes=notes,
                        shares=shares_used)
        shares_used.append([sh, fx])
        for arm, sol in sols.items():
            travel += float(np.abs(np.diff(np.asarray(sol), axis=0)).sum())
            for q in sol:
                wr[arm] = [min(wr[arm][0], q[4]), max(wr[arm][1], q[4])]
            cur[arm] = sol[-1]
        for k in range(0, len(poses), 3):
            for arm in holds:
                worst_clear = min(worst_clear, hand_clear(arm, TAs[k], poses[k]))
        P, TA = poses[-1], TAs[-1]
    big = max(max(abs(v) for v in wr[a]) for a in wr if wr[a][0] < 9)
    cost = .05 * travel + max(0., big - 1.2) + (3. if worst_clear < clear_min else 0.) + max(0., .02 - worst_clear) * 50
    return dict(ok=bool(worst_clear >= clear_min), cost=float(cost), travel=round(travel, 2), wrist=wr, shares=shares_used,
                birth_q=birth_q,
                hand_clearance_m=round(float(worst_clear), 4), notes=notes,
                end={a: np.round(q, 2).tolist() for a, q in cur.items()})


def segment_plan(kin, d, i0, P, TA, holds, grip_T, cur, world, step_m, step_deg):
    """The moves from i0 up to the next hand change, every arm's targets in one list, one IK-branch DP per arm.
    -> {move index: (per-arm joint lists, A poses, relative poses), "share": {i: share}} or None."""
    moves = d["moves"]
    j = i0 + 1
    while j < len(moves) and not moves[j].get("hold"):
        j += 1
    a_held = any(piece_of(g) == "A" for g in holds.values()) and not d.get("a_on_table")
    per, Pc, TAc, shares = [], P, TA, {}
    for i in range(i0, j):
        mv = moves[i]
        sh = float(mv.get("share", 0.)) if a_held else 0.
        shares[i] = sh
        poses, _, _ = paths.dense(Pc, [mv], step_m=step_m, step_deg=step_deg)
        P0i = np.linalg.inv(Pc)
        TAs = [TAc @ (se3_power(np.linalg.inv(Pk @ P0i), sh) if sh else np.eye(4)) for Pk in poses]
        if pieces_low(d, TAs, poses) < TABLE_Z + .005:
            return None
        per.append((i, poses, TAs))
        Pc, TAc = poses[-1], TAs[-1]
    out = {"share": shares}
    sols = {}
    for arm, g in holds.items():
        Ts, lens = [], []
        for i, poses, TAs in per:
            Ts += [world(piece_of(g), A_, Pk) @ grip_T[arm] for A_, Pk in zip(TAs, poses)]
            lens.append(len(poses))
        sol, rep = solve_path(kin, arm, Ts, cur[arm], anchor_every=5, n_random=10, max_step=.35)
        if sol is None or min(hand_low(T_) for T_ in Ts[::4] + [Ts[-1]]) < TABLE_Z + .01:
            return None
        sols[arm] = (sol, lens)
    for n, (i, poses, TAs) in enumerate(per):
        sub = {}
        for arm, (sol, lens) in sols.items():
            a = sum(lens[:n])
            sub[arm] = sol[a:a + lens[n]]
        out[i] = (sub, TAs, poses)
    return out


def search(tier, grid, kw=None):
    from rcb_topogym import Session
    s = Session("interlocked_claws", variation_id="interlocked_claws_very_easy", seed=0, gripper_space="native",
                strict=False)
    kin = Kin(s.env._world.model)
    out = []
    t0 = time.time()
    for yaw, x, y, z, share in itertools.product(grid["yaw"], grid["x"], grid["y"], grid["z"], grid["share"]):
        T_A = np.eye(4)
        T_A[:3, :3] = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
        T_A[:3, 3] = [x, y, z]
        best = None
        for g, hd in itertools.product(grid.get("grips") or [None], grid.get("heads") or [None]):
            k2 = dict(kw or {})
            if g is not None:
                k2["b_grip"] = g
            if hd is not None:
                k2["head"] = hd
            try:
                d = tiers.TIERS[tier](T_A=T_A, **k2)
            except ValueError as e:               # no clear handle root at this placement
                r = dict(ok=False, cost=200., stage=f"design:{e}")
            else:
                r = score_full(kin, d, share)
            r.update(yaw=yaw, x=x, y=y, z=z, share=share, b_grip=g, head=hd)
            print(json.dumps(r, default=float), round(time.time() - t0, 1), flush=True)
            if best is None or r["cost"] < best["cost"]:
                best = r
        out.append(best)
    out.sort(key=lambda r: r["cost"])
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("tier"); ap.add_argument("out")
    ap.add_argument("--yaw", type=float, nargs="+", default=[-90, -45, 0, 45, 90])
    ap.add_argument("--x", type=float, nargs="+", default=[-.08, 0., .08])
    ap.add_argument("--y", type=float, nargs="+", default=[-.16, -.08, 0.])
    ap.add_argument("--z", type=float, nargs="+", default=[.20, .26])
    ap.add_argument("--share", type=float, nargs="+", default=[0., .5])
    ap.add_argument("--grips", nargs="+", default=None, help="B grip options: top, axial, side:<deg>")
    ap.add_argument("--heads", nargs="+", default=None, help="head grip options: above, arm:<side>[:tilt]")
    ap.add_argument("--kw", default="{}")
    a = ap.parse_args()
    res = search(a.tier, dict(yaw=a.yaw, x=a.x, y=a.y, z=a.z, share=a.share, grips=a.grips, heads=a.heads),
                 kw=json.loads(a.kw))
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print("best", json.dumps(res[:5], default=float))
