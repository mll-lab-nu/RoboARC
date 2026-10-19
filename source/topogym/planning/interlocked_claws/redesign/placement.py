"""Placement search of a redesigned claws assembly in the robot's workspace (kinematic, no physics).

For a design (pieces, engaged pose, moves), every candidate placement of A in the assembly frame (yaw, x, y, z) and
share of the relative motion taken by the left hand is scored by following the dense relative path with both arms:
the right hand on B's grip, the left hand on a grip that is placed on A where the left arm reaches it naturally (the
grip's position on A is given; its approach direction is chosen per placement: from the left shoulder toward the grip,
tilted 15 deg down). Score: IK residuals, wrist margins (the wrist stays unfolded), joint travel.

  python -m topogym.planning.interlocked_claws.redesign.placement <tier> <out.json>
"""
from __future__ import annotations

import itertools
import json
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from topogym.planning.interlocked_claws.redesign import paths, tiers
from .robot_plan import Kin

SOURCE_TO_WORLD = np.array([-.1356786, 0., .9196429 - .02])   # authored source frame -> world (mount, riser 0)
SHOULDER = {"left": np.array([-.469, -.019, .099]), "right": np.array([.469, -.019, .099])}


def T_world(T_source):
    T = np.array(T_source, float)
    T[:3, 3] = T[:3, 3] + SOURCE_TO_WORLD
    return T


def natural_grip(p, arm, tilt_deg=15.):
    """A TCP frame at p (source frame) whose approach points from the arm's shoulder toward p (horizontal part), tilted
    down; pads closing horizontally."""
    d = p[:2] - SHOULDER[arm][:2]
    yaw = np.arctan2(d[1], d[0])
    t = np.deg2rad(tilt_deg)
    a = np.array([np.cos(yaw) * np.cos(t), np.sin(yaw) * np.cos(t), -np.sin(t)])
    c = np.array([-np.sin(yaw), np.cos(yaw), 0.])
    z = np.cross(a, c)
    if z[2] < 0:
        c, z = -c, -z
    T = np.eye(4); T[:3, :3] = np.stack([a, c, z], 1); T[:3, 3] = p
    return T


def se3_power(D, lam):
    from scipy.linalg import logm, expm
    L = np.real(logm(D))
    return np.real(expm(lam * L))


def score_design(kin, d, share=0., seeds=None):
    """Follow the design's moves (until its first hand change) with both arms (IK-branch DP); the left hand on A's
    grip, the right on B's. -> dict(ok, worst, wrist ranges, joint travel, cost)."""
    from .robot_plan import solve_path
    TA0 = T_world(d["A_in_assembly"])
    P = np.asarray(d["engaged"], float)
    gA = np.asarray(d["A"].grips[d["hands"]["left"]], float)
    gB = np.asarray(d["B"].grips[d["hands"]["right"]], float)
    cur = {}
    qa, ea, ra = kin.ik_multi("left", TA0 @ gA, seeds["left"])
    qb, eb, rb = kin.ik_multi("right", TA0 @ P @ gB, seeds["right"])
    if max(ea, eb) > 1e-3 or max(ra, rb) > .01:
        return dict(ok=False, cost=50. + 10 * (ea + eb), stage="birth", worst=[ea, ra, eb, rb])
    cur = {"left": qa, "right": qb}
    wr = {"left": [qa[4], qa[4]], "right": [qb[4], qb[4]]}
    travel, T_A = 0., TA0
    for i, mv in enumerate(d["moves"]):
        if mv.get("hold"):
            break
        poses, _, _ = paths.dense(P, [mv], step_m=.004, step_deg=2.)
        sh = float(mv.get("share", share))
        P0i = np.linalg.inv(P)
        TAs = []
        for Pk in poses:
            N = se3_power(np.linalg.inv(Pk @ P0i), sh) if sh else np.eye(4)
            TAs.append(T_A @ N)
        for arm, Ts in (("left", [A @ gA for A in TAs]), ("right", [A @ Pk @ gB for A, Pk in zip(TAs, poses)])):
            sol, rep = solve_path(kin, arm, Ts, cur[arm], anchor_every=5, n_random=8)
            if sol is None:
                return dict(ok=False, cost=20. + 10 * (len(d["moves"]) - i), stage=f"move{i}:{arm}", rep=rep)
            travel += float(np.abs(np.diff(np.asarray(sol), axis=0)).sum())
            for q in sol:
                wr[arm] = [min(wr[arm][0], q[4]), max(wr[arm][1], q[4])]
            cur[arm] = sol[-1]
        P, T_A = poses[-1], TAs[-1]
    big = max(abs(wr["left"][0]), abs(wr["left"][1]), abs(wr["right"][0]), abs(wr["right"][1]))
    return dict(ok=True, cost=float(.05 * travel + max(0., big - 1.2)), wrist=wr, travel=round(travel, 2),
                end={a: np.round(q, 2).tolist() for a, q in cur.items()})


SEEDS = {"left": [np.array(x) for x in ([0, -.6, .9, 0, .2, 0], [-.3, -.6, .9, .5, .3, -.5], [.3, -.6, .9, -.5, .3, .5],
                                        [0, -.3, .5, 0, .5, 0], [0, -.9, 1.3, 0, -.4, 0], [0, -.1, .1, 0, 1.4, 0])],
         "right": [np.array(x) for x in ([0, -.6, .9, 0, .2, 0], [-.3, -.6, .9, .5, .3, -.5], [.3, -.6, .9, -.5, .3, .5],
                                         [0, -.3, .5, 0, .5, 0], [0, -.9, 1.3, 0, -.4, 0], [0, -.1, .1, 0, 1.4, 0])]}


def search_design(tier, grid, kw=None, verbose=True):
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
        d = tiers.TIERS[tier](T_A=T_A, **(kw or {}))
        r = score_design(kin, d, share, SEEDS)
        r.update(yaw=yaw, x=x, y=y, z=z, share=share)
        out.append(r)
        if verbose:
            print(json.dumps(r, default=float), round(time.time() - t0, 1), flush=True)
    out.sort(key=lambda r: r["cost"])
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("tier"); ap.add_argument("out")
    ap.add_argument("--yaw", type=float, nargs="+", default=[-90, -60, -30, 0, 30])
    ap.add_argument("--x", type=float, nargs="+", default=[-.12, -.04, .04])
    ap.add_argument("--y", type=float, nargs="+", default=[-.12, -.04, .04])
    ap.add_argument("--z", type=float, nargs="+", default=[.26, .32])
    ap.add_argument("--share", type=float, nargs="+", default=[0., .5])
    ap.add_argument("--kw", default="{}")
    a = ap.parse_args()
    tier, path = a.tier, a.out
    res = search_design(tier, dict(yaw=a.yaw, x=a.x, y=a.y, z=a.z, share=a.share), kw=json.loads(a.kw))
    json.dump(res, open(path, "w"), indent=1)
    print("best", json.dumps(res[:5]))
