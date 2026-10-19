"""Object-level certificate and breakers of a redesigned claws tier (no robot, no physics).

  python -m topogym.planning.interlocked_claws.redesign.check_design <tier> <out.json>

Checks:
  - path: the designed disengagement (dense exact moves) keeps >= the certified clearance (geom.check_path);
  - start: the engaged pose is free (distance > 0) and NOT separated; end: separated by the task's predicate (>= 10 mm);
  - breakers from the engaged pose: straight translations along 26 lattice directions + 40 random directions (0.5 m),
    rotations about A's axes and B's axes through B's origin and the claw (+-180 deg), the straight line (lerp + slerp)
    to the designed end pose; and, from the start of every move, the same translations (a skipped-move shortcut);
  - move necessity: the path with each single move removed (the rest re-applied) is not free.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from topogym.planning.interlocked_claws.redesign import paths, tiers
from . import geom


def check(design, clearance_certified=None, n_random=40, verbose=True):
    t0 = time.time()
    A, B = design["A"], design["B"]
    pair = geom.PairChecker(A.boxes, B.boxes)
    ca, cb = A.corners(), B.corners()
    c_cert = design["clearance"] * .5 if clearance_certified is None else clearance_certified
    poses, starts, which = paths.dense(design["engaged"], design["moves"])
    res = dict(tier=design["tier"], name=design["name"], boxes_A=len(A.boxes), boxes_B=len(B.boxes),
               clearance_design_m=design["clearance"], clearance_certified_m=c_cert)
    res["path"] = geom.check_path(pair, poses, c_cert)
    d0 = pair.distance(design["engaged"])
    res["start"] = dict(distance_m=d0, separating_gap_m=geom.separating_gap(ca, np.eye(4), cb, design["engaged"]))
    res["end"] = dict(separating_gap_m=geom.separating_gap(ca, np.eye(4), cb, poses[-1]))
    res["end"]["separated"] = res["end"]["separating_gap_m"] >= .01
    # breakers from the start
    br = []
    for k, d in enumerate(geom.directions(n_random, seed=1)):
        esc, at = geom.escape_translation(pair, ca, cb, design["engaged"], d, .5, 0.)
        br.append(dict(kind="translate", dir=np.round(d, 3).tolist(), escaped=esc, blocked_at_m=at))
    T0 = np.asarray(design["engaged"])
    pivots = {"B_origin": T0[:3, 3], "A_origin": np.zeros(3)}
    for pname, pv in pivots.items():
        for ax in np.eye(3):
            for sgn in (1, -1):
                ang = geom.escape_rotation(pair, T0, sgn * ax, pv, 180., 0.)
                br.append(dict(kind="rotate", pivot=pname, axis=(sgn * ax).tolist(), blocked_at_deg=ang,
                               escaped=False if ang is not None else None))
    u = geom.first_collision(pair, T0, poses[-1], 0., 600)
    br.append(dict(kind="straight_line_to_goal", blocked_at_u=u, escaped=u is None))
    res["breakers_start"] = dict(n=len(br), escaped=sum(1 for b in br if b["escaped"]), list=br)
    # from the start of each move: translations (skipped moves)
    per_move = []
    for i, s in enumerate(starts):
        Ts = poses[s]
        esc = []
        for d in geom.directions(0):
            e, at = geom.escape_translation(pair, ca, cb, Ts, d, .5, 0.)
            if e:
                esc.append(np.round(d, 3).tolist())
        per_move.append(dict(move=i, label=design["moves"][i].get("label"), escapes=esc))
    res["breakers_per_move"] = per_move
    # each move removed
    necessity = []
    for i in range(len(design["moves"])):
        mv = [m for j, m in enumerate(design["moves"]) if j != i]
        ps, _, _ = paths.dense(design["engaged"], mv)
        ok = geom.check_path(pair, ps, 0.)
        sep = geom.separating_gap(ca, np.eye(4), cb, ps[-1]) >= .01
        necessity.append(dict(removed=i, label=design["moves"][i].get("label"), path_free=ok["passed"],
                              separated_at_end=bool(sep), needed=not (ok["passed"] and sep)))
    res["move_necessity"] = necessity
    res["queries"] = pair.queries
    res["seconds"] = round(time.time() - t0, 1)
    res["pass"] = bool(res["path"]["passed"] and d0 > 0 and not res["start"]["separating_gap_m"] >= .01
                       and res["end"]["separated"] and res["breakers_start"]["escaped"] == 0
                       and all(n["needed"] for n in necessity))
    if verbose:
        print(json.dumps(dict(tier=res["tier"], pass_=res["pass"], path=res["path"], start=res["start"], end=res["end"],
                              breakers_escaped=res["breakers_start"]["escaped"],
                              per_move_escapes=[(p["label"], len(p["escapes"])) for p in per_move],
                              necessity=[(n["label"], n["needed"]) for n in necessity], seconds=res["seconds"]),
                         default=float))
    return res


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    tier, out = argv[0], argv[1]
    design = tiers.make(tier)
    res = check(design)
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))


if __name__ == "__main__":
    main()
