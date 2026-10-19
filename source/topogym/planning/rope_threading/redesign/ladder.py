"""Measurable ladder steps for the redesigned rope_threading tiers (numpy, capsule model).

Per leg (ring): over a Fibonacci sphere of gripper approach directions (x closing-axis rolls for near-needle
approaches), the fraction of orientations from which each arm reaches
  - the receive pinch (RECEIVE_STATION_M from the tip, the tip POKE_M past the ring plane) and
  - the feeder's push-end pose (FEED_STATION_M behind the tip),
IK-exact and collision-free (conservative capsules vs fixtures and table), and the largest clearance among them (the
best witness). Plus the ring's aperture margin (usable centreline radius minus the rope radius), the tail before the
first ring and the tip overhang. A harder tier should show fewer feasible orientations / smaller clearances / smaller
margins.

usage: python -m topogym.planning.rope_threading.redesign.ladder <out.json> [tier ...]
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

from topogym.planning.rope_threading.redesign import scenes as S
from . import aloha_kin as K
from . import oracle as O
from topogym.planning.rope_threading.redesign.legs import legs_for as _legs_for


def orientations(d, n_dirs=48):
    """Gripper rotations: approach directions on a Fibonacci sphere; the closing axis perpendicular to the rope where
    defined (both signs), else 4 rolls."""
    out = []
    for i in range(n_dirs):
        z = 1 - 2 * (i + 0.5) / n_dirs
        r = math.sqrt(max(0.0, 1 - z * z))
        th = math.pi * (1 + 5 ** 0.5) * i
        a = np.array([r * math.cos(th), r * math.sin(th), z])
        c = np.cross(a, d)
        if np.linalg.norm(c) >= 0.2:
            for sgn in (1.0, -1.0):
                out.append(K.gripper_frame(a, closing=sgn * c))
        else:
            w = S._unit(np.cross(a, [0.3, 0.5, 0.8]))
            for k in range(4):
                cc = math.cos(k * np.pi / 4) * w + math.sin(k * np.pi / 4) * np.cross(a, w)
                out.append(K.gripper_frame(a, closing=cc))
    return out


def reach_stats(design, arm, point, tangent, opening=0.012):
    d = S._unit(tangent)
    ok, best, tried = 0, -np.inf, 0
    for Rm in orientations(d):
        tried += 1
        q, pe, re = K.ik(arm, point, rotation=Rm)
        if pe > 0.003 or re > 0.05:
            continue
        good, g_arm, g_f, _ = O.config_clear(arm, q, design, opening)
        gap = min(g_arm[0], g_f[0])
        if good:
            ok += 1
            best = max(best, gap)
    return dict(feasible_fraction=round(ok / tried, 3), feasible=ok, tried=tried,
                best_clearance_m=None if ok == 0 else round(float(best), 4))


def tier_ladder(tier):
    d = S.build(tier)
    rows = []
    for leg in _legs_for(d):
        ring = [r for r in d.rings if r.name == leg["ring"]][0]
        dv = S._unit(ring.travel)
        pinch = ring.center + (S.POKE_M - S.RECEIVE_STATION_M) * dv
        push_end = ring.center - (S.FEED_STATION_M - S.POKE_M) * dv
        other = "left" if leg["receiver"] == "right" else "right"
        usable = ring.radius - (S.TUBE_RADIUS_M + S.ROPE_RADIUS_M - 0.001) - 0.0001
        rows.append(dict(ring=ring.name, axis=np.round(dv, 2).tolist(), inner_radius_mm=round(1000 * ring.inner_radius, 1),
                         aperture_margin_mm=round(1000 * (usable - S.ROPE_RADIUS_M), 1),
                         receiver=leg["receiver"], feeder=leg["feeder"],
                         receive_designated=reach_stats(d, leg["receiver"], pinch, dv),
                         receive_other=reach_stats(d, other, pinch, dv),
                         feed_end=reach_stats(d, leg["feeder"], push_end, dv)))
    geo = S.check_design(d)
    return dict(tier=tier, legs=rows, tail_m=round(geo["tail_before_first_ring_m"], 3),
                tip_m=round(geo["tip_after_last_ring_m"], 3), rope_m=round(d.rope_length, 3),
                min_goal_fixture_gap_mm=round(1000 * geo["goal_min_fixture_gap_m"], 1),
                distinct_axes=len({tuple(np.round(np.abs(S._unit(r.travel)), 1)) for r in d.rings}))


if __name__ == "__main__":
    out = sys.argv[1]
    res = {}
    for tier in sys.argv[2:] or list(S.TIERS):
        res[tier] = tier_ladder(tier)
        print(tier, json.dumps({k: v for k, v in res[tier].items() if k != "legs"}), flush=True)
        for r in res[tier]["legs"]:
            print("   ", r["ring"], r["inner_radius_mm"], "margin", r["aperture_margin_mm"], "recv", r["receive_designated"],
                  "other", r["receive_other"]["feasible_fraction"], "feed", r["feed_end"], flush=True)
    json.dump(res, open(out, "w"), indent=1)
