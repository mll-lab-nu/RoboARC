"""Goal of the redesigned knot_untying tiers (a task design, argued in docs/rcb/redesign/knot_untying.md).

The family's goal, unchanged, on every rope (common.KnotUntyingTask.evaluate_state): no projected crossings or
overlapping segments (planar_clearance at 2 r), on the table (z in [table - 1 mm, table + 2 r + 3 mm]), quiet
(<= 1 mm/s; the 0.5 s stable window and the release / robot-clear settling stay the Session's own), plus two goal terms
of the redesign:
  - between ropes: the same planar clearance (2 r) between any two ropes (two ropes lying across each other are not
    "untangled");
  - layout: every rope's centre line inside its region polygon (task-local xy), when the tier declares one.
Both are goal terms only: nothing ends an episode early (clamp, never fail), no new failure reason is introduced.
"""
from __future__ import annotations

import numpy as np


def inside(xy, polygon):
    """Points inside a polygon (even-odd rule); xy (N, 2)."""
    xy = np.asarray(xy, float)
    poly = np.asarray(polygon, float)
    x, y = xy[:, 0], xy[:, 1]
    res = np.zeros(len(xy), bool)
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        res ^= ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / np.where(y2 != y1, y2 - y1, 1e-18) + x1)
    return res


def mutual_planar_distance(a_xy, b_xy):
    """Smallest planar distance between two polylines (exact segment-segment)."""
    from topogym.utils.geometry.passage import segment_distance_2d
    a, b = np.asarray(a_xy, float), np.asarray(b_xy, float)
    best = np.inf
    # coarse prefilter on vertices, then exact on nearby segment pairs
    from scipy.spatial import cKDTree
    tree = cKDTree(b)
    for i in range(len(a) - 1):
        near = tree.query_ball_point(a[i], .03)
        for j in near:
            for jj in (j - 1, j):
                if 0 <= jj < len(b) - 1:
                    best = min(best, segment_distance_2d(a[i], a[i + 1], b[jj], b[jj + 1]))
    if not np.isfinite(best):
        best = float(tree.query(a)[0].min())
    return float(best)


def evaluate(ropes, *, radius, table_height_m, regions=None, origin_xy=(0., 0.), maximum_speed_m_s=.001,
             table_clearance_m=.003, penetration_tolerance_m=.001):
    """ropes: {name: dict(points (N, 3) world material centre line, stations (N,), speeds (M,))}.
    Returns (success_now, metrics); the stable window is the Session's."""
    from topogym.utils.geometry.passage import planar_clearance
    metrics, ok = dict(ropes={}), True
    origin = np.asarray(origin_xy, float)
    for name, r in ropes.items():
        p = np.asarray(r["points"], float)
        clear, geo = planar_clearance(p, np.asarray(r["stations"], float), 2 * radius)
        z = p[:, 2]
        on_table = bool(z.min() >= table_height_m - penetration_tolerance_m
                        and z.max() <= table_height_m + 2 * radius + table_clearance_m)
        speed = float(np.max(r["speeds"])) if len(r["speeds"]) else 0.
        rec = dict(planar_clear=bool(clear), on_table=on_table, quiet=speed <= maximum_speed_m_s,
                   maximum_speed_m_s=speed, **{k: v for k, v in geo.items()})
        if regions and name in regions:
            inn = inside(p[:, :2] - origin, regions[name])
            rec.update(in_region=bool(inn.all()), outside_fraction=float(1 - inn.mean()))
        metrics["ropes"][name] = rec
        ok &= rec["planar_clear"] and on_table and rec["quiet"] and rec.get("in_region", True)
    names = list(ropes)
    pairs = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = mutual_planar_distance(np.asarray(ropes[names[i]]["points"])[:, :2], np.asarray(ropes[names[j]]["points"])[:, :2])
            pairs[f"{names[i]}|{names[j]}"] = d
            ok &= d >= 2 * radius
    metrics["between_ropes_min_planar_distance_m"] = pairs
    metrics["goal_terms"] = ["family: planar_clear, on_table, quiet (per rope)", "between ropes >= 2 r",
                             "in_region (tiers with regions)"]
    return bool(ok), metrics
