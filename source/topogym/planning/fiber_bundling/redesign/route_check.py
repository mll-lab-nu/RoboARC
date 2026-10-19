"""Object-level check of a redesign tier's route terms (no stepping): the task's route_terms on sheath centre lines
sampled like the real one (link centres 40 mm apart, two lead links behind the cradle exit, the front):
- taut: the derived oracle route (derive.py) -> every peg routed;
- over_top: the derived route, but where it passes a lay-over peg the cable crosses straight over the peg's top,
  5 mm to the required side of its centre -> must not count (the below-top rule; the 2D-only rule is printed too);
- shortcut: the taut cable with no pegs, from the cradle exit to where it kinks into the ring axis, then the axis
  -> a lay-over peg on the wrong side.
Run it through run_registered.py.

usage: route_check.py <tier> <instance.json> <this directory> <out.json>
"""
import json
import sys

import numpy as np
from rcb_topogym import Session

tier, inst, plan_dir, out = sys.argv[1:5]
sys.path.insert(0, plan_dir)
from derive import CRADLE_EXIT, FEED_PAST_M, ROUTES, arclength, at, derive  # noqa: E402

s = Session("fiber_bundling", instance=inst, strict=True)
task, w = s.env._task, s.env._world
state = w.read_state()
spec = ROUTES[tier]
d = derive(tier, spec["segments"], ring_inner_radius_m=spec["ring_inner_radius_m"], lay=spec["lay"],
           extra_pull_m=spec.get("extra_pull_m", 0.), trunk_length_m=spec.get("trunk_length_m", 0.24),
           trunk_segments=spec.get("trunk_segments", 6), taut=spec.get("taut"))
pegs = task.config["redesign"]["pegs"]
origin = np.asarray(task.origin, float)
r_trunk = task.layout.trunk_radius_m


def sampled(P2, z=None):
    """Lead links 40 and 80 mm behind the cradle exit, then points every 40 mm along P2 (local xy) to its end."""
    S = arclength(P2)
    pts = [np.r_[CRADLE_EXIT - [0.08, 0.], 0.], np.r_[CRADLE_EXIT - [0.04, 0.], 0.]]
    for x in list(np.arange(0., S[-1], 0.04)) + [S[-1]]:
        p, _ = at(P2, x)
        pts.append(np.r_[p, 0.])
    P3 = np.asarray(pts)
    if z is not None:
        P3[:, 2] = z(P3)
    return P3 + origin


def terms(P3):
    task._sheath_polyline = lambda _state: P3
    return task.route_terms(state)


P = np.asarray(d["polyline"])
res = dict(tier=tier, cases={})
res["cases"]["taut"] = terms(sampled(P))
# over the top: for each lay-over peg, the route replaced near the peg by a straight run across the peg's top
lay_over = [c["peg"] for c in d["checks"]["pegs_clear"] if c["lay_over"]]
for peg in pegs:
    if peg["name"] not in lay_over:
        continue
    c = np.asarray(peg["xy_m"], float)
    S = arclength(P)
    k = int(np.argmin(np.linalg.norm(P - c, axis=1)))
    a_, _ = at(P, max(S[k] - 0.06, 0.))
    b_, _ = at(P, min(S[k] + 0.06, S[-1]))
    u = (b_ - a_) / np.linalg.norm(b_ - a_)
    n = np.array([-u[1], u[0]])                                   # left of the run
    off = 0.005 * (n if peg["side"] == "left" else -n)            # the peg 5 mm to the cable's required side
    q = c - off                                                   # the cable's point nearest the peg centre
    run = np.asarray([q + u * t for t in np.linspace(-0.06, 0.06, 13)])
    over = np.vstack([P[:max(k - 30, 1)], run, P[min(k + 30, len(P) - 1):]])
    top = float(peg["top_m"]) + r_trunk + 0.002
    z = lambda P3, q=q: np.where(np.linalg.norm(P3[:, :2] - q, axis=1) < 0.05, top, 0.)    # local coordinates
    t = terms(sampled(over, z))
    flat = terms(sampled(over))                                   # the same xy with the cable at the axis height
    res["cases"][f"over_top_{peg['name']}"] = dict(terms=t, routed_by_2d_only={r["name"]: r["side"] == r["required"]
                                                   and r["gap_m"] <= 0.02 for r in flat})
# the shortcut: from the cradle exit straight to 0.06 m before the ring centre, then along the ring axis
ring_c = np.asarray(d["config"]["redesign"]["ring"]["center_m"][:2], float)
yaw = np.radians(d["config"]["redesign"]["ring"]["yaw_deg"])
ax = np.array([np.cos(yaw), np.sin(yaw)])
kink = ring_c - 0.06 * ax
short = np.vstack([np.linspace(CRADLE_EXIT, kink, 40), np.linspace(kink, ring_c + FEED_PAST_M * ax, 10)[1:]])
res["cases"]["shortcut"] = terms(sampled(short))
print(json.dumps(res))
open(out, "w").write(json.dumps(res, indent=1))
