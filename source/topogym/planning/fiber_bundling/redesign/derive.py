"""Inverse generation of the fiber_bundling redesign tiers (docs/rcb/redesign/fiber_bundling.md).

The oracle's motion comes first: a top-view route for the sheath, from the cradle exit to the front's final pose. A
free-bend tier's route is made of straight runs and bends (`route_points`). A formboard tier's route is the cable
pulled taut around its pegs (`taut_route`): the feed pulls the cable through the cradle, so only a peg that deflects
the taut line holds a bend. The scene follows from the route:
- the bundling ring sits on the route, FEED_PAST_M before its end, its axis on the route's tangent there;
- a formboard peg sits where the taut cable wraps it on the side the predicate requires (SEAT_GAP_M between peg and
  sheath surfaces); a peg that starts on the other side of the straight start cable marks a lay-over;
- the route is as long as the released feed pulls (RELEASED_PULL_M), plus what the S-route needs.
Then the derived scene is checked against the robot: where the hand holds the sheath at the end, its distance from
the Panda's shoulder; the taut cable with no pegs (what an unrouted feed leaves) must pass some peg on the wrong side
or cut through it.

Coordinates: the task origin (the trunk root on the cable axis, world (-0.1, -0.25, 1.1596) in the released scene),
x along the released cable, y to the robot's side, metres. No simulator here.

usage: derive.py <out_dir>    -> <out_dir>/tiers.json, <out_dir>/<tier>_route.png
"""
import json
import sys
from pathlib import Path

import numpy as np

FRONT_WORLD_X = 0.14                                        # the front face at the start, world x (kept for every tier)
AXIS_WORLD_YZ = np.array([-0.25, 1.1596])                   # the cable axis at the start, world y, z
SHOULDER_WORLD = np.array([-0.57, 0., 0.9196 + 0.333])     # Panda base + link1 height
CRADLE_EXIT = np.array([0.085, 0.])                         # cradle centre 0.02 + half its 0.13 m length
RELEASED_PULL_M = 0.14                                      # the released feed moves the front 0.14 m
FEED_PAST_M = 0.03                                          # the feed ends with the front 30 mm past the ring plane
GRASP_BEHIND_FRONT_M = 0.06                                 # the hand holds the second-to-last link's middle
TRUNK_R, PEG_R, SEAT_GAP_M = 0.012, 0.012, 0.002
PEG_TOP_M = 0.045                                           # a formboard peg's top above the cable axis (default)
CAP_M = 0.005          # a peg's head overhang: the combs and the lay roll ratchet a wrapped cable up a bare shaft and
                       # over its top (lab_layover_v7: over peg 1 during the roll); under a head it stays wrapped
FAN_R, FAN_L = 0.04, 0.08                                   # the strand fan at the start: radius, length
RING_OUTER_M = 0.0135 + 0.008                               # released ring: inner radius + tube diameter
RING_BASE = (0.10, 0.13)                                    # the ring mount's base plate (x, y), RingMount.parts
CRADLE_BASE = ((0.02, 0.), (0.15, 0.11))                    # the cradle's base plate: centre, size (origin frame)
# The released references hold the sheath 0.834 m from the shoulder at their feed's end: no tier asks for more.
REACH_LIMIT_M = 0.834


def route_points(segments, step=0.002):
    """Polyline of a route: from the cradle exit heading +x; ("line", L) or ("bend", radius, degrees, peg_side)."""
    p, h = CRADLE_EXIT.astype(float).copy(), 0.0
    pts, bends = [p.copy()], []
    for seg in segments:
        if seg[0] == "line":
            n = max(1, int(round(seg[1] / step)))
            for _ in range(n):
                p = p + seg[1] / n * np.array([np.cos(h), np.sin(h)])
                pts.append(p.copy())
        else:
            _, rad, deg = seg[:3]
            sgn = np.sign(deg)
            center = p + rad * sgn * np.array([-np.sin(h), np.cos(h)])
            bends.append(dict(center=center.copy(), radius=rad, degrees=deg, side="left" if sgn > 0 else "right"))
            n = max(2, int(round(abs(np.radians(deg)) * rad / step)))
            for k in range(1, n + 1):
                a = h + np.radians(deg) * k / n
                p = center + rad * sgn * np.array([np.sin(a), -np.cos(a)])
                pts.append(p.copy())
            h += np.radians(deg)
    return np.asarray(pts), bends, h


def _rot(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def taut_route(pegs, final_heading_deg, length, step=0.002):
    """The cable pulled taut from the cradle exit around formboard pegs, in order: straight runs tangent to each peg's
    circle (radius SEATED: peg + sheath + gap), an arc around each peg (a left turn where the peg is on the cable's
    left), and a last run along final_heading_deg until the route is `length` long. pegs: [(center_xy, side)].
    -> (polyline, per-peg turn angles in deg). The feed pulls the cable through the cradle, so the routed cable ends taut:
    a bend formed elsewhere is dragged off by the feed, a peg that deflects the taut line keeps it."""
    rho = SEATED
    sgn = lambda side: 1. if side == "left" else -1.
    pts = [CRADLE_EXIT.astype(float)]
    p = pts[0]
    turns = []
    # incoming tangent: from the point p to circle 0, passing with the peg on the given side
    c0, s0 = np.asarray(pegs[0][0], float), sgn(pegs[0][1])
    d = c0 - p; D = np.linalg.norm(d)
    u = _rot(d / D, -s0 * np.arcsin(rho / D))
    t_in = p + u * np.sqrt(D * D - rho * rho)
    heading_in = np.arctan2(u[1], u[0])
    seg_ends = [(t_in, heading_in)]
    for i in range(len(pegs)):
        c, s = np.asarray(pegs[i][0], float), sgn(pegs[i][1])
        if i + 1 < len(pegs):
            c2, s2 = np.asarray(pegs[i + 1][0], float), sgn(pegs[i + 1][1])
            d = c2 - c; D = np.linalg.norm(d); dh = d / D
            if s == s2:                                   # outer tangent: parallel to the centre line
                u = dh
                off = -s * np.array([-u[1], u[0]]) * rho  # the cable on the side away from the pegs
                a_out, a_in = c + off, c2 + off
            else:                                         # inner tangent (an S): crosses between the circles
                beta = np.arcsin(2 * rho / D)
                u = _rot(dh, s * beta)                       # turned towards the first peg's side
                n = -s * np.array([-u[1], u[0]])
                a_out, a_in = c + n * rho, c2 - n * rho
            h_out = np.arctan2(u[1], u[0])
        else:
            h_out = np.radians(final_heading_deg)
            u = np.array([np.cos(h_out), np.sin(h_out)])
            a_out = c - s * np.array([-u[1], u[0]]) * rho
        t_in, h_in = seg_ends[-1]
        turn = (h_out - h_in + np.pi) % (2 * np.pi) - np.pi
        if np.sign(turn) != s and abs(turn) > 1e-9:       # the cable must turn towards the peg's side
            turn += s * 2 * np.pi
        turns.append(float(np.degrees(turn)))
        # straight run to the arc's start, then the arc
        seg = np.linalg.norm(t_in - p)
        for k in range(1, max(1, int(seg / step)) + 1):
            pts.append(p + (t_in - p) * k / max(1, int(seg / step)))
        a0 = np.arctan2(*(t_in - c)[::-1])
        n_arc = max(2, int(abs(turn) * rho / step))
        for k in range(1, n_arc + 1):
            a = a0 + turn * k / n_arc
            pts.append(c + rho * np.array([np.cos(a), np.sin(a)]))
        p = pts[-1]
        if i + 1 < len(pegs):
            seg_ends.append((a_in, h_out))
    P = np.asarray(pts)
    used = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
    rest = max(length - used, 0.)
    u = np.array([np.cos(np.radians(final_heading_deg)), np.sin(np.radians(final_heading_deg))])
    n = max(1, int(rest / step))
    P = np.vstack([P, [p + u * rest * k / n for k in range(1, n + 1)]])
    return P, turns


def arclength(P):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def at(P, s):
    S = arclength(P)
    i = int(np.clip(np.searchsorted(S, s) - 1, 0, len(P) - 2))
    u = (s - S[i]) / max(S[i + 1] - S[i], 1e-12)
    p = P[i] + u * (P[i + 1] - P[i])
    t = (P[i + 1] - P[i]) / np.linalg.norm(P[i + 1] - P[i])
    return p, t


def derive(tier, segments, *, ring_inner_radius_m, lay=None, extra_pull_m=0.0, trunk_length_m=0.24, trunk_segments=6,
           taut=None):
    """Scene from route: ring pose, pegs, checks. taut: {"pegs": [(xy, side[, top_m])], "heading_deg": h} routes the
    cable taut around formboard pegs (taut_route) instead of `segments`."""
    FRONT0 = trunk_length_m
    ORIGIN_WORLD = np.r_[FRONT_WORLD_X - trunk_length_m, AXIS_WORLD_YZ]
    total = (FRONT0 - CRADLE_EXIT[0]) + RELEASED_PULL_M + extra_pull_m
    if taut:
        P, turns = taut_route(taut["pegs"], taut["heading_deg"], total)
        bends = [dict(center=np.asarray(peg[0], float), radius=SEATED, degrees=turn, side=peg[1],
                      top=peg[2] if len(peg) > 2 else PEG_TOP_M) for peg, turn in zip(taut["pegs"], turns)]
        segs = [["taut", taut["pegs"], taut["heading_deg"]]]
    else:
        used = sum(s[1] if s[0] == "line" else abs(np.radians(s[2])) * s[1] for s in segments)
        segs = list(segments) + [("line", max(total - used, 0.0))]
        P, bends, heading = route_points(segs)
    S = arclength(P)
    end_p, end_t = at(P, S[-1])
    ring_c, _ = at(P, S[-1] - FEED_PAST_M)
    yaw = float(np.degrees(np.arctan2(end_t[1], end_t[0])))
    hand, _ = at(P, S[-1] - GRASP_BEHIND_FRONT_M)
    hand_w = ORIGIN_WORLD + np.r_[hand, 0.]
    reach = float(np.linalg.norm(hand_w - SHOULDER_WORLD))
    pegs = []
    for i, b in enumerate(b for b in bends if b.get("radius") <= PEG_R + TRUNK_R + SEAT_GAP_M + 1e-9):
        pegs.append(dict(name=f"formboard_peg_{i}", xy_m=[round(float(b["center"][0]), 4), round(float(b["center"][1]), 4)],
                         radius_m=PEG_R, top_m=float(b.get("top", PEG_TOP_M)), side=b["side"], reach_m=0.02,
                         cap_m=CAP_M))
    # the shortcut: the taut cable with no pegs, from the cradle exit straight to where it must kink into the ring axis
    # (one and a half links before the ring centre): an unrouted feed's cable
    a, c = CRADLE_EXIT, ring_c - 0.06 * end_t
    short = []
    for peg in pegs:
        q = np.asarray(peg["xy_m"])
        ab = c - a
        t = float(np.clip(np.dot(q - a, ab) / np.dot(ab, ab), 0., 1.))
        d = float(np.linalg.norm(q - (a + t * ab)))
        cross = ab[0] * (q - (a + t * ab))[1] - ab[1] * (q - (a + t * ab))[0]
        side = "left" if cross > 0 else "right"
        short.append(dict(peg=peg["name"], shortcut_clearance_m=round(d - PEG_R - TRUNK_R, 4), shortcut_side=side,
                          blocks_or_wrong=bool(d < PEG_R + TRUNK_R or side != peg["side"])))
    # Fixtures keep clear of the start state (the straight sheath on y = 0 up to the front, the strand fan beyond it),
    # of the pickup (the open Robotiq around the grasped link: +-0.06 m along the sheath, 0.07 m across), of the
    # cradle and of each other.
    def start_gap(q, r):
        x, y = q
        sheath = abs(y) - TRUNK_R if 0. <= x <= FRONT0 else np.hypot(min(abs(x), abs(x - FRONT0)), y) - TRUNK_R
        fan = np.hypot(max(0., x - FRONT0 - FAN_L, FRONT0 - x), max(0., abs(y) - FAN_R)) if x >= FRONT0 - 0.01 else 1.
        return float(min(sheath, fan) - r)
    grasp_x = FRONT0 - GRASP_BEHIND_FRONT_M
    clear = []
    for peg in pegs:
        q = np.asarray(peg["xy_m"])
        pick = max(abs(q[0] - grasp_x) - 0.06, abs(q[1]) - 0.07) - PEG_R
        start_side = "left" if q[1] > 0 else "right"         # the start cable runs along y = 0, heading +x
        clear.append(dict(peg=peg["name"], start_gap_m=round(start_gap(q, PEG_R), 4), pickup_gap_m=round(float(pick), 4),
                          start_side=start_side, lay_over=bool(start_side != peg["side"])))
    ring_gap = start_gap(ring_c, RING_OUTER_M)
    def boxes_overlap(c1, s1, c2, s2):
        return all(abs(a - b) < (u + v) / 2 for a, b, u, v in zip(c1, c2, s1, s2))
    base_hits = [peg["name"] for peg in pegs if boxes_overlap(ring_c, RING_BASE, peg["xy_m"], (2 * PEG_R, 2 * PEG_R))]
    if boxes_overlap(ring_c, RING_BASE, *CRADLE_BASE):
        base_hits.append("cradle_base")
    geometry = dict(fiber_count=16, ring_inner_radius_m=ring_inner_radius_m)
    if trunk_length_m != 0.24:
        geometry.update(trunk_length_m=trunk_length_m, trunk_segments=trunk_segments)
    cfg = dict(geometry=geometry, center_xy_m=[round(float(ORIGIN_WORLD[0]), 4), float(ORIGIN_WORLD[1])],
               redesign=dict(ring=dict(yaw_deg=round(yaw, 2), center_m=[round(float(ring_c[0]), 4), round(float(ring_c[1]), 4), 0.]),
                             pegs=pegs, lay=lay))
    checks = dict(route_length_m=round(float(S[-1]), 4), pull_m=round(float(S[-1]) - (FRONT0 - CRADLE_EXIT[0]), 4),
                  final_heading_deg=round(yaw, 2), hand_world=[round(float(x), 4) for x in hand_w],
                  hand_to_shoulder_m=round(reach, 4), reach_ok=bool(reach <= REACH_LIMIT_M + 1e-9),
                  ring_world=[round(float(x), 4) for x in ORIGIN_WORLD + np.r_[ring_c, 0.]],
                  ring_start_gap_m=round(ring_gap, 4), ring_base_overlaps=base_hits, pegs_clear=clear,
                  fixtures_ok=bool(ring_gap > 0.005 and not base_hits and all(c["start_gap_m"] > 0.0015 and c["pickup_gap_m"] > 0.
                                                                             for c in clear)),
                  shortcut=short)
    return dict(tier=tier, segments=[list(s) for s in segs], config=cfg, checks=checks, polyline=P.round(5).tolist())


# Lay targets: the red fiber starts at -90 deg; each target needs a roll that the arm can hold through the whole feed
# at that tier's ring (the oracle's IK probes, after the route and combs: the roll range at the feed pose, released
# ring about +-60, easy's -48..+68, medium's -70..+46, hard's -52..+42 deg; the feed with the roll held completes for
# -45..+30 deg at easy's ring, -60..+30 at medium's, -60..+15 at hard's), and lies >= 40 deg from -90 (where an oracle
# that never rolls leaves it), at least twice its tolerance. At the turned rings only negative rolls qualify.
# The oracle routes (inverse generation's first step). Bend radii: a formboard bend is seated on its peg
# (PEG_R + TRUNK_R + SEAT_GAP_M); a free bend uses 0.05 m.
SEATED = PEG_R + TRUNK_R + SEAT_GAP_M
ROUTES = {
    "very_easy": dict(segments=[], ring_inner_radius_m=0.0135, lay=None),
    "easier": dict(segments=[], ring_inner_radius_m=0.0135, lay=dict(strand=0, angle_deg=-30., tolerance_deg=20.)),
    "easy": dict(segments=[("line", 0.035), ("bend", 0.05, 30.)], ring_inner_radius_m=0.0128,
                 lay=dict(strand=0, angle_deg=-130., tolerance_deg=20.)),
    # Formboard tiers: the cable routed taut around the pegs (the feed pulls it through the cradle, so only a peg that
    # deflects the taut line holds a bend; a peg already on its required side at the start would be wrapped by any pull
    # toward the ring). Medium's peg stands 6 mm right of the start cable and must end on the cable's left: the cable
    # is lifted over it and wraps it from below (a lay-over and a 67 deg seated detour). Hard: peg 0 6 mm right of the
    # start cable stays on its right (an S's first bend), peg 1 further right ends on its left (the lay-over).
    # Hard's peg 0 is a tall post (top 0.12 m above the axis): the open cradle lets a lifted cable rise along its whole
    # length (pivoting ~0.45 m behind the root: at peg 0 it rises ~0.8x the hand's lift, at peg 1 ~0.9x; lab_layover_v3),
    # so a lift that clears peg 1 (45 mm) would clear a 45 mm peg 0 too and the lay-over would drop the cable on peg 0's
    # wrong side. The post keeps the S's first bend: the lay-over crosses peg 1 only (hand lift ~0.10 m).
    "medium": dict(segments=[], taut=dict(pegs=[((0.20, -0.03), "left")], heading_deg=40.), ring_inner_radius_m=0.0128,
                   lay=dict(strand=0, angle_deg=-140., tolerance_deg=15.), trunk_length_m=0.36, trunk_segments=9),
    # Hard v2 (2026-10-11): the post's top 0.085 m and peg 1's 0.035 m (were 0.12 / 0.045). At the comb pose the
    # Panda's wrist (link7, rigid with the hand) passes over the post, and the gripper's spring link over peg 1's head:
    # the 0.12 m post's head blocked every comb roll below -20 deg (needed by the strands on that side) and the lay roll
    # at the standoff, and peg 1's head took 40-50 N from the gripper in roll-0 presses (hard16 lost its grasp). The
    # oracle's kinematic sweep (oracle.py --lay-sweep) gives clearance down to -30 deg there with these heights. The
    # lay-over lift becomes ~0.075 m: the cable rises ~0.06 m at the post (its top 0.085) and ~0.068 m at peg 1 (0.035).
    "hard": dict(segments=[], taut=dict(pegs=[((0.12, -0.03), "right", 0.085), ((0.20, -0.06), "left", 0.035)], heading_deg=40.),
                 ring_inner_radius_m=0.0126, lay=dict(strand=0, angle_deg=-130., tolerance_deg=12.), extra_pull_m=0.05,
                 trunk_length_m=0.36, trunk_segments=9),
}


def plot(d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = np.asarray(d["polyline"])
    cfg = d["config"]["redesign"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.add_patch(plt.Rectangle((0.02 - 0.065, -0.021), 0.13, 0.042, color="0.6", alpha=.5, label="cradle"))
    front0 = d["config"]["geometry"].get("trunk_length_m", 0.24)
    ax.plot([0, front0], [0, 0], color="0.4", lw=6, alpha=.35, label="sheath at start")
    ax.plot(P[:, 0], P[:, 1], color="tab:blue", lw=2, label="oracle route (sheath centre line)")
    c = np.asarray(cfg["ring"]["center_m"][:2]); yaw = np.radians(cfg["ring"]["yaw_deg"])
    n = np.array([-np.sin(yaw), np.cos(yaw)])
    r = d["config"]["geometry"]["ring_inner_radius_m"] + 0.004
    ax.plot(*np.c_[c - r * n, c + r * n], color="tab:orange", lw=4, label="ring (top view)")
    for peg in cfg["pegs"]:
        ax.add_patch(plt.Circle(peg["xy_m"], peg["radius_m"], color="0.25"))
        ax.annotate(f'{peg["name"][-5:]} (top {1e3 * peg["top_m"]:.0f} mm): cable on its '
                    f'{"right" if peg["side"] == "left" else "left"}', peg["xy_m"],
                    textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.plot([CRADLE_EXIT[0], c[0]], [CRADLE_EXIT[1], c[1]], "k--", lw=1, label="straight shortcut")
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=7, loc="upper left")
    ax.set_title(f'{d["tier"]}: route {d["checks"]["route_length_m"]} m, ring yaw {cfg["ring"]["yaw_deg"]} deg, '
                 f'hand-shoulder {d["checks"]["hand_to_shoulder_m"]} m', fontsize=8)
    ax.set_xlabel("x from the trunk root (m)"); ax.set_ylabel("y (m, robot side +)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    res = {}
    for tier, spec in ROUTES.items():
        d = derive(tier, spec["segments"], ring_inner_radius_m=spec["ring_inner_radius_m"], lay=spec["lay"],
                   extra_pull_m=spec.get("extra_pull_m", 0.), trunk_length_m=spec.get("trunk_length_m", 0.24),
                   trunk_segments=spec.get("trunk_segments", 6), taut=spec.get("taut"))
        plot(d, out / f"{tier}_route.png")
        res[tier] = {k: v for k, v in d.items() if k != "polyline"}
    (out / "tiers.json").write_text(json.dumps(res, indent=1))
    for tier, d in res.items():
        print(tier, json.dumps(d["config"]["redesign"]), json.dumps(d["checks"]))


if __name__ == "__main__":
    main(sys.argv[1])
