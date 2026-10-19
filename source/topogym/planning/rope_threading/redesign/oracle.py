"""Inverse-generation oracle for the redesigned rope_threading tiers: the planned hand motion that the scenes
(topogym/tasks/rope_threading/redesign/scenes.py) were generated from, checked kinematically on the ALOHA.

The plan is a relay over the tier's legs (one per ring, in order): the feeder holds the rope FEED_STATION_M behind the tip
on the ring's axis line and pushes until the tip is POKE_M past the ring plane; the receiver pinches the tip
RECEIVE_STATION_M from its end beyond the ring (another arm: a hand-over; the same arm after letting go and walking around
the ring: a same-hand regrasp); the receiver then draws the tip along the goal centreline to 5 cm before the next ring
(follow-the-leader: the rope slides along the goal path, the object model of the family's old object_oracle). Before a
feed the hand that holds the tip regrasps at the feeding station, with a keeper hand holding the rope on its own side
where one is named.

Kinematic checks (laptop numpy, aloha_kin): every waypoint has an IK solution with the rope between the pads (closing
axis perpendicular to the rope), within joint limits, and the arm's conservative capsules clear the table and every
fixture; consecutive solutions are continuous (the joint path is reported). `forced_handover` shows, per grommet, that
the feeding arm has no collision-free pinch pose beyond the bulkhead (over a sphere of gripper orientations) while the
receiving arm has one. Physics is checked separately (mj_object_demo.py on the cloud host).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from topogym.planning.rope_threading.redesign import scenes as S
from . import aloha_kin as K
from topogym.planning.rope_threading.redesign.legs import legs_for as _legs_for

HOLD_SPEED = 0.12          # m/s TCP while holding rope (the old relay's --speed .12)
FREE_SPEED = 0.25          # m/s TCP free (--free-speed .25)
GRASP_DWELL_S = 0.6        # close + acquire (contact_weld_v1 stable_time 0.02 s; the relay waited for the solve)
RELEASE_DWELL_S = 0.4
APPROACH_OFFSET = 0.07     # pre-grasp / retreat distance along the gripper's approach axis
DRAW_MIN_Z = 0.11          # a hand drawing the tip keeps this height over the table
ARM_CLEAR = 0.008          # required clearance of the conservative arm capsules from fixtures / table (mesh check: hulls up to ~4 mm outside)
FINGER_CLEAR = 0.006       # finger capsules vs fixtures (mesh check v1 found the finger hulls 1-2 mm beyond the capsules)
WRIST_ANGLE_MAX = 2.0     # rad: core excludes the forearm-gripper_base pair, so no wrist_angle beyond ~116 deg


# ----------------------------------------------------------------------------------------------- geometry helpers
def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def grasp_rotations(tangent, n=16, needle=True):
    """Gripper rotations that hold a rope of direction `tangent` between the pads: cross grasps (approach
    perpendicular to the rope, sampled around it) and needle grasps (approach along the rope, closing axis sampled)."""
    t = _unit(tangent)
    w1 = np.cross(t, [0, 0, 1.0])
    if np.linalg.norm(w1) < 1e-6:
        w1 = np.array([1.0, 0, 0])
    w1 = _unit(w1)
    w2 = np.cross(t, w1)
    out = []
    for k in range(n):
        phi = 2 * np.pi * k / n
        a = math.cos(phi) * w1 + math.sin(phi) * w2
        Rc = K.gripper_frame(a, rope_tangent=t)
        out.append(("cross", Rc))
        # the same pinch rolled 180 deg about the approach axis (the wrist camera on the other side)
        out.append(("cross_flipped", Rc @ np.diag([1.0, -1.0, -1.0])))
    if needle:
        for sgn in (1.0, -1.0):          # fingers pointing along the rope's travel, or against it (at a hole)
            for k in range(8):
                phi = 2 * np.pi * k / 8
                c = math.cos(phi) * w1 + math.sin(phi) * w2
                out.append(("needle" if sgn > 0 else "needle_reverse", K.gripper_frame(sgn * t, closing=c)))
    return out


TABLE_TOUCH_FINGERS = 0.020 + (K.FINGER_RADIUS - 0.011)   # fingertips may touch the table when pinching a rope lying
# on it (pads at rope height); grown with the finger capsule (coordinator 10-10) so a pick stays the same pose


_BOUNDS = {}


def _fixture_bounds(design):
    """Per design: the fixtures and rings with a bounding sphere each (centre, radius), cached."""
    key = id(design)
    if key not in _BOUNDS:
        items = []
        for f in design.fixtures():
            if isinstance(f, S.Box):
                items.append((f, f.center, float(np.linalg.norm(f.half))))
            else:
                items.append((f, 0.5 * (f.a + f.b), 0.5 * float(np.linalg.norm(f.b - f.a)) + f.radius))
        rings = [(r, r.center, r.outer_radius) for r in design.rings]
        _BOUNDS[key] = (items, rings)
    return _BOUNDS[key]


def fixture_clearance(caps, design, skip_rings=()):
    """Minimum clearance of arm capsules to every fixture and the table; returns (min gap, name, capsule). Fixtures
    whose bounding sphere cannot beat the current minimum are skipped (exact result)."""
    best = (np.inf, None, None)
    items, rings = _fixture_bounds(design)
    for cname, a, b, r in caps:
        pts = a + np.linspace(0, 1, 9)[:, None] * (b - a)
        touch = 0.0
        if cname.startswith("finger"):
            # coordinator 10-10: the fingertips may touch the table only in a pick pose (the TCP, 0.06 / 0.075 along
            # the finger capsule, within 3 cm of the table): the real-mesh check found fingers 3.7 mm in the table
            # while drawing the tip (an allowance meant for picks applied to every pose)
            z_tcp = float(a[2] + (0.13 - K.FINGER_LEN[0]) / (K.FINGER_LEN[1] - K.FINGER_LEN[0]) * (b[2] - a[2]))
            touch = TABLE_TOUCH_FINGERS if z_tcp < 0.03 else 0.0
        zgap = float(np.min(pts[:, 2])) - r + touch
        if cname != "base" and zgap < best[0]:
            best = (zgap, "table", cname)
        mid, half = 0.5 * (a + b), 0.5 * float(np.linalg.norm(b - a))
        for f, c, rad in items:
            if float(np.linalg.norm(c - mid)) - rad - half - r >= best[0]:
                continue
            if isinstance(f, S.Box):
                g = float(np.min(f.distance(pts))) - r
            else:
                g = K.segment_distance(a, b, f.a, f.b) - r - f.radius
            if g < best[0]:
                best = (g, f.name, cname)
        for ring, c, rad in rings:
            if ring.name in skip_rings or float(np.linalg.norm(c - mid)) - rad - half - r >= best[0]:
                continue
            g = float(np.min(ring.distance(pts))) - r
            if g < best[0]:
                best = (g, ring.name, cname)
    return best


def config_clear(arm, q, design, opening, other=None, skip_rings=()):
    """Capsule clearance of one arm configuration: (ok, arm gap, finger gap, other-arm gap)."""
    caps = K.arm_capsules(arm, q, opening)
    arm_caps = [c for c in caps if not c[0].startswith("finger")]
    fing = [c for c in caps if c[0].startswith("finger")]
    g_arm = fixture_clearance(arm_caps, design, skip_rings)
    g_f = fixture_clearance(fing, design, skip_rings)
    g_o = (np.inf, None, None)
    if other is not None:
        for n1, a1, b1, r1 in caps:
            for n2, a2, b2, r2 in other:
                g = K.segment_distance(a1, b1, a2, b2) - r1 - r2
                if g < g_o[0]:
                    g_o = (g, n2, n1)
    g_s = self_clearance(caps)
    if g_s[0] < g_o[0]:
        g_o = g_s
    ok = g_arm[0] >= ARM_CLEAR and g_f[0] >= FINGER_CLEAR and g_o[0] >= SELF_CLEAR
    if ok:
        # the real hulls too, when RD_MESH_SCENE is set (the cloud host): the capsules missed 5-12 mm of the gripper / base hulls
        from .mesh_validity import get as _mesh
        mv = _mesh(design)
        if mv is not None:
            other_q = getattr(other, "q", None) if other is not None else None
            tcp_z = float(K.tcp_pose(arm, q)[2, 3])
            mok, worst = mv.check(arm, q, opening, other_q=_OTHER_Q.get("left" if arm == "right" else "right"),
                                  allow_table_fingers=tcp_z < 0.03)
            if not mok:
                ok = False
                g_arm = (worst[0] - mv.clear, worst[2], "mesh:" + str(worst[1]))
    return ok, g_arm, g_f, g_o


_OTHER_Q = {}      # the other arm's current joints for the real-hull check (realize() keeps it up to date)


SELF_CLEAR = 0.005
_SELF_PAIRS = [(a, b) for a in ("wrist", "gripper_mount", "gripper_rail", "camera", "finger_l", "finger_r")
               for b in ("base", "shoulder", "upper_arm")] + \
              [(a, "forearm") for a in ("camera", "finger_l", "finger_r")]


def self_clearance(caps):
    """Smallest gap between non-adjacent capsules of one arm (coordinator 10-10: the RRT vias folded an upper arm onto
    the base, 26 mm in the real-mesh check; the plan checks had no self term). Returns (gap, name, name)."""
    by = {c[0]: c for c in caps}
    best = (np.inf, None, None)
    for a, b in _SELF_PAIRS:
        if a in by and b in by:
            _, a1, b1, r1 = by[a]
            _, a2, b2, r2 = by[b]
            g = K.segment_distance(a1, b1, a2, b2) - r1 - r2
            if g < best[0]:
                best = (g, a, b)
    return best


GRASP_KINDS = {"cross": np.array([0, 0, -1.0]), "cross_flipped": np.array([0, 0, 1.0]),
               "needle": np.array([1.0, 0, 0]), "needle_reverse": np.array([-1.0, 0, 0])}


def held_kind(R, tangent):
    """Which grasp family a gripper rotation R holds a rope of (tip-ward) direction `tangent` in: the tip direction in
    the gripper frame is fixed by the grasp (rolling the hand 180 deg would swing the rope around)."""
    h = np.asarray(R).T @ _unit(tangent)
    return max(GRASP_KINDS, key=lambda k: float(h @ GRASP_KINDS[k]))


def solve_pose(arm, pos, tangent, design, q_prev=None, rot_prev=None, prefer_down=True, opening=0.02,
               allow_needle=True, skip_rings=(), other=None, good_step=0.35, kinds=None):
    """IK for a TCP at `pos` holding a rope of direction `tangent` (any grasp rotation about the rope). Among the
    collision-free solutions the one closest to q_prev (branch continuity) is returned (early exit below good_step
    rad); else the best infeasible one (feasible False)."""
    cands = grasp_rotations(tangent, needle=allow_needle)
    if kinds is not None:
        cands = [(k, R) for k, R in cands if k in kinds]
    scored = []
    for kind, R in cands:
        score = 0.0
        if rot_prev is not None:
            score += float(np.linalg.norm(K.Rotation.from_matrix(rot_prev.T @ R).as_rotvec()))
        if prefer_down:
            score += 0.3 * (1 + R[2, 0])          # approach pointing down is cheapest
        scored.append((score, kind, R))
    scored.sort(key=lambda x: x[0])
    best_bad, best_ok = None, None
    n_ok = 0
    for score, kind, R in scored:
        q, pe, re = K.ik(arm, pos, rotation=R, seed=q_prev, fast=True)
        if pe > 0.002 or re > 0.03:
            if best_bad is None or pe + re < best_bad["pos_err"] + best_bad["rot_err"]:
                best_bad = dict(feasible=False, q=q, R=R, kind=kind, pos_err=pe, rot_err=re, why="ik")
            continue
        if q[4] > WRIST_ANGLE_MAX:
            # core excludes the forearm-gripper_base pair: no wrist_angle beyond ~116 deg (main, 10-10)
            if best_bad is None or best_bad.get("why") == "ik":
                best_bad = dict(feasible=False, q=q, R=R, kind=kind, pos_err=pe, rot_err=re, why="wrist_angle")
            continue
        ok, g_arm, g_f, g_o = config_clear(arm, q, design, opening, other, skip_rings)
        res = dict(feasible=ok, q=q, R=R, kind=kind, pos_err=pe, rot_err=re, arm_gap=g_arm[0], arm_gap_with=g_arm[1:],
                   finger_gap=g_f[0], finger_gap_with=g_f[1:], other_gap=g_o[0], why=None if ok else "collision")
        if ok:
            step = float(np.max(np.abs(q - q_prev))) if q_prev is not None else 0.0
            res["joint_step"] = step
            n_ok += 1
            if best_ok is None or step < best_ok["joint_step"]:
                best_ok = res
            if q_prev is None or step < good_step or n_ok >= 4:
                break
            continue
        if best_bad is None or best_bad.get("why") == "ik" or min(g_arm[0], g_f[0]) > min(best_bad.get("arm_gap", -1),
                                                                                            best_bad.get("finger_gap", -1)):
            best_bad = res
    return best_ok if best_ok is not None else best_bad


def segment_clear(arm, qa, qb, design, opening, other=None, step=0.03):
    """Capsule check of the joint-space segment qa -> qb (interior samples every `step` rad of the largest joint)."""
    n = max(2, int(math.ceil(float(np.max(np.abs(qb - qa))) / step)))
    worst = None
    for s_ in np.linspace(0, 1, n + 1)[1:-1]:
        ok, g_arm, g_f, g_o = config_clear(arm, qa + s_ * (qb - qa), design, opening, other)
        if not ok:
            gap = min(g_arm[0], g_f[0], g_o[0])
            if worst is None or gap < worst[0]:
                worst = (gap, g_arm[1:] if g_arm[0] == gap else (g_f[1:] if g_f[0] == gap else g_o[1:]), float(s_))
    return worst is None, worst


# ----------------------------------------------------------------------------------------------- the relay plan
class Planner:
    def __init__(self, design: S.TierDesign):
        self.d = design
        self.G = design.goal
        self.sG = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.G, axis=0), axis=1))]
        self.L = float(self.sG[-1])
        self.rings = sorted(design.rings, key=lambda r: r.s_goal)
        self.phases = []
        self.holding = {"left": None, "right": None}     # material station (distance from the tip) or None
        self.q = {"left": K.HOME.copy(), "right": K.HOME.copy()}
        self.tcp = {arm: K.tcp_pose(arm, K.HOME) for arm in ("left", "right")}

    # the rope's material point sigma (from the tip) when the tip is at goal arc length p (follow-the-leader)
    def at(self, p, sigma):
        s = float(np.clip(p - sigma, 0, self.L))
        i = int(np.clip(np.searchsorted(self.sG, s), 1, len(self.G) - 1))
        w = (s - self.sG[i - 1]) / max(self.sG[i] - self.sG[i - 1], 1e-12)
        pos = self.G[i - 1] * (1 - w) + self.G[i] * w
        tan = _unit(self.G[min(i + 1, len(self.G) - 1)] - self.G[max(i - 2, 0)])
        return pos, tan

    def start_at(self, sigma):
        """The start rope's point `sigma` from the tip (material end) and its tangent (towards the tip)."""
        P = self.d.start
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        target = s[-1] - sigma
        i = int(np.clip(np.searchsorted(s, target), 1, len(P) - 1))
        w = (target - s[i - 1]) / max(s[i] - s[i - 1], 1e-12)
        return P[i - 1] * (1 - w) + P[i] * w, _unit(P[i] - P[i - 1])

    def phase(self, name, kind, hand, poses, gripper=None, ring=None, note="", sigma=None):
        """poses: list of (position, rope tangent or None (keep orientation), speed). sigma: the material station
        (distance from the tip) the phase's last pose grasps, for measured targeting."""
        self.phases.append(dict(name=name, kind=kind, hand=hand, poses=poses, gripper=gripper, ring=ring, note=note,
                                sigma=sigma))

    def plan(self):
        d, R = self.d, self.rings
        legs = {leg["ring"]: leg for leg in _legs_for(d)}
        st, poke, rcv, appr = S.FEED_STATION_M, S.POKE_M, S.RECEIVE_STATION_M, S.FEED_APPROACH_M
        first = R[0]
        feeder0 = legs[first.name]["feeder"]
        # pick up the start rope at the feeding station
        p0, t0 = self.start_at(st)
        p0 = p0 + np.array([0.0, 0.0, 0.014 - p0[2]])          # pads at the rope, fingertips on the table
        self.phase("approach start rope", "approach", feeder0, [(p0 + [0, 0, APPROACH_OFFSET], t0, FREE_SPEED),
                                                               (p0, t0, FREE_SPEED * 0.5)], sigma=st)
        self.phase("grasp", "grasp", feeder0, [], gripper="close")
        self.holding[feeder0] = st
        pf0 = first.center - (st + appr) * _unit(first.travel)
        lift = p0 + [0, 0, 0.10]
        mid = 0.5 * (lift + pf0) + [0, 0, 0.05]
        self.phase("lift and transport tip to the first feed line", "transport", feeder0,
                   [(lift, t0, HOLD_SPEED), (mid, _unit(0.5 * (t0 + _unit(first.travel))), HOLD_SPEED),
                    (pf0, _unit(first.travel), HOLD_SPEED)])
        tip_holder, tip_sigma = feeder0, st
        p_tip = first.s_goal - appr
        for k, ring in enumerate(R):
            leg = legs[ring.name]
            F, Rc, Kp = leg["feeder"], leg["receiver"], leg.get("keeper")
            dvec = _unit(ring.travel)
            # 1) the feeder must hold the station behind the tip on this ring's lead-in line
            if not (tip_holder == F and abs(tip_sigma - st) < 1e-9):
                station, stan = self.at(p_tip, st)
                if Kp is not None:
                    kp_pos, kp_tan, kp_sigma = self.keeper_point(Kp, p_tip)
                    self.phase(f"keeper grasp ({ring.name})", "approach", Kp, [(kp_pos + self.retreat_vec(Kp, kp_tan, at=kp_pos), kp_tan, FREE_SPEED),
                                                                              (kp_pos, kp_tan, FREE_SPEED * 0.5)], sigma=kp_sigma)
                    self.phase("keeper close", "grasp", Kp, [], gripper="close")
                    self.holding[Kp] = "keeper"
                if tip_holder == F:
                    self.phase(f"release tip for regrasp ({ring.name})", "release", F, [], gripper="open")
                    self.holding[F] = None
                    cur = self.tcp_pos(F)
                    self.phase(f"regrasp at feeding station ({ring.name})", "regrasp", F,
                               [(cur + self.retreat_vec(F, stan, at=cur), None, FREE_SPEED),
                                (station + self.retreat_vec(F, stan, at=station), stan, FREE_SPEED), (station, stan, FREE_SPEED * 0.5)],
                               sigma=st)
                    self.phase("close", "grasp", F, [], gripper="close", ring=ring.name)
                else:
                    self.phase(f"second hand takes the feeding station ({ring.name})", "approach", F,
                               [(station + self.retreat_vec(F, stan, at=station), stan, FREE_SPEED), (station, stan, FREE_SPEED * 0.5)],
                               sigma=st)
                    self.phase("close", "grasp", F, [], gripper="close", ring=ring.name)
                    self.phase("previous holder lets go", "release", tip_holder, [], gripper="open")
                    self.holding[tip_holder] = None
                    self.phase("previous holder retreats", "retreat", tip_holder,
                               [(self.tcp_pos(tip_holder) + self.retreat_vec(tip_holder, None), None, FREE_SPEED)])
                self.holding[F] = st
                if Kp is not None:
                    self.phase("keeper lets go", "release", Kp, [], gripper="open")
                    self.holding[Kp] = None
                    self.phase("keeper retreats", "retreat", Kp,
                               [(self.tcp_pos(Kp) + self.retreat_vec(Kp, None), None, FREE_SPEED), (self.park(Kp), None, FREE_SPEED)])
                tip_holder, tip_sigma = F, st
            # 2) push along the axis: the tip from appr before the plane to poke past it
            pa = ring.center - (st + appr) * dvec
            pb = ring.center - (st - poke) * dvec
            cur = self.tcp_pos(F)
            if k > 0 and float(np.linalg.norm(cur - pa)) > 0.02:
                # line the rope up on the ring's axis, let go and re-grip from above: the grasp taken where the rope
                # still curves may not be one the push can end with (the hand next to a wall or ring)
                self.phase(f"line up for {ring.name}", "transport", F, [(pa, dvec, HOLD_SPEED)], ring=ring.name)
                self.phase(f"let go to re-grip ({ring.name})", "release", F, [], gripper="open", ring=ring.name)
                self.phase(f"re-grip on {ring.name}'s axis", "regrasp", F,
                           [(pa + np.array([0.0, 0.0, 0.05]), dvec, FREE_SPEED), (pa, dvec, FREE_SPEED * 0.5)],
                           ring=ring.name, sigma=st)
                self.phase("close", "grasp", F, [], gripper="close", ring=ring.name)
            self.phase(f"feed {ring.name}", "feed", F, [(pa, dvec, HOLD_SPEED), (pb, dvec, HOLD_SPEED * 0.6)], ring=ring.name)
            p_tip = ring.s_goal + poke
            # 3) the receiver pinches the tip beyond the ring
            pr = ring.center + (poke - rcv) * dvec
            if Rc != F:
                self.phase(f"receiver approaches beyond {ring.name}", "approach", Rc,
                           [(pr + self.retreat_vec(Rc, dvec, ring, at=pr), dvec, FREE_SPEED), (pr, dvec, FREE_SPEED * 0.5)], ring=ring.name,
                           sigma=rcv)
                self.phase("receiver closes", "grasp", Rc, [], gripper="close", ring=ring.name)
                self.phase("feeder lets go (hand-over)", "release", F, [], gripper="open", ring=ring.name)
                self.holding[F] = None
                self.phase("feeder retreats", "retreat", F,
                           [(self.tcp_pos(F) - 0.06 * dvec + (self.retreat_vec(F, None, at=self.tcp_pos(F)) * 0.6), None, FREE_SPEED),
                            (self.park(F), None, FREE_SPEED)])
            else:
                self.phase(f"feeder lets go in front of {ring.name}", "release", F, [], gripper="open", ring=ring.name)
                self.holding[F] = None
                back = self.tcp_pos(F) - 0.06 * dvec
                around = self.around_point(F, ring)
                self.phase(f"walk around {ring.name}", "regrasp", F,
                           [(back, None, FREE_SPEED), (around, None, FREE_SPEED),
                            (pr + self.retreat_vec(F, dvec, ring, at=pr), dvec, FREE_SPEED), (pr, dvec, FREE_SPEED * 0.5)], ring=ring.name,
                           sigma=rcv)
                self.phase("close beyond the ring", "grasp", F, [], gripper="close", ring=ring.name)
            self.holding[Rc] = rcv
            tip_holder, tip_sigma = Rc, rcv
            # 4) draw the tip along the goal to 5 cm before the next ring (or to the end)
            nxt = R[k + 1] if k + 1 < len(R) else None
            p_end = (nxt.s_goal - S.DRAW_STOP_M) if nxt is not None else self.L
            ps = np.linspace(p_tip, p_end, max(2, int(math.ceil((p_end - p_tip) / 0.03)) + 1))[1:]
            poses = []
            for p in ps:
                pos, tan = self.at(p, rcv)
                # the drawing hand need not hug the goal curve: it stays >= DRAW_MIN_Z over the table (low poses
                # force wrist turns next to the table), the rope behind it just sags a little more
                pos = np.array([pos[0], pos[1], max(pos[2], DRAW_MIN_Z)])
                poses.append((pos, tan, HOLD_SPEED))
            self.phase(f"draw tip {'to ' + nxt.name if nxt else 'to the final overhang'}", "transport", Rc, poses,
                       ring=ring.name)
            p_tip = p_end
        self.phase("hold until quiet", "hold", tip_holder, [], note="holding is allowed; release not required")
        return self.phases

    # --- helpers that choose free-space points
    def tcp_pos(self, arm):
        for ph in reversed(self.phases):
            if ph["hand"] == arm and ph["poses"]:
                return np.asarray(ph["poses"][-1][0], float)
        return self.tcp[arm][:3, 3].copy()

    def covered(self, p, margin=0.03):
        """Is point p under a shelf (inside its footprint + margin, below it)?"""
        for o in self.d.obstacles:
            if o.role == "shelf" and isinstance(o, S.Box):
                if (abs(p[0] - o.center[0]) < o.half[0] + margin and abs(p[1] - o.center[1]) < o.half[1] + margin
                        and p[2] < o.center[2]):
                    return True
        return False

    def retreat_vec(self, arm, tangent, ring=None, at=None):
        """Pre-grasp offset: mostly up, plus away from the bulkhead towards the arm's own side; sideways (towards the
        arm's side) next to a vertical eyelet or under a shelf."""
        side = -1.0 if arm == "left" else 1.0
        v = np.array([0.03 * side, 0.0, APPROACH_OFFSET])
        if ring is not None and abs(_unit(ring.travel)[2]) > 0.7:      # vertical eyelet: approach from the side
            v = np.array([0.06 * side, 0.0, 0.0])
        if at is not None and self.covered(at):
            v = np.array([0.07 * side, -0.02, 0.0])
        return v

    def around_point(self, arm, ring):
        """A free point to pass a ring when walking around it: beside it on the arm's side (vertical eyelets and rings
        under a shelf), else above its outer rim."""
        d = _unit(ring.travel)
        side = -1.0 if arm == "left" else 1.0
        if abs(d[2]) > 0.7:                # vertical eyelet: beside it, below the rim level
            return ring.center + np.array([0.11 * side, 0.0, -0.06])
        if self.covered(ring.center):
            h = np.cross(d, [0, 0, 1.0])
            h = _unit(h) * (1 if h[0] * side >= 0 else -1)
            return ring.center + (ring.outer_radius + 0.08) * h
        return ring.center + np.array([0.0, 0.0, ring.outer_radius + 0.07]) + 0.03 * side * np.array([1, 0, 0])

    def keeper_point(self, arm, p_tip):
        """The keeper holds the rope on its own half: the last goal point before the first grommet on its side."""
        side = -1.0 if arm == "left" else 1.0
        best = None
        for s in np.arange(0.05, max(p_tip, 0.06), 0.01):
            pos, tan = self.at(p_tip, p_tip - s)
            if side * pos[0] > 0.08 and pos[2] > 0.03:
                best = (pos, tan, float(p_tip - s))
        if best is None:
            pos, tan = self.at(p_tip, p_tip - 0.05)
            best = (pos, tan, float(p_tip - 0.05))
        return best

    def park(self, arm):
        side = -1.0 if arm == "left" else 1.0
        return np.array([0.25 * side, -0.15, 0.28])


def realize(design, phases, opening_open=0.06, opening_closed=0.012):
    """Joint-space realisation of a plan: IK per pose with branch continuity, every joint segment capsule-checked
    (fixtures, table, the other arm at rest), colliding segments re-routed through via poses (lifted 6 / 12 / 20 cm,
    or sideways towards the arm's own side under a shelf). Returns steps, gripper events, failures, phase spans."""
    q = {"left": K.HOME.copy(), "right": K.HOME.copy()}
    _OTHER_Q.clear()
    _OTHER_Q.update({arm: K.HOME.copy() for arm in q})
    R = {arm: K.tcp_pose(arm, K.HOME)[:3, :3] for arm in q}
    pos = {arm: K.tcp_pose(arm, K.HOME)[:3, 3].copy() for arm in q}
    grip = {"left": "open", "right": "open"}
    holdk = {"left": None, "right": None}
    last_tan = {"left": None, "right": None}
    steps, events, fails, spans = [], [], [], []
    planner_cover = Planner(design).covered
    for i, ph in enumerate(phases):
        arm = ph["hand"]
        other_arm = "left" if arm == "right" else "right"
        if ph["gripper"]:
            if ph["gripper"] == "close" and last_tan[arm] is not None:
                _choose_family(arm, i, phases, steps, q, R, pos, last_tan, design, opening_closed, opening_open, grip,
                               spans=spans)
            events.append(dict(phase=i, arm=arm, event="grasp" if ph["gripper"] == "close" else "release",
                               kind=ph["kind"], ring=ph.get("ring")))
            grip[arm] = ph["gripper"]
            holdk[arm] = held_kind(R[arm], last_tan[arm]) if ph["gripper"] == "close" and last_tan[arm] is not None else None
            if ph["gripper"] == "close":
                events[-1]["grasp_kind"] = holdk[arm]
            continue
        n_before = len(steps)
        for (p, tan, speed) in ph["poses"]:
            p = np.asarray(p, float)
            opening = opening_closed if grip[arm] == "close" else opening_open
            other = K.arm_capsules(other_arm, q[other_arm], opening_closed if grip[other_arm] == "close" else opening_open)
            tangent = tan if tan is not None else _rope_dir_from(R[arm])
            kinds = [holdk[arm]] if (grip[arm] == "close" and holdk[arm] is not None and tan is not None) else None
            res = solve_pose(arm, p, tangent, design, q_prev=q[arm], rot_prev=R[arm], opening=opening, other=other,
                             kinds=kinds)
            if tan is not None:
                last_tan[arm] = _unit(tan)
            if not res["feasible"]:
                fails.append(dict(phase=i, name=ph["name"], hand=arm, pos=np.round(p, 3).tolist(), why=res["why"],
                                  pos_err=round(res["pos_err"], 4), rot_err=round(res["rot_err"], 3),
                                  arm_gap=round(res.get("arm_gap", np.nan), 4), arm_gap_with=res.get("arm_gap_with"),
                                  finger_gap=round(res.get("finger_gap", np.nan), 4),
                                  finger_gap_with=res.get("finger_gap_with")))
            ok, worst = segment_clear(arm, q[arm], res["q"], design, opening, other)
            vias = []
            if not ok:
                vias = _subdivide(arm, pos[arm], R[arm], q[arm], p, res, design, opening, other, kinds,
                                  holding=grip[arm] == "close")
            if not ok and vias is None:
                vias = _find_vias(arm, pos[arm], R[arm], q[arm], p, res["R"], res["q"], design, opening, other,
                                  planner_cover, kinds)
                if vias is None:
                    fails.append(dict(phase=i, name=ph["name"], hand=arm, pos=np.round(p, 3).tolist(), why="segment",
                                      gap=round(worst[0], 4), with_=worst[1], at=round(worst[2], 2)))
                    vias = []
            for (vp, vR, vq) in vias:
                steps.append(dict(phase=i, arm=arm, pos=vp, R=vR, q=vq, speed=FREE_SPEED if grip[arm] == "open" else HOLD_SPEED,
                                  kind="via", gripper=grip[arm]))
            steps.append(dict(phase=i, arm=arm, pos=p, R=res["R"], q=res["q"], speed=speed, kind="pose", gripper=grip[arm]))
            q[arm], R[arm], pos[arm] = res["q"], res["R"], p
            _OTHER_Q[arm] = np.asarray(res["q"], float).copy()
        spans.append(dict(phase=i, name=ph["name"], hand=arm, steps=(n_before, len(steps))))
    return steps, events, fails, spans


def _subdivide(arm, pa, Ra, qa, pb, res_b, design, opening, other, kinds, holding, depth=3):
    """A colliding joint segment re-solved through Cartesian midpoints (the held rope's direction interpolated): the
    joint path then follows the TCP path instead of sweeping (e.g. through the table) on a large joint step."""
    if depth == 0:
        return None
    Rb, qb = res_b["R"], res_b["q"]
    if kinds:
        h = GRASP_KINDS[kinds[0]]
        tm = _unit(Ra @ h + Rb @ h) if np.linalg.norm(Ra @ h + Rb @ h) > 1e-6 else Rb @ h
    else:
        tm = _rope_dir_from(Rb)
    pm = 0.5 * (pa + pb)
    rm = solve_pose(arm, pm, tm, design, q_prev=qa, rot_prev=Ra, opening=opening, other=other, kinds=kinds)
    if not rm["feasible"]:
        return None
    out = []
    for (p0, R0, q0, p1, r1) in ((pa, Ra, qa, pm, rm), (pm, rm["R"], rm["q"], pb, res_b)):
        ok, _ = segment_clear(arm, q0, r1["q"], design, opening, other)
        if ok:
            if r1 is rm:
                out.append((pm, rm["R"], rm["q"]))
            continue
        sub = _subdivide(arm, p0, R0, q0, p1, r1, design, opening, other, kinds, holding, depth - 1)
        if sub is None:
            return None
        out += sub
        if r1 is rm:
            out.append((pm, rm["R"], rm["q"]))
    return out


def _holding_poses(phases, i, arm):
    """(position, tip-ward tangent) of the arm's poses from phase i on while it holds (until its next release)."""
    out = []
    for ph in phases[i + 1:]:
        if ph["hand"] != arm:
            continue
        if ph["gripper"] == "open":
            break
        out += [(np.asarray(p, float), np.asarray(t, float)) for (p, t, _) in ph["poses"] if t is not None]
    return out


def _choose_family(arm, i, phases, steps, q, R, pos, last_tan, design, opening_closed, opening_open, grip, spans=None):
    """At a grasp: keep the grasp family that the arm's whole holding phase can be realised with (look-ahead over its
    holding poses, every second one); if another family does better, re-solve the grasp pose with it in place."""
    other_arm = "left" if arm == "right" else "right"
    other = K.arm_capsules(other_arm, q[other_arm], opening_closed if grip[other_arm] == "close" else opening_open)
    ahead = _holding_poses(phases, i, arm)[:24]
    mine = [k for k, st in enumerate(steps) if st["arm"] == arm]
    if not mine or not ahead:
        return
    last = steps[mine[-1]]
    q_before = steps[mine[-2]]["q"] if len(mine) > 1 else K.HOME.copy()
    R_before = steps[mine[-2]]["R"] if len(mine) > 1 else K.tcp_pose(arm, K.HOME)[:3, :3]
    current = held_kind(R[arm], last_tan[arm])
    scores = {}
    for fam in GRASP_KINDS:
        g = solve_pose(arm, last["pos"], last_tan[arm], design, q_prev=q_before, rot_prev=R_before, opening=opening_open,
                       other=other, kinds=[fam])
        if not g["feasible"]:
            continue
        qq, RR, fails = g["q"], g["R"], 0
        for (p, t) in ahead:
            r_ = solve_pose(arm, p, t, design, q_prev=qq, rot_prev=RR, opening=opening_closed, other=other, kinds=[fam])
            if r_["feasible"]:
                qq, RR = r_["q"], r_["R"]
            else:
                fails += 1
        scores[fam] = (fails, 0 if fam == current else 1, g)
    if not scores:
        return
    # coordinator 10-10: a family change re-solves the grasp pose in place; the segment into the new pose must be
    # clear too (an unchecked one put a finger 5.7 mm into post P2 in the real-mesh check). p13: when it is not, the
    # approach is re-routed through via poses (midpoints, then lifted / sideways vias, then RRT) instead of keeping a
    # family whose holding phase fails (p12: rejecting the change left 11-17 failing poses per tier)
    p_before = steps[mine[-2]]["pos"] if len(mine) > 1 else K.tcp_pose(arm, K.HOME)[:3, 3]
    for fam in sorted(scores, key=lambda k: scores[k][:2]):
        if fam == current:
            return
        g = scores[fam][2]
        vias = []
        if not segment_clear(arm, q_before, g["q"], design, opening_open, other)[0]:
            if os.environ.get("RD_FAMILY_VIAS", "1") != "1":
                continue
            vias = _subdivide(arm, np.asarray(p_before, float), R_before, q_before, np.asarray(last["pos"], float), g,
                              design, opening_open, other, None, holding=False)
            if vias is None:
                vias = _find_vias(arm, np.asarray(p_before, float), R_before, q_before, np.asarray(last["pos"], float),
                                  g["R"], g["q"], design, opening_open, other, lambda *a_, **k_: False, None)
            if vias is None:
                continue
        at = mine[-1]
        for j, (vp, vR, vq) in enumerate(vias):
            steps.insert(at + j, dict(phase=last["phase"], arm=arm, pos=vp, R=vR, q=vq, speed=FREE_SPEED, kind="via",
                                      gripper="open"))
        if vias and spans is not None:          # the phase spans recorded so far: shift / extend past the insertion
            n = len(vias)
            for sp in spans:
                a0, b0 = sp["steps"]
                sp["steps"] = (a0 + n if a0 > at else a0, b0 + n if b0 > at else b0)
        last.update(q=g["q"], R=g["R"], kind="pose")
        q[arm], R[arm] = g["q"], g["R"]
        _OTHER_Q[arm] = np.asarray(g["q"], float).copy()
        return


def _find_vias(arm, pa, Ra, qa, pb, Rb, qb, design, opening, other, covered, kinds=None):
    side = -1.0 if arm == "left" else 1.0
    offsets = []
    for h in (0.06, 0.12, 0.20):
        offsets.append((np.array([0, 0, h]), np.array([0, 0, h])))
    for dx in (0.10, 0.18):
        offsets.append((np.array([side * dx, 0, 0.03]), np.array([side * dx, 0, 0.03])))
    for h in (0.12, 0.20):
        offsets.append((np.array([side * 0.10, 0, h]), np.array([side * 0.10, 0, h])))
    for oa, ob in offsets:
        va, vb = pa + oa, pb + ob
        # the via keeps the held rope's direction: the tip direction of the end poses, same grasp family
        ta = Ra @ GRASP_KINDS[kinds[0]] if kinds else _rope_dir_from(Ra)
        tb = Rb @ GRASP_KINDS[kinds[0]] if kinds else _rope_dir_from(Rb)
        ra = solve_pose(arm, va, ta, design, q_prev=qa, rot_prev=Ra, opening=opening, other=other, kinds=kinds)
        if not ra["feasible"]:
            continue
        rb = solve_pose(arm, vb, tb, design, q_prev=ra["q"], rot_prev=Rb, opening=opening, other=other, kinds=kinds)
        if not rb["feasible"]:
            continue
        chain = [qa, ra["q"], rb["q"], qb]
        if all(segment_clear(arm, x, y, design, opening, other)[0] for x, y in zip(chain[:-1], chain[1:])):
            return [(va, ra["R"], ra["q"]), (vb, rb["R"], rb["q"])]
    return _rrt_vias(arm, qa, qb, design, opening, other, kinds=kinds, Ra=Ra, Rb=Rb)


def _rrt_vias(arm, qa, qb, design, opening, other, *, kinds=None, Ra=None, Rb=None, iters=4000, step=0.25, seed=0,
              max_dev_deg=60.0):
    """Joint-space RRT-Connect between two collision-free configurations (coordinator, 10-10: the heuristic via
    offsets did not clear 2-10 segments per tier). Every node and edge passes the same capsule checks
    (config_clear / segment_clear). While holding the rope (kinds given), a node must keep the held rope direction
    within max_dev_deg of the end poses' directions (the rope is carried, not swung around). The path is shortcut
    and returned as vias [(tcp position, R, q)], or None."""
    rng = np.random.default_rng(seed)
    lo, hi = K.LIMITS[:, 0], K.LIMITS[:, 1]
    held = GRASP_KINDS[kinds[0]] if kinds else None
    if held is not None:
        da, db = Ra @ held, Rb @ held
        cos_ok = math.cos(math.radians(max_dev_deg))

    def valid(q):
        if q[4] > WRIST_ANGLE_MAX:
            return False
        if held is not None:
            t = K.tcp_pose(arm, q)[:3, :3] @ held
            if max(float(t @ da), float(t @ db)) < cos_ok:
                return False
        return config_clear(arm, q, design, opening, other)[0]

    def edge(x, y):
        return segment_clear(arm, x, y, design, opening, other, step=0.03)[0]

    trees = [[(np.asarray(qa, float), -1)], [(np.asarray(qb, float), -1)]]
    span = np.abs(np.asarray(qb) - np.asarray(qa)) + 0.5

    def nearest(tree, q):
        d = [float(np.linalg.norm(n[0] - q)) for n in tree]
        return int(np.argmin(d))

    def extend(tree, q):
        i = nearest(tree, q)
        x = tree[i][0]
        d = q - x
        n = float(np.linalg.norm(d))
        y = q if n <= step else x + d * (step / n)
        if valid(y) and edge(x, y):
            tree.append((y, i))
            return len(tree) - 1, n <= step
        return None, False

    def path_to(tree, i):
        out = []
        while i != -1:
            out.append(tree[i][0])
            i = tree[i][1]
        return out[::-1]

    for it in range(iters):
        if rng.random() < 0.3:
            q = rng.uniform(lo, hi)
        else:
            c = 0.5 * (np.asarray(qa) + np.asarray(qb))
            q = np.clip(c + rng.normal(0, 1, 6) * span, lo, hi)
        a, b = trees[it % 2], trees[(it + 1) % 2]
        ia, _ = extend(a, q)
        if ia is None:
            continue
        # connect the other tree toward the new node
        target = a[ia][0]
        while True:
            ib, reached = extend(b, target)
            if ib is None:
                break
            if reached or float(np.linalg.norm(b[ib][0] - target)) < 1e-9:
                pa, pb = path_to(a, ia), path_to(b, ib)
                path = (pa + pb[::-1]) if it % 2 == 0 else (pb + pa[::-1])
                # shortcut
                k = 0
                while k < len(path) - 2:
                    j = len(path) - 1
                    while j > k + 1 and not edge(path[k], path[j]):
                        j -= 1
                    path = path[:k + 1] + path[j:]
                    k += 1
                out = []
                for qv in path[1:-1]:
                    T = K.tcp_pose(arm, qv)
                    out.append((T[:3, 3].copy(), T[:3, :3].copy(), np.asarray(qv, float)))
                return out
    return None


def check_plan(design, phases, verbose=False):
    """IK + continuous collision checks of the realised plan, metrics. Returns a report dict."""
    steps, events, fails, spans = realize(design, phases)
    path = {"left": 0.0, "right": 0.0}
    jpath = {"left": 0.0, "right": 0.0}
    last = {arm: (K.tcp_pose(arm, K.HOME)[:3, 3], K.HOME.copy()) for arm in path}
    t_total = 0.0
    for st in steps:
        p0, q0 = last[st["arm"]]
        seg = float(np.linalg.norm(st["pos"] - p0))
        dq = float(np.max(np.abs(st["q"] - q0)))
        path[st["arm"]] += seg
        jpath[st["arm"]] += float(np.linalg.norm(st["q"] - q0))
        t_total += max(seg / st["speed"], dq / K.MAX_JOINT_SPEED, 0.04)
        last[st["arm"]] = (st["pos"], st["q"])
    t_total += sum(GRASP_DWELL_S if e["event"] == "grasp" else RELEASE_DWELL_S for e in events)
    kinds = {}
    for ph in phases:
        kinds[ph["kind"]] = kinds.get(ph["kind"], 0) + 1
    return dict(tier=design.tier, phases=len(phases), phase_kinds=kinds, waypoints=sum(1 for s_ in steps),
                vias=sum(1 for s_ in steps if s_["kind"] == "via"),
                grasp_count=sum(1 for e in events if e["event"] == "grasp"),
                release_count=sum(1 for e in events if e["event"] == "release"),
                handover_count=sum(1 for leg in _legs_for(design) if leg["feeder"] != leg["receiver"]),
                same_hand_ring_regrasps=sum(1 for leg in _legs_for(design) if leg["feeder"] == leg["receiver"]),
                regrasp_phases=sum(1 for ph in phases if ph["kind"] == "regrasp"),
                keeper_grasps=sum(1 for ph in phases if ph["name"].startswith("keeper grasp")),
                tcp_path_m={k: round(v, 3) for k, v in path.items()}, tcp_path_total_m=round(sum(path.values()), 3),
                joint_path_rad={k: round(v, 2) for k, v in jpath.items()}, duration_estimate_s=round(t_total, 1),
                infeasible_waypoints=fails, feasible=not fails)


def timeline(design, phases, world=True):
    """The realised plan as a timed, serial timeline in the tooling lane's plan format (antihack.trajectory.
    metrics_from_plan): per arm TCP waypoints [t, x, y, z] (world frame), grasp / release events, plus each arm's joint
    waypoints. Segment time = max(TCP distance / speed, joint step / pi rad/s, one control); grasp / release dwells."""
    steps, evs, fails, spans = realize(design, phases)
    off = (K.ROOT_WORLD + np.array([0, 0, K.TABLE_Z_WORLD])) if world else np.zeros(3)
    q = {"left": K.HOME.copy(), "right": K.HOME.copy()}
    pos = {arm: K.tcp_pose(arm, K.HOME)[:3, 3].copy() for arm in q}
    tcp = {arm: [[0.0, *(pos[arm] + off)]] for arm in q}
    joints = {arm: [[0.0, *q[arm]]] for arm in q}
    events, t = [], 0.0
    span_t = []
    ev_by_phase = {}
    for e in evs:
        ev_by_phase.setdefault(e["phase"], []).append(e)
    by_phase = {}
    for st in steps:
        by_phase.setdefault(st["phase"], []).append(st)
    for i, ph in enumerate(phases):
        t0 = t
        for e in ev_by_phase.get(i, []):
            t += GRASP_DWELL_S if e["event"] == "grasp" else RELEASE_DWELL_S
            events.append(dict(t=round(t, 3), arm=e["arm"], event=e["event"], object="rope"))
        for st in by_phase.get(i, []):
            arm = st["arm"]
            if tcp[arm][-1][0] < t:
                tcp[arm].append([round(t, 3), *(pos[arm] + off)])
                joints[arm].append([round(t, 3), *q[arm]])
            seg = float(np.linalg.norm(st["pos"] - pos[arm]))
            dq = float(np.max(np.abs(st["q"] - q[arm])))
            t += max(seg / st["speed"], dq / K.MAX_JOINT_SPEED, 0.04)
            q[arm], pos[arm] = st["q"], st["pos"]
            tcp[arm].append([round(t, 3), *(pos[arm] + off)])
            joints[arm].append([round(t, 3), *q[arm]])
        span_t.append(dict(name=ph["name"], hand=ph["hand"], kind=ph["kind"], t0=round(t0, 3), t1=round(t, 3),
                           sigma=ph.get("sigma"), gripper=ph.get("gripper"), ring=ph.get("ring"),
                           steps=[dict(kind=st["kind"], pos=np.round(st["pos"], 5).tolist(), R=np.round(st["R"], 7).tolist(),
                                       q=np.round(st["q"], 6).tolist(), speed=st["speed"]) for st in by_phase.get(i, [])]))
    t += 0.5
    for arm in q:
        tcp[arm].append([round(t, 3), *(pos[arm] + off)])
        joints[arm].append([round(t, 3), *q[arm]])
    plan = dict(control_dt_s=0.04, arms={arm: dict(tcp=[[float(v) for v in r] for r in tcp[arm]]) for arm in tcp},
                events=events)
    jpath = {arm: float(np.sum(np.linalg.norm(np.diff(np.asarray(joints[arm])[:, 1:], axis=0), axis=1))) for arm in joints}
    return plan, dict(joints={arm: [[float(v) for v in r] for r in joints[arm]] for arm in joints},
                      joint_path_rad=jpath, duration_s=t, spans=span_t, failures=fails,
                      steps=[dict(phase=st["phase"], arm=st["arm"], kind=st["kind"], pos=np.round(st["pos"], 4).tolist(),
                                  q=np.round(st["q"], 5).tolist(), R=np.round(st["R"], 6).tolist()) for st in steps])


def _rope_dir_from(R):
    """The rope direction a gripper rotation holds (perpendicular to the closing axis): its z axis for a cross grasp."""
    return R[:, 2]


def forced_handover(design, ring_name, feeder, receiver, n_dirs=60):
    """For a bulkhead grommet: is there any collision-free pinch pose for the feeding arm at the receive point beyond
    the wall (over n_dirs approach directions x 8 rolls)? And for the receiving arm?"""
    ring = [r for r in design.rings if r.name == ring_name][0]
    d = _unit(ring.travel)
    pr = ring.center + (S.POKE_M - S.RECEIVE_STATION_M) * d
    out = {}
    for arm in (feeder, receiver):
        ok = None
        tried = 0
        # Fibonacci sphere of approach directions; closing axis perpendicular to the rope where possible
        for i in range(n_dirs):
            z = 1 - 2 * (i + 0.5) / n_dirs
            r = math.sqrt(max(0, 1 - z * z))
            th = math.pi * (1 + 5 ** 0.5) * i
            a = np.array([r * math.cos(th), r * math.sin(th), z])
            for roll in range(4):
                c = np.cross(a, d)
                if np.linalg.norm(c) < 0.2:          # nearly needle: sample the closing axis
                    w = _unit(np.cross(a, [0.3, 0.5, 0.8]))
                    c = math.cos(roll * np.pi / 4) * w + math.sin(roll * np.pi / 4) * np.cross(a, w)
                elif roll > 0:
                    break
                Rm = K.gripper_frame(a, closing=c)
                tried += 1
                qs, pe, re = K.ik(arm, pr, rotation=Rm)
                if pe > 0.003 or re > 0.05:
                    continue
                caps = K.arm_capsules(arm, qs, 0.012)
                g = fixture_clearance(caps, design)
                if g[0] >= 0.0:
                    ok = dict(approach=np.round(a, 3).tolist(), q=np.round(qs, 3).tolist(), gap=round(g[0], 4))
                    break
            if ok:
                break
        out[arm] = dict(reachable=bool(ok), witness=ok, orientations_tried=tried)
    return dict(ring=ring_name, pinch_point=np.round(pr, 3).tolist(), feeder=feeder, receiver=receiver,
                forced=bool(not out[feeder]["reachable"] and out[receiver]["reachable"]), arms=out)


def run(tier, verbose=False):
    d = S.build(tier)
    phases = Planner(d).plan()
    rep = check_plan(d, phases, verbose)
    return d, phases, rep


def save_timeline(tier, out):
    """Realise a tier's plan and write its timeline (plan format + phases with steps / grasp stations) as JSON."""
    d = S.build(tier)
    phases = Planner(d).plan()
    plan, extra = timeline(d, phases)
    rep = check_plan(d, phases)
    Path(out).write_text(json.dumps(dict(tier=tier, plan=plan, joints=extra["joints"], spans=extra["spans"],
                                         failures=extra["failures"], duration_s=extra["duration_s"],
                                         joint_path_rad=extra["joint_path_rad"], check=rep), default=float))
    return rep


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--save":
        for tier in sys.argv[3:] or list(S.TIERS):
            rep = save_timeline(tier, Path(sys.argv[2]) / f"timeline_{tier}.json")
            print(tier, "feasible", rep["feasible"], "vias", rep["vias"], "fails", len(rep["infeasible_waypoints"]), flush=True)
        raise SystemExit(0)
    for tier in sys.argv[1:] or list(S.TIERS):
        d, phases, rep = run(tier)
        print(json.dumps({k: v for k, v in rep.items() if k != "infeasible_waypoints"}, default=float))
        for f in rep["infeasible_waypoints"][:12]:
            print("   INFEASIBLE", json.dumps(f, default=str))
