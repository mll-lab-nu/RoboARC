"""The v1 relay plan of the redesigned rope_threading tiers: "lay and pick" (rope agent, 2026-10-10 night).

Geometry v1 (scenes.TIERS_V1) puts every ring a hand lets go of low (on a pedestal), so after a feed the rope rests on
the table on both sides of it. The relay over the tier's legs (one per ring, in material order):

  0. the first feeder picks the start rope FEED_STATION_M behind its tip (top-down, the rope lying on the table);
  1. lift and carry: the tip FEED_APPROACH_M before the ring's plane on its axis (the held rope along the axis);
  2. feed: push along the axis until the tip is POKE_M past the plane (closed loop at execution: the tip measured and
     turned onto the axis first, robot_relay.py);
  3. receive: another arm pinches the tip RECEIVE_STATION_M from its end beyond the ring (a grommet in the wall: a
     forced hand-over; the feeder then lets go and parks), or the feeder lets go, walks around the ring and picks the tip
     beyond it (a low ring: a forced same-hand regrasp), or nothing (the last ring: the feeder holds; holding is allowed);
  4. draw and lay: the tip holder draws the tip along the goal path to FEED_APPROACH_M before the next ring and lays the
     rope down on the table along that ring's axis (its ground line), lets go;
  5. the next feeder (always the tip holder: the rings alternate the wall's halves by design) picks the laid rope
     FEED_STATION_M behind the tip; back to 1.

Kinematics: the phases are realised by oracle.realize (IK with branch continuity, capsule checks of every pose and joint
segment against the fixtures, the table and the other arm, vias / RRT where a segment collides, grasp families fixed while
holding), so `check_plan` / `timeline` / the anti-hack metrics work on them unchanged. At execution (robot_relay.py)
every grasp is re-aimed at the rope's measured material station and every feed is closed loop.
"""
from __future__ import annotations

import os
import math

import numpy as np

from topogym.planning.rope_threading.redesign import scenes as S
from . import oracle as O
from topogym.planning.rope_threading.redesign.legs import legs_for as _legs_for

PICK_Z = 0.014              # TCP height of a pick of a rope lying on the table (pads at the rope, fingertips touch)
LIFT_M = 0.09               # lift after a pick before carrying
DRAW_Z_MIN = 0.06           # a drawing hand keeps at least this height until the lay
LAY_DESCENT_M = 0.10        # along the lead-in, the drawing hand descends from its height to PICK_Z over this length
SAME_HAND_BACK_M = 0.06     # after a same-hand release the hand backs off along -axis by this much, then lifts
LAY_GAP_M = 0.12            # the laid tip lies this far before the next ring's plane (plan 1: FEED_APPROACH_M = 6 cm put the
                            # drawing hand's wrist camera into the ring / grommet; the lift and carry closes the gap)
LAY_GAP_VERTICAL_M = 0.10   # beyond the outer rim of a vertical eyelet's footprint
SAME_HAND_STATION_M = 0.20  # feeding grasp before a same-hand ring (rr6: from 14 cm the tip stood 8.5 cm past the plane and
                            # retracted to 3-4 cm when the hand let go, too short to pick beyond the ring)
HAND_STOP_M = S.FEED_STATION_M - S.POKE_M   # the feeding hand stops this far before the plane (5.5 cm)
LAY_RUN_EXTRA_M = 0.12      # the drawing hand runs straight along the lay line this much more than the feeding station
LAY_WALL_CLEAR_M = float(os.environ.get("RR_LAY_WALL_CLEAR", "0.08"))   # m: a lay run starts this far from the bulkhead (plan8 0.08; plan9 0.12 lost G3 on hard at every lay before C)
                            # (rr12 easy / medium: with 2 cm the rope behind the tip cut the corner to the grommet, so the
                            # material station lay 20 cm off the lay line)
RETRACT_M = 0.045           # the tip's retraction when a same-hand feeder lets go (rr6, very_easy / easier / easy A)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def lay_dir(ring, prev_exit=None):
    """The ground direction along which the rope is laid in front of `ring`: its axis projected on the table, or, for a
    (nearly) vertical axis, the plan-view direction from the previous ring's exit to it."""
    d = _unit(ring.travel)
    h = np.array([d[0], d[1], 0.0])
    if np.linalg.norm(h) > 0.45:
        return _unit(h)
    if prev_exit is None:
        return np.array([-1.0, 0.0, 0.0])
    v = np.asarray(ring.center, float) - np.asarray(prev_exit, float)
    v[2] = 0.0
    return _unit(v)


class RelayPlanner(O.Planner):
    """Phases of the v1 relay (same phase format as oracle.Planner.plan)."""

    def __init__(self, design):
        super().__init__(design)
        self.legs = {leg["ring"]: leg for leg in _legs_for(design)}
        self.lay = {}           # ring name -> dict(tip, dir, station): the laid rope in front of it (nominal)

    # nominal positions ------------------------------------------------------------------------------------------
    def station(self, ring):
        """The feeding grasp's distance from the tip for this ring: longer before a same-hand ring."""
        leg = self.legs[ring.name]
        return SAME_HAND_STATION_M if (leg["feeder"] == leg["receiver"] and not leg.get("last")) else S.FEED_STATION_M

    def poke(self, ring):
        return self.station(ring) - HAND_STOP_M

    def lay_geometry(self, ring, prev_exit):
        """The laid rope in front of a ring: tip point and station point on the table (nominal)."""
        st, appr = self.station(ring), (ring.lay_gap or LAY_GAP_M)
        dl = lay_dir(ring, prev_exit)
        d = _unit(ring.travel)
        if abs(d[2]) > 0.8:
            # a vertical ring: the tip lies outside its footprint on the approach side
            tip = np.asarray(ring.center, float) - (ring.outer_radius + LAY_GAP_VERTICAL_M) * dl
        else:
            tip = np.asarray(ring.center, float) - appr * np.array([d[0], d[1], 0.0]) / max(np.linalg.norm(d[:2]), 1e-9)
        tip = np.array([tip[0], tip[1], S.ROPE_Z_M])
        station = tip - st * dl
        return dict(tip=tip, dir=dl, station=station)

    def feed_line(self, ring):
        """Hand positions at the start and the end of the push (the hand holds FEED_STATION_M behind the tip)."""
        st, appr, poke = self.station(ring), S.FEED_APPROACH_M, self.poke(ring)
        d = _unit(ring.travel)
        return ring.center - (st + appr) * d, ring.center - (st - poke) * d, d

    def pick_point_beyond(self, ring):
        """Nominal pinch of the tip beyond a ring after the push (RECEIVE_STATION_M from the tip): on the axis for a
        hand-over (the feeder still holds), on the table (or at the ring's exit height if higher) for a same-hand pick."""
        d = _unit(ring.travel)
        leg = self.legs[ring.name]
        back = RETRACT_M if leg["feeder"] == leg["receiver"] else 0.0
        p = ring.center + (self.poke(ring) - back - S.RECEIVE_STATION_M) * d
        return p

    # the plan -----------------------------------------------------------------------------------------------------
    def plan(self):
        R = self.rings
        rcv = S.RECEIVE_STATION_M
        first = R[0]
        st = self.station(first)
        F0 = self.legs[first.name]["feeder"]
        # 0) pick the start rope at the feeding station
        p0, t0 = self.start_at(st)
        p0 = np.array([p0[0], p0[1], PICK_Z])
        self.phase("approach start rope", "approach", F0,
                   [(p0 + [0, 0, O.APPROACH_OFFSET], t0, O.FREE_SPEED), (p0, t0, O.FREE_SPEED * 0.5)], sigma=st)
        self.phase("grasp", "grasp", F0, [], gripper="close")
        self.holding[F0] = st
        holder = F0
        prev_exit = None
        for k, ring in enumerate(R):
            leg = self.legs[ring.name]
            F, Rc, last = leg["feeder"], leg["receiver"], bool(leg.get("last"))
            assert holder == F, f"{ring.name}: the tip holder {holder} is not its feeder {F}"
            pa, pb, d = self.feed_line(ring)
            # 1) lift and carry the tip to the feed line
            cur = self.tcp_pos(F)
            lift = cur + np.array([0.0, 0.0, LIFT_M])
            mid = 0.5 * (lift + pa) + np.array([0.0, 0.0, 0.04])
            t_now = self.lay[ring.name]["dir"] if ring.name in self.lay else _unit(self.start_at(st)[1])
            self.phase(f"lift and carry to {ring.name}'s feed line", "transport", F,
                       [(lift, t_now, O.HOLD_SPEED), (mid, _unit(t_now + d), O.HOLD_SPEED), (pa, d, O.HOLD_SPEED)],
                       ring=ring.name)
            # 2) feed
            self.phase(f"feed {ring.name}", "feed", F, [(pa, d, O.HOLD_SPEED), (pb, d, O.HOLD_SPEED * 0.6)],
                       ring=ring.name)
            if last:
                self.phase("hold until quiet", "hold", F, [], note="holding is allowed; release not required")
                break
            # 3) receive beyond the ring
            pr = self.pick_point_beyond(ring)
            if Rc != F:
                self.phase(f"receiver approaches beyond {ring.name}", "approach", Rc,
                           [(pr + self.retreat_vec(Rc, d, ring, at=pr), d, O.FREE_SPEED), (pr, d, O.FREE_SPEED * 0.5)],
                           ring=ring.name, sigma=rcv)
                self.phase("receiver closes", "grasp", Rc, [], gripper="close", ring=ring.name)
                self.phase("feeder lets go (hand-over)", "release", F, [], gripper="open", ring=ring.name)
                self.holding[F] = None
                self.phase("feeder retreats", "retreat", F,
                           [(self.tcp_pos(F) - 0.06 * d + self.retreat_vec(F, None, at=self.tcp_pos(F)) * 0.6, None,
                             O.FREE_SPEED), (self.park(F), None, O.FREE_SPEED)])
            else:
                self.phase(f"feeder lets go in front of {ring.name}", "release", F, [], gripper="open", ring=ring.name)
                self.holding[F] = None
                back = self.tcp_pos(F) - SAME_HAND_BACK_M * d + np.array([0.0, 0.0, 0.04])
                up = back + np.array([0.0, 0.0, max(0.0, ring.center[2] + ring.outer_radius + 0.08 - back[2])])
                around = self.around_point(F, ring)
                prt = np.array([pr[0], pr[1], max(PICK_Z, float(pr[2]) - 0.04 if abs(d[2]) < 0.3 else PICK_Z)])
                if ring.center[2] < 0.12:
                    prt[2] = PICK_Z                       # a low ring: the tip comes to rest on the table beyond it
                if d[2] > 0.2:
                    # a ring tilted upward (medium / hard T): the tip leaves it rising and falls forward onto the table
                    # beyond it; the pick also keeps the hand clear of the ring's lower rim
                    prt[:2] += 0.03 * _unit(np.array([d[0], d[1]]))
                    prt[2] = PICK_Z
                self.phase(f"walk around {ring.name}", "regrasp", F,
                           [(back, None, O.FREE_SPEED), (up, None, O.FREE_SPEED), (around, None, O.FREE_SPEED),
                            (prt + np.array([0.0, 0.0, O.APPROACH_OFFSET]), _unit(np.array([d[0], d[1], 0.0])), O.FREE_SPEED),
                            (prt, _unit(np.array([d[0], d[1], 0.0])), O.FREE_SPEED * 0.5)],
                           ring=ring.name, sigma=rcv)
                self.phase("close beyond the ring", "grasp", F, [], gripper="close", ring=ring.name)
            holder = Rc
            self.holding[Rc] = rcv
            # 4) draw the tip to the next ring's lay line and lay it down
            nxt = R[k + 1]
            exit_pt = ring.center + self.poke(ring) * d
            lay = self.lay_geometry(nxt, exit_pt)
            self.lay[nxt.name] = lay
            self.phase(f"draw tip to {nxt.name}", "transport", Rc, self.draw_poses(ring, nxt, lay), ring=ring.name)
            self.phase(f"lay down in front of {nxt.name}", "release", Rc, [], gripper="open", ring=nxt.name)
            self.holding[Rc] = None
            up = self.tcp_pos(Rc) + np.array([0.0, 0.0, O.APPROACH_OFFSET])
            self.phase("lift off the laid rope", "retreat", Rc, [(up, None, O.FREE_SPEED)])
            # 5) pick the laid rope at the feeding station
            ps = np.array([lay["station"][0], lay["station"][1], PICK_Z])
            self.phase(f"pick at {nxt.name}'s feeding station", "regrasp", Rc,
                       [(ps + np.array([0.0, 0.0, O.APPROACH_OFFSET]), lay["dir"], O.FREE_SPEED),
                        (ps, lay["dir"], O.FREE_SPEED * 0.5)], ring=nxt.name, sigma=self.station(nxt))
            self.phase("close", "grasp", Rc, [], gripper="close", ring=nxt.name)
            self.holding[Rc] = self.station(nxt)
            prev_exit = exit_pt
        return self.phases

    def draw_poses(self, ring, nxt, lay):
        """Hand poses drawing the tip (held RECEIVE_STATION_M from its end) from beyond `ring` along the goal path to the
        lay point in front of `nxt`, then down onto the table along the lay line."""
        rcv = S.RECEIVE_STATION_M
        p_start = ring.s_goal + self.poke(ring)
        lay_tip = lay["tip"]
        # the arc on the goal path nearest the lay line's start (FEED_STATION_M + 2 cm behind the laid tip)
        back_len = self.station(nxt) + LAY_RUN_EXTRA_M
        while abs(float((lay_tip - back_len * lay["dir"])[0])) < LAY_WALL_CLEAR_M and back_len > 0.06:
            back_len -= 0.01             # hard's C: the lay line would start at the bulkhead (G3 -> C is 0.32 m)
        while float((lay_tip - back_len * lay["dir"])[1]) > 0.40 and back_len > self.station(nxt) + 0.02:
            back_len -= 0.01             # plan6 medium / hard T: the run started at the table's back edge (IK, wrist on the table)
        back = lay_tip - back_len * lay["dir"]
        lo = ring.s_goal + 0.05
        hi = nxt.s_goal
        ss = np.linspace(lo, hi, 400)
        pts = np.array([self.at(s_, 0.0)[0] for s_ in ss])
        j = int(np.argmin(np.linalg.norm(pts[:, :2] - back[:2], axis=1)))
        p_join = float(ss[j])
        poses = []
        n = max(2, int(math.ceil(max(p_join - p_start, 0.0) / 0.04)))
        for p in np.linspace(p_start, p_join, n + 1)[1:]:
            pos, tan = self.at(p, rcv)
            pos = np.array([pos[0], pos[1], max(pos[2], DRAW_Z_MIN)])
            poses.append((pos, tan, O.HOLD_SPEED))
        # along the lay line: from `back` to the laid tip, the hand RECEIVE_STATION_M behind the tip, descending
        z0 = poses[-1][0][2] if poses else DRAW_Z_MIN
        L = float(np.linalg.norm(lay_tip[:2] - back[:2]))
        m = max(2, int(math.ceil(L / 0.03)))
        for i in range(1, m + 1):
            f = i / m
            tip_xy = back[:2] + f * (lay_tip[:2] - back[:2])
            hand_xy = tip_xy - rcv * lay["dir"][:2]
            z = max(S.ROPE_Z_M + 0.010, z0 + (O_PICK_Z() - z0) * min(1.0, f * L / LAY_DESCENT_M))
            poses.append((np.array([hand_xy[0], hand_xy[1], z]), lay["dir"], O.HOLD_SPEED * 0.8))
        return poses


def O_PICK_Z():
    return PICK_Z


def run(tier):
    d = S.build(tier)
    rp = RelayPlanner(d)
    phases = rp.plan()
    rep = O.check_plan(d, phases)
    return d, phases, rep


if __name__ == "__main__":
    import json
    import sys
    for tier in sys.argv[1:] or list(S.TIERS_V1):
        d, phases, rep = run(tier)
        print(json.dumps({k: v for k, v in rep.items() if k != "infeasible_waypoints"}, default=float), flush=True)
        for f in rep["infeasible_waypoints"][:20]:
            print("   INFEASIBLE", json.dumps(f, default=str), flush=True)
