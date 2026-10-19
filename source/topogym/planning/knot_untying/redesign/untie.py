"""Object-level untying policies on the MuJoCo lab: the redesign's oracle and the anti-hack breakers.

Hands are the lab's compliant virtual hands (8 N weld grasp, pinch only where a parallel gripper fits); every hand
target is kept inside its ALOHA arm's measured vertical-pinch reach (0.585 m from the arm's base in the table plane,
robot_ik reach map), otherwise the motion stops there ("reach limit"), as the family Runner stops on an IK failure.

Policies
  oracle      anchored node deletion (HULK with the pin on the over strand next to the crossing, not the far end):
              from the free end, the first crossing where the rope passes UNDER is deleted by pulling the strand
              35 mm past it (toward the body) along the rope, in reach-limited strokes with regrasps, while the other
              hand pins the over strand beside the crossing; R-I curls are removed by drawing the free end against
              the clamp; then (tiers with a region) the rope is laid into the region by hand-over-hand placement
  pull_free   breaker: pinch the free end and pull it straight away from the clamp (the "pull both ends apart" of an
              anchored rope; also "pull the nearest end")
  route_rule  breaker: the family's own decision rule (route_rule.choose, hold far / direction away + straighten), as
              robot_oracle.Runner applies it (5 cm pull increments up to 0.55 m, 3 cm lift, up to 16 decisions); a
              grasp the clamp makes impossible is skipped (the clamp is the pin), never forced
  drag        breaker: pinch the knot body and drag the whole rope toward the goal region / away from the clamp
"""
from __future__ import annotations

import json
import time

import numpy as np

from .mjlab import to_world, to_local
from topogym.planning.knot_untying.redesign.tiers import keep_out_violations

BASES_LOCAL = {"left": np.array([-.469, -.019]), "right": np.array([.469, -.019])}
REACH_M = .585
PRE = .05


class Stop(Exception):
    pass


class Untie:
    def __init__(self, lab, scene, *, limit=8., speed=.06, verbose=True):
        self.lab, self.scene = lab, scene
        for h in lab.hands:
            lab.hands[h]["limit"] = limit
        self.anchors = {r["name"]: (r.get("anchor") or {}).get("end") for r in scene["ropes"]}
        self.names = list(lab.rods)
        self.speed = speed
        self.decisions = []
        self.phases = []
        self.verbose = verbose
        self.t_start = lab.t
        self.wall0 = time.time()
        self.deferred = set()        # ropes left lying on top of others: the layout lifts them off
        # breakers pull at full strength, until the grasp slips (knot agent 10-10: with the weld hands of the redesign,
        # ~8 N, a naive policy is stronger too; main's condition 2); the oracle keeps its own stroke stops
        self.breaker_stop = None if limit >= 6. else 4.8

    # ------------------------------------------------------------ observation
    def say(self, *a):
        if self.verbose:
            print(f"[{self.lab.t:7.2f}s {time.time() - self.wall0:6.0f}w]", *a, flush=True)

    def visits(self, rope, *, skip_deferred=True):
        """Crossing visits of `rope` (sorted by s); crossings with a deferred rope (lying on top, to be lifted off by
        the layout) are left out unless skip_deferred is False."""
        k = self.names.index(rope)
        out = []
        for i, c in enumerate(self.lab.crossings()):
            a_over = c["za"] > c["zb"]
            if c["a"][0] == k:
                out.append(dict(s=c["a"][1], over=a_over, other=(self.names[c["b"][0]], c["b"][1]), idx=i, xy=c["xy"]))
            if c["b"][0] == k:
                out.append(dict(s=c["b"][1], over=not a_over, other=(self.names[c["a"][0]], c["a"][1]), idx=i,
                                xy=c["xy"]))
        if skip_deferred and self.deferred:
            out = [v for v in out if v["other"][0] == rope or v["other"][0] not in self.deferred]
        return sorted(out, key=lambda v: v["s"])

    def first_under(self, rope):
        """(visit, remaining tail m) of the first crossing from the free end where the rope passes under (None, None if
        there is none). Withdrawing the tail shrinks the remaining length; when that crossing is gone the next one is
        further from the end, so the remaining length jumps up."""
        fae = self.free_at_end(rope)
        vs = self.visits(rope)
        walk = sorted(vs, key=lambda v: -v["s"]) if fae else vs
        target = next((v for v in walk if not v["over"]), None)
        if target is None:
            return None, None
        L = self.lab.length(rope)
        return target, (L - target["s"]) if fae else target["s"]

    def free_at_end(self, rope):
        return self.anchors.get(rope) != "end"

    def local(self, rope, s):
        return to_local(self.lab.point(rope, s))

    def reachable(self, hand, p_local, margin=0.):
        return float(np.linalg.norm(np.asarray(p_local)[:2] - BASES_LOCAL[hand])) <= REACH_M - margin

    def arm_for(self, p_local, prefer=None, busy=()):
        options = [h for h in ("left", "right") if h not in busy and self.reachable(h, p_local)]
        if not options:
            return None
        if prefer in options:
            return prefer
        return min(options, key=lambda h: np.linalg.norm(np.asarray(p_local)[:2] - BASES_LOCAL[h]))

    def pinch_station(self, rope, s, toward=0, search=.05, step=.005, exclude=(), min_clear=None, one_sided=False):
        """A station near s where a parallel gripper fits (lab pinch clearance); toward > 0 prefers larger s;
        one_sided: search only in the `toward` direction."""
        L = self.lab.length(rope)
        offs = [0.]
        for x in np.arange(step, search + 1e-9, step):
            if one_sided:
                offs += [x if toward >= 0 else -x]
            else:
                offs += ([x, -x] if toward >= 0 else [-x, x])
        need = self.lab.pinch_clearance if min_clear is None else min_clear
        for o in offs:
            st = float(np.clip(s + o, .03, L - .012))
            if any(abs(st - e) < .02 for e in exclude):
                continue
            if self.lab.pinch_clearance_at(rope, st) >= need:
                return st
        return None

    # ------------------------------------------------------------ hand primitives (targets in world frame)
    def phase(self, hand, label, t0):
        self.phases.append(dict(hand=hand, label=label, t0=round(t0, 4), t1=round(self.lab.t, 4)))

    def goto(self, hand, p_world, speed=None, label="transport"):
        t0 = self.lab.t
        if not self.reachable(hand, to_local(p_world)):
            raise Stop(f"{hand} cannot reach {to_local(p_world).round(3).tolist()}")
        self.lab.move({hand: np.asarray(p_world, float)}, speed or self.speed)
        self.phase(hand, label, t0)

    def take(self, hand, rope, s, label="grasp"):
        """Approach above the station, descend, pinch (weld), as one approach + grasp phase."""
        lab = self.lab
        p = lab.point(rope, s)
        if not self.reachable(hand, to_local(p)):
            raise Stop(f"{hand} cannot reach station {s:.3f} of {rope}")
        t0 = lab.t
        here = lab.hand_pos(hand)
        above = p + [0, 0, PRE]
        if np.linalg.norm(here - above) > 1e-3:
            mid = here.copy()
            mid[2] = max(here[2], above[2])
            lab.move({hand: mid}, .25)
            lab.move({hand: above}, .25)
        lab.move({hand: p + [0, 0, .004]}, .08)
        self.phase(hand, "approach", t0)
        t1 = lab.t
        ok = lab.grasp(hand, rope, s)
        if not ok:
            raise Stop(f"{hand}: no room to pinch {rope} at s={s:.3f}")
        lab.wait(.15)
        self.phase(hand, label, t1)
        return True

    def lift(self, hand, dz=.02, speed=.05):
        t0 = self.lab.t
        self.lab.move({hand: self.lab.hand_pos(hand) + [0, 0, dz]}, speed)
        self.phase(hand, "lift", t0)

    def drop(self, hand, lower_to_table=True):
        """Lower the held strand to the table, open, retreat 5 cm up."""
        lab = self.lab
        t0 = lab.t
        g = lab.hands[hand]["grasp"]
        if g is not None and lower_to_table:
            p = lab.hand_pos(hand)
            z_table = to_world(np.array([0, 0, lab.rods[g["rope"]].radius + .001]))[2]
            if p[2] > z_table + .001:
                lab.move({hand: np.r_[p[:2], z_table]}, .03)
        lab.release(hand)
        lab.wait(.1)
        lab.move({hand: lab.hand_pos(hand) + [0, 0, PRE]}, .2)
        self.phase(hand, "release", t0)

    def stroke(self, hand, direction, length, *, step=.01, speed=None, stop_force=None, watch=None, margin=0.):
        """Move the holding hand along a (unit, world) direction up to `length`, in steps; stops at the reach limit,
        on a slip, when `stop_force` is exceeded, or when watch() returns True. Returns (moved, why)."""
        lab = self.lab
        d = np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        moved, why = 0., "length"
        t0 = lab.t
        while moved < length - 1e-9:
            if lab.hands[hand]["grasp"] is None:
                why = "slip"
                break
            nxt = lab.hand_pos(hand) + min(step, length - moved) * d
            if not self.reachable(hand, to_local(nxt), margin):
                why = "reach"
                break
            lab.move({hand: nxt}, speed or self.speed, min_time=.02)
            moved += min(step, length - moved)
            if lab.hands[hand]["grasp"] is None:
                why = "slip"
                break
            f = float(np.mean(lab.hands[hand]["window"]))
            if stop_force is not None and f > stop_force:
                why = "force"
                break
            if watch is not None and watch():
                why = "watch"
                break
        self.phase(hand, "transport", t0)
        return moved, why

    def holding(self, hand):
        return self.lab.hands[hand]["grasp"] is not None

    def release_all(self):
        for h in self.lab.hands:
            if self.holding(h):
                self.drop(h)

    def count(self):
        return len(self.lab.crossings())

    # ------------------------------------------------------------ clutter
    def posts(self):
        """Tall fixtures (posts, and boxes taller than 5 cm) as planar discs (centre, radius) in task-local xy."""
        out = []
        for f in self.scene.get("fixtures", []):
            if f.get("role") == "post":
                out.append((np.asarray(f["pos"][:2], float), float(f["radius"])))
        return out

    def path_clearance(self, p0_local, direction, length):
        """Smallest planar distance from the straight stroke segment to a post surface (inf without posts)."""
        p0 = np.asarray(p0_local, float)[:2]
        d = np.asarray(direction, float)[:2]
        d = d / np.linalg.norm(d)
        best = np.inf
        for c, r in self.posts():
            t = np.clip(np.dot(c - p0, d), 0, length)
            best = min(best, float(np.linalg.norm(p0 + t * d - c)) - r)
        return best

    def clear_direction(self, p0_local, direction, length, need=.03):
        """The stroke direction nearest the requested one (0, +-15, +-30, +-45, +-60 deg) whose straight path keeps
        `need` from every post (the pulled strand and the pinching fingers pass beside it)."""
        d0 = np.asarray(direction, float)
        base = np.arctan2(d0[1], d0[0])
        for ang in (0, 15, -15, 30, -30, 45, -45, 60, -60):
            a = base + np.deg2rad(ang)
            d = np.array([np.cos(a), np.sin(a), 0.])
            if self.path_clearance(p0_local, d, length) >= need:
                return d
        return d0

    # ------------------------------------------------------------ the oracle's moves
    def delete_first_under(self, rope, k, *, stroke_max=.18, lift=.02, pin=True, pull_offset=.035):
        """One anchored node deletion (with regrasping strokes)."""
        lab = self.lab
        rec = dict(k=k, kind="delete", rope=rope, crossings_before=self.count(), strokes=[])
        fae = self.free_at_end(rope)
        sign = -1. if fae else 1.            # toward the body
        L = lab.length(rope)
        rem_prev = None
        for stroke_i in range(14):
            target, rem = self.first_under(rope)
            if target is None:
                rec["result"] = "deleted (no under-crossing left from the free end)"
                rec["done"] = True
                break
            if rem_prev is not None and rem > rem_prev + .05:
                rec["result"] = "deleted (the next under-crossing is further from the end)"
                rec["done"] = True
                break
            rem_prev = rem
            s_u = target["s"]
            s_pull = self.pinch_station(rope, s_u + sign * pull_offset, toward=sign, search=.04, one_sided=True)
            if s_pull is None and not rec.get("loosened"):
                rec["loosened"] = self.loosen(rope)
                continue
            if s_pull is None:
                # still no room close to the crossing: pull further along the body side (the strand between the
                # crossing and the pinch is drawn along with it)
                s_pull = self.pinch_station(rope, s_u + sign * (pull_offset + .04), toward=sign, search=.06,
                                            one_sided=True)
            if s_pull is None:
                rec["result"] = "no pinch room past the crossing"
                break
            p_pull = self.local(rope, s_pull)
            pull_hand = self.arm_for(p_pull)
            if pull_hand is None:
                rec["result"] = "pull station out of reach"
                break
            pin_hand, s_pin = None, None
            if pin:
                orope, so = target["other"]
                # pin the over strand beside the crossing, on the side away from the pull station
                cand = self.pinch_station(orope, so, toward=0, search=.05, exclude=(s_pull,) if orope == rope else ())
                if cand is not None:
                    other = "left" if pull_hand == "right" else "right"
                    if self.reachable(other, self.local(orope, cand)):
                        pin_hand, s_pin = other, cand
                    else:
                        alt = self.arm_for(p_pull, busy=(pull_hand,))
                        if alt is not None and self.reachable(pull_hand, self.local(orope, cand)):
                            pin_hand, pull_hand, s_pin = pull_hand, alt, cand
            direction = lab.point(rope, s_pull) - lab.point(rope, s_u)
            direction[2] = 0.
            direction /= np.linalg.norm(direction)
            direction = self.clear_direction(to_local(lab.point(rope, s_pull)), direction, stroke_max)
            st = dict(s_under=round(s_u, 4), s_pull=round(s_pull, 4), pull_hand=pull_hand, pin_hand=pin_hand,
                      s_pin=None if s_pin is None else round(s_pin, 4), dir=direction.round(3).tolist())
            try:
                if pin_hand is not None and not self.holding(pin_hand):
                    self.take(pin_hand, target["other"][0], s_pin, label="grasp_pin")
                self.take(pull_hand, rope, s_pull, label="grasp" if stroke_i == 0 else "regrasp")
            except Stop as e:
                st["error"] = str(e)
                rec["strokes"].append(st)
                self.release_all()
                rec["result"] = "grasp failed"
                break
            if rec.get("turn"):
                a = np.arctan2(direction[1], direction[0]) + np.deg2rad(rec["turn"])
                direction = np.array([np.cos(a), np.sin(a), 0.])
            self.lift(pull_hand, lift)
            n0 = self.count()

            def deleted(r0=rem):
                t, r = self.first_under(rope)
                return t is None or r > r0 + .05
            moved, why = self.stroke(pull_hand, direction, stroke_max, step=.005, stop_force=3.5, watch=deleted)
            st.update(moved=round(moved, 3), stop=why, crossings=self.count(),
                      max_force=round(float(lab.max_force[pull_hand]), 2))
            rec["strokes"].append(st)
            self.say("  stroke", json.dumps(st))
            self.drop(pull_hand)
            if why == "watch":
                continue                     # the next loop pass confirms the deletion and ends
            if why in ("force", "slip") and moved < .05:
                # jammed: loosen once, then try the stroke turned 40 deg either side
                tries = rec.setdefault("jam_recovery", [])
                if not rec.get("loosened"):
                    rec["loosened"] = self.loosen(rope)
                    tries.append("loosen")
                elif len(tries) < 3:
                    rec["turn"] = 40. if len(tries) == 1 else -40.
                    tries.append(f"turn {rec['turn']:+.0f} deg")
                else:
                    rec["result"] = f"jammed ({why}) after loosening and turned strokes"
                    break
        if pin:
            for h in list(lab.hands):
                if self.holding(h):
                    self.drop(h)
        rec["crossings_after"] = self.count()
        return rec

    def loosen(self, rope, push=.025, lift=.015):
        """Loosen a snug knot: pinch both sides just outside the knot (standing side and tail side) and push them
        toward the knot centre by `push` each (feeding rope into the core), release. Returns a record."""
        from .birth import knot_spans
        lab = self.lab
        spans = knot_spans(lab, rope)
        rec = dict(kind="loosen", rope=rope)
        if not spans:
            rec["result"] = "no knot"
            return rec
        fae = self.free_at_end(rope)
        span = spans[-1] if fae else spans[0]            # the knot nearest the free end
        s_a = self.pinch_station(rope, span[0] - .02, toward=-1, search=.06, one_sided=True)
        s_b = self.pinch_station(rope, span[1] + .02, toward=1, search=.06, one_sided=True)
        rec.update(span=list(span), stations=[s_a, s_b])
        if s_a is None or s_b is None:
            rec["result"] = "no pinch room beside the knot"
            return rec
        pa, pb = self.local(rope, s_a), self.local(rope, s_b)
        ha = self.arm_for(pa)
        hb = self.arm_for(pb, busy=(ha,))
        c = lab.point(rope, .5 * (span[0] + span[1]))
        if ha is None and hb is None:
            hb = self.arm_for(pb)
        if ha is None or hb is None:
            # one arm reaches the knot: push one side at a time (tail side first)
            pushes = [(h, st) for h, st in ((self.arm_for(pb), s_b), (self.arm_for(pa), s_a)) if h is not None]
            if not pushes:
                rec["result"] = "knot sides out of reach"
                return rec
            for h, st in pushes:
                try:
                    self.take(h, rope, st, label="grasp_loosen")
                except Stop as e:
                    rec.setdefault("errors", []).append(str(e))
                    continue
                self.lift(h, lift)
                d = (c - lab.hand_pos(h)) * [1, 1, 0]
                n = np.linalg.norm(d)
                t0 = lab.t
                lab.move({h: lab.hand_pos(h) + d / max(n, 1e-9) * min(push, max(0., n - .015))}, .02)
                self.phase(h, "loosen", t0)
                lab.wait(.2)
                self.drop(h)
            rec["result"] = "pushed one side at a time"
            rec["crossings_after"] = self.count()
            self.decisions.append(rec)
            return rec
        try:
            self.take(ha, rope, s_a, label="grasp_loosen")
            self.take(hb, rope, s_b, label="grasp_loosen")
        except Stop as e:
            rec["result"] = str(e)
            self.release_all()
            return rec
        self.lift(ha, lift)
        self.lift(hb, lift)
        moves = {}
        for h in (ha, hb):
            d = (c - lab.hand_pos(h)) * [1, 1, 0]
            n = np.linalg.norm(d)
            moves[h] = lab.hand_pos(h) + d / max(n, 1e-9) * min(push, max(0., n - .015))
        t0 = lab.t
        lab.move(moves, .02)
        self.phase(ha, "loosen", t0)
        self.phase(hb, "loosen", t0)
        lab.wait(.2)
        self.drop(ha)
        self.drop(hb)
        rec["result"] = "pushed"
        rec["crossings_after"] = self.count()
        self.decisions.append(rec)
        self.say("loosen", json.dumps(rec, default=float))
        return rec

    def draw_free_end(self, rope, k, *, length=.25, lift=.02, margin=0.):
        """R-I curls on an anchored rope: draw the free end away from the clamp (the clamp is the other hand)."""
        lab = self.lab
        L = lab.length(rope)
        fae = self.free_at_end(rope)
        s_end = self.pinch_station(rope, L - .025 if fae else .025, toward=-1 if fae else 1, search=.10,
                                   one_sided=True)
        s_anchor = 0. if fae else L
        rec = dict(k=k, kind="draw_free_end", rope=rope, crossings_before=self.count())
        if s_end is None:
            rec["result"] = "no pinch room near the free end"
            return rec
        p = self.local(rope, s_end)
        hand = self.arm_for(p)
        rec["hand"] = hand
        if hand is None:
            rec["result"] = "free end out of reach"
            return rec
        try:
            self.take(hand, rope, s_end)
        except Stop as e:
            rec["result"] = str(e)
            return rec
        self.lift(hand, lift)
        d = lab.point(rope, s_end) - lab.point(rope, s_anchor)
        d[2] = 0
        moved, why = self.stroke(hand, d, length, stop_force=4.5, watch=lambda: self.count() == 0, margin=margin)
        rec.update(moved=round(moved, 3), stop=why)
        self.drop(hand)
        rec["crossings_after"] = self.count()
        return rec

    def run_oracle(self, rope, *, max_decisions=10, stroke_max=.18):
        """Untie `rope` from its free end: node deletions (pin beside the crossing, strokes, regrasps); the free end is
        drawn against the clamp when no under-crossing remains or a deletion made no progress."""
        for k in range(1, max_decisions + 1):
            vs = self.visits(rope)
            if not vs:
                break
            n = len(vs)
            if all(v["over"] and v["other"][0] != rope for v in vs):
                # only lying on top of another rope: the layout lifts it off into its region
                self.decisions.append(dict(k=k, kind="defer_to_layout", rope=rope, visits=n))
                self.deferred.add(rope)
                break
            if not any(not v["over"] for v in vs):
                rec = self.draw_free_end(rope, k)
            else:
                rec = self.delete_first_under(rope, k, stroke_max=stroke_max)
                if not rec.get("done") and len(self.visits(rope)) >= n:
                    self.decisions.append(rec)
                    rec = self.draw_free_end(rope, k)
            rec["visits_after"] = len(self.visits(rope))
            self.decisions.append(rec)
            self.say("decision", json.dumps({x: rec.get(x) for x in ("k", "kind", "crossings_before", "crossings_after",
                                                                       "visits_after", "result")}))
        self.release_all()
        return len(self.visits(rope)) == 0

    # ------------------------------------------------------------ v7 moves (coordinator 10-10 19:00; main approved 18:57)
    def post_tops(self):
        """(centre xy, radius, top z) of every post, task-local."""
        out = []
        for f in self.scene.get("fixtures", []):
            if f.get("role") == "post":
                top = float(f["pos"][2]) + .5 * float(f.get("height", 2 * float(f["pos"][2])))
                out.append((np.asarray(f["pos"][:2], float), float(f["radius"]), top))
        return out

    def _rest_xy(self, rope, s0, s1):
        """Task-local xy of every rope's nodes except `rope`'s material in [s0, s1]."""
        pts = []
        for name, rod in self.lab.rods.items():
            nodes = to_local(rod.nodes(self.lab.d))[:, :2]
            if name == rope:
                nodes = nodes[(rod.s_nodes < s0) | (rod.s_nodes > s1)]
            pts.append(nodes)
        return np.concatenate(pts) if pts else np.zeros((0, 2))

    def clear_tail(self, rope, k, *, lift=.06, margin=.05, min_len=.08, max_len=.25, moving=.35):
        """The "clear the tail" move (main 18:57): the deletions pull slack that falls back across the rope as new
        loops (easier / easy: 5 of 8 deletions added crossings). The hand takes the free end, lifts it `lift` (above
        every strand) and carries it up to `max_len` along the clearest direction, so the end and the rope behind it
        are drawn out of the knot. Direction: the one of 24 from the end whose drawn line keeps the largest distance
        from the rest of the rope (all but the last `moving` m, which follows the end) and from the posts; the line
        stays on the table (|x| <= 0.42, |y| <= 0.30), off the bases' keep-out, and inside the carrying arm's reach
        with `margin` to spare. (19:35: the first version measured the tail from the crossing nearest the free end,
        which is often at the end itself, and skipped the move as "tail too short".)"""
        lab = self.lab
        L = lab.length(rope)
        fae = self.free_at_end(rope)
        rec = dict(k=k, kind="clear_tail", rope=rope, crossings_before=self.count())
        s_end = self.pinch_station(rope, L - .025 if fae else .025, toward=-1 if fae else 1, search=.10, one_sided=True)
        if s_end is None:
            rec["result"] = "no pinch room near the free end"
            return rec
        p_end = self.local(rope, s_end)
        hand = self.arm_for(p_end)
        if hand is None:
            rec["result"] = "free end out of reach"
            return rec
        b = p_end[:2]
        t0, t1 = (L - moving, L) if fae else (0., moving)
        rest = self._rest_xy(rope, t0, t1)
        posts = self.post_tops()
        best = None
        def box_out(q):           # distance outside the parking box |x| <= 0.42, |y| <= 0.30 (0 inside)
            return float(np.linalg.norm(np.maximum(np.abs(q) - np.array([.42, .30]), 0.)))
        for ang in np.deg2rad(np.arange(0, 360, 15)):
            d = np.array([np.cos(ang), np.sin(ang)])
            ell = 0.
            prev_out = box_out(b)
            prev_r = float(np.linalg.norm(b - BASES_LOCAL[hand]))
            for x in np.arange(.02, max_len + 1e-9, .01):
                q = b + x * d
                out_q = box_out(q)
                r_q = float(np.linalg.norm(q - BASES_LOCAL[hand]))
                inside = out_q == 0. and self.reachable(hand, np.r_[q, 0.], margin)
                # an end lying beyond the box or the reach margin (19:35: medium's end at y 0.38, 2 cm inside the
                # reach limit) may move back towards them
                ok = (self.reachable(hand, np.r_[q, 0.]) and not len(keep_out_violations(q[None]))
                      and (inside or (out_q <= prev_out and r_q <= prev_r)))
                if not ok:
                    break
                prev_out, prev_r = out_q, r_q
                if inside:
                    ell = x
            if ell < min_len:
                continue
            line = b + np.outer(np.arange(.03, ell + 1e-9, .01), d)
            clear = float(np.min(np.linalg.norm(line[:, None, :] - rest[None], axis=2))) if len(rest) else 1.
            for c, r, _ in posts:
                clear = min(clear, float(np.min(np.linalg.norm(line - c, axis=1))) - r)
            score = min(clear, .10) + .2 * ell
            if best is None or score > best[0]:
                best = (score, d, ell, clear)
        if best is None:
            rec["result"] = "no reachable clear direction"
            return rec
        _, d, ell, clear = best
        park = b + ell * d
        z = lab.rods[rope].radius + .001
        rec.update(hand=hand, park=park.round(3).tolist(), length=round(ell, 3), clearance=round(clear, 3))
        try:
            self.take(hand, rope, s_end, label="grasp_tail")
            self.lift(hand, lift, speed=.08)
            self.goto(hand, to_world(np.r_[park, z + lift]), speed=.08, label="clear_tail")
            self.drop(hand)
        except Stop as e:
            rec["result"] = str(e)
            self.release_all()
        rec["crossings_after"] = self.count()
        return rec

    def post_wraps(self, rope, near=.05, min_deg=270.):
        """Wraps of `rope` round a post: runs of nodes within `near` of the post's surface whose angle about the post
        turns by at least `min_deg`."""
        rod = self.lab.rods[rope]
        nodes = to_local(rod.nodes(self.lab.d))
        out = []
        for pi, (c, r, top) in enumerate(self.post_tops()):
            idx = np.flatnonzero(np.linalg.norm(nodes[:, :2] - c, axis=1) < r + near)
            if len(idx) < 3:
                continue
            for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                if len(run) < 3:
                    continue
                ang = np.unwrap(np.arctan2(nodes[run, 1] - c[1], nodes[run, 0] - c[0]))
                w = float(np.degrees(ang[-1] - ang[0]))
                if abs(w) >= min_deg:
                    out.append(dict(post=pi, centre=c, radius=r, top=top, winding=round(w, 1),
                                    s0=float(rod.s_nodes[run[0]]), s1=float(rod.s_nodes[run[-1]])))
        return out

    def lift_over_post(self, rope, k, wrap, *, clear=.04, out=.08, speed=.04):
        """The "lift over the post" move (main 18:57: medium's last 2 crossings are a wrap round a 16 cm post that no
        pull removes; 62 of 75 strokes slipped). Pinch the wrap at its middle, lift straight up to `clear` above the
        post top (the loop slides up the post with it), carry it `out` away from the post on the pinch side, and lay
        it down there."""
        lab = self.lab
        rec = dict(k=k, kind="lift_over_post", rope=rope, post=wrap["post"], winding=wrap["winding"],
                   crossings_before=self.count())
        st = self.pinch_station(rope, .5 * (wrap["s0"] + wrap["s1"]), search=.05)
        if st is None:
            rec["result"] = "no pinch room on the wrap"
            return rec
        p = self.local(rope, st)
        hand = self.arm_for(p)
        if hand is None:
            rec["result"] = "wrap out of reach"
            return rec
        c = wrap["centre"]
        d = p[:2] - c
        d = d / max(np.linalg.norm(d), 1e-9)
        q = p[:2] + out * d
        z_up = wrap["top"] + clear
        z = lab.rods[rope].radius + .001
        rec.update(hand=hand, station=round(st, 3))
        try:
            self.take(hand, rope, st, label="grasp_wrap")
            t0 = lab.t
            lab.move({hand: to_world(np.r_[p[:2], z_up])}, speed)
            self.phase(hand, "lift_wrap", t0)
            if not self.holding(hand):
                rec["result"] = "slip while lifting"
            else:
                if not self.reachable(hand, np.r_[q, 0.]):
                    q = p[:2] + .5 * out * d
                self.goto(hand, to_world(np.r_[q, z_up]), speed=speed, label="carry_off_post")
                self.goto(hand, to_world(np.r_[q, z + .03]), speed=speed, label="lower")
                self.drop(hand)
        except Stop as e:
            rec["result"] = str(e)
            self.release_all()
        rec["wraps_after"] = len(self.post_wraps(rope))
        rec["crossings_after"] = self.count()
        return rec

    def run_oracle2(self, rope, *, max_decisions=16, stroke_max=.18):
        """run_oracle with the v7 moves: a wrap round a post is lifted off first; every deletion is followed by
        clearing the tail; the free end is drawn only as far as an arm keeps reaching it (5 cm margin)."""
        lifted = {}
        stalled = False
        for k in range(1, max_decisions + 1):
            vs = self.visits(rope)
            if not vs:
                break
            n = len(vs)
            if all(v["over"] and v["other"][0] != rope for v in vs):
                self.decisions.append(dict(k=k, kind="defer_to_layout", rope=rope, visits=n))
                self.deferred.add(rope)
                break
            # knot agent 10-10: a wrap round a post is a last resort (the design's order: the stopper, the knot's
            # tucks, then the post); from the birth, lifting first took medium 7 -> 11 crossings
            wraps = [w for w in self.post_wraps(rope) if lifted.get(w["post"], 0) < 2]
            if wraps and (stalled or not any(not v["over"] for v in vs)):
                w = max(wraps, key=lambda x: abs(x["winding"]))
                lifted[w["post"]] = lifted.get(w["post"], 0) + 1
                rec = self.lift_over_post(rope, k, w)
                stalled = False
            elif not any(not v["over"] for v in vs):
                rec = self.draw_free_end(rope, k, margin=.05)
            else:
                rec = self.delete_first_under(rope, k, stroke_max=stroke_max)
                stalled = not rec.get("done") and len(self.visits(rope)) >= n
                if self.visits(rope):
                    self.decisions.append(rec)
                    self.say("decision", json.dumps({x: rec.get(x) for x in ("k", "kind", "crossings_before",
                                                                               "crossings_after", "result")}))
                    rec = self.clear_tail(rope, k)
            rec["visits_after"] = len(self.visits(rope))
            self.decisions.append(rec)
            self.say("decision", json.dumps({x: rec.get(x) for x in ("k", "kind", "crossings_before", "crossings_after",
                                                                       "visits_after", "result", "park", "winding")}))
        self.release_all()
        return len(self.visits(rope)) == 0

    # ------------------------------------------------------------ layout into the region
    def layout(self, rope, path_local, *, spacing=.14, lift=.05, settle=.3):
        """Hand-over-hand placement of the rope along `path_local` (a polyline in task-local xy starting at the clamp,
        inside the region): station s_i of the rope is carried to the path point at arc length s_i (lifted `lift`,
        over curbs), from the clamp outward; the previous placed station is pinned by the other arm while the next is
        carried when that arm reaches it; a carry that leaves the carrying arm's reach is handed over in the lens."""
        lab = self.lab
        P = np.asarray(path_local, float)
        P = P[np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6]]     # drop repeated points
        seg = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        L = lab.length(rope)
        rec = dict(kind="layout", rope=rope, placements=[], handovers=0)
        z_rope = lab.rods[rope].radius + .001
        s_list = list(np.arange(spacing, min(L - .03, seg[-1]) + 1e-9, spacing))
        # the path's turns get their own placements (start / apex / end of every bend): a chord across a U-turn
        # would otherwise lay the two legs against each other
        d = np.diff(P, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])
        turn = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)
        bends = [float(seg[i + 1]) for i in np.flatnonzero(turn > np.deg2rad(10))]
        if bends:
            groups, cur = [], [bends[0]]
            for b in bends[1:]:
                if b - cur[-1] < .06:
                    cur.append(b)
                else:
                    groups.append(cur)
                    cur = [b]
            groups.append(cur)
            extra = []
            for g in groups:
                extra += [g[0], .5 * (g[0] + g[-1]), g[-1]]
            s_list = sorted(set([round(x, 3) for x in s_list + [x for x in extra if .03 < x < L - .03]]))
            pruned = []
            for x in s_list:
                if not pruned or x - pruned[-1] >= .03:
                    pruned.append(x)
            s_list = pruned
        if L - .03 > s_list[-1] + .03:
            s_list.append(L - .03)
        prev = None
        for s_i in s_list:
            tgt = np.r_[np.interp(min(s_i, seg[-1]), seg, P[:, 0]), np.interp(min(s_i, seg[-1]), seg, P[:, 1]), z_rope]
            st = self.pinch_station(rope, s_i, search=.04)
            if st is None:
                rec["placements"].append(dict(s=s_i, error="no pinch room"))
                continue
            here = self.local(rope, st)
            carrier = self.arm_for(here, prefer=self.arm_for(tgt))
            if carrier is None:
                rec["placements"].append(dict(s=s_i, error="station out of reach"))
                continue
            pin = None
            if prev is not None:
                other = "left" if carrier == "right" else "right"
                if self.reachable(other, self.local(rope, prev)) and self.lab.pinch_clearance_at(rope, prev) >= lab.pinch_clearance:
                    try:
                        self.take(other, rope, prev, label="grasp_pin")
                        pin = other
                    except Stop:
                        pin = None
            try:
                self.take(carrier, rope, st)
                self.lift(carrier, lift)
                hops = []
                if not self.reachable(carrier, tgt):
                    # hand-over in the lens: carry to the lens point nearest the target, the other arm takes over
                    other = "left" if carrier == "right" else "right"
                    if pin == other:
                        self.drop(other)
                        pin = None
                    lens = np.array([0., float(np.clip(tgt[1], -.25, .25)), 0.])
                    self.goto(carrier, to_world(np.r_[lens[:2], z_rope + lift]))
                    s_take = self.pinch_station(rope, st + (.02 if st < L - .05 else -.02), search=.02)
                    if s_take is None:
                        raise Stop("no room for the hand-over pinch")
                    p_take = self.lab.point(rope, s_take)
                    self.lab.move({other: p_take + [0, 0, PRE]}, .12)
                    self.lab.move({other: p_take}, .05)
                    self.lab.grasp(other, rope, s_take)
                    self.lab.wait(.1)
                    self.lab.release(carrier)
                    self.lab.move({carrier: self.lab.hand_pos(carrier) + [0, 0, PRE]}, .1)
                    rec["handovers"] += 1
                    hops.append("handover")
                    carrier = other
                self.goto(carrier, to_world(np.r_[tgt[:2], z_rope + lift]))
                self.drop(carrier)
            except Stop as e:
                rec["placements"].append(dict(s=s_i, error=str(e)))
                self.release_all()
                continue
            if pin is not None and self.holding(pin):
                self.drop(pin)
            rec["placements"].append(dict(s=round(s_i, 3), station=round(st, 3), target=tgt[:2].round(3).tolist(),
                                          carrier=carrier, hops=hops))
            prev = st
            lab.wait(settle)
        self.release_all()
        rec["crossings_after"] = self.count()
        self.decisions.append(rec)
        return rec

    def layout_fixup(self, rope, region_local, *, passes=4, inset=.035, lift=.05, settle=.3, min_run=.02):
        """After layout (knot agent 10-11): the rope parts still outside the region polygon (task-local xy) are carried
        back in, one contiguous outside run per carry (pinched at the run's middle, carried to the nearest point of the
        region's bounding box shrunk by `inset`, lifted `lift`, over curbs), up to `passes` rounds."""
        lab = self.lab
        poly = np.asarray(region_local, float)
        x0, y0 = poly.min(axis=0) + inset
        x1, y1 = poly.max(axis=0) - inset

        def inside(q):
            n, c = len(poly), np.zeros(len(q), bool)
            for i in range(n):
                a, b = poly[i], poly[(i + 1) % n]
                cond = (a[1] > q[:, 1]) != (b[1] > q[:, 1])
                xs = a[0] + (q[:, 1] - a[1]) * (b[0] - a[0]) / np.where(b[1] - a[1] == 0, 1e-12, b[1] - a[1])
                c ^= cond & (q[:, 0] < xs)
            return c
        rec = dict(kind="layout_fixup", rope=rope, carries=[])
        z_rope = lab.rods[rope].radius + .001
        for _ in range(passes):
            rod = lab.rods[rope]
            S = np.asarray(rod.s_nodes)
            Q = np.asarray([to_local(p)[:2] for p in rod.nodes(lab.d)])
            out = ~inside(Q)
            if not out.any():
                break
            runs, cur = [], None
            for i, o in enumerate(out):
                if o and cur is None:
                    cur = i
                if not o and cur is not None:
                    runs.append((cur, i - 1)); cur = None
            if cur is not None:
                runs.append((cur, len(out) - 1))
            runs = [r for r in runs if S[r[1]] - S[r[0]] >= min_run] or runs
            runs.sort(key=lambda r: -(S[r[1]] - S[r[0]]))
            i0, i1 = runs[0]
            s_mid = float(.5 * (S[i0] + S[i1]))
            st = self.pinch_station(rope, s_mid, search=.04)
            if st is None:
                rec["carries"].append(dict(s=round(s_mid, 3), error="no pinch room")); break
            here = self.local(rope, st)
            tgt = np.r_[np.clip(here[0], x0, x1), np.clip(here[1], y0, y1), z_rope]
            carrier = self.arm_for(here, prefer=self.arm_for(tgt))
            if carrier is None or not self.reachable(carrier, tgt):
                rec["carries"].append(dict(s=round(st, 3), error="out of reach")); break
            try:
                self.take(carrier, rope, st)
                self.lift(carrier, lift)
                self.goto(carrier, to_world(np.r_[tgt[:2], z_rope + lift]))
                self.drop(carrier)
            except Stop as e:
                rec["carries"].append(dict(s=round(st, 3), error=str(e)))
                self.release_all()
                break
            rec["carries"].append(dict(s=round(st, 3), run=[round(float(S[i0]), 3), round(float(S[i1]), 3)],
                                       target=tgt[:2].round(3).tolist(), carrier=carrier))
            lab.wait(settle)
        self.release_all()
        rod = lab.rods[rope]
        rec["outside_fraction"] = float((~inside(np.asarray([to_local(p)[:2] for p in rod.nodes(lab.d)]))).mean())
        rec["crossings_after"] = self.count()
        self.decisions.append(rec)
        return rec

    # ------------------------------------------------------------ breakers
    def run_pull_free(self, rope, length=.40):
        lab = self.lab
        L = lab.length(rope)
        fae = self.free_at_end(rope)
        s_end = self.pinch_station(rope, L - .02 if fae else .02, toward=-1 if fae else 1, search=.10, one_sided=True)
        rec = dict(kind="pull_free", rope=rope, crossings_before=self.count())
        if s_end is None:
            rec["result"] = "no pinch room near the free end"
            self.decisions.append(rec)
            return rec
        p = self.local(rope, s_end)
        hand = self.arm_for(p)
        rec["hand"] = hand
        if hand is None:
            rec["result"] = "free end out of reach"
            self.decisions.append(rec)
            return rec
        self.take(hand, rope, s_end)
        self.lift(hand, .02)
        d = lab.point(rope, s_end) - lab.point(rope, 0. if fae else L)
        d[2] = 0
        moved, why = self.stroke(hand, d, length, stop_force=self.breaker_stop)
        rec.update(moved=round(moved, 3), stop=why, max_force=round(float(lab.max_force[hand]), 2),
                   crossings_during=self.count())
        if self.holding(hand):
            self.drop(hand)
        rec["crossings_after"] = self.count()
        self.decisions.append(rec)
        return rec

    def run_route_rule(self, rope, *, hold="far", direction="away", max_pull=.55, lift=.03, max_decisions=16,
                       straighten=.9):
        """The family's rule (topogym/planning/knot_untying/route_rule.py) as robot_oracle.Runner uses it."""
        from topogym.planning.knot_untying.route_rule import choose, only_local_loops
        from topogym.assets.geometry.knot_untying import projected_crossings as family_crossings
        lab = self.lab
        rod = lab.rods[rope]
        L = lab.length(rope)
        for k in range(1, max_decisions + 1):
            line = to_local(rod.nodes(lab.d))
            cr = family_crossings(line)
            if not cr:
                break
            rec = dict(k=k, crossings_before=len(cr))
            if only_local_loops(cr, rod.s_nodes) and not (self.decisions and self.decisions[-1].get("kind") == "straighten"):
                rec["kind"] = "straighten"
                ends = (.035, L - .035)
                sides = []
                for s in ends:
                    st = self.pinch_station(rope, s, search=.0)
                    sides.append(st)
                rec["end_grasps"] = sides
                held = []
                for s in sides:
                    if s is None:
                        continue
                    hand = self.arm_for(self.local(rope, s), busy=held)
                    if hand is None:
                        continue
                    try:
                        self.take(hand, rope, s)
                        held.append(hand)
                    except Stop as e:
                        rec.setdefault("errors", []).append(str(e))
                for h in held:
                    self.lift(h, .015)
                if len(held) == 2:
                    pa, pb = lab.hand_pos(held[0]), lab.hand_pos(held[1])
                    axis = (pb - pa) * [1, 1, 0]
                    axis /= np.linalg.norm(axis)
                    for _ in range(40):
                        span = float(np.linalg.norm((lab.hand_pos(held[1]) - lab.hand_pos(held[0]))[:2]))
                        if span >= straighten * L or self.count() == 0:
                            break
                        a, b = lab.hand_pos(held[0]) - .025 * axis, lab.hand_pos(held[1]) + .025 * axis
                        if not (self.reachable(held[0], to_local(a)) and self.reachable(held[1], to_local(b))):
                            rec["stopped"] = "reach limit"
                            break
                        lab.move({held[0]: a, held[1]: b}, self.speed)
                        if not all(self.holding(h) for h in held):
                            rec["stopped"] = "slip"
                            break
                elif len(held) == 1:
                    # one end clamped: the rule's spread degenerates to pulling the held end away from the clamp
                    h = held[0]
                    d = lab.hand_pos(h) - lab.point(rope, 0. if self.free_at_end(rope) else L)
                    d[2] = 0
                    self.stroke(h, d, .3)
                    rec["stopped"] = "one end clamped: single-end spread"
                self.release_all()
            else:
                action = choose(line, rod.s_nodes, cr, L, hold, direction)
                rec["kind"] = "withdraw"
                if action is None:
                    rec["result"] = "no action"
                    self.decisions.append(rec)
                    break
                rec.update({x: action[x] for x in ("end", "s_under", "s_pull", "s_anchor")})
                s_pull = self.pinch_station(rope, action["s_pull"], search=.03)
                s_anc = self.pinch_station(rope, action["s_anchor"], search=.0)
                rec["anchor_pinch"] = s_anc
                held = []
                try:
                    if s_anc is not None:
                        ha = self.arm_for(self.local(rope, s_anc))
                        if ha is not None:
                            self.take(ha, rope, s_anc, label="grasp_pin")
                            held.append(ha)
                            self.lift(ha, .002)
                    if s_pull is None:
                        raise Stop("no pinch room at the pull band")
                    hp = self.arm_for(self.local(rope, s_pull), busy=held)
                    if hp is None:
                        raise Stop("pull band out of reach of a free arm")
                    self.take(hp, rope, s_pull)
                    held.append(hp)
                except Stop as e:
                    rec["error"] = str(e)
                    self.release_all()
                    self.decisions.append(rec)
                    continue
                self.lift(hp, lift)
                d = np.asarray(action["direction"], float)
                d = np.r_[d[:2], 0.] if len(d) == 3 else np.r_[d, 0.]
                moved, why = self.stroke(hp, d, max_pull, step=.05 / 4, stop_force=self.breaker_stop)
                rec.update(moved=round(moved, 3), stop=why, max_force=round(float(lab.max_force[hp]), 2))
                self.release_all()
            rec["crossings_after"] = self.count()
            self.decisions.append(rec)
            self.say("route_rule", json.dumps({x: rec.get(x) for x in ("k", "kind", "crossings_before",
                                                                         "crossings_after", "stop", "error", "stopped")}))
        return self.count() == 0

    def run_replay_old(self, npz_path, *, align=True, close_below=.0065, open_above=.0075, capture=.012,
                       control_dt=.025):
        """The strict reference retargeted: its recorded TCP paths (tooling lane's antihack.record: tcp (N, 2, 7) world
        pos + wxyz) and grip events (commanded opening width crossing `close_below` / `open_above`: knot grasps are
        physical, no weld flag) replayed by the lab hands row by row (25 ms each). align: the paths are translated so the
        old knot's crossing centroid lands on the new knot's (the old oracle's best chance). A closing hand grasps the
        rope link nearest its pad contact point if it lies within `capture` and a parallel gripper fits there."""
        from scipy.spatial.transform import Rotation
        from topogym.tasks.knot_untying.redesign.geometry import projected_crossings
        lab = self.lab
        z = np.load(npz_path, allow_pickle=False)
        tcp = np.asarray(z["tcp"], float)
        width = np.asarray(z["targets"], float)[:, [6, 13]]
        arms = [str(a) for a in z["arms"]]
        contact = np.empty((len(tcp), 2, 3))
        for k in range(2):
            q = tcp[:, k, 3:7]
            contact[:, k] = tcp[:, k, :3] + Rotation.from_quat(q[:, [1, 2, 3, 0]]).apply([.01381, 0., -.00055])
        shift = np.zeros(3)
        rec = dict(kind="replay_old", source=str(npz_path), rows=len(tcp), aligned=bool(align), grasps=[], missed=[])
        if align:
            old_nodes = np.asarray(z["obj:rope/rope/nodes"][0], float)
            c_old = np.mean([c["xy"] for c in projected_crossings([old_nodes])], axis=0)
            new_lines = [r.nodes(lab.d) for r in lab.rods.values()]
            cr = projected_crossings(new_lines)
            c_new = np.mean([c["xy"] for c in cr], axis=0) if cr else c_old
            shift[:2] = np.asarray(c_new) - np.asarray(c_old)
            rec["shift_world_m"] = shift.round(4).tolist()
        steps = int(round(control_dt / lab.P["dt"]))
        closed = [False, False]
        for h in lab.hands:
            lab.place_hand(h, contact[0, arms.index(h)] + shift, lab.d.mocap_quat[lab.hands[h]["mocap"]].copy())
        for i in range(len(tcp)):
            for k, h in enumerate(arms):
                lab.set_hand(h, contact[i, k] + shift)
                w = width[i, k]
                if not closed[k] and w < close_below:
                    closed[k] = True
                    p = lab.hand_actual(h)
                    best = None
                    for name, rod in lab.rods.items():
                        c = rod.centres(lab.d)
                        j = int(np.argmin(np.linalg.norm(c - p, axis=1)))
                        dist = float(np.linalg.norm(c[j] - p))
                        if best is None or dist < best[0]:
                            sc = .5 * (rod.s_nodes[j] + rod.s_nodes[j + 1])
                            best = (dist, name, float(sc))
                    if best is not None and best[0] <= capture and lab.pinch_clearance_at(best[1], best[2]) >= lab.pinch_clearance:
                        # weld at the current hand pose (no teleport): the pinch closes where the hand is
                        lab.grasp(h, best[1], best[2], check_pinch=False)
                        rec["grasps"].append(dict(row=i, hand=h, rope=best[1], s=round(best[2], 3), dist_m=round(best[0], 4)))
                    else:
                        rec["missed"].append(dict(row=i, hand=h, nearest_m=None if best is None else round(best[0], 4),
                                                  pinch_room=None if best is None else round(lab.pinch_clearance_at(best[1], best[2]), 4)))
                elif closed[k] and w > open_above:
                    closed[k] = False
                    lab.release(h)
            lab.step(steps)
        self.release_all()
        rec["crossings_after"] = self.count()
        self.decisions.append(rec)
        return rec

    def run_pull_both(self, ropes, *, lift=.03, length=.30):
        """Breaker: the two free ends (two ropes: each rope's free end) grasped by the two arms, lifted 3 cm, pulled
        apart along their connecting line (as the tooling lane's pull_both_ends, which solves the released easy), laid
        down and released. One rope: its free end against the clamp (= pull_free)."""
        lab = self.lab
        ends = []
        for rope in ropes:
            L = lab.length(rope)
            fae = self.free_at_end(rope)
            st = self.pinch_station(rope, L - .02 if fae else .02, toward=-1 if fae else 1, search=.10, one_sided=True)
            if st is not None:
                ends.append((rope, st))
        if len(ropes) == 1 and len(ends) == 1:
            # one anchored rope (knot agent 10-10): its two "ends" are the free end and the standing part just outside
            # the clamp (the clamped links themselves are under the staple): the naive "pull both ends apart"
            rope = ropes[0]
            L = lab.length(rope)
            fae = self.free_at_end(rope)
            st = self.pinch_station(rope, .04 if fae else L - .04, toward=1 if fae else -1, search=.12, one_sided=True)
            if st is not None:
                ends.insert(0, (rope, st))
        rec = dict(kind="pull_both_ends", ends=ends, crossings_before=self.count())
        if len(ends) < 2:
            rec["result"] = "fewer than two free ends: see pull_free"
            self.decisions.append(rec)
            return rec
        (ra, sa), (rb, sb) = ends[:2]
        pa, pb = self.local(ra, sa), self.local(rb, sb)
        ha = self.arm_for(pa)
        hb = self.arm_for(pb, busy=(ha,))
        if ha is None or hb is None:
            rec["result"] = f"ends not reachable by two arms ({ha}, {hb})"
            self.decisions.append(rec)
            return rec
        try:
            self.take(ha, ra, sa)
            self.take(hb, rb, sb)
        except Stop as e:
            rec["result"] = str(e)
            self.release_all()
            self.decisions.append(rec)
            return rec
        self.lift(ha, lift)
        self.lift(hb, lift)
        axis = (lab.hand_pos(hb) - lab.hand_pos(ha)) * [1, 1, 0]
        axis /= np.linalg.norm(axis)
        moved = 0.
        while moved < length:
            a_, b_ = lab.hand_pos(ha) - .01 * axis, lab.hand_pos(hb) + .01 * axis
            if not (self.reachable(ha, to_local(a_)) and self.reachable(hb, to_local(b_))):
                rec["stop"] = "reach"
                break
            lab.move({ha: a_, hb: b_}, self.speed, min_time=.02)
            moved += .02
            if not (self.holding(ha) and self.holding(hb)):
                rec["stop"] = "slip"
                break
            if self.breaker_stop is not None and max(float(np.mean(lab.hands[h]["window"])) for h in (ha, hb)) > self.breaker_stop:
                rec["stop"] = "force"
                break
        rec.update(spread_m=round(moved, 3), max_force=round(max(float(lab.max_force[h]) for h in (ha, hb)), 2))
        self.release_all()
        rec["crossings_after"] = self.count()
        self.decisions.append(rec)
        return rec

    def run_drag(self, rope, delta_local=(0., .2, 0.)):
        lab = self.lab
        vs = self.visits(rope)
        s_mid = float(np.mean([v["s"] for v in vs])) if vs else lab.length(rope) / 2
        s = self.pinch_station(rope, s_mid, search=.12)
        rec = dict(kind="drag", s=s, crossings_before=self.count())
        if s is None:
            rec["result"] = "no pinch room near the knot"
            self.decisions.append(rec)
            return rec
        hand = self.arm_for(self.local(rope, s))
        self.take(hand, rope, s)
        self.lift(hand, .03)
        d = np.asarray(delta_local, float)
        moved, why = self.stroke(hand, d, float(np.linalg.norm(d)))
        rec.update(moved=round(moved, 3), stop=why)
        if self.holding(hand):
            self.drop(hand)
        rec["crossings_after"] = self.count()
        self.decisions.append(rec)
        return rec
