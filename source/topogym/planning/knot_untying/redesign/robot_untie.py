"""The redesign's untying oracle (untie.Untie: run_oracle2, the v7 moves, layout) on the real ALOHA: Untie's decision
logic unchanged, its hand primitives re-implemented for robot hands (robot_lab.RobotLab):

  take    vertical pinch at the live rope station: the jaw flip whose IK (at the station and 5 cm above it) is safe
          (wrist rule), clear of the fixtures and >= 12 mm from the other arm (exact geom distance), nearest the arm's
          joints; else an idle other hand backs off, or the station slides along the rope (away from a holding hand);
          up to the transit height (5 cm above the station, or above a post on the way), across with the jaws turning
          to the pinch orientation -- or, if the arm cannot follow that way clear of the other arm and the fixtures
          (checked in plan), over the post and down 7 cm past it, or a joint-space line to the planned pre-grasp
          joints (approach_plan) --, converge (learned TCP bias), re-aim on the live rope, open to the room the strand
          has (2 x (clearance - 3 mm), 14..30 mm), descend to the rope centre (fingertips 1 mm above the table),
          converge, close; the grasp is the World's contact_weld_v1 binding (read, never assumed). One re-aimed retry;
          then Stop (Untie's callers treat it as a failed grasp).
  delete_first_under   Untie's, with the pin and the pull planned together (two ALOHA grippers must fit side by side).
  stroke  Untie.stroke's contract as one continuous motion; stops on slip, force, reach, watch, or the arms closing in.
  goto    Untie.goto + the robot's reach stop (Stop).
  drop    Untie.drop (lower to the table, open, 5 cm up): unchanged, the backend opens the jaws.
  layout  Untie.layout with a two-gripper hand-over (7 cm apart along the rope).
"""
from __future__ import annotations

import json

import numpy as np

from .mjlab import to_local, to_world
from .robot_lab import BLOCKED_M, CONTACT_IN_TCP_M, HOLD_W, _fixture_distance, vertical_pinch_R
from .untie import BASES_LOCAL, PRE, Stop, Untie
from topogym.planning.knot_untying.redesign.tiers import keep_out_violations


PINLESS_STROKE_M = .08     # a stroke without a pin: along the strand's tangent, at most 8 cm
TRANSIT_TOL_M = .008       # free-space transit moves accept this much IK error (a high transit near the reach limit)
REACH_BY_HEIGHT = ((.03, None), (.11, .55), (np.inf, .50))  # (up to this height above the table, reach m; None:
                           # the lab's disc): a vertical pinch reaches 0.60 m at the table, 0.55 m at 5-10 cm, 0.50 m at
                           # 19 cm from the arm's base (measured from the home pose, reach_annulus.py, 10-11)
POST_CLEAR_M = .035        # transit height over a post: its top + this (the fingertips 2.3 mm below the contact point)
PRESS_DEPTH_M = .005       # a press pin's fingertips 5 mm into the strand's top
CARRY_ROT_M = .25          # a carry (the strand held, free space) may tilt the gripper this much off its pinch
                           # orientation (easier_v8: tail carries refused at 0.13-0.18 rad near the front edge)
STROKE_HIGH_LIFT_M = .065  # a stroke whose path meets a curb at the lab's 2 cm lift goes over it (curb top 4.5 cm)
CURB_STANDOFF_M = .04      # a layout target keeps this far from a curb in plan (rope centre)
CLAMP_KEEP_M = .15         # clear_tail's drawn tail keeps this far from the clamps in plan (knot 10-11)
CLAMP_PULL_M = .10         # a pull station this close to a clamp in plan is the last choice (the short strand twists)


class RobotUntie(Untie):
    def __init__(self, lab, scene, *, limit=8., speed=.06, free_speed=.25, descend_speed=.12, verbose=True,
                 press_pins=False, use_pins=True):
        super().__init__(lab, scene, limit=limit, speed=speed, verbose=verbose)
        self.free_speed, self.descend_speed = free_speed, descend_speed
        self.press_pins = bool(press_pins)      # pins may also press the over strand (closed fingers, free heading)
        self.use_pins = bool(use_pins)          # False: every deletion stroke goes without a pin (tangent, <= 8 cm)
        self.grasp_records = []
        self.routes = []                        # approach routes taken (approach())
        # move variants (run_robot_oracle.py flags; the defaults are the reference behaviour)
        self.stroke_lift = .02                  # deletion strokes lift the strand this much (the lab's)
        self.pull_offset = .035                 # the pull station this far past the crossing (the lab's)
        self.high_lift = STROKE_HIGH_LIFT_M     # None: never lift a stroke over a curb
        self.clamp_keep = CLAMP_KEEP_M          # clear_tail's tail line keeps this far from the clamps
        self.pin_order = "near"                 # "far": a far pin on the standing side before a near one
        self.clear_tail_on = True               # False: no clear_tail after a deletion
        self.endgame = False                    # True: a stalled deletion ends with lift_across (robot endgame)
        self.fast = False                       # True (--fast, the user's 10-12 speed rule): free moves, lifts, drops,
                                                # presses, tail carries and joint moves faster; strokes keep their speed
        self.turbo = 1.                         # with fast: those speeds x turbo (10-14, route length; the rows stay
                                                # within the standard joint speed limit: the runner's limiter)
        self.stall_adapt = False                # True: a stroke stopped short ("arms" / "fixture" / "reach") changes
                                                # the next stroke: no pin / turned 40 deg / the other hand pulls
        self.stall_adapt_grasp = False          # True: also a failed grasp (other hand, then no pin) and clear_tail's
                                                # end taken by the other arm (robot_untie f16db2e8's --stall-adapt)
        self.retry_room = False                 # True: a pinch the World did not bind retries at the roomiest
                                                # station within 3 cm (not at the same place)

    # ------------------------------------------------------------ planning helpers
    def reachable(self, hand, p_local, margin=0.):
        """Untie's reach disc (0.585 m from the arm's base in the table plane, the lab's vertical-pinch reach map) for a
        point up to 3 cm above the table; higher up a vertical pinch reaches less (REACH_BY_HEIGHT). The approach to a
        station flies its own checked route (approach_plan), so only the station's height counts here."""
        p = np.asarray(p_local, float)
        z = float(p[2]) if p.shape[0] > 2 else 0.
        r = next(r for h, r in REACH_BY_HEIGHT if z <= h)
        if r is None:
            return super().reachable(hand, p_local, margin)
        return float(np.linalg.norm(p[:2] - BASES_LOCAL[hand])) <= r - margin

    def transit_z(self, a_world, b_world, base_z):
        """A transit height between two points: base_z, or 5 cm above the top of any post the straight path passes
        within 6 cm of (the gripper and the forearm clear it)."""
        z = base_z
        a, b = to_local(a_world)[:2], to_local(b_world)[:2]
        d = b - a
        L = float(np.linalg.norm(d))
        for c, r, top in self.post_tops():
            t = np.clip(np.dot(c - a, d) / max(L * L, 1e-12), 0, 1) if L > 1e-9 else 0.
            if float(np.linalg.norm(a + t * d - c)) - r < .06:
                z = max(z, to_world(np.array([0, 0, top + POST_CLEAR_M]))[2])
        return z

    # ------------------------------------------------------------ two real grippers
    GAP_M = .016           # the two arms keep this much clearance (forearms, wrists, grippers); 12 mm let two
                           # grippers touch once both PD arms lagged 5-8 mm (easy_v8, 10-11)
    FIXTURE_GAP_M = .004   # and a planned pinch keeps the arm this far from the fixtures (posts, curb, clamp staple)

    HOLD_EXTRA_M = .010    # next to an arm that holds (a pin / a press resting on the rope) stations keep 10 mm more
    # path / stroke margins (the runner's --tight lowers them all: inst8, 10-11, 137 of ~310 deletions ended "grasp
    # failed", most on a station or approach that missed these by 1-3 mm)
    PATH_NEED_M = .005     # approach / carry / drop paths keep this much from the other arm (an arm already closer: 2 mm)
    PATH_FIX_M = .003      # ... and this much from the fixtures
    STROKE_NEED_M = .006   # a planned stroke keeps this much from the other arm
    STROKE_FIX_M = .004    # ... and this much from the fixtures
    STOP_ARMS_M = .005     # a live stroke stops below this arm gap ("arms")
    STOP_FIX_M = .003      # ... or below this fixture gap ("fixture")

    def station_gap(self, hand):
        """The arm gap a station of `hand` must keep: GAP_M, + HOLD_EXTRA_M when the other arm holds the rope (its PD
        arm rests off its command on the rope and the descending arm lags 5-8 mm: easy_v8, the finger bars touched at
        a planned 16 mm), + whatever a blocked descent next to it added for this take (self._gap_extra)."""
        o = self.other(hand)
        if getattr(self, "_relaxed", False):
            return self.GAP_M + getattr(self, "_gap_extra", 0.)
        return self.GAP_M + (self.HOLD_EXTRA_M if self.holding(o) else 0.) + getattr(self, "_gap_extra", 0.)

    def other(self, hand):
        return "left" if hand == "right" else "right"

    def retreat_other(self, hand):
        """Move the other (idle) hand out of the way: 15 cm above the table, 12 cm back toward its own base."""
        o = self.other(hand)
        if self.holding(o):
            return False
        lab = self.lab
        q = to_local(lab.hand_pos(o))
        base = np.array([-.469 if o == "left" else .469, -.019])
        dxy = base - q[:2]
        n = float(np.linalg.norm(dxy))
        xy = q[:2] + (dxy / n * min(.12, max(0., n - .25)) if n > 1e-9 else 0.)
        top = max(q[2], .15)
        if q[2] < top - 1e-3:
            lab.move({o: to_world(np.r_[q[:2], top])}, self.free_speed, label="retreat_other", tol=TRANSIT_TOL_M)
        lab.move({o: to_world(np.r_[xy, top])}, self.free_speed, label="retreat_other", tol=TRANSIT_TOL_M)
        return True

    def pose_options(self, hand, rope, s):
        """Every safe vertical pinch of `hand` at the live station s: [(flip, R, contact target, q_pre, q_at)] with the
        IK reached at the station and 5 cm above, the wrist rule kept and the arm clear of the fixtures (the other arm
        is not considered: plan_pin_pull pairs the options)."""
        lab = self.lab
        p = lab.point(rope, s)
        t = lab.tangent(rope, s)[:2]
        c = p.copy()
        c[2] = max(c[2], lab.floor_z(hand))
        q_now = lab.q[hand]
        seeds = [q_now]
        for k in (-1, 1):
            alt = q_now.copy()
            alt[5] = alt[5] + k * np.pi
            seeds.append(alt)
        out = []
        for flip in (1, -1):
            R = vertical_pinch_R(t, flip)
            for seed in seeds:
                q_pre, e1, r1 = lab.solve_contact(hand, c + [0, 0, PRE], R, seed=seed)
                q_at, e2, r2 = lab.solve_contact(hand, c, R, seed=q_pre)
                if max(e1, e2) > .001 or max(r1, r2) > .02 or lab.ik.wrist_unsafe(q_pre) or lab.ik.wrist_unsafe(q_at):
                    continue
                if self.pinch_width_ok(hand, rope, s, q_at, q_pre) is None:
                    continue
                out.append((flip, R, c, q_pre, q_at))
                break
        return out

    def pinch_width_ok(self, hand, rope, s, q_at, q_pre):
        """The pad gap a pinch at s opens to (take()'s room rule: 2 x (clearance - 3 mm), 14..30 mm) if the arm keeps
        FIXTURE_GAP_M from the fixtures at q_at and q_pre, else None. With the lab's real_fingers the fingers are
        opened to that gap in the check, and a smaller gap (down to 14 mm) is tried before giving up."""
        lab = self.lab
        room = lab.pinch_clearance_at(rope, s)
        w0 = float(np.clip(2 * (room - .003), .014, lab.approach_width))
        if not getattr(lab, "real_fingers", False):
            return w0 if min(lab.fixture_gap(hand, q_at), lab.fixture_gap(hand, q_pre)) >= self.FIXTURE_GAP_M else None
        for w in [w0] + [x for x in (.026, .022, .018, .014) if x < w0 - 1e-9]:
            with lab.plan_width(**{hand: w}):
                if min(lab.fixture_gap(hand, q_at), lab.fixture_gap(hand, q_pre)) >= self.FIXTURE_GAP_M:
                    return w
        return None

    def press_options(self, hand, rope, s, depth=PRESS_DEPTH_M):
        """Press-pin poses of `hand` over the live station s: closed vertical fingers, any of 12 headings (a press need
        not close across the strand, so the gripper's finger bar and camera mount can turn away from the other arm):
        [(heading_deg, R, contact target, q_pre, q_at)], IK reached, wrist rule, clear of the fixtures."""
        lab = self.lab
        p = lab.point(rope, s)
        c = p.copy()
        c[2] = p[2] + lab.rods[rope].radius - depth + lab._tip_below_contact
        q_now = lab.q[hand]
        out = []
        # the closed fingers' 54 mm edge runs along the heading: across the strand first (it squeezes the whole
        # rope width against the table and cannot roll it out), along it last
        tr = lab.tangent(rope, s)[:2]
        tr = tr / max(np.linalg.norm(tr), 1e-12)
        heads = sorted(range(0, 360, 30), key=lambda g: abs(np.cos(np.deg2rad(g)) * tr[0] + np.sin(np.deg2rad(g)) * tr[1]))
        for deg in heads:
            a = np.deg2rad(deg)
            R = vertical_pinch_R([np.cos(a), np.sin(a)], 1)
            best = None
            for seed in (q_now, q_now + np.r_[0, 0, 0, 0, 0, np.pi], q_now - np.r_[0, 0, 0, 0, 0, np.pi]):
                q_pre, e1, r1 = lab.solve_contact(hand, c + [0, 0, PRE], R, seed=seed)
                q_at, e2, r2 = lab.solve_contact(hand, c, R, seed=q_pre)
                if max(e1, e2) > .001 or max(r1, r2) > .02 or lab.ik.wrist_unsafe(q_pre) or lab.ik.wrist_unsafe(q_at):
                    continue
                best = (q_pre, q_at)
                break
            if best is None:
                continue
            with lab.plan_width(**{hand: 0.}):            # a press comes down with the jaws shut
                if min(lab.fixture_gap(hand, best[1]), lab.fixture_gap(hand, best[0])) < self.FIXTURE_GAP_M:
                    continue
            out.append((deg, R, c, best[0], best[1]))
        return out

    def take_press(self, hand, rope, s, heading_deg, label="press_pin"):
        """Approach above the station with the jaws closed at the planned heading, converge, press (RobotLab.press)."""
        lab = self.lab
        if self.holding(hand):
            self.drop(hand, lower_to_table=False)
        t0 = lab.t
        a = np.deg2rad(heading_deg)
        R = vertical_pinch_R([np.cos(a), np.sin(a)], 1)
        p = lab.point(rope, s)
        c = p.copy()
        c[2] = p[2] + lab.rods[rope].radius - PRESS_DEPTH_M + lab._tip_below_contact
        above = c + [0, 0, PRE]
        q_pre = None
        for seed in (lab.q[hand], lab.q[hand] + [0, 0, 0, 0, 0, np.pi], lab.q[hand] - [0, 0, 0, 0, 0, np.pi]):
            q, e, r = lab.solve_contact(hand, above, R, seed=seed)
            if e <= .001 and r <= .02 and not lab.ik.wrist_unsafe(q):
                if q_pre is None or np.max(np.abs(q - lab.q[hand])) < np.max(np.abs(q_pre - lab.q[hand])):
                    q_pre = q
        if q_pre is None:                       # the planner's own pre-press joints for this heading
            for deg, R_o, c_o, q_pre_o, q_at_o in self.press_options(hand, rope, s):
                if deg == heading_deg:
                    q_pre, R, c = q_pre_o, R_o, c_o
                    above = c + [0, 0, PRE]
                    break
        if q_pre is None:
            raise Stop(f"{hand}: no IK 5 cm above the press at s={s:.3f} of {rope}")
        lab.set_width({hand: 0.}, .1, label="close")
        self.approach(hand, above, R, q_pre, what=f"press s={s:.3f} of {rope}")
        lab.converge(hand, tol=.001, max_s=.3, label="converge_above")
        if not lab.press(hand, rope, s, speed=.08 * self.turbo if self.fast else .05):
            lab.move({hand: lab.hand_pos(hand) + [0, 0, PRE]}, .3 * self.turbo if self.fast else .2, label="unpress",
                     tol=TRANSIT_TOL_M)
            raise Stop(f"{hand}: the press at s={s:.3f} of {rope} did not get down (reach)")
        self.grasp_records.append(dict(hand=hand, rope=rope, s=round(float(s), 4), press=True, heading_deg=heading_deg))
        self.phase(hand, label, t0)
        return True

    def drop(self, hand, lower_to_table=True):
        """Untie.drop (lower the held strand to the table, open, 5 cm up); a press is just lifted off. Above a fixture
        (a strand drawn over the curb) the strand is lowered only as far as the arm keeps 3 mm from it (legs_check),
        then let go there, instead of being pushed into the curb."""
        lab = self.lab
        g = lab.hands[hand]["grasp"]
        if g is not None and g.get("press"):
            t0 = lab.t
            lab.release(hand)
            lab.move({hand: lab.hand_pos(hand) + [0, 0, PRE]}, .3 * self.turbo if self.fast else .2, label="unpress",
                     tol=TRANSIT_TOL_M)
            self.phase(hand, "release", t0)
            return
        if g is not None and lower_to_table and lab.fixture_geoms:
            p = lab.hand_pos(hand).copy()
            z_table = to_world(np.array([0, 0, lab.rods[g["rope"]].radius + .001]))[2]
            if p[2] > z_table + .001 and self.legs_check(hand, [(p, np.r_[p[:2], z_table], False)], lab.R[hand]) \
                    is not None:
                a, b = z_table, float(p[2])
                for _ in range(7):
                    mid = .5 * (a + b)
                    if self.legs_check(hand, [(p, np.r_[p[:2], mid], False)], lab.R[hand]) is None:
                        b = mid
                    else:
                        a = mid
                if b < p[2] - .001:
                    lab.move({hand: np.r_[p[:2], b]}, .03, label="lower_onto_fixture", tol=.004)
                if self.fast:
                    self._fast_drop(hand, lower_to_table=False)
                else:
                    super().drop(hand, lower_to_table=False)
                return
        if self.fast:
            self._fast_drop(hand, lower_to_table=lower_to_table)
        else:
            super().drop(hand, lower_to_table=lower_to_table)

    def _fast_drop(self, hand, lower_to_table=True):
        """Untie.drop with the --fast speeds: the strand lowered at 6 cm/s (3), a 50 ms wait after the jaws opened (100;
        release() already steps until the weld is gone), 5 cm up at 0.3 m/s (0.2)."""
        lab = self.lab
        t0 = lab.t
        g = lab.hands[hand]["grasp"]
        if g is not None and lower_to_table:
            p = lab.hand_pos(hand)
            z_table = to_world(np.array([0, 0, lab.rods[g["rope"]].radius + .001]))[2]
            if p[2] > z_table + .001:
                lab.move({hand: np.r_[p[:2], z_table]}, .06 * self.turbo, label="lower")
        lab.release(hand)
        lab.wait(.05)
        lab.move({hand: lab.hand_pos(hand) + [0, 0, PRE]}, .3 * self.turbo, label="release_up")
        self.phase(hand, "release", t0)

    def stroke_clear(self, pull_hand, q_at, R, c, direction, length, q_o, lift=.02, need=None, need_fixture=None,
                     step=.015):
        """How far the pull hand's planned stroke (lift `lift`, then `length` along `direction`, from the pinch pose at
        contact c) keeps `need` from the other arm standing at q_o and `need_fixture` from the fixtures (the gripper's
        camera mount meets a 16 cm post's top: hard, 10-11), with the IK within 3 mm; sampled every `step`. -> m."""
        lab = self.lab
        need = self.STROKE_NEED_M if need is None else need
        need_fixture = self.STROKE_FIX_M if need_fixture is None else need_fixture
        q = q_at
        d = np.asarray(direction, float)
        d = d / max(np.linalg.norm(d), 1e-12)
        ok_len = 0.
        xs = list(np.arange(0., length, step)) + [length]
        for x in xs:
            pt = c + [0, 0, lift] + x * d
            if not self.reachable(pull_hand, to_local(pt)):    # the live stroke's own reach rule (height-aware)
                return ok_len
            q, ep, er = lab.solve_contact(pull_hand, pt, R, seed=q)
            if ep > .003 or lab.gripper_gap(pull_hand, q, q_o=q_o) < need:
                return ok_len
            if lab.fixture_geoms:
                with lab.plan_width(**{pull_hand: HOLD_W}):      # the stroke's hand holds the rope
                    if lab.fixture_gap(pull_hand, q) < need_fixture:
                        return ok_len
            ok_len = float(x)
        return ok_len

    def stroke_clear_best(self, pull_hand, q_at, R, c, direction, length, q_o, lift=None):
        """(clear length m, lift m) of the stroke at the lab's lift, or lifted over a curb (STROKE_HIGH_LIFT_M) when that
        is clear >= 3 cm further: the lab's strokes draw a strand toward its clamp, behind easier / easy's curb, which
        lab hands pass and the robot's fingers meet (easier v7, 10-11: pulls cut to 3-6 cm in front of the curb)."""
        lift = self.stroke_lift if lift is None else lift
        low = self.stroke_clear(pull_hand, q_at, R, c, direction, length, q_o, lift=lift)
        if (low >= length - 1e-9 or self.high_lift is None
                or not any(f.get("role") == "curb" for f in self.scene.get("fixtures", []))):
            return low, lift
        high = self.stroke_clear(pull_hand, q_at, R, c, direction, length, q_o, lift=self.high_lift)
        return (high, self.high_lift) if high >= low + .03 else (low, lift)

    def stroke_direction(self, rope, s_pull, s_u, length, pinless):
        """The deletion's stroke direction (the lab's: from the crossing to the pull station; pinless: along the
        strand's own tangent toward the body), turned clear of the posts as the lab turns it."""
        lab = self.lab
        sign = -1. if self.free_at_end(rope) else 1.
        if pinless:
            direction = sign * lab.tangent(rope, s_pull)
        else:
            direction = lab.point(rope, s_pull) - lab.point(rope, s_u)
        direction = np.asarray(direction, float).copy()
        direction[2] = 0.
        direction /= np.linalg.norm(direction)
        return self.clear_direction(to_local(lab.point(rope, s_pull)), direction, length)

    def plan_pin_pull(self, rope, s_pull0, sign, pull_hand, orope, so, *, pull_search=.04, pin_search=.10,
                      far_search=.25, s_u=None, stroke_len=.18, min_stroke=.10):
        """A pull station (s_pull0, or up to `pull_search` further past the crossing) and a pin station on the over
        strand (within `pin_search` of its crossing visit `so`, nearest first: the lab pins within 5 cm, a robot pin
        may have to sit up to 10 cm away; then up to `far_search` toward the over strand's clamp) whose two real
        grippers fit side by side (GAP_M), with
        their jaw flips; a pin the other arm already holds is kept when the pull fits beside it.
        -> dict(pull_hand, s_pull, pull_flip, pin_hand, s_pin, pin_flip, kept_pin) or None (no pin fits)."""
        lab = self.lab
        L = lab.length(rope)
        Lo = lab.length(orope)
        pin_hand = self.other(pull_hand)
        held = lab.hands[pin_hand]["grasp"]
        pulls = []
        for j in range(int(round(pull_search / .005)) + 1):
            st = float(np.clip(s_pull0 + sign * j * .005, .03, L - .012))
            if lab.pinch_clearance_at(rope, st) >= lab.pinch_clearance and st not in pulls:
                pulls.append(st)
        clamps = self.clamp_centres()
        if len(clamps):              # a pull next to a clamp is the last choice (the short strand twists in the jaws)
            far = [x for x in pulls
                   if float(np.min(np.linalg.norm(self.local(rope, x)[:2] - clamps, axis=1))) >= CLAMP_PULL_M]
            pulls = far + [x for x in pulls if x not in far]
        pins = []
        for j in range(int(round(pin_search / .005)) + 1):
            for sg in ((1.,) if j == 0 else (1., -1.)):
                st = float(np.clip(so + sg * j * .005, .03, Lo - .012))
                if orope == rope and any(abs(st - x) < .02 for x in pulls[:1]):
                    continue
                if lab.pinch_clearance_at(orope, st) >= lab.pinch_clearance:
                    pins.append(st)
        # then further along the over strand toward its clamp (up to far_search): a far pin on the standing side still
        # holds the knot body while the tail is drawn, as HULK's far pin does
        anchor = self.anchors.get(orope)
        if anchor in ("start", "end"):
            toward = -1. if anchor == "start" else 1.
            for j in range(1, int(round((far_search - pin_search) / .01)) + 1):
                st = so + toward * (pin_search + j * .01)
                if st < .06 or st > Lo - .06:
                    break
                if lab.pinch_clearance_at(orope, st) >= lab.pinch_clearance:
                    pins.append(float(st))
        if self.pin_order == "far":     # the far pins (toward the over strand's clamp) first
            near_n = sum(1 for x in pins if abs(x - so) <= pin_search + 1e-9)
            pins = pins[near_n:] + pins[:near_n]
        pin_cache = {}
        for s_pull in pulls:
            if not self.reachable(pull_hand, self.local(rope, s_pull)):
                continue
            pull_opts = self.pose_options(pull_hand, rope, s_pull)
            if not pull_opts:
                continue
            if held is not None and held["rope"] == orope and abs(held["s"] - so) <= far_search + .005:
                for fl, R, c, q_pre, q_at in pull_opts:
                    if min(lab.gripper_gap(pull_hand, q_at), lab.gripper_gap(pull_hand, q_pre)) < (self.GAP_M + self.HOLD_EXTRA_M):
                        continue
                    here = lab.hand_pos(pull_hand)
                    above = c + [0, 0, PRE]
                    if self.path_blocked(pull_hand, here, self.transit_z(here, above, above[2]), above, R) is not None:
                        continue
                    clear = None
                    if s_u is not None:
                        d_ = self.stroke_direction(rope, s_pull, s_u, stroke_len, pinless=False)
                        clear, s_lift = self.stroke_clear_best(pull_hand, q_at, R, c, d_, stroke_len, lab.q[pin_hand])
                        if clear < min(min_stroke, stroke_len) - 1e-9:
                            continue
                    return dict(pull_hand=pull_hand, s_pull=s_pull, pull_flip=fl, pin_hand=pin_hand,
                                s_pin=held["s"], pin_flip=None, pin_kind="press" if held.get("press") else "pinch",
                                kept_pin=True, stroke_clear_m=clear, stroke_lift_m=None if clear is None else s_lift)
            kinds = ("pinch", "press") if self.press_pins else ("pinch",)
            for s_pin in pins:
                if not self.reachable(pin_hand, self.local(orope, s_pin)):
                    continue
                for kind in kinds:
                    key = (kind, s_pin)
                    if key not in pin_cache:
                        pin_cache[key] = (self.pose_options(pin_hand, orope, s_pin) if kind == "pinch"
                                          else self.press_options(pin_hand, orope, s_pin))
                    for pfl, _, _, _, pq_at in pin_cache[key]:
                        for fl, R, c, q_pre, q_at in pull_opts:
                            gap = lab.gripper_gap(pull_hand, q_at, q_o=pq_at)
                            if gap < (self.GAP_M + self.HOLD_EXTRA_M):
                                continue
                            gap = min(gap, lab.gripper_gap(pull_hand, q_pre, q_o=pq_at))
                            if gap < (self.GAP_M + self.HOLD_EXTRA_M):
                                continue
                            here = lab.hand_pos(pull_hand)
                            above = c + [0, 0, PRE]
                            if self.path_blocked(pull_hand, here, self.transit_z(here, above, above[2]), above, R,
                                                 q_o=pq_at) is not None:
                                continue
                            clear, s_lift = None, None
                            if s_u is not None:      # the stroke itself must not run into the pin's arm
                                d_ = self.stroke_direction(rope, s_pull, s_u, stroke_len, pinless=False)
                                clear, s_lift = self.stroke_clear_best(pull_hand, q_at, R, c, d_, stroke_len, pq_at)
                                if clear < min(min_stroke, stroke_len) - 1e-9:
                                    continue
                            return dict(pull_hand=pull_hand, s_pull=s_pull, pull_flip=fl, pin_hand=pin_hand,
                                        s_pin=s_pin, pin_flip=pfl, pin_kind=kind, kept_pin=False,
                                        gap_mm=round(gap * 1000, 1), stroke_clear_m=clear, stroke_lift_m=s_lift)
        return None

    def pinch_plan(self, hand, rope, s, need_gap=None, flip=None):
        """(R, contact target, q at the pre-grasp, flip, gap) of the jaw flip whose IK at the live station and 5 cm
        above it is safe (wrist rule) and nearest the arm's joints, keeping `need_gap` from the other arm as it is
        commanded now (None: no gap rule). Raises Stop if none."""
        lab = self.lab
        p = lab.point(rope, s)
        t = lab.tangent(rope, s)[:2]
        c = p.copy()
        c[2] = max(c[2], lab.floor_z(hand))
        best = None
        q_now = lab.q[hand]
        seeds = [q_now]
        for k in (-1, 1):
            alt = q_now.copy()
            alt[5] = alt[5] + k * np.pi
            seeds.append(alt)
        tried = []
        for fl in ((1, -1) if flip is None else (flip,)):
            R = vertical_pinch_R(t, fl)
            for seed in seeds:
                q_pre, e1, r1 = lab.solve_contact(hand, c + [0, 0, PRE], R, seed=seed)
                q_at, e2, r2 = lab.solve_contact(hand, c, R, seed=q_pre)
                if max(e1, e2) > .001 or max(r1, r2) > .02 or lab.ik.wrist_unsafe(q_pre) or lab.ik.wrist_unsafe(q_at):
                    continue
                gap = min(lab.gripper_gap(hand, q_at), lab.gripper_gap(hand, q_pre))
                w_ok = self.pinch_width_ok(hand, rope, s, q_at, q_pre)
                fix = min(lab.fixture_gap(hand, q_at), lab.fixture_gap(hand, q_pre))
                tried.append(dict(flip=fl, gap_mm=round(gap * 1000, 1), fixture_mm=round(fix * 1000, 1),
                                  **({} if not getattr(lab, "real_fingers", False) else
                                     dict(width_mm=None if w_ok is None else round(w_ok * 1000, 1)))))
                if need_gap is not None and gap < need_gap:
                    continue
                if w_ok is None:
                    continue
                travel = float(np.max(np.abs(q_pre - q_now)))
                if best is None or travel < best[0]:
                    best = (travel, R, c, q_pre, fl, gap)
                break
        if best is None:
            raise Stop(f"{hand}: no safe vertical pinch at s={s:.3f} of {rope} (tried {tried})")
        return best[1], best[2], best[3], best[4], best[5]

    def plan_station(self, hand, rope, s, search=.08, flip=None):
        """(s, plan) for `hand` at s, or at the nearest station up to `search` along the rope (away from the other
        gripper) where a parallel gripper fits and the arms keep station_gap() (GAP_M + 10 mm next to a holding arm);
        an idle other hand in the way is moved back first. Nothing within reach of that rule: the same search with
        GAP_M alone (easy v7, 10-11: the 26 mm rule left no station at 24.8 mm; a blocked descent re-plans anyway)."""
        try:
            return self._plan_station(hand, rope, s, search=search, flip=flip)
        except Stop as e:
            if self.station_gap(hand) <= self.GAP_M + getattr(self, "_gap_extra", 0.) + 1e-9:
                raise
            first = e
        saved = getattr(self, "_relaxed", False)
        self._relaxed = True
        try:
            return self._plan_station(hand, rope, s, search=search, flip=flip)
        except Stop:
            pass
        finally:
            self._relaxed = saved
        # still nothing: an idle other arm goes back to its home joints (the arms had tangled: easier_v8 inst8 looped
        # with planned gaps of -40..-55 mm), then one more try
        if self.home_other(hand):
            return self._plan_station(hand, rope, s, search=search, flip=flip)
        raise first

    def home_other(self, hand):
        """The idle other arm to its home joints (the instance's reset targets) in joint space, if that way keeps
        2 mm from the fixtures and stays above the rope; -> True if it moved."""
        o = self.other(hand)
        home = getattr(self, "home_q", {}).get(o)
        if home is None or self.holding(o):
            return False
        lab = self.lab
        q0 = lab.q[o].copy()
        if float(np.max(np.abs(np.asarray(home) - q0))) < 1e-3:
            return False
        why = self.joint_route_check(o, q0, home, low_z=min(lab.hand_pos(o)[2], to_world([0, 0, .05])[2]), need=-1.)
        if why is not None and "arms_mm" not in why:
            self.routes.append(dict(t=round(lab.t, 3), hand=o, to="home", kind="refused", why=why))
            return False
        travel = float(np.max(np.abs(np.asarray(home) - q0)))
        lab.joint_move({o: np.asarray(home, float)}, max(.5 / self.turbo, 1.875 * travel / (1.5 * self.turbo)) if self.fast else
                       max(.8, 1.875 * travel / 1.0), label="home_other")
        self.routes.append(dict(t=round(lab.t, 3), hand=o, to="home", kind="joint"))
        return True

    def _plan_station(self, hand, rope, s, search=.08, flip=None):
        lab = self.lab
        o = self.other(hand)
        try:
            return s, self.pinch_plan(hand, rope, s, need_gap=self.station_gap(hand), flip=flip)
        except Stop as e:
            first = e
        if not self.holding(o) and self.retreat_other(hand):
            try:
                return s, self.pinch_plan(hand, rope, s, need_gap=self.station_gap(hand), flip=flip)
            except Stop as e:
                first = e
        if flip is not None:          # a planned flip that no longer fits: any flip
            try:
                return s, self.pinch_plan(hand, rope, s, need_gap=self.station_gap(hand))
            except Stop:
                pass
        L = lab.length(rope)
        if self.holding(o):          # slide away from the holding hand
            q_o = lab.hand_actual(o)
            up = float(np.linalg.norm(lab.point(rope, min(L - .012, s + .01))[:2] - q_o[:2]))
            down = float(np.linalg.norm(lab.point(rope, max(.03, s - .01))[:2] - q_o[:2]))
            signs = (1.,) if up >= down else (-1.,)
        else:                        # a fixture in the way: the nearer free station either side
            signs = (1., -1.)
        for x in np.arange(.01, search + 1e-9, .01):
            for sign in signs:
                st = float(np.clip(s + sign * x, .03, L - .012))
                if lab.pinch_clearance_at(rope, st) < lab.pinch_clearance:
                    continue
                try:
                    return st, self.pinch_plan(hand, rope, st, need_gap=self.station_gap(hand))
                except Stop:
                    continue
        raise Stop(f"{hand}: no station within {search * 100:.0f} cm of s={s:.3f} of {rope} keeps the arm clear ({first})")

    def path_blocked(self, hand, here, zt, above, R, n=6, need=None, q_o=None):
        """The approach take() will fly (up to the transit height, across with the jaws turning, down to 5 cm above
        the station), sampled n times per leg with IK seeded along it: None if the arm keeps `need` from the other arm
        everywhere, else where it does not (dict)."""
        from scipy.spatial.transform import Rotation, Slerp
        lab = self.lab
        need = self.PATH_NEED_M if need is None else need
        R0 = lab.R[hand]
        sl = Slerp([0., 1.], Rotation.from_matrix(np.stack([R0, R])))
        legs = []
        if zt > here[2] + 1e-3:
            legs.append((here, np.r_[here[:2], zt], False))
        a = np.r_[here[:2], max(here[2], zt)] if zt > here[2] + 1e-3 else here
        legs.append((a, np.r_[above[:2], zt], True))
        if zt > above[2] + 1e-3:
            legs.append((np.r_[above[:2], zt], above, False))
        q = lab.q[hand].copy()
        turned = False
        for a_, b_, turning in legs:
            for k in range(1, n + 1):
                u = k / n
                Rk = sl([u])[0].as_matrix() if turning else (R if turned else R0)
                q, ep, er = lab.solve_contact(hand, a_ + (b_ - a_) * u, Rk, seed=q)
                gap = lab.gripper_gap(hand, q, q_o=q_o)
                if gap < need:
                    return dict(at=np.round(a_ + (b_ - a_) * u, 3).tolist(), gap_mm=round(gap * 1000, 1))
            turned = turned or turning
        return None

    # ------------------------------------------------------------ the approach route (free space above the rope)
    @staticmethod
    def _legs(here, zt, above, Q=None):
        """Cartesian approach legs [(from, to, turning)]: up to zt over `here` (if lower), across at zt with the jaws
        turning (to above's xy, or to Q past a post), then down to `above`."""
        legs = []
        if zt > here[2] + 1e-3:
            legs.append((here, np.r_[here[:2], zt], False))
        a = np.r_[here[:2], max(here[2], zt)]
        b = np.r_[(above if Q is None else Q)[:2], zt]
        legs.append((a, b, True))
        if float(np.linalg.norm(above - b)) > 1e-3:
            legs.append((b, above, False))
        return legs

    def legs_check(self, hand, legs, R, need=None, tol=.004, tol_rot=.12, step=.02):
        """Fly the legs in plan (IK seeded along them every <= `step`, the jaws slerping to R on the turning leg):
        None if the arm follows (IK within tol / tol_rot, wrist rule) keeping `need` from the other arm (as
        commanded) and 3 mm from the fixtures all along, else why not (dict); `q_end` is set to the last joints."""
        from scipy.spatial.transform import Rotation, Slerp
        lab = self.lab
        R0 = lab.R[hand]
        sl = Slerp([0., 1.], Rotation.from_matrix(np.stack([R0, R])))
        q = lab.q[hand].copy()
        # an arm that already stands closer than the rules (a stroke stopped inside the 3 mm, a press) may move on while
        # keeping 2 mm (hard_nostop inst8: the last crossing's pull arm started 4-5 mm from the pinning arm and every
        # route dipped to 3-4 mm, so the deletion could never start)
        need = self.PATH_NEED_M if need is None else need
        g0 = lab.gripper_gap(hand, q)
        if g0 < need:
            need = max(.002, g0 - .003)
        need_fix = min(self.PATH_FIX_M, (lab.fixture_gap(hand, q) if lab.fixture_geoms else 1.) - .001)
        turned = False
        for a_, b_, turning in legs:
            m = max(4, int(np.ceil(float(np.linalg.norm(b_ - a_)) / step)))
            for k in range(1, m + 1):
                u = k / m
                Rk = sl([u])[0].as_matrix() if turning else (R if turned else R0)
                x = a_ + (b_ - a_) * u
                q, ep, er = lab.solve_contact(hand, x, Rk, seed=q)
                why = None
                if ep > tol or er > tol_rot or lab.ik.wrist_unsafe(q):
                    why = dict(reach_mm=round(ep * 1000, 1), rot=round(er, 3), wrist=lab.ik.wrist_unsafe(q))
                else:
                    gap = lab.gripper_gap(hand, q)
                    if gap < need:
                        why = dict(arms_mm=round(gap * 1000, 1))
                    elif lab.fixture_geoms:
                        fix = lab.fixture_gap(hand, q)
                        if fix < need_fix:
                            why = dict(fixture_mm=round(fix * 1000, 1))
                if why is not None:
                    why["at"] = np.round(to_local(x), 3).tolist()
                    return why
            turned = turned or turning
        self.q_end = q
        return None

    def past_posts(self, here, above):
        """Where (world xy) the straight way here -> above has left every post near it by 7 cm, or None (a post at
        the far end)."""
        a, b = to_local(here)[:2], to_local(above)[:2]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            return None
        t_out = None
        for c, r, top in self.post_tops():
            t = float(np.clip(np.dot(c - a, d) / (L * L), 0, 1))
            if float(np.linalg.norm(a + t * d - c)) - r < .06:
                t_out = max(t_out or 0., t + (r + .07) / L)
        if t_out is None or t_out >= .95:
            return None
        return to_world(np.r_[a + t_out * d, 0.])

    def rope_top_near(self, xy_world, r=.04):
        """The highest rope surface (world z) within r of a point in plan, or -inf."""
        lab = self.lab
        top = -np.inf
        for rod in lab.rods.values():
            nodes = rod.nodes(lab.d)
            near = np.linalg.norm(nodes[:, :2] - np.asarray(xy_world)[:2], axis=1) < r
            if near.any():
                top = max(top, float(nodes[near, 2].max()) + rod.radius)
        return top

    def joint_route_check(self, hand, q0, q_goal, low_z, n=12, need=None):
        """The joint-space straight line q0 -> q_goal sampled n times: None if the wrist rule holds, the contact point
        stays above low_z - 1 cm and 2 cm above any rope within 4 cm, and the arm keeps `need` from the other arm (as
        commanded) and 3 mm from the fixtures, else why not (dict)."""
        lab = self.lab
        q0 = np.asarray(q0, float)
        q_goal = np.asarray(q_goal, float)
        need = self.PATH_NEED_M if need is None else need
        g0 = lab.gripper_gap(hand, q0)
        if g0 < need:
            need = max(.002, g0 - .003)
        need_fix = min(self.PATH_FIX_M, (lab.fixture_gap(hand, q0) if lab.fixture_geoms else 1.) - .001)
        for k in range(1, n + 1):
            q = q0 + (q_goal - q0) * (k / n)
            why = None
            c, _ = lab.ik.contact(hand, q)
            if lab.ik.wrist_unsafe(q):
                why = dict(wrist=True)
            elif c[2] < low_z - .01:
                why = dict(low_mm=round((c[2] - low_z) * 1000, 1))
            elif c[2] - .003 < self.rope_top_near(c) + .02:
                why = dict(rope_mm=round((c[2] - .003 - self.rope_top_near(c)) * 1000, 1))
            else:
                gap = lab.gripper_gap(hand, q)
                if gap < need:
                    why = dict(arms_mm=round(gap * 1000, 1))
                elif lab.fixture_geoms and lab.fixture_gap(hand, q) < need_fix:
                    why = dict(fixture_mm=round(lab.fixture_gap(hand, q) * 1000, 1))
            if why is not None:
                why["k"] = k
                why["at"] = np.round(to_local(c), 3).tolist()
                return why
        return None

    def approach_plan(self, hand, above, R, q_pre):
        """The way take() / take_press() fly from the hand's commanded pose to `above` (5 cm over the station, jaws
        at R), the first of these the arm can follow clear of the other arm and the fixtures (legs_check):
          "cartesian"   up to the transit height (5 cm over the station, or over a post on the way), across, down;
          "over_post"   up over the post, across until 7 cm past it, then straight down to `above` (a vertical gripper
                        reaches less far up high: hard, 10-11);
          "joint"       up to `above`'s height if lower, then a joint-space line to the planned pre-grasp joints q_pre
                        (no IK along the way: a wrist turn or the reach limit the Cartesian ways cannot follow),
                        checked by joint_route_check.
        -> ((kind, legs, q_goal), None) or (None, {kind: why})."""
        lab = self.lab
        here = lab.hand_pos(hand)
        zt = self.transit_z(here, above, above[2])
        whys = {}
        legs = self._legs(here, zt, above)
        why = self.legs_check(hand, legs, R)
        if why is None:
            return ("cartesian", legs, None), None
        whys["cartesian"] = why
        # the turn made higher up and a straight vertical descent: the camera mount (11-15 cm above the pads) then
        # passes a 16 cm post's top and comes down where the planned pre-grasp pose has it (hard v7, 10-11)
        for extra in (.06, .12):
            z_hi = max(zt, above[2] + extra)
            if z_hi <= zt + 1e-3:
                continue
            legs = self._legs(here, z_hi, above)
            why = self.legs_check(hand, legs, R)
            if why is None:
                return (f"cartesian_high_{int(round(extra * 100))}cm", legs, None), None
            whys[f"cartesian_high_{int(round(extra * 100))}cm"] = why
        if zt > above[2] + 1e-3:
            Q = self.past_posts(here, above)
            if Q is not None:
                legs = self._legs(here, zt, above, Q=Q)
                why = self.legs_check(hand, legs, R)
                if why is None:
                    return ("over_post", legs, None), None
                whys["over_post"] = why
        legs, q0 = [], lab.q[hand].copy()
        if above[2] > here[2] + 1e-3:
            legs = [(here, np.r_[here[:2], above[2]], False)]
            why = self.legs_check(hand, legs, lab.R[hand])
            if why is not None:
                whys["joint"] = dict(lift=why)
                return None, whys
            q0 = self.q_end
        why = self.joint_route_check(hand, q0, q_pre, low_z=min(here[2], above[2]))
        if why is None:
            return ("joint", legs, q_pre), None
        whys["joint"] = why
        return None, whys

    def approach(self, hand, above, R, q_pre, what=""):
        """Fly to `above` by approach_plan's route (an idle other hand backs off if none is clear); a live reach stop
        on a Cartesian route finishes in joint space if that is clear. Raises Stop. -> the route's kind."""
        lab = self.lab
        plan, whys = self.approach_plan(hand, above, R, q_pre)
        if plan is None and self.retreat_other(hand):
            plan, whys = self.approach_plan(hand, above, R, q_pre)
        if plan is None:
            raise Stop(f"{hand}: no clear way to {what} ({whys})")
        kind, legs, q_goal = plan
        for a_, b_, turning in legs:
            if turning:
                lab.set_orientation(hand, R)
            lab.move({hand: b_}, self.free_speed, label="approach" if turning else
                     ("approach_up" if b_[2] > a_[2] + 1e-3 else "approach_down"), tol=TRANSIT_TOL_M)
            if lab.reach_stopped[hand]:
                break
        if lab.reach_stopped[hand]:
            why = self.joint_route_check(hand, lab.q[hand], q_pre, low_z=min(lab.hand_pos(hand)[2], above[2]))
            if why is not None:
                raise Stop(f"{hand}: reach stop on the {kind} way to {what}; the joint way is not clear ({why})")
            lab.reach_stopped[hand] = None
            kind, q_goal = kind + "+joint", q_pre
        if q_goal is not None:
            travel = float(np.max(np.abs(np.asarray(q_goal) - lab.q[hand])))
            lab.joint_move({hand: q_goal}, max(.4 / self.turbo, 1.875 * travel / (1.8 * self.turbo)) if self.fast else
                           max(.6, 1.875 * travel / 1.2), label="approach_joint")
        self.routes.append(dict(t=round(lab.t, 3), hand=hand, to=what, kind=kind,
                                tried={k: v for k, v in (whys or {}).items()}))
        return kind

    # ------------------------------------------------------------ primitives
    def take(self, hand, rope, s, label="grasp", flip=None):
        lab = self.lab
        s_req = s
        if not self.reachable(hand, to_local(lab.point(rope, s))):
            raise Stop(f"{hand} cannot reach station {s:.3f} of {rope}")
        if self.holding(hand):
            self.drop(hand, lower_to_table=False)
        t0 = lab.t
        s, (R, c, q_pre, flip, gap) = self.plan_station(hand, rope, s, flip=flip)
        above = c + [0, 0, PRE]
        route = self.approach(hand, above, R, q_pre, what=f"s={s:.3f} of {rope}")
        rec = dict(hand=hand, rope=rope, s=round(float(s), 4), s_requested=round(float(s_req), 4), flip=flip,
                   gap_mm=round(gap * 1000, 1), route=route, attempts=[])
        ok = False
        for attempt in range(2):
            aim_z = PRE if attempt == 0 else .015
            if (attempt == 1 and self.retry_room and rec["attempts"] and rec["attempts"][-1].get("bound") is False):
                # the jaws closed where planned but the World found no bilateral pinch (a strand or a fixture a few mm
                # away took a finger: easier_v8 inst8, "no_bilateral_candidate" twice at one station): the retry goes
                # to the roomiest station within 3 cm, from 5 cm above
                here_room = lab.pinch_clearance_at(rope, s)
                L_ = lab.length(rope)
                cands = [float(np.clip(s + sg * x, .03, L_ - .012)) for x in np.arange(.005, .0301, .005)
                         for sg in (1., -1.)]
                cands = [(lab.pinch_clearance_at(rope, x), x) for x in cands
                         if self.reachable(hand, to_local(lab.point(rope, x)))]
                if cands:
                    room_best, s_best = max(cands)
                    if room_best >= max(here_room + .002, lab.pinch_clearance):
                        rec.setdefault("re_stationed_room", []).append(round(s_best, 4))
                        s = s_best
                        aim_z = PRE
            # re-aim on the live rope (it relaxes while the hand comes, or the other hand's pinch moved it): a station
            # that lost its room for the fingers moves to the nearest one within 3 cm that has it
            if lab.pinch_clearance_at(rope, s) < lab.pinch_clearance:
                alt = self.pinch_station(rope, s, search=.03)
                rec.setdefault("re_stationed", []).append(None if alt is None else round(float(alt), 4))
                if alt is None or not self.reachable(hand, to_local(lab.point(rope, float(alt)))):
                    break
                s = float(alt)
            if not self.reachable(hand, to_local(lab.point(rope, s))):
                # the strand moved out of this arm's reach since the plan (another hand's pin / press): the nearest
                # station within 4 cm that is in reach and has finger room (medium_v8, 10-11)
                alts = [float(np.clip(s + sg * x, .03, lab.length(rope) - .012)) for x in np.arange(.005, .0401, .005)
                        for sg in (1., -1.)]
                alts = [a for a in alts if self.reachable(hand, to_local(lab.point(rope, a)))
                        and lab.pinch_clearance_at(rope, a) >= lab.pinch_clearance]
                rec.setdefault("re_stationed_reach", []).append(None if not alts else round(alts[0], 4))
                if not alts:
                    break
                s = alts[0]
            p = lab.point(rope, s)
            t = lab.tangent(rope, s)[:2]
            R = vertical_pinch_R(t, flip)
            c = p.copy()
            c[2] = max(c[2], lab.floor_z(hand))
            room = lab.pinch_clearance_at(rope, s)
            width = float(np.clip(2 * (room - .003), .014, lab.approach_width))
            if getattr(lab, "real_fingers", False) and lab.fixture_geoms:
                # open only as wide as keeps the opened fingers off the fixtures (a curb next to the station)
                q_pre_l, _, _ = lab.solve_contact(hand, c + [0, 0, PRE], R, seed=lab.q[hand])
                q_at_l, _, _ = lab.solve_contact(hand, c, R, seed=q_pre_l)
                w_ok = self.pinch_width_ok(hand, rope, s, q_at_l, q_pre_l)
                width = .014 if w_ok is None else min(width, w_ok)
            lab.set_orientation(hand, R)
            # 4 mm of IK error is still a pinch (the jaws open 14-30 mm round an 8 mm rope); at the reach edge the
            # 2 mm default stopped medium_v8's aim at 2.3 mm
            if aim_z > .015 + 1e-9 and attempt == 1:
                lab.move({hand: lab.hand_pos(hand) + [0, 0, PRE - .015]}, self.free_speed, label="re-aim_up",
                         tol=TRANSIT_TOL_M)
            lab.move({hand: c + [0, 0, aim_z]}, self.free_speed, min_time=.1, label="aim",
                     tol=.004, tol_rot=.05)
            lab.set_width({hand: width}, .1, label="open")
            e_up = lab.converge(hand, tol=.001, max_s=.3, label="converge_above")     # learn the arm's bias up here
            contacts = lab.arm_contacts(hand, top=3) if float(np.linalg.norm(e_up)) > BLOCKED_M else []
            if contacts:
                rec["attempts"].append(dict(blocked_above_mm=(e_up * 1000).round(1).tolist(), contacts=contacts))
                o = self.other(hand)
                if attempt == 0 and any(str(c[1]).startswith(o + "/") for c in contacts):
                    # blocked by the other arm: re-plan the station keeping 15 mm more from it, and come again
                    self._gap_extra = getattr(self, "_gap_extra", 0.) + .015
                    try:
                        s, (R, c_new, q_pre, flip, gap) = self.plan_station(hand, rope, s, flip=None)
                        rec["re_planned"] = dict(s=round(float(s), 4), gap_mm=round(gap * 1000, 1))
                        lab.move({hand: lab.hand_pos(hand) + [0, 0, PRE]}, self.free_speed, label="re-plan_up",
                                 tol=TRANSIT_TOL_M)
                        self.approach(hand, c_new + [0, 0, PRE], R, q_pre, what=f"s={s:.3f} of {rope} (re-planned)")
                        continue
                    except Stop as e:
                        rec["re_plan_error"] = str(e)[:200]
                    finally:
                        self._gap_extra = 0.
                break                                   # blocked above the station: do not come down there
            lab.move({hand: c}, self.descend_speed, label="descend", tol=.004, tol_rot=.05)
            err = lab.converge(hand, max_s=.3)
            ok = lab.grasp(hand, rope, s)
            rec["attempts"].append(dict(width=round(width, 4), tcp_err_mm=(err * 1000).round(2).tolist(), bound=ok,
                                        bias_mm=(lab.bias[hand] * 1000).round(2).tolist(), target=c.round(4).tolist(),
                                        rope_z_mm=round((p[2] - lab.floor_z(hand)) * 1000, 2)))
            if ok:
                break
            lab.release(hand)
            lab.move({hand: lab.hand_pos(hand) + [0, 0, .015]}, self.descend_speed, label="re-aim")
        self.phase(hand, "approach", t0)
        self.grasp_records.append(rec)
        if not ok:
            raise Stop(f"{hand}: no weld bound at s={s:.3f} of {rope} after 2 attempts")
        t1 = lab.t
        lab.wait(.1, label="hold")
        self.phase(hand, label, t1)
        return True

    def curb_standoff(self, p_local, standoff=None):
        """p_local (task-local, xy used) moved straight away from any curb (role "curb", an axis-aligned box) it lies
        within `standoff` of in plan, to that distance; z kept."""
        standoff = CURB_STANDOFF_M if standoff is None else standoff
        q = np.array(p_local, float)
        for f in self.scene.get("fixtures", []):
            if f.get("role") != "curb" or f.get("kind", "box") != "box":
                continue
            c = np.asarray(f["pos"][:2], float)
            h = np.asarray(f["size"][:2], float)
            near = np.clip(q[:2], c - h, c + h)
            v = q[:2] - near
            dist = float(np.linalg.norm(v))
            if dist >= standoff:
                continue
            if dist < 1e-9:                      # inside the curb's footprint: out through the nearer long side
                v = np.array([0., 1. if q[1] >= c[1] else -1.])
                near = np.array([q[0], c[1] + h[1] * v[1]])
                dist = 0.
            q[:2] = near + v / max(float(np.linalg.norm(v)), 1e-12) * standoff
        return q

    def carry(self, hand, p_world, speed=None, label="carry", raise_by=(0., .04, .08)):
        """Transport the held strand to p_world (its point at the lift height) on the first clear route: straight
        across at p_world's height, or up by 4 / 8 cm first, across, and down there (the camera mount, 11-15 cm above
        the pads, passes a 16 cm post's top only just at a 5 cm lift); checked like an approach (legs_check: reach,
        wrist rule, 5 mm from the other arm, 3 mm from the fixtures). Raises Stop. -> the extra height used (m)."""
        lab = self.lab
        if not self.reachable(hand, to_local(p_world)):
            raise Stop(f"{hand} cannot reach {to_local(p_world).round(3).tolist()}")
        goal = np.asarray(p_world, float)
        whys = {}
        for retreat in (False, True):
            if retreat:     # every route met the other arm: an idle other arm backs off, then try again (easier_v8)
                if not any("arms_mm" in w for w in whys.values()) or not self.retreat_other(hand):
                    break
            dz = self._carry_once(hand, goal, speed, label, raise_by, whys)
            if dz is not None:
                return dz
        raise Stop(f"{hand}: no clear carry to {to_local(goal).round(3).tolist()} ({whys})")

    def _carry_once(self, hand, goal, speed, label, raise_by, whys):
        lab = self.lab
        here = lab.hand_pos(hand).copy()
        for dz in raise_by:
            legs = []
            if dz > 0:
                legs.append((here, here + [0, 0, dz], False))
            a = here + [0, 0, dz]
            b = goal + [0, 0, dz]
            legs.append((a, b, False))
            if dz > 0:
                legs.append((b, goal, False))
            why = self.legs_check(hand, legs, lab.R[hand], tol_rot=CARRY_ROT_M)
            if why is not None:
                whys[round(dz, 3)] = why
                continue
            t0 = lab.t
            for a_, b_, _ in legs:
                lab.move({hand: b_}, speed or self.speed, label=label, tol=.004, tol_rot=CARRY_ROT_M)
                if lab.reach_stopped[hand]:
                    raise Stop(f"{hand}: reach stop carrying to {to_local(goal).round(3).tolist()}")
            self.phase(hand, label, t0)
            return dz
        return None

    def goto(self, hand, p_world, speed=None, label="transport"):
        """Untie.goto (transport a held strand) on a checked route: carry() (straight, or 4 / 8 cm higher when the arm
        would meet a fixture or the other arm); Stop when none is clear or the reach ends."""
        self.carry(hand, p_world, speed=speed, label=label)

    def lift(self, hand, dz=.02, speed=.05):
        t0 = self.lab.t
        if self.fast:
            speed = max(speed, .08 * self.turbo)
        self.lab.move({hand: self.lab.hand_pos(hand) + [0, 0, dz]}, speed, label="lift")
        self.phase(hand, "lift", t0)

    def stroke(self, hand, direction, length, *, step=.01, speed=None, stop_force=None, watch=None, margin=0.):
        """Untie.stroke's contract (the holding hand along `direction` up to `length`; stops at the reach limit, on a
        slip, above `stop_force`, or when watch() is True; -> (moved, why)) as one continuous motion at `speed` (ramped
        over the first 4 rows): the slip and the force are checked after every row, the reach and watch() every
        `step` (the lab checks them after each of its 5-10 mm moves)."""
        lab = self.lab
        d = np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        v = float(speed or self.speed)
        moved, why, since = 0., "length", 0.
        t0 = lab.t
        k = 0
        while moved < length - 1e-9:
            if lab.hands[hand]["grasp"] is None:
                why = "slip"
                break
            k += 1
            ds = min(v * lab.dt * min(1., k / 4.), length - moved)
            nxt = lab.hand_pos(hand) + ds * d
            if since + ds >= step - 1e-9 and not self.reachable(hand, to_local(nxt), margin):
                why = "reach"
                break
            if not lab.track({hand: nxt}, label="stroke")[hand]:
                why = "reach"
                break
            moved += ds
            since += ds
            if lab.hands[hand]["grasp"] is None:
                why = "slip"
                break
            f = float(np.mean(lab.hands[hand]["window"]))
            if stop_force is not None and f > stop_force:
                why = "force"
                break
            if since >= step - 1e-9:
                since = 0.
                if lab.gripper_gap(hand, lab.q[hand]) < self.STOP_ARMS_M:
                    why = "arms"           # the other arm is in the way: stop before the grippers touch
                    break
                if lab.fixture_geoms and lab.fixture_gap(hand, lab.q[hand]) < self.STOP_FIX_M:
                    why = "fixture"        # a post / the curb / the staple is in the way of the arm
                    break
                if watch is not None and watch():
                    why = "watch"
                    break
        self.phase(hand, "transport", t0)
        return moved, why

    def back_off_fixture(self, hand, dist=.025):
        """After a stroke that stopped at a fixture: carry the held strand `dist` straight away from it in plan before
        letting go, so the strand does not lie against the curb / post and the next pinch finds room for the fingers
        between them (easy, 10-11: a strand drawn against the back curb left no pinch station within 8 cm). -> the
        distance moved (m)."""
        lab = self.lab
        near = lab.fixture_nearest(hand, lab.q[hand])
        if near is None:
            return 0.
        dist, pa, pf = near
        away = np.r_[(pa - pf)[:2], 0.] * (1. if dist >= 0 else -1.)    # penetrating: MuJoCo's points swap sides
        n = float(np.linalg.norm(away))
        if n < 1e-6:
            return 0.
        start = lab.hand_pos(hand).copy()
        lab.move({hand: start + away / n * dist}, self.speed, label="back_off", tol=.004)
        return round(float(np.linalg.norm(lab.hand_pos(hand) - start)), 3)

    def park(self, hands=None, speed=None):
        """Release what is held and lift every hand 8 cm above where it is (clear of the rope)."""
        for h in (hands or self.lab.sides):
            if self.holding(h):
                self.drop(h)
        for h in (hands or self.lab.sides):
            p = self.lab.hand_pos(h)
            self.lab.move({h: p + [0, 0, max(0., .08 - (p[2] - to_world(np.zeros(3))[2]))]}, speed or self.free_speed,
                          tol=TRANSIT_TOL_M,
                          label="park")

    # ------------------------------------------------------------ the deletion (Untie's, with a joint pin / pull plan)
    def delete_first_under(self, rope, k, *, stroke_max=.18, lift=None, pin=True, pull_offset=None):
        """Untie.delete_first_under (an anchored node deletion: the strand pulled 35 mm past its first under-crossing
        from the free end, in strokes with regrasps, the over strand pinned beside the crossing), where the pin and the
        pull are planned together: two ALOHA grippers do not fit side by side everywhere (a 15 cm finger bar and a
        camera mount), so the pull may move up to 4 cm further past the crossing and the pin up to 5 cm along the over
        strand until both fit (GAP_M); without such a pair the stroke goes without a pin (the lab's strokes without a
        free pin hand)."""
        lab = self.lab
        lift = self.stroke_lift if lift is None else lift
        pull_offset = self.pull_offset if pull_offset is None else pull_offset
        rec = dict(k=k, kind="delete", rope=rope, crossings_before=self.count(), strokes=[])
        self.current = rec                   # the runner keeps it if the episode ends inside this move
        fae = self.free_at_end(rope)
        sign = -1. if fae else 1.            # toward the body
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
                s_pull = self.pinch_station(rope, s_u + sign * (pull_offset + .04), toward=sign, search=.06,
                                            one_sided=True)
            if s_pull is None:
                # the strand past the crossing runs packed beside others (easier_v8, 10-11: no 10 mm room within
                # 13.5 cm, every line stalled at 4 crossings): look further along, up to 25 cm past the crossing
                s_pull = self.pinch_station(rope, s_u + sign * (pull_offset + .10), toward=sign, search=.12,
                                            one_sided=True)
                if s_pull is not None:
                    rec.setdefault("far_pull", []).append(round(float(s_pull), 4))
            if s_pull is None:
                rec["result"] = "no pinch room past the crossing"
                break
            pull_hand = self.arm_for(self.local(rope, s_pull))
            if pull_hand is None:
                rec["result"] = "pull station out of reach"
                break
            if rec.get("swap_hand"):            # stall_adapt: the other hand pulls (the first one ran out of reach)
                alt = self.other(pull_hand)
                if self.reachable(alt, self.local(rope, s_pull)):
                    pull_hand = alt
            plan = None
            if pin and self.use_pins and not rec.get("pinless"):
                orope, so = target["other"]
                # a pinned stroke clear for 10 cm, else for 5 cm (a fixture or the pin's arm cuts it short), before
                # going without a pin
                for min_stroke in (.10, .05):
                    plan = self.plan_pin_pull(rope, s_pull, sign, pull_hand, orope, so, s_u=s_u, stroke_len=stroke_max,
                                              min_stroke=min_stroke)
                    if plan is None:
                        alt = self.other(pull_hand)
                        if self.reachable(alt, self.local(rope, s_pull)):
                            plan = self.plan_pin_pull(rope, s_pull, sign, alt, orope, so, s_u=s_u,
                                                      stroke_len=stroke_max, min_stroke=min_stroke)
                    if plan is not None:
                        break
            pin_hand = s_pin = None
            pull_flip = pin_flip = None
            if plan is not None:
                pull_hand, s_pull, pull_flip = plan["pull_hand"], plan["s_pull"], plan["pull_flip"]
                pin_hand, s_pin, pin_flip = plan["pin_hand"], plan["s_pin"], plan["pin_flip"]
            for h in lab.sides:               # a held strand that is not this stroke's pin is let go
                if self.holding(h) and not (h == pin_hand and plan is not None and plan["kept_pin"]):
                    self.drop(h)
            # no pin fits beside the pull (two ALOHA grippers): draw the strand along its own tangent, toward the body,
            # in shorter strokes, so the over strand is dragged as little as possible (knot 22:55)
            length = stroke_max if pin_hand is not None else min(stroke_max, PINLESS_STROKE_M)
            if plan is not None and plan.get("stroke_clear_m") is not None:
                length = min(length, max(.05, float(plan["stroke_clear_m"])))   # end before the planned obstacle
            direction = self.stroke_direction(rope, s_pull, s_u, length, pinless=pin_hand is None)
            st = dict(s_under=round(s_u, 4), s_pull=round(s_pull, 4), pull_hand=pull_hand, pin_hand=pin_hand, length=length,
                      s_pin=None if s_pin is None else round(s_pin, 4), dir=direction.round(3).tolist(),
                      pin_plan=None if plan is None else {x: plan.get(x) for x in ("kept_pin", "gap_mm", "pin_kind",
                                                                                    "stroke_clear_m")})
            stroke_lift = lift
            if plan is not None and plan.get("stroke_lift_m") is not None:
                stroke_lift = float(plan["stroke_lift_m"])
            if pin_hand is None:
                # a pinless stroke: the jaw flip whose stroke runs clearest of the fixtures and the other arm (the flip
                # decides where the camera mount points)
                best = None
                for fl, R_, c_, _, q_at_ in self.pose_options(pull_hand, rope, s_pull):
                    clr, s_lift = self.stroke_clear_best(pull_hand, q_at_, R_, c_, direction, length,
                                                         lab.q[self.other(pull_hand)])
                    if best is None or clr > best[0] + 1e-9:
                        best = (clr, fl, s_lift)
                if best is not None:
                    pull_flip = best[1]
                    stroke_lift = best[2]
                    st["stroke_clear_m"] = round(best[0], 3)
                    if best[0] >= .03:
                        length = min(length, best[0])            # end before the planned obstacle
                        st["length"] = round(length, 3)
            try:
                if pin_hand is not None and not self.holding(pin_hand):
                    if plan.get("pin_kind") == "press":
                        self.take_press(pin_hand, target["other"][0], s_pin, pin_flip)
                    else:
                        self.take(pin_hand, target["other"][0], s_pin, label="grasp_pin", flip=pin_flip)
                self.take(pull_hand, rope, s_pull, label="grasp" if stroke_i == 0 else "regrasp", flip=pull_flip)
            except Stop as e:
                st["error"] = str(e)
                rec["strokes"].append(st)
                self.release_all()
                if self.stall_adapt_grasp:
                    # a failed grasp (the other arm in the way: hard_nostop inst10 looped on "no station keeps the arm
                    # clear", gap -20 mm) is tried again with the other hand pulling, then without a pin
                    ad = rec.setdefault("stall_adapt", [])
                    if not rec.get("swap_hand"):
                        rec["swap_hand"] = True
                        ad.append("other hand after a failed grasp")
                        continue
                    if not rec.get("pinless"):
                        rec["pinless"] = True
                        ad.append("no pin after a failed grasp")
                        continue
                rec["result"] = "grasp failed"
                break
            if rec.get("turn"):
                a = np.arctan2(direction[1], direction[0]) + np.deg2rad(rec["turn"])
                direction = np.array([np.cos(a), np.sin(a), 0.])
            st["lift"] = round(stroke_lift, 3)
            self.lift(pull_hand, stroke_lift)

            def deleted(r0=rem):
                t_, r_ = self.first_under(rope)
                return t_ is None or r_ > r0 + .05
            moved, why = self.stroke(pull_hand, direction, length, step=.005, stop_force=3.5, watch=deleted)
            st.update(moved=round(moved, 3), stop=why, crossings=self.count(),
                      max_force=round(float(lab.max_force[pull_hand]), 2))
            if why == "fixture":
                st["back_off_m"] = self.back_off_fixture(pull_hand)
            rec["strokes"].append(st)
            self.say("  stroke", json.dumps(st))
            self.drop(pull_hand)
            if why == "watch":
                continue
            if why in ("arms", "reach", "fixture") and moved < .03:     # the robot cannot stroke there: give up after 3
                rec["stalls"] = rec.get("stalls", 0) + 1
                if self.stall_adapt:
                    # change the next stroke instead of repeating it (hard_nostop inst10: "stalled (arms)" three times
                    # with the pin's arm beside the pull): no pin after an arms stop, a turned stroke after a fixture
                    # stop, the other hand after a reach stop; four tries in all
                    ad = rec.setdefault("stall_adapt", [])
                    if why == "arms" and not rec.get("pinless"):
                        rec["pinless"] = True
                        ad.append("pinless")
                        for h in lab.sides:
                            if self.holding(h):
                                self.drop(h)
                    elif why == "fixture" and rec.get("turn") is None:
                        rec["turn"] = 40.
                        ad.append("turn +40")
                    elif why == "fixture" and rec.get("turn") == 40.:
                        rec["turn"] = -40.
                        ad.append("turn -40")
                    elif why == "reach" and not rec.get("swap_hand"):
                        rec["swap_hand"] = True
                        ad.append("other hand")
                    if rec["stalls"] < 4:
                        continue
                if rec["stalls"] >= 3:
                    rec["result"] = f"stalled ({why}) three times"
                    break
                continue
            if why in ("force", "slip") and moved < .05:
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
        if self.endgame and not rec.get("done"):
            # the deletion stalled (no pinch room / grasps / jams): the robot endgame moves the strand bodily
            for h in list(lab.hands):
                if self.holding(h):
                    self.drop(h)
            rec["endgame"] = self.lift_across(rope, k)
        rec["crossings_after"] = self.count()
        return rec

    def separate_close(self, rope, *, need=.012, nonlocal_s=.04, push=.025, lift=.015, max_moves=4):
        """Robot finish (very_easy inst8, 10-11: 0 crossings but two nonlocal strands 6.2 mm apart, so the goal's
        one-diameter separation failed): while two nonlocal points of `rope` (more than `nonlocal_s` apart along it),
        or a point of it and another rope, lie closer than `need` in plan, the closer point that has finger room
        (nearest the free end first) is pinched, lifted `lift`, carried `push` straight away from the other strand
        and laid down. -> the records."""
        lab = self.lab
        recs = []
        z_rope = lab.rods[rope].radius + .001
        fae = self.free_at_end(rope)
        for _ in range(max_moves):
            rod = lab.rods[rope]
            P = to_local(rod.nodes(lab.d))[:, :2]
            S = np.asarray(rod.s_nodes, float)
            D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
            D = np.where(np.abs(S[:, None] - S[None, :]) > nonlocal_s, D, np.inf)
            i, j = np.unravel_index(int(np.argmin(D)), D.shape)
            dmin, other_pt = float(D[i, j]), P[j]
            for name, rod2 in lab.rods.items():
                if name == rope:
                    continue
                Q = to_local(rod2.nodes(lab.d))[:, :2]
                D2 = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
                a, b = np.unravel_index(int(np.argmin(D2)), D2.shape)
                if float(D2[a, b]) < dmin:
                    dmin, i, other_pt, j = float(D2[a, b]), a, Q[b], None
            if dmin >= need:
                break
            cands = [i] + ([j] if j is not None else [])
            cands.sort(key=lambda n: -S[n] if fae else S[n])            # nearest the free end first
            rec = dict(kind="separate", dmin_mm=round(dmin * 1000, 1))
            done = False
            for n in cands:
                st = self.pinch_station(rope, float(S[n]), search=.02)
                if st is None:
                    continue
                away = P[n] - (P[j] if (n == i and j is not None) else (P[i] if n == j else other_pt))
                norm_ = float(np.linalg.norm(away))
                if norm_ < 1e-6:
                    continue
                away = away / norm_
                p = self.local(rope, st)
                target = p[:2] + away * push
                hand = self.arm_for(p)
                if hand is None or not self.reachable(hand, np.r_[target, 0.]):
                    continue
                try:
                    self.take(hand, rope, st, label="grasp_separate")
                    self.lift(hand, lift, speed=.03)
                    self.goto(hand, to_world(np.r_[target, z_rope + lift]), speed=.03, label="separate")
                    self.drop(hand)
                    rec.update(hand=hand, station=round(float(st), 4), to=target.round(3).tolist())
                    done = True
                    break
                except Stop as e:
                    rec.setdefault("errors", []).append(str(e)[:160])
                    self.release_all()
            rec["crossings_after"] = self.count()
            recs.append(rec)
            self.say("  separate", json.dumps(rec, default=str)[:300])
            if not done:
                break
        return recs

    def flatten_high(self, rope, *, push=.05, lift=.02, max_moves=4, margin=.002):
        """Robot finish (hard_nostop inst10, 10-12: 0 crossings and planar clear after the deletions, but no success: a
        strand resting on a fixture fails the goal's flatness (no point higher than one diameter + 3 mm above the table),
        and the layout that followed re-tangled the rope): while a node of `rope` lies higher than that (less `margin`),
        the strand is pinched next to its highest node, lifted `lift`, carried `push` in plan away from the nearest
        fixture (the fixture distance's gradient) and laid on the table. -> the records."""
        lab = self.lab
        recs = []
        rod = lab.rods[rope]
        r = rod.radius
        z_rope = r + .001
        limit = 2 * r + .003 - margin
        tried = set()
        for _ in range(max_moves):
            P = to_local(rod.nodes(lab.d))
            S = np.asarray(rod.s_nodes, float)
            z = P[:, 2]
            order = [int(i) for i in np.argsort(-z) if z[i] > limit]
            if not order:
                break
            n = next((i for i in order if round(float(S[i]), 2) not in tried), None)
            if n is None:
                break
            tried.add(round(float(S[n]), 2))
            rec = dict(kind="flatten", z_mm=round(float(z[n]) * 1000, 1), s=round(float(S[n]), 4))
            away = None
            if lab.fixtures:
                def fd(q):
                    return min(_fixture_distance(f, q) for f in lab.fixtures)
                h = .002
                g = np.array([fd(P[n] + [h, 0, 0]) - fd(P[n] - [h, 0, 0]), fd(P[n] + [0, h, 0]) - fd(P[n] - [0, h, 0])])
                if float(np.linalg.norm(g)) > 1e-9 and fd(P[n]) < .05:
                    away = g / float(np.linalg.norm(g))
                    rec["fixture_mm"] = round(fd(P[n]) * 1000, 1)
            if away is None:
                rec["result"] = "no fixture near the high point"
                recs.append(rec)
                continue
            st = self.pinch_station(rope, float(S[n]), search=.03)
            if st is None:
                rec["result"] = "no pinch room"
                recs.append(rec)
                continue
            p = self.local(rope, st)
            target = p[:2] + away * push
            hand = self.arm_for(p)
            if hand is None or not self.reachable(hand, np.r_[target, 0.]):
                alt = None if hand is None else self.other(hand)
                if alt is not None and self.reachable(alt, p) and self.reachable(alt, np.r_[target, 0.]):
                    hand = alt
                else:
                    rec["result"] = "out of reach"
                    recs.append(rec)
                    continue
            try:
                self.take(hand, rope, st, label="grasp_flatten")
                self.lift(hand, lift, speed=.03)
                self.goto(hand, to_world(np.r_[target, max(float(p[2]), z_rope) + lift]), speed=.03, label="flatten")
                self.drop(hand)
                rec.update(hand=hand, station=round(float(st), 4), to=target.round(3).tolist())
            except Stop as e:
                rec["result"] = str(e)[:200]
                self.release_all()
            rec["crossings_after"] = self.count()
            rec["z_max_after_mm"] = round(float(to_local(rod.nodes(lab.d))[:, 2].max()) * 1000, 1)
            recs.append(rec)
            self.say("  flatten", json.dumps(rec, default=str)[:300])
        return recs

    def lift_across(self, rope, k, *, lift=.08, carry=.10):
        """Robot endgame (knot 10-11: a robot reference need not stay inside the lab's moves). At the first
        under-crossing from the free end the over strand is pinched next to the crossing, lifted `lift`, carried `carry`
        across (perpendicular to its own tangent, to the side whose landing point lies clearest of the rest of the rope
        and the posts), and laid down; an over strand without pinch room or reach: the under strand is moved the same
        way. -> a record (crossings before / after, what moved where)."""
        lab = self.lab
        rec = dict(kind="lift_across", crossings_before=self.count())
        target, _ = self.first_under(rope)
        if target is None:
            rec["result"] = "no under-crossing"
            return rec
        z_rope = lab.rods[rope].radius + .001
        for r_, s_ in (tuple(target["other"]), (rope, float(target["s"]))):
            st = self.pinch_station(r_, s_, search=.03)
            if st is None:
                rec.setdefault("skipped", []).append(f"{r_}@{s_:.3f}: no pinch room")
                continue
            p = self.local(r_, st)
            hand = self.arm_for(p)
            if hand is None:
                rec.setdefault("skipped", []).append(f"{r_}@{st:.3f}: out of reach")
                continue
            tg = np.asarray(lab.tangent(r_, st), float)[:2]
            tg = tg / max(float(np.linalg.norm(tg)), 1e-9)
            nrm = np.array([-tg[1], tg[0]])
            L = lab.length(r_)
            rest = self._rest_xy(r_, max(0., st - .15), min(L, st + .15))
            best = None
            for sgn in (1., -1.):
                q = p[:2] + sgn * carry * nrm
                if not self.reachable(hand, np.r_[q, lift]):
                    continue
                clear = float(np.min(np.linalg.norm(rest - q, axis=1))) if len(rest) else 1.
                for c, r, _ in self.post_tops():
                    clear = min(clear, float(np.linalg.norm(q - c)) - r)
                if best is None or clear > best[0]:
                    best = (clear, q)
            if best is None:
                rec.setdefault("skipped", []).append(f"{r_}@{st:.3f}: no reachable side")
                continue
            q = best[1]
            try:
                self.take(hand, r_, st, label="grasp_across")
                self.lift(hand, lift, speed=.04)
                self.goto(hand, to_world(np.r_[q, z_rope + lift]), speed=.04, label="carry_across")
                self.drop(hand)
                rec.update(hand=hand, moved=r_, station=round(float(st), 4), to=q.round(3).tolist(),
                           clearance=round(best[0], 3))
                break
            except Stop as e:
                rec.setdefault("errors", []).append(str(e)[:160])
                self.release_all()
        rec["crossings_after"] = self.count()
        self.say("  lift_across", json.dumps(rec, default=str)[:400])
        return rec

    def lift_over_post(self, rope, k, wrap, *, clear=.04, out=.08, speed=.04):
        """Untie.lift_over_post with one robot hand (knot 10-11: medium_v8's 8 cm eye post; the lab's two-point lift
        needs the grippers 3-5 cm apart): pinch the wrap at its middle (take(): a station clear of the post for the
        fingers), lift to `clear` above the post top -- straight up, or up and 2 / 4 cm out from the post when the
        arm would meet it (checked) --, carry `out` away from the post on the pinch side and lower (carry(): checked
        routes), let go."""
        lab = self.lab
        rec = dict(k=k, kind="lift_over_post", rope=rope, post=wrap["post"], winding=wrap["winding"],
                   crossings_before=self.count())
        self.current = rec
        st = self.pinch_station(rope, .5 * (wrap["s0"] + wrap["s1"]), search=.05)
        if st is None:
            rec["result"] = "no pinch room on the wrap"
            return rec
        p = self.local(rope, st)
        hand = self.arm_for(p)
        if hand is None:
            rec["result"] = "wrap out of reach"
            return rec
        c = np.asarray(wrap["centre"], float)
        d = p[:2] - c
        d = d / max(float(np.linalg.norm(d)), 1e-9)
        z_up = float(wrap["top"]) + clear
        z = lab.rods[rope].radius + .001
        rec.update(hand=hand, station=round(st, 3))
        try:
            self.take(hand, rope, st, label="grasp_wrap")
            t0 = lab.t
            here = to_local(lab.hand_pos(hand))
            lifted = None
            for off in (0., .02, .04):
                goal = to_world(np.r_[here[:2] + off * d, z_up])
                if self.legs_check(hand, [(lab.hand_pos(hand).copy(), goal, False)], lab.R[hand]) is None:
                    lab.move({hand: goal}, speed, label="lift_wrap", tol=.004)
                    lifted = off
                    break
            if lifted is None:
                raise Stop(f"{hand}: no clear lift over the post")
            rec["lift_out_m"] = lifted
            self.phase(hand, "lift_wrap", t0)
            if not self.holding(hand):
                rec["result"] = "slip while lifting"
            else:
                base = to_local(lab.hand_pos(hand))[:2]
                q = base + out * d
                if not self.reachable(hand, np.r_[q, z_up]):
                    q = base + .5 * out * d
                self.carry(hand, to_world(np.r_[q, z_up]), speed=speed, label="carry_off_post")
                self.carry(hand, to_world(np.r_[q, z + .03]), speed=speed, label="lower")
                self.drop(hand)
        except Stop as e:
            rec["result"] = str(e)
            self.release_all()
        rec["wraps_after"] = len(self.post_wraps(rope))
        rec["crossings_after"] = self.count()
        return rec

    def clamp_centres(self):
        """Task-local xy of the clamps' plates (fixtures with role "clamp", name ending in _plate; else every clamp part)."""
        fx = [f for f in self.scene.get("fixtures", []) if f.get("role") == "clamp"]
        plates = [f for f in fx if str(f.get("name", "")).endswith("_plate")] or fx
        return np.array([f["pos"][:2] for f in plates], float).reshape(-1, 2)

    def clear_tail(self, rope, k, *, lift=.06, margin=.05, min_len=.08, max_len=.25, moving=.35,
                   clamp_keep=None):
        """Untie.clear_tail (take the free end, lift it, carry it up to max_len along the clearest of 24 directions),
        with every drawn line kept `clamp_keep` from the clamps in plan (knot 10-11: the lab's parking box sent
        easier's tail beside its clamp, and the next deletion then pulled the short strand next to the clamp,
        which twisted out of the jaws three times)."""
        clamps = self.clamp_centres()
        clamp_keep = self.clamp_keep if clamp_keep is None else clamp_keep
        lab = self.lab
        L = lab.length(rope)
        fae = self.free_at_end(rope)
        rec = dict(k=k, kind="clear_tail", rope=rope, crossings_before=self.count())
        if not self.clear_tail_on:
            rec["result"] = "skipped (variant: no clear_tail)"
            rec["crossings_after"] = rec["crossings_before"]
            return rec
        s_end = self.pinch_station(rope, L - .025 if fae else .025, toward=-1 if fae else 1, search=.10, one_sided=True)
        if s_end is None:
            rec["result"] = "no pinch room near the free end"
            return rec
        p_end = self.local(rope, s_end)
        hand = self.arm_for(p_end)
        if hand is None:
            rec["result"] = "free end out of reach"
            return rec
        hands = [hand]
        if self.stall_adapt_grasp and self.reachable(self.other(hand), p_end):
            hands.append(self.other(hand))      # the other arm takes the end if the first cannot (stall_adapt_grasp)
        out = None
        for hand in hands:
            out = self._clear_tail_with(rope, hand, s_end, p_end, fae, L, rec, clamps, clamp_keep, lift, margin,
                                        min_len, max_len, moving)
            if out == "done":
                break
            if hand != hands[-1]:
                rec.setdefault("tried", []).append(dict(hand=hand, result=rec.pop("result", None)))
        if out != "nodir":                      # (as before: no direction leaves crossings_after unset)
            rec["crossings_after"] = self.count()
        return rec

    def _clear_tail_with(self, rope, hand, s_end, p_end, fae, L, rec, clamps, clamp_keep, lift, margin, min_len,
                         max_len, moving):
        lab = self.lab
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
            if len(clamps) and float(np.min(np.linalg.norm(line[:, None, :] - clamps[None], axis=2))) < clamp_keep:
                continue
            clear = float(np.min(np.linalg.norm(line[:, None, :] - rest[None], axis=2))) if len(rest) else 1.
            for c, r, _ in posts:
                clear = min(clear, float(np.min(np.linalg.norm(line - c, axis=1))) - r)
            score = min(clear, .10) + .2 * ell
            if best is None or score > best[0]:
                best = (score, d, ell, clear)
        if best is None:
            rec["result"] = "no reachable clear direction"
            return "nodir"
        _, d, ell, clear = best
        park = b + ell * d
        z = lab.rods[rope].radius + .001
        rec.update(hand=hand, park=park.round(3).tolist(), length=round(ell, 3), clearance=round(clear, 3))
        try:
            self.take(hand, rope, s_end, label="grasp_tail")
        except Stop as e:
            rec["result"] = str(e)
            self.release_all()
            return "failed"
        try:
            self.lift(hand, lift, speed=.12 * self.turbo if self.fast else .08)
            self.goto(hand, to_world(np.r_[park, z + lift]), speed=.12 * self.turbo if self.fast else .08, label="clear_tail")
            self.drop(hand)
        except Stop as e:
            rec["result"] = str(e)
            self.release_all()
        return "done"


    # ------------------------------------------------------------ layout (Untie.layout with a robot hand-over)
    def layout(self, rope, path_local, *, spacing=.14, lift=.05, settle=.3, handover_gap=.12, pin_min_m=.12,
               max_placements=None, curb_standoff=CURB_STANDOFF_M, carry_slack=.02, carry_speed=.03):
        """Untie.layout (hand-over-hand placement along the region's path, from the clamp outward; the other arm pins
        the previous placed station when it reaches it), with the hand-over done by two real grippers: the carrier
        brings its station to the bimanual lens, the other arm pinches the lifted strand `handover_gap` along the rope
        from the carrier (two ALOHA grippers do not fit 2 cm apart, the lab's gap; take() slides it further if they
        still clash), the carrier lets go, and the other arm carries its own station to that station's point on the
        path. The previous placed station is pinned only when it lies `pin_min_m` or more from the next one in plan
        (closer, the two grippers' finger bars would clash); a pin that leaves the carrier no pinch or no clear carry
        is let go and the station carried alone. Every carry flies a checked route (carry()): at the lift height, or
        higher by 4 / 8 cm, the arm clear of the fixtures and the other arm. A station goes to the path point at its
        own arc length, kept `curb_standoff` from a curb in plan, and every carry ends `carry_slack` short along the
        path: lifted 5 cm, the 14 cm segment from the previous placement cannot span 14 cm in plan, so a full carry
        drags the placed stations back 1-3 cm per placement toward the clamp and the curb (layout_t1 / t2, 10-11).
        Carries go at `carry_speed` (3 cm/s): at 6 cm/s the arm lagged 13-17 mm dragging the strand over the curb, the
        fingers pressed the neighbouring links and the weld was lost by relative_pose_error at 0.2 N (layout_t2).
        max_placements stops early (a mechanics test)."""
        lab = self.lab
        carry_speed = getattr(self, "layout_carry_speed", None) or carry_speed
        lift = getattr(self, "layout_lift", None) or lift
        P = np.asarray(path_local, float)
        P = P[np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6]]
        seg = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        L = lab.length(rope)
        rec = dict(kind="layout", rope=rope, placements=[], handovers=0)
        z_rope = lab.rods[rope].radius + .001
        s_list = list(np.arange(spacing, min(L - .03, seg[-1]) + 1e-9, spacing))
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

        def target(s_):
            a = min(s_, seg[-1])
            return np.r_[np.interp(a, seg, P[:, 0]), np.interp(a, seg, P[:, 1]), z_rope]
        prev = None
        for s_i in s_list:
            if max_placements is not None and len(rec["placements"]) >= max_placements:
                break
            st = self.pinch_station(rope, s_i, search=.04)
            if st is None:
                rec["placements"].append(dict(s=s_i, error="no pinch room"))
                continue
            tgt = self.curb_standoff(target(st), curb_standoff)
            here = self.local(rope, st)
            carrier = self.arm_for(here, prefer=self.arm_for(tgt))
            if carrier is None:
                rec["placements"].append(dict(s=s_i, error="station out of reach"))
                continue
            pin = None
            pin_note = None
            if prev is not None:
                other = "left" if carrier == "right" else "right"
                apart = float(np.linalg.norm(self.local(rope, prev)[:2] - here[:2]))
                if apart < pin_min_m:
                    pin_note = f"{apart:.3f} m from the station"
                elif not self.reachable(other, self.local(rope, prev)):
                    pin_note = f"{other} cannot reach it"
                elif lab.pinch_clearance_at(rope, prev) < lab.pinch_clearance:
                    pin_note = "no pinch room"
                else:
                    try:
                        self.take(other, rope, prev, label="grasp_pin")
                        pin = other
                    except Stop as e:
                        pin_note = str(e)[:160]
            hops = []
            try:
                try:
                    self.take(carrier, rope, st)
                except Stop:
                    if pin is None:
                        raise
                    self.drop(pin)               # the two grippers do not fit: carry this station without the pin
                    pin = None
                    hops.append("pin_let_go")
                    self.take(carrier, rope, st)
                self.lift(carrier, lift)
                if not self.reachable(carrier, tgt):
                    other = "left" if carrier == "right" else "right"
                    if pin == other:
                        self.drop(other)
                        pin = None
                    lens = np.array([0., float(np.clip(tgt[1], -.25, .25))])
                    hops.append(dict(to_lens=self.carry(carrier, to_world(np.r_[lens, z_rope + lift]), speed=carry_speed,
                                                        label="to_lens")))
                    side = 1. if st + handover_gap < L - .03 else -1.
                    s_take = self.pinch_station(rope, st + side * handover_gap, search=.02)
                    if s_take is None:
                        raise Stop("no room for the hand-over pinch")
                    self.take(other, rope, s_take, label="grasp_handover")
                    self.drop(carrier, lower_to_table=False)
                    rec["handovers"] += 1
                    hops.append(dict(handover=round(float(s_take), 3)))
                    carrier, st = other, s_take
                    tgt = self.curb_standoff(target(s_take), curb_standoff)
                end = self.curb_standoff(target(max(0., st - carry_slack)), curb_standoff)   # carry_slack short
                try:
                    hops.append(dict(carry=self.carry(carrier, to_world(np.r_[end[:2], z_rope + lift]), speed=carry_speed)))
                except Stop:
                    if pin is None:
                        raise
                    self.drop(pin)               # the pin's arm blocks every carry route: carry without it
                    pin = None
                    hops.append("pin_let_go")
                    hops.append(dict(carry=self.carry(carrier, to_world(np.r_[end[:2], z_rope + lift]), speed=carry_speed)))
                self.drop(carrier)
            except Stop as e:
                rec["placements"].append(dict(s=s_i, error=str(e), hops=hops))
                self.say("  placement", json.dumps(rec["placements"][-1], default=str))
                self.release_all()
                continue
            if pin is not None and self.holding(pin):
                self.drop(pin)
            rec["placements"].append(dict(s=round(s_i, 3), station=round(st, 3), target=tgt[:2].round(3).tolist(),
                                          carrier=carrier, pin=pin, pin_note=pin_note, hops=hops, crossings=self.count(),
                                          placed_err_mm=round(float(np.linalg.norm(self.local(rope, st)[:2] - tgt[:2]))
                                                              * 1000, 1)))
            self.say("  placement", json.dumps(rec["placements"][-1], default=str))
            prev = st
            lab.wait(settle)
            if getattr(self, "layout_stop_untied", False) and self.count() == 0:
                # the layout as an endgame: it untied the rope (hard_nostop inst10: 1 -> 0 at one placement, then later
                # placements re-tangled it), so stop here and leave the rest to the region fix-up / flatten / the goal
                rec["stopped_untied_at"] = round(float(s_i), 3)
                break
        self.release_all()
        rec["crossings_after"] = self.count()
        self.decisions.append(rec)
        return rec
