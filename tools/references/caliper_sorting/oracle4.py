"""The caliper on its stand (2026-10-18): the whole-episode kinematic planner of the reference (oracle.py replays it in
the physics). Design record: runs/caliper_sorting_design_20261014/STAND_FEASIBILITY.md.

The caliper lies fixed face up on its stand, jaws out over the table, born locked. The left hand works the caliper: it
loosens and locks the slider's locking screw (pinching the knob and turning it), moves the slider (pinching its two side
faces) and reads the vernier with its wrist camera. The right hand brings the sleeves: rack -> outside jaws -> inside
jaws -> its bin. Per sleeve:
  3  left: jaws open to 80 mm                      2  right: the sleeve out of the rack
  4  right: the sleeve between the outside jaws, its rim pressed on the fixed jaw
  5  left: the slider closed onto the sleeve       17 left: knob locked
  6  right: the sleeve lifted out of the locked jaws
  7  left: the outside diameter read               18 left: knob loosened
  8  left: jaws closed to 5 mm
  9  right: the sleeve's bore onto the knife edges, its wall pressed on the fixed edge
  10 left: the slider opened onto the bore         19 left: knob locked
  11 right: the sleeve pulled off the edges along its axis
  12 left: the inside diameter read                20 left: knob loosened
  13 right: the sleeve into the bin whose label matches the two readings
21 is the start's loosening (the caliper is born locked), 15 the end (arms home). One hand moves at a time.
The episode is a generator (plan()): it yields after each sleeve's measurement and receives the two readings (od, id)
that decide a re-measure (a reading more than FAR from every label value, at most MAX_REMEASURE times) and the bin
(the label nearest to both); before each loosening and after each turn of the knob it yields ('knob',) and receives
the knob's angle (None: the plan's own). run() drives it with its own readings (vernier_reading of the planned
slider) and its own knob angles. IK every DT, warm-started; free moves are joint interpolations, contact moves
straight lines.
Usage: oracle4.py [n_sleeves [seed]] -> the plan's frames in oracle4_<seed>.pkl next to this file"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aloha_kin as ak                  # noqa: E402
import geom as c3                       # noqa: E402
import plan3 as p3                      # noqa: E402
import scenario3 as sc3                 # noqa: E402

DT = 0.02
V_LIN, V_ANG = 0.12, 0.75             # a joint move's peak speed (pi/2 V_ANG) stays under 0.03 rad per 25 ms control step
V_FINE = 0.02
J_STEP = 0.024                          # rad per 20 ms frame: 0.03 rad per 25 ms control step
SAFE_Z = 0.20
FAR = float(os.environ.get('ORACLE_FAR', 0.12))   # a reading farther than this from every label value: measured again
MAX_REMEASURE = 2
# the grasps (STAND_FEASIBILITY.md: the stand at the work origin + (-0.18, 0.07) m, 90 mm up)
SL_GRASP = dict(xg=47.0, tau=20.0, flip=False)      # left on the slider: fingertips on its side faces at slider x 47
SLEEVE_TILT, SLEEVE_FLIP = 30.0, True               # right on a sleeve's end faces, approach tilted 30 deg about its axis
KNOB_TILT, KNOB_FLIP = -40.0, False                 # left on the knob: the body leans 40 deg toward the tail (at 30 or
                                                    # 25 deg the hands met at the bore's lock)
KNOB_YAWS = (0.0, 90.0)                             # fingers across the beam, else along it
KNOB_YAWS_BORE = (90.0, 0.0)                        # with the sleeve on the knife edges, its face 30 mm from the knob:
                                                    # fingers along the beam first (across, the turn swung a finger's
                                                    # root into the face)
KNOB_TIP_GAP, FINGER_TIP = 3.0, 16.15               # mm: fingertips above the vernier plate; fingertips past the TCP
KNOB_W = 15.0                                       # deg / s
# locking: +30 deg (the lock takes 20 deg off the loose stop; the pinch lags); while the knob stands below LOCK_MIN after
# a turn, it is turned again by the shortfall + 3 deg. Loosening: back by the knob's angle + FREE_MARGIN, again while it
# stands above FREE_AGAIN (the user, 2026-10-18: no further, so the fingers do not slide on a knob at its loose stop)
KNOB_LOCK_TURN, LOCK_MIN = 30.0, 22.0
FREE_MARGIN, FREE_AGAIN, KNOB_RETRIES = 5.0, 5.0, 2
READ_ROLL = 180.0                                   # deg: the hand turned about the reading camera's line of sight
PRELOAD = 0.3                                       # mm: the sleeve placed this far into the fixed jaw / fixed edge
OD_Y = -0.028                                       # m, C y of the sleeve's centre between the outside jaws
ID_Y, ID_OUT = 0.031, 0.045                         # m: the edges 8 mm into the bore; pulled off along its axis by 45 mm
ID_X0 = 2.5                                         # mm, C x of the bore's centre while the edges (at s = 5) go in
BIN_SLOTS = (-0.020, 0.020)                         # m: two places in a bin, either side of its centre line (y)
# sleeve frames TS are world-aligned at the sleeve centre (grasps are defined that way); the sleeve body is TS @ R_AXIS
R_AXIS = Rot.from_quat([np.sin(-np.pi / 4), 0, 0, np.cos(-np.pi / 4)]).as_matrix()


def interp_T(A, B, u):
    T = np.eye(4)
    T[:3, 3] = (1 - u) * A[:3, 3] + u * B[:3, 3]
    T[:3, :3] = Slerp([0, 1], Rot.from_matrix([A[:3, :3], B[:3, :3]]))(u).as_matrix()
    return T


def lift(T, dz):
    T = T.copy()
    T[2, 3] += dz
    return T


def at_z(T, z):
    T = T.copy()
    T[2, 3] = z
    return T


class IKFail(Exception):
    pass


class Episode:
    def __init__(self, n_sleeves=4, seed=0):
        self.scen = sc3.scene(seed)
        self.L = c3.layout(self.scen)
        self.R = ak.Robot()
        self.rng = np.random.default_rng(0)
        self.abort_on_fail = False
        self.sl = self.scen['sleeves'][:max(1, min(n_sleeves, len(self.scen['sleeves'])))]
        self.q = {s: self.R.q_home[s].copy() for s in 'LR'}
        self.f = {s: p3.Q_OPEN for s in 'LR'}
        self.TC = p3.T_cal(self.L['caliper'])
        self.s = 0.0
        self.knob = c3.G.KNOB_TRAVEL_DEG          # the knob's planned angle off the loose stop: born on its tight stop
        self.TS = [p3.T_from(np.eye(3), c3.sleeve_in_rack(self.L, x['slot'], x['ro'])) for x in self.scen['sleeves']]
        self.GS = [p3.G_sleeve(x['ro'], SLEEVE_TILT, SLEEVE_FLIP) for x in self.scen['sleeves']]
        self.GL = p3.G_slider(**SL_GRASP)
        self.held = {'L': None, 'R': None}
        self.target = {'L': None, 'R': None}
        self.settle_flag = False
        self.reading_flag = False            # frames of the 1 s pause in front of the vernier (the physics reads there)
        self.bin_count = [0] * 4
        self.frames, self.t, self.fail, self.readings = [], 0.0, [], []
        self.decisions, self.remeasures, self.knob_pre, self.knob_turns = [], [], [], []
        self.waits = {}

    # ---------------------------------------------------------------- bookkeeping
    def tcp(self, s):
        return self.R.fk(s, self.q[s])

    def _record(self, stage):
        if not self.frames or self.frames[-1]['stage'] != stage:
            print('%7.2f s  %-72s IK failures %d' % (self.t, stage[:72], len(self.fail)), flush=True)
        self.frames.append(dict(t=self.t, qL=self.q['L'].copy(), qR=self.q['R'].copy(), fL=self.f['L'], fR=self.f['R'],
                                TC=self.TC.copy(), s=self.s, TS=[T.copy() for T in self.TS], stage=stage,
                                heldL=self.held['L'][0] if self.held['L'] else None,
                                heldR=self.held['R'][0] if self.held['R'] else None,
                                tgtL=self.target['L'], tgtR=self.target['R'], settle=self.settle_flag,
                                reading=self.reading_flag))
        self.t += DT

    def _solve(self, s, T, stage):
        q, ep, er = self.R.ik(s, T, self.q[s])
        if ep > 1e-3 or er > 1e-2 or np.abs(q - self.q[s]).max() > 0.25:
            best = self.R.ik_best(s, T, self.rng, n=8, q_prefer=self.q[s])
            if best is None or np.abs(best[0] - self.q[s]).max() > 0.25:
                self.fail.append((stage, s, float(ep), float(er), float(np.abs(q - self.q[s]).max())))
                if self.abort_on_fail:
                    raise IKFail(stage)
                return q
            q = best[0]
        return q

    def _carry(self):
        for side in 'LR':
            h = self.held[side]
            if h is None:
                continue
            obj, G = h
            Th = self.tcp(side)
            if obj == 'slider':
                Tsl = np.linalg.inv(self.TC) @ Th @ np.linalg.inv(G)
                self.s = float(Tsl[0, 3] * 1000)
            else:
                self.TS[obj] = Th @ np.linalg.inv(G)

    # ---------------------------------------------------------------- primitives
    def jmove(self, s, goal, stage, tmin=0.5, qgoal=None):
        if qgoal is None:
            q, ep, er = self.R.ik(s, goal, self.q[s], iters=300)
            if ep > 1e-3 or er > 1e-2:
                best = self.R.ik_best(s, goal, self.rng, n=24, q_prefer=self.q[s])
                if best is None:
                    self.fail.append((stage, s, 'goal unreachable', np.round(goal[:3, 3], 3).tolist()))
                    if self.abort_on_fail:
                        raise IKFail(stage)
                    return
                q = best[0]
            qgoal = q
        A = self.tcp(s)
        dist = np.linalg.norm(goal[:3, 3] - A[:3, 3]) if goal is not None else 0.0
        dq = np.abs(qgoal - self.q[s]).max()
        n = max(2, int(np.ceil(max(dist / V_LIN, dq / V_ANG, tmin) / DT)))
        q0 = self.q[s].copy()
        for i in range(1, n + 1):
            u = 0.5 - 0.5 * np.cos(np.pi * i / n)
            self.q[s] = q0 + (qgoal - q0) * u
            self._carry()
            self._record(stage)

    def move(self, s, goal, stage, v=V_LIN, w=V_ANG, tmin=0.25):
        A = self.tcp(s)
        dist = np.linalg.norm(goal[:3, 3] - A[:3, 3])
        ang = np.linalg.norm(Rot.from_matrix(goal[:3, :3] @ A[:3, :3].T).as_rotvec())
        n = max(2, int(np.ceil(max(dist / v, ang / w, tmin) / DT)))
        # re-timed (more frames) until no joint moves more than J_STEP per frame
        q_start, n_fail = self.q[s].copy(), len(self.fail)
        while True:
            qs = []
            for i in range(1, n + 1):
                u = 0.5 - 0.5 * np.cos(np.pi * i / n)
                self.q[s] = self._solve(s, interp_T(A, goal, u), stage)
                qs.append(self.q[s].copy())
            step = float(np.abs(np.diff(np.vstack([q_start] + qs), axis=0)).max())
            self.q[s] = q_start.copy()
            if step <= J_STEP or n > 20000:
                break
            del self.fail[n_fail:]
            n = int(np.ceil(n * step / J_STEP * 1.02))
        for q in qs:
            self.q[s] = q
            self._carry()
            self._record(stage)

    def fingers(self, s, target, stage, dur=0.35):
        a = self.f[s]
        n = max(2, int(round(dur / DT)))
        for i in range(1, n + 1):
            self.f[s] = a + (target - a) * i / n
            self._record(stage)

    def wait(self, dur, stage, settle=False):
        self.settle_flag = settle
        for _ in range(max(1, int(round(dur / DT)))):
            self._record(stage)
        self.settle_flag = False

    def attach(self, s, obj):
        if obj == 'slider':
            G = np.linalg.inv(p3.slider_T(self.TC, self.s)) @ self.tcp(s)
        else:
            G = np.linalg.inv(self.TS[obj]) @ self.tcp(s)
        self.held[s] = (obj, G)

    def grasp(self, s, T, q_close, obj, stage, approach=0.05, v=0.06, via_up=0.0):
        """Approach T from `approach` m back along the TCP's approach axis (through a point via_up m above that when
        given), close, attach."""
        Tpre = T.copy()
        Tpre[:3, 3] -= approach * T[:3, 0]
        if via_up:
            self.jmove(s, lift(Tpre, via_up), stage)
            self.move(s, Tpre, stage, v=0.08)
        else:
            self.jmove(s, Tpre, stage)
        self.target[s] = obj
        self.move(s, T, stage, v=v)
        self.fingers(s, q_close, stage)
        self.attach(s, obj)
        self.target[s] = None

    def release(self, s, stage, back=0.05):
        self.target[s] = self.held[s][0]
        self.held[s] = None
        self.fingers(s, p3.Q_OPEN, stage)
        T = self.tcp(s)
        T[:3, 3] -= back * T[:3, 0]
        self.move(s, T, stage, v=0.08)
        self.target[s] = None

    # ---------------------------------------------------------------- waiting places
    def park(self, s, stage):
        """A hand waits clear of the other's work: the left 10 cm above its slider grasp at s = 110 mm (over the beam's
        tail), the right 22 cm up between the rack and the caliper, the sleeve (if it holds one) as it is held."""
        if s == 'L':
            T = lift(p3.slider_T(self.TC, 110.0) @ self.GL, 0.10)
        else:
            k = self.held['R'][0] if self.held['R'] else 0
            c = np.array([self.L['caliper'][0] + 0.17, self.L['caliper'][1] - 0.05, 0.24])
            T = p3.T_from(np.eye(3), c) @ self.GS[k]
        self.jmove(s, T, stage)

    # ---------------------------------------------------------------- the left hand on the caliper
    def set_jaws(self, s_target, stage, v=0.03):
        """Left hand pinches the slider's side faces, moves it to s_target, lets go, waits over the tail."""
        T = p3.slider_T(self.TC, self.s) @ self.GL
        self.grasp('L', T, p3.Q_ROLLER, 'slider', stage, approach=0.04, via_up=0.06)
        self.move('L', p3.slider_T(self.TC, s_target) @ self.GL, stage, v=v)
        self.release('L', stage, back=0.04)
        self.park('L', stage)

    def close_onto(self, s_end, s_near, stage, v_near):
        """Left hand moves the slider onto a sleeve: to s_near at 3 cm/s, the last part at v_near (the physics replay
        creeps on until the jaws press the sleeve), lets go."""
        T = p3.slider_T(self.TC, self.s) @ self.GL
        self.grasp('L', T, p3.Q_ROLLER, 'slider', stage, approach=0.04, via_up=0.06)
        self.move('L', p3.slider_T(self.TC, s_near) @ self.GL, stage, v=0.03)
        self.move('L', p3.slider_T(self.TC, s_end) @ self.GL, stage, v=v_near)
        self.release('L', stage, back=0.04)
        self.park('L', stage)

    def read(self, what, stage):
        s = self.s
        k = p3.vernier_line(s)
        T = p3.slider_T(self.TC, s) @ p3.G_read(s, k, READ_ROLL)
        # the reading pose's largest-margin IK, the way in planned backward from it along the approach axis (the fingers
        # slide in under the vernier plate's edge; from above they would pass through the slider)
        best = self.R.ik_best('L', T, self.rng, n=32, q_prefer=self.q['L'])
        q_T = best[0] if best is not None else self.q['L']
        for back in (0.10, 0.07, 0.05, 0.04):
            Tpre = T.copy()
            Tpre[:3, 3] -= back * T[:3, 0]
            q_pre, ep, er = self.R.track('L', q_T, T, Tpre)
            if ep <= 1e-3 and er <= 1e-2:
                break
        # high above the way in first (a joint move straight from the waiting place swept the fingers through the
        # slider), then down to its start and in
        # the joint moves from and back to the waiting place happen high up: straight up from the way in's start as far
        # as the arm follows (a joint move straight to it swept the fingers through the beam and the slider)
        for up in (SAFE_Z - Tpre[2, 3], 0.08, 0.05, 0.03, 0.02):
            q_hi, ep, er = self.R.track('L', q_pre, Tpre, lift(Tpre, up))
            if ep <= 1e-3 and er <= 1e-2:
                break
        self.fingers('L', p3.Q_CLOSED_EMPTY, stage, dur=0.2)
        self.jmove('L', None, stage, qgoal=q_hi)
        self.move('L', Tpre, stage, v=0.06)
        self.move('L', T, stage, v=0.06)
        self.reading_flag = True
        self.wait(1.0, stage)
        self.reading_flag = False
        val = c3.vernier_reading(s, self.scen)
        self.readings.append((what, s, val))
        self.move('L', Tpre, stage, v=0.06)
        self.move('L', lift(Tpre, up), stage, v=0.08)
        self.fingers('L', p3.Q_OPEN, stage, dur=0.2)
        self.park('L', stage)
        return val

    def knob_T(self):
        """The knob's frame: on its axis at the vernier plate's top face, axes as the slider's."""
        K = c3.KNOB
        return p3.slider_T(self.TC, self.s) @ p3.T_from(np.eye(3), np.array([K['x'], K['y'], c3.SLIDER['z_top']]) * 1e-3)

    def _knob_turn_poses(self, delta_deg, pre, yaw0):
        """The hand's poses on the knob through a turn: the grasp pose turned by yaw0 - pre x delta (deg about the knob's
        axis; yaw0 0: fingers across the beam), then frames to +(1 - pre) x delta."""
        Tk = self.knob_T()
        th = np.radians(KNOB_TILT)
        a = Tk[:3, :3] @ np.array([np.sin(th), 0, -np.cos(th)])          # th < 0: the body leans toward the tail
        R = p3.tcp_frame(a, Tk[:3, :3] @ np.array([0, -1.0 if KNOB_FLIP else 1.0, 0]))
        tip = Tk[:3, 3] + Tk[:3, 2] * (KNOB_TIP_GAP + 5.0 * abs(np.sin(th))) * 1e-3
        axis, P = Tk[:3, :3] @ np.array([0, 0, -1.0]), Tk[:3, 3].copy()
        T0 = p3.T_from(R, tip - a * FINGER_TIP * 1e-3)
        rot = lambda deg: p3.T_from(Rot.from_rotvec(axis * np.radians(deg)).as_matrix(), np.zeros(3))     # noqa: E731
        about = lambda deg: p3.T_from(np.eye(3), P) @ rot(deg) @ p3.T_from(np.eye(3), -P)                # noqa: E731
        T = about(yaw0 - delta_deg * pre) @ T0
        n = max(2, int(np.ceil(abs(delta_deg) / KNOB_W / DT)))
        return T, [about(delta_deg * (0.5 - 0.5 * np.cos(np.pi * i / n))) @ T for i in range(1, n + 1)]

    def _knob_grasp_q(self, delta_deg, pre, yaw0):
        """The arm's joints at the grasp of a pre-turned knob turn (its largest-margin IK) if warm-started IK follows the
        whole turn from them, else None."""
        T, path = self._knob_turn_poses(delta_deg, pre, yaw0)
        best = self.R.ik_best('L', T, self.rng, n=32, q_prefer=self.q['L'])
        if best is None:
            return None
        q = best[0]
        for Ti in path:
            q, ep, er = self.R.ik('L', Ti, q)
            if ep > 1e-3 or er > 1e-2:
                return None
        return best[0]

    def turn_knob(self, delta_deg, stage, pres=(1.0, 0.5, 0.0), yaws=KNOB_YAWS):
        """Left hand: down onto the knob from above, pinch it, turn it by delta_deg about its axis (+ tightens), let go,
        come back up. The knob is round, so the hand may take it already turned (by -pre x delta, the first of `pres`
        and of the yaws the arm can follow through the whole turn). The designed contact (target 'knob') runs from the
        approach's last 5 cm to the way up."""
        yaw0, pre, q_T = next(((y, p, q) for y in yaws for p in pres
                               for q in [self._knob_grasp_q(delta_deg, p, y)] if q is not None), (yaws[0], 0.0, None))
        T, path = self._knob_turn_poses(delta_deg, pre, yaw0)
        self.knob_pre.append((stage[:2].strip(), delta_deg, yaw0, pre))
        if q_T is None:
            best = self.R.ik_best('L', T, self.rng, n=32, q_prefer=self.q['L'])
            q_T = best[0] if best is not None else self.q['L']
        for back in (0.05, 0.035, 0.02):                # the approach planned backward from the grasp
            Tpre = T.copy()
            Tpre[:3, 3] -= back * T[:3, 0]
            q_pre, ep, er = self.R.track('L', q_T, T, Tpre)
            if ep <= 1e-3 and er <= 1e-2:
                break
        self.jmove('L', None, stage, qgoal=q_pre)
        self.target['L'] = 'knob'
        self.move('L', T, stage, v=0.03)
        self.wait(0.5, stage)
        self.fingers('L', p3.Q_KNOB, stage, dur=0.5)
        self.wait(0.3, stage)
        for Ti in path:
            self.q['L'] = self._solve('L', Ti, stage)
            self._record(stage)
        self.knob = float(np.clip(self.knob + delta_deg, 0.0, c3.G.KNOB_TRAVEL_DEG))
        self.wait(0.3, stage)
        self.fingers('L', p3.Q_OPEN, stage, dur=0.4)
        for back in (0.05, 0.035, 0.02):          # back up the approach line as far as the arm can follow it
            Tb = self.tcp('L')
            Tb[:3, 3] -= back * Tb[:3, 0]
            _, ep, er = self.R.track('L', self.q['L'], self.tcp('L'), Tb)
            if ep <= 1e-3 and er <= 1e-2:
                break
        self.move('L', Tb, stage, v=0.06)
        self.target['L'] = None
        self.park('L', stage)

    def knob_angle(self):
        """(generator) The knob's angle now, asked of whoever runs the episode (None: the plan's own)."""
        ang = yield ('knob',)
        if ang is not None:
            self.knob = float(ang)
        return self.knob

    def lock(self, stage, yaws=KNOB_YAWS):
        """(generator) Lock the knob: +KNOB_LOCK_TURN, then again by the shortfall + 3 deg while it stands below
        LOCK_MIN (at most KNOB_RETRIES times)."""
        turn = KNOB_LOCK_TURN
        for k in range(1 + KNOB_RETRIES):
            before = yield from self.knob_angle()
            self.knob_turns.append(dict(stage=stage[:2].strip(), before=before, turn=turn, again=k))
            self.turn_knob(turn, stage, yaws=yaws)
            ang = yield from self.knob_angle()
            self.knob_turns[-1]['after'] = ang
            if ang >= LOCK_MIN:
                return
            turn = min(LOCK_MIN - ang + 3.0, c3.G.KNOB_TRAVEL_DEG - ang)

    def loosen(self, stage, only_if_locked=False):
        """(generator) Loosen the knob: back by its angle + FREE_MARGIN; again the same way while it stands above
        FREE_AGAIN (at most KNOB_RETRIES times). only_if_locked: nothing when it frees the slider already."""
        ang = yield from self.knob_angle()
        if only_if_locked and ang <= c3.G.RELEASE_DEG:
            return
        for k in range(1 + KNOB_RETRIES):
            turn = -(ang + FREE_MARGIN)
            self.knob_turns.append(dict(stage=stage[:2].strip(), before=ang, turn=turn, again=k))
            self.turn_knob(turn, stage, pres=(0.0, 0.25, 0.5, 1.0))
            ang = yield from self.knob_angle()
            self.knob_turns[-1]['after'] = ang
            if ang <= FREE_AGAIN:
                return

    # ---------------------------------------------------------------- the right hand and the sleeves
    def c_point(self, x, y, z=0.0):
        """A point given in the caliper frame C (m), in the mount frame."""
        return (self.TC @ np.r_[x, y, z, 1.0])[:3]

    def sleeve_to(self, k, centre, stage, v=V_LIN):
        self.move('R', p3.T_from(np.eye(3), centre) @ self.GS[k], stage, v=v)

    def pick(self, k, stage):
        self.grasp('R', self.TS[k] @ self.GS[k], p3.Q_SLEEVE, k, stage, approach=0.06)
        self.move('R', at_z(self.tcp('R'), SAFE_Z + 0.05), stage)

    def od_in(self, k, stage):
        """The sleeve down between the open outside jaws 2 mm off the fixed jaw, held still (the physics replay lets it
        settle), then onto the fixed jaw (its rim PRELOAD into it)."""
        ro = self.sl[k]['ro']
        gap = self.c_point((ro + 2.0) * 1e-3, OD_Y)
        self.jmove('R', p3.T_from(np.eye(3), gap + [0, 0, 0.09]) @ self.GS[k], stage)
        self.sleeve_to(k, gap, stage, v=0.05)
        self.wait(0.5, stage, settle=True)
        self.sleeve_to(k, self.c_point((ro - PRELOAD) * 1e-3, OD_Y), stage, v=V_FINE / 4)

    def od_out(self, k, stage):
        """The preload off first: the sleeve's centre to the middle between the locked jaws (pressed on the fixed jaw as it
        is lifted, its friction there, 35 mm off the sleeve's axis, twisted it in the pinch about that axis harder than
        the end faces' friction could hold: the grasp audit, 2026-10-18); then straight up out of the jaws (above its
        widest line the sleeve narrows: nothing slides along them), slowly, then to the waiting place."""
        ro = self.sl[k]['ro']
        self.sleeve_to(k, self.c_point(ro * 1e-3, OD_Y), stage, v=V_FINE / 4)
        self.wait(1.0, stage)
        self.sleeve_to(k, self.TS[k][:3, 3] + [0, 0, 0.07], stage, v=0.009)
        self.park('R', stage)

    def id_in(self, k, stage):
        """The bore onto the knife edges (closed to 5 mm) along its axis, then sideways until its wall presses the fixed
        edge (PRELOAD)."""
        ri = self.sl[k]['ri']
        outside = self.c_point(ID_X0 * 1e-3, ID_Y + ID_OUT)
        self.jmove('R', p3.T_from(np.eye(3), outside + [0, 0, 0.06]) @ self.GS[k], stage)
        self.sleeve_to(k, outside, stage, v=0.05)
        self.wait(0.5, stage, settle=True)
        self.sleeve_to(k, self.c_point(ID_X0 * 1e-3, ID_Y), stage, v=V_FINE / 2)
        self.sleeve_to(k, self.c_point((ri + PRELOAD) * 1e-3, ID_Y), stage, v=V_FINE / 4)

    def id_out(self, k, stage):
        """Off the locked edges along its axis at 1 cm/s, then to the waiting place."""
        self.sleeve_to(k, self.TS[k][:3, 3] + self.TC[:3, :3] @ np.array([0, ID_OUT, 0]), stage, v=0.01)
        self.park('R', stage)

    def to_bin(self, k, kb, stage):
        """Set down standing on edge in the bin's next place, let go, up."""
        ro = self.sl[k]['ro']
        slot = BIN_SLOTS[self.bin_count[kb] % len(BIN_SLOTS)]
        self.bin_count[kb] += 1
        c_set = np.array([self.L['bins'][kb][0], self.L['bins'][kb][1] + slot, (c3.BINS['wall'] + ro) * 1e-3])
        Tb = p3.T_from(np.eye(3), c_set + np.array([0, 0, 0.003])) @ self.GS[k]
        self.jmove('R', at_z(Tb, SAFE_Z + 0.05), stage)
        self.move('R', lift(Tb, 0.04), stage, v=0.08)
        self.move('R', Tb, stage, v=0.03)
        self.TS[k] = p3.T_from(np.eye(3), c_set)
        self.release('R', stage, back=0.0)
        self.target['R'] = k
        self.move('R', lift(self.tcp('R'), 0.10), stage, v=0.08)
        self.target['R'] = None

    # ---------------------------------------------------------------- the measurements
    def measure_od(self, k, again=0):
        tag = f'sleeve {k + 1}'
        ag = f' (measured again, {again})' if again else ''
        ro = self.sl[k]['ro']
        if abs(self.s - c3.S_OPEN) > 0.5:
            self.set_jaws(c3.S_OPEN, f'3 {tag}: left hand opens the jaws to {c3.S_OPEN:.0f} mm{ag}', v=0.06)
        self.od_in(k, f'4 {tag}: right hand puts the sleeve between the outside jaws, rim on the fixed jaw{ag}')
        self.close_onto(2 * ro, 2 * ro + 1.0, f'5 {tag}: left hand closes the slider onto the sleeve (outside diameter){ag}',
                        v_near=0.002)
        yield from self.lock(f'17 {tag}: left hand locks the slider (outside diameter){ag}')
        self.od_out(k, f'6 {tag}: right hand lifts the sleeve out of the locked jaws{ag}')
        val = self.read('OD', f'7 {tag}: left wrist camera reads the vernier (outside diameter){ag}')
        yield from self.loosen(f'18 {tag}: left hand loosens the locking screw{ag}', only_if_locked=True)
        return val

    def measure_id(self, k, again=0):
        tag = f'sleeve {k + 1}'
        ag = f' (measured again, {again})' if again else ''
        ri = self.sl[k]['ri']
        self.set_jaws(5.0, f'8 {tag}: left hand closes the jaws to 5 mm{ag}', v=0.06)
        self.id_in(k, f'9 {tag}: right hand puts the bore onto the knife edges, wall on the fixed edge{ag}')
        self.close_onto(2 * ri, 2 * ri - 1.0, f'10 {tag}: left hand opens the slider onto the bore (inside diameter){ag}',
                        v_near=0.002)
        yield from self.lock(f'19 {tag}: left hand locks the slider (inside diameter){ag}', yaws=KNOB_YAWS_BORE)
        self.id_out(k, f'11 {tag}: right hand pulls the sleeve off the knife edges{ag}')
        val = self.read('ID', f'12 {tag}: left wrist camera reads the vernier (inside diameter){ag}')
        yield from self.loosen(f'20 {tag}: left hand loosens the locking screw{ag}')
        return val

    def far(self, od, idd):
        e_od = min(abs(od - self.scen['classes'][c][1]) for c in self.scen['bins'])
        e_id = min(abs(idd - self.scen['classes'][c][0]) for c in self.scen['bins'])
        return e_od > FAR + 1e-9, e_id > FAR + 1e-9

    def choose_bin(self, od, idd):
        """The bin whose label is nearest to the readings: min over labels of max(|ID - ID_k|, |OD - OD_k|)."""
        dist = [max(abs(idd - self.scen['classes'][c][0]), abs(od - self.scen['classes'][c][1])) for c in self.scen['bins']]
        return int(np.argmin(dist))

    def plan(self):
        st = '21 left hand loosens the locking screw (the caliper is born locked)'
        self.park('R', '0 start')
        yield from self.loosen(st)
        yield ('started',)
        for k in range(len(self.sl)):
            tag = f'sleeve {k + 1}'
            if abs(self.s - c3.S_OPEN) > 0.5:
                self.set_jaws(c3.S_OPEN, f'3 {tag}: left hand opens the jaws to {c3.S_OPEN:.0f} mm', v=0.06)
            self.pick(k, f'2 {tag}: right hand takes the sleeve out of the rack')
            yield from self.measure_od(k)
            yield from self.measure_id(k)
            od, idd = yield ('measured', k)
            n_re = 0
            while n_re < MAX_REMEASURE:
                far_od, far_id = self.far(od, idd)
                if not (far_od or far_id):
                    break
                n_re += 1
                self.remeasures.append(dict(sleeve=k, round=n_re, od=od, id=idd, again_od=far_od, again_id=far_id))
                if far_od:
                    yield from self.measure_od(k, again=n_re)
                if far_id:
                    yield from self.measure_id(k, again=n_re)
                od2, id2 = yield ('remeasured', k)
                od, idd = (od2 if far_od else od), (id2 if far_id else idd)
            kb = self.choose_bin(od, idd)
            self.decisions.append(dict(sleeve=k, od=od, id=idd, bin=kb, remeasures=n_re))
            self.to_bin(k, kb, f'13 {tag}: right hand puts the sleeve into bin {kb + 1} (readings OD {od:.1f}, ID {idd:.1f})')
            yield ('placed', k)
        st = '15 arms home'
        self.jmove('R', None, st, qgoal=self.R.q_home['R'])
        self.jmove('L', None, st, qgoal=self.R.q_home['L'])
        self.wait(0.5, st)
        yield ('done',)

    def run(self):
        """Kinematic episode: the plan driven by its own readings and knob angles."""
        g = self.plan()
        msg = next(g)
        while msg[0] != 'done':
            if msg[0] in ('measured', 'remeasured'):
                last = {w: v for w, _, v in self.readings}
                msg = g.send((last['OD'], last['ID']))
            else:
                msg = next(g)
        return self


def main(n=4, seed=0):
    E = Episode(n, seed).run()
    with open(HERE / f'oracle4_{seed}.pkl', 'wb') as fh:
        pickle.dump(dict(frames=E.frames, dt=DT, fail=E.fail, readings=E.readings, scenario=E.scen,
                         decisions=E.decisions, remeasures=E.remeasures, knob_turns=E.knob_turns), fh)
    print('frames %d, duration %.1f s, IK failures %d' % (len(E.frames), E.t, len(E.fail)))
    for f in E.fail[:12]:
        print('   IK fail', f)
    for r in E.readings:
        print('   reading', r)
    print('   knob pre-turns (stage, turn, yaw, pre):', E.knob_pre)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(int(a[0]) if a else 4, int(a[1]) if len(a) > 1 else 0)
