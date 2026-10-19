"""Robot-level oracle of the v1 rope_threading tiers (rope agent, 2026-10-10 night): the realised lay-and-pick relay
(relay_plan.RelayPlanner -> oracle.timeline) executed in a MuJoCo Session as ordinary JOINT_ABS rows, closed loop on the
rope's measured state (the privileged oracle reads the chain's nodes, as the family's feedback relay measures its rope):

  - every grasp is re-aimed at the rope's measured material station (position and tangent, the same grasp family),
    checked against the fixtures (capsules) and confirmed by the grasp assist; a miss opens, re-measures and retries;
  - every feed is closed loop: the held tip is measured and turned onto the ring's axis FEED_APPROACH_M before the
    plane (up to 3 corrections), pushed along the axis in IK steps, and the crossing is read from the task's own metrics;
    a feed that did not cross pulls back, re-aligns and pushes again;
  - after every release the relay waits for the rope to settle (<= QUIET_MS m/s, at most 3 s) before it is measured;
  - the last ring is held: the task allows holding; the episode ends when the task's success holds for its stable window.

No state edits, no teleporting: the rows are the only input, saved for deterministic fresh-process replays.

usage (the cloud host): python -m topogym.planning.rope_threading.redesign.robot_relay <tier> <instance.json> <timeline.json>
              <out_dir> [--stop-after-phase N] [--hold-s 20]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from topogym.planning.rope_threading.redesign import scenes as S
from . import aloha_kin as K
from . import oracle as O

GRIP_OPEN, GRIP_CLOSED = 0.037, 0.002
DT = 0.04
JOINT_SPEED = 0.95                   # rad/s per joint: 0.038 rad per 0.04 s control, inside the standard service's
                                     # 0.04 rad per step (the servo limit is pi; the privileged references used 0.8 pi)
STEP_LIMIT_RAD = 0.038               # every recorded row's arm joints move at most this from the previous row
ARM_COLS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
QUIET_MS = 0.02                      # rope speed (m/s) after a release before it is measured again
OPENING_OPEN, OPENING_CLOSED = 0.06, 0.012
BIAS = True
SAMEHAND_LOWER_M = 0.0               # lower a same-hand feeder by this much before it lets go (variant)
SAMEHAND_RELEASE_ROWS = 0            # open a same-hand feeder's gripper over this many controls (0: at once)
DUMP_CKPT = True                     # write branch points (rows + executor state) at every block start
LIFT_SPEED = 0.06                    # m/s of the straight lift off a laid rope
LIFT_WAIT_S = 0.0                    # s waited (still) after letting go of a laid rope before lifting
LET_GO_S = 30.0                      # s waited after letting go when holding did not settle the rope
LAY_MAX_DEG = 45.0                   # a laid tip further off its lay line fails the block (rewind)
RR_PICK_FLOOR = 0.014                # the lowest TCP height a lay correction goes to
SLOW_STEP_S = 8.0                    # s of wall time per control that counts as a jammed simulation
TRACE = True                         # print the tick records (tip, speed) every log_every rows
ALIGN_ITERS = 5
ALIGN_MAX_STEP = math.radians(10)
ALIGN_GAIN = 0.6
YAW_TRIES = (0.0, 20.0, -20.0, 40.0, -40.0)      # deg about the vertical for a pick of a rope on the table
ROLL_TRIES = (0.0, 20.0, -20.0, 40.0, -40.0, 60.0, -60.0)   # deg about the rope for a pinch in the air
ALIGN_PUSH_MAX_DEG = 20.0
ALIGN_PUSH_MAX_ERR = 0.02
PICK_TCP_Z = 0.014                   # TCP height of a pick of a rope lying on the table
BIAS_MAX = 0.025
BIAS_CONTACT = 0.012                 # rad: a lag beyond the current bias by more than this is contact, not load
BIAS_MIN_Z = 0.04                    # m: the bias is refreshed only with the TCP this high over the table
WRIST_SPEED = 1.2                    # rad/s for forearm_roll / wrist_angle / wrist_rotate (record: at 2.5 rad/s forearm_roll
                                     # reached only ~1.9 rad/s and wrist_angle trailed by 0.3 rad)
PACE_LAG = 0.25                      # rad: a joint move waits while any joint of the arm lags its target by more than this
PACE_MAX_HOLD = 25                   # controls waited per interpolation step at most
VIA_UPS = (0.06, 0.10, 0.15, 0.20)   # m: lift heights of a re-routed move onto a pre-grasp
EMPTY_ONLY = True                    # wrist limit + pacing only while the hand is open
SKIP_BLOCKED_M = 0.005               # m: a pre-grasp move blocked by more than this (no lifted route) is not made
FEED_FORCING_ABORT = True            # a feed attempt whose alignment / pull-back is reported forcing is given up
MIN_PROTRUDE_M = 0.08                # m: a wall hand-over wants the tip this far past the wall
INCH_M = 0.03
INCH_TRIES = 0                       # off: every inchworm lost the rope on re-closing (route7 easier 4x, hard 6-32x)
HELD_TWIN = False                    # (off: not yet route-validated) a held-rope plan segment that would turn the wrist family uses the pi-twin grasp
HELD_TWIN_RAD = 1.5
HELD_RESEED = True                   # a held-rope plan segment that would flip the wrist branch is re-solved in the current branch
PLAN_SEED = False                    # a re-aimed grasp is solved from the plan's grasp joints (its wrist family), not the current
PLAN_BRANCH = False                  # (opt-in) an in-air pinch takes the clear candidate nearest the plan's grasp joints
POKE_EXTRA_M = 0.0                   # (opt-in) m pushed further through a same-hand ring (the tip retracts after the release)
GRIP_ROWS = 0                        # (opt-in) at most this many controls per gripper open / close (0: as called)
CART_SPEED_SCALE = 1.0               # (opt-in) scales the TCP speed of straight lines (descents, lifts, corrections)
FOLLOW_SPEED_SCALE = 1.0             # (opt-in) scales the plan's TCP speeds (the joint-step rule still caps each row)
LAY_RELEASE_ROWS = 0                 # (opt-in) open the fingers over this many controls when laying the rope down
PLAN_BRANCH_MAX_SIGMA = 1.0          # PLAN_BRANCH only for pinches at most this far from the tip (0.08: the receives, not station picks)
CHORD_MIN_ALIGN = 0.8                # a station pick's chord to the tip must run this well along the lay line (cos)
RECORD_APPLIED = True                # the recorded rows are the Session's applied targets (hold_action after each step)
PATH_CHECK = True                    # the joint move onto a pre-grasp is capsule-checked; a blocked one tries the next pose
PUSH_CLEAR = 0.004                   # m: capsule clearance required during a feed push (the planner's 8 / 6 mm elsewhere)
GRASP_SIGMA_VARIANTS = (0.0, 0.010, 0.020, -0.008)     # a retried block grasps this much farther from the tip (m)
FEED_VARIANTS = ((0.05, 0.0), (0.035, 0.0), (0.05, 0.015), (0.03, 0.010))   # (push speed m/s, extra poke m)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _rot_between(a, b):
    a, b = _unit(a), _unit(b)
    v = np.cross(a, b)
    s_, c_ = float(np.linalg.norm(v)), float(a @ b)
    if s_ < 1e-9:
        return np.eye(3)
    return Rotation.from_rotvec(v / s_ * math.atan2(s_, c_)).as_matrix()


class RobotRelay:
    def __init__(self, tier, instance, timeline, out_dir, log_every=25):
        import rcb_topogym as rt
        from .session_build import register, use_tmp
        use_tmp()
        register(tier)
        self.tier, self.out = tier, Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.s = rt.Session("rope_threading", instance=instance, strict=False)
        inst = json.loads(Path(instance).read_text())
        home = np.asarray(inst["reset_joint_targets"], dtype=np.float64)
        self.home_row = home.astype(np.float32)
        self.q = {"left": home[0:6].copy(), "right": home[7:13].copy()}
        self.g = {"left": float(home[6]), "right": float(home[13])}
        self.design = S.build(tier)
        self.rings = {r.name: r for r in self.design.rings}
        self.required = [f"{n}{'+' if sg > 0 else '-'}" for n, sg in self.design.required_crossings()]
        self.tl = json.loads(Path(timeline).read_text())
        self.off = K.ROOT_WORLD + np.array([0.0, 0.0, K.TABLE_Z_WORLD])
        self.rows, self.events, self.log = [], [], []
        self.done = False
        self.t0 = time.time()
        self.prev_nodes = None
        self.speed = 0.0
        self.log_every = log_every
        self.last_metrics = {}
        self.rewinds = []
        self.last_lay_dir = None
        self.check_lay_dir = None
        self.pick_lay_dir = None
        self.jam, self.slow = False, 0
        self._forcing_seen, self.forcing_now, self.forcing_ticks = None, False, 0
        self.held_sigma = {}
        self.variant = 0
        self.paced_rows = 0
        self.applied_changed = 0
        self.forcing_log = []           # the executor's own record of forcing ticks in the kept rows (rewinds cut it)
        # load / gravity bias per arm (main 10-10 22:15, the claws agent's finding: open-loop ALOHA rows sag 5-10 mm at
        # the TCP, wrist joints ~0.02 rad off): rows command q + bias, bias = last sent - measured joints, refreshed
        # whenever the arm is nearly still (clamped to BIAS_MAX)
        self.bias = {"left": np.zeros(6), "right": np.zeros(6)}

    # ------------------------------------------------------------------------------------------------ io / state
    def row(self):
        ql = self.q["left"] + self.bias["left"]
        qr = self.q["right"] + self.bias["right"]
        return np.r_[ql, self.g["left"], qr, self.g["right"]].astype(np.float32)

    def update_bias(self, arms=("left", "right"), vel_max=0.05):
        if not BIAS:
            return
        pos = np.asarray(self.s.robot.positions(), float)
        vel = np.asarray(self.s.robot.velocities(), float)
        for arm in arms:
            sl = slice(0, 6) if arm == "left" else slice(7, 13)
            if float(np.max(np.abs(vel[sl]))) > vel_max:
                continue
            if self.measured_tcp(arm)[2] < BIAS_MIN_Z:
                continue            # near the table the fingers may press on it: that is not a load to compensate
            sent = np.asarray(self.s._targets, float)[sl]
            dev = sent - pos[sl]
            if float(np.max(np.abs(dev - self.bias[arm]))) > BIAS_CONTACT:
                # a large lag is a hand pressing on something, not sag: compensating it would press harder (easier's
                # oracle pressed V with 128 N for 4 s); keep the bias, and drop back towards zero
                self.bias[arm] = 0.5 * self.bias[arm]
                continue
            self.bias[arm] = np.clip(dev, -BIAS_MAX, BIAS_MAX)

    def step(self, n=1):
        for _ in range(n):
            if self.done or self.jam:
                return
            target = self.row()
            if STEP_LIMIT_RAD > 0:
                # the standard service's joint-step rule (rcb_topogym JOINT_STEP_LIMIT_RAD = 0.04 rad per control step,
                # relative to the previous row; row 0 relative to the hold action): a row whose arm joints jump further
                # (a fast joint move, a load-bias refresh) is reached through evenly spaced rows of its own, each a
                # control step in this closed loop, so the recorded rows run unchanged under the standard service
                prev = self.rows[-1] if self.rows else self.home_row
                d = target[ARM_COLS].astype(np.float64) - prev[ARM_COLS].astype(np.float64)
                k = int(math.ceil(float(np.max(np.abs(d))) / STEP_LIMIT_RAD - 1e-9))
                if k > 1:
                    for j in range(1, k):
                        inter = target.copy()
                        inter[ARM_COLS] = (prev[ARM_COLS].astype(np.float64) + d * (j / k)).astype(np.float32)
                        self._step_row(inter)
                        if self.done or self.jam:
                            return
            self._step_row(target)

    def _step_row(self, row):
        if True:
            self.rows.append(row)
            t_step = time.process_time()           # CPU time: a starved process (load 170, nice 19) is not a jam
            self.done = bool(self.s.step(self.rows[-1]))
            if RECORD_APPLIED:
                # record the joint targets the Session applied (its joint / gripper limit clip, joint speed slew, table
                # and static guards; float32-exact): the standard service then has nothing to clip or guard on replay
                applied = np.asarray(self.s.hold_action(), dtype=np.float32)
                if not np.array_equal(applied, self.rows[-1]):
                    self.applied_changed += 1
                self.rows[-1] = applied
            lf = getattr(self.s, "last_forcing", None)
            if lf is not None and lf is not self._forcing_seen:
                # an arm drive at its cap while pressing (the Session's forcing evidence): the moving primitives stop
                self._forcing_seen = lf
                self.forcing_now = True
                self.forcing_ticks += 1
                try:
                    info = lf[1] if isinstance(lf, tuple) else dict(lf)
                    self.forcing_log.append(dict(row=len(self.rows) - 1, arm=lf[0] if isinstance(lf, tuple) else None,
                                                 joint=info.get("joint"), force_n=round(float(info.get("force_n", 0.0)), 1),
                                                 partner=info.get("partner")))
                except Exception:                  # evidence format: keep the row only
                    self.forcing_log.append(dict(row=len(self.rows) - 1))
            if time.process_time() - t_step > SLOW_STEP_S:
                self.slow += 1
                if self.slow >= 3:
                    # rr7 hard: a rope jammed in a grommet window took ~36 s per control: treat it as a failed block
                    self.jam = True
                    self.event(event="physics_jam", cpu_seconds=round(time.process_time() - t_step, 1))
            else:
                self.slow = 0
            N = self.nodes()
            if self.prev_nodes is not None:
                self.speed = float(np.max(np.linalg.norm(N - self.prev_nodes, axis=1)) / DT)
            self.prev_nodes = N
            if len(self.rows) % self.log_every == 0:
                self.metrics("tick", quiet_log=not TRACE)

    def nodes(self):
        w = self.s.env._world
        r = w._ropes["rope"]
        return r.node_positions(w.data) - self.off

    def held(self, arm):
        g = self.s.env._world.read_state().get("grasp") or {}
        b = ((g.get("hands") or {}).get(arm) or {}).get("binding")
        return None if not b else tuple(b["target"])

    def material_point(self, sigma, N=None):
        N = self.nodes() if N is None else N
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(N, axis=0), axis=1))]
        target = s[-1] - sigma
        i = int(np.clip(np.searchsorted(s, target), 1, len(N) - 1))
        w = (target - s[i - 1]) / max(s[i] - s[i - 1], 1e-12)
        return N[i - 1] * (1 - w) + N[i] * w, _unit(N[i] - N[i - 1])

    def metrics(self, label, quiet_log=False):
        m = dict(self.s.env._task.result.metrics)
        seq = [f"{a}{'+' if b > 0 else '-'}" for a, b in m.get("crossing_sequence", [])]
        self.last_metrics = dict(sequence=seq, quiet=m.get("quiet"), thickness_clear=m.get("thickness_clear"),
                                 ambiguous=m.get("ambiguous"))
        N = self.nodes()
        tip, tdir = N[-1], _unit(N[-1] - N[-20])
        rec = dict(label=label, row=len(self.rows), sequence=seq, quiet=m.get("quiet"), speed=round(self.speed, 4),
                   tip=np.round(tip, 3).tolist(), tip_dir=np.round(tdir, 2).tolist(),
                   held={a: self.held(a) for a in ("left", "right")}, wall_s=round(time.time() - self.t0, 1))
        self.log.append(rec)
        if not quiet_log:
            print(json.dumps(rec, default=str), flush=True)
        return rec

    def prefix(self):
        seq = self.last_metrics.get("sequence", [])
        n = 0
        for a, b in zip(seq, self.required):
            if a != b:
                break
            n += 1
        return n, seq

    def event(self, **kw):
        kw = dict(row=len(self.rows), **kw)
        self.events.append(kw)
        print(json.dumps(dict(event=kw), default=str), flush=True)

    # ------------------------------------------------------------------------------------------------ checkpoints
    def checkpoint(self, label):
        """A complete checkpoint of the current tick (env history, segment storage) plus this executor's and the
        Session's own counters. Rewinding to it and going on yields rows that are one forward sequence from the start
        (the discarded branch's rows are dropped); the fresh-process replay of the rows is the verdict."""
        env = self.s.env
        env._checkpoint_current()
        s = self.s
        return dict(label=label, ref=dict(env._info()["state_ref"]), rows=len(self.rows),
                    q={a: v.copy() for a, v in self.q.items()}, g=dict(self.g), speed=self.speed,
                    bias={a: v.copy() for a, v in self.bias.items()}, held_sigma=dict(self.held_sigma),
                    pick_lay_dir=None if self.pick_lay_dir is None else self.pick_lay_dir.copy(),
                    prev_nodes=None if self.prev_nodes is None else self.prev_nodes.copy(),
                    sess=dict(n_steps=s.n_steps, sim_time_s=s.sim_time_s, physics_steps=s.physics_steps,
                              _targets=np.array(s._targets, copy=True), episode_end=s.episode_end,
                              episode_done=s.episode_done, _reward=s._reward, instability=s.instability,
                              last_step_ticks=s.last_step_ticks),
                    last_metrics=dict(self.last_metrics))

    def rewind(self, cp, why):
        self.s.env.privileged.rewind_to(cp["ref"])
        for k, v in cp["sess"].items():
            setattr(self.s, k, v.copy() if isinstance(v, np.ndarray) else v)
        del self.rows[cp["rows"]:]
        self.forcing_log = [f for f in self.forcing_log if f["row"] < cp["rows"]]
        self.q = {a: v.copy() for a, v in cp["q"].items()}
        self.g = dict(cp["g"])
        self.bias = {a: v.copy() for a, v in cp["bias"].items()}
        self.held_sigma = dict(cp["held_sigma"])
        self.speed = cp["speed"]
        self.prev_nodes = None if cp["prev_nodes"] is None else cp["prev_nodes"].copy()
        self.last_metrics = dict(cp["last_metrics"])
        self.done = False
        self.jam, self.slow = False, 0
        self.last_lay_dir = self.check_lay_dir = None
        self.pick_lay_dir = cp.get("pick_lay_dir")
        self.rewinds.append(dict(to=cp["label"], row=cp["rows"], why=why))
        self.event(event="rewind", to=cp["label"], to_row=cp["rows"], why=why)

    # ------------------------------------------------------------------------------------------------ motion
    def other(self, arm):
        return "left" if arm == "right" else "right"

    def opening(self, arm):
        return OPENING_CLOSED if self.g[arm] < 0.02 else OPENING_OPEN

    def other_caps(self, arm):
        o = self.other(arm)
        return K.arm_capsules(o, self.q[o], self.opening(o))

    def clear(self, arm, q):
        ok, g_arm, g_f, g_o = O.config_clear(arm, q, self.design, self.opening(arm), self.other_caps(arm))
        return ok, dict(arm=round(float(g_arm[0]), 4), arm_with=g_arm[1:], finger=round(float(g_f[0]), 4),
                        finger_with=g_f[1:], other=round(float(g_o[0]), 4))

    def move_q(self, arm, q_target, tcp_dist=0.0, speed=0.12):
        self.update_bias()
        q0 = self.q[arm].copy()
        q_target = np.clip(np.asarray(q_target, float), K.LIMITS[:, 0], K.LIMITS[:, 1])
        empty = self.g[arm] >= GRIP_OPEN - 1e-4 or not EMPTY_ONLY
        # the wrist limit and the pacing apply to empty-hand moves only (pace2: slowing the draws / lays changed the rope's
        # dynamics and the laid tips whipped back 153-157 deg at A); the P2 / bulkhead presses were empty-hand transits
        vmax = np.array([JOINT_SPEED] * 3 + [min(JOINT_SPEED, WRIST_SPEED) if empty else JOINT_SPEED] * 3)
        dq = float(np.max(np.abs(q_target - q0) / vmax))
        n = max(1, int(math.ceil(max(dq / DT, tcp_dist / max(speed, 1e-3) / DT))))
        for k in range(1, n + 1):
            self.q[arm] = q0 + (q_target - q0) * k / n
            self.step()
            if self.done:
                return
            if PACE_LAG > 0 and empty:
                # lag-paced (new head 10-11, very_easy / easier: a fast joint move left forearm_roll 0.79 rad behind its
                # target; the lagging arm left the checked joint line and its finger hit P2 at 85-133 N): the next step
                # waits until the arm is back near the commanded line
                f0, hold = self.forcing_ticks, 0
                while hold < PACE_MAX_HOLD and self.lag(arm) > PACE_LAG and self.forcing_ticks == f0:
                    self.step()
                    hold += 1
                    if self.done or self.jam:
                        return
                self.paced_rows += hold

    def lag(self, arm):
        """Largest joint lag (rad) of the arm behind its sent target (the load bias excluded)."""
        sl = slice(0, 6) if arm == "left" else slice(7, 13)
        pos = np.asarray(self.s.robot.positions(), float)[sl]
        return float(np.max(np.abs(self.q[arm] + self.bias[arm] - pos)))

    def tcp(self, arm):
        return K.tcp_pose(arm, self.q[arm])

    def solve(self, arm, pos, R, seed=None, check=True):
        """IK for a full pose seeded by the current joints; (q, ok, info)."""
        q, pe, re = K.ik(arm, pos, rotation=R, seed=self.q[arm] if seed is None else seed)
        if pe > 0.003 or re > 0.05:
            return q, False, dict(why="ik", pos_err=round(pe, 4), rot_err=round(re, 3))
        if q[4] > 2.0:          # core excludes the forearm-gripper pair: no wrist_angle beyond ~116 deg
            return q, False, dict(why="wrist_angle", q4=round(float(q[4]), 3))
        if not check:
            return q, True, {}
        ok, info = self.clear(arm, q)
        return q, ok, dict(why=None if ok else "collision", **info)

    def path_margin(self, arm, qa, qb, step=0.05):
        """Smallest clearance margin (m, beyond the planner's 8 / 6 / 5 mm) on the joint line qa -> qb (a start that is
        itself in a near-contact skips the first quarter)."""
        n = max(2, int(math.ceil(float(np.max(np.abs(qb - qa))) / step)))
        start_ok = self.clear(arm, qa)[0]
        worst, where = np.inf, None
        for s_ in np.r_[np.linspace(0.0, 1.0, n + 1)[1:-1], 1.0]:
            if not start_ok and s_ < 0.25:
                continue
            _, info = self.clear(arm, qa + s_ * (qb - qa))
            m = min(info["arm"] - O.ARM_CLEAR, info["finger"] - O.FINGER_CLEAR, info["other"] - O.SELF_CLEAR)
            if m < worst:
                worst, where = m, dict(at=round(float(s_), 2), **info)
        return float(worst), where

    def lifted_route(self, arm, q_pre, p_pre, R_pre):
        """[q_up, q_via, q_pre]: straight up from the current pose, across at that height, down onto the pre-grasp; None
        when no lift height gives a clear route."""
        T = self.tcp(arm)
        for up in VIA_UPS:
            q_up, ok1, _ = self.solve(arm, T[:3, 3] + np.array([0.0, 0.0, up]), T[:3, :3], check=False)
            q_via, ok2, _ = self.solve(arm, p_pre + np.array([0.0, 0.0, up]), R_pre, seed=q_pre)
            if not (ok1 and ok2):
                continue
            chain = [self.q[arm], q_up, q_via, q_pre]
            if all(self.path_margin(arm, a, b)[0] >= 0.0 for a, b in zip(chain[:-1], chain[1:])):
                return chain[1:]
        return None

    def path_clear(self, arm, qa, qb, step=0.05):
        """Capsule check of the joint line qa -> qb (a start that is itself in a near-contact, e.g. fingers at the ring
        just let go of, skips the first quarter)."""
        n = max(2, int(math.ceil(float(np.max(np.abs(qb - qa))) / step)))
        start_ok = self.clear(arm, qa)[0]
        for s_ in np.linspace(0.0, 1.0, n + 1)[1:-1]:
            if not start_ok and s_ < 0.25:
                continue
            ok, info = self.clear(arm, qa + s_ * (qb - qa))
            if not ok:
                return False, dict(at=round(float(s_), 2), **info)
        return True, {}

    def move_pose(self, arm, pos, R, speed=0.10, check=True):
        q, ok, info = self.solve(arm, pos, R, check=check)
        if not ok:
            return False, info
        dist = float(np.linalg.norm(self.tcp(arm)[:3, 3] - pos))
        self.move_q(arm, q, dist, speed)
        return True, info

    def cart_line(self, arm, p1, R, speed=0.05, step=0.008, check=True):
        speed = speed * CART_SPEED_SCALE
        """Straight TCP line from the current position to p1 at orientation R (IK every `step`); stops before the first
        infeasible step. Returns the fraction done."""
        p0 = self.tcp(arm)[:3, 3].copy()
        L = float(np.linalg.norm(p1 - p0))
        n = max(1, int(math.ceil(L / step)))
        self.forcing_now = False
        qs = []
        for i in range(1, n + 1):
            p = p0 + (p1 - p0) * i / n
            q, ok, info = self.solve(arm, p, R, check=check)
            if not ok:
                self.event(arm=arm, event="cart_line_stop", at=round(i / n, 3), info=info)
                return (i - 1) / n
            qs.append(self.q[arm].copy())
            self.move_q(arm, q, L / n, speed)
            if self.forcing_now:
                # easy / easier oracles pressed a post / the eyelet at 80-146 N: no straight line pushes on through
                self.event(arm=arm, event="cart_line_forcing", at=round(i / n, 3), forcing=str(self.s.last_forcing)[:200])
                self.back_off(arm, qs)
                return (i - 1) / n
            if self.done:
                return i / n
        return 1.0

    def back_off(self, arm, qs, k=2):
        """After a forcing stop: command the joints of k steps earlier at once (the stopped target kept pressing for
        0.6 s in the very_easy / easier references), then go on."""
        if qs:
            self.q[arm] = qs[max(0, len(qs) - k)].copy()
            self.step(4)
        self.forcing_now = False

    def grip(self, arm, value, rows=8):
        self.g[arm] = value
        self.step(min(rows, GRIP_ROWS) if GRIP_ROWS > 0 else rows)

    def wait_quiet(self, max_s=3.0, thresh=None):
        thresh = QUIET_MS if thresh is None else thresh   # read at call time: --set QUIET_MS reaches it
        t = 0.0
        while t < max_s and not self.done:
            self.step(2)
            t += 2 * DT
            if self.speed <= thresh:
                break
        return t

    # ------------------------------------------------------------------------------------------------ blocks
    def held_twin(self, arm, steps):
        """For a held rope: the plan's steps re-solved with the grasp turned by pi about the gripper's approach axis (the
        same pinch of the same rope line, fingers swapped) when that keeps the wrist where it is. easier (inst5): the
        re-aimed receive beyond G1 ended in the other wrist family; the plan's draw then turned the wrist ~3 rad with
        the rope in hand and wrapped it round the wrist (up to 426 N on the rope). Returns the q list, or None."""
        if not HELD_TWIN or not steps or self.g[arm] >= GRIP_OPEN - 1e-4:
            return None
        q0 = self.q[arm]
        q_plan = np.asarray(steps[0]["q"], float)
        d_plan = float(np.max(np.abs(q_plan[3:] - q0[3:])))
        if d_plan < HELD_TWIN_RAD:
            return None
        flip = Rotation.from_rotvec(np.array([math.pi, 0.0, 0.0])).as_matrix()
        out, seed = [], q0
        for st in steps:
            R = K.tcp_pose(arm, np.asarray(st["q"], float))[:3, :3] @ flip
            q, ok, _ = self.solve(arm, np.asarray(st["pos"], float), R, seed=seed, check=False)
            if not ok:
                return None
            out.append(q)
            seed = q
        d_twin = float(np.max(np.abs(out[0][3:] - q0[3:])))
        if d_twin > d_plan - 1.0:
            return None
        self.event(arm=arm, event="held_twin", wrist_change_plan=round(d_plan, 2), wrist_change_twin=round(d_twin, 2))
        return out

    def held_reseed(self, arm, steps):
        """For a held rope: the plan's poses re-solved in the arm's current IK branch when the plan's first step would flip
        the wrist (forearm_roll / wrist_rotate by ~pi, wrist_angle mirrored: the same TCP pose in the other branch). The
        re-aimed receive beyond a grommet can land in the other branch; the plan's draw then turned the wrist ~3 rad with
        the rope in hand. Under the 0.04 rad / step rule that turn takes ~80 controls and swung the rope over the wall;
        the laid tip then whipped 144-177 deg off its lay line at every A lay. Returns the q list, or None."""
        if not HELD_RESEED or not steps or self.g[arm] >= GRIP_OPEN - 1e-4:
            return None
        q0 = self.q[arm]
        if float(np.max(np.abs(np.asarray(steps[0]["q"], float)[3:] - q0[3:]))) < HELD_TWIN_RAD:
            return None
        out, seed = [], q0
        for st in steps:
            T = K.tcp_pose(arm, np.asarray(st["q"], float))
            q, ok, info = self.solve(arm, T[:3, 3], T[:3, :3], seed=seed, check=False)
            if not ok or float(np.max(np.abs(q[3:] - seed[3:]))) > 1.0:
                self.event(arm=arm, event="held_reseed_fail", step=len(out), info=info,
                           q=np.round(q, 3).tolist(), seed=np.round(seed, 3).tolist())
                return None
            out.append(q)
            seed = q
        self.event(arm=arm, event="held_reseed", wrist_change_plan=round(float(np.max(np.abs(np.asarray(steps[0]["q"], float)[3:] - q0[3:]))), 2),
                   steps=len(out))
        return out

    def follow(self, arm, steps):
        cur = self.tcp(arm)[:3, 3]
        twin = self.held_reseed(arm, steps) or self.held_twin(arm, steps)
        for i, st in enumerate(steps):
            p = np.asarray(st["pos"], float)
            q = np.asarray(st["q"], float) if twin is None else twin[i]
            if PATH_CHECK and self.g[arm] >= GRIP_OPEN - 1e-4 and float(np.max(np.abs(q - self.q[arm]))) > 0.2:
                # an empty-hand planned move starts from where the hand really is (after a lift-off, a re-aimed grasp),
                # not from the planned pose (easier: the right gripper pressed V at 220 N going to the pick beyond it)
                m, where = self.path_margin(arm, self.q[arm], q)
                if m < 0.0:
                    route = self.lifted_route(arm, q, p, K.tcp_pose(arm, q)[:3, :3])
                    self.event(arm=arm, event="follow_lifted" if route is not None else "follow_blocked",
                               margin=round(m, 4), info=where)
                    for qv in (route[:-1] if route is not None else []):
                        self.move_q(arm, qv, 0.06, 0.08)
            self.move_q(arm, q, float(np.linalg.norm(p - cur)), float(st["speed"]) * FOLLOW_SPEED_SCALE)
            cur = p
            if self.done:
                return

    def reaim(self, arm, steps, sigma):
        """The planned approach (steps; the last one grasps) moved onto the measured material station: the last pose at
        the measured point (a rope lying on the table: TCP 10 mm above its centre), its gripper turned about the held
        family's rope axis onto the measured tangent; the approach poses shifted with it."""
        p_meas, t_meas = self.material_point(sigma)
        last = steps[-1]
        R_last = np.asarray(last["R"], float)
        held_dir = None
        for k in O.GRASP_KINDS:
            h = R_last @ O.GRASP_KINDS[k]
            if held_dir is None or abs(h @ t_meas) > abs(held_dir @ t_meas):
                held_dir = h
        if held_dir @ t_meas < 0:
            t_meas = -t_meas
        Rc = _rot_between(held_dir, t_meas) @ R_last
        target = p_meas.copy()
        if p_meas[2] < 0.012:
            target[2] = PICK_TCP_Z              # pads at the rope, fingertips at the table
        pre = [np.asarray(st["pos"], float) - np.asarray(last["pos"], float) for st in steps[:-1]]
        return target, Rc, pre

    def grasp_at(self, arm, sigma, steps, tries=3):
        """Approach (re-aimed), close, confirm; retry with a fresh measurement."""
        sigma = sigma + GRASP_SIGMA_VARIANTS[self.variant % len(GRASP_SIGMA_VARIANTS)]
        if sigma > 0.08 and self.pick_lay_dir is not None:
            # rr19 hard T: the rope between the planned station and the tip had folded back, the carry then held the tip
            # behind the hand: take the station whose chord to the tip runs along the lay line (nearest to the nominal)
            N = self.nodes()
            tip = N[-1]
            best = None
            for c in np.arange(0.10, 0.285, 0.005):
                pt = self.material_point(c, N)[0]
                ch = tip - pt
                ch[2] = 0.0
                if np.linalg.norm(ch) < 0.5 * c:
                    continue
                al = float(_unit(ch) @ self.pick_lay_dir)
                if al >= CHORD_MIN_ALIGN and (best is None or abs(c - sigma) < abs(best - sigma)):
                    best = float(c)
            if best is not None and abs(best - sigma) > 1e-6:
                self.event(arm=arm, event="station_chord_shift", sigma=round(sigma, 3), to=round(best, 3))
                sigma = best
            self.pick_lay_dir = None
        if sigma > 0.08 and self.variant > 0:
            # a station pick: the material station nearest the planned pick point (the laid rope rarely lies exactly
            # as planned: rr7 medium / very_easy B picks were out of reach with the planned station)
            planned = np.asarray(steps[-1]["pos"], float)
            N = self.nodes()
            cands = np.arange(sigma - 0.03, sigma + 0.061, 0.005)
            dist = [float(np.linalg.norm(self.material_point(c, N)[0][:2] - planned[:2])) for c in cands]
            sig2 = float(cands[int(np.argmin(dist))])
            if abs(sig2 - sigma) > 1e-6:
                self.event(arm=arm, event="station_shift", sigma=round(sigma, 3), to=round(sig2, 3),
                           dist_mm=round(1000 * min(dist), 1))
            sigma = sig2
        for attempt in range(tries):
            if self.done:
                return False
            target, Rc, pre = self.reaim(arm, steps, sigma)
            if attempt > 0:
                target = target + np.array([0.0, 0.0, -0.003 * attempt]) if target[2] > 0.02 else target
            # pre-grasp: backed off along the gripper's approach axis with the final rotation already set (the plan's
            # family change at a grasp swung the wrist by pi right above the rope, diag_pick 10-10 23:10), then a
            # straight TCP line onto the target
            ok_all = False
            cands = []
            if target[2] < 0.03:
                rots = [Rotation.from_rotvec(np.array([0.0, 0.0, math.radians(y)])).as_matrix() for y in YAW_TRIES]
            else:
                # a pinch in the air: the hand may turn about the rope (rr17 medium: the re-aimed receive beyond G1 needed
                # wrist_angle 2.08 rad > 2.0)
                t_axis = self.material_point(sigma)[1]
                rots = [Rotation.from_rotvec(t_axis * math.radians(r)).as_matrix() for r in ROLL_TRIES]
            for Rrot in rots:
                Ry = Rrot @ Rc
                approach = Ry[:, 0]
                for off in [-0.07 * approach, -0.05 * approach + np.array([0.0, 0.0, 0.03])] + ([pre[-1]] if pre else []):
                    cands.append((Ry, off))
            picked, blocked = None, []
            if PLAN_BRANCH and steps and target[2] > 0.03 and sigma <= PLAN_BRANCH_MAX_SIGMA:
                # a pinch in the air (the receive beyond a grommet): of the clear candidates, the one whose grasp joints are
                # nearest the plan's (its IK branch), so the held draw that follows needs no wrist flip with the rope in
                # hand (under the 0.04 rad / step rule that flip swung the rope over the wall and the A lay whipped)
                q_plan = np.asarray(steps[-1]["q"], float)
                best, why = None, {}
                for Ry, off in cands:
                    for seed in (q_plan, None):
                        q_pre, ok_pre, inf = self.solve(arm, target + off, Ry, seed=seed)
                        tag = "plan" if seed is not None else "cur"
                        near = float(np.max(np.abs(q_pre[3:] - q_plan[3:]))) < 1.0
                        if not ok_pre:
                            k = f"{tag}:{inf.get('why')}:{'near' if near else 'far'}"
                            why[k] = why.get(k, 0) + 1
                            continue
                        q_t, ok_t, inf2 = self.solve(arm, target, Ry, seed=q_pre, check=False)
                        if not ok_t:
                            k = f"{tag}:target_{inf2.get('why')}:{'near' if near else 'far'}"
                            why[k] = why.get(k, 0) + 1
                            continue
                        if PATH_CHECK and self.path_margin(arm, self.q[arm], q_pre)[0] < 0.0:
                            k = f"{tag}:path:{'near' if near else 'far'}"
                            why[k] = why.get(k, 0) + 1
                            continue
                        dist = float(np.max(np.abs(q_t[3:] - q_plan[3:])))
                        if best is None or dist < best[0]:
                            best = (dist, Ry, off, q_pre)
                if best is not None:
                    self.event(arm=arm, event="plan_branch_pick", wrist_dist=round(best[0], 3), rejected=why)
                    picked = (best[1], best[2], best[3], None)
                if (best is None or best[0] > 1.0) and len(steps) >= 2 and \
                        float(np.linalg.norm(target - np.asarray(steps[-1]["pos"], float))) < 0.02:
                    # no re-aimed candidate in the plan's branch: the plan's own approach pose and grasp rotation, onto the
                    # measured target (the tip is where the plan expected it, within 2 cm)
                    q_pre = np.asarray(steps[-2]["q"], float)
                    R_plan = K.tcp_pose(arm, np.asarray(steps[-1]["q"], float))[:3, :3]
                    off = K.tcp_pose(arm, q_pre)[:3, 3] - target
                    self.event(arm=arm, event="plan_pose_pick", off_m=round(float(np.linalg.norm(off)), 3))
                    picked = (R_plan, off, q_pre, None)
            for Ry, off in (cands if picked is None else []):
                q_pre, ok_pre, info = self.solve(arm, target + off, Ry,
                                                 seed=np.asarray(steps[-1]["q"], float) if PLAN_SEED and steps else None)
                if ok_pre:
                    q_t, ok_t, _ = self.solve(arm, target, Ry, seed=q_pre, check=False)
                    if not ok_t:
                        continue
                    if PATH_CHECK:
                        m, where = self.path_margin(arm, self.q[arm], q_pre)
                        if m < 0.0:
                            # the new head's very_easy: the joint move onto a pre-grasp swung the hand into post P2
                            blocked.append((m, Ry, off, q_pre, where))
                            continue
                    picked = (Ry, off, q_pre, None)
                    break
            if picked is None and blocked:
                blocked.sort(key=lambda b: -b[0])
                for m, Ry, off, q_pre, where in blocked[:3]:
                    route = self.lifted_route(arm, q_pre, target + off, Ry)
                    if route is not None:
                        self.event(arm=arm, event="pre_move_lifted", margin=round(m, 4), info=where)
                        picked = (Ry, off, q_pre, route)
                        break
                if picked is None:
                    m, Ry, off, q_pre, where = blocked[0]
                    if m < -SKIP_BLOCKED_M:
                        # hard (inst5): the direct move with the finger 19 mm into P1's capsule jammed the hand on the
                        # post at 197 N; the planned approach (checked from the real start, lifted if needed) instead
                        self.event(arm=arm, event="pre_move_skipped", margin=round(m, 4), info=where)
                    else:
                        self.event(arm=arm, event="pre_move_blocked", margin=round(m, 4), info=where)
                        picked = (Ry, off, q_pre, None)
            for Ry, off, q_pre, route in ([picked] if picked is not None else []):
                if True:
                    Rc = Ry
                    if route is not None:
                        for qv in route[:-1]:
                            self.move_q(arm, qv, 0.06, 0.08)
                    dist = float(np.linalg.norm(self.tcp(arm)[:3, 3] - (target + off)))
                    self.move_q(arm, q_pre, dist, 0.12)
                    if self.g[arm] < GRIP_OPEN - 1e-4:
                        # the home pose's gripper is almost closed (0.0084): open before the descent (diag_pick 3: the
                        # nearly closed fingers landed on the rope and pushed it aside)
                        self.grip(arm, GRIP_OPEN, 8)
                    self.settle(arm, rows=6)
                    frac = self.cart_line(arm, target, Rc, 0.035, step=0.006)
                    rem = float(np.linalg.norm(self.tcp(arm)[:3, 3] - target))
                    if frac <= 0.98 and rem < 0.045:
                        # the capsule check's finger-table allowance stops a few cm above a rope lying on the table
                        frac = 1.0 if self.cart_line(arm, target, Rc, 0.03, step=0.006, check=False) > 0.98 else frac
                    ok_all = frac > 0.98
                    if not ok_all:
                        self.event(arm=arm, event="reaim_infeasible", attempt=attempt, info=dict(why="cart_line",
                                                                                                fraction=round(frac, 3)))
                    break
            else:
                self.event(arm=arm, event="reaim_infeasible", attempt=attempt, info=info)
            if not ok_all:
                if self.g[arm] < GRIP_OPEN - 1e-4:
                    self.grip(arm, GRIP_OPEN, 8)
                self.follow(arm, steps)
            if True:
                # settle, refresh the load bias at the grasp pose and correct onto the (re-measured) target
                self.settle(arm)
                target2, Rc2, _ = self.reaim(arm, steps, sigma)
                if attempt > 0 and target2[2] > 0.02:
                    target2 = target2 + np.array([0.0, 0.0, -0.003 * attempt])
                if float(np.linalg.norm(target2 - target)) < 0.02:
                    target = target2               # the rotation stays the one the approach was solved with
                self.cart_line(arm, target, Rc, 0.02, step=0.004, check=False)
                self.settle(arm)
            diag = self.grasp_diag(arm, target, sigma)
            self.grip(arm, GRIP_CLOSED, 10)
            h = self.held(arm)
            self.event(arm=arm, event="grasp", attempt=attempt, sigma=sigma, held=h, q=np.round(self.q[arm], 3).tolist(),
                       target=np.round(target, 4).tolist(), diag=diag,
                       opening_after=round(float(self.s.gripper_width(arm)), 4))
            if h is not None:
                self.held_sigma[arm] = float(sigma)
                return True
            self.grip(arm, GRIP_OPEN, 6)
            up = self.tcp(arm)[:3, 3] + np.array([0.0, 0.0, 0.04])
            self.move_pose(arm, up, self.tcp(arm)[:3, :3], 0.1, check=False)
            self.wait_quiet(1.0)
        return False

    def settle(self, arm, rows=10, vel=0.03):
        for _ in range(rows):
            v = np.asarray(self.s.robot.velocities(), float)
            sl = slice(0, 6) if arm == "left" else slice(7, 13)
            if float(np.max(np.abs(v[sl]))) < vel:
                break
            self.step()
        self.update_bias((arm,))

    def measured_tcp(self, arm):
        pos, _ = self.s.ee_pose(arm)
        return np.asarray(pos, float) - self.off

    def grasp_diag(self, arm, target, sigma):
        m = self.measured_tcp(arm)
        N = self.nodes()
        p, _ = self.material_point(sigma, N)
        dn = np.linalg.norm(N - m, axis=1)
        return dict(tcp_meas=np.round(m, 4).tolist(), tcp_err_mm=round(1000 * float(np.linalg.norm(m - target)), 1),
                    station_now=np.round(p, 4).tolist(), station_to_tcp_mm=round(1000 * float(np.linalg.norm(p - m)), 1),
                    nearest_node_mm=round(1000 * float(dn.min()), 1), nearest_node=int(np.argmin(dn)), n_nodes=len(N),
                    bias=np.round(self.bias[arm], 4).tolist())

    def release(self, arm, settle=True, slow_rows=0):
        if slow_rows > 0:
            # a slow opening: the fingers part over slow_rows controls instead of one step
            g0 = self.g[arm]
            for k in range(1, slow_rows + 1):
                self.g[arm] = g0 + (GRIP_OPEN - g0) * k / slow_rows
                self.step()
            self.step(3)
        else:
            self.grip(arm, GRIP_OPEN, 6)
        self.event(arm=arm, event="release", slow_rows=slow_rows)
        if settle:
            self.wait_quiet(3.0)

    def tip_state(self, arm):
        """The held rope's tip and its direction from the held point (the material point FEED_STATION_M behind the tip
        when the hand holds there; else the node nearest the TCP)."""
        N = self.nodes()
        tip = N[-1]
        h = self.held(arm)
        k = None
        if h is not None and len(h) > 1:
            try:
                k = int(h[1])
            except (TypeError, ValueError):
                k = None
        if k is None or k >= len(N) - 1:
            k = int(np.argmin(np.linalg.norm(N - self.tcp(arm)[:3, 3], axis=1)))
        held_pt = 0.5 * (N[k] + N[min(k + 1, len(N) - 1)])
        return tip, _unit(tip - held_pt), held_pt

    def push(self, arm, p1, R, speed, d, step=0.006, lag_max=0.03):
        """A straight push of the held rope along d (IK every `step`), stopped when the tip lags the hand by more than
        lag_max along d (the rope is blocked: pressing on would crush it against a wall or ring). Returns the fraction."""
        p0 = self.tcp(arm)[:3, 3].copy()
        tip0 = self.nodes()[-1].copy()
        L = float(np.linalg.norm(p1 - p0))
        n = max(1, int(math.ceil(L / step)))
        self.forcing_now = False
        self._relaxed_noted = False
        qs = []
        for i in range(1, n + 1):
            p = p0 + (p1 - p0) * i / n
            q, ok, info = self.solve(arm, p, R)
            if not ok and info.get("why") == "collision" and i / n >= 0.5:
                # the capsule model (conservative by a few mm) stops the hand short of the ring: go on with the IK
                # check only; the Session's forcing evidence stops the push if the hand really presses (rr fan1 easy:
                # stopped at 75 %, camera 3.7 mm from A on the capsules)
                q, ok, info2 = self.solve(arm, p, R, check=False)
                if ok and not getattr(self, "_relaxed_noted", False):
                    self.event(arm=arm, event="push_relaxed", at=round(i / n, 3), info=info)
                    self._relaxed_noted = True
            if not ok:
                self.event(arm=arm, event="push_stop", at=round(i / n, 3), info=info)
                return (i - 1) / n
            qs.append(self.q[arm].copy())
            self.move_q(arm, q, L / n, speed)
            if self.forcing_now:
                self.event(arm=arm, event="push_forcing", at=round(i / n, 3), forcing=str(self.s.last_forcing)[:200])
                self.back_off(arm, qs)
                return (i - 1) / n
            if self.done or self.jam:
                return i / n
            if i % 4 == 0:
                lag = float((p - p0) @ d) - float((self.nodes()[-1] - tip0) @ d)
                if lag > lag_max:
                    self.event(arm=arm, event="push_blocked", at=round(i / n, 3), lag_m=round(lag, 4))
                    return i / n
        return 1.0

    def correct_lay(self, arm, steps, tol=0.01):
        """Closed loop at the end of a draw: the tip moved onto the planned lay point (the hand holds it
        RECEIVE_STATION_M behind the tip along the lay line) by a horizontal hand shift (rr9: an open-loop draw laid the
        tip 8 cm off and the rope over a post)."""
        p_last, p_prev = np.asarray(steps[-1]["pos"], float), np.asarray(steps[-2]["pos"], float)
        dl = p_last - p_prev
        dl[2] = 0.0
        if np.linalg.norm(dl) < 1e-6:
            return
        self.last_lay_dir = _unit(dl)
        self.pick_lay_dir = _unit(dl)            # the next station pick checks the rope runs from it to the tip this way
        exp_tip = p_last + S.RECEIVE_STATION_M * _unit(dl)
        exp_tip[2] = S.ROPE_Z_M + 0.002
        for it in range(2):
            self.wait_quiet(1.0, 0.03)
            tip = self.nodes()[-1]
            delta = exp_tip - tip
            # rr10: a draw that ended with the tip 5 cm up let the bent rope whip back 30 cm at the release: the tip is
            # brought down to the table as well (never the hand below its pick height)
            delta[2] = max(delta[2], RR_PICK_FLOOR - self.tcp(arm)[2, 3]) if delta[2] < 0 else 0.0
            err = float(np.linalg.norm(delta))
            self.event(arm=arm, event="lay_correction", it=it, tip_err_mm=round(1000 * err, 1))
            if err < tol or err > 0.15:
                return
            T = self.tcp(arm)
            frac = self.cart_line(arm, T[:3, 3] + delta, T[:3, :3], 0.04, step=0.008)
            if frac < 0.5:
                self.cart_line(arm, T[:3, 3] + delta, T[:3, :3], 0.04, step=0.008, check=False)

    def feed(self, arm, ring, tries=3):
        d = _unit(ring.travel)
        v = self.variant % len(FEED_VARIANTS)
        push_speed, extra = FEED_VARIANTS[v]
        # the push ends with the hand HAND_STOP before the plane: the tip goes (held station - HAND_STOP) past it
        station = self.held_sigma.get(arm) or S.FEED_STATION_M
        appr, poke = S.FEED_APPROACH_M, station - (S.FEED_STATION_M - S.POKE_M) + extra + (POKE_EXTRA_M if ring.mount != "bulkhead" else 0.0)
        target = ring.center - appr * d
        need = len(self.prefix()[1]) + 0      # crossings before this feed
        k_ring = [n for n, _ in self.design.required_crossings()].index(ring.name)
        saved_clear = (O.ARM_CLEAR, O.FINGER_CLEAR)
        O.ARM_CLEAR, O.FINGER_CLEAR = PUSH_CLEAR, PUSH_CLEAR      # alignment and push: 4 mm capsule clearance
        try:
            return self._feed(arm, ring, tries, d, v, push_speed, appr, poke, target, k_ring)
        finally:
            O.ARM_CLEAR, O.FINGER_CLEAR = saved_clear

    def _feed(self, arm, ring, tries, d, v, push_speed, appr, poke, target, k_ring):
        for attempt in range(tries):
            align = []
            best = None                                # (score, q) of the best alignment seen
            pos_only = False
            for it in range(ALIGN_ITERS):
                self.wait_quiet(2.0, 0.03)            # a swinging rope (0.37 m/s after the carry) is no measurement
                tip, dtip, _ = self.tip_state(arm)
                err = float(np.linalg.norm(tip - target))
                ang = math.degrees(math.acos(np.clip(dtip @ d, -1, 1)))
                score = err + 0.002 * ang
                align.append(dict(it=it, tip_err_m=round(err, 4), angle_deg=round(ang, 1)))
                if best is not None and score > best[0] + 0.004:
                    # the last correction made it worse (rr8 hard: 10.8 -> 29.5 deg): go back, translate only from now
                    self.move_q(arm, best[1], 0.02, 0.05)
                    self.step(6)
                    pos_only = True
                    align[-1]["reverted"] = True
                    continue
                if best is None or score < best[0]:
                    best = (score, self.q[arm].copy())
                if err < 0.006 and ang < 8.0:
                    break
                T = self.tcp(arm)
                Rc = np.eye(3) if pos_only else _rot_between(dtip, d)
                gain, max_step = (1.0, math.radians(15)) if v == 0 else (ALIGN_GAIN, ALIGN_MAX_STEP)
                rv = Rotation.from_matrix(Rc).as_rotvec() * gain
                ang_r = float(np.linalg.norm(rv))
                if ang_r > max_step:                  # damped: at most max_step per iteration (rr7's 15 deg on v0)
                    rv = rv * max_step / ang_r
                Rc = Rotation.from_rotvec(rv).as_matrix()
                R_new = Rc @ T[:3, :3]
                p_new = target - Rc @ (tip - T[:3, 3])
                q_before, f0 = self.q[arm].copy(), self.forcing_ticks
                ok, info = self.move_pose(arm, p_new, R_new, 0.05)
                if not ok:
                    # position only (keep the orientation)
                    p_new2 = T[:3, 3] + (target - tip)
                    ok, info = self.move_pose(arm, p_new2, T[:3, :3], 0.05)
                    align[-1]["fallback"] = info if not ok else "position_only"
                    if not ok:
                        break
                if FEED_FORCING_ABORT and self.forcing_ticks > f0:
                    # hard (inst5): the alignment at G3's window pressed the bulkhead's lintel / strips at 110-174 N:
                    # back to the joints before this correction, and this feed attempt is given up
                    self.back_off(arm, [q_before], k=1)
                    self.event(arm=arm, event="feed_forcing", ring=ring.name, attempt=attempt, it=it,
                               forcing=str(self.s.last_forcing)[:200])
                    return False
                self.step(6)
            if best is not None and align and (align[-1].get("reverted") or align[-1]["tip_err_m"] > 0.015):
                self.move_q(arm, best[1], 0.02, 0.05)
                self.step(6)
                tip, dtip, _ = self.tip_state(arm)
                align.append(dict(it="best", tip_err_m=round(float(np.linalg.norm(tip - target)), 4),
                                  angle_deg=round(math.degrees(math.acos(np.clip(dtip @ d, -1, 1))), 1)))
            if align and align[-1]["angle_deg"] > 60.0:
                # the held rope points away from the ring (the tip lies behind the hand): no push can thread it
                self.event(arm=arm, event="feed_reversed_tip", ring=ring.name, align=align)
                return False
            if align and (align[-1]["angle_deg"] > ALIGN_PUSH_MAX_DEG or align[-1]["tip_err_m"] > ALIGN_PUSH_MAX_ERR):
                # pushing a misaligned tip presses the rope into the wall / ring (rr7 hard: a jammed simulation)
                self.event(arm=arm, event="feed_misaligned", ring=ring.name, align=align)
                return False
            T = self.tcp(arm)
            saved = (O.ARM_CLEAR, O.FINGER_CLEAR)
            O.ARM_CLEAR, O.FINGER_CLEAR = PUSH_CLEAR, PUSH_CLEAR
            try:
                frac = self.push(arm, T[:3, 3] + (appr + poke) * d, T[:3, :3], push_speed, d)
            finally:
                O.ARM_CLEAR, O.FINGER_CLEAR = saved
            self.step(5)
            self.metrics(f"after feed {ring.name} (attempt {attempt})")
            n, seq = self.prefix()
            self.event(arm=arm, event="feed", ring=ring.name, attempt=attempt, variant=v, align=align, push_fraction=round(frac, 3),
                       prefix=n, sequence=seq)
            if n >= k_ring + 1:
                if ring.mount == "bulkhead" and k_ring < len(self.required) - 1:
                    self.inchworm(arm, ring, d, push_speed)
                return True
            # pull back and retry
            T = self.tcp(arm)
            f0 = self.forcing_ticks
            self.cart_line(arm, T[:3, 3] - (appr + poke) * frac * d, T[:3, :3], push_speed, check=False)
            self.step(10)
            if FEED_FORCING_ABORT and self.forcing_ticks > f0:
                self.event(arm=arm, event="feed_forcing", ring=ring.name, attempt=attempt, it="pull_back",
                           forcing=str(self.s.last_forcing)[:200])
                return False
        return False

    def inchworm(self, arm, ring, d, speed):
        """A wall hand-over needs the tip well past the wall (the receiver's fingers stop ~6.5 cm short of it: hard on inst5
        failed every G3 receive with the tip 5.6 cm through, succeeded with 8.7 cm): while the tip is less than
        MIN_PROTRUDE_M past the plane, open, slide back INCH_M along the rope, close and push INCH_M again."""
        for k in range(INCH_TRIES):
            tip = self.nodes()[-1]
            prot = float((tip - ring.center) @ d)
            if prot >= MIN_PROTRUDE_M or self.done:
                break
            T = self.tcp(arm)
            self.grip(arm, GRIP_OPEN, 8)
            self.cart_line(arm, T[:3, 3] - INCH_M * d, T[:3, :3], 0.03, check=False)
            self.grip(arm, GRIP_CLOSED, 10)
            if self.held(arm) is None:
                self.event(arm=arm, event="inchworm_lost", ring=ring.name, step=k, protrusion_m=round(prot, 4))
                break
            T = self.tcp(arm)
            self.push(arm, T[:3, 3] + INCH_M * d, T[:3, :3], speed, d)
            self.step(5)
            self.event(arm=arm, event="inchworm", ring=ring.name, step=k, protrusion_before_m=round(prot, 4),
                       protrusion_after_m=round(float((self.nodes()[-1] - ring.center) @ d), 4))

    # ------------------------------------------------------------------------------------------------ the run
    def dump_ckpt(self, i, name, last_steps, last_sigma, expected):
        """Branch point for fan-out attempts (main 10-11 12:50): the rows so far (one forward sequence) and the
        executor's state at the start of block i, so a fresh process can replay the rows and go on from span i."""
        d = self.out / "ckpt"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"span_{i:03d}.json"
        if f.exists():
            return
        np.savez(d / f"span_{i:03d}.npz", actions=np.asarray(self.rows, dtype=np.float32))
        lj = lambda v: None if v is None else np.asarray(v).tolist()
        f.write_text(json.dumps(dict(
            span=i, name=name, rows=len(self.rows), expected=expected, last_steps=last_steps, last_sigma=last_sigma,
            q={a: v.tolist() for a, v in self.q.items()}, g=dict(self.g), bias={a: v.tolist() for a, v in self.bias.items()},
            held_sigma=dict(self.held_sigma), pick_lay_dir=lj(self.pick_lay_dir), sequence=self.last_metrics.get("sequence")),
            default=str))

    def resume(self, ckpt_json):
        """Replay a branch point's rows open loop in this fresh Session and restore the executor state; returns the
        init dict for run(start_span=...)."""
        meta = json.loads(Path(ckpt_json).read_text())
        rows = np.load(str(ckpt_json)[:-5] + ".npz")["actions"].astype(np.float32)
        for r in rows:
            self.rows.append(r)
            self.done = bool(self.s.step(r))
            lf = getattr(self.s, "last_forcing", None)
            if lf is not None and lf is not self._forcing_seen:
                self._forcing_seen = lf
                info = lf[1] if isinstance(lf, tuple) else {}
                self.forcing_log.append(dict(row=len(self.rows) - 1, prefix=True, joint=info.get("joint"),
                                             force_n=round(float(info.get("force_n", 0.0)), 1)))
            if self.done:
                break
        self.prev_nodes = self.nodes()
        self.q = {a: np.asarray(v, float) for a, v in meta["q"].items()}
        self.g = {a: float(v) for a, v in meta["g"].items()}
        self.bias = {a: np.asarray(v, float) for a, v in meta["bias"].items()}
        self.held_sigma = {a: float(v) for a, v in meta["held_sigma"].items()}
        self.pick_lay_dir = None if meta.get("pick_lay_dir") is None else np.asarray(meta["pick_lay_dir"], float)
        self.metrics(f"resumed at span {meta['span']} ({meta['name']}) after {len(rows)} rows")
        return dict(start_span=int(meta["span"]), last_steps=meta["last_steps"], last_sigma=meta["last_sigma"],
                    expected=int(meta["expected"]))

    def run(self, stop_after_phase=None, hold_s=20.0, max_variants=4, init=None, start_variant=0, span_variants=None):
        """Execute the timeline's spans. Blocks start at every grasping approach and every feed: a complete checkpoint
        is taken there. A failed grasp, a failed feed or a ring lost (the task's own crossing prefix drops below the
        rings fed so far, after the rope settled) rewinds to the latest block start with variants left and retries it
        with the next variant (grasp: target shifted along the rope; feed: push speed / depth). The rows of the
        discarded branches are dropped, so the saved rows are one forward sequence."""
        import copy
        spans = self.tl["spans"]
        last_steps = {"left": None, "right": None}
        last_sigma = {"left": None, "right": None}
        expected = 0
        cps, tries = {}, {}
        i = 0
        if init is None:
            self.step(5)
            self.metrics("start")
        else:
            i = init["start_span"]
            last_steps, last_sigma, expected = init["last_steps"], init["last_sigma"], init["expected"]
            tries[i] = start_variant           # the first block of a branch starts at this variant
        stop_reason = None
        while i < len(spans):
            if self.done:
                break
            if stop_after_phase is not None and i > stop_after_phase:
                stop_reason = "stop_after_phase"
                break
            sp = spans[i]
            arm = sp["hand"]
            name = sp["name"]
            block_start = bool((sp.get("sigma") is not None and sp.get("steps")) or (sp.get("kind") == "feed" and sp.get("ring")))
            if block_start and i not in cps:
                if span_variants and i in span_variants and i not in tries:
                    tries[i] = span_variants[i]     # a block known (from branch searches) to need this variant first
                cps[i] = (self.checkpoint(f"{i} {name}"),
                          dict(last_steps=copy.deepcopy(last_steps), last_sigma=dict(last_sigma), expected=expected))
                if DUMP_CKPT:
                    self.dump_ckpt(i, name, last_steps, last_sigma, expected)
            blk = max([k for k in cps if k <= i], default=None)
            self.variant = tries.get(blk, 0) if blk is not None else 0
            fail = None
            if sp.get("gripper") == "close":
                ok = self.grasp_at(arm, last_sigma[arm], last_steps[arm]) if last_steps[arm] else False
                self.metrics(f"{i} {name} ({arm}) ok={ok} v{self.variant}")
                if not ok:
                    fail = "grasp"
            elif sp.get("gripper") == "open":
                same_hand = name.startswith("feeder lets go in front of")
                if same_hand and SAMEHAND_LOWER_M > 0:
                    # before a same-hand feeder lets go: lower the hand (and the rope behind the ring) towards the
                    # table, so the rope behind it lies down instead of springing back (hard / medium T retraction)
                    T = self.tcp(arm)
                    dz = min(SAMEHAND_LOWER_M, max(0.0, float(T[2, 3]) - 0.02))
                    if dz > 0.002:
                        self.cart_line(arm, T[:3, 3] - np.array([0.0, 0.0, dz]), T[:3, :3], 0.02, step=0.005,
                                       check=False)
                        self.event(arm=arm, event="samehand_lowered", dz=round(dz, 3))
                self.release(arm, slow_rows=SAMEHAND_RELEASE_ROWS if same_hand else (LAY_RELEASE_ROWS if name.startswith("lay down") else 0))
                self.metrics(f"{i} {name} ({arm})")
                if name.startswith("lay down") and self.last_lay_dir is not None:
                    N = self.nodes()
                    td = N[-1] - N[-20]
                    td[2] = 0.0
                    ang = math.degrees(math.acos(np.clip(_unit(td) @ self.last_lay_dir, -1, 1)))
                    self.check_lay_dir = self.last_lay_dir          # checked again after the lift-off
                    self.last_lay_dir = None
                    if ang > LAY_MAX_DEG:
                        # rr13 medium: the laid tip swung 70 deg off the lay line; the pick 20 cm behind it then misses
                        fail = f"laid tip {ang:.0f} deg off the lay line"

            elif sp.get("kind") == "feed" and sp.get("ring"):
                ok = self.feed(arm, self.rings[sp["ring"]])
                if ok:
                    expected = [n for n, _ in self.design.required_crossings()].index(sp["ring"]) + 1
                else:
                    fail = "feed"
            elif sp.get("kind") == "hold":
                stop_reason = "hold"
                break
            elif sp.get("steps"):
                steps = sp["steps"]
                if sp.get("sigma") is not None:
                    # an approach that ends in a grasp: its last poses are executed (re-aimed) by the grasp phase
                    pre = steps[:-1]
                    if len(pre) > 2:
                        self.follow(arm, pre[:-2])
                    last_steps[arm] = steps[-3:] if len(steps) >= 3 else steps
                    last_sigma[arm] = float(sp["sigma"])
                elif sp.get("kind") == "retreat" and name.startswith("lift off") and self.g[arm] >= GRIP_OPEN - 1e-4:
                    # leave the rope straight up first (a joint-space lift swept the open fingers through the laid
                    # rope), then the planned retreat from there
                    if LIFT_WAIT_S > 0:
                        self.step(int(LIFT_WAIT_S / DT))
                    T = self.tcp(arm)
                    up = T[:3, 3] + np.array([0.0, 0.0, 0.07])
                    self.cart_line(arm, up, T[:3, :3], LIFT_SPEED, step=0.01, check=False)
                    self.wait_quiet(2.0, 0.01)
                    self.metrics(f"{i} after lifting off ({name})")
                    if self.check_lay_dir is not None:
                        N = self.nodes()
                        td = N[-1] - N[-20]
                        td[2] = 0.0
                        ang = math.degrees(math.acos(np.clip(_unit(td) @ self.check_lay_dir, -1, 1)))
                        self.check_lay_dir = None
                        if ang > LAY_MAX_DEG:
                            # rr19 medium: the laid rope whipped back 38 cm while the hand lifted off it
                            fail = f"laid tip {ang:.0f} deg off the lay line after the lift-off"

                    if len(steps) > 1:
                        self.follow(arm, steps[1:])
                else:
                    self.follow(arm, steps)
                    if name.startswith("draw tip to") and len(steps) >= 2:
                        self.correct_lay(arm, steps)
                    if sp.get("kind") in ("transport",):
                        self.metrics(f"{i} after {name}")
            if fail is None and self.jam:
                fail = "physics jam"
            if fail is None and expected and sp.get("gripper") != "close" and sp.get("kind") != "approach":
                n, seq = self.prefix()
                if n < expected:
                    self.wait_quiet(1.5)
                    self.metrics(f"{i} recheck after {name}")
                    n, seq = self.prefix()
                    if n < expected:
                        fail = f"ring lost ({n} < {expected}: {seq})"
            if fail is not None and not self.done:
                cand = [k for k in cps if k <= i and tries.get(k, 0) < max_variants - 1]
                if not cand:
                    self.event(arm=arm, event="give_up", phase=i, why=fail)
                    stop_reason = f"gave up at span {i}: {fail}"
                    break
                j = max(cand)
                tries[j] = tries.get(j, 0) + 1
                cp, ex = cps[j]
                self.rewind(cp, f"span {i} ({name}): {fail}; variant {tries[j]} of span {j}")
                last_steps, last_sigma, expected = copy.deepcopy(ex["last_steps"]), dict(ex["last_sigma"]), ex["expected"]
                for k in [k for k in cps if k > j]:
                    del cps[k]
                    tries.pop(k, None)
                i = j
                continue
            i += 1
        self.stop_reason = stop_reason
        # hold until the episode ends (success) or hold_s passes
        n_hold = int(hold_s / DT)
        for k in range(n_hold):
            if self.done:
                break
            self.step()
            if k % 25 == 24:
                self.metrics(f"hold {round((k + 1) * DT, 1)} s")
        if not self.done and stop_reason == "hold" and LET_GO_S > 0:
            # rr16 easier: the whole sequence held, but the rope did not settle under the holding hand (contact chatter
            # at the eyelet, 4-23 mm/s for 20 s): let go, lift the hand clear, wait (letting go is allowed)
            for arm in ("left", "right"):
                if self.held(arm) is not None or self.g[arm] < GRIP_OPEN - 1e-4:
                    self.grip(arm, GRIP_OPEN, 6)
                    self.event(arm=arm, event="final_release")
                    T = self.tcp(arm)
                    self.cart_line(arm, T[:3, 3] + np.array([0.0, 0.0, 0.06]), T[:3, :3], 0.04, step=0.01, check=False)
            for k in range(int(LET_GO_S / DT)):
                if self.done:
                    break
                self.step()
                if k % 25 == 24:
                    self.metrics(f"after letting go {round((k + 1) * DT, 1)} s")
        final = self.metrics("final")
        st = self.s.status()
        rows = np.asarray(self.rows, dtype=np.float32)
        prev_rows = np.vstack([self.home_row[None], rows[:-1]]).astype(np.float64)
        steps = np.abs(rows.astype(np.float64) - prev_rows)[:, ARM_COLS].max(1) if len(rows) else np.zeros(0)
        step_census = dict(limit_rad=0.04, over_limit_rows=int((steps > 0.04 + 1e-6).sum()),
                           max_step_rad=round(float(steps.max()) if len(steps) else 0.0, 5))
        np.savez(self.out / "rows.npz", actions=rows, action_layout=np.array(self.s.action_layout))
        rep = dict(tier=self.tier, success=bool(st.get("success")), episode_end=st.get("episode_end"), rows=int(len(rows)),
                   stop_reason=self.stop_reason, rewinds=self.rewinds, forcing_ticks_seen=self.forcing_ticks, paced_rows=self.paced_rows, step_census=step_census, applied_changed_rows=self.applied_changed, forcing_kept=dict(ticks=len(self.forcing_log), max_force_n=max([f.get('force_n', 0.0) for f in self.forcing_log], default=0.0), first=self.forcing_log[:20]),
                   sim_time_s=round(len(rows) * DT, 2), required=self.required, final=final,
                   final_speed_m_s=round(self.speed, 5), events=self.events, review=st.get("review"),
                   forcing=st.get("forcing"), wall_s=round(time.time() - self.t0, 1), state_sha=st.get("state_sha"),
                   log=self.log)
        (self.out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
        self.s.close()
        return rep


if __name__ == "__main__":
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1)          # kill -USR1 <pid>: dump the stack into the log
    ap = argparse.ArgumentParser()
    ap.add_argument("tier")
    ap.add_argument("instance")
    ap.add_argument("timeline")
    ap.add_argument("out")
    ap.add_argument("--stop-after-phase", type=int, default=None)
    ap.add_argument("--hold-s", type=float, default=20.0)
    ap.add_argument("--no-bias", action="store_true", help="no load-bias compensation of the commanded joints")
    ap.add_argument("--max-variants", type=int, default=4)
    ap.add_argument("--resume", default=None, help="a branch point (ckpt/span_NNN.json) to replay and go on from")
    ap.add_argument("--start-variant", type=int, default=0, help="the variant the branch's first block starts with")
    ap.add_argument("--set", action="append", default=[], help="NAME=VALUE: override a module constant of this executor")
    ap.add_argument("--span-variant", action="append", default=[], help="SPAN=VARIANT: the first variant of that block")
    a = ap.parse_args()
    if a.no_bias:
        BIAS = False
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k not in globals():
            raise SystemExit(f"unknown constant {k}")
        globals()[k] = type(globals()[k])(float(v)) if not isinstance(globals()[k], bool) else v.lower() in ("1", "true")
    rr = RobotRelay(a.tier, a.instance, a.timeline, a.out)
    init = rr.resume(a.resume) if a.resume else None
    sv = {int(k): int(v) for k, v in (x.split("=") for x in a.span_variant)}
    rep = rr.run(a.stop_after_phase, a.hold_s, a.max_variants, init=init, start_variant=a.start_variant,
                 span_variants=sv)
    rep["span_variants"] = sv
    rep["overrides"] = a.set
    rep["resumed_from"] = a.resume
    (Path(a.out) / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("log", "events")}, default=str))
