"""The gear-train oracle on a clock_assembling Session (closed loop, privileged): the design's oracle
(runs/watch_assembly_design_20261014/scripts/v3/l1_sim.py, its L3 controller l3_phys.py) on topogym's control stack.

It plans every move from the measured state (part poses, the task mechanism's rules) and commands only joint and gripper
targets, one row per 40 Hz control step:
  - a TCP pose per hand per control step (straight lines and slerps, or a probe tip on a straight line), turned into arm
    joint targets by damped least-squares IK on a scratch copy of the world's model (the arm's own branch, re-seeded when a
    joint would leave its range); where the branch must change, the arm changes it in joint space high above the scene;
  - each target plus feedforwards of the drive's velocity lag (joint damping / kp), its Coulomb friction (friction loss /
    kp, toward the remaining joint error) and the held part's weight; every arm joint's command moves at most 0.035 rad per
    control step (the standard mode's 0.04 rad limit never splits a row);
  - grasps by closing the gripper and waiting until the assisted grasp holds the part; presses with the open fingertips.
Frames: the oracle works in the robot mount frame M (the design's frame: table top z = 0); the world is M + the mount."""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

from topogym.assets.geometry import clock_gear_train as geo

EX, EY, EZ = np.eye(3)
DOWN = -EZ
HANDS = {"L": "left", "R": "right"}
ARM_J = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
GRIP_OPEN, GRIP_CLOSED = .037, .002
STEP_MAX = .0349                # rad: an arm joint's command change per control step (float32 rounding stays under 0.035)
MARGIN = .15                    # rad kept from every joint limit by the IK
GRASP_Z = {"barrel": None, "center": .038, "third": .039, "fourth": .042}   # pad contact height on the arbor (W)
SCREW_YAW = {"screw1": -3.141592653589793, "screw2": -.5235987755982991}    # design L2: wrist room for the strokes
PRESS_FORCE = 4.                # N, the open fingertips on the bridge
NUDGE_FORCE = 3.                # N, the probe on a wheel
LAG_HOLD = .05                  # rad: a move waits while an arm's commands lag its IK solution by more


def T(p, R):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, p
    return M


def Rcols(x, y):
    x = np.asarray(x, float) / np.linalg.norm(x)
    y = np.asarray(y, float) - np.dot(y, x) * x
    y /= np.linalg.norm(y)
    return np.c_[x, y, np.cross(x, y)]


def rotvec(Rm):
    return Rot.from_matrix(Rm).as_rotvec()


class Kin:
    """IK of both arms on a private MjData of the world's model (the TCP: site '<arm>/gripper', in M)."""

    def __init__(self, m, mount, q_birth):
        import mujoco
        self.mj, self.m, self.d = mujoco, m, mujoco.MjData(m)
        self.mount = np.asarray(mount, float)
        J = mujoco.mjtObj.mjOBJ_JOINT
        jid = {h: [mujoco.mj_name2id(m, J, f"{s}/{j}") for j in ARM_J] for h, s in HANDS.items()}
        self.qadr = {h: np.array([m.jnt_qposadr[j] for j in ids]) for h, ids in jid.items()}
        self.dadr = {h: np.array([m.jnt_dofadr[j] for j in ids]) for h, ids in jid.items()}
        self.lim = {h: np.array([m.jnt_range[j] for j in ids]) for h, ids in jid.items()}
        self.site = {h: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{s}/gripper") for h, s in HANDS.items()}
        self.q_home = {h: np.asarray(q_birth[h], float).copy() for h in HANDS}
        self._jp, self._jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self.rng = np.random.default_rng(12345)

    def _pose(self, h):
        R = self.d.site_xmat[self.site[h]].reshape(3, 3)
        return T(self.d.site_xpos[self.site[h]] - self.mount, R.copy())

    def fk(self, h, q):
        self.d.qpos[self.qadr[h]] = q
        self.mj.mj_kinematics(self.m, self.d)
        return self._pose(h)

    def ik(self, h, Tt, q0, iters=150):
        lo, hi = self.lim[h][:, 0] + MARGIN, self.lim[h][:, 1] - MARGIN
        q = np.clip(np.asarray(q0, float), lo, hi)
        da, sid = self.dadr[h], self.site[h]
        target = Tt[:3, 3] + self.mount
        ep = er = np.ones(3)
        for _ in range(iters):
            self.d.qpos[self.qadr[h]] = q
            self.mj.mj_kinematics(self.m, self.d)
            self.mj.mj_comPos(self.m, self.d)
            ep = target - self.d.site_xpos[sid]
            er = rotvec(Tt[:3, :3] @ self.d.site_xmat[sid].reshape(3, 3).T)
            if np.linalg.norm(ep) < 2e-5 and np.linalg.norm(er) < 2e-4:
                break
            self.mj.mj_jacSite(self.m, self.d, self._jp, self._jr, sid)
            Jm = np.r_[self._jp[:, da], self._jr[:, da]]
            dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + 1e-5 * np.eye(6), np.r_[ep, er])
            q = np.clip(q + np.clip(dq, -.2, .2), lo, hi)
        return q, float(np.linalg.norm(ep)), float(np.linalg.norm(er))

    def ik_near(self, h, Tt, q_ref, q0=None, iters=30, pos_tol=.001, rot_tol=.02, mu=.01):
        """IK that stays near q_ref (pose error in units of pos_tol / rot_tol + mu |q - q_ref|^2)."""
        lo, hi = self.lim[h][:, 0] + MARGIN, self.lim[h][:, 1] - MARGIN
        q_ref = np.asarray(q_ref, float)
        q = np.clip(np.asarray(q_ref if q0 is None else q0, float), lo, hi)
        da, sid = self.dadr[h], self.site[h]
        target = Tt[:3, 3] + self.mount
        wgt = np.r_[np.full(3, 1 / pos_tol), np.full(3, 1 / rot_tol)]
        for _ in range(iters):
            self.d.qpos[self.qadr[h]] = q
            self.mj.mj_kinematics(self.m, self.d)
            self.mj.mj_comPos(self.m, self.d)
            e = np.r_[target - self.d.site_xpos[sid], rotvec(Tt[:3, :3] @ self.d.site_xmat[sid].reshape(3, 3).T)]
            self.mj.mj_jacSite(self.m, self.d, self._jp, self._jr, sid)
            Jm = wgt[:, None] * np.r_[self._jp[:, da], self._jr[:, da]]
            dq = np.linalg.solve(Jm.T @ Jm + (mu + 1e-6) * np.eye(6), Jm.T @ (wgt * e) + mu * (q_ref - q))
            q = np.clip(q + np.clip(dq, -.2, .2), lo, hi)
            if np.abs(dq).max() < 1e-6:
                break
        self.d.qpos[self.qadr[h]] = q
        self.mj.mj_kinematics(self.m, self.d)
        ep = target - self.d.site_xpos[sid]
        er = rotvec(Tt[:3, :3] @ self.d.site_xmat[sid].reshape(3, 3).T)
        return q, float(np.linalg.norm(ep)), float(np.linalg.norm(er))

    def ik_all(self, h, Tt, n=16):
        """Distinct IK solutions (> 0.3 rad apart) from home and n seeded random starts, each with its joint margin."""
        lo, hi = self.lim[h][:, 0] + MARGIN, self.lim[h][:, 1] - MARGIN
        sols = []
        for q0 in [self.q_home[h]] + [self.rng.uniform(lo, hi) for _ in range(n)]:
            q, ep, er = self.ik(h, Tt, q0, iters=200)
            if ep < 1e-3 and er < 1e-2 and all(np.abs(q - o).max() > .3 for o, _ in sols):
                sols.append((q, self.margin(h, q)))
        return sols

    def margin(self, h, q):
        return float(np.minimum(q - self.lim[h][:, 0], self.lim[h][:, 1] - q).min())

    def reach(self, h, Tt, seeds=()):
        for q0 in (self.q_home[h],) + tuple(seeds):
            q, ep, er = self.ik(h, Tt, q0)
            if ep < 1e-3 and er < 1e-2:
                return True
        return False


class RulesView:
    """The task mechanism's rules (topogym.tasks.clock_assembling.mechanism) read on the live data, in W (the movement
    frame; the mechanism measures from the movement's origin in the world)."""

    def __init__(self, rules, data):
        self.r, self.d = rules, data

    seated = property(lambda self: self.r.seated)
    gate_closed = property(lambda self: self.r.gate_closed)
    tight = property(lambda self: self.r.tight)

    def entered(self, n):
        return self.r.entered(self.d, n)

    def tip_offset(self, n):
        return self.r.tip_offset(self.d, n)

    def wheel_local(self, n):
        p, R = self.r.wheel_local(self.d, n)
        return p, Rot.from_matrix(R)

    def bridge_seat_error(self):
        return self.r.bridge_seat_error(self.d)


class ClockOracle:
    SAFE_Z = .22                  # M: travel height between places (above the barrel arbor, the bridge and the stands)
    PARK = {"L": np.array([-.22, .02, .22]) + np.r_[geo.OFFSET, 0.], "R": np.array([.22, .02, .22]) + np.r_[geo.OFFSET, 0.]}
    PROBE_CLEAR_Z = geo.H_BRIDGE + geo.T_BRIDGE + .007 + .008   # W: the probe tip's travel height

    def __init__(self, session, log=print):
        import mujoco
        self.mj, self.s, self.log = mujoco, session, log
        self.w = session.env._world
        self.m, self.d = self.w.model, self.w.data
        task = session.env._task
        self.mount = np.asarray(task.origin(), float)
        self.P_W = geo.P_W                                  # M
        self.tick_s = 1. / session.control_hz
        self.rules = RulesView(self.w._mechanism, self.d)
        layout = session.action_layout
        self.ia = {h: [layout.index(f"{s}/{j}") for j in ARM_J] for h, s in HANDS.items()}
        self.ig = {h: layout.index(f"{s}_native_gripper") for h, s in HANDS.items()}
        self.target = np.asarray(session._targets, float).copy()
        q_birth = {h: self.target[self.ia[h]].copy() for h in HANDS}
        self.kin = Kin(self.m, self.mount, q_birth)
        J = mujoco.mjtObj.mjOBJ_JOINT
        self.qadr = self.kin.qadr
        self.dadr = self.kin.dadr
        act = {h: [next(a for a in range(self.m.nu) if self.m.actuator_trnid[a, 0] == mujoco.mj_name2id(self.m, J, f"{s}/{j}"))
                   for j in ARM_J] for h, s in HANDS.items()}
        self.kp = {h: np.array([self.m.actuator_gainprm[a, 0] for a in act[h]]) for h in HANDS}
        self.damp = {h: self.m.dof_damping[self.dadr[h]].copy() for h in HANDS}
        self.fric = {h: self.m.dof_frictionloss[self.dadr[h]].copy() for h in HANDS}
        self.bid = {p: self.w._mechanism.b[p] for p in geo.PARTS}
        self.bid["barrel_arbor"] = self.w._mechanism.b_arbor        # the barrel is held by its arbor's square
        self.robot_bodies = {h: {b for b in range(self.m.nbody)
                                 if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(s + "/")}
                             for h, s in HANDS.items()}
        self.grip = {h: float(self.target[self.ig[h]]) for h in HANDS}
        self.q_sol = {h: q_birth[h].copy() for h in HANDS}
        self.q_prev = {h: q_birth[h].copy() for h in HANDS}
        self.pose = {h: self.kin.fk(h, q_birth[h]) for h in HANDS}
        self.joint_mode = {h: None for h in HANDS}          # a joint-space move's remaining waypoints
        self.held = {"L": None, "R": None}
        self.pressing = False
        self.ended = False
        self.steps, self.phase, self.events = [], "", []
        self.stat = dict(ik_fail=0, reseeds=0, switches=0, worst_ik_m=0., settle_steps=0, worst_cmd_step=0., lag_steps=0)
        self.probe_G = self.bridge_G = None
        self.kept = None
        self.track = {"L": [], "R": []}

    # ---------------------------------------------------------------- state readouts (M frame)
    def part_T(self, p):
        b = self.bid[p]
        return T(np.asarray(self.d.xpos[b]) - self.mount, np.asarray(self.d.xmat[b]).reshape(3, 3).copy())

    def tcp_actual(self, h):
        return T(np.asarray(self.d.site_xpos[self.kin.site[h]]) - self.mount,
                 np.asarray(self.d.site_xmat[self.kin.site[h]]).reshape(3, 3).copy())

    def held_part(self, h):
        b = self.w._grasp_manager.hands[HANDS[h]]["binding"]
        return None if b is None else tuple(b["target"])[1]

    def q_meas(self, h):
        return np.asarray(self.d.qpos[self.qadr[h]], float).copy()

    def _forces(self, bodies_a, bodies_b):
        f6, tot = np.zeros(6), 0.
        for i in range(int(self.d.ncon)):
            c = self.d.contact[i]
            b1, b2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
            if (b1 in bodies_a and b2 in bodies_b) or (b2 in bodies_a and b1 in bodies_b):
                self.mj.mj_contactForce(self.m, self.d, i, f6)
                tot += abs(f6[0])
        return tot

    def hand_force_on(self, h, part):
        return self._forces(self.robot_bodies[h], {self.bid[part]})

    def probe_push(self):
        """Contact force on the probe from everything but the robot (N)."""
        pb = self.bid["probe"]
        f6, tot = np.zeros(6), 0.
        robot = self.robot_bodies["L"] | self.robot_bodies["R"]
        for i in range(int(self.d.ncon)):
            c = self.d.contact[i]
            b1, b2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
            if pb in (b1, b2) and (b2 if b1 == pb else b1) not in robot:
                self.mj.mj_contactForce(self.m, self.d, i, f6)
                tot += abs(f6[0])
        return tot

    # ---------------------------------------------------------------- the control step
    def payload_tau(self, h):
        p = self.held[h]
        if p is None:
            return np.zeros(6)
        if p == "barrel_arbor":      # carrying the barrel by its square (the mainspring locked); winding: no payload
            if self.w._mechanism.fixed:
                return np.zeros(6)
            p = "barrel"
        b = self.bid[p]
        mass = float(self.m.body_subtreemass[b])
        jp = np.zeros((3, self.m.nv))
        self.mj.mj_jac(self.m, self.d, jp, None, np.asarray(self.d.subtree_com[b]), self.mj.mj_name2id(
            self.m, self.mj.mjtObj.mjOBJ_BODY, f"{HANDS[h]}/gripper_link"))
        return -(jp[:, self.dadr[h]].T @ (mass * self.m.opt.gravity))

    def solve(self, h):
        """The arm's joint solution for its commanded pose (continuing its branch)."""
        if self.joint_mode[h]:
            q = self.joint_mode[h].pop(0)
            self.pose[h] = self.kin.fk(h, q)
            return q
        q, ep, er = self.kin.ik_near(h, self.pose[h], self.q_sol[h], q0=self.q_sol[h])
        if ep > .002 or er > .03:                      # more iterations on the same branch first
            q, ep, er = self.kin.ik(h, self.pose[h], self.q_sol[h])
        if ep > .002 or er > .03:
            q2 = self.reseed(h)
            if q2 is not None:
                q, ep = q2, 0.
        if ep > .001:
            self.stat["ik_fail"] += 1
        self.stat["worst_ik_m"] = max(self.stat["worst_ik_m"], ep)
        return q

    def reseed(self, h):
        """The branch followed so far cannot reach the pose (a joint at its range): the reachable solution nearest to it."""
        self.stat["reseeds"] += 1
        best = None
        for q0 in [self.q_sol[h], self.kin.q_home[h]] + [q for q, _ in self.kin.ik_all(h, self.pose[h], n=12)]:
            q, ep, er = self.kin.ik(h, self.pose[h], q0)
            if ep < 1e-3 and er < 1e-2:
                key = float(np.max(np.abs(q - self.q_sol[h])))
                if best is None or key < best[0]:
                    best = (key, q)
        self.note(f"{h} arm re-seeded its IK in '{self.phase}' "
                  f"({'none reachable' if best is None else f'{best[0]:.2f} rad away'}; TCP target {np.round(self.pose[h][:3, 3], 3).tolist()})")
        return None if best is None else best[1]

    def tick(self):
        """One control step: both arms' joint targets (with feedforward, rate-limited), both grippers."""
        if self.ended:
            return True
        a = self.target.copy()
        for h in "LR":
            q = self.solve(h)
            qd = (q - self.q_sol[h]) / self.tick_s
            self.q_sol[h] = q
            tau = self.damp[h] * qd + self.payload_tau(h)
            tau += self.fric[h] * np.tanh((q - self.q_meas(h)) / .002)
            cmd = q + tau / self.kp[h]
            step = np.clip(cmd - self.q_prev[h], -STEP_MAX, STEP_MAX)
            self.stat["worst_cmd_step"] = max(self.stat["worst_cmd_step"], float(np.abs(step).max()))
            self.q_prev[h] = self.q_prev[h] + step
            a[self.ia[h]] = self.q_prev[h]
            a[self.ig[h]] = self.grip[h]
        a = a.astype(np.float32).astype(np.float64)
        self.q_prev = {h: a[self.ia[h]].copy() for h in "LR"}
        self.target = a
        done = self.s.step(a)
        for h in "LR":
            self.held[h] = self.held_part(h)
            if not (self.pressing and h == "R"):      # how far the TCP is from where the oracle commands it
                err = float(np.linalg.norm(self.pose[h][:3, 3] - self.tcp_actual(h)[:3, 3]))
                self.track[h].append((err, self.held[h] is not None, self.phase))
        if done:
            self.ended = True
            self.log(f"episode end at control step {self.s.n_steps}: {self.s.episode_end}")
        return done

    def lag(self, h):
        return float(np.max(np.abs(self.q_sol[h] - self.q_prev[h])))

    def settle(self, s_max=3.):
        """Wait until both arms' commands have caught up with their solutions."""
        for _ in range(int(s_max / self.tick_s)):
            if all(self.lag(h) < .02 and not self.joint_mode[h] for h in "LR"):
                break
            self.stat["settle_steps"] += 1
            if self.tick():
                break

    def wait(self, s):
        for _ in range(max(1, int(round(s / self.tick_s)))):
            if self.tick():
                break

    def step_name(self, name):
        self.phase = name
        self.steps.append((self.s.n_steps, name))
        self.log(f"[{self.s.n_steps:6d}] {name}")

    def note(self, text):
        self.events.append((self.s.n_steps, text))
        self.log(f"[{self.s.n_steps:6d}]   {text}")

    # ---------------------------------------------------------------- motions
    def move(self, h, target, v=.08, w=1.5, until=None, tool=None):
        """Hand h to pose target: the point tool (hand frame; default the TCP) on a straight line at v m/s, the orientation
        by slerp at w rad/s; then wait for the arm to catch up."""
        A = self.pose[h].copy()
        t = np.zeros(3) if tool is None else np.asarray(tool, float)
        pa, pb = A[:3, :3] @ t + A[:3, 3], target[:3, :3] @ t + target[:3, 3]
        dp = np.linalg.norm(pb - pa)
        ang = (Rot.from_matrix(target[:3, :3]) * Rot.from_matrix(A[:3, :3]).inv()).magnitude()
        n = max(1, int(math.ceil(max(dp / v, ang / w) / self.tick_s)))
        sl = Slerp([0, 1], Rot.from_matrix([A[:3, :3], target[:3, :3]]))
        for k in range(1, n + 1):
            u = k / n
            u = u * u * (3 - 2 * u)
            R = sl(u).as_matrix()
            self.pose[h] = T(pa + u * (pb - pa) - R @ t, R)
            if self.tick():
                return True
            if until is not None and until():
                return True
            for _ in range(int(2. / self.tick_s)):        # the arm's commands lag (their rate limit): hold the pose
                if self.lag(h) < LAG_HOLD:
                    break
                self.stat["lag_steps"] += 1
                if self.tick():
                    return True
        self.settle()
        return False

    def turn(self, h, angle, w=1.5, until=None, axis=EZ):
        k = max(1, int(math.ceil(abs(angle) / math.radians(80))))
        for _ in range(k):
            M = self.pose[h].copy()
            M[:3, :3] = Rot.from_rotvec(np.asarray(axis, float) / np.linalg.norm(axis) * angle / k).as_matrix() @ M[:3, :3]
            if self.move(h, M, v=.01, w=w, until=until):
                return True
        return False

    def up(self, M, dz):
        N = M.copy()
        N[2, 3] += dz
        return N

    def carry_z(self, h):
        """Travel height of hand h's TCP: SAFE_Z; the barrel hangs 8 cm under the hand; a held wheel's lower pivot passes
        2 cm over the barrel's square."""
        p = self.held[h]
        if p in ("barrel", "barrel_arbor"):
            return self.SAFE_Z + .035
        if p in GRASP_Z and GRASP_Z[p] is not None:
            hang = GRASP_Z[p] + .013 + geo.LOW_PIVOT_L
            return max(self.SAFE_Z, self.P_W[2] + geo.BARREL_SQUARE_TOP + .02 + hang)
        return self.SAFE_Z

    def _track(self, h, q0, poses):
        """IK continuity along poses from q0: (ok, q at the end)."""
        q = np.asarray(q0, float)
        for P in poses:
            q1, ep, er = self.kin.ik(h, P, q, iters=60)
            if ep > 1e-3 or er > 1e-2 or np.abs(q1 - q).max() > .3:     # unreachable, or a jump to another branch
                return False, q1
            q = q1
        return True, q

    def _line(self, A, B, n):
        sl = Slerp([0, 1], Rot.from_matrix([A[:3, :3], B[:3, :3]]))
        return [T(A[:3, 3] + u * (B[:3, 3] - A[:3, 3]), sl(u).as_matrix()) for u in np.linspace(0, 1, n + 1)[1:]]

    def joint_move(self, h, q_goal):
        """Hand h's arm to q_goal in joint space (high above the scene), at most STEP_MAX per control step."""
        q0 = self.q_sol[h]
        n = max(1, int(math.ceil(np.abs(q_goal - q0).max() / (.8 * STEP_MAX))))
        path = []
        for k in range(1, n + 1):
            u = k / n
            u = u * u * (3 - 2 * u)
            path.append(q0 + u * (q_goal - q0))
        self.joint_mode[h] = path
        self.stat["switches"] += 1
        while self.joint_mode[h]:
            if self.tick():
                return True
        self.settle()
        return False

    def go(self, h, target, v=.15, v_end=.04):
        """Long move: up to the travel height, across, down to target (the last few cm at v_end). When the arm's present
        branch cannot follow the way across and down, it changes branch in joint space at the top, choosing the solution
        whose way down is continuous, nearest the present one."""
        cur = self.pose[h]
        z_safe = self.carry_z(h)
        if cur[2, 3] < z_safe:
            if self.move(h, self.up(cur, z_safe - cur[2, 3]), v=.1):
                return True
        hi = target.copy()
        hi[2, 3] = max(z_safe, target[2, 3])
        across = self._line(self.pose[h], hi, 40)
        down = self._line(hi, target, 20)
        ok, _ = self._track(h, self.q_sol[h], across + down)
        if not ok:
            best = None
            for q, mg in self.kin.ik_all(h, hi, n=24):
                good, _ = self._track(h, q, down)
                if good:
                    key = (float(np.max(np.abs(q - self.q_sol[h]))), -mg)
                    if best is None or key < best[0]:
                        best = (key, q)
            if best is not None:
                self.note(f"{h} arm changes branch at the top ({best[0][0]:.2f} rad)")
                if self.joint_move(h, best[1]):
                    return True
                self.pose[h] = hi.copy()
        if self.move(h, hi, v=v):
            return True
        return self.move(h, target, v=v_end)

    def grasp(self, h, part):
        self.grip[h] = GRIP_CLOSED
        for _ in range(int(1.5 / self.tick_s)):
            if self.tick():
                return
            if self.held[h] == part:
                break
        self.wait(.15)
        if self.held[h] != part:
            self.note(f"{h} hand failed to take the {part} (holds {self.held[h]})")

    def release(self, h):
        self.grip[h] = GRIP_OPEN
        self.wait(.35)

    # ---------------------------------------------------------------- the oracle's pieces (the design's l1_sim.Sim)
    def wheel_grasp_z(self, n):
        if n == "barrel":
            return (geo.SQUARE_BOTTOM + geo.BARREL_SQUARE_TOP) / 2
        return GRASP_Z[n]

    def place_wheel(self, n):
        self.step_name(f"place {n}")
        h, W = "L", self.P_W
        P = self.part_T(n)
        zc = self.wheel_grasp_z(n)
        close = EX if n != "barrel" else P[:3, 0] * np.array([1, 1, 0])     # the barrel: across its square's flats
        g = T(P[:3, 3] + np.array([0, 0, zc + .013]), Rcols(DOWN, close))
        self.grip[h] = GRIP_OPEN
        self.go(h, self.up(g, .04))
        self.move(h, g, v=.04)
        self.grasp(h, "barrel_arbor" if n == "barrel" else n)
        self.move(h, self.up(g, .07), v=.06)
        G_rel = np.linalg.inv(self.part_T(n)) @ self.tcp_actual(h)           # the TCP in the part's frame
        R_part = Rot.from_euler("z", self._yaw(self.part_T(n)[:3, :3])).as_matrix()   # upright
        target_part = T(W + np.array([geo.POS[n][0], geo.POS[n][1], 0.]), R_part)
        above = target_part.copy()
        above[2, 3] += .012
        self.go(h, self.up(above @ G_rel, .03))
        self.move(h, above @ G_rel, v=.04)
        turns = 0
        for _ in range(6):
            goal = self.up(target_part @ G_rel, .0002)          # just above the seat: pushing below it bends the arbor
            self.move(h, goal, v=.006, until=lambda: self.rules.seated[n])
            self.wait(.15)
            if self.rules.seated[n]:
                break
            if n in geo.PARENT and self.rules.gate_closed[n]:
                self.step_name(f"turn {n} half a leaf (gate)")
                half = math.pi / geo.GEARS[n]["pin"] * (1 if turns % 2 == 0 else -1)
                turns += 1
                M = self.pose[h].copy()
                M[:3, :3] = Rot.from_euler("z", half).as_matrix() @ M[:3, :3]
                self.move(h, M, v=.01, w=.4, until=lambda: not self.rules.gate_closed[n])
                R_part = Rot.from_euler("z", self._yaw(self.part_T(n)[:3, :3])).as_matrix()
                target_part = T(W + np.array([geo.POS[n][0], geo.POS[n][1], 0.]), R_part)
                G_rel = np.linalg.inv(self.part_T(n)) @ self.tcp_actual(h)
        self.release(h)
        self.move(h, self.up(self.pose[h], .04), v=.06)
        self.wait(.2)
        self.note(f"{n} seated {self.rules.seated[n]}")
        return self.rules.seated[n]

    @staticmethod
    def _yaw(R):
        return math.atan2(R[1, 0], R[0, 0])

    def place_bridge(self):
        """The right hand pinches the bridge across the middle of its center-third bar (bridge_bar2, 4.9 mm from the
        bridge's centre of mass, so the bridge hangs level between the fingers instead of out from them), brings it down
        over the steady pins and the barrel arbor until it rests (on its pillars or on pivot tips that miss their holes),
        then lets go: its own weight is the light press of the watchmaker's finger. Closing toward the bridge's -y keeps
        the right arm on one IK branch from the stand to the movement."""
        self.step_name("place bridge")
        h = "R"
        B = self.part_T("bridge")
        pinch = .5 * (geo.BRIDGE_PTS["C"] + geo.BRIDGE_PTS["T"])       # the middle of bridge_bar2, bridge frame
        close = B[:3, :3] @ -EY                                         # across the bar
        g = T(B[:3, 3] + B[:3, :3] @ np.r_[pinch, .006 + .013], Rcols(DOWN, close))
        self.grip[h] = GRIP_OPEN
        self.go(h, self.up(g, .04))
        self.move(h, g, v=.04)
        self.grasp(h, "bridge")
        self.move(h, self.up(g, .08), v=.06)
        self.bridge_G = np.linalg.inv(self.part_T("bridge")) @ self.tcp_actual(h)
        rest = geo.SHOULDER_Z + geo.UP_PIVOT_L + .0003
        target = T(self.P_W + np.array([0, 0, rest]), np.eye(3))
        self.go(h, self.up(target @ self.bridge_G, .06), v_end=.12)
        self.move(h, self.up(target @ self.bridge_G, .015), v=.04)
        self.move(h, target @ self.bridge_G, v=.004)
        self.wait(.3)
        self.release(h)
        self.move(h, self.up(self.pose[h], .05), v=.05)
        self.wait(.5)

    def press_at(self, xy, depth=.006):
        """The right hand's open fingertips press the bridge top around W point xy: down at 5 mm/s until 4 N (an open
        gripper never acquires a grasp)."""
        bz = self.part_T("bridge")[2, 3] + geo.T_BRIDGE
        p = np.array([self.P_W[0] + xy[0], self.P_W[1] + xy[1], bz])
        Rp = Rcols(DOWN, EX)
        self.grip["R"] = GRIP_OPEN
        self.go("R", T(p + np.array([0, 0, .0161 + .02]), Rp), v_end=.06)
        self.pressing = True
        touch = [None]

        def enough():
            f = self.hand_force_on("R", "bridge")
            if f > .3 and touch[0] is None:
                touch[0] = self.pose["R"][2, 3]
            return f >= PRESS_FORCE or (touch[0] is not None and touch[0] - self.pose["R"][2, 3] > depth)
        self.move("R", T(p + np.array([0, 0, .0161 - depth]), Rp), v=.005, until=enough)

    def press_off(self):
        self.move("R", self.up(self.pose["R"], .03), v=.03)
        self.pressing = False

    def lower_bridge(self):
        self.step_name("press bridge down")
        self.press_at(.5 * (geo.BRIDGE_PTS["C"] + geo.BRIDGE_PTS["T"]))
        self.wait(.6)
        self.press_off()
        self.wait(.3)

    def bar_clear(self, q, margin=.006):
        """W point q (2D) at least half a bar width + margin from every bridge bar and boss."""
        for a, b in geo.BRIDGE_BARS:
            pa, pb = geo.BRIDGE_PTS[a], geo.BRIDGE_PTS[b]
            v = pb - pa
            s = np.clip(np.dot(q - pa, v) / np.dot(v, v), 0, 1)
            if np.linalg.norm(q - (pa + s * v)) < geo.BAR_W / 2 + margin:
                return False
        for k, r in geo.BOSS_R.items():
            if np.linalg.norm(q - geo.BRIDGE_PTS[k]) < r + margin:
                return False
        return True

    def probe_pose(self, tip, d_rod):
        """The probe's pose with its tip point at tip (M) and its rod along d_rod (hand -> tip)."""
        d_rod = d_rod / np.linalg.norm(d_rod)
        z = EZ - np.dot(EZ, d_rod) * d_rod
        z /= np.linalg.norm(z)
        Rp = np.c_[d_rod, np.cross(z, d_rod), z]
        return T(tip - d_rod * (geo.PROBE_L + geo.PROBE_TIP - .0012), Rp)

    def take_probe(self):
        self.step_name("take probe")
        h = "L"
        P = self.part_T("probe")
        g = P @ T(np.array([.010 - .013, 0, 0]), np.eye(3))    # held like a screwdriver: the gripper along the rod
        back = g.copy()
        back[:3, 3] -= g[:3, 0] * .04
        self.grip[h] = GRIP_OPEN
        self.go(h, self.up(back, .03))
        self.move(h, back, v=.06)
        self.move(h, g, v=.03)
        self.grasp(h, "probe")
        self.probe_G = np.linalg.inv(self.part_T("probe")) @ self.tcp_actual(h)
        self.move(h, self.up(self.pose[h], .06), v=.06)

    def probe_tip_cmd(self):
        P = self.pose["L"] @ np.linalg.inv(self.probe_G)
        return P[:3, 3] + P[:3, 0] * (geo.PROBE_L + geo.PROBE_TIP)

    def probe_tip_actual(self):
        P = self.part_T("probe")
        return P[:3, 3] + P[:3, 0] * (geo.PROBE_L + geo.PROBE_TIP - .0012)

    def probe_to(self, tip, d_rod, v=.03, high=False):
        """The probe's tip to tip with the rod along d_rod; the tip moves on straight lines (the hand turns about it)."""
        target = self.probe_pose(tip, d_rod) @ self.probe_G
        tool = (np.linalg.inv(self.probe_G) @ np.r_[geo.PROBE_L + geo.PROBE_TIP, 0, 0, 1])[:3]
        if high:          # the tip rises over the bridge, travels, comes down: never sweeps low across the movement
            a, b = self.probe_tip_cmd()[:2] - self.P_W[:2], np.asarray(tip, float)[:2] - self.P_W[:2]
            sq = np.asarray(geo.POS["barrel"])
            s_ = np.clip(np.dot(sq - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0, 1)
            near_square = np.linalg.norm(a + s_ * (b - a) - sq) < .015
            z_clear = self.P_W[2] + max(self.PROBE_CLEAR_Z, (geo.BARREL_SQUARE_TOP + .010) if near_square else 0.)
            self.move("L", self.up(self.pose["L"], max(0., z_clear - self.probe_tip_cmd()[2])), v=.1)
            above = np.array(tip, float)
            above[2] = max(above[2], z_clear)
            self.move("L", self.probe_pose(above, d_rod) @ self.probe_G, v=v, w=1., tool=tool)
        self.move("L", target, v=v, w=1., tool=tool)

    def nudge(self, n, stop_lat=None, keep=False, attempts=4):
        """Tilt wheel n so its upper pivot (the barrel: its arbor) moves under its bridge hole. A wheel standing on its lower
        pivot tilts the way it is pushed: the probe either presses the top face of its rim on the side the pivot has to
        go to ('press') or pushes the rim's edge from the far side toward it ('side'), whichever comes closest to the
        needed direction with the rod (75 deg, from outside) clear of every bridge bar and boss and the arm reaching."""
        if not keep:
            self.step_name(f"nudge {n}")
        W, a75 = self.P_W, math.radians(75)

        def done():
            if stop_lat is None:
                return self.rules.entered(n)
            return bool(np.linalg.norm(self.rules.tip_offset(n)[0][:2]) <= stop_lat)
        for _ in range(attempts):
            if done():
                return True
            lat, _ = self.rules.tip_offset(n)
            e = lat[:2]
            en = e / (np.linalg.norm(e) + 1e-12)
            p, _ = self.rules.wheel_local(n)
            g = geo.GEARS[n]
            band = g["drum"] if n == "barrel" else g["band"]
            r_root = geo.pitch_r(n) - 1.25 * g["m"] - geo.COL_SHRINK
            Rr = r_root - (.004 if n == "barrel" else .003)
            axis_xy = p[:2]

            def rod_ok(q, out, z0):
                return all(self.bar_clear(q + out * dz / math.tan(a75), margin=.005)
                           for dz in np.linspace(0., geo.H_BRIDGE + geo.T_BRIDGE + .016 - z0, 12))

            def others_clear(q):
                for o in geo.WHEELS:
                    if o != n and np.linalg.norm(q - np.asarray(geo.POS[o])) < geo.pitch_r(o) + geo.GEARS[o]["m"] + .004:
                        return False
                return True

            def rod_dir(w_):
                return -np.r_[w_, 0.] * math.cos(a75) - EZ * math.sin(a75)

            def reach(tips, w_):
                return all(self.kin.reach("L", self.probe_pose(W + t, rod_dir(w_)) @ self.probe_G, (self.q_sol["L"],))
                           for t in tips)
            choice = None
            zc = p[2] + .5 * (band[0] + band[1])
            tilt_needed = min(np.linalg.norm(e) / (geo.SHOULDER_Z + geo.UP_PIVOT_L + .002) + .004, 2 * geo.TILT_MAX)
            for beta in (0, 15, -15, 30, -30, 45, -45, 60, -60):
                rot = Rot.from_euler("z", math.radians(beta))
                for mode in ("press", "side"):
                    u = rot.apply(np.r_[-en if mode == "press" else en, 0])[:2]
                    for gamma in (0, 30, -30, 60, -60):
                        w_ = Rot.from_euler("z", math.radians(gamma)).apply(np.r_[u, 0])[:2]
                        if mode == "press":
                            for frac in (1., .85, .7, .55):
                                q = axis_xy + u * Rr * frac
                                t = np.r_[q, p[2] + band[1] + .0012]
                                a = t - rod_dir(w_) * .03
                                end = t - EZ * (tilt_needed * Rr * frac + .0004)
                                if rod_ok(q, w_, band[1]) and reach([t, end, a, np.r_[a[:2], self.PROBE_CLEAR_Z]], w_):
                                    choice = (mode, u, w_, q, frac)
                                    break
                        else:
                            q = axis_xy + u * (r_root + .0012 + .004)
                            t = np.r_[q, zc]
                            end = t - np.r_[u, 0.] * (.004 + tilt_needed * zc + .0004)
                            back = t + np.r_[u, 0.] * .003 + EZ * .02
                            if rod_ok(q, w_, zc) and others_clear(q) and reach([t, end, back, np.r_[q, self.PROBE_CLEAR_Z]], w_):
                                choice = (mode, u, w_, q, 1.)
                        if choice is not None:
                            break
                    if choice is not None:
                        break
                if choice is not None:
                    break
            if choice is None:
                self.note(f"no clear rim point for {n}")
                return False
            mode, u, w_, q, frac = choice
            d_rod = rod_dir(w_)
            if mode == "press":
                touch = W + np.r_[q, p[2] + band[1] + .0012]
                travel, step = tilt_needed * Rr * frac + .0004, -EZ
                self.probe_to(touch - d_rod * .03, d_rod, v=.08, high=True)
                self.probe_to(touch, d_rod, v=.02)
            else:
                touch = W + np.r_[q, zc]
                travel, step = .004 + tilt_needed * zc + .0004, -np.r_[u, 0.]
                self.probe_to(touch + EZ * .02, d_rod, v=.08, high=True)
                self.probe_to(touch, d_rod, v=.02)
            self.note(f"nudge {n}: {mode} at {math.degrees(math.acos(np.clip(np.dot(u, -en if mode == 'press' else en), -1, 1))):.0f} deg")
            corr = 0.
            for k in range(int(travel / .002 / self.tick_s)):
                if mode == "side":     # keep the tip at the rim's mid-height (2 mm thick): correct the arm's sag
                    corr = float(np.clip(corr + .2 * (touch[2] - self.probe_tip_actual()[2]), -.003, .003))
                tip = touch + step * .002 * self.tick_s * (k + 1) + EZ * corr
                self.pose["L"] = self.probe_pose(tip, d_rod) @ self.probe_G
                if self.tick():
                    return False
                if done() or self.probe_push() > NUDGE_FORCE:
                    break
            if keep and done():
                self.kept = (mode, touch, d_rod, u)
                return True
            self.wait(.3)
            if mode == "press":
                self.probe_to(touch - d_rod * .03, d_rod, v=.04)
            else:
                self.probe_to(touch + np.r_[u, 0.] * .003, d_rod, v=.02)
                self.probe_to(touch + np.r_[u, 0.] * .003 + EZ * .02, d_rod, v=.04)
        return done()

    def ease_and_nudge(self, n):
        """A pivot stays caught under the edge of its hole's entry cone: the right hand lifts the bridge 1.2 mm by its grasp
        bar, the probe brings the pivot within the cone and holds the wheel there, the bridge goes down, the probe backs out."""
        self.step_name(f"ease bridge for {n}")
        self.probe_to(self.P_W + np.array([-.17, .06, .06]), np.array([.3, 0., -1.]), v=.12, high=True)
        g = self.part_T("bridge") @ self.bridge_G
        self.grip["R"] = GRIP_OPEN
        self.go("R", self.up(g, .04), v_end=.06)
        self.move("R", g, v=.02)
        self.grasp("R", "bridge")
        self.move("R", self.up(self.pose["R"], .0012), v=.002)
        self.kept = None
        self.nudge(n, stop_lat=geo.PIVOT_HOLE_R + geo.CHAMFER - .0004, keep=True)
        self.move("R", self.up(self.pose["R"], -.0030), v=.002)
        self.wait(.3)
        self.release("R")
        self.move("R", self.up(self.pose["R"], .05), v=.05)
        if self.kept is not None:
            mode, touch, d_rod, u = self.kept
            if mode == "press":
                self.probe_to(touch - d_rod * .03, d_rod, v=.03)
            else:
                self.probe_to(touch + np.r_[u, 0.] * .003, d_rod, v=.02)
                self.probe_to(touch + np.r_[u, 0.] * .003 + EZ * .02, d_rod, v=.04)
        self.note(f"eased the bridge for {n}: entered {self.rules.entered(n)}")
        return self.rules.entered(n)

    def drive_screw(self, s, k, h):
        self.step_name(f"screw {s}")
        P = self.part_T(s)
        g = T(P[:3, 3] + np.array([0, 0, geo.HEAD_H - .0035 + .013]),
              Rot.from_euler("z", -SCREW_YAW[s]).as_matrix() @ Rcols(DOWN, EX))      # a round head: any yaw
        self.grip[h] = GRIP_OPEN
        self.go(h, self.up(g, .04))
        self.move(h, g, v=.04)
        self.grasp(h, s)
        self.move(h, self.up(g, .06), v=.06)
        G_rel = np.linalg.inv(self.part_T(s)) @ self.tcp_actual(h)
        bT = self.part_T("bridge")
        hole = bT @ np.r_[geo.PILLARS[k] + geo.SCREW_OFF, geo.T_BRIDGE, 1.]
        target = T(hole[:3], self.part_T(s)[:3, :3])
        self.go(h, self.up(target @ G_rel, .05))
        self.move(h, self.up(target @ G_rel, .004), v=.03)
        self.move(h, self.up(target @ G_rel, -.0005), v=.006)
        self.wait(.2)
        for stroke in range(6):
            if self.rules.tight[s]:
                break
            self.step_name(f"screw {s} stroke {stroke + 1}")
            self.turn(h, -math.radians(240), w=2., until=lambda: self.rules.tight[s])
            if self.rules.tight[s]:
                break
            self.release(h)
            self.move(h, self.up(self.pose[h], .004), v=.02)
            self.turn(h, math.radians(240), w=3.)
            self.move(h, self.up(self.pose[h], -.004), v=.02)
            self.grasp(h, s)
        self.release(h)
        self.move(h, self.up(self.pose[h], .05), v=.06)
        self.note(f"{s} tight {self.rules.tight[s]}")

    def wind_and_run(self, strokes=3, stroke=math.radians(90.)):
        """The run test: the left hand winds the barrel by its square (counterclockwise from above: the click lets the
        arbor turn that way only), regrasping between strokes as on a screw; then lets go, lifts and withdraws to its
        park pose (clear of the train, so the run is the mainspring's alone and the cameras see the train turn): the
        mainspring drives the train, and the rules time the run (RUN_SETTLE_S after the last advance, then RUN_WINDOW_S)."""
        h = "L"
        A = self.part_T("barrel_arbor")
        g = T(A[:3, 3] + A[:3, :3] @ np.array([0, 0, self.wheel_grasp_z("barrel") + .013]),
              Rcols(DOWN, A[:3, 0] * np.array([1, 1, 0])))
        self.step_name("wind the barrel")
        self.grip[h] = GRIP_OPEN
        self.go(h, self.up(g, .04))
        self.move(h, g, v=.03)
        for k in range(strokes):
            self.step_name(f"wind stroke {k + 1}")
            self.grasp(h, "barrel_arbor")
            self.turn(h, stroke, w=1.2, axis=self.part_T("barrel")[:3, 2])
            self.release(h)
            if k + 1 < strokes:                 # the open fingers back to the square's next flats (a quarter turn on)
                self.move(h, self.up(self.pose[h], .003), v=.02)
                self.turn(h, -stroke, w=2.)
                self.move(h, self.up(self.pose[h], -.003), v=.02)
        self.move(h, self.up(self.pose[h], .03), v=.05)
        self.step_name("run: the mainspring drives the train")
        if self.go(h, T(self.PARK[h], Rcols(DOWN, EX)), v=.10, v_end=.10):
            return
        for _ in range(int(9. / self.tick_s)):
            if self.tick():
                return

    def park(self, hands="LR"):
        self.step_name("park")
        for h in hands:
            self.go(h, T(self.PARK[h], Rcols(DOWN, EX)), v=.10, v_end=.10)

    def tracking(self):
        """Per hand: the TCP's distance from the commanded pose (median / 90 % / 99 % / max, mm), all control steps and
        those holding a part."""
        out = {}
        for h, rows in self.track.items():
            e = np.array([r[0] for r in rows]) * 1e3
            held = np.array([r[1] for r in rows], bool)
            q = lambda x: [round(float(v), 1) for v in np.percentile(x, [50, 90, 99, 100])] if len(x) else None
            worst = sorted({r[2]: max(x[0] for x in rows if x[2] == r[2]) for r in rows}.items(), key=lambda kv: -kv[1])[:4]
            out[h] = dict(all=q(e), holding=q(e[held]), worst_phases_mm={p: round(v * 1e3, 1) for p, v in worst})
        return out

    # ---------------------------------------------------------------- the whole episode
    def unfold(self):
        """From the birth (home) pose to the park pose with the hand pointing down, in joint space: the solution nearest
        home among those with the largest joint margins (home's TCP points forward; a Cartesian turn from it to pointing
        down would flip the wrist on the way)."""
        for h in "LR":
            P = T(self.PARK[h], Rcols(DOWN, EX))
            sols = self.kin.ik_all(h, P, n=24)
            sols = [q for q, mg in sols if mg >= .3] or [q for q, _ in sols]
            q = min(sols, key=lambda q: float(np.abs(q - self.kin.q_home[h]).max()))
            self.joint_move(h, q)
            self.pose[h] = P.copy()

    def run(self, upto="all"):
        self.step_name("start")
        self.grip = {h: GRIP_OPEN for h in "LR"}
        self.wait(.3)
        self.unfold()
        for n in geo.WHEELS:
            self.place_wheel(n)
            if self.ended:
                return
        self.move("L", T(np.array([-.25, .05, .22]) + np.r_[geo.OFFSET, 0.], Rcols(DOWN, EX)), v=.15)
        if upto == "wheels":
            return
        self.place_bridge()
        self.park("R")
        self.note(f"bridge placed, entered {[n for n in geo.WHEELS if self.rules.entered(n)]}")
        probe = not all(self.rules.entered(n) for n in geo.WHEELS)     # every pivot in: no probe needed
        if probe:
            self.take_probe()
            for _ in range(2):                 # a pivot that went in can come out again while the bridge settles
                for n in geo.WHEELS:
                    if not self.rules.entered(n):
                        self.nudge(n, attempts=2)
                for n in ("center", "third", "fourth"):     # still out: caught under an entry cone's edge
                    if not self.rules.entered(n):
                        self.ease_and_nudge(n)
                if all(self.rules.entered(n) for n in geo.WHEELS) or self.ended:
                    break
            self.note(f"nudged, entered {[n for n in geo.WHEELS if self.rules.entered(n)]}")
            self.probe_to(self.P_W + np.array([-.17, .06, .06]), np.array([.3, 0., -1.]), v=.12, high=True)
        self.lower_bridge()
        self.park("R")
        if probe:
            self.step_name("probe back")
            P0 = geo.opening_poses()["probe"]
            target = T(P0[0] + np.array([0, 0, .002]), np.eye(3))
            self.go("L", self.up(target @ self.probe_G, .05))
            self.move("L", target @ self.probe_G, v=.03)
            self.release("L")
            back = self.pose["L"].copy()
            back[:3, 3] -= back[:3, 0] * .04
            self.move("L", back, v=.04)
            self.move("L", self.up(self.pose["L"], .05), v=.06)
        self.note(f"bridge seat error {np.round(self.rules.bridge_seat_error(), 5).tolist()}")
        if upto == "press":
            return
        self.drive_screw("screw1", "P1", "L")
        self.park("L")
        self.drive_screw("screw2", "P2", "R")
        self.park("R")
        self.wind_and_run()
        self.park()
        self.wait(1.)
