"""The v2 very_easy oracle's own ALOHA kinematics: joint-branch choice, joint paths and the joint targets it commands.

A 6-joint arm reaches most hand poses with several joint solutions (for example the wrist flipped: forearm roll
+180 deg, wrist angle negated, wrist rotate +180 deg); for this assembly only some of them stay inside the joint limits
for the whole episode (2026-10-09 IK study). The oracle therefore plans each arm's joint path itself, from every start
solution of the first target and both ways of pinching the bar (0 / 180 deg about the fingers), keeps a path that
reaches every target, drives the arm onto it from home through a joint-space move, and then solves every commanded
pose with this model's IK started from its previous solution, so the arm stays on the planned solution. For a lattice
design that asks for it (record "anchor": "arms"), the pair's placement during the assembly is also planned here, by
both arms together (OracleArms.placement): the route fixes only silver's pose relative to gold.

The model is the simulator's (a robot-only MuJoCo scene of the same robot configuration, built by topogym.world.mujoco
with the tier's profile) but a separate copy: it never reads or writes the simulation. IK is the robot entity
view's damped least squares (world/mujoco/views.py) run to 1e-5 m / 1e-4 rad; a candidate counts as a solution only
within 1e-4 m / 1e-3 rad of the target, otherwise the query raises "Unreachable EE pose".
"""
import numpy as np
from scipy.spatial.transform import Rotation

MAX_STEP_RAD = .12        # joint change allowed between consecutive targets before the interval is split
SPLITS = 3                # an interval is split into 4 at most this many times (64 intermediate poses)
# Largest arm joint target change per control step. The standard service moves a joint target at most 0.04 rad per
# control step (rcb_topogym JOINT_STEP_LIMIT_RAD) and spreads a larger change over more steps, which changes the
# reference's timing; the margin covers float32 rounding.
STEP_LIMIT_RAD = .036
SMOOTH_PEAK = 1.875       # the peak speed of the smoothstep profile 10u^3 - 15u^4 + 6u^5, relative to its mean
# IK: at most 100 damped least-squares steps (damping 1e-3, i.e. 1e-6 on J J^T; at most 0.2 rad per joint and step)
# until 1e-5 m / 1e-4 rad; a solution within 1e-4 m / 1e-3 rad.
IK = dict(max_solver_iters=100, damping=1e-3, pos_tol=1e-5, rot_tol=1e-4, max_step_size=.2)
POSITION_TOLERANCE_M, ORIENTATION_TOLERANCE_RAD = 1e-4, 1e-3
# The pair's placement found by both arms (OracleArms.placement): the closed chain to 1e-6 m / 1e-5 rad within 60
# damped least-squares steps per sample, each joint kept CHAIN_LIMIT_MARGIN_RAD inside its range. Below the chain,
# the pieces' lowest points rise out of a TABLE_BAND_M band above the floor (at most TABLE_STEP_M per sample); below
# that the hands split every route motion evenly (the pair's common motion, the mean of the two TCP twists, is zero)
# and the pair drifts down the posture cost (distance from the assembled configuration, closeness to the joint limits)
# by at most PAIR_STEP_M / PAIR_STEP_RAD per sample. SPLIT_DAMPING lets the even split give way near singular
# configurations (2026-10-10, five lattice designs: 0.05 kept every piece >= 49 mm above the table and gold's world
# motion <= 15 mm per sample; 0.1 gave larger joint-limit margins but let one design into the table).
CHAIN_TOLERANCE, CHAIN_ITERS, CHAIN_LIMIT_MARGIN_RAD = (1e-6, 1e-5), 60, .02
TABLE_BAND_M, TABLE_STEP_M = .03, .005
POSTURE_GAIN, PAIR_STEP_M, PAIR_STEP_RAD = .05, .001, np.deg2rad(.5)
SPLIT_DAMPING, CHAIN_DAMPING = .05, 1e-3


def _rotation(wxyz):
    return Rotation.from_quat(np.asarray(wxyz, float)[[1, 2, 3, 0]])


def _compose(a, b):
    return np.asarray(a[0], float) + a[1].apply(np.asarray(b[0], float)), a[1]*b[1]


def _inverse(a):
    r = a[1].inv()
    return -r.apply(np.asarray(a[0], float)), r


class _Chain:
    """Both arms holding the pair (OracleArms.placement), on a private MjData of the planner's model: the left TCP
    must be the right TCP composed with K = G^-1 R S (G, S: the TCP poses in the gold and silver frames, pinch roll
    included; R: silver's pose relative to gold), 6 equations on the 12 arm joints."""
    SIDES = ("left", "right")

    def __init__(self, arms, grasps, points, floor_m):
        import mujoco
        self.mujoco, self.arms, self.grasps, self.points, self.floor = mujoco, arms, grasps, points, floor_m
        self.model = arms.world.model
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[arms.entity._qadr] = arms.home
        self.qadr = {s: arms.entity._qadr[arms.arm(s)[0]] for s in self.SIDES}
        self.dadr = {s: arms.entity._dadr[arms.arm(s)[1]] for s in self.SIDES}
        self.body = {s: int(arms.robot._hands[s].idx) for s in self.SIDES}
        self.offset = {s: (np.asarray(g.tcp_position_in_hand_m, float), _rotation(g.tcp_orientation_in_hand_wxyz))
                       for s, g in arms.robot.groups.items()}
        self.lo = np.concatenate([arms.limits[s][:, 0] + CHAIN_LIMIT_MARGIN_RAD for s in self.SIDES])
        self.hi = np.concatenate([arms.limits[s][:, 1] - CHAIN_LIMIT_MARGIN_RAD for s in self.SIDES])
        self.jacp, self.jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))

    def set(self, x):
        for k, s in enumerate(self.SIDES):
            self.data.qpos[self.qadr[s]] = x[6*k:6*k+6]
        self.mujoco.mj_kinematics(self.model, self.data)
        self.mujoco.mj_comPos(self.model, self.data)

    def tcp(self, s):
        b = self.body[s]
        r = Rotation.from_matrix(self.data.xmat[b].reshape(3, 3))
        return self.data.xpos[b] + r.apply(self.offset[s][0]), r*self.offset[s][1]

    def jac(self, s, point):
        """(linear, angular) Jacobian rows of a point moving with side s's hand, over the 12 arm joints."""
        self.mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, np.asarray(point, float), self.body[s])
        out = np.zeros((6, 12))
        k = self.SIDES.index(s)
        out[:3, 6*k:6*k+6], out[3:, 6*k:6*k+6] = self.jacp[:, self.dadr[s]], self.jacr[:, self.dadr[s]]
        return out

    def piece(self, s):
        return _compose(self.tcp(s), _inverse(self.grasps[s]))

    def table_rows(self):
        """Per piece whose lowest point is inside the band above the floor: that point's z row and the rise asked
        for."""
        rows = []
        for s in self.SIDES:
            pose = self.piece(s)
            world = pose[0] + pose[1].apply(self.points[s])
            low = int(np.argmin(world[:, 2]))
            short = self.floor + TABLE_BAND_M - world[low, 2]
            if short > 0:
                rows.append((self.jac(s, world[low])[2], min(short, TABLE_STEP_M)))
        return rows

    def posture_gradient(self, x, centre):
        half, mid = (self.hi - self.lo)/2, (self.hi + self.lo)/2
        return 2*(x - centre)/half**2 + 8*((x - mid)/half)**7/half

    def solve(self, x, K, centre):
        """The 12 arm joints satisfying the closed chain for K, from x (task priorities: the chain, the table band,
        the even split with the posture drift); None when it does not converge."""
        x = x.copy()
        for it in range(CHAIN_ITERS):
            self.set(x)
            left, right = self.tcp("left"), self.tcp("right")
            want = _compose(right, K)
            f = np.r_[want[0] - left[0], (want[1]*left[1].inv()).as_rotvec()]
            if it and np.linalg.norm(f[:3]) < CHAIN_TOLERANCE[0] and np.linalg.norm(f[3:]) < CHAIN_TOLERANCE[1]:
                return x
            J = self.jac("right", want[0]) - self.jac("left", left[0])
            P = .5*(self.jac("left", left[0]) + self.jac("right", right[0]))       # the mean TCP twist
            c, rows = np.zeros(6), []
            if it == 0:
                drift = P @ -self.posture_gradient(x, centre)
                c = drift*min(PAIR_STEP_M/max(np.linalg.norm(drift[:3]), 1e-12),
                              PAIR_STEP_RAD/max(np.linalg.norm(drift[3:]), 1e-12), POSTURE_GAIN)
                rows = self.table_rows()
            free = np.ones(12, bool)
            for _ in range(3):
                Jf, Pf = J*free, P*free
                pinv = Jf.T @ np.linalg.inv(Jf @ Jf.T + CHAIN_DAMPING**2*np.eye(6))
                dx = -pinv @ f
                N = np.diag(free.astype(float)) - pinv @ Jf
                if rows:
                    T = np.array([r for r, _ in rows])*free
                    A = T @ N
                    pa = A.T @ np.linalg.inv(A @ A.T + CHAIN_DAMPING**2*np.eye(len(rows)))
                    dx += pa @ (np.array([e for _, e in rows]) - T @ dx)
                    N = N - pa @ A
                A = Pf @ N
                dx += A.T @ np.linalg.solve(A @ A.T + SPLIT_DAMPING**2*np.eye(6), c - Pf @ dx)
                dx = np.clip(dx, -.08, .08)
                beyond = ((x + dx < self.lo) & (dx < 0)) | ((x + dx > self.hi) & (dx > 0))
                if not beyond.any():
                    break
                free &= ~beyond              # a joint at its limit stops; the others take over
            x = np.clip(x + dx, self.lo, self.hi)
        return None

    def track(self, start, relatives):
        """The 12 arm joints at every relative pose (assembled first) from `start`, or None."""
        G, S = self.grasps["right"], self.grasps["left"]
        x, out = start, []
        for relative in relatives:
            x = self.solve(x, _compose(_inverse(G), _compose(relative, S)), start)
            if x is None:
                return None
            out.append(x)
        return out


class OracleArms:
    def __init__(self):
        import tempfile
        from topogym.config.scene import load_scene_config
        from topogym.simulator.config import SimulatorConfig
        from topogym.world.mujoco import MujocoWorld, make_robot, profile_for
        cfg = load_scene_config(SimulatorConfig(task_id="cast_coil", robot_id="aloha", enable_rgb=False,
                                                compute_backend="cpu"))
        with tempfile.TemporaryDirectory() as folder:
            world = MujocoWorld.from_tabletop(cfg.env, cfg.robot_control, profile=profile_for("cast_coil", "cast_coil_v2_very_easy"),
                                              work_dir=folder)
            world.build()
        self.world, self.entity = world, world.entities["robot"]
        self.robot = make_robot(self.entity, cfg.robot_control)
        self.robot.reset_home()
        self.home = self.entity.get_qpos()
        self.limits = {side: np.array([world.model.jnt_range[self.entity.get_joint(n).mj_id] for n in group.arm_joints])
                       for side, group in self.robot.groups.items()}

    def arm(self, side):
        """(q indices, dof indices) of the side's arm joints in the robot's q vector."""
        n = len(self.robot.groups[side].arm_joints)
        return self.robot._q_indices[side][:n], self.robot._indices[side][:n]

    def full(self, joints):
        """A robot q vector from {side: arm joint angles} (other coordinates at home)."""
        q = self.home.copy()
        for side, values in joints.items():
            q[self.arm(side)[0]] = values
        return q

    def tcp(self, side, q):
        """(position, Rotation) of the side's TCP for the robot q vector."""
        group, hand = self.robot.groups[side], self.robot._hands[side]
        positions, quaternions = self.entity.forward_kinematics(q)
        turn = _rotation(quaternions[hand.idx_local])
        return positions[hand.idx_local] + turn.apply(group.tcp_position_in_hand_m), \
            turn*_rotation(group.tcp_orientation_in_hand_wxyz)

    def solve(self, side, target, seed):
        """The side's arm joint angles placing its TCP at `target` = (position, Rotation), descending from the arm
        joint angles `seed` = {side: angles}."""
        return self._solve(side, target, self.full(seed))[self.arm(side)[0]]

    def _solve(self, side, target, q):
        """The robot q vector placing the side's TCP at `target`, descending from the robot q vector `q`."""
        group, hand = self.robot.groups[side], self.robot._hands[side]
        position, tcp_turn = target
        hand_turn = tcp_turn*_rotation(group.tcp_orientation_in_hand_wxyz).inv()
        hand_position = position - hand_turn.apply(group.tcp_position_in_hand_m)
        candidate = self.entity.inverse_kinematics(hand, pos=hand_position, quat=hand_turn.as_quat()[[3, 0, 1, 2]],
                                                   init_qpos=q, dofs_idx_local=self.arm(side)[1], **IK)
        if not np.isfinite(candidate).all():
            raise ValueError("IK produced a non-finite candidate")
        positions, quaternions = self.entity.forward_kinematics(candidate)
        residual = float(np.linalg.norm(positions[hand.idx_local]-hand_position))
        angle = float((_rotation(quaternions[hand.idx_local]).inv()*hand_turn).magnitude())
        if residual > POSITION_TOLERANCE_M or angle > ORIENTATION_TOLERANCE_RAD:
            raise ValueError(f"Unreachable EE pose: position residual={residual:.6g} m, orientation={angle:.6g} rad")
        return candidate

    def margin(self, side, q):
        joints = q[self.arm(side)[0]]
        lower, upper = self.limits[side][:, 0], self.limits[side][:, 1]
        return float(np.min(np.minimum(joints-lower, upper-joints)))

    def _reach(self, side, a, b, q, depth=0):
        from topogym.planning.motions.cast_coil_official import interpolate
        index = self.arm(side)[0]
        try:
            candidate = self._solve(side, b, q)
            if np.abs(candidate[index]-q[index]).max() <= MAX_STEP_RAD:
                return candidate
        except ValueError:
            pass
        if depth == SPLITS:
            return None
        last = a
        for u in (.25, .5, .75, 1.):
            middle = interpolate(a, b, u)
            q = self._reach(side, last, middle, q, depth+1)
            if q is None:
                return None
            last = middle
        return q

    def track(self, side, targets, start):
        """Joint path (robot q vector per target) following `targets` from the solution `start`, or None."""
        path, q = [start], start
        for a, b in zip(targets[:-1], targets[1:]):
            q = self._reach(side, a, b, q)
            if q is None:
                return None
            path.append(q)
        return path

    def starts(self, side, target, tries=60, seed=0):
        """Distinct joint solutions (robot q vectors) of one target, from home and random seeds."""
        rng = np.random.default_rng(seed)
        index = self.arm(side)[0]
        found = []
        for k in range(tries):
            seed_q = self.home.copy()
            if k:
                seed_q[index] = rng.uniform(self.limits[side][:, 0], self.limits[side][:, 1])
            try:
                q = self._solve(side, target, seed_q)
            except ValueError:
                continue
            if all(np.abs(q[index]-other[index]).max() > .2 for other in found):
                found.append(q)
        return found

    def plan(self, side, targets):
        """The kept (hold, arm joint path, margin): every start solution and both pinch rolls; the largest joint-limit
        margin."""
        best = None
        for hold in (0, 2):
            flip = Rotation.from_euler("x", 90*hold, degrees=True)
            rolled = [(p, r*flip) for p, r in targets]
            for start in self.starts(side, rolled[0]):
                path = self.track(side, rolled, start)
                if path is None:
                    continue
                margin = min(self.margin(side, q) for q in path)
                if best is None or margin > best[2]:
                    best = (hold, path, margin)
        if best is None:
            raise RuntimeError(f"No joint path of the {side} arm reaches every planned target")
        hold, path, margin = best
        return hold, [q[self.arm(side)[0]] for q in path], margin

    def plan_segments(self, side, segments):
        """plan() for target lists joined by joint-space moves (a carry whose hand poses in between are free): the
        pinch roll is one for all of them (the grasp does not change), each segment starts from the best of its first
        target's solutions. The kept (hold, [arm joint path per segment], margin) has the largest smallest margin."""
        best = None
        for hold in (0, 2):
            flip = Rotation.from_euler("x", 90*hold, degrees=True)
            paths, margin = [], np.inf
            for targets in segments:
                rolled = [(p, r*flip) for p, r in targets]
                kept = None
                for start in self.starts(side, rolled[0]):
                    path = self.track(side, rolled, start)
                    if path is None:
                        continue
                    smallest = min(self.margin(side, q) for q in path)
                    if kept is None or smallest > kept[1]:
                        kept = (path, smallest)
                if kept is None:
                    break
                paths.append(kept[0])
                margin = min(margin, kept[1])
            else:
                if best is None or margin > best[2]:
                    best = (hold, paths, margin)
        if best is None:
            raise RuntimeError(f"No joint paths of the {side} arm reach every planned segment")
        hold, paths, margin = best
        return hold, [[q[self.arm(side)[0]] for q in path] for path in paths], margin

    def placement(self, record, tcp_in_piece, points, floor_m):
        """The assembly's piece poses [(assembly step, gold, silver)], start (apart) to assembled on the samples of
        planning/motions/cast_coil_official.assembly, with the pair's world placement found by both arms together
        (record "anchor": "arms") instead of fixed shares of every motion: from the oracle's assembled placement, the
        12 arm joints follow silver's pose relative to gold along the disassembly (_Chain), for both pinch rolls of
        each hand and the two start solutions of each arm with the largest joint-limit margins; the kept tracking has
        the largest smallest margin. silver = gold composed with the route's relative pose, exactly.
        tcp_in_piece / points: {side: TCP pose in its piece's frame / piece points}; floor_m: the lowest height the
        pieces should keep (the band above it is kept clear when the arms can)."""
        from topogym.planning.motions import cast_coil_official as M
        samples = M.disassembly(record)
        relatives = [relative for _, _, relative in samples]
        gold = M.assembled_gold_pose(record)
        assembled = {"right": gold, "left": M.compose(gold, relatives[0])}
        best = None
        for holds in ((0, 0), (0, 2), (2, 0), (2, 2)):
            grasps = {s: _compose(tcp_in_piece[s], (np.zeros(3), Rotation.from_euler("x", 90*h, degrees=True)))
                      for s, h in zip(("left", "right"), holds)}
            chain = _Chain(self, grasps, points, floor_m)
            starts = {s: sorted(self.starts(s, _compose(assembled[s], grasps[s]), tries=20),
                                key=lambda q, s=s: -self.margin(s, q))[:2] for s in ("left", "right")}
            for a in starts["left"]:
                for b in starts["right"]:
                    path = chain.track(np.r_[a[self.arm("left")[0]], b[self.arm("right")[0]]], relatives)
                    if path is None:
                        continue
                    margin = CHAIN_LIMIT_MARGIN_RAD + min(float(np.min(np.minimum(x - chain.lo, chain.hi - x)))
                                                          for x in path)
                    if best is None or margin > best[0]:
                        best = (margin, chain, path)
        if best is None:
            raise RuntimeError("No placement of the pair keeps both arms on the route")
        _, chain, path = best
        n, out = len(record["route"]), []
        for (index, _, relative), x in zip(reversed(samples), reversed(path)):
            chain.set(x)
            gold = chain.piece("right")
            out.append((n - index if index >= 0 else n, gold, M.compose(gold, relative)))
        return out

    def joint_move(self, start, ends, seconds=2., dt=.025):
        """{side: arm joint configurations} from the arm joint angles `start` = {side: angles} to `ends`, the arms in
        step: first every waist (arms folded), then the other joints. Each part lasts at least `seconds`, and long
        enough that no joint moves more than STEP_LIMIT_RAD per control step at the profile's peak speed."""
        parts = {}
        for side, end in ends.items():
            a, b = np.array(start[side], float), np.array(end, float)
            middle = a.copy()
            middle[0] = b[0]
            parts[side] = ((a, middle), (middle, b))
        out = {side: [] for side in ends}
        for part in range(2):
            travel = max(float(np.abs(parts[s][part][1]-parts[s][part][0]).max()) for s in ends)
            count = max(1, round(seconds/dt), int(np.ceil(SMOOTH_PEAK*travel/STEP_LIMIT_RAD)))
            for k in range(1, count+1):
                u = k/count
                u = u**3*(10+u*(-15+6*u))
                for side in ends:
                    first, last = parts[side][part]
                    out[side].append(first + u*(last-first))
        return out
