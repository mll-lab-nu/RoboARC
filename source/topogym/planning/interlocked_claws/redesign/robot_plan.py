"""Robot-level planning tools for the redesigned claws tiers: ALOHA IK on the scene's own MuJoCo model (bounded least
squares, continuity prior), kinematic collision checks of a whole configuration (arms + both pieces placed), and the
conversion of joint waypoints to Session rows at a joint-speed limit.

Everything here is kinematic: the Session (physics, welds, drives) executes the rows and is the verdict.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ARMS = ("left", "right")
JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
LIMITS = np.array(((-3.14158, 3.14158), (-1.85005, 1.25664), (-1.76278, 1.6057), (-3.14158, 3.14158),
                   (-1.8675, 2.23402), (-3.14158, 3.14158)))
TCP_IN_HAND = np.array([.13, 0., -.003])
WRIST_SAFE = (-1.8, 1.9)              # keep the wrist from folding toward the forearm (core: never > 116 deg = 2.02 rad)
WRIST_ROTATE_SAFE = 2.8               # and keep wrist_rotate away from +-180 deg (core, 10-10 21:15)
OPEN, CLOSED = .037, .002
FINGER_ON_GRIP = .0222


class Kin:
    """Kinematics + collision queries on a private MjData of a Session's model."""

    def __init__(self, model, pieces=("piece_0", "piece_1")):
        import mujoco
        self.mj, self.m = mujoco, model
        self.d = mujoco.MjData(model)
        self.qadr, self.hand, self.fadr = {}, {}, {}
        for arm in ARMS:
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}/{j}") for j in JOINTS]
            self.qadr[arm] = np.array([int(model.jnt_qposadr[i]) for i in ids])
            self.hand[arm] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}/gripper_link")
            self.fadr[arm] = np.array([int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                                                                 f"{arm}/{f}")])
                                       for f in ("left_finger", "right_finger")])
        self.piece_qadr, self.piece_bodies = {}, {}
        for p in pieces:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{p}::root")
            self.piece_qadr[p] = int(model.jnt_qposadr[j])
            self.piece_bodies[p] = {b for b in range(model.nbody)
                                    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(p + "::")}
        self.robot_bodies = {arm: {b for b in range(model.nbody)
                                   if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(arm + "/")}
                             for arm in ARMS}
        self.finger_bodies = {arm: {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}/{f}")
                                    for f in ("left_finger_link", "right_finger_link")} for arm in ARMS}
        self.static = {b for b in range(model.nbody)
                       if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "") in ("table", "ground", "table::table")}
        self.q0 = model.qpos0.copy()
        self.com = {}                             # piece -> composite COM in its design frame (set by the caller)

    # ------------------------------------------------------------------ state
    def set(self, q_arms=None, fingers=None, pieces=None, base=None):
        d = self.d
        d.qpos[:] = self.q0 if base is None else base
        for arm, q in (q_arms or {}).items():
            d.qpos[self.qadr[arm]] = q
        for arm, f in (fingers or {}).items():
            d.qpos[self.fadr[arm]] = f
        for p, T in (pieces or {}).items():         # T: the piece's design frame; the body frame is at its COM
            a = self.piece_qadr[p]
            T = np.asarray(T, float)
            d.qpos[a:a + 3] = T[:3, 3] + T[:3, :3] @ self.com.get(p, np.zeros(3))
            d.qpos[a + 3:a + 7] = Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]
        self.mj.mj_kinematics(self.m, d)

    def tcp(self, arm, q):
        self.set({arm: q})
        R = self.d.xmat[self.hand[arm]].reshape(3, 3).copy()
        return np.asarray(self.d.xpos[self.hand[arm]]) + R @ TCP_IN_HAND, R

    def tcp_frame(self, arm, q):
        p, R = self.tcp(arm, q)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
        return T

    # ------------------------------------------------------------------ IK
    def ik(self, arm, T_target, seed, prior=None, w_prior=.002, w_rot=.25, iters=400, wrist=WRIST_SAFE):
        T_target = np.asarray(T_target, float)
        lo, hi = LIMITS[:, 0].copy(), LIMITS[:, 1].copy()
        lo[4], hi[4] = max(lo[4], wrist[0]), min(hi[4], wrist[1])
        lo[5], hi[5] = -WRIST_ROTATE_SAFE, WRIST_ROTATE_SAFE
        x0 = np.clip(np.asarray(seed, float), lo + 1e-6, hi - 1e-6)
        pr = None if prior is None else np.asarray(prior, float)

        def res(x):
            p, R = self.tcp(arm, x)
            e = np.r_[p - T_target[:3, 3], w_rot * Rotation.from_matrix(T_target[:3, :3].T @ R).as_rotvec()]
            return e if pr is None else np.r_[e, w_prior * (x - pr)]
        r = least_squares(res, x0, bounds=(lo, hi), max_nfev=iters, diff_step=1e-7, xtol=1e-12, ftol=1e-12,
                          gtol=1e-12)
        p, R = self.tcp(arm, r.x)
        ep = float(np.linalg.norm(p - T_target[:3, 3]))
        er = float(Rotation.from_matrix(T_target[:3, :3].T @ R).magnitude())
        return r.x, ep, er

    def ik_multi(self, arm, T_target, seeds, **kw):
        best = None
        for n, s in enumerate(seeds):           # among exact solutions the earliest seed's wins (callers order them)
            q, ep, er = self.ik(arm, T_target, s, **kw)
            score = (1e-3 * n if ep < 1e-3 and er < 1e-2 else 1. + ep + .05 * er)
            if best is None or score < best[0]:
                best = (score, q, ep, er)
        return best[1], best[2], best[3]

    # ------------------------------------------------------------------ collisions
    def contacts(self, q_arms, fingers, pieces, held=(), margin=0.):
        """Contacts (dist < -margin) of the configuration, labelled; the fingers of a hand on the piece it holds and
        the two pieces with each other are left out (the object-level checker certifies the pieces)."""
        import mujoco
        self.set(q_arms, fingers, pieces)
        mujoco.mj_collision(self.m, self.d)
        out = []
        held_pairs = set()
        for arm, piece in held:
            held_pairs |= {(f, b) for f in self.finger_bodies[arm] for b in self.piece_bodies[piece]}
        name = lambda b: mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or str(b)
        allp = set().union(*self.piece_bodies.values())
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.dist >= -margin:
                continue
            b1, b2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
            if (b1, b2) in held_pairs or (b2, b1) in held_pairs:
                continue
            if b1 in allp and b2 in allp:
                continue
            out.append((name(b1), name(b2), round(float(c.dist), 4)))
        return out


def rows_from_waypoints(waypoints, layout, speed=.25, min_steps=1, dt=.025):
    """Joint waypoints [{"left": q6, "right": q6, "lg": native, "rg": native}] -> Session rows (native gripper space):
    linear in joint space, each segment at most `speed` rad/s on its fastest arm joint (grippers at most 0.02 m/s)."""
    cols = {name: i for i, name in enumerate(layout)}
    def row_of(w):
        r = np.zeros(len(layout), np.float64)
        for arm in ARMS:
            for j, jn in enumerate(JOINTS):
                r[cols[f"{arm}/{jn}"]] = w[arm][j]
            r[cols[f"{arm}_native_gripper"]] = w["lg" if arm == "left" else "rg"]
        return r
    rows = [row_of(waypoints[0])]
    for w in waypoints[1:]:
        a, b = rows[-1], row_of(w)
        arm_cols = [cols[f"{arm}/{jn}"] for arm in ARMS for jn in JOINTS]
        g_cols = [cols[f"{arm}_native_gripper"] for arm in ARMS]
        n = max(min_steps, int(np.ceil(np.max(np.abs(b[arm_cols] - a[arm_cols])) / (speed * dt))),
                int(np.ceil(np.max(np.abs(b[g_cols] - a[g_cols])) / (.02 * dt))))
        for u in np.linspace(0., 1., n + 1)[1:]:
            rows.append(a + u * (b - a))
    return np.asarray(rows, np.float32)


def solve_path(kin, arm, targets, seed, *, anchor_every=6, n_random=12, max_step=.6, rng_seed=0, wrist=WRIST_SAFE,
               tol=(1.5e-3, .02)):
    """Joint path of one arm through a list of TCP targets (4x4), choosing among IK branches globally.

    Anchors every `anchor_every` targets get several IK solutions (from the previous anchor's solutions, the start
    seed and random seeds); a dynamic programme picks one per anchor minimising the largest joint step (steps over
    max_step rad between anchors are infeasible); the targets between anchors are then tracked from the chosen
    solutions. -> (joints per target, report)."""
    rng = np.random.default_rng(rng_seed)
    idx = list(range(0, len(targets), anchor_every))
    if idx[-1] != len(targets) - 1:
        idx.append(len(targets) - 1)
    lo, hi = LIMITS[:, 0], LIMITS[:, 1]
    layers = []
    prev = [np.asarray(seed, float)]
    for a in idx:
        sols = []
        seeds = list(prev) + [np.asarray(seed, float)] + list(rng.uniform(lo, hi, size=(n_random, 6)))
        for sd in seeds:
            q, ep, er = kin.ik(arm, targets[a], sd, wrist=wrist)
            if ep < tol[0] and er < tol[1] and not any(np.abs(q - s).max() < 1e-2 for s in sols):
                sols.append(q)
        layers.append(sols)
        prev = sols if sols else prev
    # dynamic programme over the layers (cost: the largest single-joint step; the first layer must start at `seed`)
    start = np.asarray(seed, float)
    c0 = np.array([np.abs(q - start).max() for q in layers[0]]) if layers[0] else np.array([])
    cost = [np.where(c0 <= max_step, c0, np.inf)]
    back = [None]
    for L in range(1, len(layers)):
        c = np.full(len(layers[L]), np.inf)
        b = np.full(len(layers[L]), -1)
        for j, q in enumerate(layers[L]):
            for i, p in enumerate(layers[L - 1]):
                step = np.abs(q - p).max()
                if step > max_step:
                    continue
                v = max(cost[L - 1][i], step)
                if v < c[j]:
                    c[j], b[j] = v, i
        cost.append(c)
        back.append(b)
    report = dict(anchors=len(idx), solutions=[len(x) for x in layers])
    if not len(cost[-1]) or not np.isfinite(cost[-1]).any():
        # the deepest layer still reachable
        reach = max((L for L in range(len(cost)) if len(cost[L]) and np.isfinite(cost[L]).any()), default=-1)
        report.update(ok=False, reached_anchor=reach, fail_target=idx[min(reach + 1, len(idx) - 1)])
        return None, report
    j = int(np.argmin(cost[-1]))
    chosen = [None] * len(layers)
    for L in range(len(layers) - 1, -1, -1):
        chosen[L] = layers[L][j]
        if L > 0:
            j = int(back[L][j])
    out, worst = [], [0., 0.]
    q = start
    for n, a in enumerate(idx):
        lo_i = 0 if n == 0 else idx[n - 1] + 1
        for t in range(lo_i, a):
            q, ep, er = kin.ik(arm, targets[t], q, prior=q, wrist=wrist)
            worst = [max(worst[0], ep), max(worst[1], er)]
            out.append(q.copy())
        q = chosen[n]
        out.append(q.copy())
    steps = np.abs(np.diff(np.asarray(out), axis=0)).max() if len(out) > 1 else 0.
    report.update(ok=bool(worst[0] < 3e-3 and worst[1] < .04), worst=worst, max_anchor_step=float(cost[-1].min()),
                  max_target_step=float(steps))
    return out, report
