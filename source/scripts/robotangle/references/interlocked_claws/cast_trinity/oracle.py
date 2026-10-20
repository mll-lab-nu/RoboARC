import numpy as np
# Oracle code for `rcb_topogym.py run-oracle --task interlocked_claws --code <this file>` on the Cast Trinity setting
# (interlocked_claws_cast_trinity; docs/rcb/families/interlocked_claws_cast_trinity.md). It recorded the -i01 reference
# (scripts/robotangle/oracles/interlocked-claws-cast-trinity-i01). run-oracle runs with the repository as the working
# directory; the plan (plan.json, plan.npz) and the two-piece planner it imports (the other modules here, the design
# record's code as it ran) sit next to this file. Input: a list of steps:
#   follow   both hands hold; per 40 Hz control step both arms' planned joints (Q) and the servoed piece's planned pose
#            in the other held piece's frame (XR); the servo arm integrates that pose error through its Jacobian (as
#            cast_abc_oracle.py), the other arm integrates its joint error.
#   clear    a held piece away from one it was just separated from, along the straight move (26 directions in the other
#            piece's frame) that leaves the largest gap and keeps the pair unlinked.
#   release  one hand opens to the piece's thickness + an extra opening (the family's assisted grasp releases); both arms
#            hold while the released piece settles (steps); optionally the hand then leaves along its contacts' normals.
#   retreat  one hand moves its TCP straight back along its approach axis (metres).
#   carry    one hand moves its piece (with whatever hangs on it) by a rigid motion given in the world (4x4 delta, or a
#            distance along the last separation's way out), the TCP following the piece (online IK every control step).
#   take     one hand takes a piece where it actually is (measured; held by the other hand or hanging): the first of the
#            piece's valid grasps that the arm reaches and whose open hand, straight approach and joint-space transit
#            touch nothing (checked on a scratch copy of the simulation data), and, if asked, from which the next
#            separation can be planned; transit, approach, closing (the assisted grasp acquires on bilateral contact).
#   pairplan closed loop: a separation planned from where the two held pieces actually are (RRT with both hands at their
#            actual grasps as obstacles), followed by one arm with a servo of its piece relative to the other, then a
#            push along the passage.
#   place    one hand carries its piece on along the last push, up, over to a point and down to just above the table,
#            opens and backs off.
#   hold     both arms hold their joints (steps).
# Only joint targets and gripper targets are commanded; no pose is ever set.
import itertools, json, os, sys, time
import mujoco
from scipy.spatial.transform import Rotation as R, Slerp
from topogym.assets.geometry import cast_trinity as ct

HERE = "scripts/robotangle/references/interlocked_claws/cast_trinity"
PLAN = json.loads(open(os.environ.get("TRI_PLAN", HERE + "/plan.json")).read())
NP = np.load(os.environ.get("TRI_PLAN_NPZ", HERE + "/plan.npz"))
os.environ.setdefault("ALOHA_XML", os.path.join(os.environ.get("TOPOGYM_ASSET_ROOT", "assets"), "robots", "aloha", "aloha.xml"))
GRIP, OPEN_EXTRA, FINGER_OFFSET = .002, .020, .0157
G_HOLD, K_SERVO, LIM, RATE = .10, .10, .05, .003
LIM_ROUTE = .15                                       # rad: the correction limit while following a separation route
PRE = .06                                             # m: the straight approach of a grasp
CMD_STEP = .035                                       # rad: the largest change of an arm joint's command per step
STEP_MAX = .015                                       # rad per control step for online IK targets
V_MOVE, W_MOVE = .015, .15                            # m/s, rad/s for online straight-line moves
s = session
w = s.env._world
m = w.model
iL, iR = list(range(0, 6)), list(range(7, 13))
SIDE = {"left": slice(0, 6), "right": slice(6, 12)}
NAME = {"T": "trinity_t", "H": "trinity_h", "B": "trinity_b"}
tau, fric, lo, hi = [], [], [], []
for n in [s.action_layout[i] for i in iL + iR]:
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    a = next(k for k in range(m.nu) if m.actuator_trnid[k, 0] == j)
    tau.append(m.dof_damping[m.jnt_dofadr[j]] / m.actuator_gainprm[a, 0])
    fric.append(m.dof_frictionloss[m.jnt_dofadr[j]] / m.actuator_gainprm[a, 0])
    lo.append(m.jnt_range[j, 0]); hi.append(m.jnt_range[j, 1])
tau, fric, lo, hi = map(np.array, (tau, fric, lo, hi))
state = dict(I=np.zeros(12), S=np.zeros(12), grip=[GRIP, GRIP], prev=None, ended=False, worst=0., lim=LIM)
report = dict(steps=[])


def body_T(name):
    b = w.read_state()["bodies"][name]
    T = np.eye(4)
    T[:3, :3] = R.from_quat(np.asarray(b["quaternion_wxyz"])[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = b["position_m"]
    return T


def tcp_T(side, q12):
    return s.robot._fk_full(s.robot.q_full(q12))[side + "_tcp"]


def jacobian(side, q12):
    T0, sl = tcp_T(side, q12), SIDE[side]
    J = np.zeros((6, 6))
    for k in range(6):
        q = q12.copy()
        q[sl.start + k] += 1e-6
        T1 = tcp_T(side, q)
        J[:3, k] = (T1[:3, 3] - T0[:3, 3]) / 1e-6
        J[3:, k] = R.from_matrix(T1[:3, :3] @ T0[:3, :3].T).as_rotvec() / 1e-6
    return J, T0


def ik(side, T, q12, iters=40):
    q = q12.copy()
    sl = SIDE[side]
    for _ in range(iters):
        J, T0 = jacobian(side, q)
        e = np.r_[T[:3, 3] - T0[:3, 3], R.from_matrix(T[:3, :3] @ T0[:3, :3].T).as_rotvec()]
        if np.linalg.norm(e[:3]) < 2e-5 and np.linalg.norm(e[3:]) < 2e-4:
            break
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e)
        q[sl] = np.clip(q[sl] + np.clip(dq, -.2, .2), lo[sl], hi[sl])
    return q


def measured():
    q = s.joint_positions()
    return np.r_[q[iL], q[iR]]


# hypothetical arm configurations are checked on a scratch copy of the simulation data (the live data is never touched)
d2 = mujoco.MjData(m)
QADR = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in [s.action_layout[i] for i in iL + iR]])
FADR = {side: [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, side + "/" + f)] for f in ("left_finger", "right_finger")]
        for side in SIDE}
OWNER = []
for g_ in range(m.ngeom):
    b_ = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g_]) or "").split("::")[0]
    OWNER.append(next((k for k, v in NAME.items() if v == b_), b_.split("/")[0]))
OWNER = np.array(OWNER)
TG = [g_ for g_ in range(m.ngeom) if OWNER[g_] == "table"][0]
TABLE_TOP = float(w.data.geom_xpos[TG][2] + m.geom_size[TG][2])


def touches(side, q12, grip, piece=None):
    """what side's arm and hand (fingers at grip, native) would penetrate with its joints at q12 and everything else where
    it is now: a set of owners ("T" "H" "B", the other arm, "table", ...); the piece about to be taken counts from 0.5 mm"""
    d2.qpos[:] = w.data.qpos
    d2.qpos[QADR[SIDE[side]]] = q12[SIDE[side]]
    d2.qpos[FADR[side]] = grip
    mujoco.mj_kinematics(m, d2)
    mujoco.mj_collision(m, d2)
    n = int(d2.ncon)
    g = np.asarray(d2.contact.geom[:n]).reshape(-1, 2)
    dist = np.asarray(d2.contact.dist[:n])
    a, b = OWNER[g[:, 0]], OWNER[g[:, 1]]
    o = np.where(a == side, b, a)
    hit = ((a == side) != (b == side)) & (dist < np.where(o == piece, -5e-4, 0.))
    return set(o[hit])


FREE = {k: m.jnt_qposadr[m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, NAME[k] + "::" + NAME[k])]] for k in NAME}


def scene_hits(q12, poses, held):
    """both arms at q12 and the given pieces at world poses (the rest where they are now): what the arms would
    penetrate by more than 1 mm besides the piece each holds (held: side -> piece), as a set of (side, owner)"""
    d2.qpos[:] = w.data.qpos
    d2.qpos[QADR] = q12
    for k, Tk in poses.items():
        a = FREE[k]
        d2.qpos[a:a + 3] = Tk[:3, 3]
        d2.qpos[a + 3:a + 7] = R.from_matrix(Tk[:3, :3]).as_quat()[[3, 0, 1, 2]]
    mujoco.mj_kinematics(m, d2)
    mujoco.mj_collision(m, d2)
    n = int(d2.ncon)
    g = np.asarray(d2.contact.geom[:n]).reshape(-1, 2)
    deep = np.asarray(d2.contact.dist[:n]) < -1e-3
    out = set()
    for a, b in zip(OWNER[g[deep, 0]], OWNER[g[deep, 1]]):
        for sd in held:
            if a == sd and b != sd and b != held[sd]:
                out.add((sd, str(b)))
            if b == sd and a != sd and a != held[sd]:
                out.add((sd, str(a)))
    return out


def reach(side, T, q12, iters=300):
    """the robot's bounded least-squares ik from q12 (then from its reset pose): (joints, position error m, rotation
    error rad)"""
    f = s.robot.tcp_frames[side]
    qq, err = s.robot.ik({f: np.r_[T[:3, 3], R.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]]}, q12, 1e-4, 1e-3, iters)
    return np.asarray(qq, dtype=np.float64), float(err[f][0]), float(err[f][1])


def reach_any(side, T, q12):
    """reach() from q12, then (if that misses) from q12 with the forearm rolled by +-90 deg and / or the wrist by 180 deg"""
    j = SIDE[side].start
    best = None
    for d4, d6 in ((0, 0), (np.pi / 2, 0), (-np.pi / 2, 0), (0, np.pi), (np.pi / 2, np.pi), (-np.pi / 2, np.pi)):
        q0 = q12.copy()
        q0[j + 3] = (q0[j + 3] + d4 + np.pi) % (2 * np.pi) - np.pi
        q0[j + 5] = (q0[j + 5] + d6 + np.pi) % (2 * np.pi) - np.pi
        r = reach(side, T, np.clip(q0, lo, hi), 120)
        if best is None or r[1] + .05 * r[2] < best[1] + .05 * best[2]:
            best = r
        if r[1] < 1e-3 and r[2] < .02:
            break
    return best


def clear_path(side, qa, qb, grip, piece=None):
    """the joint-space line qa -> qb, sampled every 0.02 rad, touches nothing (qa itself is not checked)"""
    n = max(4, int(np.ceil(np.abs(qb - qa).max() / .02)))
    return all(not touches(side, qa + (qb - qa) * t, grip, piece) for t in np.linspace(0, 1, n + 1)[1:])


def transit(side, qa, qb, grip, piece=None):
    """a joint-space route qa -> qb that touches nothing: direct, or through points 10 or 20 cm above both ends, or 8 cm
    back along each end's approach axis and 5 cm up"""
    if clear_path(side, qa, qb, grip, piece):
        return [qb]
    for up, back in ((.10, 0), (.20, 0), (.05, .08)):
        Ta, Tb = tcp_T(side, qa), tcp_T(side, qb)
        Ta[:3, 3] += [0, 0, up] - back * Ta[:3, 0]
        Tb[:3, 3] += [0, 0, up] - back * Tb[:3, 0]
        qa2, e1, _ = reach(side, Ta, qa)
        qb2, e2, _ = reach(side, Tb, qb)
        if max(e1, e2) < 2e-3 and clear_path(side, qa, qa2, grip, piece) and clear_path(side, qa2, qb2, grip, piece) \
                and clear_path(side, qb2, qb, grip, piece):
            return [qa2, qb2, qb]
    return None


def away_from_contacts(side):
    """the mean direction (unit, world) in which side's hand leaves the pieces it touches now (live contacts, read only),
    or None"""
    d = w.data
    n = int(d.ncon)
    g = np.asarray(d.contact.geom[:n]).reshape(-1, 2)
    a, b = OWNER[g[:, 0]], OWNER[g[:, 1]]
    v = np.zeros(3)
    for i in range(n):
        if a[i] == side and b[i] in NAME:
            v -= d.contact.frame[i][:3]                  # the normal points from geom1 (the hand) to geom2 (the piece)
        elif b[i] == side and a[i] in NAME:
            v += d.contact.frame[i][:3]
    return v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-9 else None


def free_hand(side, q, grip):
    """if side's hand (fingers at grip) touches something where it is: a straight TCP move back along its approach axis
    and / or up (3 or 6 cm) to the nearest pose that touches nothing; returns the new targets"""
    qm = measured()
    if not touches(side, qm, grip):
        return q
    T0 = tcp_T(side, qm)
    for back, up in ((.03, 0), (0, .03), (.03, .03), (.06, 0), (0, .06), (.06, .06)):
        T1 = T0.copy()
        T1[:3, 3] += -back * T0[:3, 0] + [0, 0, up]
        q1, ep, er = reach(side, T1, qm)
        if ep < 1e-3 and not touches(side, q1, grip):
            return drive_tcp(side, lin(tcp_T(side, q), T1, .01, .1), q)
    return q


def joint_move(q, goal, side):
    """side's joints from q to goal in a straight joint-space line (at most STEP_MAX / 2 per step)"""
    sl = SIDE[side]
    nb = max(20, int(np.ceil(np.abs(goal[sl] - q[sl]).max() / (STEP_MAX / 2))))
    start = q.copy()
    for b in range(1, nb + 1):
        q = start.copy()
        q[sl] = start[sl] + (goal[sl] - start[sl]) * b / nb
        if command(q):
            break
    return q


MODS = {}


def planner(paths):
    """the two-piece planner (the modules next to this file, or the step's paths), imported once"""
    if not MODS:
        sys.path[:0] = [p_ for p_ in (paths or [os.path.abspath(HERE)]) if p_ not in sys.path]
        import geom12, puzzle, k1, chain34, tilt_follow, grasp_scan12
        MODS.update(g=geom12, P=puzzle.Puzzle(geom12), k1=k1, plan_pair=chain34.plan_pair, HAND=chain34.HAND,
                    resample=tilt_follow.resample, candidates=grasp_scan12.candidates)
    return MODS


CTR = {}


def centre(piece):
    """the piece's bounding-box centre in its own frame"""
    if piece not in CTR:
        V = np.asarray(ct.solid(piece).vertices)
        CTR[piece] = (V.min(0) + V.max(0)) / 2
    return CTR[piece]


def hand_verts(G, width):
    """the hand's collision hulls (vertices) with its TCP at G, fingers open to width"""
    return [V @ G[:3, :3].T + G[:3, 3] for V in MODS["HAND"].hull_verts(width)]


def held_verts(piece):
    """the hand that holds piece, at its actual grasp: hull vertices in the piece's frame"""
    side, width = state["holder"][piece]
    return hand_verts(np.linalg.inv(body_T(NAME[piece])) @ tcp_T(side, measured()), width)


BACK = .010                                            # m: routes end this far back out of the passage, along its axis
MARGIN = .0005                                         # m: the least gap between the two pieces along a planned route


def plan_from_here(X, Y, hands, secs, seeds):
    """X's two-piece route relative to Y from where the pieces are (poses in Y's frame) to BACK short of the lying
    passage, or None"""
    Ym = body_T(NAME[Y])
    poses = {k: np.linalg.inv(Ym) @ body_T(NAME[k]) for k in "THB"}
    for seed in seeds:
        path = MODS["plan_pair"](MODS["P"], poses, X, Y, hands[X], hands[Y], secs, seed, BACK, MARGIN)
        if path is not None:
            return path
    return None


def links():
    def lk(A, B):
        a0, a1 = A, np.roll(A, -1, axis=0)
        b0, b1 = B, np.roll(B, -1, axis=0)
        r00, r01, r10, r11 = b0[None] - a0[:, None], b1[None] - a0[:, None], b0[None] - a1[:, None], b1[None] - a1[:, None]
        un = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
        n1, n2, n3, n4 = un(np.cross(r00, r01)), un(np.cross(r01, r11)), un(np.cross(r11, r10)), un(np.cross(r10, r00))
        sg = np.sign(np.sum(np.cross((a1 - a0)[:, None], (b1 - b0)[None]) * r00, -1))
        return float(np.sum(sum(np.arcsin(np.clip(np.sum(p * q, -1), -1, 1)) for p, q in ((n1, n2), (n2, n3), (n3, n4), (n4, n1))) * sg) / (4 * np.pi))
    L = {k: (lambda T: ct.centerline(k) @ T[:3, :3].T + T[:3, 3])(body_T(NAME[k])) for k in "THB"}
    return {a + b: int(round(lk(L[a], L[b]))) for a, b in (("T", "H"), ("T", "B"), ("H", "B"))}


def command(target, servo=None, hold=("left", "right")):
    """one control step toward target (12 joints) with feedforwards and corrections. servo = (side, piece, other, X):
    that arm servos piece's pose to X in other's frame. Returns True at the episode's end."""
    if state["prev"] is None:
        state["prev"] = target.copy()
    vel = (target - state["prev"]) * s.control_hz
    state["prev"] = target.copy()
    cmd = np.clip(target + tau * vel + fric * np.tanh(vel / .01) + state["I"] + state["S"], lo, hi)
    if state.get("cmd") is not None:                     # at most CMD_STEP a joint per control step (the standard
        cmd = state["cmd"] + np.clip(cmd - state["cmd"], -CMD_STEP, CMD_STEP)   # service's limit is 0.04 rad)
    state["cmd"] = cmd
    a = np.zeros(14)
    a[iL], a[6], a[iR], a[13] = cmd[:6], state["grip"][0], cmd[6:], state["grip"][1]
    if s.step(a):
        state["ended"] = True
        return True
    qm = measured()
    for side in hold:
        sl = SIDE[side]
        state["I"][sl] = np.clip(state["I"][sl] + G_HOLD * (target[sl] - qm[sl]), -state["lim"], state["lim"])
    if servo is not None:
        side, piece, other, X = servo
        Om, Pm = body_T(NAME[other]), body_T(NAME[piece])
        Pd = Om @ X
        dR = R.from_matrix(Pd[:3, :3] @ Pm[:3, :3].T).as_rotvec()
        J, T0 = jacobian(side, qm)
        tw = np.r_[Pd[:3, 3] - Pm[:3, 3] + np.cross(dR, T0[:3, 3] - Pm[:3, 3]), dR]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), tw)
        sl = SIDE[side]
        state["S"][sl] = np.clip(state["S"][sl] + np.clip(K_SERVO * dq, -RATE, RATE), -state["lim"], state["lim"])
        state["worst"] = max(state["worst"], float(np.linalg.norm((np.linalg.inv(Om) @ Pm)[:3, 3] - X[:3, 3])))
    return False


def lin(A, B, v=V_MOVE, wv=W_MOVE):
    ang = R.from_matrix(B[:3, :3] @ A[:3, :3].T).magnitude()
    n = max(2, int(np.ceil(max(np.linalg.norm(B[:3, 3] - A[:3, 3]) / v, ang / wv) * s.control_hz)) + 1)
    sl = Slerp([0, 1], R.from_matrix(np.stack([A[:3, :3], B[:3, :3]])))
    out = []
    for t in np.linspace(0, 1, n)[1:]:
        T = np.eye(4)
        T[:3, :3] = sl(t).as_matrix()
        T[:3, 3] = (1 - t) * A[:3, 3] + t * B[:3, 3]
        out.append(T)
    return out


def drive_tcp(side, poses, q, servo=None, tail=30):
    """move one arm's TCP through poses (online IK, at most STEP_MAX per joint per step); the other arm holds."""
    for k, T in enumerate(poses + [poses[-1]] * tail):
        goal = ik(side, T, q)
        q = q.copy()
        sl = SIDE[side]
        q[sl] += np.clip(goal[sl] - q[sl], -STEP_MAX, STEP_MAX)
        if command(q, servo=servo):
            return q
        if k >= len(poses) and np.abs(goal[sl] - q[sl]).max() < 1e-4:
            break
    return q


def rel(a, b):
    X = np.linalg.inv(body_T(NAME[a])) @ body_T(NAME[b])
    return [round(1000 * float(v), 1) for v in X[:3, 3]] + [round(float(np.degrees(v)), 1) for v in R.from_matrix(X[:3, :3]).as_rotvec()]


def fingers():
    """per hand: the grasp manager's phase / rejection, and each finger's contacts (owner: deepest penetration, mm)"""
    out = {}
    gh = (w.read_state().get("grasp") or {}).get("hands") or {}
    d = w.data
    n = int(d.ncon)
    geom, dist = np.asarray(d.contact.geom[:n]), np.asarray(d.contact.dist[:n])
    for side in ("left", "right"):
        h = gh.get(side) or {}
        f = {}
        for fn in ("left_finger_link", "right_finger_link"):
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, side + "/" + fn)
            c = {}
            for (g1, g2), dd in zip(geom, dist):
                b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
                if b in (b1, b2):
                    o = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2 if b1 == b else b1).split("::")[0]
                    c[o] = round(max(c.get(o, -1e9), -1000 * float(dd)), 2)
            f[fn[0]] = c
        out[side] = dict(phase=h.get("phase"), rejection=h.get("rejection"), intent=h.get("intent"), fingers=f)
    return out


def piece_contacts():
    """live contacts of the pieces with anything but themselves and the hand holding them: pair -> [count, deepest
    penetration mm]; and the servo / hold offsets (largest |rad| per arm)"""
    d = w.data
    n = int(d.ncon)
    g = np.asarray(d.contact.geom[:n]).reshape(-1, 2)
    dist = np.asarray(d.contact.dist[:n])
    held = {v[0]: k for k, v in state["holder"].items()}
    out = {"offsets": {sd: [round(float(np.abs(state["S"][SIDE[sd]]).max()), 3), round(float(np.abs(state["I"][SIDE[sd]]).max()), 3)]
                       for sd in SIDE}}
    for (a, b), dd in zip(zip(OWNER[g[:, 0]], OWNER[g[:, 1]]), dist):
        if (a in NAME or b in NAME) and a != b and held.get(a) != b and held.get(b) != a:
            k_ = "-".join(sorted((str(a), str(b))))
            c = out.setdefault(k_, [0, 0.])
            c[0] += 1
            c[1] = round(max(c[1], -1000 * float(dd)), 2)
    return out


def note(kind, **kw):
    kw.update(step=s.n_steps, kind=kind, links=links(), H_in_T=rel("T", "H"), B_in_H=rel("H", "B"),
              held={side: s.object_in_hand(side) for side in ("left", "right")})
    if os.environ.get("TRI_FINGERS"):
        kw["hands"] = fingers()
    report["steps"].append(kw)
    print(json.dumps(kw), file=sys.stderr, flush=True)


t0 = time.time()
q = measured()
state["prev"] = q.copy()
state["cmd"] = np.asarray(s.hold_action(), dtype=np.float64)[iL + iR]     # the birth targets the drives hold now
state["holder"] = {k: tuple(v) for k, v in PLAN.get("holders", {}).items()}     # piece: (hand, grasp width)
state["away"] = np.zeros(3)
STOP = int(os.environ.get("TRI_STOP_AFTER", "999"))       # 只跑步骤表的前几步（诊断用）
for si, step in enumerate(PLAN["steps"]):
    if state["ended"] or si > STOP:
        break
    kind = step["kind"]
    if kind == "hold":
        for _ in range(step["steps"]):
            if command(q):
                break
    elif kind == "follow":
        Q, X = NP[step["Q"]], NP[step["XR"]]
        sv = step["servo"]                                  # [side, piece, other]
        start = measured()
        nb = max(20, int(np.ceil(np.abs(Q[0] - start).max() / (STEP_MAX / 2))))
        for b in range(1, nb + 1):                          # blend from where the arms are onto the plan
            if command(start + (Q[0] - start) * b / nb, servo=(sv[0], sv[1], sv[2], X[0])):
                break
        errs = []
        for k in range(len(Q)):
            if command(Q[k], servo=(sv[0], sv[1], sv[2], X[k])):
                break
            if k % 10 == 0 or k >= len(Q) - 80:          # 伺服的那件相对另一件：和计划差多少（位置 mm、转角度）
                Rel = np.linalg.inv(body_T(NAME[sv[2]])) @ body_T(NAME[sv[1]])
                errs.append((k, 1000 * float(np.linalg.norm(Rel[:3, 3] - X[k][:3, 3])),
                             float(np.degrees(R.from_matrix(Rel[:3, :3].T @ X[k][:3, :3]).magnitude()))))
        q = Q[-1].copy()
        e = np.array([[a, b, c] for a, b, c in errs])
        note("follow errors", name=step.get("name", ""), max_mm=round(float(e[:, 1].max()), 2), max_deg=round(float(e[:, 2].max()), 2),
             last80_max_mm=round(float(e[e[:, 0] >= len(Q) - 80][:, 1].max()), 2),
             last80_max_deg=round(float(e[e[:, 0] >= len(Q) - 80][:, 2].max()), 2),
             trace=[[int(a), round(b, 2), round(c, 2)] for a, b, c in errs[::max(1, len(errs) // 25)]])
    elif kind == "release":
        gi = 0 if step["side"] == "left" else 1
        state["grip"][gi] = (step["width"] + step.get("extra", OPEN_EXTRA) + FINGER_OFFSET) / 2
        state["holder"] = {k: v for k, v in state["holder"].items() if v[0] != step["side"]}
        q = measured() if step.get("from_measured") else q
        for _ in range(step["steps"]):
            if command(q):
                break
        if step.get("separate"):                             # off the pieces still touching the hand, along the contacts'
            n_ = away_from_contacts(step["side"])            # normals, slowly (a piece resting on it is not dragged)
            if n_ is not None:
                T0 = tcp_T(step["side"], q)
                T1 = T0.copy()
                T1[:3, 3] += step["separate"] * n_
                q = drive_tcp(step["side"], lin(T0, T1, .005, .05), q)
            note("separated", side=step["side"], normal=None if n_ is None else [round(float(v), 3) for v in n_])
    elif kind == "retreat":
        qm = q.copy()
        T0 = tcp_T(step["side"], qm)
        T1 = T0.copy()
        T1[:3, 3] -= step["distance"] * T0[:3, 0]
        q = drive_tcp(step["side"], lin(T0, T1), qm)
    elif kind == "carry":
        side, piece = step["side"], step["piece"]
        P0 = body_T(NAME[piece])
        G_act = np.linalg.inv(P0) @ tcp_T(side, q)
        D = np.asarray(step["delta"]) if "delta" in step else np.eye(4)   # world rigid motion of the piece (about its pose)
        if step.get("away"):                                 # or that far along the last separation's push
            D[:3, 3] = step["away"] * state["away"]
        q = drive_tcp(side, [Pk @ G_act for Pk in lin(P0, D @ P0, step.get("v", V_MOVE), step.get("w", W_MOVE))], q)
    elif kind == "clear":
        # a held piece away from one it was just separated from (held by the other hand): of 26 straight translations of
        # `dist` m along the axes and diagonals of the other piece's frame, the one that leaves the largest gap between
        # the two (their exact geometry at the measured poses) and keeps the pair unlinked all the way (it may slide
        # along the other piece at first, as in the push)
        side, piece, other = step["side"], step["piece"], step["other"]
        planner(step["paths"])
        Om = body_T(NAME[other])
        base = {k: np.linalg.inv(Om) @ body_T(NAME[k]) for k in "THB"}
        key = "".join(sorted(piece + other, key="THB".index))
        gap0, best = MODS["P"].clearance(base, ((key[0], key[1]),))[key], None
        for dd in itertools.product((-1, 0, 1), repeat=3):
            if not any(dd):
                continue
            u = np.asarray(dd, float) / np.linalg.norm(dd)
            gaps, linked = [], False
            for t in (.2, .4, .6, .8, 1.):
                poses = dict(base)
                poses[piece] = base[piece].copy()
                poses[piece][:3, 3] += u * t * step["dist"]
                gaps.append(MODS["P"].clearance(poses, ((key[0], key[1]),))[key])
                linked = linked or MODS["P"].links(poses)[key] != 0
            if linked:
                continue
            if best is None or gaps[-1] > best[0]:
                best = (gaps[-1], Om[:3, :3] @ u)                  # in the world
        note("clear", piece=piece, other=other, gap_before_mm=round(1000 * gap0, 2),
             gap_after_mm=None if best is None else round(1000 * best[0], 2),
             direction=None if best is None else [round(float(v), 2) for v in best[1]])
        if best is not None:
            P0 = body_T(NAME[piece])
            G_act = np.linalg.inv(P0) @ tcp_T(side, q)
            P1 = P0.copy()
            P1[:3, 3] += best[1] * step["dist"]
            q = drive_tcp(side, [Pk @ G_act for Pk in lin(P0, P1, .005, .05)], q)
    elif kind == "present":
        # one hand moves its piece (what hangs on it comes along) by the translation, from a 3 x 3 x 3 grid of `step` m,
        # that leaves the most of the hanging piece's valid grasps (a sample) within the other arm's reach (ik only)
        side, held, piece, taker = step["side"], step["held"], step["piece"], step["taker"]
        planner(step["paths"])
        cands = [c for c in MODS["candidates"](MODS["g"], piece) if c[0] in set(step["valid"])]
        cands = cands[::max(1, len(cands) // step.get("sample", 32))]
        P0, H0, qm = body_T(NAME[held]), body_T(NAME[piece]), measured()
        G_act = np.linalg.inv(P0) @ tcp_T(side, qm)
        best, t_p = None, time.time()
        for d in itertools.product(*[(0., -step.get("grid", .08), step.get("grid", .08))] * 3):
            Pd = P0.copy()
            Pd[:3, 3] += d
            if reach(side, Pd @ G_act, qm)[1] > 1e-3:
                continue
            Hd = H0.copy()
            Hd[:3, 3] += d
            n = sum(1 for _, Gp, _ in cands if (lambda r: r[1] < 1e-3 and r[2] < .02)(reach(taker, Hd @ Gp, qm, 100)))
            if best is None or n > best[0]:
                best = (n, np.asarray(d))
        note("present", shift_mm=[round(1000 * float(v)) for v in best[1]], reachable=best[0], of=len(cands),
             s=round(time.time() - t_p, 1))
        Pd = P0.copy()
        Pd[:3, 3] += best[1]
        q = drive_tcp(side, [Pk @ G_act for Pk in lin(P0, Pd, .01, .1)], q)
        for _ in range(step.get("settle", 60)):
            if command(q):
                break
    elif kind == "take":
        # a hand takes a piece where it is (held by the other hand, or hanging): the first candidate grasp (the preferred
        # one, then the piece's other valid grasps nearest the hand first) that the arm reaches (1 mm, 1 deg) and whose
        # open hand at the grasp, straight approach from PRE back and joint-space transit (direct, or through points above
        # both ends) touch nothing; with "plan", only one from which the next separation can be planned. Then the transit,
        # the approach (aimed at where the piece is then) and closing until the assisted grasp acquires.
        side, piece = step["side"], step["piece"]
        gi = 0 if side == "left" else 1
        planner(step["paths"])
        q = free_hand(side, q, state["grip"][gi])
        Pm, qm = body_T(NAME[piece]), measured()
        here = tcp_T(side, qm)[:3, 3]
        base = w.data.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, side + "/shoulder_link")]

        def outward(c):                                      # the approach axis along the line from the shoulder: reachable
            Tc = Pm @ c[1]
            v = Tc[:3, 3] - base
            return float(Tc[:3, 0] @ v / np.linalg.norm(v))
        cands = sorted([c for c in MODS["candidates"](MODS["g"], piece) if c[0] in set(step["valid"])],
                       key=lambda c: (c[0] != step.get("prefer"), c[0] not in step.get("prefer_list", ()), -outward(c)))
        chosen, checks, t_take, why, seen = None, 0, time.time(), {}, {}
        by_name = {c[0]: c for c in cands}

        def geo(c):
            """(why not, transit route, joints at the grasp) of candidate c"""
            if c[0] in seen:
                return seen[c[0]]
            name, Gp, width = c
            grip = (width + OPEN_EXTRA + FINGER_OFFSET) / 2
            Tg = Pm @ Gp
            qg, ep, er = reach_any(side, Tg, qm)
            hit = touches(side, qg, grip, piece)
            bad = "ik" if ep > 1e-3 or er > .02 else ("grasp:" + ",".join(sorted(hit)) if hit else "")
            qk = qg
            for t in np.linspace(0, 1, 7)[1:] if not bad else []:   # the approach, checked from the grasp backwards
                Tk = Tg.copy()
                Tk[:3, 3] -= t * PRE * Tg[:3, 0]
                qk, ep, er = reach(side, Tk, qk, 60)
                hit = touches(side, qk, grip, piece)
                if ep > 1e-3 or er > .02 or hit:
                    bad = "approach ik" if ep > 1e-3 or er > .02 else "approach:" + ",".join(sorted(hit))
                    break
            route = transit(side, qm, qk, grip, piece) if not bad else None
            seen[c[0]] = (bad or ("transit" if route is None else ""), route, qg)
            return seen[c[0]]

        def roll_room(qg):                                   # how far the forearm roll and the wrist rotation are from
            j = SIDE[side].start                             # their +-180 deg limits: room for the routes' turns
            return min(np.pi - abs(qg[j + 3]), np.pi - abs(qg[j + 5]))
        for c in cands:
            if time.time() - t_take > step.get("max_s", 300):
                break
            bad, route, qg = geo(c)
            if bad:
                why[bad] = why.get(bad, 0) + 1
                continue
            twin = by_name.get(c[0][:-1] + ("+" if c[0].endswith("-") else "-"))   # the same grasp turned 180 deg
            order = [c]
            if twin is not None and not geo(twin)[0]:
                order = sorted([c, twin], key=lambda c_: -roll_room(geo(c_)[2]))
            for c_ in order:
                name, Gp, width = c_
                if step.get("plan"):
                    checks += 1
                    sp = step["plan"]
                    hv = {piece: hand_verts(Gp, width)}
                    hv.update({k: held_verts(k) for k in (sp["X"], sp["Y"]) if k != piece})
                    if plan_from_here(sp["X"], sp["Y"], hv, sp.get("secs", 40), (1,)) is None:
                        why["no plan"] = why.get("no plan", 0) + 1
                        continue
                chosen = (name, Gp, width, (width + OPEN_EXTRA + FINGER_OFFSET) / 2, geo(c_)[1])
                break
            if chosen is not None or checks >= step.get("max_checks", 6):
                break
        note("take chosen", side=side, piece=piece, grasp=None if chosen is None else chosen[0], candidates=len(cands),
             rejected=why, s=round(time.time() - t_take, 1), tcp=[round(1000 * float(v), 1) for v in here],
             roll_room_rad=None if chosen is None else round(float(roll_room(seen[chosen[0]][2])), 2),
             piece_at=[round(1000 * float(v), 1) for v in Pm[:3, 3]])
        if chosen is None:
            state["failed"] = "no grasp on " + piece
            break
        name, Gp, width, grip, route = chosen
        state["grip"][gi] = grip
        for _ in range(20):
            if command(q):
                break
        for qt in route:
            q = joint_move(q, qt, side)
        Pm = body_T(NAME[piece])                             # the piece may have moved a little: aim at where it is now
        Tg = Pm @ Gp
        q = drive_tcp(side, lin(tcp_T(side, q), Tg, .01, .1), q)
        T0 = tcp_T(side, measured())
        aim = dict(tcp_err_mm=round(1000 * float(np.linalg.norm(T0[:3, 3] - Tg[:3, 3])), 2),
                   tcp_err_deg=round(float(np.degrees(R.from_matrix(Tg[:3, :3] @ T0[:3, :3].T).magnitude())), 2),
                   piece_moved_mm=round(1000 * float(np.linalg.norm(body_T(NAME[piece])[:3, 3] - Pm[:3, 3])), 2))
        state["grip"][gi] = GRIP
        for k in range(80):
            if command(q):
                break
            if k >= 8 and s.object_in_hand(side) == NAME[piece]:
                break
        state["holder"][piece] = (side, width)
        note("taken", side=side, piece=piece, grasp=name, **aim)
    elif kind == "pairplan":
        # closed loop: plan X's two-piece route relative to Y from where they actually are (RRT with both hands, at their
        # actual grasps, as obstacles: chain34.plan_pair), then push along the lying
        # passage's direction. One arm moves its piece along the route (online IK, servo of that piece relative to the
        # measured other one), the other holds: X's arm, or Y's (Y along the inverse route) when only that one reaches.
        planner(step["paths"])
        X, Y = step["piece"], step["other"]
        t_plan = time.time()
        path = plan_from_here(X, Y, {X: held_verts(X), Y: held_verts(Y)}, step.get("secs", 120), step.get("seeds", (1, 2, 3)))
        note("planned", piece=X, other=Y, found=path is not None, poses=0 if path is None else len(path),
             plan_s=round(time.time() - t_plan, 1))
        if path is None:
            state["failed"] = "no route " + X + Y
            break
        ref = MODS["resample"](path, dp=.00025, da=np.radians(.125))   # about 10 mm/s and 5 deg/s at 40 Hz
        state["lim"] = LIM_ROUTE                            # an arm carrying a piece through a long turn needs more (kept
                                                             # after: clipping the offsets back would jerk the arms)
        key = "".join(sorted(X + Y, key="THB".index))
        Ym, Xm = body_T(NAME[Y]), body_T(NAME[X])
        poses = {k: np.linalg.inv(Ym) @ body_T(NAME[k]) for k in "THB"}
        K, Dv, _ = MODS["k1"].lying(MODS["g"], MODS["P"], X, Y, MODS["P"].links(poses)[key])[0]   # the planner's passage
        back = BACK if np.linalg.norm(ref[-1][:3, 3] - (K[:3, 3] - BACK * Dv)) < 1e-6 else 0.
        push = [ref[-1]] * 40                                # hold aligned at the route's end, then straight in along
        for i in range(1, int(round((back + .032) / .00025)) + 1):   # the axis at 10 mm/s to 32 mm past the passage
            Tp = ref[-1].copy()
            Tp[:3, 3] += Dv * i * .00025
            push.append(Tp)
        qm = measured()
        sx, sy = state["holder"][X][0], state["holder"][Y][0]
        GX, GY = np.linalg.inv(Xm) @ tcp_T(sx, qm), np.linalg.inv(Ym) @ tcp_T(sy, qm)
        pivot = ((Xm @ np.r_[centre(X), 1.])[:3] + (Ym @ np.r_[centre(Y), 1.])[:3]) / 2
        more = []                                            # two more 32 mm pushes, if the pair is still linked after it
        for i in range(1, 129):
            Tp = push[-1].copy()
            Tp[:3, 3] += Dv * i * .0005
            more.append(Tp)
        route = ref + push + more
        motion, prev = [], None                              # X's world motion G (G X0 = Y0 Xr) if Y stayed still,
        for Xr in route:                                     # its rotation vector unwrapped along the route
            G = Ym @ Xr @ np.linalg.inv(Xm)
            rv = R.from_matrix(G[:3, :3]).as_rotvec()
            if prev is not None and np.linalg.norm(rv) > 1e-9:
                alt = rv - 2 * np.pi * rv / np.linalg.norm(rv)
                if np.linalg.norm(alt - prev) < np.linalg.norm(rv - prev):
                    rv = alt
            prev = rv
            motion.append((G, rv))

        def shared(s_):
            """both TCP targets and the servo pose along the route when X takes the fraction s_ of G (about the pieces'
            midpoint) and Y the rest (s_ = 1: only X's arm moves, 0: only Y's); the relative route is unchanged"""
            out = []
            for (G, rv), Xr in zip(motion, route):
                M = np.eye(4)
                M[:3, :3] = R.from_rotvec(s_ * rv).as_matrix()
                M[:3, 3] = pivot - M[:3, :3] @ pivot + s_ * (G[:3, :3] @ pivot + G[:3, 3] - pivot)
                out.append((M @ Xm @ GX, M @ np.linalg.inv(G) @ Ym @ GY, Xr))
            return out
        n_main = len(ref) + len(push)
        Xrel0 = np.linalg.inv(Ym) @ Xm

        def placed(c, yaw):
            """the whole held pair turned by yaw about the vertical through the pivot and moved by c"""
            W = np.eye(4)
            W[:3, :3] = R.from_rotvec([0., 0., yaw]).as_matrix()
            W[:3, 3] = pivot - W[:3, :3] @ pivot + np.asarray(c, float)
            return W

        Z = [k for k in "THB" if k not in (X, Y)][0]
        lk0 = links()
        Zm = body_T(NAME[Z])
        Zrides = Y if lk0["".join(sorted(Y + Z, key="THB".index))] else X if lk0["".join(sorted(X + Z, key="THB".index))] else None
        held_by = {sx: X, sy: Y}
        hits_seen = {}

        def worst_of(tg, cap=2.):
            """the largest ik residual (mm + deg) along tg, or 100 + where an arm would hit something else, or 50 + at a
            joint jump; and how far along tg it got (fraction) and why it stopped"""
            qq, worst = qm.copy(), 0.
            for i_, (TX, TY, _) in enumerate(tg):
                q_prev = qq.copy()
                for sd, T in ((sx, TX), (sy, TY)):
                    qq, ep, er = reach(sd, T, qq, 60)
                    worst = max(worst, 1000 * ep + np.degrees(er))
                if worst > cap:
                    return worst, i_ / len(tg), "ik"
                if np.abs(qq - q_prev).max() > .5:           # a jump between checked poses: a branch switch
                    hits_seen[("joints", "jump")] = hits_seen.get(("joints", "jump"), 0) + 1
                    return 50. + worst, i_ / len(tg), "jump " + str(int(np.argmax(np.abs(qq - q_prev))))
                Xw, Yw = TX @ np.linalg.inv(GX), TY @ np.linalg.inv(GY)
                pz = {X: Xw, Y: Yw}
                if Zrides is not None:
                    pz[Z] = pz[Zrides] @ np.linalg.inv(Xm if Zrides == X else Ym) @ Zm
                hit = scene_hits(qq, pz, held_by)
                if hit:
                    for h_ in hit:
                        hits_seen[h_] = hits_seen.get(h_, 0) + 1
                    return 100. + worst, i_ / len(tg), "hit " + ",".join(sorted("%s-%s" % h_ for h_ in hit))
            return worst, 1., ""
        best, t_s, tried = None, time.time(), []
        star = [(0., 0., 0.)] + [tuple(v) for v in np.vstack([np.eye(3), -np.eye(3)]) * .08]
        for yaw in np.radians([0., 45., -45., 90., -90., 180.]):
            for c in star:
                W = placed(c, yaw)
                pre = [(Wk @ Xm @ GX, Wk @ Ym @ GY, Xrel0) for Wk in lin(np.eye(4), W, .01, .1)]
                for s_ in (1., .5, 0., .25, .75):
                    tg = [(W @ TX, W @ TY, Xr) for TX, TY, Xr in shared(s_)[:n_main]]
                    worst, frac, why_ = worst_of(pre[::6] + tg[::6] + [tg[-1]])
                    tried.append((round(float(np.degrees(yaw))), [round(1000 * v) for v in c], s_, round(worst, 1), round(frac, 2), why_))
                    if best is None or (frac, -worst) > (best[4], -best[0]):
                        best = (worst, s_, c, yaw, frac)
                    if worst < 2:
                        break
                if best[0] < 2:
                    break
            if best[0] < 2:
                break
        worst, s_, c, yaw, _ = best
        note("share search", piece=X, other=Y, tried=len(tried), furthest=sorted(tried, key=lambda t_: (-t_[4], t_[3]))[:12])
        if os.environ.get("TRI_SEARCH_ONLY"):
            state["failed"] = "search only"
            break
        note("share", piece=X, other=Y, share=s_, move_mm=[round(1000 * v) for v in c], yaw_deg=round(float(np.degrees(yaw))),
             worst_ik_mm_plus_deg=round(worst, 2), route_deg=round(float(np.degrees(np.linalg.norm(motion[n_main - 1][1]))), 1),
             s=round(time.time() - t_s, 1), hits={"%s-%s" % k_: v_ for k_, v_ in hits_seen.items()})
        W = placed(c, yaw)
        tg = [(Wk @ Xm @ GX, Wk @ Ym @ GY, Xrel0) for Wk in lin(np.eye(4), W, .01, .1)]
        n_pre = len(tg)
        tg += [(W @ TX, W @ TY, Xr) for TX, TY, Xr in shared(s_)]
        n_main += n_pre
        q = measured()
        QR, qq = [], q.copy()                               # both arms' joints for every pose (the robot's ik, chained)
        for TX, TY, _ in tg:
            qq = reach(sx, TX, qq, 60)[0]
            qq = reach(sy, TY, qq, 60)[0]
            QR.append(qq.copy())
        QR = np.array(QR)
        note("route joints", poses=len(QR), max_step_rad=round(float(np.abs(np.diff(QR, axis=0)).max()), 4))
        segments = [(0, n_main)] + [(n_main + 64 * e, n_main + 64 * (e + 1)) for e in range(2)]
        trace, n_dev = [], 0
        for si_, (a_, b_) in enumerate(segments):
            if si_ > 0:
                if links()[key] == 0:
                    break
                note("pushing on", piece=X, other=Y, extra_mm=32 * si_)
            for kk, i_ in enumerate(list(range(a_, b_)) + [b_ - 1] * 40):
                Xr = tg[i_][2]
                q = q + np.clip(QR[i_] - q, -STEP_MAX, STEP_MAX)
                if command(q, servo=(sx, X, Y, Xr)):
                    break
                if kk % 20 == 0 or (kk % 4 == 0 and n_dev < 3):
                    Rel = np.linalg.inv(body_T(NAME[Y])) @ body_T(NAME[X])
                    e_ = [a_ + kk, round(1000 * float(np.linalg.norm(Rel[:3, 3] - Xr[:3, 3])), 1),
                          round(float(np.degrees(R.from_matrix(Rel[:3, :3].T @ Xr[:3, :3]).magnitude())), 1)]
                    if kk % 20 == 0:
                        trace.append(e_)
                    if e_[1] > 3 and n_dev < 3:                  # the first moments the pieces leave the route
                        n_dev += 1
                        note("deviation", piece=X, other=Y, at=e_, contacts=piece_contacts())
                if si_ == 0 and kk == n_pre + len(ref) - 1:
                    note("at the passage", piece=X, other=Y, trace=trace, contacts=piece_contacts())
        if links()[key] != 0:
            state["failed"] = "still linked " + key
            note("route trace", piece=X, other=Y, trace=trace, contacts=piece_contacts())
            break
        state["away"] = body_T(NAME[Y])[:3, :3] @ Dv            # X's way out of Y, in the world
    elif kind == "place":
        # the piece away from the one it left: 3 cm on along the separation's push, up `lift`, over to `xy` and down until
        # its lowest point is `drop` above the table; open, back off along the approach axis and up
        side, piece = step["side"], step["piece"]
        gi = 0 if side == "left" else 1
        P0 = body_T(NAME[piece])
        G_act = np.linalg.inv(P0) @ tcp_T(side, q)
        A = P0.copy()
        A[:3, 3] += .03 * state["away"]
        B_ = A.copy()
        B_[2, 3] += step.get("lift", .12)
        C = B_.copy()
        C[:2, 3] = step["xy"]
        V = np.asarray(ct.solid(piece).vertices)
        C[2, 3] -= (V @ C[:3, :3].T + C[:3, 3])[:, 2].min() - (TABLE_TOP + step.get("drop", .03))
        q = drive_tcp(side, [Pk @ G_act for Pk in lin(P0, A, .02, .2) + lin(A, B_, .03, .3) + lin(B_, C, .03, .3)], q)
        state["grip"][gi] = (state["holder"][piece][1] + OPEN_EXTRA + FINGER_OFFSET) / 2
        state["holder"].pop(piece)
        for _ in range(60):
            if command(q):
                break
        T0 = tcp_T(side, q)
        T1 = T0.copy()
        T1[:3, 3] -= .06 * T0[:3, 0]
        T2 = T1.copy()
        T2[2, 3] += .06
        q = drive_tcp(side, lin(T0, T1) + lin(T1, T2), q)
    note(kind, name=step.get("name", ""))
while not state["ended"] and not state.get("failed") and STOP >= len(PLAN["steps"]):
    if command(q):
        break
    if s.n_steps > PLAN.get("max_steps", 200000):
        break
print(json.dumps({"oracle_n_steps": s.n_steps, "episode_end": s.episode_end, "success": s.success(), "links_end": links(),
                  "failed": state.get("failed"),
                  "max_servo_error_mm": round(1000 * state["worst"], 3), "wall_s": round(time.time() - t0, 1)}),
      file=sys.stderr, flush=True)
