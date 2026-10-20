# Oracle code for `rcb_topogym.py run-oracle --task interlocked_claws --code <this file>` on the Rotor setting
# (interlocked_claws_rotor; docs/simulator/families/interlocked_claws_rotor.md).
#
# Input: rotor_plan.npz next to this file (ROTOR_PLAN overrides): per 40 Hz control step the planned joints of both arms
# (Q: the reference route's inverse kinematics with the regrasps' re-orientations and hand motions, time-scaled to at most
# 0.2 rad/s per joint, 20 mm/s and 0.2 rad/s per TCP) and the planned pose of the silver piece in the gold piece's frame
# (XR, metres), in segments: 'move' (both hands hold) and per regrasp 'leave' (the regrasping hand's retreat and
# transit); per regrasp the arm, the new grasp (GNEW: TCP in the regrasped piece's frame) and the route pose to return to
# (XK). Per control step each arm joint is commanded to its target plus feedforwards of its drive's velocity lag
# (joint damping / kp x target speed) and Coulomb friction (frictionloss / kp in the direction of the motion), plus a slow
# correction: an arm that only holds integrates its joint tracking error; while both hands hold, the right arm servos the
# silver's pose relative to the measured gold (the planned relative pose put on the measured gold, minus the measured
# silver, carried to the right TCP and through the right arm's Jacobian).
# A regrasp: the hand opens to its grasp thickness + 20 mm (the family's assisted grasp releases) and both arms hold
# while the released piece settles; the hand retreats and moves along the plan; then, from the piece's measured pose, it
# approaches the new grasp (inverse kinematics along a straight TCP line from 50 mm in front), closes (the assisted grasp
# acquires on bilateral finger contact), and carries the piece from where it settled back to the route pose relative to
# the other piece (its own pose servo); the next 'move' segment is joined over BLEND steps. Only joint targets and
# gripper targets are commanded; no pose is ever set.
import json, os, sys, time
import mujoco
from scipy.spatial.transform import Rotation as R

PLAN = os.environ.get("ROTOR_PLAN", "tools/references/interlocked_claws/rotor_plan.npz")
SETTLE, SETTLE_RELEASE, GRASP, BLEND, HOLD = 80, 60, 24, 40, 240
STEP_MAX = .02                                               # rad per control step for run-time inverse kinematics targets
LOG = os.environ.get("ROTOR_LOG", "1") == "1"
GRIP, OPEN_EXTRA, FINGER_OFFSET = .002, .020, .0157          # closed target; opening beyond the grasp thickness; finger rails
G_HOLD, K_SERVO, LIM, RATE = .10, .10, .05, .003            # integral gain, servo gain, offset limit (rad), offset rate (rad/step)
V_APPROACH, W_APPROACH, V_CARRY, W_CARRY = .010, .10, .005, .05   # m/s, rad/s at the control rate
s = session
P = np.load(PLAN)
Q, XR, SEG, KIND, EV = P["Q"], P["XR"], P["SEG"], P["KIND"], P["EV"]
ARM, GNEW, XK, GAP_OLD, GAP_NEW = P["ARM"], P["GNEW"], P["XK"], P["GAP_OLD"], P["GAP_NEW"]
hz = s.control_hz
w = s.env._world
m = w.model
layout = s.action_layout
iL, iR = list(range(0, 6)), list(range(7, 13))
tau, fric, lo, hi = [], [], [], []
for n in [layout[i] for i in iL + iR]:
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    a = next(k for k in range(m.nu) if m.actuator_trnid[k, 0] == j)
    tau.append(m.dof_damping[m.jnt_dofadr[j]] / m.actuator_gainprm[a, 0])
    fric.append(m.dof_frictionloss[m.jnt_dofadr[j]] / m.actuator_gainprm[a, 0])
    lo.append(m.jnt_range[j, 0]); hi.append(m.jnt_range[j, 1])
tau, fric, lo, hi = map(np.array, (tau, fric, lo, hi))
SIDE = {"left": slice(0, 6), "right": slice(6, 12)}
PIECE = {"left": "piece_1", "right": "piece_0"}   # the gold piece (left hand) and the silver one (right)


def body_T(name):
    b = w.read_state()["bodies"][name]
    T = np.eye(4)
    T[:3, :3] = R.from_quat(np.asarray(b["quaternion_wxyz"])[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = b["position_m"]
    return T


def tcp_T(arm, q12):
    return s.robot._fk_full(s.robot.q_full(q12))[arm + "_tcp"]


def jacobian(arm, q12):
    T0, sl = tcp_T(arm, q12), SIDE[arm]
    J = np.zeros((6, 6))
    for k in range(6):
        q = q12.copy()
        q[sl.start + k] += 1e-6
        T1 = tcp_T(arm, q)
        J[:3, k] = (T1[:3, 3] - T0[:3, 3]) / 1e-6
        J[3:, k] = R.from_matrix(T1[:3, :3] @ T0[:3, :3].T).as_rotvec() / 1e-6
    return J, T0


def twist_to(arm, q12, T):
    """joint step of one arm that moves its TCP from FK(q12) to T (damped least squares, one step)"""
    J, T0 = jacobian(arm, q12)
    e = np.r_[T[:3, 3] - T0[:3, 3], R.from_matrix(T[:3, :3] @ T0[:3, :3].T).as_rotvec()]
    return J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e), e


def ik(arm, T, q12, iters=30):
    q = q12.copy()
    for _ in range(iters):
        dq, e = twist_to(arm, q, T)
        if np.linalg.norm(e[:3]) < 2e-5 and np.linalg.norm(e[3:]) < 2e-4:
            break
        q[SIDE[arm]] = np.clip(q[SIDE[arm]] + np.clip(dq, -.2, .2), lo[SIDE[arm]], hi[SIDE[arm]])
    return q


def lin(A, B, v, wv):
    """poses from A to B, straight line and slerp, at most v m and wv rad per second at the control rate"""
    ang = R.from_matrix(B[:3, :3] @ A[:3, :3].T).magnitude()
    n = max(2, int(np.ceil(max(np.linalg.norm(B[:3, 3] - A[:3, 3]) / v, ang / wv) * hz)) + 1)
    rots = R.from_matrix(np.stack([A[:3, :3], B[:3, :3]]))
    from scipy.spatial.transform import Slerp
    sl = Slerp([0, 1], rots)
    out = []
    for t in np.linspace(0, 1, n):
        T = np.eye(4)
        T[:3, :3] = sl(t).as_matrix()
        T[:3, 3] = (1 - t) * A[:3, 3] + t * B[:3, 3]
        out.append(T)
    return out


state = dict(I=np.zeros(12), S=np.zeros(12), grip=[GRIP, GRIP], prev=Q[0].copy(), worst=0., ended=False)


def log(what, **kw):
    if LOG:
        rel = np.linalg.inv(body_T("piece_1")) @ body_T("piece_0")
        print(json.dumps(dict(step=s.n_steps, what=what, silver_in_gold_mm=np.round(1000 * rel[:3, 3], 2).tolist(), **kw)),
              file=sys.stderr, flush=True)


def toward(arm, q, T):
    """the next control step's joints toward the IK solution for TCP pose T, at most STEP_MAX per joint; (q, residual m)"""
    goal = ik(arm, T, q)
    d = goal[SIDE[arm]] - q[SIDE[arm]]
    q = q.copy()
    q[SIDE[arm]] += np.clip(d, -STEP_MAX, STEP_MAX)
    T0 = tcp_T(arm, goal)
    return q, float(np.linalg.norm(T0[:3, 3] - T[:3, 3])), float(np.abs(d).max())


def measured():
    q = s.joint_positions()
    return np.r_[q[iL], q[iR]]


def command(target, servo_arm=None, xrel=None, hold=("left", "right")):
    """one control step toward target (12 joints): feedforwards + corrections; servo_arm servos the silver-in-gold pose
    xrel (right: moves the silver; left: moves the gold); arms in hold integrate their joint error. Returns True at the
    episode's end."""
    vel = (target - state["prev"]) * hz
    state["prev"] = target.copy()
    cmd = target + tau * vel + fric * np.tanh(vel / .01) + state["I"] + state["S"]
    cmd = np.clip(cmd, lo, hi)
    a = np.zeros(14)
    a[iL], a[6], a[iR], a[13] = cmd[:6], state["grip"][0], cmd[6:], state["grip"][1]
    if s.step(a):
        state["ended"] = True
        return True
    qm = measured()
    for arm in hold:
        sl = SIDE[arm]
        state["I"][sl] = np.clip(state["I"][sl] + G_HOLD * (target[sl] - qm[sl]), -LIM, LIM)
    if servo_arm is not None:
        Gm, Sm = body_T("piece_1"), body_T("piece_0")
        if servo_arm == "right":
            Td, Tm = Gm @ xrel, Sm
        else:
            Td, Tm = Sm @ np.linalg.inv(xrel), Gm
        dR = R.from_matrix(Td[:3, :3] @ Tm[:3, :3].T).as_rotvec()
        J, T0 = jacobian(servo_arm, qm)
        tw = np.r_[Td[:3, 3] - Tm[:3, 3] + np.cross(dR, T0[:3, 3] - Tm[:3, 3]), dR]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), tw)
        sl = SIDE[servo_arm]
        state["S"][sl] = np.clip(state["S"][sl] + np.clip(K_SERVO * dq, -RATE, RATE), -LIM, LIM)
        rel = np.linalg.inv(Gm) @ Sm
        state["worst"] = max(state["worst"], float(np.linalg.norm(rel[:3, 3] - xrel[:3, 3])))
    return False


t0 = time.time()
report = dict(regrasps=[])
for _ in range(SETTLE):
    if command(Q[0], "right", XR[0], hold=("left",)):
        break
segs = np.unique(SEG)
for si in segs:
    if state["ended"]:
        break
    idx = np.nonzero(SEG == si)[0]
    kind, e = str(KIND[si]), int(EV[si])
    if kind == "move":
        log("move", segment=int(si), steps=int(len(idx)), worst_mm=round(1000 * state["worst"], 3))
        if si > 0 and str(KIND[si - 1]) == "leave":          # join the plan from where the regrasp left the arms
            start = measured()
            gap = float(np.abs(Q[idx[0]] - start).max())
            log("blend", largest_gap_rad=round(gap, 4))
            state["I"][:], state["S"][:] = 0., 0.
            nb = max(BLEND, int(np.ceil(gap / (STEP_MAX / 2))))   # at most 0.4 rad/s
            for b in range(1, nb + 1):
                if command(start + (Q[idx[0]] - start) * b / nb, "right", XR[idx[0]], hold=("left",)):
                    break
        for k in idx:
            if command(Q[k], "right", XR[k], hold=("left",)):
                break
        continue
    # a regrasp: release, settle, leave along the plan, approach the measured piece, grasp, carry back
    arm = "left" if ARM[e] == 0 else "right"
    other = "right" if arm == "left" else "left"
    gi = 0 if arm == "left" else 1
    log("release", event=e, arm=arm)
    hold_target = state["prev"] + state["S"]               # the holding arm keeps its servoed pose while the piece settles
    state["prev"], state["S"][:] = hold_target.copy(), 0.
    state["grip"][gi] = (GAP_OLD[e] + OPEN_EXTRA + FINGER_OFFSET) / 2
    trail = [np.linalg.inv(body_T("piece_1")) @ body_T("piece_0")]   # the released piece's way, relative

    def note():
        X = np.linalg.inv(body_T("piece_1")) @ body_T("piece_0")
        if np.linalg.norm(X[:3, 3] - trail[-1][:3, 3]) > 2e-4 or R.from_matrix(X[:3, :3] @ trail[-1][:3, :3].T).magnitude() > 2e-3:
            trail.append(X)
    for _ in range(SETTLE_RELEASE):
        if command(hold_target, hold=("left", "right")):
            break
        note()
    settled = np.linalg.inv(body_T("piece_1")) @ body_T("piece_0")
    state["grip"][gi] = (max(GAP_OLD[e], GAP_NEW[e]) + OPEN_EXTRA + FINGER_OFFSET) / 2
    for k in idx:                                         # retreat and transit: the regrasping arm moves, the other holds
        tgt = hold_target.copy()
        tgt[SIDE[arm]] = Q[k][SIDE[arm]]
        if command(tgt, hold=("left", "right")):
            break
        note()
    if state["ended"]:
        break
    # approach the new grasp on the piece where it actually is
    qm = measured()
    Tn = body_T(PIECE[arm]) @ GNEW[e]
    pre = Tn.copy()
    pre[:3, 3] -= .05 * Tn[:3, 0]
    q = qm.copy()
    q[SIDE[other]] = hold_target[SIDE[other]]
    path = lin(tcp_T(arm, qm), pre, V_APPROACH, W_APPROACH)[1:] + lin(pre, Tn, V_APPROACH, W_APPROACH)[1:]
    log("approach", event=e, arm=arm, steps=len(path))
    worst_res, worst_jump = 0., 0.
    for T in path + [path[-1]] * 40:                       # the last pose repeated: room to catch up at STEP_MAX
        q, res, jump = toward(arm, q, T)
        worst_res, worst_jump = max(worst_res, res), max(worst_jump, jump)
        if command(q, hold=("left", "right")):
            break
        note()
        if T is path[-1] and jump < 1e-4:
            break
    log("grasp", event=e, ik_residual_mm=round(1000 * worst_res, 3), largest_ik_jump_rad=round(worst_jump, 4))
    state["grip"][gi] = GRIP
    for _ in range(GRASP):
        if command(q, hold=("left", "right")):
            break
        note()
    if state["ended"]:
        break
    # carry the piece back from where it settled to the route pose relative to the other piece
    qm = measured()
    G_act = np.linalg.inv(body_T(PIECE[arm])) @ tcp_T(arm, qm)
    X0 = np.linalg.inv(body_T("piece_1")) @ body_T("piece_0")
    q = qm.copy()
    q[SIDE[other]] = hold_target[SIDE[other]]
    state["S"][:] = 0.
    log("carry back", event=e, arm=arm, from_route_mm=round(1000 * float(np.linalg.norm(X0[:3, 3] - XK[e][:3, 3])), 2))
    worst_jump = 0.
    way = [X0] + trail[::-1] + [XK[e]]                    # back the way the piece came, then onto the route pose
    back = []
    for A_, B_ in zip(way, way[1:]):
        back += lin(A_, B_, V_CARRY, W_CARRY)[1:]
    log("carry path", event=e, trail_poses=len(trail), steps=len(back))
    for X in back:
        if arm == "right":
            T = body_T("piece_1") @ X @ G_act
        else:
            T = body_T("piece_0") @ np.linalg.inv(X) @ G_act
        q, res, jump = toward(arm, q, T)
        worst_jump = max(worst_jump, jump)
        if command(q, arm, X, hold=(other,)):
            break
    done = np.linalg.inv(body_T("piece_1")) @ body_T("piece_0")
    log("back on route", event=e, largest_ik_jump_rad=round(worst_jump, 4))
    report["regrasps"].append(dict(sample=int(P["K"][e]), arm=arm, step=s.n_steps,
                                   settle_from_route_mm=round(1000 * float(np.linalg.norm(settled[:3, 3] - XK[e][:3, 3])), 2),
                                   back_on_route_mm=round(1000 * float(np.linalg.norm(done[:3, 3] - XK[e][:3, 3])), 3)))
while not state["ended"] and s.n_steps < len(Q) + 20000:
    if command(Q[-1], "right", XR[-1], hold=("left",)):
        break
    HOLD -= 1
    if HOLD <= 0:
        break
print(json.dumps({"oracle_n_steps": s.n_steps, "episode_end": s.episode_end, "success": s.success(),
                  "max_relative_position_error_mm": round(1000 * state["worst"], 3), "regrasps": report["regrasps"],
                  "wall_s": round(time.time() - t0, 1)}), file=sys.stderr, flush=True)
