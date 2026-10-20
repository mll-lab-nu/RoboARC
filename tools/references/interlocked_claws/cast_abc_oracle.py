# Oracle code for `rcb_topogym.py run-oracle --task interlocked_claws --code <this file>` on the Cast ABC setting
# (interlocked_claws_cast_abc; docs/rcb/families/interlocked_claws.md, "Cast ABC").
#
# Input: cast_abc_plan.npz next to this file (ABC_PLAN overrides): per 40 Hz control step the planned joints of both arms
# (RL, RR: the reference route's inverse kinematics, time-scaled to at most 0.1 rad/s per joint and 10 mm/s, 0.1 rad/s of
# the C ring relative to the plate) and the planned pose of the C ring in the AB plate's frame (XR, metres).
# Per control step each arm joint is commanded to its planned target plus feedforwards of its drive's velocity lag
# (joint damping / kp x planned speed) and Coulomb friction (frictionloss / kp in the direction of the planned motion),
# plus a slow correction: the left arm integrates its joint tracking error (the plate's weight makes the low-gain drives
# sag); the right arm integrates the C ring's pose error relative to the measured plate (the planned relative pose put on
# the measured plate, minus the measured C, carried to the right TCP and through the right arm's Jacobian). Grippers stay
# closed; the pieces are held by the family's assisted grasp. Only joint targets are commanded; no pose is ever set.
# Order: 80 steps on the first plan row (the corrections take up the sag after birth), the plan, then the last row held
# until the episode ends (at most 240 steps).
import json, os, sys, time
from pathlib import Path
import mujoco
from scipy.spatial.transform import Rotation as R

PLAN = os.environ.get("ABC_PLAN", "tools/references/interlocked_claws/cast_abc_plan.npz")
SETTLE, HOLD, GRIP = 80, 240, .002
G_LEFT, K_SERVO, LIM, RATE = .10, .10, .05, .003   # integral gain, servo gain, offset limit (rad), offset rate (rad/step)
s = session
P = np.load(PLAN)
RL, RR, XR, SEG = P["RL"], P["RR"], P["XR"], P["seg"]
w = s.env._world
m = w.model
layout = s.action_layout
iL, iR = list(range(0, 6)), list(range(7, 13))
tau, fric = [], []
for n in [layout[i] for i in iL + iR]:
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    a = next(k for k in range(m.nu) if m.actuator_trnid[k, 0] == j)
    tau.append(m.dof_damping[m.jnt_dofadr[j]] / m.actuator_gainprm[a, 0])
    fric.append(m.dof_frictionloss[m.jnt_dofadr[j]] / m.actuator_gainprm[a, 0])
tau, fric = np.array(tau), np.array(fric)


def body_T(name):
    b = w.read_state()["bodies"][name]
    T = np.eye(4)
    T[:3, :3] = R.from_quat(np.asarray(b["quaternion_wxyz"])[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = b["position_m"]
    return T


def tcp_T(q12):
    return s.robot._fk_full(s.robot.q_full(q12))["right_tcp"]


def jacobian(q12):
    T0 = tcp_T(q12)
    J = np.zeros((6, 6))
    for k in range(6):
        q = q12.copy()
        q[6 + k] += 1e-6
        T1 = tcp_T(q)
        J[:3, k] = (T1[:3, 3] - T0[:3, 3]) / 1e-6
        J[3:, k] = R.from_matrix(T1[:3, :3] @ T0[:3, :3].T).as_rotvec() / 1e-6
    return J, T0


rows = np.vstack([np.repeat(np.c_[RL[:1], RR[:1]], SETTLE, axis=0), np.c_[RL, RR], np.repeat(np.c_[RL[-1:], RR[-1:]], HOLD, axis=0)])
xrel = np.concatenate([np.repeat(XR[:1], SETTLE, axis=0), XR, np.repeat(XR[-1:], HOLD, axis=0)])
vel = np.vstack([np.zeros((1, 12)), np.diff(rows, axis=0)]) * s.control_hz
I_L, S_R, worst = np.zeros(6), np.zeros(6), 0.
t0 = time.time()
for k in range(len(rows)):
    r = rows[k]
    cmd = r + tau * vel[k] + fric * np.tanh(vel[k] / .01)
    cmd[:6] += I_L
    cmd[6:] += S_R
    a = np.zeros(14)
    a[iL], a[6], a[iR], a[13] = cmd[:6], GRIP, cmd[6:], GRIP
    if s.step(a):
        break
    q = s.joint_positions()
    qm = np.r_[q[iL], q[iR]]
    I_L = np.clip(I_L + G_LEFT * (r[:6] - qm[:6]), -LIM, LIM)
    Pm, Cm = body_T("ab_plate"), body_T("c_ring")
    Cd = Pm @ xrel[k]
    dR = R.from_matrix(Cd[:3, :3] @ Cm[:3, :3].T).as_rotvec()
    J, T0 = jacobian(qm)
    twist = np.r_[Cd[:3, 3] - Cm[:3, 3] + np.cross(dR, T0[:3, 3] - Cm[:3, 3]), dR]
    dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), twist)
    S_R = np.clip(S_R + np.clip(K_SERVO * dq, -RATE, RATE), -LIM, LIM)
    worst = max(worst, float(np.linalg.norm((np.linalg.inv(Pm) @ Cm)[:3, 3] - xrel[k][:3, 3])))
print(json.dumps({"oracle_n_steps": s.n_steps, "episode_end": s.episode_end, "success": s.success(),
                  "max_relative_position_error_mm": round(1000 * worst, 3), "wall_s": round(time.time() - t0, 1)}),
      file=sys.stderr, flush=True)
