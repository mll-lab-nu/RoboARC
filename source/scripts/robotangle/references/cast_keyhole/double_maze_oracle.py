# Oracle code for `rcb_topogym.py run-oracle --task cast_keyhole --code <this file>` on the double-maze setting
# (cast_keyhole_double_maze; docs/rcb/families/cast_keyhole_double_maze.md).
#
# Input: double_maze_plan.npz next to this file (DM_PLAN overrides): per 40 Hz control step the planned joints of both
# arms (RL, RR: from the birth joints to the designed route's first sample, the route's inverse kinematics, then the
# silver moved on out of the gold's opening and into the goal region while the gold moves behind the separator; solved
# for the plates where they sit in the hands once the grasp welds have bound; time-scaled to at most 0.1 rad/s per joint, 10 mm/s and
# 0.1 rad/s of the silver relative to the gold, 20 mm/s of either plate in the world), the planned centres of mass of
# both plates in the world (CG, CS) and the silver's link pose in the gold's link frame (XR, metres).
# Per control step each arm joint is commanded to its planned target plus feedforwards of its drive's velocity lag
# (joint damping / kp x planned speed), Coulomb friction (frictionloss / kp in the direction of the planned motion) and
# the held plate's weight (the static sag J^T W / kp of the plate at its planned pose, W its weight as a wrench at the
# tool centre point; the arms' own links are gravity-compensated), plus a slow correction: the left arm integrates its
# joint tracking error; the right arm integrates the silver's pose error relative to the measured gold (the planned
# relative pose put on the measured gold, minus the measured silver, carried to the right TCP and through the right
# arm's Jacobian). Grippers stay closed; the plates are held by the family's assisted grasp. Only joint targets are
# commanded; no pose is ever set.
# Order: BIND steps on the birth joints with the weight feedforward of the birth state (measured before the first
# step, ramped in over RAMP steps; it does not depend on the plan, so the plates settle into the hands the same way each
# time: the welds bind during the first steps), SETTLE steps on the first plan row with the corrections running, the plan,
# then the last row held until the episode ends (at most HOLD steps). The plan is solved for the plates where they sit
# in the hands after the BIND steps. DM_DIAG names an optional JSON-lines file of tracking diagnostics, the first line the
# plates' link poses in their TCP frames after the BIND steps (not needed for the recording).
import json, os, sys, time
import mujoco
from scipy.spatial.transform import Rotation as R

PLAN = os.environ.get("DM_PLAN", "scripts/robotangle/references/cast_keyhole/double_maze_plan.npz")
DIAG = os.environ.get("DM_DIAG")
BIND, SETTLE, RAMP, HOLD, GRIP = 8, 72, 3, 240, .002
G_LEFT, K_SERVO, LIM_L, LIM_R, RATE = .10, .10, .05, .08, .003   # integral gain, servo gain, offset limits (rad), offset rate (rad/step)
s = session
P = np.load(PLAN)
RL, RR, XR, CG, CS = P["RL"], P["RR"], P["XR"], P["CG"], P["CS"]
w = s.env._world
m = w.model
layout = s.action_layout
iL, iR = list(range(0, 6)), list(range(7, 13))
kp, tau, fric = [], [], []
for n in [layout[i] for i in iL + iR]:
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    a = next(k for k in range(m.nu) if m.actuator_trnid[k, 0] == j)
    kp.append(m.actuator_gainprm[a, 0])
    tau.append(m.dof_damping[m.jnt_dofadr[j]] / m.actuator_gainprm[a, 0])
    fric.append(m.dof_frictionloss[m.jnt_dofadr[j]] / m.actuator_gainprm[a, 0])
kp, tau, fric = np.array(kp), np.array(tau), np.array(fric)
weight = {n: float(m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{n}::{n}")]) * np.asarray(m.opt.gravity, float)
          for n in ("keyhole_gold", "keyhole_silver")}


def tcp(q12, side):
    return s.robot._fk_full(s.robot.q_full(q12))[f"{side}_tcp"]


def jacobian(q12, side):
    off = 0 if side == "left" else 6
    T0 = tcp(q12, side)
    J = np.zeros((6, 6))
    for k in range(6):
        q = q12.copy()
        q[off + k] += 1e-6
        T1 = tcp(q, side)
        J[:3, k] = (T1[:3, 3] - T0[:3, 3]) / 1e-6
        J[3:, k] = R.from_matrix(T1[:3, :3] @ T0[:3, :3].T).as_rotvec() / 1e-6
    return J, T0


def pose(body):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(np.asarray(body["quaternion_wxyz"])[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = body["position_m"]
    return T


def weight_sag(r, cg, cs):
    """Static joint sag of both arms at joints r under their plates' weight, centres of mass cg, cs (rad, 12)."""
    out = np.zeros(12)
    for side, name, c, off in (("left", "keyhole_gold", cg, 0), ("right", "keyhole_silver", cs, 6)):
        J, T0 = jacobian(r, side)
        F = weight[name]
        out[off:off + 6] = (J.T @ np.r_[F, np.cross(c - T0[:3, 3], F)]) / kp[off:off + 6]
    return out


rows = np.vstack([np.repeat(np.c_[RL[:1], RR[:1]], BIND + SETTLE, axis=0), np.c_[RL, RR], np.repeat(np.c_[RL[-1:], RR[-1:]], HOLD, axis=0)])
plan_index = np.r_[np.zeros(BIND + SETTLE, int), np.arange(len(RL)), np.full(HOLD, len(RL) - 1)]
hold = np.asarray(s.hold_action(), float)
q_birth = np.r_[hold[iL], hold[iR]]
birth = w.read_state()["bodies"]
sag_birth = weight_sag(q_birth, np.asarray(birth["keyhole_gold"]["position_m"]), np.asarray(birth["keyhole_silver"]["position_m"]))
vel = np.vstack([np.zeros((1, 12)), np.diff(rows, axis=0)]) * s.control_hz
I_L, S_R = np.zeros(6), np.zeros(6)
worst = dict(position_mm=0., rotation_deg=0.)
diag = open(DIAG, "w") if DIAG else None
links = None
t0 = time.time()
for k in range(len(rows)):
    r, i = rows[k], plan_index[k]
    if k < BIND:
        cmd = q_birth - min(1., (k + 1) / RAMP) * sag_birth
        a = hold.copy()
        a[iL], a[iR] = cmd[:6], cmd[6:]
        if s.step(a):
            break
        if k == BIND - 1 and diag is not None:
            q = s.joint_positions()
            fk = s.robot._fk_full(s.robot.q_full(np.r_[q[iL], q[iR]]))
            st = w.read_state()["bodies"]
            diag.write(json.dumps({"in_hand_after_bind": {side: (np.linalg.inv(fk[f"{side}_tcp"]) @ pose(st[name])).tolist()
                                   for side, name in (("left", "keyhole_gold"), ("right", "keyhole_silver"))}}) + "\n")
        continue
    cmd = r + tau * vel[k] + fric * np.tanh(vel[k] / .01) - weight_sag(r, CG[i], CS[i])
    cmd[:6] += I_L
    cmd[6:] += S_R
    a = np.zeros(14)
    a[iL], a[6], a[iR], a[13] = cmd[:6], GRIP, cmd[6:], GRIP
    if s.step(a):
        break
    q = s.joint_positions()
    qm = np.r_[q[iL], q[iR]]
    I_L = np.clip(I_L + G_LEFT * (r[:6] - qm[:6]), -LIM_L, LIM_L)
    state = w.read_state()
    Gm, Sm = pose(state["bodies"]["keyhole_gold"]), pose(state["bodies"]["keyhole_silver"])
    Sd = Gm @ XR[i]
    dR = R.from_matrix(Sd[:3, :3] @ Sm[:3, :3].T).as_rotvec()
    J, T0 = jacobian(qm, "right")
    twist = np.r_[Sd[:3, 3] - Sm[:3, 3] + np.cross(dR, T0[:3, 3] - Sm[:3, 3]), dR]
    dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), twist)
    S_R = np.clip(S_R + np.clip(K_SERVO * dq, -RATE, RATE), -LIM_R, LIM_R)
    e_p, e_r = 1000 * float(np.linalg.norm(Sd[:3, 3] - Sm[:3, 3])), float(np.degrees(np.linalg.norm(dR)))
    if k >= BIND + SETTLE:
        worst["position_mm"], worst["rotation_deg"] = max(worst["position_mm"], e_p), max(worst["rotation_deg"], e_r)
    if diag is not None:
        if links is None:
            links = (state["bodies"]["keyhole_gold"]["link_indices"][0], state["bodies"]["keyhole_silver"]["link_indices"][0])
        c = state["contacts"]
        la, lb = np.asarray(c.get("link_a", [])), np.asarray(c.get("link_b", []))
        pp = (((la == links[0]) & (lb == links[1])) | ((la == links[1]) & (lb == links[0])))
        pen = float(np.asarray(c["penetration"])[pp].max()) * 1e3 if pp.any() else None
        force = float(np.linalg.norm(np.asarray(c["force"])[pp], axis=1).max()) if pp.any() else None
        diag.write(json.dumps(dict(k=k, i=int(i), seg=str(P["seg"][i]), e_mm=round(e_p, 3), e_deg=round(e_r, 3),
                                   plate_contacts=int(pp.sum()), plate_pen_mm=None if pen is None else round(pen, 3),
                                   plate_force_n=None if force is None else round(force, 2),
                                   I_L=np.round(I_L, 4).tolist(), S_R=np.round(S_R, 4).tolist(),
                                   sag=np.round(qm - r, 4).tolist(), grasp=[h.get("phase") for h in state.get("grasp", {}).get("hands", {}).values()]
                                   if isinstance(state.get("grasp", {}).get("hands"), dict) else None)) + "\n")
        if k % 200 == 0:
            diag.flush()
if diag is not None:
    diag.close()
print(json.dumps({"oracle_n_steps": s.n_steps, "episode_end": s.episode_end, "success": s.success(),
                  "max_relative_position_error_mm": round(worst["position_mm"], 3),
                  "max_relative_rotation_error_deg": round(worst["rotation_deg"], 3), "wall_s": round(time.time() - t0, 1)}),
      file=sys.stderr, flush=True)
