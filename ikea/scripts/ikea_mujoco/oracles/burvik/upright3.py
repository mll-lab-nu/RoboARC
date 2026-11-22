"""BURVIK P6, closed loop: turn the assembled table upright, on kinematically planned joint paths.

Why not the base rows: they lift with the left hand on leg_2 off the leg's centre (one finger pushes it) and the right
hand on the handle, whose arm is at full reach with its elbow on the table: the right hand cannot rise, the table
wedges in the jig and both grasps slip. A Cartesian servo of both hands (upright.py) drives the arms into their joint
limits halfway through the turn. Here every motion is a joint path from joint-limited IK (plan_up.py), checked for
collisions with the table moved rigidly in a model copy (probe z_search_up.py: grasps and path chosen there):
  1. the base rows to K_PRE (the left hand above leg_2, open). The right hand opens, goes in joint space to the base
     rows' last right-arm pose (high over the legs; the sweep is collision free), then by IK onto leg_0 at world x X_R,
     its approach tilted PHI_R from straight down toward -y, the fingers across the leg; closed;
  2. the left hand by IK onto leg_2 at world x X_L, pointing down (PHI_L); closed;
  3. the table's path: up by LIFT, then 90 deg about world y (the handle up, the legs down) about the grasp line while
     that line moves so that the top's centre ends at (PLACE_X, PLACE_Y, PLACE_Z); both arms' joints follow IK
     solutions of the two grasps along it (sampled every SAMPLE ticks, interpolated linearly in between);
  4. down along the same plan until a leg touches the bench; hold; open both hands; back each out along its approach
     axis; joint blend to the base rows' last row."""
import os
import numpy as np
import mujoco
from seat import seat_rot
from servo import rotvec
from upright import arm_touch, ease, slerp_R, contact_forces, bench_contact, Hands, CLOSED, OPEN
from plan_up import ik, grasp_pose, PARTS

K_PRE = 119750
X_L = float(os.environ.get("BURVIK_UP_XL", "0.0"))
X_R = float(os.environ.get("BURVIK_UP_XR", "0.10"))
PHI_L = float(os.environ.get("BURVIK_UP_PHIL", "0"))
PHI_R = float(os.environ.get("BURVIK_UP_PHIR", "60"))
LIFT = float(os.environ.get("BURVIK_UP_LIFT", "0.25"))
PLACE_X = float(os.environ.get("BURVIK_UP_X", "0.0"))
PLACE_Y = float(os.environ.get("BURVIK_UP_Y", "0.0"))
PLACE_Z = float(os.environ.get("BURVIK_UP_Z", "0.575"))
N_LIFT, N_TURN, N_DOWN = (int(x) for x in os.environ.get("BURVIK_UP_N", "1600,3200,6000").split(","))      # ticks; N_DOWN covers DOWN_M (stopped at the bench contact)
DOWN_M = 0.06
SAMPLE = 50
MODE = os.environ.get("BURVIK_UP_MODE", "servo")
KN = float(os.environ.get("BURVIK_UP_KN", "0.01"))   # null-space pull toward the plan, per tick
DEBUG = os.environ.get("BURVIK_UP_DEBUG") == "1"
DBG_A, DBG_B = (int(x) for x in os.environ.get("BURVIK_UP_DBG", "1800,3200").split(","))


def joint_move(o, side, q1, n, grips):
    """the side's arm targets from the last row's to q1 (eased) over n ticks; the other arm's held."""
    q0 = np.array(o.rows[-1])[o.arm[side]]
    for i in range(n):
        row = np.array(o.rows[-1]); row[o.arm[side]] = q0 + ease(i + 1, n) * (q1 - q0)
        for s, g in grips.items():
            row[o.grip[s]] = g
        o.step(row)


def ik_here(o, side, p, R, seed):
    d2 = mujoco.MjData(o.m); d2.qpos[:] = o.d.qpos
    return ik(o.m, d2, o, side, p, R, seed)


def grasp(B, side, leg, x, phi, grips_other, above=0.08):
    """by IK to 80 mm back along the approach (joint move), in on the true leg axis (Cartesian), closed."""
    o, m, d = B.o, B.o.m, B.o.d; H = Hands(B)
    mujoco.mj_kinematics(m, d)
    P, R = grasp_pose(m, d, leg, x, side, phi)
    q, ep, ew = ik_here(o, side, P - above * R[:, 2], R, d.qpos[o.qadr[side]])
    B.note(phase="p6_ik_pre", side=side, e_mm=round(1000 * ep, 2), e_deg=round(np.degrees(ew), 2))
    dq = np.max(np.abs(q - d.qpos[o.qadr[side]]))
    joint_move(o, side, q, max(800, int(dq / 0.6 / o.s.control_dt)), dict(grips_other, **{side: OPEN}))
    for i in range(400):                                          # settle on the pre-grasp pose
        H.go({side: (P - above * R[:, 2], R)}, dict(grips_other, **{side: OPEN}))
    for i in range(1400):                                         # in along the approach axis, on the true leg axis
        mujoco.mj_kinematics(m, d); P, R = grasp_pose(m, d, leg, x, side, phi)
        e = H.go({side: (P - above * (1 - ease(i, 1200)) * R[:, 2], R)}, dict(grips_other, **{side: OPEN}))
    for i in range(500):
        mujoco.mj_kinematics(m, d); P, R = grasp_pose(m, d, leg, x, side, phi)
        e = H.go({side: (P, R)}, dict(grips_other, **{side: CLOSED}))
    B.note(phase="p6_grasp", side=side, leg=leg, e_mm=round(1000 * e[side], 2), fingers=contact_forces(m, d, {m.body(leg).id}, side + "_"))


def upright_p6(B):
    o = B.o; m, d = o.m, o.d
    B.follow(117800, K_PRE, hold=False)
    gl = float(o.rows[-1][o.grip["left"]])
    for i in range(300):                                          # the right hand opens on the handle
        row = np.array(o.rows[-1]); row[o.grip["right"]] = OPEN; o.step(row)
    q0 = np.array(o.rows[-1])[o.arm["right"]]; qs = np.array(B.base[-1][o.arm["right"]], dtype=np.float64)
    qs[6] += 2 * np.pi * np.round((q0[6] - qs[6]) / (2 * np.pi))
    joint_move(o, "right", qs, 3000, {"right": OPEN, "left": gl})
    grasp(B, "right", "leg_0", X_R, PHI_R, {"left": gl})
    grasp(B, "left", "leg_2", X_L, PHI_L, {"right": CLOSED})
    B.snap("p6_take", K_PRE)
    upright_move(B)


def upright_move(B):
    o = B.o; m, d = o.m, o.d
    top = m.body("top").id; parts = {m.body(n).id for n in PARTS}
    mujoco.mj_kinematics(m, d)
    c0 = d.xpos[top].copy(); R0 = d.xmat[top].reshape(3, 3).copy(); Rrel = seat_rot(R0, 90.) @ R0.T
    g = {s: (d.site_xpos[o.site[s]].copy(), d.site_xmat[o.site[s]].reshape(3, 3).copy()) for s in ("left", "right")}
    G0 = 0.5 * (g["left"][0] + g["right"][0]); G1 = G0 + np.array([0., 0., LIFT])
    cf = np.array([PLACE_X, PLACE_Y, PLACE_Z]); Gf = cf - Rrel @ (c0 - G0)

    def motion(t):
        """the table's rigid motion at tick t of the plan: x -> R x + p."""
        if t <= N_LIFT:
            return np.eye(3), np.array([0., 0., LIFT * ease(t, N_LIFT)])
        if t <= N_LIFT + N_TURN:
            a = ease(t - N_LIFT, N_TURN); Ra = slerp_R(np.eye(3), Rrel, a); G = G1 + a * (Gf - G1)
            return Ra, G - Ra @ G0
        a = min(1., (t - N_LIFT - N_TURN) / N_DOWN)
        return Rrel, Gf - Rrel @ G0 - np.array([0., 0., DOWN_M * a])

    T = N_LIFT + N_TURN + N_DOWN
    ts = list(range(0, T + 1, SAMPLE))
    q = {s: np.array(o.rows[-1])[o.arm[s]] for s in ("left", "right")}
    plan = {s: [] for s in q}; worst = 0.
    d2 = mujoco.MjData(m); d2.qpos[:] = d.qpos
    for t in ts:
        Rm, p = motion(t)
        for s in ("left", "right"):
            q[s], ep, ew = ik(m, d2, o, s, Rm @ g[s][0] + p, Rm @ g[s][1], q[s]); plan[s].append(q[s].copy()); worst = max(worst, ep)
    plan = {s: np.array(v) for s, v in plan.items()}
    jump = {s: round(float(np.max(np.abs(np.diff(plan[s], axis=0)))), 4) for s in plan}
    B.note(phase="p6_plan", samples=len(ts), worst_ik_mm=round(1000 * worst, 3), max_jump_rad=jump,
           at=({s: int(np.argmax(np.max(np.abs(np.diff(plan[s], axis=0)), axis=1))) for s in plan}))
    # the plan is in actual joint positions: the targets keep the current targets' offset from them (the position
    # servos' load offset; no jump at the start)
    off0 = {s: np.array(o.rows[-1])[o.arm[s]] - plan[s][0] for s in plan}
    # The grasps creep (soft contacts under the table's weight torque): the hands are driven to where they must be for
    # the table to be on its plan GIVEN the grasps as they are now: each tick, the TCP's pose in the table's true frame
    # (the current grasp), composed with the planned table pose, solved for the joints by damped least squares warm
    # started from the previous solution (MODE "track"; MODE "plan": the precomputed joint plan alone).
    qd = {s: plan[s][0].copy() for s in plan}
    lim = {s: o.s.step_limit[o.arm[s]] * 0.9 for s in plan}
    d3 = mujoco.MjData(m); d3.qpos[:] = d.qpos
    t = 0; contact_t = None
    while t < T:
        k = t // SAMPLE; f = (t - k * SAMPLE) / SAMPLE
        row = np.array(o.rows[-1])
        if MODE == "servo":
            # u2a's servo (the table's pose error -> its rigid twist, plus the plan's own motion this tick -> each TCP ->
            # damped least squares, added to the last targets: integral action, so the grasps' creep is made up), with
            # the null space of each arm's TCP task pulled toward the planned joints (keeps the arms off their limits
            # and on the collision-checked configurations)
            Rm, p = motion(t); Rn, pn = motion(t + 1)
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            Rt = d.xmat[top].reshape(3, 3); pt = d.xpos[top].copy()
            e_p = Rm @ c0 + p - pt; e_w = rotvec(Rm @ R0 @ Rt.T)
            v_ff = (Rn @ c0 + pn) - (Rm @ c0 + p); w_ff = rotvec(Rn @ Rm.T)
            v = v_ff + np.clip(0.08 * e_p, -0.0004, 0.0004); w = w_ff + np.clip(0.08 * e_w, -0.002, 0.002)
            for s in ("left", "right"):
                jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d, jp, jr, o.site[s])
                va = o.vadr[s]; J = np.vstack([jp[:, va], jr[:, va]])
                tw = np.concatenate([v + np.cross(w, d.site_xpos[o.site[s]] - pt), w])
                Jp = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(6))
                qt = (1 - f) * plan[s][k] + f * plan[s][min(k + 1, len(ts) - 1)]
                dq = Jp @ tw + (np.eye(7) - Jp @ J) @ (KN * (qt - d.qpos[o.qadr[s]]))
                row[o.arm[s]] = row[o.arm[s]] + np.clip(dq, -lim[s], lim[s]); row[o.grip[s]] = CLOSED
        elif MODE == "track":
            Rm, p = motion(t + 1)
            mujoco.mj_kinematics(m, d)
            Rt = d.xmat[top].reshape(3, 3); pt = d.xpos[top]
            Rp = Rm @ R0; pp = Rm @ c0 + p                                  # the planned table pose
            for s in ("left", "right"):
                gp = Rt.T @ (d.site_xpos[o.site[s]] - pt); gR = Rt.T @ d.site_xmat[o.site[s]].reshape(3, 3)
                qd[s], _, _ = ik(m, d3, o, s, pp + Rp @ gp, Rp @ gR, qd[s], iters=3)
                row[o.arm[s]] = qd[s] + off0[s]; row[o.grip[s]] = CLOSED
        else:
            for s in ("left", "right"):
                qt = (1 - f) * plan[s][k] + f * plan[s][min(k + 1, len(ts) - 1)]
                row[o.arm[s]] = qt + off0[s]; row[o.grip[s]] = CLOSED
        corr = {s: row[o.arm[s]] - off0[s] - ((1 - f) * plan[s][k] + f * plan[s][min(k + 1, len(ts) - 1)]) for s in plan}
        o.step(row); t += 1
        if DEBUG and (t % 400 == 0 or DBG_A <= t <= DBG_B and t % 100 == 0):
            Rm, p = motion(t)
            mujoco.mj_kinematics(m, d)
            tcp_err = {s: round(1000 * float(np.linalg.norm(Rm @ g[s][0] + p - d.site_xpos[o.site[s]])), 2) for s in g}
            lag = {s: round(float(np.max(np.abs(np.array(o.rows[-1])[o.arm[s]] - d.qpos[o.qadr[s]]))), 4) for s in g}
            B.note(phase="dbg_move", t=t, tcp_err_mm=tcp_err, joint_lag=lag, top_err_mm=round(1000 * float(np.linalg.norm(Rm @ c0 + p - d.xpos[top])), 2),
                   fl=contact_forces(m, d, {m.body("leg_2").id}, "left_"), fr=contact_forces(m, d, {m.body("leg_0").id}, "right_"),
                   world=contact_forces(m, d, parts), arms={sd: arm_touch(m, d, sd) for sd in ("left", "right")},
                   corr={sd: round(float(np.max(np.abs(corr[sd]))), 3) for sd in corr})
        if t > N_LIFT + N_TURN and bench_contact(m, d, parts):
            contact_t = t; break
    Rm, p = motion(t)
    B.note(phase="p6_down", t=t, contact=contact_t, top=np.round(d.xpos[top], 4).tolist(), top_err_mm=round(1000 * float(np.linalg.norm(Rm @ c0 + p - d.xpos[top])), 2),
           top_z=np.round(d.xmat[top].reshape(3, 3)[:, 2], 4).tolist(), fl=contact_forces(m, d, {m.body("leg_2").id}, "left_"),
           fr=contact_forces(m, d, {m.body("leg_0").id}, "right_"), world=contact_forces(m, d, parts))
    for i in range(200):
        o.step(np.array(o.rows[-1]))
    # release, back out, home
    lim = {s: o.s.step_limit[o.arm[s]] * 0.9 for s in ("left", "right")}
    for i in range(400):
        row = np.array(o.rows[-1]); row[o.grip["left"]] = OPEN; row[o.grip["right"]] = OPEN; o.step(row)
    mujoco.mj_kinematics(m, d)
    ax = {s: d.site_xmat[o.site[s]].reshape(3, 3)[:, 2].copy() for s in ("left", "right")}
    for i in range(1000):
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        row = np.array(o.rows[-1])
        for s in ("left", "right"):
            row[o.arm[s]] += np.clip(o.dls(s, -ax[s] * 0.0001, np.zeros(3)), -lim[s], lim[s])
        o.step(row)
    start = np.array(o.rows[-1]); end = np.array(B.base[-1], dtype=np.float64)
    for s in ("left", "right"):
        j = o.arm[s][6]; end[j] = end[j] + 2 * np.pi * np.round((start[j] - end[j]) / (2 * np.pi))
    n = 3200
    for i in range(n):
        a = ease(i + 1, n); row = (1 - a) * start + a * end
        row[o.grip["left"]] = OPEN; row[o.grip["right"]] = OPEN
        o.step(row)
    for i in range(400):
        o.step(np.array(o.rows[-1]))
    pp, Rt = o.body("top")
    B.note(phase="p6", top=np.round(pp, 4).tolist(), top_z=np.round(Rt[:, 2], 4).tolist(),
           robot=contact_forces(m, d, parts, "left_") | contact_forces(m, d, parts, "right_"))
