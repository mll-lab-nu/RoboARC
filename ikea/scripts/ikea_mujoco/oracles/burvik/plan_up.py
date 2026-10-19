"""Kinematic planning for BURVIK's upright turn (no physics): joint-limited IK of both arms along the table's path, with
the table moved rigidly in a model copy, and a collision check of the arms against the table, the fixtures and each
other. Used to choose the grasps (leg, x, approach angle) and the path before the dynamic run (upright.py).

  ik(m, d, side, p, R, q_seed)  -> q (7), error (m, rad)    damped least squares, clamped to the joint ranges
  grasp_pose(...)               -> TCP (p, R) on a leg: the approach tilted phi from straight down toward the arm's side
  check(...)                    -> per-sample IK errors, joint-limit margins, collisions
"""
import numpy as np
import mujoco
from servo import rotvec

PARTS = ("top", "leg_0", "leg_1", "leg_2", "leg_3", "handle")


def arm_ids(m, o, side):
    jid = [int(m.actuator_trnid[a, 0]) for a in o.arm[side]]
    return (np.array([m.jnt_qposadr[j] for j in jid]), np.array([m.jnt_dofadr[j] for j in jid]),
            np.array([m.jnt_range[j] for j in jid]), np.array([bool(m.jnt_limited[j]) for j in jid]))


def ik(m, d, o, side, p, R, q_seed, iters=200, lam=1e-3, margin=0.05):
    qa, va, rng, lim = arm_ids(m, o, side)
    lo = np.where(lim, rng[:, 0] + margin, -np.inf); hi = np.where(lim, rng[:, 1] - margin, np.inf)
    q = np.clip(np.array(q_seed, float), lo, hi); site = o.site[side]
    jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    for _ in range(iters):
        d.qpos[qa] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        e_p = p - d.site_xpos[site]; e_w = rotvec(R @ d.site_xmat[site].reshape(3, 3).T)
        if np.linalg.norm(e_p) < 1e-5 and np.linalg.norm(e_w) < 1e-4:
            break
        mujoco.mj_jacSite(m, d, jp, jr, site)
        J = np.vstack([jp[:, va], jr[:, va]]); e = np.concatenate([e_p, e_w])
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
        q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)
    d.qpos[qa] = q; mujoco.mj_kinematics(m, d)
    e_p = float(np.linalg.norm(p - d.site_xpos[site])); e_w = float(np.linalg.norm(rotvec(R @ d.site_xmat[site].reshape(3, 3).T)))
    return q, e_p, e_w


def leg_frame(m, d, leg):
    b = m.body(leg).id; R = d.xmat[b].reshape(3, 3); return d.xpos[b].copy(), -R[:, 2]


def grasp_pose(m, d, leg, x, side, phi_deg):
    """TCP pose on leg at world x: approach = cos(phi) * down + sin(phi) * toward the other arm's side (left: +y),
    made perpendicular to the leg; fingers (TCP y) across the leg."""
    p0, a = leg_frame(m, d, leg)
    P = p0 + a * (x - p0[0]) / a[0]
    ph = np.radians(phi_deg); s = 1. if side == "left" else -1.
    z = np.cos(ph) * np.array([0, 0, -1.]) + np.sin(ph) * np.array([0, s, 0.])
    z = z - (z @ a) * a; z /= np.linalg.norm(z)
    y = np.cross(z, a); y /= np.linalg.norm(y)
    return P, np.column_stack([np.cross(y, z), y, z])


def table_qpos(m, d, T_rel, c0, ids):
    """qpos with every table part moved by the rigid motion T_rel = (R, t) about the world origin (x -> R x + t)."""
    q = d.qpos.copy(); R, t = T_rel
    qr = np.zeros(4); mujoco.mju_mat2Quat(qr, R.reshape(9))
    for b in ids:
        j = m.body_jntadr[b]; a = m.jnt_qposadr[j]
        q[a:a + 3] = R @ d.qpos[a:a + 3] + t
        qq = np.zeros(4); mujoco.mju_mulQuat(qq, qr, d.qpos[a + 3:a + 7]); q[a + 3:a + 7] = qq
    return q


def collisions(m, d, o, sides=("left", "right"), ignore_fingers_on=()):
    """contacts with penetration of the arms (links, hand, fingers) against anything not of the same arm, except the
    fingers of `side` on its grasped leg (ignore_fingers_on: [(side, leg)])."""
    mujoco.mj_collision(m, d)
    out = []
    for c in d.contact[:d.ncon]:
        if c.dist > 0:
            continue
        n1, n2 = m.body(int(m.geom_bodyid[c.geom1])).name, m.body(int(m.geom_bodyid[c.geom2])).name
        s1 = next((s for s in sides if n1.startswith(s + "_")), None); s2 = next((s for s in sides if n2.startswith(s + "_")), None)
        if s1 is None and s2 is None or s1 == s2:
            continue
        skip = any((s1 == s and "finger" in n1 and n2 == leg) or (s2 == s and "finger" in n2 and n1 == leg) for s, leg in ignore_fingers_on)
        if not skip:
            out.append((n1, n2, round(1000 * float(c.dist), 1)))
    return out
