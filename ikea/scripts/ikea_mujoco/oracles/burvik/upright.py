"""BURVIK P6, closed loop: turn the assembled table upright.

The base rows lift and turn the table with the left hand on leg_2 (its fingers off the leg's centre: one finger pushes
the leg) and the right hand still on the handle, whose elbow rests on the table: the right hand cannot rise, the table
wedges in the jig and both grasps slip. Here:
  1. the base rows to K_PRE (the left hand above leg_2, open); the right hand opens, backs out of the handle and goes
     beside the top onto leg_0 (the other upper leg) at world x X_R, centred, from the arm's own side (approach along y), the fingers across the leg;
  2. the left hand onto leg_2 at world x X_L, the same way; both closed;
  3. both arms servoed as one rigid body with the top (seat.py's rule: the top's pose error -> its twist -> each TCP's
     rigid twist -> damped least squares), along pose targets interpolated in time: straight up by LIFT, then 90 deg
     about world y (the handle up, the legs down) about the grasp line while that line moves so that the top ends
     at PLACE_Z over (PLACE_X, PLACE_Y); then down at DESCENT m/s until a leg touches the bench;
  4. hold, open both hands, back each out along its approach axis, blend to the base rows' last row (home)."""
import os
import numpy as np
import mujoco
from servo import rotvec
from seat import seat_rot

K_PRE = 119750                       # the base rows' left hand above leg_2, open, pointing down
X_L = float(os.environ.get("BURVIK_UP_XL", "-0.057"))     # world x of the left grasp on leg_2
X_R = float(os.environ.get("BURVIK_UP_XR", "0.12"))       # world x of the right grasp on leg_0
LIFT = float(os.environ.get("BURVIK_UP_LIFT", "0.25"))
PLACE_X = float(os.environ.get("BURVIK_UP_X", "0.0"))
PLACE_Y = float(os.environ.get("BURVIK_UP_Y", "0.0"))
PLACE_Z = float(os.environ.get("BURVIK_UP_Z", "0.575"))    # the top's centre just above its standing height
DESCENT = 0.005
CLOSED, OPEN = 0., 255.
DEBUG = os.environ.get("BURVIK_UP_DEBUG") == "1"


def ease(i, n):
    return 0.5 - 0.5 * np.cos(np.pi * min(1., i / n))


def slerp_R(R0, R1, a):
    w = rotvec(R1 @ R0.T); th = np.linalg.norm(w)
    if th * a < 1e-12:
        return R0.copy()
    k = w / th; t = a * th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return (np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K) @ R0


def contact_forces(m, d, bodies, other_prefix=None):
    """{name: N} of contacts between `bodies` and anything outside them (optionally only bodies named other_prefix*)."""
    out = {}
    for c in d.contact[:d.ncon]:
        if c.efc_address < 0:
            continue
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if (b1 in bodies) != (b2 in bodies):
            g = c.geom2 if b1 in bodies else c.geom1
            bn = m.body(int(m.geom_bodyid[g])).name
            if other_prefix and not bn.startswith(other_prefix):
                continue
            n = m.geom(int(g)).name or bn
            out[n] = round(out.get(n, 0) + abs(float(d.efc_force[c.efc_address])), 1)
    return out


def bench_contact(m, d, bodies):
    bench = m.geom("bench").id
    for c in d.contact[:d.ncon]:
        if c.efc_address >= 0 and bench in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == bench else c.geom1
            if int(m.geom_bodyid[other]) in bodies:
                return True
    return False


def arm_touch(m, d, side):
    out = {}
    for c in d.contact[:d.ncon]:
        n1, n2 = m.body(int(m.geom_bodyid[c.geom1])).name, m.body(int(m.geom_bodyid[c.geom2])).name
        if c.efc_address >= 0 and (n1.startswith(side + "_") != n2.startswith(side + "_")):
            k = n1 + "|" + n2; out[k] = round(out.get(k, 0) + abs(float(d.efc_force[c.efc_address])), 1)
    return out


def leg_point(m, d, leg, x):
    """the point of the leg's axis at world x, and the axis (unit, toward the tip)."""
    b = m.body(leg).id; R = d.xmat[b].reshape(3, 3); a = -R[:, 2]; p = d.xpos[b]
    return p + a * (x - p[0]) / a[0], a


class Hands:
    def __init__(self, B):
        self.B = B; self.o = B.o; self.m, self.d = B.o.m, B.o.d
        self.lim = {s: self.o.s.step_limit[self.o.arm[s]] * 0.9 for s in ("left", "right")}

    def go(self, targets, grips, kp=0.1, vmax=0.0003, wmax=0.003):
        """one tick: each side in targets (p*, R*) servoed (TCP pose), the others' arm targets held; grips per side."""
        o, m, d = self.o, self.m, self.d
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        row = np.array(o.rows[-1]); err = {}
        for s, (p_star, Rs) in targets.items():
            e_p = p_star - d.site_xpos[o.site[s]]; e_w = rotvec(Rs @ d.site_xmat[o.site[s]].reshape(3, 3).T)
            v = kp * e_p; w = kp * e_w
            if np.linalg.norm(v) > vmax: v *= vmax / np.linalg.norm(v)
            if np.linalg.norm(w) > wmax: w *= wmax / np.linalg.norm(w)
            row[o.arm[s]] += np.clip(o.dls(s, v, w), -self.lim[s], self.lim[s]); err[s] = float(np.linalg.norm(e_p))
        for s, g in grips.items():
            row[o.grip[s]] = g
        o.step(row)
        return err

    def path(self, side, waypoints, grip, grips_other, n_per_m=8000, n_rot=800):
        """the side's TCP along waypoints [(p, R)], each leg interpolated (ease) over ticks set by its length."""
        o, d = self.o, self.d
        mujoco.mj_kinematics(self.m, d)
        p0 = d.site_xpos[o.site[side]].copy(); R0 = d.site_xmat[o.site[side]].reshape(3, 3).copy()
        for p1, R1 in waypoints:
            n = max(n_rot, int(np.linalg.norm(p1 - p0) * n_per_m))
            for i in range(n + 150):
                a = ease(i, n)
                e = self.go({side: (p0 + a * (p1 - p0), slerp_R(R0, R1, a))}, dict(grips_other, **{side: grip}))
                if DEBUG and i % 400 == 0:
                    self.B.note(phase="dbg_path", side=side, i=i, n=n, e_mm=round(1000 * e[side], 1), tcp=np.round(d.site_xpos[o.site[side]], 3).tolist(),
                                q=np.round(d.qpos[o.qadr[side]], 2).tolist(), touch=arm_touch(self.m, d, side))
            p0, R0 = p1, R1
        return p0, R0

    def grasp_R(self, side, leg_axis):
        """from the arm's own side (left: toward +y, right: toward -y; an axis the upright turn leaves fixed), the fingers
        (TCP y) across the leg, the roll nearest the current one."""
        d, o = self.d, self.o
        zt = np.array([0., 1. if side == "left" else -1., 0.]); yt = np.cross(zt, leg_axis); yt /= np.linalg.norm(yt)
        if float(yt @ d.site_xmat[o.site[side]].reshape(3, 3)[:, 1]) < 0:
            yt = -yt
        return np.column_stack([np.cross(yt, zt), yt, zt])

    def regrasp(self, side, leg, x, grips_other, via=None, above=0.08):
        o, m, d = self.o, self.m, self.d
        mujoco.mj_kinematics(m, d)
        P, a = leg_point(m, d, leg, x); Rg = self.grasp_R(side, a)
        wps = []
        Rc = d.site_xmat[o.site[side]].reshape(3, 3).copy()
        for i, v in enumerate(via or []):                            # transfer waypoints, the grasp roll reached on the way
            wps.append((np.asarray(v, float), slerp_R(Rc, Rg, (i + 1) / len(via))))
        wps += [(P - above * Rg[:, 2], Rg)]
        self.path(side, wps, OPEN, grips_other)
        for i in range(1200 + 200):                                   # in along the approach axis, onto the true axis
            mujoco.mj_kinematics(m, d); P, _ = leg_point(m, d, leg, x)
            e = self.go({side: (P - above * (1 - ease(i, 1200)) * Rg[:, 2], Rg)}, dict(grips_other, **{side: OPEN}))
        for i in range(500):
            mujoco.mj_kinematics(m, d); P, _ = leg_point(m, d, leg, x)
            e = self.go({side: (P, Rg)}, dict(grips_other, **{side: CLOSED}))
        self.B.note(phase="p6_regrasp", side=side, leg=leg, e_mm=round(1000 * e[side], 2),
                    fingers=contact_forces(m, d, {m.body(leg).id}, side + "_"))


def upright_p6(B, kp=0.08, kr=0.08, vmax=0.0004, wmax=0.002):
    o = B.o; m, d = o.m, o.d
    top = m.body("top").id
    parts = {m.body(n).id for n in ("top", "leg_0", "leg_1", "leg_2", "leg_3", "handle")}
    H = Hands(B); lim = H.lim
    B.follow(117800, K_PRE, hold=False)
    gl = float(o.rows[-1][o.grip["left"]])
    # 1. the right hand off the handle: open, then onto leg_0
    for i in range(300):
        H.go({}, {"right": OPEN, "left": gl})
    # joint-space to the base rows' last right-arm pose (high over the legs' side; the sweep is collision free)
    q0 = np.array(o.rows[-1])[o.arm["right"]]; qs = np.array(B.base[-1][o.arm["right"]], dtype=np.float64)
    qs[6] += 2 * np.pi * np.round((q0[6] - qs[6]) / (2 * np.pi))
    for i in range(3200):
        row = np.array(o.rows[-1]); row[o.arm["right"]] = q0 + ease(i + 1, 3000) * (qs - q0)
        row[o.grip["right"]] = OPEN; row[o.grip["left"]] = gl; o.step(row)
    H.regrasp("right", "leg_0", X_R, {"left": gl})
    # 2. the left hand onto leg_2
    H.regrasp("left", "leg_2", X_L, {"right": CLOSED})
    B.snap("p6_take", K_PRE)
    upright_move(B, kp, kr, vmax, wmax)


def upright_move(B, kp=0.08, kr=0.08, vmax=0.0004, wmax=0.002):
    """steps 3-4, from both hands closed on the legs (snap p6_take)."""
    o = B.o; m, d = o.m, o.d
    top = m.body("top").id
    parts = {m.body(n).id for n in ("top", "leg_0", "leg_1", "leg_2", "leg_3", "handle")}
    H = Hands(B); lim = H.lim
    closed = {"left": CLOSED, "right": CLOSED}

    state = {}

    def tick(p_star, R_star, grips):
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        p = d.xpos[top].copy(); Rm = d.xmat[top].reshape(3, 3)
        e_p = p_star - p; e_w = rotvec(R_star @ Rm.T)
        v = kp * e_p; w = kr * e_w
        if np.linalg.norm(v) > vmax: v *= vmax / np.linalg.norm(v)
        if np.linalg.norm(w) > wmax: w *= wmax / np.linalg.norm(w)
        row = np.array(o.rows[-1], dtype=np.float64)
        for s in ("left", "right"):
            tcp = d.site_xpos[o.site[s]]
            dq = np.clip(o.dls(s, v + np.cross(w, tcp - p), w), -lim[s], lim[s])
            row[o.arm[s]] = row[o.arm[s]] + dq
            row[o.grip[s]] = grips[s]
        o.step(row)
        state["ep"], state["ew"] = float(np.linalg.norm(e_p)), float(np.degrees(np.linalg.norm(e_w)))

    def dbg(tag, i):
        if DEBUG and i % 200 == 0:
            B.note(phase=tag, i=i, top=np.round(d.xpos[top], 4).tolist(), e_mm=round(state["ep"] * 1000, 2), e_deg=round(state["ew"], 2),
                   tcp={s: np.round(d.site_xpos[o.site[s]], 3).tolist() for s in ("left", "right")},
                   fl=contact_forces(m, d, {m.body("leg_2").id}, "left_"), fr=contact_forces(m, d, {m.body("leg_0").id}, "right_"),
                   world=contact_forces(m, d, parts), ql=np.round(d.qpos[o.qadr["left"]], 2).tolist(), qr=np.round(d.qpos[o.qadr["right"]], 2).tolist())

    mujoco.mj_kinematics(m, d)
    c0 = d.xpos[top].copy(); R0 = d.xmat[top].reshape(3, 3).copy()
    G0 = 0.5 * (d.site_xpos[o.site["left"]] + d.site_xpos[o.site["right"]])
    R1 = seat_rot(R0, 90.0)                       # the top's z axis (the handle side) to world +z
    Rrel = R1 @ R0.T
    G1 = G0 + np.array([0., 0., LIFT])
    cf = np.array([PLACE_X, PLACE_Y, PLACE_Z])
    Gf = cf - Rrel @ (c0 - G0)
    n_lift, n_turn = 2600, 5200
    for i in range(n_lift + 200):
        tick(c0 + np.array([0., 0., LIFT * ease(i, n_lift)]), R0, closed); dbg("dbg_lift", i)
    B.note(phase="p6_lifted", top=np.round(d.xpos[top], 4).tolist(), e_mm=round(state["ep"] * 1000, 2), e_deg=round(state["ew"], 2),
           fl=contact_forces(m, d, {m.body("leg_2").id}, "left_"), fr=contact_forces(m, d, {m.body("leg_0").id}, "right_"))
    for i in range(n_turn + 400):
        a = ease(i, n_turn); Ra = slerp_R(np.eye(3), Rrel, a); G = G1 + a * (Gf - G1)
        tick(G + Ra @ (c0 - G0), Ra @ R0, closed); dbg("dbg_turn", i)
    B.note(phase="p6_turned", top=np.round(d.xpos[top], 4).tolist(), top_z=np.round(d.xmat[top].reshape(3, 3)[:, 2], 4).tolist(),
           e_mm=round(state["ep"] * 1000, 2), e_deg=round(state["ew"], 2),
           fl=contact_forces(m, d, {m.body("leg_2").id}, "left_"), fr=contact_forces(m, d, {m.body("leg_0").id}, "right_"))
    z = PLACE_Z; t = 0
    while not bench_contact(m, d, parts) and t < 40000:
        z -= DESCENT * o.s.control_dt; t += 1
        tick(np.array([PLACE_X, PLACE_Y, z]), R1, closed)
    zc = float(d.xpos[top][2])
    for i in range(300):
        tick(np.array([PLACE_X, PLACE_Y, zc - 0.0005]), R1, closed)
    B.note(phase="p6_down", ticks=t, top=np.round(d.xpos[top], 4).tolist(), e_mm=round(state["ep"] * 1000, 2), e_deg=round(state["ew"], 2),
           world=contact_forces(m, d, parts))
    # 4. release: open both hands, arms held; back out; home
    for i in range(400):
        H.go({}, {"left": OPEN, "right": OPEN})
    mujoco.mj_kinematics(m, d)
    ax = {s: d.site_xmat[o.site[s]].reshape(3, 3)[:, 2].copy() for s in ("left", "right")}
    for i in range(1000):
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        row = np.array(o.rows[-1])
        for s in ("left", "right"):
            row[o.arm[s]] += np.clip(o.dls(s, -ax[s] * 0.0001, np.zeros(3)), -lim[s], lim[s])
        o.step(row)
    start = np.array(o.rows[-1]); end = np.array(B.base[-1], dtype=np.float64)
    for s in ("left", "right"):                   # wrists by whole turns (no unwinding)
        j = o.arm[s][6]; end[j] = end[j] + 2 * np.pi * np.round((start[j] - end[j]) / (2 * np.pi))
    n = 3200
    for i in range(n):
        a = ease(i + 1, n); row = (1 - a) * start + a * end
        row[o.grip["left"]] = OPEN; row[o.grip["right"]] = OPEN
        o.step(row)
    for i in range(400):
        o.step(np.array(o.rows[-1]))
    p, Rm = o.body("top")
    B.note(phase="p6", top=np.round(p, 4).tolist(), top_z=np.round(Rm[:, 2], 4).tolist(),
           robot=contact_forces(m, d, parts, "left_") | contact_forces(m, d, parts, "right_"))
