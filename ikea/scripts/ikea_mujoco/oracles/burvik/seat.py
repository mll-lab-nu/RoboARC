"""BURVIK P1, closed loop: seat the top in the jig.

The base rows release the top ~25 mm above the jig's pads with its rim behind the guards' front faces; it then drops onto
the guards' tops (lean ~6 deg forward onto the posts) or tips over. The jig's own seat is the slot between the guards
(behind) and the posts (in front): the rim on the two pads, leaning ~0.5 deg onto the posts. Measured: a top set there
settles at rest (`probe/seat2.py`).

Here both arms (the top held in both hands) are servoed as one rigid body: the top's pose error gives its twist, each
TCP gets that rigid twist (v + w x (tcp - p)), mapped through damped least squares, added to the arm's last executed
targets. Phases: from base row K_TAKE, align over the slot (target lean SEAT_LEAN deg, centre SEAT_X, 0, SEAT_Z +
ABOVE); lower at DESCENT m/s until the rim touches a pad (or SEAT_Z - 1 mm); hold; open both hands (the base open
value); blend back to the base rows of the retreat."""
import os
import numpy as np
import mujoco
from servo import rotvec

K_TAKE = 64500
SEAT_X = float(os.environ.get("BURVIK_SEAT_X", "-0.1600"))
SEAT_Y = float(os.environ.get("BURVIK_SEAT_Y", "0.0"))
SEAT_Z = float(os.environ.get("BURVIK_SEAT_Z", "0.3920"))
SEAT_LEAN = float(os.environ.get("BURVIK_SEAT_LEAN", "0.3"))
ABOVE = 0.004
DESCENT = float(os.environ.get("BURVIK_SEAT_DESCENT", "0.002"))


def seat_rot(R, lean_deg):
    """the minimal rotation of R that puts the top's z axis (the handle side) at world (-cos L, 0, sin L)."""
    L = np.radians(lean_deg)
    a = R[:, 2]; b = np.array([-np.cos(L), 0., np.sin(L)])
    c = np.cross(a, b); s = np.linalg.norm(c)
    if s < 1e-12:
        return R.copy()
    k = c / s; th = np.arctan2(s, float(a @ b))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return (np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K) @ R


def pad_contact(m, d, top):
    for c in d.contact[:d.ncon]:
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if {b1, b2} == {top, 0} and c.efc_address >= 0:
            g = c.geom1 if b1 == 0 else c.geom2
            if m.geom(int(g)).name.endswith("_pad"):
                return True
    return False


def seat_p1(B, kp=0.08, kr=0.08, vmax=0.00025, wmax=0.001):
    o = B.o; m, d = o.m, o.d
    top = m.body("top").id
    o.run(B.base[64000:K_TAKE])
    lim = {s: o.s.step_limit[o.arm[s]] * 0.9 for s in ("left", "right")}
    grip = {s: float(B.base[K_TAKE][o.grip[s]]) for s in ("left", "right")}
    mujoco.mj_kinematics(m, d)
    R_star = seat_rot(d.xmat[top].reshape(3, 3), SEAT_LEAN)
    z_t = SEAT_Z + ABOVE; phase = "align"; t = 0; hold = 0

    def tick(p_star, grips):
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
        return float(np.linalg.norm(e_p)), float(np.degrees(np.linalg.norm(e_w)))

    while True:
        t += 1
        ep, ew = tick(np.array([SEAT_X, SEAT_Y, z_t]), grip)
        if phase == "align" and ep < 0.0005 and ew < 0.2:
            phase = "lower"; B.note(phase="seat_aligned", ticks=t, e_mm=round(ep * 1000, 3), e_deg=round(ew, 3))
        elif phase == "lower":
            if pad_contact(m, d, top) or z_t <= SEAT_Z - 0.001:
                phase = "hold"
            else:
                z_t -= DESCENT * o.s.control_dt
        elif phase == "hold":
            hold += 1
            if hold >= 200:
                break
        if t > 20000:
            B.note(phase="seat_timeout", e_mm=round(ep * 1000, 3), e_deg=round(ew, 3)); break
    p, Rm = o.body("top")
    B.note(phase="seat_lowered", ticks=t, top=np.round(p, 4).tolist(), top_z=np.round(Rm[:, 2], 4).tolist(), pad=pad_contact(m, d, top))
    # open both hands, arms held
    opened = {s: float(B.base[67600][o.grip[s]]) for s in ("left", "right")}
    for i in range(300):
        row = np.array(o.rows[-1]); row[o.grip["left"]] = opened["left"]; row[o.grip["right"]] = opened["right"]
        o.step(row)
    # the base retreat, shifted by the arms' offset at its start, the offset ramped out
    k0, k1, k2 = 67600, 68800, 70000
    off = {s: np.array(o.rows[-1])[o.arm[s]] - B.base[k0][o.arm[s]] for s in ("left", "right")}
    for k in range(k0, k2):
        a = min(1., (k - k0) / (k1 - k0))
        row = np.array(B.base[k], dtype=np.float64)
        for s in ("left", "right"):
            row[o.arm[s]] += (1 - a) * off[s]
        o.step(row)
    p, Rm = o.body("top")
    lean = float(np.degrees(np.arcsin(Rm[2, 2])))
    B.note(phase="p1", top=np.round(p, 4).tolist(), top_z=np.round(Rm[:, 2], 4).tolist(), lean_deg=round(lean, 3),
           v=round(float(np.linalg.norm(d.qvel[m.jnt_dofadr[m.body_jntadr[top]]:][:3])), 5))
