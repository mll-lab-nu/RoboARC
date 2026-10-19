"""BURVIK P6, closed loop: turn the assembled table upright the way the demo does, on our grasps.

The demo's upright turn (base rows 120500-130500) moves both hands as ONE rigid body (the right hand's pose in the left
hand's frame is constant to 0.0 mm / 0.00 deg): the left hand on leg_2 next to the top, the right hand on the handle's
crossbar; lift, a fast 90 deg turn about world y (124000-125500), a move over the stand, down, release. Replayed as
joint rows it fails here: the base rows' left grasp is 18 mm off leg_2's centre (one finger pushes the leg), the table
does not come up, and the right hand slides off the handle.
Here:
  1. the base rows to K_PRE (the left hand above leg_2, open; the right hand on the handle, as since P3);
  2. the left hand onto leg_2 at world x X_L, centred, pointing down, closed (upright3.grasp: IK to 80 mm above, in, closed);
  3. the demo's rigid motion M(k) (from the base rows' left TCP, forward kinematics) applied to OUR two hand poses at
     the take; joints by joint-limited IK every SAMPLE rows seeded with the base rows' own joints (the demo's arm
     configurations), interpolated in between; the targets keep the take's load offset; plus u2a's integral servo of
     the table's true pose against M(k) * table0 (KC per tick), which makes up for the grasps' creep;
  4. at the first leg-bench contact the motion's descent stops (the hands hold that height); hold; open both hands;
     back each out along its approach axis; joint blend to the base rows' last row (home)."""
import os
import numpy as np
import mujoco
from servo import rotvec
from upright import arm_touch, ease, slerp_R, contact_forces, bench_contact, Hands, CLOSED, OPEN
from plan_up import ik, PARTS
from seat import seat_rot
from upright3 import grasp

K_PRE = 119750
K_GO = int(os.environ.get("BURVIK_UP4_GO", "121000"))       # base row the motion is taken from (both grasps closed)
K_END = int(os.environ.get("BURVIK_UP4_END", "130400"))     # base row before the demo opens its hands
STRIDE = float(os.environ.get("BURVIK_UP4_STRIDE", "1.0"))  # base rows per tick (< 1: slower than the demo)
X_L = float(os.environ.get("BURVIK_UP_XL", "-0.058"))
KC = float(os.environ.get("BURVIK_UP4_KC", "0.05"))
SAMPLE = 25
RDIST_TICKS = int(os.environ.get("BURVIK_UP4_RTICKS", "200"))   # x 0.1 mm
RDIR = os.environ.get("BURVIK_UP4_RDIR", "back")   # the right hand's first move off the handle after the release
RETREAT = os.environ.get("BURVIK_UP4_RETREAT", "polar")   # the left hand's retreat after the release ("axis": along its approach only)
PLACE = os.environ.get("BURVIK_UP4_PLACE", "ik")    # "ik": set the table down before the release; "drop": release in the air
DEBUG = os.environ.get("BURVIK_UP_DEBUG") == "1"


def fk_tcp(m, o, d, row):
    for s in ("left", "right"):
        d.qpos[o.qadr[s]] = row[o.arm[s]]
    mujoco.mj_kinematics(m, d)
    return {s: (d.site_xpos[o.site[s]].copy(), d.site_xmat[o.site[s]].reshape(3, 3).copy()) for s in ("left", "right")}


def upright_p6(B):
    o = B.o; m, d = o.m, o.d
    B.follow(117800, K_PRE, hold=False)
    if os.environ.get("BURVIK_UP4_RECONF", "1") == "1":
        reconf_right(B)
    grasp(B, "left", "leg_2", X_L, 0., {"right": float(o.rows[-1][o.grip["right"]])})
    B.snap("p6_take", K_PRE)
    upright_move(B)


def reconf_right(B, back=0.06):
    """the right hand re-takes the handle in the demo's arm configuration: since P3 the right arm holds the handle in
    another IK branch (joint offsets up to 1.4 rad from the demo's), whose elbow rests on the top and cannot lift.
    Open, back out along the approach axis, joint move to the demo-branch IK of the same pre-grasp pose (seeded with
    the base row K_GO's right arm), in along the approach axis, close."""
    o = B.o; m, d = o.m, o.d; H = Hands(B); s = "right"; gl = float(o.rows[-1][o.grip["left"]])
    for i in range(300):
        row = np.array(o.rows[-1]); row[o.grip[s]] = OPEN; o.step(row)
    mujoco.mj_kinematics(m, d)
    P = d.site_xpos[o.site[s]].copy(); R0_ = d.site_xmat[o.site[s]].reshape(3, 3).copy()
    for i in range(800):
        H.go({s: (P - back * ease(i, 600) * R0_[:, 2], R0_)}, {s: OPEN, "left": gl})
    # the new grasp: the same point on the crossbar, the demo's hand orientation there (its arm branch)
    R = fk_tcp(m, o, mujoco.MjData(m), B.base[K_GO])[s][1]
    d2 = mujoco.MjData(m); d2.qpos[:] = d.qpos
    q, ep, ew = ik(m, d2, o, s, P - back * R[:, 2], R, np.asarray(B.base[K_GO][o.arm[s]], float))
    q0 = np.array(o.rows[-1])[o.arm[s]]
    B.note(phase="p6_reconf_ik", e_mm=round(1000 * ep, 2), dq=np.round(q - d.qpos[o.qadr[s]], 3).tolist(),
           d_demo=np.round(q - np.asarray(B.base[K_GO][o.arm[s]], float), 3).tolist(), turn_deg=round(float(np.degrees(np.linalg.norm(rotvec(R @ R0_.T)))), 1))
    n = max(1500, int(np.max(np.abs(q - q0)) / 0.5 / o.s.control_dt))
    off = q0 - d.qpos[o.qadr[s]]                           # the servo's load offset
    for i in range(n + 300):
        row = np.array(o.rows[-1]); row[o.arm[s]] = q0 + ease(i, n) * (q + off - q0); row[o.grip[s]] = OPEN; row[o.grip["left"]] = gl
        o.step(row)
    for i in range(1400):
        H.go({s: (P - back * (1 - ease(i, 1200)) * R[:, 2], R)}, {s: OPEN, "left": gl})
    for i in range(500):
        H.go({s: (P, R)}, {s: CLOSED, "left": gl})
    B.note(phase="p6_reconf", grasp=contact_forces(m, d, {m.body("handle").id}, "right_"), touch=arm_touch(m, d, s))


def upright_move(B):
    if os.environ.get("BURVIK_UP4_JOINT", "1") == "1":
        return upright_move_joint(B)
    return upright_move_rigid(B)


def upright_move_joint(B):
    """the base rows' joints from K_GO, each arm shifted by its joint offset at the take (our grasp vs the demo's)."""
    o = B.o; m, d = o.m, o.d
    top = m.body("top").id; parts = {m.body(n).id for n in PARTS}
    off = {s: np.array(o.rows[-1])[o.arm[s]] - np.asarray(B.base[K_GO][o.arm[s]], float) for s in ("left", "right")}
    for s in off:                                   # the wrist by whole turns
        off[s][6] -= 2 * np.pi * np.round(off[s][6] / (2 * np.pi))
    B.note(phase="p6_joint_off", off={s: np.round(off[s], 3).tolist() for s in off})
    hold = None
    for k in range(K_GO, K_END):
        row = np.array(B.base[k], dtype=np.float64) if hold is None else np.array(o.rows[-1])
        if hold is None:
            for s in ("left", "right"):
                row[o.arm[s]] += off[s]; row[o.grip[s]] = CLOSED
        o.step(row)
        if hold is None and bench_contact(m, d, parts):
            hold = k; B.note(phase="p6_bench", k=k, top=np.round(d.xpos[top], 4).tolist())
        if DEBUG and k % 400 == 0:
            B.note(phase="dbg_joint", k=k, top=np.round(d.xpos[top], 3).tolist(), top_z=np.round(d.xmat[top].reshape(3, 3)[:, 2], 3).tolist(),
                   fl=contact_forces(m, d, {m.body("leg_2").id}, "left_"), fr=contact_forces(m, d, {m.body("handle").id}, "right_"),
                   arms={sd: {k_: v_ for k_, v_ in arm_touch(m, d, sd).items() if "finger" not in k_} for sd in ("left", "right")})
    if hold is None:
        place_down(B, parts)
    release_home(B, parts)


def place_down(B, parts, descent=0.005, kp=0.08, vmax=0.0004, wmax=0.002):
    """both hands as one rigid grip (seat.py's servo): the top levelled (its normal to world +z) and lowered at
    `descent` m/s until a leg touches the bench; then held 300 ticks."""
    o = B.o; m, d = o.m, o.d; top = m.body("top").id
    lim = {s: o.s.step_limit[o.arm[s]] * 0.9 for s in ("left", "right")}
    mujoco.mj_kinematics(m, d)
    R_up = seat_rot(d.xmat[top].reshape(3, 3), 90.); p_t = d.xpos[top].copy(); t = 0

    def tick(p_star):
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        p = d.xpos[top].copy(); e_p = p_star - p; e_w = rotvec(R_up @ d.xmat[top].reshape(3, 3).T)
        v = kp * e_p; w = kp * e_w
        if np.linalg.norm(v) > vmax: v *= vmax / np.linalg.norm(v)
        if np.linalg.norm(w) > wmax: w *= wmax / np.linalg.norm(w)
        row = np.array(o.rows[-1])
        for s in ("left", "right"):
            row[o.arm[s]] += np.clip(o.dls(s, v + np.cross(w, d.site_xpos[o.site[s]] - p), w), -lim[s], lim[s]); row[o.grip[s]] = CLOSED
        o.step(row)
        return e_p, e_w
    for i in range(600):                                   # level first
        tick(p_t)
    while not bench_contact(m, d, parts) and t < 40000:
        p_t = p_t - np.array([0., 0., descent * o.s.control_dt]); t += 1
        e_p, e_w = tick(p_t)
    for i in range(300):
        tick(d.xpos[top].copy())
    B.note(phase="p6_placed", ticks=t, top=np.round(d.xpos[top], 4).tolist(), tilt_deg=round(float(np.degrees(np.arccos(min(1., d.xmat[top].reshape(3, 3)[2, 2])))), 3),
           world=contact_forces(m, d, parts))


def upright_move_rigid(B):
    o = B.o; m, d = o.m, o.d
    top = m.body("top").id; parts = {m.body(n).id for n in PARTS}
    dk = mujoco.MjData(m)
    T0 = fk_tcp(m, o, dk, B.base[K_GO])
    mujoco.mj_kinematics(m, d)
    h0 = {s: (d.site_xpos[o.site[s]].copy(), d.site_xmat[o.site[s]].reshape(3, 3).copy()) for s in ("left", "right")}
    c0 = d.xpos[top].copy(); R0 = d.xmat[top].reshape(3, 3).copy()

    def M(kf):
        """the demo's rigid motion from K_GO to base row kf (interpolated): x -> R x + p (world)."""
        i = int(kf); f = kf - i
        Ta = fk_tcp(m, o, dk, (1 - f) * B.base[i] + f * B.base[min(i + 1, len(B.base) - 1)])
        R = Ta["left"][1] @ T0["left"][1].T
        return R, Ta["left"][0] - R @ T0["left"][0]

    n = int((K_END - K_GO) / STRIDE)
    ts = list(range(0, n + 1, SAMPLE))
    q = {s: np.array(d.qpos[o.qadr[s]]) for s in ("left", "right")}
    plan = {s: [] for s in q}; worst = 0.
    d2 = mujoco.MjData(m); d2.qpos[:] = d.qpos
    # seed offset: our arms' joints minus the base row's at the take (the IK stays in the demo's configuration family)
    seed_off = {s: q[s] - np.asarray(B.base[K_GO][o.arm[s]], float) for s in q}
    for t in ts:
        kf = K_GO + t * STRIDE; R, p = M(kf)
        for s in ("left", "right"):
            i = int(kf); seed = np.asarray(B.base[i][o.arm[s]], float) + seed_off[s]
            q[s], ep, ew = ik(m, d2, o, s, R @ h0[s][0] + p, R @ h0[s][1], seed if t == 0 else 0.5 * (seed + q[s]))
            plan[s].append(q[s].copy()); worst = max(worst, ep)
    plan = {s: np.array(v) for s, v in plan.items()}
    jump = {s: round(float(np.max(np.abs(np.diff(plan[s], axis=0)))), 4) for s in plan}
    B.note(phase="p6_plan", samples=len(ts), worst_ik_mm=round(1000 * worst, 3), max_jump_rad=jump)
    off0 = {s: np.array(o.rows[-1])[o.arm[s]] - plan[s][0] for s in plan}
    corr = {s: np.zeros(7) for s in plan}; lim = {s: o.s.step_limit[o.arm[s]] * 0.3 for s in plan}
    t = 0; hold_z = None
    while t < n:
        k = t // SAMPLE; f = (t - k * SAMPLE) / SAMPLE
        R, p = M(K_GO + t * STRIDE)
        if hold_z is not None:                       # on the bench: no further descent
            p = p.copy(); p[2] = max(p[2], hold_z)
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        e_p = R @ c0 + p - d.xpos[top]; e_w = rotvec(R @ R0 @ d.xmat[top].reshape(3, 3).T)
        v = np.clip(KC * e_p, -0.0004, 0.0004); w = np.clip(KC * e_w, -0.002, 0.002)
        row = np.array(o.rows[-1])
        for s in ("left", "right"):
            if KC > 0:
                corr[s] += np.clip(o.dls(s, v + np.cross(w, d.site_xpos[o.site[s]] - d.xpos[top]), w), -lim[s], lim[s])
            qt = (1 - f) * plan[s][k] + f * plan[s][min(k + 1, len(ts) - 1)]
            if hold_z is not None:
                qt = q_hold[s]
            row[o.arm[s]] = qt + off0[s] + corr[s]; row[o.grip[s]] = CLOSED
        o.step(row); t += 1
        if hold_z is None and bench_contact(m, d, parts):
            hold_z = float(p[2]); q_hold = {s: (1 - f) * plan[s][k] + f * plan[s][min(k + 1, len(ts) - 1)] for s in plan}
            B.note(phase="p6_bench", t=t, top=np.round(d.xpos[top], 4).tolist())
        if DEBUG and t % 400 == 0:
            B.note(phase="dbg_move", t=t, top_err_mm=round(1000 * float(np.linalg.norm(e_p)), 2), top_err_deg=round(float(np.degrees(np.linalg.norm(e_w))), 2),
                   fl=contact_forces(m, d, {m.body("leg_2").id}, "left_"), fr=contact_forces(m, d, {m.body("handle").id}, "right_"),
                   arms={sd: {k_: v_ for k_, v_ in arm_touch(m, d, sd).items() if "finger" not in k_} for sd in ("left", "right")},
                   corr={sd: round(float(np.max(np.abs(corr[sd]))), 3) for sd in corr})
    if hold_z is None and PLACE == "ik":
        place_ik(B, parts)
    release_home(B, parts)


def feet_shift(m, d, c0, Rrel, step=0.005, span=0.06):
    """the smallest horizontal shift of the levelled table (rotation Rrel about c0) whose four leg tips are farthest from
    the stand's posts (top_stand_*; best min distance first, then the smallest shift)."""
    posts = [d.geom_xpos[m.geom(f"top_stand_{i}").id][:2].copy() for i in range(4)]
    feet = []
    for n in ("leg_0", "leg_1", "leg_2", "leg_3"):
        b = m.body(n).id; tip = d.xpos[b] - 0.42 * d.xmat[b].reshape(3, 3)[:, 2]
        feet.append((Rrel @ (tip - c0) + c0)[:2])
    best = None
    for dx in np.arange(-span, span + 1e-9, step):
        for dy in np.arange(-span, span + 1e-9, step):
            sh = np.array([dx, dy])
            dmin = min(float(np.linalg.norm(f + sh - p)) for f in feet for p in posts)
            key = (round(dmin, 3), -float(np.linalg.norm(sh)))
            if best is None or key > best[0]:
                best = (key, sh, dmin)
    return best[1], best[2]


def place_ik(B, parts, n_level=1600, descent=0.005, sample=25):
    """the table set down on its feet before the release. Its path: level (normal to world +z) while moving horizontally
    to where its feet are clear of the stand's posts (feet_shift), then straight down at `descent` m/s. Each `sample`
    ticks both arms are re-solved by joint-limited IK for that path composed with the grasps AS THEY ARE NOW (each hand's
    pose in the table's true frame: the grasps creep, so this keeps the table, not the hands, on the path), interpolated,
    the load offset kept; stopped at the first leg-bench contact, held 300 ticks."""
    o = B.o; m, d = o.m, o.d; top = m.body("top").id
    mujoco.mj_kinematics(m, d)
    c0 = d.xpos[top].copy(); R0 = d.xmat[top].reshape(3, 3).copy(); Rup = seat_rot(R0, 90.); Rrel = Rup @ R0.T
    shift, clear = feet_shift(m, d, c0, Rrel)
    q = {s: d.qpos[o.qadr[s]].copy() for s in ("left", "right")}
    off = {s: np.array(o.rows[-1])[o.arm[s]] - q[s] for s in q}
    d2 = mujoco.MjData(m); d2.qpos[:] = d.qpos

    def pose(t):                                    # the table's planned pose at tick t
        a = 0.5 - 0.5 * np.cos(np.pi * min(1., t / n_level))
        z = -descent * o.s.control_dt * max(0, t - n_level)
        return slerp_R(R0, Rup, a), c0 + np.array([a * shift[0], a * shift[1], z])

    def solve(t):
        mujoco.mj_kinematics(m, d)
        Rt = d.xmat[top].reshape(3, 3); pt = d.xpos[top]; Rp, pp = pose(t)
        for s in q:
            gp = Rt.T @ (d.site_xpos[o.site[s]] - pt); gR = Rt.T @ d.site_xmat[o.site[s]].reshape(3, 3)
            q[s], _, _ = ik(m, d2, o, s, pp + Rp @ gp, Rp @ gR, q[s])
        return {s: q[s].copy() for s in q}

    prev = {s: d.qpos[o.qadr[s]].copy() for s in q}; nxt = solve(sample); t = 0; contact = None
    while t < n_level + 30000:
        f = (t % sample) / sample
        row = np.array(o.rows[-1])
        for s in q:
            row[o.arm[s]] = (1 - f) * prev[s] + f * nxt[s] + off[s]; row[o.grip[s]] = CLOSED
        o.step(row); t += 1
        if t % sample == 0:
            prev = nxt; nxt = solve(t + sample)
        if t > n_level and bench_contact(m, d, parts):
            contact = t; break
    for i in range(300):
        o.step(np.array(o.rows[-1]))
    B.note(phase="p6_placed", ticks=t, contact=contact, shift_mm=np.round(1000 * shift, 1).tolist(), feet_clear_mm=round(1000 * clear, 1),
           top=np.round(d.xpos[top], 4).tolist(), tilt_deg=round(float(np.degrees(np.arccos(min(1., d.xmat[top].reshape(3, 3)[2, 2])))), 3),
           world=contact_forces(m, d, parts))


def polar_retreat(B, side, r_out=0.36, n_out=1200, n_arc=2400):
    """the side's open hand away from the standing table without sweeping over it: straight out from the table's
    vertical axis to radius r_out at the hand's height, then along that circle to the azimuth of the base rows' last
    (home) TCP, its orientation held; the joint blend home then starts outside the table."""
    o = B.o; m, d = o.m, o.d; top = m.body("top").id; H = Hands(B)
    mujoco.mj_kinematics(m, d)
    c = d.xpos[top][:2].copy(); p0 = d.site_xpos[o.site[side]].copy(); R = d.site_xmat[o.site[side]].reshape(3, 3).copy()
    home = fk_tcp(m, o, mujoco.MjData(m), B.base[-1])[side][0]
    a0 = np.arctan2(p0[1] - c[1], p0[0] - c[0]); r0 = float(np.linalg.norm(p0[:2] - c))
    a1 = np.arctan2(home[1] - c[1], home[0] - c[0]); da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
    grips = {"left": OPEN, "right": OPEN}
    def at(r, a): return np.array([c[0] + r * np.cos(a), c[1] + r * np.sin(a), p0[2]])
    for i in range(n_out + 100):
        H.go({side: (at(r0 + ease(i, n_out) * (max(r_out, r0) - r0), a0), R)}, grips)
    for i in range(n_arc + 100):
        H.go({side: (at(max(r_out, r0), a0 + ease(i, n_arc) * da), R)}, grips)
    B.note(phase="p6_retreat", side=side, r0_mm=round(1000 * r0), arc_deg=round(float(np.degrees(da)), 1),
           touch={k: v for k, v in arm_touch(m, d, side).items()})


def release_home(B, parts):
    o = B.o; m, d = o.m, o.d; top = m.body("top").id
    B.snap("p6_placed", len(B.base))
    B.note(phase="p6_down", top=np.round(d.xpos[top], 4).tolist(), top_z=np.round(d.xmat[top].reshape(3, 3)[:, 2], 4).tolist(),
           world=contact_forces(m, d, parts))
    for i in range(300):
        o.step(np.array(o.rows[-1]))
    lim = {s: o.s.step_limit[o.arm[s]] * 0.9 for s in ("left", "right")}
    for i in range(400):
        row = np.array(o.rows[-1]); row[o.grip["left"]] = OPEN; row[o.grip["right"]] = OPEN; o.step(row)
    mujoco.mj_kinematics(m, d)
    ax = {s: d.site_xmat[o.site[s]].reshape(3, 3)[:, 2].copy() for s in ("left", "right")}
    if RDIR != "back":                          # the right hand first leaves the handle's crossbar this way
        Rr = d.site_xmat[o.site["right"]].reshape(3, 3)
        v = {"up": np.array([0., 0., 1.]), "down": np.array([0., 0., -1.]), "tcpx+": Rr[:, 0], "tcpx-": -Rr[:, 0],
             "tcpy+": Rr[:, 1], "tcpy-": -Rr[:, 1]}[RDIR]
        for i in range(RDIST_TICKS):
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            row = np.array(o.rows[-1])
            row[o.arm["right"]] += np.clip(o.dls("right", v * 0.0001, np.zeros(3)), -lim["right"], lim["right"])
            o.step(row)
        mujoco.mj_kinematics(m, d)
        B.note(phase="p6_right_off", dir=RDIR, table=np.round(d.xpos[top], 4).tolist(), touch=arm_touch(m, d, "right"))
        ax = {s: d.site_xmat[o.site[s]].reshape(3, 3)[:, 2].copy() for s in ("left", "right")}
    p_open = {s: d.site_xpos[o.site[s]].copy() for s in ax}
    for i in range(1000):
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        row = np.array(o.rows[-1])
        for s in ("left", "right"):
            row[o.arm[s]] += np.clip(o.dls(s, -ax[s] * 0.0001, np.zeros(3)), -lim[s], lim[s])
        o.step(row)
    mujoco.mj_kinematics(m, d)
    B.note(phase="p6_backout", moved_mm={s: np.round(1000 * (d.site_xpos[o.site[s]] - p_open[s]), 1).tolist() for s in ax},
           approach={s: np.round(ax[s], 3).tolist() for s in ax}, tcp={s: np.round(d.site_xpos[o.site[s]], 3).tolist() for s in ax},
           table=np.round(d.xpos[top], 3).tolist(), touch={s: arm_touch(m, d, s) for s in ax})
    if RETREAT == "polar":
        polar_retreat(B, "left")
    start = np.array(o.rows[-1]); end = np.array(B.base[-1], dtype=np.float64)
    for s in ("left", "right"):
        j = o.arm[s][6]; end[j] = end[j] + 2 * np.pi * np.round((start[j] - end[j]) / (2 * np.pi))
    nb = 3200
    for i in range(nb):
        a = ease(i + 1, nb); row = (1 - a) * start + a * end
        row[o.grip["left"]] = OPEN; row[o.grip["right"]] = OPEN
        o.step(row)
    for i in range(400):
        o.step(np.array(o.rows[-1]))
    pp, Rt = o.body("top")
    B.note(phase="p6", top=np.round(pp, 4).tolist(), top_z=np.round(Rt[:, 2], 4).tolist(),
           robot=contact_forces(m, d, parts, "left_") | contact_forces(m, d, parts, "right_"))
