"""kyrre closed-loop reference: the session wrapper (records the EXECUTED rows, data.ctrl after the clamp), true-state
measurements (screws vs their holes, the hex key's short tip vs a screw's socket) and damped-least-squares Jacobian IK
on the TCP sites. Adapted from oracles/stenerik/st_lib.py."""
from __future__ import annotations
import json, math, os, pickle
import numpy as np
import mujoco
import rcb_ikea as R

ARM = {'left': np.arange(0, 7), 'right': np.arange(8, 15)}
GRIP = {'left': 7, 'right': 15}
ARMCOLS = np.r_[ARM['left'], ARM['right']]
DEMO = os.environ.get('KY_DEMO', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kyrre_demo.npz'))
KEY_SHORT_TIP = np.array([0.025, 0., 0.0881])    # hex key frame: the short arm's tip (layout.json hex_key.info)
SOCKET_FLAT0 = 0.                                # screw frame: a socket flat's normal at angle 0 (head_hex walls at 60 k deg)


def demo_rows():
    return R.load_trajectory(DEMO, 16)


class Rig:
    def __init__(self, session=None):
        self.s = session if session is not None else R.Session('kyrre')
        self.m, self.d = self.s.model, self.s.data
        self.M = self.s.S.mech
        self.ex: list = []
        m = self.m
        self.lim = np.asarray(self.s.step_limit, dtype=np.float64)       # rad per tick (inf for grippers)
        self.site = {a: m.site(a + '_tcp').id for a in ARM}
        self.dof = {a: np.array([int(m.jnt_dofadr[m.joint(f'{a}_joint{k}').id]) for k in range(1, 8)]) for a in ARM}
        self.qadr = {a: np.array([int(m.jnt_qposadr[m.joint(f'{a}_joint{k}').id]) for k in range(1, 8)]) for a in ARM}
        self.bid = {m.body(i).name: i for i in range(m.nbody)}
        self.ops = list(self.M.ops)
        self.opi = {o['part']: i for i, o in enumerate(self.ops)}
        self.screws = [o['part'] for o in self.ops if o['mode'] == 'screw']
        self.jp = np.zeros((3, m.nv)); self.jr = np.zeros((3, m.nv))
        self.finger = {a: set(i for i in range(m.nbody) if m.body(i).name.startswith(a + '_') and
                              ('finger' in m.body(i).name or 'hand' in m.body(i).name)) for a in ARM}
        self.key = self.bid['hex_key']

    # ---- stepping / state
    def step(self, row):
        self.s.step(np.asarray(row, dtype=np.float64))
        self.ex.append(self.d.ctrl.copy())

    def move(self, target):
        """Reach `target` (16) from the current ctrl within the joint-speed limits: a straight line in joint space over
        the ceil(max jump / limit) ticks it needs (one tick when within the limits); the grippers jump at once."""
        cur = self.d.ctrl.copy(); target = np.asarray(target, dtype=np.float64)
        jump = np.abs(target - cur)[ARMCOLS] / self.lim[ARMCOLS]
        n = max(1, int(math.ceil(float(jump.max()) - 1e-9)))
        for k in range(1, n + 1):
            r = target.copy()
            if k < n:
                r[ARMCOLS] = cur[ARMCOLS] + (target[ARMCOLS] - cur[ARMCOLS]) * (k / n)
            self.step(r)
        return n

    def hold(self, n):
        for _ in range(n):
            self.step(self.d.ctrl.copy())

    def get(self):
        st = self.s.get_state(); st['ex'] = list(self.ex); return st

    def set(self, st):
        self.s.set_state(st); self.ex = list(st.get('ex', st['rows'])); mujoco.mj_forward(self.m, self.d)

    def save(self, st, path):
        with open(path + '.tmp', 'wb') as f:
            pickle.dump(st, f)
        os.replace(path + '.tmp', path)

    def load(self, path):
        with open(path, 'rb') as f:
            st = pickle.load(f)
        self.set(st); return st

    @property
    def ctrl(self):
        return self.d.ctrl.copy()

    def q(self, arm):
        return self.d.qpos[self.qadr[arm]].copy()

    def qv(self, arm):
        return self.d.qvel[self.dof[arm]].copy()

    # ---- measurements
    def pose(self, b):
        i = self.bid[b] if isinstance(b, str) else b
        return self.d.xpos[i].copy(), self.d.xmat[i].reshape(3, 3).copy()

    def welded(self, part):
        return bool(self.d.eq_active[self.M.eq_ids[self.opi[part]]])

    def op_view(self, part):
        """(gap m, lateral m, tilt deg, weld on) of the part's operation (the mechanism's own geometry)."""
        gap, lat, ang = self.M.geometry_all()[:3]; i = self.opi[part]
        return float(gap[i]), float(lat[i]), float(ang[i]), self.welded(part)

    def touching(self, arm):
        m, d = self.m, self.d; n = d.ncon; out = set()
        gb = m.geom_bodyid; F = self.finger[arm]
        for i in range(n):
            x, y = int(gb[d.contact.geom1[i]]), int(gb[d.contact.geom2[i]])
            if x in F and y not in F: out.add(m.body(y).name)
            elif y in F and x not in F: out.add(m.body(x).name)
        return sorted(out)

    def fbody(self, a, b):
        """Sum of the normal contact forces between bodies a and b (N)."""
        m, d = self.m, self.d; n = d.ncon
        if not n: return 0.
        gb = m.geom_bodyid; c = d.contact
        ia = self.bid[a] if isinstance(a, str) else a; ib = self.bid[b] if isinstance(b, str) else b
        x, y = gb[c.geom1[:n]], gb[c.geom2[:n]]
        sel = ((x == ia) & (y == ib)) | ((x == ib) & (y == ia))
        adr = c.efc_address[:n][sel]
        return float(np.abs(d.efc_force[adr[adr >= 0]]).sum())

    # ---- the hex key against a screw
    def key_tip(self, s=None):
        """World position of a key tip and its arm's direction (bend -> tip): the short arm's, or with a screw s given,
        the tip nearer to that screw's head (the demo drives some screws with the long arm in the socket)."""
        p, Rk = self.pose(self.key)
        short = (p + Rk @ KEY_SHORT_TIP, Rk[:, 0].copy())
        if s is None:
            return short
        long_ = (p.copy(), -Rk[:, 2].copy())
        ps = self.pose(s)[0]
        return min((short, long_), key=lambda t: np.linalg.norm(t[0] - ps))

    def key_vs(self, s):
        """The key's short tip against screw s: lateral offset (world, perpendicular to the screw axis), height of the tip
        above the screw frame origin along its axis (m), the short arm's angle to the screw axis (deg; 0 = pointing down
        the axis), the hex misalignment (rad, in [-pi/6, pi/6)) of the key's flats to the socket's flats."""
        tip, kx = self.key_tip(s); ps, Rs = self.pose(s); z = Rs[:, 2]
        v = tip - ps; h = float(v @ z); lat = v - h * z
        ang = math.degrees(math.acos(max(-1., min(1., float(-kx @ z)))))
        nk = self.pose(self.key)[1][:, 1]; nk = nk - (nk @ z) * z; xs = Rs[:, 0]
        a = math.atan2(float(np.cross(xs, nk) @ z), float(xs @ nk)) - SOCKET_FLAT0
        return lat, h, ang, (a + math.pi / 6) % (math.pi / 3) - math.pi / 6

    def screw_turn(self, s):
        """The screw's yaw about its own axis (rad), for counting turns."""
        Rs = self.pose(s)[1]; return math.atan2(Rs[1, 0], Rs[0, 0])

    # ---- Jacobian IK
    def jac(self, arm):
        mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.site[arm])
        return np.vstack([self.jp[:, self.dof[arm]], self.jr[:, self.dof[arm]]])

    def ik(self, arm, v, w=np.zeros(3), lam=1e-3):
        J = self.jac(arm)
        x = np.r_[v, w]
        return J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), x)


class Follower:
    """Demo rows with per-arm joint offsets (7 each) added; an offset can be ramped out between two demo rows (joints
    1-6 fully, j7 to its nearest whole turn: j7 has no limit, whole turns change nothing)."""

    def __init__(self, rig, D):
        self.rig, self.D = rig, D
        self.off = {a: np.zeros(7) for a in ARM}; self.ramp = {a: None for a in ARM}

    def set_from(self, i, arms=('left', 'right')):
        """The offsets that make demo row i the current ctrl."""
        for a in arms:
            self.off[a] = self.rig.ctrl[ARM[a]] - self.D[i][ARM[a]]; self.ramp[a] = None

    def ramp_out(self, r0, r1, arms=('left', 'right')):
        for a in arms:
            tgt = np.zeros(7); tgt[6] = 2 * math.pi * round(self.off[a][6] / (2 * math.pi))
            self.ramp[a] = (r0, r1, self.off[a].copy(), tgt)

    def offset(self, a, i):
        rp = self.ramp[a]
        if rp is None: return self.off[a]
        r0, r1, x, y = rp
        if i <= r0: return x
        if i >= r1:
            self.off[a] = y.copy(); self.ramp[a] = None; return self.off[a]
        u = (i - r0) / (r1 - r0); u = u * u * (3 - 2 * u)
        return x + (y - x) * u

    def row(self, i):
        r = self.D[i].copy()
        for a in ARM:
            r[ARM[a]] += self.offset(a, i)
        return r

    def run(self, a, b, hook=None, stride=1):
        i = a
        while i < b:
            r = self.row(i)
            if hook is not None:
                r = hook(i, r)
            self.rig.move(r)
            i += stride if callable(stride) is False else stride(i)

    def state(self):
        return dict(off={a: v.tolist() for a, v in self.off.items()}, ramp={a: (None if v is None else (v[0], v[1], v[2].tolist(), v[3].tolist())) for a, v in self.ramp.items()})

    def set_state(self, st):
        self.off = {a: np.array(v) for a, v in st['off'].items()}
        self.ramp = {a: (None if v is None else (v[0], v[1], np.array(v[2]), np.array(v[3]))) for a, v in st['ramp'].items()}


# ---------------------------------------------------------------- the hex-key ratchet (closed loop)
CTRL_DT = R.SUBSTEPS * 0.000625


def tip_cmd(rig, arm, v_tip, w=np.zeros(3)):
    """Joint increments (7) that move the key's short tip by v_tip (world) and rotate the hand by w (world)."""
    tip = rig.key_tip()[0]; tcp = rig.d.site_xpos[rig.site[arm]]
    return rig.ik(arm, v_tip - np.cross(w, tip - tcp), w)


def nudge(rig, arm, dq, j7_extra=0., grip=None):
    """Apply joint increments dq (7) to the arm's targets (plus an extra j7 increment) within the joint-speed limits."""
    t = rig.ctrl; t[ARM[arm]] += dq; t[ARM[arm][6]] += j7_extra
    if grip is not None: t[GRIP[arm]] = grip
    return rig.move(t)


def axial_force(rig, s):
    """The key's push on screw s along the screw axis (N, positive = down the axis)."""
    m, d = rig.m, rig.d; n = d.ncon; tot = np.zeros(3); f6 = np.zeros(6)
    ia, ib = rig.key, rig.bid[s]; gb = m.geom_bodyid
    for i in range(n):
        c = d.contact[i]; x, y = int(gb[c.geom1]), int(gb[c.geom2])
        if (x, y) == (ia, ib): sg = 1.
        elif (x, y) == (ib, ia): sg = -1.
        else: continue
        mujoco.mj_contactForce(m, d, i, f6)
        tot += sg * (c.frame.reshape(3, 3).T @ f6[:3])
    return float(-tot @ rig.pose(s)[1][:, 2])


def cmd_fk(rig, arm):
    """The TCP pose of the arm's joint TARGETS (forward kinematics of data.ctrl on a scratch copy) and its Jacobian."""
    if not hasattr(rig, '_d2'):
        rig._d2 = mujoco.MjData(rig.m)
    d2 = rig._d2; d2.qpos[:] = rig.d.qpos; d2.qpos[rig.qadr[arm]] = rig.d.ctrl[ARM[arm]]
    mujoco.mj_kinematics(rig.m, d2); mujoco.mj_comPos(rig.m, d2)
    jp, jr = np.zeros((3, rig.m.nv)), np.zeros((3, rig.m.nv))
    mujoco.mj_jacSite(rig.m, d2, jp, jr, rig.site[arm])
    J = np.vstack([jp[:, rig.dof[arm]], jr[:, rig.dof[arm]]])
    return d2.site_xpos[rig.site[arm]].copy(), d2.site_xmat[rig.site[arm]].reshape(3, 3).copy(), J


def rotvec(Rm):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, Rm.reshape(-1)); 
    if q[0] < 0: q = -q
    v = np.zeros(3); mujoco.mju_quat2Vel(v, q, 1.0); return v


def servo_pose(rig, arm, p_des, R_des, cap_p=1.5e-3, cap_r=0.02, j7_extra=0., lam=1e-3, grip=None, q_ref=None, k_null=0.05):
    """One tick: move the arm's joint targets one damped Newton step toward the TCP pose (p_des, R_des) in COMMAND space
    (the targets' own forward kinematics: no integration of the servo lag), each step capped; with q_ref, the arm's
    redundancy (the null space of the TCP task) is pulled toward that posture (joints 1-6), so the elbow and forearm stay
    where the demo held them; then within the limits."""
    p, Rm, J = cmd_fk(rig, arm)
    e = p_des - p; n = np.linalg.norm(e)
    if n > cap_p: e *= cap_p / n
    w = rotvec(R_des @ Rm.T); n = np.linalg.norm(w)
    if n > cap_r: w *= cap_r / n
    Jp = J.T @ np.linalg.inv(J @ J.T + lam ** 2 * np.eye(6))
    dq = Jp @ np.r_[e, w]
    if q_ref is not None:
        z = np.zeros(7); z[:6] = k_null * (q_ref[:6] - rig.d.ctrl[ARM[arm]][:6])
        dq = dq + (np.eye(7) - Jp @ J) @ z
    return nudge(rig, arm, dq, j7_extra=j7_extra, grip=grip)


def Rz_about(axis, a):
    """Rotation matrix by angle a about the unit vector axis."""
    q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), a); Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, q)
    return Rm.reshape(3, 3)


class KeyServo:
    """The hand holding the hex key, servoed so the key's short tip sits on screw s's TRUE axis at a height h above the
    screw origin, with the hand turned about that axis by a stroke angle. The tip's place in the hand is measured every
    tick (true state); the hand pose is commanded in command space (servo_pose); a slow integral on the true tip error
    takes out the servo's static error."""

    def __init__(self, rig, arm, s, ki=0.02, ki_cap=1e-3, track_key=False):
        self.rig, self.arm, self.s, self.ki, self.ki_cap, self.track_key = rig, arm, s, ki, ki_cap, track_key
        self.Rk0 = self.key_frame_on_axis(); self.q_ref = rig.ctrl[ARM[arm]].copy()
        p, Rm, _ = cmd_fk(rig, arm); self.R0 = Rm.copy(); self.theta = 0.; self.bias = np.zeros(3); self.kz = 0.; self.zoff = 0.

    def key_frame_on_axis(self):
        """The key's current orientation with its short arm turned exactly onto the screw axis (pointing down it)."""
        rig = self.rig; Rk = rig.pose(rig.key)[1]; z = rig.pose(self.s)[1][:, 2]; x = Rk[:, 0]
        a = np.cross(x, -z); n = np.linalg.norm(a)
        if n < 1e-12: return Rk
        return Rz_about(a / n, math.asin(min(1., n))) @ Rk

    def tip_in_tcp(self):
        rig = self.rig; tip = rig.key_tip()[0]; p = rig.d.site_xpos[rig.site[self.arm]]; Rm = rig.d.site_xmat[rig.site[self.arm]].reshape(3, 3)
        return Rm.T @ (tip - p)

    def step(self, h, dtheta=0., v_lat=np.zeros(3), push=None):
        """One tick: the stroke angle += dtheta (rad; + = tightening = clockwise seen from above), tip target at height h
        (with push=(f_lo, f_hi): plus an offset along the axis, stepped 1 um per tick to hold the key's axial push on the
        screw between f_lo and f_hi N)."""
        rig = self.rig; ps, Rs = rig.pose(self.s); z = Rs[:, 2]
        lat, hh, ang, hx = rig.key_vs(self.s)
        if push is not None:
            f = axial_force(rig, self.s)
            self.zoff += -1e-6 if f < push[0] else (1e-6 if f > push[1] else 0.)
            self.zoff = min(3e-4, max(-1e-3, self.zoff)); h = h + self.zoff
        else:
            self.zoff = 0.
        err = -lat + (self.kz * (h - hh) * z if self.kz else 0.)
        self.bias = self.bias + self.ki * err; n = np.linalg.norm(self.bias)
        if n > self.ki_cap: self.bias *= self.ki_cap / n
        self.theta += dtheta
        if self.track_key:     # the KEY's orientation is servoed (its in-hand tilt taken out), not the hand's
            Rk = rig.pose(rig.key)[1]; Rt = rig.d.site_xmat[rig.site[self.arm]].reshape(3, 3)
            R_des = Rz_about(z, -self.theta) @ self.Rk0 @ (Rk.T @ Rt)
        else:
            R_des = Rz_about(z, -self.theta) @ self.R0
        tip_des = ps + h * z + self.bias + v_lat
        p_des = tip_des - R_des @ self.tip_in_tcp()
        servo_pose(rig, self.arm, p_des, R_des, q_ref=self.q_ref)
        return lat, hh, ang, hx


def key_sweep(rig, s, step_deg=5., lift=2.5e-3, margin=0.004):
    """Which turns of the key about screw s's axis are free: the key (in its current pose, lifted by `lift` along the axis)
    rotated by phi (deg; + = tightening) in a scratch copy of the state collides (within `margin` m) with nothing but
    the robot and screw s. Returns (phis, free flags) for phi in [0, 360)."""
    m = rig.m
    d2 = mujoco.MjData(m); d2.qpos[:] = rig.d.qpos; d2.qvel[:] = 0
    j = [j for j in range(m.njnt) if m.jnt_bodyid[j] == rig.key and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
    a = m.jnt_qposadr[j]
    ps, Rs = rig.pose(s); z = Rs[:, 2]; p0 = d2.qpos[a:a + 3].copy() + lift * z; q0 = d2.qpos[a + 3:a + 7].copy()
    robot = set(R.robot_bodies(m).tolist()); kg = set(np.flatnonzero(m.geom_bodyid == rig.key).tolist())
    old = m.geom_margin.copy(); m.geom_margin[list(kg)] = margin
    phis = np.arange(0., 360., step_deg); free = []
    try:
        for phi in phis:
            Rr = Rz_about(z, -math.radians(phi)); qr = np.zeros(4); mujoco.mju_mat2Quat(qr, Rr.reshape(-1))
            d2.qpos[a:a + 3] = ps + Rr @ (p0 - ps); qn = np.zeros(4); mujoco.mju_mulQuat(qn, qr, q0); d2.qpos[a + 3:a + 7] = qn
            mujoco.mj_forward(m, d2)
            hit = False
            for i in range(d2.ncon):
                g1, g2 = d2.contact.geom1[i], d2.contact.geom2[i]
                if (g1 in kg) == (g2 in kg): continue
                o = int(m.geom_bodyid[g2 if g1 in kg else g1])
                if o in robot or o == rig.bid[s]: continue
                hit = True; break
            free.append(not hit)
    finally:
        m.geom_margin[:] = old
    return phis, np.array(free)


def _start_in(theta, lo, hi, delta):
    """theta shifted by whole flats (60 deg) to the start nearest to it from which a stroke of delta stays in [lo, hi]
    (the handle's free sector, in the tightening sense); None if no such start exists."""
    cands = [theta + k * math.pi / 3 for k in range(-6, 7)]
    ok = [c for c in cands if lo - 1e-9 <= c and c + delta <= hi + 1e-9]
    return min(ok, key=lambda c: abs(c - theta)) if ok else None


def free_sector(rig, s, margin_deg=15., lift=4e-3):
    """The key handle's free sector about screw s from its current orientation (key_sweep, 5 deg steps): (lo, hi) in
    radians in the tightening sense, the contiguous free range around 0 shrunk by margin_deg on each side."""
    phis, free = key_sweep(rig, s, step_deg=5., lift=lift, margin=0.003)
    n = len(phis); hi = 0
    while hi + 1 < n and free[hi + 1]: hi += 1
    lo = 0
    while lo + 1 < n and free[(n - lo - 1) % n]: lo += 1
    if hi == n - 1:
        return -math.inf, math.inf
    return -math.radians(5. * lo - margin_deg), math.radians(5. * hi - margin_deg)


def _foreign_contact(rig, s, arm):
    """The key or the key hand touches something other than screw s, the key, the hand's own fingers."""
    m, d = rig.m, rig.d
    if not hasattr(rig, '_armbodies'):
        rig._armbodies = {a: set(i for i in range(m.nbody) if m.body(i).name.startswith(a + '_')) for a in ARM}
    hand = rig._armbodies[arm] | {rig.key}; ok = hand | {rig.bid[s]}
    for i in range(d.ncon):
        b1, b2 = int(m.geom_bodyid[d.contact.geom1[i]]), int(m.geom_bodyid[d.contact.geom2[i]])
        if (b1 in hand and b2 not in ok) or (b2 in hand and b1 not in ok):
            return m.body(b2 if b1 in hand else b1).name
    return None


def _near(theta, home, band=math.radians(45.)):
    """theta, or shifted by whole flats (60 deg) when it has drifted more than `band` from home: the same flat alignment,
    every stroke kept in the sector the handle is free in (the hysteresis keeps a flat error near +-30 deg from
    flipping it back and forth)."""
    while theta - home > band: theta -= math.pi / 3
    while home - theta > band: theta += math.pi / 3
    return theta


def ratchet(rig, arm, s, delta=2 * math.pi / 3, w_on=1.0, w_back=2.4, push=(0.8, 2.0), h_eng=-2.0e-4, h_lift=2.4e-3,
            max_strokes=80, log=None, verbose=False, v_low=3e-5, max_lead=math.radians(4.), stroke_track=False, stroke_ki=0.02, start_back=False, h_out=1.2e-2, sector=None, min_stroke=math.radians(30.)):
    """Drive screw s with the hex key in `arm` (the demo's grip: the short arm along the hand's axis in the socket, the
    long arm a handle): strokes of `delta` about the TRUE screw axis in the tightening sense (KeyServo: the key's tip on
    the axis, the key's orientation turned about it), the push on the screw held in `push` N; then the key lifted out of
    the socket, turned back by `delta` (a multiple of 60 deg, so the flats line up again; the remaining flat error is
    taken out from the true state), lowered into the socket, and so on until the screw's retention weld latches.
    The handle's sweep is limited by the robot's own links (about 130 deg free from the demo's start), hence strokes.
    Returns a dict (strokes, ticks, weld tick)."""
    k0 = len(rig.ex); out = dict(screw=s, strokes=0, misses=0)
    ks = KeyServo(rig, arm, s, track_key=True); th_home = ks.theta     # every stroke starts within 30 deg of here
    if sector is not None:
        lo, hi = sector; out['sector_deg'] = [round(math.degrees(lo)), round(math.degrees(hi))]

    def _pick(th):
        """With a free sector: the lowest start (by whole flats from th) that leaves a stroke of at least min_stroke."""
        if sector is None: return _near(th, th_home)
        cands = [th + k * math.pi / 3 for k in range(-8, 9)]
        ok = [c for c in cands if lo - 1e-9 <= c <= hi - min_stroke]
        return min(ok) if ok else _near(th, th_home)
    if sector is not None and not start_back:
        ks.theta = _pick(ks.theta)
    if start_back:          # the key sits at the END of a stroke (the demo's): out of the socket, back to the stroke's start
        ks.kz = 1.
        for t in range(300):
            lat, h, ang, hx = ks.step(h_lift)
            if h > 1.5e-3: break
        while ks.theta > -delta + 1e-9:
            ks.step(h_lift, dtheta=-min(w_back * CTRL_DT, ks.theta + delta))
        th_home = -delta
    for st in range(max_strokes):
        # 1. hover over the socket until the tip is on the axis and the flats line up (true state), then lower into it
        #    (the z integrator on: the servo's sag taken out)
        ks.kz = 1.; ok = False
        if rig.key_vs(s)[1] > 1e-3:
            settled = 0
            for t in range(400):
                lat, h, ang, hx = ks.step(h_lift)
                settled = settled + 1 if np.linalg.norm(lat) < 3e-4 else 0
                if settled >= 15 and abs(hx) > math.radians(0.5):      # correct the flats only once the hand has settled
                    ks.theta = _pick(ks.theta + hx); settled = 0
                if verbose and t % 20 == 0:
                    print('  hover', st, t, 'lat %.3f h %.3f hx %.2f theta %.3f' % (np.linalg.norm(lat) * 1e3, h * 1e3, math.degrees(hx), ks.theta), flush=True)
                if np.linalg.norm(lat) < 1e-4 and abs(hx) < math.radians(1.) and t > 20: break
        h0 = max(rig.key_vs(s)[1], h_eng)
        for t in range(700):
            lat, h, ang, hx = ks.step(max(h_eng - 5e-5, h0 - v_low * t))
            if verbose and t % 25 == 0:
                print('  lower', st, t, 'lat %.3f h %.3f ang %.2f hx %.2f f %.2f theta %.3f' % (np.linalg.norm(lat) * 1e3, h * 1e3, ang, math.degrees(hx), axial_force(rig, s), ks.theta), flush=True)
            if h < h_eng + 3e-5 and abs(hx) < math.radians(2.):
                ok = True; break
            if t > 300 and h > 5e-4:
                break
        if not ok:
            out['misses'] += 1
            for t in range(60): ks.step(h_lift)
            ks.theta = _pick(ks.theta + rig.key_vs(s)[3])    # line the flats up, then try again
            for t in range(60): ks.step(h_lift)
            if out['misses'] > 30: break
            continue
        for t in range(20): ks.step(h_eng - 5e-5)
        # 2. the stroke (the push held in its band)
        ks.kz = 0.; ks.zoff = 0.; base = h_eng - 5e-5; a0 = ks.theta; welded = False
        if sector is not None: delta = max(min_stroke, hi - a0)       # as far as the free sector allows
        if not stroke_track:             # the stroke as a rigid turn of the HAND about the screw axis (as the demo turns
            ks.track_key = False         # it): no in-hand compensation, no lateral integrator while engaged
            ks.R0 = Rz_about(rig.pose(s)[1][:, 2], ks.theta) @ cmd_fk(rig, arm)[1]; ki0 = ks.ki; ks.ki = stroke_ki
        y0 = handle_yaw(rig, s); turned = 0.; yp = y0; stall = 0
        while ks.theta < a0 + delta - 1e-9 and stall < 2000:
            if rig.welded(s): welded = True; break
            g = rig.op_view(s)
            w = w_on if g[0] > 8e-4 else min(w_on, 0.5)
            y = handle_yaw(rig, s); turned += -((y - yp + math.pi) % (2 * math.pi) - math.pi); yp = y
            # torque-limited: the commanded turn leads the key's actual turn by at most max_lead (the key stalls on a
            # hard spot instead of being forced out of the socket)
            dth = min(w * CTRL_DT, a0 + delta - ks.theta) if (ks.theta - a0) - turned < max_lead else 0.
            stall = stall + 1 if dth == 0. else 0
            ks.step(base, dtheta=dth, push=push)
            fc = _foreign_contact(rig, s, arm)
            if fc is not None:                    # the handle or hand met something: end this stroke here
                out['guard'] = out.get('guard', 0) + 1; out['guard_last'] = fc; break
        if not stroke_track:
            ks.track_key = True; ks.ki = ki0
        out['strokes'] += 1
        g = rig.op_view(s)
        rec = dict(stroke=st, tick=len(rig.ex) - k0, gap_mm=round(g[0] * 1e3, 3), lat_mm=round(g[1] * 1e3, 3), tilt=round(g[2], 2),
                   hx=round(math.degrees(rig.key_vs(s)[3]), 2), weld=g[3], misses=out['misses'],
                   host_mm=[round(float(x) * 1e3, 2) for x in rig.pose(rig.ops[rig.opi[s]]['host'])[0]])
        if log is not None: log.append(rec)
        if verbose: print(json.dumps(rec), flush=True)
        if welded or rig.welded(s):
            for t in range(40):
                if rig.welded(s): break
                ks.step(base, push=push)
            out['weld_tick'] = len(rig.ex) - k0
            # the stroke to its end (the latched screw no longer collides: the key turns free in its socket), so the
            # handle ends where the demo's own strokes end
            while ks.theta < a0 + delta - 1e-9:
                ks.step(base, dtheta=min(w_on * CTRL_DT, a0 + delta - ks.theta))
            break
        # 3. out of the socket and back by delta
        ks.kz = 1.; ph = {}
        for t in range(300):
            lat, h, ang, hx = ks.step(h_lift)
            if h > 1.5e-3: break                            # clear of the 1.2 mm socket
        ph['lift'] = t
        a1 = ks.theta - delta if sector is None else _pick(ks.theta)          # the key turned with the screw: still lined up
        while abs(ks.theta - a1) > 1e-9:
            ks.step(h_lift, dtheta=max(-w_back * CTRL_DT, min(w_back * CTRL_DT, a1 - ks.theta)))
    for t in range(400):                                      # 4. straight up, out of the socket
        lat, h, ang, hx = ks.step(min(h_out, h_lift + 2e-5 * t))
        if h > h_out - 5e-4: break
    out.update(ok=rig.welded(s), ticks=len(rig.ex) - k0, theta=ks.theta)
    return out


def handle_yaw(rig, s):
    """The key handle's direction about screw s's axis (rad, world): the arm that is not in the socket, from the bend."""
    p0, Rk = rig.pose(rig.key); bend = p0 + Rk @ np.array([0., 0., KEY_SHORT_TIP[2]]); ps, Rs = rig.pose(s); z = Rs[:, 2]
    short = p0 + Rk @ KEY_SHORT_TIP
    v = (p0 - bend) if np.linalg.norm(short - ps) <= np.linalg.norm(p0 - ps) else (short - bend)   # the arm not in the socket
    v = v - (v @ z) * z
    return math.atan2(v[1], v[0])


def place_hook(rig, fol, s, target, gain=0.02, cap=5e-5, near=0.025, gap_min=6e-3, tilt_max=20.):
    """While a hand carries screw s close to its hole (within `near`, the screw not yet in the thread: gap > gap_min):
    steer the screw onto the hole's TRUE axis (lateral error + tilt, damped-least-squares IK on that arm), accumulated
    in the follower's offset for that arm. `target` = (position, axis) of the hole in the world."""
    tp, ax = target

    def hook(i, r):
        g = rig.op_view(s)
        if g[3] or g[0] < gap_min or g[2] > tilt_max: return r      # only a screw carried upright to its hole
        p, Rs = rig.pose(s); dl = p - tp
        if np.linalg.norm(dl) > near: return r
        for arm in ('left', 'right'):
            if s in rig.touching(arm): break
        else:
            return r
        e = dl - (dl @ ax) * ax
        v = -gain * e; n = np.linalg.norm(v)
        if n > cap: v *= cap / n
        w = gain * np.cross(Rs[:, 2], ax); n = np.linalg.norm(w)
        if n > 2e-4: w *= 2e-4 / n
        tcp = rig.d.site_xpos[rig.site[arm]]
        dq = rig.ik(arm, v - np.cross(w, p - tcp), w)
        fol.off[arm] = fol.off[arm] + dq
        hook.steered = arm
        r = r.copy(); r[ARM[arm]] += dq
        return r
    hook.steered = None
    return hook


def key_hook(rig, fol, s, arm, gain=0.05, cap=2e-5, wcap=4e-3, h_free=1.6e-3, h_hi=8e-3, near=4e-3, dz=2e-6,
             f_lo=1.0, f_hi=3.0, h_seat=-1e-4, log=None, use_lat=True, use_hx=True, use_z=True):
    """The demo's hex-key strokes on screw s, corrected on the true state (accumulated in the follower's offset for the
    key arm, so every later stroke starts from the corrected pose):
      lifted (the key's nearer tip h_free..h_hi above the screw head, within `near` of its axis): the tip steered onto the
        screw's TRUE axis and the key turned about the axis until its flats line up with the socket's;
      in a stroke (below h_free): the tip pushed down the axis in steps of dz while it sits above h_seat (the socket's
        bottom is about -0.2 mm) and the push on the screw is below f_lo, lifted back while the push exceeds f_hi; no
        turning (the demo's own stroke turns the key)."""
    st = dict(z=0., n=0)

    def hook(i, r):
        lat, h, ang, hx = rig.key_vs(s)
        if h > h_hi or np.linalg.norm(lat) > near:
            return r
        z = rig.pose(s)[1][:, 2]; tip = rig.key_tip(s)[0]; tcp = rig.d.site_xpos[rig.site[arm]]
        y = handle_yaw(rig, s); dy = abs((y - st.get('y', y) + math.pi) % (2 * math.pi) - math.pi); st['y'] = y
        if h > h_free:
            if not use_lat: return r
            v = -gain * lat; n = np.linalg.norm(v)
            if n > cap: v *= cap / n
            w = np.zeros(3)
            if use_hx and h < h_free + 6e-4 and dy < 5e-4:     # lining the flats up just above the socket, not in a swing
                w = -gain * hx * z; n = np.linalg.norm(w)
                if n > wcap: w *= wcap / n
        else:
            if not use_z: return r
            f = axial_force(rig, s)
            vz = -dz if (h > h_seat and f < f_lo) else (dz if f > f_hi else 0.)
            if vz == 0.: return r
            v = vz * z; w = np.zeros(3); st['z'] += vz; st['n'] += 1
        dq = rig.ik(arm, v - np.cross(w, tip - tcp), w)
        fol.off[arm] = fol.off[arm] + dq
        r = r.copy(); r[ARM[arm]] += dq
        return r
    hook.st = st
    return hook


def key_hook3(rig, fol, D, s, arm, gain=0.05, cap=2e-5, wcap=3e-3, h_hi=8e-3, near=4e-3, lift=1.5e-3, lift_th=8e-4,
              push_f=2.0, push_max=1.5e-3, dz=4e-6, ramp=2e-5, win=3000):
    """The demo's own hex-key strokes on screw s, made to seat the key in the socket (true state; every correction is
    accumulated in the follower's offset for the key arm, so later strokes keep it):
      always (the key's nearer tip below h_hi above the screw head, within `near` of its axis): the tip steered onto the
        screw's TRUE axis;
      the demo's lifted phase (its commanded TCP more than lift_th above the lowest point of its last `win` rows): the
        hand raised by `lift` more (a key seated deeper than the demo's must still clear the 1.2 mm socket before the
        demo turns it back), and, once the key is not turning, turned about the screw axis until its flats line up;
      the demo's stroke phase: the hand lowered (dz per tick, at most push_max) until the tip is in the socket (below
        -0.1 mm) or pushes push_f N."""
    zc = {}; st = dict(z=0., hist=[], y=None)
    d2 = mujoco.MjData(rig.m)

    def cmd_z(r):
        if r not in zc:
            d2.qpos[:] = rig.d.qpos; d2.qpos[rig.qadr[arm]] = D[r][ARM[arm]]; mujoco.mj_kinematics(rig.m, d2)
            zc[r] = float(d2.site_xpos[rig.site[arm]][2])
        return zc[r]

    def hook(i, r):
        lat, h, ang, hx = rig.key_vs(s)
        if h > h_hi or np.linalg.norm(lat) > near:
            st['hist'] = []; return r
        z = rig.pose(s)[1][:, 2]; tip = rig.key_tip(s)[0]; tcp = rig.d.site_xpos[rig.site[arm]]
        zi = cmd_z(i); st['hist'].append((i, zi)); st['hist'] = [x for x in st['hist'] if x[0] > i - win]
        lifted = zi > min(x[1] for x in st['hist']) + lift_th
        y = handle_yaw(rig, s); dy = abs((y - (st['y'] if st['y'] is not None else y) + math.pi) % (2 * math.pi) - math.pi); st['y'] = y
        v = -gain * lat; n = np.linalg.norm(v)
        if n > cap: v *= cap / n
        w = np.zeros(3)
        if lifted:
            tz = lift
            if dy < 5e-4 and h > 1.4e-3:
                w = -gain * hx * z; n = np.linalg.norm(w)
                if n > wcap: w *= wcap / n
        else:
            f = axial_force(rig, s)
            tz = st['z'] - dz if (h > -1e-4 and f < push_f) else st['z']
            tz = max(tz, -push_max)
        step = max(-ramp, min(ramp, tz - st['z'])); st['z'] += step
        v = v + step * z
        dq = rig.ik(arm, v - np.cross(w, tip - tcp), w)
        fol.off[arm] = fol.off[arm] + dq
        r = r.copy(); r[ARM[arm]] += dq
        return r
    hook.st = st
    return hook


def arm_sector(rig, s, arm, lift=3e-3, step_deg=5., span_deg=180., margin=0.004, margin_deg=10.):
    """The free sector of a hand-and-key turn about screw s's axis, the ARM included: for each turn phi (tightening sense,
    step_deg steps out to +-span_deg), the arm's joints are solved (damped least squares, the null space held at the
    current posture) for the hand turned by phi about the screw axis with the key lifted by `lift`, on a scratch copy of
    the state; the turn is free when the robot (links, hand, key) is not within `margin` of anything but screw s.
    Returns (lo, hi) in radians: the contiguous free range around 0, shrunk by margin_deg each side."""
    m = rig.m; d2 = mujoco.MjData(m); d2.qpos[:] = rig.d.qpos; d2.qvel[:] = 0
    qa = rig.qadr[arm]; q0 = rig.d.qpos[qa].copy()
    kj = [j for j in range(m.njnt) if m.jnt_bodyid[j] == rig.key and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]; ka = m.jnt_qposadr[kj]
    mujoco.mj_kinematics(m, d2)
    ps, Rs = rig.pose(s); z = Rs[:, 2]
    tcp0 = d2.site_xpos[rig.site[arm]].copy(); Rt0 = d2.site_xmat[rig.site[arm]].reshape(3, 3).copy()
    kp0 = rig.d.qpos[ka:ka + 3].copy(); kq0 = rig.d.qpos[ka + 3:ka + 7].copy()
    robot = set(R.robot_bodies(m).tolist()) | {rig.key}
    old = m.geom_margin.copy()
    rg = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in robot]; m.geom_margin[rg] = margin
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    def free_at(phi):
        Rr = Rz_about(z, -phi); pt = ps + Rr @ (tcp0 - ps) + lift * z; Rt = Rr @ Rt0
        q = q0.copy()
        for it in range(60):
            d2.qpos[qa] = q; mujoco.mj_kinematics(m, d2); mujoco.mj_comPos(m, d2)
            mujoco.mj_jacSite(m, d2, jp, jr, rig.site[arm]); J = np.vstack([jp[:, rig.dof[arm]], jr[:, rig.dof[arm]]])
            e = np.r_[pt - d2.site_xpos[rig.site[arm]], rotvec(Rt @ d2.site_xmat[rig.site[arm]].reshape(3, 3).T)]
            if np.linalg.norm(e) < 1e-6: break
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(6))
            nz = np.zeros(7); nz[:6] = 0.2 * (q0[:6] - q[:6])
            q = q + Jp @ e + (np.eye(7) - Jp @ J) @ nz
        lo_, hi_ = m.jnt_range[[m.joint(f'{arm}_joint{k}').id for k in range(1, 8)]].T
        lim = np.asarray(m.jnt_limited[[m.joint(f'{arm}_joint{k}').id for k in range(1, 8)]]).astype(bool)
        if (lim & ((q < lo_ + 0.02) | (q > hi_ - 0.02))).any():
            return False
        d2.qpos[qa] = q
        qr = np.zeros(4); mujoco.mju_mat2Quat(qr, Rr.reshape(-1)); qn = np.zeros(4); mujoco.mju_mulQuat(qn, qr, kq0)
        d2.qpos[ka:ka + 3] = ps + Rr @ (kp0 - ps) + lift * z; d2.qpos[ka + 3:ka + 7] = qn
        mujoco.mj_forward(m, d2)
        for i in range(d2.ncon):
            b1, b2 = int(m.geom_bodyid[d2.contact.geom1[i]]), int(m.geom_bodyid[d2.contact.geom2[i]])
            if (b1 in robot) != (b2 in robot):
                o = b2 if b1 in robot else b1
                if o != rig.bid[s]:
                    return False
        return True

    try:
        hi = 0.
        while hi + step_deg <= span_deg and free_at(math.radians(hi + step_deg)): hi += step_deg
        lo = 0.
        while lo + step_deg <= span_deg and free_at(-math.radians(lo + step_deg)): lo += step_deg
    finally:
        m.geom_margin[:] = old
    return -math.radians(max(0., lo - margin_deg)), math.radians(max(0., hi - margin_deg))


class DemoTrack:
    """The demo run on the author's geometry (prism rule off; a guide only): qpos every `every` rows (track_np.py)."""

    def __init__(self, path, rig):
        z = np.load(path); self.Q, self.rows = z['qpos'], z['rows']; self.rig = rig; m = rig.m
        self.d2 = mujoco.MjData(m)

    def at(self, row):
        k = int(np.clip(np.searchsorted(self.rows, row), 0, len(self.rows) - 1))
        self.d2.qpos[:] = self.Q[k]; mujoco.mj_kinematics(self.rig.m, self.d2)
        return self.d2

    def pose(self, row, body):
        d2 = self.at(row); b = self.rig.bid[body]
        return d2.xpos[b].copy(), d2.xmat[b].reshape(3, 3).copy()

    def tcp(self, row, arm):
        d2 = self.at(row); return d2.site_xpos[self.rig.site[arm]].copy(), d2.site_xmat[self.rig.site[arm]].reshape(3, 3).copy()


def _align(a, b):
    """The shortest rotation taking unit vector a onto unit vector b."""
    v = np.cross(a, b); n = np.linalg.norm(v); c = float(a @ b)
    if n < 1e-12: return np.eye(3)
    return Rz_about(v / n, math.atan2(n, c))


def grasp_hook(rig, fol, track, D, arm, part, r0, r1, gain=0.1, cap_p=1e-4, cap_r=1e-3, axis=None, max_turn_deg=15.):
    """Demo rows r0..r1 (a hand's approach to `part` and its grasp): the hand is steered so it meets the part where the
    part REALLY is: the demo's COMMANDED hand pose at this row moved with the part's displacement since the demo (its
    position plus the turn of its `axis`, from the demo track) is the target for our commanded hand (forward kinematics
    of the targets, offsets included); the correction is accumulated in the follower's offset for that arm (kept after
    the window: the caller ramps it out once the part is placed). Once the hand holds the part, the hook stops."""
    d3 = mujoco.MjData(rig.m)

    def hook(i, r):
        if not (r0 <= i < r1) or hook.held:
            return r
        if part in rig.touching(arm) and i > r0 + 20:
            hook.held = True; return r
        pd, Rd = track.pose(i, part); pn, Rn = rig.pose(part)
        Ra = _align(Rd[:, axis], Rn[:, axis]) if axis is not None else np.eye(3)
        if math.degrees(math.acos(min(1., (np.trace(Ra) - 1) / 2))) > max_turn_deg: Ra = np.eye(3)
        d3.qpos[:] = rig.d.qpos; d3.qpos[rig.qadr[arm]] = D[i][ARM[arm]]; mujoco.mj_kinematics(rig.m, d3)
        hp = d3.site_xpos[rig.site[arm]]; hR = d3.site_xmat[rig.site[arm]].reshape(3, 3)
        p_des = pn + Ra @ (hp - pd); R_des = Ra @ hR
        p, Rm, J = cmd_fk(rig, arm)
        e = gain * (p_des - p)
        if np.linalg.norm(e) > cap_p: e *= cap_p / np.linalg.norm(e)
        w = gain * rotvec(R_des @ Rm.T)
        if np.linalg.norm(w) > cap_r: w *= cap_r / np.linalg.norm(w)
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), np.r_[e, w])
        fol.off[arm] = fol.off[arm] + dq; hook.steered = True
        r = r.copy(); r[ARM[arm]] += dq
        return r
    hook.held = False; hook.steered = False
    return hook


def carry_hook(rig, fol, track, D, arm, part, r0, r1, gain=0.2, cap_p=2e-4, cap_r=2e-3, settle=150, every_tick=True):
    """Demo rows r0..r1 while `arm` holds `part`: once the grasp has settled (`settle` ticks held), the part's pose in
    the hand is compared with the demo track's (its pose in the demo's hand at the same row) and the demo's COMMANDED
    hand trajectory is corrected by that constant transform, so the part travels the demo's path; the correction is
    applied in command space (forward kinematics of the targets: no servo-lag windup), a fraction per tick, accumulated
    in the follower's offset for that arm. It ends when the hand lets go."""
    d3 = mujoco.MjData(rig.m); st = dict(n=0, X=None)

    def hook(i, r):
        if not (r0 <= i < r1) or part not in rig.touching(arm):
            st['n'] = 0; return r
        st['n'] += 1
        if st['n'] < settle: return r
        if st['X'] is None or every_tick:
            pt = rig.d.site_xpos[rig.site[arm]]; Rt = rig.d.site_xmat[rig.site[arm]].reshape(3, 3); pp, Rp = rig.pose(part)
            hp, hR = track.tcp(i, arm); dp, dR = track.pose(i, part)
            # in-hand poses (hand frame): ours and the demo's
            Ho = (Rt.T @ (pp - pt), Rt.T @ Rp); Hd = (hR.T @ (dp - hp), hR.T @ dR)
            # X: hand correction (in the hand frame) so that hand_cmd' * Ho = hand_cmd * Hd  ->  X = Hd * Ho^-1
            RX = Hd[1] @ Ho[1].T; pX = Hd[0] - RX @ Ho[0]
            st['X'] = (pX, RX); hook.X = (pX.tolist(), RX.tolist())
        pX, RX = st['X']
        d3.qpos[:] = rig.d.qpos; d3.qpos[rig.qadr[arm]] = D[i][ARM[arm]]; mujoco.mj_kinematics(rig.m, d3)
        hp = d3.site_xpos[rig.site[arm]]; hR = d3.site_xmat[rig.site[arm]].reshape(3, 3)
        p_des = hp + hR @ pX; R_des = hR @ RX
        p, Rm, J = cmd_fk(rig, arm)
        e = gain * (p_des - p)
        if np.linalg.norm(e) > cap_p: e *= cap_p / np.linalg.norm(e)
        w = gain * rotvec(R_des @ Rm.T)
        if np.linalg.norm(w) > cap_r: w *= cap_r / np.linalg.norm(w)
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), np.r_[e, w])
        fol.off[arm] = fol.off[arm] + dq; hook.steered = True
        r = r.copy(); r[ARM[arm]] += dq
        return r
    hook.steered = False; hook.X = None
    return hook
