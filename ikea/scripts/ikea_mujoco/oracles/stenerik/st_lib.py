"""stenerik closed-loop reference: the session wrapper (records the EXECUTED rows, data.ctrl after the clamp), true-state
measurements (screws vs inserts, the screwdriver tip vs a screw head, contact forces) and damped-least-squares Jacobian
IK on the TCP sites."""
from __future__ import annotations
import math, os, pickle
import numpy as np
import mujoco
import rcb_ikea as R

ARM = {'left': np.arange(0, 7), 'right': np.arange(8, 15)}
GRIP = {'left': 7, 'right': 15}
ARMCOLS = np.r_[ARM['left'], ARM['right']]
GRIPCOLS = np.array([GRIP['left'], GRIP['right']])
# the Franka Hand slew (rcb_ikea.GRIPPER_WIDTH_SPEED_M_S: 0.1 m/s of finger width = 0.797 command units per tick here):
# 1 = Rig.move holds its target until the hand commands have reached it (ST_GRIP_WAIT=0: the hands slew in place)
GRIP_WAIT = int(os.environ.get('ST_GRIP_WAIT', '1'))
DEMO = os.environ.get('ST_DEMO', '/workspace/qn_ikea_b2/stenerik/refs/demo.npz')
HEAD_TOP = 0.01675          # screw frame: the head's top face (the cross recess goes down to 0.01555)


def demo_rows():
    return R.load_trajectory(DEMO, 16)


class Rig:
    def __init__(self):
        if os.environ.get('NOCLAMP'):      # a study only (the phase guide on the author's scene): no joint-speed clamp
            R.JOINT_SPEED_LIMITS_RAD_S = {}
        self.s = R.Session('stenerik')
        self.m, self.d = self.s.model, self.s.data
        self.M = self.s.S.mech
        self.ex: list = []
        m = self.m
        self.lim = np.asarray(self.s.step_limit, dtype=np.float64)       # per tick: rad (arms), command units (hands)
        self.grip_wait_ticks = 0
        self.site = {a: m.site(a + '_tcp').id for a in ARM}
        self.dof = {a: np.array([int(m.jnt_dofadr[m.joint(f'{a}_joint{k}').id]) for k in range(1, 8)]) for a in ARM}
        self.qadr = {a: np.array([int(m.jnt_qposadr[m.joint(f'{a}_joint{k}').id]) for k in range(1, 8)]) for a in ARM}
        self.bid = {m.body(i).name: i for i in range(m.nbody)}
        self.screws = ['screw_%d' % i for i in range(1, 7)]
        self.jp = np.zeros((3, m.nv)); self.jr = np.zeros((3, m.nv))
        gb = m.geom_bodyid
        self.finger = {a: set(i for i in range(m.nbody) if m.body(i).name.startswith(a + '_') and
                              ('finger' in m.body(i).name or 'hand' in m.body(i).name)) for a in ARM}

    # ---- stepping / state
    def step(self, row):
        row = np.asarray(row, dtype=np.float64)
        self.s.step(row)
        self.ex.append(self.d.ctrl.copy())

    def move(self, target, grip_first=True):
        """Reach `target` (16) from the current ctrl within the speed limits: the arms on a straight line in joint space
        over the ceil(max jump / limit) ticks they need (one tick when within the limits); the hands slew toward the
        target's command at the Franka Hand limit (rcb_ikea.GRIPPER_WIDTH_SPEED_M_S, Session.applied) and the target is
        then HELD until both hand commands have reached it (GRIP_WAIT), so a grasp or a release completes before the
        caller goes on (the demo's next row, the next servo tick), as with the instant command the demo was made with."""
        cur = self.d.ctrl.copy(); target = np.asarray(target, dtype=np.float64)
        jump = np.abs(target - cur)[ARMCOLS] / self.lim[ARMCOLS]
        n = max(1, int(math.ceil(float(jump.max()) - 1e-9)))
        for k in range(1, n + 1):
            r = target.copy()
            if k < n:
                r[ARMCOLS] = cur[ARMCOLS] + (target[ARMCOLS] - cur[ARMCOLS]) * (k / n)
            self.step(r)
        if GRIP_WAIT:
            tg = np.clip(target[GRIPCOLS], self.s._clo[GRIPCOLS], self.s._chi[GRIPCOLS])
            while np.any(np.abs(tg - self.d.ctrl[GRIPCOLS]) > 1e-12):
                self.step(target); n += 1
                self.grip_wait_ticks += 1
        return n

    def get(self):
        st = self.s.get_state(); st['ex'] = list(self.ex); return st

    def set(self, st):
        self.s.set_state(st); self.ex = list(st.get('ex', st['rows'])); mujoco.mj_forward(self.m, self.d)

    def save(self, st, path):
        with open(path, 'wb') as f:
            pickle.dump(st, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.set(pickle.load(f))

    @property
    def ctrl(self):
        return self.d.ctrl.copy()

    def q(self, arm):
        return self.d.qpos[self.qadr[arm]].copy()

    def qv(self, arm):
        return self.d.qvel[self.dof[arm]].copy()

    # ---- measurements
    def pose(self, b):
        i = self.bid[b]; return self.d.xpos[i].copy(), self.d.xmat[i].reshape(3, 3).copy()

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

    def touching(self, arm):
        """Part bodies the arm's hand / fingers touch."""
        m, d = self.m, self.d; n = d.ncon; out = set()
        gb = m.geom_bodyid; F = self.finger[arm]
        for i in range(n):
            x, y = int(gb[d.contact.geom1[i]]), int(gb[d.contact.geom2[i]])
            if x in F and y not in F: out.add(m.body(y).name)
            elif y in F and x not in F: out.add(m.body(x).name)
        return sorted(out)

    def screw_state(self, s):
        """(nearest insert k, gap m along the crossbar axis, lateral m, tilt deg, weld on)"""
        M = self.M; pc, qc = M.pose('crossbar'); from ikea_runtime import rotate, axis_angle_deg
        ax = rotate(qc, [0, 0, 1]); p, q = M.pose(s); best = None
        for k in range(6):
            tp, _ = M.insert_target(k); dl = p - tp; gap = float(dl @ ax); lat = float(np.linalg.norm(dl - gap * ax))
            if best is None or np.linalg.norm(dl) < best[4]:
                best = (k, gap, lat, axis_angle_deg(rotate(q, [0, 0, 1]), ax), float(np.linalg.norm(dl)))
        return best[0], best[1], best[2], best[3], M.active('ret_crossbar_screw_%s' % s[-1])

    def tip(self, s):
        """The screwdriver tip (its body origin) against screw s's head: lateral offset vector (world, perpendicular to
        the screw axis), the height above the head top along the axis (m; about -1.0e-3 when seated in the recess)."""
        pd = self.d.xpos[self.bid['screwdriver']]; ps, Rs = self.pose(s); ax = Rs[:, 2]
        v = pd - (ps + Rs @ np.array([0, 0, HEAD_TOP])); h = float(v @ ax)
        return v - h * ax, h

    def nearest_screw_to_tip(self):
        pd = self.d.xpos[self.bid['screwdriver']]
        return min(self.screws, key=lambda s: np.linalg.norm(self.pose(s)[0] + self.pose(s)[1][:, 2] * HEAD_TOP - pd))

    # ---- Jacobian IK
    def jac(self, arm):
        mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.site[arm])
        return np.vstack([self.jp[:, self.dof[arm]], self.jr[:, self.dof[arm]]])

    def ik(self, arm, v, w=np.zeros(3), lam=1e-3, joints=7):
        J = self.jac(arm)[:, :joints]
        x = np.r_[v, w]; dq = np.zeros(7)
        dq[:joints] = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), x)
        return dq


def align_state(rig):
    """Each support vs its side target on the crossbar: (support, gap mm, lateral mm, angle deg); the crossbar position."""
    from ikea_runtime import rotate, quat_angle_deg
    M = rig.M; pc, qc = M.pose('crossbar'); ax = rotate(qc, [0, 0, 1]); st = []
    for side in 'LR':
        sup = M.side_support(side); ps, qs = M.pose(sup); tp = pc + rotate(qc, M.t['side_support_rel_crossbar'][side]); dl = ps - tp
        st.append((sup, round(float(dl @ ax) * 1e3, 3), round(float(np.linalg.norm(dl - (dl @ ax) * ax)) * 1e3, 3),
                   round(quat_angle_deg(qs, qc), 3)))
    return dict(sup=st, cb=(pc * 1e3).round(2).tolist())


# ---------------------------------------------------------------- closed-loop screw driving (right hand + screwdriver)
KP7, KD7 = 2000., 200.          # right_actuator7 gains (scene.xml)
SPIN = 2.5 * R.SUBSTEPS * 0.000625   # j7 target advance per tick at 2.5 rad/s (limit 2.61)


def fvec(rig, a, b):
    """Net contact force on body b from body a (world, N)."""
    m, d = rig.m, rig.d; n = d.ncon; tot = np.zeros(3); f6 = np.zeros(6)
    ia, ib = rig.bid[a], rig.bid[b]; gb = m.geom_bodyid
    for i in range(n):
        c = d.contact[i]; x, y = int(gb[c.geom1]), int(gb[c.geom2])
        if (x, y) == (ia, ib): s = 1.
        elif (x, y) == (ib, ia): s = -1.
        else: continue
        mujoco.mj_contactForce(m, d, i, f6)
        tot += s * (c.frame.reshape(3, 3).T @ f6[:3])
    return tot


def driver_axis(rig):
    return rig.d.xmat[rig.bid['screwdriver']].reshape(3, 3)[:, 2].copy()


def tcp_cmd(rig, v_tip, w=np.zeros(3), arm='right'):
    """Right-arm joint increments that move the screwdriver tip by v_tip (world) and rotate the hand by w."""
    tip = rig.d.xpos[rig.bid['screwdriver']]; tcp = rig.d.site_xpos[rig.site[arm]]
    v_tcp = v_tip - np.cross(w, tip - tcp)
    return rig.ik(arm, v_tcp, w)


def nudge(rig, dq, j7=None, grip=None, settle=0):
    """Apply joint increments dq (7) to the right arm's target (and optionally set j7 / the gripper) within the limits."""
    t = rig.ctrl; t[ARM['right']] += dq
    if j7 is not None: t[14] = j7
    if grip is not None: t[15] = grip
    n = rig.move(t)
    for _ in range(settle):
        rig.step(rig.ctrl)
    return n


def cmd_tcp(rig, arm='right'):
    """The TCP position of the arm's joint TARGETS (forward kinematics of data.ctrl on a scratch copy)."""
    if not hasattr(rig, '_d2'):
        rig._d2 = mujoco.MjData(rig.m)
    d2 = rig._d2; d2.qpos[:] = rig.d.qpos; d2.qpos[rig.qadr[arm]] = rig.d.ctrl[ARM[arm]]
    mujoco.mj_kinematics(rig.m, d2)
    return d2.site_xpos[rig.site[arm]].copy()


def slip(rig, arm='right'):
    """How far the screwdriver has slid up toward the hand along its own axis (m) — the tip relative to the TCP."""
    tip = rig.d.xpos[rig.bid['screwdriver']]; tcp = rig.d.site_xpos[rig.site[arm]]
    return float((tip - tcp) @ driver_axis(rig))


def centre(rig, s, h_target, passes=12, settle=40, align=True, tol_lat=5e-5, tol_deg=0.3, cap_deg=3.):
    """Put the screwdriver tip on screw s's TRUE axis at h_target above the head and its axis along the screw axis:
    damped-least-squares IK passes (each rotation capped at cap_deg), each followed by `settle` held ticks, until the
    lateral error is below tol_lat and the axis angle below tol_deg."""
    for _ in range(passes):
        lat, h = rig.tip(s); ax = rig.pose(s)[1][:, 2]; ad = driver_axis(rig)
        ang = math.degrees(math.acos(min(1., float(ad @ ax))))
        if np.linalg.norm(lat) < tol_lat and abs(h - h_target) < 2e-4 and (ang < tol_deg or not align):
            break
        w = np.cross(ad, ax) if align else np.zeros(3); n = np.linalg.norm(w)
        if n > math.radians(cap_deg): w *= math.radians(cap_deg) / n
        nudge(rig, tcp_cmd(rig, -lat + (h_target - h) * ax, w), settle=settle)
    lat, h = rig.tip(s)
    return float(np.linalg.norm(lat)), h, math.degrees(math.acos(min(1., float(driver_axis(rig) @ rig.pose(s)[1][:, 2]))))


def wing_offset(rig, s):
    """The angle (rad, in [-pi/4, pi/4)) about the screw axis from the nearest cross slot of screw s's head (the slots run
    along the screw's local x and y) to the screwdriver's wing (along its local x)."""
    Rs = rig.pose(s)[1]; z = Rs[:, 2]; xd = rig.d.xmat[rig.bid['screwdriver']].reshape(3, 3)[:, 0]
    xp = xd - (xd @ z) * z; xs = Rs[:, 0]
    a = math.atan2(float(np.cross(xs, xp) @ z), float(xs @ xp))
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


def align_wings(rig, s, h_target, iters=4, tol=math.radians(1.)):
    """Turn j7 so the screwdriver's wings line up with the head's cross slots (measured, both frames true state), then
    centre again."""
    for _ in range(iters):
        a = wing_offset(rig, s)
        if abs(a) < tol: break
        t = rig.ctrl; t0 = t[14]; t[14] = t0 + a; rig.move(t)          # +j7 turns the driver clockwise seen from above
        for _ in range(60): rig.step(rig.ctrl)
        if abs(wing_offset(rig, s)) > abs(a):                            # the other sense: undo twice
            t = rig.ctrl; t[14] = t0 - a; rig.move(t)
            for _ in range(60): rig.step(rig.ctrl)
        centre(rig, s, h_target, passes=6)
    return wing_offset(rig, s)


STRAIGHTEN = float(os.environ.get('ST_STRAIGHTEN', '2.0'))   # deg: a screw leaning more than this is stood up first (0: off)


def straighten(rig, s, gain=0.05, cap=1e-5, cap_w=math.radians(0.05), tol=0.8, max_ticks=4000, settle=40):
    """A screw that leans in its insert (perched on the thread start), with the screwdriver tip in its head recess: move
    the tip toward where the head would be if the screw stood on the insert axis and turn the driver axis toward the
    insert axis (damped least squares IK, small steps), so the head is pulled upright about the screw's tip; until the
    tilt is below `tol` deg. Returns what happened."""
    from ikea_runtime import rotate
    M = rig.M; k = rig.screw_state(s)[0]; t0 = rig.screw_state(s)[3]; t = 0
    while t < max_ticks:
        kk, gap, lat, tilt, on = rig.screw_state(s)
        if tilt < tol or on: break
        pc, qc = M.pose('crossbar'); ax = rotate(qc, [0, 0, 1]); tp, _ = M.insert_target(k)
        tip = rig.d.xpos[rig.bid['screwdriver']]
        goal = tp + ax * (gap + HEAD_TOP - 1.0e-3)          # the tip 1 mm down in the recess of an upright screw
        v = gain * (goal - tip); n = np.linalg.norm(v)
        if n > cap: v *= cap / n
        w = gain * np.cross(driver_axis(rig), ax); n = np.linalg.norm(w)
        if n > cap_w: w *= cap_w / n
        nudge(rig, tcp_cmd(rig, v, w)); t += 1
        if rig.tip(s)[1] > -3e-4:                           # the tip left the recess: stop
            break
    for _ in range(settle):
        rig.step(rig.ctrl)
    st = rig.screw_state(s)
    return dict(tilt0=round(float(t0), 3), tilt=round(float(st[3]), 3), gap_mm=round(st[1] * 1e3, 3), lat_mm=round(st[2] * 1e3, 3), ticks=t,
                tip_h_mm=round(rig.tip(s)[1] * 1e3, 3))


def drive(rig, s, log, f_lo=2., f_hi=6., max_ticks=40000, recentre=40, seat_gap=8e-5, hold_after=0.2, lead_max=0.05,
          every=400, h_above=2e-3, settle=40, lead_z=1e-3, tau_snug=0.5):
    """Drive screw s (already standing in its insert, the screwdriver in the right hand over it): centre the tip on the
    true screw axis, lower into the recess, spin j7 at 2.5 rad/s with a 5-15 N axial push (2 um IK steps), re-centre
    every `recentre` ticks; once the screw is within seat_gap of its seat the j7 target follows the joint (zero torque)
    until the retention weld engages, then hold_after s more. Returns a dict of what happened."""
    k0 = len(rig.ex); out = dict(screw=s)
    out['centre_above'] = centre(rig, s, h_above, settle=settle)
    out['wing_offset_deg'] = math.degrees(align_wings(rig, s, h_above))
    ax = rig.pose(s)[1][:, 2]
    # lower until the tip bears on the head (or sits in the recess)
    t = 0
    while t < 3000:
        f = -float(fvec(rig, 'screwdriver', s) @ ax)
        lat, h = rig.tip(s)
        if f > 0.8 or h < -0.8e-3: break
        nudge(rig, tcp_cmd(rig, -0.005e-3 * ax - 0.3 * lat)); t += 1
        if t % 10 == 0 and every < 400:
            log.append(dict(t=-t, f=round(f, 2), h_mm=round(h * 1e3, 3), lat_mm=round(float(np.linalg.norm(lat)) * 1e3, 3),
                            scr=[round(float(x), 5) for x in rig.screw_state(s)[:4]]))
    out['lowered'] = dict(ticks=t, h_mm=round(rig.tip(s)[1] * 1e3, 3), f=round(f, 2))
    if STRAIGHTEN and rig.screw_state(s)[3] > STRAIGHTEN and rig.tip(s)[1] < -6e-4:
        out['straighten'] = straighten(rig, s)
    weld = 'ret_crossbar_screw_%s' % s[-1]; seated_at = None; t = 0; mode = 'spin' if rig.tip(s)[1] < -6e-4 else 'seek'
    t_seek = 0
    z0 = slip(rig)                     # the screwdriver's place in the hand: never push it up the fingers
    out['slip_max_mm'] = 0.
    while t < max_ticks:
        kk, gap, latc, tilt, on = rig.screw_state(s)
        ax = rig.pose(s)[1][:, 2]
        f = -float(fvec(rig, 'screwdriver', s) @ ax)
        dz = 0.; sl = slip(rig) - z0; out['slip_max_mm'] = max(out['slip_max_mm'], sl * 1e3)
        lead = float((rig.d.site_xpos[rig.site['right']] - cmd_tcp(rig)) @ ax)     # how far the target is below the TCP
        lo, hi = (0.5, 1.5) if mode == 'seek' else (f_lo, f_hi)
        if f > hi or lead > lead_z: dz = 2e-6
        elif f < lo: dz = -2e-6
        lat, h = rig.tip(s)
        if mode == 'spin' and h > -5e-4 and gap > seat_gap:
            mode = 'seek'; t_seek = 0; out['popped'] = out.get('popped', 0) + 1
        if mode == 'seek':
            t_seek += 1
            if h < -6e-4:
                mode = 'spin'; out.setdefault('seated_in_recess_at', t)
            elif t_seek > 1200 or np.linalg.norm(lat) > 3e-4:
                # the wings did not find the slots within ~1 rad (or the tip, pinned by friction on the head, drifted off
                # the axis): lift clear, centre again, and seek again
                out['recentred'] = out.get('recentred', 0) + 1
                centre(rig, s, 1e-3, passes=6); t += 1; t_seek = 0
                continue
        if h > -5e-4:        # on the head, not in the recess yet: follow the tip's orbit (the driver sits off the j7
            v = dz * ax - 0.3 * lat          # axis in the fingers) every tick so the wings can drop into the slots
        else:
            v = dz * ax - (0.2 * lat if t % recentre == 0 else 0.)
        q7, v7 = rig.q('right')[6], rig.qv('right')[6]
        if mode == 'spin' and (gap < seat_gap or on):
            mode = 'free'; out['free_at'] = dict(t=t, gap_mm=gap * 1e3)
        if mode == 'seek':                  # slow turn (0.8 rad/s) under a light push until the wings drop into the slots
            j7 = min(rig.ctrl[14] + 0.32 * SPIN, q7 + KD7 * v7 / KP7 + lead_max)
        elif mode == 'spin':
            j7 = min(rig.ctrl[14] + SPIN, q7 + KD7 * v7 / KP7 + lead_max)
        elif not on:                         # near the seat: a light snugging torque (tau_snug) until the weld latches
            j7 = q7 + (tau_snug + KD7 * v7) / KP7
        else:                                # latched: zero torque, so the weld holds an unloaded screw
            j7 = q7 + KD7 * v7 / KP7
        nudge(rig, tcp_cmd(rig, v), j7=j7)
        t += 1
        if on and seated_at is None:
            seated_at = t; out['weld_at'] = dict(t=t, gap_mm=gap * 1e3)
        if seated_at is not None and t - seated_at >= int(hold_after / rig.s.control_dt):
            break
        if t % every == 0:
            if every < 400: print(t, mode, round(gap * 1e3, 3), round(float(np.linalg.norm(lat)) * 1e3, 3), round(h * 1e3, 3), round(f, 2), out.get('recentred'), flush=True)
            log.append(dict(t=t, mode=mode, lead_mm=round(lead * 1e3, 3), slip_mm=round(sl * 1e3, 3), gap_mm=round(gap * 1e3, 4), lat_mm=round(latc * 1e3, 4), tilt=round(tilt, 3),
                            f=round(f, 2), tip_h_mm=round(h * 1e3, 3), tip_lat_mm=round(float(np.linalg.norm(lat)) * 1e3, 3),
                            q7=round(float(q7), 3), al=align_state(rig), tau7=round(float(rig.d.actuator_force[14]), 2), met=rig.M.metrics.get('screw|%s|%d' % (s, kk))))
    out.update(ok=seated_at is not None, ticks=len(rig.ex) - k0)
    return out


# ---------------------------------------------------------------- following the demo with a right-arm offset
class Follower:
    """Demo rows with a right-arm joint offset (7) added; the offset can be ramped out (j1-6 fully, j7 by its fractional
    part of a whole turn: j7 has no limit, whole turns change nothing) between two demo rows."""

    def __init__(self, rig, D):
        self.rig, self.D = rig, D
        self.off = np.zeros(7); self.ramp = None       # (r0, r1, off_at_r0, off_target)

    def set_from(self, i):
        """The offset that makes demo row i the current ctrl (right arm)."""
        self.off = self.rig.ctrl[ARM['right']] - self.D[i][ARM['right']]; self.ramp = None

    def ramp_out(self, r0, r1):
        tgt = np.zeros(7); tgt[6] = 2 * math.pi * round(self.off[6] / (2 * math.pi))
        self.ramp = (r0, r1, self.off.copy(), tgt)

    def offset(self, i):
        if self.ramp is None: return self.off
        r0, r1, a, b = self.ramp
        if i <= r0: return a
        if i >= r1:
            self.off = b.copy(); self.ramp = None; return self.off
        u = (i - r0) / (r1 - r0); u = u * u * (3 - 2 * u)
        return a + (b - a) * u

    # the left arm's offset (the pad placements): ramped out over [r0, r1] by ramp_left
    offL = np.zeros(7); rampL = None

    def ramp_left(self, r0, r1):
        self.rampL = (r0, r1, self.offL.copy())

    def offset_left(self, i):
        if self.rampL is None: return self.offL
        r0, r1, a = self.rampL
        if i <= r0: return a
        if i >= r1:
            self.offL = np.zeros(7); self.rampL = None; return self.offL
        u = (i - r0) / (r1 - r0); u = u * u * (3 - 2 * u)
        return a * (1 - u)

    def row(self, i):
        r = self.D[i].copy(); r[ARM['right']] += self.offset(i); r[ARM['left']] += self.offset_left(i); return r

    def run(self, a, b, hook=None):
        for i in range(a, b):
            r = self.row(i)
            if hook is not None:
                r = hook(i, r)
            self.rig.move(r)


def place_hook(rig, fol, s, k, i0, i1, gain=0.02, cap=5e-5, near=0.025):
    """While carrying screw s (demo rows i0..i1): steer the screw onto insert k's axis (lateral error and tilt, true
    state), accumulating the correction in the follower's offset."""
    from ikea_runtime import rotate

    def hook(i, r):
        if not (i0 <= i < i1):
            return r
        M = rig.M; pc, qc = M.pose('crossbar'); ax = rotate(qc, [0, 0, 1]); tp, _ = M.insert_target(k)
        p, Rs = rig.pose(s); axs = Rs[:, 2]
        dl = p - tp
        if np.linalg.norm(dl) > near:
            return r
        e = dl - (dl @ ax) * ax
        v = -gain * e; n = np.linalg.norm(v)
        if n > cap: v *= cap / n
        w = gain * np.cross(axs, ax)
        tcp = rig.d.site_xpos[rig.site['right']]
        dq = rig.ik('right', v - np.cross(w, p - tcp), w)
        fol.off = fol.off + dq
        r = r.copy(); r[ARM['right']] += dq
        return r
    return hook


def tip_hook(rig, fol, s, i0, i1, gain=0.05, cap=2e-5, near_lat=4e-3, near_h=0.03):
    """While the demo brings the screwdriver onto screw s and spins it (demo rows i0..i1): keep the tip on the TRUE screw
    axis (lateral error only; the demo's own height and spin), accumulating the correction in the follower's offset."""
    def hook(i, r):
        if not (i0 <= i < i1):
            return r
        lat, h = rig.tip(s)
        if np.linalg.norm(lat) > near_lat or h > near_h:
            return r
        v = -gain * lat; n = np.linalg.norm(v)
        if n > cap: v *= cap / n
        dq = tcp_cmd(rig, v); dq[6] = 0.
        fol.off = fol.off + dq
        r = r.copy(); r[ARM['right']] += dq
        return r
    return hook


def untorque(rig, ticks=120, back=0.04, back_ticks=60):
    """Release the screwdriver's wind-up before the lift: the right j7 target follows the joint (zero actuator torque)
    for `ticks`, then backs off by `back` rad (against the drive direction) so the wings sit free in the recess."""
    for _ in range(ticks):
        t = rig.ctrl; q7, v7 = rig.q('right')[6], rig.qv('right')[6]; t[14] = q7 + KD7 * v7 / KP7; rig.move(t)
    t0 = rig.ctrl[14]
    for k in range(1, back_ticks + 1):
        t = rig.ctrl; t[14] = t0 - back * k / back_ticks; rig.move(t)
    for _ in range(40):
        rig.step(rig.ctrl)


def pad_view(rig, p):
    """Pad p against its nearest pad site (support, type): gap along the support normal (m), lateral (m), box angle
    (deg), the pad-support contact force (N), backing / adhesive welds."""
    from ikea_runtime import rotate, qmul, quat_angle_deg
    M = rig.M; pp, qp = M.pose(p); qb = qmul(qp, M.pad_box[p]); best = None
    for sup in M.supports:
        for typ in ('flat', 'slope'):
            tp, tq, qs = M.site_target(sup, typ); ax = rotate(qs, [0, 0, 1]); dl = pp - tp; gap = float(dl @ ax)
            lat = float(np.linalg.norm(dl - gap * ax)); d = float(np.linalg.norm(dl))
            if best is None or d < best['d']:
                best = dict(d=d, site='%s|%s' % (sup, typ), gap_mm=round(gap * 1e3, 3), lat_mm=round(lat * 1e3, 3),
                            ang=round(quat_angle_deg(qb, tq), 2), f=round(M.force(p, sup), 2))
    best['backed'] = M.active('backing_%s' % p)
    best['bonded'] = any(M.active('adhesive_%s_%s' % (s_, p)) for s_ in M.supports)
    best['d'] = round(best['d'] * 1e3, 1)
    return best


def pad_hook(rig, fol, i0, i1, gain=0.02, cap=3e-5, near=0.04):
    """Demo rows i0..i1 of the pad phase: while a hand holds a pad that is off the backing and not yet bonded (the left
    hand carries pads 1-2, the right hand pads 3-4, the other hand holds the backing strip), steer it onto the nearest
    free pad site (lateral error in the site plane and the box orientation, true state; the demo's own approach height),
    accumulating the correction in that arm's follower offset."""
    from ikea_runtime import rotate, qmul, qinv
    M = rig.M

    def free(p):
        return not M.active('backing_%s' % p) and not any(M.active('adhesive_%s_%s' % (s_, p)) for s_ in M.supports)

    def hook(i, r):
        if not (i0 <= i < i1):
            return r
        for arm in ('left', 'right'):
            held = [p for p in M.pads if p in rig.touching(arm) and free(p)]
            if held:
                break
        else:
            return r
        p = held[0]; pp, qp = M.pose(p); best = None
        for sup in M.supports:
            for typ in ('flat', 'slope'):
                if '%s|%s' % (sup, typ) in M.pad_at: continue
                tp, tq, qs = M.site_target(sup, typ); dd = np.linalg.norm(pp - tp)
                if best is None or dd < best[0]: best = (dd, tp, tq, qs)
        if best is None or best[0] > near:
            return r
        _, tp, tq, qs = best; ax = rotate(qs, [0, 0, 1]); dl = pp - tp; e = dl - (dl @ ax) * ax
        v = -gain * e; n = np.linalg.norm(v)
        if n > cap: v *= cap / n
        qb = qmul(qp, M.pad_box[p]); dq_ = qmul(tq, qinv(qb))
        if dq_[0] < 0: dq_ = -dq_
        w = gain * 2 * dq_[1:]                         # small-angle rotation vector toward the site's box frame
        tcp = rig.d.site_xpos[rig.site[arm]]
        dq = rig.ik(arm, v - np.cross(w, pp - tcp), w)
        if arm == 'left': fol.offL = fol.offL + dq
        else: fol.off = fol.off + dq
        r = r.copy(); r[ARM[arm]] += dq
        return r
    return hook
