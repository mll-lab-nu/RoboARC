"""variera closed-loop reference: the session wrapper (records the EXECUTED rows, data.ctrl after the clamp), true-state
measurements and damped-least-squares Jacobian servos on the TCP sites."""
from __future__ import annotations
import math, os, pickle
import numpy as np
import mujoco
import rcb_ikea as R

ARM = {'left': np.arange(0, 7), 'right': np.arange(8, 15)}
GRIP = {'left': 7, 'right': 15}
ARMCOLS = np.r_[ARM['left'], ARM['right']]
BODIES = ['shelf', 'frame_1', 'frame_2', 'screw_1', 'screw_2', 'screw_3', 'screw_4', 'screwdriver']


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def rot_err(R_des, R_cur):
    """rotation vector (world) taking R_cur to R_des."""
    Re = R_des @ R_cur.T
    ang = math.acos(max(-1., min(1., (np.trace(Re) - 1) / 2)))
    ax = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    n = np.linalg.norm(ax)
    if n < 1e-12:
        return np.zeros(3), ang
    return ax / n * ang, ang


class Rig:
    def __init__(self):
        self.s = R.Session('variera')
        self.m, self.d = self.s.model, self.s.data
        m = self.m
        self.ex: list = []
        self.lim = np.asarray(self.s.step_limit, dtype=np.float64)
        self.site = {a: m.site(a + '_tcp').id for a in ARM}
        self.dof = {a: np.array([int(m.jnt_dofadr[m.joint(f'{a}_joint{k}').id]) for k in range(1, 8)]) for a in ARM}
        self.qadr = {a: np.array([int(m.jnt_qposadr[m.joint(f'{a}_joint{k}').id]) for k in range(1, 8)]) for a in ARM}
        self.bid = {m.body(i).name: i for i in range(m.nbody)}
        self.jp = np.zeros((3, m.nv)); self.jr = np.zeros((3, m.nv))
        self.mon = self.s.S.monitor

    def step(self, row):
        self.s.step(np.asarray(row, dtype=np.float64))
        self.ex.append(self.d.ctrl.copy())

    def get(self):
        st = self.s.get_state(); st['ex'] = list(self.ex); return st

    def set(self, st):
        self.s.set_state(st); self.ex = list(st['ex'])

    @property
    def ctrl(self):
        return self.d.ctrl.copy()

    def pose(self, b):
        i = self.bid[b]; return self.d.xpos[i].copy(), self.d.xmat[i].reshape(3, 3).copy()

    def tcp(self, arm):
        s = self.site[arm]; return self.d.site_xpos[s].copy(), self.d.site_xmat[s].reshape(3, 3).copy()

    def jac(self, arm):
        mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.site[arm])
        return np.vstack([self.jp[:, self.dof[arm]], self.jr[:, self.dof[arm]]])

    def ik(self, arm, v, w=np.zeros(3), lam=1e-3):
        J = self.jac(arm); x = np.r_[v, w]
        return J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), x)

    def supported(self, obj):
        """True when the part touches anything but a robot link (it rests on / against something)."""
        m, d = self.m, self.d; n = d.ncon
        if not n: return False
        gb = m.geom_bodyid; b = self.bid[obj]
        x = gb[d.contact.geom1[:n]]; y = gb[d.contact.geom2[:n]]
        other = np.where(x == b, y, np.where(y == b, x, -1))
        other = other[other >= 0]
        return bool(any(not m.body(int(o)).name.startswith(('left_', 'right_')) for o in other))

    def touching(self, arm):
        m, d = self.m, self.d; n = d.ncon; out = set(); gb = m.geom_bodyid
        for i in range(n):
            x, y = m.body(int(gb[d.contact.geom1[i]])).name, m.body(int(gb[d.contact.geom2[i]])).name
            if x.startswith(arm + '_') and not y.startswith(arm + '_'): out.add(y)
            elif y.startswith(arm + '_') and not x.startswith(arm + '_'): out.add(x)
        return sorted(out)


class Guide:
    """The phase guide: per source row, world poses of BODIES and both TCPs (the old reference replayed without the hand
    slew, where it passes)."""

    def __init__(self, path):
        z = np.load(path)
        self.pose = z['pose']; self.tcp = z['tcp']; self.names = list(z['names'])

    def body(self, i, b):
        k = self.names.index(b); p = self.pose[i, k]
        return p[:3], quat_to_mat(p[3:])

    def tcpw(self, i, arm):
        t = self.tcp[i, 0 if arm == 'left' else 1]
        return t[:3], t[3:].reshape(3, 3)


def held_track_dq(rig, guide, i, arm, obj, frame, kp, kr, cap, dp=None, planar=True):
    """Joint increment that moves the HELD obj toward the guide's pose of obj relative to `frame` ('world' or a body) at
    source row i: the object's error twist about its origin, carried to the TCP through the lever arm, damped LSQ."""
    pg, Rg = guide.body(i, obj)
    if frame == 'world':
        p_des, R_des = pg, Rg
    else:
        pfg, Rfg = guide.body(i, frame); pf, Rf = rig.pose(frame)
        p_des = pf + Rf @ (Rfg.T @ (pg - pfg)); R_des = Rf @ (Rfg.T @ Rg)
    if dp is not None:
        p_des = p_des + dp
    p, Rc = rig.pose(obj)
    w, ang = rot_err(R_des, Rc)
    om = kr * w; v_obj = kp * (p_des - p)
    if planar and rig.supported(obj):      # resting on something: slide only (world x, y and yaw), no push into it
        v_obj[2] = 0.; om[:2] = 0.
    ptcp = rig.d.site_xpos[rig.site[arm]]
    v = v_obj + np.cross(om, ptcp - p)
    dq = np.clip(rig.ik(arm, v, om), -cap, cap)
    return dq, float(np.linalg.norm(p_des - p)), math.degrees(ang)


def approach_track_dq(rig, guide, i, arm, obj, kp, kr, cap):
    """Joint increment that moves the TCP toward the guide's TCP pose relative to obj at source row i (a grasp approach:
    the grasp lands where the guide's did on the part)."""
    pg, Rg = guide.body(i, obj); tg, Rtg = guide.tcpw(i, arm)
    po, Ro = rig.pose(obj)
    p_des = po + Ro @ (Rg.T @ (tg - pg)); R_des = Ro @ (Rg.T @ Rtg)
    p, Rt = rig.tcp(arm)
    w, ang = rot_err(R_des, Rt)
    dq = np.clip(rig.ik(arm, kp * (p_des - p), kr * w), -cap, cap)
    return dq, float(np.linalg.norm(p_des - p)), math.degrees(ang)
