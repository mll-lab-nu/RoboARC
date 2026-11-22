"""Shared pieces of the gladom closed-loop reference: a session wrapper that records the EXECUTED rows (data.ctrl after
rcb_ikea's clamp, the Franka Hand slew included) and true-state measurements of the hex driver, the screws and their
thread operations."""
from __future__ import annotations
import math, os, sys
import numpy as np
import mujoco
import rcb_ikea as R

ARM = {'left': np.arange(0, 7), 'right': np.arange(8, 15)}
GRIP = {'left': 7, 'right': 15}
DOF = {'left': np.arange(0, 7), 'right': np.arange(9, 16)}      # arm hinge dofs (fingers at 7-8, 16-17)
DEMO = os.environ.get('GLADOM_DEMO', 'refs/gladom.npz')
SCREWS = {'s1-123451': 'screw_long', 's2-122202': 'screw_bottom', 's3-122202-left': 'screw_ring_l',
          's3-122202-right': 'screw_ring_r'}


def demo_rows():
    return np.load(DEMO)['ctrl']


class Rig:
    def __init__(self, noslew=False):
        self.s = R.Session('gladom')
        if noslew:   # diagnostics only: the hand unlimited (the rule before the slew)
            self.s.step_limit = R.joint_speed_limits(self.s.actuators) * self.s.control_dt
            self.s._limited = np.isfinite(self.s.step_limit)
        self.m, self.d = self.s.model, self.s.data
        self.mech = self.s.S.mech
        self.ex: list = []
        self.site = {a: self.m.site(a + '_tcp').id for a in ARM}
        self.jp = np.zeros((3, self.m.nv)); self.jr = np.zeros((3, self.m.nv))
        self.bid = {n: self.m.body(n).id for n in ('hex_driver', 'frame_a', 'frame_b', 'ring', 'tray', *SCREWS.values())}
        self.fj = {a: [self.m.joint(f'{a}_finger_joint{k}').qposadr[0] for k in (1, 2)] for a in ARM}
        self.j7 = {a: self.m.joint(f'{a}_joint7').qposadr[0] for a in ARM}
        self.ops = {o['key']: o for o in self.s.S.task['operations']}

    def step(self, row):
        self.s.step(row)
        self.ex.append(self.d.ctrl.copy())

    def get(self):
        st = self.s.get_state(); st['ex'] = list(self.ex); st['rows'] = []; return st

    def set(self, st):
        st = dict(st); st['rows'] = list(st['ex'])
        self.s.set_state(st); self.ex = list(st['ex'])

    @property
    def ctrl(self):
        return self.d.ctrl.copy()

    # ---- measurements
    def width(self, arm):
        return float(self.d.qpos[self.fj[arm][0]] + self.d.qpos[self.fj[arm][1]])

    def axis(self, name):
        return self.d.xmat[self.bid[name]].reshape(3, 3)[:, 2].copy()

    def screw(self, key):
        """gap / lateral / angle to the claimed or first candidate target, turns, engaged."""
        mech = self.mech; o = self.ops[key]
        me = mech.measure(o)
        return dict(gap=me['gap'], lat=me['lateral'], ang=me['angle'], speed=me['speed'], turns=float(mech.turns[key]),
                    engaged=mech.active(key), dwell=float(mech.dwell[key]))

    def summary(self):
        out = dict(t=round(float(self.d.time), 4))
        for a in ARM:
            out[a] = dict(w=round(self.width(a), 5), g=round(float(self.d.ctrl[GRIP[a]]), 3),
                          j7=round(float(self.d.qpos[self.j7[a]]), 4), j7c=round(float(self.d.ctrl[ARM[a][6]]), 4))
        hd = self.bid['hex_driver']
        out['drv'] = dict(p=np.round(self.d.xpos[hd], 5).tolist(), z=np.round(self.axis('hex_driver'), 4).tolist())
        for k in SCREWS:
            out[k] = {kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in self.screw(k).items()}
        return out

    def jac(self, arm):
        mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.site[arm])
        return np.vstack([self.jp[:, DOF[arm]], self.jr[:, DOF[arm]]])
