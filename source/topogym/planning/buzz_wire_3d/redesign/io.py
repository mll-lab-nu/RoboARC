"""Robot I/O for the redesign oracle: a MuJoCo Session (the real thing) or a workstation testbed (the planning model's
physics with the grasp weld emulated). Both send 8-D JOINT_ABS rows (7 arm targets + the Robotiq's native command,
0 open .. 0.8 closed) at 40 Hz and expose read-only measurements."""
from __future__ import annotations

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

ARM = tuple(f"joint{i}" for i in range(1, 8))


class BaseIO:
    model = None
    data = None

    def _ids(self):
        m = self.model
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in ARM])
        self.wand = m.body("wand::wand").id
        self.hand = m.body("hand").id

    def q(self):
        return np.array(self.data.qpos[self.qadr])

    def wand_pose(self):
        return np.array(self.data.xpos[self.wand]), R.from_quat(np.array(self.data.xquat[self.wand])[[1, 2, 3, 0]])

    def hand_pose(self):
        return np.array(self.data.xpos[self.hand]), R.from_matrix(np.array(self.data.xmat[self.hand]).reshape(3, 3))

    def wand_speed(self):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.wand, v, 0)
        return float(np.linalg.norm(v[3:])), float(np.linalg.norm(v[:3]))


def set_springref(model, value):
    """The Robotiq spring links' reference angle (core 10-11: today's import has springref 0 = Genesis parity, the
    fingertips curl into a V and hold the wand at the tips; at 2.62, the Menagerie value, the pads close parallel). Set
    before the first step; returns the joints changed."""
    import os
    names = []
    for j in range(model.njnt):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if n.endswith("spring_link_joint"):
            model.qpos_spring[model.jnt_qposadr[j]] = value
            names.append(n)
    return names


class SessionIO(BaseIO):
    def __init__(self, session, log=None, springref=None):
        import os
        self.s = session
        self.w = session.env._world
        self.model, self.data = self.w.model, self.w.data
        sr = springref if springref is not None else os.environ.get("BUZZ_SPRINGREF")
        self.springref = None if sr in (None, "", "none") else float(sr)
        if self.springref is not None:
            set_springref(self.model, self.springref)
        self._ids()
        self.rows = 0
        self.log = log or (lambda **kv: None)
        self.done = False

    def step(self, q, grip):
        row = np.r_[np.asarray(q, float), float(grip)].astype(np.float32)
        self.done = bool(self.s.step(row))
        self.rows += 1
        return self.done

    def holding(self):
        st = self.w.read_state().get("grasp") or {}
        return bool(st.get("active"))

    def metrics(self):
        return dict(self.s.env._task.result.metrics)

    def touched(self):
        return self.s.episode_end is not None and not self.s.success()


class TestbedIO(BaseIO):
    """The planning model's physics (same contact classes as the buzz profile, exact diagonal), 25 ms control steps of
    40 x 0.625 ms, the grasp weld emulated: it activates when both finger pads touch the wand's grip while the command
    closes (two consecutive ticks), with the current relative pose; it releases when the command opens."""

    def __init__(self, model, q0, log=None):
        self.model = model
        self.data = mujoco.MjData(model)
        self._ids()
        m, d = model, self.data
        self.act = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a): a for a in range(m.nu)}
        for j, qq in zip(self.qadr, q0):
            d.qpos[j] = qq
        self.cmd_q, self.cmd_g = np.asarray(q0, float), 0.0
        self._apply()
        mujoco.mj_forward(m, d)
        self.weld = m.equality("grasp/arm/wand/wand").id
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        self.pad = {b: set(g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(b).id and m.geom_contype[g])
                    for b in ("left_finger", "right_finger")}
        self.grip_geoms = set(g for g in range(m.ngeom) if m.geom_bodyid[g] == self.wand)
        self.wire_metal = np.array(["wire::metal" in n or "wire_metal" in n for n in names])
        self.wand_metal = np.array([("wand::loop" in n or "wand::stem" in n or "wand_metal" in n) for n in names])
        self.rows = 0
        self.touch = False
        self.qual = 0
        self.prev_g = 0.0
        self.log = log or (lambda **kv: None)

    def _apply(self):
        for n, qq in zip(ARM, self.cmd_q):
            self.data.ctrl[self.act[f"arm/drive/{n}"]] = qq
        self.data.ctrl[self.act["arm/drive/finger_joint1"]] = self.cmd_g
        self.data.ctrl[self.act["arm/drive/finger_joint2"]] = self.cmd_g

    def holding(self):
        return bool(self.data.eq_active[self.weld])

    def _bilateral(self):
        touch = {b: False for b in self.pad}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.dist > 0:
                continue
            for b, gs in self.pad.items():
                if (c.geom1 in gs and c.geom2 in self.grip_geoms) or (c.geom2 in gs and c.geom1 in self.grip_geoms):
                    touch[b] = True
        return all(touch.values())

    def _activate(self):
        m, d = self.model, self.data
        ph, Rh = self.hand_pose()
        pw, Rw = self.wand_pose()
        # weld eq_data: anchor (hand origin in the wand frame), relpose pos (unused with anchor), quat, torquescale
        rel = Rw.inv() * Rh
        m.eq_data[self.weld, 0:3] = Rw.inv().apply(ph - pw)
        m.eq_data[self.weld, 3:6] = 0.0
        m.eq_data[self.weld, 6:10] = (Rh.inv() * Rw).as_quat()[[3, 0, 1, 2]]
        m.eq_data[self.weld, 10] = 1.0
        d.eq_active[self.weld] = 1

    def step(self, q, grip):
        self.cmd_q, self.cmd_g = np.asarray(q, float), float(grip)
        opening = self.cmd_g < self.prev_g - 1e-6 or self.cmd_g <= 1e-3
        if opening and self.holding():
            self.data.eq_active[self.weld] = 0
        self._apply()
        for tick in range(10):
            for _ in range(4):
                mujoco.mj_step(self.model, self.data)
                for i in range(self.data.ncon):
                    c = self.data.contact[i]
                    if c.dist <= 0 and ((self.wire_metal[c.geom1] and self.wand_metal[c.geom2]) or
                                        (self.wire_metal[c.geom2] and self.wand_metal[c.geom1])):
                        self.touch = True
            if not self.holding() and self.cmd_g > 0.3 and not opening:
                self.qual = self.qual + 1 if self._bilateral() else 0
                if self.qual >= 2:
                    self._activate()
        self.prev_g = self.cmd_g
        self.rows += 1
        return False

    def touched(self):
        return self.touch

    def metrics(self):
        return {}
