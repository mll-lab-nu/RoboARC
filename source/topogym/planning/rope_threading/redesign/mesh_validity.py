"""Real-mesh validity for the rope_threading redesign planner (coordinator, 10-10; the cloud host: needs MuJoCo and the robot
assets). The capsule model (aloha_kin.arm_capsules) kept 8 mm from fixtures, yet the real-mesh check (mesh_check.py)
found the ALOHA's collision hulls 5-12 mm into fixtures / its own base where the capsules were clear. Here every
planner configuration is also checked on the real hulls: the frozen scene's robot and table with the tier's fixtures
(mesh_check.build), robot geoms given a `clear` margin so MuJoCo reports any pair closer than that, and a configuration
is clear when no such contact remains among: the arm vs fixtures / table / the other arm, and the arm's gripper cluster
vs its own base, shoulder and upper arm. Fingertips may touch the table in a pick (allow_table_fingers). The fingers
stand at 30 mm (open) or 6 mm (closed), as in mesh_check.

Enabled in oracle.config_clear when RD_MESH_SCENE names a frozen rope_threading scene dir (module-level cache)."""
from __future__ import annotations

import os

import numpy as np

ARM_JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
GRIPPER = ("wrist_link", "gripper_base", "gripper_link", "left_finger_link", "right_finger_link", "lower_forearm_link")
TRUNK = ("base_link", "shoulder_link", "upper_arm_link", "upper_forearm_link")
_CACHE = {}


class MeshValidity:
    def __init__(self, design, scene_dir, clear=0.005):
        import mujoco
        from .mesh_check import build
        self.mj = mujoco
        self.m = build(design, scene_dir)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.body_name = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
        self.geom_name = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}" for g in range(m.ngeom)]
        self.arm_of_geom = []
        for g in range(m.ngeom):
            b = self.body_name[int(m.geom_bodyid[g])]
            arm = b.split("/")[0] if b.startswith(("left/", "right/")) else None
            self.arm_of_geom.append(arm)
            if arm is not None:
                m.geom_margin[g] = max(float(m.geom_margin[g]), clear)
        self.qadr = {arm: [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}/{j}")]) for j in ARM_JOINTS]
                     for arm in ("left", "right")}
        self.fadr = {arm: [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}/{f}")])
                           for f in ("left_finger", "right_finger")] for arm in ("left", "right")}
        self.clear = clear
        self.home = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0])

    def check(self, arm, q, opening=0.02, other_q=None, allow_table_fingers=False):
        """(ok, worst (depth beyond the margin, what, with)) of one arm configuration on the real hulls."""
        mj, m, d = self.mj, self.m, self.d
        d.qpos[:] = m.qpos0
        other = "left" if arm == "right" else "right"
        for a, v in zip(self.qadr[arm], q):
            d.qpos[a] = v
        for a, v in zip(self.qadr[other], self.home if other_q is None else other_q):
            d.qpos[a] = v
        for a in self.fadr[arm]:
            d.qpos[a] = 0.030 if opening > 0.02 else 0.006
        mj.mj_kinematics(m, d)
        mj.mj_collision(m, d)
        worst = (np.inf, None, None)
        for c in d.contact[:d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            a1, a2 = self.arm_of_geom[g1], self.arm_of_geom[g2]
            if arm not in (a1, a2):
                continue
            mine, oth = (g1, g2) if a1 == arm else (g2, g1)
            bm = self.body_name[int(m.geom_bodyid[mine])].split("/", 1)[1]
            on = self.geom_name[oth]
            bo = self.body_name[int(m.geom_bodyid[oth])]
            if self.arm_of_geom[oth] == arm:            # self contact: only the gripper cluster vs the trunk counts
                bo_short = bo.split("/", 1)[1]
                if not ((bm in GRIPPER and bo_short in TRUNK) or (bm in TRUNK and bo_short in GRIPPER)):
                    continue
            if allow_table_fingers and on == "table" and bm in ("left_finger_link", "right_finger_link"):
                continue
            gap = float(c.dist)                       # < clear (the margin) means closer than allowed
            if gap < worst[0]:
                worst = (gap, bm, on if not bo.startswith(("left/", "right/")) else bo)
        return worst[0] >= self.clear - 1e-9, worst


def get(design):
    scene = os.environ.get("RD_MESH_SCENE")
    if not scene:
        return None
    key = id(design)
    if key not in _CACHE:
        _CACHE[key] = MeshValidity(design, scene)
    return _CACHE[key]
