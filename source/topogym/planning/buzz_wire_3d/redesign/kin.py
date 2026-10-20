"""Panda + Robotiq kinematics on any MuJoCo model of a buzz_wire_3d scene (a Session's world model or a planning
model): TCP forward kinematics, damped-least-squares IK with joint limits, and collision / clearance queries of the
robot against the scene's static bodies. Read-only: a private MjData, never the Session's.

TCP: the hand body's origin + (0, 0, 0.1029) in the hand frame (topogym.robot.panda_robotiq.description). The fingers
close along the hand's y axis; the approach direction is the hand's z axis.
"""
from __future__ import annotations

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

TCP_IN_HAND = np.array([0., 0., .1029])
ARM = tuple(f"joint{i}" for i in range(1, 8))
HOME = np.array([0, .3926990817, 0, -1.9634954085, 0, 2.3561944902, -.7853981634])
ROBOT_BODIES = ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "tool_mount", "r85_base_mount",
                "r85_base", "r85_right_driver", "r85_right_coupler", "r85_right_spring_link", "r85_right_follower",
                "left_finger", "r85_right_silicone_pad", "r85_left_driver", "r85_left_coupler", "r85_left_spring_link",
                "r85_left_follower", "right_finger", "r85_left_silicone_pad", "hand")
GRIPPER = ("finger_joint1", "finger_joint2")
FOUR_BAR = ("finger_joint1", "r85_right_coupler_joint", "r85_right_spring_link_joint", "r85_right_follower_joint",
            "finger_joint2", "r85_left_coupler_joint", "r85_left_spring_link_joint", "r85_left_follower_joint")
# The Robotiq's settled four-bar per pad gap (m), closing on nothing in the buzz physics (workstation, 2026-10-10): pad gap,
# then the FOUR_BAR joint values. Width grasp of the 9.8 mm grip ~ command 0.69, thickness (7 mm) ~ 0.72.
# The settled four-bar per pad gap in the live Session at the Robotiq's springref 2.62 (the shared setting since
# c64294012: the pads close parallel), measured by tools/redesign/buzz_wire_3d/grip_table.py: pad gap,
# then the FOUR_BAR joint values. The 9.8 mm grip width ~ command 0.68, the 7 mm thickness ~ 0.70.
GRIP_TABLE = [[0.08333, [0.02247, 0.0, 0.02236, -0.0214, 0.02247, 0.0, 0.02236, -0.0214]], [0.07897, [0.0685, 0.0, 0.0682, -0.06529, 0.0685, 0.0, 0.0682, -0.06529]], [0.07417, [0.11761, 0.0, 0.11713, -0.1122, 0.11761, 0.0, 0.11713, -0.1122]], [0.06918, [0.16708, 0.0, 0.16646, -0.15954, 0.16708, 0.0, 0.16646, -0.15954]], [0.06404, [0.2166, 0.0, 0.21587, -0.207, 0.2166, 0.0, 0.21587, -0.207]], [0.05878, [0.26612, 0.0, 0.26531, -0.25452, 0.26612, 0.0, 0.26531, -0.25452]], [0.05339, [0.31564, 0.0, 0.31479, -0.30208, 0.31564, 0.0, 0.31479, -0.30208]], [0.04789, [0.36516, 0.0, 0.36429, -0.34969, 0.36516, 0.0, 0.36429, -0.34969]], [0.0423, [0.41468, 0.0, 0.41382, -0.39733, 0.41468, 0.0, 0.41382, -0.39733]], [0.03663, [0.4642, 0.0, 0.46339, -0.44498, 0.4642, 0.0, 0.46339, -0.44498]], [0.03088, [0.51372, 0.0, 0.51298, -0.49265, 0.51372, 0.0, 0.51298, -0.49265]], [0.02508, [0.56323, 0.0, 0.5626, -0.54032, 0.56323, 0.0, 0.5626, -0.54032]], [0.01923, [0.61275, 0.0, 0.61226, -0.58799, 0.61275, 0.0, 0.61226, -0.58799]], [0.01335, [0.66227, 0.0, 0.66195, -0.63565, 0.66227, 0.0, 0.66195, -0.63565]], [0.00744, [0.71178, 0.0, 0.71167, -0.68329, 0.71178, 0.0, 0.71167, -0.68329]], [0.00154, [0.7613, 0.0, 0.76144, -0.73089, 0.7613, 0.0, 0.76144, -0.73089]]]
# springref 0 (the old import: the fingertips curl), kept for reference
GRIP_TABLE_SR0 = [(0.0854, [0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 0.0]), (0.07119, [0.12274, -0.05527, 0.07574, 0.00738, 0.12274, -0.05527, 0.07574, 0.00738]), (0.05071, [0.2943, -0.14228, 0.17835, 0.02781, 0.2943, -0.14228, 0.17835, 0.02781]), (0.03498, [0.42304, -0.22415, 0.2444, 0.07153, 0.42304, -0.22415, 0.2444, 0.07153]), (0.02174, [0.53328, -0.315, 0.2839, 0.15008, 0.53328, -0.315, 0.2839, 0.15008]), (0.01367, [0.60592, -0.40429, 0.28126, 0.26975, 0.60592, -0.40429, 0.28126, 0.26975]), (0.00833, [0.66425, -0.50013, 0.25098, 0.43306, 0.66425, -0.50013, 0.25098, 0.43306]), (0.00488, [0.7175, -0.59897, 0.20221, 0.63837, 0.7175, -0.59897, 0.20221, 0.63837]), (0.00295, [0.76871, -0.68717, 0.14748, 0.8722, 0.76871, -0.68717, 0.14748, 0.8722])]


class PandaKin:
    def __init__(self, model, *, static_prefixes=("wire::", "wire_base::", "obst::", "table", "ground"),
                 wand_body="wand::wand", margin=0.0):
        self.m = model
        self.d = mujoco.MjData(model)
        self.qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in ARM])
        self.dadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in ARM])
        self.lo = np.array([model.jnt_range[model.joint(n).id][0] for n in ARM])
        self.hi = np.array([model.jnt_range[model.joint(n).id][1] for n in ARM])
        self.hand = model.body("hand").id
        self.gadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in GRIPPER])
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
        self.robot_body = np.array([n in ROBOT_BODIES for n in names])
        self.static_body = np.array([any(n.startswith(p) for p in static_prefixes) for n in names])
        self.wand_bid = model.body(wand_body).id if wand_body in names else -1
        if self.wand_bid >= 0:
            self.wand_qadr = model.jnt_qposadr[model.body_jntadr[self.wand_bid]]
        gb = model.geom_bodyid
        self.robot_geom = self.robot_body[gb] & (model.geom_contype > 0)
        self.static_geom = self.static_body[gb] & (model.geom_contype > 0)
        self.wand_geom = (gb == self.wand_bid) & (model.geom_contype > 0)
        self.margin = margin
        self.set_gripper(0.0)

    # ---------------------------------------------------------------- state
    def set_gripper(self, closure):
        """closure in the native unit (0 open .. 0.8 closed), both driver joints; the passive four-bar stays at 0
        (a planning approximation: the fingers' sweep is covered by the clearance margin)."""
        self.d.qpos[self.gadr] = closure

    def set_gap(self, gap_m):
        """The whole four-bar at the settled configuration whose pad gap is gap_m (interpolated in GRIP_TABLE)."""
        gaps = np.array([r[0] for r in GRIP_TABLE])[::-1]
        vals = np.array([r[1] for r in GRIP_TABLE])[::-1]
        q = np.array([np.interp(gap_m, gaps, vals[:, k]) for k in range(vals.shape[1])])
        for n, v in zip(FOUR_BAR, q):
            self.d.qpos[self.m.jnt_qposadr[self.m.joint(n).id]] = v

    def set_q(self, q):
        self.d.qpos[self.qadr] = q

    def set_wand(self, p, rot):
        if self.wand_bid >= 0:
            self.d.qpos[self.wand_qadr:self.wand_qadr + 3] = p
            self.d.qpos[self.wand_qadr + 3:self.wand_qadr + 7] = rot.as_quat()[[3, 0, 1, 2]]

    def fk(self, q):
        self.set_q(q)
        mujoco.mj_kinematics(self.m, self.d)
        Rh = self.d.xmat[self.hand].reshape(3, 3)
        return self.d.xpos[self.hand] + Rh @ TCP_IN_HAND, R.from_matrix(Rh.copy())

    # ---------------------------------------------------------------- IK
    def ik(self, p, rot, seed, *, iters=200, tol_p=2e-4, tol_r=2e-3, damping=1e-3, wr=0.3, step=0.5,
           q_prior=None, prior_w=0.0):
        """Damped least squares to the TCP pose (p, rot), joint limits enforced by clamping; -> (q, ok, pos_err,
        rot_err). wr weights the rotation residual (m per rad)."""
        q = np.clip(np.asarray(seed, float).copy(), self.lo, self.hi)
        jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
        target = rot.as_matrix()
        for it in range(iters):
            self.set_q(q)
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            Rh = self.d.xmat[self.hand].reshape(3, 3)
            ptcp = self.d.xpos[self.hand] + Rh @ TCP_IN_HAND
            ep = p - ptcp
            er = R.from_matrix(target @ Rh.T).as_rotvec()
            ne, nr = np.linalg.norm(ep), np.linalg.norm(er)
            if ne < tol_p and nr < tol_r:
                return q, True, ne, nr
            mujoco.mj_jac(self.m, self.d, jp, jr, ptcp, self.hand)
            J = np.vstack([jp[:, self.dadr], wr * jr[:, self.dadr]])
            e = np.r_[ep, wr * er]
            JJ = J @ J.T + damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJ, e)
            if prior_w and q_prior is not None:
                N = np.eye(7) - np.linalg.pinv(J) @ J
                dq += prior_w * N @ (q_prior - q)
            n = np.linalg.norm(dq)
            if n > step:
                dq *= step / n
            q = np.clip(q + dq, self.lo, self.hi)
        self.set_q(q)
        mujoco.mj_kinematics(self.m, self.d)
        Rh = self.d.xmat[self.hand].reshape(3, 3)
        ptcp = self.d.xpos[self.hand] + Rh @ TCP_IN_HAND
        ne = np.linalg.norm(p - ptcp); nr = np.linalg.norm(R.from_matrix(target @ Rh.T).as_rotvec())
        return q, bool(ne < tol_p and nr < tol_r), ne, nr

    # ---------------------------------------------------------------- collisions
    def contacts(self, q, *, wand=None, include_self=True):
        """Penetrating / near pairs at configuration q (the wand placed at `wand` = (p, rot) if given) -> list of
        (dist, kind, geom1, geom2); kind: robot-static, robot-robot, wand-static, wand-robot."""
        self.set_q(q)
        if wand is not None:
            self.set_wand(*wand)
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_collision(self.m, self.d)
        out = []
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            r1, r2 = self.robot_geom[g1], self.robot_geom[g2]
            s1, s2 = self.static_geom[g1], self.static_geom[g2]
            w1, w2 = self.wand_geom[g1], self.wand_geom[g2]
            if (r1 and s2) or (r2 and s1):
                kind = "robot-static"
            elif r1 and r2:
                if not include_self:
                    continue
                kind = "robot-robot"
            elif (w1 and s2) or (w2 and s1):
                kind = "wand-static"
            elif (w1 and r2) or (w2 and r1):
                kind = "wand-robot"
            else:
                continue
            out.append((float(c.dist), kind, g1, g2))
        return out

    def geom_name(self, g):
        return mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}:" + (
            mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[g]) or "")
