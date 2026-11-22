"""Thin optional wrappers. They only change the action or observation format; every step still goes through the
environment's own step() (the graded tooling's Session.step and limits). No skills (grasp, place, navigate) here.

EEPoseAction    end-effector pose actions (absolute or delta) for the puzzles, solved with the tooling's own IK
                (rcb_topogym.Session.ik, the IK of the standard mode's ee_pose action) and its IK-jump rule.
"""
from __future__ import annotations

import math

import numpy as np

from . import spaces


def _quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def axis_angle_quat(v) -> np.ndarray:
    """Rotation vector (axis x angle, rad) -> quaternion wxyz."""
    v = np.asarray(v, dtype=np.float64)
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([[math.cos(a / 2)], math.sin(a / 2) * v / a])


class EEPoseAction:
    """End-effector actions for a puzzle environment (PuzzleEnv). Per arm (the order of env.arms) 8 numbers:

      mode="absolute": x y z qw qx qy qz gripper  (the TCP pose in the robot frame, as the proprio tcp_pose and the
                       standard mode's ee_pose rows; gripper in the joint action's command space)
      mode="delta":    dx dy dz rx ry rz _ gripper (a translation (m) and a rotation vector (rad) applied to the measured
                       TCP pose, both in the robot frame; the 7th entry is ignored; gripper absolute)

    Each step solves the arm joint targets with Session.ik (bounded least squares on the robot's own kinematics, seeded
    at the measured joints, no collision checking) and passes the joint row to env.step(), so the environment's rules
    (the joint-speed rule, the robot's limits, the table guard) apply as to any joint action. As in the standard mode, an
    arm whose IK residual exceeds 2 cm / 20 deg, or whose solution is more than rcb_topogym.IK_JUMP_LIMIT_RAD (0.5 rad)
    from the measured joints on any joint (another IK branch), holds its targets; info["ee_plan"] says why."""

    def __init__(self, env, mode: str = "delta"):
        base = env.unwrapped
        if not hasattr(base, "session") or base.task.suite != "puzzles":
            raise TypeError("EEPoseAction works on the RoboARC puzzles (the IKEA tooling has no IK; its standard mode has "
                            "no ee_pose action either)")
        if mode not in ("absolute", "delta"):
            raise ValueError("mode must be 'absolute' or 'delta'")
        self.env, self.base, self.mode = env, base, mode
        self.observation_space = env.observation_space
        n = 8 * len(base.arms)
        lo, hi = np.full(n, -np.inf, np.float64), np.full(n, np.inf, np.float64)
        for i, arm in enumerate(base.arms):
            g = base.session.robot.grip_index[arm]
            lo[8 * i + 7], hi[8 * i + 7] = base.limits[g]
        self.action_space = spaces.Box(lo, hi, (n,), np.float64)
        self.ik_jump_limit_rad = float(base.M.IK_JUMP_LIMIT_RAD)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kw):
        return self.env.reset(**kw)

    def joint_row(self, action) -> tuple[np.ndarray, list]:
        """The joint action row for an EE action (IK per arm), and the per-arm plan notes."""
        b, S = self.base, self.base.session
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size != self.action_space.shape[0]:
            raise ValueError(f"an EE action has {self.action_space.shape[0]} entries (8 per arm: {b.arms})")
        row = np.asarray(S.hold_action(), dtype=np.float64).copy()
        plan = []
        for i, arm in enumerate(b.arms):
            seg = a[8 * i:8 * i + 8]
            p_r, q_r = b.to_robot(*S.ee_pose(arm))           # the measured TCP pose, robot frame
            if self.mode == "absolute":
                pos, quat = seg[:3], seg[3:7] / max(1e-9, float(np.linalg.norm(seg[3:7])))
            else:
                pos, quat = p_r + seg[:3], _quat_mul(axis_angle_quat(seg[3:6]), q_r)
            note = {"arm": arm}
            qa, ep, er = S.ik(arm, *b.to_world(pos, quat))
            sl = S.robot.arm_slice[arm]
            qa = np.clip(qa, b.limits[sl, 0], b.limits[sl, 1])
            jump, joint = S.ik_jump(arm, qa)
            if ep > 0.02 or er > 0.35:
                note["status"] = f"held: no IK solution within 2 cm / 20 deg (residual {ep * 1000:.0f} mm / {math.degrees(er):.0f} deg)"
            elif jump > self.ik_jump_limit_rad:
                note["status"] = f"held: the IK solution is {jump:.2f} rad from the measured joints on {joint} (another branch)"
            else:
                row[sl] = qa
                note["status"] = "ok"
            row[S.robot.grip_index[arm]] = float(seg[7])
            plan.append(note)
        return row, plan

    def step(self, action):
        row, plan = self.joint_row(action)
        obs, rew, term, trunc, info = self.env.step(row)
        info["ee_plan"] = plan
        return obs, rew, term, trunc, info

    @property
    def unwrapped(self):
        return self.base
