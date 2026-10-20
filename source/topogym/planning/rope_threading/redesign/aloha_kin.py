"""ALOHA (two ViperX 300 6-DoF arms) kinematics in numpy for the rope_threading redesign planner.

Task-local frame (as every rope_threading instance places it): origin = the robot root on the table surface
(world (-0.1356786, 0, table_z)), x towards the right arm, z up. The chain below is copied from the MuJoCo scene the port
builds (instance_assets/mujoco/scene.xml of rope-threading-*-i00: body offsets, joint axes, the gripper site), so the
TCP here is the Session's TCP (gripper_link + (0.13, 0, -0.003), topogym/robot/aloha/description.py).

Gripper frame (= gripper_link frame): x = approach (fingers point along +x), y = closing axis (the two finger pads are
separated along y), z = the finger-plane normal. A rope held between the pads runs perpendicular to y.

No MuJoCo here (workstation-safe): FK is a product of exponentials of the MJCF chain; `verify_against_mujoco` (the cloud host) checks it
against mj_kinematics on the real model. Collision checks use conservative capsules for the arm links (`arm_capsules`).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT_WORLD = np.array([-0.1356786, 0.0, 0.0])          # + table_z in z (0.9196429 in the instances)
TABLE_Z_WORLD = 0.9196429
BASES = {"left": (np.array([-0.469, -0.019, 0.0209]), 0.0),            # base position (task frame), yaw (rad)
         "right": (np.array([0.469, -0.019, 0.0209]), np.pi)}
LIMITS = np.array([[-3.14158, 3.14158], [-1.85005, 1.25664], [-1.76278, 1.6057],
                   [-3.14158, 3.14158], [-1.8675, 2.23402], [-3.14158, 3.14158]])
HOME = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0])
# (child offset in parent frame, joint axis in child frame) from base_link down to gripper_link
CHAIN = [(np.array([0.0, 0.0, 0.079]), np.array([0.0, 0.0, 1.0])),        # waist (shoulder_link)
         (np.array([0.0, 0.0, 0.04805]), np.array([0.0, 1.0, 0.0])),      # shoulder (upper_arm_link)
         (np.array([0.05955, 0.0, 0.3]), np.array([0.0, 1.0, 0.0])),      # elbow (upper_forearm_link)
         (np.array([0.2, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),          # forearm_roll (lower_forearm_link)
         (np.array([0.1, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),          # wrist_angle (wrist_link)
         (np.array([0.069744, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))]     # wrist_rotate (gripper_link)
TCP_IN_HAND = np.array([0.13, 0.0, -0.003])
MAX_JOINT_SPEED = np.pi                                  # rad/s (the ALOHA's servo limit used by the port)


def _rot(axis, angle):
    return Rotation.from_rotvec(np.asarray(axis, float) * float(angle)).as_matrix()


def frames(arm: str, q) -> list:
    """World-free task-frame 4x4 poses of base_link, shoulder_link, upper_arm, upper_forearm, lower_forearm, wrist,
    gripper_link (7 frames) for joint vector q (6)."""
    base, yaw = BASES[arm]
    T = np.eye(4)
    T[:3, :3] = _rot([0, 0, 1], yaw)
    T[:3, 3] = base
    out = [T.copy()]
    for (offset, axis), angle in zip(CHAIN, np.asarray(q, float)):
        step = np.eye(4)
        step[:3, 3] = offset
        step[:3, :3] = _rot(axis, angle)
        T = T @ step
        out.append(T.copy())
    return out


def tcp_pose(arm: str, q) -> np.ndarray:
    """4x4 task-frame pose of the TCP (the gripper_link orientation, the gripper site position)."""
    G = frames(arm, q)[-1]
    T = G.copy()
    T[:3, 3] = G[:3, :3] @ TCP_IN_HAND + G[:3, 3]
    return T


def gripper_frame(approach, closing=None, rope_tangent=None) -> np.ndarray:
    """Rotation matrix (columns x approach, y closing, z) from an approach direction and either the closing axis or the
    rope tangent the pads must hold (closing = approach x tangent; needle grasps need `closing`)."""
    a = np.asarray(approach, float)
    a = a / np.linalg.norm(a)
    if closing is None:
        t = np.asarray(rope_tangent, float)
        y = np.cross(a, t)
        if np.linalg.norm(y) < 1e-6:
            raise ValueError("needle grasp: give the closing axis")
    else:
        y = np.asarray(closing, float) - np.dot(closing, a) * a
    y = y / np.linalg.norm(y)
    z = np.cross(a, y)
    return np.column_stack([a, y, z])


def ik(arm: str, position, rotation=None, *, approach=None, seed=None, weight_rot=0.3, iters=200,
       free_roll=False, fast=False) -> tuple[np.ndarray, float, float]:
    """Bounded least squares IK for one arm. rotation: full 3x3 target; or approach (unit 3-vector) only, with
    free_roll the roll about the approach axis is free. Returns q, position error (m), orientation error (rad)."""
    p_t = np.asarray(position, float)
    seeds = [np.asarray(seed, float)] if seed is not None else []
    seeds += [HOME, np.array([0, -0.5, 0.6, 0, 0.6, 0]), np.array([0, 0.2, -0.2, 0, 1.2, 0]),
              np.array([0.6, -0.4, 0.5, 0, 0.8, 0]), np.array([-0.6, -0.4, 0.5, 0, 0.8, 0])]
    if fast:                         # candidate screening: the previous joints and two generic seeds only
        seeds = seeds[:3]
        iters = min(iters, 100)
    # aim the waist at the target from each seed
    base, yaw = BASES[arm]
    d = p_t - base
    local = _rot([0, 0, 1], -yaw) @ d
    waist = np.arctan2(local[1], local[0])

    def residual(q):
        T = tcp_pose(arm, q)
        r = list(T[:3, 3] - p_t)
        if rotation is not None:
            r += list(weight_rot * Rotation.from_matrix(np.asarray(rotation).T @ T[:3, :3]).as_rotvec())
        elif approach is not None:
            a = np.asarray(approach, float) / np.linalg.norm(approach)
            r += list(weight_rot * (T[:3, 0] - a))
        return np.asarray(r)

    best = None
    for s in seeds:
        s = np.array(s, float)
        if seed is None or s is not seeds[0]:
            s[0] = waist
        s = np.clip(s, LIMITS[:, 0] + 1e-6, LIMITS[:, 1] - 1e-6)
        sol = least_squares(residual, s, bounds=(LIMITS[:, 0], LIMITS[:, 1]), max_nfev=iters, xtol=1e-12, ftol=1e-12,
                            gtol=1e-12)
        T = tcp_pose(arm, sol.x)
        perr = float(np.linalg.norm(T[:3, 3] - p_t))
        if rotation is not None:
            rerr = float(np.linalg.norm(Rotation.from_matrix(np.asarray(rotation).T @ T[:3, :3]).as_rotvec()))
        elif approach is not None:
            a = np.asarray(approach, float) / np.linalg.norm(approach)
            rerr = float(np.arccos(np.clip(T[:3, 0] @ a, -1, 1)))
        else:
            rerr = 0.0
        score = perr + 0.05 * rerr
        if best is None or score < best[0]:
            best = (score, sol.x.copy(), perr, rerr)
        if perr < 1e-4 and rerr < 1e-3:
            break
    return best[1], best[2], best[3]


# ------------------------------------------------------------------------------------------- conservative capsules
# (frame index, start in frame, end in frame, radius): enclose the ViperX link meshes with margin (checked on the cloud host
# against the MuJoCo meshes: verify_against_mujoco). Frame indices into frames(): 0 base, 1 shoulder, 2 upper_arm,
# 3 upper_forearm, 4 lower_forearm, 5 wrist, 6 gripper_link.
ARM_CAPSULES = [
    ("base", 0, [0, 0, 0.0], [0, 0, 0.07], 0.06),
    ("shoulder", 1, [0, 0, 0.0], [0, 0, 0.05], 0.05),
    ("upper_arm", 2, [0, 0, 0.0], [0.05955, 0, 0.3], 0.04),
    ("forearm", 3, [0, 0, 0.0], [0.3, 0, 0], 0.035),
    ("wrist", 5, [-0.02, 0, 0], [0.069744, 0, 0], 0.035),
    ("gripper_mount", 6, [0.0, 0, 0], [0.055, 0, 0], 0.035),       # wrist mount / servo housing
    ("gripper_rail", 6, [0.05, -0.05, 0], [0.05, 0.05, 0], 0.025),  # the finger rail across the closing axis
    # (coordinator 10-10: the capsules are the fast pre-check; the real hulls decide (mesh_validity, RD_MESH_SCENE).
    # Enlarging them to cover the hulls made the grommet feeds infeasible, so they keep their original size.)
    ("camera", 6, [0.008, 0, 0.078], [0.042, 0, 0.078], 0.022),    # the wrist D405 (centre ~ (0.025, 0, 0.0825))
]
# the fingers: plates from the gripper body to just past the TCP, at +-half opening along y
FINGER_LEN = (0.07, 0.145)          # along gripper x (TCP at 0.13)
FINGER_RADIUS = 0.011


def arm_capsules(arm: str, q, opening=0.02) -> list:
    """[(name, A, B, radius)] task-frame capsules approximating the arm, gripper and fingers at q."""
    F = frames(arm, q)
    out = []
    for name, i, a, b, r in ARM_CAPSULES:
        T = F[i]
        out.append((name, T[:3, :3] @ np.asarray(a) + T[:3, 3], T[:3, :3] @ np.asarray(b) + T[:3, 3], r))
    G = F[6]
    for side, sgn in (("finger_l", 1), ("finger_r", -1)):
        a = G[:3, :3] @ np.array([FINGER_LEN[0], sgn * (opening / 2 + FINGER_RADIUS), -0.003]) + G[:3, 3]
        b = G[:3, :3] @ np.array([FINGER_LEN[1], sgn * (opening / 2 + FINGER_RADIUS), -0.003]) + G[:3, 3]
        out.append((side, a, b, FINGER_RADIUS))
    return out


def segment_distance(p1, q1, p2, q2) -> float:
    """Minimum distance between segments p1q1 and p2q2."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a <= 1e-15 and e <= 1e-15:
        return float(np.linalg.norm(r))
    if a <= 1e-15:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e <= 1e-15:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-15 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm(p1 + d1 * s - (p2 + d2 * t)))


def point_box_distance(p, center, half, R=np.eye(3)) -> float:
    local = np.asarray(R).T @ (np.asarray(p) - center)
    d = np.maximum(np.abs(local) - half, 0)
    return float(np.linalg.norm(d))


def segment_box_distance(a, b, center, half, R=np.eye(3), samples=24) -> float:
    ts = np.linspace(0, 1, samples)
    return min(point_box_distance(a + t * (b - a), center, half, R) for t in ts)


def joint_path_time(qs) -> float:
    """Minimum time to follow a joint path at the joint speed limit (piecewise linear)."""
    qs = np.asarray(qs)
    return float(np.sum(np.max(np.abs(np.diff(qs, axis=0)), axis=1)) / MAX_JOINT_SPEED) if len(qs) > 1 else 0.0


def world_from_task(p):
    return np.asarray(p) + ROOT_WORLD + np.array([0, 0, TABLE_Z_WORLD])


def task_from_world(p):
    return np.asarray(p) - ROOT_WORLD - np.array([0, 0, TABLE_Z_WORLD])
