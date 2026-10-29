"""DROID's camera poses on RoboARC's one-arm puzzles (`python -m roboarc.vla.run --camera-poses droid`).

A DROID-trained policy sees DROID's cameras: two exterior cameras beside the robot and a wrist camera on the Robotiq.
The one-arm puzzles (the Franka + Robotiq 2F-85 scene) frame their own views. `install()` gives every scene this
process builds DROID's poses, in memory: the cameras front_oblique, agentview and wrist keep their names, resolution
and clipping planes; their poses and fields of view become DROID's. No task file and no installed file changes, and the
physics stays as it is (the cameras are the renderer's only), so the frozen-runtime checks hold.

The poses are those of DROID's simulated evaluation scene (sim-evals, the Isaac scene of DROID's robot), relative to
the robot:

    exterior   sim-evals' external_cam_2 (the robot's right: front_oblique, exterior image 1 of the droid convention)
               and external_cam (the robot's left: agentview): positions in the robot's base frame (link0),
               orientations as OpenGL cameras (looking along -z, up +y); focal length 2.1 mm on a 3.024 mm high aperture
    wrist      sim-evals' wrist camera in the Robotiq's base_link (OpenGL), focal length 2.8 mm on the same aperture; the
               base_link in DROID's flange (DROID's Isaac USD); DROID's flange 51.8 mm behind RoboARC's `hand` frame
               along its z axis with the same axes, for the gripper pose of the DroidAdapter's q7 offset (pi/4)
"""
from __future__ import annotations

import os
import sys

import numpy as np

EXTERIOR = {"front_oblique": ((0.05, -0.57, 0.66), (0.805, 0.399, -0.195, -0.393)),   # position, quaternion w x y z
            "agentview": ((0.05, 0.57, 0.66), (-0.393, -0.195, 0.399, 0.805))}
WRIST_IN_ROBOTIQ = ((0.011, -0.031, -0.074), (-0.420, 0.570, 0.576, -0.409))
ROBOTIQ_IN_FLANGE = np.array([[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0.0182], [0, 0, 0, 1.0]])
FLANGE_IN_HAND = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.0518], [0, 0, 0, 1.0]])
EXTERIOR_FOV = float(2 * np.degrees(np.arctan(3.024 / 2 / 2.1)))    # vertical, degrees (71.5)
WRIST_FOV = float(2 * np.degrees(np.arctan(3.024 / 2 / 2.8)))       # 56.7


def _pose(pos, quat) -> np.ndarray:
    """4 x 4 transform from a position and a quaternion (w, x, y, z)."""
    w, x, y, z = np.asarray(quat, dtype=np.float64) / np.linalg.norm(quat)
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                 [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                 [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = pos
    return T


def _view(T: np.ndarray) -> tuple[list, list, list]:
    """An OpenGL camera's pose -> RoboARC's camera fields: its position, a point it looks at, its up vector."""
    p, forward, up = T[:3, 3], -T[:3, 2], T[:3, 1]
    return p.tolist(), (p + 0.3 * forward).tolist(), up.tolist()


def apply(scene) -> None:
    """In place: DROID's poses for the cameras of a one-arm puzzle's scene (what topogym's load_scene_config returns)."""
    cams = scene.env.cameras
    missing = sorted({"front_oblique", "agentview", "wrist"} - set(cams))
    if missing:
        raise ValueError(f"DROID's camera poses are for the one-arm puzzles' cameras; this scene has no {missing}")
    mount = scene.env.robot_mount
    if any(float(a) for a in mount.euler):
        raise ValueError("DROID's camera poses need a robot base with the world's axes")
    base = np.eye(4)
    base[:3, 3] = [float(v) for v in mount.pos]
    for name, (pos, quat) in EXTERIOR.items():
        p, lookat, up = _view(base @ _pose(pos, quat))
        cams[name].update(pos=p, lookat=lookat, up=up, fov=EXTERIOR_FOV)
    p, lookat, up = _view(FLANGE_IN_HAND @ ROBOTIQ_IN_FLANGE @ _pose(*WRIST_IN_ROBOTIQ))
    cams["wrist"]["attach"].update(offset_pos=p, offset_lookat=lookat, offset_up=up)
    cams["wrist"]["fov"] = WRIST_FOV


def install() -> None:
    """From now on every scene this process builds has DROID's camera poses: wraps topogym's load_scene_config, which
    the simulator, its render host and its camera calibration import each time they build a scene."""
    from .._runtime import IMAGE_TOPOGYM
    topogym = os.environ.get("ROBOARC_TOPOGYM") or str(IMAGE_TOPOGYM)
    if topogym not in sys.path:
        sys.path.append(topogym)
    from topogym.config import scene
    load = scene.load_scene_config
    if getattr(load, "droid_cameras", False):
        return

    def load_scene_config(config):
        s = load(config)
        apply(s)
        return s

    load_scene_config.droid_cameras = True
    scene.load_scene_config = load_scene_config
