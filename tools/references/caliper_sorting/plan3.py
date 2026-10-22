"""The caliper on its stand (2026-10-18): the grasps and the caliper frames the planner (oracle4.py) uses.

The caliper is fixed face up on its stand, in the measuring orientation (C x toward the tail = mount -x, C y toward the
inside jaws = mount -y); the slider frame is C moved by the jaw opening s along C x.
Grasps (TCP poses; the TCP frame: x the approach, y the closing direction, z the side the wrist camera sticks out to):
  left  on the slider: the two fingertips squarely on its two side faces (30.2 mm apart, closing along C y), the
        fingertip spheres at the faces' mid-height; approach -C z tilted about C y (+: the gripper body toward the tail)
  left  on the knob (oracle4.knob_poses): from above, fingers across or along the beam, leaning toward the tail
  left  reading pose: its wrist camera 70 mm from the scale over the coinciding vernier line, the view 55 deg from the
        face's normal on the -C y side (the hand-held design's reading camera, caliper_scene2)
  right end-face grasp of a sleeve: the pads on the two end faces (closing along the sleeve's axis), 27 mm above the
        axis; the approach tilted about the axis
All lengths in m unless noted."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import geom as v2

MM = 1e-3
PAD = 0.013                                             # pad centre past the TCP along the approach
SPHERE_PAST_TCP = 0.0155                                # the fingertip spheres (g0, g1) past the TCP
R_CAL = Rot.from_quat([0, 0, 1, 0]).as_matrix()        # C in the mount frame: half a turn about z
Q_ROLLER = 0.0151 + 0.00825                             # finger joint on the slider body, 30.2 mm wide (gap = 2 (q - 8.25 mm))
Q_KNOB = 0.0035 + 0.00825                               # the locking screw's knob, 7 mm across
Q_SLEEVE = 0.010 + 0.00825                              # 20 mm sleeve length (end-face grasp)
Q_OPEN = 0.030
Q_CLOSED_EMPTY = 0.0084


def T_from(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def tcp_frame(approach, closing):
    a = np.asarray(approach, float)
    a /= np.linalg.norm(a)
    c = np.asarray(closing, float)
    c = c - (c @ a) * a
    c /= np.linalg.norm(c)
    return np.c_[a, c, np.cross(a, c)]


def rot_y(deg):
    return Rot.from_euler('y', deg, degrees=True).as_matrix()


def rot_x(deg):
    return Rot.from_euler('x', deg, degrees=True).as_matrix()


# ---------------------------------------------------------------- grasps (in the object frames)
def G_slider(xg=44.0, tau=20.0, flip=False):
    """Left hand on the slider (slider frame): fingertip spheres at slider x xg (mm) on the two side faces, at their
    mid-height (z -1.4 mm); closing along C y; approach -C z tilted tau deg about C y (+: body toward the tail)."""
    a = rot_y(tau) @ np.array([0, 0, -1.0])
    R = tcp_frame(a, (0, -1.0 if flip else 1.0, 0))
    contact = np.array([xg, v2.GRIP_R[1], -1.4]) * MM
    return T_from(R, contact - SPHERE_PAST_TCP * a)


def G_sleeve(ro, tilt=0.0, flip=False):
    """Right hand end-face grasp near the top of a sleeve whose axis is world y (sleeve frame: axis = local z; world
    pose quat_axis_y maps local z -> world y, local y -> world -z).  Returned in WORLD-aligned terms relative to the
    sleeve centre: the caller composes T = [I, centre] @ G."""
    a = rot_y(tilt) @ np.array([0, 0, -1.0])
    R = tcp_frame(a, (0, -1.0 if flip else 1.0, 0))
    return T_from(R, np.array([0, 0, (ro - 8.0) * MM]) - PAD * a)


def G_read(s_mm, k_line, roll=0.0):
    """TCP pose in the SLIDER frame that puts the hand's wrist camera at the reading camera for vernier line k_line
    (caliper_scene2's read camera); roll (deg) turns the hand about the camera's line of sight (0: the gripper below the
    camera, its fingers under the vernier plate's edge; 90: beside it, toward the tail)."""
    xr = v2.Z0 + 1.9 * k_line
    pos, right, up, fwd = v2.read_cam_geometry(v2.READ_D, xr)
    cam = pos * MM
    x = 0.907 * fwd + 0.422 * up
    z = -0.422 * fwd + 0.907 * up
    y = np.cross(z, x)
    T = T_from(np.c_[x, y, z], cam + 0.1047 * x - 0.0855 * z)
    if roll:
        Rr = Rot.from_rotvec(fwd * np.radians(roll)).as_matrix()
        T = T_from(Rr @ T[:3, :3], cam + Rr @ (T[:3, 3] - cam))
    return T


def vernier_line(s_mm):
    lc = 0.1
    return int(round(((s_mm % 1.0)) / lc)) % 10


# ---------------------------------------------------------------- caliper frames (mount frame)
def T_cal(p, R=R_CAL):
    return T_from(R, np.asarray(p, float))


def slider_T(Tc, s_mm):
    return Tc @ T_from(np.eye(3), np.array([s_mm * MM, 0, 0]))
