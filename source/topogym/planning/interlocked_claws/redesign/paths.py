"""Exact rigid motions for the redesigned claws tiers: moves -> dense relative poses (B in A's frame).

A move is a dict:
  {"kind": "slide", "vector": [dx, dy, dz]}                          translation in A's frame
  {"kind": "turn", "axis": [ax, ay, az], "deg": d, "pivot": [x, y, z]} rotation about an axis through a point (A's frame)
  {"kind": "turn_body", "axis_body": [...], "deg": d}                  rotation about B's own axis through B's origin
  {"kind": "screw", "axis": [...], "deg": d, "pitch": p, "pivot": [...]}  rotation + translation p per turn along axis
Each move may carry "label" and "why" (for the design note and the diagrams).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _rot(axis, deg):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return Rotation.from_rotvec(np.deg2rad(deg) * axis).as_matrix()


def apply(T, move, u=1.):
    """Pose after the fraction u in [0, 1] of the move, starting from T (4x4)."""
    T = np.asarray(T, float).copy()
    k = move["kind"]
    if k == "slide":
        T[:3, 3] += u * np.asarray(move["vector"], float)
    elif k in ("turn", "screw"):
        R = _rot(move["axis"], u * move["deg"])
        p = np.asarray(move["pivot"], float)
        T[:3, :3] = R @ T[:3, :3]
        T[:3, 3] = R @ (T[:3, 3] - p) + p
        if k == "screw":
            a = np.asarray(move["axis"], float); a /= np.linalg.norm(a)
            T[:3, 3] += a * move["pitch"] * u * move["deg"] / 360.
    elif k == "turn_body":
        R = _rot(T[:3, :3] @ np.asarray(move["axis_body"], float), u * move["deg"])
        T[:3, :3] = R @ T[:3, :3]
    else:
        raise ValueError(k)
    return T


def inverse(move):
    m = dict(move)
    for k in ("label", "why"):                 # disengagement wording, when the insertion move gives one
        if f"{k}_out" in m:
            m[k] = m.pop(f"{k}_out")
    k = m["kind"]
    if k == "slide":
        m["vector"] = (-np.asarray(m["vector"], float)).tolist()
    elif k in ("turn", "turn_body", "screw"):
        m["deg"] = -float(m["deg"])
    return m


def dense(T0, moves, step_m=.002, step_deg=1.5):
    """-> (poses, index of the first pose of each move, per-pose move index)."""
    poses, starts, which = [np.asarray(T0, float)], [], [0]
    T = np.asarray(T0, float)
    for i, mv in enumerate(moves):
        starts.append(len(poses) - 1)
        if mv["kind"] == "slide":
            n = max(1, int(np.ceil(np.linalg.norm(mv["vector"]) / step_m)))
        else:
            n = max(1, int(np.ceil(abs(mv["deg"]) / step_deg)))
            if mv["kind"] == "screw":
                n = max(n, int(np.ceil(abs(mv["pitch"] * mv["deg"] / 360.) / step_m)))
        for u in np.linspace(0., 1., n + 1)[1:]:
            poses.append(apply(T, mv, u))
            which.append(i)
        T = poses[-1]
    return poses, starts, which


def insertion_to_disengagement(T_free, insertion):
    """Inverse generation: from the free pose and the insertion moves -> (engaged pose, disengagement moves)."""
    T = np.asarray(T_free, float)
    for mv in insertion:
        T = apply(T, mv)
    return T, [inverse(m) for m in reversed(insertion)]
