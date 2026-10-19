"""Object-level geometry checks of the redesigned claws tiers: exact distances between the compound-box pieces (FCL on
their boxes), continuous path certificates, the task's separation predicate, and simple escape breakers.

Frames: piece A (teal, `other`) is the reference; B (brown, `target`) moves; a relative pose is (p, R) = B's frame in
A's frame (4x4 matrices).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _fcl():
    import fcl
    return fcl


def _boxes_object(boxes):
    """One FCL collision manager of a piece's boxes (each box an exact fcl.Box)."""
    fcl = _fcl()
    objs = []
    for b in boxes:
        T = np.asarray(b["pose"], float)
        o = fcl.CollisionObject(fcl.Box(*b["size"]), fcl.Transform(T[:3, :3], T[:3, 3]))
        objs.append((o, T))
    return objs


class PairChecker:
    """Distances between piece A (fixed at the identity, or at a given pose) and piece B at a relative pose."""

    def __init__(self, boxes_a, boxes_b, extra_static=()):
        fcl = _fcl()
        self.a = _boxes_object(list(boxes_a) + list(extra_static))
        self.b = _boxes_object(boxes_b)
        self.manager_a = fcl.DynamicAABBTreeCollisionManager()
        self.manager_a.registerObjects([o for o, _ in self.a])
        self.manager_a.setup()
        corners = []
        for bx in boxes_b:
            T = np.asarray(bx["pose"], float)
            s = np.asarray(bx["size"], float) / 2
            c = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float) * s
            corners.append(c @ T[:3, :3].T + T[:3, 3])
        self.b_corners = np.concatenate(corners)
        self.radius_b = float(np.linalg.norm(self.b_corners, axis=1).max())
        self.queries = 0

    def _place_b(self, T):
        fcl = _fcl()
        T = np.asarray(T, float)
        for o, L in self.b:
            M = T @ L
            o.setTransform(fcl.Transform(M[:3, :3], M[:3, 3]))

    def distance(self, T) -> float:
        """Minimum distance between A and B at relative pose T (<= 0: touching or intersecting)."""
        fcl = _fcl()
        self._place_b(T)
        self.queries += 1
        best = np.inf
        for o, _ in self.b:
            req = fcl.DistanceRequest()
            data = fcl.DistanceData(request=req)
            self.manager_a.distance(o, data, fcl.defaultDistanceCallback)
            best = min(best, data.result.min_distance)
            if best <= 0:
                break
        return float(best)

    def collides(self, T, margin=0.) -> bool:
        return self.distance(T) <= margin


def interp(T0, T1, u):
    """Linear translation + slerp rotation between two 4x4 poses."""
    T0, T1 = np.asarray(T0, float), np.asarray(T1, float)
    r = Slerp([0., 1.], Rotation.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]])))(u)
    out = np.eye(4)
    out[:3, :3] = r.as_matrix()
    out[:3, 3] = (1 - u) * T0[:3, 3] + u * T1[:3, 3]
    return out


def motion_bound(T0, T1, radius):
    """Largest displacement of any point of B (within `radius` of B's origin) between two poses."""
    dp = np.linalg.norm(np.asarray(T1)[:3, 3] - np.asarray(T0)[:3, 3])
    ang = Rotation.from_matrix(np.asarray(T0)[:3, :3].T @ np.asarray(T1)[:3, :3]).magnitude()
    return float(dp + radius * ang)


def check_segment(pair: PairChecker, T0, T1, clearance=0., depth=0, max_depth=22):
    """Conservative certificate that the straight (lerp + slerp) motion T0 -> T1 keeps the distance >= clearance.
    -> (ok, min sampled distance, failing u or None)."""
    stack = [(0., 1., depth)]
    cache = {}

    def d(u):
        if u not in cache:
            cache[u] = pair.distance(interp(T0, T1, u))
        return cache[u]
    worst = np.inf
    while stack:
        a, b, k = stack.pop()
        m = (a + b) / 2
        da, dm, db = d(a), d(m), d(b)
        worst = min(worst, da, dm, db)
        if min(da, dm, db) < clearance:
            u = a if da < clearance else (m if dm < clearance else b)
            return False, worst, u
        travel = motion_bound(interp(T0, T1, a), interp(T0, T1, m), pair.radius_b)
        if dm - clearance > travel + 1e-7:
            continue
        if k >= max_depth:
            return False, worst, m
        stack.append((a, m, k + 1))
        stack.append((m, b, k + 1))
    return True, worst, None


def check_path(pair: PairChecker, poses, clearance=0.):
    """Every consecutive pair of poses certified; -> dict(passed, min_distance, failure)."""
    worst = np.inf
    for i, (T0, T1) in enumerate(zip(poses[:-1], poses[1:])):
        ok, w, u = check_segment(pair, T0, T1, clearance)
        worst = min(worst, w)
        if not ok:
            return dict(passed=False, min_distance_m=worst, failure=dict(segment=i, u=u))
    return dict(passed=True, min_distance_m=worst, failure=None)


def first_collision(pair: PairChecker, T0, T1, clearance=0., steps=400):
    """Sampled: the first u in (0, 1] at which the straight motion T0 -> T1 comes closer than clearance (None: free)."""
    for u in np.linspace(0., 1., steps + 1)[1:]:
        if pair.distance(interp(T0, T1, u)) < clearance:
            return float(u)
    return None


# ------------------------------------------------------------------------------------------- separation predicate
def separating_gap(corners_a, Ta, corners_b, Tb):
    """The task's separation certificate (terminal.separation_candidate): the largest gap along the world axes, both
    body axes and the centroid direction between the two point clouds."""
    Ta, Tb = np.asarray(Ta, float), np.asarray(Tb, float)
    a = corners_a @ Ta[:3, :3].T + Ta[:3, 3]
    b = corners_b @ Tb[:3, :3].T + Tb[:3, 3]
    delta = b.mean(0) - a.mean(0)
    axes = np.vstack([np.eye(3), Ta[:3, :3].T, Tb[:3, :3].T, delta / max(np.linalg.norm(delta), 1e-12)])
    pa, pb = a @ axes.T, b @ axes.T
    return float(np.maximum(pb.min(0) - pa.max(0), pa.min(0) - pb.max(0)).max())


# ------------------------------------------------------------------------------------------------ breakers
def directions(n_extra=0, seed=0):
    """The 26 lattice directions (faces, edges, corners) plus optional random unit vectors."""
    d = [np.array([i, j, k], float) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
    d = [v / np.linalg.norm(v) for v in d]
    if n_extra:
        r = np.random.default_rng(seed).normal(size=(n_extra, 3))
        d += [v / np.linalg.norm(v) for v in r]
    return d


def escape_translation(pair: PairChecker, corners_a, corners_b, T0, direction, distance=.5, clearance=0.,
                       steps=500, min_gap=.01):
    """Translate B straight from T0 along `direction` (A's frame): -> (escaped, collided_at_m)."""
    T1 = np.array(T0, float)
    T1[:3, 3] = T1[:3, 3] + distance * np.asarray(direction, float)
    for u in np.linspace(0., 1., steps + 1)[1:]:
        T = interp(T0, T1, u)
        if pair.distance(T) < clearance:
            return False, float(u * distance)
        if separating_gap(corners_a, np.eye(4), corners_b, T) >= min_gap:
            return True, None
    return False, None


def escape_rotation(pair: PairChecker, T0, axis, pivot, max_deg=180., clearance=0., steps=360):
    """Rotate B about an axis through `pivot` (A's frame): -> the first colliding angle in degrees (None: free)."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    for ang in np.linspace(0., max_deg, steps + 1)[1:]:
        R = Rotation.from_rotvec(np.deg2rad(ang) * axis).as_matrix()
        T = np.array(T0, float)
        T[:3, :3] = R @ T[:3, :3]
        T[:3, 3] = R @ (T[:3, 3] - pivot) + pivot
        if pair.distance(T) < clearance:
            return float(ang)
    return None


def apply_moves(T0, moves):
    """A move list -> the pose list. move: ("t", [dx, dy, dz]) a translation in A's frame; ("r", axis, deg, pivot) a
    rotation about the axis through pivot (A's frame; pivot "B" = B's current origin); ("s", axis, deg, pitch, pivot)
    a screw (rotation + translation of pitch metres per turn along the axis)."""
    poses = [np.array(T0, float)]
    for mv in moves:
        T = poses[-1].copy()
        if mv[0] == "t":
            T[:3, 3] += np.asarray(mv[1], float)
        elif mv[0] in ("r", "s"):
            axis = np.asarray(mv[1], float) / np.linalg.norm(mv[1])
            deg = float(mv[2])
            pivot = T[:3, 3].copy() if (len(mv) < (4 if mv[0] == "r" else 5) or isinstance(mv[-1], str)) \
                else np.asarray(mv[-1], float)
            R = Rotation.from_rotvec(np.deg2rad(deg) * axis).as_matrix()
            T[:3, :3] = R @ T[:3, :3]
            T[:3, 3] = R @ (T[:3, 3] - pivot) + pivot
            if mv[0] == "s":
                T[:3, 3] += axis * float(mv[3]) * deg / 360.
        else:
            raise ValueError(mv)
        poses.append(T)
    return poses


def densify(poses, max_step_m=.003, max_step_deg=2., radius=.1):
    """Insert samples so that consecutive poses differ by at most max_step (translation + radius x angle)."""
    out = [np.asarray(poses[0], float)]
    for T0, T1 in zip(poses[:-1], poses[1:]):
        dp = np.linalg.norm(np.asarray(T1)[:3, 3] - np.asarray(T0)[:3, 3])
        ang = np.rad2deg(Rotation.from_matrix(np.asarray(T0)[:3, :3].T @ np.asarray(T1)[:3, :3]).magnitude())
        n = max(1, int(np.ceil(max(dp / max_step_m, ang / max_step_deg))))
        for u in np.linspace(0., 1., n + 1)[1:]:
            out.append(interp(T0, T1, u))
    return out
