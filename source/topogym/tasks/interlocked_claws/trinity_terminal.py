"""Cast Trinity: every pair of the three pieces separated by a plane, all three held quiet.

The family's two-piece certificate (terminal.separation_candidate) applied pair by pair: a plane with a gap of at least
minimum_separation_m between the two pieces, tried along the world axes, both pieces' body axes and the direction between
their centres. While one piece hooks through another's window or slit no such plane exists."""
import itertools

import numpy as np
from scipy.spatial.transform import Rotation


def pairwise_separation_candidate(state, vertices, *, minimum_separation_m=.01):
    clouds, rotations, linear, angular = {}, {}, [], []
    for name, local in vertices.items():
        body = state["bodies"][name]
        p = np.asarray(body["position_m"], float)
        q = np.asarray(body["quaternion_wxyz"], float)
        v = np.asarray(body["linear_velocity_m_s"], float)
        w = np.asarray(body["angular_velocity_rad_s"], float)
        if not np.isfinite(np.r_[p, q, v, w]).all() or np.linalg.norm(q) < 1e-12:
            return False, "Non-finite or invalid physical state", {}
        r = Rotation.from_quat(q[[1, 2, 3, 0]])
        clouds[name] = r.apply(local)+p
        rotations[name] = r.as_matrix().T
        linear.append(float(np.linalg.norm(v)))
        angular.append(float(np.linalg.norm(w)))
    if len(clouds) != 3:
        raise ValueError("Cast Trinity requires exactly three complete piece geometries")
    gaps = {}
    for a, b in itertools.combinations(clouds, 2):
        delta = clouds[b].mean(0)-clouds[a].mean(0)
        axes = np.vstack([np.eye(3), rotations[a], rotations[b], delta/max(np.linalg.norm(delta), 1e-12)])
        pa, pb = clouds[a]@axes.T, clouds[b]@axes.T
        gaps[f"{a}|{b}"] = float(np.maximum(pb.min(0)-pa.max(0), pa.min(0)-pb.max(0)).max())
    separated = min(gaps.values()) >= minimum_separation_m
    quiet = max(linear) <= .001 and max(angular) <= .02
    return separated and quiet, None, dict(
        evaluation_status="available", separated=separated, quiet=quiet,
        separating_plane_gap_m=min(gaps.values()), pair_separating_plane_gaps_m=gaps,
        minimum_separation_m=minimum_separation_m,
        maximum_linear_speed_m_s=max(linear), maximum_angular_speed_rad_s=max(angular),
        holding_allowed=True, goal_kind="pairwise_whole_piece_separation")
