"""Visual-only dashed goal boundary generated from the exact acceptance region."""
from pathlib import Path
import numpy as np
import trimesh
from topogym.tasks.base import EntitySpec

# Drawn glowing (an emissive surface) and 5 mm wide instead of 2 mm, the stripes grown outward so that their inner edge
# stays: the marker stands out on every table top and under every light of the appearance seeds (the user, 2026-10-27).
GOAL_GREEN = (.1, .85, .2)
GOAL_GLOW = 2.5


def write_goal_region(folder, region, *, line_width_m=.005, dash_length_m=.02, gap_length_m=.01):
    if not np.isfinite([line_width_m, dash_length_m, gap_length_m]).all() or min(line_width_m, dash_length_m, gap_length_m) <= 0:
        raise ValueError("Positive finite marker dimensions required")
    # The inner edge of each stripe is the effective containment boundary.
    effective = np.asarray(region.half_size_m)-region.boundary_margin_m
    half = effective+line_width_m/2
    meshes = []
    for axis in (0, 1):
        other = 1-axis
        for sign in (-1, 1):
            start = -half[axis]
            while start < half[axis]:
                end = min(start+dash_length_m, half[axis])
                extents = [line_width_m, line_width_m, .0002]
                extents[axis] = end-start
                xy = np.zeros(2)
                xy[axis], xy[other] = (start+end)/2, sign*half[other]
                world_xy = region.rotation @ xy+region.center_xy_m
                mesh = trimesh.creation.box(extents=extents)
                matrix = np.eye(4)
                matrix[:2, :2] = region.rotation
                matrix[:3, 3] = [*world_xy, region.table_height_m+.0003]
                meshes.append(mesh.apply_transform(matrix))
                start = end+gap_length_m
    path = Path(folder)/"goal_region.obj"
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.util.concatenate(meshes).export(path)
    return EntitySpec("goal_region", "mesh", path, fixed=True,
                      morph_options={"collision": False, "convexify": False, "decimate": False,
                                     "file_meshes_are_zup": True, "align": False},
                      surface={"color": (*GOAL_GREEN, 1.), "emissive": tuple(GOAL_GLOW * c for c in GOAL_GREEN)})
