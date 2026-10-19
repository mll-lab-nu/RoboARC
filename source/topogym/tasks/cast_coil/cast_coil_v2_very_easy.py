"""cast_coil v2 very_easy: the official Cast Coil pieces, no handles.

Each piece is two 3 x 4 frames with a 1 x 2 window plus one round pin (assets/geometry/cast_coil_official.py). The pieces
start standing on the table: the window axis (piece y) along world +x, towards the arms, and upside down (piece z
down). A body frame is its piece's cell frame, so the
assembled pieces' frames coincide and CastCoilTask's goal (relative position < 1 mm, rotation < 0.02 rad, released,
still and on the table) applies unchanged, with zero body origins. Runs on MuJoCo (topogym.world.mujoco, profile
cast_coil_v2_very_easy); the first tier of the v2 designs, next to the released tiers.
"""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from topogym.assets.geometry.cast_coil_official import PIECE_NAMES, block_centre_cells, generate_assets
from topogym.assets.variation_resources import read_json
from topogym.tasks.base import EntitySpec, TaskSpec
from .common import CastCoilTask

# Columns: piece x -> world +y, piece y (window axis) -> world +x, piece z -> world -z.
REST_ORIENTATION = Rotation.from_matrix(np.column_stack(([0., 1., 0.], [1., 0., 0.], [0., 0., -1.])))


class CastCoilV2VeryEasyTask(CastCoilTask):
    variation_id = "cast_coil_v2_very_easy"
    family_id = "coil_design_6"
    model_revision = "coil_design_6_v1"
    difficulty = "very_easy"
    fixed_parameters = ("objective",)
    hash_when_changed = ()
    geometry_resource = "variations/cast_coil/design_6.json"
    defaults = dict(objective="assemble", stable_duration_s=.5,
                    terminal=dict(release_clearance_m=.005, maximum_drift_m=.001, table_clearance_m=.003),
                    gold_center_xy_m=[.03, .12], silver_center_xy_m=[-.30, .12], initial_floor_clearance_m=.0005,
                    friction=.35)

    def solution(self):
        """The pieces' record, without the route."""
        return read_json(self.geometry_resource)

    def grasp_assist(self):
        """contact_weld_v1 on the two pinched bar cells, centred on the finger pads."""
        from topogym.utils.grasp import GraspAssistConfig, GraspTarget
        record = self.solution()
        return GraspAssistConfig("contact_weld_v1", tuple(
            GraspTarget(name, name, tuple(np.asarray(record["pieces"][piece]["grasp_pad_cells"], float)*record["cell_m"]),
                        .025) for piece, name in PIECE_NAMES.items()))

    def declare(self, asset_dir, table_height_m):
        self.table_height_m = float(table_height_m)
        cfg = self.config
        if cfg["objective"] != "assemble":
            raise ValueError("The official Cast Coil is an assembly task")
        clearance = float(cfg["initial_floor_clearance_m"])
        if not np.isfinite(clearance) or clearance < 0:
            raise ValueError("Invalid initial floor clearance")
        record = self.solution()
        report = generate_assets(Path(asset_dir), record)
        centre = block_centre_cells(record)*float(record["cell_m"])
        self.local_vertices, self.body_origins = {}, {}
        entities, bodies = [], {}
        for name, key in (("coil_gold", "gold_center_xy_m"), ("coil_silver", "silver_center_xy_m")):
            body = report["bodies"][name]
            xy = np.asarray(cfg[key], dtype=float)
            if xy.shape != (2,) or not np.isfinite(xy).all():
                raise ValueError(f"{key} must be a finite XY position")
            points = np.asarray(body["points_m"], float)
            lowest = REST_ORIENTATION.apply(points)[:, 2].min()
            position = np.r_[xy - REST_ORIENTATION.apply(centre)[:2], self.table_height_m + clearance - lowest]
            self.local_vertices[name] = points
            self.body_origins[name] = np.zeros(3)
            entities.append(EntitySpec(name, "urdf", body["urdf"], tuple(position),
                                       tuple(REST_ORIENTATION.as_quat()[[3, 0, 1, 2]]),
                                       material={"friction": float(cfg["friction"])},
                                       morph_options={"align": False, "convexify": False, "decimate": False,
                                                      "merge_fixed_links": False}, surface={"smooth": False}))
            bodies[name] = {k: v for k, v in body.items() if k != "points_m"}
        bounds = [REST_ORIENTATION.apply(self.local_vertices[e.name]) + e.position_m for e in entities]
        gap = np.maximum(bounds[1].min(0)-bounds[0].max(0), bounds[0].min(0)-bounds[1].max(0))
        if np.max(gap[:2]) < .01:
            raise ValueError("The two pieces must start at least 10 mm apart")
        # The Genesis rigid options build the instance's Genesis scene (the render bridge's Nyx host, never stepped);
        # MuJoCo's settings are the tier's profile (topogym/world/mujoco/profiles.py, cast_coil_v2_very_easy).
        return TaskSpec(tuple(entities), {"rigid": {"enable_self_collision": True, "iterations": 100,
                                                    "constraint_timeconst": .001, "noslip_iterations": 5}},
                        dict(task_id=self.task_id, instance_id=self.instance_id, objective=cfg["objective"],
                             asset_report=dict(report, bodies=bodies), goal_requires_release=True))
