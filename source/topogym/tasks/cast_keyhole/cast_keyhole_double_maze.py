"""cast_keyhole double maze (design dm4b, an experimental setting, 2026-10-12): two 9 mm plates whose five-column slot
grids are threaded through each other (topogym.assets.geometry.cast_keyhole_double_maze). Both plates start in the hands
(contact_weld_v1 with initial_intent "close", as interlocked_claws' in-hand settings); the goal is the family's placement:
the entire silver plate inside the fixed goal area, the gold plate behind the separator, held quiet."""
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from topogym.tasks.base import EntitySpec, TaskSpec

from .common import CastKeyholeTask


class CastKeyholeDoubleMazeTask(CastKeyholeTask):
    variation_id = "cast_keyhole_double_maze"
    family_id = "double_maze"
    model_revision = "double_maze_dm4b"
    difficulty = None  # Experimental; the tier is set after the model evaluation.
    fixed_parameters = ("scale",)
    geometry_resource = "variations/cast_keyhole/double_maze/geometry.json"
    pregrasp_weld = True

    def build_assets(self, folder):
        from topogym.assets.geometry.cast_keyhole_double_maze import build_double_maze
        assembly, report, self.initial_robot_q = build_double_maze(folder, self.geometry_resource)
        return assembly, report

    def declare(self, asset_dir, table_height_m):
        self.table_height_m = float(table_height_m)
        cfg, folder = self.config, Path(asset_dir)
        assembly, report = self.build_assets(folder)
        if cfg["piece_tint"] != self.defaults["piece_tint"]:
            # Appearance only: rescale each piece's baked GLB colour after generation (as the family).
            from topogym.assets.scene_visuals import scale_glb_color
            for body in assembly.bodies:
                metal = "gold" if "gold" in body.name else "silver"
                for path in sorted((folder / body.name).glob("*.glb")):
                    scale_glb_color(path, cfg["piece_tint"][metal])
        # Entities in the ALOHA model frame; finalize_spec carries them with the robot mount into the world.
        entities, self.local_vertices, targets = [], {}, []
        for body, item in zip(assembly.bodies, report["bodies"]):
            center = np.asarray(item["origin_m"])          # the link frame: the plate frame moved to the centre of mass
            pose = np.asarray(body.pose)
            self.local_vertices[body.name] = body.visual.vertices - center
            entities.append(EntitySpec(body.name, "urdf", folder / body.name / f"{body.name}.urdf",
                tuple(pose[:3, :3] @ center + pose[:3, 3]), tuple(Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]),
                material={"friction": .35}, morph_options={"align": False, "convexify": True,
                "decimate": False, "merge_fixed_links": False}, surface={"smooth": False}))
            targets.append(dict(entity=body.name, link=body.name, center_m=(np.asarray(body.grasp)[:3, 3] - center).tolist(),
                                radius_m=.03))
        from topogym.evaluation.task_metrics import rigid_asset_vertices
        self.placement_vertices = {e.name: rigid_asset_vertices(e) for e in entities}
        self.placement_targets = ("keyhole_silver",)
        self.placement_remainder = ("keyhole_gold",)
        self.weld_targets = tuple(targets)
        return TaskSpec(tuple(entities), {"rigid": {"enable_self_collision": True,
            "iterations": 100, "max_collision_pairs": 8192, "constraint_timeconst": .001}},
            dict(task_id=self.task_id, instance_id=self.instance_id, asset_report=report,
                 separation_measure="whole target in fixed region; remainder behind separator",
                 initial_state="both plates in the hands, interlocked", holding_allowed=True,
                 recommended_grasp_assist=dict(model="contact_weld_v1", initial_intent="close", targets=list(targets))))

    def finalize_spec(self, spec, *, asset_dir, scene_config):
        # Task-owned segment0 configuration (as interlocked_claws.InterlockedClawsTask.finalize_spec): the robot's home is
        # the opening's joints, the entities move with the robot mount, both grippers are commanded closed.
        if scene_config.robot_control.robot.name != "aloha":
            raise ValueError("The double-maze opening requires the original ALOHA")
        q = np.asarray(self.initial_robot_q).reshape(2, 8).copy()
        scene_config.robot_control.home.left = q[0].tolist()
        scene_config.robot_control.home.right = q[1].tolist()
        mount = scene_config.env.robot_mount
        rotation = Rotation.from_euler("xyz", mount.euler, degrees=True)
        offset = np.asarray(mount.pos, dtype=float) + rotation.apply([0., 0.,
            float(scene_config.robot_control.robot.get("base_riser_height_m", 0.)) - .02])
        entities = tuple(replace(e, position_m=tuple(offset + rotation.apply(e.position_m)),
            quaternion_wxyz=tuple((rotation * Rotation.from_quat(np.asarray(e.quaternion_wxyz)[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]))
            for e in spec.entities)
        targets = {side: {"gripper_position_m": float(scene_config.robot_control.gripper.closed_qpos)}
                   for side in ("left", "right")}
        spec = replace(spec, entities=entities, initial_robot_targets=targets, metadata={**spec.metadata,
            "initial_robot_qpos": q.tolist(), "robot_geometry_modified": False, "initial_robot_targets": targets,
            "initial_grasp_model": "physical_finger_contact_no_weld",
            "initial_state": "segment0 pregrasp; original ALOHA; both plates in the hands"})
        return super().finalize_spec(spec, asset_dir=asset_dir, scene_config=scene_config)
