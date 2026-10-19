"""Cast Cage Hedgehog: a faithful model of Hanayama's Cast Cage (N. Yoshigahara, 1996) at four times its real size, with a
push rod as a tool.

The cage is a closed barrel (outer radius 75 mm, 260 mm tall with its collars) with four side windows, each two lobes
joined by an S-shaped waist. W0 (+x in the cage frame) is traced from a photograph of the original; W90, W180 and W270
are W0 mirrored, upside down and half-turned, 6 mm narrower all round. The star has six tapered spikes U, E, N, D, W,
S. It starts at rest inside the cage with spike E 3 mm out of W0. The goal is the star in the goal region. The rod is
neither a placement target nor part of the remainder that must stay on the system side.

The cage and star are the design's exact convex pieces (topogym/assets/geometry/hedgehog_cast_cage.py). Cage placement and rod rest come from the robot placement study
(2026-10-11). At the Panda's standard home the wrist would be inside the cage on its mount, so the robot starts from
this setting's own home (robot_home_q): the standard home with the base joint turned 30 deg toward the rod.
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from topogym.assets.geometry.hedgehog_cast_cage import (
    RESOURCE, ROD_HANDLE_M, load_design, rod_mesh, rod_rest, write_rod,
)
from topogym.assets.geometry.hedgehog_in_cage import write_body
from topogym.solvers.collision.triangle_meshes import TriangleMeshPair
from topogym.tasks.base import EntitySpec, merge_parameters
from .common import HedgehogInCageTask

CAGE_COLOR = (.06, .06, .07, 1.)     # the original's black barrel
STAR_COLOR = (.72, .33, .16, 1.)     # and its copper star (the scene's "copper" surface colour)


class HedgehogCastCageTask(HedgehogInCageTask):
    """The Cast Cage and its star, a push rod on the table; cage placement, star and rod poses fixed by the setting."""

    variation_id = "hedgehog_in_cage_cast_cage"
    difficulty = None  # Explicit experimental entry; the tiers are unchanged.
    family_id = "cast_cage"
    model_revision = "cast_cage_v1"
    # The public goal (topogym.tasks.goal_description): the piece is a star, and one sentence about the rod (the user,
    # 2026-10-12: the text mentions the rod; the rod falling off the table stays a failure, as for every task body).
    placement_target_description = "the whole star, including every spike"
    placement_scene_note = ("A thin rod with a box handle lies on the table; you may use it as a tool. It is a task piece: "
                            "if it falls off the table (more than 100 mm below the tabletop), the episode ends in failure.")
    # ball_pose: the star in the cage frame (origin at the cavity centre), at rest: the design's start rest D25
    # settled in this task's MuJoCo physics. With its
    # friction saturated at spike W it slides for 115 s, until spike D wedges between the floor and the wall, and rests
    # 3.30 mm / 3.9 deg from D25 (spike E 2.7 mm out of W0, the others inside); then moved 0.12 mm / 0.06 deg to the
    # nearest pose 0.02 mm clear of the cage (the smallest twist clearing MuJoCo's contact distances).
    # rod_pose: the rod lying on the table, its grip point (the handle's centre) at grip_xy_m in the world, its tip toward
    # yaw_deg (from world +x). robot_home_q: the Panda's seven arm joints at reset (rad), the standard home
    # (topogym/config/profiles/robot_control_configs/panda_robotiq.yaml) with joint 1 at +30 deg; its hand is then 139 mm
    # from the cage, 147 mm from the star and 134 mm from the rod (the standard home is 48 mm inside the cage and 38 mm
    # inside the mount).
    defaults = merge_parameters(HedgehogInCageTask.defaults, {
        "scale": 4., "center_xy_m": [.05, .05], "yaw_deg": 120., "mount_height_m": .24, "mount_width_m": .10,
        "finish": "metal",
        "ball_pose": {"position_m": [0.001481, -0.006543, -0.036756],
                      "quaternion_wxyz": [0.902177, -0.333163, 0.218527, -0.165302]},
        "rod_pose": {"grip_xy_m": [-.25, .25], "yaw_deg": 46.2},
        "robot_home_q": [0.5235987756, 0.3926990817, 0., -1.9634954085, 0., 2.3561944902, -0.7853981634],
    })
    fixed_parameters = ("scale", "center_xy_m", "yaw_deg", "mount_height_m", "mount_width_m", "ball_pose", "rod_pose",
                        "robot_home_q")

    def build_assets(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        design = load_design()
        cage_parts, star_parts = design["cage"], list(design["star"].values())
        cage, star = trimesh.util.concatenate(cage_parts), trimesh.util.concatenate(star_parts)
        finish = self.config["finish"]
        report = dict(template_id="hanayama_cast_cage", scale=float(design["scale"]), geometry_resource=RESOURCE,
                      geometry_status="faithful Cast Cage model (runs/hedgehog_cast_cage_20261010/faithful)",
                      cage_geometry=dict(outer_radius_m=float(design["outer_radius_m"]),
                                         inner_radius_m=float(design["inner_radius_m"]),
                                         height_m=2 * float(design["cage_bottom_m"]),
                                         windows=dict(zip(design["window_names"].tolist(),
                                                          design["window_centres_deg"].tolist()))),
                      cage=write_body(folder, "cage", cage, cage_parts, CAGE_COLOR, 2700, finish),
                      hedgehog=write_body(folder, "hedgehog", star, star_parts, STAR_COLOR, 2700, finish),
                      push_rod=write_rod(folder),
                      initial_pose=self.checked_ball_pose(cage, star, design),
                      reference_route_verified=False, physics_qualified=False)
        (folder / "asset_report.json").write_text(json.dumps(report, indent=2) + "\n")
        return cage, star, -float(design["cage_bottom_m"]), report

    def checked_ball_pose(self, cage, star, design):
        """The configured star pose, rejected if the star would start inside the cage walls; with how far each spike tip
        reaches beyond the cage's outer wall (negative: inside) and the window nearest it."""
        pose = self.config["ball_pose"]
        q = np.asarray(pose["quaternion_wxyz"], dtype=float)
        rotation = Rotation.from_quat(q[[1, 2, 3, 0]])
        transform = np.eye(4)
        transform[:3, :3] = rotation.as_matrix()
        transform[:3, 3] = pose["position_m"]
        clearance = TriangleMeshPair((cage, star)).clearance((np.eye(4), transform))
        if clearance < 0:
            raise ValueError(f"{self.variation_id}: the star pose intersects the cage")
        windows = dict(zip(design["window_names"].tolist(), design["window_centres_deg"].tolist()))
        tips = {}
        for name, axis, length in zip(design["spike_names"].tolist(), design["spike_directions"], design["spike_lengths_m"]):
            tip = rotation.apply(axis * length) + pose["position_m"]
            azimuth = np.degrees(np.arctan2(tip[1], tip[0]))
            window = min(windows, key=lambda w: abs((azimuth - windows[w] + 180) % 360 - 180))
            tips[name] = dict(beyond_outer_wall_m=float(np.hypot(*tip[:2]) - design["outer_radius_m"]), window=window)
        return dict(position_m=list(pose["position_m"]), quaternion_wxyz=q.tolist(), clearance_to_cage_m=clearance,
                    spike_tips=tips, gravity_settled=True)

    def declare(self, asset_dir, table_height_m):
        """The family's mount, cage and star, then the rod. Added after the family's declaration, the rod is neither in
        placement_targets nor in placement_remainder; it is a task body for the physical rules (local_vertices)."""
        spec = super().declare(asset_dir, table_height_m)
        pose = self.config["rod_pose"]
        xy = np.asarray(pose["grip_xy_m"], dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("Invalid push rod grip_xy_m")
        pitch, height = rod_rest()
        rotation = Rotation.from_euler("z", float(pose["yaw_deg"]), degrees=True) * Rotation.from_euler("y", pitch)
        self.local_vertices["push_rod"] = np.asarray(rod_mesh().vertices)
        rod = EntitySpec("push_rod", "urdf", Path(asset_dir) / "push_rod.urdf", (*xy, self.table_height_m + height),
                         tuple(rotation.as_quat()[[3, 0, 1, 2]]), material={"friction": .3},
                         morph_options={"convexify": True, "decimate": False, "merge_fixed_links": False, "align": False})
        return replace(spec, entities=(*spec.entities, rod))

    def finalize_spec(self, spec, *, asset_dir, scene_config):
        """The robot's reset pose: the profile's home with this setting's arm joints (robot_home_q), its gripper entries
        unchanged; then the family's goal region."""
        home = list(scene_config.robot_control.home.arm)
        home[:7] = self.config["robot_home_q"]
        scene_config.robot_control.home.arm = home
        spec = replace(spec, metadata={**spec.metadata, "initial_robot_q": home})
        return super().finalize_spec(spec, asset_dir=asset_dir, scene_config=scene_config)

    def grasp_targets(self):
        """The assisted grasp's regions (common.recommended_grasp_assist) for the star and the rod."""
        from topogym.utils.grasp import rigid_grasp_region
        star = trimesh.util.concatenate(list(load_design()["star"].values()))
        return (rigid_grasp_region("hedgehog_ball", "hedgehog", star.vertices),
                rigid_grasp_region("push_rod", "push_rod", trimesh.creation.box(ROD_HANDLE_M).vertices))
