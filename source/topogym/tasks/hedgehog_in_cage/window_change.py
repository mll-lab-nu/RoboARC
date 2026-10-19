"""Window-change Hedgehog: the CLASSIC cage and ball with narrower window widths.

A harder design for the very_easy tier, registered as its own experimental variation; very_easy itself stays the
below_easy three-window cage. Window widths (chord at the inner wall): 0 deg 43 mm, 72 / 288 deg 56.5 mm, 144 / 216 deg
61 mm. The cage is turned 40 deg about the vertical (``yaw_deg``).
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from topogym.assets.geometry.hedgehog_in_cage import (
    HedgehogLayout, cage_meshes, hedgehog_meshes, load_template, write_body,
)
from topogym.solvers.collision.triangle_meshes import TriangleMeshPair
from topogym.tasks.base import merge_parameters
from .hedgehog_in_cage_easy import HedgehogInCageEasyTask


class HedgehogWindowChangeTask(HedgehogInCageEasyTask):
    """The CLASSIC cage with this setting's window widths; ball pose and cage yaw fixed."""

    variation_id = "hedgehog_in_cage_window_change"
    difficulty = None  # Explicit experimental entry (the harder very_easy design); the tiers are unchanged.
    model_revision = "window_change_v1"
    family_id = "window_change"
    # Easy-tier ball settled under gravity in Genesis (CPU float64, 25 ms tick / 40 substeps, physics tick 857 of the
    # easy reference), moved 0.33 mm to the nearest pose 0.22 mm clear of the cage (it rests on a bar's inner face).
    # Windows: [centre_deg, chord width at the inner wall in mm].
    defaults = merge_parameters(HedgehogInCageEasyTask.defaults, {
        "yaw_deg": 40.,
        "windows": [[0., 43.], [72., 56.5], [144., 61.], [216., 61.], [288., 56.5]],
        "ball_pose": {"position_m": [-0.005976, -0.006835, -0.041739],
                      "quaternion_wxyz": [-0.453751, -0.662801, 0.586951, -0.101455]},
    })
    fixed_parameters = ("scale", "yaw_deg", "windows", "ball_pose")

    def template(self):
        """The CLASSIC template with this setting's windows (template units = mm / scale)."""
        template = deepcopy(load_template())
        windows, scale = np.asarray(self.config["windows"], dtype=float), float(self.config["scale"])
        template["template_id"] += "+window_change"
        template["cage"].update(window_count=len(windows), window_centers_deg=windows[:, 0].tolist(),
                                window_widths=(windows[:, 1] / scale).tolist())
        return template

    def build_assets(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "template.json"
        path.write_text(json.dumps(self.template(), indent=2) + "\n")
        layout = HedgehogLayout(load_template(path), float(self.config["scale"]))  # reuse its validation
        cage, cage_parts = cage_meshes(layout)
        ball, ball_parts = hedgehog_meshes(layout)
        c, u = layout.template["cage"], layout.unit
        finish = self.config["finish"]
        report = dict(template_id=layout.template["template_id"], scale=layout.scale,
                      geometry_status="window_change design from the CLASSIC reconstruction",
                      cage_geometry=dict(outer_radius_m=c["outer_diameter"] * u / 2, windows=self.config["windows"],
                                         collar_height_m=c["collar_height"] * u),
                      cage=write_body(folder, "cage", cage, cage_parts, (.30, .36, .42, 1), 2700, finish),
                      hedgehog=write_body(folder, "hedgehog", ball, ball_parts, (.80, .83, .86, 1), 2700, finish),
                      initial_pose=self.checked_ball_pose(cage_parts, ball_parts),
                      reference_route_verified=False, physics_qualified=False)
        (folder / "asset_report.json").write_text(json.dumps(report, indent=2) + "\n")
        return cage, ball, -c["height"] * u / 2, report

    def checked_ball_pose(self, cage_parts, ball_parts):
        """The configured pose, rejected if the ball would start inside the cage walls."""
        pose = self.config["ball_pose"]
        q = np.asarray(pose["quaternion_wxyz"], dtype=float)
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        transform[:3, 3] = pose["position_m"]
        pair = TriangleMeshPair((trimesh.util.concatenate(cage_parts), trimesh.util.concatenate(ball_parts)))
        clearance = pair.clearance((np.eye(4), transform))
        if clearance < 0:
            raise ValueError(f"{self.variation_id}: the ball pose intersects the cage")
        return dict(position_m=list(pose["position_m"]), quaternion_wxyz=q.tolist(),
                    clearance_to_cage_m=clearance, gravity_settled=True)
