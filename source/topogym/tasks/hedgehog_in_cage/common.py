"""Hedgehog cage/ball declarations; control capabilities belong to the registry."""
from pathlib import Path
from topogym.tasks.placement import PlacementGoalMixin

import numpy as np
from scipy.spatial.transform import Rotation

from topogym.assets.geometry.hedgehog_in_cage import generate_assets, cage_meshes, hedgehog_meshes
from topogym.tasks.base import EntitySpec, RigidTaskDefinition, TaskSpec
from ..scene_presets import defaults, record_preset
from topogym.assets.scene_visuals import surface


def recommended_grasp_assist(variation_id, seed=0, *, task_parameters=None):
    """Allow ball-surface regrasp without a pickup file or cage binding."""
    from topogym.config.registry import task_definition
    from topogym.utils.grasp import GraspAssistConfig, rigid_grasp_region
    task = task_definition("hedgehog_in_cage", variation_id, seed)(task_parameters, seed=seed)
    if hasattr(task, "grasp_targets"):      # a setting with its own grasp regions (owl.py: the owl and the rod)
        targets = task.grasp_targets()
        return GraspAssistConfig("contact_weld_v1", targets,
                                 max_tcp_distance_m=max(.06, max(t.radius_m for t in targets)+.03))
    if task.asset_resource:
        import io
        import zipfile
        import trimesh
        from topogym.assets.library import resource_path
        with zipfile.ZipFile(resource_path(task.asset_resource)) as bundle:
            ball = trimesh.load(io.BytesIO(bundle.read("hedgehog.obj")), file_type="obj", process=False)
    else:
        from topogym.assets.geometry.hedgehog_in_cage import HedgehogLayout, load_template
        ball, _ = hedgehog_meshes(HedgehogLayout(load_template(), scale=float(task.config["scale"])))
    target = rigid_grasp_region("hedgehog_ball", "hedgehog", ball.vertices)
    # A surface contact may lie far from the ball's origin (at a spike). Keep
    # the TCP gate consistent with the region plus the finger-to-TCP offset.
    return GraspAssistConfig("contact_weld_v1", (target,),
                            max_tcp_distance_m=max(.06, target.radius_m+.03))


# The finishes of the family's parts at appearance seeds 1 and 2 (scene_variation's hedgehog_in_cage axes; the user,
# 2026-10-27: a seed changes the colour and the material of every part no text names): linear RGB albedo, metalness,
# roughness. A part at its released finish (the axis default: the woods' textures, the Cast Cage's black and copper,
# the coated mount) is not refinished and keeps its files byte for byte. No colour here is near the green goal region,
# and each stands apart from every table top of topogym.world.look (CIEDE2000 >= 15).
FINISHES = dict(
    black_anodised=dict(color=(.03, .03, .035, 1.), metallic=.5, roughness=.4),
    blue_powder_coat=dict(color=(.02, .04, .30, 1.), metallic=.3, roughness=.45),
    crimson_enamel=dict(color=(.40, .03, .03, 1.), metallic=.3, roughness=.4),
    gunmetal=dict(color=(.10, .10, .11, 1.), metallic=1., roughness=.35),
    cobalt_anodised=dict(color=(.04, .10, .45, 1.), metallic=.7, roughness=.3),
    red_lacquer=dict(color=(.55, .05, .04, 1.), metallic=0., roughness=.25),
    gold_plated=dict(color=(.80, .55, .03, 1.), metallic=1., roughness=.15),
    black_nickel=dict(color=(.035, .035, .04, 1.), metallic=1., roughness=.25),
    black_oxide=dict(color=(.035, .035, .04, 1.), metallic=.6, roughness=.5),
    cobalt_paint=dict(color=(.04, .10, .45, 1.), metallic=.3, roughness=.4),
    black_iron=dict(color=(.035, .035, .04, 1.), metallic=.4, roughness=.55),
    yellow_paint=dict(color=(.80, .55, .03, 1.), metallic=.3, roughness=.35),
    black_plastic=dict(color=(.03, .03, .035, 1.), metallic=0., roughness=.6),
    orange_plastic=dict(color=(.80, .25, .03, 1.), metallic=0., roughness=.5),
)


def finish(name):
    """A FINISHES entry (a fresh dict, as scene_visuals.surface)."""
    if name not in FINISHES:
        raise ValueError(f"Unknown hedgehog_in_cage finish: {name!r}")
    return dict(FINISHES[name])


def refinish_body(folder, name, look):
    """Paint a written body (write_body's or write_rod's <name>.glb and <name>.urdf) in a FINISHES look: the GLB's base
    colour, metalness and roughness (its wood texture dropped) and the URDF visual's colour, which Nyx draws."""
    from topogym.assets.scene_visuals import scale_glb_color, scale_urdf_color
    scale_glb_color(folder / f"{name}.glb", 1., color=look["color"], metallic=look["metallic"],
                    roughness=look["roughness"], texture=False)
    scale_urdf_color(folder / f"{name}.urdf", 1., color=look["color"])


class HedgehogInCageTask(PlacementGoalMixin, RigidTaskDefinition):
    task_id = "hedgehog_in_cage"
    placement_description = 'the hedgehog ball including every spike'
    goal_side = (1, -1)  # -y, the nominal scene's side; pose variation cannot flip it
    defaults = defaults(task_id, scale=3., center_xy_m=[-.1356786, .05], yaw_deg=0.,
                    mount_height_m=.04, mount_width_m=.25, stable_duration_s=2.,
                    minimum_separation_m=.01, wood_tint=1., mount_finish="coated_metal")
    hash_when_changed = ("wood_tint", "mount_finish")
    # config key -> a body written by build_assets that the key's finish paints (refinish_body); none here
    finish_bodies = {}

    asset_resource = None

    terminal_revision = "hedgehog_placement_v4_grasp_contact_2mm"

    def terminal_metadata(self):
        from .qualification import placement_qualification
        metadata = super().terminal_metadata()
        qualification = placement_qualification(self)
        if qualification is not None:
            # A historical match describes evidence scope, not current-run
            # calibration/reachability or permission to report success.
            metadata.update(qualification=dict(qualification))
        return metadata

    def build_assets(self, folder):
        if not self.asset_resource:
            layout, report = generate_assets(folder, float(self.config["scale"]), finish=self.config["finish"])
            cage, _ = cage_meshes(layout)
            ball, _ = hedgehog_meshes(layout)
            return cage, ball, -layout.template["cage"]["height"]*layout.unit/2, report
        import json
        import trimesh
        from topogym.assets.variation_resources import read_json, unpack_assets
        unpack_assets(self.asset_resource, folder)
        prototype = read_json(self.prototype_resource)
        report = json.loads((folder/"asset_report.json").read_text())
        report.update(initial_pose=self.initial_ball_pose,
                      source_resource=self.asset_resource, spike_count=prototype["spike_count"],
                      historical_certificate_transferred=False)
        cage = trimesh.load(folder/"cage.obj", force="mesh", process=False)
        ball = trimesh.load(folder/"hedgehog.obj", force="mesh", process=False)
        return cage, ball, prototype["cage_bottom_m"], report

    def declare(self, asset_dir, table_height_m):
        self.table_height_m = float(table_height_m)
        cfg, folder = self.config, Path(asset_dir)
        preset = record_preset(self, folder)
        cage, ball, cage_bottom, report = self.build_assets(folder)
        if cfg["wood_tint"] != 1.:
            # Appearance only: darken the baked wood textures of cage and ball. Nyx draws a URDF visual with the
            # URDF's own <material><color> (here "1 1 1 1"), not the GLB's baseColorFactor, so the wood bodies'
            # URDF colours scale too.
            from topogym.assets.scene_visuals import scale_glb_color, scale_urdf_color
            for path in sorted(folder.glob("*.glb")):
                scale_glb_color(path, cfg["wood_tint"], materials=("wood",))
            for path in sorted(folder.glob("*.urdf")):
                if '<material name="wood"' in path.read_text():
                    scale_urdf_color(path, cfg["wood_tint"])
        for key, body in self.finish_bodies.items():
            if cfg[key] != self.defaults[key]:
                refinish_body(folder, body, finish(cfg[key]))
        xy = np.asarray(cfg["center_xy_m"], dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("Invalid Hedgehog center_xy_m")
        pad = float(cfg["mount_height_m"])
        if pad <= 0 or not np.isfinite(pad) or cfg["mount_width_m"] <= 0:
            raise ValueError("Invalid cage mount dimensions")
        origin = np.r_[xy, self.table_height_m+pad-cage_bottom]
        rotation = Rotation.from_euler("z", float(cfg["yaw_deg"]), degrees=True)
        initial = report["initial_pose"]
        ball_rotation = rotation*Rotation.from_quat(np.asarray(initial["quaternion_wxyz"])[[1, 2, 3, 0]])
        position = origin+rotation.apply(initial["position_m"])
        self.local_vertices = {"hedgehog_cage": cage.vertices, "hedgehog_ball": ball.vertices}
        options = {"convexify": True, "decimate": False, "merge_fixed_links": False}
        entities = (
            EntitySpec("hedgehog_mount", "box", None,
                (*xy, self.table_height_m+pad/2), fixed=True, surface=self.mount_surface(),
                morph_options={"size": (cfg["mount_width_m"], cfg["mount_width_m"], pad)}),
            EntitySpec("hedgehog_cage", "urdf", folder/"cage.urdf", tuple(origin),
                tuple(rotation.as_quat()[[3, 0, 1, 2]]), fixed=True,
                material={"friction": .3}, morph_options={**options,
                    "convexify": getattr(self, "cage_convexify", True)}),
            # align False: the ball's free joint at its URDF origin, as in the MuJoCo scene (Genesis would reframe it
            # at its centre of mass, 2.3 mm away, and the render bridge would draw it there).
            EntitySpec("hedgehog_ball", "urdf", folder/"hedgehog.urdf", tuple(position),
                tuple(ball_rotation.as_quat()[[3, 0, 1, 2]]), material={"friction": .3},
                morph_options={**options, "align": False}))
        # Placement rule caches the union of collision and visible geometry.
        from topogym.evaluation.task_metrics import rigid_asset_vertices
        self.placement_vertices = {e.name: rigid_asset_vertices(e) for e in entities}
        self.placement_targets = ('hedgehog_ball',)
        self.placement_remainder = tuple(e.name for e in entities if e.name not in self.placement_targets)
        return TaskSpec(entities, {"rigid": {"enable_self_collision": True, "iterations": 100,
            "max_collision_pairs": 8192, "constraint_timeconst": .00125}}, dict(**preset, task_id=self.task_id, instance_id=self.instance_id,
            asset_report=report, required_control_review="cage reachability and suitable gripper"))

    def mount_surface(self):
        name = self.config["mount_finish"]
        return surface("coated_metal") if name == self.defaults["mount_finish"] else finish(name)

    def evaluate_goal_region(self, state, region):
        """Measure the fixed placement rule; physical qualification is separate."""
        from topogym.evaluation.task_metrics import placement_candidate
        result = placement_candidate(state, region, self.placement_vertices,
            self.placement_targets, self.placement_remainder)
        return result

    def goal_region_observation(self, region):
        return region.observation(self.placement_targets, self.placement_remainder)

    def evaluate_state(self, state):
        return self.evaluate_placement(state)
