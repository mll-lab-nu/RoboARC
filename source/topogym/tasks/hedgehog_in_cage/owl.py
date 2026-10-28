"""Bird in Cage: a wooden owl in a wooden cage on the family's mount; its two wing plates turn together on one hinge, and
a thin rod lies on the table (geometry: topogym.assets.geometry.hedgehog_owl).

An experimental variation of the family (difficulty None, as window_change); the tiers are unchanged. The goal is the
family's placement rule with the owl as the target: every part of the owl (body, beak, legs, tail, both wing plates)
inside the goal volume, the cage and its mount on the system side, the owl quiet for stable_duration_s. Where the rod
ends up does not matter (neither target nor remainder).

The owl starts standing inside the cage on its legs and tail, its back to the widest gap (between the two bars at
+-41 deg of the cage frame), the tail resting on the floor disc's rim, the wings'
straight edges along the body (wing angle 4.72 rad). This pose settled under gravity. The cage stands on the hedgehog
mount at the family's centre, turned `yaw_deg` about the vertical (0: the widest gap faces world +x, away from the
robot; the default 315 turns it toward +x -y). The rod lies flat on the table beside the mount, on the side away from the goal region, its handle box down.
"""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from topogym.assets.geometry import hedgehog_owl as geometry
from topogym.tasks.base import EntitySpec, TaskSpec, merge_parameters
from ..scene_presets import record_preset
from .common import HedgehogInCageTask, finish

# The owl at appearance seeds 1 and 2 (owl_finish): the texts call it wooden, so it stays wood and changes its wood
# (the user, 2026-10-27): oiled wenge (dark) or raw beech (pale), the released woods' colours (body and tail, wings,
# pegs: beak, legs and eyes) in that order, the pegs in a contrasting wood. Body and wings differ from the released
# and from each other by CIEDE2000 >= 20. The rod (rod_finish): its shaft's and its handle's common.FINISHES.
OWL_FINISHES = dict(wenge=dict(colors=((.10, .06, .035, 1.), (.14, .085, .05, 1.), (.80, .66, .42, 1.)), metallic=0.,
                               roughness=.45),
                    beech=dict(colors=((.80, .62, .42, 1.), (.94, .79, .58, 1.), (.30, .18, .10, 1.)), metallic=0.,
                               roughness=.7))
ROD_FINISHES = dict(yellow=("yellow_paint", "black_plastic"), cobalt=("cobalt_anodised", "orange_plastic"))


def paint_visual_geoms(path, looks):
    """Give the visual-only geoms of an MJCF file (the drawn twins; the colliding geoms keep their bytes) a material:
    looks = {released rgba text: (rgba, metallic, roughness)}. Nyx draws a geom's material's colour, metalness and
    roughness (renderer.pbr_materials) instead of its rgba."""
    import re
    names = {}

    def paint(match):
        geom = match.group(0)
        rgba = re.search(r'rgba="([^"]+)"', geom)
        if 'contype="0" conaffinity="0" mass="0"' not in geom or rgba is None or rgba.group(1) not in looks:
            return geom
        names.setdefault(rgba.group(1), f"look_{len(names)}")
        return geom.replace(rgba.group(0), f'material="{names[rgba.group(1)]}"')
    text = re.sub(r"<geom\b[^>]*/>", paint, path.read_text())
    # appended right before </asset>, no whitespace: the file without its materials and visual geoms (what the physics
    # reads) keeps its bytes
    materials = "".join(f'<material name="{name}" rgba="{geometry._rgba(looks[key][0])}" metallic="{looks[key][1]:g}" '
                        f'roughness="{looks[key][2]:g}"/>' for key, name in names.items())
    path.write_text(text.replace("</asset>", materials + "</asset>", 1))


class HedgehogOwlTask(HedgehogInCageTask):
    """Owl, cage and rod; the owl's start pose in the cage frame is fixed by the design."""

    variation_id = "hedgehog_in_cage_owl"
    difficulty = None  # Explicit experimental entry; the tiers are unchanged.
    model_revision = "owl_v6_rung37_v1"
    family_id = "bird_in_cage_owl"
    placement_description = "the whole owl, including both wings, the tail, the legs and the beak"
    placement_target_description = placement_description
    placement_remainder_description = "the cage and its mount"
    placement_scene_note = ("A thin rod with a box handle lies on the table; you may use it as a tool. It is a task piece: "
                            "if it falls off the table (more than 100 mm below the tabletop), the episode ends in failure.")
    defaults = merge_parameters(HedgehogInCageTask.defaults, {
        "scale": geometry.SCALE,
        "yaw_deg": 315.,
        # In the cage frame (origin at the floor's centre, z up, the widest gap on +x); wing_rad is the hinge angle.
        "owl_pose": {"position_m": [-0.006935, 0.0, 0.075362],
                     "quaternion_wxyz": [0.0, 0.025658, -0.0, 0.999671], "wing_rad": 4.720585},
        # World xy of the handle box's centre; yaw_deg: the direction from the handle to the tip (0: world +x).
        "rod_pose": {"handle_xy_m": [-0.31, 0.24], "yaw_deg": 0.},
        # the finishes at appearance seeds 1 and 2: the cage's (common.FINISHES), the owl's (OWL_FINISHES) and the rod's
        # (ROD_FINISHES); the defaults name the released light wood cage, wooden owl and red rod
        "birdcage_finish": "light_wood", "owl_finish": "wood", "rod_finish": "red",
    })
    hash_when_changed = (*HedgehogInCageTask.hash_when_changed, "birdcage_finish", "owl_finish", "rod_finish")
    fixed_parameters = ("scale", "owl_pose")

    def owl_rotation(self, rotation):
        q = np.asarray(self.config["owl_pose"]["quaternion_wxyz"], dtype=float)
        return rotation * Rotation.from_quat(q[[1, 2, 3, 0]] / np.linalg.norm(q))

    def declare(self, asset_dir, table_height_m):
        from topogym.utils.grasp import TASK_CONTACT_PENETRATION_M
        self.table_height_m = float(table_height_m)
        cfg, folder = self.config, Path(asset_dir)
        preset = record_preset(self, folder)
        pose = cfg["owl_pose"]
        report = geometry.write_assets(folder, pose)
        if cfg["wood_tint"] != 1.:
            # Appearance only (the family's wood_tint, which this declare does not inherit): the cage's wood GLB (its
            # URDF names no colour, so Nyx draws the GLB's baseColorFactor) and the owl's wooden visual-only geoms (the
            # drawn twins; the colliding geoms in group 3 keep their bytes).
            import re
            from topogym.assets.scene_visuals import scale_glb_color
            tint = float(cfg["wood_tint"])
            scale_glb_color(folder / "owl_cage_meshes" / "visual.glb", tint, materials=("wood",))
            wood = {geometry._rgba(c) for c in (geometry.WOOD_D, geometry.WOOD_W)}

            def darker(match):
                if match.group(1) not in wood:
                    return match.group(0)
                r, g, b, a = (float(v) for v in match.group(1).split())
                return f'rgba="{geometry._rgba((min(1., r*tint), min(1., g*tint), min(1., b*tint), a))}"'

            def visual(match):
                geom = match.group(0)
                return re.sub(r'rgba="([^"]+)"', darker, geom) if 'contype="0" conaffinity="0" mass="0"' in geom else geom
            owl = folder / "owl.xml"
            owl.write_text(re.sub(r"<geom\b[^>]*/>", visual, owl.read_text()))
        if cfg["birdcage_finish"] != self.defaults["birdcage_finish"]:
            from topogym.assets.scene_visuals import scale_glb_color
            look = finish(cfg["birdcage_finish"])
            scale_glb_color(folder / "owl_cage_meshes" / "visual.glb", 1., color=look["color"], metallic=look["metallic"],
                            roughness=look["roughness"], texture=False)
        if cfg["owl_finish"] != self.defaults["owl_finish"]:
            look = OWL_FINISHES[cfg["owl_finish"]]
            paint_visual_geoms(folder / "owl.xml", {geometry._rgba(c): (wood, look["metallic"], look["roughness"])
                                                    for c, wood in zip((geometry.WOOD_D, geometry.WOOD_W, geometry.PEG),
                                                                       look["colors"])})
        if cfg["rod_finish"] != self.defaults["rod_finish"]:
            shaft, handle = (finish(name) for name in ROD_FINISHES[cfg["rod_finish"]])
            paint_visual_geoms(folder / "rod.xml", {geometry._rgba(c): (look["color"], look["metallic"], look["roughness"])
                                                    for c, look in ((geometry.ROD_RGBA, shaft), (geometry.HANDLE_RGBA, handle))})
        if report["initial_pose"]["clearance_to_cage_m"] < -TASK_CONTACT_PENETRATION_M:
            raise ValueError(f"{self.variation_id}: the owl starts deeper in the cage than the task allows")
        xy = np.asarray(cfg["center_xy_m"], dtype=float)
        pad = float(cfg["mount_height_m"])
        if xy.shape != (2,) or not np.isfinite(xy).all() or pad <= 0 or cfg["mount_width_m"] <= 0:
            raise ValueError("Invalid owl cage placement")
        # The cage frame's origin is the floor disc's top face; the disc stands on the mount.
        origin = np.r_[xy, self.table_height_m + pad + geometry.DISC_T * geometry.SCALE * geometry.MM]
        rotation = Rotation.from_euler("z", float(cfg["yaw_deg"]), degrees=True)
        owl_position = origin + rotation.apply(pose["position_m"])
        owl_rotation = self.owl_rotation(rotation)
        rod = cfg["rod_pose"]
        rod_xy = np.asarray(rod["handle_xy_m"], dtype=float)
        if rod_xy.shape != (2,) or not np.isfinite(rod_xy).all():
            raise ValueError("Invalid rod handle_xy_m")
        rod_position = np.r_[rod_xy, self.table_height_m + geometry.HANDLE_HALF_M[2]]   # the handle box on the table
        rod_rotation = Rotation.from_euler("z", float(rod["yaw_deg"]), degrees=True)
        wxyz = lambda r: tuple(r.as_quat()[[3, 0, 1, 2]])
        entities = (
            EntitySpec("hedgehog_mount", "box", None, (*xy, self.table_height_m + pad / 2), fixed=True,
                       surface=self.mount_surface(), morph_options={"size": (cfg["mount_width_m"], cfg["mount_width_m"], pad)}),
            EntitySpec("owl_cage", "urdf", folder / "owl_cage.urdf", tuple(origin), wxyz(rotation), fixed=True,
                       material={"friction": .4},
                       morph_options={"convexify": True, "decimate": False, "merge_fixed_links": False}),
            EntitySpec("owl", "mjcf", folder / "owl.xml", tuple(owl_position), wxyz(owl_rotation),
                       initial_joint_positions={"wing": float(pose["wing_rad"])}),
            # align False: the rod's free joint at its handle's centre, as in the MuJoCo scene (Genesis would reframe this
            # lone free body at its centre of mass, 27 mm along the shaft, and the render bridge would draw it there). The
            # owl needs none: Genesis does not reframe a free root with a jointed child (its wings).
            EntitySpec("rod", "mjcf", folder / "rod.xml", tuple(rod_position), wxyz(rod_rotation),
                       morph_options={"align": False}))
        links = geometry.owl_link_points()
        # Per link, in the link frames: the placement rule reads the owl's links where they are (the wings turn).
        self.owl_link_vertices = links
        owl_start = np.vstack([links["body"], geometry.wings_in_owl_frame(links["wings"], pose["wing_rad"])])
        from topogym.evaluation.task_metrics import rigid_asset_vertices
        self.local_vertices = {"owl_cage": geometry.cage_points(), "owl": links["body"], "rod": geometry.rod_points()}
        self.placement_vertices = {"hedgehog_mount": rigid_asset_vertices(entities[0]), "owl_cage": geometry.cage_points(),
                                   "owl": owl_start}
        self.placement_targets = ("owl",)
        self.placement_remainder = ("hedgehog_mount", "owl_cage")
        return TaskSpec(entities, {"rigid": {"enable_self_collision": True, "iterations": 100,
            "max_collision_pairs": 8192, "constraint_timeconst": .00125}}, dict(**preset, task_id=self.task_id,
            instance_id=self.instance_id, asset_report=report, required_control_review="cage reachability and the rod as a tool"))

    def grasp_targets(self):
        """Assisted-grasp regions: the owl's body link (beak, legs and tail included) and the rod."""
        from topogym.utils.grasp import rigid_grasp_region
        return (rigid_grasp_region("owl", "body", geometry.owl_link_points()["body"]),
                rigid_grasp_region("rod", "rod", geometry.rod_points()))

    def evaluate_goal_region(self, state, region):
        """The family's placement measurement with the owl read per link: the body link and the wings link each at their
        own pose, every owl link quiet."""
        from topogym.evaluation.task_metrics import body_points, quiet_body
        from topogym.utils.geometry.regions import placement_measurements
        links = state["articulations"]["owl"]["links"]
        target = {"owl": np.vstack([body_points(links[name], points) for name, points in self.owl_link_vertices.items()])}
        remainder = {name: body_points(state["bodies"][name], self.placement_vertices[name]) for name in self.placement_remainder}
        quiet = all(quiet_body(links[name]) for name in self.owl_link_vertices)
        metrics = placement_measurements(region, target, remainder)
        metrics.update(quiet=bool(quiet), connections_intact=True, evaluation_status="available", holding_allowed=True)
        reached = metrics["all_target_inside"] and metrics["remainder_on_system_side"] and quiet
        return bool(reached), None, metrics
