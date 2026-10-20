"""The redesign tiers as tasks: the family's own BuzzWireTask (its predicate, terminal revision, tolerances and
contract unchanged) on a tier's Design. Two scene additions, both declared geometry with their own contact roles:

- rest sleeves (`rest_sleeve`): short insulated sleeves on the wire, as on real buzz-wire toys. The game's rule is
  metal on metal (`wand_metal` on `wire_metal`, common.py); a sleeve is not metal, so the ring may rest on it. They are
  the only places where the wand can be put down mid-route.
- wooden obstacles (`wood_stop`, the role of the toy's other wooden parts), fixed to the deck.

Nothing else changes: the wire is the family's SpatialWire through the tier's controls, the wand / deck / endpoint
mounts / start pose / finish pose come from topogym.assets.geometry.buzz_wire_3d."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from topogym.tasks.buzz_wire_3d.common import BuzzWireTask
from topogym.tasks.base import merge_parameters, TaskSpec, EntitySpec
from topogym.tasks.scene_presets import record_preset
from .design import Design, sleeve_segments

SLEEVE_ROLE = "rest_sleeve"
OBSTACLE_ROLE = "wood_stop"


def _vec(v):
    return " ".join(f"{float(x):.12g}" for x in np.ravel(v))


def _add_collision(link, role, index, geometry: tuple, xyz, rpy=(0., 0., 0.)):
    node = ET.SubElement(link, "collision", name=f"terminal__{role}__{index}")
    ET.SubElement(node, "origin", xyz=_vec(xyz), rpy=_vec(rpy))
    g = ET.SubElement(node, "geometry")
    ET.SubElement(g, geometry[0], **geometry[1])
    return node


def add_sleeves(folder: Path, design: Design, rail) -> list:
    """Append the sleeves to wire.urdf as exact capsules (a cylinder + two end spheres per rail segment, the family's
    own capsule convention), and their visual (rest_sleeves.glb, matte black rubber). -> segments."""
    from topogym.assets.geometry.buzz_wire_3d import replace_visuals, tube_mesh
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals
    segs = sleeve_segments(rail, design.sleeves)
    if not segs:
        return []
    path = folder / "wire.urdf"
    tree = ET.parse(path)
    link = tree.find("link")
    r = design.sleeve_radius_m
    k = len(link.findall("collision"))
    for a, b in segs:
        a, b = np.asarray(a), np.asarray(b)
        length = float(np.linalg.norm(b - a))
        z = (b - a) / length
        rot = Rotation.align_vectors([z], [[0, 0, 1]])[0]
        _add_collision(link, SLEEVE_ROLE, k, ("cylinder", dict(radius=f"{r:.12g}", length=f"{length:.12g}")),
                       (a + b) / 2, rot.as_euler("xyz")); k += 1
        _add_collision(link, SLEEVE_ROLE, k, ("sphere", dict(radius=f"{r:.12g}")), a); k += 1
        _add_collision(link, SLEEVE_ROLE, k, ("sphere", dict(radius=f"{r:.12g}")), b); k += 1
    ET.indent(tree)
    tree.write(path, encoding="unicode", xml_declaration=True)
    meshes = []
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(rail, axis=0), axis=1))]
    for s0, s1 in design.sleeves:
        idx = np.flatnonzero((arc >= s0) & (arc <= s1))
        pts = rail[idx[0]:idx[-1] + 1]
        tube = tube_mesh(pts, r, sides=20)
        caps = [trimesh.creation.icosphere(subdivisions=2, radius=r).apply_translation(p) for p in (pts[0], pts[-1])]
        meshes.append(trimesh.util.concatenate([tube, *caps]))
    mesh = trimesh.util.concatenate(meshes)
    mesh.visual = TextureVisuals(uv=np.zeros((len(mesh.vertices), 2)), material=PBRMaterial(
        name="rest_sleeve_rubber", baseColorFactor=(0.08, 0.08, 0.09, 1), metallicFactor=0.0, roughnessFactor=0.85))
    mesh.export(folder / "rest_sleeves.glb")
    replace_visuals(path, ["rail_metal.glb", "posts_wood.glb", "rest_sleeves.glb"])
    return segs


def obstacle_mesh(ob):
    if ob["kind"] == "box":
        m = trimesh.creation.box(ob["size"])
    elif ob["kind"] == "cylinder":
        m = trimesh.creation.cylinder(radius=ob["radius"], height=ob["height"], sections=48)
    else:
        raise ValueError(ob["kind"])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", ob.get("euler_deg", (0, 0, 0)), degrees=True).as_matrix()
    T[:3, 3] = ob["pos"]
    m.apply_transform(T)
    return m


def add_obstacles(folder: Path, design: Design, layout) -> int:
    """Append wooden obstacles to base.urdf (collision primitives, role wood_stop) and their visual. The base link's
    origin is the old deck centre: z offset = base thickness / 2 above the toy's bottom centre (family convention)."""
    from topogym.assets.geometry.buzz_wire_3d import replace_visuals, toy_wood
    if not design.obstacles:
        return 0
    path = folder / "base.urdf"
    tree = ET.parse(path)
    link = tree.find("link")
    k = len(link.findall("collision"))
    dz = np.array([0, 0, layout.base_size_m[2] / 2])
    for ob in design.obstacles:
        rpy = Rotation.from_euler("xyz", ob.get("euler_deg", (0, 0, 0)), degrees=True).as_euler("xyz")
        if ob["kind"] == "box":
            geo = ("box", dict(size=_vec(ob["size"])))
        else:
            geo = ("cylinder", dict(radius=f"{ob['radius']:.12g}", length=f"{ob['height']:.12g}"))
        _add_collision(link, OBSTACLE_ROLE, k, geo, np.asarray(ob["pos"]) - dz, rpy); k += 1
    ET.indent(tree)
    tree.write(path, encoding="unicode", xml_declaration=True)
    mesh = trimesh.util.concatenate([obstacle_mesh(ob) for ob in design.obstacles])
    mesh.apply_translation(-dz)
    toy_wood(mesh, tile=max(layout.base_size_m) * 1.04).export(folder / "obstacles_wood.glb")
    replace_visuals(path, ["base_wood.glb", "obstacles_wood.glb"])
    return len(design.obstacles)


class BuzzWireRedesignTask(BuzzWireTask):
    """A redesign tier: subclasses set `design_factory` (a callable -> Design), `variation_id`, `difficulty`."""
    design_factory = None
    family_id = "redesign_2026_10"
    model_revision = "redesign_v0"
    variant_parameters = dict(linear_scale=1.0, geometry={}, base_elevation_m=.125)
    defaults = merge_parameters(BuzzWireTask.defaults, variant_parameters)
    fixed_parameters = ("linear_scale", "geometry", "base_elevation_m")

    # Patch 01 (main approved 10-11 for the redesign tiers): descriptive only, no rule changes
    INSULATED_NOTE = ("Black rubber sleeves on the wire are insulated: the ring may rest on them. The final bend and the "
                      "post at the finish are insulated as well.")
    instruction = BuzzWireTask.instruction + " " + INSULATED_NOTE

    def public_instruction(self) -> str:
        text = super().public_instruction()
        if self.INSULATED_NOTE in text:
            return text
        head, sep, rest = text.partition("\n\n")
        return head + " " + self.INSULATED_NOTE + (sep + rest if sep else "")

    def design(self) -> Design:
        return type(self).design_factory()

    def build_layout(self):
        return self.design().layout()

    def declare(self, asset_dir, table_height_m):
        from topogym.assets.geometry.buzz_wire_3d import collision_roles, generate_assets, endpoint_pose
        design = self.design()
        if not np.allclose(self.config["center_xy_m"], design.center_xy_m, atol=1e-12):
            raise ValueError("the tier's center_xy_m default must be its design's deck centre")
        self.table_height_m = float(table_height_m)
        preset = record_preset(self, asset_dir)
        layout = self.layout = design.layout()
        self.route, report = generate_assets(asset_dir, layout)
        folder = Path(asset_dir)
        # Appearance only, as BuzzWireTask.declare (which this declare replaces): the baked wood and the rail's colour.
        from topogym.assets.scene_visuals import scale_glb_color
        for path in sorted(folder.glob("*.glb")):
            if self.config["wood_tint"] != 1.:
                scale_glb_color(path, self.config["wood_tint"], materials=("wood",))
            if self.config["rail_rgb"] != self.defaults["rail_rgb"]:
                scale_glb_color(path, 1., materials=("rail_metal", "brushed_copper"),
                                color=[*self.config["rail_rgb"], 1.])
        sleeves = add_sleeves(folder, design, self.route.rail)
        obstacles = add_obstacles(folder, design, layout)
        report = dict(report, redesign=dict(design=design.name, tier=design.tier, sleeves=[list(x) for x in design.sleeves],
                                            sleeve_radius_m=design.sleeve_radius_m, sleeve_segments=len(sleeves),
                                            obstacles=design.obstacles, notes=design.notes))
        (folder / "geometry_report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
        files = sorted(p.name for p in folder.iterdir() if p.is_file() and p.name != "sha256.json")
        (folder / "sha256.json").write_text(json.dumps(
            {f: hashlib.sha256((folder / f).read_bytes()).hexdigest() for f in files}, indent=2) + "\n")
        self.origin = np.r_[self.config["center_xy_m"], table_height_m]
        self.rail = self.route.rail + self.origin
        position, rotation = endpoint_pose(self.route, 0)
        quaternion = rotation.as_quat()[[3, 0, 1, 2]]
        entities = (
            EntitySpec("wire_base", "urdf", folder / "base.urdf",
                       tuple(self.origin + [0, 0, layout.base_size_m[2] / 2]), fixed=True,
                       morph_options={"convexify": True, "decimate": False},
                       contact_roles=collision_roles(folder / "base.urdf")),
            EntitySpec("wire", "urdf", folder / "wire.urdf", tuple(self.origin), fixed=True,
                       contact_roles=collision_roles(folder / "wire.urdf")),
            EntitySpec("wand", "urdf", folder / "wand.urdf", tuple(self.origin + position), tuple(quaternion),
                       material={"friction": 0.3}, contact_roles=collision_roles(folder / "wand.urdf")),
        )
        return TaskSpec(
            entities,
            {"rigid": {"enable_self_collision": True, "multiplier_collision_broad_phase": 64,
                       "constraint_timeconst": 0.00125}},
            dict(**preset, linear_scale=float(self.config["linear_scale"]), supported_endpoints=True,
                 task_id=self.task_id, geometry_report=report, goal_evaluation_complete=self.goal_complete),
        )


    def finalize_spec(self, spec, *, asset_dir, scene_config):
        """A tier's robot_mount (x, y, yaw) sets the Panda's table-top mount through the scene config, before the scene
        is built, so the recording, its replay and the agent's task share it (the frozen instance keeps the scene)."""
        from dataclasses import replace
        from .design import mount_problems
        design = self.design()
        if design.robot_mount is not None:
            x, y, yaw = map(float, design.robot_mount)
            bad = mount_problems((x, y, yaw), design.center_xy_m, design.base_size_m)
            if bad:
                raise ValueError(f"illegal robot mount {design.robot_mount}: {bad}")
            mount = scene_config.env.robot_mount
            mount.pos = [x, y, float(mount.pos[2])]
            mount.euler = [float(mount.euler[0]), float(mount.euler[1]), yaw]
            spec = replace(spec, metadata={**spec.metadata, "robot_mount": dict(pos=[x, y, float(mount.pos[2])],
                                                                              euler_deg=[0.0, 0.0, yaw])})
        return super().finalize_spec(spec, asset_dir=asset_dir, scene_config=scene_config)


def tier_class(variation_id: str, difficulty: str, factory, *, model_revision="redesign_v0"):
    """A concrete tier task class (its defaults carry the design's deck centre)."""
    design = factory()
    return type(f"BuzzWireRedesign_{difficulty}", (BuzzWireRedesignTask,), dict(
        variation_id=variation_id, difficulty=difficulty, design_factory=staticmethod(factory),
        model_revision=model_revision,
        defaults=merge_parameters(BuzzWireRedesignTask.defaults, dict(center_xy_m=list(design.center_xy_m)))))
