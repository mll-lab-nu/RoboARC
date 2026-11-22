#!/usr/bin/env python3
"""Appearance seeds of the fixed-scene bases (scene_variation.appearance) change only visual geoms and meshes: declared
specs and the physics content of the generated files stay those of seed 0; no physics runs. Every release base's seeds 1
and 2 differ, the dim one not too dark, and none keeps seed 0's table frame; a blanket look keeps the frozen birth's
scene identity, and its texture is an asset file; the goal marker is 5 mm wide, its inner edge the containment limit."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]          # the repository; the simulator source is under source/
sys.path[:0] = [str(ROOT / "source")]

from topogym.config.registry import task_definition
from topogym.tasks.scene_variation import appearance, axes

BASES = (("bed_bolt", "bed_bolt_crossbar"), ("bed_bolt", "bed_bolt_slotted"), ("caliper_sorting", "caliper_sorting_vernier"),
         ("clock_assembling", "clock_assembling_gear_train"), ("battery_replacement", "battery_replacement_xbox_aa"),
         ("interlocked_claws", "interlocked_claws_cast_trinity"), ("horseshoe_ring", "horseshoe_ring_easy"),
         ("lock_picking", "lock_picking_very_easy"), ("wire_splicing", "wire_splicing_eye_wrap"),
         ("bed_bolt", "bed_bolt_torx"), ("bed_bolt", "bed_bolt_square"), ("bed_bolt", "bed_bolt_phillips"),
         ("hedgehog_in_cage", "hedgehog_in_cage_easier"), ("hedgehog_in_cage", "hedgehog_in_cage_easy"),
         ("hedgehog_in_cage", "hedgehog_in_cage_cast_cage"), ("hedgehog_in_cage", "hedgehog_in_cage_owl"),
         ("hedgehog_in_cage", "hedgehog_in_cage_window_change"),
         ("buzz_wire_3d", "buzz_wire_3d_easier"), ("buzz_wire_3d", "buzz_wire_3d_medium"),
         ("buzz_wire_3d", "buzz_wire_3d_rd_very_easy"), ("buzz_wire_3d", "buzz_wire_3d_rd_easier"),
         ("buzz_wire_3d", "buzz_wire_3d_rd_hard"),
         ("cast_coil", "cast_coil_medium"), ("cast_coil", "cast_coil_hard"), ("cast_coil", "cast_coil_v2_very_easy"),
         ("cast_coil", "cast_coil_v2_easier"), ("cast_coil", "cast_coil_v2_easy"),
         ("cast_keyhole", "cast_keyhole_easier"), ("cast_keyhole", "cast_keyhole_easy"), ("cast_keyhole", "cast_keyhole_medium"),
         ("cast_keyhole", "cast_keyhole_hard"), ("cast_keyhole", "cast_keyhole_double_maze"),
         ("interlocked_claws", "interlocked_claws_easier"), ("interlocked_claws", "interlocked_claws_hard"),
         ("interlocked_claws", "interlocked_claws_rd_easy"), ("interlocked_claws", "interlocked_claws_rd_medium"),
         ("interlocked_claws", "interlocked_claws_rd_hard"), ("interlocked_claws", "interlocked_claws_cast_abc"),
         ("interlocked_claws", "interlocked_claws_rotor"),
         ("horseshoe_ring", "horseshoe_ring_very_easy"), ("horseshoe_ring", "horseshoe_ring_easier"),
         ("horseshoe_ring", "horseshoe_ring_medium"), ("horseshoe_ring", "horseshoe_ring_hard"),
         ("lock_picking", "lock_picking_easy"), ("lock_picking", "lock_picking_medium"), ("lock_picking", "lock_picking_hard"),
         ("rope_threading", "rope_threading_rd_very_easy"), ("rope_threading", "rope_threading_rd_easier"),
         ("rope_threading", "rope_threading_rd_easy"), ("rope_threading", "rope_threading_rd_medium"),
         ("rope_threading", "rope_threading_rd_hard"),
         ("slotted_post_ring", "slotted_post_ring_very_easy"), ("slotted_post_ring", "slotted_post_ring_easier"),
         ("slotted_post_ring", "slotted_post_ring_easy_chain"), ("slotted_post_ring", "slotted_post_ring_medium_chain"),
         ("slotted_post_ring", "slotted_post_ring_hard_chain"),
         ("orbits", "orbits_very_easy"), ("orbits", "orbits_easier"),
         ("knot_untying", "knot_untying_easier"), ("knot_untying", "knot_untying_medium"), ("knot_untying", "knot_untying_hard"))


def release_bases():
    """(base, task, variation) of every base in tools/release/mujoco_release.json (release_rebuild.bases_of's names)."""
    spec = json.loads((ROOT / "tools/release/mujoco_release.json").read_text())
    for f, r in spec["families"].items():
        for t in r.get("tiers", []):
            yield f"{f}-{t}", f, r.get("variations", {}).get(t, f"{f}_{t}")
        for name, n in r.get("new_bases", {}).items():
            yield name, n.get("task", f), n["variation"]


def visual_geom(element):
    """An MJCF geom that neither collides nor carries mass: mass 0 (mjcf_entity.merge_mjcf_entity drops it), or no mass
    and density 0 (orbits' wooden handle; MuJoCo's default density is 1000)."""
    if element.tag != "geom" or int(element.get("contype", "1")) or int(element.get("conaffinity", "1")):
        return False
    mass = element.get("mass")
    return float(mass) == 0. if mass is not None else float(element.get("density", "1000")) == 0.


def physics(path, folder):
    """A generated file's content as the physics reads it: MJCF without its visual geoms (visual_geom), the meshes only
    they use and an <asset> these leave empty (a look may give materials to an MJCF entity that has no assets); URDF
    without <visual>; OBJ as it is (physics_view() leaves out those no physics element reads); images and GLBs none."""
    data = path.read_bytes().replace(str(folder).encode(), b"@DIR@")
    if path.suffix == ".obj":
        return data
    if path.suffix not in (".xml", ".urdf"):
        return None
    root = ET.fromstring(data)
    for parent in root.iter():
        for child in list(parent):
            if visual_geom(child) or child.tag in ("visual", "material", "texture"):
                parent.remove(child)
    used = {g.get("mesh") for g in root.iter("geom")}
    for parent in list(root.iter()):
        for asset in [a for a in parent if a.tag == "asset"]:
            for mesh in [m for m in asset if m.tag == "mesh" and m.get("name") not in used]:
                asset.remove(mesh)
            if not len(asset):
                parent.remove(asset)
    return ET.tostring(root)


def physics_meshes(paths, entities):
    """The resolved paths of the meshes a physics element reads: a declared entity's file, a URDF <collision>'s mesh, an
    MJCF mesh that a geom other than a visual one uses."""
    read = {Path(e.asset_path).resolve() for e in entities if e.asset_path}
    for path in paths:
        if path.suffix not in (".xml", ".urdf"):
            continue
        root = ET.parse(path).getroot()
        if path.suffix == ".urdf":
            read |= {(path.parent / m.get("filename")).resolve() for c in root.iter("collision") for m in c.iter("mesh")}
            continue
        compiler = root.find("compiler")
        meshdir = path.parent / ("" if compiler is None else compiler.get("meshdir", compiler.get("assetdir", "")))
        files = {m.get("name") or Path(m.get("file")).stem: m.get("file") for m in root.iter("mesh") if m.get("file")}
        read |= {(meshdir / files[g.get("mesh")]).resolve() for g in root.iter("geom")
                 if g.get("mesh") in files and not visual_geom(g)}
    return read


def obj_geometry(path):
    """sha256 of an OBJ file without its material lines (mtllib, usemtl)."""
    return hashlib.sha256(b"".join(line for line in path.read_bytes().splitlines(keepends=True)
                                   if not line.startswith((b"mtllib", b"usemtl")))).hexdigest()


def physics_view(folder, entities):
    """The physics content of the files under ``folder`` by name (physics()). Files without physics content (GLBs,
    images) may be renamed: Horseshoe's ring GLB is named after its finish. So may an OBJ no physics element reads (a
    visual mesh; interlocked_claws' copy that carries a material), but not its geometry: the set of every OBJ's
    geometry stays, as the placement and separation rules read the visual meshes' vertices (rigid_asset_vertices)."""
    files = {str(p.relative_to(folder)): p for p in Path(folder).rglob("*") if p.is_file()}
    read = physics_meshes(files.values(), entities)
    view = {k: physics(p, folder) for k, p in files.items() if p.suffix != ".obj" or p.resolve() in read}
    view["OBJ geometry"] = sorted({obj_geometry(p) for p in files.values() if p.suffix == ".obj"})
    return {k: v for k, v in view.items() if v is not None}


def declare(cls, params, seed):
    """(the spec without the entities' surfaces, the instance id and visual records (metadata *_visual: Horseshoe's ring
    finish), physics content by file, full content by file)."""
    with tempfile.TemporaryDirectory() as folder:
        task = cls(params, seed=seed)
        spec = task.declare(Path(folder), .75)
        plain = repr(([{**vars(e), "surface": None} for e in spec.entities], spec.physics_options,
                      {k: v for k, v in spec.metadata.items() if k != "instance_id" and not k.endswith("_visual")}, spec.attachments,
                      spec.initial_state, spec.initial_robot_targets)).replace(folder, "@DIR@")
        files = {str(p.relative_to(folder)): p for p in Path(folder).rglob("*") if p.is_file()}
        return (plain, physics_view(folder, spec.entities),
                {k: hashlib.sha256(p.read_bytes().replace(folder.encode(), b"@DIR@")).hexdigest() for k, p in files.items()})


class AppearanceSeedTests(unittest.TestCase):
    def test_seed_zero_is_the_recorded_scene(self):
        for family, variation in BASES:
            self.assertEqual(appearance(task_definition(family, variation), 0), {})

    def test_consecutive_seeds_differ_on_every_choice_axis(self):
        for family, variation in BASES:
            cls = task_definition(family, variation)
            one, two = appearance(cls, 1), appearance(cls, 2)
            for axis in axes(cls):
                if axis["kind"] == "choice":
                    self.assertNotEqual(one[axis["path"]], two[axis["path"]], (variation, axis["path"]))
                    self.assertNotEqual(one[axis["path"]], cls.defaults[axis["path"]], (variation, axis["path"]))

    def test_every_release_base_has_two_different_looks(self):
        for base, task, variation in release_bases():
            cls = task_definition(task, variation)
            with self.subTest(base=base):
                self.assertEqual(appearance(cls, 0), {})
                self.assertTrue(appearance(cls, 1))
                self.assertNotEqual(appearance(cls, 1), appearance(cls, 2))

    def test_every_release_base_has_two_scene_looks_apart_from_seed_zero(self):
        """Seeds 1 and 2 differ in every part of the scene's look (topogym.world.look) and neither takes the released
        table, floor, key light or a colour temperature near the released one; the draw is a pure function."""
        from topogym.world.look import FLOOR_FINISHES, FRAME_FINISHES, TABLE_FINISHES, check_scene_look
        for base, task, variation in release_bases():
            cls = task_definition(task, variation)
            one, two = appearance(cls, 1)["scene_look"], appearance(cls, 2)["scene_look"]
            with self.subTest(base=base):
                self.assertEqual(appearance(cls, 1)["scene_look"], one)
                for look in (one, two):
                    self.assertEqual(check_scene_look(look), look)
                    self.assertNotEqual(look["table"], next(iter(TABLE_FINISHES)))
                    self.assertNotEqual(look["floor"], next(iter(FLOOR_FINISHES)))
                    self.assertNotEqual(look["lighting"]["key"], 0)
                    self.assertTrue(3600 <= look["lighting"]["cct_k"] <= 4300 or 8000 <= look["lighting"]["cct_k"] <= 10000)
                self.assertNotEqual(TABLE_FINISHES[one["table"]]["cls"], TABLE_FINISHES[two["table"]]["cls"])
                self.assertNotEqual(FLOOR_FINISHES[one["floor"]]["cls"], FLOOR_FINISHES[two["floor"]]["cls"])
                for look in (one, two):
                    self.assertNotEqual(TABLE_FINISHES[look["table"]]["cls"], "plastic")
                    self.assertNotEqual(look["frame"], next(iter(FRAME_FINISHES)))
                self.assertNotEqual(one["frame"], two["frame"])
                self.assertNotEqual(one["lighting"]["key"], two["lighting"]["key"])
                self.assertNotEqual(one["lighting"]["cct_k"] < 6000, two["lighting"]["cct_k"] < 6000)
                self.assertNotEqual(one["lighting"]["brightness"] < 1, two["lighting"]["brightness"] < 1)

    def test_seeds_one_and_two_refinish_the_table_frame(self):
        """Seed 0's frame is dark_wood, compact_lab's own finish of the table's rim and apron; every release base's seeds
        1 and 2 draw two other frame finishes."""
        from omegaconf import OmegaConf
        from topogym.world.look import FRAME_FINISHES
        lab = OmegaConf.load(ROOT / "source/topogym/config/profiles/presentation_configs/compact_lab.yaml")
        released = next(r for r in OmegaConf.to_container(lab.presentation.appearance) if r["pattern"] == "table_frame*")
        self.assertEqual(next(iter(FRAME_FINISHES)), "dark_wood")
        self.assertEqual({k: v for k, v in FRAME_FINISHES["dark_wood"].items() if k != "cls"},
                         {k: v for k, v in released.items() if k != "pattern"})
        for base, task, variation in release_bases():
            cls = task_definition(task, variation)
            frames = [appearance(cls, seed)["scene_look"]["frame"] for seed in (1, 2)]
            with self.subTest(base=base):
                self.assertNotIn("dark_wood", frames)
                self.assertNotEqual(frames[0], frames[1])

    def test_a_dim_seed_stays_bright_enough(self):
        """Of seeds 1 and 2 the dim one (brightness below 1) keeps at least its family's floor (DIM_FLOOR: bed_bolt 0.9,
        else 0.65) and, on a dark table top (lum below DARK_TOP_LUM), 0.85."""
        from topogym.world.look import DARK_TOP_LUM, DIM_FLOOR, TABLE_FINISHES
        seen = set()
        for base, task, variation in release_bases():
            cls = task_definition(task, variation)
            for seed in (1, 2):
                look = appearance(cls, seed)["scene_look"]
                brightness, dark = look["lighting"]["brightness"], TABLE_FINISHES[look["table"]]["lum"] < DARK_TOP_LUM
                if brightness >= 1:
                    continue
                with self.subTest(base=base, seed=seed):
                    self.assertGreaterEqual(brightness, DIM_FLOOR.get(task, .65))
                    if dark:
                        self.assertGreaterEqual(brightness, .85)
                seen |= {task, "dark top" if dark else "light top"}
        self.assertIn("bed_bolt", DIM_FLOOR)
        self.assertLessEqual({"bed_bolt", "dark top"}, seen)      # both rules were exercised

    def test_the_goal_marker_is_5_mm_wide_outside_the_containment_limit(self):
        """write_goal_region's dashes are 5 mm wide, grown outward: in the region's frame each dash's inner edge is the
        effective boundary half_size - boundary_margin (GoalRegion.contains' limit) and its outer edge 5 mm beyond; the
        marker glows (an emissive surface)."""
        import numpy as np
        import trimesh
        from topogym.assets.geometry.goal_region import write_goal_region
        from topogym.utils.geometry.regions import GoalRegion
        region = GoalRegion(center_xy_m=(.1, -.05), half_size_m=(.12, .09), table_height_m=.75, placement_height_m=.2,
                            separator_normal_xy=(1., 0.), separator_offset_m=-1., yaw_deg=30., boundary_margin_m=.004)
        limit = np.asarray(region.half_size_m) - region.boundary_margin_m
        with tempfile.TemporaryDirectory() as folder:
            entity = write_goal_region(folder, region)
            dashes = trimesh.load(entity.asset_path, process=False).split(only_watertight=False)
        self.assertGreater(len(dashes), 20)
        for dash in dashes:
            xy = np.abs((dash.vertices[:, :2] - region.center_xy_m) @ region.rotation)
            across = [k for k in (0, 1) if abs(xy[:, k].min() - limit[k]) < 1e-6 and abs(xy[:, k].max() - limit[k] - .005) < 1e-6]
            self.assertEqual(len(across), 1, xy)                 # the dash lies across one border, 5 mm wide
            self.assertLessEqual(xy[:, 1 - across[0]].max(), limit[1 - across[0]] + .0025 + 1e-6)
        corner = region.rotation @ limit + region.center_xy_m          # the inner edges' corner, then 0.1 mm beyond it
        beyond = region.rotation @ (limit + .0001) + region.center_xy_m
        self.assertTrue(region.contains([[*corner, region.table_height_m]]))
        self.assertFalse(region.contains([[*beyond, region.table_height_m]]))
        self.assertTrue(all(c > 0 for c in entity.surface["emissive"]))

    def test_a_scene_look_changes_the_renderer_config_only(self):
        from omegaconf import OmegaConf
        from topogym.config.scene import load_scene_config
        from topogym.simulator.config import SimulatorConfig
        for family, variation in BASES[:3]:
            cls = task_definition(family, variation)
            plain = OmegaConf.to_container(load_scene_config(SimulatorConfig(task_id=family, variation_id=variation)))
            looked = OmegaConf.to_container(load_scene_config(SimulatorConfig(
                task_id=family, variation_id=variation, task_parameters=dict(scene_look=appearance(cls, 1)["scene_look"]))))
            with self.subTest(variation=variation):
                self.assertEqual({k: v for k, v in looked.items() if k != "env"},
                                 {k: v for k, v in plain.items() if k != "env"})
                self.assertEqual({k: v for k, v in looked["env"].items() if k not in ("renderer", "appearance")},
                                 {k: v for k, v in plain["env"].items() if k not in ("renderer", "appearance")})
                changed = {k for k in set(looked["env"]["renderer"]) | set(plain["env"]["renderer"])
                           if looked["env"]["renderer"].get(k) != plain["env"]["renderer"].get(k)}
                self.assertEqual(changed, {"lights", "env_intensity", "env_tint", "pbr_materials"})
                self.assertEqual(looked["env"]["appearance"][:len(plain["env"]["appearance"])], plain["env"]["appearance"])
                self.assertEqual([r["pattern"] for r in looked["env"]["appearance"][len(plain["env"]["appearance"]):]],
                                 ["table", "lab_floor", "table_frame*"])

    def test_a_blanket_look_keeps_the_birth_identity(self):
        """The PBD birth recipe's scene identity (declared as pbd_initial_state_generation._recipe does) is the nominal
        scene's for every look, while the drawn texture is an asset file of the tree, which an instance's asset_identity
        covers."""
        from topogym.assets.library import describe_build
        from topogym.config.scene import load_scene_config
        from topogym.simulator.config import SimulatorConfig
        from topogym.tasks.base import merge_parameters
        from topogym.utils.io import canonical_hash
        from topogym.world.initial_state import scene_identity
        flat = dict(initialization_mode="flat_for_preparation", initial_state_path=None, initial_state_resource=None,
                    initial_state_sha256=None)
        for family, variation in (("blanket_flatten", "blanket_flatten_easy"), ("blanket_fold", "blanket_fold_very_easy")):
            cls = task_definition(family, variation)
            looks = {}
            for seed in (0, 1, 2):
                params = merge_parameters(flat, appearance(cls, seed))
                config = SimulatorConfig(task_id=family, variation_id=variation, task_parameters=params)
                scene = load_scene_config(config)
                table = scene.env.world.table
                with tempfile.TemporaryDirectory() as folder:
                    task = cls(params, seed=0)
                    spec = task.declare(Path(folder), float(table.pos[2] + table.size[2] / 2))
                    build = describe_build(family, Path(folder), task.config, 0)
                    looks[seed] = (canonical_hash(scene_identity(config, scene, task, spec, build)), dict(build.sha256),
                                   spec.entities[0].surface.get("diffuse_texture_path", "").replace(str(Path(folder).resolve()), "@DIR@"))
            for seed in (1, 2):
                with self.subTest(variation=variation, seed=seed):
                    self.assertEqual(looks[seed][0], looks[0][0])                   # one birth for every look
                    self.assertEqual(set(looks[seed][1]) - set(looks[0][1]), {"blanket_appearance_diffuse.png"})
                    self.assertEqual({k: v for k, v in looks[seed][1].items() if k in looks[0][1]}, looks[0][1])
                    self.assertEqual(looks[seed][2], "@DIR@/blanket_appearance_diffuse.png")
            self.assertNotEqual(looks[1][1]["blanket_appearance_diffuse.png"], looks[2][1]["blanket_appearance_diffuse.png"])

    def test_a_look_changes_the_file_nyx_draws(self):
        """Nyx draws a URDF visual in the URDF's own colour (not its GLB's baseColorFactor), an MJCF collision geom of
        groups 0-2 in one grey (whatever its rgba), a GLB under a URDF without a colour by its baseColorFactor. Seed 1's
        look changes the drawn file of these bases (renders of 2026-10-20 showed seeds 1 and 2 pixel-identical to seed 0
        before) and leaves the declared spec and the physics content alone."""
        drawn = {("hedgehog_in_cage", "hedgehog_in_cage_window_change"): "cage.urdf",
                 ("hedgehog_in_cage", "hedgehog_in_cage_owl"): "owl_cage_meshes/visual.glb",
                 ("buzz_wire_3d", "buzz_wire_3d_rd_very_easy"): "rail_metal.glb",
                 ("orbits", "orbits_very_easy"): "cord.xml"}
        for (family, variation), name in drawn.items():
            cls = task_definition(family, variation)
            spec0, physics0, full0 = declare(cls, {}, 0)
            spec1, physics1, full1 = declare(cls, appearance(cls, 1), 0)
            with self.subTest(variation=variation):
                self.assertEqual(spec1, spec0)
                self.assertEqual(physics1, physics0)
                self.assertNotEqual(full1[name], full0[name])

    def test_the_physics_view_leaves_out_what_only_looks_change(self):
        """physics_view on hand-made files. Materials given to an MJCF entity that had no <asset>, the colour of a
        visual geom of density 0 and a URDF visual mesh swapped for a copy that carries a material leave it alone; a
        collider's size, a visual geom made to collide, a collision mesh's or a visual mesh's vertices change it."""
        obj, moved = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", "v 0 0 0\nv 2 0 0\nv 0 1 0\nf 1 2 3\n"
        plain = ('<mujoco><worldbody><body name="b"><geom type="box" size=".01 .01 .01"/><geom type="sphere" size=".02" '
                 'contype="0" conaffinity="0" density="0" rgba="1 0 0 1"/></body></worldbody></mujoco>')
        looked = ('<mujoco><asset><material name="m" rgba="0 0 1 1"/></asset><worldbody><body name="b"><geom type="box" '
                  'size=".01 .01 .01"/><geom type="sphere" size=".02" contype="0" conaffinity="0" density="0" material="m"/>'
                  '</body></worldbody></mujoco>')

        def view(mjcf, visual="p.obj", files=None):
            with tempfile.TemporaryDirectory() as folder:
                (Path(folder) / "a.xml").write_text(mjcf)
                (Path(folder) / "p.urdf").write_text(
                    f'<robot name="p"><link name="p"><visual><geometry><mesh filename="{visual}"/></geometry></visual>'
                    '<collision><geometry><mesh filename="c.obj"/></geometry></collision></link></robot>')
                for name, text in {"p.obj": obj, "c.obj": obj, **(files or {})}.items():
                    (Path(folder) / name).write_text(text)
                return physics_view(folder, [])

        base = view(plain)
        copy = {"p_finish.obj": "mtllib p_finish.mtl\nusemtl finish\n" + obj, "p_finish.mtl": "newmtl finish\nKd 1 0 0\n"}
        self.assertEqual(view(looked), base)
        self.assertEqual(view(plain, "p_finish.obj", copy), base)
        self.assertNotEqual(view(plain.replace('size=".01 .01 .01"', 'size=".02 .01 .01"')), base)
        self.assertNotEqual(view(plain.replace('contype="0" ', "")), base)
        self.assertNotEqual(view(plain, files={"c.obj": moved}), base)
        self.assertNotEqual(view(plain, "p_finish.obj", {**copy, "p_finish.obj": copy["p_finish.obj"].replace(obj, moved)}),
                            base)

    def test_appearance_seeds_and_task_seeds_leave_physics_alone(self):
        for family, variation in BASES:
            cls = task_definition(family, variation)
            spec0, physics0, full0 = declare(cls, {}, 0)
            for params, seed in ((appearance(cls, 1), 0), (appearance(cls, 2), 0), ({}, 1), ({}, 2)):
                with self.subTest(variation=variation, params=params, seed=seed):
                    spec, physics_, full = declare(cls, params, seed)
                    self.assertEqual(spec, spec0)
                    self.assertEqual(physics_, physics0)
                    self.assertEqual(full != full0, bool(params))   # a task seed alone changes no file


if __name__ == "__main__":
    unittest.main()
