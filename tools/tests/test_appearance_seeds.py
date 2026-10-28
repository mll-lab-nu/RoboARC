#!/usr/bin/env python3
"""Appearance seeds of the fixed-scene bases (scene_variation.appearance) change only visual geoms and meshes: declared
specs and the physics content of the generated files stay those of seed 0; no physics runs. Every release base's seeds 1
and 2 differ; a blanket look keeps the frozen birth's scene identity, and its texture is an asset file."""
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
         ("lock_picking", "lock_picking_very_easy"), ("wire_splicing", "wire_splicing_eye_wrap"))


def release_bases():
    """(base, task, variation) of every base in tools/release/mujoco_release.json (release_rebuild.bases_of's names)."""
    spec = json.loads((ROOT / "tools/release/mujoco_release.json").read_text())
    for f, r in spec["families"].items():
        for t in r.get("tiers", []):
            yield f"{f}-{t}", f, r.get("variations", {}).get(t, f"{f}_{t}")
        for name, n in r.get("new_bases", {}).items():
            yield name, n.get("task", f), n["variation"]


def physics(path, folder):
    """A generated file's content as the physics reads it: MJCF without the geoms that neither collide nor carry mass
    (mjcf_entity.merge_mjcf_entity drops them) and the meshes only they use; URDF without <visual>; images and GLBs none."""
    data = path.read_bytes().replace(str(folder).encode(), b"@DIR@")
    if path.suffix == ".obj":
        return data
    if path.suffix not in (".xml", ".urdf"):
        return None
    root = ET.fromstring(data)
    for parent in root.iter():
        for child in list(parent):
            visual = child.tag == "geom" and not (int(child.get("contype", "1")) or int(child.get("conaffinity", "1"))) \
                and float(child.get("mass", "1")) == 0.
            if visual or child.tag in ("visual", "material", "texture"):
                parent.remove(child)
    used = {g.get("mesh") for g in root.iter("geom")}
    for asset in root.iter("asset"):
        for mesh in [m for m in asset if m.tag == "mesh" and m.get("name") not in used]:
            asset.remove(mesh)
    return ET.tostring(root)


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
        # files without physics content (GLBs, images) may be renamed: Horseshoe's ring GLB is named after its finish
        physics_ = {k: physics(p, folder) for k, p in files.items()}
        return (plain, {k: v for k, v in physics_.items() if v is not None},
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
        from topogym.world.look import FLOOR_FINISHES, TABLE_FINISHES, check_scene_look
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
                self.assertNotEqual(one["lighting"]["key"], two["lighting"]["key"])
                self.assertNotEqual(one["lighting"]["cct_k"] < 6000, two["lighting"]["cct_k"] < 6000)
                self.assertNotEqual(one["lighting"]["brightness"] < 1, two["lighting"]["brightness"] < 1)

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
                                 ["table", "lab_floor"])

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
