"""Session-level task classes of the redesigned rope_threading tiers.

The scene is `scenes.build(tier)` (rings with their own axes, the bulkhead, posts, the shelf, the start rope); the
success predicate is RopeThreadingTask.evaluate_state, unchanged (ordered signed aperture crossings along the material
coordinate, thickness clear, unambiguous, quiet), reading the rings through a ThreadingLayout-compatible view
(`RedesignLayout`). The rope and contact settings are the dev profile the MuJoCo strict references run with
(contact_relaxed_v16: 5.5 mm mesh, E 5 MPa, rest = the start shape, finger-only robot contact, 1 mm aperture contact
allowance, round-ring thickness check, quiet <= 10 mm/s for 0.5 s).

Nothing here is registered: topogym/planning/rope_threading/redesign/session_build.py registers a variation and its
MuJoCo profile entry in-process (the rings split into their exact convex tube segments, as rope_chain_v1 does for
R1-R3) and freezes an instance; patches/rope_threading_registration.patch proposes the permanent entries.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from topogym.tasks.base import EntitySpec, TaskSpec, merge_parameters
from topogym.tasks.scene_presets import record_preset
from topogym.assets.materials import rope_fem_parameters, rope_ipc_parameters
from topogym.assets.geometry.rope_volume import write_rope_volume
from topogym.assets.scene_visuals import surface
from .. import common
from ..common import RopeThreadingTask
from . import frozen as S

DEV_PARAMETERS = dict(rope_mesh=dict(spacing=.0055, sides=12), rope_material=dict(E=5e6),
                      rope_reference_shape="initial", rope_robot_contact="fingers",
                      aperture_contact_allowance_m=.001, aperture_check_profile="round_ring",
                      terminal=dict(maximum_speed_m_s=.01))


class RedesignLayout:
    """The ThreadingLayout surface RopeThreadingTask.evaluate_state and finalize_spec use, for a redesign tier: rings
    in the entity frame (world x, y; z above the table), the required signed order, the rope radius, the start."""
    ring_shape = "closed"

    COLORS = {(0.2, 0.7, 0.8): "cyan", (0.35, 0.8, 0.4): "green", (0.75, 0.5, 0.85): "purple", (0.95, 0.75, 0.2): "amber",
              (0.9, 0.35, 0.35): "red"}

    def __init__(self, design):
        self.design = design
        self._rings = [dict(r) for r in design.rings]       # frozen in goal order, entity frame
        for r in self._rings:          # public names, as goal_description names R1-R3 ("cyan ring (R1)")
            color = self.COLORS.get(tuple(round(float(c), 2) for c in r["rgba"][:3]), "grey")
            r["display_name"] = f"{color} {'grommet in the wall' if r['mount'] == 'bulkhead' else 'ring'} ({r['name']})"

    @property
    def radius(self):
        return S.ROPE_RADIUS_M

    @property
    def rope_radius_m(self):
        return S.ROPE_RADIUS_M

    def rings(self):
        return [dict(r) for r in self._rings]

    def required_crossings(self):
        return [tuple(x) for x in self.design.required_crossings]

    def centerline(self):
        return self.design.start + S.TASK_ROOT_WORLD


def _mesh_file(folder, name, mesh, rgba):
    from topogym.assets.geometry.rope_threading import _material
    _material(mesh, rgba)
    mesh.export(folder / f"{name}.obj", mtl_name=f"{name}.mtl")
    mesh.export(folder / f"{name}.glb")
    return folder / f"{name}.glb"


class RedesignThreadingTask(RopeThreadingTask):
    """Base of the five redesign tiers; subclasses set redesign_tier and the variation identity."""
    redesign_tier = None
    defaults = merge_parameters(RopeThreadingTask.defaults, DEV_PARAMETERS)
    instruction = ("Thread the rope through every ring in the prescribed order and direction (the grommets in the wall "
                   "and the rings on either side of it), then keep the whole rope still.")

    def declare(self, asset_dir, table_height_m):
        import trimesh
        from topogym.assets.geometry.rope_threading import ring_mesh
        from topogym.utils.geometry.rope import make_elastic_tube_mesh, write_obj
        self.table_height_m = float(table_height_m)
        design = S.load(self.redesign_tier)
        self.layout = RedesignLayout(design)
        for ring in self.layout.rings():
            common.usable_rope_aperture_radius(ring, S.ROPE_RADIUS_M, self.config["terminal"],
                                               self.config["aperture_contact_allowance_m"])
        folder = Path(asset_dir)
        folder.mkdir(parents=True, exist_ok=True)
        preset = record_preset(self, folder)
        centerline = self.layout.centerline()
        volume = write_rope_volume(folder / "rope_volume.npz", centerline, S.ROPE_RADIUS_M, **self.config["rope_mesh"])
        if self.config["rope_reference_shape"] == "initial":
            with np.load(volume, allow_pickle=False) as data:
                arrays = {k: data[k].copy() for k in data.files}
            np.savez_compressed(volume, **arrays, elastic_rest=arrays["spawn"])
        # through the module attribute: a frozen instance's volume replaces the generated one (rcb_topogym)
        self.centerline_map = common.load_centerline_map(volume)
        vertices, faces = make_elastic_tube_mesh(centerline, radius=S.ROPE_RADIUS_M, radial_segments=12)
        write_obj(folder / "rope.obj", vertices, faces)
        entities = [EntitySpec("rope", "mesh", folder / "rope.obj", (0, 0, table_height_m), physics_model="fem_elastic",
                               volume_path=volume, material=rope_fem_parameters() | self.config["rope_material"],
                               surface=surface(self.config["rope_finish"]),
                               morph_options={"convexify": False, "decimate": False})]
        ring_style = dict(fixed=True, material={"coup_type": "ipc_only"},
                          morph_options={"convexify": False, "decimate": False, "file_meshes_are_zup": True})
        for ring in self.layout.rings():
            mesh = ring_mesh(ring, 24)
            path = _mesh_file(folder, ring["name"], mesh, ring["rgba"])
            entities.append(EntitySpec(ring["name"], "mesh", path, (0, 0, table_height_m),
                                       surface={"color": tuple(ring["rgba"]), "metallic": .15, "roughness": .5},
                                       **ring_style))
            if "stand" in ring:
                a, b = np.asarray(ring["stand"]["fromto"][:3]), np.asarray(ring["stand"]["fromto"][3:])
                stand = trimesh.creation.capsule(height=float(np.linalg.norm(b - a)), radius=ring["stand"]["radius"],
                                                 count=[12, 12])
                stand.apply_translation(-stand.bounds.mean(axis=0))
                axis = (b - a) / np.linalg.norm(b - a)
                R = trimesh.geometry.align_vectors([0, 0, 1.0], axis)
                stand.apply_transform(R)
                stand.apply_translation((a + b) / 2)
                path = _mesh_file(folder, ring["name"] + "_stand", stand, (.6, .6, .6, 1.))
                entities.append(EntitySpec(ring["name"] + "_stand", "mesh", path, (0, 0, table_height_m),
                                           surface={"color": (.6, .6, .6, 1.), "metallic": .15, "roughness": .5},
                                           **ring_style))
                foot = trimesh.creation.box(extents=2 * np.asarray(ring["base"]["half"]))
                foot.apply_translation(np.asarray(ring["base"]["center"]))
                path = _mesh_file(folder, ring["name"] + "_base", foot, (.45, .48, .5, 1.))
                entities.append(EntitySpec(ring["name"] + "_base", "mesh", path, (0, 0, table_height_m),
                                           surface={"color": (.45, .48, .5, 1.), "metallic": .15, "roughness": .5},
                                           **ring_style))
        for o in design.obstacles:
            if o.kind == "box":
                from scipy.spatial.transform import Rotation
                q = Rotation.from_matrix(o.R).as_quat()[[3, 0, 1, 2]]
                c = o.center + S.TASK_ROOT_WORLD + np.array([0, 0, table_height_m])
                entities.append(EntitySpec(o.name, "box", None, tuple(map(float, c)), tuple(map(float, q)), fixed=True,
                                           material={"coup_type": "ipc_only"},
                                           morph_options={"size": tuple(float(2 * h) for h in o.half)},
                                           surface={"color": tuple(o.rgba), "metallic": .1, "roughness": .6}))
            else:
                a = o.a + S.TASK_ROOT_WORLD
                b = o.b + S.TASK_ROOT_WORLD
                if abs(a[0] - b[0]) > 1e-9 or abs(a[1] - b[1]) > 1e-9:
                    raise ValueError(f"{o.name}: only vertical posts are declared as cylinders")
                h = float(b[2] - a[2])
                c = (a + b) / 2 + np.array([0, 0, table_height_m])
                entities.append(EntitySpec(o.name, "cylinder", None, tuple(map(float, c)), (1., 0., 0., 0.), fixed=True,
                                           material={"coup_type": "ipc_only"},
                                           morph_options={"radius": float(o.radius), "height": h},
                                           surface={"color": tuple(o.rgba), "metallic": .1, "roughness": .6}))
        report = dict(tier=self.redesign_tier, rope_length_m=design.rope_length,
                      ring_aperture_diameters_m={r["name"]: 2 * r["inner_radius_m"] for r in self.layout.rings()},
                      target_material_order=[n + ("+" if s > 0 else "-") for n, s in self.layout.required_crossings()],
                      selected_ring_ids=[r["name"] for r in self.layout.rings()], obstacles=[o.name for o in design.obstacles])
        return TaskSpec(tuple(entities), {"fem": {}, "ipc": rope_ipc_parameters()},
                        dict(**preset, task_id=self.task_id, fem_rope_profile=self.config["rope_mesh"].get("profile", "v13_material_mesh"),
                             geometry_report=report, required_reference_shape=self.config["rope_reference_shape"],
                             goal_evaluation_complete=bool(self.goal_complete)))


def _tier_class(tier):
    name = "RopeThreadingRD" + "".join(p.capitalize() for p in tier.split("_"))
    return type(name, (RedesignThreadingTask,), dict(
        redesign_tier=tier, variation_id=f"rope_threading_rd_{tier}", family_id="bulkhead_lacing",
        model_revision=f"bulkhead_lacing_{tier}_rd_v0", difficulty=tier, __module__=__name__))


RopeThreadingRDVeryEasy = _tier_class("very_easy")
RopeThreadingRDEasier = _tier_class("easier")
RopeThreadingRDEasy = _tier_class("easy")
RopeThreadingRDMedium = _tier_class("medium")
RopeThreadingRDHard = _tier_class("hard")
RopeThreadingRDMediumAlt = _tier_class("medium_alt")
_ALT = {t: _tier_class(t) for t in ("medium_t0", "medium_t0ccw", "medium_tm10", "medium_tp10", "hard_t0", "hard_t0ccw", "hard_tm10", "hard_tp10")}
globals().update({c.__name__: c for c in _ALT.values()})      # registered by module:attribute
CLASSES = {"very_easy": RopeThreadingRDVeryEasy, "easier": RopeThreadingRDEasier, "easy": RopeThreadingRDEasy,
           "medium": RopeThreadingRDMedium, "hard": RopeThreadingRDHard, "medium_alt": RopeThreadingRDMediumAlt, **_ALT}
