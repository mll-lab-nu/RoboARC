"""Slotted post, mobile ring/beads/discs and FEM cord with attached end stops."""

from pathlib import Path
from topogym.tasks.placement import PlacementGoalMixin
import numpy as np
from topogym.assets.geometry.slotted_post_ring import (
    SlottedPostLayout,
    generate_assets,
    make_meshes,
    centerline,
)
from topogym.assets.materials import rope_fem_parameters, rope_ipc_parameters
from topogym.assets.geometry.rope_volume import write_rope_volume
from ..base import TaskDefinition, EntitySpec, TaskSpec, AttachmentSpec
from ..scene_presets import defaults, record_preset
from topogym.assets.scene_visuals import surface, visual_collision_urdf
from topogym.evaluation.task_metrics import placement_candidate, rigid_asset_vertices, ipc_integrity


class SlottedPostRingTask(PlacementGoalMixin, TaskDefinition):
    task_id = "slotted_post_ring"
    placement_description = 'the entire removable metal ring'
    goal_side = (1, 1)  # +y, the nominal scene's side; pose variation cannot flip it
    placement_attachment_names = ('cord_disc_left', 'cord_disc_right')
    physics_model = "fem_ipc"
    # cord_model "chain" (2026-10-09, the MuJoCo port; rcb_topogym sets it for MuJoCo instances): this setting's cord as
    # the team's 10-07 rigid capsule chain (fem_chain.py, slotted_post_chain.py), along this setting's authored cord,
    # with its geometry, goal region and rules unchanged and no grasp assistance. "fem" (the default, out of the instance
    # identity): the FEM+IPC cord, so frozen Genesis instances keep their bytes.
    defaults = defaults(task_id, geometry={}, stable_duration_s=0.5, cord_finish="hemp", cord_model="fem")
    hash_when_changed = ("cord_finish", "cord_model")
    # The chain of the FEM settings (cord_model "chain"):
    # - capsules of the cord's radius (rope_radius_m, 3.5 mm) and links of 16.5 mm (slotted_post_chain.LINK_LENGTH_M),
    #   on ball joints at the authored cord's arc-length points from disc eye to disc eye (chain_roots): the chain's
    #   length is the cord's between the eyes. 16.5 mm is the team's link: the beads' 9 mm bores leave 1 mm radially and
    #   hold at least two links (#108 measured 29.6 mm links jamming bead 1); a capsule chain is exactly the cord's radius
    #   everywhere (the joints are the capsules' shared end spheres).
    # - bending stiffness from the FEM material (rope_fem_parameters: E 50 kPa): k = E pi r^4 / (4 L) = 3.6e-4 N m/rad per
    #   joint, about the authored shape (MuJoCo's ball-joint spring reference; the FEM rest shape is the authored mesh too).
    #   It is small: a 10-link span's own weight bends it with ~3e-3 N m. Damping 2e-4 N m s/rad and armature 1e-6 kg m2
    #   are the team chain's (the FEM has no viscous term to convert); density 650 kg/m3 on the cylinders and friction 0.3
    #   are the FEM cord's.
    # - what the chain cannot do: stretch (E 50 kPa stretches the 7 mm cord ~50 % at 1 N); the FEM-only integrity rules (tet inversion, attachment stretch) have no chain counterpart:
    #   the discs hang on ball joints and cannot detach.
    chain_young_modulus_pa = 5e4

    def __init__(self, config=None, seed=0):
        super().__init__(config, seed)
        if self.config["cord_model"] not in ("fem", "chain"):
            raise ValueError("slotted_post_ring cord_model must be fem or chain")
        if self.config["cord_model"] == "chain":
            if type(self).physics_model != "fem_ipc":
                raise ValueError(f"{self.variation_id} has no FEM cord to replace")
            self.physics_model = "rigid_articulated"

    @property
    def chain_cord(self):
        return type(self).physics_model == "fem_ipc" and self.config["cord_model"] == "chain"

    def make_layout(self):
        """This setting's geometry and cord route."""
        return SlottedPostLayout(**self.config["geometry"])

    def declare(self, asset_dir, table_height_m):
        if self.chain_cord:
            return self._declare_chain(asset_dir, table_height_m)
        self.table_height_m = float(table_height_m)
        preset = record_preset(self, asset_dir)
        xy = np.asarray(self.config["center_xy_m"], dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("Invalid Slotted Post Ring center_xy_m")
        origin = tuple(np.r_[xy, table_height_m])
        self.layout, report = generate_assets(asset_dir, self.make_layout())
        parts, _ = make_meshes(self.layout)
        folder = Path(asset_dir)
        volume = write_rope_volume(folder / "cord_volume.npz", centerline(self.layout),
                                   self.layout.rope_radius_m, end_stops=True)
        entities = []
        for name in parts:
            if name.startswith("stop_knot"):
                continue  # Must become part of the same FEM cord, not free rigid decorations.
            soft = name == "cord"
            entities.append(
                EntitySpec(
                    name,
                    "mesh" if soft else "urdf",
                    folder / f"{name}.obj" if soft else visual_collision_urdf(folder, name),
                    origin,
                    fixed=name in {"base", "post"},
                    physics_model="fem_elastic" if soft else "rigid",
                    volume_path=volume if soft else None,
                    material=rope_fem_parameters()
                    if soft
                    else {"coup_type": "ipc_only"},
                    surface=surface(self.config["cord_finish"]) if soft else {},
                    morph_options={"convexify": False, "decimate": False,
                                   **({"file_meshes_are_zup": True, "align": False,
                                       "merge_fixed_links": False} if not soft else {})},
                )
            )
        self.placement_vertices = {e.name: rigid_asset_vertices(e) for e in entities if e.name != "cord"}
        self.placement_targets = ("target_ring",)
        self.placement_remainder = tuple(e.name for e in entities if e.name != "target_ring")
        return TaskSpec(
            tuple(entities),
            {"fem": {}, "ipc": rope_ipc_parameters()},
            dict(
                **preset,
                assembly_origin_m=list(origin),
                task_id=self.task_id,
                fem_rope_profile="v13_material_mesh",
                geometry_report=report,
                permanent_connections=[
                    "cord endpoint to disc",
                    "FEM end stops",
                    "mobile bead/cord contact",
                ],
                goal_evaluation_complete=bool(self.goal_complete),
            ),
            attachments=tuple(AttachmentSpec(f"cord_disc_{side}", "cord", f"disc_{side}", end, self.layout.rope_radius_m) for side, end in (("left", "start"), ("right", "end"))),
        )

    # -- cord_model "chain": the below-easy chain construction and goal measurement on this setting's layout
    def _declare_chain(self, asset_dir, table_height_m):
        """The below-easy chain scene (SlottedBelowEasyTask.declare) of this setting's layout. The goal region is chosen
        from the FEM declaration's clouds (placement_clouds), so it is the FEM setting's region."""
        import tempfile
        from . import fem_chain
        spec = fem_chain.declare(self, asset_dir, table_height_m)
        folder, origin = Path(asset_dir), spec.entities[0].position_m
        parts, cord = make_meshes(self.layout)
        rigid = {name: rigid_asset_vertices(EntitySpec(name, "urdf", visual_collision_urdf(folder, name), origin))
                 for name in parts if name not in {"cord"} and not name.startswith("stop_knot")}
        with tempfile.TemporaryDirectory() as tmp:
            volume = write_rope_volume(Path(tmp)/"cord_volume.npz", centerline(self.layout), self.layout.rope_radius_m,
                                       end_stops=True)
            with np.load(volume) as data:
                spawn = np.asarray(data["spawn"])
        offset = np.asarray(origin)
        self.region_clouds = dict(
            target_ring=rigid["target_ring"]+offset, base=rigid["base"]+offset, post=rigid["post"]+offset,
            cord=np.vstack([spawn, *(rigid[n] for n in ("bead_left", "bead_right", "disc_left", "disc_right"))])+offset)
        return spec

    def render_host_binding(self):
        """How the render bridge shows the chain as this setting's FEM cord (after _declare_chain): the FEM scene is the
        render host, its cord skinned on the chain's links by material arc length (the links are equal arc spans of the
        authored cord from eye to eye, as the FEM cord's material coordinate runs) and its rigid parts carried by the
        MuJoCo bodies that hold them."""
        cord = centerline(self.layout)
        arc = float(np.linalg.norm(np.diff(cord, axis=0), axis=1).sum())
        return dict(entity="cord", chain=[f"cord::{name}" for name in self.link_names],
                    link_s_m=np.linspace(0., arc, len(self.link_names)+1).tolist(),
                    bodies={"target_ring": "target_ring::target_ring",
                            **{name: f"cord::{name}" for name in ("bead_left", "bead_right", "disc_left", "disc_right")}})

    def chain_parts(self):
        from . import fem_chain
        return fem_chain.chain_parts(self)

    def link_cover(self, length):
        from . import fem_chain
        return fem_chain.link_cover(self, length)

    def placement_clouds(self, spec):
        if self.chain_cord:
            return self.region_clouds
        return super().placement_clouds(spec)

    def evaluate_state(self, state):
        return self.evaluate_placement(state)

    def evaluate_goal_region(self, state, region):
        """Measure whole-ring placement and permanent-connection integrity."""
        if self.chain_cord:      # the ring inside and still (1 mm/s, 0.02 rad/s), the rest on the system side
            from . import fem_chain
            return fem_chain.evaluate_goal_region(self, state, region)
        return placement_candidate(state, region, self.placement_vertices,
            self.placement_targets, self.placement_remainder, soft_names=("cord",),
            attachment_names=("cord_disc_left", "cord_disc_right"))

    def goal_region_observation(self, region):
        return region.observation(self.placement_targets, self.placement_remainder)


def recommended_grasp_assist(variation_id, seed=0, *, task_parameters=None):
    """The rigid-chain settings' own contact weld (the ring only; tasks/slotted_post_ring/slotted_post_ring_hard_chain.py
    grasp_assist). The FEM tiers keep their unassisted contract."""
    from topogym.config.registry import task_definition
    task = task_definition("slotted_post_ring", variation_id, seed)(task_parameters, seed=seed)
    # The FEM settings stay unassisted, their cord a chain (cord_model "chain") or not; the rigid-chain settings (the
    # below-easy and #108 chain settings) declare their own.
    if type(task).physics_model != "rigid_articulated":
        return None
    return task.grasp_assist()
