"""Orbits FEM cord, rigid fixtures and permanent material endpoints on the handle."""

from pathlib import Path
from topogym.tasks.placement import PlacementGoalMixin
import hashlib
import json
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from topogym.assets.geometry.orbits import OrbitsLayout, generate_assets, make_meshes
from topogym.assets.materials import rope_fem_parameters, rope_ipc_parameters
from topogym.assets.geometry.rope_volume import write_rope_volume
from ..base import TaskDefinition, EntitySpec, TaskSpec, AttachmentSpec
from ..scene_presets import defaults, record_preset
from topogym.assets.scene_visuals import surface, visual_collision_urdf
from topogym.evaluation.task_metrics import placement_candidate, rigid_asset_vertices, ipc_integrity


# Shared with development planners; these are the existing local task values,
# in metres after linear_scale. Imported oracle geometry does not replace them.
CORD_PROFILES = {
    "compact_external_handle_v1": dict(handle_x_m=.41, cord_length_m=.80),
    # very_easy stiff loop (2026-10-07): 0.90 m around the 35 mm bight.
    "below_easy_stiff_loop_v1": dict(handle_x_m=.32, cord_length_m=.90),
    "easier_ring_hole_v2": dict(handle_x_m=.42, cord_length_m=1.25),
}


def recommended_grasp_assist(variation_id, seed=0, *, task_parameters=None):
    """Contact weld on any chain link for the rigid-chain tiers (very_easy, easier; user, 2026-10-07): friction pinches
    on the chain let it slip out under load. The FEM tiers keep their unassisted contract."""
    from topogym.config.registry import task_definition
    from topogym.utils.grasp import GraspAssistConfig, GraspTarget
    from .below_easy import OrbitsBelowEasyTask
    task = task_definition("orbits", variation_id, seed)(task_parameters, seed=seed)
    # Only these two tiers' chain (its link layout comes from chain_roots); other tiers keep their own contract.
    if not isinstance(task, OrbitsBelowEasyTask):
        return None
    _, _, roots = task.chain_roots()
    lengths = np.linalg.norm(np.diff(roots, axis=0), axis=1)
    # An 8 mm sphere about each link's middle (teammate standard): the ~23 mm finger pads often touch two links, and a
    # pinch touching two targets does not weld. The weld may drift 15 mm before it lets go (default 10 mm; user,
    # 2026-10-08): a grip loaded with about 0.3 N slid the
    # cord about 1 cm along itself between the closed fingers (10.5 mm in the graded image, Harbor batch 23) with no
    # visible slip; a dropped cord drifts tens of centimetres.
    return GraspAssistConfig("contact_weld_v1", tuple(
        GraspTarget("cord", f"link_{i:02d}", (length/2, 0., 0.), .008) for i, length in enumerate(lengths)),
        max_translation_error_m=.015)


def profile_value(profile, key, ring_count):
    value = profile[key]
    return float(value[ring_count] if isinstance(value, dict) else value)


def profile_cord_length(profile, ring_count):
    return profile_value(profile, "cord_length_m", ring_count)


class OrbitsTask(PlacementGoalMixin, TaskDefinition):
    task_id = "orbits"
    placement_description = 'the entire cord and wooden handle'
    goal_side = (1, -1)  # -y, the nominal scene's side; pose variation cannot flip it
    placement_attachment_names = ('cord_handle_start', 'cord_handle_end')
    physics_model = "fem_ipc"
    # cord_model "chain" (2026-10-09, the MuJoCo port; rcb_topogym sets it for MuJoCo instances): this setting's cord as
    # the team's rigid capsule chain on the handle, built as the very_easy / easier chain is (below_easy.py,
    # orbits_chain.py), along this setting's authored cord, with its geometry, goal region and rules unchanged and no
    # grasp assistance. "fem" (the default, out of the instance identity): the FEM+IPC cord, so frozen Genesis instances
    # keep their bytes.
    defaults = defaults(task_id, geometry={}, stable_duration_s=0.5, cord_profile="reference",
                        cord_finish="golden_rope", cord_model="fem")
    hash_when_changed = ("cord_finish", "cord_model")
    # The chain of the FEM settings (cord_model "chain"):
    # - capsules of the cord's radius (cord_radius_m: 1.87 mm after the 0.85 scale) and links of 16.5 mm (the team's
    #   link, as very_easy / easier), on ball joints at the authored cord's arc-length points (resample_links) from above
    #   one handle eye to above the other (chain_polyline: the FEM cord's ends inside the eyes become the ball joints at
    #   the handle top), the loop closed at its middle by a <connect>. A capsule chain is exactly the cord's radius
    #   everywhere (the joints are the capsules' shared end spheres), so it keeps the cord's clearances round the posts,
    #   the stopper ball and through the rings.
    # - bending stiffness from the FEM material (rope_fem_parameters: E 50 kPa): k = E pi r^4 / (4 L) = 2.9e-5 N m/rad per
    #   joint, about the authored loop (very_easy / easier use 100 MPa). Damping 2e-4 N m s/rad and armature 1e-6 kg m2 are
    #   the team chain's (the FEM has no viscous term to convert; very_easy / easier's 1e-5 armature compensated Genesis'
    #   soft weld and tie, and these settings have no weld and a nearly hard MuJoCo tie); density 650 kg/m3 and friction
    #   0.3 (the fixture too, the FEM cord's IPC friction) are the FEM cord's.
    # - the FEM rule "an attachment stretched more than 2 mm" becomes the middle tie opened more than 2 mm (the handle's
    #   two ball joints cannot open); a chain cannot stretch (E 50 kPa stretches this cord ~180 % at 1 N).
    CHAIN_RADIUS_M = .00187
    LINK_LENGTH_M = .0165
    JOINT_DAMPING = 2e-4
    JOINT_ARMATURE = 1e-6
    YOUNG_MODULUS_PA = 5e4
    LINEAR_LIMIT_M_S = .001          # placement_candidate's limits: every cord point and the handle at most 1 mm/s,
    ANGULAR_LIMIT_RAD_S = .02        # the handle's rotation at most 0.02 rad/s
    TIE_LIMIT_M = .002

    def __init__(self, config=None, seed=0):
        supplied = dict(config or {})
        if "center_xy_m" in supplied and "placement_anchor" not in supplied:
            if "base_center_xy_m" in supplied:
                raise ValueError("Specify placement_anchor when supplying both Orbits centers")
            supplied["placement_anchor"] = "assembly_center"
        super().__init__(supplied, seed)
        if self.config["cord_model"] not in ("fem", "chain"):
            raise ValueError("orbits cord_model must be fem or chain")
        if self.config["cord_model"] == "chain":
            if type(self).physics_model != "fem_ipc":
                raise ValueError(f"{self.variation_id} has no FEM cord to replace")
            self.physics_model = "rigid_articulated"

    @property
    def chain_cord(self):
        return type(self).physics_model == "fem_ipc" and self.config["cord_model"] == "chain"

    def authored_cord(self):
        """(layout, initial cord or None: the layout's own reference cord) of this setting's cord_profile."""
        layout = OrbitsLayout(**self.config["geometry"]).scaled(self.config["linear_scale"])
        initial_cord = None
        if self.config["cord_profile"] == "profile_v2":
            from dataclasses import replace
            from topogym.assets.geometry.orbits_initial import natural_working_state, resample
            initial_cord, _ = natural_working_state(layout, handle_x_m=.60, spacing=.0075)
            # Preserve the recorded polyline while providing a conservative
            # clearance cover and adequate FEM tube sampling.
            initial_cord = resample(initial_cord, spacing=.001)
            layout = replace(layout, handle_x_m=.60)
        elif self.config["cord_profile"] in CORD_PROFILES:
            from dataclasses import replace
            from topogym.assets.geometry.orbits_initial import natural_working_state, resample
            # Read the same profile as the development planner.
            # Both legs stop at the near-side eyes;
            # the handle's centre lies beyond the entire open bight, never inside it.
            profile = CORD_PROFILES[self.config["cord_profile"]]
            handle_x_m = profile_value(profile, "handle_x_m", layout.ring_count)
            initial_cord, _ = natural_working_state(layout, handle_x_m=handle_x_m,
                cord_length_m=profile_cord_length(profile, layout.ring_count), spacing=.002)
            initial_cord = resample(initial_cord, spacing=.001)
            layout = replace(layout, handle_x_m=handle_x_m)
        elif self.config["cord_profile"] != "reference":
            raise ValueError("Unknown Orbits cord_profile")
        return layout, initial_cord

    def declare(self, asset_dir, table_height_m):
        if self.chain_cord:
            return self._declare_chain(asset_dir, table_height_m)
        self.table_height_m = float(table_height_m)
        preset = record_preset(self, asset_dir)
        layout, initial_cord = self.authored_cord()
        self.layout, report = generate_assets(
            asset_dir, layout, initial_cord=initial_cord
        )
        parts, centerline = make_meshes(self.layout, initial_cord=initial_cord)
        report.update(handle_outside_bight=bool(self.layout.handle_x_m > centerline[:, 0].max()),
            cord_length_m=float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum()),
            full_solution_length_qualified=False, material_rest_shape='straight',
            initialization_stress_free=False)
        folder = Path(asset_dir)
        yaw = float(self.config["yaw_deg"])
        if not np.isfinite(yaw):
            raise ValueError("Invalid Orbits yaw_deg")
        rotation = Rotation.from_euler("z", yaw, degrees=True)
        points = np.vstack([rotation.apply(mesh.vertices) for mesh in parts.values()])
        center = (points.min(0)+points.max(0))/2
        base_points = rotation.apply(parts["base"].vertices)
        base_center = (base_points.min(0)+base_points.max(0))/2
        anchor = self.config["placement_anchor"]
        if anchor == "base_center":
            coordinate_key, local_anchor = "base_center_xy_m", base_center
        elif anchor == "assembly_center":
            coordinate_key, local_anchor = "center_xy_m", center
        else:
            raise ValueError("Orbits placement_anchor must be base_center or assembly_center")
        xy = np.asarray(self.config[coordinate_key], dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError(f"Invalid Orbits {coordinate_key}")
        self.origin = np.r_[xy-local_anchor[:2], table_height_m]
        # Bake the same rigid rotation into every visual, collider and FEM spawn.
        # FEM factory keeps identity orientation; endpoint identities do not change.
        if yaw:
            matrix = np.eye(4); matrix[:3, :3] = rotation.as_matrix()
            for name in parts:
                for suffix in ("obj", "glb"):
                    path = folder / f"{name}.{suffix}"
                    mesh = trimesh.load(path, force="mesh", process=False)
                    mesh.apply_transform(matrix); mesh.export(path)
            scene = trimesh.load(folder / "scene.glb", force="scene", process=False)
            scene.apply_transform(matrix)
            scene.export(folder / "scene.glb")
            np.save(folder / "cord_centerline.npy", rotation.apply(centerline))
        volume = write_rope_volume(folder / "cord_volume.npz", centerline,
                                   self.layout.cord_radius_m, end_stops=False)
        if yaw:
            # Pose the existing tube; rebuilding frames after rotation can twist
            # cross-sections. Material rest coordinates and endpoint IDs stay fixed.
            with np.load(volume) as source:
                data = {key: source[key] for key in source.files}
            for key in ("spawn", "centerline"):
                data[key] = rotation.apply(data[key])
            np.savez_compressed(volume, **data)
        entities = []
        for name in parts:
            soft = name == "cord"
            fixed = name not in {"cord", "handle"}
            entities.append(
                EntitySpec(
                    name,
                    "mesh" if soft else "urdf",
                    folder / f"{name}.obj" if soft else visual_collision_urdf(folder, name),
                    tuple(self.origin),
                    fixed=fixed,
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
        self.placement_targets = ("cord", "handle")
        self.placement_remainder = tuple(e.name for e in entities if e.name not in self.placement_targets)
        # The generator hashes its original files before the task poses them.
        # Refresh the manifest after all runtime meshes and volumes are final.
        (folder / "sha256.json").write_text(json.dumps({
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.iterdir()) if path.is_file() and path.name != "sha256.json"
        }, sort_keys=True, indent=2) + "\n")
        return TaskSpec(
            tuple(entities),
            {"fem": {}, "ipc": rope_ipc_parameters()},
            dict(
                **preset, placement_anchor=anchor, placement_xy_m=xy.tolist(),
                assembly_center_xy_m=(center[:2]+self.origin[:2]).tolist(),
                base_center_xy_m=(base_center[:2]+self.origin[:2]).tolist(),
                base_long_axis_world=rotation.apply([1., 0., 0.]).tolist(),
                assembly_bounds_world_m=(np.array([points.min(0), points.max(0)])+self.origin).tolist(),
                yaw_deg=yaw,
                linear_scale=self.config["linear_scale"], origin_m=self.origin.tolist(),
                task_id=self.task_id,
                fem_rope_profile="v13_material_mesh",
                orbits_scene_contract=dict(
                    ring_count=self.layout.ring_count,
                    post_count=self.layout.ring_count + 1,
                    ring_ids=[f"ring_{i+1}" for i in range(self.layout.ring_count)],
                    ring_centers_world_m=(rotation.apply(np.column_stack([
                        self.layout.ring_x, np.zeros(self.layout.ring_count),
                        self.layout.ring_z])) + self.origin).tolist() if self.layout.ring_count else [],
                    trapped_post_id=f"post_{self.layout.ring_count+1}",
                    initial_winding_absolute=[0]*self.layout.ring_count + [1],
                    cord_profile=self.config["cord_profile"],
                    cord_length_m=report["cord_length_m"],
                    handle_x_local_m=self.layout.handle_x_m,
                    difficulty_qualified=False, solution_verified=False,
                    stability_qualified=False,
                ),
                geometry_report=report, geometry_report_frame="unrotated_template",
                permanent_connections=["cord endpoints to handle eyes"],
                goal_evaluation_complete=bool(self.goal_complete),
            ),
            attachments=tuple(AttachmentSpec(f"cord_handle_{end}", "cord", "handle", end, self.layout.cord_radius_m) for end in ("start", "end")),
        )

    # -- cord_model "chain": the very_easy / easier chain construction on this setting's cord
    def chain_roots(self):
        """(layout, the FEM declaration's initial cord (None: the layout's reference cord), the chain's link roots)."""
        from topogym.assets.geometry.orbits_chain import chain_polyline, resample_links
        layout, initial_cord = self.authored_cord()
        layout = layout.validate()
        if abs(layout.cord_radius_m-self.CHAIN_RADIUS_M) > 1e-9:
            raise ValueError(f"{self.variation_id}: the chain radius must be the cord's {layout.cord_radius_m}")
        _, cord = make_meshes(layout, initial_cord=initial_cord)
        self.fem_centerline = cord
        return layout, initial_cord, resample_links(chain_polyline(layout, cord, self.CHAIN_RADIUS_M), self.LINK_LENGTH_M)

    def ring_collision_urdf(self, folder, name):
        """A ring's URDF in the chain scene (OrbitsBelowEasyTask.declare): for the FEM settings' chains (cord_model
        "chain") the ring's own mesh, as before (MuJoCo collides it as its exact convex segments, profiles.py); the
        below-easy settings override it with a loop of capsules (below_easy.py)."""
        return visual_collision_urdf(folder, name)

    def _declare_chain(self, asset_dir, table_height_m):
        """The very_easy / easier chain scene (OrbitsBelowEasyTask.declare) of this setting's cord. The goal region is
        chosen from the FEM declaration's clouds (placement_clouds), so it is the FEM setting's region."""
        import tempfile
        from .below_easy import OrbitsBelowEasyTask
        spec = OrbitsBelowEasyTask.declare(self, asset_dir, table_height_m)
        cord_entity = next(e for e in spec.entities if e.name == "cord")
        rotation = Rotation.from_quat(np.asarray(cord_entity.quaternion_wxyz)[[1, 2, 3, 0]])
        origin = np.asarray(cord_entity.position_m)
        handle = rigid_asset_vertices(EntitySpec("handle", "urdf", visual_collision_urdf(Path(asset_dir), "handle"),
                                                 tuple(origin)))
        with tempfile.TemporaryDirectory() as tmp:
            volume = write_rope_volume(Path(tmp)/"cord_volume.npz", self.fem_centerline, self.layout.cord_radius_m,
                                       end_stops=False)
            with np.load(volume) as data:
                spawn = np.asarray(data["spawn"])
        self.region_clouds = dict(cord=rotation.apply(spawn)+origin, handle=rotation.apply(handle)+origin)
        for name in self.placement_remainder:
            self.region_clouds[name] = rotation.apply(self.placement_vertices[name])+origin
        self.placement_targets = ("cord", "handle")      # the FEM setting's targets; "handle" is the cord entity's root
        return spec

    def render_host_binding(self):
        """How the render bridge shows the chain as this setting's FEM cord (after _declare_chain): the FEM scene is the
        render host, its cord skinned on the chain's links by material arc length (the chain spans the FEM cord from
        where it leaves the first eye to where it enters the second, in equal arc steps) and its handle carried by the
        chain's root."""
        from topogym.assets.geometry.orbits_chain import chain_polyline
        cord = self.fem_centerline
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(cord, axis=0), axis=1))]
        free = chain_polyline(self.layout, cord, self.CHAIN_RADIUS_M)
        start = int(np.flatnonzero(np.all(np.isclose(cord, free[0], rtol=0., atol=1e-12), axis=1))[0])
        span = float(np.linalg.norm(np.diff(free, axis=0), axis=1).sum())
        return dict(entity="cord", chain=[f"cord::{name}" for name in self.link_names],
                    link_s_m=(arc[start]+np.linspace(0., span, len(self.link_names)+1)).tolist(),
                    bodies={"handle": "cord::handle"})

    def link_cover(self, length):
        from .below_easy import OrbitsBelowEasyTask
        return OrbitsBelowEasyTask.link_cover(self, length)

    def cord_points(self, state):
        from .below_easy import OrbitsBelowEasyTask
        return OrbitsBelowEasyTask.cord_points(self, state)

    def placement_clouds(self, spec):
        if self.chain_cord:
            return self.region_clouds
        return super().placement_clouds(spec)

    def _evaluate_chain(self, state, region):
        """placement_candidate's measurement on the chain: the cord (every link's cover) and the handle inside, the
        fixture on the system side, both ends of every link and the handle at most 1 mm/s, the handle at most 0.02 rad/s,
        the middle tie closed within 2 mm (else connection_or_volume_invalid, the FEM attachment rule)."""
        from topogym.evaluation.task_metrics import body_points, quiet_body
        from topogym.utils.geometry.regions import placement_measurements
        links = state["articulations"]["cord"]["links"]
        points, tie = self.cord_points(state)
        handle, cord = points[:len(self.handle_vertices)], points[len(self.handle_vertices):]
        remainder = {name: body_points(state["bodies"][name], self.placement_vertices[name])
                     for name in self.placement_remainder}
        metrics = placement_measurements(region, {"cord": cord, "handle": handle}, remainder)
        speeds = []
        for name, length in zip(self.link_names, self.link_lengths):
            link = links[name]
            arm = Rotation.from_quat(np.asarray(link["quaternion_wxyz"])[[1, 2, 3, 0]]).apply([[0., 0., 0.], [length, 0., 0.]])
            v = np.asarray(link["linear_velocity_m_s"])+np.cross(np.asarray(link["angular_velocity_rad_s"]), arm)
            speeds.append(float(np.linalg.norm(v, axis=1).max()))
        quiet = (max(speeds) <= self.LINEAR_LIMIT_M_S
                 and quiet_body(links["handle"], self.LINEAR_LIMIT_M_S, self.ANGULAR_LIMIT_RAD_S))
        intact = tie <= self.TIE_LIMIT_M
        metrics.update(quiet=bool(quiet), maximum_link_speed_m_s=max(speeds), connections_intact=intact,
                       tie_residual_m=tie, evaluation_status="available", holding_allowed=True)
        reached = metrics["all_target_inside"] and metrics["remainder_on_system_side"] and quiet and intact
        return bool(reached), None if intact else "connection_or_volume_invalid", metrics

    def evaluate_state(self, state):
        return self.evaluate_placement(state)

    def evaluate_goal_region(self, state, region):
        """All cord vertices and the whole handle; no arbitrary loop closure."""
        if self.chain_cord:
            return self._evaluate_chain(state, region)
        return placement_candidate(state, region, self.placement_vertices,
            self.placement_targets, self.placement_remainder, soft_names=("cord",),
            attachment_names=("cord_handle_start", "cord_handle_end"))

    def goal_region_observation(self, region):
        return region.observation(self.placement_targets, self.placement_remainder)
