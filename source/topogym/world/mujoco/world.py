"""One MuJoCo scene, one physical clock and exact in-episode state restoration: GenesisWorld's interface on MuJoCo.

Clock: a World tick is physics_dt_s (2.5 ms) of `substeps` MuJoCo steps (timestep = physics_dt_s / solver_substeps), as
Genesis' solver substeps. After a tick:
  - kinematics (poses, velocities) are refreshed at the end of the tick (mj_kinematics, mj_comPos, mj_comVel);
  - contacts, contact forces and drive forces are those of the tick's last substep (its collision and constraint
    solve), as Genesis reports them;
  - both are cached as the tick's readout, which capture_state stores and restore_state puts back, so a restored
    state reads exactly as it did when it was captured.
State: mj_getState(mjSTATE_INTEGRATION) (time, qpos, qvel, act, warmstart, ctrl, applied forces, eq_active, ...) plus
the grasp welds' anchors (model.eq_data rows, which acquisition writes), their slip windows, the grasp manager's state,
the elastoplastic joints' rest orientations (model.qpos_spring rows of profile plastic, which the return mapping
writes) and a task mechanism's own state (set_mechanism). Stepping is single-threaded and deterministic: the same state
and controls give the same bits.
A non-finite state or a MuJoCo instability warning (bad qacc / qpos / qvel, after which MuJoCo resets its data) faults
the world and raises FloatingPointError, as GenesisWorld.read_state does on a non-finite state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from topogym.world.finite import require_finite

POSE_CONVENTION = "MuJoCo body (authored link) frames; wxyz; world-frame velocities, linear at the body origin"


class _RenderedCamera:
    """A camera of the renderer's rig that first shows the world's current state: SensorSuite's camera interface
    (render, res, fov, far, transform, invalidate_frame, wrist mounts through update_dynamic_camera)."""

    def __init__(self, world, camera):
        self._world, self._camera = world, camera

    @property
    def _attached_link(self):
        return getattr(self._camera, "_attached_link", None)

    @property
    def _followed_entity(self):
        return getattr(self._camera, "_followed_entity", None)

    def move_to_attach(self):
        self._world.sync_renderer()
        self._camera.move_to_attach()

    def update_following(self):
        self._world.sync_renderer()
        self._camera.update_following()

    def invalidate_frame(self):
        invalidate = getattr(self._camera, "invalidate_frame", None)
        if invalidate is not None:
            invalidate()

    def render(self, *args, **kwargs):
        self._world.sync_renderer()
        return self._camera.render(*args, **kwargs)

    def __getattr__(self, name):          # res, fov, far, near, transform, ...
        if name.startswith("__"):
            raise AttributeError(name)
        self._world.sync_renderer()
        return getattr(self._camera, name)


@dataclass(frozen=True)
class MujocoWorldState:
    world_id: str
    physics_step: int
    integration: np.ndarray          # mj_getState(..., mjSTATE_INTEGRATION)
    eq_data: np.ndarray              # model.eq_data (grasp weld anchors are written at acquisition)
    readout: dict                    # the tick's cached contacts and drive forces
    grasp_state: Any
    contact_events: Any = None       # per-substep contact evidence of tasks with contact roles (contacts.py)
    grasp_slip: Any = None           # the grasp welds' slip windows (grasp.py)
    eq_obj2id: Any = None            # model.eq_obj2id (a cloth's grasp pins are re-targeted at acquisition: grasp_cloth.py)
    plastic_rest: Any = None         # the elastoplastic joints' rest orientations (profile plastic)
    mechanism: Any = None            # a task mechanism's snapshot (set_mechanism), with the model fields it writes


class MujocoWorld:
    """GenesisWorld's lifecycle on MuJoCo: from_tabletop -> add_entities -> build -> advance / read_state / capture /
    restore -> close. The scene is built from the declarations (scene.build_scene) or loaded from a frozen scene
    directory (an instance's instance_assets/mujoco)."""

    def __init__(self, *, env_cfg, robot_cfg, physics_dt_s, substeps, seed, profile, scene=None, work_dir=None,
                 grasp_assist=None, hull_cache=None, fem_birth=None):
        self.env_cfg, self.robot_cfg = env_cfg, robot_cfg
        self.physics_dt_s = float(physics_dt_s)
        self.substeps = int(substeps)
        self.cloth_every = 1        # MuJoCo steps per outside (VBD) cloth step (profile mujoco_substeps; build)
        self.profile = profile
        self.frozen_scene = None if scene is None else Path(scene)
        self.work_dir = None if work_dir is None else Path(work_dir)
        self.grasp_assist = grasp_assist
        self.hull_cache = hull_cache
        self.fem_birth = fem_birth          # the soft entities' frozen birth (an IPC settled-state archive), new scenes only
        self._ropes = {}                    # rope entity -> rope_chain.RopeChain (profile "rope")
        self._cloths = {}                   # cloth entity -> cloth.Cloth (profile "cloth")
        self.rng = np.random.default_rng(seed)
        self.entities: dict = {}
        self._specs = []
        self.physics_step = 0
        self.terminal_contract_version = 1
        self.diagnostics_dir = None
        self._world_id = uuid4().hex
        self._built = self._closed = self._faulted = False
        self._grasp_manager = None
        self._config_finite = None          # read_state: the frozen grasp config checked finite once
        self._link_rows = {}                # read_state: entity -> (link names, body ids)
        self.model = self.data = self.manifest = self.scene_dir = None
        self._readout = None
        self._velocities = None
        # Rendering: renderer_factory() -> an object with .world.cameras (Nyx cameras of a scene that is never stepped)
        # and .update(mj_model, mj_data); the session sets it (rcb_topogym: the render bridge of the frozen instance).
        self.renderer_factory = None
        self._renderer, self._cameras = None, {}
        self._version, self._rendered_version = 0, -1
        self._contact_collector = None
        self._release_geometry = None
        self._drag = None
        self._plastic = None                # profile plastic: elastoplastic ball joints (their rest orientations are state)
        self._clamp = None                  # profile speed_clamp: extra damping on MJCF chain links above a speed / spin
        self.clamp_stats = None             # its activations since the world was built (as RopeChain.clamp_stats)
        self._attachments = ()              # tasks.base.AttachmentSpec: rope ends welded to rigid entities
        self._attached = {}                 # attachment -> (rigid body, welded link bodies, their birth offsets)
        self._mechanism = None              # a task's mechanism stepped with the physics (set_mechanism)

    @classmethod
    def from_tabletop(cls, env_cfg, robot_cfg, *, physics_options=None, compute_backend="cpu", precision="64", seed=0,
                      cameras=False, scene=None, work_dir=None, grasp_assist=None, profile=None, hull_cache=None,
                      fem_birth=None):
        from omegaconf import OmegaConf
        if compute_backend != "cpu" or str(precision) != "64":
            raise ValueError("MuJoCo physics runs on the CPU in float64")
        env_cfg = OmegaConf.create(OmegaConf.to_container(env_cfg, resolve=True))
        if int(env_cfg.sim.get("n_envs", 0)) != 0:
            raise ValueError("MujocoWorld requires one unbatched environment")
        options = dict(physics_options or {})
        if profile is None:
            raise ValueError("MujocoWorld needs the family's MuJoCo profile")
        # a rope family's FEM / IPC options configure the Genesis rope; its MuJoCo chain comes from the profile's "rope"
        ported = {"rigid"} | ({"fem", "ipc"} if profile.get("rope") is not None else set())
        # a cloth family's PBD options configure the Genesis cloth; its MuJoCo flex comes from the profile's "cloth"
        ported |= {"pbd", "pbd_table_friction"} if profile.get("cloth") is not None else set()
        if set(options) - ported:
            raise NotImplementedError(f"MuJoCo core: physics option groups {sorted(set(options) - ported)} are not ported")
        if profile.get("cloth") is not None and "pbd" in options:
            from .cloth import settings_of
            if abs(float(options["pbd"]["particle_size"]) - float(settings_of(profile)["particle_size_m"])) > 1e-12:
                raise ValueError("MuJoCo cloth: the profile's particle size differs from the task's PBD particle size")
        return cls(env_cfg=env_cfg, robot_cfg=robot_cfg, physics_dt_s=float(env_cfg.sim.dt),
                   substeps=int(env_cfg.sim.substeps), seed=seed, profile=profile, scene=scene, work_dir=work_dir,
                   grasp_assist=grasp_assist, hull_cache=hull_cache, fem_birth=fem_birth)

    @property
    def sim_time_s(self):
        return self.physics_step * self.physics_dt_s

    @property
    def cameras(self):
        """The Nyx camera rig, through a renderer that shows this world's state (docs/rcb/MUJOCO_CORE.md, Rendering):
        built on first use from renderer_factory (None: headless, no cameras)."""
        if self._renderer is None and self.renderer_factory is not None:
            self._renderer = self.renderer_factory()
            self._cameras = {name: _RenderedCamera(self, camera) for name, camera in self._renderer.world.cameras.items()}
        return self._cameras

    def sync_renderer(self):
        """Show the current state in the renderer (once per state change)."""
        if self._renderer is not None and self._rendered_version != self._version:
            self._renderer.update(self.model, self.data)
            self._rendered_version = self._version

    # -- construction
    def add_entities(self, specifications):
        self._require_open()
        if self._built:
            raise RuntimeError("Task entities must be declared before World.build")
        names = {s.name for s in self._specs}
        for spec in specifications:
            if spec.name in names or spec.name in ("robot", "table", "ground"):
                raise ValueError(f"Duplicate world entity: {spec.name}")
            names.add(spec.name)
            self._specs.append(spec)

    def add_attachments(self, specifications):
        """Permanent rope-end attachments (tasks.base.AttachmentSpec): a rope chain's end links within band_length_m of
        the endpoint welded to the rigid entity at their birth pose (scene.SceneBuilder.add_attachments). A frozen scene
        must hold the same attachments (_check_declarations)."""
        self._require_open()
        if self._built:
            raise RuntimeError("Attachments must be declared before World.build")
        self._attachments = (*self._attachments, *specifications)

    def prepare_grasp(self, config):
        if config.model == "pbd_local_patch_v1" and self.profile.get("cloth") is not None:
            return          # a cloth's grasp pins are part of its scene (cloth.add_cloth)
        raise NotImplementedError(f"MuJoCo core: grasp model {config.model} is not ported")

    def build(self):
        self._require_open()
        if self._built:
            raise RuntimeError("World.build may only be called once")
        import mujoco
        from topogym.robot.description import describe_robot
        from .scene import build_scene, load_scene
        if self.terminal_contract_version != 1:
            raise NotImplementedError("MuJoCo core: terminal contract 2 (substep contact coverage) is not ported")
        try:
            if self.frozen_scene is None:
                if self.work_dir is None:
                    raise ValueError("A new MuJoCo scene needs a work directory")
                description = describe_robot(self.robot_cfg)
                home = {g.name: list(map(float, g.home)) for g in description.groups}
                birth = {}
                if self.fem_birth is not None:
                    from .rope_chain import birth_from_archive
                    birth = {spec.name: birth_from_archive(self.fem_birth, spec.name) for spec in self._specs
                             if spec.physics_model == "fem_elastic"}
                if self.fem_birth is not None and any(spec.physics_model == "pbd_cloth" for spec in self._specs):
                    from .cloth import birth_from_instance     # a cloth's birth: the released instance's PBD particles
                    birth.update({spec.name: birth_from_instance(self.fem_birth, spec.name) for spec in self._specs
                                  if spec.physics_model == "pbd_cloth"})
                build_scene(self.work_dir, env_cfg=self.env_cfg, robot_cfg=self.robot_cfg, description=description,
                            specs=self._specs, grasp_assist=self.grasp_assist, profile=self.profile, home=home,
                            hull_cache=self.hull_cache, fem_birth=birth, attachments=self._attachments)
                self.scene_dir = self.work_dir
            else:
                self.scene_dir = self.frozen_scene
            from .robot import SELF_CONTACT_EXCLUDES
            robot_id = describe_robot(self.robot_cfg).robot_id
            self.model, self.manifest = load_scene(self.scene_dir, robot_asset_root=self.robot_cfg.robot.get("asset_root"),
                                                   excludes=SELF_CONTACT_EXCLUDES.get(robot_id, ()))
            # The exact constraint diagonal (mjENBL_DIAGEXACT) on every model, frozen scenes included: MuJoCo's
            # approximate diagonal (body / dof inverse weights at qpos0) mis-regularised contacts on the Robotiq's
            # passive four-bar and returned energy (buzz G8-G11); the regulariser then follows each row's true A_ii.
            # A profile may keep MuJoCo's approximate diagonal (profile diag_exact False: a family's own validated
            # physics, orbits' chain settings).
            if self.profile.get("diag_exact", True):
                self.model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_DIAGEXACT)
            # The robot's fingertip pads meet the other finger's pads, never its hull (robot.FINGER_PADS), frozen scenes
            # included: a pad driven into the other finger's hull made rows the exact diagonal no longer bounds (BL1).
            from .robot import separate_finger_pads, set_spring_refs
            separate_finger_pads(self.model, robot_id)
            # The robot's joint spring references (robot.SPRING_REFS), frozen scenes included: the Robotiq's spring links
            # at the MJCF's 2.62, where frozen scenes had imported them about 0 and the fingertips curled (2026-10-11).
            set_spring_refs(self.model, robot_id)
            # The contact classes as the profile builds them now, frozen scenes included (scene.migrate_contacts): the soft
            # class's constant impedance and static_contact.keep (2026-10-11).
            from .scene import migrate_contacts
            migrate_contacts(self.model, self.profile)
            # profile mujoco_substeps k (blanket_vbd_gpu, 2026-10-12): MuJoCo takes k steps of physics_dt / (substeps k)
            # per outside cloth step of physics_dt / substeps. The ALOHA's finger coupling (1.25 ms) is passive only at
            # a MuJoCo step of at most 0.625 ms (MuJoCo's refsafe clamps a time constant to 2 timesteps; at 2.5 ms it
            # returned energy: validator BLF2c / 3c / 4a, eq:1). The cloth step keeps its 2.5 ms; its colliders and pins
            # take the fresh kinematics after the k-th MuJoCo step (mj_kinematics), so they move over the cloth step
            # from the tick's start pose to its end pose (before: both from the step's start state, one step behind).
            k = int(self.profile.get("mujoco_substeps") or 1)
            if k > 1:
                self.cloth_every = k
                self.substeps *= k
                self.model.opt.timestep = self.physics_dt_s / self.substeps
            self._check_declarations()
            self.data = mujoco.MjData(self.model)
            from .views import EntityView
            fixed = {s.name: bool(s.fixed) for s in self._specs}
            for name, record in self.manifest["entities"].items():
                if record["kind"] == "pbd_cloth":
                    from .cloth import Cloth, ClothView
                    from .cloth_vbd import ClothVBD, VBD_FORMAT     # a VBD cloth (Newton) steps next to MuJoCo
                    backend = ClothVBD if record["cloth"]["format"] == VBD_FORMAT else Cloth
                    self._cloths[name] = backend(self, name, record, self.scene_dir)
                    self.entities[name] = ClothView(self, self._cloths[name])
                    continue
                self.entities[name] = EntityView(self, name, record,
                                                 fixed=fixed.get(name, record["kind"] in ("static",)))
                if record["kind"] == "rope_chain":
                    from .rope_chain import RopeChain
                    self._ropes[name] = RopeChain(self, name, record, self.scene_dir)
            for spec in self._specs:
                entity = self.entities[spec.name]
                for joint_name, value in dict(spec.initial_joint_positions).items():
                    joint = entity.get_joint(joint_name)
                    if len(joint.dofs_idx_local) != 1 or not np.isfinite(value):
                        raise ValueError("Initial scalar joint position requires one finite coordinate")
                    entity.set_dofs_position([float(value)], joint.dofs_idx_local, zero_velocity=True)
            self._built = True
            self._attached = self._attached_links()
            self._drag = self._drag_links(self.manifest["profile"].get("drag"))
            self._plastic = self._plastic_joints(self.manifest["profile"].get("plastic"))
            # a runtime rule of the family's live profile, frozen scenes included (as a rope's clamp, rope_chain.py)
            self._clamp = self._clamp_links((self.profile or {}).get("speed_clamp"))
            self._state_changed()
            for spec in self._specs:
                if spec.name in self._cloths and spec.material_map_path is not None:
                    # the cloth at its flat rest now: its particles' material coordinates, as GenesisWorld binds them
                    from topogym.world.physics.pbd import bind_material_map
                    bind_material_map(self.entities[spec.name], spec.material_map_path)
            if any(spec.contact_roles for spec in self._specs):
                from topogym.robot.description import describe_robot
                from .contacts import SubstepContactCollector, resolve
                roles, complete, parts, robot_complete = resolve(self, self._specs, self.manifest,
                                                                 describe_robot(self.robot_cfg))
                self._contact_collector = SubstepContactCollector(self, roles, geometry_complete=complete,
                                                                  robot_parts=parts,
                                                                  robot_geometry_complete=robot_complete)
        except BaseException:
            self._faulted = True
            raise

    def _check_declarations(self):
        """A frozen scene must be the scene of these declarations: the same entities at the same authored poses."""
        from scipy.spatial.transform import Rotation
        import mujoco
        declared = {s.name for s in self._specs} | {"robot", "table"} | ({"ground"} if "ground" in self.manifest["entities"] else set())
        if set(self.manifest["entities"]) != declared:
            raise ValueError(f"MuJoCo scene entities {sorted(self.manifest['entities'])} differ from the declared {sorted(declared)}")
        attached = sorted((a.name, a.soft_entity, a.rigid_entity, a.endpoint, float(a.band_length_m)) for a in self._attachments)
        built = sorted((r["name"], name, r["rigid"], r["endpoint"], float(r["band_length_m"]))
                       for name, record in self.manifest["entities"].items() if record.get("kind") == "rope_chain"
                       for r in record["rope"].get("attachments", ()))
        if attached != built:
            raise ValueError(f"MuJoCo scene attachments {built} differ from the declared {attached}")
        m = self.model
        for spec in self._specs:
            record = self.manifest["entities"][spec.name]
            if record["kind"] == "rope_chain":
                # a rope chain is the rope's frozen birth, not a body at the declared pose: the same volume and origin
                import hashlib
                rope = record["rope"]
                if (hashlib.sha256(Path(spec.volume_path).read_bytes()).hexdigest() != rope["volume_sha256"]
                        or np.max(np.abs(np.asarray(spec.position_m, dtype=float) - rope["declared_position_m"])) > 1e-9):
                    raise ValueError(f"MuJoCo rope chain of {spec.name} was built from another volume or origin")
                continue
            if record["kind"] == "pbd_cloth":
                # a cloth's vertex bodies sit on its authored sheet: the same mesh file
                import hashlib
                if hashlib.sha256(Path(spec.asset_path).read_bytes()).hexdigest() != record["cloth"]["asset_sha256"]:
                    raise ValueError(f"MuJoCo cloth of {spec.name} was built from another mesh")
                continue
            body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, record["root"])
            pos, quat = np.asarray(m.body_pos[body]), np.asarray(m.body_quat[body])
            want_q = np.asarray(spec.quaternion_wxyz, dtype=float)
            want_q = want_q / np.linalg.norm(want_q)
            want_r, want_p = Rotation.from_quat(want_q[[1, 2, 3, 0]]), np.asarray(spec.position_m, dtype=float)
            local = self.manifest["entities"][spec.name].get("root_pose")
            if local is not None:       # an MJCF entity whose first root is a free body off the entity origin
                want_p = want_p + want_r.apply(local["pos"])
                want_r = want_r * Rotation.from_quat(np.asarray(local["quat"])[[1, 2, 3, 0]])
            angle = (Rotation.from_quat(quat[[1, 2, 3, 0]]).inv() * want_r).magnitude()
            if np.max(np.abs(pos - want_p)) > 1e-9 or angle > 1e-9:
                raise ValueError(f"MuJoCo scene pose of {spec.name} differs from its declaration")

    def configure_grasp(self, robot, config):
        self._require_built()
        if self.physics_step or self._grasp_manager is not None:
            raise RuntimeError("Grasp assistance must be configured once at episode birth")
        if config.model == "none":
            return
        from topogym.world.physics.grasp_manager import GraspManager
        if config.model == "pbd_local_patch_v1" and self._cloths:
            from .grasp_cloth import MujocoClothGraspBackend, MujocoVBDClothGraspBackend
            vbd = any(getattr(c, "steps_outside", False) for c in self._cloths.values())
            backend = MujocoVBDClothGraspBackend if vbd else MujocoClothGraspBackend
            self._grasp_manager = GraspManager(backend(self, robot, config), config)
            return
        if config.model != "contact_weld_v1":
            raise NotImplementedError(f"MuJoCo core: grasp model {config.model} is not ported")
        from .grasp import MujocoGraspBackend
        self._grasp_manager = GraspManager(MujocoGraspBackend(self, robot, config), config)

    def set_mechanism(self, mechanism):
        """A task's own mechanism, stepped with the physics (TaskDefinition.mujoco_mechanism; docs/rcb/MUJOCO_CORE.md,
        "Task mechanisms"). It may switch the task's own equalities (data.eq_active, model.eq_data rows) and write model
        fields of the task's own geoms and joints (contact bits, stiffness); it never touches the robot. Its calls:
          tick(model, data)     after every tick's MuJoCo steps, before the tick's readout;
          capture() / restore(snapshot)  with every checkpoint (capture_state / restore_state): its own state and the
                                model fields it writes;
          read()                a dict of finite numbers and flags, read_state()["mechanism"].
        Set once, after build and before any tick. A world without one steps, captures and reads exactly as before."""
        self._require_built()
        if self.physics_step or self._mechanism is not None:
            raise RuntimeError("A task mechanism is set once, at episode birth")
        self._mechanism = mechanism
        self._state_changed()

    def configure_final_settling(self, rigid_targets, *, exact_robot_contact=False, fixed_flags=False):
        """Read-only release-confirmation evidence (release.py) for tasks that require release; fixed_flags: the robot
        boxes' world-fixed flags too (the active-parts policy)."""
        self._require_built()
        from .release import MujocoReleaseGeometry
        self._release_geometry = MujocoReleaseGeometry(self, rigid_targets, exact_robot_contact=exact_robot_contact,
                                                       ropes=self._ropes, fixed_flags=fixed_flags)

    def restore_cloth_births(self, declaration):
        """A declared frozen initial state (the task's InitialStateSpec: the PBD archive) on MuJoCo: every cloth at the
        birth its scene froze from that archive's released instance (cloth.add_cloth), at rest. Called after the task's
        bind, as GenesisWorld restores the archive then (topogym/world/initial_state.py)."""
        self._require_built()
        if not self._cloths:
            raise NotImplementedError("MuJoCo core: a declared initial state without a cloth is not ported")
        cloths = {}
        for name, cloth in self._cloths.items():
            birth = cloth.record["cloth"]["birth"]
            if birth.get("archive_sha256") not in (None, declaration.sha256):
                raise ValueError(f"the MuJoCo cloth {name} was born from archive {birth['archive_sha256']}, "
                                 f"the task declares {declaration.sha256}")
            cloth.apply_birth(self.data)
            cloths[name] = dict(birth)
        self._state_changed()
        return dict(codec="mujoco_cloth_birth/1", manifest_sha256=declaration.sha256, cloths=cloths)

    # -- a settled birth frozen with the instance (SimulatorConfig.mujoco_settled_state)
    SETTLED_CODEC = "mujoco_settled_state/1"

    def export_settled_state(self, path, *, settled_ticks: int) -> None:
        """Write this world's state after Env.reset's settling (reset_physics_steps ticks) to `path` (.npz): MuJoCo's
        integration state (mjSTATE_INTEGRATION: time, qpos, qvel, act, warm start, controls, ...), the equality data,
        the elastoplastic rest orientations and the scene's file hashes. restore_settled_state puts exactly these bits
        back, so a build of the frozen instance starts where the settling ended without running it again (Env.reset's
        reset_clock then recomputes the readout from them, as after the settling)."""
        import json
        import mujoco
        self._require_built()
        if self._cloths:
            raise NotImplementedError("a settled state with cloths (their tick state) is not exported")
        if self._mechanism is not None:
            raise NotImplementedError("a settled state with a task mechanism (its own state) is not exported")
        m, d = self.model, self.data
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(m, spec))
        mujoco.mj_getState(m, d, state, spec)
        np.savez(path, codec=self.SETTLED_CODEC, integration=state, eq_data=np.array(m.eq_data),
                 settled_ticks=int(settled_ticks), scene_files=json.dumps(self.manifest["files"], sort_keys=True),
                 plastic_rest=np.zeros(0) if self._plastic is None else np.array(m.qpos_spring[self._plastic["qidx"]]))

    def restore_settled_state(self, path) -> dict:
        """Restore a state written by export_settled_state on this scene (its file hashes must match)."""
        import json
        import mujoco
        self._require_built()
        with np.load(path, allow_pickle=False) as z:
            if str(z["codec"]) != self.SETTLED_CODEC:
                raise ValueError(f"{path}: not a {self.SETTLED_CODEC} archive")
            if json.loads(str(z["scene_files"])) != self.manifest["files"]:
                raise ValueError(f"{path}: settled on another MuJoCo scene")
            m, d = self.model, self.data
            m.eq_data[:] = z["eq_data"]
            if self._plastic is not None:
                m.qpos_spring[self._plastic["qidx"]] = z["plastic_rest"]
            mujoco.mj_setState(m, d, z["integration"], mujoco.mjtState.mjSTATE_INTEGRATION)
            report = dict(codec=self.SETTLED_CODEC, settled_ticks=int(z["settled_ticks"]))
        self._state_changed()
        return report

    def reset_clock(self):
        self._require_built()
        if self._faulted:
            raise RuntimeError("Cannot restart the episode clock in a faulted world")
        if self._grasp_manager is not None:
            raise RuntimeError("Configure grasp assistance after restarting the episode clock")
        self.physics_step = 0
        self._state_changed()
        if self._contact_collector is not None:
            self._contact_collector.birth()

    # -- stepping
    def advance(self):
        import mujoco
        self._require_built()
        if self._faulted:
            raise RuntimeError("World is faulted; restore an existing state before advancing")
        m, d = self.model, self.data
        try:
            if self._grasp_manager is not None:
                self._grasp_manager.before_tick(self.physics_step + 1)
            bad = sum(int(d.warning[w].number) for w in self._BAD)
            if self._drag is not None:
                self._apply_drag()
            collector = self._contact_collector
            if collector is not None:
                collector.begin(self.physics_step + 1)
            grasp = None if self._grasp_manager is None else self._grasp_manager.backend
            ropes = tuple(self._ropes.values())
            for cloth in self._cloths.values():  # the particles' velocity readout: their displacement over this tick
                cloth.tick_start = cloth.vertices(d)
            outside = tuple(c for c in self._cloths.values() if getattr(c, "steps_outside", False))
            for cloth in outside:               # a VBD cloth: its self-contact BVH rebuilt at every tick start
                cloth.tick_begin()
            plastic, clamp = self._plastic, self._clamp
            k = self.cloth_every
            for i in range(self.substeps):
                for rope in ropes:          # the rod's couples at this step's pose (rope_chain.py)
                    rope.apply(d)
                if clamp is not None:       # damping on the links above the profile's speed / spin (speed_clamp)
                    self._clamp_apply(clamp)
                if plastic is not None:     # the springs' rest orientations at this step's pose (return mapping)
                    self._plastic_return(plastic)
                if i % k == 0:
                    for cloth in outside:   # the colliders' start poses; the pins' lagged reaction on the hand
                        cloth.before_step(m, d)
                mujoco.mj_step(m, d)
                if k == 1:
                    for cloth in outside:   # one cloth step over the same dt, colliders from start to end pose
                        cloth.after_step(m, d, m.opt.timestep)
                elif (i + 1) % k == 0 and outside:
                    mujoco.mj_kinematics(m, d)          # the poses after these k steps (profile mujoco_substeps)
                    for cloth in outside:   # one cloth step over the k steps' dt
                        cloth.after_step(m, d, k * m.opt.timestep)
                if collector is not None:
                    collector.sample(i)
                if grasp is not None:
                    grasp.sample()
            if collector is not None:
                collector.end()
            if sum(int(d.warning[w].number) for w in self._BAD) != bad:
                raise FloatingPointError("MuJoCo reported an unstable state (bad qacc / qpos / qvel) and reset its data")
            if self._mechanism is not None:  # a task mechanism, after the tick's steps and before its readout
                self._mechanism.tick(m, d)
            self.physics_step += 1
            self._finish_tick()
            if self._grasp_manager is not None:
                self._grasp_manager.after_tick(self.physics_dt_s)
        except BaseException:
            self._faulted = True
            raise

    def _attached_links(self):
        """Every attachment's rigid body, welded link bodies and the links' origins in the rigid body's frame at birth
        (qpos0: the scene's body poses), for its residual in read_state."""
        import mujoco
        m = self.model
        records = [r for record in self.manifest["entities"].values() if record.get("kind") == "rope_chain"
                   for r in record["rope"].get("attachments", ())]
        if not records:
            return {}
        d0 = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d0)
        out = {}
        for r in records:
            rigid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.manifest["entities"][r["rigid"]]["root"])
            links = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in r["links"]])
            out[r["name"]] = (rigid, links, (d0.xpos[links] - d0.xpos[rigid]) @ d0.xmat[rigid].reshape(3, 3))
        return out

    def _drag_links(self, drag):
        """Profile drag {links: fnmatch pattern on body names, k}: the moving bodies it names, their trees' roots, k."""
        if not drag:
            return None
        import mujoco
        from fnmatch import fnmatchcase
        m = self.model
        bodies = [b for b in range(m.nbody) if int(m.body_weldid[b]) != 0
                  and fnmatchcase(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "", drag["links"])]
        if not bodies:
            raise ValueError(f"profile drag: {drag['links']!r} names no moving body")
        bodies = np.asarray(bodies)
        return bodies, np.asarray(m.body_rootid)[bodies], float(drag["k"])

    def _apply_drag(self):
        """Quadratic air drag on light links (profile drag): f = -k |v| v at each link's centre of mass, from the tick's
        start state (its kinematics are current: _kinematics / _state_changed), held through the tick's substeps in
        xfrc_applied (part of the integration state, so capture / restore carry it)."""
        bodies, roots, k = self._drag
        d = self.data
        cvel = np.asarray(d.cvel)[bodies]
        v = cvel[:, 3:] + np.cross(cvel[:, :3], np.asarray(d.xipos)[bodies] - np.asarray(d.subtree_com)[roots])
        d.xfrc_applied[bodies, :3] = -k * np.linalg.norm(v, axis=1)[:, None] * v

    def _clamp_links(self, clamp):
        """Profile speed_clamp {links: fnmatch pattern on body names, v_m_s, w_rad_s, rate}: the speed clamp of a rope
        (rope_chain.RopeChain._clamp) for the links of an MJCF chain, whose links hang on joints of one tree instead of
        being free bodies. The moving bodies it names, their trees' roots and dofs, and the thresholds."""
        if not clamp:
            return None
        import mujoco
        from fnmatch import fnmatchcase
        m = self.model
        bodies = [b for b in range(m.nbody) if int(m.body_weldid[b]) != 0 and m.body_mass[b] > 0
                  and fnmatchcase(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "", clamp["links"])]
        if not bodies:
            raise ValueError(f"profile speed_clamp: {clamp['links']!r} names no moving body")
        bodies = np.asarray(bodies)
        roots = np.asarray(m.body_rootid)[bodies]
        dofs = np.flatnonzero(np.isin(np.asarray(m.body_rootid)[np.asarray(m.dof_bodyid)], roots))
        self.clamp_stats = dict(substeps=0, link_substeps=0, speed_link_substeps=0, spin_link_substeps=0,
                                first_tick=None, last_tick=None, max_speed_m_s=0., max_spin_rad_s=0., loss_j=0.)
        return dict(bodies=bodies, roots=roots, dofs=dofs, mass=np.asarray(m.body_mass)[bodies].copy(),
                    inertia=np.asarray(m.body_inertia)[bodies].copy(), v=float(clamp["v_m_s"]), w=float(clamp["w_rad_s"]),
                    k=float(clamp.get("rate", .5)) / float(m.opt.timestep), qfrc=np.zeros(m.nv), written=False)

    def _clamp_apply(self, c):
        """Before every MuJoCo step: damping on the excess only, as a rope's clamp. On a link whose centre-of-mass speed
        exceeds v_m_s, a force -rate m / dt (1 - v_m_s / |v|) v at its centre of mass; on a link whose spin exceeds
        w_rad_s, a couple -rate I / dt (1 - w_rad_s / |w|) w. Both oppose the link's motion, so they only remove
        energy; they go through the tree's Jacobian into qfrc_applied (nothing else writes these dofs), where the stress
        observer books their work as dissipation. While no link exceeds a threshold it writes nothing (the dofs are
        cleared once after an activation), so a run that never reaches them is bit-identical to one without the clamp.
        mj_step leaves the kinematics at the previous step's state, so they are refreshed first (mj_step computes them
        again from qpos and qvel)."""
        import mujoco
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        bodies = c["bodies"]
        cvel = np.asarray(d.cvel)[bodies]
        w, com = cvel[:, :3], np.asarray(d.xipos)[bodies]
        v = cvel[:, 3:] + np.cross(w, com - np.asarray(d.subtree_com)[c["roots"]])
        sp, om = np.linalg.norm(v, axis=1), np.linalg.norm(w, axis=1)
        hot_v, hot_w = sp > c["v"], om > c["w"]
        if not (hot_v.any() or hot_w.any()):
            if c["written"]:                                            # the previous activation's forces
                d.qfrc_applied[c["dofs"]] = 0.
                c["written"] = False
            return
        k, q, power = c["k"], c["qfrc"], 0.
        q[:] = 0.
        R = np.asarray(d.ximat)[bodies].reshape(-1, 3, 3)               # the links' inertial frames
        for i in np.flatnonzero(hot_v | hot_w):
            F, T = np.zeros(3), np.zeros(3)
            if hot_v[i]:
                F = -k * c["mass"][i] * (1. - c["v"] / sp[i]) * v[i]
            if hot_w[i]:
                T = R[i] @ (-k * (1. - c["w"] / om[i]) * c["inertia"][i] * (R[i].T @ w[i]))
            mujoco.mj_applyFT(m, d, F, T, com[i], int(bodies[i]), q)
            power += float(F @ v[i] + T @ w[i])
        d.qfrc_applied[c["dofs"]] = q[c["dofs"]]
        c["written"] = True
        st = self.clamp_stats
        tick = self.physics_step + 1
        st["substeps"] += 1
        st["link_substeps"] += int(np.sum(hot_v | hot_w))
        st["speed_link_substeps"] += int(hot_v.sum())
        st["spin_link_substeps"] += int(hot_w.sum())
        st["first_tick"] = tick if st["first_tick"] is None else st["first_tick"]
        st["last_tick"] = tick
        st["max_speed_m_s"] = max(st["max_speed_m_s"], float(sp.max()))
        st["max_spin_rad_s"] = max(st["max_spin_rad_s"], float(om.max()))
        st["loss_j"] -= power * float(m.opt.timestep)                  # its power at this step's start velocity

    def _plastic_joints(self, plastic):
        """Profile plastic {joints: fnmatch pattern on joint names, yield_angle_deg}: elastoplastic ball joints. Each
        named joint's spring (its stiffness, toward model.qpos_spring) is elastic up to the yield angle; beyond it the
        rest orientation follows the joint (_plastic_return), so a joint bent past its yield torque (stiffness x yield
        angle) stays bent, with at most the yield angle of elastic give. The rest orientations start at the birth pose."""
        if not plastic:
            return None
        import mujoco
        from fnmatch import fnmatchcase
        m, d = self.model, self.data
        joints = [j for j in range(m.njnt)
                  if fnmatchcase(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "", plastic["joints"])]
        if not joints:
            raise ValueError(f"profile plastic: {plastic['joints']!r} names no joint")
        bad = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints
               if int(m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_BALL) or not m.jnt_stiffness[j] > 0]
        if bad:
            raise ValueError(f"profile plastic: not a ball joint with a stiffness: {bad[:3]}")
        qadr = np.asarray([int(m.jnt_qposadr[j]) for j in joints])
        qidx = (qadr[:, None] + np.arange(4)).ravel()
        # one block of qpos when the joints are consecutive (a cable's joints are): views instead of gathers
        block = slice(int(qidx[0]), int(qidx[-1]) + 1) if np.array_equal(qidx, np.arange(qidx[0], qidx[-1] + 1)) else qidx
        m.qpos_spring[qidx] = d.qpos[qidx]  # rest = the birth orientation: no spring force at birth
        half = np.radians(float(plastic["yield_angle_deg"])) / 2
        return dict(qidx=qidx, block=block, rest_of={int(j): int(a) for j, a in zip(joints, qadr)},
                    cos=float(np.cos(half)), sin=float(np.sin(half)))

    def _plastic_return(self, p):
        """The return mapping, before every MuJoCo step: where a joint has turned more than the yield angle away from
        its rest orientation, the rest orientation follows to exactly the yield angle behind it, about the same axis
        (the spring then pulls with its yield torque). rel = conj(rest) * q is a unit quaternion whose scalar part is
        the dot product rest . q, and the turn exceeds the yield angle exactly when |rest . q| < cos(yield / 2).
        Element-wise products, sums and square roots only, in a fixed order, so every host computes the same bits."""
        m, d = self.model, self.data
        block = p["block"]
        q = d.qpos[block].reshape(-1, 4)
        r = m.qpos_spring[block].reshape(-1, 4)
        w = r[:, 0] * q[:, 0] + r[:, 1] * q[:, 1] + r[:, 2] * q[:, 2] + r[:, 3] * q[:, 3]
        hit = np.abs(w) < p["cos"]
        if not hit.any():
            return
        q, r, w = q[hit], r[hit], w[hit]
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        rw, rx, ry, rz = r[:, 0], -r[:, 1], -r[:, 2], -r[:, 3]          # conj(rest)
        x = rw * qx + rx * qw + ry * qz - rz * qy                         # rel = conj(rest) * q, in the rest frame
        y = rw * qy - rx * qz + ry * qw + rz * qx
        z = rw * qz + rx * qy - ry * qx + rz * qw
        s = np.sqrt(x * x + y * y + z * z) * np.where(w < 0, -1., 1.)    # the shortest turn's axis: (x, y, z) / s
        ax, ay, az = x / s, y / s, z / s
        c, sn = p["cos"], p["sin"]                                        # new rest = q * quat(axis, -yield angle)
        nw = qw * c + qx * ax * sn + qy * ay * sn + qz * az * sn
        nx = -qw * ax * sn + qx * c - qy * az * sn + qz * ay * sn
        ny = -qw * ay * sn + qx * az * sn + qy * c - qz * ax * sn
        nz = -qw * az * sn - qx * ay * sn + qy * ax * sn + qz * c
        n = np.sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
        rest = m.qpos_spring[block].reshape(-1, 4).copy()
        rest[hit] = np.stack([nw, nx, ny, nz], axis=1) / n[:, None]
        m.qpos_spring[block] = rest.ravel()

    @property
    def _BAD(self):
        import mujoco
        w = mujoco.mjtWarning
        return (w.mjWARN_BADQACC, w.mjWARN_BADQPOS, w.mjWARN_BADQVEL)

    def _finish_tick(self):
        """The tick's readout: the last substep's contacts and drive forces; then end-of-tick kinematics."""
        self._readout = self._read_contacts()
        self._kinematics()

    def _kinematics(self):
        self._version += 1
        import mujoco
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        if self._cloths:
            mujoco.mj_flex(m, d)        # the flex vertices at the tick's end (the render bridge's bind_flex reads them)
        self._velocities = None

    def _state_changed(self):
        """After a state change outside stepping (build, joint assignment, reset_clock): a forward pass gives the
        readout of the current state."""
        import mujoco
        mujoco.mj_forward(self.model, self.data)
        self._readout = self._read_contacts()
        self._velocities = None
        self._version += 1

    def _read_contacts(self) -> dict:
        import mujoco
        m, d = self.model, self.data
        n = int(d.ncon)
        c = d.contact
        if n:
            keep = np.flatnonzero(np.asarray(c.exclude[:n]) == 0)
            if self._cloths:
                # a cloth's own contacts (its spheres, its flex) are not rigid contacts: Genesis reports none for PBD
                keep = keep[self._rigid_contacts(c, n, keep)]
            g1, g2 = np.asarray(c.geom1[:n])[keep], np.asarray(c.geom2[:n])[keep]
            frames = np.asarray(c.frame[:n]).reshape(-1, 3, 3)[keep]
            local = np.zeros((len(keep), 3))
            f6 = np.zeros(6)
            for k, i in enumerate(keep):
                mujoco.mj_contactForce(m, d, int(i), f6)
                local[k] = f6[:3]
            force = np.einsum("nij,ni->nj", frames, local)       # contact frame rows -> world: on geom2's body
            out = dict(link_a=np.asarray(m.geom_bodyid[g1], dtype=np.int64),
                       link_b=np.asarray(m.geom_bodyid[g2], dtype=np.int64),
                       geom_a=g1.astype(np.int64), geom_b=g2.astype(np.int64),
                       force=force, penetration=-np.asarray(c.dist[:n], dtype=np.float64)[keep],
                       position=np.asarray(c.pos[:n], dtype=np.float64)[keep].copy())
        else:
            out = dict(link_a=np.zeros(0, np.int64), link_b=np.zeros(0, np.int64), geom_a=np.zeros(0, np.int64),
                       geom_b=np.zeros(0, np.int64), force=np.zeros((0, 3)), penetration=np.zeros(0),
                       position=np.zeros((0, 3)))
        return dict(contacts=out, qfrc_actuator=np.array(d.qfrc_actuator, dtype=np.float64))

    def _rigid_contacts(self, c, n, keep) -> np.ndarray:
        """Mask over `keep`: the contacts that touch no cloth (no flex side, no cloth vertex body)."""
        m = self.model
        if not hasattr(self, "_cloth_body"):
            self._cloth_body = np.zeros(m.nbody, dtype=bool)
            for cloth in self._cloths.values():
                self._cloth_body[cloth.bodies] = True
        flex = np.asarray(c.flex[:n]).reshape(n, 2)[keep]
        geoms = np.stack([np.asarray(c.geom1[:n])[keep], np.asarray(c.geom2[:n])[keep]], axis=1)
        body = np.where(geoms >= 0, np.asarray(m.geom_bodyid)[np.maximum(geoms, 0)], 0)
        return (flex < 0).all(axis=1) & ~self._cloth_body[body].any(axis=1)

    def contacts(self) -> dict:
        """The tick's contacts (the last substep's): link_a / link_b (body ids = link idx), force (world, on link_b),
        penetration (m, > 0 overlapping), position (world)."""
        return self._readout["contacts"]

    def qfrc_actuator(self) -> np.ndarray:
        return self._readout["qfrc_actuator"]

    def body_velocities(self):
        """(angular, linear) world-frame velocities of every body, the linear one at the body origin."""
        if self._velocities is None:
            d, m = self.data, self.model
            cvel = np.asarray(d.cvel)
            com = np.asarray(d.subtree_com)[np.asarray(m.body_rootid)]
            ang = cvel[:, :3].copy()
            lin = cvel[:, 3:] + np.cross(ang, np.asarray(d.xpos) - com)
            self._velocities = (ang, lin)
        return self._velocities

    # -- readout
    def read_state(self):
        self._require_built()
        d = self.data
        ang, lin = self.body_velocities()
        bodies, articulations = {}, {}
        for name, entity in self.entities.items():
            if name in self._ropes:         # a rope reads as its FEM view (below), as on Genesis
                continue
            if name in self._cloths:        # a cloth reads as its particles (below), as on Genesis
                continue
            def pose(link):
                b = link.idx
                return dict(position_m=np.array(d.xpos[b]), quaternion_wxyz=np.array(d.xquat[b]),
                            linear_velocity_m_s=lin[b].copy(), angular_velocity_rad_s=ang[b].copy())
            bodies[name] = dict(pose(entity.base_link), link_indices=tuple(l.idx for l in entity.links))
            q, v = entity.get_qpos(), entity.get_dofs_velocity()
            joints = {j.name: dict(position=q[j.qs_idx_local].copy(), velocity=v[j.dofs_idx_local].copy())
                      for j in entity.joints}
            if self._plastic is not None:   # an elastoplastic joint's rest orientation is part of the state
                for j in entity.joints:
                    a = self._plastic["rest_of"].get(j.mj_id)
                    if a is not None:
                        joints[j.name]["rest"] = np.array(self.model.qpos_spring[a:a + 4])
            # every link pose of the entity from one gather per array (rows of fresh arrays: the same values as a
            # copy per link; speed lane, 2026-10-13)
            names, idx = self._link_rows.get(name) or self._link_rows.setdefault(
                name, ([l.name for l in entity.links], np.array([l.idx for l in entity.links], dtype=np.int64)))
            P, Q, L, A = np.asarray(d.xpos)[idx], np.asarray(d.xquat)[idx], lin[idx], ang[idx]
            articulations[name] = dict(joints=joints, links={
                nm: dict(position_m=P[k], quaternion_wxyz=Q[k], linear_velocity_m_s=L[k], angular_velocity_rad_s=A[k])
                for k, nm in enumerate(names)})
        contacts = {k: v.copy() for k, v in self.contacts().items()}
        fem, ipc = {}, {}
        if self._ropes:
            # the ropes' FEM view (skinned from the chains) and their integrity evidence (rigid links: volume ratio 1)
            fem = {name: rope.fem_state(d, (ang, lin)) for name, rope in self._ropes.items()}
            # an attachment's residual: how far its welded link origins moved off their birth offsets in the rigid body
            residual = {name: float(np.linalg.norm((d.xpos[links] - d.xpos[rigid]) @ d.xmat[rigid].reshape(3, 3) - rest,
                                                   axis=1).max()) for name, (rigid, links, rest) in self._attached.items()}
            ipc = dict(native_valid=True, volume_ratio={name: rope.integrity() for name, rope in self._ropes.items()},
                       attachment_residual_m=residual)
        particles = {name: cloth.particle_state(d, self.physics_dt_s) for name, cloth in self._cloths.items()}
        result = dict(bodies=bodies, particles=particles, fem=fem, articulations=articulations, contacts=contacts, ipc=ipc,
                      pose_convention=POSE_CONVENTION, physics_step=self.physics_step, sim_time_s=self.sim_time_s,
                      assist_active=False, assist_model="none")
        if self._grasp_manager is not None:
            grasp = self._grasp_manager.read()
            result.update(assist_active=grasp["active"], assist_model=self._grasp_manager.config.model, grasp=grasp)
        if self._release_geometry is not None:
            result["release_geometry"] = self._release_geometry.read()
        if self._contact_collector is not None:
            result["contact_events"] = self._contact_collector.read()
        if self._mechanism is not None:
            result["mechanism"] = self._mechanism.read()
        result["terminal_contract_version"] = self.terminal_contract_version
        try:
            if self._sources_finite(lin, ang):
                # bodies / articulations hold only copies of these finite arrays (poses, joints, rest orientations),
                # ints and names: walking the rest gives the same verdict, and on a failure the same first bad path
                rest = {k: v for k, v in result.items() if k not in ("bodies", "articulations")}
                if "grasp" in rest and self._grasp_config_finite(rest["grasp"]["config"]):
                    # the grasp config is frozen (the same values every tick), checked once
                    rest["grasp"] = {k: v for k, v in rest["grasp"].items() if k != "config"}
                require_finite(rest)
            else:
                require_finite(result)
        except FloatingPointError:
            self._faulted = True
            raise
        return result

    def _grasp_config_finite(self, config) -> bool:
        if self._config_finite is None:
            try:
                require_finite(config)
                self._config_finite = True
            except FloatingPointError:
                self._config_finite = False
        return self._config_finite

    def _sources_finite(self, lin, ang) -> bool:
        """Whether every array read_state copies its bodies and articulations from is finite (a few whole-array checks
        instead of the per-tick walk over hundreds of small copies; the walk stays the verdict when one is not)."""
        d = self.data
        ok = all(np.isfinite(a).all() for a in (d.xpos, d.xquat, lin, ang, d.qpos, d.qvel))
        if ok and self._plastic is not None:
            ok = bool(np.isfinite(self.model.qpos_spring).all())
        return bool(ok)

    # -- checkpoints
    def capture_state(self):
        import mujoco
        self._require_built()
        if self._faulted:
            raise RuntimeError("Cannot capture a partially failed physical step")
        m, d = self.model, self.data
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(m, spec))
        mujoco.mj_getState(m, d, state, spec)
        readout = dict(contacts={k: v.copy() for k, v in self._readout["contacts"].items()},
                       qfrc_actuator=self._readout["qfrc_actuator"].copy())
        if self._cloths:
            readout["cloth_tick_start"] = {name: None if c.tick_start is None else c.tick_start.copy()
                                           for name, c in self._cloths.items()}
            outside = {name: c.capture() for name, c in self._cloths.items() if getattr(c, "steps_outside", False)}
            if outside:
                readout["cloth_vbd"] = outside
        return MujocoWorldState(self._world_id, self.physics_step, state, np.array(m.eq_data), readout,
                                None if self._grasp_manager is None else self._grasp_manager.capture(),
                                None if self._contact_collector is None else self._contact_collector.read(),
                                None if self._grasp_manager is None else self._grasp_manager.backend.capture_slip(),
                                eq_obj2id=np.array(m.eq_obj2id) if self._cloths else None,
                                plastic_rest=None if self._plastic is None else np.array(m.qpos_spring[self._plastic["qidx"]]),
                                mechanism=None if self._mechanism is None else self._mechanism.capture())

    def restore_state(self, state):
        import mujoco
        self._require_built()
        if not isinstance(state, MujocoWorldState) or state.world_id != self._world_id:
            raise ValueError("WorldState belongs to another built world")
        if (state.grasp_state is None) != (self._grasp_manager is None):
            raise ValueError("WorldState grasp configuration mismatch")
        try:
            m, d = self.model, self.data
            m.eq_data[:] = state.eq_data
            if state.eq_obj2id is not None:
                m.eq_obj2id[:] = state.eq_obj2id
            if self._plastic is not None:
                if state.plastic_rest is None:
                    raise ValueError("WorldState lacks the elastoplastic joints' rest orientations")
                m.qpos_spring[self._plastic["qidx"]] = state.plastic_rest
            if self._mechanism is not None:
                if state.mechanism is None:
                    raise ValueError("WorldState lacks the task mechanism's state")
                self._mechanism.restore(state.mechanism)
            mujoco.mj_setState(m, d, state.integration, mujoco.mjtState.mjSTATE_INTEGRATION)
            # The instability counters restart at zero, as mj_resetData does: they lie outside mjSTATE_INTEGRATION, and
            # MuJoCo's auto-reset after a bad state zeroes them and then counts once, so a count kept from an earlier fault
            # hid the next one (1 -> 1) and the episode went on from qpos0.
            for w in self._BAD:
                d.warning[w].number = 0
            self._kinematics()
            self._readout = dict(contacts={k: v.copy() for k, v in state.readout["contacts"].items()},
                                 qfrc_actuator=state.readout["qfrc_actuator"].copy())
            for name, start in state.readout.get("cloth_tick_start", {}).items():
                self._cloths[name].tick_start = None if start is None else start.copy()
            for name, cloth_state in state.readout.get("cloth_vbd", {}).items():
                self._cloths[name].restore(cloth_state)
            self.physics_step = state.physics_step
            if self._grasp_manager is not None:
                self._grasp_manager.restore(state.grasp_state)
                self._grasp_manager.backend.restore_slip(state.grasp_slip)
            if self._contact_collector is not None:
                if state.contact_events is None:
                    raise ValueError("WorldState lacks the contact evidence of this world")
                self._contact_collector.restore(state.contact_events)
            self._faulted = False
        except BaseException:
            self._faulted = True
            raise

    def discard_states(self, states):
        pass

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._renderer is not None:
            self._renderer.close()
            self._renderer, self._cameras = None, {}
        self.entities.clear()
        self.model = self.data = None

    def _require_open(self):
        if self._closed:
            raise RuntimeError("World is closed")

    def _require_built(self):
        self._require_open()
        if not self._built:
            raise RuntimeError("World has not been built")
