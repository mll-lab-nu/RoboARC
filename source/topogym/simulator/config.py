"""The single simulator input configuration; task-specific values stay explicit."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Mapping, Any

from topogym.utils.snapshots import freeze
from topogym.utils.grasp import (ForceClosureGraspTarget, GraspAssistConfig, GraspTarget, MultiRopeGraspAssistConfig,
                                 SoftGraspTarget, WeakHoldGraspTarget)


# "segment" (default): a complete checkpoint at the episode start and at each SegmentRunner segment start, a light
# record (tick identity, robot targets, task state) at every other tick; rewind reaches only those checkpoints.
# "memory" / "disk": a complete checkpoint at every physics tick (in memory / pickled to a private temporary
# directory); rewind reaches any tick. Callers that rewind to arbitrary ticks select "memory".
DEFAULT_HISTORY_STORAGE = "segment"
HISTORY_STORAGES = {"segment", "memory", "disk"}


# RigidOptions fields an instance may override, with their types. use_contact_island=False: one global constraint solve
# instead of per-island solves; on the CPU backend 1.3x faster (2026-10-02 study).
RIGID_OPTION_OVERRIDES = {"use_contact_island": bool}
# IPCCouplerOptions fields an instance may override (2026-10-02 study: Newton 1e-3 / PCG 1e-5 with one solver substep;
# libuipc's linear_bvh broad phase instead of its default info_stackless_bvh: the same contacts, ~10 % faster; tooling
# v18 knot K2: a semi-implicit Newton solve with a minimum number of iterations per frame).
IPC_OPTION_OVERRIDES = {"newton_tolerance": float, "linear_system_tolerance": float, "collision_detection_method": str,
                        "newton_semi_implicit_enable": bool, "newton_min_iterations": int}
IPC_BROAD_PHASES = ("linear_bvh",)


@dataclass(frozen=True)
class SimulatorConfig:
    task_id: str = "cast_coil"
    robot_id: str = "auto"
    variation_id: str = "base"
    task_parameters: Mapping[str, Any] = field(default_factory=dict)
    compute_backend: str = "cuda"
    precision: str = "64"
    physics_dt_s: float = .0025
    solver_substeps: int = 4
    terminal_contract_version: int = 1
    control_decimation: int = 10
    max_physics_steps: int | None = 10000
    history_storage: str = field(default_factory=lambda: DEFAULT_HISTORY_STORAGE)
    reset_physics_steps: int = 0
    # Qualified overrides of the task's Genesis RigidOptions (RIGID_OPTION_OVERRIDES); merged in Env.reset.
    rigid_options: Mapping[str, Any] = field(default_factory=dict)
    # Qualified overrides of the task's IPCCouplerOptions (IPC_OPTION_OVERRIDES); merged in Env.reset.
    ipc_options: Mapping[str, Any] = field(default_factory=dict)
    # A settled IPC scene frozen with the instance (world/physics/ipc_settled_state.py): restored at every build instead
    # of settling for reset_physics_steps ticks.
    ipc_settled_state: str | None = None
    enable_rgb: bool = True
    enable_depth: bool = False
    observation_mode: str = "JOINT"
    control_mode: str = "JOINT_ABS"
    enable_privileged: bool = False
    seed: int | None = None
    render_backend: str = "nyx"
    asset_root: Path | None = None
    work_dir: Path | None = None
    grasp_assist: GraspAssistConfig = field(default_factory=GraspAssistConfig)
    # The physics engine: "genesis", or "mujoco" for a family ported to topogym.world.mujoco (CPU float64). A MuJoCo
    # instance freezes its scene (MJCF, meshes, CoACD hulls) as a directory; mujoco_scene names it (None: build it).
    physics_engine: str = "genesis"
    mujoco_scene: str | None = None
    # A new MuJoCo scene's soft-body birth (topogym/world/mujoco/rope_chain.py): an IPC settled-state archive (manifest)
    # whose FEM vertices give each rope chain its birth shape. Read only when the scene is built (make-instance).
    mujoco_birth: str | None = None
    # A MuJoCo instance's settled birth (MujocoWorld.export_settled_state, written by make-instance after settling for
    # the family's reset_physics_steps): restored at every build instead of settling again (reset_physics_steps 0).
    mujoco_settled_state: str | None = None
    # A MuJoCo profile chosen by name (topogym/world/mujoco/profiles.py NAMED_PROFILES) instead of the family's default.
    mujoco_profile: str | None = None

    def __post_init__(self):
        if type(self.terminal_contract_version) is not int or self.terminal_contract_version not in (1, 2):
            raise ValueError("terminal_contract_version must be 1 (tick) or 2 (rigid substeps)")
        if self.terminal_contract_version == 2 and self.task_id not in {
                "cast_coil", "cast_keyhole", "hedgehog_in_cage", "horseshoe_ring", "interlocked_claws"}:
            raise ValueError("terminal contract 2 is not qualified for this task/backend")
        if self.history_storage not in HISTORY_STORAGES:
            raise ValueError("history_storage must be segment, memory or disk")
        if isinstance(self.grasp_assist, Mapping):
            data = dict(self.grasp_assist)
            try:
                target_type = SoftGraspTarget if data.get("model") in {"ipc_soft_band_v1", "pbd_local_patch_v1"} else GraspTarget
                # the opt-in subclasses (topogym/utils/grasp.py) are read back by their own field names
                data["targets"] = tuple((WeakHoldGraspTarget if "weak_hold_travel_m" in v else
                                         ForceClosureGraspTarget if "friction_coefficient" in v else target_type)(**dict(v))
                                        if isinstance(v, Mapping) else v for v in data.get("targets", ()))
                config_type = MultiRopeGraspAssistConfig if "weld_each_rope" in data else GraspAssistConfig
                object.__setattr__(self, "grasp_assist", config_type(**data))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid grasp_assist configuration: {error}") from error
        if not isinstance(self.grasp_assist, GraspAssistConfig):
            raise ValueError("grasp_assist must be a GraspAssistConfig")
        if self.observation_mode not in {"JOINT", "EE"}:
            raise ValueError("observation_mode must be JOINT or EE")
        if self.control_mode not in {"JOINT_ABS", "JOINT_DELTA", "EE_ABS", "EE_DELTA"}:
            raise ValueError("Unsupported control_mode")
        if not isinstance(self.enable_privileged, bool):
            raise ValueError("enable_privileged must be a boolean")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**32):
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a nonempty registered name")
        if self.robot_id not in {"auto", "aloha", "panda_robotiq"}:
            raise ValueError("robot_id must be auto, aloha or panda_robotiq")
        if not isinstance(self.variation_id, str) or not self.variation_id:
            raise ValueError("variation_id must be a nonempty name")
        if self.compute_backend not in {"cpu", "cuda"}:
            raise ValueError("compute_backend must be cpu or cuda")
        if self.precision not in {"32", "64"}:
            raise ValueError("precision must be '32' or '64'")
        if self.render_backend != "nyx":
            raise ValueError("Simulator RGB uses Nyx with compact_lab; render_backend must be nyx")
        if isinstance(self.physics_dt_s, bool) or not math.isfinite(self.physics_dt_s) or self.physics_dt_s <= 0:
            raise ValueError("physics_dt_s must be finite and positive")
        for name in ("solver_substeps", "control_decimation", "max_physics_steps", "reset_physics_steps"):
            value = getattr(self, name)
            if name == "max_physics_steps" and value is None:
                continue
            minimum = 0 if name == "reset_physics_steps" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not isinstance(self.enable_rgb, bool):
            raise ValueError("enable_rgb must be a boolean")
        if type(self.enable_depth) is not bool or self.enable_depth and not self.enable_rgb:
            raise ValueError("Depth requires enable_rgb=True and a boolean enable_depth")
        # Nyx renders on a CUDA device for either physics backend; with CPU physics the plugin copies the visual
        # state to its CUDA buffers per render (2026-10-02 study: same frames up to renderer noise).
        options = dict(self.rigid_options)
        unknown = set(options) - set(RIGID_OPTION_OVERRIDES)
        if unknown or any(type(options[k]) is not RIGID_OPTION_OVERRIDES[k] for k in options):
            raise ValueError(f"rigid_options must use qualified overrides {RIGID_OPTION_OVERRIDES}; got {options}")
        object.__setattr__(self, "rigid_options", freeze(options))
        options = dict(self.ipc_options)
        unknown = set(options) - set(IPC_OPTION_OVERRIDES)
        valid = dict(collision_detection_method=lambda v: v in IPC_BROAD_PHASES, newton_semi_implicit_enable=lambda v: True)
        if unknown or any(type(v) is not IPC_OPTION_OVERRIDES[k]
                          or not valid.get(k, lambda v: math.isfinite(v) and v > 0)(v) for k, v in options.items()):
            raise ValueError(f"ipc_options must use qualified overrides {IPC_OPTION_OVERRIDES}; got {options}")
        object.__setattr__(self, "ipc_options", freeze(options))
        if self.ipc_settled_state is not None and (not isinstance(self.ipc_settled_state, str) or self.reset_physics_steps
                                                   or self.compute_backend != "cuda"):
            raise ValueError("ipc_settled_state must be a manifest path, with reset_physics_steps 0 on CUDA")
        if self.physics_engine not in {"genesis", "mujoco"}:
            raise ValueError("physics_engine must be genesis or mujoco")
        if self.physics_engine == "mujoco" and (self.compute_backend != "cpu" or self.precision != "64"
                                                or self.ipc_options or self.ipc_settled_state is not None):
            raise ValueError("MuJoCo physics runs on the CPU in float64, without IPC")
        if self.mujoco_scene is not None and (self.physics_engine != "mujoco" or not isinstance(self.mujoco_scene, str)):
            raise ValueError("mujoco_scene must be a scene directory of a MuJoCo instance")
        if self.mujoco_birth is not None and (self.physics_engine != "mujoco" or not isinstance(self.mujoco_birth, str)):
            raise ValueError("mujoco_birth must be a settled-state manifest of a MuJoCo instance")
        if self.mujoco_settled_state is not None and (self.physics_engine != "mujoco" or self.reset_physics_steps
                                                      or not isinstance(self.mujoco_settled_state, str)):
            raise ValueError("mujoco_settled_state must be a MuJoCo instance's settled-state archive, with "
                             "reset_physics_steps 0")
        if self.mujoco_profile is not None and (self.physics_engine != "mujoco" or not isinstance(self.mujoco_profile, str)):
            raise ValueError("mujoco_profile names a MuJoCo profile")
        object.__setattr__(self, "task_parameters", freeze(dict(self.task_parameters)))
        for name in ("asset_root", "work_dir"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())
