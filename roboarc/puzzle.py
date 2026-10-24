"""The RoboARC puzzles (MuJoCo 3.14.0, the tooling rcb_topogym.Session) as a RoboARCEnv.

Physics, limits and success are the graded tooling's own: the environment holds one rcb_topogym.Session built from the
task's frozen instance, as the privileged task, its verifier and the standard sim service do, and every control step
is Session.step(row). The only rule this module adds is the standard mode's joint-speed rule (rules="standard"), with
the standard service's own code path (environment/sim/server.py, run_joint_position)."""
from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from ._runtime import (IMAGE_INSTANCES, IMAGE_STAGE, IMAGE_TOPOGYM, RuntimeMismatch, cache_dir, check_graded_runtime,
                       in_puzzle_image, repo_root)
from .core import RoboARCEnv, warn_or_raise
from .registry import TaskInfo

_SESSION_BUILT = []          # one Session per process (the tooling's rule)


def _import_tooling(root: Path):
    """rcb_topogym from the repository's pinned copy (scripts/roboarc/shared/, the bytes every task dir holds: the runtime
    binding pins its sha256), with topogym from the image (/opt/topogym) or ROBOARC_TOPOGYM."""
    topogym = os.environ.get("ROBOARC_TOPOGYM") or (str(IMAGE_TOPOGYM) if IMAGE_TOPOGYM.is_dir() else None)
    if topogym and topogym not in sys.path:
        sys.path.append(topogym)
    shared = str(root / "scripts" / "roboarc" / "shared")
    if shared not in sys.path:
        sys.path.insert(0, shared)
    import rcb_topogym
    if Path(rcb_topogym.__file__).resolve().parent != Path(shared).resolve():
        raise RuntimeMismatch(f"rcb_topogym was imported from {rcb_topogym.__file__}, not from the repository's {shared}")
    return rcb_topogym


def stage_instance(task: TaskInfo, *, bind: bool = True, seed: int = 0) -> Path:
    """The task's frozen instance of instance seed `seed` in a fresh per-process directory under the cache (instance.json,
    instance_assets/ and, with bind=True, the seed's runtime_binding.json, which makes the Session check the whole
    runtime): staged and checked by the image's stage_instance.py (as the task Dockerfiles do; `--seed-file` for a seed
    other than 0), or copied from ROBOARC_INSTANCES/<base>/ (a seed S > 0: <base>/seeds/<S>/)."""
    seed = int(seed)
    if seed not in task.seeds:
        raise ValueError(f"{task.id} allows the instance seeds {list(task.seeds)}, not {seed}")
    root = cache_dir() / "instances"
    root.mkdir(parents=True, exist_ok=True)
    dest = Path(tempfile.mkdtemp(prefix=f"{task.base}-s{seed}-", dir=root))
    atexit.register(shutil.rmtree, dest, True)
    env_dir = Path(task.task_dirs["privileged"]) / "environment"
    binding = env_dir / ("runtime_binding.json" if seed == 0 else f"runtime_binding.s{seed}.json")
    rel = Path(task.base) if seed == 0 else Path(task.base) / "seeds" / str(seed)
    local = os.environ.get("ROBOARC_INSTANCES")
    if local:
        src = Path(local) / rel
        if not (src / "instance.json").is_file():
            raise FileNotFoundError(f"no frozen instance for {task.base} seed {seed} under ROBOARC_INSTANCES={local}")
        shutil.copyfile(src / "instance.json", dest / "instance.json")
        shutil.copytree(src / "instance_assets", dest / "instance_assets", symlinks=True)
        if bind:
            shutil.copyfile(binding, dest / "runtime_binding.json")
        return dest / "instance.json"
    if not in_puzzle_image():
        raise RuntimeMismatch(f"no frozen instances here ({IMAGE_INSTANCES} is the image's): run inside "
                              f"{task.image}, or set ROBOARC_INSTANCES to a local copy (instances/<base>/ per base)")
    if not bind:     # stage_instance.py checks against a binding; without one, copy the files as it does
        shutil.copyfile(IMAGE_INSTANCES / rel / "instance.json", dest / "instance.json")
        shutil.copytree(IMAGE_INSTANCES / rel / "instance_assets", dest / "instance_assets", symlinks=True)
        return dest / "instance.json"
    cmd = [sys.executable, str(IMAGE_STAGE), task.base, str(dest), "--binding", str(env_dir / "runtime_binding.json")]
    if seed:         # the seed's instance against its own binding (runtime_binding.s<S>.json beside --binding)
        (dest / "seed").write_text(f"{seed}\n")
        cmd += ["--seed-file", str(dest / "seed")]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeMismatch(f"staging the frozen instance of {task.base} seed {seed} failed: "
                              f"{(p.stderr or p.stdout).strip()[-800:]}")
    return dest / "instance.json"


def _render_in_cache(world, base: str) -> None:
    """Build the world's Nyx rig (topogym builds it on the first frame, through world.renderer_factory) with the working
    directory set to a private scratch directory under the cache, then return to the caller's working directory. The
    tooling's Nyx export writes its scene folder (__nyx_cache__/) relative to the working directory, which may be
    read-only (a mounted checkout) or shared; the files stay in the scratch directory, which is removed at exit."""
    factory = getattr(world, "renderer_factory", None)
    if factory is None:
        return
    scratch = Path(tempfile.mkdtemp(prefix=f"render-{base}-", dir=cache_dir()))
    atexit.register(shutil.rmtree, scratch, True)

    def build():
        here = os.getcwd()
        os.chdir(scratch)
        try:
            return factory()
        finally:
            os.chdir(here)

    world.renderer_factory = build


def _qmul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


class PuzzleEnv(RoboARCEnv):
    """A RoboARC puzzle task. `session` is the underlying rcb_topogym.Session (the privileged tooling: objects(),
    primitives(), safety(), ... are all there); stepping it directly bypasses this environment's bookkeeping."""

    def __init__(self, task: TaskInfo, *, renderer="nyx", cameras=None, depth=False, image_size=None, render_mode=None,
                 rules="standard", max_episode_steps=None, strict=True, instance_seed: int = 0):
        super().__init__(task, renderer=renderer, cameras=cameras, depth=depth, image_size=image_size,
                         render_mode=render_mode, max_episode_steps=max_episode_steps, strict=strict)
        if _SESSION_BUILT:
            raise RuntimeError("one RoboARC puzzle environment per process (the simulator is one scene per process); "
                               "use subprocess vector environments for several")
        self.rules = rules
        root = repo_root()
        problems = check_graded_runtime(strict=False)
        if not in_puzzle_image() and not os.environ.get("ROBOARC_INSTANCES"):
            problems.append(f"not inside the puzzle image {task.image} (no {IMAGE_INSTANCES})")
        warn_or_raise(problems, strict, RuntimeMismatch)
        # what the tooling reads at import / at its first frame: no agent-side session log; rendering off unless Nyx
        os.environ.setdefault("RCB_SESSION_LOG", "")
        if renderer == "nyx":
            if os.environ.get("RCB_RENDER") == "off":
                raise ValueError("RCB_RENDER=off is set in the environment: pass renderer='none' instead, or unset it")
        else:
            os.environ["RCB_RENDER"] = "off"       # Nyx never starts; the camera rig and calibration stay
        M = self.M = _import_tooling(root)
        self.instance_seed = int(instance_seed)
        instance = stage_instance(task, bind=strict, seed=self.instance_seed)
        _SESSION_BUILT.append(task.id)
        S = self.session = M.Session(task.name, instance=str(instance), strict=strict, max_steps=self.max_episode_steps)
        generated = bool(getattr(S, "instance_generated", False))
        realized = bool(getattr(S, "instance_realized", False))
        if strict and (not S.instance_matches or not (S.instance_exact or generated or realized) or not S.runtime_binding):
            raise RuntimeMismatch("the frozen instance did not build exactly (the verifier's condition: instance_matches, "
                                  "exact or realized, runtime binding checked)")
        if renderer == "nyx":
            _render_in_cache(S.env._world, task.base)
        self.instance_path = instance
        self.birth_hash = S.instance_sha
        self.instruction = str(S.spec["text"])     # the task's public goal text (the standard mode's task_text)
        self.control_hz = float(S.control_hz)
        self.physics_substeps = int(S.physics_substeps)
        r = S.robot
        self.arms = list(r.arms)
        self.action_layout = list(S.action_layout)
        self.limits = np.array([[float(l[0]), float(l[1])] for l in r.limits], dtype=np.float64)
        grips = [r.grip_index[a] for a in self.arms]
        self._grip_cols = grips
        self._arm_cols = np.array([j for j in range(len(self.action_layout)) if j not in grips])
        self.joint_step_limit_rad = float(M.JOINT_STEP_LIMIT_RAD)
        # the agent-facing robot frame of the standard mode (origin at the robot's kinematic root link, world axes)
        spec = S.robot_spec()
        T_WR = M._T(spec["base"]["pose"][:3], spec["base"]["pose"][3:])
        T_RW = np.eye(4); T_RW[:3, :3] = T_WR[:3, :3].T; T_RW[:3, 3] = -T_WR[:3, :3].T @ T_WR[:3, 3]
        Q_WR = M.mat_to_quat(T_WR[:3, :3]); self._Q_RW = Q_WR * np.array([1.0, -1.0, -1.0, -1.0]); self._Q_WR = Q_WR
        self.T_world_from_robot, self._T_RW = T_WR, T_RW
        self.base_link = r.entity.links[0].name
        n, A = len(self.action_layout), len(self.arms)
        ee = [f"{a}_{k}" for a in self.arms for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]
        self.proprio_layout = (self.action_layout + ee + [f"{x}_cmd" for x in self.action_layout]
                               + [f"{x}_vel" for x in self.action_layout] + [f"{a}_gripper_opening_m" for a in self.arms]
                               + [f"{x}_effort" for x in self.action_layout])
        self.proprio_groups = {
            "joint_pos": [0, n, "measured joint positions in the action layout (arm joints rad; gripper: the command space, "
                                f"{S.gripper_space})"],
            "tcp_pose": [n, 7 * A, "measured tool-frame (TCP) pose per arm: x y z (m) + qw qx qy qz, in the robot frame"],
            "joint_cmd": [n + 7 * A, n, "the joint targets being tracked (after every limit): the last applied row"],
            "joint_vel": [2 * n + 7 * A, n, "measured joint velocities (rad/s; gripper: its rate)"],
            "gripper_opening": [3 * n + 7 * A, A, "measured gap between the finger pads per arm (m)"],
            "joint_effort": [3 * n + 8 * A, n, "the force each PD drive applied (N m; gripper: its drive)"],
        }
        self._cam_info = S.camera_info()
        native = {c: tuple(v["image_size"]) for c, v in self._cam_info.items()}
        depth_max = {c: float(v["depth_range_m"][1]) for c, v in self._cam_info.items()}
        self._setup_spaces(self.limits[:, 0], self.limits[:, 1], np.float32, native, depth_max)
        self._raster = None
        S.record_start()

    # ---------------------------------------------------------------- frames
    def to_robot(self, pos, quat):
        """A world pose (position, quaternion wxyz) -> the robot frame."""
        T = self._T_RW
        return T[:3, :3] @ np.asarray(pos, dtype=np.float64) + T[:3, 3], _qmul(self._Q_RW, np.asarray(quat, dtype=np.float64))

    def to_world(self, pos, quat):
        """A robot-frame pose -> the simulator's world frame."""
        T = self.T_world_from_robot
        return T[:3, :3] @ np.asarray(pos, dtype=np.float64) + T[:3, 3], _qmul(self._Q_WR, np.asarray(quat, dtype=np.float64))

    # ---------------------------------------------------------------- hooks
    def _reset_sim(self) -> None:
        S = self.session
        S.reset()
        S.record_start()

    def _clamp(self, a: np.ndarray) -> tuple[np.ndarray, bool]:
        """The row this step commands: clipped to the joint limits, then (rules="standard") the standard mode's
        joint-speed rule: every arm joint target moves at most JOINT_STEP_LIMIT_RAD (0.04 rad) per control step from the
        current targets (server.py run_joint_position: componentwise clip of the step); grippers are not limited."""
        row = np.clip(a, self.limits[:, 0], self.limits[:, 1])
        limited = False
        if self.rules == "standard":
            cmd = np.asarray(self.session.hold_action(), dtype=np.float64)
            cols, lim = self._arm_cols, self.joint_step_limit_rad
            d = row[cols] - cmd[cols]
            if np.any(np.abs(d) > lim + 1e-6):
                row[cols] = cmd[cols] + np.clip(d, -lim, lim)
                limited = True
        return row, limited

    def _step_sim(self, a: np.ndarray) -> dict:
        S = self.session
        row, limited = self._clamp(a)
        S.step(np.asarray(row, dtype=np.float32))   # exactly what the standard service hands the Session (float32)
        notes = (S.last_step_meta or {}).get("limits") or []
        end = S.episode_end
        return {"success": S.success(), "terminated": end is not None, "episode_end": end,
                "progress": float(S._reward), "applied_action": np.asarray(S.hold_action(), dtype=np.float32),
                "speed_limited": bool(limited or notes), "robot_limits": notes, "table_guard": S.last_guard,
                "infra_error": end == M_INSTABILITY}

    def _control_steps(self) -> int:
        return int(self.session.n_steps)

    @property
    def sim_time_s(self) -> float:
        return float(self.session.sim_time_s)

    def _proprio(self) -> np.ndarray:
        S = self.session
        tcp = [np.concatenate(self.to_robot(*S.ee_pose(a))) for a in self.arms]
        return np.concatenate([S.joint_positions(), *tcp, S.hold_action(), S.joint_velocities(),
                               [S.gripper_width(a) for a in self.arms], S.joint_efforts()]).astype(np.float32)

    def _images(self, cameras, depth):
        S = self.session
        if self.renderer == "nyx":
            f = S.env.sensor_frame(depth=depth)
            rgb = {c: np.ascontiguousarray(np.asarray(f.rgb[c])[..., :3]) for c in cameras}
            dep = {}
            if depth:
                for c in cameras:
                    d = np.array(f.depth[c], dtype=np.float32)
                    d = d[..., 0] if d.ndim == 3 else d
                    d[~np.isfinite(d) | (d <= 0) | (d > 65.535)] = 0.0
                    dep[c] = d
            return rgb, dep
        if self.renderer == "mujoco":
            from .render_mujoco import Raster, fovy_from_intrinsics
            info = S.camera_info(list(cameras))
            if self._raster is None:
                w = max(v["image_size"][0] for v in info.values()); h = max(v["image_size"][1] for v in info.values())
                self._raster = Raster(S.env._world.model, w, h)
                atexit.register(self._raster.close)            # before the interpreter tears down the GL bindings
            self._raster.update(S.env._world.data)
            rgb, dep = {}, {}
            for c, v in info.items():
                img, z = self._raster.render(v["cam2world_gl"], fovy_from_intrinsics(v["intrinsic_cv"], v["image_size"][1]),
                                             v["image_size"], depth, clip=v["depth_range_m"])
                rgb[c] = img
                if depth:
                    dep[c] = z
            return rgb, dep
        raise RuntimeError("renderer='none': no images")

    # ---------------------------------------------------------------- extras
    def camera_info(self) -> dict:
        """Per camera, at the current state: intrinsic_cv, extrinsic_cv / cam2world_gl in the robot frame, image_size,
        mount, depth_range_m (the standard mode's calibration)."""
        out = {}
        for c, v in self.session.camera_info().items():
            E = np.eye(4); E[:3] = np.asarray(v["extrinsic_cv"], dtype=np.float64)
            out[c] = {"intrinsic_cv": v["intrinsic_cv"], "extrinsic_cv": (E @ self.T_world_from_robot)[:3].tolist(),
                      "cam2world_gl": (self._T_RW @ np.asarray(v["cam2world_gl"], dtype=np.float64)).tolist(),
                      "image_size": v["image_size"], "moves": v["moves"], "depth_range_m": v["depth_range_m"],
                      "mount": v["mount"] if v["mount"]["type"] == "link" else {"type": "static"}}
        return out

    def rules_text(self) -> dict:
        S, M = self.session, self.M
        return {"rules": self.rules,
                "joint_speed_rule": (f"arm joint targets move at most {self.joint_step_limit_rad:g} rad per control step "
                                     f"({self.joint_step_limit_rad * self.control_hz:g} rad/s; the standard mode's rule)")
                                    if self.rules == "standard" else None,
                "robot_limits": "the tooling's Session.step: joint limits clipped, the robot's joint speed limits (and the "
                                "Panda's Cartesian and command-lead limits), the table guard (no gripper target lower "
                                f"than {M.table_guard_words()}), the static guard; all clamp, never fail",
                "success": "the task's own predicate held continuously for its confirmation window (Session.success())",
                "episode_end": "success, the task's failure predicate, or a simulator breakdown ('simulator_instability': "
                               "an infrastructure error, not the task's failure); later steps do not move the scene"}

    def state_sha(self) -> str:
        """The state hash the graders compare (rcb_topogym.Session.state_sha)."""
        return self.session.state_sha()

    def success(self) -> bool:
        return bool(self.session.success())

    def evaluate(self) -> dict:
        """The tooling's status (success, progress, episode_end, review / forcing evidence, state hash) and its safety
        readings for the episode so far."""
        S = self.session
        return {"status": S.status(), "safety": S.safety(episode=True)}

    def trajectory(self) -> np.ndarray:
        """The rows applied since the last reset (float32, after every limit): a deliverable for the privileged mode."""
        rec = self.session._recording or []
        return np.stack(rec).astype(np.float32) if rec else np.zeros((0, len(self.action_layout)), np.float32)

    def save_trajectory(self, path, actions=None) -> str:
        """Write `actions` (default: trajectory()) as the privileged mode's /app/output/trajectory.npz."""
        acts = self.trajectory() if actions is None else np.asarray(actions, dtype=np.float32)
        self.session.save_trajectory(acts, str(path), meta={"source": "roboarc api"})
        return str(path)

    def grade(self, actions) -> dict:
        """Replay `actions` from the frozen state exactly as the privileged verifier does (Session.replay with
        grading=True; privileged rules, the grading hold for its families) and return its status. Resets the env."""
        S = self.session
        st = S.replay(np.asarray(actions, dtype=np.float32), grading=True)
        st["strict_goal"] = self.M.strict_goal([st])
        self._terminated, self._truncated = True, False     # the episode is the replay's: reset() before stepping
        return st

    def get_state(self) -> dict:
        """A snapshot of the episode: a complete simulator checkpoint of the current tick (topogym's history), the
        tooling's episode counters, and the rows applied since reset (set_state can always rebuild from those)."""
        S = self.session
        S.env._checkpoint_current()
        ref = dict(S.env._info()["state_ref"])
        return {"suite": "puzzles", "task": self.task.id, "state_ref": ref, "state_sha": S.state_sha(),
                "rows": self.trajectory(), "ticks": list(getattr(S, "_recorded_ticks", [])),
                "session": {k: _copy(getattr(S, k)) for k in _SESSION_FIELDS if hasattr(S, k)},
                "env": {"success_seen": self._success_seen, "terminated": self._terminated, "truncated": self._truncated}}

    def set_state(self, state: dict) -> str:
        """Restore a get_state() snapshot of this environment. Fast path: rewind topogym's history to the snapshot's
        checkpoint; when that checkpoint is gone (a reset or an earlier rewind discarded it) or the restored state hash
        differs, the rows are replayed from the frozen state instead. -> "checkpoint" | "replay"."""
        if state.get("task") != self.task.id or state.get("suite") != "puzzles":
            raise ValueError("this state belongs to another task")
        S = self.session
        how = "checkpoint"
        try:
            S.env.privileged.rewind_to(state["state_ref"])
            for k, v in state["session"].items():
                setattr(S, k, _copy(v))
            S._recording = [np.asarray(r, dtype=np.float32) for r in state["rows"]]
            S._recorded_ticks = list(state["ticks"])
            if S.state_sha() != state["state_sha"]:
                raise ValueError("restored state differs")
        except (ValueError, KeyError, RuntimeError):
            how = "replay"
            S.reset(); S.record_start()
            for r in np.asarray(state["rows"], dtype=np.float32):
                S.step(r)
            if S.state_sha() != state["state_sha"]:
                raise RuntimeError("the replayed rows did not reach the snapshot's state hash")
        e = state["env"]
        self._success_seen, self._terminated, self._truncated = e["success_seen"], e["terminated"], e["truncated"]
        return how

    def close(self) -> None:
        if not self._closed:
            if self._raster is not None:
                self._raster.close()
            self.session.close()
        super().close()


M_INSTABILITY = "simulator_instability"
_SESSION_FIELDS = ("n_steps", "physics_steps", "sim_time_s", "last_step_ticks", "_reward", "_targets", "episode_done",
                   "episode_end", "instability", "last_step_meta", "last_guard", "last_static_guard", "last_forcing")


def _copy(v):
    import copy
    return copy.deepcopy(v)
