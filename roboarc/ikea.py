"""The IKEA MuJoCo tasks (plain MuJoCo 3.14.0, the tooling rcb_ikea.Session) as a RoboARCEnv.

Physics, limits and verdicts are the graded tooling's own: every control step is rcb_ikea.Session.step(row) (the
control range, then the speed clamp: the arm joints and the Franka Hand slew, the one table both modes and the grader's
replay apply), and the end-of-episode rule is the standard service's (environment/sim/server.py: verdict() after a
check, then final_check() on a snapshot). Cameras: the standard mode's rig (ikea_rig.py, rig.json), RGB from Nyx
(ikea_nyx.py, the -sim image) or MuJoCo's raster, depth from MuJoCo's raster."""
from __future__ import annotations

import atexit
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from ._runtime import (IMAGE_IKEA_SCENES, IMAGE_NYX_PYTHON, RuntimeMismatch, cache_dir, check_graded_runtime,
                       in_ikea_image, repo_root)
from .core import RoboARCEnv, warn_or_raise
from .registry import TaskInfo

_SCENE_LOADED = []


def _import_tooling(root: Path):
    shared = root / "ikea" / "scripts" / "ikea_mujoco" / "shared"
    standard = root / "ikea" / "scripts" / "ikea_mujoco" / "standard"
    for p in (str(shared), str(standard)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import rcb_ikea
    if Path(rcb_ikea.__file__).resolve().parent != shared.resolve():
        raise RuntimeMismatch(f"rcb_ikea was imported from {rcb_ikea.__file__}, not from the repository's {shared}")
    return rcb_ikea


def _quat(xmat) -> np.ndarray:
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(xmat, dtype=np.float64).reshape(9))
    return q


class IkeaEnv(RoboARCEnv):
    """An IKEA MuJoCo task. `session` is the underlying rcb_ikea.Session (model, data, status(), final_check(), ...)."""

    def __init__(self, task: TaskInfo, *, renderer="nyx", cameras=None, depth=False, image_size=None, render_mode=None,
                 max_episode_steps=None, strict=True, check_every: int = 40, instance_seed: int = 0):
        super().__init__(task, renderer=renderer, cameras=cameras, depth=depth, image_size=image_size,
                         render_mode=render_mode, max_episode_steps=max_episode_steps, strict=strict)
        if int(instance_seed) not in task.seeds:
            raise ValueError(f"{task.id} allows the instance seeds {list(task.seeds)}, not {instance_seed}")
        self.instance_seed = int(instance_seed)
        if _SCENE_LOADED:
            raise RuntimeError("one IKEA environment per process (each scene's modules are top-level imports); use "
                               "subprocess vector environments for several")
        self.rules = "standard"
        self.step_after_end = True
        self.check_every = max(1, int(check_every))
        root = repo_root()
        problems = check_graded_runtime(strict=False)
        scenes = Path(os.environ.get("IKEA_SCENES") or IMAGE_IKEA_SCENES)
        if not (scenes / task.name).is_dir():
            problems.append(f"no IKEA scene {task.name} under {scenes}: run inside {task.render_image}"
                            " (or set IKEA_SCENES to the scenes)")
        if renderer == "nyx" and not Path(os.environ.get("RCB_NYX_PYTHON", str(IMAGE_NYX_PYTHON))).is_file():
            raise RuntimeMismatch(f"renderer='nyx' needs the Nyx render process of {task.render_image} "
                                  "(RCB_NYX_PYTHON); use that image, or renderer='mujoco' / 'none'")
        if renderer != "none" and os.environ.get("MUJOCO_GL", "") == "" and not os.environ.get("DISPLAY"):
            problems.append("MUJOCO_GL is not set (the camera rig's depth and raster need egl or osmesa); the -sim image "
                            "sets MUJOCO_GL=egl")
        R = self.R = _import_tooling(root)
        binding_path = Path(task.task_dirs["privileged"]) / "tests" / "runtime_binding.json"
        self.binding = json.loads(binding_path.read_text())
        import mujoco
        runtime = {"mujoco": mujoco.__version__, "numpy": np.__version__, "python": sys.version.split()[0]}
        if (scenes / task.name).is_dir():
            files = R.scene_files(str(scenes / task.name))
            if files != self.binding["scene_files"]:
                problems.append("the scene's files differ from the task's runtime binding")
        if runtime != self.binding.get("runtime"):
            problems.append(f"runtime {runtime} differs from the bound {self.binding.get('runtime')} ({task.image})")
        warn_or_raise(problems, strict, RuntimeMismatch)
        _SCENE_LOADED.append(task.id)
        self.session = R.Session(task.name, str(scenes / task.name))
        self.birth_hash = self.session.state_sha()
        if strict and (self.birth_hash != self.binding["initial_state_sha256"] or self.session.nu != int(self.binding["nu"])
                       or self.session.actuators != self.binding["actuators"]):
            raise RuntimeMismatch(f"the scene built to {self.birth_hash[:12]}, not the bound initial state "
                                  f"{self.binding['initial_state_sha256'][:12]}")
        self._bind_model()
        s, m = self.session, self.session.model
        self.control_hz = 1.0 / float(s.control_dt)
        self.physics_substeps = int(R.SUBSTEPS)
        self.action_layout = list(s.actuators)
        lo = np.where(np.asarray(m.actuator_ctrllimited).astype(bool), m.actuator_ctrlrange[:, 0], -np.inf)
        hi = np.where(np.asarray(m.actuator_ctrllimited).astype(bool), m.actuator_ctrlrange[:, 1], np.inf)
        self.limits = np.stack([lo, hi], axis=1).astype(np.float64)
        self.step_limit = np.asarray(s.step_limit, dtype=np.float64)
        # cameras: the standard mode's rig (rig.json of the standard task), posed from a scratch copy of the joints
        std = Path(task.task_dirs.get("standard", ""))
        rig_path = std / "environment" / "sim" / "rig.json"
        self.rig_spec = json.loads(rig_path.read_text())
        text = std / "environment" / "sim" / "task_text.json"
        self.instruction = json.loads(text.read_text())["task_text"] if text.is_file() else task.description
        import ikea_rig
        self.rig = ikea_rig.Rig(m, copy.deepcopy(self.rig_spec))
        self.rig.update(s.data)
        self.nyx = None
        atexit.register(self._close_gl)     # before the interpreter tears down the GL bindings
        if renderer == "nyx":
            import ikea_nyx
            self.nyx = ikea_nyx.NyxRig(m, self.rig, copy.deepcopy(self.rig_spec),
                                       tempfile.mkdtemp(prefix=f"ikea_nyx_{task.name}_", dir=cache_dir()),
                                       seed=self.instance_seed)
        native = {c: (self.rig.W, self.rig.H) for c in self.rig.names}
        depth_max = {c: float(self.rig.far) for c in self.rig.names}
        self._setup_spaces(lo, hi, np.float64, native, depth_max)
        self._frames: dict = {}
        self._since_check = 0
        self.settle_failed = 0
        self._end = None

    def _bind_model(self) -> None:
        """Names and indices of the robot in the current model (server.py's: arms left_ / right_, grippers, tcp sites,
        the robot frame = the left arm's base link)."""
        import mujoco
        s = self.session
        m, d = s.model, s.data
        act = list(s.actuators)
        sep = act[0][len("left")] if act and act[0].startswith("left") else "_"
        self.arms = ["left", "right"]
        grip, arm_joints, fingers = {}, {}, {}
        for a in self.arms:
            idx = [i for i, n in enumerate(act) if n.startswith(a + sep)]
            g = [i for i in idx if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_TENDON or "gripper" in act[i]]
            grip[a] = g[0]
            arm_joints[a] = [int(m.actuator_trnid[i, 0]) for i in idx if i != g[0]]
            fingers[a] = [j for j in range(m.njnt) if m.joint(j).name.startswith(a + sep) and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE]
        kind = "viperx" if self.task.name == "svallet" else "panda"
        self._tcp = {a: m.site(f"{a}{sep}tcp" if kind == "panda" else f"{a}{sep}gripper").id for a in self.arms}
        root = m.body(f"left{sep}link0" if kind == "panda" else f"left{sep}base_link").id
        T_WR = np.eye(4); T_WR[:3, :3] = np.asarray(d.xmat[root]).reshape(3, 3); T_WR[:3, 3] = d.xpos[root]
        T_RW = np.eye(4); T_RW[:3, :3] = T_WR[:3, :3].T; T_RW[:3, 3] = -T_WR[:3, :3].T @ T_WR[:3, 3]
        self.T_world_from_robot, self._T_RW = T_WR, T_RW
        self._Q_RW = _quat(T_WR[:3, :3]) * np.array([1.0, -1.0, -1.0, -1.0])
        self._qadr = np.array([m.jnt_qposadr[j] for a in self.arms for j in arm_joints[a]])
        self._vadr = np.array([m.jnt_dofadr[j] for a in self.arms for j in arm_joints[a]])
        self._fingers = fingers
        self._grip_cols = [grip[a] for a in self.arms]
        jn = [m.joint(j).name for a in self.arms for j in arm_joints[a]]
        nj, nu = len(jn), len(act)
        tcp_names = [m.site(self._tcp[a]).name for a in self.arms]
        self.proprio_layout = (jn + [f"{n}_vel" for n in jn] + [f"{a}{sep}gripper_opening_m" for a in self.arms]
                               + [f"{t}_{k}" for t in tcp_names for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]
                               + [f"{n}_cmd" for n in act] + [f"{n}_effort" for n in act])
        self.proprio_groups = {
            "joint_pos": [0, nj, "measured arm joint positions (rad), left arm then right arm"],
            "joint_vel": [nj, nj, "measured arm joint velocities (rad/s)"],
            "gripper_opening": [2 * nj, 2, "measured opening of each gripper (m): the sum of its finger joint positions"],
            "tcp_pose": [2 * nj + 2, 14, "measured tool-frame pose per arm: x y z (m) + qw qx qy qz, in the robot frame"],
            "joint_cmd": [2 * nj + 16, nu, "the actuator commands being applied (after the control range and speed clamp)"],
            "joint_effort": [2 * nj + 16 + nu, nu, "the force of each actuator"],
        }

    def to_robot(self, pos, quat):
        import mujoco
        T = self._T_RW
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, self._Q_RW, np.asarray(quat, dtype=np.float64))
        return T[:3, :3] @ np.asarray(pos, dtype=np.float64) + T[:3, 3], q

    # ---------------------------------------------------------------- hooks
    def _reset_sim(self) -> None:
        """rcb_ikea.Session.reset(): a fresh build of the scene, bit-identical to the grader's initial state."""
        self.session.reset()
        self._bind_model()
        self._frames = {}
        self._since_check = 0
        self.settle_failed = 0
        self._end = None

    def _verdict(self) -> dict:
        """server.py verdict(): Session.status() side-effect free (data and the mechanism's eq_data restored)."""
        import mujoco
        s = self.session
        m, d = s.model, s.data
        snap = mujoco.MjData(m)
        mujoco.mj_copyData(snap, m, d); eqd = m.eq_data.copy()
        try:
            return s.status()
        finally:
            mujoco.mj_copyData(d, m, snap); m.eq_data[:] = eqd

    def _settled(self) -> dict:
        """server.py settled(): Session.final_check() (the last row held SETTLE_S more on a copy), side-effect free."""
        import mujoco
        s = self.session
        m, d = s.model, s.data
        snap = mujoco.MjData(m)
        mujoco.mj_copyData(snap, m, d); eqd = m.eq_data.copy()
        try:
            return s.final_check()
        finally:
            mujoco.mj_copyData(d, m, snap); m.eq_data[:] = eqd

    def _step_sim(self, a: np.ndarray) -> dict:
        s = self.session
        before = s.slewed_ticks
        clipped = bool(np.any((a < self.limits[:, 0]) | (a > self.limits[:, 1])))
        s.step(a)
        self._frames = {}
        out = {"applied_action": np.array(s.data.ctrl, dtype=np.float64), "speed_limited": s.slewed_ticks > before,
               "clipped": clipped, "episode_end": self._end}
        success = False
        if self._end is None:
            if s.validity.breakdown is not None:
                self._end = "sim_breakdown"
            else:
                self._since_check += 1
                if self._since_check >= self.check_every:
                    self._since_check = 0
                    v = self._verdict()
                    if v["breakdown"] is not None:
                        self._end = "sim_breakdown"
                    elif v["success"]:
                        if self._settled()["success"]:
                            self._end = "success"
                            success = True
                        else:
                            self.settle_failed += 1
        out["episode_end"] = self._end
        out["breakdown"] = s.validity.breakdown
        out["settle_failed"] = self.settle_failed
        out["success"] = success or self._end == "success"
        out["terminated"] = self._end is not None
        return out

    def _control_steps(self) -> int:
        return len(self.session.rows)

    @property
    def sim_time_s(self) -> float:
        return float(self.session.data.time)

    def _proprio(self) -> np.ndarray:
        s = self.session
        m, d = s.model, s.data
        tcp = [np.concatenate(self.to_robot(d.site_xpos[self._tcp[a]], _quat(d.site_xmat[self._tcp[a]]))) for a in self.arms]
        grip = [float(sum(d.qpos[m.jnt_qposadr[j]] for j in self._fingers[a])) for a in self.arms]
        return np.concatenate([d.qpos[self._qadr], d.qvel[self._vadr], grip, *tcp, d.ctrl, d.actuator_force]).astype(np.float32)

    def _images(self, cameras, depth):
        s = self.session
        self.rig.update(s.data, key=(len(s.rows), id(s.data)))
        todo = [c for c in cameras if c not in self._frames]
        if todo:
            if self.renderer == "nyx":
                self._frames.update(self.nyx.render(self.rig.kd, todo))
            elif self.renderer == "mujoco":
                self._frames.update({c: self.rig.rgb(c) for c in todo})
            else:
                raise RuntimeError("renderer='none': no images")
        rgb = {c: self._frames[c] for c in cameras}
        dep = {c: self.rig.depth(c) for c in cameras} if depth else {}
        return rgb, dep

    # ---------------------------------------------------------------- extras
    def camera_info(self) -> dict:
        s = self.session
        self.rig.update(s.data, key=(len(s.rows), id(s.data)))
        out = {}
        for c in self.rig.names:
            E = np.eye(4); E[:3] = self.rig.extrinsic_cv(c)
            out[c] = {"intrinsic_cv": self.rig.intrinsic_cv(c), "extrinsic_cv": (E @ self.T_world_from_robot)[:3].tolist(),
                      "cam2world_gl": (self._T_RW @ self.rig.cam2world_gl(c)).tolist(), "image_size": [self.rig.W, self.rig.H],
                      "moves": self.rig.mounts[c]["type"] != "static", "depth_range_m": [self.rig.near, self.rig.far]}
        return out

    def rules_text(self) -> dict:
        R = self.R
        return {"rules": "standard (= privileged for IKEA)",
                "speed_rule": "rcb_ikea.Session.applied(): the control range, then every limited target (the arm joints; "
                              "the Franka Hand command at 0.1 m/s of finger width) moves at most its limit x control step "
                              "from the previous target; clamp, never fail",
                "speed_limits_per_s": {n: float(v * self.control_hz) for n, v in zip(self.action_layout, self.step_limit)
                                       if np.isfinite(v)},
                "success": f"success.py AND released AND at rest AND no breakdown, then again after holding the last row "
                           f"{R.SETTLE_S:g} s on a copy (final_check); checked every {self.check_every} control steps",
                "episode_end": "success (settled), sim_breakdown (a non-finite state, a BADQACC/BADQPOS/BADQVEL warning, a "
                               f"part contact deeper than {R.BREAKDOWN_DEPTH_M * 1000:g} mm); stepping on is allowed (physics "
                               "continues, as in the privileged mode)"}

    def state_sha(self) -> str:
        return self.session.state_sha()

    def success(self) -> bool:
        return bool(self._verdict()["success"])

    def evaluate(self) -> dict:
        """The grader's verdict on the rows so far: status() now and final_check() (after the 1 s hold), both side-effect
        free."""
        return {"status": self._verdict(), "final_check": self._settled()}

    def trajectory(self) -> np.ndarray:
        rows = self.session.rows
        return np.array(rows, dtype=np.float64).reshape(-1, len(self.action_layout))

    def save_trajectory(self, path, actions=None) -> str:
        return self.session.save_trajectory(None if actions is None else np.asarray(actions, dtype=np.float64), str(path))

    def grade(self, actions) -> dict:
        """Replay `actions` from a fresh build exactly as each of the verifier's replays does (Session.replay ->
        final_check). Resets the env."""
        out = self.session.replay(np.asarray(actions, dtype=np.float64))
        self._bind_model()
        self._terminated, self._truncated = True, False
        return out

    def get_state(self) -> dict:
        return {"suite": "ikea", "task": self.task.id, "session": self.session.get_state(), "state_sha": self.state_sha(),
                "env": {"success_seen": self._success_seen, "terminated": self._terminated, "truncated": self._truncated,
                        "since_check": self._since_check, "settle_failed": self.settle_failed, "end": self._end}}

    def set_state(self, state: dict) -> str:
        if state.get("task") != self.task.id or state.get("suite") != "ikea":
            raise ValueError("this state belongs to another task")
        self.session.set_state(copy.deepcopy(state["session"]))
        e = state["env"]
        self._success_seen, self._terminated, self._truncated = e["success_seen"], e["terminated"], e["truncated"]
        self._since_check, self.settle_failed, self._end = e["since_check"], e["settle_failed"], e["end"]
        self._frames = {}
        if self.state_sha() != state["state_sha"]:
            raise RuntimeError("the restored state differs from the snapshot")
        return "checkpoint"

    def _close_gl(self) -> None:
        r = getattr(self.rig, "renderer", None)
        if r is not None:
            self.rig.renderer = None
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        if not self._closed:
            self._close_gl()
            if self.nyx is not None:
                try:
                    self.nyx.close()
                except Exception:  # noqa: BLE001
                    pass
        super().close()
