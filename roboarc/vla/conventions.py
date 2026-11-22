"""openpi's observation and action conventions on RoboARC's robots (docs/VLA.md).

    aloha     two-arm robots (ALOHA 2 puzzles; IKEA: the ViperX pair of svallet, the Franka pair of the others): openpi's
              ALOHA example. {"state": (2 * (J + 1),), "images": {cam_high, cam_low, cam_left_wrist, cam_right_wrist:
              (3, 224, 224) uint8}, "prompt"}; actions (H, 2 * (J + 1)) absolute joint targets (rad) with the gripper
              normalised 0 = closed .. 1 = open. Joints in the ALOHA (Interbotix) convention: RoboARC's ViperX arms have
              its joint names, zeros and limits, so nothing is flipped (openpi's adapt_to_pi flips belong to the server).
    droid     one Franka arm with a Robotiq 2F-85 (puzzles): openpi's DROID example.
              {"observation/exterior_image_1_left", "observation/wrist_image_left": (224, 224, 3) uint8,
              "observation/joint_position": (7,), "observation/gripper_position": (1,), "prompt"}; the gripper normalised
              0 = open .. 1 = closed (DROID's). Actions (H, 8): `droid` absolute joint positions + gripper position (the
              action space of openpi's joint-position DROID configs, the one openpi recommends for simulation);
              `droid-velocity` DROID's original joint velocities (normalised: a joint moves velocity x 0.2 rad per
              1/15 s step from its measured position, the largest component scaled to |1|) + the gripper position
              binarised at 0.5, as DROID's RobotEnv and openpi's DROID client execute them.

A convention maps the environment's observation to openpi's and an action chunk to control-step rows. Rows are
executed one per control step at `policy_hz` = the environment's control rate (40 Hz) unless asked otherwise: then the
chunk is resampled in time (joints linearly between consecutive actions, the gripper held), so `policy_hz=50` executes
an ALOHA chunk of 50 Hz actions over the same simulated time. Every RoboARC limit (the standard mode's joint-speed rule,
the robots' speed limits, the table guard, ...) still applies to the rows: clamp, never fail."""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

ALOHA_CAMERAS = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
DROID_MAX_JOINT_DELTA = 0.2      # rad per DROID control step (droid/robot_ik/robot_ik_solver.py, max_joint_delta)
DROID_CONTROL_HZ = 15.0          # DROID's control rate (openpi examples/droid/main.py)
CONVENTIONS = ("aloha", "droid", "droid-velocity")
IKEA_POLICY_HZ = 50.0            # the default action rate on IKEA (openpi's ALOHA rate; IKEA steps at 400 Hz)


# ------------------------------------------------------------------------------------------------- images and prompts
def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """(H, W, C) uint8 -> (height, width, C): scaled to fit (PIL bilinear) and centred on black, the algorithm of
    tf.image.resize_with_pad as openpi_client.image_tools.resize_with_pad computes it (checked in tests/api/test_vla.py)."""
    from PIL import Image
    if image.shape[-3:-1] == (height, width):
        return image
    im = Image.fromarray(np.ascontiguousarray(image))
    cur_w, cur_h = im.size
    ratio = max(cur_w / width, cur_h / height)
    rh, rw = int(cur_h / ratio), int(cur_w / ratio)
    im = im.resize((rw, rh), resample=Image.BILINEAR)
    out = Image.new(im.mode, (width, height), 0)
    out.paste(im, (max(0, int((width - rw) / 2)), max(0, int((height - rh) / 2))))
    return np.asarray(out)


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def task_prompt(task, full: bool = False) -> str:
    """The language instruction: the first sentence of the standard task's **Task:** text (instruction.md), the
    short goal a VLA prompt carries; full=True gives the whole **Task:** paragraph. The task's colours, names and
    sizes stay as written (the same text for every instance seed)."""
    std = Path(task.task_dirs.get("standard") or task.task_dirs["privileged"])
    text = (std / "instruction.md").read_text()
    m = re.search(r"\*\*Task:\*\*\s*(.+)", text)
    if not m:
        return task.description
    para = m.group(1).strip()
    return para if full else _SENTENCE_END.split(para, maxsplit=1)[0].strip()


# ---------------------------------------------------------------------------------------------- the robot's columns
class RobotMap:
    """Where each arm's joints and gripper sit in the environment's action row and proprioception, and the gripper's
    command range (closed, open) in the action's units."""

    def __init__(self, env):
        self.env = env
        u = env.unwrapped
        n = len(env.action_layout)
        jp = env.proprio_groups["joint_pos"]
        self.arms: list[str] = list(u.arms)
        self.grip_col: dict[str, int] = {}
        self.arm_cols: dict[str, list[int]] = {}
        self.closed: dict[str, float] = {}
        self.open: dict[str, float] = {}
        self._grip_measured: dict[str, tuple] = {}     # arm -> (group, proprio index, measured span or None)
        if env.task.suite == "puzzles":
            r = u.session.robot
            grips = u.session.robot_spec()["grippers"]
            for a in self.arms:
                sl = r.arm_slice[a]
                self.arm_cols[a] = list(range(sl.start, sl.stop))
                self.grip_col[a] = int(r.grip_index[a])
                g = grips[a]
                if g.get("command_space") == "native":
                    self.closed[a], self.open[a] = float(g["closed_target"]), float(g["open_target"])
                else:
                    self.closed[a], self.open[a] = float(g["opening_m"][0]), float(g["opening_m"][1])
            # measured joints in the action layout; the gripper column holds the measured gripper in the command space
            self._joint_index = np.arange(jp[0], jp[0] + n)
            self._grip_measured = {a: ("joint_pos", jp[0] + self.grip_col[a], None) for a in self.arms}
        else:                                                    # IKEA: actuators left then right, one gripper each
            m = u.session.model
            go = env.proprio_groups["gripper_opening"]
            idx = 0
            self._joint_index = np.zeros(n, dtype=int)
            for i, a in enumerate(self.arms):
                cols = [j for j, name in enumerate(env.action_layout) if name.startswith(a)]
                g = u._grip_cols[i]
                self.grip_col[a] = int(g)
                self.arm_cols[a] = [c for c in cols if c != g]
                lo, hi = (float(v) for v in u.limits[g])
                self.closed[a], self.open[a] = lo, hi            # Panda hand 0 = closed .. 255 = open; ViperX rail m
                for c in self.arm_cols[a]:
                    self._joint_index[c] = jp[0] + idx
                    idx += 1
                full = float(sum(m.jnt_range[j][1] for j in u._fingers[a]))     # the opening at the fingers' limits
                self._grip_measured[a] = ("gripper_opening", go[0] + i, (0.0, full))
        self.arm_joint_cols = [c for a in self.arms for c in self.arm_cols[a]]

    def joints(self, proprio: np.ndarray) -> np.ndarray:
        """Measured joint positions in the action layout (the gripper columns: see gripper_state)."""
        q = np.zeros(len(self.env.action_layout))
        for a in self.arms:
            q[self.arm_cols[a]] = proprio[self._joint_index[self.arm_cols[a]]]
        return q

    def gripper_state(self, proprio: np.ndarray, a: str, closed_is_one: bool) -> float:
        """The measured gripper, normalised: 0 = open .. 1 = closed when closed_is_one (DROID), else 0 = closed .. 1 = open
        (ALOHA)."""
        _, i, span = self._grip_measured[a]
        x = float(proprio[i])
        if span is None:                                          # the command space (puzzles)
            lo, hi = (self.open[a], self.closed[a]) if closed_is_one else (self.closed[a], self.open[a])
        else:                                                     # IKEA: the measured opening (m), 0 = closed
            lo, hi = (span[1], span[0]) if closed_is_one else span
        return (x - lo) / (hi - lo)

    def gripper_command(self, g: float, a: str, closed_is_one: bool) -> float:
        lo, hi = (self.open[a], self.closed[a]) if closed_is_one else (self.closed[a], self.open[a])
        return lo + float(g) * (hi - lo)

    def gripper_normalise(self, x: float, a: str, closed_is_one: bool) -> float:
        """A gripper command (the action's units) -> the convention's normalised value (the inverse of
        gripper_command)."""
        lo, hi = (self.open[a], self.closed[a]) if closed_is_one else (self.closed[a], self.open[a])
        return (float(x) - lo) / (hi - lo)


# ------------------------------------------------------------------------------------------------------ conventions
class Convention:
    """Maps observations to openpi's dict and action chunks to control-step rows (see the module docstring)."""
    name = "?"
    closed_is_one = False

    def __init__(self, env, *, image_size: int = 224, policy_hz: float | None = None, cameras: dict | None = None):
        self.env = env
        self.robot = RobotMap(env)
        self.image_size = int(image_size)
        self.control_hz = float(env.control_hz)
        # one action per control step at the puzzles' 40 Hz; IKEA's control step is 2.5 ms (400 Hz): openpi's 50 Hz there
        self.policy_hz = float(policy_hz or (IKEA_POLICY_HZ if env.task.suite == "ikea" else self.control_hz))
        self.cameras = dict(cameras or self.default_cameras())
        missing = [c for c in self.cameras.values() if c not in env.all_camera_names]
        if missing:
            raise ValueError(f"{env.task.id} has no camera {missing}; it has {env.all_camera_names}")
        self.action_dim = len(env.action_layout)
        self._chunk = None

    # observation
    def default_cameras(self) -> dict:
        raise NotImplementedError

    def needed_cameras(self) -> list[str]:
        return list(dict.fromkeys(self.cameras.values()))

    def image(self, obs: dict, camera: str) -> np.ndarray:
        """One camera's image at image_size x image_size (resize with pad); black when the environment renders none
        (renderer="none": for testing a policy loop only, never for an evaluation)."""
        if "rgb" not in obs:
            return np.zeros((self.image_size, self.image_size, 3), np.uint8)
        return resize_with_pad(np.asarray(obs["rgb"][camera], dtype=np.uint8), self.image_size, self.image_size)

    def observation(self, obs: dict, prompt: str) -> dict:
        raise NotImplementedError

    # actions
    def to_rows(self, actions: np.ndarray) -> np.ndarray:
        """(H, D) actions in this convention -> (H, action_dim) absolute environment rows (position conventions)."""
        raise NotImplementedError

    def from_rows(self, rows: np.ndarray) -> np.ndarray:
        """(T, action_dim) environment rows -> (T, D) actions in this convention (position conventions): the inverse
        of to_rows, for replaying a trajectory through a policy server and for datasets."""
        raise NotImplementedError

    def check_chunk(self, chunk) -> np.ndarray:
        a = np.asarray(chunk, dtype=np.float64)
        if a.ndim == 1:
            a = a[None, :]
        if a.ndim != 2 or a.shape[1] < self.action_dim or len(a) == 0:
            raise ValueError(f"the policy returned actions of shape {a.shape}; {self.name} needs (H, {self.action_dim})")
        if not np.all(np.isfinite(a[:, :self.action_dim])):
            raise ValueError("the policy returned NaN or inf actions")
        return a[:, :self.action_dim]           # openpi's outputs keep the first D dimensions (AlohaOutputs, DroidOutputs)

    def begin_chunk(self, chunk, horizon: int) -> None:
        """Start executing the first `horizon` actions of `chunk` (openpi's action horizon / open-loop horizon)."""
        a = self.check_chunk(chunk)
        n = max(1, min(int(horizon), len(a)))
        self._chunk = (a[:n], n)
        self._k = 0

    def next_row(self, proprio: np.ndarray):
        """The next control step's row, or None when the executed part of the chunk is used up. Control step k plays the
        chunk at time k / control_hz, i.e. action index u = k * policy_hz / control_hz (u < the executed actions)."""
        a, n = self._chunk
        u = self._k * self.policy_hz / self.control_hz
        if u > n - 1e-9:
            return None
        self._k += 1
        i = int(math.floor(u + 1e-9))
        f = u - i
        rows = self.to_rows(a[[i]])[0]
        if f > 1e-9 and i + 1 < n:                    # resampled: joints linear between actions i and i + 1
            nxt = self.to_rows(a[[i + 1]])[0]
            cols = self.robot.arm_joint_cols
            rows[cols] = rows[cols] + f * (nxt[cols] - rows[cols])
        return rows


class AlohaConvention(Convention):
    name = "aloha"

    def default_cameras(self) -> dict:
        return {c: c for c in ALOHA_CAMERAS if c in self.env.all_camera_names}

    def state(self, proprio: np.ndarray) -> np.ndarray:
        q = self.robot.joints(proprio)
        for a in self.robot.arms:
            q[self.robot.grip_col[a]] = self.robot.gripper_state(proprio, a, False)
        return q

    def observation(self, obs: dict, prompt: str) -> dict:
        images = {k: np.ascontiguousarray(self.image(obs, cam).transpose(2, 0, 1)) for k, cam in self.cameras.items()}
        return {"state": self.state(obs["proprio"]), "images": images, "prompt": prompt}

    def to_rows(self, actions: np.ndarray) -> np.ndarray:
        rows = np.array(actions, dtype=np.float64, copy=True)
        for a in self.robot.arms:
            c = self.robot.grip_col[a]
            rows[:, c] = [self.robot.gripper_command(g, a, False) for g in actions[:, c]]
        return rows

    def from_rows(self, rows: np.ndarray) -> np.ndarray:
        out = np.array(rows, dtype=np.float64, copy=True)
        for a in self.robot.arms:
            c = self.robot.grip_col[a]
            out[:, c] = [self.robot.gripper_normalise(x, a, False) for x in np.asarray(rows, dtype=np.float64)[:, c]]
        return out


class DroidConvention(Convention):
    """Single arm. velocity=False: absolute joint positions + gripper position (openpi's joint-position DROID configs);
    velocity=True: DROID's joint velocities, executed as DROID's RobotEnv does."""
    name = "droid"
    closed_is_one = True

    def __init__(self, env, *, velocity: bool = False, **kw):
        self.velocity = bool(velocity)
        if self.velocity:
            if kw.get("policy_hz") is None:
                kw["policy_hz"] = DROID_CONTROL_HZ
            self.name = "droid-velocity"
        super().__init__(env, **kw)
        if len(self.robot.arms) != 1:
            raise ValueError(f"the droid convention is for one arm; {env.task.id} has {self.robot.arms}")
        self.arm = self.robot.arms[0]
        self._target = None
        self._held_index = -1

    def default_cameras(self) -> dict:
        """exterior 1 = front_oblique (the closer third-person view), exterior 2 = agentview (the overview; DROID's
        second exterior camera, which openpi's DROID transforms ignore), wrist = the hand camera."""
        cams = self.env.all_camera_names
        info = self.env.unwrapped.camera_info()
        static = [c for c in cams if not info[c]["moves"]]
        ext1 = "front_oblique" if "front_oblique" in cams else static[0]
        out = {"observation/exterior_image_1_left": ext1}
        ext2 = "agentview" if "agentview" in cams and ext1 != "agentview" else next((c for c in static if c != ext1), None)
        if ext2:
            out["observation/exterior_image_2_left"] = ext2
        out["observation/wrist_image_left"] = "wrist" if "wrist" in cams else [c for c in cams if info[c]["moves"]][0]
        return out

    def observation(self, obs: dict, prompt: str) -> dict:
        out = {k: self.image(obs, cam) for k, cam in self.cameras.items()}
        q = self.robot.joints(obs["proprio"])
        out["observation/joint_position"] = q[self.robot.arm_cols[self.arm]]
        out["observation/gripper_position"] = np.array([self.robot.gripper_state(obs["proprio"], self.arm, True)])
        out["prompt"] = prompt
        return out

    def _layout(self, joints: np.ndarray, grip: np.ndarray) -> np.ndarray:
        rows = np.zeros((len(joints), self.action_dim))
        rows[:, self.robot.arm_cols[self.arm]] = joints
        rows[:, self.robot.grip_col[self.arm]] = [self.robot.gripper_command(g, self.arm, True) for g in grip]
        return rows

    def to_rows(self, actions: np.ndarray) -> np.ndarray:
        if self.velocity:
            raise TypeError("droid-velocity actions are integrated from the measured joints (next_row)")
        return self._layout(actions[:, :7], actions[:, 7])

    def from_rows(self, rows: np.ndarray) -> np.ndarray:
        if self.velocity:
            raise TypeError("joint velocities are not a function of the rows alone")
        rows = np.asarray(rows, dtype=np.float64)
        g = [self.robot.gripper_normalise(x, self.arm, True) for x in rows[:, self.robot.grip_col[self.arm]]]
        return np.concatenate([rows[:, self.robot.arm_cols[self.arm]], np.asarray(g)[:, None]], axis=1)

    def begin_chunk(self, chunk, horizon: int) -> None:
        super().begin_chunk(chunk, horizon)
        self._held_index = -1

    def next_row(self, proprio: np.ndarray):
        if not self.velocity:
            return super().next_row(proprio)
        a, n = self._chunk
        u = self._k * self.policy_hz / self.control_hz
        if u > n - 1e-9:
            return None
        self._k += 1
        i = int(math.floor(u + 1e-9))
        if i != self._held_index:                     # a new DROID step: target = measured + velocity x 0.2 rad
            v = np.clip(a[i, :7], -1.0, 1.0)          # openpi's DROID client clips every dimension to [-1, 1]
            peak = float(np.abs(v).max(initial=0.0))
            if peak > 1.0:
                v = v / peak
            q = self.robot.joints(proprio)[self.robot.arm_cols[self.arm]]
            g = 1.0 if a[i, 7] > 0.5 else 0.0          # binarised, as openpi's DROID client does
            self._target = self._layout((q + v * DROID_MAX_JOINT_DELTA)[None, :], np.array([g]))[0]
            self._held_index = i
        return self._target.copy()


def make_convention(env, name: str | None = None, **kw) -> Convention:
    """The convention for this environment's robot: `name` or, by default, aloha for a two-arm robot and droid (joint
    positions) for one arm."""
    if name is None:
        name = "aloha" if len(env.unwrapped.arms) == 2 else "droid"
    if name == "aloha":
        return AlohaConvention(env, **kw)
    if name == "droid":
        return DroidConvention(env, **kw)
    if name == "droid-velocity":
        return DroidConvention(env, velocity=True, **kw)
    raise ValueError(f"convention must be one of {CONVENTIONS}")
