"""The environment interface shared by both suites (RoboARCEnv) and roboarc.make()."""
from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np

from . import spaces
from .registry import TaskInfo, resolve

RENDERERS = ("nyx", "mujoco", "none")
RULES = ("standard", "privileged")


def _resize(img: np.ndarray, size, nearest: bool = False) -> np.ndarray:
    """(W, H) resize with PIL (bilinear for RGB, nearest for depth), as the tooling's render_frame / render_depth."""
    if size is None or (img.shape[1], img.shape[0]) == tuple(size):
        return img
    from PIL import Image
    mode = Image.NEAREST if nearest else Image.BILINEAR
    return np.asarray(Image.fromarray(img).resize(tuple(int(v) for v in size), mode), dtype=img.dtype)


class RoboARCEnv(spaces.EnvBase):
    """One RoboARC task as a Gymnasium-compatible environment (see docs/API.md).

    reset() restores the task's frozen instance (the graded birth state; `seed` is accepted and has no effect: every
    task is one fixed instance). step(action) runs one control step through the graded tooling's own step function.
    Observations: {"rgb": {camera: (H, W, 3) uint8}, "depth": {camera: (H, W) float32 m} (depth=True only),
    "proprio": (P,) float32} with the layout in `proprio_layout` / `proprio_groups` (the standard mode's)."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, task: TaskInfo, *, renderer: str, cameras, depth: bool, image_size, render_mode: str | None,
                 max_episode_steps: int | None, strict: bool):
        if renderer not in RENDERERS:
            raise ValueError(f"renderer must be one of {RENDERERS}")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        if render_mode == "rgb_array" and renderer == "none":
            raise ValueError("render_mode='rgb_array' needs a renderer ('nyx' or 'mujoco')")
        self.task = task
        self.renderer = renderer
        self.render_mode = render_mode
        self.strict = bool(strict)
        self.want_depth = bool(depth)
        self.image_size = None if image_size is None else (int(image_size[0]), int(image_size[1]))
        self.max_episode_steps = int(max_episode_steps or task.max_episode_steps)
        self._cameras_arg = cameras
        self._success_seen = False
        self._terminated = self._truncated = False
        self._closed = False
        self.step_after_end = False      # IKEA: True (physics continues after the episode end, as in the privileged mode)
        self.camera_names: list[str] = []

    # ---------------------------------------------------------------- subclass hooks
    def _reset_sim(self) -> None: raise NotImplementedError
    def _step_sim(self, action: np.ndarray) -> dict: raise NotImplementedError
    def _images(self, cameras: list[str], depth: bool) -> tuple[dict, dict]: raise NotImplementedError
    def _proprio(self) -> np.ndarray: raise NotImplementedError
    def _control_steps(self) -> int: raise NotImplementedError

    # ---------------------------------------------------------------- spaces
    def _setup_spaces(self, low, high, action_dtype, native_size: dict, depth_max: dict) -> None:
        all_cams = list(native_size)
        if self.renderer == "none":
            cams = []
        elif self._cameras_arg is None:
            cams = all_cams
        else:
            cams = [c for c in self._cameras_arg]
            bad = [c for c in cams if c not in all_cams]
            if bad:
                raise ValueError(f"unknown cameras {bad}; this task has {all_cams}")
        self.camera_names = cams
        self.all_camera_names = all_cams
        size = {c: (self.image_size or tuple(native_size[c])) for c in cams}
        obs = {}
        if cams:
            obs["rgb"] = spaces.Dict({c: spaces.Box(0, 255, (size[c][1], size[c][0], 3), np.uint8) for c in cams})
            if self.want_depth:
                obs["depth"] = spaces.Dict({c: spaces.Box(0.0, float(depth_max[c]), (size[c][1], size[c][0]), np.float32)
                                            for c in cams})
        n = len(self.proprio_layout)
        obs["proprio"] = spaces.Box(-np.inf, np.inf, (n,), np.float32)
        self.observation_space = spaces.Dict(obs)
        self.action_space = spaces.Box(np.asarray(low, dtype=action_dtype), np.asarray(high, dtype=action_dtype),
                                       (len(low),), action_dtype)

    # ---------------------------------------------------------------- Gymnasium API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Back to the frozen instance: the same birth state (birth_hash) whatever `seed` is (a fixed instance)."""
        self._check_open()
        if seed is not None:
            try:
                self.action_space.seed(seed)
            except Exception:  # noqa: BLE001
                pass
        self._reset_sim()
        self._success_seen = False
        self._terminated = self._truncated = False
        return self._observe(), self._info({"birth_hash": self.birth_hash})

    def step(self, action):
        """One control step (1 / control_hz s of simulated time) with `action` as the absolute joint targets (see
        action_layout). Out-of-limit values are clipped and the task's speed rules clamp a too-fast target; neither
        fails. -> (obs, reward, terminated, truncated, info)."""
        self._check_open()
        if self._truncated or (self._terminated and not self.step_after_end):
            raise RuntimeError("the episode has ended (terminated or truncated): call reset() before stepping again")
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != self.action_space.shape:
            raise ValueError(f"action must have {self.action_space.shape[0]} entries ({self.action_layout}), got {a.size}")
        if not np.all(np.isfinite(a)):
            raise ValueError("action contains NaN or inf")
        extra = self._step_sim(a)
        success = bool(extra.pop("success"))
        reward = 1.0 if success and not self._success_seen else 0.0
        self._success_seen = self._success_seen or success
        self._terminated = bool(extra.pop("terminated"))
        self._truncated = (not self._terminated) and self._control_steps() >= self.max_episode_steps
        info = self._info({"success": success, **extra})
        return self._observe(), reward, self._terminated, self._truncated, info

    def render(self):
        """render_mode 'rgb_array': all cameras tiled into one RGB image (3 per row)."""
        if self.render_mode != "rgb_array":
            return None
        cams = self.camera_names or self.all_camera_names
        rgb, _ = self._images(cams, False)
        tiles = [_resize(rgb[c], (424, 240)) for c in cams]
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        return np.concatenate([np.concatenate(tiles[i:i + 3], axis=1) for i in range(0, len(tiles), 3)], axis=0)

    def close(self) -> None:
        self._closed = True

    @property
    def unwrapped(self):
        return self

    # ---------------------------------------------------------------- helpers
    def _check_open(self):
        if self._closed:
            raise RuntimeError("the environment is closed")

    def _observe(self) -> dict:
        obs = {}
        if self.camera_names:
            rgb, dep = self._images(self.camera_names, self.want_depth)
            obs["rgb"] = {c: _resize(np.ascontiguousarray(rgb[c][..., :3]).astype(np.uint8), self.image_size)
                          for c in self.camera_names}
            if self.want_depth:
                obs["depth"] = {c: _resize(np.asarray(dep[c], dtype=np.float32), self.image_size, nearest=True)
                                for c in self.camera_names}
        obs["proprio"] = np.asarray(self._proprio(), dtype=np.float32)
        return obs

    def _info(self, extra: dict) -> dict:
        return {"control_step": self._control_steps(), "sim_time_s": self.sim_time_s, **extra}

    def describe(self) -> dict:
        """Everything about this task that the API exposes, as plain data (spaces, layouts, rates, limits, cameras)."""
        return {"task": self.task.to_dict(), "instruction": self.instruction, "control_hz": self.control_hz, "physics_substeps": self.physics_substeps,
                "action_layout": list(self.action_layout), "action_low": self.action_space.low.tolist(),
                "action_high": self.action_space.high.tolist(), "proprio_layout": list(self.proprio_layout),
                "proprio_groups": self.proprio_groups, "cameras": self.camera_info(),
                "max_episode_steps": self.max_episode_steps, "rules": self.rules_text(),
                "renderer": self.renderer, "graded_renderer": self.renderer == "nyx"}


def make(task_id: str, *, renderer: str = "nyx", cameras=None, depth: bool = False, image_size=None,
         render_mode: str | None = None, rules: str = "standard", max_episode_steps: int | None = None,
         strict: bool = True, **kwargs) -> RoboARCEnv:
    """Build the environment of one RoboARC task (see docs/API.md). One environment per process: the simulators are
    one-scene-per-process; use subprocess vector environments (gymnasium.vector.AsyncVectorEnv) for several.

    renderer   "nyx" (the graded renderer, GPU), "mujoco" (MuJoCo's raster of the physics model: a quick look, not the
               graded images) or "none" (no images: proprioception only, fastest).
    cameras    the cameras in the observation (default: all of the task's); depth=True adds z-depth.
    image_size (W, H) to resize every image to (default: the cameras' own 848 x 480).
    rules      "standard" (default): the standard mode's joint-speed rule on top of the tooling's; "privileged": the
               tooling's own rules only (what the privileged mode's agents and its verifier replays apply). IKEA: both
               are the same.
    strict     True (default): refuse to run outside the graded runtime (image, EPA-patched libmujoco, numpy masking,
               frozen files); False only warns.
    Further keyword arguments go to the suite's environment (IkeaEnv: check_every)."""
    info = resolve(task_id)
    if rules not in RULES:
        raise ValueError(f"rules must be one of {RULES}")
    if info.suite == "puzzles":
        from .puzzle import PuzzleEnv
        env = PuzzleEnv(info, renderer=renderer, cameras=cameras, depth=depth, image_size=image_size,
                        render_mode=render_mode, rules=rules, max_episode_steps=max_episode_steps, strict=strict, **kwargs)
    else:
        from .ikea import IkeaEnv
        env = IkeaEnv(info, renderer=renderer, cameras=cameras, depth=depth, image_size=image_size,
                      render_mode=render_mode, max_episode_steps=max_episode_steps, strict=strict, **kwargs)
    return env


def warn_or_raise(problems: list[str], strict: bool, exc=RuntimeError) -> None:
    if not problems:
        return
    if strict:
        raise exc("; ".join(problems))
    for p in problems:
        warnings.warn(f"roboarc: {p} (strict=False: not the graded runtime)", RuntimeWarning, stacklevel=3)


def env_flag(name: str, default: str = "") -> str:
    return os.environ.get(name, default)
