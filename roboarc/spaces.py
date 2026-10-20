"""Observation and action spaces. With Gymnasium installed these are gymnasium.spaces.Box / Dict and the environments
subclass gymnasium.Env; without it (the published images do not ship Gymnasium) a minimal stand-in with the same
attributes and methods is used (shape, dtype, low, high, sample(), contains(), seed(); Dict.spaces, keys(), [key])."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

try:   # pragma: no cover - which branch runs depends on the environment
    import gymnasium as _gym
    from gymnasium import spaces as _gs
    HAVE_GYMNASIUM = True
except ImportError:   # pragma: no cover
    _gym = _gs = None
    HAVE_GYMNASIUM = False


class _Box:
    """A box in R^n (gymnasium.spaces.Box's interface, the parts this package uses)."""

    def __init__(self, low, high, shape=None, dtype=np.float32, seed=None):
        self.dtype = np.dtype(dtype)
        if shape is None:
            shape = np.shape(low) if np.ndim(low) else np.shape(high)
        self.shape = tuple(int(s) for s in shape)
        self.low = np.broadcast_to(np.asarray(low, dtype=self.dtype), self.shape).copy()
        self.high = np.broadcast_to(np.asarray(high, dtype=self.dtype), self.shape).copy()
        self._rng = np.random.default_rng(seed)

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)
        return [seed]

    def sample(self):
        if self.dtype.kind in "ui":
            info = np.iinfo(self.dtype)
            lo = np.maximum(self.low, info.min).astype(np.int64)
            hi = np.minimum(self.high, info.max).astype(np.int64)
            return self._rng.integers(lo, hi + 1, size=self.shape).astype(self.dtype)
        lo = np.where(np.isfinite(self.low), self.low, -1.0).astype(np.float64)
        hi = np.where(np.isfinite(self.high), self.high, 1.0).astype(np.float64)
        return self._rng.uniform(lo, hi).astype(self.dtype)

    def contains(self, x) -> bool:
        x = np.asarray(x)
        return (x.shape == self.shape and np.can_cast(x.dtype, self.dtype, "same_kind")
                and bool(np.all(x >= self.low)) and bool(np.all(x <= self.high)))

    __contains__ = contains

    def __repr__(self):
        return f"Box({self.low.min() if self.low.size else ''}, {self.high.max() if self.high.size else ''}, {self.shape}, {self.dtype})"


class _Dict:
    """A dictionary of spaces (gymnasium.spaces.Dict's interface, the parts this package uses)."""

    def __init__(self, spaces: dict, seed=None):
        self.spaces = OrderedDict(spaces)
        self.seed(seed)

    def seed(self, seed=None):
        for i, s in enumerate(self.spaces.values()):
            s.seed(None if seed is None else seed + i)
        return [seed]

    def sample(self):
        return OrderedDict((k, s.sample()) for k, s in self.spaces.items())

    def contains(self, x) -> bool:
        return isinstance(x, dict) and set(x) == set(self.spaces) and all(self.spaces[k].contains(x[k]) for k in x)

    __contains__ = contains

    def keys(self):
        return self.spaces.keys()

    def items(self):
        return self.spaces.items()

    def __getitem__(self, key):
        return self.spaces[key]

    def __iter__(self):
        return iter(self.spaces)

    def __len__(self):
        return len(self.spaces)

    def __repr__(self):
        return "Dict(" + ", ".join(f"{k!r}: {v!r}" for k, v in self.spaces.items()) + ")"


def Box(low, high, shape=None, dtype=np.float32):
    if HAVE_GYMNASIUM:
        return _gs.Box(low=low, high=high, shape=shape, dtype=dtype)
    return _Box(low, high, shape, dtype)


def Dict(spaces: dict):
    if HAVE_GYMNASIUM:
        return _gs.Dict(spaces)
    return _Dict(spaces)


EnvBase = _gym.Env if HAVE_GYMNASIUM else object
