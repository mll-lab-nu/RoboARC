"""RoboARC's raw simulator interface: the benchmark's tasks as Gymnasium-compatible environments over the same code
the graded tasks run (see docs/API.md).

    import roboarc
    env = roboarc.make("roboarc-orbits-02-i01")          # or "orbits-02", "roboarc-ikea-lack-i00", "lack"
    obs, info = env.reset()                               # the frozen instance: info["birth_hash"]
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Import this package before numpy: it sets numpy's SIMD masking the way the graded containers do (NPY_DISABLE_CPU_FEATURES,
OPENBLAS_CORETYPE), which numpy reads when it is first imported.
"""
import os as _os
import sys as _sys

from . import _runtime as _rt

if "numpy" not in _sys.modules:
    for _k, _v in _rt.GRADED_ENV.items():
        _os.environ.setdefault(_k, _v)
if not _rt.SETTINGS_AT_NUMPY_IMPORT:
    _rt.record_numpy_settings()

__version__ = "1.1.0"

from ._runtime import RuntimeMismatch  # noqa: E402
from .registry import TaskInfo, list_tasks, resolve, task_info  # noqa: E402


def make(task_id, **kwargs):
    """Build one task's environment; see roboarc.core.make and docs/API.md for the options."""
    from .core import make as _make
    return _make(task_id, **kwargs)


def load_reference(task_id, mode="privileged"):
    """The task's reference trajectory (default: the privileged mode's solution) as an (N, action_dim) array, or None."""
    from .references import load_reference as _load
    return _load(task_id, mode)


__all__ = ["make", "list_tasks", "task_info", "resolve", "load_reference", "TaskInfo", "RuntimeMismatch", "__version__"]
