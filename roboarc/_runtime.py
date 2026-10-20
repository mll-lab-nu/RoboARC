"""Where things are and which runtime settings the graded containers use. Standard library only (no numpy here: the
package sets numpy's SIMD masking before numpy is first imported, see roboarc/__init__.py)."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

# The graded containers' numpy / OpenBLAS settings (every puzzle task's Dockerfile, both IKEA images): SIMD masking, so
# every CPU class computes the same bytes where numpy takes part (asset generation, the tooling's arithmetic).
SIMD_MASK = "AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR"
GRADED_ENV = {"NPY_DISABLE_CPU_FEATURES": SIMD_MASK, "OPENBLAS_CORETYPE": "Haswell"}

# MuJoCo 3.14.0's wheel libmujoco.so with the EPA horizon fix (source/images/roboarc/mujoco-3.14.0-epa/): every image
# ships the patched file; the stock wheel's file has the first hash.
LIBMUJOCO_STOCK_SHA256 = "5e7623e30f55bf324d4c9648379ebeba5bcd153ee0304c52eac74bf8d441000b"
LIBMUJOCO_EPA_SHA256 = "52e2a24e927ccac047b13576ff9fcbe455bc9a12b0d2e0bf1868bb7ca5be2e4d"

# Paths inside the published images
IMAGE_TOPOGYM = Path("/opt/topogym")                     # roboarc: the simulator package (PYTHONPATH)
IMAGE_INSTANCES = Path("/opt/robotangle/instances")      # roboarc: every base's frozen instance
IMAGE_STAGE = Path("/opt/robotangle/stage_instance.py")  # roboarc: stages and checks one instance
IMAGE_IKEA_SCENES = Path("/opt/ikea/scenes")             # roboarc-ikea(-sim):1.0.0: the scenes
IMAGE_NYX_PYTHON = Path("/opt/nyx/bin/python")           # roboarc-ikea-sim:1.0.0: the Nyx render process


class RuntimeMismatch(RuntimeError):
    """The process does not run the graded runtime (image, libmujoco, numpy settings, frozen files). make() raises it
    with strict=True (the default); strict=False only warns, and the results are then not the graded ones."""


def repo_root() -> Path:
    """The RoboARC checkout this package belongs to (ROBOARC_REPO overrides): the directory holding tasks/ and scripts/."""
    env = os.environ.get("ROBOARC_REPO")
    root = Path(env).resolve() if env else Path(__file__).resolve().parent.parent
    if not (root / "scripts" / "roboarc" / "shared" / "rcb_topogym.py").is_file():
        raise FileNotFoundError(
            f"{root} is not a RoboARC checkout (no scripts/roboarc/shared/rcb_topogym.py): install the package with "
            "`pip install -e .` from the repository, or set ROBOARC_REPO to the checkout")
    return root


def cache_dir() -> Path:
    """Staged frozen instances and other scratch files (ROBOARC_CACHE, default ~/.cache/roboarc, or /tmp/roboarc-cache
    when the home directory is not writable)."""
    env = os.environ.get("ROBOARC_CACHE")
    if env:
        p = Path(env)
    else:
        home = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        p = home / "roboarc"
        try:
            p.mkdir(parents=True, exist_ok=True)
            if not os.access(p, os.W_OK):
                raise PermissionError(p)
        except OSError:
            p = Path("/tmp") / f"roboarc-cache-{os.getuid()}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def libmujoco_sha256() -> str | None:
    """sha256 of the libmujoco.so the installed mujoco package loads (None when mujoco is not installed)."""
    try:
        spec = importlib.util.find_spec("mujoco")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    folder = Path(list(spec.submodule_search_locations)[0])
    libs = sorted(folder.glob("libmujoco.so*"))
    real = [p for p in libs if not p.is_symlink()] or libs
    return file_sha256(real[0]) if real else None


# The numpy settings in force when numpy was first imported (roboarc/__init__.py records them: it sets the graded values
# when it is imported first, else it can only read what the environment held).
SETTINGS_AT_NUMPY_IMPORT: dict = {}


def record_numpy_settings() -> None:
    SETTINGS_AT_NUMPY_IMPORT.update({k: os.environ.get(k) for k in GRADED_ENV})


def numpy_masking_active() -> bool:
    """Whether numpy was imported with the graded SIMD masking and OpenBLAS core type in the environment."""
    seen = SETTINGS_AT_NUMPY_IMPORT or {k: os.environ.get(k) for k in GRADED_ENV}
    return all(seen.get(k) == v for k, v in GRADED_ENV.items())


def check_graded_runtime(strict: bool, *, need_epa: bool = True) -> list[str]:
    """The checks every backend shares: the EPA-patched libmujoco and numpy's masking. Returns the problems (raises
    RuntimeMismatch with strict=True when there are any)."""
    problems = []
    if not numpy_masking_active():
        problems.append("numpy was imported without the graded settings " + " ".join(f"{k}={v}" for k, v in GRADED_ENV.items())
                        + ": import roboarc before numpy, or set them in the environment (under pytest: "
                        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, a plugin may import numpy first)")
    if need_epa:
        sha = libmujoco_sha256()
        if sha is None:
            problems.append("mujoco is not installed")
        elif sha != LIBMUJOCO_EPA_SHA256:
            problems.append(f"libmujoco.so is not MuJoCo 3.14.0 with the EPA patch (sha256 {sha[:12]}, expected "
                            f"{LIBMUJOCO_EPA_SHA256[:12]}): run `python -m roboarc patch-mujoco` or use the published image")
    if problems and strict:
        raise RuntimeMismatch("; ".join(problems) + " (strict=False runs anyway, but not the graded runtime)")
    return problems


def in_puzzle_image() -> bool:
    return IMAGE_INSTANCES.is_dir() and IMAGE_STAGE.is_file()


def in_ikea_image() -> bool:
    return IMAGE_IKEA_SCENES.is_dir()
