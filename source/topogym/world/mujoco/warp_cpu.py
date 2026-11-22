"""NVIDIA Warp's CPU setup for the VBD cloth (newton's SolverVBD on Warp's CPU backend; images/robotangle/README.md,
"VBD cloth"). Call init() before the first `import newton` and before any Warp kernel runs; it is idempotent.

- Compile target. Warp compiles CPU kernels with its embedded Clang, by default for the host (`-march=native`). init()
  pins the target instead (CPU_COMPILER_FLAGS: Clang frontend flags, since Warp only intercepts `-march=native` itself),
  so every host compiles the same code and module hashes do not depend on the host. x86-64-v3 = AVX2, FMA, BMI2, as on
  every x86 server since Haswell and Zen 1. It is part of every module's cache key.
- Kernel cache. The image compiles the kernels at build time into WARP_CACHE_PATH (/opt/warp-cache), read-only, so a
  process only loads them (a cold compile is about 30 s per process). If that cache is not writable for this user, Warp
  gets a per-user cache in the temp dir instead, whose module directories link to the image's. A module the image lacks
  then compiles there, and a non-root user never writes into the image.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

CPU_COMPILER_FLAGS = "-target-cpu x86-64-v3"

_ready = False


def link_image_cache(image: Path, own: Path) -> None:
    """Link every module directory of the image's cache into `own`, safely when many processes start at once (a
    validation round's workers, parallel trials): each link is made under a private temporary name and renamed into
    place (os.replace is atomic), so two processes linking the same module both succeed with the same link. A module
    `own` already holds (a link, or a directory this user compiled) is kept."""
    own.mkdir(parents=True, exist_ok=True)
    for module in image.iterdir():
        link = own / module.name
        if link.exists() or link.is_symlink():
            continue
        tmp = own / f".{module.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        tmp.symlink_to(module)
        try:
            os.replace(tmp, link)
        except OSError:
            tmp.unlink(missing_ok=True)
            if not (link.exists() or link.is_symlink()):      # not a lost race: a real failure
                raise


def init() -> None:
    global _ready
    if _ready:
        return
    import warp as wp
    wp.config.cpu_compiler_flags = CPU_COMPILER_FLAGS
    image = os.environ.get("WARP_CACHE_PATH")
    if image and (Path(image) / wp.config.version).is_dir() and not os.access(Path(image) / wp.config.version, os.W_OK):
        own = Path(tempfile.gettempdir()) / f"warp-cache-{os.getuid()}" / wp.config.version
        link_image_cache(Path(image) / wp.config.version, own)
        wp.config.kernel_cache_dir = str(own.parent)
    wp.init()
    _ready = True
