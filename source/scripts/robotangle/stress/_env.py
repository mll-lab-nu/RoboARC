"""Process setup for every stress tool. Import it before numpy.

- The MuJoCo task containers' numpy / OpenBLAS SIMD masking, so a run on an AVX-512 host and one on an AVX2 host end on
  the same state hash (checked 2026-10-09: Zen 4 vs i9-14900HX, 5 of 5 replays identical).
- One thread per library: MuJoCo is single-threaded, so parallelise across processes, never inside one.
- Headless: no GPU, software GL. The stress tools never render.
- The repository root and the RoboTangle tooling directory go first on sys.path, so `topogym` and `rcb_topogym` come from
  this checkout. A variable that is already set wins; that is how to run against an installed image instead."""
import os
import sys
from pathlib import Path

for key, value in {
        "NPY_DISABLE_CPU_FEATURES": "AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR",
        "OPENBLAS_CORETYPE": "Haswell",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "RCB_TORCH_THREADS": "1",
        "RCB_SESSION_LOG": "", "CUDA_VISIBLE_DEVICES": "", "LIBGL_ALWAYS_SOFTWARE": "1", "PYOPENGL_PLATFORM": "egl"}.items():
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if os.environ.get("STRESS_USE_INSTALLED") != "1":        # 1: import topogym / rcb_topogym from the environment instead
    for p in (REPO / "scripts/robotangle/shared", REPO):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def provenance() -> dict:
    """Where the physics came from: the modules actually imported, a hash of the stress tools, and the checkout's
    commit when there is one."""
    out = {}
    for name in ("topogym", "rcb_topogym", "mujoco"):
        m = sys.modules.get(name)
        if m is not None:
            out[f"{name}_file"] = getattr(m, "__file__", None)
    mj = sys.modules.get("mujoco")
    if mj is not None:
        out["mujoco_version"] = getattr(mj, "__version__", None)
    import hashlib                         # the stress tools themselves, also where there is no git checkout
    h = hashlib.sha256()
    for f in sorted(HERE.glob("*.py")):
        h.update(f.name.encode() + b"\0" + f.read_bytes())
    out["stress_tools_sha256"] = h.hexdigest()[:16]
    try:
        import subprocess
        out["repo_commit"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                                            timeout=10).stdout.strip() or None
    except Exception:
        out["repo_commit"] = None
    return out
