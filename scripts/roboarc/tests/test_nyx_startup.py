"""Unit test of the Nyx renderer's start-up retry (scripts/roboarc/shared/nyx_direct.py: rig, NyxDirect._release), with a
stub build instead of gs-nyx (numpy only; no GPU; under a second): the first start; two failed builds, then ready; three
failed builds, which raise the last error with no 4th attempt; and a build that fails half way, which frees this
module's CUDA buffers and keeps the half-built native renderer untouched (shutting it down made the next build crash).
test_server_requests.py runs it.
usage: python3 scripts/roboarc/tests/test_nyx_startup.py [repo root]"""
import os, sys, types
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "scripts/roboarc/shared"))
import nyx_direct as N  # noqa: E402

N.STARTUP_BACKOFF_S = (0.0, 0.0)
N.DirectRig = lambda nyx, headless: types.SimpleNamespace(nyx=nyx, headless=headless)
errs, n = [], 0


def builds(outcomes):
    """from_world stand-in: one outcome per call, an exception to raise or None for a ready renderer."""
    calls = []

    def from_world(world, instance=None):
        calls.append(instance)
        o = outcomes[len(calls) - 1]
        if o is not None:
            raise o
        return types.SimpleNamespace(startup=None)
    return from_world, calls


for name, outcomes, want in (("first start", [None], 1),
                             ("two failures, then ready", [RuntimeError("Nyx could not allocate the scene buffers"),
                                                           RuntimeError("CUDA driver call cuMemAlloc failed with error 2"), None], 3)):
    N.NyxDirect.from_world, calls = builds(outcomes)
    r = N.rig("inst.json", "world", "headless"); n += 1
    st = r.nyx.startup
    if len(calls) != want or st["attempts"] != want or len(st["failed_attempts"]) != want - 1 or r.headless != "headless":
        errs.append(f"{name}: {len(calls)} builds, startup {st}")
    if want == 3 and "cuMemAlloc" not in st["failed_attempts"][1]["error"]:
        errs.append(f"{name}: failed attempts not recorded: {st['failed_attempts']}")

N.NyxDirect.from_world, calls = builds([RuntimeError(f"failure {i}") for i in range(1, 5)])
try:
    N.rig("inst.json", "world", "headless")
    errs.append("three failures: no error raised")
except RuntimeError as exc:
    if str(exc) != "failure 3" or len(calls) != 3:
        errs.append(f"three failures: raised {exc!r} after {len(calls)} builds (want 'failure 3' after 3)")
n += 1

released = []


class Step:
    def __init__(self, name, fail=False):
        self.name, self.fail = name, fail

    def __call__(self):
        released.append(self.name)
        if self.fail:
            raise RuntimeError(f"{self.name} failed")


def half_build(self, scene, folder, **_):
    self.nps = types.SimpleNamespace(shutdown=Step("sdk"))
    self._sdk_started = True
    self.cuda = types.SimpleNamespace(close=Step("cuda", fail=True))
    self.renderer = types.SimpleNamespace(unload_scene=Step("unload"), shutdown=Step("renderer"))
    raise RuntimeError("Nyx could not allocate the scene buffers")


N.NyxDirect._start = half_build
try:
    nyx = N.NyxDirect("scene", "folder")
    errs.append("half build: no error raised")
except RuntimeError as exc:
    if "allocate" not in str(exc):
        errs.append(f"half build: raised {exc!r}")
n += 1
if released != ["cuda"] or len(N._FAILED_BUILDS) != 1 or N._FAILED_BUILDS[0].shutdown.name != "renderer":
    errs.append(f"half build: released {released} (want only cuda), kept {len(N._FAILED_BUILDS)} renderer(s) (want 1)")
for e in errs:
    print("FAIL", e)
print(f"{'FAIL' if errs else 'ok'} roboarc Nyx start-up retry: {n} checks, {len(errs)} failed")
sys.exit(1 if errs else 0)
