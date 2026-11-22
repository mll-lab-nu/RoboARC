"""Unit test of Nyx's device check (nyx_direct.py: device_problem, check_device, NyxDirect's build, rig), with mocked
CUDA devices, Vulkan extension lists and a stub gs-nyx instead of a GPU (numpy only; under a second). Both copies are
tested: the puzzles' scripts/roboarc/shared/nyx_direct.py and IKEA's ikea/scripts/ikea_mujoco/standard/nyx_direct.py.
The extension lists are the ones measured on 2026-10-21 (NVIDIA driver 610.43.02; Mesa 25.0.7 lavapipe), cut to the
extensions that matter:
  - an RTX GPU (ray queries): renders, the build goes on;
  - an A100 / H100 (ray-tracing pipeline and acceleration structures, no ray queries): UnsupportedDevice naming
    VK_KHR_ray_query and the GPUs Nyx runs on, raised before any native renderer exists;
  - a CPU Vulkan driver (lavapipe: ray queries, but not the CUDA device): UnsupportedDevice;
  - no Vulkan driver at all: UnsupportedDevice;
  - a check that cannot complete (an unexpected Vulkan error): no verdict, the build goes on as before;
  - NyxDirect: on an A100 the build stops at the check, after the CUDA device is open and before NyxRenderer() exists;
    on an RTX GPU it creates the renderer as before;
  - rig() (puzzles): UnsupportedDevice is raised at once (one build, no retry) and later calls raise it without a build.
usage: python3 tools/tests/test_nyx_device_check.py [repo root]"""
import importlib.util, os, sys, types
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, "..", ".."))
sys.dont_write_bytecode = True
COPIES = {"puzzles": "scripts/roboarc/shared/nyx_direct.py", "ikea": "ikea/scripts/ikea_mujoco/standard/nyx_direct.py"}

UUID = bytes(range(16))
COMMON = {"VK_KHR_external_memory_fd", "VK_KHR_external_semaphore_fd", "VK_EXT_extended_dynamic_state3",
          "VK_KHR_compute_shader_derivatives", "VK_KHR_deferred_host_operations"}
RTX = COMMON | {"VK_KHR_ray_query", "VK_KHR_acceleration_structure", "VK_KHR_ray_tracing_pipeline"}
A100 = COMMON | {"VK_KHR_acceleration_structure", "VK_KHR_ray_tracing_pipeline", "VK_NV_ray_tracing"}
LAVAPIPE = (COMMON - {"VK_KHR_external_semaphore_fd"}) | {"VK_KHR_ray_query", "VK_KHR_acceleration_structure"}
errs, n = [], 0


def load(tag: str, rel: str):
    spec = importlib.util.spec_from_file_location(f"nyx_direct_{tag}", os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                  # dataclasses look their module up there
    spec.loader.exec_module(mod)
    return mod


class Cuda:
    """_Cuda stand-in: the device's name and UUID, and what the build does with it."""
    def __init__(self, name="NVIDIA GeForce RTX 4090", device=0):
        self.name = name

    def identity(self):
        return self.name, UUID

    def current(self):
        pass

    def close(self):
        pass


def mock_vulkan(N, devices):
    """N._vulkan_devices returns `devices` (a list), or raises it (an exception)."""
    def vulkan_devices(loader=None):
        if isinstance(devices, BaseException):
            raise devices
        return devices
    N._vulkan_devices = vulkan_devices


def check(N, tag, name, devices, *, raises, words=()):
    """check_device on a mocked CUDA device and _vulkan_devices outcome."""
    global n
    n += 1
    N._UNSUPPORTED.clear()
    mock_vulkan(N, devices)
    try:
        N.check_device(Cuda(name))
        if raises:
            errs.append(f"{tag} {name}: no UnsupportedDevice")
    except N.UnsupportedDevice as exc:
        if not raises:
            errs.append(f"{tag} {name}: raised {exc}")
        elif not all(w in str(exc) for w in words):
            errs.append(f"{tag} {name}: message {str(exc)!r} lacks {words}")


def stub_gs_nyx(created: list):
    """gs_nyx.nyx_py_sdk / nyx_py_renderer stand-ins: NyxRenderer() records that it was created, then stops the build."""
    class Stop(Exception):
        pass

    class Renderer:
        def __init__(self):
            created.append("NyxRenderer")
            raise Stop()
    sdk = types.SimpleNamespace(startup=lambda: None, shutdown=lambda: None)
    renderer = types.SimpleNamespace(NyxRenderer=Renderer)
    pkg = types.ModuleType("gs_nyx")
    pkg.nyx_py_sdk, pkg.nyx_py_renderer = sdk, renderer
    sys.modules.update({"gs_nyx": pkg, "gs_nyx.nyx_py_sdk": sdk, "gs_nyx.nyx_py_renderer": renderer})
    return Stop


SCENE = types.SimpleNamespace(export=lambda folder, camera_device="cpu": dict(
    path=os.path.join(str(folder), "nyx_scene.json"), slots=[], references=[], deformables=[],
    cameras=[{"name": "cam_high", "res": [848, 480]}]))

for tag, rel in COPIES.items():
    N = load(tag, rel)
    check(N, tag, "NVIDIA GeForce RTX 4090", [("NVIDIA GeForce RTX 4090", UUID, RTX)], raises=False)
    check(N, tag, "NVIDIA A100-PCIE-40GB", [("NVIDIA A100-PCIE-40GB", UUID, A100)], raises=True,
          words=("VK_KHR_ray_query", "A100-PCIE-40GB", "RT cores", "L40S", "H100", "no software fallback"))
    check(N, tag, "NVIDIA H100 80GB HBM3", [("NVIDIA H100 80GB HBM3", UUID, A100)], raises=True, words=("VK_KHR_ray_query",))
    check(N, tag, "NVIDIA A100-PCIE-40GB", [("llvmpipe (LLVM 15.0.6, 256 bits)", bytes(16), LAVAPIPE)], raises=True,
          words=("no Vulkan device is the CUDA device", "llvmpipe"))
    check(N, tag, "NVIDIA A100-PCIE-40GB", N._NoVulkanDriver(), raises=True, words=("no Vulkan driver",))
    check(N, tag, "NVIDIA GeForce RTX 4090", RuntimeError("vkCreateInstance returned -3"), raises=False)
    # two devices, the CUDA device's twin has ray queries: renders (the other device does not matter)
    check(N, tag, "NVIDIA GeForce RTX 4090", [("llvmpipe", bytes(16), LAVAPIPE - {"VK_KHR_ray_query"}),
                                              ("NVIDIA GeForce RTX 4090", UUID, RTX)], raises=False)
    missing = N.device_problem("x", UUID, [("x", UUID, RTX - {"VK_KHR_acceleration_structure", "VK_KHR_ray_query"})])
    if N.device_problem("x", UUID, [("x", UUID, RTX)]) is not None or "VK_KHR_ray_query, VK_KHR_acceleration_structure" not in (missing or ""):
        errs.append(f"{tag} device_problem: {missing!r}")
    n += 1

    # the build: the check runs after the CUDA device is open and before the native renderer exists
    for gpu, ext, want in (("NVIDIA A100-PCIE-40GB", A100, []), ("NVIDIA GeForce RTX 4090", RTX, ["NyxRenderer"])):
        N._UNSUPPORTED.clear()
        created = []
        Stop = stub_gs_nyx(created)
        mock_vulkan(N, [(gpu, UUID, ext)])
        N._Cuda = lambda device=0, gpu=gpu: Cuda(gpu, device)
        N._preload_nvrtc = lambda: None
        try:
            N.NyxDirect(SCENE, "/nonexistent/nyx")
            errs.append(f"{tag} build on {gpu}: no error")
        except N.UnsupportedDevice as exc:
            if want:
                errs.append(f"{tag} build on {gpu}: {exc}")
        except Stop:
            if not want:
                errs.append(f"{tag} build on {gpu}: the native renderer was created")
        if created != want:
            errs.append(f"{tag} build on {gpu}: created {created}, want {want}")
        n += 1
    for k in ("gs_nyx", "gs_nyx.nyx_py_sdk", "gs_nyx.nyx_py_renderer"):
        sys.modules.pop(k, None)

    if tag != "puzzles":
        continue
    # rig(): one build that raises UnsupportedDevice, no retry; then no build at all
    N._UNSUPPORTED.clear()
    N.STARTUP_BACKOFF_S = (0.0, 0.0)
    builds = []

    def from_world(world, instance=None):
        builds.append(instance)
        N._UNSUPPORTED.append("Nyx cannot render on this host: test")
        raise N.UnsupportedDevice(N._UNSUPPORTED[0])

    N.NyxDirect.from_world = from_world
    N.DirectRig = lambda nyx, headless: types.SimpleNamespace(nyx=nyx, headless=headless)
    for k in range(2):
        try:
            N.rig("inst.json", "world", "headless")
            errs.append(f"{tag} rig call {k + 1}: no error raised")
        except N.UnsupportedDevice:
            pass
        n += 1
    if len(builds) != 1:
        errs.append(f"{tag} rig: {len(builds)} builds (want 1: no retry, and none once the device is known)")
    if not issubclass(N.UnsupportedDevice, RuntimeError):
        errs.append(f"{tag}: UnsupportedDevice is not a RuntimeError (the grader's soft video handling catches Exception)")
for e in errs:
    print("FAIL", e)
print(f"{'FAIL' if errs else 'ok'} Nyx device check (puzzles and IKEA copies): {n} checks, {len(errs)} failed")
sys.exit(1 if errs else 0)
