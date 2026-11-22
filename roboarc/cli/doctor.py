"""roboarc doctor: can this machine run RoboARC? On a host it asks the tools the benchmark needs (nvidia-smi,
vulkaninfo, docker, harbor, through scripts/harbor/host_info.py where that already reads them) and looks for the pinned
images; inside a RoboARC image it reports the graded runtime, as `roboarc check` does. Standard library only."""
from __future__ import annotations

import json
import re
import shutil

from . import _common as C

DRIVER_MIN = 580                    # the NVIDIA driver RoboARC states (docs/RUNNING.md)
HARBOR_MIN = (0, 23)
PYTHON_MIN = (3, 11)
VULKAN = ("VK_KHR_ray_query", "VK_EXT_extended_dynamic_state3", "VK_KHR_compute_shader_derivatives")
OK, FAIL, UNKNOWN = "ok", "FAIL", "?"


def check(name: str, status: str, detail: str, fix: str = "") -> dict:
    return {"name": name, "status": status, "detail": detail, "fix": fix if status != OK else ""}


def _version(text: str | None) -> tuple | None:
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text or "")
    return tuple(int(x) for x in m.groups(default="0")) if m else None


def _python3() -> dict:
    rc, out = C.capture(["python3", "-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])"])
    v = _version(out) if rc == 0 else None
    if v is None:
        return check("python3", FAIL, "no `python3` on PATH", "the repository's scripts call `python3`: put a Python "
                     f"{PYTHON_MIN[0]}.{PYTHON_MIN[1]} or newer first on PATH")
    if v[:2] < PYTHON_MIN:
        return check("python3", FAIL, f"`python3` on PATH is {out}", "the repository's scripts call `python3` and need "
                     f"{PYTHON_MIN[0]}.{PYTHON_MIN[1]} or newer: activate the environment roboarc is installed in")
    return check("python3", OK, out)


def _repository() -> dict:
    import roboarc
    from ..registry import list_tasks
    n = len(list_tasks())
    return check("repository", OK if n else FAIL, f"{C.repo()}: RoboARC {roboarc.__version__}, {n} tasks",
                 "no task directory under tasks/: is the checkout complete?")


def _gpus(host: dict) -> list[dict]:
    gpus = host.get("gpus") or []
    if not gpus:
        return [check("GPU", FAIL, "nvidia-smi lists no GPU", f"install the NVIDIA driver, {DRIVER_MIN} or newer"),
                check("NVIDIA driver", UNKNOWN, "no GPU listed")]
    names = "; ".join(f"{g['index']}: {g['name']}, {g['memory_mib']} MiB" for g in gpus)
    drivers = sorted({g["driver"] for g in gpus})
    old = [d for d in drivers if (_version(d) or (0,))[0] < DRIVER_MIN]
    return [check("GPU", OK, names),
            check("NVIDIA driver", FAIL if old else OK, ", ".join(drivers) + ("" if not old else f" (below {DRIVER_MIN})"),
                  f"install NVIDIA driver {DRIVER_MIN} or newer")]


def _vulkan() -> dict:
    name = "Vulkan ray query"
    if shutil.which("vulkaninfo") is None:
        return check(name, UNKNOWN, "vulkaninfo not found", "install the package vulkan-tools and run `roboarc doctor` "
                     "again: the renderer needs " + ", ".join(VULKAN))
    rc, out = C.capture(["vulkaninfo"], timeout=120)
    missing = [x for x in VULKAN if x not in out]
    if not missing:
        return check(name, OK, "vulkaninfo names " + ", ".join(VULKAN))
    if rc != 0:
        return check(name, UNKNOWN, f"vulkaninfo failed: {out.splitlines()[-1] if out else 'no output'}",
                     "run `vulkaninfo` yourself: it must name " + ", ".join(VULKAN))
    return check(name, FAIL, "vulkaninfo does not name " + ", ".join(missing), "the renderer needs a GPU with RT cores "
                 "(RTX 20 series or newer, RTX workstation cards, T4, L4, L40, A10, A40) and a current driver")


def _docker() -> tuple[list[dict], bool]:
    rc, out = C.capture(["docker", "info", "--format", "{{.ServerVersion}}"])
    if rc != 0:
        why = "docker not found" if rc is None else ("the Docker daemon does not answer: "
                                                      + (out.splitlines() or ["no message"])[-1])
        fix = ("install Docker Engine (https://docs.docker.com/engine/install/)" if rc is None else
               "start the daemon, and add your user to the group `docker`")
        return [check("Docker", FAIL, why, fix),
                check("NVIDIA Container Toolkit", UNKNOWN, "not looked for (Docker does not answer)")], False
    rows = [check("Docker", OK, f"server {out}")]
    rc, out = C.capture(["docker", "compose", "version", "--short"])
    rows.append(check("Docker Compose", OK if rc == 0 else FAIL, out if rc == 0 else "`docker compose` is not installed",
                      "install the Docker Compose plugin: Harbor starts every task with `docker compose`"))
    rc, runtimes = C.capture(["docker", "info", "--format", "{{json .Runtimes}}"])
    ctk, ver = C.capture(["nvidia-ctk", "--version"])
    found = (rc == 0 and '"nvidia"' in runtimes) or ctk == 0
    rows.append(check("NVIDIA Container Toolkit", OK if found else FAIL,
                      (ver.splitlines()[0] if ctk == 0 and ver else "Docker has the runtime `nvidia`") if found
                      else "no `nvidia` runtime in Docker and no nvidia-ctk on PATH",
                      "install the NVIDIA Container Toolkit "
                      "(https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)"))
    return rows, True


def _harbor(host: dict) -> dict:
    v = _version(host.get("harbor"))
    want = f"{HARBOR_MIN[0]}.{HARBOR_MIN[1]}"
    if v is None:
        return check("Harbor", FAIL, "harbor not found", f"install Harbor {want} or newer "
                     "(https://github.com/laude-institute/harbor): it runs the tasks for coding agents and for "
                     "`roboarc selftest`")
    return check("Harbor", OK if v[:2] >= HARBOR_MIN else FAIL, host["harbor"], f"Harbor {want} or newer is needed")


def _images(docker_ok: bool) -> dict:
    name = "pinned images"
    rc, out = C.capture(C.command("images/pull_all.sh", "--list"))
    if rc != 0:
        return check(name, FAIL, out.splitlines()[-1] if out else "images/pull_all.sh --list failed",
                     "images/pins.json must list the images")
    pins = [dict(zip(("ref", "pinned_id"), line.split())) for line in out.splitlines() if line.strip()]
    if not docker_ok:
        return check(name, UNKNOWN, f"{len(pins)} pinned, not looked for (Docker does not answer)")
    for p in pins:                      # Docker resolves a pinned reference only to the image of that digest
        rc, local = C.capture(["docker", "image", "inspect", "--format", "{{.Id}}", p["ref"]])
        p["local_id"] = local if rc == 0 else None
    missing = [p["ref"].split("@")[0] for p in pins if p["local_id"] is None]
    row = (check(name, FAIL, f"{len(pins) - len(missing)} of {len(pins)} are local; missing: {', '.join(missing)}",
                 "roboarc images pull") if missing else
           check(name, OK, f"{len(pins)} of {len(pins)} are local, by their pinned digests"))
    return dict(row, images=pins)


def host_checks() -> list[dict]:
    rc, out = C.capture(C.command("scripts/harbor/host_info.py"))
    try:
        host = json.loads(out) if rc == 0 else {}
    except ValueError:
        host = {}
    docker_rows, docker_ok = _docker()
    return [_python3(), _repository(), *_gpus(host), _vulkan(), *docker_rows, _harbor(host), _images(docker_ok)]


def image_checks(report: dict) -> list[dict]:
    """The graded runtime (tasks.runtime_report) as checks: one line when it is, one per problem when it is not."""
    which = "the puzzle image" if report["puzzle_image"] else "an IKEA image"
    rows = [check("image", OK, f"{which}; repository {report['repo']}")]
    if not report["problems"]:
        rows.append(check("graded runtime", OK, f"libmujoco {(report['libmujoco_sha256'] or '')[:12]} (MuJoCo 3.14.0 "
                          "with the EPA patch), numpy's graded settings"))
    return rows + [check("graded runtime", FAIL, p) for p in report["problems"]]


def _doctor(a) -> int:
    import roboarc
    from .. import _runtime as R
    from .tasks import runtime_report
    in_image = R.in_puzzle_image() or R.in_ikea_image()
    out = {"roboarc": roboarc.__version__, "where": "image" if in_image else "host"}
    if in_image:
        out["runtime"] = runtime_report()
        out["checks"] = image_checks(out["runtime"])
    else:
        out["checks"] = host_checks()
    failed = [c for c in out["checks"] if c["status"] == FAIL]
    unknown = [c for c in out["checks"] if c["status"] == UNKNOWN]
    out["ok"] = not failed
    if a.json:
        print(json.dumps(out, indent=1))
        return 0 if out["ok"] else 1
    print(f"RoboARC {out['roboarc']}, " + ("inside a RoboARC image: the graded runtime" if in_image else "this host"))
    for c in out["checks"]:
        print(f"  {c['status']:4s}  {c['name']:25s} {c['detail']}")
        if c["fix"]:
            print(f"{'':33s} -> {c['fix']}")
    notes = ([f"{len(failed)} failed"] if failed else []) + ([f"{len(unknown)} could not be made"] if unknown else [])
    print("doctor: " + (", ".join(notes) if notes else "every check passed"))
    return 0 if out["ok"] else 1


def add_parsers(sub) -> None:
    p = sub.add_parser("doctor", formatter_class=C.Formatter,
                       help="can this machine run RoboARC? (inside an image: is this the graded runtime?)",
                       description=f"""\
Can this machine run RoboARC? One line per check, and for each one that fails, what to do about it.

On a host:
  python3                    the `python3` on PATH is {PYTHON_MIN[0]}.{PYTHON_MIN[1]} or newer (the repository's scripts call it)
  repository                 the checkout this package belongs to, its version and number of tasks
  GPU, NVIDIA driver         nvidia-smi lists a GPU; the driver is {DRIVER_MIN} or newer
  Vulkan ray query           vulkaninfo names {VULKAN[0]}, {VULKAN[1]} and
                             {VULKAN[2]} (the renderer needs a GPU with RT cores)
  Docker, Docker Compose     the daemon answers; `docker compose` is installed
  NVIDIA Container Toolkit   Docker can give a container the GPU
  Harbor                     {HARBOR_MIN[0]}.{HARBOR_MIN[1]} or newer
  pinned images              every image of images/pins.json is local, found by its pinned digest
                             (`roboarc images check` compares the image ids too)
Inside a RoboARC image: the graded runtime instead (the EPA-patched libmujoco, numpy's settings).

It starts no container and pulls nothing. A check whose tool is missing (vulkaninfo) is shown as `?`, not as failed.
Exit status: 0 when no check failed, 1 otherwise.""", epilog="""\
examples:
  roboarc doctor
  roboarc doctor --json > doctor.json      # to send with a problem report""")
    p.add_argument("--json", action="store_true", help="print one JSON object instead of the table: roboarc (the "
                   "version), where (host or image), ok, checks (name, status, detail, fix; the pinned images' check "
                   "also has images: ref, pinned_id, local_id) and, inside an image, runtime (what `roboarc check` "
                   "prints)")
    p.set_defaults(fn=_doctor)
