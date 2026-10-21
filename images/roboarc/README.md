# roboarc (the release image)

`ghcr.io/embodied-first-exam/roboarc:<VERSION>` (this directory's `VERSION`) is the image every RoboARC task runs in:
the privileged agent and verifier containers are built `FROM` it, and the standard mode's `sim` service runs it. Its
recipe is `source/images/roboarc/Dockerfile`; `build.sh` builds it. It holds:

- Python 3.11.15 on Debian 12 (the official Python base image, pinned by digest);
- the simulator source (`/opt/topogym/topogym`, the pruned `topogym` package with its MuJoCo core) and its hash
  `/opt/topogym/SOURCE.json`, which every task's runtime binding names;
- the Python packages the simulator, the task tooling, the graders and the renderer use: MuJoCo 3.14.0 with the EPA
  fix (a 14-byte patch of its `libmujoco.so`; `docs/simulator/MUJOCO_EPA_PATCH.md`), numpy 2.3.5, scipy 1.17.1,
  trimesh, coacd, the Nyx renderer `gs-nyx` 0.1.3, torch 2.11.0+cu128 (nyx_direct.py computes the camera poses with
  it) with the CUDA libraries it loads, the VBD cloth packages newton 1.6.1 and warp-lang 1.18.0 and their Warp CPU
  kernels, compiled at build into `/opt/warp-cache`;
- the Debian packages they need (ffmpeg for the replay videos, the GL / EGL / OpenGL / Vulkan loaders, X11 client
  libraries) and, for the agent harness, curl, ca-certificates and ripgrep;
- the robot assets (`/opt/topogym-assets/robots`), Nyx's studio HDRI and wood maps and the ground plane's checker
  texture (`/opt/topogym-visuals`), the NVIDIA Vulkan ICD and EGL vendor files;
- every task's frozen MuJoCo instance (`/opt/robotangle/instances/<base>/`), staged into a task by
  `/opt/robotangle/stage_instance.py` and checked against the task's `runtime_binding.json`.

The task tooling (`rcb_topogym.py`, `nyx_direct.py`, `render_bridge.py`) is not in the image: each task dir carries
its own pinned copy.

| version | image id | size | built from |
|---|---|---|---|
| 1.1.0 | `sha256:dadc2d710ad2…` | 8.7 GB | `source/images/roboarc/Dockerfile` (the Python base image) |
| 1.0.0 | `sha256:18a821a18d2b…` | 12.2 GB | a delta on the RoboTangle 0.2.22 image (Genesis 1.2.1, libuipc), pinned by digest |

## Pins

Everything the image installs is pinned, so a rebuild installs the same files (two builds on different hosts give the
same file contents; only the image id differs, by file times):

- the base image `python:3.11.15-slim-bookworm` by digest (`ARG PYTHON_BASE`);
- the Debian packages by exact version (the `apt-get install` list), from `snapshot.debian.org` at a fixed time
  (`ARG DEBIAN_SNAPSHOT`, the day the earlier base installed the same versions), so their dependencies resolve to the
  same versions too; the image keeps Debian's own source list for anyone who installs more;
- the Python packages by version and wheel sha256: `source/images/roboarc/runtime.lock` (installed with uv,
  `--no-deps --require-hashes`; uv itself is a build stage, hash-pinned, not in the image) and
  `source/images/roboarc/vbd-requirements.txt` (newton, warp-lang);
- the downloaded studio maps by sha256 (`source/images/roboarc/prepare_nyx_studio.py`), the MuJoCo EPA patch by the
  sha256 of the library before and after.

1.1.0 installs the same versions and the same wheel and Debian package files as 1.0.0: every library the graded
runtime loads in 1.0.0 (the Python interpreter, glibc / libm, libstdc++, numpy's and scipy's OpenBLAS, MuJoCo, Warp,
the CUDA and graphics libraries, ffmpeg) is byte-identical in 1.1.0, and every reference replays to the end
state it reaches in 1.0.0 (`docs/history/GENESIS_REMOVAL.md`).

## Host

- Linux x86_64 (not WSL: Nyx needs the NVIDIA Vulkan driver inside the container).
- An NVIDIA GPU with RT cores (GPUs without them, e.g. A100 and H100, cannot run Nyx) and driver >= 580 with its
  graphics (GL/Vulkan) user-space libraries; Docker with the NVIDIA Container Toolkit, the default runtime left at `runc`.
- Exactly one GPU per container: Nyx crashes when two are visible. The task compose files reserve one GPU with a
  device reservation (`count: 1`, `capabilities: [gpu]`) and no `runtime: nvidia`. `NVIDIA_DRIVER_CAPABILITIES=all`
  makes the toolkit mount the driver's graphics libraries, and the image's Vulkan ICD and EGL vendor files point the
  loaders at them. `VK_ICD_FILENAMES` names the image's ICD file: a toolkit in CDI mode also mounts the host's
  `/etc/vulkan/icd.d/nvidia_icd.json`, which would list the GPU twice.
- Physics runs on the CPU (MuJoCo, float64), except the blanket families' cloth step, which runs on the GPU on a
  deterministic solver.
- Running needs no network. The image is not cache-warmed: every container compiles its kernels on first use.

## Build (from the repository root)

```bash
images/roboarc/build.sh                       # instances from the HF dataset pinned in scripts/roboarc/assets.json
images/roboarc/build.sh --instances <dir>     # or a local copy: <dir>/<base>/instance.json + instance_assets/
images/roboarc/build.sh --noinst              # the image without instances (to make new instances in it)
```

`build.sh` is the build step of `tools/release/release_rebuild.py`:

1. the frozen instances: by default the pinned revision of the private HF dataset `eai-v2-internal/robotangle-assets`
   (`instances/` only; read access needed: `hf auth login` or `HF_TOKEN`, never printed), downloaded by
   `scripts/roboarc/assets.py fetch`; or `--instances <dir>`. Either way `assets.py check-instances` requires exactly
   the pinned bases and files, each with its sha256, before anything is built;
2. `docker buildx build` of `source/images/roboarc/Dockerfile` with `source/` as the context
   (`source/.dockerignore` keeps it to the paths the recipe copies) and the instances as the named build context
   `instances`;
3. the runtime check below; the tag is removed when the installed runtime differs from `scripts/roboarc/release.json`.

It needs Docker with BuildKit (the default `docker` driver), about 25 GB of free disk and network access to Docker Hub
(the base image), snapshot.debian.org, pypi.org and download.pytorch.org (the wheels) and dl.polyhaven.org (the studio
maps).

## Check a build

The image must install exactly the runtime every task is bound to (`build.sh` runs this):

```bash
docker run --rm ghcr.io/embodied-first-exam/roboarc:$(cat images/roboarc/VERSION) python -c 'import json; from topogym.simulator.frozen_binding import installed_runtime; print(json.dumps(installed_runtime(), sort_keys=True))' > /tmp/roboarc_runtime.json
python3 -c 'import json, sys; sys.exit(0 if json.load(open("/tmp/roboarc_runtime.json")) == json.load(open("scripts/roboarc/release.json"))["runtime"] else "image differs from the tasks runtime binding: do not run")'
```

The runtime is the simulator source's hash (`SOURCE.json`, re-hashed from the installed files), the Python version
and the versions of torch, numpy, scipy, mujoco, newton and warp-lang. Every Session checks it again at start
(`runtime_binding_ok` in reward.json) and refuses a different runtime.

## Archive and push

A validated build is not rebuilt for the registry: `images/archive.sh` exports it from the local Docker store into a
deterministic OCI archive (the config kept byte for byte, so the archive's config digest is the image id; the layers
compressed with fixed settings), whose manifest digest goes into `images/pins.json`. `images/push_all.sh` pushes that
archive byte for byte with RoboARC's other images (images/README.md). For 1.1.0: archive
`roboarc-1.1.0.oci.tar`, manifest digest `sha256:85ea90032a55…`, config digest = image id `sha256:dadc2d710ad2…`.

## Files

| file | what | pinned |
|---|---|---|
| `VERSION` | the image tag. `scripts/roboarc/mk_tasks.py` writes `ghcr.io/embodied-first-exam/roboarc:<VERSION>` into every task's Dockerfiles, compose files and README | changing it re-renders all 202 task dirs (new DIGESTs) |
| `build.sh` | the build (above); the archive and the push are `images/archive.sh` and `images/push_all.sh` | no (tools; not image inputs) |

The recipe and its inputs are under `source/` (`source/README.md`). Up to 1.0.x the image was built on the RoboTangle
0.2.22 image (Genesis 1.2.1, two libuipc builds); that recipe, its libuipc build files and the records of the libuipc
builds are in RoboARC 1.0.0 (`images/roboarc/` at main 153eb8cf01).
