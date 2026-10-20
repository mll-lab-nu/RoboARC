# roboarc (the release image)

`ghcr.io/embodied-first-exam/roboarc:<VERSION>` (this directory's `VERSION`; 1.0.0 is image id `sha256:18a821a18d2b…`)
is the image every RoboARC task runs in: the privileged agent and verifier containers are built `FROM` it, and the
standard mode's `sim` service runs it. It holds:

- the simulator source (`/opt/topogym/topogym`, the pruned `topogym` package with its MuJoCo core) and its hash
  `/opt/topogym/SOURCE.json`, which every task's runtime binding names;
- the robot assets (`/opt/topogym-assets/robots`);
- the VBD cloth packages (newton 1.6.1, warp-lang 1.18.0) and their Warp CPU kernels, compiled at build into
  `/opt/warp-cache`;
- every task's frozen MuJoCo instance (`/opt/robotangle/instances/<base>/`), staged into a task by
  `/opt/robotangle/stage_instance.py` and checked against the task's `runtime_binding.json`;
- MuJoCo 3.14.0 with the EPA fix (a 14-byte patch of its `libmujoco.so`; `docs/simulator/MUJOCO_EPA_PATCH.md`);
- everything else from the Genesis base image, below.

The task tooling (`rcb_topogym.py`, `nyx_direct.py`, `render_bridge.py`) is not in the image: each task dir carries
its own pinned copy.

## The Genesis base image

The release image is a delta on top of the published RoboTangle image 0.2.22, pinned by digest in the recipe:
`ghcr.io/mll-lab-nu/efe:robotangle-0.2.22@sha256:eb1a75177bf3…` (the same image was first published as
`ghcr.io/mll-lab-nu/eai-robotangle:0.2.22`). The base provides Python 3.11.15, numpy 2.3.5 and scipy 1.17.1 (their
OpenBLAS kernels decide the bytes of the frozen assets, hence the SIMD masking every task container sets), mujoco
3.14.0 (installed as a genesis-world dependency), coacd, torch 2.11.0+cu128, the Nyx renderer (`gs-nyx`, which
`nyx_direct.py` drives), the Vulkan ICD and EGL vendor files, the Nyx studio HDRI, Genesis 1.2.1 and two builds of
libuipc.

In RoboARC 1.0.0 every base runs on MuJoCo and every family renders with `nyx_direct.py`: no task steps Genesis or
imports libuipc. They stay because the runtime identity pins them. `topogym.simulator.frozen_binding.installed_runtime()`
records the versions of genesis-world, pyuipc, torch, numpy, scipy, mujoco, newton and warp-lang and the hash of
`/opt/uipc-build.json`; every task's `runtime_binding.json` holds that runtime (`scripts/roboarc/release.json`
`runtime`, and both libuipc build records in `uipc_builds`), and a Session refuses any other one. A base without
Genesis and libuipc is a new runtime: new bindings for all 202 task dirs and every reference replayed again (a 2.0
change, not a 1.x one).

The base's recipe is kept here for the record, `Dockerfile.genesis-0.2.22`, with the files it copies from this
directory: `stage_instance.py`, `install_uipc_prefix.py` and `verify_uipc_wheel.py` are the bytes the base image
holds (their comments use the paths of the repository they were written in), and `uipc-sm86-build/` is the recipe of
the sm_86 libuipc build. It cannot be built from this repository: its context was a release tree archive and the
libuipc wheels, kept in the private HF dataset (`image/`), not here. 0.2.22 itself was published as a delta of 0.2.21
(`ghcr.io/mll-lab-nu/eai-robotangle:0.2.21@sha256:4fd681598e07…`) plus one `COPY` layer with the eight IPC instances'
`instance_assets/frozen_assets/rope_volume.npz`.

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
2. `docker buildx build` of `source/images/robotangle/Dockerfile.mujoco` with `source/` as the context
   (`source/.dockerignore` keeps it to the paths the recipe copies) and the instances as the named build context
   `instances`, FROM the base image above;
3. the runtime check below; the tag is removed when the installed runtime differs from `scripts/roboarc/release.json`.

It needs Docker with BuildKit (the default `docker` driver), about 25 GB of free disk (the 11.3 GB image plus the
BuildKit cache) and network access to ghcr.io (the base image) and pypi.org (the VBD wheels, hash-pinned).

Keep the bytes of every file the recipe copies (under `source/`): a changed input misses the build cache, and a
cache-cold build writes `.pyc` files whose source-mtime field differs from the released image's, so the image is no
longer file-for-file the release. A rebuild is checked against the release by comparing file manifests.

## Check a build

The image must install exactly the runtime every task is bound to (`build.sh` runs this):

```bash
docker run --rm ghcr.io/embodied-first-exam/roboarc:$(cat images/roboarc/VERSION) python -c 'import json; from topogym.simulator.frozen_binding import installed_runtime; print(json.dumps(installed_runtime(), sort_keys=True))' > /tmp/roboarc_runtime.json
python3 -c 'import json, sys; sys.exit(0 if json.load(open("/tmp/roboarc_runtime.json")) == json.load(open("scripts/roboarc/release.json"))["runtime"] else "image differs from the tasks runtime binding: do not run")'
```

Every Session checks it again at start (`runtime_binding_ok` in reward.json) and refuses a different runtime, including
the two libuipc build records `scripts/roboarc/release.json` `uipc_builds` names:

```bash
docker run --rm ghcr.io/embodied-first-exam/roboarc:$(cat images/roboarc/VERSION) sh -c 'sha256sum /opt/uipc-build.json /opt/uipc_sm86/uipc-build.json; cat /opt/uipc-check.json'
```

## Push

```bash
<token source> | GHCR_USER=<github user> images/roboarc/push.sh [--log <dir>] <ref>=<expected image id> ...
```

`push.sh` pushes local tags after a verified build. Each tag must be its expected image id, or nothing is pushed. The
registry token (write:packages) is read from stdin only and lives in a temporary Docker config that is removed on
exit. After each push the pushed digest must equal the remote manifest's, and the remote config digest the image id.
New packages start private.

## Files

| file | what | pinned |
|---|---|---|
| `VERSION` | the image tag. `scripts/roboarc/mk_tasks.py` writes `ghcr.io/embodied-first-exam/roboarc:<VERSION>` into every task's Dockerfiles, compose files and README | changing it re-renders all 202 task dirs (new DIGESTs) |
| `build.sh`, `push.sh` | the build (above) and the push | no (tools; not image inputs) |
| `Dockerfile.genesis-0.2.22`, `stage_instance.py`, `install_uipc_prefix.py`, `verify_uipc_wheel.py`, `uipc-sm86-build/` | the record of the Genesis base image's recipe (above) | the base image itself is pinned by digest in `source/images/robotangle/Dockerfile.mujoco` |

The release recipe and its inputs are under `source/` (`source/README.md`).

## The libuipc builds (part of the bound runtime; unused by 1.0.0's tasks)

The base holds two builds of one libuipc recipe (libuipc 292c98c3 with determinism patches, the pinned CUDA 12.8
builder image, pyuipc `0.0.25+topogym.det1.cuda128`): the audited wheel (`89-real;120-real;120-virtual`) in
site-packages with `/opt/uipc-build.json`, and an sm_86 build (`86-real;86-virtual`) in `/opt/uipc_sm86`. The tooling
chooses one by the GPU's compute capability (`rcb_topogym.py` `select_uipc_build`) and records it in the Session's
runtime. Both keep the pyuipc version and libuipc commit. The sm_86 recipe is `uipc-sm86-build/`: the release
Dockerfile's compile stages verbatim, the config `uipc-build-sm86.json` and the audit script with one change (it
requires the PTX of the config's `-virtual` architectures instead of a fixed compute_120). nvcc output is not
bit-reproducible, so a rebuilt wheel would need new bindings.

```bash
cd images/roboarc/uipc-sm86-build
docker build -f Dockerfile.uipc --build-arg UIPC_CONFIG=uipc-build-sm86.json --build-arg UIPC_BUILD_JOBS=16 \
  --target uipc-export -o type=local,dest=<out> .
```
