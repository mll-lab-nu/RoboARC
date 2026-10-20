# eai-robotangle

`ghcr.io/mll-lab-nu/eai-robotangle:<images/robotangle/VERSION>` is a public image: RoboTangle's simulator (the pruned
`topogym` package of the release commit), Genesis 1.2.1, libuipc (the audited determinism-patched wheel, plus an sm_86 build
of the same recipe), PyTorch, Nyx and every task's frozen instance. The tasks `FROM` it, and the standard `sim` service runs
it. Maintainers push it with `scripts/push_images.sh` after validating a build.

- **Nyx.** The image contains Genesis AI's Nyx renderer (the `gs-nyx` wheels on PyPI). The benchmark owner has permission to
  redistribute it in this image.
- **Build source.** The build context comes from the private Hugging Face dataset `eai-v2-internal/robotangle-assets`. It is
  internal and is never made public; only the built image is public. It holds, at the revision and sha256s pinned in
  `scripts/robotangle/assets.json`:
  - `image/release-<commit>.tar.gz`: the release tree (`scripts/robotangle/release.json` `release_commit`), the paths
    `images/topogym/Dockerfile` copies;
  - `image/uipc-prebuilt/`: the audited libuipc wheel and its `uipc-build.json` (sha256 `39781d19…`), which every task's
    runtime binding names. nvcc output is not bit-reproducible, so a wheel compiled elsewhere never matches the binding;
  - `image/uipc-sm86/`: the sm_86 build of the same recipe and its `uipc-build.json` (sha256 `712470d2…`; "The libuipc
    builds", below). The image installs it in `/opt/uipc_sm86` (`install_uipc_prefix.py`);
  - `instances/`: every task's frozen instance. `stage_instance.py` copies a task's instance next to its
    `runtime_binding.json` and checks it (privileged `environment/` and `tests/` images; the standard `sim` entrypoint).

  Access: a member of `eai-v2-internal`, logged in with `hf auth login`, or `HF_TOKEN` in the environment (never printed).

## Host

- Linux x86_64. Not WSL: Nyx needs the NVIDIA Vulkan driver inside the container.
- An NVIDIA GPU with driver >= 580, and Docker with the NVIDIA Container Toolkit. Docker's default runtime stays `runc`.
- Exactly one GPU per container: Nyx crashes when two GPUs are visible. The task compose files reserve one GPU with a
  device reservation (`count: 1`, `capabilities: [gpu]`), without `runtime: nvidia`. NVIDIA_DRIVER_CAPABILITIES=all makes
  the toolkit mount the driver's graphics libraries, and the image's Vulkan ICD and EGL vendor files point the loaders at
  them, as in `images/robopaint/`. `VK_ICD_FILENAMES` names the image's ICD file: a toolkit in CDI mode also mounts the
  host's `/etc/vulkan/icd.d/nvidia_icd.json`, which would list the GPU twice.
- `count: 1` is host GPU 0 unless the runner pins GPUs; IPC sims (knot_untying, rope_threading) run alone on a 24 GB GPU
  (`docs/benchmarks/robotangle.md`, Image).
- GPU architecture (`docs/benchmarks/robotangle.md`, Image, has the table):
  - knot_untying and rope_threading run libuipc: the audited wheel on compute capability 8.9 and 12.x, the sm_86 build on
    any other GPU (native on 8.6, e.g. RTX A6000; untested elsewhere). 7.5 (T4, RTX 20xx) can run neither;
  - GPUs without RT cores (e.g. A100, H100) cannot run Nyx, so they run no RoboTangle task;
  - the driver must include its graphics (GL/Vulkan) user-space libraries.
- blanket_flatten: on a GPU other than the one its frozen PBD states were prepared on (an RTX PRO 6000 Blackwell), the task
  generates the birth state from the recipe frozen in the instance (`docs/benchmarks/robotangle.md`, GPU portability).
- Build:
  - about 25 GB of free disk: the 11.3 GB image plus BuildKit cache;
  - BuildKit with the default `docker` driver;
  - network access to docker.io, pypi.org, download.pytorch.org and dl.polyhaven.org (the CC0 studio HDRI).
- Running needs no network.

## Build (from the repository root)

0.2.22 was published as a delta of the published 0.2.21: `FROM ghcr.io/mll-lab-nu/eai-robotangle:0.2.21@sha256:4fd681598e07a7a29201b8f30ccbee9944baaa856a5b05b045e48522c138f7ab` plus one `COPY` layer with the eight IPC instances'
`instance_assets/frozen_assets/rope_volume.npz` (mode 0644). Its files are those of a full build from the pinned dataset
revision, and a host that has 0.2.21 pulls only that layer (13 MB). A full build with the commands below gives the same
files in other layers.

```bash
scripts/robotangle/fetch_image_context.sh [<download dir>]     # images/robotangle/context/ from the pinned dataset, checked
docker build -f images/robotangle/Dockerfile -t ghcr.io/mll-lab-nu/eai-robotangle:$(cat images/robotangle/VERSION) images/robotangle/context
```

`scripts/build_images.sh` builds it from `images/robotangle/BUILD_CONTEXT` once the context exists, and skips it otherwise.
The image is not cache-warmed: every container compiles its kernels on first use, and scene ready took 67-296 s for the
rigid families on a shared 4090, against the 1800 s startup watchdog. The release tree's `images/topogym/warm_cache.sh`
does not apply here. It expects `topogym-*` task directories with their own `instance.json`, and RoboTangle's tasks stage
their instances from the image.

## Check a build

The image must install exactly the runtime every task is bound to:

```bash
docker run --rm ghcr.io/mll-lab-nu/eai-robotangle:$(cat images/robotangle/VERSION) python -c 'import json; from topogym.simulator.frozen_binding import installed_runtime; print(json.dumps(installed_runtime(), sort_keys=True))' > /tmp/robotangle_runtime.json
python3 -c 'import json, sys; sys.exit(0 if json.load(open("/tmp/robotangle_runtime.json")) == json.load(open("scripts/robotangle/release.json"))["runtime"] else "image differs from the tasks runtime binding: do not run")'
```

Every session checks it again at start (`runtime_binding_ok` in reward.json) and refuses a different runtime. The two
libuipc builds must be the ones `scripts/robotangle/release.json` `uipc_builds` names, and the image build already checked
both wheels against their build records:

```bash
docker run --rm ghcr.io/mll-lab-nu/eai-robotangle:$(cat images/robotangle/VERSION) sh -c 'sha256sum /opt/uipc-build.json /opt/uipc_sm86/uipc-build.json; cat /opt/uipc-check.json'
```

## The libuipc builds

The image holds two builds of one libuipc recipe (libuipc 292c98c3 with the determinism patches of the release's
`images/topogym/uipc-patches`, the pinned CUDA 12.8 builder image, pyuipc `0.0.25+topogym.det1.cuda128`):
- the audited wheel (`89-real;120-real;120-virtual`) in site-packages, `/opt/uipc-build.json`: every evidence run used it,
  and GPUs of compute capability 8.9 and 12.x always load it;
- the sm_86 build (`86-real;86-virtual`) in `/opt/uipc_sm86`: any other GPU loads it. The task tooling chooses at Session
  start, before uipc is imported (`rcb_topogym.py` `select_uipc_build`).

Both keep the pyuipc version and libuipc commit, which the frozen IPC settled states record; that is why the sm_86 build
restores them. Do not give a rebuild another local version label.

The sm_86 recipe is `images/robotangle/uipc-sm86-build/`: the release Dockerfile's compile stages verbatim, its config
`uipc-build-sm86.json` (the release config with the architecture list changed, plus a `recipe` record), and the release's
`verify_uipc_wheel.py` with one change: the audit requires the PTX of the config's `-virtual` architectures instead of a
fixed compute_120. A rebuild follows the same recipe but need not give the same bytes or even the same machine code (two
sm_86 builds of 2026-10-08 differ in 10 of 1230 functions), so a rebuilt wheel needs new bindings. A single wheel for
sm_86, sm_89 and sm_120 was tried and not used: its sm_89 code differed from the audited wheel's in 8 of 1230 functions
(`docs/benchmarks/robotangle.md`, Image).

```bash
cd images/robotangle/uipc-sm86-build
docker build -f Dockerfile.uipc --build-arg UIPC_CONFIG=uipc-build-sm86.json --build-arg UIPC_BUILD_JOBS=16 \
  --target uipc-export -o type=local,dest=<out> .
```

## VBD cloth (Dockerfile.mujoco)

The MuJoCo images (`images/robotangle/Dockerfile.mujoco`) carry the cloth solver of the VBD cloth backend: Newton's
`SolverVBD` on NVIDIA Warp's CPU backend.
- **Packages:** `newton` 1.6.1 and `warp-lang` 1.18.0. `vbd-requirements.txt` gives the exact versions and wheel sha256s;
  they are installed with `--no-deps --require-hashes` (newton needs only warp-lang, warp-lang only the image's numpy).
  The runtime binding records both versions (`installed_runtime`).
- **Licences:**
  - newton: Apache-2.0. Its bundled notices are in `newton/licenses/`: CC-BY-4.0, unittest-parallel, and viser with the
    Inter font.
  - warp-lang: Apache-2.0. The wheel also carries NVIDIA's redistributable parts under their own licences, used only on
    GPU: CUDA and NVRTC runtime pieces, and libmathdx. It also carries third-party notices (LLVM, Gaia, cubql, dlpack,
    NanoVDB, USD and others). All are in `warp_lang-1.18.0.dist-info/licenses/`.
- **Compile target:** Warp compiles CPU kernels with its embedded Clang, by default for the host (`-march=native`).
  `topogym/world/mujoco/warp_cpu.py` pins it to `-target-cpu x86-64-v3` (AVX2, FMA, BMI2), so every host compiles the
  same code.
  - The flag must be in Clang frontend syntax: Warp rejects `-march=x86-64-v3` and intercepts only `-march=native`.
  - A CPU without x86-64-v3 cannot load the kernels.
  - In a test kernel (dot, cross, length), Warp contracted nothing to FMA under any target, and its results were
    bit-identical under native, x86-64-v3, x86-64-v2 and generic. The pin removes host-dependent code generation, such as
    AVX-512 on Zen 4.
- **Kernel cache:** the build compiles the kernels once into `/opt/warp-cache` (`WARP_CACHE_PATH`), and it is read-only
  for everyone. A cold compile takes about 30 s per process.
  - `images/robotangle/warp_precompile.py` steps a blanket-sized cloth through every path the cloth uses. A new Warp
    module, such as a faster VBD step, must be stepped there too.
  - Code that uses the cloth calls `warp_cpu.init()` before `import newton`. When the image cache is not writable for
    the user, `init()` gives Warp a per-user cache in the temp dir whose module directories link to the image's, so a
    module the image lacks compiles there instead of failing.
- **GPU (blanket tasks on `blanket_vbd_gpu`):** the cloth runs on CUDA (`topogym/world/mujoco/vbd_gpu.py`); the
  container needs a GPU (`--gpus`, Apptainer `--nv`). Its CUDA kernels are not in the image cache: the first process
  compiles them (about 1-2 min) into the per-user cache, so mount one cache directory for all trials of a host
  (e.g. `-v <host dir>:/tmp/warp-cache-<uid>`). Results do not depend on the GPU class; a GPU shared with another
  process only slows it (measured 0.74 -> 1.07 s/control).
- **CPU resources (blanket tasks on `blanket_vbd`):** the cloth step (`topogym/world/mujoco/vbd_fast.py`) runs on 4
  threads that meet at a barrier per colour, so a control lasts as long as its slowest thread. A container
  for these tasks needs a quota of 4 CPUs (`docker run --cpus=4`, compose `cpus: 4`) with the profile's 4 threads, and
  must NOT be pinned to cores (`--cpuset-cpus`). The host must not be oversubscribed: start a trial only while the
  1-min load plus 4 is at most the number of logical CPUs.
  - The thread count and the quota change only speed. Results are bit-identical at 1 and 4 threads and under every
    limit below (the cross-CPU gate: workstation i9-14900HX, host A Ryzen 9 7950X, a cloud host EPYC 9354).
  - Measured on the 7950X (32 logical CPUs) at load 20-35, `fast/test_fast.py`, 20 controls, 3 repeats:

    | Condition | Seconds per control |
    |---|---|
    | `--cpus=4` | 0.61-0.77 (no CFS throttling) |
    | `--cpus=4`, two trials side by side | 0.70-0.74 each |
    | `--cpus=4`, 8 extra busy threads on the host | 0.94-1.22 |
    | `--cpuset-cpus` on 4 cores of one CCD | 0.97-1.15 |
    | `--cpuset-cpus` on 4 cores across both CCDs | 1.18-1.24 |
    | `--cpuset-cpus`, with a busy neighbour pinned to the same cores | 2.3-2.4 |
    | 1 thread | 1.5-1.6 |
  - Why a fixed cpuset is slower: the scheduler can no longer move a barrier thread off a core that another process,
    or a busy SMT sibling, is using. When the pinned cores are shared, the whole team waits on the straggler every
    phase.
  - On a saturated host (a cloud host at load ~125 of 128), the 4-thread fold ran at 2.45 s/control, close to its 1-thread
    2.8 s/control.
  - Pin cores only on a host partitioned so that the trial owns those cores exclusively, SMT siblings included.
