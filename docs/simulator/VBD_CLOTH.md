# VBD cloth in the release image

The release image (`source/images/robotangle/Dockerfile.mujoco`) carries the cloth solver of the VBD cloth backend: Newton's
`SolverVBD` on NVIDIA Warp's CPU backend.
- **Packages:** `newton` 1.6.1 and `warp-lang` 1.18.0. `source/images/robotangle/vbd-requirements.txt` gives the exact versions and wheel sha256s;
  they are installed with `--no-deps --require-hashes` (newton needs only warp-lang, warp-lang only the image's numpy).
  The runtime binding records both versions (`installed_runtime`).
- **Licences:**
  - newton: Apache-2.0. Its bundled notices are in `newton/licenses/`: CC-BY-4.0, unittest-parallel, and viser with the
    Inter font.
  - warp-lang: Apache-2.0. The wheel also carries NVIDIA's redistributable parts under their own licences, used only on
    GPU: CUDA and NVRTC runtime pieces, and libmathdx. It also carries third-party notices (LLVM, Gaia, cubql, dlpack,
    NanoVDB, USD and others). All are in `warp_lang-1.18.0.dist-info/licenses/`.
- **Compile target:** Warp compiles CPU kernels with its embedded Clang, by default for the host (`-march=native`).
  `source/topogym/world/mujoco/warp_cpu.py` pins it to `-target-cpu x86-64-v3` (AVX2, FMA, BMI2), so every host compiles the
  same code.
  - The flag must be in Clang frontend syntax: Warp rejects `-march=x86-64-v3` and intercepts only `-march=native`.
  - A CPU without x86-64-v3 cannot load the kernels.
  - In a test kernel (dot, cross, length), Warp contracted nothing to FMA under any target, and its results were
    bit-identical under native, x86-64-v3, x86-64-v2 and generic. The pin removes host-dependent code generation, such as
    AVX-512 on Zen 4.
- **Kernel cache:** the build compiles the kernels once into `/opt/warp-cache` (`WARP_CACHE_PATH`), and it is read-only
  for everyone. A cold compile takes about 30 s per process.
  - `source/images/robotangle/warp_precompile.py` steps a blanket-sized cloth through every path the cloth uses. A new Warp
    module, such as a faster VBD step, must be stepped there too.
  - Code that uses the cloth calls `warp_cpu.init()` before `import newton`. When the image cache is not writable for
    the user, `init()` gives Warp a per-user cache in the temp dir whose module directories link to the image's, so a
    module the image lacks compiles there instead of failing.
- **GPU (blanket tasks on `blanket_vbd_gpu`):** the cloth runs on CUDA (`source/topogym/world/mujoco/vbd_gpu.py`); the
  container needs a GPU (`--gpus`, Apptainer `--nv`). Its CUDA kernels are not in the image cache: the first process
  compiles them (about 1-2 min) into the per-user cache, so mount one cache directory for all trials of a host
  (e.g. `-v <host dir>:/tmp/warp-cache-<uid>`). Results do not depend on the GPU class; a GPU shared with another
  process only slows it (measured 0.74 -> 1.07 s/control).
- **CPU resources (blanket tasks on `blanket_vbd`):** the cloth step (`source/topogym/world/mujoco/vbd_fast.py`) runs on 4
  threads that meet at a barrier per colour, so a control lasts as long as its slowest thread. A container
  for these tasks needs a quota of 4 CPUs (`docker run --cpus=4`, compose `cpus: 4`) with the profile's 4 threads, and
  must NOT be pinned to cores (`--cpuset-cpus`). The host must not be oversubscribed: start a trial only while the
  1-min load plus 4 is at most the number of logical CPUs.
  - The thread count and the quota change only speed. Results are bit-identical at 1 and 4 threads and under every
    limit below (the cross-CPU gate: workstation i9-14900HX, host A Ryzen 9 7950X, a cloud host EPYC 9354).
  - Measured on the 7950X (32 logical CPUs) at load 20-35, a 20-control fold step benchmark, 3 repeats:

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
