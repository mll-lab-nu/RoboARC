# 1.1.0: the image and the simulator without Genesis (2026-10-20)

RoboARC was developed on the Genesis simulator (TopoGym, then RoboTangle). Before 1.0.0 every task moved to MuJoCo
3.14.0 and every family to the Nyx renderer fed by `nyx_direct.py`, but the release image stayed a delta on the
RoboTangle 0.2.22 image, so 1.0.x still shipped Genesis 1.2.1, two libuipc builds and their dependencies, and the
runtime binding of every task named them. 1.1.0 removes them without changing what any task grades.

## What the graded runtime uses

Measured on `roboarc:1.0.0` before the change, with every graded process traced (Python's audit hook for imports,
opened files, `dlopen` and executed programs, `/proc/self/maps` snapshots, `strace` of `open` / `execve`): one base per
family (14), in both modes, with Nyx rendering on, as the tasks run them (the privileged verifier's two replays and its
video; the standard sim service with the task's oracle client, then its collect hook's replay), plus every documented
Session call of the agent's API on five families.

- **Genesis: no code.** `genesis` is never imported. Two kinds of its files were read: the ground plane's checker
  texture (`genesis/assets/textures/checker.png`, which `nyx_direct.py` found next to the genesis package) and, for the
  Panda + Robotiq robot, `xml/franka_emika_panda/` (which the renderer's robot builder resolved through the genesis
  package; the physics already used the vendored copy of the same bytes). 1.1.0 ships the texture
  (`source/assets/textures/genesis/`) and resolves the Panda to the vendored copy.
- **libuipc: no code.** Only its two build records (`/opt/uipc-build.json`, `/opt/uipc_sm86/uipc-build.json`) were read,
  by the runtime check.
- **torch is used**: `nyx_direct.py` computes the camera poses with torch's kernels (on the CPU, or on CUDA for the
  chain-cord families), so that the frames match the Genesis host's to the bit. It stays, with the CUDA wheels
  `import torch` loads.
- 47 Python distributions have code loaded; 206 Debian packages' files are mapped or opened (glibc, libstdc++, the GL /
  EGL / Vulkan loaders, X11 libraries, ffmpeg and its codecs: the verifier and the collect hook encode and check the
  replay videos with `ffmpeg` / `ffprobe`).

## What changed

- **Image** (`source/images/roboarc/Dockerfile`, replacing `source/images/robotangle/Dockerfile.mujoco`): built from
  the same Python base image (pinned by digest), with the same Debian package versions (from snapshot.debian.org of the
  day the 0.2.22 base installed them) and the same wheels by sha256 (`runtime.lock`: the 0.2.22 base's lock entries
  for what is used), without what is unused: Genesis, libuipc, quadrants, gs-madrona, gs-nyx-plugin, torchvision,
  triton, opencv, vtk, matplotlib, scikit-image, numba and the other packages only they needed (85 distributions), the
  build toolchain (build-essential), git, vulkan-tools, the 9.3 GB uv download cache the 0.2.22 image kept under
  `/root/.cache`, documentation pages and apt/dpkg logs. It keeps curl, ca-certificates and ripgrep for the agent
  harness.
- **Simulator source** (`SOURCE.json` 443 -> 426 files): the Genesis world, its scene builder, its physics backends
  (FEM, IPC, the Genesis grasp models, PBD table friction, release geometry, rigid contacts), its visuals and renderer
  and a Genesis-only blanket preview are removed; the environment refuses a non-MuJoCo engine; the runtime identity
  (`frozen_binding.installed_runtime`) names torch, numpy, scipy, mujoco, newton and warp-lang, no longer genesis-world,
  pyuipc or a libuipc build record. Modules the MuJoCo path also uses keep their Genesis-era functions (unreachable).
- **Tooling v20** (`rcb_topogym.py`): no libuipc build choice by GPU, no Genesis kernel replacements; `nyx_direct.py`
  finds the checker texture in the image. The runtime bindings of all 202 puzzle task dirs name the new simulator
  source, tooling and runtime.
- **Repository**: the parts that only served porting the Genesis release were removed: the Genesis-era task texts
  and tools (`tools/release/templates/`, `package_text/`, `package_text_tool.py`, `scripts/roboarc/port_package.py`),
  the 52 Genesis `-i00` references (`tools/references/oracles_genesis/`), the Genesis release generator
  (`records/genesis_release/`), the Genesis base image's recipe and libuipc build files (`images/roboarc/`), and
  `tools/tests/test_mujoco_kinematics.py`. `tools/release/release_rebuild.py` gives a new base its template's final
  packages. All of it is in RoboARC 1.0.0 (main 153eb8cf01); [docs/rcb/README.md](../rcb/README.md) maps the paths.

## Validation

- **Same files.** The image installs the same Debian package versions and the same wheels as 1.0.0. A manifest of
  both images' filesystems: of the 5,051 files the traced 1.0.0 runtime opened or mapped (outside the task's own files),
  4,945 are byte-identical in 1.1.0, 99 are gone (the Genesis-only simulator modules, the libuipc build records, the
  Genesis package's texture and Panda files, dist-info files of removed packages) and 7 differ: `SOURCE.json`, the five
  simulator modules 1.1.0 changes and the loader cache `/etc/ld.so.cache`. Every file of the 49 kept
  Python distributions is byte-identical, except the build record of the one sdist (antlr4: `RECORD`, `WHEEL`).
  `/opt/warp-cache` (the compiled Warp kernels) is byte-identical. Two builds of the recipe on different hosts give the
  same file contents.
- **Same graded results.** Strict fresh-process replays of every reference in 1.1.0, as the privileged verifier runs
  them (`tests/test.sh`'s `grade.py --repeat 2`, rendering off): all 99 references succeed, both replays of each end on
  the same state, every birth is the frozen instance's (`instance_exact`, `assets_exact`, the runtime binding checked),
  and 97 end on the hash the freeze manifest records (`records/freeze_manifests/freeze_manifest.v3.json`). The other two,
  knot-untying-06-i01 and -09-i01, end on `0d25b901f83db04e` and `3dd3136502cb9e13`: the hashes the released image of
  0.2.0 reached in its Harbor oracle runs, and 1.0.0 reaches them too (traced state by state every 250 rows, 1.0.0 and
  1.1.0 identical throughout); the freeze manifest carries an earlier dry run's values for these two. The two bases
  without a reference build their frozen birth and return to it on reset. Before the change the same replays in 1.0.0
  gave the same hashes for one base per family in both modes, with Nyx rendering on.
- **Rendering.** The Nyx frames, depth images and camera data of the agent API (every camera of five families) are
  byte-identical between 1.0.0 and 1.1.0 (for orbits and knot_untying two runs of 1.0.0 itself can differ in a few
  pixels; runs of both images made under the same conditions are identical); the standard service with Nyx rendering
  on reaches the recorded end states.
- **The Python package.** `tests/api`: the simulator tests (29 passed) and the reference bit-identity tests (17 passed,
  2 IKEA cases skipped) in 1.1.0.
- **The release rebuild.** `tools/release/release_rebuild.py --onto <1.1.0>` with 1.1.0's own instances and references
  builds the same image id and changes no file.
