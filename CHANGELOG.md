# Changelog

RoboARC's releases. Versions before 1.0.0 were released inside the shared benchmark repository
`mll-lab-nu/embodied-first-exam` (formerly robot_coding_bench), under the image name `ghcr.io/mll-lab-nu/eai-roboarc`.
1.0.0 is the first release from this repository. Its content is the "V3" wave of that repository.

## Unreleased

- The usage gateway ships with the repository: `scripts/usage_gateway/` (from the shared benchmark repository,
  unchanged; PROVENANCE.md) and `.env.example`. Usage recording stays optional and off by default; turn it on with
  `USAGE_GATEWAY=scripts/usage_gateway` (docs/RUNNING.md "Usage recording"). Its tests and a check that it takes
  every option the runner passes run in `ci_checks/check-unit-tests.sh`.
- `scripts/set_agent_egress.py` runs its command line only as a script, so importing it (to read `HOSTS`) runs
  nothing; as a script it behaves as before.
- `tools/stress/corpus/SHA256SUMS`: the O2 reproducer's line matches its file again.
- Trials and seeds (owner decisions for 1.2.0; PROTOCOL.md "Changes from the shared protocol", RUNNING.md "Seeds",
  "Results"): the batch runner gives the K trials of a task distinct seeds without replacement, in an order fixed by the
  batch name and the task's base name (`scripts/task_seed.py assign`; the same seeds in both modes), runs each trial as a
  Harbor job of its own from a copy made for its seed (`scripts/task_seed.py materialize`, back from 1.0.0 with
  `assign`, the verifier's `tests/` in standard mode and per-seed goal images), and records the trial and its seeds in
  `<job>/seed.json`. Results are one folder per trial, `<task>/<run id>/t<k>/` (`config.json` gains `trial`, `trials`,
  `trial_dir`, `copy_checksum`; `seed_source` is `runner`), with an index (`trials.json`, `README.md`) per task and run
  id. A single-seed task run with K > 1 is no longer over the retry cap: the retry rule (H.5) counts attempts per trial.
  The seed of a trial was recorded as the default before (the runner's task copy was gone when results were written);
  a trial's task version is now the committed DIGEST the runner copied (Harbor's checksum of the copy, which differs
  from it whenever the copy does, is kept as `copy_checksum`). `scripts/roboarc/regrade.py --replay` grades a trial with its seed's
  verifier files. New check `ci_checks/check-seed-files.py`. Every task still allows one seed: no task changes.

No task directory, pinned tooling, hashed simulator file or image input changes.

## 1.1.0 (2026-10-20)

The puzzle image without Genesis: `ghcr.io/embodied-first-exam/roboarc:1.1.0` (8.7 GB; 1.0.0 was 12.2 GB), and the IKEA
images rebuilt from fully pinned inputs with only what the tasks run: `roboarc-ikea:1.1.0` (1.05 GB; 1.31 GB) and
`roboarc-ikea-sim:1.1.0` (7.67 GB; 9.08 GB). The task set's version is 1.1.0. Every task binds its new image; the
instructions, the graders' logic, the frozen instances and scenes and the references are unchanged. Results graded
under 1.0.x stand: every reference replays in the new images to the end state it reaches in 1.0.0.
Details: [docs/history/GENESIS_REMOVAL.md](docs/history/GENESIS_REMOVAL.md), `ikea/images/ikea-mujoco/README.md`.

### The image
- Built from the Python base image (`source/images/roboarc/Dockerfile`) instead of the RoboTangle 0.2.22 (Genesis)
  image. Everything is pinned: the base image by digest, the Debian packages by version from snapshot.debian.org, the
  wheels by sha256 (`source/images/roboarc/runtime.lock`, `vbd-requirements.txt`).
- The same files as 1.0.0 for everything it keeps: every library the graded runtime loads (the Python interpreter,
  glibc / libm, libstdc++, numpy's and scipy's OpenBLAS, MuJoCo 3.14.0 with the EPA patch, Warp, Newton, torch and its
  CUDA libraries, gs-nyx, the graphics loaders, ffmpeg) is byte-identical, and so is every file of the 49 Python
  packages it keeps.
- Removed as unused (traced on every family, both modes, rendering on, and on the agent's API): genesis-world, libuipc
  (pyuipc, both builds), quadrants, gs-madrona, gs-nyx-plugin, torchvision, triton, opencv, vtk, matplotlib,
  scikit-image, numba and the other packages only these needed; build-essential, git, vulkan-tools; the uv download
  cache the 0.2.22 image kept (9.3 GB); documentation pages and apt/dpkg logs. curl, ca-certificates and ripgrep stay
  for the agent harness.

### The simulator, the tooling and the bindings
- `source/topogym/`: the Genesis world, scene builder, physics backends (FEM, IPC, the Genesis grasp models), visuals
  and renderer are removed (SOURCE.json: 443 -> 426 files); the runtime identity no longer names genesis-world, pyuipc
  or a libuipc build. The ground plane's checker texture, which the renderer read from the genesis package, ships with
  the image; a Panda + Robotiq robot given in genesis-world's layout resolves to the vendored Panda (the same bytes).
- Tooling v20 (`rcb_topogym.py`): no libuipc build choice by GPU, no Genesis kernel replacements; `nyx_direct.py` finds
  the checker texture in the image.
- All 202 puzzle task dirs: new `runtime_binding.json`, tooling copies, image reference, a 1.1.0 Change Log entry and
  DIGEST; the compose files cite the new recipe.

### Records
- `records/freeze_manifests/freeze_manifest.v3.json`: knot-untying-06-i01 and knot-untying-09-i01 expect the end hashes
  `0d25b901f83db04e` and `3dd3136502cb9e13`. The old values came from a pre-release dry run; the released references end on
  the new ones in images 0.2.0 (Harbor oracle runs), 1.0.0 and 1.1.0, with equal state hashes at every 250-row checkpoint.
  The references themselves are unchanged.

### Repository
- Removed (they only served porting from Genesis; RoboARC 1.0.0 at main 153eb8cf01 keeps them): the Genesis-era task
  texts and tools (`tools/release/templates/`, `package_text/`, `package_text_tool.py`, `scripts/roboarc/port_package.py`),
  the Genesis `-i00` references (`tools/references/oracles_genesis/`), the Genesis release generator
  (`records/genesis_release/`), the Genesis base image's recipe and libuipc build files, and the Genesis kinematics
  comparison test. `release_rebuild.py` gives a new base its template's final packages.

### The IKEA images
- `roboarc-ikea:1.1.0` and `roboarc-ikea-sim:1.1.0` (`ikea/images/ikea-mujoco/`): every input is pinned, the Python base
  image by digest (as before), the Debian packages at the 1.0.0 images' versions from the Debian snapshot archive of
  their build day, the wheels by sha256 (`requirements.lock`, `sim/nyx-requirements.txt`; until 1.0.0 the base resolved
  mujoco's dependencies at build time and the sidecar installed its Debian and pip packages unpinned). The sidecar is
  `FROM` the base by digest. `build.sh` builds both reproducibly (pinned BuildKit, SOURCE_DATE_EPOCH, rewritten
  timestamps) as OCI archives with fixed manifest digests: base `sha256:2c56d9226aaf…` (id `9f9cbab8aa74`), sidecar
  `sha256:f22a1e9042c7…` (id `cbd929360257`).
- The same files as 1.0.0 for everything the graded runtime loads (traced: privileged grading with its replay video,
  the standard sim service with Nyx, the agent's tooling): MuJoCo with the EPA patch, numpy, the scenes, OSMesa and
  ffmpeg, Nyx, torch and its CUDA libraries, the graphics loaders (the dynamic linker's cache lists the added and
  removed libraries; every library resolves to the same file); every Debian package keeps its version. The verifier's
  replay videos are byte-identical to 1.0.0's.
- Removed as unused: mujoco's absl-py, etils, fsspec, websockets and zipp (its viewer, studio and test code); in the
  Nyx venv triton, sympy, mpmath, jinja2, markupsafe, filelock, fsspec, cuda-toolkit, cuda-pathfinder, absl-py, etils,
  glfw and pip; curl in the sidecar; bytecode caches; the second copy of the scenes the 1.0.0 base carried (a `chmod`
  layer).
- The verifier's replay video libraries (OSMesa, imageio with ffmpeg, Pillow) are in the base image: the privileged
  tasks' verifier image installs nothing when it is built (until 1.0.x it installed them from the live Debian archive
  and PyPI at that moment). The privileged agent's container has them too. Each IKEA task's runtime binding names the
  new base image id, and its README a 1.1.0 Change Log entry.
- Push the two images from their OCI archives byte for byte (`images/monitor/push.sh`), so the registry's digests are
  the ones above; a `docker push` re-compresses the layers.

### Validation of 1.1.0
- The 99 references, replayed strictly in fresh processes as the privileged verifier does (two replays, rendering
  off): all succeed, both replays agree, every birth is the frozen instance's; 97 end on the freeze manifest's hash, and
  knot-untying-06-i01 and -09-i01 on the hashes 1.0.0 and the released 0.2.0 image reach (the manifest holds an earlier
  dry run's values for these two). The two bases without a reference build their frozen birth.
- Every library the traced runtime loads and every file of the kept Python packages is byte-identical to 1.0.0's; two
  builds of the recipe on different hosts give the same file contents.
- The agent API's Nyx frames, depth and camera data equal 1.0.0's (five families); the standard service with Nyx
  rendering reaches the recorded end states; `tests/api`: 29 simulator tests and 17 bit-identity tests pass in 1.1.0.
- `release_rebuild.py --onto 1.1.0` with its own instances builds the same image id and changes no file; the
  generators' `--check`, the invariants, unit tests and advisories pass.
- IKEA: the 12 references replayed in fresh processes in the 1.1.0 verifier image exactly as the privileged verifier
  does (two replays and the replay video): all succeed, both replays agree, and every end and settled state equals
  1.0.0's (and the recorded hashes). Harbor oracle 1 / nop 0 on all 12 scenes in both modes (48 trials) with the 1.1.0
  images; every oracle ends on its recorded state. Every file the traced runtime loads is byte-identical to 1.0.0's;
  three builds (two from fresh clones) give byte-identical archives. Nyx's RGB frames of a moving scene differ from
  run to run on one host in 1.0.0 as in 1.1.0 (hundreds to tens of thousands of pixels, by up to 25 levels, between two
  runs of the same image; the frames at reset are identical), so they are compared by that spread; depth,
  proprioception and every physics state are identical. `tests/api` in `roboarc-ikea-sim:1.1.0` with a GPU: 29 passed
  (the IKEA reference bit-identity tests for lack and svallet among them), 19 skipped (puzzles).

## 1.0.1 (2026-10-19)

The task set's version 1.0.1 (`scripts/roboarc/release.json` `version`; the README Release line of every puzzle task).
The images stay `ghcr.io/embodied-first-exam/roboarc:1.0.0`, `roboarc-ikea:1.0.0` and `roboarc-ikea-sim:1.0.0` (the same
ids); the resource monitor's image is now RoboARC's own build of the same files (below). The graders' and the sim
services' logic, the pinned tooling (`rcb_topogym.py`, `render_bridge.py`,
`nyx_direct.py`, `rcb_ikea.py`), the runtime bindings, the frozen instances, the IKEA scenes and the references are
unchanged. Every task has a new DIGEST and a Change Log entry in its README.

### The task texts (all 226 tasks)
A re-render from the cleaned generator inputs; text only for the puzzles:
- **Citations of this repository's paths**: `docs/PROTOCOL.md`, `docs/STANDARD_MODE.md`, `docs/TAXONOMY.md`,
  `docs/simulator/TOPOGYM_PROTOCOL.md`, `tools/release/release_rebuild.py`, `images/roboarc/` and the base image's
  recipe, instead of the development tree's `docs/rcb/...`, `images/topogym/`, `scripts/robotangle/...`,
  `scripts/task_dynamics.toml`. No task text cites a missing path any more except inside files whose bytes stay as
  released (the pinned tooling, the shared eai-standard and monitor layers, IKEA's `rcb_ikea.py` / `ikea_nyx.py`);
  [docs/rcb/README.md](docs/rcb/README.md) maps those.
- **No internal notes**: the resources' `measured` strings, the no-oracle bases' unfilled "PR #N", the Change Log
  entries citing internal documents, the graders', sim service's and oracle player's docstrings naming the earlier
  repository's RoboWits files, issue numbers and GPU-physics claims, IKEA's owner / board / pull-request notes, an
  internal runner script and a branch name.
- **`[task].name` = `roboarc/<task dir>`** (Harbor's `org/name`; it was `robot-coding-bench/<task dir>`); the Harbor
  dataset name is `roboarc/roboarc` (`scripts/harbor/dataset_manifest.py`). `harbor run -p tasks -i <dir>` is
  unchanged (it matches the directory name). The Python API accepts both forms.
- For the puzzles, `instruction.md` is unchanged and every changed `.py` file has the same docstring-free AST as in
  1.0.0 (the audit below); puzzle results graded under 1.0.0 stand.
- **IKEA's verification prose** names machine classes ("a Ryzen 9 7950X workstation", "a dedicated Xeon Gold 6230R
  core", "one process at a time") where the clean-up had left host labels and broken phrases ("host B", "a dedicated
  the cluster Xeon ..."): the rendered prose fields of `ikea/scripts/ikea_mujoco/evidence.json` and
  `standard_evidence.json`; every number, hash, image id, date and run name is unchanged (21 IKEA READMEs).

### IKEA MuJoCo on the repository's task layer, under `tasks/`
The 24 IKEA task dirs moved from `ikea/tasks/` to `tasks/` (same directory names) and are rendered with the
repository's task layer (`scripts/task_conventions.py`), like the puzzles: the instruction's shared closing sections
(`## External material`, `## Resource limits`, `## Time limit`), the resource-monitor sidecar, declared resources,
limits, caps (`scripts/task_resources.toml`, rules `^roboarc-ikea-`: privileged CPU only, the standard sim one GPU; the
agent's container 4 CPUs / 10 GB, the standard sim 4 CPUs / 8 GB; the privileged verifier keeps 4 CPUs / 8 GB for its
two replays and the video replay) and seeds, a `DIGEST`, and the one egress allowlist (`scripts/set_agent_egress.py`:
the model APIs and the package registries, which the instruction already named; IKEA's 1.0.0 allowlist had the model
APIs only). `ikea/scripts/task_conventions.py` and `ikea/scripts/set_agent_egress.py` are gone; IKEA's live-grading
2.4 pair is `scripts/eai_standard/v2_4_live/` (bytes unchanged), its own variant in `check-tooling-identity.sh`. Every
`ci_checks/` check covers the IKEA tasks. **IKEA results before and after 1.0.1 are not comparable**: the instruction
and the egress changed (the graders, scenes, references and images did not).

### The resource monitor's image: RoboARC's own
Every task's `rcb_monitor` service pinned `ghcr.io/mll-lab-nu/efe:monitor-0.1.0@sha256:f2c910f5…`, a package of
another group that RoboARC does not control. It now pins
`ghcr.io/embodied-first-exam/roboarc-monitor:0.1.0@sha256:ea1b0d4b859998d6a47e2cb7a72c3a9ae85870b32c75de1bc1448e387da07093`
(`scripts/rcb_monitor/inject.py` `IMAGE`), built from `images/monitor/`:
- **The same image contents**: the base is pinned by digest (`python:3.12-slim-bookworm`, Python 3.12.15), and the
  five layers have the earlier image's diff ids (the filesystems are equal file for file); the environment, working
  directory and command are the same; the config adds three OCI labels (title, version, source).
- **Reproducible**: `images/monitor/build.sh` builds an OCI archive with a pinned BuildKit, no attestations and a fixed
  `SOURCE_DATE_EPOCH`; two builds are byte-identical, so the digest the tasks pin is known before the push.
  `images/monitor/push.sh` copies that archive unchanged (skopeo `--preserve-digests`) and checks the registry's
  digest. [images/monitor/README.md](images/monitor/README.md).
- The tasks change only in that `image:` line, their 1.0.1 Change Log entry and DIGEST. `check-protocol.py` accepts
  base images from `ghcr.io/embodied-first-exam/` only. The Genesis base image (a build-time input of
  `roboarc:1.0.0`, `source/images/robotangle/Dockerfile.mujoco`) stays in the `efe` package: no task runs it.

### Usage recording is optional; licence
Usage recording is optional: an official run no longer needs the usage gateway (PROTOCOL.md 'Changes from the shared
protocol', H.6). The runner's `GATEWAY=1` is replaced by `USAGE_GATEWAY=<dir>`; results record `usage_gateway`,
`usage_source`, `usage_recorded`, `token_budget`. Licence: MIT (LICENSE, LICENSES/THIRD_PARTY.md).

### The raw simulator interface: the Python package `roboarc`
`roboarc.make(task_id)` returns a Gymnasium-compatible environment for every task (the 101 puzzle bases and the 12 IKEA
tasks) over the graded tooling itself (`rcb_topogym.Session`, `rcb_ikea.Session`): the frozen instance at `reset()`,
one control step of the tooling per `step()`, the tasks' own limits (clamp, never fail) and success predicates, Nyx or
MuJoCo-raster camera images, proprioception in the standard mode's robot frame, state snapshots, and the privileged
verifier's replay. Design, conventions and test results: [docs/API.md](docs/API.md); examples in `examples/`, tests in
`tests/api/` (the cheap ones run in CI, job `api-tests`). Each tested reference stepped through it ends on its recorded state
hash (one base per puzzle family and two IKEA scenes; with Nyx rendering on as well). The package changes no task,
image or graded file.

### Repository clean-up
The repository now holds RoboARC only and is laid out for its two audiences: `source/` is the image build context,
development tools are in `tools/`, records in `records/`, the simulator's internals in `docs/simulator/`, the Harbor
runner in `scripts/harbor/`; new `docs/RUNNING.md` (a Harbor guide), `docs/DEVELOPING.md`, one public page per family
in `docs/families/`, and a rewritten README. The generator's packages are the shipped texts (`mk_tasks.py` halved); the
asset tools check every instance source against its pin; CI runs on `dev/**` too, checks the docs' paths and links, and
runs the simulator source hash. Leftovers of the shared repository (other benchmarks, the eai-standard 2.2 layer,
lab-side runner defaults, internal notes) and duplicate copies (references, tooling, image recipes) are gone; the
session oracles of the orbits and slotted-post-ring references are restored. The clean-up itself changed no task byte
(the re-render above did); the images rebuilt from the cleaned tree have the 1.0.0 ids. Path moves: [PROVENANCE.md](PROVENANCE.md#after-100-the-clean-up).

### Validation of 1.0.1
On a fresh clone (2026-10-20, a shared Ryzen 9 7950X / RTX 4090 host, load 25-60):
- Both generators' `--check` (0 differences), `run_invariants.sh` (226 tasks), `check-unit-tests.sh --all`,
  `run_all.sh` ("all checks passed"), `check-tooling-identity.sh` (226 task dirs, 26 shared files, the 2.4 layer in
  113 standard dirs), `check-doc-paths.py`, `tools/release/source_hash.py --check` (SOURCE.json `c9171f79…`), and every
  job of `.github/workflows/checks.yml` run literally in its own `python:3.12` container, `api-tests` included.
- A diff audit of every changed task file against 1.0.0: every changed `.py` file of the 202 puzzle and 24 IKEA task
  dirs (474 files) has the docstring-free AST it had in 1.0.0; the puzzle `task.toml`s differ only in `measured`,
  `oracle_evidence` and `[task].name`, their shell scripts, Dockerfiles and compose files only in comments; no puzzle
  `instruction.md` changed; the IKEA runtime bindings and other JSON data are unchanged apart from text fields.
- Harbor oracle and nop, both modes: one base per puzzle family (14 bases, 56 trials, rendering off; the blanket bases
  with their CUDA cloth on the GPU) and all 12 IKEA scenes (48 trials, rendering on). All 104 scored as expected
  (oracle 1, nop 0, no infrastructure error), and every oracle ended on its recorded state hash (the puzzles' freeze
  manifest; IKEA's privileged end and settled hashes and standard live hashes). The IKEA privileged verifiers used at
  most 26 % of their timeouts under the new caps (stenerik 5036 s of 19440 s, kyrre 4365 s of 25440 s).
- The `roboarc` package's simulator tests against the re-rendered tasks: 29 passed in `roboarc:1.0.0`, 27 passed and 2
  skipped (puzzle-only) in `roboarc-ikea-sim:1.0.0`.
- The monitor image and the prose fix (2026-10-20, the same host): `images/monitor/build.sh` run four times (three times
  on one checkout, once on a fresh clone) gave byte-identical archives, manifest `sha256:ea1b0d4b…`; its filesystem equals
  the earlier image's file for file (paths, modes, owners, sizes, contents, link targets); `push.sh` into a local test
  registry stored exactly that digest. Every CI job in its own `python:3.12` container on a fresh clone passes,
  as do the gates listed above. Harbor oracle and nop, both modes, on interlocked-claws-03, blanket-fold-01 and IKEA
  lack with that image (12 trials): all scored as expected, every oracle ended on the same state hashes as above, and
  the monitor wrote the same record files and summary fields as before, no violation.

## 1.0.0 (2026-10-19)

The topological puzzles: 101 bases (202 task directories) in 14 families, image
`ghcr.io/embodied-first-exam/roboarc:1.0.0`. IKEA MuJoCo: 12 tasks (24 task directories), images
`ghcr.io/embodied-first-exam/roboarc-ikea:1.0.0` and `roboarc-ikea-sim:1.0.0`.

The tasks' graded content is byte-identical to the V3 release tree (dev/robotangle acfcec1089, image
`eai-roboarc:0.2.1`). Only the image references and the version labels changed (see "Repository" below). The image's
filesystem is identical to 0.2.1's (image id 41a17481d67a): the same layer digests, and new labels only.

### IKEA MuJoCo joins RoboARC
The IKEA MuJoCo furniture-assembly benchmark is part of RoboARC from 1.0.0: 12 tasks, 24 task directories under
`ikea/tasks/`, renamed into the RoboARC namespace: `ikea-<scene>-i00-<mode>` -> `roboarc-ikea-<scene>-i00-<mode>` (the map
is below). Scenes by Xiaoyu's team (Xiaoyu, Zhen Wu). Images
`ghcr.io/embodied-first-exam/roboarc-ikea:1.0.0` and `roboarc-ikea-sim:1.0.0`, which are the V3 images
`eai-ikea-mujoco:0.3.2` (5d5090ea1910) and `eai-ikea-mujoco-sim:0.4.2` (050cd45d70f6) under new names, with the same
ids. Its V3 content (development branch dev/robotangle_mujoco 5cb4114f10):
- **New tasks:** stenerik, burvik, uppsta and kyrre (the review was approved; kyrre ships with the native-cylinder
  exception below).
- **Franka Hand slew:** `GRIPPER_WIDTH_SPEED_M_S` in every copy of `rcb_ikea.py` (one sha256, 3a63a33b6803), so the hand's
  width command moves at its real speed. New references for the eight earlier tasks were built within the slew (lack,
  eket, ikea365, vesken, gladom, variera and stegesund; svallet's ViperX reference is unchanged).
- **Welded-part contact switch-off** in the scene runtime (`ikea_runtime.py`; kyrre's and uppsta's seated parts).
- **kyrre: MuJoCo's native cylinders**, the one exception to the 32-facet prism rule (owner, 2026-10-19). The stress
  A/B showed 1.15 % non-jerk breakdowns with native cylinders against 2.3 % with the prisms, and no breakdown involved
  a screw. The verifier limit is set by the slow-class replay (a Xeon Gold 6230R core, 8283 s): 25,440 s.
- **Nyx start-up retry** in the standard sidecar's render process (`ikea_nyx.py`, the same file in every standard task;
  sha256 036be49c38e6 in V3; 37a773230f41 in 1.0.0, after the rename changed a path in its docstring and
  a host name left a comment).
- **EPA-patched libmujoco.so**, the same 14-byte patch as the puzzles' image.
- Timing is gated on the i9 / Zen 4 host class. Slow-Xeon times are reported only: gladom, variera, stenerik, uppsta
  and kyrre are over the 40 % bar only there.

Task names (the generators' `name()`; physics, references, hashes and grading unchanged):

| until 1.0.0 | 1.0.0 |
|---|---|
| `ikea-burvik-i00-privileged` | `roboarc-ikea-burvik-i00-privileged` |
| `ikea-burvik-i00-standard` | `roboarc-ikea-burvik-i00-standard` |
| `ikea-eket-i00-privileged` | `roboarc-ikea-eket-i00-privileged` |
| `ikea-eket-i00-standard` | `roboarc-ikea-eket-i00-standard` |
| `ikea-gladom-i00-privileged` | `roboarc-ikea-gladom-i00-privileged` |
| `ikea-gladom-i00-standard` | `roboarc-ikea-gladom-i00-standard` |
| `ikea-ikea365-i00-privileged` | `roboarc-ikea-ikea365-i00-privileged` |
| `ikea-ikea365-i00-standard` | `roboarc-ikea-ikea365-i00-standard` |
| `ikea-kyrre-i00-privileged` | `roboarc-ikea-kyrre-i00-privileged` |
| `ikea-kyrre-i00-standard` | `roboarc-ikea-kyrre-i00-standard` |
| `ikea-lack-i00-privileged` | `roboarc-ikea-lack-i00-privileged` |
| `ikea-lack-i00-standard` | `roboarc-ikea-lack-i00-standard` |
| `ikea-stegesund-i00-privileged` | `roboarc-ikea-stegesund-i00-privileged` |
| `ikea-stegesund-i00-standard` | `roboarc-ikea-stegesund-i00-standard` |
| `ikea-stenerik-i00-privileged` | `roboarc-ikea-stenerik-i00-privileged` |
| `ikea-stenerik-i00-standard` | `roboarc-ikea-stenerik-i00-standard` |
| `ikea-svallet-i00-privileged` | `roboarc-ikea-svallet-i00-privileged` |
| `ikea-svallet-i00-standard` | `roboarc-ikea-svallet-i00-standard` |
| `ikea-uppsta-i00-privileged` | `roboarc-ikea-uppsta-i00-privileged` |
| `ikea-uppsta-i00-standard` | `roboarc-ikea-uppsta-i00-standard` |
| `ikea-variera-i00-privileged` | `roboarc-ikea-variera-i00-privileged` |
| `ikea-variera-i00-standard` | `roboarc-ikea-variera-i00-standard` |
| `ikea-vesken-i00-privileged` | `roboarc-ikea-vesken-i00-privileged` |
| `ikea-vesken-i00-standard` | `roboarc-ikea-vesken-i00-standard` |

Its earlier releases were inside the shared repository: 8 tasks (16 dirs) with images `eai-ikea-mujoco:0.3.0` /
`-sim:0.4.0` (embodied-first-exam #299, 2026-10-17).

### New bases (reviewed and approved by the benchmark owner, 2026-10-18)
- **orbits-02-i01** (orbits, easier): runs on a 0.3125 ms MuJoCo step (solver_substeps 8), see
  [docs/simulator/families/orbits.md](docs/simulator/families/orbits.md). Reference end state b9c68d6502105df4.
- **knot-untying-07-i01** (knot_untying rd, easier): a re-timed route. The reference replays in 28.1 % of the 3 h budget
  (3037 s on a Ryzen 9 7950X), under the knot family's 30 % bar. End state 8ae062ea4c733dbc.
- **blanket-flatten-09-i01** (blanket_flatten, hard): the Genesis release's hard tier, ported to the deterministic GPU
  cloth solver (`blanket_vbd_gpu`), route hard5. End state 6a9c4796da46cf47.

### Changed
- **blanket-fold-05-i01: a new reference** in both modes. It is the released route, re-recorded with the oracle rate
  limiter's guard-stop clamp, so every arm joint step stays within the standard service's joint-speed rule (max
  0.0390 rad). It has 10195 rows (was 17189, one of them over the rule), end state 4f1b6724fc0a6512, and standard rows
  = privileged rows. Timing exception (owner, 2026-10-18): <= 30 % of the 3 h budget (2930 s = 27.1 % on a Zen 4
  trial host). Stress: 1 breakdown in 371 episodes, of the pressed-pile class, counted as an agent-caused warning
  (~0.27 %, under the ~1 % ceiling).
- **MuJoCo 3.14.0 + the EPA horizon-overflow fix.** MuJoCo 3.14.0's EPA keeps the polytope horizon in fixed 24-entry
  arrays and overflows them on complex convex contacts (a SIGSEGV in `mj_narrowphase`). Upstream fixed this in 3.15
  (google-deepmind/mujoco 6d662aed). The image carries a 14-byte patch of the wheel's own `libmujoco.so.3.14.0` that
  backports only that fix (`source/images/robotangle/mujoco-3.14.0-epa/`; sha256 5e7623e3… -> 52e2a24e…). A source
  build is not bit-identical to the wheel, so it is not used. Gate: every released reference reaches its recorded end
  state on the patched library (Zen 4, and Intel for the CPU bases; stock = patched), and 1150 / 1150 stress episodes
  were identical. So 0.2.0's results stand.
- **Nyx start-up retry** (`nyx_direct.py` sha256 52bb406f…, in every task's tooling copy). A renderer build that fails,
  for example on a full shared GPU, is built again after 10 s and then 30 s (x 1-1.5), at most 3 attempts, and each
  failed attempt is logged (the standard service's `nyx_ready` event counts them). The half-built renderer is kept,
  never shut down. A crash inside the build still goes through crash recovery.

### Evidence
V3 build (2026-10-18, image 0.2.1):
- Harbor on the image, both modes, rendering off: orbits-02, knot-07, blanket-flatten-09 and blanket-fold-05. Every
  oracle scored 1 at its reference's end state (privileged deterministic and instance_exact; standard live = replay),
  and every nop scored 0.
- Strict fresh-process replays of one base per family plus the hold-only bases (16 / 16), all at their gate end hashes.

From this repository (2026-10-19, a fresh clone, image `roboarc:1.0.0` built by `images/roboarc/build.sh`):
- The image has the same two layer digests as 0.2.1. All 153,404 filesystem entries are equal (path, mode, owner,
  mtime, size, sha256).
- `mk_tasks.py --check` (202 match), `run_invariants.sh`, unit tests, check-video, check-resources and check-cutoffs
  all pass. `run_all.sh` passes too: the invariants hold, and the advisories report nothing beyond warnings.
- Harbor, oracle and nop in both modes, rendering off, on interlocked-claws-03, hedgehog-in-cage-01 and cast-coil-01:
  12 / 12 as expected (oracle 1 at the reference end, deterministic; nop 0).
- Strict fresh-process replays in the new image (both replays of each): one base per family plus the hold-only bases
  (16) and the four V3 items (orbits-02, knot-untying-07, blanket-flatten-09, blanket-fold-05). All 20 end at their
  recorded hashes, 15 + blanket-flatten-08 equal to the 0.2.1 spot replays, and the V3 items at b9c68d6502105df4,
  8ae062ea4c733dbc, 6a9c4796da46cf47 and 4f1b6724fc0a6512.
- IKEA: both generators' check modes give 0 differences. Both images rebuilt from `ikea/` have the V3 images' ids
  (5d5090ea1910, 050cd45d70f6). The unit tests pass in
  the image. Harbor oracle and nop on svallet and lack in both modes, before and after the rename: 8 / 8 each time
  (oracle 1, nop 0). Fresh replays of svallet's and lack's references from the renamed dirs give the 0.3.2 records'
  end and settled hashes.

Not yet in: the Intel replays of the 13 blanket bases on the GPU. They gate the public image push.

### Repository
- A standalone repository: the RoboARC tasks, generator, simulator source, image recipe, CI checks and the shared task
  layer they use (provenance: [PROVENANCE.md](PROVENANCE.md)).
- Image references: `ghcr.io/mll-lab-nu/eai-roboarc:0.2.1` -> `ghcr.io/embodied-first-exam/roboarc:1.0.0`. The
  monitor sidecar and the Genesis base image are referenced by their unchanged digests in the consolidated package
  `ghcr.io/mll-lab-nu/efe` (`monitor-0.1.0`, `robotangle-0.2.22`), because the old `eai-*` packages no longer exist.
- The task README's release line names RoboARC 1.0.0 and the simulator source commit.
- IKEA task dirs: only the image references changed (the Dockerfiles' `FROM`, the compose `image:` lines, the
  bindings' `reference`, the README rows). The host names in their evidence lines were replaced by host classes.

## 0.2.1 (2026-10-18, V3 build; not released before 1.0.0)

The V3 build on the development branches of the shared repository (image `eai-roboarc:0.2.1`, 41a17481d67a, HF assets
revision 9f34f894a454). Its content is released as 1.0.0 above.

## 0.2.0 (2026-10-17, "V2"; embodied-first-exam #301)

98 bases (196 task directories). Image `eai-roboarc:0.2.0`.
- New bases: knot-untying-06-i01 (rd very_easy) and knot-untying-09-i01 (rd medium; 27.4 % of the budget, under the
  knot family's 30 % bar), blanket-flatten-07-i01 (easy) and blanket-flatten-08-i01 (medium), both ported to
  `blanket_vbd_gpu`.
- Standard references within the standard service's joint-speed rule (0.04 rad per control step) for 11 bases.
- buzz-wire-3d-03 / 04: references where the release had none.
- knot-06 / 09 re-frozen without the lab start's provenance in instance.json. The physics is unchanged; the births
  changed.
- Reference meta hygiene: no host paths in the references' metadata.

## 0.1.0 (2026-10-14; embodied-first-exam #186 / #187)

The MuJoCo release. 94 bases (188 task directories) replace the Genesis release of the same puzzles (RoboTangle,
`eai-robotangle` 0.2.22, whose results are not comparable). Image `eai-roboarc:0.1.0`, HF assets revision 88e0136e1b97.
- MuJoCo 3.14.0 on the CPU in float64 (blanket families: a deterministic CUDA cloth step), and numpy / OpenBLAS SIMD
  masking in every container that runs a session, so every CPU class builds the same frozen bytes.
- The 3 h agent budget in both modes. Standard mode on eai-standard/2.4, with no world coordinates in standard texts.
- Privileged grading replays need the strict goal within the agent's rows (the 2 s hold only confirms). Earlier
  privileged results can be re-graded with `scripts/roboarc/regrade.py`.
- Follow-ups on main: the image's own Harbor evidence (#203) and the lock rotor discs drawn where the physics has them
  (#218).
