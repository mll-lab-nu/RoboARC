# Provenance

This repository was assembled on 2026-10-19 from the RoboARC branches of the shared benchmark repository
`mll-lab-nu/embodied-first-exam` (formerly `JamesKrW/robot_coding_bench`). On 2026-10-19 that repository's main was
rewritten and its branches deleted. The commits below were kept from a local copy made before that. Its history is
not carried over: this repository starts with a fresh history.

The tables and items below describe the tree of release 1.0.0 (commit `153eb8cf01`). The clean-up after 1.0.0 moved
several of the paths they name: see [After 1.0.0](#after-100-the-clean-up).

## Source commits

| source | commit | date | what came from it |
|---|---|---|---|
| `dev/robotangle`: the RoboARC release tree | `acfcec10898b65bda1cf75ade20a6ade647ac074` | 2026-10-18 13:20 -0500 | `tasks/roboarc-*` (202), `scripts/roboarc/`, `images/roboarc/`, `images/monitor/`, the shared task layer in `scripts/` (`task_conventions.py`, `set_agent_egress.py`, `eai_standard/`, `rcb_monitor/`, the runner and its tests), `ci_checks/`, `docs/` (protocol, running, taxonomy, leakage, `benchmarks/roboarc.md`), `tasks/README.md`, `.gitignore`, the MIT notice in `LICENSES/` |
| `dev/robotangle_mujoco`: the dev source | `5cb4114f109d229963df1c95e7750fab7527e452` | 2026-10-19 05:23 -0500 | `source/topogym/`, `source/assets/robots/`, `source/images/robotangle/` (Dockerfile.mujoco, VBD requirements, Warp precompile, stage_instance.py, the libuipc sm86 build recipe), `source/images/topogym/prune.py`, `source/images/robotangle/mujoco-3.14.0-epa/` (from `images/ikea-mujoco/mujoco-3.14.0-epa/`), `source/scripts/robotangle/`, `source/scripts/topogym_rules.py`, `source/.dockerignore`, `docs/families/` (from `docs/rcb/families/`), `source/docs/` (from `docs/rcb/`) |
| `dev/robotangle_mujoco` (same commit): IKEA MuJoCo | `5cb4114f109d229963df1c95e7750fab7527e452` | 2026-10-19 05:23 -0500 | `ikea/tasks/ikea-*` (24), `ikea/scripts/ikea_mujoco/`, `ikea/images/ikea-mujoco/`, `ikea/docs/benchmarks/ikea_mujoco.md`, and the shared layer they render with: `ikea/scripts/task_conventions.py`, `ikea/scripts/set_agent_egress.py`, `ikea/scripts/eai_standard/`, `ikea/ci_checks/topogym_tooling_versions.json` |
| the release commit named in `scripts/roboarc/release.json` (`release_commit`) | `67ed48da8514eb3cb95c2d5b64cc8639125d6b7a` | 2026-10-18 12:49 -0500 | an ancestor of 5cb4114f10. Its simulator source, robot assets and image helpers are byte-identical to `source/` (the only later change in those paths is orbits-02's `dynamics` line in `source/scripts/robotangle/package_text/`). The task READMEs name it as "simulator source `67ed48da8`" |
| main before the rewrite | `19dcad58b5691b93d95646dd26a6115e69b2925f` | 2026-10-19 00:59 -0700 | the release history in CHANGELOG.md (0.1.0: #186 / #187, 0.2.0: #301) |
| the shared repository's main: the usage gateway (added after 1.1.0) | `3518d6558f56d65acb93b8bd03a3b5f2d43f2a7b` | 2026-10-19 04:49 -0700 | `scripts/usage_gateway/` (`gateway.py`, `rawlog.py`, `api_paths.json`, `tests/`) and `.env.example`, byte-identical. The newest commit whose gateway takes every option the runner passes and reads `scripts/set_agent_egress.py`'s host lists from the file's text; later versions import that module, which runs nothing on import since the same change (its command line is under `if __name__ == "__main__"`) |

## External inputs (pinned)

| input | pin |
|---|---|
| frozen instances: HF dataset `eai-v2-internal/robotangle-assets` (private) | revision `9f34f894a454d4f39ebaed5dcf8c8c9d40f828b9`; 21477 files by sha256 in `scripts/roboarc/assets.json` |
| puzzle image base (1.1.0) | `python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3` (the 0.2.22 image's own base); Debian packages from `snapshot.debian.org` 20260930T230000Z at the versions in `source/images/roboarc/Dockerfile`; wheels by sha256 in `source/images/roboarc/runtime.lock` and `vbd-requirements.txt` |
| Genesis base image (`eai-robotangle` 0.2.22; 1.0.x only) | `ghcr.io/mll-lab-nu/efe:robotangle-0.2.22@sha256:eb1a75177bf3d11f60540ef5a285c5a47d8cf35e2e99630759bb108b9f05dfad`, the same digest as the published `eai-robotangle:0.2.22` |
| monitor sidecar (`eai-monitor` 0.1.0) | `ghcr.io/mll-lab-nu/efe:monitor-0.1.0@sha256:f2c910f53f79c2bfa4a71298b10c78383cae3afc40720ca562cc845b8d20fb0c`, the same digest as the published `eai-monitor:0.1.0` |
| standard agent base | `docker.io/library/python:3.12.14-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f`; its packages by version and wheel sha256 in `scripts/eai_standard/client-requirements.txt` (from 1.1.0) |
| MuJoCo EPA fix | upstream google-deepmind/mujoco `6d662aedda693a1225a03a26c6b3ddd5e3cc4fbe`, backported as a byte patch |

| IKEA scene meshes: HF dataset `eai-v2-internal/ikea-mujoco-assets` (private) | revision `610b424b3d4152d8990e226eae60b9170ac7dbdf`; every file by sha256 in `ikea/scripts/ikea_mujoco/assets.json` |
| IKEA base | `docker.io/library/python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1` |

## What changed against the sources

1. **Image references** (`scripts/roboarc/mk_tasks.py` `IMAGE`, `images/roboarc/VERSION` 1.0.0):
   `ghcr.io/mll-lab-nu/eai-roboarc:0.2.1` -> `ghcr.io/embodied-first-exam/roboarc:1.0.0`. The monitor
   (`scripts/rcb_monitor/inject.py`) and the Genesis base (`Dockerfile.mujoco` `BASE`) now point at the consolidated
   package with unchanged digests. `ci_checks/check-protocol.py` accepts the new registry and the shared `efe` package.
2. **Version labels:** the task README's release line now reads "RoboARC 1.0.0 ... simulator source `67ed48da8`". It
   used to read "dev/robotangle_mujoco `67ed48da8`".
3. With 1 and 2, the re-rendered task directories differ from acfcec1089 only in `README.md`, the Dockerfiles'
   `FROM`, the compose files' `image:` lines and `DIGEST`. Every grader, tooling copy, binding, oracle and instruction
   is byte-identical.
4. **Image recipe:** `Dockerfile.mujoco` copies the EPA patch script from its new path. It gains OCI labels
   (`RELEASE_VERSION`), which change no file. `images/roboarc/build.sh` replaces the shared repository's
   `fetch_image_context.sh` + generated `images/roboarc/Dockerfile` + `scripts/build_images.sh`. It runs the build
   step of `release_rebuild.py`: `Dockerfile.mujoco` from `source/` with the instances as a named build context. The
   generated duplicate recipe was removed. `images/roboarc/push.sh` replaces `scripts/push_images.sh`.
5. **release_rebuild.py:** the new image name, and the dev source found under `source/` of a worktree of this
   repository. Making new instances still needs the RoboTangle Genesis task bindings (`tasks/robotangle-*`), which are
   not carried here. Rebuilds with premade instances (`--instances`, as for 0.2.0 and 0.2.1) need nothing else.
6. **CI scoped to RoboARC:** the lists of other benchmarks' generators, tests and hooks were removed from
   `check-unit-tests.sh`, `check-generated-tasks.sh` and `check-ci-selftest.py`. The self-test keeps 78 mutations of
   RoboARC tasks; the ones written for other benchmarks' features stay in the shared repository. Removed tests whose
   fixtures are other benchmarks' tasks: `scripts/tests/test_runner_results.py` and `test_task_seed.py`. Also removed
   from `run_invariants.sh` (H.5 keeps `test_retry_policy.py`). `scripts/eai_standard/selftest.py` is not run: it covers
   the eai-standard/2.2 layer and needs 2.2 task adapters, while RoboARC's tasks use 2.4. No protocol rule changed.
7. **Shared tables:** `scripts/task_resources.toml` keeps the 8 rules that match RoboARC tasks (with their
   `[[limit]]` and `[[cap]]` tables), and
   `docs/easy_set.toml` keeps RoboARC's 31 entries.
8. **Hygiene:** host names and work paths (also `/home/<user>/` paths in planner records) were replaced in documents, records and comments (`<host work dir>`,
   `host A/B/C`, `a cloud host`, `the work notes`). Files whose bytes are pinned keep their text: the task directories,
   the tooling `rcb_topogym.py` / `nyx_direct.py` / `render_bridge.py` (sha256 in `release.json`) and the image's
   simulator source (`source/topogym/` outside `planning/`, hashed into `SOURCE.json`). A few of their comments name
   a lab host by nickname or a relative work-note path.
9. **IKEA MuJoCo** (`ikea/`, a subtree laid out as its source tree, so its generators run unchanged): the image names
   (`images/ikea-mujoco/VERSION` and `sim/VERSION` 1.0.0, `bindings.json` `reference`, `mk_tasks.py`'s registry
   assert, `mk_standard_tasks.py` `SIM_IMAGE`, `sim/Dockerfile` `FROM`) -> `ghcr.io/embodied-first-exam/roboarc-ikea`
   and `roboarc-ikea-sim`. Host names in the evidence records, the review and the oracle notes were replaced as in 8,
   and the task READMEs re-rendered from them. Pinned files keep their text: `rcb_ikea.py` and the scenes
   (`ikea_nyx.py` lost a host name from a comment in 1.0.0: sha256 37a773230f41 in every standard task). Renamed (user, 2026-10-19) by the generators' `name()`: `ikea-<scene>-i00-<mode>` ->
   `roboarc-ikea-<scene>-i00-<mode>` (the full map is in CHANGELOG.md). That changes `task.toml`'s `name`, the
   README's title and twin rows, and a path in the docstrings of `ikea_nyx.py`, `ikea_rig.py` and `server.py`. The
   evidence records keep the names the runs had. Apart from these lines, the task dirs equal 5cb4114f10's. New: `images/ikea-mujoco/build.sh`,
   `ci/check.sh`, `README.md`.
10. **Not carried:** other benchmarks, the Genesis-era RoboTangle and TopoGym task
   directories, `third_party/`, the TopoGym repository's helper / difficulty-sweep / VLA tooling, AI-agent
   instruction files, and the shared repository's workflows (they ran on self-hosted runners).

## Verification of this assembly (2026-10-19)

- `mk_tasks.py --check`: 202 task dirs match a fresh render.
- Image built from this tree (`images/roboarc/build.sh`): the same two layer digests as the 0.2.1 release image
  41a17481d67a (`sha256:3810470b…`, `sha256:5f70bf18…`), so the filesystem is identical. The config differs only in
  the new labels.
- Strict fresh-process replays and Harbor oracle / nop runs on the new tag: see CHANGELOG.md, 1.0.0, Evidence.

## After 1.0.0: the clean-up

The repository was cleaned up after 1.0.0 without changing what any task grades: every task directory, the pinned
tooling (`scripts/roboarc/shared/`), the 443 hashed simulator files (`SOURCE.json`) and the image inputs kept their
bytes. Where things went:

| 1.0.0 path | now |
|---|---|
| `source/scripts/robotangle/` (release, reference, redesign and stress tooling, tests) | `tools/` (`release/`, `references/`, `redesign/`, `stress/`, `tests/`) |
| `source/scripts/robotangle/oracles/` (copies of `scripts/roboarc/oracles/`), `source/scripts/robotangle/shared/` (an older copy of the tooling and graders) | removed; `scripts/roboarc/` holds the only copies |
| `source/scripts/robotangle/freeze_manifest*.json`, the Genesis-era release generator | `records/` |
| `source/scripts/robotangle/harbor_check.py` | `scripts/roboarc/harbor_check.py` (the path the task READMEs cite) |
| the session oracles the orbits and slotted-post-ring references were recorded with (not carried in 1.0.0) | `tools/references/oracle_runners/`, from 5cb4114f10 |
| `source/docs/`, `docs/families/` (development notes) | `docs/simulator/`, `docs/simulator/families/`, `docs/history/`; `docs/families/` holds one public page per family; `docs/rcb/README.md` maps the older paths task texts cite |
| the runner tools in `scripts/` (`run_agent_batch.sh`, `write_task_results.py`, ...) | `scripts/harbor/` |
| `ikea/docs/benchmarks/ikea_mujoco.md` | `docs/benchmarks/ikea_mujoco.md` |
| `ikea/images/ikea-mujoco/mujoco-3.14.0-epa/` (a copy) | removed; both images use `source/images/robotangle/mujoco-3.14.0-epa/` |
| `ikea/scripts/eai_standard/v2_4/` (IKEA's live-grading service and client) | `scripts/eai_standard/v2_4_live/` (in 1.0.1; first `ikea/scripts/eai_standard/v2_4_live/`); IKEA's other shared files come from `scripts/eai_standard/` |
| `PLAN.md`, `docs/PROTOCOL_HISTORY.md`, `docs/CUTOFFS.md`, `docs/REVIEWING.md` | removed; what still applies is in `docs/DEVELOPING.md`, `docs/RUNNING.md` and `docs/benchmarks/roboarc.md` |

Removed as unused by the release: the other benchmarks' entries of the shared layer and their CI checks, the
eai-standard 2.2 layer, the dead parts of the simulator package that never reached the image, IKEA's work-in-progress
material and one-off diagnostics. The generator's packages became the shipped texts (`scripts/roboarc/port_package.py`
turns Genesis-era texts of a new base into a package).

Verified on the cleaned tree: both generators' `--check` (no task byte changes), the invariants, unit tests and
advisory checks; `tools/release/source_hash.py --check` (SOURCE.json `c9171f79…`); `images/roboarc/build.sh` from a fresh
clone gives the image id of 1.0.0 (18a821a18d2b, all 153,404 filesystem entries equal), and `ikea/images/ikea-mujoco/build.sh`
the ids 5d5090ea1910 and 050cd45d70f6.

## 1.0.1: the text re-render

1.0.1 re-renders the task directories with the clean-up's generator inputs (CHANGELOG.md, 1.0.1): the texts cite this
repository's paths and carry no internal notes, `[task].name` is `roboarc/<dir>`, and the IKEA tasks moved from
`ikea/tasks/` to `tasks/` onto the repository's task layer (`ikea/scripts/task_conventions.py` and
`ikea/scripts/set_agent_egress.py` removed). The pinned tooling, the graders' and the sim services' logic, the frozen
instances, the references, the IKEA scenes and the images are those of 1.0.0, except the monitor sidecar's image:
`ghcr.io/embodied-first-exam/roboarc-monitor:0.1.0@sha256:ea1b0d4b859998d6a47e2cb7a72c3a9ae85870b32c75de1bc1448e387da07093`, built by RoboARC from
`images/monitor/` (reproducibly; its five layers have the diff ids of the 1.0.0 pin `ghcr.io/mll-lab-nu/efe:monitor-0.1.0@sha256:f2c910f5…`,
whose package RoboARC does not control; the config adds three OCI labels).

## 1.1.0: no Genesis

1.1.0 builds the release image from the Python base image instead of the RoboTangle 0.2.22 (Genesis) image
(`source/images/roboarc/Dockerfile`; every input pinned: base image digest, Debian package versions from
snapshot.debian.org, wheels by sha256), removes Genesis and libuipc from the image and the Genesis-only modules from the
simulator source (SOURCE.json 443 -> 426 files), and moves the tooling to v20 (no libuipc build choice). Every library
the graded runtime loads is the same file as in 1.0.0, and every reference replays to the end state it reaches in 1.0.0
([docs/history/GENESIS_REMOVAL.md](docs/history/GENESIS_REMOVAL.md)). Removed from the repository (in 1.0.0 at main
153eb8cf01): `records/genesis_release/`, `tools/references/oracles_genesis/`, `tools/release/{templates,package_text}/`,
`tools/release/package_text_tool.py`, `scripts/roboarc/port_package.py`, `tools/tests/test_mujoco_kinematics.py`,
`images/roboarc/{Dockerfile.genesis-0.2.22,install_uipc_prefix.py,verify_uipc_wheel.py,uipc-sm86-build/}`. The
row "Genesis base image" above is the input of 1.0.x only.

The IKEA images of 1.1.0 (`ikea/images/ikea-mujoco/`, README there) are rebuilt from the same pinned inputs as 1.0.0's,
now all pinned in the recipes: the Python base image by digest, the Debian packages at the 1.0.0 images' versions from
snapshot.debian.org (2026-10-19), the wheels by sha256, the sidecar `FROM` the base by digest; reproducible OCI archives
(base `sha256:2c56d9226aaf…`, id `9f9cbab8aa74`; sidecar `sha256:f22a1e9042c7…`, id `cbd929360257`). Unused packages
are left out; every file the graded runtime loads is the same as in 1.0.0 (`5d5090ea1910`, `050cd45d70f6`).

Every image of 1.1.0 is pinned by manifest digest in every task (`images/pins.json`): `roboarc:1.1.0@sha256:85ea90032a55…`
is the validated image `dadc2d710ad2` exported without a rebuild into a deterministic OCI archive (`images/archive.sh`:
the config kept byte for byte, so the archive's config digest is the image id, and every layer's content is its diff
id; two exports byte-identical); the IKEA images and the monitor keep the digests of their reproducible builds. The
four archives are pushed byte for byte by `images/push_all.sh`. The standard agent's client packages are installed from
a hashed lock of the same versions (`--no-deps --require-hashes`).

## 1.1.1: the images' registry

1.1.1 names the same four images (same digests and ids) under `ghcr.io/mll-lab-nu/` instead of
`ghcr.io/embodied-first-exam/` (`images/pins.json`); nothing else changed. The runtime bindings and the IKEA sim
recipe's `FROM` keep the names the images were bound and built under.
