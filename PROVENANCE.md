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
| the shared repository's main: the usage gateway (added after 1.1.0) | `3518d6558f56d65acb93b8bd03a3b5f2d43f2a7b` | 2026-10-19 04:49 -0700 | `scripts/usage_gateway/` (`gateway.py`, `rawlog.py`, `api_paths.json`, `tests/`) and `.env.example`, byte-identical, except what DSH added on 2026-10-28: `api_paths.json`'s `deepseek` provider (DeepSeek's Messages API) and its comment, `tests/test_paths.py`'s DeepSeek cases, and `.env.example`'s `DEEPSEEK_API_KEY` line, and what Kimi Code added on 2026-11-06: `api_paths.json`'s `moonshot` provider (Moonshot's Chat Completions and its image and video uploads) and its comment, `gateway.py`'s upload and `record_fields` rules and its reading of Moonshot's usage, their tests in `tests/test_paths.py` and `tests/test_budget_stop.py`, and `.env.example`'s `KIMI_API_KEY` and `KIMI_BASE_URL` lines. The newest commit whose gateway takes every option the runner passes and reads `scripts/set_agent_egress.py`'s host lists from the file's text; later versions import that module, which runs nothing on import since the same change (its command line is under `if __name__ == "__main__"`) |
| TopoGym's development line `dev/robotangle_mujoco` (for 1.2.0; prepared by Jianwen and ported onto 1.1.0) | `497a9e9d7f630c40bc315a01795a8a4beec5e7f7` | 2026-10-20 01:45 -0500 | the five new task settings and what they need: `source/topogym/` (35 files: 12 changed, 23 new), the pinned tooling's new task entries (`scripts/roboarc/shared/`), `tools/release/` (`mujoco_release.json` new bases, `release_rebuild.py`, `topogym_rules.py`), the nine new bases' references (`scripts/roboarc/oracles/`) and their builders (`tools/references/`), tests (`tools/tests/`), a stress reproducer, the development notes (`docs/simulator/families/`, `docs/simulator/MUJOCO_CORE.md`): "1.2.0 (in progress)" below |
| TopoGym's `jianwen/wire-splicing` (for 1.3.0; ported by Jianwen onto 1.2.1, branch `dev/jianwen-2` `4ce9596e`) | `cb3d05437c` (the public goal of `4a2359b44b`) | 2026-10-22 | the wire_splicing family: `source/topogym/tasks/wire_splicing/`, `source/topogym/assets/geometry/wire_splicing.py`, its registry / variation / profile / goal-text entries, the two opt-in grasp options and the weld work guard (`world/mujoco/grasp.py`, `utils/grasp.py`), its appearance axis, the tooling's task entry, its packages and references, tests: "1.3.0" below |

## External inputs (pinned)

| input | pin |
|---|---|
| frozen instances: Hugging Face dataset `eai-v2-internal/robotangle-assets` | revision `b2cf9b552f310ef4cfbd91df62ccfa837757c93a`; 21477 files by sha256 in `scripts/roboarc/assets.json` |
| puzzle image base (1.1.0) | `python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3` (the 0.2.22 image's own base); Debian packages from `snapshot.debian.org` 20260930T230000Z at the versions in `source/images/roboarc/Dockerfile`; wheels by sha256 in `source/images/roboarc/runtime.lock` and `vbd-requirements.txt` |
| Genesis base image (`eai-robotangle` 0.2.22; 1.0.x only) | `ghcr.io/mll-lab-nu/efe:robotangle-0.2.22@sha256:eb1a75177bf3d11f60540ef5a285c5a47d8cf35e2e99630759bb108b9f05dfad`, the same digest as the published `eai-robotangle:0.2.22` |
| monitor sidecar (`eai-monitor` 0.1.0) | `ghcr.io/mll-lab-nu/efe:monitor-0.1.0@sha256:f2c910f53f79c2bfa4a71298b10c78383cae3afc40720ca562cc845b8d20fb0c`, the same digest as the published `eai-monitor:0.1.0` |
| standard agent base | `docker.io/library/python:3.12.14-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f`; its packages by version and wheel sha256 in `scripts/eai_standard/client-requirements.txt` (from 1.1.0) |
| MuJoCo EPA fix | upstream google-deepmind/mujoco `6d662aedda693a1225a03a26c6b3ddd5e3cc4fbe`, backported as a byte patch |

| IKEA scene meshes: Hugging Face dataset `eai-v2-internal/ikea-mujoco-assets` | revision `610b424b3d4152d8990e226eae60b9170ac7dbdf`; every file by sha256 in `ikea/scripts/ikea_mujoco/assets.json` |
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

## 1.2.0: appearance seeds and new task settings

1.2.0 adds nine new bases (four new families: bed_bolt 01-05, caliper_sorting, clock_assembling,
battery_replacement; and Cast Trinity, interlocked-claws-13) and appearance-only scene seeds. The source part was
prepared by Jianwen as a three-way, per-file merge of TopoGym's development line `497a9e9d7f` onto 1.0.0 (merge base
`a3cce97cf304f4b6d297ec3243974e1f8777791c`, 2026-10-16), then ported onto 1.1.0:

- **Paths:** 1.1.0's layout (`source/scripts/robotangle/` -> `scripts/roboarc/shared/`, `scripts/roboarc/oracles/`,
  `tools/release/`, `tools/references/`, `tools/tests/`, `tools/stress/corpus/`; `docs/families/` dev notes ->
  `docs/simulator/families/`; `source/docs/MUJOCO_CORE.md` -> `docs/simulator/MUJOCO_CORE.md`).
- **Merged with 1.1.0:** the Genesis-only modules 1.1.0 removed stay removed; `simulator/env.py` keeps both changes
  (different hunks); the tooling edits apply on top of tooling v20. SOURCE.json a7487c51 (426 files) -> 2ba65d0f
  (449 files); `rcb_topogym.py` sha256 676e1803..., `nyx_direct.py` 8bf9cd0b....
- **Conflicts resolved:** `tools/release/mujoco_release.json` (1.1.0's text, the new bases added without
  `package_text`, which 1.1.0 removed); `tools/release/release_rebuild.py` (1.1.0's packages rule; a new base whose
  template is of another family is refused until its own final packages are in `scripts/roboarc/packages/`).
- **Not carried:** the new bases' `package_text` (their final packages are written in the 1.2.0 build) and the
  NYX_DIRECT edit of the Genesis-era generator. The usage gateway and the `set_agent_egress.py` guard of the same branch
  were taken earlier (the source-commit row above).
- **Released physics:** every change to the hashed source is new-family code or an opt-in branch a released
  configuration does not take; the released references are replayed strictly in the 1.1.0 image with this source to
  confirm it (the 1.2.0 validation repeats it for all bases).
- **No task directories** change yet: the bindings name 1.1.0's source and tooling until the 1.2.0 re-render, so
  `tools/release/source_hash.py --check` and `mk_tasks.py --check` do not pass on this state alone.

- **Appearance seeds and the new families' source (second port):** Jianwen's later source syncs (TopoGym `622c7801c6`,
  appearance seeds; `738231f5cb`, clock_assembling v3 and battery_replacement's controller body) applied three-way per
  file onto this source (33 files, all clean; SOURCE.json 488fbe49, 449 files); his release tooling for seeds
  (`tools/release/derive_seed_instance.py`, `release_rebuild.py --appearance-seeds`, goal-picture renderers in
  `scripts/roboarc/goal_images/`); his seed data (every base's seeds 1 and 2 pinned in `scripts/roboarc/assets.json`
  by sha256, per-seed locks, goal pictures and re-stamped references), derived on 2026-10-20 in his image of the earlier
  layout. Those seed instances are not in the HF dataset yet; 1.2.0 derives them again in its own image, checks them,
  and uploads one revision with the owner's approval. Not carried: his runner, results writer and `task_seed.py`
  changes (this repository's per-trial runner replaced them), his task dirs and his image recipe.

- **Batches 2 and 3 of the sync:** TopoGym `3eed5d8b33` (battery_replacement's handle shell and second-round looks,
  caliper_sorting's second round) and `a849170272` (colour fixes: rope-threading's seed ropes, Cast Trinity's third
  piece), 8 files, three-way per file, all clean. SOURCE.json `cacc831d…` (449 files), with this repository's own
  change of fiber_bundling's `lay` rule (either red fiber).
- **Seeds derived in this release's image and re-pinned:** `tools/release/release_rebuild.py --appearance-seeds` derived
  every base's seeds 1 and 2 again in the 1.2.0 image without instances; births and `instance_assets/` equal Jianwen's
  first derivation (220 / 220 before the later source; afterwards differing exactly where that source changed the looks
  or the text), and the canonical `instance.json` differs from his pins only in the derive provenance
  (`derived_from.source`, `derived_from.tool`). battery-replacement-01, caliper-sorting-01 and interlocked-claws-13
  were made again in that image (births equal their references'). `scripts/roboarc/assets.json` pins the seeds and the
  new bases by sha256; the HF dataset keeps revision `9f34f894a454` until the release's revision is uploaded.
- **Not carried from his branch:** his runner and results writer (this repository's per-trial runner), his task dirs,
  his images and his 1.2.0 version labels; his handover notes stay outside the repository (the colour table is in
  `docs/simulator/seeds/`).

## 1.3.0: wire_splicing (experimental)

1.3.0 adds the wire_splicing family, prepared by Jianwen as a port of TopoGym's `jianwen/wire-splicing` `cb3d05437c`
(the 25 commits since its merge base with TopoGym's main line, `a35961e06a`): the family's own files whole, the shared
files as hunks merged three-way onto 1.2.1's tree, both sides kept where they met (13 regions). His six commits are
carried as he wrote them (`dev/jianwen-2` `4ce9596e`, on 1.2.1 `bcca8557`), onto the cutoff-exception change
`686ec7fd`; the last removes the needle meter the first ones had added (it never decided success) and makes the bending
post's rod yellow at every seed.

- **Shared code, opt-in only:** `MultiRopeGraspAssistConfig` gains `weld_links_between_pads` and `hold_between_pads`
  (default off; only wire_splicing's grasp configuration builds this class), and the MuJoCo grasp backend gains the
  weld work guard, read from the frozen scene's profile (`weld_work_guard_j`; no released instance's profile has it,
  so it is off for them). The registry, variation records, goal texts, appearance axes, scene profiles and the
  tooling's task table gain wire_splicing's entries; no existing entry changes. SOURCE.json `4d2cba21…` (454 files);
  tooling v22 (`rcb_topogym.py` `412cbe02…`).
- **Instances:** the 110 bases of 1.2.0 with their seeds are bound into the 1.3.0 image byte for byte (the 1.2.0 image's
  tree, every file checked against `assets.json`); wire-splicing-01-i01 is made in the 1.3.0 image without instances
  (birth `3f36d852dfc7bc65`, its reference's) and its seeds 1 and 2 derived there (births `1436195a04dd4ed0`,
  `1a2d05fc89ef3277`, as Jianwen's). Not in the HF dataset yet (with the 1.2.0 seeds, one revision with the owner's
  approval).
- **Changed against his branch:** the wire texts' tool-frame distance (6 cm, the instance's; the template's 15 cm had
  been kept); the base's release record in `experimental_bases` (grasp rules as its texts state them); the wire's
  standard reference recorded in `tools/release/standard_speed_rule.json`; `privileged.s1/s2.npz` rewritten by
  `release_rebuild.py` (the same entries and bytes inside, another zip container).

## 1.4.0: the nine bases of 1.2.0 released

1.4.0 adds no source, tooling or instance: the image stays 1.3.0's. The nine bases prepared in 1.2.0 (above) move from
`release.json` `experimental_bases` to `bases` after the owner's review (2026-10-24), with Jianwen's Harbor evidence on
the 1.3.0 image (`dev/jianwen-2` `bc189085`, merged with his authorship). The release records, packages, references and
instances are the ones of 1.2.0 and 1.3.0; only the packages' difficulty (`untiered` -> the owner's tier) changes.

## 2.0.0: the owner's pruning

2.0.0 adds no source, tooling or instance and removes none from the image: the image stays 1.3.0's. 33 puzzle bases
leave the release by the owner's decision (2026-10-25; the list is in CHANGELOG.md). Their task dirs, packages,
references and release records are not in this tree; the maintainers keep them on the branch
`archive/removed-in-2.0.0` (= 1.4.0, `315c10ca`). The records of the 78 remaining bases are unchanged.

## 2.1.0: the bed-bolt drive card, a wire_splicing grader fix, the blanket release check

- Three commits by Jianwen, merged with his authorship and hashes (`e762c138` merges `6c24f636`): a card on the table in
  the five bed-bolt scenes with the bolt's head and its recess drawn 4.5 times their size (`6c24f636`, visual only: the
  MuJoCo scene, the births and the references are unchanged), a fix of wire_splicing's coil count on a one-node strand
  found by his v2.13b stress round (`2b765e6a`; the reference never takes that branch), and the stress round's wall
  limit following the owner's agent-time exception (`097a3123`).
- The buzz-wire 06, 07 and 10 texts state their own loop opening (`04c7dd90`; text only).
- The blanket families' release check leaves out the world-fixed arm bases (`active_parts_clearance_v1`, terminal
  revision `release_settling_v2_active_parts`), from the failure analysis of blanket-flatten-07. The revision changes the
  ten blanket instances' identity, not their physics: `tools/release/refresh_instance.py` rebuilt them with their assets
  byte for byte, and their references are re-stamped.
- The image `roboarc:2.1.0` is 1.3.0's but for those source files (and `SOURCE.json`, and `.pyc` files whose only
  difference is their 4-byte source-mtime field) and the fifteen re-made or refreshed instances with their seeds; it
  still carries the instances of the 33 bases removed in 2.0.0, as 1.3.0 does. The bed-bolt instructions gain one
  sentence about the card, the blanket texts state the new release rule.

## 2.2.0: appearance seeds v2

- Jianwen's appearance seeds v2 (his commits up to `e0a930a3`, kept with his authorship and hashes): seeds 1 and 2
  change the room's lighting, the table top, its frame and the floor, and the task's objects' colours and finishes; the
  goal marker is 5 mm wide and emissive at every seed; texts no longer name parts by a colour that now changes; IKEA
  tasks allow seeds 0-2 (standard-mode lighting only). His branch also carries runner changes (each trial's host record,
  one list of the family classes for task resources, GPU placement, DSH).
- The image `roboarc:2.2.0` is built by `tools/release/release_rebuild.py` from that source: the nine CC0 Poly Haven
  texture sets are downloaded at build and sha256-checked; the 30 goal-marker bases' seed-0 instances are made again in
  it (their births and references unchanged); the other 48 bases' seed-0 instances are 2.1.0's; seeds 1 and 2 of all 78
  bases are derived again (`tools/release/derive_seed_instance.py`: `instance_assets/` byte for byte, physical fields
  and birth equal to seed 0's but for the task's instance id), and their references re-stamped. Like 2.1.0 it still
  carries the instances of the 33 bases removed in 2.0.0. The IKEA images are unchanged.
- The release records keep their 2.1.0 fields but for the instance package, release commit, the re-made bases'
  creation source and the colourless design strings; the seeds' Harbor records are this release's (CHANGELOG, 2.2.0,
  Evidence).

## After 2.2.1: the authoring tools

The tools for authoring new tasks and reference solutions that the sections above name (`tools/references/`,
`tools/redesign/`, `tools/stress/`, `source/topogym/planning/`, `ikea/scripts/ikea_mujoco/stress/`) are not part of the
public release. Their removal changed no task directory and no image input: `SOURCE.json` stays `1552f29d` (455 files).
`docs/easy_set.toml` (item 7 of "What changed against the sources") is retired: results count every task.
