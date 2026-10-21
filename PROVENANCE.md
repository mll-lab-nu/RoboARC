# Provenance

This repository was assembled on 2026-10-19 from the RoboARC branches of the shared benchmark repository
`mll-lab-nu/embodied-first-exam` (formerly `JamesKrW/robot_coding_bench`). On 2026-10-19 that repository's main was
rewritten and its branches deleted. The commits below were kept from a local copy made before that. Its history is
not carried over: this repository starts with a fresh history.

## Source commits

| source | commit | date | what came from it |
|---|---|---|---|
| `dev/robotangle`: the RoboARC release tree | `acfcec10898b65bda1cf75ade20a6ade647ac074` | 2026-10-18 13:20 -0500 | `tasks/roboarc-*` (202), `scripts/roboarc/`, `images/roboarc/`, `images/monitor/`, the shared task layer in `scripts/` (`task_conventions.py`, `set_agent_egress.py`, `eai_standard/`, `rcb_monitor/`, the runner and its tests), `ci_checks/`, `docs/` (protocol, running, taxonomy, leakage, `benchmarks/roboarc.md`), `tasks/README.md`, `.gitignore`, the MIT notice in `LICENSES/` |
| `dev/robotangle_mujoco`: the dev source | `5cb4114f109d229963df1c95e7750fab7527e452` | 2026-10-19 05:23 -0500 | `source/topogym/`, `source/assets/robots/`, `source/images/robotangle/` (Dockerfile.mujoco, VBD requirements, Warp precompile, stage_instance.py, the libuipc sm86 build recipe), `source/images/topogym/prune.py`, `source/images/robotangle/mujoco-3.14.0-epa/` (from `images/ikea-mujoco/mujoco-3.14.0-epa/`), `source/scripts/robotangle/`, `source/scripts/topogym_rules.py`, `source/.dockerignore`, `docs/families/` (from `docs/rcb/families/`), `source/docs/` (from `docs/rcb/`) |
| `dev/robotangle_mujoco` (same commit): IKEA MuJoCo | `5cb4114f109d229963df1c95e7750fab7527e452` | 2026-10-19 05:23 -0500 | `ikea/tasks/ikea-*` (24), `ikea/scripts/ikea_mujoco/`, `ikea/images/ikea-mujoco/`, `ikea/docs/benchmarks/ikea_mujoco.md`, and the shared layer they render with: `ikea/scripts/task_conventions.py`, `ikea/scripts/set_agent_egress.py`, `ikea/scripts/eai_standard/`, `ikea/ci_checks/topogym_tooling_versions.json` |
| the release commit named in `scripts/roboarc/release.json` (`release_commit`) | `67ed48da8514eb3cb95c2d5b64cc8639125d6b7a` | 2026-10-18 12:49 -0500 | an ancestor of 5cb4114f10. Its simulator source, robot assets and image helpers are byte-identical to `source/` (the only later change in those paths is orbits-02's `dynamics` line in `source/scripts/robotangle/package_text/`). The task READMEs name it as "simulator source `67ed48da8`" |
| main before the rewrite | `19dcad58b5691b93d95646dd26a6115e69b2925f` | 2026-10-19 00:59 -0700 | the release history in CHANGELOG.md (0.1.0: #186 / #187, 0.2.0: #301) |
| TopoGym's `dev/robotangle_mujoco` (the development line), synced on branch `dev/jianwen` on 2026-10-20 | `497a9e9d7f630c40bc315a01795a8a4beec5e7f7` | 2026-10-20 01:45 -0500 | the five new task settings and what they need in `source/topogym/`, `source/scripts/robotangle/` and `source/scripts/topogym_rules.py`; five new docs in `docs/families/`; `source/docs/MUJOCO_CORE.md`'s task mechanisms. Merged per file onto 1.0.0: "Sync of 2026-10-20" below |
| the shared repository's main (`mll-lab-nu/embodied-first-exam`), added on branch `dev/jianwen` on 2026-10-20 | `3518d6558f56d65acb93b8bd03a3b5f2d43f2a7b` | 2026-10-19 04:49 -0700 | the usage gateway the runner starts for `GATEWAY=1` (`scripts/usage_gateway/`: `gateway.py`, `rawlog.py`, `api_paths.json`, `tests/`) and `.env.example`, byte-identical. 1.0.0 shipped the runner without them. This is the newest commit whose gateway takes every option the runner passes and reads `scripts/set_agent_egress.py`'s host lists from the file's text; the next one (#424) imports that module instead. `scripts/set_agent_egress.py` has had an `if __name__ == "__main__"` guard since, so importing it no longer runs its command line (same command-line behaviour) |
| TopoGym's `jianwen/appearance-seeds-release` (dev/robotangle_mujoco 497a9e9d7f + appearance seeds), synced on branch `jianwen/seeds-release` on 2026-10-20 | `622c7801c6ae2df9514b081c17e98af9e7e3940e` | 2026-10-20 18:45 -0500 | the changes 497a9e9d7f..622c7801c6 on the path map above, applied per file as a three-way merge onto the `dev/jianwen` sync: appearance seeds and render fixes in `source/topogym/`, `stage_instance.py --seed-file` (installed by `Dockerfile.mujoco`, kept in the context by `source/.dockerignore`), `assets.py` seeds, `release_rebuild.py` `--appearance-seeds` / `--goal-pictures` merged with this copy's V2-release additions (one conflict: the release tag line, kept as `<bench>:<version>`), caliper_sorting package texts, four `docs/families/` pages. Not carried: its `ci_checks/` seed tests and `docs/develop/` |
| TopoGym's `jianwen/appearance-seeds-1.2.0` (dev/robotangle_mujoco 9115da07d8 merged with 622c7801c6), synced on branch `jianwen/seed-derive` on 2026-10-20 | `738231f5cb48bfe3f981df42a91a1289fb0bedb5` | 2026-10-20 22:02 -0500 | the changes 622c7801c6..738231f5cb on the path map above (the development line's 497a9e9d7f..9115da07d8 and the merge's appearance tables re-attached to the new looks), applied per file as a three-way merge onto the 622c7801c6 sync: clock_assembling v3 (drawn realistically, the reviewed rules, the bridge solid to the wheels, a new -i01 reference, birth ad4ee5041eeb19de), battery_replacement's Xbox controller shell and rule C counted from the slot (`WeakHoldGraspTarget` in `source/topogym/utils/grasp.py` and `world/mujoco/grasp.py`; a re-stamped -i01 reference, birth 764192d9a7e49a0f), their package texts, `references/clock_assembling/clock_oracle.py`, `test_clock_rules.py`, the two `mujoco_release.json` notes. The two `docs/families/` pages conflicted where this copy had replaced host names: those hunks take the development line's text with the hygiene of item 8. `source/topogym/` outside `planning/` is byte-identical to 738231f5cb. The development line's change leaves the other families alone: 14 contact-weld bases, one per family, made and strictly replayed on 497a9e9d7f and on 9115da07d8 have the same births and end states (`<host work dir>/neutrality_9115da07d8/results/compare2.txt`). Not carried: `tests/` |
| TopoGym's `dev/robotangle_mujoco` at `3eed5d8b33` (dev after the merge of `jianwen/appearance-seeds-1.2.0`), synced on branch `jianwen/seed-derive` on 2026-10-21 (WIP) | `3eed5d8b33efacd8f99ac5500a05efb1905039c5` | 2026-10-21 10:53 -0500 | the changes 738231f5cb..3eed5d8b33 on the path map above, per file as a three-way merge onto the 738231f5cb sync: d0de6082f4 (dev's merge of `jianwen/appearance-seeds-1.2.0`: the two `release_rebuild.py` fixes this copy already had), f6471bc717 (battery_replacement 2b: the props drawn realistically and put on the appearance axes, `tray_finish` / `box_finish` defaults renamed `dark` / `navy`, the package texts no longer name the stick's colours; physics, birth and end unchanged), 9e36ecf429 (F4: the nine new bases' package texts cite `docs/PROTOCOL.md` and `docs/STANDARD_MODE.md`), 3eed5d8b33 / e18370887a (caliper_sorting 2b: drawn like the real things, a `fixture_finish` axis, `bin_finish`'s default renamed `grey_blue`). One conflict, the battery note of `mujoco_release.json` (this copy's host-name hygiene): the development line's text, hygiene applied. Not carried: `docs/rcb/handover/`. **Stale after this sync:** battery-replacement-01-i01's and caliper-sorting-01-i01's frozen instances (seed 0 and the derived seeds 1 and 2), their pins and bindings: 2b changes the files seed 0 declares (battery `controller.xml`, `stick.xml`; caliper 11 GLB textures), so their `asset_identity` no longer matches (`docs/handover/jianwen_push_20261021.md`). Carried with it, as `source/ci_checks/`: TopoGym's `ci_checks/test-appearance-seeds.py` (byte for byte) and `ci_checks/test-release-seeds.py` (one change: `premade={}` in its fake arguments, since this copy's `write_release` names premade instances) |

## External inputs (pinned)

| input | pin |
|---|---|
| frozen instances: HF dataset `eai-v2-internal/robotangle-assets` (private) | revision `9f34f894a454d4f39ebaed5dcf8c8c9d40f828b9`: 1.0.0's 101 seed-0 instances, 21477 files by sha256 in `scripts/roboarc/assets.json`, unchanged. Since 1.2.0 assets.json also pins, by sha256 only, the 9 new bases' instances (262 files) and every base's appearance seeds 1 and 2 (43448 files): those are **not in the dataset**. Nothing is uploaded to the organisation dataset and its revision is not raised (owner, 2026-10-20); the 1.2.0 image is built from the local instances (`images/roboarc/build.sh --instances`). The seeds derive again from the seed-0 instances (`docs/benchmarks/roboarc_seeds_release.md`, steps 0 and 4) |
| Genesis base image (`eai-robotangle` 0.2.22) | `ghcr.io/mll-lab-nu/efe:robotangle-0.2.22@sha256:eb1a75177bf3d11f60540ef5a285c5a47d8cf35e2e99630759bb108b9f05dfad`, the same digest as the published `eai-robotangle:0.2.22` |
| monitor sidecar (`eai-monitor` 0.1.0) | `ghcr.io/mll-lab-nu/efe:monitor-0.1.0@sha256:f2c910f53f79c2bfa4a71298b10c78383cae3afc40720ca562cc845b8d20fb0c`, the same digest as the published `eai-monitor:0.1.0` |
| standard agent base | `docker.io/library/python:3.12.14-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f` |
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
   from `run_invariants.sh` (H.5 keeps `test_retry_policy.py`). No protocol rule changed.
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
   and the task READMEs re-rendered from them. Pinned files keep their text: `rcb_ikea.py`, `ikea_nyx.py` and the
   scenes. Renamed (user, 2026-10-19) by the generators' `name()`: `ikea-<scene>-i00-<mode>` ->
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

## Sync of 2026-10-20 (branch `dev/jianwen`)

TopoGym's development line `dev/robotangle_mujoco` at `497a9e9d7f` carried onto 1.0.0 in the layout above (the same
path map as the 1.0.0 assembly), for a later merge into main.

- **Base:** `a3cce97cf304f4b6d297ec3243974e1f8777791c` (2026-10-16), the newest commit that line shares with the shared
  repository's `dev/robotangle_mujoco` as last fetched there (`7bdb86da29`, 2026-10-17). 5cb4114f10 itself is not in
  the TopoGym repository. Mapped onto this layout, the base's files differ from 1.0.0's only where 1.0.0 changed them
  after the base: 33 files in `source/topogym/planning/` (V3 and hygiene), 75 in `source/scripts/robotangle/`, 10 in
  `docs/families/`, 9 in `source/docs/`, 6 in `source/images/robotangle/`, `source/.dockerignore`. `source/assets/robots/`,
  `prune.py`, `topogym_rules.py` and the rest of `source/topogym/` are byte-identical to the base.
- **Rule, per file:** a file only the development line changed since the base takes its bytes; a file only 1.0.0 changed
  keeps 1.0.0's; a file both changed is merged three-way (six files: `mujoco_release.json`, `release_rebuild.py`,
  `shared/nyx_direct.py`, `package_text/interlocked_claws/README.md`, `stress/corpus/SHA256SUMS`,
  `source/docs/MUJOCO_CORE.md`; none conflicted). The development line changed nothing under `ikea/` since the base:
  IKEA stays as released.
- **Kept 1.0.0's** where the development line only carried an older parallel edit of a released task's text:
  `package_text/cast_coil/cast-coil-06-i01-standard/instruction.md` (the released 0.047 s per control step).
- **Not carried:** `docs/rcb/handover/` (as in 1.0.0); `images/topogym/uipc-sm86/` (the prebuilt pyuipc wheel and its
  build record; the recipe is `source/images/robotangle/uipc-sm86-build/`); `tests/` of the TopoGym repository.
- **Hygiene** as in item 8 on the added documents and records (a lab RTX 4090 host, a cloud host, `<host work dir>`). Code
  keeps its text.
- **No task directories** for the new bases yet: they come with a release rebuild (`release_rebuild.py`, instances in the
  image, `scripts/roboarc/release.json`, `mk_tasks.py`).
- Checks: the 72 added or changed Python files compile (Python 3.12); the changed JSON files parse;
  `scripts/roboarc/mk_tasks.py --check`: 202 task dirs match a fresh render.
