# Paths cited by the frozen task files

The pinned tooling copied into the task directories (`scripts/roboarc/shared/`: `rcb_topogym.py`, `render_bridge.py`,
`nyx_direct.py`; IKEA's `rcb_ikea.py`, `ikea_nyx.py` and its `nyx_direct.py`), the eai-standard 2.4 layer and IKEA's
live-grading pair, and the hashed simulator source (`source/topogym/`, outside `planning/`) were written in the shared
development repository the benchmark came from. Their bytes are pinned (`tooling_sha256`, the runtime bindings,
SOURCE.json, the shared layers' byte identity), so the paths they cite stay as written. Since 1.0.1 every other task
file (READMEs, `task.toml`, compose files, Dockerfiles, `test.sh`, the graders' and sim services' docstrings) cites
this repository's paths; the 1.0.0 task files cited the older paths below too. This page maps each one to the file
that holds it in this repository.

## Documents

| cited path | in this repository |
|---|---|
| `docs/rcb/MUJOCO_CORE.md` | [`docs/simulator/MUJOCO_CORE.md`](../simulator/MUJOCO_CORE.md) |
| `docs/rcb/RENDER_BRIDGE.md` | [`docs/simulator/RENDER_BRIDGE.md`](../simulator/RENDER_BRIDGE.md) |
| `docs/rcb/benchmarks/topogym.md` | [`docs/simulator/TOPOGYM_PROTOCOL.md`](../simulator/TOPOGYM_PROTOCOL.md) (section names unchanged) |
| `docs/rcb/PROTOCOL.md`, `docs/protocol/README.md` (IKEA) | [`docs/PROTOCOL.md`](../PROTOCOL.md) |
| `docs/rcb/STANDARD_MODE_2_2.md`, `docs/protocol/standard_mode.md` (IKEA) | [`docs/STANDARD_MODE.md`](../STANDARD_MODE.md): the standard mode the tasks serve (eai-standard 2.4; IKEA's live-grading variant of the layer is `scripts/eai_standard/v2_4_live/`) |
| `docs/rcb/TAXONOMY.md` | [`docs/TAXONOMY.md`](../TAXONOMY.md) |
| `docs/rcb/STRESS_VALIDATION.md`, `docs/rcb/STRESS_HOWTO.md` | [`docs/simulator/STRESS_VALIDATION.md`](../simulator/STRESS_VALIDATION.md), [`docs/simulator/STRESS_HOWTO.md`](../simulator/STRESS_HOWTO.md) |
| `docs/rcb/redesign/<family>.md` | [`docs/simulator/redesign/`](../simulator/redesign/) (buzz_wire_3d, fiber_bundling, interlocked_claws, knot_untying, rope_threading; the blanket_flatten and blanket_fold notes were not carried) |
| `docs/rcb/families/<family>.md`, `docs/families/<family>.md` | `docs/simulator/families/<family>.md` (the per-family development notes) |
| `docs/rcb/MUJOCO_MIGRATION.md`, `docs/rcb/TOPOGYM_SPEEDUP_RECORD.md`, `docs/rcb/BLANKET_GENERATED_INITIAL_STATES.md` | [`docs/history/`](../history/) |
| `images/robotangle/README.md` ("VBD cloth") | [`docs/simulator/VBD_CLOTH.md`](../simulator/VBD_CLOTH.md) |
| `docs/rcb/HARBOR_FREEZE_RUNBOOK.md`, `docs/rcb/ROBOTANGLE_TRACKER.md`, `docs/develop/...`, `docs/debug_logs/...` | not carried (development notes of the earlier repository) |

## Image recipes

| cited path | in this repository |
|---|---|
| `images/roboarc/Dockerfile`, `images/topogym/`, `images/robotangle/Dockerfile`, `images/roboarc/Dockerfile.genesis-0.2.22` | the release image's recipe: [`source/images/roboarc/Dockerfile`](../../source/images/roboarc/Dockerfile), built from `source/` by `images/roboarc/build.sh`; it writes the Vulkan ICD and EGL vendor files. Up to 1.0.x the image was a delta on the RoboTangle 0.2.22 (Genesis) image, whose recipe (`images/roboarc/Dockerfile.genesis-0.2.22`) is in RoboARC 1.0.0 |
| `images/roboarc/stage_instance.py`, `images/robotangle/stage_instance.py` | [`source/images/roboarc/stage_instance.py`](../../source/images/roboarc/stage_instance.py), installed as `/opt/robotangle/stage_instance.py` |
| `images/topogym/prune.py` | [`source/images/topogym/prune.py`](../../source/images/topogym/prune.py) |
| `images/robotangle/mujoco-3.14.0-epa/`, `source/images/robotangle/mujoco-3.14.0-epa/` | [`source/images/roboarc/mujoco-3.14.0-epa/`](../../source/images/roboarc/mujoco-3.14.0-epa/), notes in [`docs/simulator/MUJOCO_EPA_PATCH.md`](../simulator/MUJOCO_EPA_PATCH.md) |
| `source/images/robotangle/Dockerfile.mujoco`, `source/images/robotangle/{vbd-requirements.txt,warp_precompile.py}` | [`source/images/roboarc/`](../../source/images/roboarc/) (`Dockerfile` replaces `Dockerfile.mujoco` in 1.1.0) |

## Scripts and records

| cited path | in this repository |
|---|---|
| `scripts/roboarc/harbor_check.py` | [`scripts/roboarc/harbor_check.py`](../../scripts/roboarc/harbor_check.py) |
| `scripts/robotangle/release_rebuild.py` | [`tools/release/release_rebuild.py`](../../tools/release/release_rebuild.py) |
| `scripts/robotangle/release.json` (runtime, `uipc_builds`) | [`scripts/roboarc/release.json`](../../scripts/roboarc/release.json) (no libuipc builds since 1.1.0); the Genesis RoboTangle release record (`records/genesis_release/`) is in RoboARC 1.0.0 |
| `scripts/robotangle/shared/` | [`scripts/roboarc/shared/`](../../scripts/roboarc/shared/) (the one copy of the pinned tooling) |
| `scripts/robotangle/oracles/<base>/` | [`scripts/roboarc/oracles/<base>/`](../../scripts/roboarc/oracles/) (released references); the Genesis `-i00` references the MuJoCo ones were ported from (`tools/references/oracles_genesis/`) are in RoboARC 1.0.0 |
| `scripts/robotangle/{references,redesign,stress}/`, `scripts/robotangle/test_*.py` | [`tools/references/`](../../tools/references/), [`tools/redesign/`](../../tools/redesign/), [`tools/stress/`](../../tools/stress/), [`tools/tests/`](../../tools/tests/) |
| `scripts/robotangle/mujoco_release.json`, `scripts/topogym_rules.py` | [`tools/release/`](../../tools/release/) |
| `scripts/robotangle/freeze_manifest.*.json` | [`records/freeze_manifests/`](../../records/freeze_manifests/) |
| `scripts/oracle_runners/*_session_oracle.py` | [`tools/references/oracle_runners/`](../../tools/references/oracle_runners/) |
| `scripts/robowits/shared/oracle_replay.py`, `scripts/robowits/oracles/<family>/oracle_qpos.npy` (the standard solution player's header) | the player is [`scripts/roboarc/shared/standard/solution/oracle_replay.py`](../../scripts/roboarc/shared/standard/solution/oracle_replay.py); its rows are each task's `solution/oracle_qpos.npy`, from `scripts/roboarc/oracles/<base>/standard.npz` |
| `scripts/task_dynamics.toml` | not carried; the `dynamics` field is defined in [`docs/PROTOCOL.md`](../PROTOCOL.md) §2.14 |
| `tasks/<family>/common.py` (the tooling's comments: a path inside the simulator package) | [`source/topogym/tasks/`](../../source/topogym/tasks/) |
| `source/dependencies/libuipc` (an error message of the tooling) | not carried: libuipc was part of the Genesis base image until 1.0.x (its builds' recipe `images/roboarc/uipc-sm86-build/` is in RoboARC 1.0.0) |
| `scripts/check_knot_clearance.py`, `scripts/prepare_blanket_initial_states.py`, `scripts/check_*fast_paths.py` and the other one-off checks named in comments | not carried (development checks; their results are quoted where they are cited) |

## Removed in 1.1.0

The Genesis-era parts that only served porting the Genesis release to MuJoCo were removed in 1.1.0
([docs/history/GENESIS_REMOVAL.md](../history/GENESIS_REMOVAL.md)). The development notes under `docs/simulator/` cite
them as they were; they are in RoboARC 1.0.0 (main 153eb8cf01).

| cited path | in RoboARC 1.0.0 |
|---|---|
| `tools/release/templates/`, `tools/release/package_text/`, `tools/release/package_text_tool.py`, `scripts/robotangle/package_text/`, `scripts/robotangle/package_text_tool.py`, `scripts/roboarc/port_package.py` | the Genesis-era task texts new bases started from, and the tools that drafted and ported them |
| `tools/references/oracles_genesis/` | the 52 Genesis `-i00` references the MuJoCo references were re-recorded from |
| `records/genesis_release/` | the Genesis RoboTangle release generator |
| `tools/tests/test_mujoco_kinematics.py` | MuJoCo's FK / IK against Genesis' on the same robot |
| `images/roboarc/{Dockerfile.genesis-0.2.22,install_uipc_prefix.py,verify_uipc_wheel.py,uipc-sm86-build}` | the Genesis base image's recipe and its libuipc builds |
| `topogym/world/{genesis_world,tabletop}.py`, `topogym/world/physics/{fem,ipc,grasp_ipc,grasp_pbd,grasp_rigid,pbd_table_friction,release_geometry,rigid_contacts}.py`, `topogym/world/visuals/`, `topogym/sensors/renderers/nyx.py` | the Genesis world, its physics backends and its renderer (`source/topogym/` in 1.0.0) |

## IKEA MuJoCo

The IKEA files cite paths of their development tree, where the IKEA suite was the repository's root.

| cited path | in this repository |
|---|---|
| `scripts/ikea_mujoco/…` (`rcb_ikea.py`, `ikea_nyx.py`; 1.0.0's texts) | [`ikea/scripts/ikea_mujoco/`](../../ikea/scripts/ikea_mujoco/) |
| `images/ikea-mujoco/…` (`ikea_nyx.py`; 1.0.0's texts) | [`ikea/images/ikea-mujoco/`](../../ikea/images/ikea-mujoco/) |
| `scripts/robotangle/shared/` (`ikea_nyx.py`: where its `nyx_direct.py` came from) | [`scripts/roboarc/shared/`](../../scripts/roboarc/shared/) (a later version of the file) |
| `scripts/eai_standard/v2_4/` (1.0.0's IKEA texts: the 2.4 service and client) | [`scripts/eai_standard/v2_4_live/`](../../scripts/eai_standard/v2_4_live/) |
| `docs/protocol/README.md` (1.0.0's IKEA texts) | [`docs/PROTOCOL.md`](../PROTOCOL.md) |
| `ikea/tasks/` (1.0.0) | `tasks/roboarc-ikea-*` |
