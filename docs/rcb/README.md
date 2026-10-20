# Paths cited by the frozen task files

The task directories (`tasks/`, `ikea/tasks/`), the pinned tooling copied into them (`scripts/roboarc/shared/`:
`rcb_topogym.py`, `render_bridge.py`, `nyx_direct.py`) and the hashed simulator source (`source/topogym/`, outside
`planning/`) were written in the shared development repository the benchmark came from. Their bytes are pinned (task
DIGESTs, `tooling_sha256`, SOURCE.json), so the paths they cite stay as written. This page maps each one to the file
that holds it in this repository.

## Documents

| cited path | in this repository |
|---|---|
| `docs/rcb/MUJOCO_CORE.md` | [`docs/simulator/MUJOCO_CORE.md`](../simulator/MUJOCO_CORE.md) |
| `docs/rcb/RENDER_BRIDGE.md` | [`docs/simulator/RENDER_BRIDGE.md`](../simulator/RENDER_BRIDGE.md) |
| `docs/rcb/benchmarks/topogym.md` | [`docs/simulator/TOPOGYM_PROTOCOL.md`](../simulator/TOPOGYM_PROTOCOL.md) (section names unchanged) |
| `docs/rcb/PROTOCOL.md`, `docs/protocol/README.md` (IKEA) | [`docs/PROTOCOL.md`](../PROTOCOL.md) |
| `docs/rcb/STANDARD_MODE_2_2.md`, `docs/protocol/standard_mode.md` (IKEA) | [`docs/STANDARD_MODE.md`](../STANDARD_MODE.md): the standard mode the tasks serve (eai-standard 2.4; IKEA ships its own copy of the layer, `ikea/scripts/eai_standard/`) |
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
| `images/roboarc/Dockerfile`, `images/topogym/` | the release image's recipe: [`source/images/robotangle/Dockerfile.mujoco`](../../source/images/robotangle/Dockerfile.mujoco), built from `source/` by `images/roboarc/build.sh`; the Vulkan ICD and EGL vendor files come from its base image, whose recipe is `images/roboarc/Dockerfile.genesis-0.2.22` |
| `images/robotangle/Dockerfile` | the base image's recipe, `images/roboarc/Dockerfile.genesis-0.2.22` (the libuipc builds: `images/roboarc/README.md`) |
| `images/roboarc/stage_instance.py` | `/opt/robotangle/stage_instance.py` in the base image; `images/roboarc/stage_instance.py` holds the same bytes |
| `images/topogym/prune.py` | [`source/images/topogym/prune.py`](../../source/images/topogym/prune.py) |
| `images/robotangle/mujoco-3.14.0-epa/` | [`source/images/robotangle/mujoco-3.14.0-epa/`](../../source/images/robotangle/mujoco-3.14.0-epa/), notes in [`docs/simulator/MUJOCO_EPA_PATCH.md`](../simulator/MUJOCO_EPA_PATCH.md) |

## Scripts and records

| cited path | in this repository |
|---|---|
| `scripts/roboarc/harbor_check.py` | [`scripts/roboarc/harbor_check.py`](../../scripts/roboarc/harbor_check.py) |
| `scripts/robotangle/release_rebuild.py` | [`tools/release/release_rebuild.py`](../../tools/release/release_rebuild.py) |
| `scripts/robotangle/release.json` (runtime, `uipc_builds`) | [`scripts/roboarc/release.json`](../../scripts/roboarc/release.json); the Genesis RoboTangle release record is [`records/genesis_release/release.json`](../../records/genesis_release/release.json) |
| `scripts/robotangle/shared/` | [`scripts/roboarc/shared/`](../../scripts/roboarc/shared/) (the one copy of the pinned tooling) |
| `scripts/robotangle/oracles/<base>/` | [`scripts/roboarc/oracles/<base>/`](../../scripts/roboarc/oracles/) (released references); the Genesis `-i00` references the MuJoCo ones were ported from: [`tools/references/oracles_genesis/`](../../tools/references/oracles_genesis/) |
| `scripts/robotangle/{references,redesign,stress}/`, `scripts/robotangle/test_*.py` | [`tools/references/`](../../tools/references/), [`tools/redesign/`](../../tools/redesign/), [`tools/stress/`](../../tools/stress/), [`tools/tests/`](../../tools/tests/) |
| `scripts/robotangle/{mujoco_release.json,package_text/,package_text_tool.py}`, `scripts/topogym_rules.py` | [`tools/release/`](../../tools/release/) |
| `scripts/robotangle/freeze_manifest.*.json` | [`records/freeze_manifests/`](../../records/freeze_manifests/) |
| `scripts/oracle_runners/*_session_oracle.py` | [`tools/references/oracle_runners/`](../../tools/references/oracle_runners/) |
| `scripts/robowits/shared/oracle_replay.py`, `scripts/robowits/oracles/<family>/oracle_qpos.npy` (the standard solution player's header) | the player is [`scripts/roboarc/shared/standard/solution/oracle_replay.py`](../../scripts/roboarc/shared/standard/solution/oracle_replay.py); its rows are each task's `solution/oracle_qpos.npy`, from `scripts/roboarc/oracles/<base>/standard.npz` |
| `scripts/task_dynamics.toml` | not carried; the `dynamics` field is defined in [`docs/PROTOCOL.md`](../PROTOCOL.md) §2.14 |
| `scripts/check_knot_clearance.py`, `scripts/prepare_blanket_initial_states.py`, `scripts/check_*fast_paths.py` and the other one-off checks named in comments | not carried (development checks; their results are quoted where they are cited) |
