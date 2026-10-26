# tools/: how the release was made

Maintainer tooling around the simulator: none of it is graded, copied into a task, or built into the image. The
graded tooling is `scripts/roboarc/shared/` (pinned by sha256), the task generator `scripts/roboarc/mk_tasks.py`, the
image recipe `source/images/roboarc/Dockerfile`.

| folder | what it holds |
|---|---|
| `release/` | the release rebuild (`release_rebuild.py` / `.sh`: instances made in the image without instances, the release image, the release files, the task dirs, the checks and the Harbor oracle / nop runs) and its inputs: `mujoco_release.json` (the MuJoCo bases and how their instances are made), `release_evidence.json`, `standard_speed_rule.json`, `topogym_rules.py` (grader-detail sentences), `freeze_manifest.example.json`; `render_reference_video.py`, `source_hash.py` (SOURCE.json from a checkout); `derive_seed_instance.py` (appearance seeds), `refresh_instance.py` (a frozen instance whose task identity changed but whose physics did not, rebuilt with its assets byte for byte), `restamp.py` (a reference's birth stamp) |
| `references/` | per family, how the released references were made or re-recorded on MuJoCo (`make_ref.py`, `rerecord.py`, run-oracle code, probes); `oracle_runners/` (the session oracles of orbits, slotted_post_ring and lock_picking). Several re-recordings started from the Genesis-era `-i00` references, which are in RoboARC 1.0.0 (`tools/references/oracles_genesis/` at main 153eb8cf01) |
| `redesign/` | the generators of the redesigned (rd) tiers: `*/export_runtime.py` writes the variation resources under `source/topogym/assets/resources/variations/`; births, frozen development instances, the anti-hack metrics, `session_kit.py` |
| `stress/` | the physics acceptance rounds (`round.py`, `search.py`, `replay.py`, `verdict.py`, `observer.py`) and their regression `corpus/` ([docs/simulator/STRESS_HOWTO.md](../docs/simulator/STRESS_HOWTO.md)) |
| `tests/` | unit and acceptance tests of the simulator and tooling |

The Harbor oracle / nop checker is `scripts/roboarc/harbor_check.py` (the path the task READMEs cite).

## Running

From the repository root, with the simulator source and the pinned tooling on the path:

    export PYTHONPATH=$PWD/source:$PWD/scripts/roboarc/shared

The tools need the image's Python packages (MuJoCo 3.14.0 with the EPA patch, newton / warp-lang, numpy, scipy,
...): run them in the release image with the checkout mounted, for example

    docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v $PWD:/repo:ro -w /repo \
      ghcr.io/embodied-first-exam/roboarc:1.0.0 python tools/tests/test_rope_energy.py

The installed simulator in the image (`/opt/topogym`) is the same source as `source/topogym` minus `planning/`; the
reference tools import `topogym.planning`, so `source/` must come first on the path. The stress tools do that
themselves (`tools/stress/_env.py`; `STRESS_USE_INSTALLED=1` uses the installed one).

## Tests

| test | needs | run |
|---|---|---|
| `test_nyx_startup.py` | numpy | anywhere (`python3 tools/tests/test_nyx_startup.py`) |
| `test_exposure_rule.py` | numpy | anywhere |
| `test_flex_energy.py`, `test_rope_energy.py` | MuJoCo 3.14.0 | in the release image (CPU) |
| `test_warp_cache_race.py` | warp-lang | in the release image (CPU) |
| `test_render_bridge.py selftest --out <dir>` | numpy, MuJoCo | in the release image (CPU); its other stages need Genesis (RoboARC 1.0.0's image) |
| `test_nyx_direct.py`, `test_guard_replay.py` | a GPU and Nyx and a frozen instance (`test_nyx_direct.py`'s `host` stage, the Genesis render host it was accepted against, needs RoboARC 1.0.0's image) | by hand (see each file's usage) |

## Release rebuild

    tools/release/release_rebuild.sh <head> --version <v> --onto <commit> [--instances <dir>] [--manifest <file>]

`<head>` and `--onto` are commits of this repository. The run needs Docker, harbor and Python >= 3.11; the HF upload of
new instances, the image push and the merge are separate steps the owner approves. The freeze manifests of the
released versions are in `records/freeze_manifests/`. A base the render tree has no packages for (a new base) gets
its template's final packages under its own name. A family whose instances port a Genesis render host (`render_host`
in `mujoco_release.json`) needs the released Genesis task's binding, which this repository does not carry: pass such
instances premade (`--instances`).
