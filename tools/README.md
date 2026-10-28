# tools/: how the release was made

Maintainer tooling around the simulator: none of it is graded, copied into a task, or built into the image. The
graded tooling is `scripts/roboarc/shared/` (pinned by sha256), the task generator `scripts/roboarc/mk_tasks.py`, the
image recipe `source/images/roboarc/Dockerfile`.

| folder | what it holds |
|---|---|
| `release/` | the release rebuild (`release_rebuild.py` / `.sh`: instances made in the image without instances, the release image, the release files, the task dirs, the checks and the Harbor oracle / nop runs) and its inputs: `mujoco_release.json` (the MuJoCo bases and how their instances are made), `release_evidence.json`, `standard_speed_rule.json`, `topogym_rules.py` (grader-detail sentences), `freeze_manifest.example.json`; `render_reference_video.py`, `source_hash.py` (SOURCE.json from a checkout); `derive_seed_instance.py` (appearance seeds), `check_seed_physics.py` (the seeds' end-state identity, the seeds' Harbor trial list), `refresh_instance.py` (a frozen instance whose task identity changed but whose physics did not, rebuilt with its assets byte for byte), `restamp.py` (a reference's birth stamp) |
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
| `test_visual_assets.py` | numpy | anywhere (the image's texture downloads cover every map `look.py` names) |
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

### A release that redoes the appearance seeds

What `release_rebuild.py` does for a source that changes every seed's looks (`topogym/world/look.py`, the families'
`appearance()`) and the frozen scene of some bases (the goal marker 5 mm wide and glowing: the 30 bases whose scene
draws it, seed 0 included; their births unchanged):

- Seed 0: a base is made in the image without instances from the head's source (`MAKE_ONE`, `rcb_topogym.py
  make-instance`) unless `--instances` or the manifest brings it premade; a premade instance is copied as it is. Nine
  bases cannot be made there, since their make reads the released instance, which that image does not carry:
  fiber-bundling 02, 04, 05 and blanket-flatten 07, 08 (`released_simulator_keys`), knot-untying 02, 04, 05
  (`render_host`, `frozen_assets`); none of them draws the marker. Built on the released instance tree, name the
  changed bases with `--remake` (manifest `remake`), so their seed 0 is made again: `--instances <released tree>
  --remake <the 30 marker bases>`. Nothing in the run requires seed 0's `instance_assets/` to stay the released bytes
  (a premade base keeps its lock only when it is the same instance); the seeds runbook's "seed 0 stays byte-identical"
  (`docs/benchmarks/roboarc_seeds_release.md`, step 4) holds for these bases' births and ends, not their scene files or
  frames.
- Seeds 1, 2 (`--appearance-seeds 1,2`): derived in the same image with the head's `appearance()`
  (`derive_seed_instance.py`; physics, birth and `instance_assets/` must equal seed 0's). A premade base's own
  `seeds/<S>/` are kept only when derived from that seed 0 on this run's source, so the released tree's seeds (the old
  looks) and seeds of a re-made seed 0 are derived again.
- Goal pictures: blanket-fold 01-05 and knot-untying 07, 09 get each seed's pictures rendered from the seed's instance in
  the release image, which must hold the textures (`source/images/roboarc/prepare_nyx_studio.py`); knot's painted line
  is 5 mm wide (`scripts/roboarc/goal_images/region_images.py`). Seed 0's pictures are the packages' own and are not
  rendered unless named with `--seed0-goal-pictures` (manifest `seed0_goal_pictures`): knot-untying 07 and 09 need it,
  since their texts now name the 5 mm line; blanket-fold draws no marker and its seed 0 looks unchanged.
- Not done by the run: the bases' records are written anew, without `seeds` (the seeds' Harbor results), so
  `[metadata.seeds] oracle` is `[0]` until the seeds' Harbor trials pass and `seeds.<S>.harbor = "passed"` is recorded in
  `release.json` (by hand, as for 1.2.0 - 2.1.0; then `mk_tasks.py` again); the Change Log entries are written into
  `release.json` `changes` by hand (kept by the run).
- The evidence per base (`check_seed_physics.py run`, in the image): the reference replayed strictly in two fresh
  processes per seed; every seed succeeds and ends as seed 0 does but for the task's instance id. The Harbor seed trials
  (`check_seed_physics.py harbor-list --changed <bases or families>`): one base per family and every changed base,
  seeds 0-2, both modes, oracle and nop, on `scripts/task_seed.py materialize` copies.
