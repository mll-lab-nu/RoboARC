# RoboARC appearance seeds: release runbook

A seeded release gives every base three scenes: seed 0 is the frozen instance as released, seeds 1 and 2 change the
looks only (colours, materials, textures; `topogym.tasks.scene_variation.appearance(cls, S)`, scene seed 0). Both twins
of a base declare `[metadata.seeds] default = 0`, `allowed = oracle = [0, 1, 2]`, `build = "runtime"`; the runner draws
the seeds (`scripts/run_agent_batch.sh`, `scripts/task_seed.py assign`) and runs a copy made by
`scripts/task_seed.py materialize`, which carries a `seed` file. Decisions: owner, 2026-10-20.

## What a seed S of a base ships

| what | where | written by |
|---|---|---|
| frozen instance | HF dataset `instances/<base>/seeds/<S>/`; in the image `/opt/robotangle/instances/<base>/seeds/<S>/` | `release_rebuild.py --appearance-seeds` |
| its pin | `scripts/roboarc/assets.json`: `instances[<base>].seeds[<S>].instance_sha256` and the files | `release_rebuild.py`, `assets.py pin` |
| its lock and binding | `packages/<base>-privileged/runtime_binding.s<S>.upstream.json`; tasks: `runtime_binding.s<S>.json` beside every `runtime_binding.json` | `release_rebuild.py`; `mk_tasks.py` |
| privileged reference | `oracles/<base>/privileged.s<S>.npz`; task: `solution/trajectory.s<S>.npz` | `release_rebuild.py` (`restamp`: the same rows, `instance_sha` = seed S's birth); `mk_tasks.py` |
| goal pictures (blanket-fold 01-05, knot-untying 07 and 09) | `packages/<base>-<mode>/tests/seeds/<S>/goal/`; both twins: `tests/seeds/<S>/goal/`; sha256 in `assets.json` `instances[<base>].seeds[<S>].goal` | `release_rebuild.py` (renderers `scripts/roboarc/goal_images/`); `mk_tasks.py` checks the sha256 |

At run time the image's `stage_instance.py --seed-file` stages the seed's instance with its binding; the privileged
`solution/solve.sh` copies the reference whose stamp is the staged instance's `state_sha` (`/app/instance.json`); the
standard oracle streams the same rows for every seed; `task_seed.py materialize` copies the seed's goal pictures over
`environment/goal/` (both modes: a standard agent image takes nothing from the simulator's image, PROTOCOL H.1, H.3).

## Prerequisites

- A topogym checkout (dev/robotangle_mujoco) with the branch `jianwen/appearance-seeds-release`: `appearance()` and the
  families' appearance axes, `release_rebuild.py --appearance-seeds` / `--goal-pictures`, and the MuJoCo image's
  `stage_instance.py --seed-file` (`images/robotangle/Dockerfile.mujoco` installs it; the Genesis 0.2.22 base image's
  copy has no `--seed-file`).
- `source/` of this repository synced from that checkout: done on `jianwen/seeds-release` (TopoGym `622c7801c6`,
  PROVENANCE.md). Its `source/scripts/robotangle/release_rebuild.py` has both this repository's V2-release additions
  (history rule, `release_evidence.json`, `simulator_overrides` of new bases, the embodied-first-exam tags) and the seed
  steps, so step 1 runs from this repository (the head and `--onto` are commits of it).
- Bases this repository cannot make: the knot_untying and rope_threading tiers (`render_host`) read the released
  Genesis binding `tasks/robotangle-<old>-privileged/environment/runtime_binding.json`, which `source/` does not carry;
  a premade base (`--instances` / the manifest) is copied as it is, so it must bring its own `seeds/<S>/`. Pass those
  as premade instances with their seeds, or run step 1 from a topogym checkout that carries this `release_rebuild.py`.
- A docker host with Harbor; one NVIDIA GPU with RT cores (Nyx) for the goal pictures and the GPU-profile bases.

## Commands

1. Make the seed instances and write the release files (from this repository):

   ```bash
   source/scripts/robotangle/release_rebuild.sh <head> --onto <head> --bench roboarc --version <v> \
     --appearance-seeds 1,2 --work <work> --gpu 0 --no-harbor [--instances <dir>]
   ```

   It makes every base's instance and `instances/<base>/seeds/{1,2}/` in the image without instances, builds the
   release image with them, and writes on the render tree (`<work>/render`, local branch `release-render/...`):
   `assets.json`, `packages/`, `oracles/`, `release.json`, the task dirs. Steps 2 and 3 run inside it.

2. Goal pictures. Bases whose packages show some (`environment/goal/`) get each seed's, rendered in the release image
   on `--gpu` into `<work>/goal/<base>/seeds/<S>/` (`GOAL_RENDERERS`: `blanket_fold` `fold_target_images.py`,
   `knot_untying` `region_images.py`). Pictures rendered elsewhere: `--goal-pictures <dir>` with
   `<dir>/<base>/seeds/<S>/` holding the pictures and the renderer's record (`target_images.json` /
   `region_images.json`). The run stops if a picture is missing or its record names another birth than the seed's
   instance, and prints the render command for that seed. Look at every picture before step 4.

3. Restamp and verify. `oracles/<base>/privileged.s<S>.npz` is the reference with `instance_sha.npy` replaced, every
   other entry byte for byte. The run's checks include `mk_tasks.py --assets <work> --check`: each seed's binding
   against its instance bytes, each re-stamped reference's birth against that instance's `state_sha`. REPORT.md lists
   per seed the birth, the instance sha256, the re-stamped reference and the pictures.

4. Upload and pin (the HF upload needs the owner's OK):

   ```bash
   hf upload eai-v2-internal/robotangle-assets <work>/instances instances --repo-type dataset \
     --commit-message "RoboARC appearance seeds (<head9>)"
   source/scripts/robotangle/release_rebuild.sh <head> --onto <head> --bench roboarc --version <v> \
     --appearance-seeds 1,2 --work <work> --hf-revision <dataset commit> --no-harbor
   ```

   The second run uses the uploaded instances and the pictures in `<work>/goal/` again and pins the revision. To pin
   from a copy of the dataset instead: `python3 scripts/roboarc/assets.py pin <copy> <rev>` (it keeps every seed and,
   while a seed's instance is unchanged, its goal pins), then `assets.py check <copy>`. Fetch the render branch into
   this repository.

5. Build the image (after the `source/` sync; `images/roboarc/VERSION` = `<v>`):

   ```bash
   images/roboarc/build.sh
   docker run --rm ghcr.io/embodied-first-exam/roboarc:<v> python /opt/robotangle/stage_instance.py 2>&1 | grep -- --seed-file
   docker run --rm ghcr.io/embodied-first-exam/roboarc:<v> ls /opt/robotangle/instances/<base>/seeds
   ```

6. Render and check the tasks (python >= 3.11):

   ```bash
   python3 scripts/roboarc/mk_tasks.py
   python3 scripts/roboarc/mk_tasks.py --check
   python3 scripts/roboarc/mk_tasks.py --assets <dataset copy> --check
   python3 ci_checks/check-seeds.py && python3 ci_checks/check-twins.py && python3 ci_checks/check-protocol.py
   RCB_CI_JOBS=1 bash ci_checks/check-unit-tests.sh
   ```

7. Evidence runs. Seed 0: the Harbor step of `release_rebuild.py` (without `--no-harbor`) on the task dirs as committed.
   Seeds 1 and 2: Harbor oracle and nop, both modes, on materialized copies:

   ```bash
   for S in 1 2; do
     for t in tasks/roboarc-*; do python3 scripts/task_seed.py materialize "$t" --seed $S --out <ev>/s$S; done
     python3 <topogym>/scripts/robotangle/harbor_check.py --tasks <ev>/s$S --image ghcr.io/embodied-first-exam/roboarc:<v> \
       --run <ev>/run_s$S --bench roboarc --gpu-bases <gpu bases> <bases>
   done
   ```

   Expected: oracle 1 and nop 0 per base, mode and seed (PROTOCOL: the reference scores 1 on every seed in
   `seeds.oracle`, which `mk_tasks.py` sets to all three). Whether a seed without a passing run is left out of
   `oracle` instead is the open decision D4.
