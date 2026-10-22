# RoboARC appearance seeds: release runbook

The steps of a seeded release, as RoboARC 1.2.0 was made (first written from Jianwen's seeded release run of
2026-10-20, then used for 1.2.0).

A seeded release gives every base three scenes: seed 0 is the frozen instance as released, seeds 1 and 2 change the
looks only (colours, materials, textures: `topogym.tasks.scene_variation.appearance(cls, S)`, the scene seed stays 0).
Both twins of a base declare `[metadata.seeds] default = 0`, `allowed = [0, 1, 2]`, `build = "runtime"` and, in
`oracle`, seed 0 and the seeds whose Harbor oracle run passed. The runner assigns the seeds without replacement
([RUNNING.md](../RUNNING.md) "Seeds") and runs each trial from a copy made by `scripts/task_seed.py materialize`, which
writes the trial's seed into the task's `seed` files; results are kept per trial (RUNNING.md "Results").

## What a seed S of a base ships

| what | where | written by |
|---|---|---|
| frozen instance | the image, `/opt/robotangle/instances/<base>/seeds/<S>/`; pinned by sha256 in `assets.json`; the HF dataset, `instances/<base>/seeds/<S>/`, once the release's revision is uploaded (step 5) | `tools/release/release_rebuild.py --appearance-seeds` (`tools/release/derive_seed_instance.py`) |
| its pin | `scripts/roboarc/assets.json`: the files, and `instances.<base>.seeds.<S>.instance_sha256` | `release_rebuild.py`, `scripts/roboarc/assets.py pin` |
| its lock and binding | `scripts/roboarc/packages/<base>-privileged/runtime_binding.s<S>.upstream.json`; in the tasks `runtime_binding.s<S>.json` beside every `runtime_binding.json` | `release_rebuild.py`; `scripts/roboarc/mk_tasks.py` |
| privileged reference | `scripts/roboarc/oracles/<base>/privileged.s<S>.npz`; in the task `solution/trajectory.s<S>.npz` | `release_rebuild.py` (re-stamped: the same rows, `instance_sha` = seed S's birth; `tools/release/restamp.py --check` verifies one); `mk_tasks.py` |
| goal pictures (blanket-fold 01-05, knot-untying 07 and 09) | `packages/<base>-<mode>/tests/seeds/<S>/goal/`; in both twins `tests/seeds/<S>/goal/`; sha256 in `assets.json` `instances.<base>.seeds.<S>.goal` | `release_rebuild.py` (renderers in `scripts/roboarc/goal_images/`); `mk_tasks.py` checks them against the pins |

At run time: `stage_instance.py --seed-file` stages the seed's instance against its binding and fails on a missing
instance or binding; the privileged `solution/solve.sh` copies the seed's reference (a reference without a birth stamp,
rope-threading 04-08, serves every seed as it is); the standard oracle streams the same rows for every seed;
`task_seed.py materialize` copies the seed's goal pictures over `environment/goal/` (both modes: a standard agent image
takes nothing from the simulator's image, PROTOCOL H.1, H.3). `ci_checks/check-seed-files.py` and `check-twins.py`
check the files of every seed.

## How a seed is made

Seeds are derived, never made again: `tools/release/derive_seed_instance.py` turns a base's seed-0 instance into
`seeds/<S>/` with `appearance(cls, S)`'s task parameters, `instance_assets/` byte for byte (the frozen MuJoCo scene, the
birth and settled-state archives, the frozen generated assets) and the build's fields (`state_sha`, `asset_identity`,
...) recomputed in the image without instances. It stops unless the physical fields equal seed 0's and the birth's tick
record differs from seed 0's in `task.instance_id` only. Premade bases (render-host tiers, redesigns, blanket births,
orbits-02) get their seeds the same way.

## Commands

1. Image without instances, from this repository's recipe (`source/images/roboarc/Dockerfile`, `images/roboarc/build.sh
   --noinst`).
2. The release run (render tree, instances, seeds, image, files):

   ```bash
   tools/release/release_rebuild.sh <head> --onto <head> --bench roboarc --version <v> --appearance-seeds 1,2 \
     --work <work> --gpu 0 --no-harbor [--instances <dir>]
   ```

   It makes or copies every base's instance, derives `instances/<base>/seeds/{1,2}/` (`<work>/logs/derive_<base>.log`),
   renders the seeds' goal pictures in the release image (or takes them from `--goal-pictures <dir>`), re-stamps the
   references, builds the release image, writes its deterministic archive (`images/archive.sh`) and its pin in
   `images/pins.json`, and writes `assets.json`, `packages/`, `oracles/`, `release.json` and the task dirs on the render
   tree. Look at every goal picture before going on.
3. Checks: `python3 scripts/roboarc/mk_tasks.py --check`, `--assets <work> --check` (each seed's binding against its
   instance bytes), `ci_checks/check-seed-files.py`, `check-twins.py`, `check-seeds.py`, the unit tests.
4. The release's evidence, per seed (owner decision D4): a strict double fresh replay of the seed's reference; the
   world and robot parts of the birth and end states bit-identical to seed 0's; the Session's strict instance checks;
   Harbor oracle 1 / nop 0 for seeds 1 and 2 in both modes on at least one base per family and every new base (on
   `task_seed.py materialize` copies, `scripts/roboarc/harbor_check.py --tasks`). Seed 0 of every released base stays
   byte-identical (births, ends, rendered frames); the blanket families' identity rule is checked on all 14 blanket
   bases. A seed enters `oracle` only with a passing Harbor oracle run.
5. The HF dataset: one new revision of `eai-v2-internal/robotangle-assets` for the release, uploaded only after step 4
   passes and with the owner's approval (`instances/<base>/seeds/<S>/` and the new bases' instances), then
   `scripts/roboarc/assets.py pin <copy> <revision>` and a rebuild that fetches from it. Until that upload the seed
   instances ship in the image only (1.2.0: the release's instance tree, pinned file by file in `assets.json`), and
   `images/roboarc/build.sh --instances <tree>` builds from such a tree.
6. The image is pushed by the owner (`images/push_all.sh`); the tasks name it by the digest in `images/pins.json`.

The nine bases added with 1.2.0 (bed-bolt-01..05, caliper-sorting-01, clock-assembling-01, battery-replacement-01,
interlocked-claws-13) are prepared with the release but stay out of `tasks/` until the owner has reviewed their demos
(`python3 scripts/roboarc/mk_tasks.py --experimental --out <dir>` renders them).
