# Push notes: dev/jianwen, 2026-10-21 (jianwen)

This push brings our appearance-seed work and the nine new bases onto `dev/jianwen`, finished parts and unfinished
ones, for the owner's integration. The integration covers the rebase onto 1.1.0 (dev/qineng), the official 1.2.0
image, the seed derivation, validation, the HF upload and the release.

**What does not ship:**
- our local image `ghcr.io/embodied-first-exam/roboarc:1.2.0` (`sha256:301adcb64b06`), renamed to
  `roboarc:1.2.0-jianwen-pre` and `-jianwen-pre-noinst` on our hosts;
- the 220 task directories rendered against it;
- the local helper branch `helper/roboarc-1.2.0`.

All of these are reference material only.

Marks used below:
- **WIP**: unfinished.
- **PRE-RELEASE**: reference only, not for shipping.
- No mark: done and meant to stay.

## 1. Commits f5a3d6f..HEAD

`dev/jianwen` was `f5a3d6f` before 2026-10-20. The pushes of 2026-10-21 come in three batches, each a fast-forward of
the one before:
- batch 1, `d6f2f23`: 19 commits;
- batch 2, `ae3be16`: 5 commits;
- batch 3: the commits after `ae3be16`.

| commit | what | mark |
|---|---|---|
| 9cdee7c | Runner: K seeds assigned without replacement (`scripts/task_seed.py assign`), one Harbor job per task, mode and seed (`<batch>-<mode>-<task>-s<S>`, later passes `-s<S>-p<n>`), `seed.json` per job, results per seed slot (`write_task_results.py`) | **WIP** (D2, section 2) |
| 302e894 | `stage_instance.py --seed-file`: a seed's instance from the image by the runner's seed file | |
| 4998035 | goal pictures per appearance seed (blanket-fold 01-05, knot-untying 07 / 09) | |
| bc15aa9 | seed goal pictures through `task_seed.py materialize` in both modes, not the image | |
| c5d91e3 | one privileged reference per appearance seed, picked by `solve.sh` (D3 (a)) | |
| 33f018c | docs: the seeds release runbook | its step 4 was replaced in def11e0 |
| b1a104c | merge of the above (jianwen/seeds-integrated) | |
| 7deaad8 | `source/`: TopoGym 622c7801c6 (the seeds code, render fixes) | |
| 616482a | VERSION 1.1.0, task dirs re-rendered for that tag | **PRE-RELEASE**, superseded |
| 10ae927 | PROVENANCE and runbook for the 622c7801c6 sync | |
| 6b99a0c | `release_rebuild --appearance-seeds` derives seeds from the base's instance (`derive_seed_instance.py`) | |
| 715ecdd | `source/`: TopoGym 738231f5cb (dev 9115da07d8: clock v3, battery rule C and the Xbox shell, merged with the seeds) | |
| 7ba658d | VERSION / README / CITATION 1.2.0 | **PRE-RELEASE** |
| fc78c76 | a reference without a birth stamp (rope-threading 04-08) serves every seed as it is | |
| 593536b | `release_rebuild`: `render_bridge_sha256` / `nyx_direct_sha256` follow the tooling | |
| 4395f81 | mk_tasks: Cast Trinity's pairwise separation text. Seed tests take the release's own seed pins | the `test_runner_seeds.py` part is **WIP** (D2) |
| 25e5a2a | release rebuild render: the 220 task dirs, `release.json`, `assets.json` and `packages/` locks, all bound to our local instances and image | **PRE-RELEASE** |
| def11e0 | docs: README counts (110 bases, 18 families), no-upload notes (PROVENANCE, image README, runbook) | |
| d6f2f23 | `release.json` evidence of the new bases (Harbor on our local image) and the CHANGELOG 1.2.0 entry | **PRE-RELEASE** |
| 94d77d1 | `source/`: TopoGym dev 3eed5d8b33 (battery 2b, caliper 2b, F4) | **WIP**: battery and caliper instances stale (section 5) |
| 808edcc | the nine new bases' texts after that sync (battery's pry stick without colours, F4 paths), 18 task dirs re-rendered | **WIP**: battery and caliper bindings stale |
| 03925b2 | `source/ci_checks/`: TopoGym's `test-appearance-seeds.py` and `test-release-seeds.py` | |
| 24ed8e2 | these notes and `docs/handover/evidence_20261021/` | |
| ae3be16 | the migration session's Chinese handoff note (draft), `docs/benchmarks/roboarc_seeds_handoff_20261021.md` | |
| f53d5f2 | `source/`: TopoGym dev a849170272 (the colour-table fixes: caliper, battery, Cast Trinity, rope_threading seeds 1 and 2; the claws 06-10 texts) | **WIP**: more seed instances stale (section 5) |
| 23c69e3 | interlocked-claws 06-10: the second piece is "a lock", no longer "teal" (packages, `release.json` designs, 10 task dirs) | |
| (batch 3, docs) | the colour table in `evidence_20261021/`, these notes updated, the handoff note's final text | |

Batch 3 carries the two top-ups that were outstanding:
- the migration session's base x seed colour table;
- caliper's seed 1 / 2 bin colours (TopoGym 28f8f4da8f, in the a849170272 sync).

## 2. D2: the runner's result layout does not match the owner's choice

What our runner does (9cdee7c, tested by `scripts/tests/test_runner_seeds.py`):
- Each (task, mode, seed) is its own Harbor job, with `<job>/seed.json`.
- The per-task results go to `<RESULTS>/<batch>/<task>/<run id>/s<S>/`, a later pass to `s<S>-p<n>/`. Each has its own
  `config.json` carrying the seed, plus a task README with the success over the seeds.

The owner chose per-episode result directories, with the seed recorded in `config.json`. Our seed-slot layer (`s<S>`)
is therefore WIP: `run_agent_batch.sh`, `write_task_results.py` and `test_runner_seeds.py` need the per-episode
layout. The assignment itself (D1: without replacement, keyed by batch and base, both modes the same seeds) matches.

## 3. Source delta since e3ea480

e3ea480 is the sync of TopoGym 497a9e9d7f that the owner's physics review covered. Four syncs followed:
- 7deaad8: TopoGym 622c7801c6;
- 715ecdd: 738231f5cb;
- 94d77d1: 3eed5d8b33;
- f53d5f2: a849170272.

`source/topogym/` outside `planning/` is byte-identical to TopoGym a849170272. Changes:

| change | TopoGym | in RoboARC | physics |
|---|---|---|---|
| the appearance seeds code: `scene_variation.appearance()`, the families' appearance axes, render fixes so a look reaches the file Nyx draws | 622c7801c6 | 7deaad8 | none (seed 0 writes the recorded files byte for byte; `test-appearance-seeds`) |
| clock_assembling v3: drawn realistically, the metric review's rules, the bridge solid to the wheels, new -i01 reference | 9115da07d8 | 715ecdd | clock only (new birth ad4ee5041eeb19de, end 9683cab251ece2c0) |
| battery_replacement rule C counted from the compartment, with the shared `WeakHoldGraspTarget` travel budget (M8, 3307d319a1, opt-in; only battery declares it) and the Xbox controller outer body | 9115da07d8 | 715ecdd | battery only (birth 764192d9a7e49a0f, end de1cb2ee33e369ed) |
| caliper_sorting rules v2 (labels legible in cam_high) | before 497a9e9d7f | already in e3ea480 | caliper only |
| bed_bolt drawn more realistically (visual geoms only) | 497a9e9d7f | already in e3ea480 | none |
| bed_bolt and Cast Trinity text fixes | before 497a9e9d7f | already in e3ea480 | none |
| battery 2b: props drawn realistically, on the appearance axes | f6471bc717 | 94d77d1 | none (birth and end unchanged; files seed 0 declares change, section 5) |
| caliper 2b: drawn like the real things, `fixture_finish` axis, `bin_finish` default `grey_blue` | e18370887a / 3eed5d8b33 | 94d77d1 | none (GLB textures change, section 5) |
| F4: the nine new bases' package texts cite `docs/PROTOCOL.md` / `docs/STANDARD_MODE.md` | 9e36ecf429 | 94d77d1, 808edcc | texts only |
| the colour-table fixes, seeds 1 and 2 only: caliper's bins dark red / dark green (28f8f4da8f); battery's pry stick always nylon, seed 1 a white controller, seed 2 Pulse Red with its white back up (b995fe32a0); Cast Trinity's third piece its own colour (`third_rgb`) and rope_threading's blue / black ropes (`scene_visuals.py` `blue_rope`, `black_rope`) (a849170272) | 28f8f4da8f, b995fe32a0, a849170272 | f53d5f2 | none (seed 0 declares the same files as on 3eed5d8b33: battery, caliper, Cast Trinity, rope_threading easy and rd very_easy checked here, every base per the migration session; `test-appearance-seeds` 6 OK from this tree) |
| interlocked-claws 06-10 texts: the second piece is "a lock", no longer "teal" (seeds 1 and 2 paint it blue and green) | 02135c5161 | f53d5f2, 23c69e3 | texts only |

The rows marked "already in e3ea480" are listed as asked. They were part of TopoGym 497a9e9d7f, which e3ea480 synced.

**Neutrality** (the evidence that the 101 released bases' seed-0 proof carries over): 14 contact-weld bases, one per
family, were made and strictly replayed on TopoGym 497a9e9d7f and on 9115da07d8. All 14 had the same birth (= the
mujoco_release.json reference's) and the same end state, with success on both:
- slotted-post-ring-01, knot-untying-03, interlocked-claws-03, cast-coil-01, hedgehog-in-cage-02, rope-threading-01,
  orbits-01;
- fiber-bundling-02, lock-picking-01, cast-keyhole-01, horseshoe-ring-02, buzz-wire-3d-04, bed-bolt-01,
  caliper-sorting-01.

Sources: `evidence_20261021/neutrality_9115da07d8_vs_497a9e9d7f_compare2.txt`, and TopoGym
`docs/rcb/handover/jianwen_push_20261019.md` M8. The 3eed5d8b33 changes touch only battery and caliper geometry, their
defaults and one comment in `scene_variation.py`. The a849170272 changes touch appearance values of seeds 1 and 2
and the claws 06-10 texts only.

The tooling copies in every task change with the sync, but `tooling_version` stays "19":
- `rcb_topogym.py`: the new families registered; a settled state exported at make-instance only for the chain families;
- `nyx_direct.py`: a fixed MJCF entity's visuals drawn at its static bodies.

## 4. Evidence we have (`docs/handover/evidence_20261021/`)

All our hosts are AMD Ryzen 9 7950X (Zen 4).

| file | what |
|---|---|
| `harbor.md` | Harbor on our local image, rendering off. The nine new bases, both modes, oracle and nop: 36 / 36 (oracle 1 at the reference's end, deterministic; nop 0). Seed copies (`task_seed.py materialize`): claws-03 and hedgehog-02 at seeds 1 and 2, both modes, oracle and nop, 16 / 16; blanket-fold-01 at seeds 1 and 2, 8 / 8. Plus rope-threading-04 seed 1 (unstamped reference), privileged oracle 1 |
| `derive_summary_110bases_20261020.json` | 220 derived seeds (110 bases x seeds 1, 2), each with three gates: physical fields equal seed 0's, the birth record differs in `task.instance_id` only, world hash equal |
| `derive_summary_battery_clock_20261020.json` | battery and clock remade on TopoGym 738231f5cb (births = the references'), their seeds derived again, the same gates |
| `replays_23bases_seeds01_20261020.json` | sampled strict replays: 23 bases (one per family and the nine new ones) at seeds 0 and 1, 46 / 46 success at the strict goal, one end world hash per base |
| `replays_battery_clock_seeds01_20261020.json` | battery and clock again after their remake, seeds 0 and 1: 4 / 4, seed 0 at the references' ends |
| `stagecheck_release_image_9.txt` | in our release image, `stage_instance.py --seed-file` with the task bindings, then a strict Session: battery-replacement-01, horseshoe-ring-01, blanket-fold-01 at seeds 0, 1, 2: 9 / 9 instance_exact and assets_exact |
| `release_rebuild_report_seeds_and_checks.md` | REPORT.md's seed table (per base and seed: birth, instance sha256, task parameters, re-stamped reference, goal pictures) and its checks |
| `seed0_declared_files_before_after_2b.md` | the cheap check of section 5 |
| `tests_20261021.md` | `test-appearance-seeds` on TopoGym 3eed5d8b33 (6 OK); `test_runner_seeds`, `test_seed_staging` and `test_seed_goal_images` on this tree |
| `neutrality_9115da07d8_vs_497a9e9d7f_compare2.txt` | section 3 |
| `colour_table.md`, `colour_table.csv` | the migration session's base x seed colour table (TopoGym a849170272): for every colour or material word a task text names, the part's colour at seeds 0, 1 and 2 and a verdict. 79 of the 110 bases name one (35 rows). It found the claws 06-10 "teal lock", which led to the text change. Host paths replaced |

The blanket families' GPU cloth needs a CUDA 13 driver (>= 580). On a 570 driver the verifier stops with "Warp sees no
CUDA device": three host-A rows in `harbor.md` show this, and those trials were rerun on host B. Fake-harbor runner run
on the helper tree (K = 3): three trials on three different seeds, the same seeds in both modes, seed files and goal
pictures per seed in each copy, results per seed slot.

## 5. Stale after the 3eed5d8b33 and a849170272 syncs (WIP, not re-derived)

These seed instances must be derived again (your integration). The files concerned are:
- the frozen instances;
- their `assets.json` pins and `packages/<base>-privileged/runtime_binding*.upstream.json` locks;
- the task dirs' runtime bindings and the seeds' re-stamped references.

| base | what must be redone | why |
|---|---|---|
| battery-replacement-01-i01 | seed 0 made again, then seeds 1 and 2 derived | 2b changes the files seed 0 declares (`controller.xml`, `stick.xml`): its `asset_identity` no longer matches; b995fe32a0 changes the seeds' colours |
| caliper-sorting-01-i01 | seed 0 made again, then seeds 1 and 2 derived | 2b changes 11 GLB textures seed 0 declares (four bins, the stand, four sleeves, the rack); 28f8f4da8f changes the seeds' bins |
| interlocked-claws-13-i01 | seeds 1 and 2 | `third_rgb` (a849170272) |
| rope-threading-01..08-i01 | seeds 1 and 2 | blue and black ropes (a849170272) |

Evidence: `evidence_20261021/seed0_declared_files_before_after_2b.md`, from `TaskDefinition.declare()` at seed 0 on
738231f5cb and on 3eed5d8b33. With the synced source, a strict Session would find battery's and caliper's
`asset_identity` changed. Births (`state_sha`) and the references' rows are unchanged per TopoGym's checks. Seed 0
declares the same files on a849170272 as on 3eed5d8b33: checked here for battery, caliper, Cast Trinity and
rope_threading (easy, rd very_easy), and reported by the migration session for every base. Per its checks, the
other twelve claws bases and slotted-post-ring are unchanged at every seed.

Text change in five released bases: interlocked-claws-06..10 call the second piece "a lock", no longer "teal", in both
modes. Their instances, references and grading do not change. Their task dirs are re-rendered (23c69e3); please
review.

## 6. Evidence that is missing (yours to do)

- replays on Intel: every host of ours is a Zen 4 7950X;
- all four checks for every derived seed (D4), including the strict double replay and the strict scene check. We ran
  the three derivation gates on all 220, strict staging on 9 and replays on 46 + 4;
- review of the base x seed colour table, delivered as `evidence_20261021/colour_table.{md,csv}`: especially the
  claws 06-10 text change, and fiber-bundling 05-10's "the red fiber" (16 fibres in 8 cycled colours, so two red ones
  at every seed; your call);
- Harbor at seed 0 for the 101 released bases with the new tooling copy.

## 7. The owner's decisions, as recorded here

- D1: seeds assigned without replacement.
- D2: `seed.json` plus per-episode result directories, with the seed in `config.json` (our layout differs: section 2).
- D3: (a), one re-stamped privileged oracle per seed.
- D4: all four checks per seed.
- The nine new bases stay experimental until the owner has watched their demos.
- D11: cast-coil-06 stays at 0.047 s per control step. This tree says 0.047 in both modes.

## 8. HF

Nothing was uploaded, to the organisation dataset `eai-v2-internal/robotangle-assets` or anywhere else, and no
`assets.py pin` ran.
- `scripts/roboarc/assets.json` keeps revision `9f34f894a454`. 1.0.0's 21477 pins are unchanged.
- The seed instances (43448 files) and the new bases' instances (262 files) are pinned by sha256 only.
- No instance files are in this push.

## 9. The nine new bases: what the push carries, what stays local

In the push:
- **references**: the -i01 references (`source/scripts/robotangle/oracles/<base>/`, `scripts/roboarc/oracles/<base>/`),
  with the re-stamped seed copies under `scripts/roboarc/oracles/` (stale for battery, caliper, Cast Trinity and rope-threading: section 5);
- **recording tools**: `source/scripts/robotangle/references/{bed_bolt,caliper_sorting,clock_assembling,battery_replacement}/`
  and `references/interlocked_claws/cast_trinity/`. Every TopoGym file at 3eed5d8b33 whose path names these families
  is in `source/`;
- **docs and notes**: the family docs (`docs/families/{bed_bolt,caliper_sorting,clock_assembling,battery_replacement,interlocked_claws_cast_trinity}.md`),
  the `mujoco_release.json` notes;
- **tests**: `test_clock_rules.py`, `test_clock_assembling.py`, `test_caliper_sorting.py`, `test_mujoco_grasp_opt_in.py`.

Only in `runs/` of our TopoGym checkout, not pushed:
- `runs/bed_bolt_design_20261012/`;
- `runs/caliper_sorting_design_20261014/`;
- `runs/watch_assembly_design_20261014/` (clock: design, REALISM_LIST, GRASP_AUDIT, harbor_test, out/);
- `runs/controller_battery_design_20261014/` (port/, realism2/, xbox_visual/, harbor_test/, scripts/restamp.py);
- `runs/cast_trinity_design_20261011/` (stress_r1);
- `runs/robotangle_mujoco/bed_bolt/`, `runs/robotangle_mujoco/metric_review_20261019/`,
  `runs/robotangle_mujoco/trinity_demo_20261017/`;
- the acceptance-round outputs on the lab hosts (`<host work dir>/bat_stress/` and the like).

The family docs cite these paths for the work notes. The demos the owner wants to watch are also in them.

## 10. How to run the two seed tests (from this tree)

```bash
cd source && python3 ci_checks/test-release-seeds.py        # Python >= 3.11, standard library only; 10 tests, < 1 s
cd source && python ci_checks/test-appearance-seeds.py      # topogym's dependencies (numpy, scipy, Pillow, MuJoCo);
                                                            # CPU only, 1-2 min; passes on MuJoCo 3.13 and in the image (3.14)
```

## 11. Topics these notes cover

- the commit list and marks;
- D2;
- the source delta since e3ea480 and its neutrality evidence;
- the evidence we have (with the colour table) and what is missing;
- the stale seed instances after 2b and the colour fixes, and the claws 06-10 text change;
- the owner's decisions;
- HF;
- the new bases' pushed and local-only parts;
- the two seed tests.
