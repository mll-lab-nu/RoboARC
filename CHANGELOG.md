# Changelog

RoboARC's releases. Versions before 1.0.0 were released inside the shared benchmark repository
`mll-lab-nu/embodied-first-exam` (formerly robot_coding_bench), under the image name `ghcr.io/mll-lab-nu/eai-roboarc`.
1.0.0 is the first release from this repository. Its content is the "V3" wave of that repository.

## Unreleased

### The batch runner places each trial on one GPU (runner change; no task, image or result changes)
- Docker gives a container that reserves `count: 1` GPU the host's first GPU: on a host with several GPUs every trial
  of `scripts/harbor/run_agent_batch.sh` ran on GPU 0 and the other GPUs stayed idle.
- With `GPU_SCHED=1` (the default, `SEQ=1`) each lane places a trial through a ledger the host's lanes and batches
  share (`scripts/harbor/gpu_slots.py`): up to `GPU_SHARE` (4) rigid-task trials per GPU, a deformable-task trial on a
  GPU of its own (`DEFORMABLE_FAMILIES`: the blanket, fiber, knot, orbits, rope, slotted-post-ring and wire families),
  no GPU for a task without one (IKEA privileged). A trial starts when the CPUs and memory its task declares
  (`[metadata.resources]`) fit beside the trials running; else the lane tries the plan's next job.
- The job's copy has its containers pinned to that GPU (`device_ids`, `scripts/harbor/pin_gpu.py`), and
  `<job>/placement.json` (in config.json as `gpu`) records it. The copy is all that changes: task digests,
  `digest_match` and earlier results are untouched, and no image is rebuilt.
- docs/RUNNING.md "GPUs and lanes"; docs/COLLABORATOR_SETUP.md starts 4 lanes per GPU. `GPU_SCHED=0` is the old
  behaviour.

### DSH, the DeepSeek Harness, in the batch runner: `AGENT=dsh` (runner change)
- `scripts/harbor/agents/dsh_acp.py`, a Harbor agent of RoboARC's own: Harbor's ACP agent with `@deepseek-ai/dsh`
  0.2.1-rc.2 installed during setup and started as `dsh --profile acp`. That version is the release line of the exam's
  DSH runs (2026-10-15 to 10-18); it picks models from its built-in catalog, where DeepSeek-V4.1-Flash is
  `deepseek-flash`, so `MODEL=deepseek/deepseek-v4.1-flash` gives the exam's run id, `dsh-deepseek_v4_1_flash-high`.
- Its instruction travels in a file (`dsh/acp_prompt_file.py`), never on a command line, so an agent's
  `pkill -f <a word of the task>` cannot kill its harness. The session's `reasoning_effort` is set from `EFFORT`. The
  model is requested as dsh's own option value (`["deepseek-official","deepseek-flash"]`), the only form Harbor's
  runner matches, and the prompt goes out only when the session runs that model (dsh's ACP default is
  deepseek-v4-flash, DeepSeek-V4-Flash); both are recorded in the trial's `agent/dsh-session.json`.
  `WEB_SEARCH=disabled` turns dsh's web_search and web_fetch off (`dsh/roboarc.patch.yml`).
- Credentials: `DEEPSEEK_API_KEY` (`.env.example`; `scripts/harbor/scan_secrets.py` scans for it). Egress: the job adds
  `api.deepseek.com`, or, with `USAGE_GATEWAY`, dsh's `DEEPSEEK_BASE_URL` points at the gateway, whose `api_paths.json`
  now forwards DeepSeek's Messages API (`POST /anthropic/v1/messages`; pending the owner's approval).
- DshAcp is not a Harbor built-in: whether its results are official is the owner's call (PROTOCOL D11). Tests:
  `scripts/harbor/tests/test_dsh_acp.py`, DSH cases in `test_usage_gateway_option.py`, `test_paths.py` and
  `account_failure_check.py --selftest`.

## 2.1.0 (2026-10-25)

The owner's decision on the five bed-bolt bases (hex, Torx, square, Phillips and slotted keys): change only the
appearance, so that the drives tell apart at a glance. A grading change for the blanket families (their release check
no longer counts the fixed arm bases), a grader fix for wire_splicing from the family's stress round, and text fixes. A
new image, `ghcr.io/mll-lab-nu/roboarc:2.1.0`, because the simulator source and fifteen frozen instances change. The
task set is 2.0.0's: 90 bases, 180 task directories.

### Blanket flatten and blanket fold: the release check counts only the moving robot parts (grading change)
- Before 2.1.0 the release check measured the cloth's bounding box (grown by 8 mm) against every robot collision box,
  the two arm bases' among them. The bases are fixed to the world and cannot be withdrawn, so a fully spread sheet also
  had to fit between them: an x-span of at most 0.7075 m and centred, which a sheet turned more than about 15 degrees
  cannot meet. The texts never said so. blanket-flatten-07's reference passed with 0.9 mm to spare.
- Now the check leaves out the boxes of robot links fixed to the world (policy `active_parts_clearance_v1`, terminal
  revision `release_settling_v2_active_parts`); every arm link, wrist, gripper and finger keeps the 5 mm clearance. This
  is the rule knot_untying already uses for its rope (world-fixed bases may touch). The instructions say so (both modes).
- Blanket successes can only increase: every state that passed before passes now. A state that failed only because the
  cloth came near a base now passes, for example flatten-07's spread sheet turned 20 degrees with the hands away
  (old check: 0 mm to a base, failed; new: 62.8 mm to the nearest moving part, passed).
- Earlier blanket failures whose only failing gate was the release clearance against a base can be re-graded from their
  trajectories with the new grader (`scripts/roboarc/regrade.py` where it applies); nothing has been re-graded.
- The terminal revision is part of the task's identity, so the ten blanket instances are refreshed
  (`tools/release/refresh_instance.py`: their scenes and archives byte for byte) and their references re-stamped
  (`instance_sha` only). Their birth and end state hashes change in the task part (the instance id and the release
  metrics) and in the release geometry's new fixed flags only; the physics and the rows are unchanged.
- The other families with a release check are unchanged: cast_coil and buzz_wire_3d keep `all_parts_clearance`,
  knot_untying its exact policy; their state is byte-identical.

### Buzz wire 3D 06, 07 and 10: the loop opening in the texts (`04c7dd90`)
- Their texts now state their own usable loop opening (8.77, 8.27 and 6.77 mm) instead of the first tiers' 12.67 mm;
  `tools/release/topogym_rules.py` computes it. Text only; the grader is unchanged.

### Collaborator note (Jianwen, `eba9248c`)
- `docs/COLLABORATOR_SETUP.md` is a collaborator's one note (agent logins, the settings of an official run, which trials
  may run again); `docs/RUNNING.md` points to it. It names the 2.1.0 image and 2.0.0's task set (90 bases, 540 trials).

### Bed bolt: the drive card (Jianwen, `6c24f636`)
- A white card lies flat on the table between the key's stand and the bolt's cradle, in front of them, with the bolt's
  head drawn on it 4.5 times its size (zinc) and the head's recess in black: hexagon, six-lobe star, square, cross or
  slot, from the recess's own dimensions. In `cam_high` the drawn head is about 50 px across.
- Visual only: drawn geoms (no collision, no mass) on the bed frame's static body, which the MuJoCo scene drops. The
  physics scene (`scene.xml` and every mesh), the births, the references and the grading are unchanged. The card is the
  same at every appearance seed.
- Clear of the references: replayed row by row, no robot, bolt or key surface comes within 87 mm of the card (the
  nearest is the left gripper's finger as it takes the key). It covers only the table in every camera: the key, the bolt
  and the hole are seen as before.
- Each bed-bolt instruction (both modes) says in one sentence: "A white card lying flat on the table between the key's
  stand and the bolt's cradle shows the bolt's head and its recess, drawn 4.5 times their size."
- No per-type handle colours (the appearance seeds already vary the key's finish).

### Wire splicing: the coil count on a straight tail (Jianwen, `2b765e6a`)
- The grader faulted (an exception inside the task's step, reported as a simulator breakdown) when the tail of the
  first wire was pulled straight out past the eye: its return strand was a single node. It now counts no turn there.
  The reference never takes that branch; its end state is the recorded one. `tools/tests/test_wire_splicing_criteria.py`
  (run by `check-unit-tests.sh`).
- `tools/stress/round.py` (`097a3123`): a reference's wall limit in the stress round follows the base's agent budget
  where the owner excepted it (`CUTOFF_EXCEPTIONS`).

### Instances, hashes and results
- The five bed-bolt instances and their seeds 1 and 2 are re-made in the image (`release_rebuild.py`,
  `derive_seed_instance.py`). They differ from 2.0.0's only in the bed frame's drawn parts: `instance.json`
  `asset_identity` (`bed_frame.xml`) and `topogym_source`, and `scene.json`'s record of the bed frame file (its hash,
  the number of dropped visual geoms) and of the robot file's cache name. The state hash covers no drawn geometry:
  every birth (seeds 0, 1 and 2) is unchanged, so the references keep their bytes (no re-stamp), and every end state is
  the recorded one.
- Results graded under 2.0.0 stand for every task but the blanket ones, whose grading is looser now (above). The
  bed-bolt tasks look different and their instruction has one more sentence; their physics and grading are unchanged.
  The other tasks change only in their README's Release line, their Change Log and their image pin (and the buzz-wire
  texts above).

### Image
- `roboarc:2.1.0` (`sha256:c813b33cff3f...`, manifest `sha256:ccc3a47a5d9e...`, `images/pins.json`): every file is
  1.3.0's but the changed source files (`geometry/bed_bolt.py`, `tasks/wire_splicing/criteria.py`,
  `evaluation/terminal/settling.py`, `world/mujoco/release.py`, `world/mujoco/world.py` and its `.pyc`,
  `simulator/env.py`, `tasks/blanket_flatten/common.py`, `tasks/goal_description.py`), `SOURCE.json`, 15 `.pyc` (their
  4-byte source-mtime field only) and the fifteen re-made or refreshed instances with their seeds. The numeric
  libraries are byte-identical. It still carries the instances of the 33 bases removed in 2.0.0, as 1.3.0 does
  (`assets.json` pins them). The IKEA images are unchanged.

### Evidence (2.1.0 image)
- Harbor, bed-bolt 01-05, both modes, seeds 0, 1 and 2: oracle 1 and nop 0 in all 60 trials, no infrastructure error;
  every oracle end (two replays, or the live episode and its replay) is the recorded one of its seed. The privileged
  oracle at seed 0 ran with rendering on (its replay video shows the card).
- Harbor, wire-splicing-01, seed 0, both modes: oracle 1 and nop 0; ends the recorded 48f312d445aec62b (privileged, two
  replays) and 164567f6effde5b0 (standard, live and replay).
- Harbor, the ten blanket bases, seed 0, both modes, oracle and nop, and seeds 1 and 2 privileged oracle: oracle 1 and
  nop 0 in all 60 trials, no infrastructure error (two runs were repeated: a GPU lost to new processes after a host
  service reload, and a standard oracle that ran out of agent time on an overloaded GPU). Every oracle's two ends are
  equal; at seed 0 the new ends are the refreshed instances' (task part) of the recorded physics: a leaf-by-leaf
  comparison of birth and end records, 1.3.0 image and 2.0.0 instance against 2.1.0, differs only in the instance id,
  the release metrics and the new fixed flags. blanket-flatten-06's privileged reference reaches success on these hosts.
- One base per other family (15 bases on the CPU; the blanket and wire families by their Harbor runs above): strict
  replays in two fresh processes, as the verifier grades, end on the recorded states; the renderer's scene inputs
  equal 1.3.0's for one base of every family but bed_bolt, whose scene adds the card.
- The release image archive is byte-identical when made twice; the CI checks pass on a fresh clone.

## 2.0.0 (2026-10-25)

The owner pruned the task set: 33 puzzle bases leave the release. RoboARC 2.0.0 has 90 task bases, 78 puzzle bases in
19 families (every family keeps at least one base) and the 12 IKEA scenes, in 180 Harbor task directories (156 puzzle,
24 IKEA). The image is unchanged: `ghcr.io/mll-lab-nu/roboarc:1.3.0` (`images/pins.json`, `images/roboarc/VERSION`),
so are the IKEA tasks and images.

### Removed bases
| family | removed (difficulty) | remaining (difficulty) |
|---|---|---|
| blanket_flatten | 01 (very_easy), 02 (very_easy), 05 (medium), 09 (hard) | 5: 03, 04, 06, 07, 08 (easier 1, easy 2, medium 1, hard 1) |
| buzz_wire_3d | 01 (very_easy), 03 (easy), 05 (hard), 08 (easy), 09 (medium) | 5: 02, 04, 06, 07, 10 (very_easy 1, easier 2, medium 1, hard 1) |
| cast_coil | 01 (very_easy), 02 (easier), 03 (easy) | 5: 04-08 (one per tier) |
| cast_keyhole | 01 (very_easy) | 5: 02-06 (easier 1, easy 1, medium 1, hard 2) |
| fiber_bundling | 01 (very_easy), 03 (easy), 06 (very_easy), 07 (easier), 10 (hard) | 5: 02, 04, 05, 08, 09 (easier 1, easy 1, medium 2, hard 1) |
| hedgehog_in_cage | 01 (very_easy), 04 (medium), 05 (hard) | 5: 02, 03, 06, 07, 08 (easier 2, easy 2, hard 1) |
| interlocked_claws | 01 (very_easy), 03 (easy), 04 (medium), 06 (very_easy), 07 (easier) | 8: 02, 05, 08-13 (easier 1, easy 1, medium 2, hard 4) |
| knot_untying | 01 (very_easy), 03 (easy) | 6: 02, 04, 05, 06, 07, 09 (very_easy 1, easier 2, medium 2, hard 1) |
| lock_picking | 01 (very_easy), 02 (easier) | 3: 03, 04, 05 (easy 1, medium 1, hard 1) |
| rope_threading | 01 (easy), 02 (medium), 03 (hard) | 5: 04-08 (one per tier) |

The other nine families are unchanged: battery_replacement 1, bed_bolt 5, blanket_fold 5, caliper_sorting 1,
clock_assembling 1, horseshoe_ring 5, orbits 2, slotted_post_ring 5, wire_splicing 1. By difficulty, 33 removed
(very_easy 12, easier 4, easy 7, medium 5, hard 5); 78 remain (very_easy 9, easier 17, easy 14, medium 17, hard 21).
Base numbers are never reused: a removed base leaves a gap in its family's numbers.

### Results
- The kept tasks are unchanged but for their README's Release line ("RoboARC 2.0.0") and `DIGEST`: their instructions,
  graders, instances, references and image are 1.4.0's, so results graded under 1.x on these tasks stand.
- Aggregate numbers (success rates over the task set, per family or per difficulty) of 1.x and 2.0.0 are not
  comparable: the set is different.
- The removed tasks are kept by the maintainers, together with their packages, references and release records, on
  the branch `archive/removed-in-2.0.0` (= RoboARC 1.4.0, `315c10ca`), and they run there on the same image.

### Files
- `scripts/roboarc/release.json`: the 33 records removed from `bases`; the version is 2.0.0 (also `pyproject.toml`,
  `CITATION.cff`, `roboarc.__version__`). Their task dirs, packages and references are removed from the tree.
- `tools/release/mujoco_release.json` keeps their entries, each family with a `removed` note per base;
  `release_rebuild.py` skips those bases. Their entries in `roboarc_names.json`, `release_evidence.json` and
  `standard_speed_rule.json` are removed. `records/` (the freeze manifests) and `scripts/roboarc/assets.json` (the
  instance pins) are unchanged.
- The `roboarc` API lists 78 puzzle bases and 90 tasks; 77 puzzle bases ship a reference (fiber-bundling-05 does
  not), each pinned by the V3 freeze manifest or its release record; `reference_pins` pins only the bases of
  `release.json`. 10 privileged references are faster than the standard joint-speed rule.
- `docs/easy_set.toml`: the 16 removed bases' entries are gone; 15 remain.
- Examples, quick starts, tests and the collaborator guide use kept bases (interlocked-claws-08 and -09,
  cast-keyhole-02, cast-coil-06; the bit-identity selection and `test_vla` accordingly). `docs/COLLABORATOR_SETUP.md`:
  a full run is 90 x 2 x 3 = 540 trials.
- Family pages, the README table and the API documentation give the new counts.

## 1.4.0 (2026-10-24)

The nine bases prepared as experimental in 1.2.0 are released into `tasks/` in both modes, after the owner's review
(2026-10-24): four new families and a thirteenth interlocked-claws base. 111 bases in 19 families, 222 puzzle task
directories (246 with IKEA's 24). The image is unchanged: `ghcr.io/mll-lab-nu/roboarc:1.3.0` (`images/pins.json` and
`images/roboarc/VERSION` stay 1.3.0), so are the IKEA tasks and images. The 1.3.0 tasks' instructions, graders,
instances and references are unchanged: results graded under 1.3.0 stand.

### New bases (by Jianwen)
| base | family | difficulty | the task |
|---|---|---|---|
| bed-bolt-01-i01 | bed_bolt (new) | medium | drive a bed frame's corner bolt home with an L-shaped hex key |
| bed-bolt-02-i01 | bed_bolt | hard | the same with an L-shaped Torx key |
| bed-bolt-03-i01 | bed_bolt | very_easy | the same with a ratcheting L-shaped square key |
| bed-bolt-04-i01 | bed_bolt | easier | the same with a ratcheting L-shaped Phillips key |
| bed-bolt-05-i01 | bed_bolt | easy | the same with a ratcheting L-shaped slotted key |
| caliper-sorting-01-i01 | caliper_sorting (new) | hard | measure four look-alike sleeves with a vernier caliper that reads in its own unit, sort them into labelled bins |
| clock-assembling-01-i01 | clock_assembling (new) | hard | assemble a watch movement's gear train, screw down its bridge, wind it and let the mainspring run it (`dynamics = "dynamic"`) |
| battery-replacement-01-i01 | battery_replacement (new) | medium | swap a game controller's two AA cells, the old ones pried out with a pry stick, and latch the cover |
| interlocked-claws-13-i01 | interlocked_claws | hard | Hanayama's Cast Trinity: separate three interlocked pieces |

- `release.json`: each record moved from `experimental_bases` to `bases` (released 2026-10-24); `experimental_bases` is
  empty. The packages' difficulty `untiered` became the owner's tier. Agent time 10800 s for each.
- Family pages `docs/families/bed_bolt.md`, `caliper_sorting.md`, `clock_assembling.md`, `battery_replacement.md`;
  `interlocked_claws.md` lists base 13.
- The `roboarc` API lists them (111 puzzle bases, 123 tasks; 109 puzzle bases ship a reference, the nine new ones pinned by their release
  records' privileged evidence).

### Tasks
- New: the 18 directories `tasks/roboarc-<base>-{privileged,standard}` of the nine bases. Each equals the
  `mk_tasks.py --experimental` render the evidence below ran on, but for `task.toml`'s difficulty and `DIGEST`.
- The 204 directories of 1.3.0 re-rendered for the version only: the README's Release line names RoboARC 1.4.0, and
  `DIGEST`.

### Evidence (the nine bases)
- **Harbor on the 1.3.0 image** (Jianwen, 2026-10-24; image id `499d6f69bd81`, simulator source `4d2cba21`, tooling
  `412cbe02`): oracle 1 and nop 0 in both modes at seeds 0, 1 and 2, no infrastructure error (108 / 108). Seed 0's
  oracle ends equal the recorded end states in both modes; at seeds 1 and 2 the two replays agree and both modes reach
  the same end. The references are unchanged (trajectory sha256 as recorded).
- From 1.2.0 (the same instances and references): the reference replays on Intel equal Zen 4 (9 / 9); reference
  timings at most 14.1 % of the 3 h budget; each seed's reference ends with the same world, robot and state as seed 0.
- Texts: the success hold (0.5 s) and the tool-frame distances were checked against the instances before the review.
  The texts state the grasping rules in more detail than the tooling's own grasp sentence (`Session.grasp_rules()`),
  without contradicting it; see the benchmark page, "Grasping". One sentence was less exact than the simulator and is
  corrected: battery-replacement-01's text let a pluck go once the cell "has moved 60 mm from where it was taken"; it
  now states the simulator's rule, one 60 mm budget per old cell and episode, shared by all its plucks and counted from
  where the cell leaves the compartment (the grading is unchanged).

### Collaborator tooling
- `scripts/send_results.sh` (by Jianwen): sends one batch's records to the public Hugging Face dataset
  `RoboARC/RoboARC-outputs`: a secret scan of the batch's jobs (redacted, then scanned again), one archive per job
  with an index and a manifest, and a scan of the whole bundle once more before the upload, since the dataset is
  public; `CREATE_PR=1` uploads as a pull request, `DRY_RUN=1` stops before the upload. A batch without a verdict yet is
  still sent. `scripts/receive_results.sh` is the maintainers' side.
- `docs/COLLABORATOR_SETUP.md`: run from `main`; a first check in both modes and at another seed, a smoke test of the
  batch runner, the full run over every base (123 bases, 738 trials) with lanes, duration, hosts and restarts, and
  sending the results back with the collaborator's own Hugging Face account.

### Docs
- README rewritten around the interfaces: coding agents (Harbor), VLA policies (in development, openpi conventions) and
  the Python API, on the same tasks, physics and graders; results are never pooled across interfaces.
- Counts updated (README, `docs/API.md`, `docs/DEVELOPING.md`, `docs/benchmarks/roboarc.md`, `docs/families/`,
  `docs/COLLABORATOR_SETUP.md`); `docs/TAXONOMY.md` names clock-assembling-01 as the one dynamic task;
  `tools/release/mujoco_release.json` records the ten bases' releases (wire-splicing-01's 1.3.0 evidence included).

### Checks
- `mk_tasks.py --check`, the generator and seed tests, unit tests, invariants, cutoffs, doc paths, CI self-test,
  `source_hash.py --check`, the API tests (catalogue counts, reference pins, joint-speed rule: 13 privileged references
  faster than the standard rule, as before).

### VLA policies (preview)
- `roboarc.vla` and `python -m roboarc.vla.run`: graded episodes driven by a policy server speaking openpi's websocket
  protocol, with openpi's ALOHA layout (two-arm robots) and DROID layout (the Franka + Robotiq families; joint
  positions, or DROID's joint velocities); the grader is the standard mode's (live success and a fresh-process replay);
  results in their own layout with a `vla:` configuration key. A stub policy server (`roboarc.vla.stub_server`: hold,
  or a replayed reference). Documentation: `docs/VLA.md`. Extra `.[vla]`: msgpack 1.1.0, websockets 15.0.1.
- `roboarc.make(..., instance_seed=S)` runs a puzzle's appearance seeds 1 and 2 (staged against their own bindings, as
  the graded tasks stage them); `TaskInfo.seeds`. `env.observe()` and `env.step_images = False` render images only
  when asked. Graded behaviour is unchanged.

## 1.3.0 (2026-10-23)

A new family, wire_splicing, with its first base wire-splicing-01-i01 (hard) in `tasks/` in both modes: 102 bases in 15
families, 204 puzzle task directories. The owner reviewed its demo on 2026-10-23. Image `ghcr.io/mll-lab-nu/roboarc:1.3.0` (manifest `sha256:2bcded00…`,
id `499d6f69bd81…`); the IKEA tasks and images are unchanged. Every released task's instance, physics, instruction,
grader and reference are unchanged: results graded under 1.2.x stand.

### wire_splicing (new family)
- The family (by Jianwen, ported from TopoGym `cb3d05437c`): two stiff wires from their terminals, 160 ball-jointed
  links with elastoplastic joints; the agent bends an eye in the red wire's bare end round the yellow rod of a bending
  post, threads the blue wire through it, winds the blue wire's bare tail at least 1.5 turns round the red wire's
  doubled end (at least one full turn round both strands together), folds the red wire's end back and lets go.
- wire-splicing-01-i01: packages, references (privileged; standard re-timed to the joint-speed rule), its record in
  `release.json` (released 2026-10-23, difficulty hard), its instance and seeds 1 and 2 (other colours of the terminals, the post and one
  bare core; the red and blue insulation and the post's yellow rod keep their colours) in the
  image and pinned in `assets.json`. Agent time 21600 s by the owner's exception (below); one control step takes about
  0.5 s. Task directories `tasks/roboarc-wire-splicing-01-i01-{privileged,standard}`; family page
  `docs/families/wire_splicing.md`.
- The `roboarc` API lists it (102 puzzle bases, 114 tasks). A reference's pin (file sha256, end hash) comes from the V3
  freeze manifest, or for a base released later from its release record's privileged evidence
  (`roboarc.registry.reference_pins`).
- Its texts state the instance's own values where the bed-bolt template's had been kept: the tool-frame distance 6 cm
  (not 15 cm) and the success hold 2 s (not 0.5 s; `stable_duration_s`), also in `tools/release/package_text_wire.py`.

### Shared code: opt-in only
- `MultiRopeGraspAssistConfig` (`topogym/utils/grasp.py`) gains `weld_links_between_pads` (a hand welds every link of a
  touched wire whose centre lies between its pads, one binding) and `hold_between_pads` (two fingers on the wires with
  every contact between the pads hold as a pinch), both default off; only wire_splicing's grasp configuration builds
  this class.
- The MuJoCo grasp backend's weld work guard (`topogym/world/mujoco/grasp.py`): a weld whose net work within its hold
  exceeds the scene profile's `weld_work_guard_j` lets go as a slip. It is read from the frozen scene's profile, and no
  other family's profile names it, so it is off for every released instance; with it off, the grasp state a checkpoint
  carries is as before.
- New entries for the family in the registry, variation records, goal texts, appearance axes, scene profiles and the
  tooling's task table (`rcb_topogym.py`, tooling v22); no existing entry changes. SOURCE.json `4d2cba21…` (454 files).
- `tools/tests/test_mujoco_grasp_opt_in.py`: the WeakHold field test follows the fields battery's rule C added.

### Owner exceptions to the cutoff cap
- The protocol's 14400 s cap now admits the bases listed in `scripts/task_conventions.py` `CUTOFF_EXCEPTIONS`, each with
  the owner's approval; `check-cutoffs.py` and `check-task-fields.sh` read the same list. The first entry is
  wire-splicing-01-i01 (owner, 2026-10-22: 21600 s; 160 ball-jointed links at about 0.5 s per control step). For an
  excepted base `mk_tasks.py` writes the cutoff and raises the grading timeouts with it: the privileged verifier to 2 x
  cutoff + 1800 s (45000 s), the standard collect hook to 1.5 x cutoff (32400 s). `cutoff_s` is part of the
  configuration results are ranked by, so its trials form their own group. The 202 task dirs did not change by it.

### Tasks
- New: `roboarc-wire-splicing-01-i01-privileged` and `-standard` (agent 21600 s, privileged verifier 45000 s, standard
  collect hook 32400 s).
- The 202 earlier puzzle directories re-rendered: the tooling copies (`rcb_topogym.py`), the runtime bindings of seeds 0-2
  (source and tooling), the image reference, `tooling_version` 22, the README's Release line and Change Log, `DIGEST`.
  Instructions, graders, instances and references are unchanged; so are the IKEA directories.

### Evidence
- **Image:** every file outside `/opt/topogym` and the new instance is byte-identical to 1.2.0's image (path, mode, size,
  sha256 of 104,536 entries; the numeric libraries unchanged); the 110 earlier bases' instances and seeds are 1.2.0's
  bytes. The simulator source differs from 1.2.0's in wire_splicing's files and the shared entries above only.
- **Released bases, seed 0:** in a 1.3.0 build that differed from this one only in wire_splicing's own files (the needle
  meter, removed before the release; the file manifests show those files and nothing else), the strict double
  fresh-process replay of all 101 references reached its recorded end state, with births = frozen and the instance,
  asset and runtime-binding checks (Zen 4); render inputs 101 / 101 identical to 1.2.0. In this image: one base per
  family again (14 / 14 ends as recorded; render inputs identical), and Nyx frames at birth and after up to 300
  reference rows within 1.2.0's own run-to-run spread (14 / 14 families; the path tracer is not pixel-deterministic
  between runs of the same image).
- **Seeds 1 and 2:** the 220 seed instances pass the four checks against their 1.3.0 bindings (assets and physical fields
  = seed 0's, strict staging, birth parts = seed 0's but `task`, the re-stamped reference); derived again in this
  image they give the same births and `instance_assets/`, and `instance.json` differs only in `derived_from.source`.
- **Harbor:** one base per family, oracle 1 / nop 0 in both modes at seed 0, every end as recorded (56 / 56); in the
  earlier build also seeds 1 and 2 of the same bases (112 / 112).
- **wire-splicing-01-i01:** Harbor oracle 1 / nop 0 in both modes at seeds 0, 1 and 2 (12 / 12; two equal replays each;
  ends 48f312d445aec62b privileged, 164567f6effde5b0 standard); its seeds pass the four checks; Intel Xeon Gold 6230R
  = Zen 4 on the strict replay (48f312d445aec62b); one privileged reference execution on an RTX 4090 Zen 4 host under
  its usual shared load: 8171 s, 37.8 % of the 21600 s budget (0.55 s per control step); along the reference the task
  succeeds only at its last row, with every condition met (no earlier state is credited).
- **Checks:** generator, IKEA, image pins, invariants, unit tests, API tests, advisories, `run_all.sh`,
  `source_hash.py --check`, doc paths and tooling identity pass on a fresh clone.

## 1.2.1 (2026-10-22)

Nyx on GPUs without RT cores: a clear error instead of a crash. The same images (`roboarc:1.2.0`, the IKEA images
1.1.0), physics, instances, references, instructions and graders: results graded under 1.2.0 stand.

- **The problem** (measured on an A100 and an H100, NVIDIA driver 610.43.02): GPUs without RT cores offer the Vulkan
  ray-tracing pipeline and acceleration structures but not `VK_KHR_ray_query`, which Nyx's path tracer uses. The
  driver's shader compiler rejected Nyx's pipelines and the native renderer then died with a segmentation fault, which
  no Python code can catch. So on such a GPU the privileged verifier crashed in its video pass before it wrote the
  verdict (the trial got no reward at all), an agent's first camera call killed its Python process, and standard mode's
  `sim` service crashed at start-up.
- **The fix**: `nyx_direct.py` checks the GPU before it creates the native renderer. It finds the Vulkan device that is
  the CUDA device (the same UUID) and requires `VK_KHR_ray_query`, `VK_KHR_acceleration_structure` and
  `VK_KHR_external_memory_fd` there (`NYX_VULKAN_EXTENSIONS`). A missing extension, no Vulkan device for the CUDA GPU
  (a CPU driver such as Mesa lavapipe) or no Vulkan driver raises `nyx_direct.UnsupportedDevice` (a `RuntimeError`)
  naming what is missing and the GPUs Nyx runs on, at once and without the start-up retry; later builds in the same
  process raise it without trying again. A check that cannot complete leaves the decision to the build, as before.
  The check takes about 0.1 s and changes nothing on a GPU with RT cores.
- **What runs where now** (docs/RUNNING.md, "GPUs without RT cores"): on GPUs with RT cores (GeForce RTX 20-series
  or newer, RTX workstation GPUs, T4, L4, L40 / L40S, A10 / A10G, A40) everything is as in 1.2.0. On V100, A100, A30,
  H100 or H200, standard mode does not run (its observations are Nyx images); the privileged verifier grades from the
  physics replays as on any host, without the replay video (`video_rendered` 0, `media.json` status "error"); agents'
  camera calls raise `UnsupportedDevice`. Such runs are unofficial: the agent sees no images. There is no software
  fallback.
- **IKEA**: the standard sidecar's Nyx render process (`ikea_nyx.py`, with IKEA's own copy of `nyx_direct.py`) has
  the same check; it reports an unsupported GPU in its ready line and the service stops at once
  (`ikea_nyx.UnsupportedGPU`, no start-up retry) instead of crashing three times. IKEA's privileged tasks do not use Nyx.
- **Tasks**: the 202 puzzle directories re-rendered (`nyx_direct.py` copies, the runtime bindings'
  `nyx_direct_sha256` for seeds 0, 1 and 2, the Release line "RoboARC 1.2.1", a Change Log entry, `DIGEST`), and the 12
  IKEA standard directories (the sim's `nyx_direct.py` and `ikea_nyx.py`, a Change Log entry, `DIGEST`). The IKEA
  privileged directories do not change. No image is rebuilt: the task tooling is not in the images (each task
  directory carries its pinned copy).
- **Tests**: `tools/tests/test_nyx_device_check.py` (mocked Vulkan extension lists and a stub gs-nyx, both
  `nyx_direct.py` copies; in `ci_checks/check-unit-tests.sh`); IKEA's `test_nyx_startup.py` covers the unsupported start.
- **Docs**: README, docs/RUNNING.md, docs/API.md, the benchmark page and the image README state the GPU requirement.
  New `docs/COLLABORATOR_SETUP.md` (by Jianwen): machine, access, a first check and the evaluation run, for a
  collaborator; linked from the README.

## 1.2.0 (2026-10-21)

Three scenes per task base (appearance seeds), the simulator source and tooling they need, and nine new bases prepared
as experimental. Image `ghcr.io/mll-lab-nu/roboarc:1.2.0` (manifest `sha256:a8561b77…`, id `b600616bee6c…`); the IKEA
tasks and images are unchanged. 226 task directories (202 puzzle, 24 IKEA), every puzzle directory re-rendered.

### Appearance seeds
- Every one of the 101 bases has seeds 0, 1 and 2 (`[metadata.seeds] allowed = [0, 1, 2]`, `build = "runtime"`, both
  twins). Seed 0 is the released scene, unchanged. Seeds 1 and 2 are the same scene and physics in other colours,
  materials and textures (`topogym.tasks.scene_variation.appearance(cls, S)`, scene seed 0), derived from the seed-0
  instance (`tools/release/derive_seed_instance.py`: `instance_assets/` byte for byte, the build's fields recomputed).
- The runner picks the seed (RUNNING.md "Seeds": the K trials of a task get distinct seeds); the image holds every
  seed's frozen instance under `/opt/robotangle/instances/<base>/seeds/<S>/` and the task stages the trial's seed
  against its own runtime binding (`runtime_binding.s<S>.json`, `stage_instance.py --seed-file`, fail closed).
- References: the privileged reference is re-stamped per seed (`solution/trajectory.s<S>.npz`: the same rows with the
  seed's birth; `solve.sh` picks the seed's); rope-threading 04-08's references record no birth and serve every seed as
  they are; the standard oracle streams the same rows for every seed. Goal pictures (blanket-fold 01-05, knot-untying
  07/09) are rendered per seed (`tests/seeds/<S>/goal/`, copied over `environment/goal/` by `task_seed.py materialize`).
- `[metadata.seeds] oracle` lists seed 0 and the seeds whose Harbor oracle run passed (below): `[0, 1, 2]` on 30 bases,
  `[0]` on the other 69 with a reference.
- Seed instances: in the image and pinned by sha256 in `scripts/roboarc/assets.json`; the HF dataset revision that
  will hold them (and the new bases' instances) is a separate step, uploaded only with the owner's approval. Until then
  `images/roboarc/build.sh` builds from a local instance tree (`--instances`).

### Simulator source and tooling
- Simulator source synced from TopoGym's development line (`497a9e9d7f`, `622c7801c6`, `738231f5cb`, `3eed5d8b33`,
  `a849170272`) onto 1.1.0's source: SOURCE.json `cacc831d…` (449 files). The appearance axes of every family (their
  defaults are the released looks), the new families' code (bed_bolt, caliper_sorting, clock_assembling,
  battery_replacement, Cast Trinity), opt-in grasp rules (force closure, weak holds: used by battery_replacement only).
  Task tooling v21 (`rcb_topogym.py`: the new families registered; `nyx_direct.py`: a fixed MJCF entity's visuals drawn).
- fiber-bundling 07-10: the `lay` rule takes either red fiber (the scene has two: strands 0 and 8, 16 strands over an
  8-colour palette); the texts say "the root of one of the two red fibers". A trajectory that met the rule with strand
  0 is graded as before; one that failed only because the other red fiber lay at the stated position now succeeds. The
  references reach their released end states under the new rule. Re-grading earlier results on these four bases needs
  a replay of the trajectory with the 1.2.0 grader: `scripts/roboarc/regrade.py` does not cover this rule (it re-grades
  the 2026-10-14 rule from the record, and the record holds only strand 0's angle); replay the trial's trajectory with
  the 1.2.0 task's `tests/grade.py` in its verifier image (as `regrade.py --replay` does for its rule), or rerun the
  trial.

### Texts that stay right in every seed
- interlocked-claws 06-10: the second piece is "a lock" (it was "a teal lock"; its colour changes with the seed).
- interlocked-claws 01/02 and cast-keyhole 01-06: the pieces are also named by their object names (`piece_0` /
  `piece_1`, `keyhole_silver` / `keyhole_gold`), so no text identifies an object by colour alone.
- Every colour word the texts name, per seed: `docs/simulator/seeds/colour_table.md`.

### New bases (experimental, untiered)
bed-bolt-01..05, caliper-sorting-01, clock-assembling-01 (dynamic: the wound mainspring drives the train),
battery-replacement-01 and interlocked-claws-13 (Cast Trinity): packages, records (`release.json`
`experimental_bases`), instances (with their seeds) in the image and references, rendered only by
`scripts/roboarc/mk_tasks.py --experimental --out <dir>`, never into `tasks/`, until the owner has reviewed their demos.
Their texts state the grasping rules the grader applies, more than the tooling's own grasp sentence reports (an open
item for that review). Evidence in 1.2.0's image: Harbor oracle 1 / nop 0, both modes, seeds 0-2 (108 / 108); the
reference replays on Intel equal Zen 4 (9 / 9); reference timings at most 14.1 % of the 3 h budget; the texts' step
times follow measured values where they were more than 25 % off.

### Evidence
- Seed instances (220 = 110 bases x seeds 1, 2), each: `instance_assets/` byte-identical to seed 0's, the physical
  instance fields and the built MuJoCo model's physics equal, staged against its own binding + a strict Session, the
  birth's tick record equal to seed 0's in every part but `task`, the re-stamped reference's birth = the seed's (220 /
  220; rope-threading 04-08 unstamped, buzz-wire-3d-05 / fiber-bundling-05 without a reference). Derived again in this
  release's image: births and `instance_assets/` equal to the first derivation's except where 1.2.0's later source
  changed the looks (battery, caliper, Trinity, rope-threading 01-08) or the text (fiber 07-10).
- End-state identity, 39 bases (one per family, interlocked-claws 06-10, fiber-bundling 07-10, rope-threading 01-08
  and the 9 new bases): each seed's reference, replayed strictly in two fresh processes, succeeds and ends with the
  same world, robot, coverage, targets and state as seed 0 (39 / 39 x seeds 0, 1, 2; only the task's instance id
  differs; Intel CPU, the blanket bases on the GPU).
- Released seed 0, all 101 bases: strict fresh replays in the 1.2.0 image reach 1.1.x's end states from the frozen
  births (101 / 101; also on Intel and the blanket bases on the GPU); the render inputs are identical to 1.1.0's
  (101 / 101: every probe built one Nyx scene from the same bytes); frames, one base per family, within the path
  tracer's own run-to-run spread (14 / 14), state records equal. Blanket (14 bases): births, ends and render inputs
  (cloth mesh, material map, texture) unchanged; the cloth's identity rule unchanged at seed 0.
- Harbor, rendering off, oracle 1 / nop 0, both modes, on materialized seed copies: the required set (one base per
  family + every touched base: interlocked-claws 06-10, fiber-bundling 07-10, rope-threading 01-08, seeds 1 and 2; the
  three re-made new bases, seeds 0-2): 276 / 276; the rest (colour-text bases at seed 0, bed-bolt 01-05 and
  clock-assembling 01 at seeds 0-2): 104 / 104.
- The image: a rebuild of the recipe with the same instances reproduces every file's content (only build-time file
  times in `.pyc` headers differ).

### Repository (since 1.1.0)
- The usage gateway ships with the repository: `scripts/usage_gateway/` (from the shared benchmark repository,
  unchanged; PROVENANCE.md) and `.env.example`. Usage recording stays optional and off by default; turn it on with
  `USAGE_GATEWAY=scripts/usage_gateway` (docs/RUNNING.md "Usage recording"). Its tests and a check that it takes
  every option the runner passes run in `ci_checks/check-unit-tests.sh`.
- `scripts/set_agent_egress.py` runs its command line only as a script, so importing it (to read `HOSTS`) runs
  nothing; as a script it behaves as before.
- `tools/stress/corpus/SHA256SUMS`: the O2 reproducer's line matches its file again.
- Everything the CI workflow uses is pinned: the runner image `ubuntu-24.04`, `actions/checkout` and
  `actions/setup-python` by commit (v4.4.0, v5.6.0), Python 3.12.15, and the jobs' Python packages by version and
  sha256 (`ci_checks/requirements-ci.lock`, installed with `--no-deps --require-hashes`).
- PROTOCOL.md: image references by digest is a rule (§1) and one of RoboARC's changes from the shared protocol (the
  tasks already comply). README / RUNNING: arm64 hosts are not supported.
- Trials and seeds (owner decisions for 1.2.0; PROTOCOL.md "Changes from the shared protocol", RUNNING.md "Seeds",
  "Results"): the batch runner gives the K trials of a task distinct seeds without replacement, in an order fixed by the
  batch name and the task's base name (`scripts/task_seed.py assign`; the same seeds in both modes), runs each trial as a
  Harbor job of its own from a copy made for its seed (`scripts/task_seed.py materialize`, back from 1.0.0 with
  `assign`, the verifier's `tests/` in standard mode and per-seed goal images), and records the trial and its seeds in
  `<job>/seed.json`. Results are one folder per trial, `<task>/<run id>/t<k>/` (`config.json` gains `trial`, `trials`,
  `trial_dir`, `copy_checksum`; `seed_source` is `runner`), with an index (`trials.json`, `README.md`) per task and run
  id. A single-seed task run with K > 1 is no longer over the retry cap: the retry rule (H.5) counts attempts per trial.
  The seed of a trial was recorded as the default before (the runner's task copy was gone when results were written);
  a trial's task version is now the committed DIGEST the runner copied (Harbor's checksum of the copy, which differs
  from it whenever the copy does, is kept as `copy_checksum`). `scripts/roboarc/regrade.py --replay` grades a trial with its seed's
  verifier files. New check `ci_checks/check-seed-files.py`. Every task still allows one seed: no task changes.

### Credit
The appearance seeds, the derive tooling, the new families and their references are Jianwen's work (TopoGym), ported
onto 1.1.x and validated for this release.

## 1.1.1 (2026-10-21)

RoboARC's four images moved to the `mll-lab-nu` organization, with the same bytes and digests:
`ghcr.io/mll-lab-nu/roboarc:1.1.0`, `roboarc-ikea:1.1.0`, `roboarc-ikea-sim:1.1.0` and `roboarc-monitor:0.1.0`
(`images/pins.json`; before: `ghcr.io/embodied-first-exam/<name>`). The packages are private and inherit the
repository's access, so a collaborator with read access to the repository can pull them (`docker login ghcr.io` with a
token that has `read:packages`). Every task directory is re-rendered: its Dockerfiles, compose files and README name the
new references, its Change Log has a 1.1.1 entry ("image references moved to ghcr.io/mll-lab-nu (same images, same
digests)"), and the puzzles' Release line names the task set 1.1.1; each task gets a new `DIGEST`. The runtime bindings
keep the image ids and the names the images were bound under, and the IKEA sim recipe keeps its `FROM` (an input of
the sim image). `ci_checks/check-image-pins.py` allows `ghcr.io/mll-lab-nu/` only (a task naming the former registry
fails). No instruction, grader, tooling, binding, instance, reference, image or archive changed: results graded under
1.1.0 stand. The repository changes listed under 1.2.0's "Repository" above change no task.

## 1.1.0 (2026-10-20)

The puzzle image without Genesis: `ghcr.io/embodied-first-exam/roboarc:1.1.0` (8.7 GB; 1.0.0 was 12.2 GB), and the IKEA
images rebuilt from fully pinned inputs with only what the tasks run: `roboarc-ikea:1.1.0` (1.05 GB; 1.31 GB) and
`roboarc-ikea-sim:1.1.0` (7.67 GB; 9.08 GB). The task set's version is 1.1.0. Every task binds its new image; the
instructions, the graders' logic, the frozen instances and scenes and the references are unchanged (except
blanket-flatten-06-i01's standard reference, under Records). Results graded under 1.0.x stand: every reference replays
in the new images to the end state it reaches in 1.0.0.
Details: [docs/history/GENESIS_REMOVAL.md](docs/history/GENESIS_REMOVAL.md), `ikea/images/ikea-mujoco/README.md`.

### The image
- Built from the Python base image (`source/images/roboarc/Dockerfile`) instead of the RoboTangle 0.2.22 (Genesis)
  image. Everything is pinned: the base image by digest, the Debian packages by version from snapshot.debian.org, the
  wheels by sha256 (`source/images/roboarc/runtime.lock`, `vbd-requirements.txt`).
- The same files as 1.0.0 for everything it keeps: every library the graded runtime loads (the Python interpreter,
  glibc / libm, libstdc++, numpy's and scipy's OpenBLAS, MuJoCo 3.14.0 with the EPA patch, Warp, Newton, torch and its
  CUDA libraries, gs-nyx, the graphics loaders, ffmpeg) is byte-identical, and so is every file of the 49 Python
  packages it keeps.
- Removed as unused (traced on every family, both modes, rendering on, and on the agent's API): genesis-world, libuipc
  (pyuipc, both builds), quadrants, gs-madrona, gs-nyx-plugin, torchvision, triton, opencv, vtk, matplotlib,
  scikit-image, numba and the other packages only these needed; build-essential, git, vulkan-tools; the uv download
  cache the 0.2.22 image kept (9.3 GB); documentation pages and apt/dpkg logs. curl, ca-certificates and ripgrep stay
  for the agent harness.

### The simulator, the tooling and the bindings
- `source/topogym/`: the Genesis world, scene builder, physics backends (FEM, IPC, the Genesis grasp models), visuals
  and renderer are removed (SOURCE.json: 443 -> 426 files); the runtime identity no longer names genesis-world, pyuipc
  or a libuipc build. The ground plane's checker texture, which the renderer read from the genesis package, ships with
  the image; a Panda + Robotiq robot given in genesis-world's layout resolves to the vendored Panda (the same bytes).
- Tooling v20 (`rcb_topogym.py`): no libuipc build choice by GPU, no Genesis kernel replacements; `nyx_direct.py` finds
  the checker texture in the image.
- All 202 puzzle task dirs: new `runtime_binding.json`, tooling copies, image reference, a 1.1.0 Change Log entry and
  DIGEST; the compose files cite the new recipe.

### Records
- `records/freeze_manifests/freeze_manifest.v3.json`: knot-untying-06-i01 and knot-untying-09-i01 expect the end hashes
  `0d25b901f83db04e` and `3dd3136502cb9e13`. The old values came from a pre-release dry run; the released references end on
  the new ones in images 0.2.0 (Harbor oracle runs), 1.0.0 and 1.1.0, with equal state hashes at every 250-row checkpoint.
  The references themselves are unchanged.
- blanket-flatten-06-i01's standard reference (`scripts/roboarc/oracles/blanket-flatten-06-i01/standard.npz`): its route
  re-recorded within the standard joint-speed rule (7577 rows, end `f6f8062d4270bdcb`; Harbor standard oracle 1 / nop
  0), replacing the privileged rows whose 3 faster rows the service stretched (oracle 0); the privileged reference is
  unchanged.
  On Intel (Xeon Gold 6338, A100 GPU) two strict fresh-process replays and the standard service's rules end on
  `f6f8062d4270bdcb` (every row executed as it is) and the privileged reference on `cc8615b8bc4a0efa`, as on AMD Zen 4.
  Stress, as for blanket-fold-05-i01's new reference (criteria v2.13b, the CPU twin of the GPU cloth step): 371
  episodes (random, directed, cem, periodic; 93,385 controls), 0 breakdown, 0 violent, 0 pressed-pile warning; the
  reference under the observer on the GPU step and nop 400 are CLEAN.
- Harbor oracle records (`scripts/roboarc/release.json` evidence, `tools/release/release_evidence.json`): the 24 that
  said `not_run`, no score claimed (privileged on 11 bases, standard on 13: blanket-flatten-04/05/07/08/09-i01,
  knot-untying-06/07/09-i01, orbits-02-i01 and slotted-post-ring-05-i01 in both modes, blanket-flatten-06-i01
  privileged, blanket-fold-02-i01 and slotted-post-ring-03/04-i01 standard) now hold their trials of the 1.1.0
  validation round on the release image: oracle 1 at the recorded end state (privileged: both replays agree;
  standard: live = replay) and nop 0. The READMEs' Reference solution and Change Log say so; no reference, instance
  or grader changed.

### Repository
- Removed (they only served porting from Genesis; RoboARC 1.0.0 at main 153eb8cf01 keeps them): the Genesis-era task
  texts and tools (`tools/release/templates/`, `package_text/`, `package_text_tool.py`, `scripts/roboarc/port_package.py`),
  the Genesis `-i00` references (`tools/references/oracles_genesis/`), the Genesis release generator
  (`records/genesis_release/`), the Genesis base image's recipe and libuipc build files, and the Genesis kinematics
  comparison test. `release_rebuild.py` gives a new base its template's final packages.

### The IKEA images
- `roboarc-ikea:1.1.0` and `roboarc-ikea-sim:1.1.0` (`ikea/images/ikea-mujoco/`): every input is pinned, the Python base
  image by digest (as before), the Debian packages at the 1.0.0 images' versions from the Debian snapshot archive of
  their build day, the wheels by sha256 (`requirements.lock`, `sim/nyx-requirements.txt`; until 1.0.0 the base resolved
  mujoco's dependencies at build time and the sidecar installed its Debian and pip packages unpinned). The sidecar is
  `FROM` the base by digest. `build.sh` builds both reproducibly (pinned BuildKit, SOURCE_DATE_EPOCH, rewritten
  timestamps) as OCI archives with fixed manifest digests: base `sha256:2c56d9226aaf…` (id `9f9cbab8aa74`), sidecar
  `sha256:f22a1e9042c7…` (id `cbd929360257`).
- The same files as 1.0.0 for everything the graded runtime loads (traced: privileged grading with its replay video,
  the standard sim service with Nyx, the agent's tooling): MuJoCo with the EPA patch, numpy, the scenes, OSMesa and
  ffmpeg, Nyx, torch and its CUDA libraries, the graphics loaders (the dynamic linker's cache lists the added and
  removed libraries; every library resolves to the same file); every Debian package keeps its version. The verifier's
  replay videos are byte-identical to 1.0.0's.
- Removed as unused: mujoco's absl-py, etils, fsspec, websockets and zipp (its viewer, studio and test code); in the
  Nyx venv triton, sympy, mpmath, jinja2, markupsafe, filelock, fsspec, cuda-toolkit, cuda-pathfinder, absl-py, etils,
  glfw and pip; curl in the sidecar; bytecode caches; the second copy of the scenes the 1.0.0 base carried (a `chmod`
  layer).
- The verifier's replay video libraries (OSMesa, imageio with ffmpeg, Pillow) are in the base image: the privileged
  tasks' verifier image installs nothing when it is built (until 1.0.x it installed them from the live Debian archive
  and PyPI at that moment). The privileged agent's container has them too. Each IKEA task's runtime binding names the
  new base image id, and its README a 1.1.0 Change Log entry.
- Push the two images from their OCI archives byte for byte (`images/push_all.sh`), so the registry's digests are
  the ones above; a `docker push` re-compresses the layers.

### Every image by digest; the client lock
- Every `FROM` and compose `image:` of the 226 task dirs names its image by manifest digest, from `images/pins.json`:
  `roboarc:1.1.0@sha256:85ea90032a55…` (the validated image `dadc2d710ad2`, exported without a rebuild into a
  deterministic OCI archive by `images/archive.sh`: its config digest is the image id and its layers' diff ids are the
  image's), `roboarc-ikea:1.1.0@sha256:2c56d9226aaf…`, `roboarc-ikea-sim:1.1.0@sha256:f22a1e9042c7…`, and as before
  `roboarc-monitor:0.1.0@sha256:ea1b0d4b8599…` and the client base. `ci_checks/check-image-pins.py` (CI job
  `image-pins`) fails a bare tag, a digest that is not a pin and another registry.
- `images/push_all.sh` pushes the four archives byte for byte (skopeo `--preserve-digests`; the archives checked
  against their pins before login, the registry's manifests and configs after), the IKEA base before its sidecar. It
  replaces `images/roboarc/push.sh` (a `docker push`, which re-compresses the layers) and `images/monitor/push.sh`.
- The standard agent's client (`scripts/eai_standard/client-requirements.txt`, copied into every standard task) is a
  hashed lock of the same versions, installed with `--no-deps --require-hashes`; a built client image has the same
  `pip freeze` and the same site-packages files as before.
- `release_rebuild.py` archives the release image and pins its digest before rendering; `harbor_check.py --tasks` runs
  a pinned task on the local image of the same id before the push.
- Each task's README has a 2026-10-21 Change Log entry; no instruction, grader, tooling, binding, instance or reference
  changed.

### Validation of 1.1.0
- The 99 references, replayed strictly in fresh processes as the privileged verifier does (two replays, rendering
  off): all succeed, both replays agree, every birth is the frozen instance's; 97 end on the freeze manifest's hash, and
  knot-untying-06-i01 and -09-i01 on the hashes 1.0.0 and the released 0.2.0 image reach (the manifest holds an earlier
  dry run's values for these two). The two bases without a reference build their frozen birth.
- Every library the traced runtime loads and every file of the kept Python packages is byte-identical to 1.0.0's; two
  builds of the recipe on different hosts give the same file contents.
- The agent API's Nyx frames, depth and camera data equal 1.0.0's (five families); the standard service with Nyx
  rendering reaches the recorded end states; `tests/api`: 29 simulator tests and 17 bit-identity tests pass in 1.1.0.
- `release_rebuild.py --onto 1.1.0` with its own instances builds the same image id and changes no file; the
  generators' `--check`, the invariants, unit tests and advisories pass.
- IKEA: the 12 references replayed in fresh processes in the 1.1.0 verifier image exactly as the privileged verifier
  does (two replays and the replay video): all succeed, both replays agree, and every end and settled state equals
  1.0.0's (and the recorded hashes). Harbor oracle 1 / nop 0 on all 12 scenes in both modes (48 trials) with the 1.1.0
  images; every oracle ends on its recorded state. Every file the traced runtime loads is byte-identical to 1.0.0's;
  three builds (two from fresh clones) give byte-identical archives. Nyx's RGB frames of a moving scene differ from
  run to run on one host in 1.0.0 as in 1.1.0 (hundreds to tens of thousands of pixels, by up to 25 levels, between two
  runs of the same image; the frames at reset are identical), so they are compared by that spread; depth,
  proprioception and every physics state are identical. `tests/api` in `roboarc-ikea-sim:1.1.0` with a GPU: 29 passed
  (the IKEA reference bit-identity tests for lack and svallet among them), 19 skipped (puzzles).

## 1.0.1 (2026-10-19)

The task set's version 1.0.1 (`scripts/roboarc/release.json` `version`; the README Release line of every puzzle task).
The images stay `ghcr.io/embodied-first-exam/roboarc:1.0.0`, `roboarc-ikea:1.0.0` and `roboarc-ikea-sim:1.0.0` (the same
ids); the resource monitor's image is now RoboARC's own build of the same files (below). The graders' and the sim
services' logic, the pinned tooling (`rcb_topogym.py`, `render_bridge.py`,
`nyx_direct.py`, `rcb_ikea.py`), the runtime bindings, the frozen instances, the IKEA scenes and the references are
unchanged. Every task has a new DIGEST and a Change Log entry in its README.

### The task texts (all 226 tasks)
A re-render from the cleaned generator inputs; text only for the puzzles:
- **Citations of this repository's paths**: `docs/PROTOCOL.md`, `docs/STANDARD_MODE.md`, `docs/TAXONOMY.md`,
  `docs/simulator/TOPOGYM_PROTOCOL.md`, `tools/release/release_rebuild.py`, `images/roboarc/` and the base image's
  recipe, instead of the development tree's `docs/rcb/...`, `images/topogym/`, `scripts/robotangle/...`,
  `scripts/task_dynamics.toml`. No task text cites a missing path any more except inside files whose bytes stay as
  released (the pinned tooling, the shared eai-standard and monitor layers, IKEA's `rcb_ikea.py` / `ikea_nyx.py`);
  [docs/rcb/README.md](docs/rcb/README.md) maps those.
- **No internal notes**: the resources' `measured` strings, the no-oracle bases' unfilled "PR #N", the Change Log
  entries citing internal documents, the graders', sim service's and oracle player's docstrings naming the earlier
  repository's RoboWits files, issue numbers and GPU-physics claims, IKEA's owner / board / pull-request notes, an
  internal runner script and a branch name.
- **`[task].name` = `roboarc/<task dir>`** (Harbor's `org/name`; it was `robot-coding-bench/<task dir>`); the Harbor
  dataset name is `roboarc/roboarc` (`scripts/harbor/dataset_manifest.py`). `harbor run -p tasks -i <dir>` is
  unchanged (it matches the directory name). The Python API accepts both forms.
- For the puzzles, `instruction.md` is unchanged and every changed `.py` file has the same docstring-free AST as in
  1.0.0 (the audit below); puzzle results graded under 1.0.0 stand.
- **IKEA's verification prose** names machine classes ("a Ryzen 9 7950X workstation", "a dedicated Xeon Gold 6230R
  core", "one process at a time") where the clean-up had left host labels and broken phrases ("host B", "a dedicated
  the cluster Xeon ..."): the rendered prose fields of `ikea/scripts/ikea_mujoco/evidence.json` and
  `standard_evidence.json`; every number, hash, image id, date and run name is unchanged (21 IKEA READMEs).

### IKEA MuJoCo on the repository's task layer, under `tasks/`
The 24 IKEA task dirs moved from `ikea/tasks/` to `tasks/` (same directory names) and are rendered with the
repository's task layer (`scripts/task_conventions.py`), like the puzzles: the instruction's shared closing sections
(`## External material`, `## Resource limits`, `## Time limit`), the resource-monitor sidecar, declared resources,
limits, caps (`scripts/task_resources.toml`, rules `^roboarc-ikea-`: privileged CPU only, the standard sim one GPU; the
agent's container 4 CPUs / 10 GB, the standard sim 4 CPUs / 8 GB; the privileged verifier keeps 4 CPUs / 8 GB for its
two replays and the video replay) and seeds, a `DIGEST`, and the one egress allowlist (`scripts/set_agent_egress.py`:
the model APIs and the package registries, which the instruction already named; IKEA's 1.0.0 allowlist had the model
APIs only). `ikea/scripts/task_conventions.py` and `ikea/scripts/set_agent_egress.py` are gone; IKEA's live-grading
2.4 pair is `scripts/eai_standard/v2_4_live/` (bytes unchanged), its own variant in `check-tooling-identity.sh`. Every
`ci_checks/` check covers the IKEA tasks. **IKEA results before and after 1.0.1 are not comparable**: the instruction
and the egress changed (the graders, scenes, references and images did not).

### The resource monitor's image: RoboARC's own
Every task's `rcb_monitor` service pinned `ghcr.io/mll-lab-nu/efe:monitor-0.1.0@sha256:f2c910f5…`, a package of
another group that RoboARC does not control. It now pins
`ghcr.io/embodied-first-exam/roboarc-monitor:0.1.0@sha256:ea1b0d4b859998d6a47e2cb7a72c3a9ae85870b32c75de1bc1448e387da07093`
(`scripts/rcb_monitor/inject.py` `IMAGE`), built from `images/monitor/`:
- **The same image contents**: the base is pinned by digest (`python:3.12-slim-bookworm`, Python 3.12.15), and the
  five layers have the earlier image's diff ids (the filesystems are equal file for file); the environment, working
  directory and command are the same; the config adds three OCI labels (title, version, source).
- **Reproducible**: `images/monitor/build.sh` builds an OCI archive with a pinned BuildKit, no attestations and a fixed
  `SOURCE_DATE_EPOCH`; two builds are byte-identical, so the digest the tasks pin is known before the push.
  `images/monitor/push.sh` copies that archive unchanged (skopeo `--preserve-digests`) and checks the registry's
  digest. [images/monitor/README.md](images/monitor/README.md).
- The tasks change only in that `image:` line, their 1.0.1 Change Log entry and DIGEST. `check-protocol.py` accepts
  base images from `ghcr.io/embodied-first-exam/` only. The Genesis base image (a build-time input of
  `roboarc:1.0.0`, `source/images/robotangle/Dockerfile.mujoco`) stays in the `efe` package: no task runs it.

### Usage recording is optional; licence
Usage recording is optional: an official run no longer needs the usage gateway (PROTOCOL.md 'Changes from the shared
protocol', H.6). The runner's `GATEWAY=1` is replaced by `USAGE_GATEWAY=<dir>`; results record `usage_gateway`,
`usage_source`, `usage_recorded`, `token_budget`. Licence: MIT (LICENSE, LICENSES/THIRD_PARTY.md).

### The raw simulator interface: the Python package `roboarc`
`roboarc.make(task_id)` returns a Gymnasium-compatible environment for every task (the 101 puzzle bases and the 12 IKEA
tasks) over the graded tooling itself (`rcb_topogym.Session`, `rcb_ikea.Session`): the frozen instance at `reset()`,
one control step of the tooling per `step()`, the tasks' own limits (clamp, never fail) and success predicates, Nyx or
MuJoCo-raster camera images, proprioception in the standard mode's robot frame, state snapshots, and the privileged
verifier's replay. Design, conventions and test results: [docs/API.md](docs/API.md); examples in `examples/`, tests in
`tests/api/` (the cheap ones run in CI, job `api-tests`). Each tested reference stepped through it ends on its recorded state
hash (one base per puzzle family and two IKEA scenes; with Nyx rendering on as well). The package changes no task,
image or graded file.

### Repository clean-up
The repository now holds RoboARC only and is laid out for its two audiences: `source/` is the image build context,
development tools are in `tools/`, records in `records/`, the simulator's internals in `docs/simulator/`, the Harbor
runner in `scripts/harbor/`; new `docs/RUNNING.md` (a Harbor guide), `docs/DEVELOPING.md`, one public page per family
in `docs/families/`, and a rewritten README. The generator's packages are the shipped texts (`mk_tasks.py` halved); the
asset tools check every instance source against its pin; CI runs on `dev/**` too, checks the docs' paths and links, and
runs the simulator source hash. Leftovers of the shared repository (other benchmarks, the eai-standard 2.2 layer,
lab-side runner defaults, internal notes) and duplicate copies (references, tooling, image recipes) are gone; the
session oracles of the orbits and slotted-post-ring references are restored. The clean-up itself changed no task byte
(the re-render above did); the images rebuilt from the cleaned tree have the 1.0.0 ids. Path moves: [PROVENANCE.md](PROVENANCE.md#after-100-the-clean-up).

### Validation of 1.0.1
On a fresh clone (2026-10-20, a shared Ryzen 9 7950X / RTX 4090 host, load 25-60):
- Both generators' `--check` (0 differences), `run_invariants.sh` (226 tasks), `check-unit-tests.sh --all`,
  `run_all.sh` ("all checks passed"), `check-tooling-identity.sh` (226 task dirs, 26 shared files, the 2.4 layer in
  113 standard dirs), `check-doc-paths.py`, `tools/release/source_hash.py --check` (SOURCE.json `c9171f79…`), and every
  job of `.github/workflows/checks.yml` run literally in its own `python:3.12` container, `api-tests` included.
- A diff audit of every changed task file against 1.0.0: every changed `.py` file of the 202 puzzle and 24 IKEA task
  dirs (474 files) has the docstring-free AST it had in 1.0.0; the puzzle `task.toml`s differ only in `measured`,
  `oracle_evidence` and `[task].name`, their shell scripts, Dockerfiles and compose files only in comments; no puzzle
  `instruction.md` changed; the IKEA runtime bindings and other JSON data are unchanged apart from text fields.
- Harbor oracle and nop, both modes: one base per puzzle family (14 bases, 56 trials, rendering off; the blanket bases
  with their CUDA cloth on the GPU) and all 12 IKEA scenes (48 trials, rendering on). All 104 scored as expected
  (oracle 1, nop 0, no infrastructure error), and every oracle ended on its recorded state hash (the puzzles' freeze
  manifest; IKEA's privileged end and settled hashes and standard live hashes). The IKEA privileged verifiers used at
  most 26 % of their timeouts under the new caps (stenerik 5036 s of 19440 s, kyrre 4365 s of 25440 s).
- The `roboarc` package's simulator tests against the re-rendered tasks: 29 passed in `roboarc:1.0.0`, 27 passed and 2
  skipped (puzzle-only) in `roboarc-ikea-sim:1.0.0`.
- The monitor image and the prose fix (2026-10-20, the same host): `images/monitor/build.sh` run four times (three times
  on one checkout, once on a fresh clone) gave byte-identical archives, manifest `sha256:ea1b0d4b…`; its filesystem equals
  the earlier image's file for file (paths, modes, owners, sizes, contents, link targets); `push.sh` into a local test
  registry stored exactly that digest. Every CI job in its own `python:3.12` container on a fresh clone passes,
  as do the gates listed above. Harbor oracle and nop, both modes, on interlocked-claws-03, blanket-fold-01 and IKEA
  lack with that image (12 trials): all scored as expected, every oracle ended on the same state hashes as above, and
  the monitor wrote the same record files and summary fields as before, no violation.

## 1.0.0 (2026-10-19)

The topological puzzles: 101 bases (202 task directories) in 14 families, image
`ghcr.io/embodied-first-exam/roboarc:1.0.0`. IKEA MuJoCo: 12 tasks (24 task directories), images
`ghcr.io/embodied-first-exam/roboarc-ikea:1.0.0` and `roboarc-ikea-sim:1.0.0`.

The tasks' graded content is byte-identical to the V3 release tree (dev/robotangle acfcec1089, image
`eai-roboarc:0.2.1`). Only the image references and the version labels changed (see "Repository" below). The image's
filesystem is identical to 0.2.1's (image id 41a17481d67a): the same layer digests, and new labels only.

### IKEA MuJoCo joins RoboARC
The IKEA MuJoCo furniture-assembly benchmark is part of RoboARC from 1.0.0: 12 tasks, 24 task directories under
`ikea/tasks/`, renamed into the RoboARC namespace: `ikea-<scene>-i00-<mode>` -> `roboarc-ikea-<scene>-i00-<mode>` (the map
is below). Scenes by Xiaoyu's team (Xiaoyu, Zhen Wu). Images
`ghcr.io/embodied-first-exam/roboarc-ikea:1.0.0` and `roboarc-ikea-sim:1.0.0`, which are the V3 images
`eai-ikea-mujoco:0.3.2` (5d5090ea1910) and `eai-ikea-mujoco-sim:0.4.2` (050cd45d70f6) under new names, with the same
ids. Its V3 content (development branch dev/robotangle_mujoco 5cb4114f10):
- **New tasks:** stenerik, burvik, uppsta and kyrre (the review was approved; kyrre ships with the native-cylinder
  exception below).
- **Franka Hand slew:** `GRIPPER_WIDTH_SPEED_M_S` in every copy of `rcb_ikea.py` (one sha256, 3a63a33b6803), so the hand's
  width command moves at its real speed. New references for the eight earlier tasks were built within the slew (lack,
  eket, ikea365, vesken, gladom, variera and stegesund; svallet's ViperX reference is unchanged).
- **Welded-part contact switch-off** in the scene runtime (`ikea_runtime.py`; kyrre's and uppsta's seated parts).
- **kyrre: MuJoCo's native cylinders**, the one exception to the 32-facet prism rule (owner, 2026-10-19). The stress
  A/B showed 1.15 % non-jerk breakdowns with native cylinders against 2.3 % with the prisms, and no breakdown involved
  a screw. The verifier limit is set by the slow-class replay (a Xeon Gold 6230R core, 8283 s): 25,440 s.
- **Nyx start-up retry** in the standard sidecar's render process (`ikea_nyx.py`, the same file in every standard task;
  sha256 036be49c38e6 in V3; 37a773230f41 in 1.0.0, after the rename changed a path in its docstring and
  a host name left a comment).
- **EPA-patched libmujoco.so**, the same 14-byte patch as the puzzles' image.
- Timing is gated on the i9 / Zen 4 host class. Slow-Xeon times are reported only: gladom, variera, stenerik, uppsta
  and kyrre are over the 40 % bar only there.

Task names (the generators' `name()`; physics, references, hashes and grading unchanged):

| until 1.0.0 | 1.0.0 |
|---|---|
| `ikea-burvik-i00-privileged` | `roboarc-ikea-burvik-i00-privileged` |
| `ikea-burvik-i00-standard` | `roboarc-ikea-burvik-i00-standard` |
| `ikea-eket-i00-privileged` | `roboarc-ikea-eket-i00-privileged` |
| `ikea-eket-i00-standard` | `roboarc-ikea-eket-i00-standard` |
| `ikea-gladom-i00-privileged` | `roboarc-ikea-gladom-i00-privileged` |
| `ikea-gladom-i00-standard` | `roboarc-ikea-gladom-i00-standard` |
| `ikea-ikea365-i00-privileged` | `roboarc-ikea-ikea365-i00-privileged` |
| `ikea-ikea365-i00-standard` | `roboarc-ikea-ikea365-i00-standard` |
| `ikea-kyrre-i00-privileged` | `roboarc-ikea-kyrre-i00-privileged` |
| `ikea-kyrre-i00-standard` | `roboarc-ikea-kyrre-i00-standard` |
| `ikea-lack-i00-privileged` | `roboarc-ikea-lack-i00-privileged` |
| `ikea-lack-i00-standard` | `roboarc-ikea-lack-i00-standard` |
| `ikea-stegesund-i00-privileged` | `roboarc-ikea-stegesund-i00-privileged` |
| `ikea-stegesund-i00-standard` | `roboarc-ikea-stegesund-i00-standard` |
| `ikea-stenerik-i00-privileged` | `roboarc-ikea-stenerik-i00-privileged` |
| `ikea-stenerik-i00-standard` | `roboarc-ikea-stenerik-i00-standard` |
| `ikea-svallet-i00-privileged` | `roboarc-ikea-svallet-i00-privileged` |
| `ikea-svallet-i00-standard` | `roboarc-ikea-svallet-i00-standard` |
| `ikea-uppsta-i00-privileged` | `roboarc-ikea-uppsta-i00-privileged` |
| `ikea-uppsta-i00-standard` | `roboarc-ikea-uppsta-i00-standard` |
| `ikea-variera-i00-privileged` | `roboarc-ikea-variera-i00-privileged` |
| `ikea-variera-i00-standard` | `roboarc-ikea-variera-i00-standard` |
| `ikea-vesken-i00-privileged` | `roboarc-ikea-vesken-i00-privileged` |
| `ikea-vesken-i00-standard` | `roboarc-ikea-vesken-i00-standard` |

Its earlier releases were inside the shared repository: 8 tasks (16 dirs) with images `eai-ikea-mujoco:0.3.0` /
`-sim:0.4.0` (embodied-first-exam #299, 2026-10-17).

### New bases (reviewed and approved by the benchmark owner, 2026-10-18)
- **orbits-02-i01** (orbits, easier): runs on a 0.3125 ms MuJoCo step (solver_substeps 8), see
  [docs/simulator/families/orbits.md](docs/simulator/families/orbits.md). Reference end state b9c68d6502105df4.
- **knot-untying-07-i01** (knot_untying rd, easier): a re-timed route. The reference replays in 28.1 % of the 3 h budget
  (3037 s on a Ryzen 9 7950X), under the knot family's 30 % bar. End state 8ae062ea4c733dbc.
- **blanket-flatten-09-i01** (blanket_flatten, hard): the Genesis release's hard tier, ported to the deterministic GPU
  cloth solver (`blanket_vbd_gpu`), route hard5. End state 6a9c4796da46cf47.

### Changed
- **blanket-fold-05-i01: a new reference** in both modes. It is the released route, re-recorded with the oracle rate
  limiter's guard-stop clamp, so every arm joint step stays within the standard service's joint-speed rule (max
  0.0390 rad). It has 10195 rows (was 17189, one of them over the rule), end state 4f1b6724fc0a6512, and standard rows
  = privileged rows. Timing exception (owner, 2026-10-18): <= 30 % of the 3 h budget (2930 s = 27.1 % on a Zen 4
  trial host). Stress: 1 breakdown in 371 episodes, of the pressed-pile class, counted as an agent-caused warning
  (~0.27 %, under the ~1 % ceiling).
- **MuJoCo 3.14.0 + the EPA horizon-overflow fix.** MuJoCo 3.14.0's EPA keeps the polytope horizon in fixed 24-entry
  arrays and overflows them on complex convex contacts (a SIGSEGV in `mj_narrowphase`). Upstream fixed this in 3.15
  (google-deepmind/mujoco 6d662aed). The image carries a 14-byte patch of the wheel's own `libmujoco.so.3.14.0` that
  backports only that fix (`source/images/robotangle/mujoco-3.14.0-epa/`; sha256 5e7623e3… -> 52e2a24e…). A source
  build is not bit-identical to the wheel, so it is not used. Gate: every released reference reaches its recorded end
  state on the patched library (Zen 4, and Intel for the CPU bases; stock = patched), and 1150 / 1150 stress episodes
  were identical. So 0.2.0's results stand.
- **Nyx start-up retry** (`nyx_direct.py` sha256 52bb406f…, in every task's tooling copy). A renderer build that fails,
  for example on a full shared GPU, is built again after 10 s and then 30 s (x 1-1.5), at most 3 attempts, and each
  failed attempt is logged (the standard service's `nyx_ready` event counts them). The half-built renderer is kept,
  never shut down. A crash inside the build still goes through crash recovery.

### Evidence
V3 build (2026-10-18, image 0.2.1):
- Harbor on the image, both modes, rendering off: orbits-02, knot-07, blanket-flatten-09 and blanket-fold-05. Every
  oracle scored 1 at its reference's end state (privileged deterministic and instance_exact; standard live = replay),
  and every nop scored 0.
- Strict fresh-process replays of one base per family plus the hold-only bases (16 / 16), all at their gate end hashes.

From this repository (2026-10-19, a fresh clone, image `roboarc:1.0.0` built by `images/roboarc/build.sh`):
- The image has the same two layer digests as 0.2.1. All 153,404 filesystem entries are equal (path, mode, owner,
  mtime, size, sha256).
- `mk_tasks.py --check` (202 match), `run_invariants.sh`, unit tests, check-video, check-resources and check-cutoffs
  all pass. `run_all.sh` passes too: the invariants hold, and the advisories report nothing beyond warnings.
- Harbor, oracle and nop in both modes, rendering off, on interlocked-claws-03, hedgehog-in-cage-01 and cast-coil-01:
  12 / 12 as expected (oracle 1 at the reference end, deterministic; nop 0).
- Strict fresh-process replays in the new image (both replays of each): one base per family plus the hold-only bases
  (16) and the four V3 items (orbits-02, knot-untying-07, blanket-flatten-09, blanket-fold-05). All 20 end at their
  recorded hashes, 15 + blanket-flatten-08 equal to the 0.2.1 spot replays, and the V3 items at b9c68d6502105df4,
  8ae062ea4c733dbc, 6a9c4796da46cf47 and 4f1b6724fc0a6512.
- IKEA: both generators' check modes give 0 differences. Both images rebuilt from `ikea/` have the V3 images' ids
  (5d5090ea1910, 050cd45d70f6). The unit tests pass in
  the image. Harbor oracle and nop on svallet and lack in both modes, before and after the rename: 8 / 8 each time
  (oracle 1, nop 0). Fresh replays of svallet's and lack's references from the renamed dirs give the 0.3.2 records'
  end and settled hashes.

Not yet in: the Intel replays of the 13 blanket bases on the GPU. They gate the public image push.

### Repository
- A standalone repository: the RoboARC tasks, generator, simulator source, image recipe, CI checks and the shared task
  layer they use (provenance: [PROVENANCE.md](PROVENANCE.md)).
- Image references: `ghcr.io/mll-lab-nu/eai-roboarc:0.2.1` -> `ghcr.io/embodied-first-exam/roboarc:1.0.0`. The
  monitor sidecar and the Genesis base image are referenced by their unchanged digests in the consolidated package
  `ghcr.io/mll-lab-nu/efe` (`monitor-0.1.0`, `robotangle-0.2.22`), because the old `eai-*` packages no longer exist.
- The task README's release line names RoboARC 1.0.0 and the simulator source commit.
- IKEA task dirs: only the image references changed (the Dockerfiles' `FROM`, the compose `image:` lines, the
  bindings' `reference`, the README rows). The host names in their evidence lines were replaced by host classes.

## 0.2.1 (2026-10-18, V3 build; not released before 1.0.0)

The V3 build on the development branches of the shared repository (image `eai-roboarc:0.2.1`, 41a17481d67a, HF assets
revision 9f34f894a454). Its content is released as 1.0.0 above.

## 0.2.0 (2026-10-17, "V2"; embodied-first-exam #301)

98 bases (196 task directories). Image `eai-roboarc:0.2.0`.
- New bases: knot-untying-06-i01 (rd very_easy) and knot-untying-09-i01 (rd medium; 27.4 % of the budget, under the
  knot family's 30 % bar), blanket-flatten-07-i01 (easy) and blanket-flatten-08-i01 (medium), both ported to
  `blanket_vbd_gpu`.
- Standard references within the standard service's joint-speed rule (0.04 rad per control step) for 11 bases.
- buzz-wire-3d-03 / 04: references where the release had none.
- knot-06 / 09 re-frozen without the lab start's provenance in instance.json. The physics is unchanged; the births
  changed.
- Reference meta hygiene: no host paths in the references' metadata.

## 0.1.0 (2026-10-14; embodied-first-exam #186 / #187)

The MuJoCo release. 94 bases (188 task directories) replace the Genesis release of the same puzzles (RoboTangle,
`eai-robotangle` 0.2.22, whose results are not comparable). Image `eai-roboarc:0.1.0`, HF assets revision 88e0136e1b97.
- MuJoCo 3.14.0 on the CPU in float64 (blanket families: a deterministic CUDA cloth step), and numpy / OpenBLAS SIMD
  masking in every container that runs a session, so every CPU class builds the same frozen bytes.
- The 3 h agent budget in both modes. Standard mode on eai-standard/2.4, with no world coordinates in standard texts.
- Privileged grading replays need the strict goal within the agent's rows (the 2 s hold only confirms). Earlier
  privileged results can be re-graded with `scripts/roboarc/regrade.py`.
- Follow-ups on main: the image's own Harbor evidence (#203) and the lock rotor discs drawn where the physics has them
  (#218).
