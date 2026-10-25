# RoboARC

RoboARC's topological puzzle families on MuJoCo 3.14.0 physics, run by the simulator package `topogym`: 111 task
bases in 19 families, each in two modes, privileged and standard (222 Harbor task directories under `tasks/`). One page
per family: [../families/](../families/README.md). This page states the benchmark's own rules on top of the task protocol
([../PROTOCOL.md](../PROTOCOL.md)). The IKEA MuJoCo tasks have their own page
([ikea_mujoco.md](ikea_mujoco.md)).

Code identifiers keep the simulator's earlier names: the package `topogym`, the tooling `rcb_topogym.py`, `TOPOGYM_*`
variables, the family keys (`interlocked_claws`), and inside the image `/opt/robotangle/` (the instances and
`stage_instance.py`).

## Tasks

`roboarc-<family>-<NN>-i01-{privileged,standard}` (PROTOCOL §2.2):
- `<family>` with dashes (`interlocked-claws`); `[metadata].family` keeps the underscores.
- `<NN>`: two digits, the base's number in its family. A number is never reused: removing a task leaves a gap. The
  difficulty is not in the name: it is `[metadata].difficulty` in task.toml ([TAXONOMY](../TAXONOMY.md)).
- `i01`: the MuJoCo instance (the frozen scene every mode of the base runs from).

Robots: ALOHA 2 (two arms with parallel grippers) or a Franka Panda with a Robotiq 2F-85, per family. Every task shares
one tooling `rcb_topogym.py`, one grader per mode and, in standard mode, one sim service and the eai-standard/2.4 layer;
the family pages list each family's robot and success rule.

- **Physics**: MuJoCo 3.14.0 on the CPU in float64; the blanket families' cloth on a deterministic CUDA step
  (`blanket_vbd_gpu`). The image's `libmujoco.so` carries a fix of the EPA horizon overflow (a 14-byte patch of the
  wheel's library, `source/images/roboarc/mujoco-3.14.0-epa/`); every released reference reaches its recorded end
  state with it, on AMD Zen 4 and Intel CPUs.
- **Determinism**: numpy / OpenBLAS SIMD paths are masked in every container that runs a session, so every CPU class
  builds the same frozen bytes; a replay of the same actions from the frozen state gives the same state hash.
- **Rendering**: Nyx (RGB, optional depth), through `render_bridge.py` and `nyx_direct.py` next to every tooling copy,
  named in the task's runtime binding. A renderer build that fails (e.g. on a full shared GPU) is retried after 10 s,
  then 30 s, at most 3 attempts, each failure logged (the standard service's `nyx_ready` event counts them).
- **Grasping**: what the tooling grades is what the texts state: the contact-review values come from `CONTACT_RULE`,
  and the grasping rule is `Session.grasp_rules()` of the instance (the weld's slip force over 20 ms, 60 N ALOHA / 65 N
  Robotiq, and the Robotiq's release after 0.2 s without finger contact; the cloth families' assisted grasp of material
  patches). The texts of bed-bolt, clock-assembling, battery-replacement and wire-splicing add what that sentence leaves
  out and the simulator applies: the grippers start empty, the bolt is no grasp target while its thread is engaged, the
  cells' pinch rules, the wire's welds between the pads and its 0.1 J weld work guard; clock-assembling's text names the
  parts of its one entity. The texts' sentence is the release record's `grasp_rules`.

## Generator, instances and image

- `scripts/roboarc/mk_tasks.py` (`--check` in CI) renders the task dirs from `scripts/roboarc/`: `release.json` (one
  record per base: family, tier, instance, the facts the texts state, the validation evidence), `packages/`,
  `oracles/` (the reference trajectories), `assets.json`, `shared/` (the tooling and the graders).
  Generated tasks end with `scripts/task_conventions.py` normalize ([../DEVELOPING.md](../DEVELOPING.md)).
- `packages/<base>-<mode>/` holds each task's texts as they ship (task.toml, README.md, instruction.md, the Dockerfiles
  and compose files, tests/test.sh, goal images). A value that comes from another source is a field, `{{NAME}}`, which
  the generator fills: the image tag (`images/roboarc/VERSION`), the agent egress (`scripts/set_agent_egress.py`), the
  tooling version, the oracle level and the `release.json` records (reference solution, nop run, provenance, Change
  Log), the privileged row cap and verifier budget, and the standard service's time per call. Before writing, the
  generator checks the filled texts against the facts they must state (the tooling's contact rule and breakdown depth,
  `release.json`'s grasping rule and step time, the difficulty, the family the grader runs). A text change is an edit
  of the package, then a re-render.
- A new base starts from the final packages of its template, a sibling base (`tools/release/release_rebuild.py`
  copies them under the new name when the base has none); its texts are then edited in `scripts/roboarc/packages/`.
  Until 1.0.x new bases started from Genesis-era texts (`tools/release/templates/`, `package_text/` and
  `scripts/roboarc/port_package.py`, in RoboARC 1.0.0).
- The frozen instances (about 700 MB) are not in git: they are in a Hugging Face dataset, pinned file by file in
  `scripts/roboarc/assets.json`. The image carries them, so running tasks needs no dataset access.
- Appearance seeds (from 1.2.0): each base has seeds 1 and 2 beside its frozen instance, the same scene
  and physics in other looks, derived from it (`tools/release/derive_seed_instance.py`), with a binding, a re-stamped
  reference and, where the task shows goal pictures, pictures of their own; the steps are in
  [roboarc_seeds_release.md](roboarc_seeds_release.md).
- Image: `ghcr.io/mll-lab-nu/roboarc:<images/roboarc/VERSION>`, built by `images/roboarc/build.sh` from
  `source/` (recipe `source/images/roboarc/Dockerfile`, the instances as a named build context) on the Python base
  image pinned by digest, with Debian packages and wheels pinned by version and hash (images/roboarc/README.md). Every task is bound to the runtime the image installs (`release.json`
  `runtime`; the build checks it).

## Time budget

10800 s (3 h) wall clock in both modes and every family (each task's `[agent] timeout_sec`; PROTOCOL §2.12), except
wire-splicing-01-i01: 21600 s (6 h) by the owner's exception (`scripts/task_conventions.py` `CUTOFF_EXCEPTIONS`; its
privileged verifier 45000 s, its standard collect hook 32400 s); `cutoff_s` is part of the configuration results are
ranked by, so its trials form their own group. The texts
say "the time limit (`## Time limit`)". The standard service's clock (`meta["time_left_s"]`, `RCB_WALL_CLOCK_S`)
counts from the agent's first request to the service (logged in `episode.jsonl`, kept over a restart), so the
instruction has the common closing sentence and no sentence of its own about the service's clock. The service ends the
episode with reason `timeout` when that count reaches 0, which is never before the harness's cutoff.

## Grading replays

Privileged mode grades the agent's `trajectory.npz` by two replays in a fresh process from the frozen state. The six
families with a grading hold (blanket-flatten, knot-untying, lock-picking, orbits, rope-threading, slotted-post-ring;
`rcb_topogym.REPLAY_TOLERANCES`) replay a trajectory with the instance's own thresholds and, if the episode has not
ended after the last row, hold the final targets for up to 2 s; during that hold blanket-flatten accepts 92 % coverage
and 80 mm/s cloth speed (95 % and 30 mm/s in the rows) and knot-untying a rope speed 0.25 mm/s higher. Standard mode
grades the live episode strictly and lets its replay confirm in the hold.

**The rule.** Privileged success needs both grading replays to succeed AND at least one of them to reach the goal with
the instance's own thresholds within the trajectory's rows (`tests/grade.py`: `strict_replay_required` 1,
`strict_goal_reached`, `strict_goal_row`; `success_in_hold_only` 1 marks a trajectory whose replays succeeded only in
the hold, not credited). The instruction says where `s.replay(actions, grading=True)` reports it
(`["grading"]["strict_success"]`). The hold exists because a recording that stops on its own success has no margin
for the replay's confirmation; requiring the strict goal makes privileged mode ask for what the task text and standard
mode ask for. This rule holds since 2026-10-14; before it, privileged success did not require the strict goal.

**Re-grading results recorded before 2026-10-14** (nothing is re-run). `scripts/roboarc/regrade.py JOBS_DIR ...
[--replay] --out regrade.jsonl` reads every RoboARC privileged trial under the Harbor jobs trees and gives its recorded
verdict, the old rule's and the new rule's:
- without `--replay`, from the trial's recorded `verifier/reward.json` alone: the new verdict is the recorded success AND
  `strict_goal_reached` (the earlier grader recorded it next to the verdict). This is exact: the replays are
  deterministic on the host class they ran on, and the rule only adds that condition;
- with `--replay`, it also replays the recorded `artifacts/app/output/trajectory.npz` with the current task's own
  `tests/grade.py` in the task's verifier image (docker, GPU) and reports both verdicts from that run; a trial whose
  recorded success the replay does not reproduce under the old rule is `inconclusive` (re-grade it on the host class it
  ran on).

Only the 94 tasks of the first release (0.1.0) are re-graded; a trial of any other task is listed as `skipped` with
the reason. Standard trials are skipped too: their grading did not change.

**Validation** (2026-10-14, an RTX 4090 host, the release image of the time, Harbor `-a oracle` / `-a nop`) with the
privileged `tests/` of this rule: the privileged oracle scores 1 with `strict_goal_reached` 1 on one task of each
family with a hold (blanket flatten, knot untying, lock picking, orbits, rope threading: tasks of the time, several
of them removed in 2.0.0; slotted-post-ring-01) and on one interlocked-claws task (a family without one),
deterministic, no infra or grader errors; nop scores 0 on that blanket-flatten task and slotted-post-ring-01
privileged. The rule itself, by construction: that blanket-flatten task's reference without its last 8 rows (a
recording that stops 8 control steps before its own success; the replay confirms after 8 steps of hold) is graded 1 by the earlier `grade.py` (`all_success` 1, `strict_goal_reached` 0) and 0 by this
one (`all_success` 1, `success_in_hold_only` 1), both deterministic. Without its last 40 rows it fails both (the hold's
2 s do not complete the confirmation).

## Safety readings in standard mode (`meta["safety"]`)

The standard service publishes `meta["safety"]` in every response, a declared exception to "no object contact forces"
(PROTOCOL H.3). The benchmark's physical-validity policy judges the agent on these readings (a success with a deep task
contact is held for human review instead of credited, a soft body that stays collapsed ends the episode, a simulator
breakdown shortly after forcing is the agent's failure), so an agent that could not see them would be judged on
something it cannot measure. The fields are exactly what `rcb_topogym._SafetyMonitor.read` returns (maxima since the
previous response, each next to its limit), and nothing else of the simulator's contact state is published:

| Field of `meta["safety"]` | What it is | Families |
|---|---|---|
| `rules`, `scope` | the policy's revision; "since the previous read" | all |
| `forcing` | `max_press_force_n` (the largest total contact force an arm's gripper, or a part it holds, pressed on anything else with), `max_drive_fraction_while_pressing`, `longest_forcing_s`, `events`, the last three events (each: the arm and joint, the control step, the drive fraction and cap, the force, and in words what pressed on what, e.g. "a part it holds" on a named task part), and the thresholds (`contact_n`, `near_cap_fraction`, `event_after_s`) | all |
| `task_contact_mm` | the deepest contact between task parts (`max`), the longest time above the sustained limit, and the review and breakdown limits | rigid |
| `held_finger_contact_mm` | the deepest contact between a finger and the part it holds, with its review limits | rigid |
| `review` | the count of review events and the last three (each: the kind, the control step, the names of the two task parts in contact, the depth in mm and the force in N); a success with one is held for review | rigid |
| `soft_volume_ratio` | the lowest volume ratio of each soft body and the longest time at or below 0 | soft (rope, cord, cloth) |
| `attachment_stretch_mm` | the worst cord attachment stretch and the longest time above its limit | soft |
| `table_guard` | the control steps whose gripper target the table guard stopped, with the last events | where the guard applies |
| `ik_jumps` | the last IK solutions that lay far from the current joints (`ee_pose` actions) | when one occurred |

These are forces, depths, ratios and the names of the parts involved, never a pose: no field says where an object is.
Names are allowed in standard mode; positions and poses in the simulator's world frame, the table height and physics
parameters are withheld. The service's `robot_felt` passes exactly these fields (`SAFETY_PUBLISHED`);
`scripts/roboarc/tests/test_server_requests.py` checks the list and the names. A change to this list is a change of the
declaration and goes through the protocol-change process (PROTOCOL §6).

## Standard mode, world frame, network

`eai-standard/2.4` ([../STANDARD_MODE.md](../STANDARD_MODE.md), Part 4). Every text the agent sees calls the service's
frame the world frame and defines it in words; it is the robot frame: origin at the robot's kinematic root link on its
mounting, axes as the simulator's world (+z up). The table height and the robot base's world position are never
disclosed. The proprio TCP poses, `ee_pose` targets, `/kinematics` (IK targets, FK poses) and the camera extrinsics
(static mounts, per-observe calibration, per-frame `extrinsic_cv`) are in it; `robot_spec.base.pose` is the identity;
no text and no `/spec` field states a world coordinate of the mounting. `joint_position` rows, physics, graders and
state hashes do not depend on the frame. Privileged mode keeps the simulator's world frame (its tooling is the
simulator); its texts state no table height or robot base position either. No world coordinates in standard texts
(`ci_checks/check-world-frame-texts.sh`). Agent egress: the allowlist (PROTOCOL Appendix H.6).

Service details: `max_frames_per_call` = 2000, per-frame depth counted as four images; where a call's length is known
only while it runs (`ee_pose` rows) the call stops taking frames at the bound and `meta.frames_skipped` counts them.
Contact and safety telemetry is published in `meta["safety"]` (above).

## Resources

Every task declares 2 GB of GPU memory as its minimum (measured peak about 1.1 GB on 24 GB cards), and 4 or 6 CPUs and
7 to 10 GB of memory by family class (rigid, rope / chain, cloth), with limits and caps (`scripts/task_resources.toml`; each task's README, "Compute resources").
Every trial needs one NVIDIA GPU with RT cores (Nyx renders through Vulkan ray queries). On a GPU without them (V100, A100, A30, H100, H200)
standard mode does not run and privileged tasks are graded without the replay video; such runs are unofficial
(docs/RUNNING.md, "GPUs without RT cores").

## Easy set

15 bases are in the easy set ([../easy_set.toml](../easy_set.toml)): run and reported, but apart from the headline
numbers (PROTOCOL §4; RUNNING.md "Results").
