# roboarc-bed-bolt-01-i01-privileged

| | |
|---|---|
| Benchmark | RoboARC |
| Mode | privileged |
| Twin | `roboarc-bed-bolt-01-i01-standard` |
| Simulator / robot | `roboarc` / `aloha-2` |
| Family / instance | `bed_bolt` / seed 0 (allowed seeds: [0, 1, 2]; §2.13) |
| Oracle | `full` |
| Agent time budget | 10800 s; tokens: protocol default (PROTOCOL §5) |
| Base image | `ghcr.io/embodied-first-exam/roboarc:1.2.0` |

## Task
RoboARC bed_bolt (ALOHA 2, MuJoCo): produce one replayable trajectory. Drive the long bolt into the blind hole in the bed frame's corner block with the L-shaped hex key until its head is within 1.5 mm of the block face. Frozen scene instance (variation bed_bolt_crossbar, seed 0).

## Difficulty
Two 6-dof aloha (viperx) arms with parallel-jaw grippers on a fixed base with a 14-D joint-target action space at 40 Hz; success needs the task's predicate (above) held for 0.5 s. Grasping: assisted grasping (contact_weld_v1).

## Reference solution
`solution/solve.sh` copies `solution/trajectory.npz` (sha256 `bd3033a462f5e74c…`) to `/app/output/`: the MuJoCo reference, which its author replayed to success with a reproducible end state outside Harbor. No Harbor oracle trial of it has been run yet, so no score is claimed here. For an appearance seed S (1, 2) it copies `solution/trajectory.s<S>.npz` instead: the same rows with `instance_sha` set to that seed's birth (`solve.sh` picks the file whose stamp is the staged instance's `state_sha`).

## Verification
Open-loop replay of `/app/output/trajectory.npz` from the frozen instance (`/tests/instance.json`, staged from the image and checked against `tests/runtime_binding.json`) in a fresh process, twice; pass iff the
task's predicate holds after every replay (`tests/grade.py`). `/logs/verifier/replay.mp4` is rendered during the second
replay (two passes, no separate video replay). The physics runs on the CPU and is bit-deterministic on one host; the final-state hashes
(`deterministic`) are still diagnostic, as for the GPU families; the rebuilt frozen scene must match (`instance_ok`)
(robot_coding_bench's GPU-backend rule).
A trajectory may have at most 200000 rows (`RCB_MAX_STEPS`: two replays must fit the verifier's timeout; a longer one is
`bad_trajectory`).

`nop`: no record yet on this base (`harbor run -p tasks/roboarc-bed-bolt-01-i01-privileged -a nop`).

## Compute resources
Minimum allocation for one trial (docs/PROTOCOL.md §2.9; `[metadata.resources]` in `task.toml`). The agent phase counts every container of the trial at once (the agent's, and in standard modes the simulator sidecar); the verifier phase runs after it.

| | Agent phase | Verifier |
|---|---|---|
| CPU cores | 2 | 2 |
| Memory | 7 GB | 5 GB |
| GPU memory | 2 GB | 2 GB |

- GPU: required, any NVIDIA GPU; recommended: L40S 48GB / RTX 4090 24GB / L4 24GB (sm_86 or newer).
- Measured: the Genesis release's numbers, kept until the MuJoCo release's own peaks are measured (its Harbor runs; MuJoCo physics on the CPU, Nyx on the GPU); GPU 2 GB: owner 10-17, board 0150 (batch_eval_live trial peak 1.1 GB on 24 GB cards)
- Limits in the agent phase (docs/PROTOCOL.md §2.10; `[metadata.resources.limits]`, shown to the agent in `instruction.md`; always measured and reported, and exceeding one fails the trial only when the run sets `RCB_LIMITS_ENFORCE=1`), every container of the trial: 16 GiB GPU memory, 16 GiB memory, 512 processes, 10 GiB of new files under `/app`, `/tmp`, `/root`; 60 s within 120 s. Expected peak of the whole trial: 16 GiB GPU memory, 16 GiB memory. Limits measured: the Genesis release's numbers, kept until the MuJoCo release's own peaks are measured (its Harbor runs; MuJoCo physics on the CPU, Nyx on the GPU).

## Details
| | |
|---|---|
| Grasping | assisted grasping (contact_weld_v1) |
| Success rule | task-native success (goal_complete True; frozen parameters; terminal revision bed_bolt_seated_v1) |

### Goal
Drive the long bolt into the blind hole in the bed frame's corner block with the L-shaped hex key until its head is within 1.5 mm of the block face.

### Success requirements
- Drive the bolt in until its head is within 1.5 mm of the block face, its tip at least 83.5 mm deep in the hole, with the thread caught, and its axis within 2 degrees of the hole's axis.
- The hole is threaded. The thread catches when the bolt's tip is 2.5 to 8.2 mm inside the hole, within 1 mm of the hole's axis and 2 degrees of its direction, with the bolt's thread lined up with the hole's (a straight push lines up somewhere in that range; turning the bolt about its axis moves where); each turn then drives the bolt 5.7 mm in. A bolt the thread does not hold cannot be pushed in past the thread: not past 8.2 mm before it has ever caught, and afterwards not deeper than it has already been turned in; it catches again only there, in the same thread. With fewer than 2 turns engaged the thread cannot carry the bolt: when no finger and no key end holds it, it droops, and past 3 degrees it lets go and falls out.
- Once the thread has engaged, pushing or pulling the bolt along its axis does not advance it. Fingers cannot turn the bolt (they slip); use the key.
- Keep the bolt's and the key's linear speed at most 1 mm/s and angular speed at most 0.02 rad/s. Holding is allowed; the key may stay in the socket or anywhere else.
- Keep all final conditions satisfied together for at least 0.5 s of simulation time. If any final condition stops holding, the confirmation interval starts again. A brief qualifying pose is not enough; let the simulation advance through the full interval.

### Task failure conditions
- A task piece falling more than 100 mm below the tabletop ends the episode in failure. Finger contact with a held movable piece may penetrate at most 2 mm; holding requires both fingers of the same hand to contact the same piece, or an active assisted grasp by that hand. All other task contacts, including piece-piece, table, arm and gripper-housing contacts, may penetrate at most 1 mm. Exceeding either limit ends the episode in failure.

### Grader details (in addition to the requirements above)
- Contact penetration as this benchmark grades it (it replaces the "at most 1 mm / 2 mm" wording above): penetration is recorded, it never ends the episode by itself. A success with deep interpenetration (a task contact deeper than 3 mm, or deeper than 2 mm for at least 0.25 s: 100 consecutive physics ticks; for a finger contact with a held piece 4 mm, or 2 mm for 0.25 s) is not credited automatically but reviewed, with the recorded evidence and frames; a clean success is credited automatically. A task contact deeper than 10 mm is a simulator breakdown (below).
- Contact penetration is read from the contacts at the end of every physics tick (2.5 ms; 10 ticks per control step). "Held" is decided anew at every tick: a zero-force touch counts, and if one finger loses contact (with no active assisted grasp), the other finger is back at the non-held limits in the same tick.
- Robot safety (this benchmark's tooling): no gripper target goes lower than 15 mm below the tabletop; a commanded target that would bring any part of a gripper lower is stopped there, on the commanded path. The fingertips reach about 16 mm beyond the tool frame (TCP) along its approach axis, open or closed. Arm drive torques are capped (the effort limits in robot_spec), so an arm pushing against an obstacle stalls at that torque. An arm with any part more than 5 mm deep in a fixed object (a fixture, the table) is stopped where it is: its joint targets become its measured joints for that control step, so it stops pushing (the safety readings report these stops as static_guard).
- Forcing is recorded, not judged: an arm drive (not a gripper drive) at 95% of its torque cap or more while that arm's gripper, or a part it holds, presses on anything else with 20 N or more in total is forcing; 0.5 s of it on every physics tick is a forcing event in the evidence (`s.safety()`, `s.status()["forcing"]`), shown to the reviewer when a success is reviewed. Forcing does not end the episode and does not change the grade by itself.
- Simulator breakdown (a non-finite state or a task contact deeper than 10 mm): the episode ends as an infrastructure error, not your failure, unless an arm was forcing (even for one tick) within the 0.25 s before it: then it is your failure (forced_breakdown: what broke and what forced it).
- Grasp regions (assisted grasping, see Grippers above; one sphere per graspable part): bolt: radius 53.2 mm; key: radius 107.9 mm. Both finger contacts must lie inside the part's sphere; a part touched elsewhere is grasped physically. Each region's centre (in its link's frame) is in `/app/instance.json` under `simulator.grasp_assist.targets`.
- A failure is reported in `s.status()["episode_end"]` as "A task body fell below the table" or "forced_breakdown: <what>" (<what>: the readable reason); `s.safety()` reports forcing, contact depths, volume ratios and table-guard stops against these limits, and `s.status()["review"]` the deep task contacts that would hold a success for review.

### Provenance
- Frozen instance: a MuJoCo instance (`simulator.physics_engine` "mujoco", its scene frozen under `instance_assets/mujoco/`), built by TopoGym source `95f678a5` (MuJoCo instance made before this image was built (TopoGym source 95f678a5; its bytes are bound into the release image eai-roboarc:1.2.0, scripts/robotangle/release_rebuild.py --instances)); bound to this release's runtime (source `4163626d`, tooling `8cb4e65a`).
- Frozen instance files (`instance.json`, `instance_assets/`): not in this directory. They are in the private HF dataset `eai-v2-internal/robotangle-assets` at `9f34f894a454`, under `instances/bed-bolt-01-i01/` (scripts/roboarc/assets.json pins each file's sha256). The image carries them, and `/opt/robotangle/stage_instance.py` copies this task's instance next to its `runtime_binding.json`, which names the instance's canonical sha256 and every asset's sha256. The tooling checks them again at every session start.
- Release: RoboARC 1.2.0, image `ghcr.io/embodied-first-exam/roboarc:1.2.0` (images/roboarc/); simulator source `4395f8165` (PROVENANCE.md).
- Name: RoboARC is the published name of this benchmark (2026-10-13). Code identifiers keep the simulator's name: the simulator package `topogym`, the tooling `rcb_topogym.py`, `TOPOGYM_*` variables.

### Change Log
- 2026-10-20: new task of roboarc (RoboARC on MuJoCo 3.14.0, docs/benchmarks/roboarc.md): a new design of bed_bolt: a bed frame's corner bolt: the long bolt carried to the blind hole and pushed in until the thread catches, then driven home with an L-shaped hex key, its short arm in the head's socket and the long arm swung 60 degrees at a time. Its task files start from the Genesis 1.2.1 task `interlocked-claws-easy-i00` (privileged mode), with this design's own task text, rules and goal, its own frozen instance and its own reference.
- 2026-10-14: tests/grade.py, the same file in every privileged task, requires again that one grading replay reach the goal with the instance's own thresholds within the trajectory's rows (failure analysis 2026-10-14, item 12; docs/benchmarks/roboarc.md, Grading replays). This family has no grading hold: a replay succeeds only within the rows, so every verdict is unchanged. reward.json gains `success_in_hold_only` (always 0 here), and `strict_replay_required` is 1.
