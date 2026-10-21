# roboarc-bed-bolt-02-i01-standard

| | |
|---|---|
| Benchmark | RoboARC |
| Mode | standard (`eai-standard/2.4`) |
| Twin | `roboarc-bed-bolt-02-i01-privileged` |
| Simulator / robot | `roboarc` / `aloha-2` |
| Family / instance | `bed_bolt` / seed 0 (allowed seeds: [0, 1, 2]; §2.13) |
| Oracle | `full` |
| Agent time budget | 10800 s; tokens: protocol default (PROTOCOL §5) |
| Base image | `ghcr.io/embodied-first-exam/roboarc:1.2.0` |

## Task
Standard-mode twin of `roboarc-bed-bolt-02-i01-privileged` (same frozen instance): the agent gets only the `eai-standard/2.4` client
(docs/rcb/STANDARD_MODE_2_2.md); the simulator runs in the `sim` sidecar (`environment/docker-compose.yaml`, GPU on the
sidecar), which owns the scene and records the episode as a (T, 14) trajectory; its collect hook
(`environment/sim/finalize.sh`) replays the trajectory once in a fresh process for success, rendering the video during
that replay. The verifier runs in a separate sandbox (`tests/Dockerfile`) over the sidecar's artifacts. Compare against
the privileged task to measure the cost of losing resets, privileged state and the tooling's primitives.

## Difficulty
Same scene, instance and success rule as `roboarc-bed-bolt-02-i01-privileged` (see its Difficulty). In standard mode the agent loses resets, privileged state and planning primitives: it sees camera images and proprioception only, acts through the client, and gets one episode, so it has to estimate the scene from images and recover from its own mistakes without starting over.

## Reference solution
`solution/solve.sh` streams the recorded rows of the reference (`solution/oracle_qpos.npy`, one `joint_position` row per control step, `hold`; the same rows as the privileged reference) through the client (`solution/oracle_replay.py`). No Harbor oracle trial of it has been run yet on this standard service, so no score is claimed here. The same rows serve every appearance seed: a seed changes only the looks.

## Verification
See `tests/grade.py`.

`nop`: no record yet on this base (`harbor run -p tasks/roboarc-bed-bolt-02-i01-standard -a nop`).

## Compute resources
Minimum allocation for one trial (docs/PROTOCOL.md §2.9; `[metadata.resources]` in `task.toml`). The agent phase counts every container of the trial at once (the agent's, and in standard modes the simulator sidecar); the verifier phase runs after it.

| | Agent phase | Verifier |
|---|---|---|
| CPU cores | 2 | 2 |
| Memory | 7 GB | 6 GB |
| GPU memory | 2 GB | 2 GB |

- GPU: required, any NVIDIA GPU; recommended: L40S 48GB / RTX 4090 24GB / L4 24GB (sm_86 or newer).
- Measured: the Genesis release's numbers, kept until the MuJoCo release's own peaks are measured (its Harbor runs; MuJoCo physics on the CPU, Nyx on the GPU); GPU 2 GB: owner 10-17, board 0150 (batch_eval_live trial peak 1.1 GB on 24 GB cards)
- Limits in the agent phase (docs/PROTOCOL.md §2.10; `[metadata.resources.limits]`, shown to the agent in `instruction.md`; always measured and reported, and exceeding one fails the trial only when the run sets `RCB_LIMITS_ENFORCE=1`), the agent's container: no GPU, 8 GiB memory, 512 processes, 20 GiB of new files under `/app`, `/tmp`, `/root`; 60 s within 120 s. Expected peak of the whole trial: 16 GiB GPU memory, 21 GiB memory. Limits measured: the Genesis release's numbers, kept until the MuJoCo release's own peaks are measured (its Harbor runs; MuJoCo physics on the CPU, Nyx on the GPU).

## Details

### Goal
Drive the long bolt into the blind hole in the bed frame's corner block with the L-shaped Torx key (a six-lobe star tip) until its head is within 1.5 mm of the block face.

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
- Forcing is recorded, not judged: an arm drive (not a gripper drive) at 95% of its torque cap or more while that arm's gripper, or a part it holds, presses on anything else with 20 N or more in total is forcing; 0.5 s of it on every physics tick is a forcing event in the evidence (`meta["safety"]`), shown to the reviewer when a success is reviewed. Forcing does not end the episode and does not change the grade by itself.
- Simulator breakdown (a non-finite state or a task contact deeper than 10 mm): the episode ends as an infrastructure error, not your failure, unless an arm was forcing (even for one tick) within the 0.25 s before it: then it is your failure (forced_breakdown: what broke and what forced it).
- Grasp regions (assisted grasping, see Grippers above; one sphere per graspable part): bolt: radius 53.2 mm; key: radius 107.9 mm. Both finger contacts must lie inside the part's sphere; a part touched elsewhere is grasped physically.

| | |
|---|---|
| mode | standard (RGB, optional depth, proprioception, calibrated cameras, no reset) |
| action space | `eai-standard/2.4`: `submit` chunks of `joint_position` (14, streamed at the 40 Hz control rate with the agent's `dt`) or `ee_pose` (16, IK per row); `observe` frames or a watch window; binary RCB1 responses |
| oracle | `full` (`solution/`) |
| verification | `success` = live success ∧ replay success: the live episode, then one fresh-process replay in the sidecar's collect hook, after the live service has exited; the state hashes are reported, not required (MuJoCo on the CPU replays bit for bit on one host; the rule does not rely on it); `/logs/verifier/replay.mp4` is rendered during that replay |
| crash recovery | `eai-standard/2.4` §13: journal + per-call checkpoints, `restart: on-failure:6`, resume class `approx` |

### Provenance
- Frozen instance: a MuJoCo instance (`simulator.physics_engine` "mujoco", its scene frozen under `instance_assets/mujoco/`), built by TopoGym source `95f678a5` (MuJoCo instance made before this image was built (TopoGym source 95f678a5; its bytes are bound into the release image eai-roboarc:1.2.0, scripts/robotangle/release_rebuild.py --instances)); bound to this release's runtime (source `4163626d`, tooling `8cb4e65a`).
- Frozen instance files (`instance.json`, `instance_assets/`): not in this directory. They are in the private HF dataset `eai-v2-internal/robotangle-assets` at `9f34f894a454`, under `instances/bed-bolt-02-i01/` (scripts/roboarc/assets.json pins each file's sha256). The image carries them, and `/opt/robotangle/stage_instance.py` copies this task's instance next to its `runtime_binding.json`, which names the instance's canonical sha256 and every asset's sha256. The tooling checks them again at every session start.
- Release: RoboARC 1.2.0, image `ghcr.io/embodied-first-exam/roboarc:1.2.0` (images/roboarc/); simulator source `4395f8165` (PROVENANCE.md).
- Name: RoboARC is the published name of this benchmark (2026-10-13). Code identifiers keep the simulator's name: the simulator package `topogym`, the tooling `rcb_topogym.py`, `TOPOGYM_*` variables.

### Change Log
- 2026-10-20: new task of roboarc (RoboARC on MuJoCo 3.14.0, docs/benchmarks/roboarc.md): a new design of bed_bolt: a bed frame's corner bolt with a Torx head: the long bolt carried to the blind hole and pushed in until the thread catches, then driven home with an L-shaped Torx key, its short arm in the head's six-lobe socket and the long arm swung 60 degrees at a time. Its task files start from the Genesis 1.2.1 task `interlocked-claws-easy-i00` (standard mode), with this design's own task text, rules and goal, its own frozen instance and its own reference.
