# roboarc-caliper-sorting-01-i01-standard

| | |
|---|---|
| Benchmark | RoboARC |
| Mode | standard (`eai-standard/2.4`) |
| Twin | `roboarc-caliper-sorting-01-i01-privileged` |
| Simulator / robot | `roboarc` / `aloha-2` |
| Family / instance | `caliper_sorting` / seed 0 (allowed seeds: [0, 1, 2]; §2.13) |
| Oracle | `full` |
| Agent time budget | 10800 s; tokens: protocol default (PROTOCOL §5) |
| Base image | `ghcr.io/mll-lab-nu/roboarc:2.2.0@sha256:96a0a895c0e4934adca39f7a370e67bf82bfb4db7ef049131a3ac0dc9d0f9057` |

## Task
Standard-mode twin of `roboarc-caliper-sorting-01-i01-privileged` (same frozen instance): the agent gets only the `eai-standard/2.4` client
(docs/STANDARD_MODE.md); the simulator runs in the `sim` sidecar (`environment/docker-compose.yaml`, GPU on the
sidecar), which owns the scene and records the episode as a (T, 14) trajectory; its collect hook
(`environment/sim/finalize.sh`) replays the trajectory once in a fresh process for success, rendering the video during
that replay. The verifier runs in a separate sandbox (`tests/Dockerfile`) over the sidecar's artifacts. Compare against
the privileged task to measure the cost of losing resets, privileged state and the tooling's primitives.

## Difficulty
Same scene, instance and success rule as `roboarc-caliper-sorting-01-i01-privileged` (see its Difficulty). In standard mode the agent loses resets, privileged state and planning primitives: it sees camera images and proprioception only, acts through the client, and gets one episode, so it has to estimate the scene from images and recover from its own mistakes without starting over.

## Reference solution
`solution/solve.sh` streams the recorded rows of the standard-mode oracle episode (`solution/oracle_qpos.npy`, one `joint_position` row per 25 ms control step, `hold`; the same rows as the privileged reference) through the client (`solution/oracle_replay.py`). That episode scored 1 (live and fresh replay) on the release image roboarc 1.3.0 (ghcr.io/mll-lab-nu/roboarc:1.3.0, sha256:499d6f69bd81), seeds 0, 1 and 2; harbor run -a oracle with rendering off (scripts/roboarc/harbor_check.py) (2026-10-24; simulator source 4d2cba21, tooling 412cbe02).

## Verification
See `tests/grade.py`.

`nop` scored 0 on this base in both modes (the release image roboarc 1.3.0 (499d6f69bd81), seeds 0, 1 and 2; harbor run -a nop with rendering off (scripts/roboarc/harbor_check.py); 2026-10-24).

## Compute resources
Minimum allocation for one trial (docs/PROTOCOL.md §2.9; `[metadata.resources]` in `task.toml`). The agent phase counts every container of the trial at once (the agent's, and in standard modes the simulator sidecar); the verifier phase runs after it.

| | Agent phase | Verifier |
|---|---|---|
| CPU cores | 4 | 2 |
| Memory | 7 GB | 6 GB |
| GPU memory | 2 GB | 2 GB |

- GPU: required, any NVIDIA GPU; recommended: L40S 48GB / RTX 4090 24GB / L4 24GB (sm_86 or newer).
- Measured: their class's values (partial MuJoCo measurements of 2026-10-13 on the class's other families, with the protocol's margin); this family's own peaks not measured yet; GPU 2 GB: a trial's peak was about 1.1 GB on 24 GB cards
- Limits in the agent phase (docs/PROTOCOL.md §2.10; `[metadata.resources.limits]`, shown to the agent in `instruction.md`; always measured and reported, and exceeding one fails the trial only when the run sets `RCB_LIMITS_ENFORCE=1`), the agent's container: no GPU, 8 GiB memory, 512 processes, 20 GiB of new files under `/app`, `/tmp`, `/root`; 60 s within 120 s. Expected peak of the whole trial: 12 GiB GPU memory, 16 GiB memory. Limits measured: their class's limits (partial MuJoCo measurements of 2026-10-13 on the class's other families, with the protocol's margin); this family's own peaks not measured yet.

## Details

### Goal
Measure the sleeves with the caliper and put every sleeve into the bin whose label gives its two caliper readings, then briefly hold everything still.

### Success requirements
- Four sleeves stand in the rack. They look alike and differ only in their inside and outside diameters. Each of the four bins carries a label 'ID r1 / OD r2' with two readings of this caliper: r1 with its knife-edge inside jaws spread in a sleeve's bore, r2 with its outside jaws closed on the sleeve's outside. A sleeve belongs in the bin whose two readings both match its own (within 0.1). A bin may stay empty; each bin holds two sleeves standing.
- The caliper is fixed face up on a stand, its jaws out over the table: it cannot be picked up. Bring each sleeve to its jaws, set the jaws with the slider and read the vernier.
- The caliper is not graduated in millimetres. Its scale is in its own unit (one main-scale division; the vernier reads tenths of it); the size of that unit and the reading with the jaws closed are this caliper's own and are not given, so a size measured in millimetres any other way cannot be matched to a label: the readings must come from the scale.
- The slider is locked at the start. Its locking screw is the knurled knob on the slider's front face, left of the vernier on the jaws' side; a white line on its top shows its angle. It turns between two stops 45 degrees apart, and clockwise (seen from the scale side) tightens. Within 10 degrees of the loose stop the slider moves freely (about 1.5 N of friction); from 20 degrees on it is locked where it stands and does not move however hard it is pushed. Turn the knob by pinching it and rotating the gripper about its axis: it turns by friction only and is not a grasp target.
- Every sleeve's centre must end inside the bin of its class: within 40 mm of the bin's centre in both horizontal directions (the bin's 80 mm x 80 mm cavity) and 3 mm to 40 mm above the tabletop.
- Every sleeve must be released: no sleeve may still be held by an assisted grasp, and no part of the robot (arm, gripper or finger) may press on a sleeve with more than 0.05 N. Whether the caliper is locked at the end does not matter.
- Keep every sleeve's linear speed at most 1 mm/s and angular speed at most 0.02 rad/s.
- Keep all final conditions satisfied together for at least 0.5 s of simulation time. If any final condition stops holding, the confirmation interval starts again. A brief qualifying pose is not enough; let the simulation advance through the full interval.

### Task failure conditions
- A task piece falling more than 100 mm below the tabletop ends the episode in failure. Finger contact with a held movable piece may penetrate at most 2 mm; holding requires both fingers of the same hand to contact the same piece, or an active assisted grasp by that hand. All other task contacts, including piece-piece, table, arm and gripper-housing contacts, may penetrate at most 1 mm. Exceeding either limit ends the episode in failure.
- A sleeve left in a bin that is not its own (its centre inside that bin as above, released as above and still at the speeds above) for 0.5 s ends the episode in failure. When you set a sleeve down to take it again, set it on the table, not in another bin.

### Grader details (in addition to the requirements above)
- Contact penetration as this benchmark grades it (it replaces the "at most 1 mm / 2 mm" wording above): penetration is recorded, it never ends the episode by itself. A success with deep interpenetration (a task contact deeper than 3 mm, or deeper than 2 mm for at least 0.25 s: 100 consecutive physics ticks; for a finger contact with a held piece 4 mm, or 2 mm for 0.25 s) is not credited automatically but reviewed, with the recorded evidence and frames; a clean success is credited automatically. A task contact deeper than 10 mm is a simulator breakdown (below).
- Contact penetration is read from the contacts at the end of every physics tick (2.5 ms; 10 ticks per control step). "Held" is decided anew at every tick: a zero-force touch counts, and if one finger loses contact (with no active assisted grasp), the other finger is back at the non-held limits in the same tick.
- Robot safety (this benchmark's tooling): no gripper target goes lower than 15 mm below the tabletop; a commanded target that would bring any part of a gripper lower is stopped there, on the commanded path. The fingertips reach about 16 mm beyond the tool frame (TCP) along its approach axis, open or closed. Arm drive torques are capped (the effort limits in robot_spec), so an arm pushing against an obstacle stalls at that torque. An arm with any part more than 5 mm deep in a fixed object (a fixture, the table) is stopped where it is: its joint targets become its measured joints for that control step, so it stops pushing (the safety readings report these stops as static_guard).
- Forcing is recorded, not judged: an arm drive (not a gripper drive) at 95% of its torque cap or more while that arm's gripper, or a part it holds, presses on anything else with 20 N or more in total is forcing; 0.5 s of it on every physics tick is a forcing event in the evidence (`meta["safety"]`), shown to the reviewer when a success is reviewed. Forcing does not end the episode and does not change the grade by itself.
- Simulator breakdown (a non-finite state or a task contact deeper than 10 mm): the episode ends as an infrastructure error, not your failure, unless an arm was forcing (even for one tick) within the 0.25 s before it: then it is your failure (forced_breakdown: what broke and what forced it).
- Grasp regions (assisted grasping, see Grippers above; one sphere per graspable part): caliper: radius 20 mm; sleeve_0: radius 40 mm; sleeve_1: radius 40 mm; sleeve_2: radius 40 mm; sleeve_3: radius 40 mm. Both finger contacts must lie inside the part's sphere; a part touched elsewhere is grasped physically.

| | |
|---|---|
| mode | standard (RGB, optional depth, proprioception, calibrated cameras, no reset) |
| action space | `eai-standard/2.4`: `submit` chunks of `joint_position` (14, streamed at the 40 Hz control rate with the agent's `dt`) or `ee_pose` (16, IK per row); `observe` frames or a watch window; binary RCB1 responses |
| oracle | `full` (`solution/`) |
| verification | `success` = live success ∧ replay success: the live episode, then one fresh-process replay in the sidecar's collect hook, after the live service has exited; the state hashes are reported, not required (MuJoCo on the CPU replays bit for bit on one host; the rule does not rely on it); `/logs/verifier/replay.mp4` is rendered during that replay |
| crash recovery | `eai-standard/2.4` §13: journal + per-call checkpoints, `restart: on-failure:6`, resume class `approx` |

### Provenance
- Frozen instance: a MuJoCo instance (`simulator.physics_engine` "mujoco", its scene frozen under `instance_assets/mujoco/`), built by TopoGym source `cacc831d` (MuJoCo instance made before this image was built (TopoGym source cacc831d; its bytes are bound into the release image ghcr.io/mll-lab-nu/roboarc:2.2.0, tools/release/release_rebuild.py --instances)); bound to this release's runtime (source `1552f29d`, tooling `f2320e4c`).
- Frozen instance files (`instance.json`, `instance_assets/`): not in this directory. They are in the private HF dataset `eai-v2-internal/robotangle-assets` at `9f34f894a454`, under `instances/caliper-sorting-01-i01/` (scripts/roboarc/assets.json pins each file's sha256). The image carries them, and `/opt/robotangle/stage_instance.py` copies this task's instance next to its `runtime_binding.json`, which names the instance's canonical sha256 and every asset's sha256. The tooling checks them again at every session start.
- Release: RoboARC 2.2.0, image `ghcr.io/mll-lab-nu/roboarc:2.2.0@sha256:96a0a895c0e4934adca39f7a370e67bf82bfb4db7ef049131a3ac0dc9d0f9057` (images/roboarc/); simulator source `aaec27121` (PROVENANCE.md).
- Name: RoboARC is the published name of this benchmark (2026-10-13). Code identifiers keep the simulator's name: the simulator package `topogym`, the tooling `rcb_topogym.py`, `TOPOGYM_*` variables.

### Change Log
- 2026-10-24: new task of roboarc (RoboARC on MuJoCo 3.14.0, docs/benchmarks/roboarc.md): a new design of caliper_sorting: four look-alike sleeves of two classes measured with a vernier caliper fixed on a stand that reads in its own unit and zero (the right hand brings each sleeve to the jaws; the left hand sets the slider, turns its locking screw and reads the vernier with its wrist camera) and sorted into the bins labelled with their two readings. Its task files start from the Genesis 1.2.1 task `interlocked-claws-easy-i00` (standard mode), with this design's own task text, rules and goal, its own frozen instance and its own reference.
- 2026-10-19: RoboARC 1.0.1, texts and the monitor's image reference only: the texts, comments and docstrings of this task cite this repository's paths (docs/PROTOCOL.md, docs/STANDARD_MODE.md, docs/TAXONOMY.md, tools/release/release_rebuild.py, images/roboarc/) and carry no internal notes, `[task].name` is `roboarc/<dir>` (it was `robot-coding-bench/<dir>`), and the Release line names the task set's version (the image stays `ghcr.io/embodied-first-exam/roboarc:1.0.0`). The resource-monitor sidecar runs RoboARC's own build of its image, `ghcr.io/embodied-first-exam/roboarc-monitor:0.1.0` (images/monitor/; the same files and command as the earlier `ghcr.io/mll-lab-nu/efe:monitor-0.1.0`). The instruction, the logic of the grader and of the sim service, the tooling, the runtime binding, the frozen instance and the reference are unchanged: results graded under 1.0.0 stand.
- 2026-10-20: RoboARC 1.1.0, the image without Genesis: `ghcr.io/embodied-first-exam/roboarc:1.1.0` is built from the Python base image with the same operating system, the same Debian package versions and the same wheels (by sha256) as 1.0.0, without genesis-world, libuipc and the packages only they used (no task of 1.0.0 ran on them: every base is MuJoCo, every family renders with nyx_direct.py). The simulator source drops its Genesis-only modules, the tooling (v20) drops the libuipc build choice and Genesis' kernel replacements, so the runtime binding names the new simulator source, tooling and runtime. The instruction, the grader, the frozen instance and the reference are unchanged; every reference replays in the new image to the end state it reaches in 1.0.0: results graded under 1.0.x stand.
- 2026-10-21: RoboARC 1.1.0, every image by digest: the Dockerfiles and compose files name the simulator image and the resource monitor's image by manifest digest, `ghcr.io/embodied-first-exam/roboarc:1.1.0@sha256:85ea9003…` (images/pins.json), so a pull gets exactly the image that was validated (until now the simulator image was named by its tag). The image, the instruction, the grader, the tooling, the frozen instance and the reference are unchanged: results graded under 1.1.0 stand.
- 2026-10-21: RoboARC 1.1.0, the agent container's client packages are installed from a hashed lock: `environment/client-requirements.txt` keeps the same versions and adds the sha256 of the file pip selects for the client base, and `environment/Dockerfile` installs it with `--no-deps --require-hashes`, so a build installs exactly those files or fails. The installed packages are the same files as before.
- 2026-10-21: RoboARC 1.1.1, image references moved to ghcr.io/mll-lab-nu (same images, same digests): the Dockerfiles, compose files and README name `ghcr.io/mll-lab-nu/<image>:<tag>@sha256:<digest>` with the digests of 1.1.0 (images/pins.json). The runtime binding is unchanged, and nothing else changed: results graded under 1.1.0 stand.
- 2026-10-21: RoboARC 1.2.0, three scenes per base (appearance seeds): seed 0 is the released scene, unchanged; seeds 1 and 2 are the same scene and physics in other colours, materials and textures. The task allows seeds 0, 1 and 2 (`[metadata.seeds]`); the runner picks the trial's seed and the image stages that seed's frozen instance against its own runtime binding (`runtime_binding.s<S>.json`, `stage_instance.py --seed-file`). `oracle` lists seed 0 and the seeds whose Harbor oracle run passed; the reference serves every seed (re-stamped with each seed's birth where it records one). Image `ghcr.io/mll-lab-nu/roboarc:1.2.0` (the seeds' simulator source and tooling); seed 0's birth, end state and render inputs are those of 1.1.x, so results graded under 1.1.x stand.
- 2026-10-22: RoboARC 1.2.1, the renderer checks the GPU before it starts: `nyx_direct.py` reads the Vulkan extensions of the task's GPU and, where `VK_KHR_ray_query` is missing (GPUs without RT cores, such as V100, A100 and H100), raises `nyx_direct.UnsupportedDevice` naming what is missing, instead of letting the native renderer crash the process. There the privileged verifier grades without its replay video (`video_rendered` 0) and the agent's camera calls raise an error it can catch; standard mode needs a GPU with RT cores. On a GPU with RT cores nothing changes. The runtime binding names the new `nyx_direct.py`; the image, the instruction, the grader, the simulator tooling, the frozen instance and the reference are unchanged: results graded under 1.2.0 stand.
- 2026-10-22: RoboARC 1.3.0, the simulator source and tooling of a new experimental family (wire_splicing, not in tasks/ until the owner has reviewed it): the family's code, its two opt-in grasp options (welding a wire's links between the pads, holding between the pads) and an opt-in weld work guard, which only that family's scene profile turns on. Image `ghcr.io/mll-lab-nu/roboarc:1.3.0` (simulator source and tooling v22). This task's instance, physics, instruction, grader and reference are unchanged, and so are its birth and end states (seeds 0, 1 and 2): results graded under 1.2.x stand.
- 2026-10-25: RoboARC 2.1.0, image `ghcr.io/mll-lab-nu/roboarc:2.1.0`: the simulator source draws a card in the five bed-bolt scenes (visual only), fixes a crash in wire_splicing's coil count on a straight tail, and lets the blanket families' release check leave out the world-fixed arm bases (see those tasks' own entries); the buzz-wire 06, 07 and 10 texts state their own loop opening. Unless this task's own entry says otherwise, its instance, physics, instruction, grader and reference are unchanged, and so are its birth and end states (seeds 0, 1 and 2): results graded under 2.0.0 stand.
- 2026-10-28: RoboARC 2.2.0, image `ghcr.io/mll-lab-nu/roboarc:2.2.0`: appearance seeds v2. Seeds 1 and 2 change the room's lighting, the table top's material, the table's frame and the floor, and the task's objects' colours and finishes; they are other scenes than the seeds 1 and 2 of 1.2.0-2.1.0, so results at those seeds (and aggregates over a task's seeds) are not comparable with earlier ones. Seed 0 looks and reads as in 2.1.0 unless this task's own entry says otherwise. The physics, the graders, the births and the reference rows are unchanged at every seed: results graded under 2.1.0 stand.
