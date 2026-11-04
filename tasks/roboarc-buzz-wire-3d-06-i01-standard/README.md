# roboarc-buzz-wire-3d-06-i01-standard

| | |
|---|---|
| Benchmark | RoboARC |
| Mode | standard (`eai-standard/2.4`) |
| Twin | `roboarc-buzz-wire-3d-06-i01-privileged` |
| Simulator / robot | `roboarc` / `franka-robotiq-2f85` |
| Family / instance | `buzz_wire_3d` / seed 0 (allowed seeds: [0, 1, 2]; §2.13) |
| Oracle | `full` |
| Agent time budget | 11400 s; tokens: protocol default (PROTOCOL §5) |
| Base image | `ghcr.io/mll-lab-nu/roboarc:2.2.0@sha256:96a0a895c0e4934adca39f7a370e67bf82bfb4db7ef049131a3ac0dc9d0f9057` |

## Task
Standard-mode twin of `roboarc-buzz-wire-3d-06-i01-privileged` (same frozen instance): the agent gets only the `eai-standard/2.4` client
(docs/STANDARD_MODE.md); the simulator runs in the `sim` sidecar (`environment/docker-compose.yaml`, GPU on the
sidecar), which owns the scene and records the episode as a (T, 8) trajectory; its collect hook
(`environment/sim/finalize.sh`) replays the trajectory once in a fresh process for success, rendering the video during
that replay. The verifier runs in a separate sandbox (`tests/Dockerfile`) over the sidecar's artifacts. Compare against
the privileged task to measure the cost of losing resets, privileged state and the tooling's primitives.

## Difficulty
Same scene, instance and success rule as `roboarc-buzz-wire-3d-06-i01-privileged` (see its Difficulty). In standard mode the agent loses resets, privileged state and planning primitives: it sees camera images and proprioception only, acts through the client, and gets one episode, so it has to estimate the scene from images and recover from its own mistakes without starting over.

## Reference solution
`solution/solve.sh` streams the recorded rows of the standard-mode oracle episode (`solution/oracle_qpos.npy`, one `joint_position` row per 25 ms control step, `hold`; the same rows as the privileged reference) through the client (`solution/oracle_replay.py`). That episode scored 1 (live and fresh replay) on the release image eai-roboarc 0.1.0 (2026-10-14; simulator source 37aa7ac0, tooling d8692e15).

## Verification
See `tests/grade.py`.

`nop` scored 0 on this base in both modes (the release image eai-roboarc 0.1.0; harbor run -a nop with rendering off (scripts/roboarc/harbor_check.py); 2026-10-14).

## Compute resources
Minimum allocation for one trial (docs/PROTOCOL.md §2.9; `[metadata.resources]` in `task.toml`). The agent phase counts every container of the trial at once (the agent's, and in standard modes the simulator sidecar); the verifier phase runs after it.

| | Agent phase | Verifier |
|---|---|---|
| CPU cores | 4 | 2 |
| Memory | 7 GB | 6 GB |
| GPU memory | 2 GB | 2 GB |

- GPU: required, any NVIDIA GPU; recommended: L40S 48GB / RTX 4090 24GB / L4 24GB (sm_86 or newer).
- Measured: partial MuJoCo measurements of 2026-10-13 (rendering off) with the protocol's margin; calibration pending; GPU 2 GB: a trial's peak was about 1.1 GB on 24 GB cards
- Limits in the agent phase (docs/PROTOCOL.md §2.10; `[metadata.resources.limits]`, shown to the agent in `instruction.md`; always measured and reported, and exceeding one fails the trial only when the run sets `RCB_LIMITS_ENFORCE=1`), the agent's container: no GPU, 8 GiB memory, 512 processes, 20 GiB of new files under `/app`, `/tmp`, `/root`; 60 s within 120 s. Expected peak of the whole trial: 12 GiB GPU memory, 16 GiB memory. Limits measured: partial MuJoCo measurements of 2026-10-13 (rendering off) with the protocol's margin; calibration pending.

## Details

### Goal
Guide the metal loop along the entire wire from start to finish without touching the wire, then release the tool on the finish support. Rubber sleeves on the wire are insulated: the ring may rest on them. The final bend and the post at the finish are insulated as well.

### Success requirements
- Start within 10 mm along the wire from its starting end and traverse continuously until within 10 mm of its finishing end. Keep one unambiguous wire crossing inside the loop. Skipping branches or taking the loop off the route before completing traversal does not count; restart from the beginning if tracking is lost.
- After completing traversal, lower the tool onto the finish support. The completed traversal remains valid when the loop leaves the wire to rest on the support.
- Match the finish resting pose within 5 mm and 10 degrees. A half-turn about the handle axis is an equivalent orientation. The tool must actually contact the finish support.
- Keep all metal parts of the tool clear of the metal wire throughout the episode. Wooden-handle contact is not a metal-wire failure. Remove all robot contact before final confirmation.
- Release all target parts and disengage any assisted grasp. Withdraw the robot until there is at least 5 mm of verified clearance from every target part; opening the fingers while still touching or surrounding the target is insufficient.
- During confirmation, every target point or rigid body's linear speed must be at most 1 mm/s and its displacement from the start of confirmation at most 1 mm. Each target rigid body's angular speed must be at most 0.02 rad/s and angular drift at most 0.02 rad.
- Keep all final conditions satisfied together for at least 1 s of simulation time. If any final condition stops holding, the confirmation interval starts again. A brief qualifying pose is not enough; let the simulation advance through the full interval.

### Task failure conditions
- Any metal-tool touch of the metal wire ends the episode in failure, including brief or zero-force touches.
- The tool falling more than 100 mm below the tabletop ends the episode in failure.

### Grader details (in addition to the requirements above)
- Robot safety (this benchmark's tooling): no gripper target goes lower than 15 mm below the tabletop; a commanded target that would bring any part of a gripper lower is stopped there, on the commanded path. The fingertips reach about 6 mm beyond the tool frame (TCP) along its approach axis when open and about 19 mm when closed (the Robotiq fingertips move forward as they close). Arm drive torques are capped (the effort limits in robot_spec), so an arm pushing against an obstacle stalls at that torque. An arm with any part more than 5 mm deep in a fixed object (a fixture, the table) is stopped where it is: its joint targets become its measured joints for that control step, so it stops pushing (reported in clipped[]).
- Forcing is recorded, not judged: an arm drive (not a gripper drive) at 95% of its torque cap or more while that arm's gripper, or a part it holds, presses on anything else with 20 N or more in total is forcing; 0.5 s of it on every physics tick is a forcing event in the evidence (`meta["safety"]`), shown to the reviewer when a success is reviewed. Forcing does not end the episode and does not change the grade by itself.
- The release clearance is measured between axis-aligned bounding boxes: one around each collision geometry of the whole robot (every link of both arms) and one around each collision geometry of the tool, so the true gap usually has to be larger than 5 mm.
- The loop's usable opening: the wire's centreline must cross the loop's disk less than 8.77 mm from the loop's centre (loop inner radius 10.24 mm minus the 1.47 mm wire radius).
- Continuous traversal: from one physics tick to the next, the crossing point may move along the wire by at most twice the loop's own motion in that tick (its displacement plus 8.77 mm times its rotation angle) plus 3 mm. A larger jump, no crossing, two crossings or a tangency cancels the traversal; it must then start again within 10 mm of the starting end, the end where the tool rests at the start.
- The finish support is the deck, and the finish resting pose is the start resting pose repeated at the wire's finishing end: the tool lies on the deck with its loop at that end of the wire and its handle pointing inward, the way it lies at the start.
- Metal contact is sampled at all 4 solver substeps of every tick. If any tick since the start had incomplete contact coverage (a simulator-side condition), the episode can no longer succeed.
- Grasp regions (assisted grasping, see Grippers above; one sphere per graspable part): wand: radius 25 mm, on the wand's grip. Both finger contacts must lie inside the part's sphere; a part touched elsewhere is grasped physically.

| | |
|---|---|
| mode | standard (RGB, optional depth, proprioception, calibrated cameras, no reset) |
| action space | `eai-standard/2.4`: `submit` chunks of `joint_position` (8, streamed at the 40 Hz control rate with the agent's `dt`) or `ee_pose` (8, IK per row); `observe` frames or a watch window; binary RCB1 responses |
| oracle | `full` (`solution/`) |
| verification | `success` = live success ∧ replay success: the live episode, then one fresh-process replay in the sidecar's collect hook, after the live service has exited; the state hashes are reported, not required (MuJoCo on the CPU replays bit for bit on one host; the rule does not rely on it); `/logs/verifier/replay.mp4` is rendered during that replay |
| crash recovery | `eai-standard/2.4` §13: journal + per-call checkpoints, `restart: on-failure:6`, resume class `approx` |

### Provenance
- Frozen instance: a MuJoCo instance (`simulator.physics_engine` "mujoco", its scene frozen under `instance_assets/mujoco/`), built by TopoGym source `d1eb6f13` (MuJoCo instance made before this image was built (TopoGym source d1eb6f13; its bytes are bound into the release image ghcr.io/mll-lab-nu/roboarc:2.2.0, tools/release/release_rebuild.py --instances)); bound to this release's runtime (source `1552f29d`, tooling `f2320e4c`).
- Frozen instance files (`instance.json`, `instance_assets/`): not in this directory. They are in the private HF dataset `eai-v2-internal/robotangle-assets` at `b2cf9b552f31`, under `instances/buzz-wire-3d-06-i01/` (scripts/roboarc/assets.json pins each file's sha256). The image carries them, and `/opt/robotangle/stage_instance.py` copies this task's instance next to its `runtime_binding.json`, which names the instance's canonical sha256 and every asset's sha256. The tooling checks them again at every session start.
- Release: RoboARC 2.2.1, image `ghcr.io/mll-lab-nu/roboarc:2.2.0@sha256:96a0a895c0e4934adca39f7a370e67bf82bfb4db7ef049131a3ac0dc9d0f9057` (images/roboarc/); simulator source `e0a930a38` (PROVENANCE.md).
- Name: RoboARC is the published name of this benchmark (2026-10-13). Code identifiers keep the simulator's name: the simulator package `topogym`, the tooling `rcb_topogym.py`, `TOPOGYM_*` variables.

### Change Log
- 2026-10-14: new task of roboarc (RoboARC on MuJoCo 3.14.0, docs/benchmarks/roboarc.md): a new design of buzz_wire_3d: a redesigned wire, with insulated rest sleeves and an insulated finish post. Its task files start from the Genesis 1.2.1 task `buzz-wire-3d-very-easy-i00` (standard mode), with this design's own task text, rules and goal, its own frozen instance and its own reference.
- 2026-10-28: roboarc 2.2.0: the instruction names the parts whose colour now changes with the appearance seed by their shape or role, not by a colour; the rules, limits and numbers it states are unchanged.
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
- 2026-10-30: RoboARC 2.2.1, the same image `ghcr.io/mll-lab-nu/roboarc:2.2.0`: the standard service returns mp4 video again. Its `encode_mp4` imported imageio, which the image does not ship, so a `submit` or `get_obs` with `video="mp4"` or `"both"` (and `submit`'s default, `"mp4"`) ended the episode as an infrastructure error; it now encodes with the image's ffmpeg binary (H.264, yuv420p, faststart, as before). The time limit's text also says that the agent works alone: a reply that ends its turn or asks a question ends the session. The instance, physics, grader and reference are unchanged, and so are the birth and end states: privileged results graded under 2.2.0 stand, and so do standard ones that never asked for mp4.
