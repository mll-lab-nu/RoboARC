# roboarc-blanket-fold-02-i01-standard

| | |
|---|---|
| Benchmark | RoboARC |
| Mode | standard (`eai-standard/2.4`) |
| Twin | `roboarc-blanket-fold-02-i01-privileged` |
| Simulator / robot | `roboarc` / `aloha-2` |
| Family / instance | `blanket_fold` / seed 0 (allowed seeds: [0, 1, 2]; §2.13) |
| Oracle | `full` |
| Agent time budget | 10800 s; tokens: protocol default (PROTOCOL §5) |
| Base image | `ghcr.io/mll-lab-nu/roboarc:2.1.0@sha256:ccc3a47a5d9e9ee94485b896070b93d737004d0cb132dc27b93bd2f19a8c917e` |

## Task
Standard-mode twin of `roboarc-blanket-fold-02-i01-privileged` (same frozen instance): the agent gets only the `eai-standard/2.4` client
(docs/STANDARD_MODE.md); the simulator runs in the `sim` sidecar (`environment/docker-compose.yaml`, GPU on the
sidecar), which owns the scene and records the episode as a (T, 14) trajectory; its collect hook
(`environment/sim/finalize.sh`) replays the trajectory once in a fresh process for success, rendering the video during
that replay. The verifier runs in a separate sandbox (`tests/Dockerfile`) over the sidecar's artifacts. Compare against
the privileged task to measure the cost of losing resets, privileged state and the tooling's primitives.

## Difficulty
Same scene, instance and success rule as `roboarc-blanket-fold-02-i01-privileged` (see its Difficulty). In standard mode the agent loses resets, privileged state and planning primitives: it sees camera images and proprioception only, acts through the client, and gets one episode, so it has to estimate the scene from images and recover from its own mistakes without starting over.

## Reference solution
`solution/solve.sh` streams the recorded rows of the standard-mode oracle episode (`solution/oracle_qpos.npy`, one `joint_position` row per 25 ms control step, `hold`; the same rows as the privileged reference) through the client (`solution/oracle_replay.py`). That episode scored 1 (live and fresh replay) on the release image roboarc 1.1.0 (2026-10-21; simulator source a7487c51, tooling cae5375e).

## Verification
See `tests/grade.py`.

`nop` scored 0 on this base in both modes (the release image eai-roboarc 0.1.0; harbor run -a nop with rendering off (scripts/roboarc/harbor_check.py); 2026-10-14).

## Compute resources
Minimum allocation for one trial (docs/PROTOCOL.md §2.9; `[metadata.resources]` in `task.toml`). The agent phase counts every container of the trial at once (the agent's, and in standard modes the simulator sidecar); the verifier phase runs after it.

| | Agent phase | Verifier |
|---|---|---|
| CPU cores | 6 | 2 |
| Memory | 8 GB | 6 GB |
| GPU memory | 2 GB | 2 GB |

- GPU: required, any NVIDIA GPU; recommended: L40S 48GB / RTX 4090 24GB / L4 24GB (sm_86 or newer).
- Measured: partial MuJoCo measurements of 2026-10-13 (rendering off) with the protocol's margin; calibration pending; GPU 2 GB: a trial's peak was about 1.1 GB on 24 GB cards
- Limits in the agent phase (docs/PROTOCOL.md §2.10; `[metadata.resources.limits]`, shown to the agent in `instruction.md`; always measured and reported, and exceeding one fails the trial only when the run sets `RCB_LIMITS_ENFORCE=1`), the agent's container: no GPU, 8 GiB memory, 512 processes, 20 GiB of new files under `/app`, `/tmp`, `/root`; 60 s within 120 s. Expected peak of the whole trial: 20 GiB GPU memory, 20 GiB memory. Limits measured: partial MuJoCo measurements of 2026-10-13 (rendering off) with the protocol's margin; calibration pending.

## Details

### Goal
Fold the blanket into the target shape on the table, then release it and withdraw. The blanket (600 mm x 480 mm) starts lying flat on the table. Fold it in half: turn the far half (the 240 mm of material along the far long edge) over onto the near half about the middle line parallel to the long edges, so that the far edge lies on the near edge and the far corners on the near corners. The folded stack is 600 mm x 240 mm, two layers, the far half upside down on top. Then dog-ear the top layer: turn the top layer's near-left corner (the blanket's original far-left corner) over onto the top layer, the top layer alone, so that its corner point (12 mm in from both edges) lands 100 mm to the right and 100 mm farther than where it lay after the half fold; the crease runs at about 45 degrees across the corner. The target images show the required final state: this is how the folded cloth must look and where it must lie; fold it like this. Near, far, left and right are as seen in the overhead camera cam_high (near = the lower side of its image).
The target images: `/app/goal/target_cam_high.png` (seen from the overhead camera cam_high) and `/app/goal/target_cam_low.png` (from the low oblique camera cam_low).

### Success requirements
- Corners: each of the blanket's four corner points (the material point 12 mm in from both edges at each corner) must lie within 55 mm, in the top view, of its place in the target fold.
- Layer order: the blanket's panels are its regions between the target's creases (the middle line and the dog-ear crease). For each point of the blanket, the panel directly beneath it (material within 8 mm horizontally and 1-20 mm below) must be the same as in the target fold for at least 88% of all points, and for at least 80% of the points that have material beneath them in the target.
- Outline: after the best rigid top-view fit of the blanket onto the target, the 20 mm grid cells it covers must overlap the target's with an intersection over union of at least 0.85.
- Flat: the folded cloth must lie flat on the tabletop, its bottom layer fully in contact with the table surface: every point of the blanket within 35 mm of its height in the target fold, and no point more than 30 mm above the target's highest point.
- Placement: placed where the target images show it, within 40 mm (the blanket's mean top-view position against the target's) and 5 degrees (the fitted rotation).
- Keep every cloth point no more than 8 mm below the tabletop. No material edge may stretch beyond 1.15 times its original length.
- Release all target parts and disengage any assisted grasp. Withdraw the robot's moving parts until there is at least 5 mm of verified clearance between every target part and the grippers and arm links; the world-fixed arm bases are not counted. Opening the fingers while still touching or surrounding the target is insufficient.
- During confirmation, every target point or rigid body's linear speed must be at most 30 mm/s and its displacement from the start of confirmation at most 3 mm.
- Keep all final conditions satisfied together for at least 0.5 s of simulation time. If any final condition stops holding, the confirmation interval starts again. A brief qualifying pose is not enough; let the simulation advance through the full interval.

### Task failure conditions
- Any cloth point falling more than 100 mm below the tabletop ends the episode in failure.

### Grader details (in addition to the requirements above)
- Robot safety (this benchmark's tooling): no gripper target goes lower than 15 mm below the tabletop; a commanded target that would bring any part of a gripper lower is stopped there, on the commanded path. Over the blanket the floor is the cloth's top: a target that would bring a gripper part other than its finger pads lower than the top of the cloth under it (less about one 1.5 mm layer) is stopped there too, so a gripper cannot crush the cloth into the table; the finger pads of a hand closing to pinch may reach about 6 mm into the cloth. The fingertips reach about 16 mm beyond the tool frame (TCP) along its approach axis, open or closed. Arm drive torques are capped (the effort limits in robot_spec), so an arm pushing against an obstacle stalls at that torque. An arm with any part more than 5 mm deep in a fixed object (a fixture, the table) is stopped where it is: its joint targets become its measured joints for that control step, so it stops pushing (reported in clipped[]).
- Forcing is recorded, not judged: an arm drive (not a gripper drive) at 95% of its torque cap or more while that arm's gripper, or a part it holds, presses on anything else with 20 N or more in total is forcing; 0.5 s of it on every physics tick is a forcing event in the evidence (`meta["safety"]`), shown to the reviewer when a success is reviewed. Forcing does not end the episode and does not change the grade by itself.
- Simulator breakdown (a cloth particle faster than 20 m/s): the episode ends as an infrastructure error, not your failure, unless an arm was forcing (even for one tick) within the 0.25 s before it: then it is your failure (forced_breakdown: what broke and what forced it).
- The blanket is 0.6 m x 0.48 m. Heights are measured at the cloth particles' centres.
- The release clearance is measured from one box around all cloth particles, grown by 8 mm: the collision box of every robot part that can move (arm links, wrists, grippers, fingers) must stay at least 5 mm from it, i.e. at least 13 mm outside the cloth's bounding box, above its highest point too. The two arm bases are fixed to the world and are not counted: the cloth may lie against them.
- Soft grasp patches: blanket: spatial radius 18 mm, material radius 18 mm, finger proximity 6 mm, maximum material strain ratio 1.5, exposed patch required. Both fingers must be near the same material patch; penetration and native constraint health checks still apply. Two hands may hold disjoint patches of one object. Material bounds, when present, restrict the admissible region.
- Grading: success needs the live episode to succeed (with the thresholds above) AND one fresh-process replay of its rows to succeed; a live failure is final whatever the replay shows. Its cloth step runs on the GPU on a deterministic solver and the rest on the CPU (MuJoCo, float64), so that replay reproduces the live episode on the same host; the rule is kept as a margin: it runs the rows with the thresholds above and, if the episode has not ended after the last row, holds the final targets for up to 2 s (80 control steps), stopping at the first success or failure, so a confirmation that starts a few physics ticks later than in the live episode can still complete; during that hold it accepts a blanket coverage of 92% instead of 95% and cloth speeds up to 80 mm/s instead of 30 mm/s (the displacement limit during the confirmation stays). Nothing else is relaxed: every other condition, the failure conditions and the confirmation time are the same.

| | |
|---|---|
| mode | standard (RGB, optional depth, proprioception, calibrated cameras, no reset) |
| action space | `eai-standard/2.4`: `submit` chunks of `joint_position` (14, streamed at the 40 Hz control rate with the agent's `dt`) or `ee_pose` (16, IK per row); `observe` frames or a watch window; binary RCB1 responses |
| oracle | `full` (`solution/`) |
| verification | `success` = live success ∧ replay success: the live episode, then one fresh-process replay in the sidecar's collect hook, after the live service has exited; the state hashes are reported, not required (the deterministic GPU cloth step and MuJoCo replay bit for bit on one host; the rule does not rely on it); `/logs/verifier/replay.mp4` is rendered during that replay; after a live success it is a grading replay (tooling v19): if this family's episode has not ended after the last live row, the final targets are held for up to 2 s with the replay tolerances of the grader details; a live failure is final |
| crash recovery | `eai-standard/2.4` §13: journal + per-call checkpoints, `restart: on-failure:6`, resume class `approx` |

### Provenance
- Frozen instance: a MuJoCo instance (`simulator.physics_engine` "mujoco", its scene frozen under `instance_assets/mujoco/`), built by TopoGym source `e2e2f564` (MuJoCo instance made before this image was built (TopoGym source e2e2f564; its bytes are bound into the release image ghcr.io/mll-lab-nu/roboarc:2.1.0, tools/release/release_rebuild.py --instances)); bound to this release's runtime (source `cf508cee`, tooling `412cbe02`).
- Frozen instance files (`instance.json`, `instance_assets/`): not in this directory. They are in the private HF dataset `eai-v2-internal/robotangle-assets` at `9f34f894a454`, under `instances/blanket-fold-02-i01/` (scripts/roboarc/assets.json pins each file's sha256). The image carries them, and `/opt/robotangle/stage_instance.py` copies this task's instance next to its `runtime_binding.json`, which names the instance's canonical sha256 and every asset's sha256. The tooling checks them again at every session start.
- Release: RoboARC 2.1.0, image `ghcr.io/mll-lab-nu/roboarc:2.1.0@sha256:ccc3a47a5d9e9ee94485b896070b93d737004d0cb132dc27b93bd2f19a8c917e` (images/roboarc/); simulator source `ce493b06e` (PROVENANCE.md).
- Name: RoboARC is the published name of this benchmark (2026-10-13). Code identifiers keep the simulator's name: the simulator package `topogym`, the tooling `rcb_topogym.py`, `TOPOGYM_*` variables.

### Change Log
- 2026-10-14: new task of roboarc (RoboARC on MuJoCo 3.14.0, docs/benchmarks/roboarc.md): a new design of blanket_fold: from flat; the target: the half fold with the top layer's corner dog-eared. Its task files start from the Genesis 1.2.1 task `blanket-flatten-very-easy-i00` (standard mode), with this design's own task text, rules and goal, its own frozen instance and its own reference.
- 2026-10-21: roboarc 1.1.0: the reference's Harbor oracle record (README, Reference solution): it scored 1 on the release image roboarc 1.1.0, where until now no score was claimed; the reference, the frozen instance and the task text are unchanged.
- 2026-10-25: roboarc 2.1.0: the release check counts only the robot parts that can move (arm links, wrists, grippers, fingers): the two arm bases are fixed to the world and no longer count, so a spread or folded blanket may lie against them (grader policy active_parts_clearance_v1, terminal revision release_settling_v2_active_parts; the instruction says so). The rule only gets looser: every earlier success stays one. The new terminal revision changes the instance's identity, so the frozen instance is refreshed (its scene and archives byte for byte) and the reference re-stamped (instance_sha only): the birth and end state hashes change in the task part only (the instance id and the release metrics) and in the release geometry's new fixed flags; the physics is unchanged.
- 2026-10-19: RoboARC 1.0.1, texts and the monitor's image reference only: the texts, comments and docstrings of this task cite this repository's paths (docs/PROTOCOL.md, docs/STANDARD_MODE.md, docs/TAXONOMY.md, tools/release/release_rebuild.py, images/roboarc/) and carry no internal notes, `[task].name` is `roboarc/<dir>` (it was `robot-coding-bench/<dir>`), and the Release line names the task set's version (the image stays `ghcr.io/embodied-first-exam/roboarc:1.0.0`). The resource-monitor sidecar runs RoboARC's own build of its image, `ghcr.io/embodied-first-exam/roboarc-monitor:0.1.0` (images/monitor/; the same files and command as the earlier `ghcr.io/mll-lab-nu/efe:monitor-0.1.0`). The instruction, the logic of the grader and of the sim service, the tooling, the runtime binding, the frozen instance and the reference are unchanged: results graded under 1.0.0 stand.
- 2026-10-20: RoboARC 1.1.0, the image without Genesis: `ghcr.io/embodied-first-exam/roboarc:1.1.0` is built from the Python base image with the same operating system, the same Debian package versions and the same wheels (by sha256) as 1.0.0, without genesis-world, libuipc and the packages only they used (no task of 1.0.0 ran on them: every base is MuJoCo, every family renders with nyx_direct.py). The simulator source drops its Genesis-only modules, the tooling (v20) drops the libuipc build choice and Genesis' kernel replacements, so the runtime binding names the new simulator source, tooling and runtime. The instruction, the grader, the frozen instance and the reference are unchanged; every reference replays in the new image to the end state it reaches in 1.0.0: results graded under 1.0.x stand.
- 2026-10-21: RoboARC 1.1.0, every image by digest: the Dockerfiles and compose files name the simulator image and the resource monitor's image by manifest digest, `ghcr.io/embodied-first-exam/roboarc:1.1.0@sha256:85ea9003…` (images/pins.json), so a pull gets exactly the image that was validated (until now the simulator image was named by its tag). The image, the instruction, the grader, the tooling, the frozen instance and the reference are unchanged: results graded under 1.1.0 stand.
- 2026-10-21: RoboARC 1.1.0, the agent container's client packages are installed from a hashed lock: `environment/client-requirements.txt` keeps the same versions and adds the sha256 of the file pip selects for the client base, and `environment/Dockerfile` installs it with `--no-deps --require-hashes`, so a build installs exactly those files or fails. The installed packages are the same files as before.
- 2026-10-21: RoboARC 1.1.1, image references moved to ghcr.io/mll-lab-nu (same images, same digests): the Dockerfiles, compose files and README name `ghcr.io/mll-lab-nu/<image>:<tag>@sha256:<digest>` with the digests of 1.1.0 (images/pins.json). The runtime binding is unchanged, and nothing else changed: results graded under 1.1.0 stand.
- 2026-10-21: RoboARC 1.2.0, three scenes per base (appearance seeds): seed 0 is the released scene, unchanged; seeds 1 and 2 are the same scene and physics in other colours, materials and textures. The task allows seeds 0, 1 and 2 (`[metadata.seeds]`); the runner picks the trial's seed and the image stages that seed's frozen instance against its own runtime binding (`runtime_binding.s<S>.json`, `stage_instance.py --seed-file`). `oracle` lists seed 0 and the seeds whose Harbor oracle run passed; the reference serves every seed (re-stamped with each seed's birth where it records one). Image `ghcr.io/mll-lab-nu/roboarc:1.2.0` (the seeds' simulator source and tooling); seed 0's birth, end state and render inputs are those of 1.1.x, so results graded under 1.1.x stand.
- 2026-10-22: RoboARC 1.2.1, the renderer checks the GPU before it starts: `nyx_direct.py` reads the Vulkan extensions of the task's GPU and, where `VK_KHR_ray_query` is missing (GPUs without RT cores, such as V100, A100 and H100), raises `nyx_direct.UnsupportedDevice` naming what is missing, instead of letting the native renderer crash the process. There the privileged verifier grades without its replay video (`video_rendered` 0) and the agent's camera calls raise an error it can catch; standard mode needs a GPU with RT cores. On a GPU with RT cores nothing changes. The runtime binding names the new `nyx_direct.py`; the image, the instruction, the grader, the simulator tooling, the frozen instance and the reference are unchanged: results graded under 1.2.0 stand.
- 2026-10-22: RoboARC 1.3.0, the simulator source and tooling of a new experimental family (wire_splicing, not in tasks/ until the owner has reviewed it): the family's code, its two opt-in grasp options (welding a wire's links between the pads, holding between the pads) and an opt-in weld work guard, which only that family's scene profile turns on. Image `ghcr.io/mll-lab-nu/roboarc:1.3.0` (simulator source and tooling v22). This task's instance, physics, instruction, grader and reference are unchanged, and so are its birth and end states (seeds 0, 1 and 2): results graded under 1.2.x stand.
- 2026-10-25: RoboARC 2.1.0, image `ghcr.io/mll-lab-nu/roboarc:2.1.0`: the simulator source draws a card in the five bed-bolt scenes (visual only), fixes a crash in wire_splicing's coil count on a straight tail, and lets the blanket families' release check leave out the world-fixed arm bases (see those tasks' own entries); the buzz-wire 06, 07 and 10 texts state their own loop opening. Unless this task's own entry says otherwise, its instance, physics, instruction, grader and reference are unchanged, and so are its birth and end states (seeds 0, 1 and 2): results graded under 2.0.0 stand.
