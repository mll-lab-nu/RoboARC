# roboarc-horseshoe-ring-01-i01-privileged

| | |
|---|---|
| Benchmark | RoboARC |
| Mode | privileged |
| Twin | `roboarc-horseshoe-ring-01-i01-standard` |
| Simulator / robot | `roboarc` / `aloha-2` |
| Family / instance | `horseshoe_ring` / seed 0 (allowed seeds: [0, 1, 2]; §2.13) |
| Oracle | `full` |
| Agent time budget | 10800 s; tokens: protocol default (PROTOCOL §5) |
| Base image | `ghcr.io/mll-lab-nu/roboarc:2.1.0@sha256:ccc3a47a5d9e9ee94485b896070b93d737004d0cb132dc27b93bd2f19a8c917e` |

## Task
RoboARC horseshoe_ring (ALOHA 2, MuJoCo): produce one replayable trajectory. Separate and place the entire removable ring inside the green dashed goal region. Keep the horseshoes and chains on the puzzle/system side. Frozen scene instance (variation very_easy, seed 0).

## Difficulty
Two 6-dof aloha (viperx) arms with parallel-jaw grippers on a fixed base with a 14-D joint-target action space at 40 Hz; success needs the task's predicate (above) held for 0.5 s. Grasping: assisted grasping (contact_weld_v1).

## Reference solution
`solution/solve.sh` copies `solution/trajectory.npz` (sha256 `665961f263543849…`) to `/app/output/`: the MuJoCo reference (the released route re-recorded on MuJoCo with a held tail), which the verifier scored 1 in two fresh-process replays on the release image eai-roboarc 0.1.0 (2026-10-14; simulator source 37aa7ac0, tooling d8692e15).

## Verification
Open-loop replay of `/app/output/trajectory.npz` from the frozen instance (`/tests/instance.json`, staged from the image and checked against `tests/runtime_binding.json`) in a fresh process, twice; pass iff the
task's predicate holds after every replay (`tests/grade.py`). `/logs/verifier/replay.mp4` is rendered during the second
replay (two passes, no separate video replay). The physics runs on the CPU and is bit-deterministic on one host; the final-state hashes
(`deterministic`) are still diagnostic, as for the GPU families; the rebuilt frozen scene must match (`instance_ok`).
A trajectory may have at most 192000 rows (`RCB_MAX_STEPS`: two replays must fit the verifier's timeout; a longer one is
`bad_trajectory`).

`nop` scored 0 on this base in both modes (the release image eai-roboarc 0.1.0; harbor run -a nop with rendering off (scripts/roboarc/harbor_check.py); 2026-10-14).

## Compute resources
Minimum allocation for one trial (docs/PROTOCOL.md §2.9; `[metadata.resources]` in `task.toml`). The agent phase counts every container of the trial at once (the agent's, and in standard modes the simulator sidecar); the verifier phase runs after it.

| | Agent phase | Verifier |
|---|---|---|
| CPU cores | 4 | 2 |
| Memory | 7 GB | 5 GB |
| GPU memory | 2 GB | 2 GB |

- GPU: required, any NVIDIA GPU; recommended: L40S 48GB / RTX 4090 24GB / L4 24GB (sm_86 or newer).
- Measured: partial MuJoCo measurements of 2026-10-13 (rendering off) with the protocol's margin; calibration pending; GPU 2 GB: a trial's peak was about 1.1 GB on 24 GB cards
- Limits in the agent phase (docs/PROTOCOL.md §2.10; `[metadata.resources.limits]`, shown to the agent in `instruction.md`; always measured and reported, and exceeding one fails the trial only when the run sets `RCB_LIMITS_ENFORCE=1`), every container of the trial: 12 GiB GPU memory, 24 GiB memory, 512 processes, 10 GiB of new files under `/app`, `/tmp`, `/root`; 60 s within 120 s. Expected peak of the whole trial: 12 GiB GPU memory, 24 GiB memory. Limits measured: partial MuJoCo measurements of 2026-10-13 (rendering off) with the protocol's margin; calibration pending.

## Details
| | |
|---|---|
| Grasping | assisted grasping (contact_weld_v1) |
| Success rule | task-native success (goal_complete True; frozen parameters; terminal revision placement_region_v3_grasp_contact_2mm) |

### Goal
Separate and place the entire removable ring inside the green dashed goal region. Keep the horseshoes and chains on the puzzle/system side.

### Success requirements
- Put every part of the entire removable ring inside the fixed goal volume, leaving at least 1 mm inside its horizontal border. The centre alone is insufficient; no tip, spike, ring segment or cord segment may remain outside.
- Keep the target above the tabletop (contact tolerance 1 mm below the surface) and within the goal volume's height. It need not touch the table. The goal volume extends 240.517 mm above the tabletop.
- Keep the horseshoes and chains, and every other remaining puzzle component, entirely on the system side of the fixed separator. Moving the whole puzzle into the goal is not completion.
- Keep each target rigid part's linear speed at most 1 mm/s and angular speed at most 0.02 rad/s. Holding the target with the gripper is allowed; release is not required.
- Keep every permanent chain/eye interlock intact; free only the removable target ring.
- Keep all final conditions satisfied together for at least 0.5 s of simulation time. If any final condition stops holding, the confirmation interval starts again. A brief qualifying pose is not enough; let the simulation advance through the full interval.

### Task failure conditions
- Losing any permanent chain/eye connection ends the episode in failure.
- A task piece falling more than 100 mm below the tabletop ends the episode in failure. Finger contact with a held movable piece may penetrate at most 2 mm; holding requires both fingers of the same hand to contact the same piece, or an active assisted grasp by that hand. All other task contacts, including piece-piece, table, arm and gripper-housing contacts, may penetrate at most 1 mm. Exceeding either limit ends the episode in failure.

### Grader details (in addition to the requirements above)
- Goal square in this instance: the inner edge of the dashed green line is the containment limit (the whole line lies outside it); it encloses a 248.5 mm x 248.5 mm square. Every point counts: for a rigid part every vertex of its visual and collision meshes.
- The separator is an invisible, fixed vertical plane parallel to the goal square's near edge, 5 mm beyond the point of all remaining components that was closest to the goal square at the start; the square's near edge is at least 30 mm beyond it. The horseshoes and chain links are free to move (their speed is not checked), but no point of them may cross that plane.
- Contact penetration as this benchmark grades it (it replaces the "at most 1 mm / 2 mm" wording above): penetration is recorded, it never ends the episode by itself. A success with deep interpenetration (a task contact deeper than 3 mm, or deeper than 2 mm for at least 0.25 s: 100 consecutive physics ticks; for a finger contact with a held piece 4 mm, or 2 mm for 0.25 s) is not credited automatically but reviewed, with the recorded evidence and frames; a clean success is credited automatically. A task contact deeper than 10 mm is a simulator breakdown (below).
- Contact penetration is read from the contacts at the end of every physics tick (2.5 ms; 10 ticks per control step). "Held" is decided anew at every tick: a zero-force touch counts, and if one finger loses contact (with no active assisted grasp), the other finger is back at the non-held limits in the same tick.
- Robot safety (this benchmark's tooling): no gripper target goes lower than 15 mm below the tabletop; a commanded target that would bring any part of a gripper lower is stopped there, on the commanded path. The fingertips reach about 16 mm beyond the tool frame (TCP) along its approach axis, open or closed. Arm drive torques are capped (the effort limits in robot_spec), so an arm pushing against an obstacle stalls at that torque. An arm with any part more than 5 mm deep in a fixed object (a fixture, the table) is stopped where it is: its joint targets become its measured joints for that control step, so it stops pushing (the safety readings report these stops as static_guard).
- Forcing is recorded, not judged: an arm drive (not a gripper drive) at 95% of its torque cap or more while that arm's gripper, or a part it holds, presses on anything else with 20 N or more in total is forcing; 0.5 s of it on every physics tick is a forcing event in the evidence (`s.safety()`, `s.status()["forcing"]`), shown to the reviewer when a success is reviewed. Forcing does not end the episode and does not change the grade by itself.
- Simulator breakdown (a non-finite state or a task contact deeper than 10 mm): the episode ends as an infrastructure error, not your failure, unless an arm was forcing (even for one tick) within the 0.25 s before it: then it is your failure (forced_breakdown: what broke and what forced it).
- The four permanent connections (the upper and lower eye of each horseshoe with its chain link) are checked at every tick by the linking number of the two loops.
- Grasp regions (assisted grasping, see Grippers above; one sphere per graspable part): horseshoe_left: radius 90.6 mm, enclosing the whole part; horseshoe_right: radius 90.6 mm, enclosing the whole part; target_ring: radius 82.6 mm, enclosing the whole part. Both finger contacts must lie inside the part's sphere; a part touched elsewhere is grasped physically. Each region's centre (in its link's frame) is in `/app/instance.json` under `simulator.grasp_assist.targets`.
- A failure is reported in `s.status()["episode_end"]` as "A task body fell below the table", "Physical chain/eye connection lost" or "forced_breakdown: <what>" (<what>: the readable reason); `s.safety()` reports forcing, contact depths, volume ratios and table-guard stops against these limits, and `s.status()["review"]` the deep task contacts that would hold a success for review.

### Provenance
- Frozen instance: a MuJoCo instance (`simulator.physics_engine` "mujoco", its scene frozen under `instance_assets/mujoco/`), built by TopoGym source `6fb23e0d` (MuJoCo instance made before this image was built (TopoGym source 6fb23e0d; its bytes are bound into the release image ghcr.io/mll-lab-nu/roboarc:2.1.0, tools/release/release_rebuild.py --instances)); bound to this release's runtime (source `cf508cee`, tooling `412cbe02`).
- Frozen instance files (`instance.json`, `instance_assets/`): not in this directory. They are in the private HF dataset `eai-v2-internal/robotangle-assets` at `9f34f894a454`, under `instances/horseshoe-ring-01-i01/` (scripts/roboarc/assets.json pins each file's sha256). The image carries them, and `/opt/robotangle/stage_instance.py` copies this task's instance next to its `runtime_binding.json`, which names the instance's canonical sha256 and every asset's sha256. The tooling checks them again at every session start.
- Release: RoboARC 2.1.0, image `ghcr.io/mll-lab-nu/roboarc:2.1.0@sha256:ccc3a47a5d9e9ee94485b896070b93d737004d0cb132dc27b93bd2f19a8c917e` (images/roboarc/); simulator source `ce493b06e` (PROVENANCE.md).
- Name: RoboARC is the published name of this benchmark (2026-10-13). Code identifiers keep the simulator's name: the simulator package `topogym`, the tooling `rcb_topogym.py`, `TOPOGYM_*` variables.

### Change Log
- 2026-10-14: new task of roboarc (RoboARC on MuJoCo 3.14.0, docs/benchmarks/roboarc.md): the scene, goal and rules of the Genesis 1.2.1 task `horseshoe-ring-very-easy-i00` (privileged mode), ported to MuJoCo: its own frozen instance and reference, the numpy / OpenBLAS SIMD masking in every container that runs a Session, and the render bridge next to the tooling (Nyx frames).
- 2026-10-14: tests/grade.py, the same file in every privileged task, requires again that one grading replay reach the goal with the instance's own thresholds within the trajectory's rows (docs/benchmarks/roboarc.md, Grading replays). This family has no grading hold: a replay succeeds only within the rows, so every verdict is unchanged. reward.json gains `success_in_hold_only` (always 0 here), and `strict_replay_required` is 1.
- 2026-10-19: RoboARC 1.0.1, texts and the monitor's image reference only: the texts, comments and docstrings of this task cite this repository's paths (docs/PROTOCOL.md, docs/STANDARD_MODE.md, docs/TAXONOMY.md, tools/release/release_rebuild.py, images/roboarc/) and carry no internal notes, `[task].name` is `roboarc/<dir>` (it was `robot-coding-bench/<dir>`), and the Release line names the task set's version (the image stays `ghcr.io/embodied-first-exam/roboarc:1.0.0`). The resource-monitor sidecar runs RoboARC's own build of its image, `ghcr.io/embodied-first-exam/roboarc-monitor:0.1.0` (images/monitor/; the same files and command as the earlier `ghcr.io/mll-lab-nu/efe:monitor-0.1.0`). The instruction, the logic of the grader and of the sim service, the tooling, the runtime binding, the frozen instance and the reference are unchanged: results graded under 1.0.0 stand.
- 2026-10-20: RoboARC 1.1.0, the image without Genesis: `ghcr.io/embodied-first-exam/roboarc:1.1.0` is built from the Python base image with the same operating system, the same Debian package versions and the same wheels (by sha256) as 1.0.0, without genesis-world, libuipc and the packages only they used (no task of 1.0.0 ran on them: every base is MuJoCo, every family renders with nyx_direct.py). The simulator source drops its Genesis-only modules, the tooling (v20) drops the libuipc build choice and Genesis' kernel replacements, so the runtime binding names the new simulator source, tooling and runtime. The instruction, the grader, the frozen instance and the reference are unchanged; every reference replays in the new image to the end state it reaches in 1.0.0: results graded under 1.0.x stand.
- 2026-10-21: RoboARC 1.1.0, every image by digest: the Dockerfiles and compose files name the simulator image and the resource monitor's image by manifest digest, `ghcr.io/embodied-first-exam/roboarc:1.1.0@sha256:85ea9003…` (images/pins.json), so a pull gets exactly the image that was validated (until now the simulator image was named by its tag). The image, the instruction, the grader, the tooling, the frozen instance and the reference are unchanged: results graded under 1.1.0 stand.
- 2026-10-21: RoboARC 1.1.1, image references moved to ghcr.io/mll-lab-nu (same images, same digests): the Dockerfiles, compose files and README name `ghcr.io/mll-lab-nu/<image>:<tag>@sha256:<digest>` with the digests of 1.1.0 (images/pins.json). The runtime binding is unchanged, and nothing else changed: results graded under 1.1.0 stand.
- 2026-10-21: RoboARC 1.2.0, three scenes per base (appearance seeds): seed 0 is the released scene, unchanged; seeds 1 and 2 are the same scene and physics in other colours, materials and textures. The task allows seeds 0, 1 and 2 (`[metadata.seeds]`); the runner picks the trial's seed and the image stages that seed's frozen instance against its own runtime binding (`runtime_binding.s<S>.json`, `stage_instance.py --seed-file`). `oracle` lists seed 0 and the seeds whose Harbor oracle run passed; the reference serves every seed (re-stamped with each seed's birth where it records one). Image `ghcr.io/mll-lab-nu/roboarc:1.2.0` (the seeds' simulator source and tooling); seed 0's birth, end state and render inputs are those of 1.1.x, so results graded under 1.1.x stand.
- 2026-10-22: RoboARC 1.2.1, the renderer checks the GPU before it starts: `nyx_direct.py` reads the Vulkan extensions of the task's GPU and, where `VK_KHR_ray_query` is missing (GPUs without RT cores, such as V100, A100 and H100), raises `nyx_direct.UnsupportedDevice` naming what is missing, instead of letting the native renderer crash the process. There the privileged verifier grades without its replay video (`video_rendered` 0) and the agent's camera calls raise an error it can catch; standard mode needs a GPU with RT cores. On a GPU with RT cores nothing changes. The runtime binding names the new `nyx_direct.py`; the image, the instruction, the grader, the simulator tooling, the frozen instance and the reference are unchanged: results graded under 1.2.0 stand.
- 2026-10-22: RoboARC 1.3.0, the simulator source and tooling of a new experimental family (wire_splicing, not in tasks/ until the owner has reviewed it): the family's code, its two opt-in grasp options (welding a wire's links between the pads, holding between the pads) and an opt-in weld work guard, which only that family's scene profile turns on. Image `ghcr.io/mll-lab-nu/roboarc:1.3.0` (simulator source and tooling v22). This task's instance, physics, instruction, grader and reference are unchanged, and so are its birth and end states (seeds 0, 1 and 2): results graded under 1.2.x stand.
- 2026-10-25: RoboARC 2.1.0, image `ghcr.io/mll-lab-nu/roboarc:2.1.0`: the simulator source draws a card in the five bed-bolt scenes (visual only), fixes a crash in wire_splicing's coil count on a straight tail, and lets the blanket families' release check leave out the world-fixed arm bases (see those tasks' own entries); the buzz-wire 06, 07 and 10 texts state their own loop opening. Unless this task's own entry says otherwise, its instance, physics, instruction, grader and reference are unchanged, and so are its birth and end states (seeds 0, 1 and 2): results graded under 2.0.0 stand.
