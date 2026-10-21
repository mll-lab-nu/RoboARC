# roboarc-ikea-burvik-i00-privileged

| | |
|---|---|
| Benchmark | RoboARC |
| Mode | privileged |
| Twin | `roboarc-ikea-burvik-i00-standard` |
| Simulator / robot | `mujoco` / `franka-panda-bimanual` |
| Family / instance | `ikea_burvik` / seed 0 |
| Oracle | `full` |
| Agent time budget | 10800 s; tokens: protocol default (PROTOCOL §5) |
| Base image | `ghcr.io/embodied-first-exam/roboarc-ikea:1.1.0@sha256:2c56d9226aaf97a325b610d464a1cd81e87db1a6285e923539397f493e21136b` |

## Task
IKEA BURVIK side table (AA-2445102-1), seven operations in plain MuJoCo 3.14.0 (CPU, float64) with two 7-DoF Franka Panda arms, each with the MuJoCo Menagerie Panda hand (two-finger, tendon-coupled): produce one replayable trajectory. Assemble the BURVIK side table: screw two legs into the upside-down top, turn the top on its side, push the handle through the two remaining top holes, screw the other two legs into the handle's sleeves, and turn the table upright.

## Difficulty
Bimanual furniture assembly with physical grasps (no assist), 16-D actuator action at 400 Hz
(4 physics steps of 0.625 ms per row); the reference needs 162534 rows
(406.34 s of simulated time).

## Reference solution
`solution/solve.sh` copies `solution/trajectory.npz` (sha256 `6445eb0f7dc19ca2…`) to `/app/output/`: the task's one reference, shared with the standard twin (its `solution/oracle_actions.npz`, the same rows): a closed-loop reference built on the final physics by `ikea/scripts/ikea_mujoco/oracles/burvik/build2.py` and `move_from_take.py`: the demo (time-dilated to the joint-speed limits) as the phase guide, each row held while the slewed Franka Hand is more than two ticks from its command; the top seated in the jig's slot by a two-hand servo (the demo perched it on the guards, leaning 6 degrees, so the hand threading leg_3 met the handle rack); the handle and the two handle legs servoed onto their axes in the top's frame, each leg spun by the wrist with a bounded axial press until its weld engages; for the upright turn the right hand re-takes the handle in the demo's arm configuration and the left hand takes leg_2 on its centre, then the demo's rigid two-hand motion is executed on these grasps through joint-limited IK; the table set down level on its feet, clear of the stand, before the hands open; the rows are the targets executed within the joint-speed and hand-speed limits. It never trips the speed clamp (arm joints and Franka Hand).

## Verification
Open-loop replay of `/app/output/trajectory.npz` from the scene's initial state in two fresh processes at the same time
(`tests/grade.py`, `tests/rcb_ikea.py replay`: per row `data.ctrl[:] = row; S.step(4)` with the scene's mechanism, then
`final_check()`: `success.py`, released, at rest and no breakdown at the end of the rows and again after a 1.0 s hold of
the last row). Pass iff both replays pass and end in the same state hashes (`deterministic`). The
scene in the verifier image must match `tests/runtime_binding.json` file by file (`instance_exact`) and build to its
initial-state hash (`instance_ok`). Replay video (evidence only, never the verdict): a third process replays the same rows and renders `/logs/verifier/replay.mp4` (CPU OSMesa, 512 x 288, at most 750 frames over the episode and the 1.0 s hold; `video_rendered`).

Harbor checks (2026-10-18, a Ryzen 9 7950X workstation (shared, loaded), Docker, Harbor 0.23.0, fresh per-run file mtimes, image 0.3.0 sha256:ed950d079b63, the replay-video verifier, image `ghcr.io/embodied-first-exam/roboarc-ikea:1.0.0` id `5d5090ea1910`): `oracle` scored 1 (two replays, 1646.1 s wall); `nop` scored 0. Image 0.3.1 (the shared welded-part contact switch-off, off in this scene) on the merged head: a fresh replay of the reference ends and settles on the same hashes as in 0.3.0 (`60878a5edc5d` / `02732bf5d33e`, on a Ryzen 9 7950X workstation, 2026-10-18). Image 0.3.2 (kyrre's native-cylinder scene; this scene byte-identical to 0.3.1), 2026-10-19: a fresh replay of the reference ends and settles on the same hashes (`60878a5edc5d` / `02732bf5d33e`, on the same workstation). Reference timing against the 10800 s budget: one unobserved replay 3780-3783 s (35.0 %, inside the 40 % bar: 87.6 % of its 4320 s) on a dedicated Xeon Gold 6230R core (2.1 GHz, a slow trial-class CPU), end and settled hashes equal to the Harbor runs'; 1694 s (15.7 %) on a quiet i9-14900HX host; 1679-1684 s on a Ryzen 9 7950X workstation (Docker, image 0.3.0, two replays at once).

## Compute resources
Minimum allocation for one trial (docs/PROTOCOL.md §2.9; `[metadata.resources]` in `task.toml`). The agent phase counts every container of the trial at once (the agent's, and in standard modes the simulator sidecar); the verifier phase runs after it.

| | Agent phase | Verifier |
|---|---|---|
| CPU cores | 2 | 3 |
| Memory | 2 GB | 3 GB |
| GPU memory | 0 GB | 0 GB |

- GPU: not needed.
- Measured: peak RSS of one Session 0.27-0.45 GB per scene (3000 reference rows + final_check, workstation, 2026-10-15); the verifier runs two replays and the video replay at once; x 1.2 and rounded up
- Limits in the agent phase (docs/PROTOCOL.md §2.10; `[metadata.resources.limits]`, shown to the agent in `instruction.md`; always measured and reported, and exceeding one fails the trial only when the run sets `RCB_LIMITS_ENFORCE=1`), every container of the trial: no GPU, 8 GiB memory, 512 processes, 10 GiB of new files under `/app`, `/tmp`, `/root`; 60 s within 120 s. Expected peak of the whole trial: 0 GiB GPU memory, 8 GiB memory. Limits measured: the privileged peaks with room for an agent's own replays (four Sessions at once).

## Details
| | |
|---|---|
| Grasping | physical (finger friction), no assist |
| Success rule | the scene's `success.py` + the terminal rules (released, at rest, no breakdown), at the end of the rows and after a 1.0 s hold |
| Scene | `/opt/ikea/scenes/burvik/` in the image; digest `49dbcc4041d4f2be…` (ikea/scripts/ikea_mujoco/assets.json) |
| Initial state | sha256 `d563d833bd219a22…` of the mjSTATE_INTEGRATION vector |

### Goal
Assemble the BURVIK side table: screw two legs into the upside-down top, turn the top on its side, push the handle through the two remaining top holes, screw the other two legs into the handle's sleeves, and turn the table upright.

### Success requirements
- All five retention joints (the four legs and the handle to the top) are latched.
- Every leg and the handle is at its assembled pose in the top's frame: gap within 0.8 mm, lateral within 1 mm, tilt within 2 degrees (legs: thread axis; handle: full rotation, both rod ends within 1 mm of the two top holes), relative speed below 0.025 m/s.
- The top ends upright: its normal within 2 degrees of world +z.

### Mechanism
Each leg end is a right-hand M8 thread (1 mm lead, SDF mesh) in an SDF nut: about 12.5 turns from first engagement to the seat. Retention welds (legs and handle to the top) switch on once the scene's gates hold for 0.15 s.

### Provenance
- Scene and demo: the IKEA MuJoCo tree (`scenes/burvik/`, `demos/burvik/actions.npz`), MuJoCo 3.14.0, RoboARC rigid settings
  (timestep 0.625 ms, Newton, 100 iterations, elliptic cones).
- The scene files are not in this repository: the image carries them (`ikea/images/ikea-mujoco/README.md`, build
  context from `ikea/scripts/ikea_mujoco/stage_context.py`, pinned by `ikea/scripts/ikea_mujoco/assets.json`).

### Change Log
- 2026-10-19: RoboARC 1.0.1: the task is on the repository's task layer and under `tasks/` (in 1.0.0 it was under `ikea/`),
  and its `[task].name` is `roboarc/<dir>` (it was `robot-coding-bench/<dir>`). For the agent: the instruction's closing
  sections are the shared ones (`## External material`, `## Resource limits`, `## Time limit`); the network allowlist
  adds the package registries (PyPI, the Debian / Ubuntu archives) that `## External material` already named (the
  repository's one list, `scripts/set_agent_egress.py` `HOSTS`); a resource-monitor sidecar (its image
  `ghcr.io/embodied-first-exam/roboarc-monitor:0.1.0`, images/monitor/) records the trial's resources (a limit
  fails the trial only when the run sets `RCB_LIMITS_ENFORCE=1`); the agent's container gets the
  declared caps. The task declares its compute resources, limits and seeds, and has a `DIGEST`; its texts cite this
  repository's paths. The scene, the logic of the graders and of the sim service, the tooling (`rcb_ikea.py`), the
  runtime binding, the reference and the images (`:1.0.0`) are unchanged. Results before and after 1.0.1 are not
  comparable: the instruction and the egress changed.
- 2026-10-20: RoboARC 1.1.0, the images `ghcr.io/embodied-first-exam/roboarc-ikea:1.1.0` and `roboarc-ikea-sim:1.1.0`:
  rebuilt from fully pinned inputs (the Python base image by digest, Debian packages by version from the Debian
  snapshot archive, wheels by sha256) and with only what the tasks run. Every file the graded runtime loads (MuJoCo
  with its EPA patch, numpy, the scenes, the replay video's libraries, Nyx, torch and its CUDA libraries) is
  byte-identical to 1.0.0's; left out are unused packages (mujoco's viewer and test dependencies, the Nyx environment's
  triton, sympy and pip, curl), bytecode caches and the duplicate scene layer. The verifier's replay video libraries
  (OSMesa, imageio with ffmpeg, Pillow) are in the base image, so the verifier image no longer installs them when it
  is built; the agent's container has them too. The runtime binding names the new image id. The instruction, the
  scene, the logic of the graders and of the sim service, the tooling and the reference are unchanged; every
  reference replays to the end and settled states it reaches in 1.0.0: results graded under 1.0.x stand.
- 2026-10-21: RoboARC 1.1.0, every image by digest: the Dockerfiles and compose files name the IKEA images and the
  resource monitor's image by manifest digest (images/pins.json), so a pull gets exactly the images that were
  validated (until now the IKEA images were named by their tags). In standard mode the agent container's client
  packages are installed from a hashed lock (`environment/client-requirements.txt`, the same versions with the sha256
  of each file, installed with `--no-deps --require-hashes`): the same files as before. The images, the instruction,
  the scene, the graders, the tooling and the reference are unchanged: results graded under 1.1.0 stand.
