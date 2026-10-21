# roboarc-ikea-eket-i00-privileged

| | |
|---|---|
| Benchmark | RoboARC |
| Mode | privileged |
| Twin | `roboarc-ikea-eket-i00-standard` |
| Simulator / robot | `mujoco` / `franka-panda-bimanual` |
| Family / instance | `ikea_eket` / seed 0 |
| Oracle | `full` |
| Agent time budget | 10800 s; tokens: protocol default (PROTOCOL §5) |
| Base image | `ghcr.io/embodied-first-exam/roboarc-ikea:1.1.0@sha256:2c56d9226aaf97a325b610d464a1cd81e87db1a6285e923539397f493e21136b` |

## Task
IKEA EKET cabinet (AA-1914763-5), manual steps 1-4 in plain MuJoCo 3.14.0 (CPU, float64) with two 7-DoF Franka Panda arms, each with the MuJoCo Menagerie Panda hand (two-finger, tendon-coupled): produce one replayable trajectory. Build the EKET cabinet frame: slide the two side panels into the top panel, then the back panel into its grooves, then the bottom panel onto the sides, and leave the cabinet standing on the table on its own.

## Difficulty
Bimanual furniture assembly with physical grasps (no assist), 16-D actuator action at 400 Hz
(4 physics steps of 0.625 ms per row); the reference needs 83181 rows
(207.95 s of simulated time).

## Reference solution
`solution/solve.sh` copies `solution/trajectory.npz` (sha256 `eb838b01a6df5fcd…`) to `/app/output/`: the task's one reference, shared with the standard twin (its `solution/oracle_actions.npz`, the same rows): a closed-loop reference built on the final physics by `ikea/scripts/ikea_mujoco/oracles/eket/build_ref.py`: the demo as the phase guide, each held panel servoed onto the unclamped demo's recorded relative-pose track (damped-least-squares Jacobian IK on the TCP) and paused at each slide until its gate accepts; each demo row held (the guide tracks still running) until the slewed Franka Hand reaches its command; the rows are the targets executed after the joint-speed and hand-speed clamp. It never trips the speed clamp (arm joints and Franka Hand).

## Verification
Open-loop replay of `/app/output/trajectory.npz` from the scene's initial state in two fresh processes at the same time
(`tests/grade.py`, `tests/rcb_ikea.py replay`: per row `data.ctrl[:] = row; S.step(4)` with the scene's mechanism, then
`final_check()`: `success.py`, released, at rest and no breakdown at the end of the rows and again after a 1.0 s hold of
the last row). Pass iff both replays pass and end in the same state hashes (`deterministic`). The
scene in the verifier image must match `tests/runtime_binding.json` file by file (`instance_exact`) and build to its
initial-state hash (`instance_ok`). Replay video (evidence only, never the verdict): a third process replays the same rows and renders `/logs/verifier/replay.mp4` (CPU OSMesa, 512 x 288, at most 750 frames over the episode and the 1.0 s hold; `video_rendered`).

Harbor checks (2026-10-18, a Ryzen 9 7950X workstation (shared, heavily loaded: load 40-90 on 32 threads), Docker, Harbor 0.23.0, image 0.3.1 sha256:b9552184c2de, the replay-video verifier, the Franka Hand slew, image `ghcr.io/mll-lab-nu/eai-ikea-mujoco:0.3.1` id `b9552184c2de`): `oracle` scored 1 (two replays, 1454.4 s wall); `nop` scored 0. Image 0.3.2 (kyrre's native-cylinder scene; this scene byte-identical to 0.3.1), 2026-10-19: a fresh replay of the reference ends and settles on the same hashes (`25be8deaa1fe` / `1d2906d52c38`, on a Ryzen 9 7950X workstation). Reference timing against the 10800 s budget: one unobserved replay of the hand-slew reference 220 s (2.0 %; 5.1 % of the 4320 s bar) on a quiet i9-14900HX host (one process at a time, image 0.3.1's scene); 652 s on a loaded Ryzen 9 7950X workstation (load 40-90); 575 s on a dedicated Xeon Gold 6230R core (2.1 GHz, a slow trial-class CPU; reported only), end and settled hashes equal on every host and in the Harbor replays.

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
| Scene | `/opt/ikea/scenes/eket/` in the image; digest `cd515879e4eee6da…` (ikea/scripts/ikea_mujoco/assets.json) |
| Initial state | sha256 `9815e6dc4b4a9856…` of the mjSTATE_INTEGRATION vector |

### Goal
Build the EKET cabinet frame: slide the two side panels into the top panel, then the back panel into its grooves, then the bottom panel onto the sides, and leave the cabinet standing on the table on its own.

### Success requirements
- The seven operations happen in this order, each measured against the top panel: side A entry and slide, side B entry and slide, back slide, bottom entry and slide. Each one must reach its goal pose within the scene's gates (goal error 0.25 to 0.8 mm, angle 0.1 to 1 degree, nearly at rest, with contact with its counterpart panels).
- The four frame joints (side A and side B to the top, the bottom to both sides) are latched, and the five panels are finite.
- At the end the panels are released (no robot contact, no applied force), rest on the table (at least one panel touches the table top, none touches the floor) and move slower than 0.01 m/s and 0.05 rad/s.
- At the end every panel, the back included (it has no frame joint), is still in its assembled place: within 10 mm and 2 degrees of its goal pose relative to the top panel (the benchmark's end-pose rule, rcb_ikea.END_POSE_RULES).

### Mechanism
Mitres, keyholes, dowels and back grooves are contact. The four frame joints (welds) latch with the measured relative pose when the side A slide, the side B slide and the bottom slide pass their gates.

### Provenance
- Scene and demo: the IKEA MuJoCo tree (`scenes/eket/`, `demos/eket_vlim/actions.npz`), MuJoCo 3.14.0, RoboARC rigid settings
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
