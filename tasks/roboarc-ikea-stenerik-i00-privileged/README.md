# roboarc-ikea-stenerik-i00-privileged

| | |
|---|---|
| Benchmark | RoboARC |
| Mode | privileged |
| Twin | `roboarc-ikea-stenerik-i00-standard` |
| Simulator / robot | `mujoco` / `franka-panda-bimanual` |
| Family / instance | `ikea_stenerik` / seed 0 |
| Oracle | `full` |
| Agent time budget | 10800 s; tokens: protocol default (PROTOCOL §5) |
| Base image | `ghcr.io/embodied-first-exam/roboarc-ikea:1.1.0@sha256:2c56d9226aaf97a325b610d464a1cd81e87db1a6285e923539397f493e21136b` |

## Task
IKEA STENERIK laptop stand (AA-2372559-2), all 17 operations in plain MuJoCo 3.14.0 (CPU, float64) with two 7-DoF Franka Panda arms, each with the MuJoCo Menagerie Panda hand (two-finger, tendon-coupled): produce one replayable trajectory. Build the STENERIK laptop stand: set up the two side supports and the crossbar, drive the six screws through the supports into the crossbar's threaded inserts, then peel the four pads off the backing strip and press them onto the supports' pad sites.

## Difficulty
Bimanual furniture assembly with physical grasps (no assist), 16-D actuator action at 400 Hz
(4 physics steps of 0.625 ms per row); the reference needs 215782 rows
(539.45 s of simulated time).

## Reference solution
`solution/solve.sh` copies `solution/trajectory.npz` (sha256 `7a1e6e502d633bfe…`) to `/app/output/`: the task's one reference, shared with the standard twin (its `solution/oracle_actions.npz`, the same rows): a closed-loop reference built on the final physics by `ikea/scripts/ikea_mujoco/oracles/stenerik/build.py` and `pads.py`: the demo as the phase guide, executed within the joint-speed limits; each carried screw steered onto its insert axis, the screwdriver centred on the true screw axis (screws 5-6 driven closed loop: wings aligned to the slots, spin with a light axial push, torque-free at the seat until the weld engages), the wind-up released before each lift; each carried pad steered onto its site, and each press servoed on the true pad-support contact force (the demo row held) until the bond latches; the rows are the targets executed after the joint-speed clamp. It never trips the speed clamp (arm joints and Franka Hand).

## Verification
Open-loop replay of `/app/output/trajectory.npz` from the scene's initial state in two fresh processes at the same time
(`tests/grade.py`, `tests/rcb_ikea.py replay`: per row `data.ctrl[:] = row; S.step(4)` with the scene's mechanism, then
`final_check()`: `success.py`, released, at rest and no breakdown at the end of the rows and again after a 1.0 s hold of
the last row). Pass iff both replays pass and end in the same state hashes (`deterministic`). The
scene in the verifier image must match `tests/runtime_binding.json` file by file (`instance_exact`) and build to its
initial-state hash (`instance_ok`). Replay video (evidence only, never the verdict): a third process replays the same rows and renders `/logs/verifier/replay.mp4` (CPU OSMesa, 512 x 288, at most 750 frames over the episode and the 1.0 s hold; `video_rendered`).

Harbor checks (2026-10-18, a Ryzen 9 7950X workstation (shared, load ~15), Docker, Harbor 0.23.0, image 0.3.0 sha256:ed950d079b63, the replay-video verifier, image `ghcr.io/mll-lab-nu/eai-ikea-mujoco:0.3.0` id `ed950d079b63`): `oracle` scored 1 (two replays, 3324.0 s wall); `nop` scored 0. Image 0.3.1 (the shared welded-part contact switch-off, off in this scene) on the merged head: a fresh replay of the reference ends and settles on the same hashes as in 0.3.0 (`a690aa1b973b` / `20a59284ab83`, on a Ryzen 9 7950X workstation, 2026-10-18). Image 0.3.2 (kyrre's native-cylinder scene; this scene byte-identical to 0.3.1), 2026-10-19: a fresh replay of the reference ends and settles on the same hashes (`a690aa1b973b` / `20a59284ab83`, on the same workstation). Reference timing against the 10800 s budget: one unobserved replay of the hand-slew reference (222,600 rows) 2649 s (24.5 %) on a quiet i9-14900HX host (61.3 % of the 4320 s bar); 3292-3313 s (30.6 %) on a Ryzen 9 7950X workstation (load ~15, two at once); 6944-7082 s (64.3-65.6 %, OVER the 40 % bar, as before the hand rule: 6253-6286 s) on a dedicated Xeon Gold 6230R core (2.1 GHz, a slow trial-class CPU); end and settled hashes equal to the Harbor runs' on every host.

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
| Scene | `/opt/ikea/scenes/stenerik/` in the image; digest `cc2dd3a687a0484a…` (ikea/scripts/ikea_mujoco/assets.json) |
| Initial state | sha256 `3dc8f1b6f99efce3…` of the mjSTATE_INTEGRATION vector |

### Goal
Build the STENERIK laptop stand: set up the two side supports and the crossbar, drive the six screws through the supports into the crossbar's threaded inserts, then peel the four pads off the backing strip and press them onto the supports' pad sites.

### Success requirements
- Each of the crossbar's six inserts holds a retained screw (its weld to the crossbar latched), aligned to the insert and to the support hole it passes through: axial gap below 0.1 mm, lateral within 0.6 mm, thread axis within 1.5 degrees, slower than 0.02 m/s. Screws, pads and the two supports are interchangeable.
- Both supports are joined to the crossbar (join weld latched) and at their side pose: gap below 0.1 mm, lateral within 0.6 mm, orientation within 1.5 degrees, slower than 0.02 m/s.
- Each of the four pad sites holds a pad that is bonded to that support, no longer bonded to the backing, and aligned (same gates).
- At the end no robot touches any part (the backing strip excepted), every part is slower than 0.025 m/s and the bench carries the parts: on a copy of the state, 2 s with the arms holding their targets, then a 2 N downward load on the crossbar, the bench reaction equals the parts' weight (plus the load) within max(0.05 N, 2 %).

### Mechanism
The M4 x 0.5 right-hand threads are exact convex pieces: a screw advances only by being turned. Retention welds switch on with the measured relative pose once the scene's gates hold: a screw seated in its insert (contact with the crossbar and the support, at least 0.4 turns and some tightening work, held 0.15 s); each support joins the crossbar with the third screw of its side; a pad on its site (0.30 s). The pads start bonded to the backing and come free once pulled off it with at least 1.5 N for 0.25 s.

### Provenance
- Scene and demo: the IKEA MuJoCo tree (`scenes/stenerik/`, `demos/stenerik/actions.npz`), MuJoCo 3.14.0, RoboARC rigid settings
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
