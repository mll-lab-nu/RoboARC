# roboarc-ikea-gladom-i00-privileged

| | |
|---|---|
| Benchmark | RoboARC |
| Mode | privileged |
| Twin | `roboarc-ikea-gladom-i00-standard` |
| Simulator / robot | `mujoco` / `franka-panda-bimanual` |
| Family / instance | `ikea_gladom` / seed 0 (allowed seeds: [0, 1, 2]; §2.13) |
| Oracle | `full` |
| Agent time budget | 11400 s; tokens: protocol default (PROTOCOL §5) |
| Base image | `ghcr.io/mll-lab-nu/roboarc-ikea:1.1.0@sha256:2c56d9226aaf97a325b610d464a1cd81e87db1a6285e923539397f493e21136b` |

## Task
IKEA GLADOM tray table (AA-1832218-3), full seven-operation scope in plain MuJoCo 3.14.0 (CPU, float64) with two 7-DoF Franka Panda arms, each with the MuJoCo Menagerie Panda hand (two-finger, tendon-coupled): produce one replayable trajectory. Build the GLADOM tray table: cross frame B into frame A, drive the long screw and the bottom screw, place the ring and drive its two screws, and set the tray on the ring.

## Difficulty
Bimanual furniture assembly with physical grasps (no assist), 16-D actuator action at 400 Hz
(4 physics steps of 0.625 ms per row); the reference needs 186611 rows
(466.5 s of simulated time).

## Reference solution
`solution/solve.sh` copies `solution/trajectory.npz` (sha256 `35190e106aeb5c08…`) to `/app/output/`: the task's one reference, shared with the standard twin (its `solution/oracle_actions.npz`, the same rows): a closed-loop reference built on the final physics by `ikea/scripts/ikea_mujoco/oracles/gladom/build.py`: the demo as the phase guide, each row held until the slewed Franka Hand reaches its command; each screw steered upright onto the stand's slot and lowered onto it before the fingers open; the hex driver's tip kept on the hand axis (its true offset in the fingers compensated by damped-least-squares IK, also while the wrist spins) and each screw driven on until its weld latches; the rows are the targets executed within the joint-speed and hand-speed limits. It never trips the speed clamp (arm joints and Franka Hand).

## Verification
Open-loop replay of `/app/output/trajectory.npz` from the scene's initial state in two fresh processes at the same time
(`tests/grade.py`, `tests/rcb_ikea.py replay`: per row `data.ctrl[:] = row; S.step(4)` with the scene's mechanism, then
`final_check()`: `success.py`, released, at rest and no breakdown at the end of the rows and again after a 1.0 s hold of
the last row). Pass iff both replays pass and end in the same state hashes (`deterministic`). The
scene in the verifier image must match `tests/runtime_binding.json` file by file (`instance_exact`) and build to its
initial-state hash (`instance_ok`). Replay video (evidence only, never the verdict): a third process replays the same rows and renders `/logs/verifier/replay.mp4` (CPU OSMesa, 512 x 288, at most 750 frames over the episode and the 1.0 s hold; `video_rendered`).

Harbor checks (2026-10-18, a Ryzen 9 7950X workstation (shared, loaded ~35-65), Docker, Harbor 0.23.0, image 0.3.0 sha256:ed950d079b63, the replay-video verifier, image `ghcr.io/mll-lab-nu/eai-ikea-mujoco:0.3.0` id `ed950d079b63`): `oracle` scored 1 (two replays, 5091.2 s wall); `nop` scored 0. Image 0.3.1 (the shared welded-part contact switch-off, off in this scene) on the merged head: a fresh replay of the reference ends and settles on the same hashes as in 0.3.0 (`88048a4ff775` / `5c2edffa330c`, on a Ryzen 9 7950X workstation, 2026-10-18). Image 0.3.2 (kyrre's native-cylinder scene; this scene byte-identical to 0.3.1), 2026-10-19: a fresh replay of the reference ends and settles on the same hashes (`88048a4ff775` / `5c2edffa330c`, on the same workstation). Reference timing: one unobserved replay 2630 s (24.3 %; 61 % of the 4320 s bar) on a quiet i9-14900HX host core (Intel AVX2, one process at a time); 8069 / 8082 s (74.8 %, OVER the 40 % bar) on Xeon Gold 6230R cores (2.1 GHz, a slow trial-class CPU; two at once on one node); 4550 s on a loaded Ryzen 9 7950X workstation (load 50-65 on 32 threads); end and settled hashes equal everywhere

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
| Scene | `/opt/ikea/scenes/gladom/` in the image; digest `0acbefad697b3b28…` (ikea/scripts/ikea_mujoco/assets.json) |
| Initial state | sha256 `700c53dd5fd9fe2b…` of the mjSTATE_INTEGRATION vector |

### Goal
Build the GLADOM tray table: cross frame B into frame A, drive the long screw and the bottom screw, place the ring and drive its two screws, and set the tray on the ring.

### Success requirements
- The retention joints of frame B, the ring and the four screws are all latched.
- Every part except frame A is at its assembled pose relative to frame A: gap within 0.8 mm along frame A's z axis, lateral within 1 mm, tilt within 2 degrees (screws: axis tilt; frame B and the ring may be rotated 180 degrees, the tray 7.5 degrees), relative speed below 0.025 m/s. The two ring screws may use either ring receiver, but distinct ones.
- The tray is free (never jointed) and supported by the ring (contact above 0.01 N in at least 12 of 36 samples over the last 0.15 s, and above 0.01 N on average over them); no robot contact with any furniture part.
- All of the above hold continuously for 0.15 s.

### Mechanism
Screw threads and frame A's four nut sleeves are SDF meshes (0.5 mm pitch, right-hand): a screw advances only by being turned. Retention welds to frame A switch on when the scene's acceptance gates hold. The tray is never jointed.

### Provenance
- Scene and demo: the IKEA MuJoCo tree (`scenes/gladom/`, `demos/gladom/actions.npz`), MuJoCo 3.14.0, RoboARC rigid settings
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
- 2026-10-21: RoboARC 1.1.1, image references moved to ghcr.io/mll-lab-nu (same images, same digests): the
  Dockerfiles, compose files and README name `ghcr.io/mll-lab-nu/<image>:<tag>@sha256:<digest>` with the digests of
  1.1.0 (images/pins.json). The runtime binding is unchanged, and nothing else changed: results graded under 1.1.0
  stand.
- 2026-10-27: three appearance seeds: the task allows seeds 0, 1 and 2 (`[metadata.seeds]`, `build = "runtime"`: the
  runner picks the trial's seed and writes it to the files `seed`), as its standard twin, whose camera images seeds 1
  and 2 light differently. This mode shows the agent no images and its replay video keeps the scene's own lights, so
  here every seed is the same task. The scene, the physics, the instruction, the grader, the tooling and the reference
  are unchanged: results graded under 1.1.x stand.
