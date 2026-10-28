# roboarc-ikea-ikea365-i00-standard

| | |
|---|---|
| Benchmark | RoboARC |
| Mode | standard (`eai-standard/2.4`) |
| Twin | `roboarc-ikea-ikea365-i00-privileged` |
| Simulator / robot | `mujoco` / `franka-panda-bimanual` |
| Family / instance | `ikea_ikea365` / seed 0 (allowed seeds: [0, 1, 2]; §2.13) |
| Oracle | `full` |
| Agent time budget | 10800 s; tokens: protocol default (PROTOCOL §5) |
| Base image | `ghcr.io/mll-lab-nu/roboarc-ikea-sim:1.1.0@sha256:f22a1e9042c7877d33703960313cf548f2af7f15d6fcfa61db83315bfdacba1e` |

## Task
IKEA 365+ serving stand (AA-2568481-2) in plain MuJoCo 3.14.0 (CPU, float64) with two 7-DoF Franka Panda arms, each with the MuJoCo Menagerie Panda hand (two-finger, tendon-coupled), through the robot service (eai-standard/2.4; 2.1 / 2.2
requests are served unchanged): RGB from four ALOHA-style cameras (optional depth), proprioception, `joint_position`
chunks, stateless kinematics, every pose in the robot's frame; no reset and no privileged state. Rebuild the IKEA 365+ serving stand: take the screw out of member A and park it, lay member B across member A, bring the screw back and screw it in through both members until it seats, turn the stand over, and place the bowl on it.

## Difficulty
The privileged twin's bimanual assembly with physical grasps, from cameras and proprioception only, in one episode
without reset; 16-D actuator action at 400 Hz, joint targets within the robot's joint
velocity limits (joints 1-4 2.175 rad/s, joints 5-7 2.61 rad/s, the hand command 182.143 units/s). The reference needs 96000 control steps (240.0 s of simulated
time).

## Reference solution
`solution/solve.sh` runs `solution/oracle_replay.py`: it submits the control steps of `solution/oracle_actions.npz` through
the client as `joint_position` chunks of 10 s, `hold`, one control step per row, as float64 (the client's transport;
`Robot.submit` sends float32), so the service executes exactly those rows. The rows are the task's one reference, the same file as the privileged twin's `solution/trajectory.npz`: a closed-loop reference built on the final physics by `ikea/scripts/ikea_mujoco/oracles/ikea365/build_ref.py`: the demo through the clamp as the phase guide; the right hand threads the returned screw itself before releasing it; the screwdriver is centred on the true screw axis, spins with a 5-15 N axial push and goes torque-free at the seat so the welds engage on an unloaded screw; guide tracks (damped-least-squares IK) steer the hands onto member B, the turned stand and the bowl; each demo row held (the guide tracks still running) until the slewed Franka Hand reaches its command, and the returned screw servoed onto its return gate's pose in member A (the slew shifts the screw in the grasp); the rows are the targets executed after the joint-speed and hand-speed clamp,
so the service's speed limiter never has to act (`n_clipped` 0).

## Verification
The sim service (`environment/sim/server.py`) steps the scene's own runtime (`rcb_ikea.Session.step`: `data.ctrl[:] =
row; S.step(4)`), applies the joint-speed rule, evaluates the grader's terminal rules (`rcb_ikea.Session.status()`:
success.py, released, at rest, no breakdown) after every call and, when they hold, `Session.final_check()` (the same after
a 1.0 s hold of the last row), both on a snapshot restored bit-exactly afterwards, ends the episode at the first settled
success or at a breakdown, and journals every executed control step; at the end it writes
`/data/trajectory.npz` (all executed control steps, float64: the replayable deliverable) and the final state hash; an
episode that ends otherwise (timeout) is judged as it stands by the same two checks. Before the episode the service checks
the scene file by file against `runtime_binding.json` and by its initial-state hash (`instance_mismatch`, infrastructure).
The verifier (`tests/grade.py`, separate sandbox, no network, the shared client base) transcribes `final.json`: pass iff
the live episode succeeded (`graded_by_live` 1, `replays` 0). No grading replay runs (since 2026-10-17).

Before 2026-10-17 (graded then by the verifier's two replays; history, not re-run since): Harbor checks on a Ryzen 9 7950X / RTX 4090 workstation (shared, heavily loaded: load 40-90), Harbor 0.23.0, image 0.3.1 sha256:b9552184c2de / -sim 0.4.1 (Nyx RGB, 54694704aa75); graded by the live episode (2026-10-18): `oracle` scored 1 (the hand-slew reference: live settled success at control step 96000 (24 calls), live end hash 3250d0d9de31 (a chunk end before the reference's last row), video 1; live service time 1285 s on the loaded host); `nop` scored 0 (video 1).

## Compute resources
Minimum allocation for one trial (docs/PROTOCOL.md §2.9; `[metadata.resources]` in `task.toml`). The agent phase counts every container of the trial at once (the agent's, and in standard modes the simulator sidecar); the verifier phase runs after it.

| | Agent phase | Verifier |
|---|---|---|
| CPU cores | 2 | 2 |
| Memory | 3 GB | 2 GB |
| GPU memory | 1 GB | 0 GB |

- GPU: required, any NVIDIA GPU; recommended: RTX 4090 24GB / L4 24GB (an NVIDIA GPU with Vulkan and EGL).
- Measured: Harbor standard oracle of svallet on an RTX 4090 host, 2026-10-17 (docker stats and nvidia-smi per process every 20 s): the sim sidecar peaked at 1.8 GiB of memory and 0.8 GiB of GPU memory (the Nyx render process), the agent's container (the client) below 0.1 GiB; the privileged verifier of the same task (two replays and the video replay) at 1.6 GiB; x 1.5 and rounded up
- Limits in the agent phase (docs/PROTOCOL.md §2.10; `[metadata.resources.limits]`, shown to the agent in `instruction.md`; always measured and reported, and exceeding one fails the trial only when the run sets `RCB_LIMITS_ENFORCE=1`), the agent's container: no GPU, 8 GiB memory, 512 processes, 20 GiB of new files under `/app`, `/tmp`, `/root`; 60 s within 120 s. Expected peak of the whole trial: 4 GiB GPU memory, 16 GiB memory. Limits measured: the sidecar's measured peaks (1.8 GiB, 0.8 GiB of GPU memory, svallet) with room for the longer scenes and an agent's own client processes; the agent's container runs the client only.

## Details
| | |
|---|---|
| Grasping | physical (finger friction), no assist |
| Success rule | the scene's `success.py` + the terminal rules (released, at rest, no breakdown), on the live episode: after every call and at its end, at the last row and again after a 1.0 s hold of it (on a copy) |
| Frame | every pose the agent gets or gives is in the robot frame: origin at the left arm's base link, +z up (`spec.world`); the scene's own world frame never leaves the sim service |
| Speed rule | targets within the robot's real limits (joints 1-4 2.175 rad/s, joints 5-7 2.61 rad/s, the hand command 182.143 units/s; the table and its sources: `rcb_ikea.JOINT_SPEED_LIMITS_RAD_S` / `GRIPPER_WIDTH_SPEED_M_S`), clamped per control step, as the privileged twin's Session; faster rows take the control steps they need (`meta.clipped`), as RoboARC's rule |
| Cameras | `cam_high`, `cam_low` (static), `cam_left_wrist`, `cam_right_wrist` (on the hands); 848 x 480, fovy 58 deg; added at service start (`environment/sim/ikea_rig.py`, `rig.json`), the model is unchanged; RGB: Nyx, RoboARC's renderer and look (path tracer, 32 spp, denoised; studio HDRI and point lights; `environment/sim/ikea_nyx.py` with RoboARC's `nyx_direct.py`); depth: MuJoCo's raster; on the sidecar's GPU |
| Video | `/data/live.mp4` (cam_high, 10 frames per simulated second, recorded during the live episode), copied to the verifier logs; never affects the verdict |
| Scene | `/opt/ikea/scenes/ikea365/` in the images; digest `c16a27fdfc97e8c9…` |
| Initial state | sha256 `4e9e040022202b3e…` of the mjSTATE_INTEGRATION vector |

### Budget
The wall clock only, the privileged twin's cutoff (PROTOCOL v2.0 H.7): `[agent] timeout_sec` = 10800 s. The service
counts the same limit from the agent's first request (`spec.budget.wall_clock_s`, `meta.time_left_s`) and ends the
episode with reason `timeout` when it runs out; that count never runs out before Harbor's.

### Deviations
- `ee_pose` is not offered (only `joint_position`, the scene's actuator layout); `/kinematics` gives FK / IK.
- Graded by the live episode (the settled check on the live simulator); no grading replay, in the sidecar or the
  verifier (since 2026-10-17).

### Goal
Rebuild the IKEA 365+ serving stand: take the screw out of member A and park it, lay member B across member A, bring the screw back and screw it in through both members until it seats, turn the stand over, and place the bowl on it.

### Success requirements
- The seven operations are accepted in this order: remove the screw, park the screw, cross the members, return the screw, reinstall the screw, turn the stand, place the bowl.
- At the end the screw is retained in member A and aligned with its seated pose relative to both members, and member B is retained and aligned with member A (gates: axial 0.1 mm, lateral 0.6 mm, axis 1.5 degrees; member B may be rotated 180 degrees).
- Released and settled: no robot contact with any part, every part slower than 0.025 m/s, the bench carries the parts; the bowl touches the members and member A is turned over (its z axis pointing down).
- A support probe on a copy of the final state: the bench must carry the whole weight, and the weight plus a 2 N downward test load on member A (the grader applies it; it never acts during your episode), with the arms holding their last commands.

The benchmark's terminal rules, the same in every IKEA MuJoCo task, come on top of these requirements. When your rows end: (1) the requirements above hold; (2) the parts are released: no robot link touches any part; (3) every part is at rest: at most 0.01 m/s and 0.05 rad/s; (4) the simulation never broke down: no non-finite state, no MuJoCo BADQACC / BADQPOS / BADQVEL warning, no contact of a part deeper than 10 mm, at any time. The grader then holds your last row for 1.0 s more and requires (1)-(4) again at its end.

### Mechanism
The M4 x 0.5 thread is exact convex pieces: the screw advances only by being turned. Two retention welds (screw to member A, member B to member A) switch on with the current relative pose once the reinstalled screw is seated.

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
- 2026-10-22: RoboARC 1.2.1, the Nyx render process checks the GPU before it starts the renderer
  (`environment/sim/nyx_direct.py`: the Vulkan extensions of the sidecar's GPU, `VK_KHR_ray_query` first): on a GPU
  without RT cores (such as V100, A100 and H100) it reports what is missing and the sim service stops at once with that
  error (`ikea_nyx.UnsupportedGPU`, no start-up retry), instead of the render process crashing three times. This mode
  needs a GPU with RT cores. On a GPU with RT cores nothing changes; the images, the instruction, the scene, the
  graders, the tooling and the reference are unchanged: results graded under 1.1.x stand.
- 2026-10-27: three appearance seeds: the task allows seeds 0, 1 and 2 (`[metadata.seeds]`, `build = "runtime"`: the
  runner picks the trial's seed and writes it to the files `seed`). Seed 0 is the released look, unchanged. Seeds 1 and
  2 light the scene differently in the Nyx images (`environment/sim/rig.json` `seeds`, which the sim service picks by
  `environment/sim/seed`): another of the four ceiling lights is the key light (three times the others, the only one
  casting shadows), the lights hang at another height, their white light is warm (3600-4300 K) or cool (8000-10000 K),
  and they and the environment map are dimmer (x 0.65-0.78) or brighter (x 1.25-1.50); the two seeds differ in each of
  these. The scene, the physics, the instruction, the graders, the tooling and the reference are the same for every
  seed: results graded under 1.1.x stand.
