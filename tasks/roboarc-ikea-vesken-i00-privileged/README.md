# roboarc-ikea-vesken-i00-privileged

| | |
|---|---|
| Benchmark | RoboARC |
| Mode | privileged |
| Twin | `roboarc-ikea-vesken-i00-standard` |
| Simulator / robot | `mujoco` / `franka-panda-bimanual` |
| Family / instance | `ikea_vesken` / seed 0 |
| Oracle | `full` |
| Agent time budget | 10800 s; tokens: protocol default (PROTOCOL §5) |
| Base image | `ghcr.io/embodied-first-exam/roboarc-ikea:1.0.0` |

## Task
IKEA VESKEN shelf unit (AA-1842283-5), body steps 1-8 in plain MuJoCo 3.14.0 (CPU, float64) with two 7-DoF Franka Panda arms, each with the MuJoCo Menagerie Panda hand (two-finger, tendon-coupled): produce one replayable trajectory. Build the VESKEN shelf unit body: snap the four feet into the bottom tray from below, the four columns into the bottom tray, the second tray onto the columns, four columns and a tray for each of the two upper levels, and the four caps into the top tray.

## Difficulty
Bimanual furniture assembly with physical grasps (no assist), 16-D actuator action at 400 Hz
(4 physics steps of 0.625 ms per row); the reference needs 389504 rows
(973.8 s of simulated time).

## Reference solution
`solution/solve.sh` copies `solution/trajectory.npz` (sha256 `20c14565036f2ea3…`) to `/app/output/`: the task's one reference, shared with the standard twin (its `solution/oracle_actions.npz`, the same rows): a closed-loop reference built on the final physics by `ikea/scripts/ikea_mujoco/oracles/vesken/vesken_oracle.py`: the demo as the phase guide, one operation per snap; each grasp approach offset to where the part really is, each carried part held on its reference pose (near the host: on its connector error) by an integral loop through damped-least-squares IK, a gentle press until the weld engages, one arm move re-pathed in joint space to avoid a self-contact, the right hand lifting the bottom tray clear of a passing left wrist, two wrist rolls that keep caps from tipping or being hit; the rows are the targets executed within the joint-speed and Franka Hand speed limits. It never trips the speed clamp (arm joints and Franka Hand).

## Verification
Open-loop replay of `/app/output/trajectory.npz` from the scene's initial state in two fresh processes at the same time
(`tests/grade.py`, `tests/rcb_ikea.py replay`: per row `data.ctrl[:] = row; S.step(4)` with the scene's mechanism, then
`final_check()`: `success.py`, released, at rest and no breakdown at the end of the rows and again after a 1.0 s hold of
the last row). Pass iff both replays pass and end in the same state hashes (`deterministic`). The
scene in the verifier image must match `tests/runtime_binding.json` file by file (`instance_exact`) and build to its
initial-state hash (`instance_ok`). Replay video (evidence only, never the verdict): a third process replays the same rows and renders `/logs/verifier/replay.mp4` (CPU OSMesa, 512 x 288, at most 750 frames over the episode and the 1.0 s hold; `video_rendered`).

Harbor checks (2026-10-18, a Ryzen 9 7950X workstation (shared, loaded), Docker, Harbor 0.23.0, image 0.3.0 sha256:ed950d079b63, the replay-video verifier, image `ghcr.io/mll-lab-nu/eai-ikea-mujoco:0.3.0` id `ed950d079b63`): `oracle` scored 1 (two replays, 792.1 s wall); `nop` scored 0. Image 0.3.1 (the shared welded-part contact switch-off, off in this scene) on the merged head: a fresh replay of the reference ends and settles on the same hashes as in 0.3.0 (`943aa32ee13e` / `29685e37f6fe`, on a Ryzen 9 7950X workstation, 2026-10-18). Image 0.3.2 (kyrre's native-cylinder scene; this scene byte-identical to 0.3.1), 2026-10-19: a fresh replay of the reference ends and settles on the same hashes (`943aa32ee13e` / `29685e37f6fe`, on the same workstation). Reference timing against the 10800 s budget: one unobserved replay 1822 s (16.9 %, inside the 40 % bar) on a dedicated Xeon Gold 6230R core (2.1 GHz, a slow trial-class CPU), 983 s on a Xeon Platinum 8592+ core, 748-761 s on a Ryzen 9 7950X workstation (loaded); end and settled hashes equal to the Harbor runs'.

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
| Scene | `/opt/ikea/scenes/vesken/` in the image; digest `0164bf56e1ea3810…` (ikea/scripts/ikea_mujoco/assets.json) |
| Initial state | sha256 `66aaaacbc3da0fb3…` of the mjSTATE_INTEGRATION vector |

### Goal
Build the VESKEN shelf unit body: snap the four feet into the bottom tray from below, the four columns into the bottom tray, the second tray onto the columns, four columns and a tray for each of the two upper levels, and the four caps into the top tray.

### Success requirements
- All 23 snap connections are made in the source order (each depends on the previous: feet, columns, tray, columns, tray, columns, tray, caps) and, at the end, each connector is within its gates (axial gap 0.6 mm, lateral 0.8 mm, outward-axis tilt 2 degrees).
- The feet stand on the bench (summed vertical contact force above 0.1 N) and the bottom tray is within 3 degrees of its upright, unrotated assembly orientation.
- No arm link touches any furniture part or jaw at the end.

### Mechanism
32 spring jaws (slide joints with springs, 1200 N/m, 2.5 mm travel) are deflected by the hooked snap pins through contact. A retention weld per connection switches on when the acceptance gates hold for 0.15 s (snap passed, jaw returned, seated, at rest).

### Provenance
- Scene and demo: the IKEA MuJoCo tree (`scenes/vesken/`, `demos/vesken_vlim/actions.npz`), MuJoCo 3.14.0, RoboARC rigid settings
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
