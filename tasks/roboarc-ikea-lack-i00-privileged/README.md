# roboarc-ikea-lack-i00-privileged

| | |
|---|---|
| Benchmark | RoboARC |
| Mode | privileged |
| Twin | `roboarc-ikea-lack-i00-standard` |
| Simulator / robot | `mujoco` / `franka-panda-bimanual` |
| Family / instance | `ikea_lack` / seed 0 |
| Oracle | `full` |
| Agent time budget | 10800 s; tokens: protocol default (PROTOCOL §5) |
| Base image | `ghcr.io/embodied-first-exam/roboarc-ikea:1.0.0` |

## Task
IKEA LACK side table (AA-207276-4, four legs) in plain MuJoCo 3.14.0 (CPU, float64) with two 7-DoF Franka Panda arms, each with the MuJoCo Menagerie Panda hand (two-finger, tendon-coupled): produce one replayable trajectory. Screw all four legs onto the four threaded studs of the tabletop.

## Difficulty
Bimanual furniture assembly with physical grasps (no assist), 16-D actuator action at 400 Hz
(4 physics steps of 0.625 ms per row); the reference needs 67697 rows
(169.24 s of simulated time).

## Reference solution
`solution/solve.sh` copies `solution/trajectory.npz` (sha256 `21764c5a84d77fba…`) to `/app/output/`: the task's one reference, shared with the standard twin (its `solution/oracle_actions.npz`, the same rows): the robot demo of the IKEA MuJoCo tree (`demos/lack/actions.npz`), time-dilated to the robots' real joint-speed limits by `scripts/ikea_mujoco/standard/std_reference.py` (each row whose gripper command steps faster than the Franka Hand's speed held until the hand reaches it, `--grip wait`), so the speed clamp never changes a row. The demo itself commands faster than the limits and fails under the clamp.

## Verification
Open-loop replay of `/app/output/trajectory.npz` from the scene's initial state in two fresh processes at the same time
(`tests/grade.py`, `tests/rcb_ikea.py replay`: per row `data.ctrl[:] = row; S.step(4)` with the scene's mechanism, then
`final_check()`: `success.py`, released, at rest and no breakdown at the end of the rows and again after a 1.0 s hold of
the last row). Pass iff both replays pass and end in the same state hashes (`deterministic`). The
scene in the verifier image must match `tests/runtime_binding.json` file by file (`instance_exact`) and build to its
initial-state hash (`instance_ok`). Replay video (evidence only, never the verdict): a third process replays the same rows and renders `/logs/verifier/replay.mp4` (CPU OSMesa, 512 x 288, at most 750 frames over the episode and the 1.0 s hold; `video_rendered`).

Harbor checks (2026-10-18, host B (Ryzen 9 7950X; shared, heavily loaded: load 40-90 on 32 threads), Docker, Harbor 0.23.0, harbor_run_safe.sh (fresh mtime base), image 0.3.1 sha256:b9552184c2de, the replay-video verifier, the merged dev/robotangle_mujoco head (the Franka Hand slew), image `ghcr.io/mll-lab-nu/eai-ikea-mujoco:0.3.1` id `b9552184c2de`): `oracle` scored 1 (two replays, 772.1 s wall); `oracle` scored 1 (two replays, 626.2 s wall); `nop` scored 0. Image 0.3.2 (kyrre's native-cylinder scene; this scene byte-identical to 0.3.1), 2026-10-19: a fresh replay of the reference ends and settles on the same hashes (`684fe3c488e0` / `a4bc18f244c3`, host B). Reference timing against the 10800 s budget: one unobserved replay of the hand-slew reference 372 s (3.4 %; 8.6 % of the 4320 s bar) on the quiet i9-14900HX host (one process under the shared workstation lock, image 0.3.1's scene); 1342 s on the loaded host B Ryzen 9 7950X (load 40-90); 1127 s on a cluster Xeon Gold 6230R core including the re-time step (reported only), end and settled hashes equal on every host and in the Harbor replays.

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
| Scene | `/opt/ikea/scenes/lack/` in the image; digest `e669195502062a45…` (scripts/ikea_mujoco/assets.json) |
| Initial state | sha256 `0bddce57c9bcaf34…` of the mjSTATE_INTEGRATION vector |

### Goal
Screw all four legs onto the four threaded studs of the tabletop.

### Success requirements
- The tabletop lies studs up. Each leg (nut end down) is screwed onto a stud; any leg may go on any stud, one leg per stud.
- Every leg is seated and latched on a distinct stud, and at the end, measured in the tabletop frame: the leg origin is 0 to 12 mm above the slab top, within 20 mm laterally of its stud, and the leg axis is within 10 degrees of the tabletop axis.

### Mechanism
The M16 x 2 right-hand threads are exact convex pieces: a leg advances only by being turned (about six turns from first thread contact to the seat). A leg that is seated on a vacant stud and descends slower than 0.05 m/s is latched to the tabletop at its current pose (a weld, the source's fixed joint).

### Provenance
- Scene and demo: the IKEA MuJoCo tree (`scenes/lack/`, `demos/lack/actions.npz`), MuJoCo 3.14.0, RoboARC rigid settings
  (timestep 0.625 ms, Newton, 100 iterations, elliptic cones).
- The scene files are not in this repository: the image carries them (`images/ikea-mujoco/README.md`, build context from
  `scripts/ikea_mujoco/stage_context.py`, pinned by `scripts/ikea_mujoco/assets.json`).
