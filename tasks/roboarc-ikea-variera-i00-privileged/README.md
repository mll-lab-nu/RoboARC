# roboarc-ikea-variera-i00-privileged

| | |
|---|---|
| Benchmark | RoboARC |
| Mode | privileged |
| Twin | `roboarc-ikea-variera-i00-standard` |
| Simulator / robot | `mujoco` / `franka-panda-bimanual` |
| Family / instance | `ikea_variera` / seed 0 |
| Oracle | `full` |
| Agent time budget | 10800 s; tokens: protocol default (PROTOCOL §5) |
| Base image | `ghcr.io/embodied-first-exam/roboarc-ikea:1.0.0` |

## Task
IKEA VARIERA shelf insert (AA-741461-4), step 1 in plain MuJoCo 3.14.0 (CPU, float64) with two 7-DoF Franka Panda arms, each with the MuJoCo Menagerie Panda hand (two-finger, tendon-coupled): produce one replayable trajectory. Assemble the VARIERA shelf insert: place the two side frames, lower the shelf onto them, and drive the four screws through the shelf holes into the frames' receivers until they seat.

## Difficulty
Bimanual furniture assembly with physical grasps (no assist), 16-D actuator action at 400 Hz
(4 physics steps of 0.625 ms per row); the reference needs 243159 rows
(607.9 s of simulated time).

## Reference solution
`solution/solve.sh` copies `solution/trajectory.npz` (sha256 `2e821fcfb52beacb…`) to `/app/output/`: the task's one reference, shared with the standard twin (its `solution/oracle_actions.npz`, the same rows): a closed-loop reference built on the final physics by `ikea/scripts/ikea_mujoco/oracles/variera/build.py`: the time-dilated demo (with its release tail) as the phase guide, each row held until the slewed Franka Hand reaches its command; each carried side frame held on the unslewed run's pose by damped-least-squares IK and released in steps (eased off to a light touch, then a small gap, then open) so the standing frame does not rock off its place; where the demo's screwdriver spin ends short of the seat, the spin goes on with the tip on the true screw axis until the head seats, then torque-free before the lift; the rows are the targets executed after the speed clamp. It never trips the speed clamp (arm joints and Franka Hand).

## Verification
Open-loop replay of `/app/output/trajectory.npz` from the scene's initial state in two fresh processes at the same time
(`tests/grade.py`, `tests/rcb_ikea.py replay`: per row `data.ctrl[:] = row; S.step(4)` with the scene's mechanism, then
`final_check()`: `success.py`, released, at rest and no breakdown at the end of the rows and again after a 1.0 s hold of
the last row). Pass iff both replays pass and end in the same state hashes (`deterministic`). The
scene in the verifier image must match `tests/runtime_binding.json` file by file (`instance_exact`) and build to its
initial-state hash (`instance_ok`). Replay video (evidence only, never the verdict): a third process replays the same rows and renders `/logs/verifier/replay.mp4` (CPU OSMesa, 512 x 288, at most 750 frames over the episode and the 1.0 s hold; `video_rendered`).

Harbor checks (2026-10-18, host B (Ryzen 9 7950X; shared, loaded), Docker, Harbor 0.23.0, image 0.3.0 sha256:ed950d079b63, the replay-video verifier, image `ghcr.io/mll-lab-nu/eai-ikea-mujoco:0.3.0` id `ed950d079b63`): `oracle` scored 1 (two replays, 3731.4 s wall); `nop` scored 0. Image 0.3.1 (the shared welded-part contact switch-off, off in this scene) on the merged head: a fresh replay of the reference ends and settles on the same hashes as in 0.3.0 (`95134e1fbd5c` / `327c16c5f007`, host B, 2026-10-18). Image 0.3.2 (kyrre's native-cylinder scene; this scene byte-identical to 0.3.1), 2026-10-19: a fresh replay of the reference ends and settles on the same hashes (`95134e1fbd5c` / `327c16c5f007`, host B). Reference timing against the 10800 s budget: one unobserved replay 2664 s (24.7 %) on a quiet i9-14900HX host; 3646-3688 s (34.0 %) on host B's Ryzen 9 7950X (loaded, load 50-60); 8024-8696 s (74.3-80.5 %, OVER the 40 % bar) on a dedicated the cluster Xeon Gold 6230R core (2.1 GHz, a slow trial-class CPU); end and settled hashes equal to the Harbor runs' on all three.

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
| Scene | `/opt/ikea/scenes/variera/` in the image; digest `23a2fca95f651e68…` (ikea/scripts/ikea_mujoco/assets.json) |
| Initial state | sha256 `067105ad8f8329ec…` of the mjSTATE_INTEGRATION vector |

### Goal
Assemble the VARIERA shelf insert: place the two side frames, lower the shelf onto them, and drive the four screws through the shelf holes into the frames' receivers until they seat.

### Success requirements
- Four distinct receivers each hold a screw that was turned relative to that receiver (at least 0.4 measured turns, with a descent matching turns x pitch), its head seated at the shelf hole (axial 0.2 mm, lateral 0.6 mm, axis 1.5 degrees).
- The frames are aligned under the shelf, parts are slower than 0.02 m/s, the assembly stands on the bench on its own, and no robot touches any furniture part.
- All of the above hold for 0.5 s.

### Mechanism
Screw threads and receivers are non-convex SDF meshes (0.5 mm pitch, right-hand): a screw advances only by being turned. There are no welds or other non-contact mechanisms.

### Provenance
- Scene and demo: the IKEA MuJoCo tree (`scenes/variera/`, `demos/variera/actions.npz`), MuJoCo 3.14.0, RoboARC rigid settings
  (timestep 0.625 ms, Newton, 100 iterations, elliptic cones).
- The scene files are not in this repository: the image carries them (`ikea/images/ikea-mujoco/README.md`, build
  context from `ikea/scripts/ikea_mujoco/stage_context.py`, pinned by `ikea/scripts/ikea_mujoco/assets.json`).
