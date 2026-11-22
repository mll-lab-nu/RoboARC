# roboarc-ikea-uppsta-i00-privileged

| | |
|---|---|
| Benchmark | RoboARC |
| Mode | privileged |
| Twin | `roboarc-ikea-uppsta-i00-standard` |
| Simulator / robot | `mujoco` / `franka-panda-bimanual` |
| Family / instance | `ikea_uppsta` / seed 0 (allowed seeds: [0, 1, 2]; §2.13) |
| Oracle | `full` |
| Agent time budget | 11400 s; tokens: protocol default (PROTOCOL §5) |
| Base image | `ghcr.io/mll-lab-nu/roboarc-ikea:1.1.0@sha256:2c56d9226aaf97a325b610d464a1cd81e87db1a6285e923539397f493e21136b` |

## Task
IKEA UPPSTÅ toy hammering block (AA-2300462-4), all manual steps and the 8 pegs in plain MuJoCo 3.14.0 (CPU, float64) with two 7-DoF Franka Panda arms, each with the MuJoCo Menagerie Panda hand (two-finger, tendon-coupled): produce one replayable trajectory. Build the UPPSTÅ hammering block: set two dowels into one board end, press end block b onto them and drive screw b with the hex key; turn it over, set the other two dowels, press end block a on and drive screw a; stand the block on its feet and push the eight pegs into their holes.

## Difficulty
Bimanual furniture assembly with physical grasps (no assist), 16-D actuator action at 800 Hz
(4 physics steps of 0.3125 ms per row); the reference needs 430225 rows
(537.78 s of simulated time).

## Reference solution
`solution/solve.sh` copies `solution/trajectory.npz` (sha256 `69cb5194277c444e…`) to `/app/output/`: the task's one reference, shared with the standard twin (its `solution/oracle_actions.npz`, the same rows): a closed-loop reference built on the final physics by `ikea/scripts/ikea_mujoco/oracles/uppsta/search.py` (`uppsta_oracle.py`): the guide is the author's demo (each 2.5 ms tick reached over its two rows) run in the scene without the prism rule, where it assembles the block; segment by segment (the arms' releases), each grasp approach offset to where the part really is, each carried part held on its guide pose (relative to the board near it) by an integral loop through damped-least-squares IK, the hex key's tip servoed onto the hole's axis, a screw's spin matched modulo 60 degrees, a part pressed a little deeper before its release where it would tip over; away from every grasp and release the demo is followed at every 4th row where the segment's gate allows (`ref/tempo.npz`); the rows are the targets executed within the joint-speed limits and the Franka Hand's speed. It never trips the speed clamp (arm joints and Franka Hand).

## Verification
Open-loop replay of `/app/output/trajectory.npz` from the scene's initial state in two fresh processes at the same time
(`tests/grade.py`, `tests/rcb_ikea.py replay`: per row `data.ctrl[:] = row; S.step(4)` with the scene's mechanism, then
`final_check()`: `success.py`, released, at rest and no breakdown at the end of the rows and again after a 1.0 s hold of
the last row). Pass iff both replays pass and end in the same state hashes (`deterministic`). The
scene in the verifier image must match `tests/runtime_binding.json` file by file (`instance_exact`) and build to its
initial-state hash (`instance_ok`). Replay video (evidence only, never the verdict): a third process replays the same rows and renders `/logs/verifier/replay.mp4` (CPU OSMesa, 512 x 288, at most 750 frames over the episode and the 1.0 s hold; `video_rendered`).

Harbor checks (2026-10-18, a Ryzen 9 7950X workstation (shared, load ~15-20), Docker, Harbor 0.23.0, fresh per-run file mtimes, image 0.3.1 sha256:b9552184c2de, the replay-video verifier, image `ghcr.io/embodied-first-exam/roboarc-ikea:1.0.0` id `5d5090ea1910`): `oracle` scored 1 (two replays, 4157.2 s wall); `nop` scored 0. Merged head with the Franka Hand slew in the shared rule (2026-10-18): a fresh replay in image 0.3.1 ends and settles on the same hashes (`a748a42dc36d` / `a2d8a234abc3`, on a Ryzen 9 7950X workstation, 7676 s at load 40-90): the reference was built within the slew. Image 0.3.2 (kyrre's native-cylinder scene; this scene byte-identical to 0.3.1), 2026-10-19: a fresh replay of the reference ends and settles on the same hashes (`a748a42dc36d` / `a2d8a234abc3`, on the same workstation). Reference timing: 3199 s (29.6 % of the 10800 s budget, 74 % of the 4320 s bar) on a quiet i9-14900HX host (one process at a time); 3519 s (32.6 %) on a Ryzen 9 7950X workstation (Docker, image 0.3.1, host load 12-17); one unobserved replay 8414-8425 s (78.0 %, OVER the 40 % bar) on a dedicated Xeon Gold 6230R core (2.1 GHz, a slow trial-class CPU), end and settled hashes equal to the Harbor runs', as gladom / variera / stenerik on that CPU.

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
| Scene | `/opt/ikea/scenes/uppsta/` in the image; digest `8c884957c9be172b…` (ikea/scripts/ikea_mujoco/assets.json) |
| Initial state | sha256 `3b69cfd57f5a5330…` of the mjSTATE_INTEGRATION vector |

### Goal
Build the UPPSTÅ hammering block: set two dowels into one board end, press end block b onto them and drive screw b with the hex key; turn it over, set the other two dowels, press end block a on and drive screw a; stand the block on its feet and push the eight pegs into their holes.

### Success requirements
- Both screws are seated and their seat welds engaged, each in its own hole within 0.6 mm along and across the hole axis, its axis within 2 degrees (spin free).
- Measured in the board frame against the scene's 'assembled' keyframe: both end blocks within 0.6 mm and 1 degree; every dowel position holds a dowel within 0.6 mm, its axis within 2 degrees (the four dowels are interchangeable).
- Every peg hole holds a peg (the eight are interchangeable) within 1 mm of the hole axis, its axis within 2 degrees, pushed down to the up position (at most 0.5 mm above it).
- The block stands on its feet (the board's assembled up direction within 1 degree of world +z) and no robot body touches any part at the end.

### Mechanism
Each screw's coarse right-hand thread (2.5 mm pitch) runs in a helical thread in the board end: a screw advances only by being turned, by the hex key in its socket. Dowels and pegs hold by compliant flutes (spring-loaded slide joints, an interference fit) and friction. A screw's seat weld is armed once it is on its hole's axis within 3 mm of its seat; once its head touches and it has stopped relative to the board for 0.05 s, the screw and the end block it clamps are welded to the board at their current pose, and the screw and its hole's thread stop colliding. The scene's tick is 8 physics steps (2.5 ms); a row here is 4 of them (1.25 ms, 800 Hz).

### Provenance
- Scene and demo: the IKEA MuJoCo tree (`scenes/uppsta/`, `demos/uppsta_vlim/actions.npz`), MuJoCo 3.14.0, RoboARC rigid settings
  (timestep 0.3125 ms, Newton, 100 iterations, elliptic cones).
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
