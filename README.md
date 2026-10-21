# RoboARC

RoboARC is a benchmark of physical manipulation for coding agents, simulated on MuJoCo 3.14.0: topological
puzzles (wire and ring puzzles, interlocked cast puzzles, knots, ropes, cords and cloth) and IKEA furniture assembly. In every task a coding agent writes the code
that drives a simulated robot (two ALOHA 2 arms, a Franka Panda with a Robotiq 2F-85, or two Franka Pandas) and delivers one replayable
trajectory for a frozen scene instance. A separate verifier replays and grades it.

Website: <https://roboarc.embodied-first-exam.ai>

Version **1.2.0** (see [CHANGELOG.md](CHANGELOG.md)):
- the topological puzzles: 110 task bases in 18 families, each in two modes, so 220 Harbor task directories under
  `tasks/`. Every base has three appearance seeds (0, 1 and 2): the same scene, physics, rules and reference
  solution, other colours and materials; the runner picks the seed (`docs/RUNNING.md`, "Seeds");
- **IKEA MuJoCo**: 12 furniture-assembly tasks with two robot arms and physical grasps, each in two modes, so 24 task
  directories under `ikea/tasks/`. The scenes are by Xiaoyu's team (Xiaoyu, Zhen Wu). See [ikea/README.md](ikea/README.md).

## Families

| family | bases | robot | difficulty tiers |
|---|---:|---|---|
| battery-replacement | 1 | ALOHA 2 | no tier yet (a new design) |
| bed-bolt | 5 | ALOHA 2 | no tier yet (new designs) |
| blanket-flatten | 9 | ALOHA 2 | very_easy 2, easier 1, easy 2, medium 2, hard 2 |
| blanket-fold | 5 | ALOHA 2 | very_easy .. hard, 1 each |
| buzz-wire-3d | 10 | Franka + Robotiq 2F-85 | 2 per tier |
| caliper-sorting | 1 | ALOHA 2 | no tier yet (a new design) |
| cast-coil | 8 | ALOHA 2 | very_easy 2, easier 2, easy 2, medium 1, hard 1 |
| cast-keyhole | 6 | ALOHA 2 | very_easy 1, easier 1, easy 1, medium 1, hard 2 |
| clock-assembling | 1 | ALOHA 2 | no tier yet (a new design) |
| fiber-bundling | 10 | Franka + Robotiq 2F-85 | 2 per tier |
| hedgehog-in-cage | 8 | Franka + Robotiq 2F-85 | very_easy 1, easier 2, easy 2, medium 1, hard 2 |
| horseshoe-ring | 5 | ALOHA 2 | 1 per tier |
| interlocked-claws | 13 | ALOHA 2 | very_easy 2, easier 2, easy 2, medium 3, hard 3; 13 (Cast Trinity) no tier yet |
| knot-untying | 8 | ALOHA 2 | very_easy 2, easier 2, easy 1, medium 2, hard 1 |
| lock-picking | 5 | Franka + Robotiq 2F-85 | 1 per tier |
| orbits | 2 | ALOHA 2 | very_easy 1, easier 1 |
| rope-threading | 8 | ALOHA 2 | very_easy 1, easier 1, easy 2, medium 2, hard 2 |
| slotted-post-ring | 5 | ALOHA 2 | 1 per tier |

Task names are `roboarc-<family>-<NN>-i01-<mode>`. `<NN>` is the base's number in its family. Its difficulty is
`[metadata].difficulty` in `task.toml`. One page per family is in [docs/families/](docs/families/), and the benchmark's
rules (grading replays, safety readings, the standard service) are in [docs/benchmarks/roboarc.md](docs/benchmarks/roboarc.md).

IKEA MuJoCo (`ikea/tasks/roboarc-ikea-<scene>-i00-<mode>`): lack, svallet, gladom, variera, eket, ikea365, vesken, stegesund,
stenerik, burvik, uppsta and kyrre. Each is an IKEA product assembled by two Franka Panda arms (svallet: ALOHA 2) in
plain MuJoCo 3.14.0. Its page is [ikea/docs/benchmarks/ikea_mujoco.md](ikea/docs/benchmarks/ikea_mujoco.md).

## Modes

- **privileged** (`-privileged`): the agent's container holds the simulator (the `topogym` package and the tooling
  `rcb_topogym.py`). The agent may reset, read the full state and replay as often as it likes. It delivers
  `/app/output/trajectory.npz`. The verifier replays it twice in a fresh process.
- **standard** (`-standard`, protocol eai-standard/2.4): the agent's container has only Python and a client. The
  simulator runs in a `sim` sidecar the agent cannot open. The agent gets RGB (optional depth), proprioception and
  calibrated cameras, and submits joint or end-effector chunks. There is no reset and no privileged state. The service
  records and grades the live episode, and a fresh-process replay confirms it.

Both modes run on the same frozen instance, with the same 3 h agent budget (`[agent] timeout_sec` = 10800 s).

## Quick start

Requirements:
- Linux x86_64 with an NVIDIA GPU that has RT cores (driver >= 580; Nyx renders through Vulkan);
- Docker with the NVIDIA Container Toolkit;
- [Harbor](https://github.com/laude-institute/harbor) 0.23 or newer;
- Python >= 3.11 for the repository tools.

The image's hardware notes are in [images/roboarc/README.md](images/roboarc/README.md).

1. **Get the images.** The puzzle tasks run `ghcr.io/embodied-first-exam/roboarc:1.2.0`. The IKEA tasks run
   `ghcr.io/embodied-first-exam/roboarc-ikea:1.0.0`, and their standard `sim` sidecar runs `roboarc-ikea-sim:1.0.0`.
   Pull them, or build them from this repository (step 3).
2. **Run a task with Harbor:**

   ```bash
   harbor run -p tasks/roboarc-interlocked-claws-03-i01-privileged -a oracle      # the reference solution: reward 1
   harbor run -p tasks/roboarc-interlocked-claws-03-i01-standard  -a nop         # does nothing: reward 0
   harbor run -p tasks/roboarc-orbits-02-i01-standard -a codex -m openai/<model> # a coding agent
   harbor run -p ikea/tasks/roboarc-ikea-lack-i00-privileged -a oracle           # an IKEA task
   ```

   Every trial needs one GPU. The knot-untying and rope-threading families need a 24 GB GPU each. Batch runs, agent
   configuration, egress and result files are described in [docs/RUNNING.md](docs/RUNNING.md). The task contract and
   scoring are in [docs/PROTOCOL.md](docs/PROTOCOL.md).
3. **Build the image** (optional, ~25 GB of disk):

   ```bash
   images/roboarc/build.sh                      # frozen instances from the pinned HF dataset (read access needed)
   images/roboarc/build.sh --instances <dir>    # or from a local copy (instances/<base>/ per base)
   ikea/images/ikea-mujoco/build.sh             # the two IKEA images (scene meshes from their pinned HF dataset)
   ```

   The build checks every instance file against `scripts/roboarc/assets.json`. It also checks that the image installs
   the runtime every task is bound to.

## Use the simulator directly

*Placeholder: a Python package `roboarc` (top-level `roboarc/`, reserved) will expose the tasks as Gymnasium-style
environments for VLA and RL use, next to Harbor (which stays the evaluation path).* It will import the same code the
graded tasks run, not a copy:
- the puzzles: the simulator package `source/topogym/` (`PYTHONPATH=source`) and the tooling
  `source/scripts/robotangle/shared/rcb_topogym.py` (`Session`), with each base's frozen instance from the image
  (`/opt/robotangle/instances/`) or the pinned dataset;
- IKEA: `ikea/scripts/ikea_mujoco/shared/rcb_ikea.py` (`Session`) on the scenes in `ikea/images/ikea-mujoco/scenes/`.

## Repository layout

| path | what |
|---|---|
| `tasks/roboarc-*` | the 220 Harbor task directories. They are generated: never edit them by hand |
| `scripts/roboarc/` | the generator `mk_tasks.py` and its inputs: `release.json`, `packages/` (per-task texts), `oracles/` (reference trajectories), `shared/` (the pinned tooling and graders), `assets.json` (instance pins) |
| `scripts/` | the shared task layer the tasks use: `task_conventions.py`, `eai_standard/` (the standard client and service layer), `rcb_monitor/` (the resource-limits sidecar), the runner (`run_agent_batch.sh`, `write_task_results.py`, `retry_policy.py`) |
| `ci_checks/` | the repository checks (invariants, advisories, unit tests) |
| `images/roboarc/` | the image: `build.sh`, `push.sh`, `stage_instance.py`, `VERSION`, and the recipe of the Genesis base image |
| `images/monitor/` | the recipe of the monitor sidecar image |
| `roboarc/` | reserved for the Python package (see "Use the simulator directly") |
| `source/` | the simulator and release source: `topogym/` (physics, tasks and the planners that made the references), `assets/robots/`, `images/robotangle/Dockerfile.mujoco` (the image recipe, with the MuJoCo EPA patch), `scripts/robotangle/` (release tooling, references, stress tests) |
| `docs/` | the protocol, running guide, benchmark page and family pages |
| `ikea/` | IKEA MuJoCo: `tasks/` (24), `scripts/ikea_mujoco/` (generators, references, evidence, review), `images/ikea-mujoco/` (scenes and the two image recipes), `ci/check.sh`, `docs/benchmarks/ikea_mujoco.md`, and the shared task layer it renders with (`scripts/task_conventions.py`, `eai_standard/`, which are the dev branch's versions) |

The code keeps the simulator's historical names: the package `topogym`, the tooling `rcb_topogym.py` and the paths
`source/scripts/robotangle/`, `/opt/robotangle/`. RoboARC was developed as TopoGym, then RoboTangle.
[PROVENANCE.md](PROVENANCE.md) says where every part comes from.

## Checks

```bash
python3 scripts/roboarc/mk_tasks.py --check           # every task dir equals a fresh render (0 differences)
bash ci_checks/run_invariants.sh                      # the protocol's invariants (PROTOCOL Appendix H)
bash ci_checks/check-unit-tests.sh                    # unit tests (numpy + requests for the service tests)
python3 ci_checks/check-video.py tasks/roboarc-*      # every task collects its video
python3 ci_checks/check-resources.py tasks/roboarc-*  # declared resources and caps
bash ci_checks/run_all.sh                             # all of the above plus the advisory checks
bash ikea/ci/check.sh [--image-tests]                 # IKEA: both generators' check modes (+ unit tests in its image)
```

`.github/workflows/checks.yml` runs these checks on every push and pull request. They need no GPU and no image. Strict
replays and Harbor trials need the image and a GPU.

## Assets

The frozen scene instances (about 700 MB) are not in git. They are in the private Hugging Face dataset
`eai-v2-internal/robotangle-assets`, pinned by commit and per-file sha256 in `scripts/roboarc/assets.json`. The image
carries them, so running tasks needs no dataset access. Only building the image does. The IKEA scenes' meshes are in
the private dataset `eai-v2-internal/ikea-mujoco-assets`, pinned in `ikea/scripts/ikea_mujoco/assets.json`.

## IKEA's task layer

IKEA's tasks follow the development branch's task layer: no monitor sidecar and no DIGEST. So the RoboARC invariants
above do not cover them; `ikea/ci/check.sh` and the Harbor checks do.

## Citation and licence

See [CITATION.cff](CITATION.cff) (a placeholder until the paper). The shared task layer derives from the MIT-licensed
robot_coding_bench ([LICENSES/robot_coding_bench-MIT.txt](LICENSES/robot_coding_bench-MIT.txt)). The licence of
RoboARC itself is to be decided.
