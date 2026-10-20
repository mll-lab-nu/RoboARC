# RoboARC

RoboARC is a benchmark of physical manipulation simulated on MuJoCo 3.14.0: topological puzzles (wire and ring puzzles,
interlocked cast puzzles, knots, ropes, cords and cloth) and IKEA furniture assembly. Each task is one frozen scene
instance and a robot (two ALOHA 2 arms, a Franka Panda with a Robotiq 2F-85, or two Franka Pandas) that has to solve it.

The tasks serve two kinds of solvers through the same simulator code:
- **coding agents**, through Harbor tasks: the agent writes the code that drives the robot, and a separate verifier
  grades the result ([For coding agents](#for-coding-agents-harbor));
- **policies** (VLA, reinforcement learning, imitation, scripted control), through the Python package `roboarc`:
  `roboarc.make(task_id)` returns a Gymnasium-compatible environment ([For policies](#for-policies-python)).

Website: <https://roboarc.embodied-first-exam.ai>

Version **1.1.0** (see [CHANGELOG.md](CHANGELOG.md); the puzzle image is `1.1.0`, the IKEA images are still `1.0.0`), 226 Harbor task directories
under `tasks/`:
- the topological puzzles: 101 task bases in 14 families, each in two modes, so 202 task directories;
- **IKEA MuJoCo**: 12 furniture-assembly tasks with two robot arms and physical grasps, each in two modes, so 24 task
  directories. The scenes are by Xiaoyu's team (Xiaoyu, Zhen Wu).

## Tasks

| family | bases | robot | difficulty tiers |
|---|---:|---|---|
| blanket-flatten | 9 | ALOHA 2 | very_easy 2, easier 1, easy 2, medium 2, hard 2 |
| blanket-fold | 5 | ALOHA 2 | very_easy .. hard, 1 each |
| buzz-wire-3d | 10 | Franka + Robotiq 2F-85 | 2 per tier |
| cast-coil | 8 | ALOHA 2 | very_easy 2, easier 2, easy 2, medium 1, hard 1 |
| cast-keyhole | 6 | ALOHA 2 | very_easy 1, easier 1, easy 1, medium 1, hard 2 |
| fiber-bundling | 10 | Franka + Robotiq 2F-85 | 2 per tier |
| hedgehog-in-cage | 8 | Franka + Robotiq 2F-85 | very_easy 1, easier 2, easy 2, medium 1, hard 2 |
| horseshoe-ring | 5 | ALOHA 2 | 1 per tier |
| interlocked-claws | 12 | ALOHA 2 | very_easy 2, easier 2, easy 2, medium 3, hard 3 |
| knot-untying | 8 | ALOHA 2 | very_easy 2, easier 2, easy 1, medium 2, hard 1 |
| lock-picking | 5 | Franka + Robotiq 2F-85 | 1 per tier |
| orbits | 2 | ALOHA 2 | very_easy 1, easier 1 |
| rope-threading | 8 | ALOHA 2 | very_easy 1, easier 1, easy 2, medium 2, hard 2 |
| slotted-post-ring | 5 | ALOHA 2 | 1 per tier |

Puzzle task directories are `tasks/roboarc-<family>-<NN>-i01-<mode>`; `<NN>` numbers the bases of a family and the
difficulty is `[metadata].difficulty` in `task.toml`. One page per family: [docs/families/](docs/families/). The
benchmark's rules (grading replays, safety readings, the standard service): [docs/benchmarks/roboarc.md](docs/benchmarks/roboarc.md).

IKEA MuJoCo (`tasks/roboarc-ikea-<scene>-i00-<mode>`): lack, svallet, gladom, variera, eket, ikea365, vesken,
stegesund, stenerik, burvik, uppsta and kyrre, each an IKEA product assembled by two Franka Panda arms (svallet: two
ALOHA 2 arms). Its page: [docs/benchmarks/ikea_mujoco.md](docs/benchmarks/ikea_mujoco.md).

## For coding agents (Harbor)

Each task exists in two modes, on the same frozen instance, with the same 3 h agent budget (`[agent] timeout_sec` =
10800 s):
- **privileged** (`-privileged`): the agent's container holds the simulator. The agent may reset, read the full state
  and replay as often as it likes, and delivers `/app/output/trajectory.npz`. The verifier replays it in a fresh
  process.
- **standard** (`-standard`, protocol eai-standard/2.4): the agent's container has only Python and a client; the
  simulator runs in a `sim` sidecar the agent cannot open. The agent gets RGB (optional depth), proprioception and
  calibrated cameras, and submits joint or end-effector chunks; no reset, no privileged state. The service grades the
  live episode; for the puzzles a fresh-process replay must confirm it.

Requirements:
- Linux x86_64 and an NVIDIA GPU with RT cores (driver >= 580; Nyx renders through Vulkan). Each puzzle task declares
  one GPU with 2 GB; IKEA's standard tasks reserve one GPU for rendering and its privileged tasks run on CPU only. See each task's resources in its `task.toml`.
- Docker with the NVIDIA Container Toolkit; [Harbor](https://github.com/laude-institute/harbor) 0.23 or newer;
  Python >= 3.11 for the repository tools.

The puzzle tasks run `ghcr.io/embodied-first-exam/roboarc:1.1.0`; the IKEA tasks run
`ghcr.io/embodied-first-exam/roboarc-ikea:1.0.0`, and their standard `sim` sidecar `roboarc-ikea-sim:1.0.0`. Pull them,
or build them ([docs/DEVELOPING.md](docs/DEVELOPING.md)).

```bash
python3 scripts/stamp_mtimes.py tasks   # once per checkout: fixed file times, so Harbor's build cache hits
harbor run -p tasks -i roboarc-interlocked-claws-03-i01-privileged -a oracle   # the reference solution: reward 1
harbor run -p tasks -i roboarc-interlocked-claws-03-i01-standard -a nop        # does nothing: reward 0
harbor run -p tasks -i roboarc-orbits-02-i01-standard -a codex -m openai/<model>   # a coding agent
harbor run -p tasks -i roboarc-ikea-lack-i00-privileged -a oracle             # an IKEA task
```

Batch runs, agent configuration, network egress, result files and re-grading: [docs/RUNNING.md](docs/RUNNING.md). The
task contract and scoring: [docs/PROTOCOL.md](docs/PROTOCOL.md) and [docs/STANDARD_MODE.md](docs/STANDARD_MODE.md).

## For policies (Python)

The package `roboarc` exposes every task (the 101 puzzle bases and the 12 IKEA scenes) as a Gymnasium-compatible
environment, without Harbor. It runs the graded tooling itself (`rcb_topogym.Session` for the puzzles, `rcb_ikea.Session`
for IKEA), so physics, limits and success are the graded tasks' own: `reset()` loads the task's frozen instance,
`step()` is one control step, and a reference trajectory stepped through it ends on exactly the state hash the release
recorded. Harbor stays the evaluation path for agents.

```python
import roboarc                                        # before numpy: it sets the graded numpy settings
env = roboarc.make("roboarc-cast-keyhole-01-i01")     # renderer="nyx" (GPU) | "mujoco" (quick look) | "none"
obs, info = env.reset()                               # obs: {"rgb": {camera: (480, 848, 3) uint8}, "proprio": (P,)}
obs, reward, terminated, truncated, info = env.step(action)   # absolute joint targets per arm + gripper
```

Run it inside the task's image with the repository mounted:

```bash
docker run --rm --gpus 1 -e NVIDIA_DRIVER_CAPABILITIES=all -v $PWD:/repo:ro -w /repo -e PYTHONPATH=/repo:/opt/topogym \
  -e HOME=/tmp -u $(id -u):$(id -g) ghcr.io/embodied-first-exam/roboarc:1.1.0 python examples/vla_loop.py roboarc-cast-keyhole-01-i01
docker run --rm -v $PWD:/repo:ro -w /repo -e PYTHONPATH=/repo:/opt/topogym -e HOME=/tmp -u $(id -u):$(id -g) \
  ghcr.io/embodied-first-exam/roboarc:1.1.0 python -m roboarc replay roboarc-interlocked-claws-03-i01     # bit-identity check
```

IKEA runs in `roboarc-ikea:1.0.0` (physics only) or `roboarc-ikea-sim:1.0.0` (with Nyx images). `pip install -e .`
installs the package (the extras `.[puzzles]` and `.[ikea]` pin the images' versions); the frozen instances and scenes
come with the images. The interface and its conventions (actions, observations, frames, the speed rules, which clamp and
never fail), rendering and GPU needs, the wrappers and the tests: [docs/API.md](docs/API.md); examples in
[examples/](examples/).

## Repository layout

| path | what |
|---|---|
| `tasks/roboarc-*` | the 226 task directories: 202 puzzle tasks and 24 IKEA tasks (`roboarc-ikea-*`); generated, never edit them by hand |
| `ikea/` | IKEA MuJoCo: the generators, references and evidence (`ikea/scripts/ikea_mujoco/`), the scenes and the two image recipes (`ikea/images/ikea-mujoco/`); see [ikea/README.md](ikea/README.md) |
| `roboarc/`, `pyproject.toml` | the Python package `roboarc` ([docs/API.md](docs/API.md)); `examples/`, `tests/api/` its examples and tests |
| `scripts/roboarc/` | the puzzle generator `mk_tasks.py` and its inputs: `release.json`, `packages/` (the per-task texts), `oracles/` (reference trajectories), `shared/` (the pinned tooling and graders), `assets.json` (instance pins); `regrade.py`, `harbor_check.py` |
| `scripts/` | the task layer the tasks cite (`task_conventions.py`, `set_agent_egress.py`, `task_resources.toml`, `eai_standard/`, `rcb_monitor/`) and build helpers (`stamp_mtimes.py`, `check_image_files.py`) |
| `scripts/harbor/` | the Harbor batch runner and its helpers ([docs/RUNNING.md](docs/RUNNING.md)) |
| `source/` | the image build context: the simulator package `topogym/` (with `planning/`, the planners that made the references), `assets/robots/`, the recipe `source/images/roboarc/Dockerfile` with its pinned inputs and the MuJoCo EPA patch |
| `images/roboarc/`, `images/monitor/` | the release image's `build.sh` / `push.sh` and its README (what it holds, how it is pinned); the monitor sidecar's recipe |
| `tools/` | development tools: the release rebuild, reference recording, redesign and stress tests ([tools/README.md](tools/README.md)) |
| `records/` | freeze manifests and archived release records |
| `ci_checks/`, `.github/` | the repository checks and the CI workflow |
| `docs/` | protocol, running guide, developer guide, API, benchmark and family pages; `docs/simulator/` the simulator's internals; `docs/rcb/` a map of older paths that task texts cite |

The simulator keeps its historical names: the package `topogym`, the tooling `rcb_topogym.py` and the image path
`/opt/robotangle/`. RoboARC was developed as TopoGym, then RoboTangle, on the Genesis simulator; its tasks moved to
MuJoCo before 1.0.0, and since 1.1.0 the image and the simulator source contain no Genesis
([docs/history/GENESIS_REMOVAL.md](docs/history/GENESIS_REMOVAL.md)). Renaming them would change the pinned bytes every
task is bound to. [PROVENANCE.md](PROVENANCE.md) says where every part comes from.

## Checks

```bash
python3 scripts/roboarc/mk_tasks.py --check           # every puzzle task dir equals a fresh render
bash ikea/ci/check.sh                                 # the same for IKEA's two generators
bash ci_checks/run_invariants.sh                      # the protocol's invariants (PROTOCOL Appendix H)
bash ci_checks/check-unit-tests.sh --all              # unit tests (numpy, scipy, requests)
bash ci_checks/run_all.sh                             # all of the above plus the advisory checks
python3 -m pytest tests/api -m "not sim and not heavy"   # the Python package's fast tests
```

`.github/workflows/checks.yml` runs them on pushes to `main` and `dev/**` and on pull requests; they need no GPU and no
image. Strict replays, the package's simulator tests and Harbor trials need the images and a GPU
([docs/DEVELOPING.md](docs/DEVELOPING.md)).

## Assets

The frozen scene instances (about 700 MB) and the IKEA scenes' meshes are not in git. The images carry them, so running
tasks and the Python package inside the images needs nothing else. Building the images reads them from Hugging Face
datasets pinned by commit and per-file sha256 (`scripts/roboarc/assets.json`, `ikea/scripts/ikea_mujoco/assets.json`).

## Citation and licence

See [CITATION.cff](CITATION.cff) (a placeholder until the paper). RoboARC is released under the MIT licence
([LICENSE](LICENSE)). Third-party material (robot models, adapted kernels, textures, lighting) keeps its own licence;
see [LICENSES/THIRD_PARTY.md](LICENSES/THIRD_PARTY.md), which also lists the items whose licence is still to be
confirmed. The shared task layer derives from the MIT-licensed robot_coding_bench
([LICENSES/robot_coding_bench-MIT.txt](LICENSES/robot_coding_bench-MIT.txt)).
