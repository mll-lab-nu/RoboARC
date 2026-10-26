# RoboARC

RoboARC is a benchmark of long-horizon physical manipulation, simulated on MuJoCo 3.14.0 and rendered with Nyx. Its
tasks are topological puzzles (wire and ring puzzles, interlocked cast puzzles, knots, ropes, cords, cloth, bolts,
gears, wires) and IKEA furniture assembly. Each task is one frozen scene, a robot (two ALOHA 2 arms, a Franka Panda
with a Robotiq 2F-85, or two Franka Pandas) and a grader that decides from the simulated state whether the task was
done.

Website: <https://roboarc.embodied-first-exam.ai> · Version **2.1.0** ([CHANGELOG.md](CHANGELOG.md))

## Interfaces

The same tasks, physics and graders are offered through separate interfaces, one per kind of solver. A result is
always reported with the interface (and its settings) it was obtained under: results of different interfaces are not
compared or pooled.

| interface | for | the solver | status |
|---|---|---|---|
| [Coding agents (Harbor)](#coding-agents-harbor) | LLM coding agents (Codex, Claude Code, ...) | writes and runs code that drives the robot, in a container, within a wall-clock budget; a separate verifier grades the result | released |
| [VLA policies](#vla-policies) | vision-language-action models | a policy server receives camera images, robot state and a language instruction and returns action chunks (openpi conventions) | preview |
| [Python API](#python-api) | RL, imitation learning, scripted control, your own loops | `roboarc.make(task_id)`: a Gymnasium-compatible environment | released |

An agent that calls a VLA policy as a tool (Harbor + VLA) is planned for later.

## Tasks

78 task bases in 19 families of topological puzzles, and 12 IKEA furniture scenes. Every puzzle base has three
appearance seeds (0, 1, 2): the same scene and physics in other colours and materials.

| family | bases | robot | difficulty |
|---|---:|---|---|
| battery-replacement | 1 | ALOHA 2 | medium 1 |
| bed-bolt | 5 | ALOHA 2 | 1 per tier |
| blanket-flatten | 5 | ALOHA 2 | easier 1, easy 2, medium 1, hard 1 |
| blanket-fold | 5 | ALOHA 2 | very_easy .. hard, 1 each |
| buzz-wire-3d | 5 | Franka + Robotiq 2F-85 | very_easy 1, easier 2, medium 1, hard 1 |
| caliper-sorting | 1 | ALOHA 2 | hard 1 |
| cast-coil | 5 | ALOHA 2 | 1 per tier |
| cast-keyhole | 5 | ALOHA 2 | easier 1, easy 1, medium 1, hard 2 |
| clock-assembling | 1 | ALOHA 2 | hard 1 |
| fiber-bundling | 5 | Franka + Robotiq 2F-85 | easier 1, easy 1, medium 2, hard 1 |
| hedgehog-in-cage | 5 | Franka + Robotiq 2F-85 | easier 2, easy 2, hard 1 |
| horseshoe-ring | 5 | ALOHA 2 | 1 per tier |
| interlocked-claws | 8 | ALOHA 2 | easier 1, easy 1, medium 2, hard 4 |
| knot-untying | 6 | ALOHA 2 | very_easy 1, easier 2, medium 2, hard 1 |
| lock-picking | 3 | Franka + Robotiq 2F-85 | easy 1, medium 1, hard 1 |
| orbits | 2 | ALOHA 2 | very_easy 1, easier 1 |
| rope-threading | 5 | ALOHA 2 | 1 per tier |
| slotted-post-ring | 5 | ALOHA 2 | 1 per tier |
| wire-splicing | 1 | ALOHA 2 | hard 1 |

A base is named `roboarc-<family>-<NN>-i01`; `<NN>` numbers the bases of a family, and the difficulty is metadata
(`[metadata].difficulty`), never part of the name. One page per family, with its success rule in plain words:
[docs/families/](docs/families/README.md). The rules shared by every family (grading, safety readings, contact limits):
[docs/benchmarks/roboarc.md](docs/benchmarks/roboarc.md).

**IKEA MuJoCo** (`roboarc-ikea-<scene>-i00`): lack, svallet, gladom, variera, eket, ikea365, vesken, stegesund,
stenerik, burvik, uppsta and kyrre, each an IKEA product assembled by two Franka Panda arms (svallet: two ALOHA 2 arms)
with physical grasps. The scenes are by Xiaoyu's team (Xiaoyu, Zhen Wu); see
[docs/benchmarks/ikea_mujoco.md](docs/benchmarks/ikea_mujoco.md).

## Requirements

Every interface runs the simulator in the released images:
- Linux x86_64 with an NVIDIA GPU that has RT cores (Nyx renders through Vulkan ray queries; driver >= 580): GeForce
  RTX 20-series or newer, RTX workstation GPUs, T4, L4, L40 / L40S, A10 / A10G, A40. Data-centre GPUs without RT cores
  (V100, A100, A30, H100, H200) run the physics but render no images, and there is no software fallback
  ([docs/RUNNING.md](docs/RUNNING.md#gpus-without-rt-cores)). arm64 hosts are not supported.
- Docker with the NVIDIA Container Toolkit.
- The images: `ghcr.io/mll-lab-nu/roboarc:2.1.0` (the puzzles), `roboarc-ikea:1.1.0` and `roboarc-ikea-sim:1.1.0`
  (IKEA), each pinned by manifest digest ([images/pins.json](images/pins.json), [images/README.md](images/README.md)).
  The packages are private and inherit this repository's access: log in with `docker login ghcr.io -u <github user>`
  and a token with the `read:packages` scope, then pull them all with `bash images/pull_all.sh` (each is checked
  against its pinned image id). Or build them ([docs/DEVELOPING.md](docs/DEVELOPING.md)).

## Coding agents (Harbor)

Each task base ships as two [Harbor](https://github.com/laude-institute/harbor) task directories under `tasks/`,
`roboarc-<base>-privileged` and `roboarc-<base>-standard` (180 in all: 156 puzzle tasks and 24 IKEA tasks):
- **privileged**: the simulator is in the agent's container. The agent may reset, read the full state and replay as
  often as it likes, and delivers one trajectory (`/app/output/trajectory.npz`), which the verifier replays twice in a
  fresh process.
- **standard** (protocol eai-standard/2.4): the simulator runs in a sidecar the agent cannot open. The agent gets RGB
  (optional depth), proprioception and calibrated cameras, and submits joint or end-effector chunks, with no reset and
  no privileged state. The service grades the live episode, and for the puzzles a fresh-process replay confirms it.

The agent's budget is wall clock, 3 h per task (10800 s); wire-splicing-01 has 6 h by the owner's exception, and its
trials form their own group. Harbor 0.23 or newer and Python >= 3.11 for the repository tools.

```bash
python3 scripts/stamp_mtimes.py tasks   # once per checkout: fixed file times, so Harbor's build cache hits
harbor run -p tasks -i roboarc-interlocked-claws-08-i01-privileged -a oracle   # the reference solution: reward 1
harbor run -p tasks -i roboarc-interlocked-claws-08-i01-standard -a nop        # does nothing: reward 0
harbor run -p tasks -i roboarc-orbits-02-i01-standard -a codex -m openai/<model>   # a coding agent
```

- Running batches, agent settings, network egress, seeds, result files and re-grading: [docs/RUNNING.md](docs/RUNNING.md).
- The task contract and scoring: [docs/PROTOCOL.md](docs/PROTOCOL.md), [docs/STANDARD_MODE.md](docs/STANDARD_MODE.md).
- A collaborator's setup, a full evaluation run and sending the results back: [docs/COLLABORATOR_SETUP.md](docs/COLLABORATOR_SETUP.md).

## VLA policies

Preview. A VLA model served as an openpi policy server (Physical Intelligence's
[openpi](https://github.com/Physical-Intelligence/openpi), `openpi_client`'s websocket protocol) gets the task's camera
images, the robot's state and a one-sentence instruction, returns action chunks, and each episode is graded by the
task's own grader. The ALOHA 2 and IKEA tasks use openpi's ALOHA layout, the single-arm Franka tasks its DROID layout.

```bash
python -m roboarc.vla.run --task roboarc-interlocked-claws-08-i01 --policy ws://POLICY_HOST:8000 \
  --seed 0,1,2 --episodes 1 --run-id <policy name> --out <results dir>      # inside the image, with one GPU
```

Setup, observation and action layouts per robot, prompts, episodes and grading, results and GPU needs:
[docs/VLA.md](docs/VLA.md).

## Python API

The package `roboarc` exposes every task as a Gymnasium-compatible environment, without Harbor. It runs the graded
tooling itself, so physics, limits and success are the graded tasks' own: `reset()` loads the frozen instance, `step()`
is one control step, and a reference trajectory stepped through it ends on exactly the state hash the release recorded.

```python
import roboarc                                        # before numpy: it sets the graded numpy settings
env = roboarc.make("roboarc-cast-keyhole-02-i01")     # renderer="nyx" (GPU) | "mujoco" (quick look) | "none"
obs, info = env.reset()                               # obs: {"rgb": {camera: (480, 848, 3) uint8}, "proprio": (P,)}
obs, reward, terminated, truncated, info = env.step(action)   # absolute joint targets per arm + gripper
```

Run it inside the task's image with the repository mounted:

```bash
docker run --rm --gpus 1 -e NVIDIA_DRIVER_CAPABILITIES=all -v $PWD:/repo:ro -w /repo -e PYTHONPATH=/repo:/opt/topogym \
  -e HOME=/tmp -u $(id -u):$(id -g) ghcr.io/mll-lab-nu/roboarc:2.1.0 python examples/vla_loop.py roboarc-cast-keyhole-02-i01
docker run --rm -v $PWD:/repo:ro -w /repo -e PYTHONPATH=/repo:/opt/topogym -e HOME=/tmp -u $(id -u):$(id -g) \
  ghcr.io/mll-lab-nu/roboarc:2.1.0 python -m roboarc replay roboarc-interlocked-claws-08-i01     # bit-identity check
```

IKEA runs in `roboarc-ikea:1.1.0` (physics only) or `roboarc-ikea-sim:1.1.0` (with images). `pip install -e .` installs
the package (the extras `.[puzzles]` and `.[ikea]` pin the images' versions). Actions, observations, frames, the speed
rules (which clamp and never fail), rendering and GPU needs, wrappers and tests: [docs/API.md](docs/API.md); examples
in [examples/](examples/).

## Repository layout

| path | what |
|---|---|
| `tasks/roboarc-*` | the Harbor task directories: 156 puzzle tasks and 24 IKEA tasks (`roboarc-ikea-*`); generated, never edit them by hand |
| `roboarc/`, `pyproject.toml` | the Python package `roboarc` ([docs/API.md](docs/API.md)), with `roboarc.vla` for VLA policies ([docs/VLA.md](docs/VLA.md)); `examples/`, `tests/api/` its examples and tests |
| `ikea/` | IKEA MuJoCo: the generators, references and evidence (`ikea/scripts/ikea_mujoco/`), the scenes and the two image recipes (`ikea/images/ikea-mujoco/`); see [ikea/README.md](ikea/README.md) |
| `scripts/roboarc/` | the puzzle generator `mk_tasks.py` and its inputs: `release.json`, `packages/` (the per-task texts), `oracles/` (reference trajectories), `shared/` (the pinned tooling and graders), `assets.json` (instance pins); `regrade.py`, `harbor_check.py` |
| `scripts/` | the task layer the Harbor tasks cite (`task_conventions.py`, `set_agent_egress.py`, `task_resources.toml`, `eai_standard/`, `rcb_monitor/`), build helpers (`stamp_mtimes.py`, `check_image_files.py`) and `send_results.sh` / `receive_results.sh` |
| `scripts/harbor/` | the Harbor batch runner and its helpers ([docs/RUNNING.md](docs/RUNNING.md)) |
| `source/` | the image build context: the simulator package `topogym/` (with `planning/`, the planners that made the references), `assets/robots/`, the recipe `source/images/roboarc/Dockerfile` with its pinned inputs and the MuJoCo EPA patch |
| `images/` | the image pins (`pins.json`), `pull_all.sh`, `archive.sh` and `push_all.sh` ([images/README.md](images/README.md)); `roboarc/`: the release image's `build.sh` and its README; `monitor/`: the monitor sidecar's recipe |
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

The frozen scene instances (about 700 MB) and the IKEA scenes' meshes are not in git. The images carry them, so every
interface works inside the images with nothing else. Building the images reads them from Hugging Face datasets pinned
by commit and per-file sha256 (`scripts/roboarc/assets.json`, `ikea/scripts/ikea_mujoco/assets.json`).

## Citation and licence

See [CITATION.cff](CITATION.cff) (a placeholder until the paper). RoboARC is released under the MIT licence
([LICENSE](LICENSE)). Third-party material (robot models, adapted kernels, textures, lighting) keeps its own licence;
see [LICENSES/THIRD_PARTY.md](LICENSES/THIRD_PARTY.md), which also lists the items whose licence is still to be
confirmed. The shared task layer derives from the MIT-licensed robot_coding_bench
([LICENSES/robot_coding_bench-MIT.txt](LICENSES/robot_coding_bench-MIT.txt)).
