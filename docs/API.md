# RoboARC raw simulator interface (`roboarc`)

The Python package `roboarc` exposes the benchmark's simulators directly, so VLA policies, reinforcement learning and
scripted controllers can run on the same tasks without Harbor. Harbor stays the evaluation path: the task directories,
their graders and their images are unchanged, and nothing here changes how a task is scored.

```python
import roboarc                                   # import it before numpy (see "Fidelity")
env = roboarc.make("roboarc-orbits-02-i01")      # Gymnasium-compatible
obs, info = env.reset()                          # the task's frozen instance: info["birth_hash"]
obs, reward, terminated, truncated, info = env.step(action)
```

Contents: [Design](#design) · [Running](#running) · [Tasks](#tasks) · [make()](#make) · [The environment](#the-environment) ·
[Actions](#actions) · [Observations](#observations) · [Episodes](#episodes-success-and-time-limits) ·
[Limits](#limits-clamp-never-fail) · [Fidelity](#fidelity-no-physics-fork) · [Rendering](#rendering) ·
[Wrappers](#wrappers) · [Examples](#examples) · [Tests](#tests) · [Limitations](#limitations)

## Design

- **One interface for both suites.** `roboarc.make(task_id)` returns a `RoboARCEnv` for any of the 101 puzzle bases and
  the 12 IKEA MuJoCo tasks. It subclasses `gymnasium.Env` when Gymnasium is installed; without it (the published images
  do not ship Gymnasium) it has the same methods and spaces with the same attributes.
- **No physics fork.** The environment holds the graded tooling's own session and calls it:
  - the puzzles: `rcb_topogym.Session` from `scripts/roboarc/shared/rcb_topogym.py`, the file every task directory
    copies (its sha256 is pinned in each task's `runtime_binding.json`), over the image's `topogym` package, built from
    the task's frozen instance (`/opt/robotangle/stage_instance.py`, as the task Dockerfiles stage it);
  - IKEA: `rcb_ikea.Session` from `ikea/scripts/ikea_mujoco/shared/rcb_ikea.py` (the file every IKEA task copies), on
    the image's scenes, checked against the task's runtime binding as the verifier checks them.

  `step(action)` is one call of that session's `step`. The substeps, timesteps, solver settings, limits, guards,
  success predicates and failure predicates are the session's; this package adds no physics and no physics settings.
- **The graded birth.** `reset()` is the session's own reset: the frozen instance's state, whose hash
  (`info["birth_hash"]`) is the one the graded tasks start from. `seed` is accepted and has no effect: every task is one
  fixed instance.
- **Bit identity.** Stepping a task's reference trajectory through the API from `reset()` reaches exactly the state hash
  the release recorded for it ([Tests](#tests)).
- **Thin.** No skills (grasp, place, navigate). The only optional wrapper maps end-effector poses to joint targets with
  the tasks' own IK ([Wrappers](#wrappers)).

## Running

The supported runtime is the published images: they hold the simulator, the frozen instances, the EPA-patched
libmujoco and the exact dependency versions the runtime bindings require.

| suite | image (physics) | images with Nyx RGB |
|---|---|---|
| puzzles | `ghcr.io/embodied-first-exam/roboarc:1.0.0` (Python 3.11) | the same image |
| IKEA | `ghcr.io/embodied-first-exam/roboarc-ikea:1.0.0` (Python 3.12) | `ghcr.io/embodied-first-exam/roboarc-ikea-sim:1.0.0` |

Mount the repository (read-only is fine) and put it on `PYTHONPATH` (or `pip install --no-deps -e /repo` in a derived
image). Run from the repository root of the checkout:

```bash
# a puzzle task, physics only (CPU; rendering off): its reference replayed and checked against the recorded end
# hash (about 30 s; orbits-02's 7918 rows take about 30 min on one core)
docker run --rm -v $PWD:/repo:ro -w /repo -e PYTHONPATH=/repo:/opt/topogym -e HOME=/tmp -u $(id -u):$(id -g) \
  ghcr.io/embodied-first-exam/roboarc:1.0.0 python -m roboarc replay roboarc-interlocked-claws-03-i01

# with Nyx images (one GPU, Vulkan through the NVIDIA container toolkit)
docker run --rm --gpus 1 -e NVIDIA_DRIVER_CAPABILITIES=all -v $PWD:/repo:ro -w /repo -e PYTHONPATH=/repo:/opt/topogym \
  -e HOME=/tmp -u $(id -u):$(id -g) ghcr.io/embodied-first-exam/roboarc:1.0.0 python examples/vla_loop.py roboarc-cast-keyhole-01-i01

# an IKEA task with Nyx images
docker run --rm --gpus 1 -e NVIDIA_DRIVER_CAPABILITIES=all -v $PWD:/repo:ro -w /repo -e PYTHONPATH=/repo -e HOME=/tmp \
  -u $(id -u):$(id -g) ghcr.io/embodied-first-exam/roboarc-ikea-sim:1.0.0 python examples/vla_loop.py lack
```

Scratch files (staged instances, the Nyx scene export) go to `ROBOARC_CACHE` (default `~/.cache/roboarc`, here under
`HOME=/tmp`), never to the working directory. When the container runs as a non-root user (`-u`), Nyx prints
`Failed to export shader library` at exit: it cannot cache its compiled shaders inside the image's site-packages, so it
compiles them at every start (a few seconds); the frames are not affected.

- `python -m roboarc check` reports whether the process runs the graded runtime; `python -m roboarc list` lists the
  tasks; `python -m roboarc info <task> --sim` prints a task's spaces, layouts and cameras.
- `pip install -e .` from the repository installs the package itself (its only dependency is numpy). The extras pin the
  images' versions: `.[puzzles]` (numpy 2.3.5, scipy 1.17.1, mujoco 3.14.0, warp-lang 1.18.0, newton 1.6.1, pillow 11.3.0),
  `.[ikea]` (numpy 2.5.4, mujoco 3.14.0), `.[gym]` (Gymnasium), `.[test]` (pytest). Outside the images two more things
  are needed: the EPA patch of the mujoco wheel (`python -m roboarc patch-mujoco`, the same 14-byte patch the images
  apply; `source/images/robotangle/mujoco-3.14.0-epa/`) and, for the puzzles, the frozen instances
  (`ROBOARC_INSTANCES=<dir>` with `<dir>/<base>/instance.json` + `instance_assets/`) and the image's own runtime (its
  libuipc build among them: the runtime binding checks it). `make(..., strict=False)` runs without these checks; the
  results are then not the graded ones.
- Hardware: the MuJoCo physics runs on the CPU (one core per environment). A GPU is needed for Nyx rendering (RT cores;
  NVIDIA driver 570 worked) and, for **blanket-flatten and blanket-fold, for the physics too**: their cloth runs on a
  deterministic CUDA solver (Warp, built for CUDA 13: a driver for CUDA 13, or NVIDIA's CUDA forward-compatibility
  libraries `cuda-compat` on `LD_LIBRARY_PATH` with an older driver; the bit-identity runs below used driver 570 with
  them). Memory per environment (container peak, measured on full reference replays with rendering off): puzzles
  0.6-2 GB, blanket 3.2 GB (its CUDA cloth), IKEA 0.3-0.5 GB; with Nyx on, about 1 GB for a puzzle and 1.1-1.3 GB for
  IKEA (whose Nyx runs in its own process), plus the renderer's GPU memory.
- Environment variables: `ROBOARC_REPO` (the checkout, default: the package's parent), `ROBOARC_CACHE` (staged
  instances, default `~/.cache/roboarc`), `ROBOARC_INSTANCES` (frozen instances outside the image), `IKEA_SCENES`
  (default `/opt/ikea/scenes`), `MUJOCO_GL` (`egl` on a headless GPU, `osmesa` on a CPU-only host).

## Tasks

```python
roboarc.list_tasks()                  # 113 TaskInfo: 101 puzzle bases, 12 IKEA tasks
roboarc.list_tasks("puzzles", family="orbits")
roboarc.task_info("orbits-02")        # aliases: see below
```

Canonical ids: `roboarc-<family>-<NN>-i01` (e.g. `roboarc-orbits-02-i01`) and `roboarc-ikea-<name>-i00` (e.g.
`roboarc-ikea-lack-i00`). Also accepted: the Harbor task names (`roboarc-orbits-02-i01-standard`,
`robot-coding-bench/roboarc-orbits-02-i01-privileged`), the bases without the prefix (`orbits-02-i01`, `orbits-02`) and
the IKEA names (`ikea-lack`, `lack`). Both modes of a task share one environment: the frozen instance is the same.

`TaskInfo` (no simulator needed): `id`, `suite` (`puzzles` | `ikea`), `family`, `name` (the tooling's task key),
`base`, `difficulty`, `robot`, `description`, `image`, `render_image`, `max_episode_steps` (the verifier's row cap),
`agent_time_limit_s` (the Harbor agent budget, 10800 s), `task_dirs`, `reference` (the reference trajectory, if any),
`reference_end_hash` and, for IKEA, `reference_settled_hash`. 99 of the 101 puzzle bases ship a reference
(buzz-wire-3d-05 and fiber-bundling-05 do not); every IKEA task does.

## make()

```python
roboarc.make(task_id, *, renderer="nyx", cameras=None, depth=False, image_size=None, render_mode=None,
             rules="standard", max_episode_steps=None, strict=True, **kwargs)
```

| option | meaning |
|---|---|
| `renderer` | `"nyx"`: the graded renderer (GPU); `"mujoco"`: MuJoCo's raster of the physics model, a quick look, **not the graded images**; `"none"`: no images (proprioception only; the fastest) |
| `cameras` | the cameras in the observation (default: all of the task's, `env.all_camera_names`; e.g. `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist` on the two-arm robots) |
| `depth` | add z-depth images (metres along the optical axis, 0 = no reading) |
| `image_size` | `(W, H)`: resize every image (bilinear RGB, nearest depth); default the cameras' own 848 x 480 |
| `render_mode` | `"rgb_array"`: `env.render()` returns all cameras tiled |
| `rules` | puzzles: `"standard"` (default) adds the standard mode's joint-speed rule to the tooling's limits; `"privileged"` applies the tooling's limits only (what privileged agents and their verifier's replays get). IKEA: one rule set |
| `max_episode_steps` | truncation (default: the task's verifier row cap) |
| `strict` | `True` (default): refuse to run outside the graded runtime (`roboarc.RuntimeMismatch`); `False`: warn and run |
| `check_every` | IKEA only: control steps between success checks (default 40 = 0.1 s) |

One environment per process: both simulators are one scene per process (the tooling's rule). For several
environments use subprocess vector environments (`gymnasium.vector.AsyncVectorEnv`).

## The environment

| member | what |
|---|---|
| `reset(seed=None, options=None)` | the frozen instance -> `(obs, info)`; `info["birth_hash"]` |
| `step(action)` | one control step -> `(obs, reward, terminated, truncated, info)` |
| `render()` | `render_mode="rgb_array"`: the cameras tiled (3 per row, 424 x 240 each) |
| `get_state()` / `set_state(state)` | a snapshot of the episode and its restore (below) |
| `close()` | release the simulator |
| `observation_space`, `action_space` | `Dict` / `Box` (below) |
| `action_layout`, `proprio_layout`, `proprio_groups` | names of the action entries and of the proprioception vector |
| `control_hz`, `physics_substeps`, `max_episode_steps` | rates and the step cap |
| `instruction` | the task's goal text (the standard mode's task text), for language-conditioned policies |
| `camera_info()` | per camera at the current state: `intrinsic_cv` (3 x 3, pixels), `extrinsic_cv` (3 x 4: robot frame -> camera, OpenCV axes x right, y down, z forward), `cam2world_gl` (4 x 4: camera -> robot frame, OpenGL axes x right, y up, z backward; the inverse of `extrinsic_cv` with y and z flipped), `image_size` (W, H), `moves` (a wrist camera), `depth_range_m`; puzzles also `mount` |
| `state_sha()` | the state hash the graders compare |
| `success()` | the task's own success verdict now |
| `evaluate()` | the tooling's full verdict: puzzles `{"status", "safety"}`; IKEA `{"status", "final_check"}` |
| `trajectory()`, `save_trajectory(path)` | the rows applied since `reset()` (after every limit), saved as the privileged mode's deliverable (`/app/output/trajectory.npz`) |
| `grade(actions)` | replay `actions` from the frozen state exactly as the privileged verifier's replay does (`Session.replay(grading=True)` / `rcb_ikea.Session.replay`); resets the environment |
| `describe()` | all of the above as plain data |
| `task` | the `TaskInfo` |
| `session` | the underlying tooling session (`rcb_topogym.Session` / `rcb_ikea.Session`): the privileged tooling, for anything else (stepping it directly bypasses the environment's bookkeeping) |

`get_state()` / `set_state()`: the puzzles snapshot topogym's complete checkpoint of the current tick plus the tooling's
episode counters and the rows applied since `reset()`; `set_state` rewinds to the checkpoint and, when that checkpoint
no longer exists (a later `reset()` or an earlier rewind discarded it) or the restored hash differs, replays the rows
from the frozen state instead. It returns `"checkpoint"` or `"replay"`; either way the continuation is the same as
without the snapshot (tested). IKEA snapshots `rcb_ikea.Session.get_state()` (physics, mechanism, validity monitor,
rows). A snapshot belongs to its environment's task.

## Actions

`action_space` is a `Box` over `action_layout`: **absolute joint position targets per arm, then that arm's gripper
command**, tracked by the robot's position controllers. Out-of-range values are clipped.

| robot (tasks) | layout | gripper command |
|---|---|---|
| two ALOHA 2 / ViperX arms (most puzzles) | `left/waist .. left/wrist_rotate, left_gripper, right/..., right_gripper` (14) | opening width in metres, 0 = closed .. about 0.057 = open |
| Franka Panda + Robotiq 2F-85 (buzz-wire-3d, fiber-bundling, hedgehog-in-cage, lock-picking) | 7 joints + gripper (8) | the frozen instance's gripper space (`env.session.gripper_space`; see `env.session.robot_spec()["grippers"]`) |
| two Franka Pandas + Menagerie Panda hands (IKEA, all but svallet) | `left_actuator1..7, left_actuator8, right_...` (16) | 0 = closed .. 255 = open (the hand squeezes at its force limit when commanded 0) |
| two ViperX 300 s (IKEA svallet) | 6 joints + gripper per arm (14) | finger rail target, 0.002 m = closed .. 0.037 m = open |

`env.session.robot_spec()` (puzzles) has the joints, limits, gains, tool frames and grasping rule of each robot; the
puzzles' grasping is the task's own (`assisted` contact welds for the families that declare them, else friction).
dtype: float32 for the puzzles (the tooling executes float32 rows), float64 for IKEA (executed in float64).

## Observations

```python
obs = {"rgb":   {camera: (H, W, 3) uint8},       # renderer "nyx" or "mujoco"
       "depth": {camera: (H, W) float32},        # depth=True: z-depth in metres, 0 = no reading
       "proprio": (P,) float32}                  # env.proprio_layout
```

The proprioception vector is the standard mode's (the `proprio` of its service), group by group (`proprio_groups`:
name -> `[first index, count, meaning]`):

| group | puzzles | IKEA |
|---|---|---|
| `joint_pos` | measured joints in the action layout (gripper in its command space) | measured arm joints |
| `tcp_pose` | measured tool-frame pose per arm, x y z qw qx qy qz | the same (TCP sites between the fingertips) |
| `joint_cmd` | the targets being tracked (the last applied row) | the actuator commands applied |
| `joint_vel` | measured velocities | measured arm joint velocities |
| `gripper_opening` | the gap between the finger pads (m) | the sum of the finger joints (m) |
| `joint_effort` | the PD drives' forces | the actuator forces |

Frames: every pose and camera extrinsic the API gives is in the **robot frame** of the standard mode, quaternions wxyz:
puzzles: origin at the robot's kinematic root link, the simulator's world axes (+z up); IKEA: the left arm's base link
(Panda `left_link0`, ViperX `left_base_link`) with that link's axes (+z up, x the arm's forward direction). So a policy sees
what the standard mode's service gives, and an observed TCP pose is a valid end-effector target. The simulator's own
world frame stays available through `env.session` (and `env.T_world_from_robot`).

## Episodes, success and time limits

- `info["success"]`: the task's own success verdict after this step:
  - puzzles: `Session.success()`, the task's predicate held continuously for its confirmation window (most tasks:
    0.5 s), the success the verifier's replays require;
  - IKEA: the standard mode's rule: `success.py` and released and at rest and no breakdown, then the same again after
    holding the last row 1 s on a copy (`rcb_ikea.Session.final_check()`), checked every `check_every` control steps.
- `terminated`: the episode ended: success, the task's failure predicate (puzzles, e.g. touching the buzz wire) or a
  simulator breakdown (`info["episode_end"]`: `"success"`, the failure's reason, `"simulator_instability"` /
  `"sim_breakdown"`). A puzzle's scene stops at its episode end (later steps would not move it; `step()` then asks for
  `reset()`). An IKEA episode may be stepped on after its end, as the privileged mode's physics goes on.
  `info["infra_error"]` marks a simulator breakdown that the graders count as an infrastructure error.
- `truncated`: `max_episode_steps` control steps (default: the verifier's row cap of the task, `TaskInfo.max_episode_steps`).
  The Harbor agent budget is wall clock (`TaskInfo.agent_time_limit_s`, 3 h); the API does not enforce it.
- `reward`: 1.0 on the step the task first succeeds, else 0.0 (sparse). Puzzles also report the task's own dense
  `info["progress"]` (0..1).
- Other `info` keys: `control_step`, `sim_time_s`, `applied_action` (the row the simulator received after every limit),
  `speed_limited`, puzzles `robot_limits` and `table_guard` (the tooling's notices), IKEA `clipped`, `breakdown`,
  `settle_failed`.

## Limits: clamp, never fail

Every limit clamps the command and the episode goes on; none fails it. The API applies them where the graded runs do:

- puzzles, in `rcb_topogym.Session.step` (every mode, every replay): the joint limits (clip); the robots' joint speed
  limits (ALOHA pi rad/s per joint; Panda 2.175 / 2.61 rad/s) and, for the Panda, its Cartesian TCP limits (1.7 m/s,
  2.5 rad/s) and the command-lead bound, each shortening the step along its direction; the table guard (no gripper
  target lower than 15 mm below the tabletop: the target stops on the commanded path); the static guard (an arm deep in
  a fixed body stops where it is); the arm torque caps.
- puzzles, `rules="standard"` (default): the standard mode's **joint-speed rule** on top: an arm joint target moves at
  most 0.04 rad per control step from the current targets (`rcb_topogym.JOINT_STEP_LIMIT_RAD`; at 40 Hz 1.6 rad/s),
  componentwise as the standard service clamps a streamed row (`environment/sim/server.py`, `run_joint_position`, a row
  of one control step held); the grippers are not limited. A farther target is reached over the following steps if it
  is sent again. The standard mode's references obey it (`roboarc.load_reference(task, "standard")`; one exception:
  blanket-flatten-06's standard reference is its privileged file, with 3 rows faster than the rule, which the rule
  clamps). 12 privileged references move faster in places (blanket-flatten-06, buzz-wire-3d-09,
  fiber-bundling-07..10, interlocked-claws-12, rope-threading-04..08): their verifier replays them under the
  privileged rules, and their standard reference is a re-timed copy. Replay a privileged reference with
  `rules="privileged"` (`python -m roboarc replay` picks the rules of the reference's mode).
- IKEA, in `rcb_ikea.Session.step` (both modes and the grader's replays): the control range (clip), then the **speed
  clamp**: every arm joint target moves at most its real limit x control step from the previous target (Franka Panda
  2.175 / 2.61 rad/s, ViperX pi rad/s) and the **Franka Hand slew**: the hand command changes at most 0.1 m/s of finger
  width (`rcb_ikea.JOINT_SPEED_LIMITS_RAD_S`, `GRIPPER_WIDTH_SPEED_M_S`). The ViperX gripper is not limited.

`info["speed_limited"]` says whether a limit shortened this step; `info["applied_action"]` is what ran.

## Fidelity: no physics fork

- The physics, its settings and every rule are the graded tooling's (see [Design](#design)); the package only reads the
  repository's pinned copies of the tooling (`scripts/roboarc/shared/`, `ikea/scripts/ikea_mujoco/shared/` and
  `standard/`), which are byte-identical to the task directories' copies.
- `make()` (with `strict=True`) refuses a runtime the graded tasks would refuse:
  - the puzzles: the task's `runtime_binding.json` staged next to the frozen instance makes the Session check the
    instance, the tooling, the render bridge and the installed runtime (source tree, dependency versions, libuipc
    build), and the environment requires the verifier's start condition (`instance_matches`, exact or realized, a
    checked binding);
  - IKEA: the scene's files against the binding, the runtime (MuJoCo, numpy and Python versions) and the initial state
    hash, as the verifier does;
  - both: MuJoCo's `libmujoco.so` must be the EPA-patched 3.14.0 file (sha256 `52e2a24e...`), and numpy must have been
    imported with the graded settings `NPY_DISABLE_CPU_FEATURES=AVX512F,...` and `OPENBLAS_CORETYPE=Haswell`.
    `import roboarc` sets these when numpy is not imported yet; otherwise set them in the environment. Under pytest set
    `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (an installed pytest plugin imports numpy before any test code).
- Rendering never changes the physics: both renderers read a scratch copy of the joint positions (checked: a reference
  replayed with Nyx frames at every step ends on the same hash as without rendering, [Tests](#tests)).

## Rendering

| renderer | what | needs | cost |
|---|---|---|---|
| `nyx` | the graded images: Nyx path tracing at the task's camera rig (puzzles: `nyx_direct.py`; IKEA: `ikea_nyx.py` in its own render process) | a GPU with RT cores, the NVIDIA container toolkit with `NVIDIA_DRIVER_CAPABILITIES=all` (Vulkan); IKEA: the `-sim` image | a control step with 4 cameras (848 x 480) and depth, physics included, on an RTX 4090: about 60 ms (puzzles and IKEA); start-up 7-10 s |
| `mujoco` | MuJoCo's OpenGL raster of the **physics model** at the same camera poses: a quick look for debugging, **not the graded renderer** (other shapes and colours, simple lighting; a GPU cloth shows its MuJoCo stand-in) | an OpenGL context: `MUJOCO_GL=egl` (headless GPU) or `osmesa` (CPU only, needs libOSMesa) | the same step with EGL: about 40 ms (puzzles), 10-16 ms (IKEA) |
| `none` | no images | nothing | none |

Depth is MuJoCo's raster in both cases (the graded depth is the same raster). Policies meant for the graded tasks should
be trained and evaluated on `nyx` images.

## Wrappers

`roboarc.wrappers.EEPoseAction(env, mode="delta" | "absolute")` (puzzles): per arm 8 numbers, an absolute TCP pose
`x y z qw qx qy qz gripper` in the robot frame, or a delta `dx dy dz rx ry rz _ gripper` (translation in m, rotation
vector in rad, applied to the measured TCP pose). Each step solves the arm joints with the tooling's own IK
(`Session.ik`: bounded least squares on the robot's kinematics, seeded at the measured joints, no collision checking)
and steps the environment with that joint row, so every limit above applies. As in the standard mode's `ee_pose`
action, an arm whose IK residual exceeds 2 cm / 20 deg, or whose solution jumps more than 0.5 rad on any joint (another
IK branch), holds its targets; `info["ee_plan"]` says why. IKEA has no such wrapper: its tooling has no IK and its
standard mode no `ee_pose` action.

## Examples

- `examples/random_policy.py <task>`: a random walk around the current targets (the cheapest smoke test).
- `examples/replay_reference.py <task> [--video out.mp4]`: the reference through the API, checked against its recorded
  end hash.
- `examples/vla_loop.py <task>`: where a VLA plugs in: images + proprioception + instruction in, a chunk of joint
  targets out.

The command line: `python -m roboarc list | info <task> [--sim] | check | replay <task> [--traj F] [--mode privileged|standard]
[--rules standard|privileged] [--renderer none|mujoco|nyx] | patch-mujoco`.

## Tests

```bash
python -m pytest tests/api -m "not sim and not heavy"         # anywhere (numpy, pytest): catalogue, references, spaces
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/api -m "not heavy"   # in an image: + one environment end to end
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/api -m heavy         # in an image: the bit-identity replays
```

- `test_catalogue.py`, `test_spaces.py`: the 113 tasks and their metadata, the aliases, every puzzle reference against
  the freeze manifest, the space stand-ins. They run in CI.
- `test_env_sim.py` (marker `sim`; one task per image: `ROBOARC_TEST_PUZZLE`, default interlocked-claws-03;
  `ROBOARC_TEST_IKEA`, default svallet): the spaces contain the observations; `reset()` gives the frozen instance's
  hash every time, whatever the seed; holds are deterministic; out-of-range and too-fast actions are clamped, malformed
  ones raise; the joint-speed rule; a `get_state()` / `set_state()` round trip continues bit-identically (both restore
  paths); the reference reaches the task's own success, its saved trajectory is the privileged deliverable, and the
  verifier's replay (`grade`) credits it; `camera_info()`: `cam2world_gl` is the inverse of `extrinsic_cv` with the
  axis flip, and both TCPs project into the static cameras.
- `test_bit_identity.py` (marker `heavy`): each reference stepped through `roboarc.make()` from `reset()` in a fresh
  process (`python -m roboarc replay`) must end exactly on its recorded hash: the privileged reference under the
  privileged rules (what its verifier replays). Default selection: one base per puzzle family plus two IKEA scenes, and
  fiber-bundling-07 (a privileged reference faster than the standard rule); `ROBOARC_BITID_TASKS` overrides it. It also
  replays the standard references of fiber-bundling-07 and slotted-post-ring-01 under the standard rules (no step
  clamped, success).

Results of 2026-10-19 (fresh process per task, one CPU core each, 8 at a time; rendering off unless noted; the blanket
families with their CUDA cloth on an RTX 4090):

| task | rows | end hash (recorded = API) | wall |
|---|---:|---|---:|
| blanket-flatten-08 | 1480 | d539d4d85d80ab0e | 741 s |
| blanket-fold-01 | 884 | b443db6478ebfe08 | 363 s |
| buzz-wire-3d-06 | 10424 | a28ee3880f00a88a | 256 s |
| cast-coil-01 | 1595 | 4eb24b63a7463ca0 | 58 s |
| cast-keyhole-01 | 5097 | 17b34a5ea8c891ee | 98 s |
| fiber-bundling-01 | 3722 | c2d5a617ceddf9ef | 186 s |
| hedgehog-in-cage-01 | 1923 | 65295842c0910fad | 46 s |
| horseshoe-ring-05 | 7817 | 0284c3211ec6452f | 313 s |
| interlocked-claws-03 | 812 | 30545aa2bfa43760 | 27 s |
| interlocked-claws-03, Nyx frames at every step | 812 | 30545aa2bfa43760 | 54 s |
| knot-untying-03 | 753 | 191d932660caa264 | 219 s |
| lock-picking-01 | 4275 of 4303 | c2071db8651ca442 | 91 s |
| orbits-01 | 3330 | 27d5ce82a052442d | 262 s |
| rope-threading-02 | 3366 of 3386 | 740d4048045ab81d | 420 s |
| slotted-post-ring-01 | 436 of 437 | 5a730ec14ba1b22c | 121 s |
| IKEA lack | 67697 | 684fe3c488e01d1a, settled a4bc18f244c3687f | 395 s |
| IKEA svallet | 69414 | 19d049469045e1a2, settled 6413de2d8af8cc39 | 200 s |

All 17 match. Every puzzle reference reached `success` (the episode ends there: "x of y" rows), both IKEA references
pass `final_check`, and no step was shortened by any limit. Rendering with Nyx leaves the hash unchanged.

## Limitations

- Each task is one frozen instance: there is no randomisation of the scene (`seed` has no effect), as in the benchmark.
- One environment per process (use subprocess vector environments).
- The standard mode's service has more than the per-step interface: chunked submissions with interpolation, its wall
  clock, its live-episode grading and replay. The API's step is the service's control step; a policy that runs well
  here sees the same physics, observations and rules there.
- The graded renderer needs a GPU; the `mujoco` renderer is for quick looks only.
