# VLA policies

Preview. RoboARC evaluates vision-language-action (VLA) policies that are served the way Physical Intelligence's
[openpi](https://github.com/Physical-Intelligence/openpi) serves them: a policy server speaking `openpi_client`'s
websocket protocol receives camera images, the robot's state and a language instruction, and returns a chunk of
actions. RoboARC executes the chunk on the task's frozen scene, asks again, and grades the episode with the task's own
grader. A VLA team needs no code of its own: start the policy server, point RoboARC at it. The commands are the
subcommands of `roboarc vla`: `run` (graded episodes, inside the task's image), `stub` (a stand-in policy server) and
`droid` (a DROID-trained model on the one-arm puzzles, on the host).

Contents: [Setup](#setup) · [Running](#running) · [Robots and layouts](#robots-and-layouts) · [Prompts](#prompts) ·
[Episodes and grading](#episodes-and-grading) · [Results](#results) · [GPU and timings](#gpu-and-timings) ·
[Determinism](#determinism) · [Checking a setup](#checking-a-setup) · [Scope](#scope)

The tasks (families, difficulty, success rules) are described in [docs/families/](families/README.md) and the rules
shared by every family in [docs/benchmarks/roboarc.md](benchmarks/roboarc.md); hardware, install and images in
[RUNNING.md](RUNNING.md#requirements).

## Setup

1. The images and a GPU with RT cores ([RUNNING.md](RUNNING.md#requirements): `roboarc images pull
   --accept-licences`, then `roboarc doctor`). The commands below run the pinned puzzle image:

   ```bash
   IMAGE=$(bash images/pull_all.sh --list | awk '$1 ~ "/roboarc:" {print $1}')
   ```
2. The protocol's two Python packages for the image, at the versions openpi's own lock pins: `msgpack==1.1.0` and
   `websockets==15.0.1` (`pip install -e .[vla]`). The images ship without them: install them into a directory and put
   it on `PYTHONPATH`, or build a derived image:

   ```bash
   mkdir -p .vla-lib && docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD/.vla-lib:/vla-lib \
     "$IMAGE" python -m pip install --target /vla-lib msgpack==1.1.0 websockets==15.0.1
   ```

   RoboARC speaks the protocol with its own code (`roboarc/vla/wire.py`), tested byte for byte against `openpi_client`
   and end to end against openpi's `WebsocketPolicyServer`: `openpi_client` pins `numpy < 2`, and the simulator's
   numpy (2.3.5) is part of the runtime the graders check.
3. A policy server reachable from the container, e.g. openpi's
   [`serve_policy.py`](https://github.com/Physical-Intelligence/openpi/blob/main/scripts/serve_policy.py) (port 8000).

## Running

<!-- unverified: a real policy server, and the IKEA line (the `roboarc-ikea-sim` image). The puzzle line was run
     against the stand-in server `roboarc vla stub` -->
```bash
RESULTS=$PWD/vla_results && mkdir -p "$RESULTS"      # where the results go: any directory, made before the run
docker run --rm --gpus 1 -e NVIDIA_DRIVER_CAPABILITIES=all -u $(id -u):$(id -g) -e HOME=/tmp \
  -v $PWD:/repo:ro -v $PWD/.vla-lib:/vla-lib:ro -v $RESULTS:/results -w /repo \
  -e PYTHONPATH=/repo:/opt/topogym:/vla-lib "$IMAGE" \
  python -m roboarc vla run --task interlocked-claws-08 --policy ws://POLICY_HOST:8000 \
    --seed 0,1,2 --episodes 1 --run-id my-pi05-aloha --out /results
```

Inside an image the command `roboarc` is `python -m roboarc`; `python -m roboarc.vla.run` is the module behind
`roboarc vla run` and takes the same arguments. Add `--network host`, or a Docker network shared with the policy
server, so the container reaches it. IKEA tasks run in the pinned `roboarc-ikea-sim` image (`bash images/pull_all.sh
--list` prints its reference) with `PYTHONPATH=/repo:/vla-lib` (the libraries built for its Python 3.12).

| option | meaning |
|---|---|
| `--task` | a task id or alias (`roboarc.list_tasks()`, `roboarc list`) |
| `--policy` | the server: `ws://host:port` or `host:port`; an API key from `OPENPI_API_KEY` |
| `--seed` | instance seeds: `0`, `0,1,2` or `all` (every task allows 0-2: a puzzle's seeds 1 and 2 are the same scene and physics in another look, lighting, table and objects' colours and materials; an IKEA task's the same scene under other lighting) |
| `--episodes` | episodes per seed (a deterministic policy repeats itself; for sampling policies) |
| `--run-id` | the policy's name in the results tree |
| `--convention` | `aloha`, `droid` or `droid-velocity` (default: by robot, below) |
| `--horizon` | actions executed per chunk before asking again (openpi's action horizon; default aloha 25, droid 8) |
| `--policy-hz` | the rate of the chunk's actions (default: one action per control step, 40 Hz; see below) |
| `--max-steps`, `--time-limit-s` | the episode's limit in control steps or simulated seconds (default: below) |
| `--image-size` | the square image size (224) |
| `--camera-poses` | `droid`: DROID's camera poses on the one-arm puzzles; `molmoact2`: the same with `agentview` aimed as in AI2's MolmoAct2 simulation ([DROID-trained policies](#droid-trained-policies)) |
| `--prompt`, `--full-prompt` | override the instruction, or send the whole task paragraph ([Prompts](#prompts)) |
| `--video` | render `video.mp4` while grading (GPU) |
| `--export-reference F` | write the task's reference solution in the convention's action units (for [checking a setup](#checking-a-setup)) |

## Robots and layouts

Every image is resized with padding to 224 x 224 (openpi's `image_tools.resize_with_pad`: the 848 x 480 camera image
scaled to 224 x 126 and centred on black) from the task's own cameras, rendered with Nyx as for every other interface.
Joint angles are in radians; every pose-free quantity is the robot's own.

### Two arms: the `aloha` convention (openpi's ALOHA example)

ALOHA 2 puzzles (two 6-DoF ViperX arms), IKEA svallet (two ViperX arms) and the other IKEA scenes (two 7-DoF Franka
Pandas with Panda hands: the same layout with 7 joints per arm).

```python
{"state": (14,) float64,                    # left 6 joints, left gripper, right 6 joints, right gripper (IKEA Pandas: 16)
 "images": {"cam_high": (3, 224, 224) uint8, "cam_low": ..., "cam_left_wrist": ..., "cam_right_wrist": ...},
 "prompt": str}
# reply: {"actions": (H, 14)}: absolute joint targets in the same layout, gripper 0 = closed .. 1 = open
```

- Joints: measured positions. RoboARC's ViperX arms have the ALOHA (Interbotix) joint names, zeros, directions and
  limits, so the values go as they are; openpi's `adapt_to_pi` joint flips and gripper transforms run in the server, as
  for a real ALOHA.
- Gripper: the measured gripper, linearly normalised over the robot's command range (ALOHA 2: finger rail 0.002 m =
  closed .. 0.037 m = open; Panda hand: 0 .. 255; the measured finger opening for IKEA), and the action's gripper mapped
  back the same way. Real ALOHA normalises the observed finger position and the commanded motor angle separately; in
  simulation both are the same linear coordinate (as in openpi's ALOHA simulation data).
- The four cameras are the tasks' own (`cam_high`, `cam_low` on the table, the wrist cameras on the gripper bases).
- A reply with more dimensions keeps the first 14 (16), as openpi's `AlohaOutputs` does.

### One Franka arm with a Robotiq 2F-85: the `droid` conventions (openpi's DROID example)

buzz-wire-3d, fiber-bundling, hedgehog-in-cage, lock-picking.

```python
{"observation/exterior_image_1_left": (224, 224, 3) uint8,     # camera `front_oblique`
 "observation/exterior_image_2_left": (224, 224, 3) uint8,     # camera `agentview` (openpi's DROID inputs ignore it)
 "observation/wrist_image_left": (224, 224, 3) uint8,          # camera `wrist`
 "observation/joint_position": (7,) float64,
 "observation/gripper_position": (1,) float64,                 # 0 = open .. 1 = closed (DROID's)
 "prompt": str}
```

- `droid` (default): the reply is `(H, 8)` absolute joint positions + gripper position, the action space of openpi's
  joint-position DROID configurations, which openpi recommends for simulation (joint velocities are hard to reproduce
  in a simulator).
- `droid-velocity`: DROID's original action space, executed as DROID's robot environment does: each action holds for
  1/15 s; its joint target is the measured joints + velocity x 0.2 rad (the components clipped to [-1, 1]); the gripper
  position is binarised at 0.5 (openpi's DROID client).
- Gripper: the Robotiq drive 0 rad = open .. 0.8 rad = closed, linearly normalised.
- Cameras: `front_oblique` (closer) and `agentview` (an overview) are the two static views, `wrist` is on the hand;
  `--cameras` remaps them (`observation/exterior_image_1_left=agentview,observation/wrist_image_left=wrist`).

### DROID-trained policies

A policy trained on DROID (openpi's pi0.5-DROID, AI2's MolmoAct2-DROID, NVIDIA's GR00T-N1.7-DROID and
Cosmos3-Nano-Policy-DROID) needs three things
DROID's robot has and the one-arm puzzles do not. RoboARC provides them without changing a task:

- **Camera poses** (`--camera-poses droid`, `roboarc/vla/droid_cameras.py`): `front_oblique`, `agentview` and `wrist`
  move to the poses of DROID's simulated evaluation scene (sim-evals), relative to the robot: the exterior cameras at
  (0.05, -/+0.57, 0.66) m from the robot base with a 71.5 deg vertical field of view, the wrist camera where DROID's sits
  on the Robotiq (56.7 deg). The change is made in memory when the scene is built: the tasks, their files and the
  physics stay as they are, the frozen-runtime checks hold, and the episodes' end states are those without it. With the
  task's own cameras pi0.5-DROID came within 3 cm of the object in 0 of 24 one-minute episodes (buzz-wire-3d-06,
  lock-picking-03; three checkpoints); with DROID's poses in 7 of 24. `--camera-poses molmoact2` is the same but
  for `agentview`, aimed as the one exterior camera of AI2's MolmoAct2 simulation (`allenai/molmoact2`
  `sim_eval/robots/franka_droid.py`: from the same position, 18 deg apart).
- **Wrist angle and start pose** (`roboarc/vla/droid_adapter.py`, a policy server around the model): RoboARC mounts the
  Robotiq at -45 deg about the flange, DROID at 0, so the model sees q7 - pi/4 and its actions get pi/4 back; the
  first 2.5 s move the arm to DROID's reset joints before the model is asked. The adapter reads only the joints and
  the control step: it holds nothing about a task.

The models behind the adapter (`roboarc/vla/droid_models.py`) share one interface: `load()` once (the weights, then one
call on a blank observation, so the first real call does not outlast the client's keepalive), then `infer(obs)` ->
`(H, 8)` in DROID's units. Each runs in its own Python environment and imports its libraries only when it loads.
Each is called as its own DROID client calls it:

- pi0.5-DROID (openpi): two images, the exterior camera of sim-evals (`agentview`, on the robot's left) and the wrist,
  at 224 x 224 with padding; 15 actions per query, 8 executed, the gripper binarised at 0.5 (openpi's DROID example).
- MolmoAct2-DROID (Hugging Face `allenai/MolmoAct2-DROID`): two images, the exterior camera of AI2's simulation
  (`agentview` with `--camera-poses molmoact2`) and the wrist, as AI2's DROID server and clients send them; each
  DROID's 16:9 frame (320 x 180: the runner pads to 320 x 320, the model takes the padding off); 15 actions per
  query, all 15 executed (the checkpoint's `n_action_steps` and AI2's clients); 10 flow steps, the gripper clipped to
  [0, 1] (AI2's DROID server); float32 as the model card's reported runs (about 24 GB),
  `molmoact2-bf16` its bfloat16 path for a smaller GPU (under 16 GB).
- GR00T-N1.7-DROID (Hugging Face `nvidia/GR00T-N1.7-DROID` through NVIDIA's `Isaac-GR00T`): two images, `agentview`
  and the wrist, each DROID's 16:9 frame at 320 x 180, one frame per query (the checkpoint's video delta indices); the
  state is the joints, the gripper and the hand's pose (`eef_9d`: Franka's `panda_link8` from its kinematics, the
  rotation stored as Isaac-GR00T's DROID data stores it; it equals that repository's `demo_data/droid_sample` to
  3e-7); 40 actions per query, 8 executed (as NVIDIA runs it in simulation, `Isaac-GR00T/examples/RoboLab`; its
  real-robot client `Isaac-GR00T/examples/DROID/main_gr00t.py` executes 15), the gripper binarised at 0.5; 4 flow
  steps, bfloat16 (6.4 GB).
- Cosmos3-Nano-Policy-DROID (Hugging Face `nvidia/Cosmos3-Nano-Policy-DROID` through NVIDIA's `cosmos-framework`, its
  RoboLab policy service): three images, the wrist and both exterior cameras (`agentview` on the robot's left,
  `front_oblique` on its right), each DROID's 16:9 frame at 640 x 360, as RoboLab's Cosmos3 client sends them (the
  service puts them in one image, the exteriors at half size below the wrist); the joints and the gripper; 32
  actions per query, all 32 executed (that client), the gripper binarised at 0.5; the service's defaults: 4 steps,
  guidance 3, a seed per query drawn from seed 0.

<!-- unverified: the four models' set-up lines (the earlier ones, with the git lfs pattern rooted: the same files,
     read with git-lfs 3.0.2) and their runs. The `--model hold` line was run -->
`roboarc vla droid` runs it all in one command (it is the script `scripts/vla_droid.sh`, with the same arguments): it
starts the adapter around the model, runs the episodes in the
puzzle image with each model's cameras (`agentview` as exterior 1 and the wrist, Cosmos3 also `front_oblique` as
exterior 2; `--camera-poses droid`, MolmoAct2 `molmoact2`), `--jobs` at a time against that one server, and stops the server.

```bash
# once: openpi at the commit RoboARC was tested with (the checkpoints download on first use, 12 GB each, into
# ~/.cache/openpi or $OPENPI_DATA_HOME); the script finds it in ~/openpi or $OPENPI
git clone https://github.com/Physical-Intelligence/openpi.git ~/openpi && cd ~/openpi
git checkout 215abfb217dbac7d5f1273282331b9b1866c0479 && GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev
# MolmoAct2: AI2's environment (no lock file: torch 2.11.0 and transformers 4.57.6 tested) + the protocol's msgpack;
# the weights download on first use (22 GB, into ~/.cache/huggingface or $HF_HOME); found in ~/molmoact2 or $MOLMOACT2
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/allenai/molmoact2.git ~/molmoact2 && cd ~/molmoact2
git checkout 66b87e64efd99dfd103241418113955cf64dfa9c && uv sync && uv pip install msgpack==1.1.0
# GR00T: NVIDIA's environment + the protocol's msgpack and websockets; the weights download on first use (6.5 GB +
# 4.6 GB for the backbone nvidia/Cosmos-Reason2-2B, whose licence must be accepted on Hugging Face); found in
# ~/Isaac-GR00T or $GROOT
git clone https://github.com/NVIDIA/Isaac-GR00T.git ~/Isaac-GR00T && cd ~/Isaac-GR00T
git checkout 1a1837f20538b7d7e21f977a11a5aee14f99803c && git lfs pull --include "/scripts/deployment/dgpu/wheels/*"
uv sync && uv pip install msgpack==1.1.0 websockets
# Cosmos3: NVIDIA's environment (Python 3.13, torch 2.13 for CUDA 13, so an NVIDIA driver 580 or newer) + the
# protocol's msgpack; the repository asks for uv 0.11.29, and its server needs uv on PATH (it fetches the weights
# with it on first use, 33 GB); found in ~/cosmos-framework or $COSMOS3
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVIDIA/cosmos-framework.git ~/cosmos-framework
cd ~/cosmos-framework && git checkout bbd4a67c3cf54cc6699464ad1259f3a0f6537da9
uvx uv@0.11.29 sync --all-extras --group=cu130-torch213-train --group=policy-server
uvx uv@0.11.29 pip install msgpack==1.1.0

roboarc vla droid --task lock-picking-03 --model jointpos --seed all --jobs 3 --out $RESULTS
roboarc vla droid --task lock-picking-03 --model molmoact2 --jobs 3 --out $RESULTS
roboarc vla droid --task hedgehog-in-cage-02 --model hold --out $RESULTS -- --max-steps 400   # checks the setup
roboarc vla droid --task lock-picking-03 --model jointpos --gpu 1 --out $RESULTS             # all of it on GPU 1
```

`--gpu N` puts the model server and every episode on one GPU, named by its `nvidia-smi` index or its UUID: the
containers get `--gpus device=N`, the model server `CUDA_VISIBLE_DEVICES=N` in `nvidia-smi`'s numbering. Without it
the containers see every GPU (`--gpus all`) and the model server takes the GPU its framework picks, which is right on
a host with one GPU. On a shared host with several, name a free one: `nvidia-smi --query-gpu=index,memory.used
--format=csv` shows them.

| `--model` | checkpoint | convention | executed per query, image size |
|---|---|---|---|
| `jointpos` (default) | openpi `pi05_droid_jointpos` | `droid` | 8, 224 |
| `velocity` | openpi `pi05_droid` | `droid-velocity` | 8, 224 |
| `polaris` | openpi `polaris/pi05_droid_jointpos_polaris` (jointpos with PolaRiS' simulated data) | `droid` | 8, 224 |
| `molmoact2` | Hugging Face `allenai/MolmoAct2-DROID` at `d8c1abd`, float32 | `droid` | 15, 320 |
| `molmoact2-bf16` | the same in bfloat16 | `droid` | 15, 320 |
| `groot` | Hugging Face `nvidia/GR00T-N1.7-DROID` at `05e7cc9`, bfloat16 | `droid` | 8, 320 |
| `cosmos3` | Hugging Face `nvidia/Cosmos3-Nano-Policy-DROID` at `24aa5d1` | `droid` | 32, 640 |
| `hold` | none: holds the arm (numpy only) | `droid` | 8, 224 |

### Rates

The puzzles step at 40 Hz (25 ms control steps, 10 physics steps each), IKEA at 400 Hz. By default each action of a
chunk is one 40 Hz control step: the rate of RoboARC's own data. A policy trained at another rate sets `--policy-hz`
(openpi's ALOHA data: 50; DROID: 15): the chunk is then played over its own duration (joints interpolated linearly
between consecutive actions, the gripper switched at its action), so 25 actions at 50 Hz take 20 control steps. On
IKEA the default is 50 Hz. Every limit of the task applies to the executed rows and clamps, never fails: the standard
mode's joint-speed rule (arm joint targets move at most 0.04 rad per control step), the robots' joint speed limits, the
table guard (all listed in [docs/API.md](API.md#limits-clamp-never-fail)).

## Prompts

The prompt is the task's goal in one sentence: the first sentence of the task statement, e.g. "Completely separate the
two claw pieces and briefly hold both still." (median 106 characters; 14 of 114 tasks over 200). `--full-prompt` sends
the whole goal paragraph (the success requirements stay out). The same text serves every seed; colours named in it are
those of the seed's scene. pi0 truncates prompts at 48 tokens (pi0.5: 200).

Tasks whose goal is a shape (blanket-fold 01-05, knot-untying 07 and 09) also have goal pictures for other interfaces;
a VLA prompt cannot carry them yet.

## Episodes and grading

- Each episode runs in a fresh process from the task's frozen instance of the chosen seed: the graded birth state.
  The policy server's `reset_pose` metadata is ignored (the robot starts where the task puts it).
- The policy is asked after every `--horizon` actions; the simulation waits for its reply (no wall clock).
- The episode ends at the task's own end (success, held for the task's confirmation window; the task's failure
  predicate; a simulator breakdown) or at the step limit: by default twice the reference solution's length, rounded up
  to 100 steps (12000 steps, 5 simulated minutes, for a task without a reference).
- Grading follows the task's standard mode: the executed rows are replayed once more, in a new process, from the frozen
  instance (after a live success, the tolerant grading replay of the GPU families). **success = live success and replay
  success**, with no infrastructure error (a simulator breakdown, live or replayed, or a replay that produced nothing
  after a live success). A success with deep task contact is held for review (`success_pending_review`) instead of
  credited. IKEA tasks grade the live episode with the settled check, as their standard mode does.
- `deterministic` (live and replayed end states equal) is reported.

## Results

VLA results have their own layout and configuration key and are never mixed with or ranked against the coding-agent
interface's results:

```
<out>/<task>/<run id>/config.json         settings + config_key ("vla:<12 hex>")
<out>/<task>/<run id>/results.json        every episode of the run id
<out>/<task>/<run id>/s<seed>/e<k>/
    result.json      success, success_pending_review, live_success, replay_success, deterministic, infra_error,
                     steps, episode_end, end hashes, timings, server metadata (the model's pinned versions)
    live.json        the live episode (prompt, server metadata, cameras, rates, review and forcing evidence)
    trajectory.npz   the executed rows (after every limit), in the format of the privileged mode's deliverable
    chunks.npz       every action chunk the policy returned and the control step it was asked at
    replay.json      the grading replay
    video.mp4        with --video
```

`config_key` hashes the evaluation's settings (convention, horizon, rates, image size, rules, renderer, step limit,
prompt, package version), not the policy: episodes compare within one key. One run id holds one configuration; a run
with other settings under the same id is refused.

## GPU and timings

Nyx renders the images (one GPU with RT cores); the physics runs on one CPU core per episode (the blanket families'
cloth on the GPU). Measured on an RTX 4090 with a loaded 32-core host:

| | ALOHA 2 (an interlocked-claws base, 4 cameras) | Franka (hedgehog-in-cage-02, 3 cameras) |
|---|---:|---:|
| control step, physics only | 34 ms | 18 ms |
| Nyx render of all cameras at 848 x 480 | 45 ms (22 frames/s) | 39 ms |
| resize with pad to 224, per image | 1.7 ms | 1.7 ms |
| a policy query (render + layout), stub server | 54 ms | 34 ms |
| per control step, horizon 25 / 8 | 36 ms | 22 ms |

MolmoAct2-DROID's server holds 11 GB in bfloat16 and about 24 GB in float32 (its model card); a query (15 actions,
1 s of simulated time) took 0.22 s on a free RTX 4090 and 0.84 s on a busy one (bfloat16). On a free RTX PRO 6000
(96 GB): GR00T-N1.7-DROID holds 6.4 GB and a query (40 actions) takes 63 ms; Cosmos3-Nano-Policy-DROID holds 33 GB and
a query (32 actions, about 2 s of simulated time) takes 1.3 s.

pi0.5-DROID through `roboarc vla droid`, model and episodes on one RTX 4090 that other jobs kept busy (8.5 GB of
its memory taken): the model server holds 8.6 GB and an episode about 1 GB. One server keeps about 4 episodes busy:
4 parallel 1600-step lock-picking episodes took 49 s each against 40 s for one alone; with 6 or 9 the GPU's memory ran
out for some, and the rest waited for the server (82-100 s each). The easiest setting of each one-arm family, 4 at a
time to the default step limit (image 2.1.0, DROID's camera poses, live episode + grading replay):

| task | control steps | live | replay | per control step (live) |
|---|---:|---:|---:|---:|
| lock-picking-03 | 32700 | 21.9 min | 10.1 min | 40 ms |
| buzz-wire-3d-06 | 20900 | 17.4 min | 9.0 min | 50 ms |
| fiber-bundling-02 | 7300 | 7.8 min | 5.4 min | 64 ms |
| hedgehog-in-cage-08 | 6000 | 4.8 min | 2.4 min | 48 ms |
| hedgehog-in-cage-02 | 3800 | 2.5 min | 1.8 min | 39 ms |

All five took 33 minutes, the model's start included. All 18 one-arm puzzles to their limits are 410500 control steps
per seed.

Images are rendered only when the policy is asked, so with a horizon of 25 rendering takes about 2 ms per control
step. The renderer's resolution and sampling are part of the frozen scene; images at 224 take the same render. The
grading replay is physics only (with `--video`, Nyx every few steps). Episode memory: about 1-2 GB (blanket 3 GB) plus
the renderer's GPU memory.

## Determinism

The physics is deterministic: the same rows from the same instance give the same end state (CPU MuJoCo; the blanket
families' deterministic GPU cloth). A deterministic policy therefore gives identical episodes; the reference solution
served through the policy loop ends on the release's recorded hash for every seed. Sampling policies (pi0 / pi0.5 draw
noise) differ from episode to episode unless their server fixes the seed.

## Checking a setup

`roboarc vla stub` (the module `roboarc.vla.stub_server`) is a stand-in policy server with openpi's protocol:

```bash
# in the image, with the docker line of "Running": /results is the mounted $RESULTS (the repository is read-only)
python -m roboarc vla run --task interlocked-claws-08 --export-reference /results/ref.npz
roboarc vla stub --policy replay --actions $RESULTS/ref.npz --horizon 50 --port 8000   # anywhere: uv pip install -e '.[vla]'
roboarc vla stub --policy hold --port 8000                                             # never succeeds
```

Served through the policy loop, the reference succeeds and ends on its recorded state hash; `hold` repeats the
observed state and fails. The policies need numpy only and also run under openpi's own server.

## Scope

- Preview: the command line and the result fields may change in a later release.
- The simulation waits for the policy's reply, so inference latency takes no simulated time.
- A shape goal reaches the policy in the prompt's words; the goal pictures serve the other interfaces
  ([Prompts](#prompts)).
- The reported runs are zero-shot. Fine-tuning on RoboARC's scenes is the expected route for base models trained on
  real robots.
