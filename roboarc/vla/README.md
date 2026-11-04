# DROID-trained VLAs on the one-arm puzzles

Two models trained on DROID, zero-shot, on RoboARC's 18 one-arm puzzles (Franka + Robotiq 2F-85), in one command:
openpi's pi0.5-DROID and AI2's MolmoAct2-DROID.

## Run

From the repository root:

```bash
scripts/vla_droid.sh --task lock-picking-03,buzz-wire-3d-06 --jobs 2 --out results
```

The script starts the model server, runs every (task, seed) episode in the puzzle image, `--jobs` at a time, and stops
the server. Each episode writes `results/<task>/<run id>/s<seed>/e1/result.json` (`success`, `steps`, ...); the run id
is `pi05-droid-<model>` or `<model>-droid` for MolmoAct2.

| option | values | default |
|---|---|---|
| `--task` | task ids, comma-separated (below) | required |
| `--model` | pi0.5: `jointpos`, `velocity`, `polaris`; `molmoact2` (float32), `molmoact2-bf16`; `hold` = no model, checks the setup | `jointpos` |
| `--seed` | `0`, `0,1,2` or `all` (the scene's colours and materials; same physics) | `0` |
| `--jobs` | episodes at a time | `1` |
| `--out` | results directory | `./vla_results` |

## Tasks

| | very easy | easier | easy | medium | hard |
|---|---|---|---|---|---|
| `buzz-wire-3d-` | 06 | 02, 07 | | 04 | 10 |
| `fiber-bundling-` | | 02 | 08 | 04, 09 | 05 |
| `hedgehog-in-cage-` | | 02, 08 | 03, 07 | | 06 |
| `lock-picking-` | | | 03 | 04 | 05 |

Each episode runs to the task's end or its step limit (twice the reference solution: 3,800 to 70,600 steps at 40 Hz).

## What is adapted

DROID's robot is the same arm and gripper. Three things differ; none of the tasks, their files or the physics change.

| | DROID | the puzzles | adapted by |
|---|---|---|---|
| cameras | exterior at (0.05, ±0.57, 0.66) m from the base; wrist camera on the Robotiq | the task's own | runner option `--camera-poses droid` ([droid_cameras.py](droid_cameras.py)) |
| wrist joint q7 | Robotiq on the flange | Robotiq at -45° about the flange | model server: the model sees q7 - π/4 ([droid_adapter.py](droid_adapter.py)) |
| start pose | DROID's reset joints | the task's | model server: the first 2.5 s move the arm there |

The actions run as DROID's, 15 per second, and each model is called as its own DROID client calls it
([droid_models.py](droid_models.py)):

| | pi0.5-DROID | MolmoAct2-DROID |
|---|---|---|
| images | exterior 1 (`agentview`, as in sim-evals), wrist: 224 x 224 with padding | exterior 1 (`agentview`, aimed as in AI2's simulation: `--camera-poses molmoact2`), wrist: 320 x 180 |
| actions per query | 15, 8 executed (`--horizon 8`) | 15, all executed (`--horizon 15`) |

## Setup (once)

1. Docker with the GPU, and read access to the puzzle image: `docker login ghcr.io`.
2. openpi, with [uv](https://docs.astral.sh/uv/) (the checkpoints download on first use, 12 GB each):

   ```bash
   git clone https://github.com/Physical-Intelligence/openpi.git ~/openpi && cd ~/openpi
   git checkout 215abfb217dbac7d5f1273282331b9b1866c0479 && GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev
   ```

3. For MolmoAct2, AI2's environment (the weights download on first use, 22 GB):

   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/allenai/molmoact2.git ~/molmoact2 && cd ~/molmoact2
   git checkout 66b87e64efd99dfd103241418113955cf64dfa9c && uv sync && uv pip install msgpack==1.1.0
   ```

4. The protocol's two packages, from the repository root:

   ```bash
   mkdir -p .vla-lib && docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD/.vla-lib:/vla-lib \
     ghcr.io/mll-lab-nu/roboarc:2.2.0 python -m pip install --target /vla-lib msgpack==1.1.0 websockets==15.0.1
   ```

Other places: `OPENPI=<dir>`, `MOLMOACT2=<dir>`, `VLA_LIB=<dir>`, `IMAGE=<image>` in the environment.

## GPU

One RTX 4090: the model server takes 8.6 GB and each episode about 1 GB, so `--jobs 4`. The four easiest settings
above (lock-picking-03, buzz-wire-3d-06, fiber-bundling-02, hedgehog-in-cage-02) took 33 minutes 4 at a time, on a
GPU shared with other jobs. MolmoAct2-DROID: 11 GB in bfloat16 (`molmoact2-bf16`), about 24 GB in float32 (the model
card); 0.22 s per query on a free 4090, 0.84 s on a busy one. More: [docs/VLA.md](../../docs/VLA.md#droid-trained-policies).
