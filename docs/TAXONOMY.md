# Taxonomy (`task.toml` `[metadata]`)

The values RoboARC's tasks use (`tasks/` and `ikea/tasks/`). See [PROTOCOL.md](PROTOCOL.md) §1, §2 for the
recommended fields and the settings a task declares. A new value is added by pull request.

| Field | Values |
|---|---|
| `author_name` | the task author's name (an organization may go in `author_org`) |
| `task_type` | `trajectory` (the deliverable is `/app/output/trajectory.npz`, replayed by the verifier) · `policy` (code the verifier runs; no RoboARC task) |
| `mode` | `privileged` · `standard`; the protocol's registry (`MODE_REGISTRY` in `scripts/task_conventions.py`) also names `standard-realtime`, which no RoboARC task uses |
| `protocol` | standard mode only: `eai-standard/2.4` ([STANDARD_MODE.md](STANDARD_MODE.md)) |
| `simulator` | `roboarc` (the puzzles: the `topogym` package on MuJoCo 3.14.0, CPU float64, the blanket cloth on a deterministic CUDA step, Nyx RGB) · `mujoco` (IKEA MuJoCo: plain MuJoCo 3.14.0) |
| `robot` | `aloha-2` (ALOHA 2, two arms with parallel grippers) · `franka-robotiq-2f85` (a Franka Panda arm with a Robotiq 2F-85 gripper) · `franka-panda-bimanual` (two Franka Panda arms with their standard hands; IKEA) |
| `family` | the puzzle family with underscores (`interlocked_claws`), shared by the family's bases; IKEA: the product (`lack`) |
| `dynamics` | `static` · `dynamic`: whether something in the scene other than the robot does work by a force other than the robot's and gravity (PROTOCOL §2); every RoboARC task is `static` |
| `oracle` | `full` (the reference solution scores 1) · `partial` · `none` (see PROTOCOL §4) |
| `oracle_evidence` | required unless `oracle = "full"`: where the evidence for solvability is |
| `expert_time_estimate_hours` | number |
| `instance_seed` | the frozen instance's seed (informational; `[metadata.seeds]` is what a runner reads) |

RoboARC's own fields:

| Field | Values |
|---|---|
| `difficulty` | `very_easy` · `easier` · `easy` · `medium` · `hard`: the task's rating within its family; task names carry no difficulty |
| `grasp` | `assisted (<model> grasp regions on <parts>)` · `physical (no assisted grasping)` |
| `success_rule` | the task-native success rule and its terminal revision (text) |
| `tooling_version` | the generation of the privileged tooling `rcb_topogym.py` the task is bound to |

## Task names
- `<base>-privileged` and `<base>-standard`: the two modes of one frozen instance.
- Puzzles: `<base>` is `roboarc-<family-with-dashes>-<NN>-i<NN>` (e.g. `roboarc-cast-coil-06-i01`). The first `<NN>` is
  the base's two-digit number within its family (never reused: a removed task leaves a gap), `i01` the MuJoCo
  instance. `[metadata].family` is the family (`cast_coil`), `difficulty` its rating.
- IKEA MuJoCo: `<base>` is `roboarc-ikea-<product>-i00`.
- `[task].name = "robot-coding-bench/<folder>"` (the Harbor namespace the tasks were published under).
- No versions or durations in names; a task's version is its digest (`tasks/<task>/DIGEST`).

## Run ids
`<harness>-<model>-<effort>`, for example `codex-gpt5-xhigh` or `claude_code-opus_4_1-max`
(`scripts/harbor/run_agent_batch.sh` derives it).
- `harness`: the Harbor agent name with `-` replaced by `_` (`codex`, `claude_code`).
- `model`: lower case, vendor prefix dropped, `gpt-<n>` joined (`gpt5`), a leading `claude-` dropped, characters other
  than letters and digits replaced by `_`. Reuse the token already used for a model in existing results.
- `effort`: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or `default` when the harness sets none.
- Provider, service tier, reset budget and other settings go in the results metadata, not in the id.
