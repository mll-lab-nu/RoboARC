# ikea/: the IKEA MuJoCo suite (developer notes)

Twelve IKEA furniture-assembly scenes in plain MuJoCo 3.14.0 (CPU, float64), each a task in both modes: 24 task
directories in the repository's `tasks/`: `tasks/roboarc-ikea-<scene>-i00-{privileged,standard}`. Scenes by Xiaoyu and Zhen Wu's team. What the
tasks are, how they are graded and their timing: `../docs/benchmarks/ikea_mujoco.md`.

| path | what |
|---|---|
| `scripts/ikea_mujoco/mk_tasks.py`, `mk_standard_tasks.py` | the generators (privileged, standard); `--check` compares without writing |
| `scripts/ikea_mujoco/*.json` | generator inputs: `tasks.json` (texts), `assets.json` (scene pins), `bindings.json` (image id, what each scene builds to), `evidence.json` / `standard_evidence.json` (Harbor checks and timings; they set verifier timeouts and row caps) |
| `scripts/ikea_mujoco/shared/`, `standard/` | copied unchanged into the tasks: the tooling `rcb_ikea.py` and the privileged grader; the sim service, Nyx render process, standard grader and oracle client (`std_reference.py` makes standard references) |
| `scripts/ikea_mujoco/oracles/<scene>/` | the shipped references (`<scene>[_ref].npz`, generator inputs) and how each was built |
| `scripts/ikea_mujoco/tests/` | grader and sim-service unit tests, run inside the base image |
| `scripts/ikea_mujoco/REVIEW.md`, `provenance.json` | the decision log; the scenes' upstream provenance (kept out of the image) |
| `images/ikea-mujoco/` | the scene code (`scenes/<scene>/`), the base and sidecar recipes, `build.sh` (README there) |
| `ci/check.sh` | the checks CI runs |

The generators render with the repository's task layer (since 1.0.1): `scripts/task_conventions.py` normalize,
`scripts/set_agent_egress.py` (the egress list), `scripts/task_resources.toml` (the `^roboarc-ikea-` rules) and
`scripts/eai_standard/` with IKEA's live-grading 2.4 pair `scripts/eai_standard/v2_4_live/`.

```bash
python3 ikea/scripts/ikea_mujoco/mk_tasks.py --check            # privileged dirs = a fresh render
python3 ikea/scripts/ikea_mujoco/mk_standard_tasks.py --check   # standard dirs = a fresh render
bash ikea/ci/check.sh [--image-tests]                           # both, plus the unit tests in the image
ikea/images/ikea-mujoco/build.sh [--assets <dir>]               # roboarc-ikea and roboarc-ikea-sim
```

Pinned bytes: the task dirs (`../tasks/roboarc-ikea-*`), `images/ikea-mujoco/scenes/`, `scripts/ikea_mujoco/shared/` and `scripts/ikea_mujoco/standard/`
are copied into tasks or images; a change there is a new task or image version. The generators render from the inputs
above, so editing an input that a task text quotes (an evidence record, a reference) changes task bytes: run both
`--check` after every change.
