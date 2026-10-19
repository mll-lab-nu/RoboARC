# IKEA MuJoCo (part of RoboARC 1.0.0)

Twelve IKEA furniture-assembly tasks in plain MuJoCo 3.14.0 on the CPU (float64). Two robot arms assemble each piece
with physical grasps (finger friction, no assist). Every task ships in both modes, so there are 24 task directories:
`tasks/roboarc-ikea-<scene>-i00-{privileged,standard}` (until 1.0.0: `ikea-<scene>-i00-<mode>`).

Scene authors: Xiaoyu's team (Xiaoyu, Zhen Wu). The tasks, references and image follow RoboARC's rules.

| path | what |
|---|---|
| `tasks/` | the 24 Harbor task directories. They are generated: never edit them by hand |
| `scripts/ikea_mujoco/` | the generators `mk_tasks.py` (privileged) and `mk_standard_tasks.py` (standard), and their inputs: `tasks.json` (texts), `bindings.json`, `assets.json` (scene pins), `evidence.json` / `standard_evidence.json`, `shared/` (the tooling `rcb_ikea.py` and the grader), `standard/` (the sim service, the Nyx render process), `oracles/` (the references and how they were made), `tails/`, `stress/`, `tests/`, `REVIEW.md` |
| `images/ikea-mujoco/` | the scenes (`scenes/<task>/`), the base image recipe (`Dockerfile`), the sidecar's (`sim/`), the EPA patch, `build.sh` |
| `scripts/task_conventions.py`, `scripts/set_agent_egress.py`, `scripts/eai_standard/`, `ci_checks/topogym_tooling_versions.json` | the shared task layer these tasks were rendered with (the development branch's versions, which differ from the RoboARC puzzles' layer in `../scripts/`) |
| `docs/benchmarks/ikea_mujoco.md` | the benchmark page: tasks, rules, timing, the exceptions |
| `ci/check.sh` | the checks |

```bash
python3 ikea/scripts/ikea_mujoco/mk_tasks.py --check            # privileged dirs = a fresh render
python3 ikea/scripts/ikea_mujoco/mk_standard_tasks.py --check   # standard dirs = a fresh render
bash ikea/ci/check.sh --image-tests                             # both, plus the unit tests in the image
ikea/images/ikea-mujoco/build.sh [--assets <dir>]               # roboarc-ikea:1.0.0 and roboarc-ikea-sim:1.0.0
harbor run -p ikea/tasks/roboarc-ikea-svallet-i00-standard -a oracle    # one task
```

Images: `ghcr.io/embodied-first-exam/roboarc-ikea:1.0.0` (privileged tasks, and the physics of the standard ones) and
`ghcr.io/embodied-first-exam/roboarc-ikea-sim:1.0.0` (the standard tasks' `sim` sidecar: plus Nyx, EGL and ffmpeg). They
are the shared repository's `eai-ikea-mujoco:0.3.2` and `eai-ikea-mujoco-sim:0.4.2`, with the same image ids
(5d5090ea1910, 050cd45d70f6).

The tasks were renamed into the RoboARC namespace for 1.0.0 (user, 2026-10-19), by the generators: only `task.toml`'s
name, the README's title and twin rows, and three docstrings in the sim service's files changed. The evidence records
in `scripts/ikea_mujoco/evidence.json` keep the names the runs had. The task layer is the development branch's: there is no resource-monitor sidecar and no DIGEST file, and
the RoboARC invariants in `../ci_checks/` do not run on these tasks.
