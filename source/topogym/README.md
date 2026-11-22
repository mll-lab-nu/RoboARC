# topogym: the simulator behind RoboARC's puzzles

The Python package the release image installs at `/opt/topogym/topogym` (`PYTHONPATH=/opt/topogym`). It started as
TopoGym on Genesis physics; the released families run on MuJoCo 3.14.0 (`world/mujoco/`) and render through Nyx. The
graded entry point is the pinned tooling, `scripts/roboarc/shared/rcb_topogym.py` (`Session`), which builds a world from
a frozen instance; the package name, `/opt/topogym` and the `TOPOGYM_*` variables are part of the released runtime and
keep their names.

| folder | content |
|---|---|
| `simulator/` | `make` / `env_list`, `SimulatorConfig`, the managed `Env` and its interaction handles, history, `frozen_binding.py` (the installed runtime every task binding names) |
| `world/` | the worlds: `mujoco/` (the MuJoCo core: scenes, contact policy, profiles, rope and cloth backends, energy books, guards; [docs/simulator/MUJOCO_CORE.md](../../docs/simulator/MUJOCO_CORE.md)), `physics/` (birth-state codecs and helpers the MuJoCo core reads; the Genesis world, its backends and visuals were removed in 1.1.0) |
| `tasks/` | the 14 families (`<family>/`: task, variations, success predicates, redesigned tiers), placement, goal descriptions |
| `assets/` | task geometry (`geometry/`), frozen resources and variation records (`resources/`), robot loading |
| `robot/`, `sensors/`, `solvers/` | robot descriptions and control modes; observations and cameras; kinematics and collision queries |
| `config/`, `evaluation/`, `recording/`, `utils/` | shared configuration profiles; measurements behind success checks; frame and transition recording; helpers |

Every file outside `*.md` is hashed into SOURCE.json (443 files in release 1.0.0): changing one changes every task's
runtime binding (`source/README.md`).

## How the references were made

The reference solutions in `scripts/roboarc/oracles/` were recorded by planners imported as `topogym.planning` (the
frozen instances and texts: `scripts/roboarc/release.json`; per-family notes: `docs/simulator/families/`). Tools for
authoring new tasks and reference solutions are not part of the public release.
