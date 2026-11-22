# roboarc

RoboARC's raw simulator interface: `roboarc.make(task_id)` returns a Gymnasium-compatible environment over the graded
tooling (`rcb_topogym.Session` for the puzzles, `rcb_ikea.Session` for IKEA). The documentation is
[docs/API.md](../docs/API.md); examples are in [examples/](../examples/) and tests in [tests/api/](../tests/api/).

| module | what |
|---|---|
| `registry.py` | the task catalogue from the task directories (`list_tasks`, `resolve`, `TaskInfo`) |
| `core.py` | `make()` and `RoboARCEnv`, the interface both suites share |
| `puzzle.py` | the puzzles: `PuzzleEnv` over `rcb_topogym.Session` (frozen-instance staging, the standard joint-speed rule) |
| `ikea.py` | IKEA: `IkeaEnv` over `rcb_ikea.Session` (binding checks, the standard mode's camera rig and success rule) |
| `render_mujoco.py` | the quick-look MuJoCo raster (not the graded renderer) |
| `wrappers.py` | `EEPoseAction`: end-effector actions through the tooling's own IK |
| `references.py` | the reference trajectories and their recorded end hashes |
| `spaces.py` | Gymnasium spaces, or a stand-in when Gymnasium is not installed |
| `_runtime.py` | paths, the graded runtime's settings and checks |
| `vla/` | VLA policies served the openpi way: conventions, the websocket protocol, the episode runner, a stub server ([docs/VLA.md](../docs/VLA.md)) |
| `cli/` | the command line `roboarc` (below); `__main__.py` makes `python -m roboarc` the same command |
| `data/ikea_reference_hashes.json` | the recorded end and settled hashes of the IKEA references |

## The command line

`pip install -e .` installs one command, `roboarc`; `python -m roboarc` is the same command, for where the console
script is not on `PATH` (inside an image with the repository on `PYTHONPATH`). `roboarc --help` lists the subcommands,
and `roboarc COMMAND --help` is each one's documentation: what it does, its arguments with their defaults, an example.

| command | what it does | what it runs |
|---|---|---|
| `roboarc doctor` | can this machine run RoboARC; inside an image, is this the graded runtime | `scripts/harbor/host_info.py`, `vulkaninfo`, `docker info`, `images/pull_all.sh --list`; `_runtime.check_graded_runtime` |
| `roboarc images pull\|check\|build` | the pinned container images, after the third-party licences are accepted once | `images/pull_all.sh` (also `--check`), `images/pins.py archives`, the three build scripts |
| `roboarc list`, `info`, `replay` | the tasks, one task's facts, a trajectory's end state hash | `registry.py`, `core.py`, `references.py` |
| `roboarc selftest` | through Harbor, the reference solution scores 1 and doing nothing scores 0 | `scripts/task_seed.py materialize`, `scripts/harbor/clamp_cpus.py`, `scripts/harbor/pin_gpu.py`, `harbor run` |
| `roboarc eval` | an evaluation batch with a coding agent | `scripts/harbor/run_agent_batch.sh`, one process per lane |
| `roboarc results summary\|send\|receive` | a batch's summary table, and its records to and from the results dataset | `scripts/send_results.sh`, `scripts/receive_results.sh` |
| `roboarc vla run\|stub\|droid` | VLA policies | `vla/run.py`, `vla/stub_server.py`, `scripts/vla_droid.sh` |
| `roboarc patch-mujoco` | the EPA byte patch for a MuJoCo 3.14.0 wheel outside the images | `source/images/roboarc/mujoco-3.14.0-epa/bytepatch_libmujoco.py` |

Rules of the wrappers (`cli/__init__.py`):

- Nothing a subcommand runs was moved or renamed: every script keeps its path, arguments and variables, and can be
  called as before. `roboarc check` still prints the graded-runtime report that `doctor` shows inside an image.
- An option sets the script's variable of the same meaning (`roboarc eval --trials 3` sets `K=3`); every other variable
  of the caller's environment reaches the script unchanged. `roboarc eval --usage-gateway` also sets `GATEWAY_RAW=0`,
  as the evaluation runs do, when the caller has not set `GATEWAY_RAW`.
- The one thing the command line adds is the consent step of `roboarc images pull` and `build`
  (`cli/licences.py`): the third-party licences in `LICENSES/THIRD_PARTY.md` are accepted once per user, with
  `--accept-licences` or at the prompt, and recorded in `~/.config/roboarc/licences-accepted.json`.
- `doctor`, `images`, `list`, `info`, `selftest`, `eval` and `results` use the standard library only, so they run on a
  host with Python 3.11 or newer and nothing else installed.

The tests are in [tests/api/test_cli.py](../tests/api/test_cli.py): they start no container, no Harbor and no batch
runner, and run in seconds.
