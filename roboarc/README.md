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
| `__main__.py` | `python -m roboarc list / info / check / replay / patch-mujoco` |
| `data/ikea_reference_hashes.json` | the recorded end and settled hashes of the IKEA references |
