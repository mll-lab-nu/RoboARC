# Simulator documentation

Design notes of the simulator behind RoboARC's puzzle families (`source/topogym/`, MuJoCo 3.14.0 physics with Nyx
frames) and of the tools that made and accepted its references. The task protocol itself is [`docs/PROTOCOL.md`](../PROTOCOL.md).

| document | what it covers |
|---|---|
| [MUJOCO_CORE.md](MUJOCO_CORE.md) | the MuJoCo core (`source/topogym/world/mujoco/`): scenes, contact policy, profiles, energy books, guards, determinism, porting a family |
| [RENDER_BRIDGE.md](RENDER_BRIDGE.md) | how frames are made: the Nyx render host and direct path (`scripts/roboarc/shared/render_bridge.py`, `nyx_direct.py`), rendering off |
| [VBD_CLOTH.md](VBD_CLOTH.md) | the VBD cloth backend in the image (newton, warp-lang, the precompiled Warp kernels, CPU/GPU resources) |
| [MUJOCO_EPA_PATCH.md](MUJOCO_EPA_PATCH.md) | the 14-byte EPA horizon fix of MuJoCo 3.14.0's `libmujoco.so` |
| [STRESS_VALIDATION.md](STRESS_VALIDATION.md), [STRESS_HOWTO.md](STRESS_HOWTO.md) | the acceptance criteria of the physics (breakdown definitions, rounds) and how to run a round (`tools/stress/`) |
| [TOPOGYM_PROTOCOL.md](TOPOGYM_PROTOCOL.md) | the record of the TopoGym (Genesis-era) task protocol whose grading-replay, tolerance and validity rules the tasks cite |
| [redesign/](redesign/) | design records of the redesigned tiers (buzz_wire_3d, fiber_bundling, interlocked_claws, knot_untying, rope_threading) |
| `families/` | per-family development notes |

Older dated records (the migration status, the speed-up record, the blanket initial-state decision) are in
[`docs/history/`](../history/). Paths cited by the frozen task files are mapped in [`docs/rcb/README.md`](../rcb/README.md).
