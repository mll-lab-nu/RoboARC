# topogym: the simulator behind RoboARC's puzzles

The Python package the release image installs at `/opt/topogym/topogym` (`PYTHONPATH=/opt/topogym`). It started as
TopoGym on Genesis physics; the released families run on MuJoCo 3.14.0 (`world/mujoco/`) and render through Nyx. The
graded entry point is the pinned tooling, `scripts/roboarc/shared/rcb_topogym.py` (`Session`), which builds a world from
a frozen instance; the package name, `/opt/topogym` and the `TOPOGYM_*` variables are part of the released runtime and
keep their names.

| folder | content |
|---|---|
| `simulator/` | `make` / `env_list`, `SimulatorConfig`, the managed `Env` and its interaction handles, history, `frozen_binding.py` (the installed runtime every task binding names) |
| `world/` | the worlds: `mujoco/` (the MuJoCo core: scenes, contact policy, profiles, rope and cloth backends, energy books, guards; [docs/simulator/MUJOCO_CORE.md](../../docs/simulator/MUJOCO_CORE.md)), `physics/` and `genesis_world.py` (the Genesis paths: render host, birth states), `visuals/` |
| `tasks/` | the 14 families (`<family>/`: task, variations, success predicates, redesigned tiers), placement, goal descriptions |
| `assets/` | task geometry (`geometry/`), frozen resources and variation records (`resources/`), robot loading |
| `robot/`, `sensors/`, `solvers/` | robot descriptions and control modes; observations and cameras; kinematics and collision queries |
| `config/`, `evaluation/`, `recording/`, `utils/` | shared configuration profiles; measurements behind success checks; frame and transition recording; helpers |
| `planning/` | the oracle sources (not in the image; below) |

Every file outside `planning/` and outside `*.md` is hashed into SOURCE.json (443 files in release 1.0.0): changing one
changes every task's runtime binding (`source/README.md`).

## planning/: how the references were made

The reference solutions in `scripts/roboarc/oracles/` were recorded by these planners, run from the tools in
`tools/references/` and `tools/redesign/` (the frozen instances and texts: `scripts/roboarc/release.json`; per-family
notes: `docs/simulator/families/`).

| family | released bases | planning | tools |
|---|---|---|---|
| blanket_flatten | 01-09 | `blanket_flatten/` (routes, robot oracle, hard port, `redesign/`; run-oracle entry `public_session.py`) | the Genesis-era routes: `tools/references/oracles_genesis/` |
| blanket_fold | 01-05 | `blanket_fold/redesign/oracle.py` (through `blanket_flatten/`'s route runner) | |
| buzz_wire_3d | 01-10 | `buzz_wire_3d/` (`redesign/` for the rd tiers) | `tools/references/buzz_wire_3d/` (`make_ref.py`, `ported/`), `tools/redesign/buzz_wire_3d/` |
| cast_coil | 01-08 | `cast_coil/` | `tools/references/cast_coil/` |
| cast_keyhole | 01-06 | `cast_keyhole/` | `tools/references/cast_keyhole/` |
| fiber_bundling | 01-10 | `fiber_bundling/` (`redesign/`) | `tools/references/fiber_bundling/` (`replan.py`) |
| hedgehog_in_cage | 01-08 | `hedgehog_in_cage/` | `tools/references/hedgehog_in_cage/` |
| horseshoe_ring | 01-05 | `horseshoe_ring/`, `harbor_bridge/` (callbacks, `reference_data/horseshoe_ring/`) | `tools/references/horseshoe_ring/` |
| interlocked_claws | 01-12 | `interlocked_claws/` (`redesign/`) | `tools/references/interlocked_claws/`, `tools/redesign/interlocked_claws/` |
| knot_untying | 8 bases, 01-09 | `knot_untying/` (`redesign/`) | `tools/redesign/knot_untying/` |
| lock_picking | 01-05 | `lock_picking/` (`public_session.py`) | `tools/references/oracle_runners/lock_picking_session_oracle.py` |
| orbits | 01-02 | `orbits/` (`native_session.py`) | `tools/references/orbits/`, `tools/references/oracle_runners/orbits_session_oracle.py` |
| rope_threading | 01-08 | `rope_threading/` (`redesign/`) | `tools/redesign/rope_threading/` |
| slotted_post_ring | 01-05 | `slotted_post_ring/` (`public_session.py`, `chain/`) | `tools/references/slotted_post_ring/`, `tools/references/oracle_runners/slotted_*` |

Shared by several planners: `base.py`, `oracle_ik.py`, `oracle_path.py`, `standard_rate.py` (the standard service's
joint-speed limit applied to a route), `motions/`, `rigid_reference/`, `grasp_observation.py`, `solvability.py`,
`instance_optimizer/`, `harbor_bridge/` (the Genesis-era Harbor oracle bridge; its callbacks and reference data are still
used by the horseshoe_ring and cast_keyhole tools). Some modules are from the Genesis era (audits, below-easy tiers,
policies over Genesis worlds) and are kept as the record of how the earlier references were made.
