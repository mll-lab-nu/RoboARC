# lock_picking: the chain lock's task texts

The chain lock replaced the released lock_picking tiers under their variation ids on 2026-10-10 (the user's decision),
so the released -i00 texts (`scripts/robotangle/packages/lock-picking-<tier>-i00-<mode>/`) describe a lock that is no
longer registered. These directories hold the chain lock's own `task.toml`, `README.md` and `instruction.md` for each
released base and mode. `release_rebuild.py` lays them over the released texts when the family's `mujoco_release.json`
entry names this directory; every other file (the Dockerfiles, the compose files, `tests/test.sh`) stays the released
one, and the rebuild writes `runtime_binding.upstream.json`.

- Each file is the released -i00 file with only the chain lock's lines replaced: the task sentence and description, the
  success rule's terminal revision (`terminal_v6_chain_lock`), the rule lines (plate faces and spacing, shaft length,
  the two tabs and the two slot depths, the rotor stops), the plate count, the scene object list, the implementation
  module (`topogym.tasks.lock_picking.chain`) with its `evaluate_state` excerpt, the variation id in `task.toml`'s
  instance comment, and the per-step wall clock (the 2026-10-09 Genesis measurement, as in every family's texts). The
  base image, the tooling version, the protocol name and every other line are the released ones, so `mk_tasks.py`
  rewrites them as for every family.
- Checks:
  - the rule lines are those `scripts/topogym_rules.py` generates from the chain lock's instances;
  - the `evaluate_state` excerpt is the current source, line for line (81 lines);
  - laid over the released texts, they give byte for byte the texts Harbor graded on 2026-10-10 (`mk_tasks.py --dev`
    render, both modes, oracle 1 and nop 0 on every tier, before the Robotiq's springref 2.62;
    `runs/test_results/lock_chain_harbor_mujoco_20261010/`, local).
- The references are in `scripts/robotangle/oracles/lock-picking-<tier>-i01/`. The family's `mujoco_release.json`
  entry is not listed yet: entries are listed once the family's acceptance rounds pass. Then:

  ```json
  "lock_picking": {"tiers": ["very_easy", "easier", "easy", "medium", "hard"],
                   "package_text": "scripts/robotangle/package_text/lock_picking",
                   "note": "lock_chain_v2: the chain lock under the released variation ids; its own task texts"}
  ```

- Design, oracle and evidence: `docs/develop/tasks/lock_picking_v2_draft_zh.md`, section 12.
