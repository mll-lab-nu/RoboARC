# cast_coil: the v2 tiers' task texts

The v2 tiers, `cast_coil_v2_very_easy` (after the official Cast Coil puzzle), `cast_coil_v2_easier` (the lattice design s43)
and `cast_coil_v2_easy` (the lattice design s121), are variation ids next to the released cast_coil tiers, which stay
registered under their own ids. They have no released base, so each gets its own base, `cast-coil-<tier>-i02`, made
from the released base of the same tier as its template (`cast-coil-very-easy-i00`, `cast-coil-easier-i00`,
`cast-coil-easy-i00`; the user's decision, 2026-10-12; `release_rebuild.py` new_bases). These directories hold each new
base's `task.toml`, `README.md` and `instruction.md` for both modes; every other file of a package (the Dockerfiles, the
compose files, `tests/test.sh`) is the template's. Each new base's `mujoco_release.json` new_bases entry names this
directory (`"package_text": "tools/release/package_text/cast_coil"`), not the family's entry: there it would send
the family's released tiers, once listed, to look for texts of their own.

- Each file is the template's released -i00 file (`tools/release/templates/cast-coil-<tier>-i00-<mode>/`) with only
  these lines replaced:
  - the names: `topogym-cast-coil-<tier>-i00-<mode>` becomes `topogym-cast-coil-<tier>-i02-<mode>` (the task name,
    the README title, the twin);
  - the task sentence (the instruction's, the description, the README's task line and goal): the v2 instances'
    `public_goal.summary`, "Assemble the gold and silver Cast Coil pieces into their matching interlocking
    configuration, then release both." The very_easy and easier templates said "into a cube with flush outer faces"
    (v2 easier assembles a 4 x 8 x 5-cell block, v2 easy 8 x 8 x 5); the easy template's sentence is the v2 one already;
  - the grasp regions' line: the v2 grasp spheres are 25 mm (`cast_coil_v2_very_easy.grasp_assist`; the very_easy and
    easier templates say 38 mm, the easy template 25 mm);
  - the contact-penetration line: the tooling's review rule, a task contact deeper than 2 mm for 0.25 s
    (`shared/rcb_topogym.py` `CONTACT_RULE` since cade53224, 2026-10-09; the released texts' 1 mm is the rule before
    it). Its words are `topogym_rules.details_section`'s for the v2 instances, and its note "(it replaces the "at most
    1 mm / 2 mm" wording above)" stays: the failure conditions above it still state the instant limits
    (`goal_description.render_goal`), which only the Session's own text replaces (`rcb_topogym.rules_text`);
  - the privileged instruction's implementation reference: `CastCoilV2VeryEasyTask`, `CastCoilV2EasierTask`,
    `CastCoilV2EasyTask` with their modules. Their `evaluate_state` is `CastCoilTask`'s, so the excerpt is unchanged;
  - the variation id in the privileged `task.toml` (description and instance comment) and the privileged README's task
    line: `cast_coil_v2_<tier>` (the standard texts name no variation). The easy template's `difficulty = "easy"` stays;
  - the per-step wall clock, in both modes, and the standard mode's "Time per call" section that
    `topogym_rules.limited_timing` derives from it, from the -i01 references' single-core strict replays on the prism
    instances (mujoco 3.14, `runs/test_results/cast_coil_v2_prism_20261012/`, 2026-10-12/13) on a lab RTX 4090 host (AMD Ryzen 9
    7950X, 32 hardware threads), one thread per replay, the host's 1-minute load average from its start to its end:
    very_easy 0.047 s (406 s for 8 688 control steps, load 37-38), easier 0.039 s (1 087 s for 28 037, load 23-31),
    easy 0.044 s (1 445 s for 32 967, load 28-31). The templates' 0.045, 0.043 and 0.05 s were the Genesis measurement.
  - Every other line is the released one (base image, tooling version, protocol name, oracle evidence, ...), so
    `mk_tasks.py` rewrites them as for every family.
- Checks (the -i01 instances; `rcb_topogym.py describe` for very_easy and easier in the image
  `eai-robotangle:0.3.0-dev-coil-v2-1011b` built from d1b764ddc; easy's public goal from its instance and its class from
  the registry):
  - every success requirement and task failure line that `goal_description.render_goal` gives for the v2 public goal is
    in the released text word for word (terminal revision `release_settling_v2_grasp_contact_2mm`, as released);
  - every grader line is `topogym_rules.details_section`'s for the v2 instances, apart from the contact-penetration
    line's kept note;
  - the `evaluate_state` excerpt is the described predicate, line for line (24 lines).
- The references are in `scripts/roboarc/oracles/cast-coil-<tier>-i02/`, recorded on the 32-facet prism instances
  (a cylinder the robot can touch is built as a convex prism, 64820cfbf): very_easy born 22e9a1e035809358, 8 688 rows,
  final state be32137b76dbae53; easier 6432913b260aaf8c, 28 037 rows, 60441db184274e4a; easy faad5b880e67eb06, 32 967
  rows, 625e699dc0e01560. The very_easy and easier oracles chose the same actions as on the cylinders. Each: a
  fresh-process strict replay to the same final state, the stress observer CLEAN, a nop of the same length CLEAN and
  not successful. Harbor graded the cylinder references of very_easy and easier on 2026-10-12, both modes, oracle 1
  and nop 0, under the template bases' names and the released texts
  (`runs/test_results/cast_coil_v2_harbor_mujoco_20261011b/`, local); on the prisms and with these texts it runs again
  through `release_rebuild.py`.
- Listed (2026-10-13, the user's go once all four items were done): the three new bases in the cast_coil entry of
  `mujoco_release.json`. On the merge of the remote (126cf1b6c): the replays above again, Harbor under each new base's
  name (test commit fba08165c: oracle 1, nop 0 in both modes, the end states above, the packaged texts equal to these
  files), and the 1,000-episode acceptance round m126_coilv2_round PASS on all three (0 confirmed BREAKDOWN under every
  reading; the two gold x silver breakdowns of easier's cylinder-instance round do not reappear: the largest gold x
  silver contact 821 N / 1.54 mm; reports in `runs/stress_20261013/m126_replays/`).
- The tiers: `topogym/tasks/cast_coil/cast_coil_v2_*.py`; their oracle: `topogym/planning/cast_coil/official_policy.py`
  with the routes in `topogym/planning/cast_coil/routes/`; the design search:
  `docs/mujoco_oracles_optim/cast_coil_atomic_ik_and_design_rules_zh.md`, section 6.
