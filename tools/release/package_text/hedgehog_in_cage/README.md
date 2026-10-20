# hedgehog_in_cage: task texts of the family's new bases

New designs of the family that have no released base each get their own base, made from a released base of the family as
its template (`scripts/robotangle/release_rebuild.py` new_bases). These directories hold each new base's `task.toml`,
`README.md` and `instruction.md` for both modes; every other file of a package (the Dockerfiles, the compose files,
`tests/test.sh`) is the template's.

## hedgehog-in-cage-07-i01 (Bird in Cage, `hedgehog_in_cage_owl`)

Template: `hedgehog-in-cage-very-easy-i00`. Each file is the template's released -i00 file
(`scripts/robotangle/packages/hedgehog-in-cage-very-easy-i00-<mode>/`) with only the owl scene's lines replaced:
- the names: `topogym-hedgehog-in-cage-very-easy-i00-<mode>` becomes `topogym-hedgehog-in-cage-07-i01-<mode>` (the task
  name, the README title, the twin);
- the task sentence (the instruction's, the description, the README's task line and goal): the owl instance's
  `public_goal.summary`, the owl as the target, "the cage and its mount" as the remainder, and the rod sentence
  (`HedgehogOwlTask.placement_scene_note`: the rod is a tool, where it ends up does not matter, it falling off the table
  ends the episode in failure; the user's decision, 2026-10-12);
- the success requirements' target and remainder words and the goal volume's height (258.708 mm), from the instance's
  `public_goal.success_conditions`; the goal square (266.7 mm);
- the grasp regions' line: owl 102.3 mm (its body link: body, beak, legs and tail; the wings are a separate link) and rod
  138 mm; the Grippers line's parts (owl, rod) and tool-frame gate (16.7962 cm), from the instance's
  `simulator.grasp_assist`;
- the contact review line says "deeper than 2 mm for at least 0.25 s", the tooling's rule (`shared/rcb_topogym.py`
  `CONTACT_RULE`, since cade53224); the released text says 1 mm. Cast Cage's texts say 2 mm as well;
- the scene object lists (`hedgehog_mount`, `owl_cage`, `owl`, `rod`, `goal_region`);
- the privileged instruction's implementation reference: `HedgehogOwlTask` in `topogym.tasks.hedgehog_in_cage.owl`. Its
  `evaluate_state` is the family's (the excerpt is unchanged); its `evaluate_goal_region` reads the owl's two links;
- the variation id in `task.toml` (description and instance comment): `hedgehog_in_cage_owl`; the `difficulty` line is
  dropped (the setting's difficulty is None, as window_change's);
- the per-step wall clock, in both modes (and the standard mode's "Time per call" line): 0.024 s, from the -i01
  reference's two fresh-process strict replays (174.7 s and 172.8 s for 7 224 control steps; MuJoCo 3.14, the a cloud host host,
  128 cores at a load of about 87). The released 0.056 s was the Genesis measurement.
- Every other line is the released one, so `mk_tasks.py` rewrites them as for every family (engine words, names,
  tooling version, reference solution, provenance, change log).
- Checks: the instance's `public_goal` summary, success and failure lines are in each instruction word for word; no
  word of very_easy's scene is left. Made by `runs/bird_in_cage_harbor_20261012/make_package_text.py` (not in the
  repository) from the instance made on 7faff8308 (birth e22fac98a9c3807a).
- The reference: `scripts/robotangle/oracles/hedgehog-in-cage-07-i01/` (79bd4c7af; run-oracle code
  `scripts/robotangle/references/hedgehog_in_cage/owl_code.py`). Acceptance round bird_owl_r1 on d27c622d7, PASS.
  Harbor under this base's name with these texts (release_rebuild dry run on 3d8436413, 2026-10-12): oracle 1 and nop 0
  in both modes. Listed in `mujoco_release.json`'s hedgehog_in_cage new_bases (the user's decision, 2026-10-12): template
  `hedgehog-in-cage-very-easy-i00`, the family's gripper space (native).
- Design, oracle and evidence: `docs/rcb/families/hedgehog_in_cage.md` (owl).

## hedgehog-in-cage-08-i01 (window change, `hedgehog_in_cage_window_change`)

Template: `hedgehog-in-cage-very-easy-i00`, which stays as it is (window change is a harder design of the very_easy cage,
registered as its own variation). Each file is the template's released -i00 file
(`scripts/robotangle/packages/hedgehog-in-cage-very-easy-i00-<mode>/`) with only the window change scene's lines
replaced:
- the names: `topogym-hedgehog-in-cage-very-easy-i00-<mode>` becomes `topogym-hedgehog-in-cage-08-i01-<mode>`
  (the task name, the README title, the twin);
- the goal volume's height (231.991 mm) and the goal square (240.0 mm). The task sentence, the success and failure lines
  and the grasp region (hedgehog_ball 89.3 mm, tool-frame gate 11.9304 cm) are the template's, which are also this
  instance's;
- the contact review line says "deeper than 2 mm for at least 0.25 s", the tooling's rule (`shared/rcb_topogym.py`
  `CONTACT_RULE`, since cade53224); the released text says 1 mm;
- the gripper action: `arm_gripper`, the opening width from 0 to 0.0854 m (the template: `arm_native_gripper`, the
  native drive target). The -i01 instance and reference are in the width gripper space (the oracle commands widths), so
  the new_bases entry says `"gripper_space": "width"`;
- the privileged instruction's implementation reference: `HedgehogWindowChangeTask` in
  `topogym.tasks.hedgehog_in_cage.window_change`. Its `evaluate_state` is the family's (the excerpt is unchanged);
- the variation id in `task.toml` (description and instance comment) and the privileged README's task line:
  `hedgehog_in_cage_window_change`; the `difficulty` line is dropped (the setting's difficulty is None);
- the per-step wall clock, in both modes (and the standard mode's "Time per call" line): 0.042 s, from one strict replay
  of the -i01 reference (123.5 s for 2 961 control steps; MuJoCo 3.14, a lab RTX 4090 host, a Ryzen 9 7950X with 32 threads at a
  load average of 35 to 39, two cores pinned, 2026-10-12). The released 0.056 s was the Genesis measurement.
- Every other line is the released one, so `mk_tasks.py` rewrites them as for every family (engine words, names,
  tooling version, reference solution, provenance, change log).
- Checks: the instance's `public_goal` summary and success lines are in each instruction and README word for word; no
  word of very_easy's scene is left. The values are those of the instance Harbor graded on 2026-10-11 (oracle 1 and nop
  0 in both modes, under the template's name with its released texts; image `eai-robotangle:0.3.0-dev-hedgehog-v4` from
  04e084ce6), from `rcb_topogym.py describe` in that image and `scripts/topogym_rules.py`. Made by
  `runs/hedgehog_release_texts_20261012/make_texts.py` (not in the repository).
- The reference: `scripts/robotangle/oracles/hedgehog-in-cage-08-i01/` (bb481d502; birth 2ecb7e00fc766fb8,
  2 961 rows; run-oracle code `scripts/robotangle/references/hedgehog_in_cage/window_change_code.py`). Acceptance round:
  hedgehog_v4 on 3297f29cb, PASS.
- Design, oracle and evidence: `docs/rcb/families/hedgehog_in_cage.md` (window_change) and
  `docs/develop/tasks/hedgehog_in_cage_window_change_zh.md`.

## hedgehog-in-cage-06-i01 (Cast Cage, `hedgehog_in_cage_cast_cage`)

Template: `hedgehog-in-cage-hard-i00`. Each file is the template's released -i00 file
(`scripts/robotangle/packages/hedgehog-in-cage-hard-i00-<mode>/`) with only the Cast Cage scene's lines replaced:
- the names: `topogym-hedgehog-in-cage-hard-i00-<mode>` becomes `topogym-hedgehog-in-cage-06-i01-<mode>`;
- the task sentence (the instruction's, the description, the README's task line and goal) and the first success
  requirement: the target is "the whole star, including every spike" (`HedgehogCastCageTask.placement_target_description`),
  followed by the rod sentence (`placement_scene_note`: the rod is a tool, where it ends up does not matter, it falling
  off the table ends the episode in failure; the user's decision, 2026-10-12). Its second half is the owl's word for
  word; it gives no example of using the rod, which here would give away the opening push;
- the goal volume's height (237.284 mm) and the goal square (245.3 mm);
- the grasp regions' line: hedgehog_ball (the star) 90.1 mm, and push_rod 37.1 mm around the rod's box handle
  (`topogym_rules.grasp_regions` says "enclosing the whole part" for every sphere; the rod's encloses only its handle);
  the Grippers line's parts (hedgehog_ball, push_rod) and tool-frame gate (12.0066 cm); `task.toml`'s grasp field;
- the scene object lists (`hedgehog_mount`, `hedgehog_cage`, `hedgehog_ball`, `push_rod`, `goal_region`);
- the contact review line at 2 mm and the gripper action as the opening width, as for window change (the -i01 instance
  and reference are in the width gripper space);
- the privileged instruction's implementation reference: `HedgehogCastCageTask` in
  `topogym.tasks.hedgehog_in_cage.cast_cage`. Its `evaluate_state` is the family's (the excerpt is unchanged);
- the variation id in `task.toml` (description and instance comment) and the privileged README's task line:
  `hedgehog_in_cage_cast_cage`; the `difficulty` line is dropped;
- the per-step wall clock, in both modes (and the standard mode's "Time per call" line): 0.041 s, from one strict replay
  of the -i01 reference (117.4 s for 2 864 control steps, alongside window change's on the same host and load). Its two
  strict replays of 2026-10-11 took 69 s each (0.024 s per step) at a load that was not recorded. The released 0.055 s
  was the Genesis measurement.
- Every other line is the released one, so `mk_tasks.py` rewrites them as for every family.
- Checks: the summary and success lines of the `public_goal` of an instance made with the goal sentence (make-instance
  from this tree, 2026-10-12) are in each instruction and README word for word; that instance's birth is the
  reference's, ad1fcefad2565341 (the sentence is not part of the birth). Harbor graded the instance made before the
  sentence on 2026-10-12 (oracle 1 and nop 0 in both modes, under the template's name with its released texts; image
  `eai-robotangle:0.3.0-dev-hedgehog-cc` from 94874eee3). The other values are from `rcb_topogym.py describe` in that
  image and `scripts/topogym_rules.py`; made by `runs/hedgehog_release_texts_20261012/make_texts.py`.
- The reference: `scripts/robotangle/oracles/hedgehog-in-cage-06-i01/` (94874eee3; birth ad1fcefad2565341,
  2 864 rows; run-oracle code `scripts/robotangle/references/hedgehog_in_cage/cast_cage_code.py`). Acceptance round on
  d1b764ddc, PASS.
- Design, oracle and evidence: `docs/rcb/families/hedgehog_in_cage.md` (cast_cage).
