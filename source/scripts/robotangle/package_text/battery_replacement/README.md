# battery_replacement: task texts of the family's new base

battery_replacement (`docs/rcb/families/battery_replacement.md`) is a new family with no released base. Its first setting,
`battery_replacement_xbox_aa` (experimental, no tier), gets its own base, `battery-replacement-01-i01` (the
`<family>-<NN>-i01` names of 2026-10-17), made from the released ALOHA base `interlocked-claws-easy-i00` as its template
(`scripts/robotangle/release_rebuild.py` new_bases). These directories hold the new base's `task.toml`, `README.md` and
`instruction.md` for both modes; every other file of a package (the Dockerfiles, the compose files, `tests/test.sh`) is the
template's, and the rebuild replaces the template's family by battery_replacement where the grader and the sidecar start.

- Not on the release list until the user says so. A temporary manifest makes the base, for local generation and for
  Harbor under its name, on a test-only commit of a temporary branch that is never merged:

  ```json
  "battery_replacement": {"tiers": [], "note": "battery_replacement_v1 (claws_v6 + the parts' own MJCF contacts): no overrides",
    "new_bases": {"battery-replacement-01-i01": {"template": "interlocked-claws-easy-i00",
      "variation": "battery_replacement_xbox_aa", "package_text": "scripts/robotangle/package_text/battery_replacement",
      "design": "a game controller's two AA cells swapped: slide the cover open, pry each old cell up with the pry stick and pinch it out (force closure only), drop it into the recycling box, seat two new cells with the right polarity, latch the cover"}}}
  ```

- Each file is Cast Trinity's text of the same template (`scripts/robotangle/package_text/interlocked_claws/
  interlocked-claws-13-i01-<mode>/`, the current wording of an ALOHA new base) with every task line replaced, each value
  from the battery_replacement instance (birth 764192d9a7e49a0f: rule C with its 60 mm per cell and episode, the controller's outer body in collision) or from what the tooling generates for it
  (`runs/controller_battery_design_20261014/port/mk_package_text.py <instance.json> <step_s>`, every replacement asserted):
  - the names (`topogym-battery-replacement-01-i01-<mode>`, the family, the variation, the Session and replay commands);
  - the task sentence and the success requirements: the instance's public goal (`goal_description.describe_goal`,
    `BatteryReplacementTask.goal_words`; the sideways tolerance 1.2 mm, the user's choice of 2026-10-17; since 2026-10-19
    the cover's line says what latches it, the 0.5 mm along the slide where the plunger sits in its groove, and the
    plunger's own condition as checked, and the stick may end on the table or in a hand: the metric review
    `runs/robotangle_mujoco/metric_review_20261019/battery_zh.md`, findings 2 and 5);
  - a short scene section after the task sentence (the cover's slide and latch, slid back "until it stops (it latches
    there)" since 2026-10-19, the review's finding 3; the old cells 4 mm below the rim, how the pry stick levers a cell;
    REGISTRATION_PROPOSAL.md section 5);
  - the grader details: `scripts/topogym_rules.py` for battery_replacement (a rigid family: the contact review rules; the
    six grasp regions, the stick's around its handle), the goal-square and separator lines dropped (no placement goal);
  - the grasp line: every part graspable (cover, new_1, new_2, old_A, old_B, stick), the tool frame within 6 cm, the
    grippers start empty and nearly closed; and one sentence on the cells' pinch rules (rule C, the user's choice of
    2026-10-18): a pinch's central angle defined, an old cell plucked at 90 deg or more, the holds below 150 deg of a cell
    sharing 60 mm per episode, counted from where the cell leaves the compartment (the user's choice of 2026-10-19, the
    review's finding 7), a new cell taken at 150 deg or more (`WeakHoldGraspTarget` / `ForceClosureGraspTarget`);
  - the objects line: the seven entities; privileged agents read the spring contacts and the plunger from the world state;
  - the implementation reference: `BatteryReplacementTask._slot` and `evaluate_state`;
  - the per-step wall clock, in both modes (the standard "Time per call" line keeps its wording): 0.04 s, MuJoCo, from
    Harbor's run of the -i01 reference under this base (2026-10-17, a lab RTX 4090 host, AMD Ryzen 9 7950X, 16 cores / 32 threads,
    load average 34.6 when the run started; image `eai-robotangle:0.3.0-dryrun-battery-replacement-1`, test-only commit
    780d3a669e): the standard live episode 255.6 s for 6 453 control steps (0.040 s; the trial's 625.7 s also holds the
    collect hook's fresh-process replay, 293.9 s, and the startup); that run's privileged grader, two replays in 636 s
    (0.049 s). Rule C's -i01 through Harbor on the same host at load average 10.7 (test-only commit 2186292da1): the
    standard live episode 137.0 s for 7 029 control steps (0.0195 s); the value stays 0.04 s, the loaded host's.
    Aside: recording the reference on the same host at load average 22-28 took 0.025 s per step (run-oracle
    164 s; the two strict replays 0.023 s).
