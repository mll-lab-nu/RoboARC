# clock_assembling: task texts of the family's new base

clock_assembling (`docs/rcb/families/clock_assembling.md`) is a new family with no released base. Its first setting,
`clock_assembling_gear_train` (experimental, no tier), gets its own base, `clock-assembling-01-i01` (the `<family>-<NN>-i01`
names of 2026-10-17), made from the
released ALOHA base `interlocked-claws-easy-i00` as its template (`scripts/robotangle/release_rebuild.py` new_bases). These
directories hold the new base's `task.toml`, `README.md` and `instruction.md` for both modes; every other file of a package
(the Dockerfiles, the compose files, `tests/test.sh`) is the template's, and the rebuild writes `runtime_binding.upstream.json`.
The template is another family's base: the rebuild replaces its family by clock_assembling where the grader and the
sidecar start (the privileged `tests/test.sh`'s `grade.py --family`, the standard compose's `RCB_TASK`) and refuses the
base while any other text still names interlocked_claws (`release_rebuild.py`, from jianwen/release-rebuild-family
2ac82650c5, in dev since 1e5c8f5443; without it Harbor grades and serves interlocked_claws).

- Not on the release list until the user says so. A temporary manifest makes the base, for local generation and for
  Harbor under its name, on a test-only commit of a temporary branch that is never merged: a clock_assembling family entry
  with `"tiers": []` and this new base, then `release_rebuild.py <commit> --version <v> --families clock_assembling`:

  ```json
  "clock_assembling": {"tiers": [], "note": "clock_v1 (claws_v6 + the movement's own MJCF contacts): no overrides",
    "new_bases": {"clock-assembling-01-i01": {"template": "interlocked-claws-easy-i00",
      "variation": "clock_assembling_gear_train", "package_text": "scripts/robotangle/package_text/clock_assembling",
      "design": "a watch movement's gear train enlarged 8 times: the barrel, center, third and fourth wheel onto their pivots (one order only), the train bridge over them with every pivot in its hole, two screws turned two turns, then the barrel wound through its ratchet and let go: the mainspring runs the train (-450)"}}}
  ```

- Each file is Cast Trinity's text of the same template (`scripts/robotangle/package_text/interlocked_claws/
  interlocked-claws-13-i01-<mode>/`, the current wording of an ALOHA new base) with every task line replaced,
  each value from the clock_assembling instance (birth e479193c42dee4e0; regenerated from it 2026-10-20 after the
  rule changes of 2026-10-19, terminal revision clock_gear_train_v3: the success requirements' wording and the success
  rule's revision changed, nothing else; again from the instance of the model revision v3, birth ad4ee5041eeb19de: every
  file byte for byte the same) or from what the tooling generates for it
  (`runs/watch_assembly_design_20261014/scripts/port/mk_package_text.py <instance.json> 0.04`):
  - the names (`topogym-clock-assembling-01-i01-<mode>`, the family, the variation, the Session and replay
    commands);
  - the task sentence and the success requirements: the instance's public goal (`goal_description.describe_goal`,
    `ClockAssemblingTask.goal_words`);
  - the grader details: `scripts/topogym_rules.py` for clock_assembling (rigid contact review, the rules' tick with the
    click and the run test, the nine grasp regions named by link), the goal-square and separator lines dropped (no
    placement goal);
  - the grasp line: every part graspable (barrel, barrel_arbor, bridge, center, fourth, probe, screw1, screw2, third), the
    tool frame within 15 cm, the grippers start empty;
  - the objects line: one entity, `clock`, its parts as links; privileged agents read a part's pose from the world state;
  - the implementation reference: `ClockAssemblingTask.evaluate_state`;
  - the per-step wall clock, in both modes (the standard "Time per call" line keeps its wording): 0.04 s, MuJoCo, from
    Harbor's run of the -i01 reference under this base on the trial host (2026-10-16, a lab RTX 4090 host, AMD Ryzen 9 7950X,
    16 cores / 32 threads, load average 24-27 an hour before; image `eai-robotangle:0.3.0-dryrun-clock-assembling-2`,
    test-only commit ccd1b0b68d): the standard live episode 350.4 s for 8 750 control steps (0.040 s); that run's
    privileged grader, two replays in 781.6 s (0.045 s). The v2 reference (8 135 control steps) and these texts through
    Harbor on the same host (2026-10-17, image `eai-robotangle:0.3.0-dryrun-clock-assembling-v2`, test-only commit
    03f8843d07): the standard live episode 436.2 s (0.054 s), the privileged grader's two replays 940.0 s (0.058 s).
    Under the stress observer on the same host both versions take 0.061 s per step, so the v2 model costs the same per
    step and the slower Harbor run most likely comes from the host's load (load average 29-46 sampled during it); the
    value stays 0.04 s. The re-recorded reference (8 171 control steps, the bridge pinched across bridge_bar2) through
    Harbor on the same host (2026-10-18, image `eai-robotangle:0.3.0-dryrun-clock-assembling-v3`, test-only commit
    003470adb9): the standard live episode 447.2 s (0.055 s), the privileged grader's replays 908.2 s (0.056 s); the
    value stays 0.04 s. With the probe's hard contacts (image `eai-robotangle:0.3.0-dryrun-clock-assembling-v4`,
    test-only commit 7a88493802): 224.5 s (0.027 s) and 474.2 s (0.029 s), a quieter host; the value stays 0.04 s. The v3
    reference (8 047 control steps; the realism change, the rule changes, the bridge solid to the wheels) through Harbor
    on this host (2026-10-20, image `eai-robotangle:0.3.0-dryrun-clock-assembling-v5`, the branch's own commit
    97a4e50f36): the standard live episode 319.3 s (0.040 s), the privileged grader's two replays 668.8 s (0.042 s); the
    value stays 0.04 s.
