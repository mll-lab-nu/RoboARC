# interlocked_claws: task texts of the family's new bases

New designs of the family that have no released base each get their own base, made from a released base of the family as
its template (`scripts/robotangle/release_rebuild.py` new_bases); one section per new base.

## interlocked-claws-11-i01 (Cast ABC, `interlocked_claws_cast_abc`)

Cast ABC (`interlocked_claws_cast_abc`, an experimental variation without a tier; `docs/rcb/families/interlocked_claws_cast_abc.md`)
has no released base. Its own base, `interlocked-claws-11-i01`, is made from the released `interlocked-claws-easy-i00`
as its template (`release_rebuild.py` new_bases). These directories hold the new base's `task.toml`, `README.md` and
`instruction.md` for both modes; every other file of a package (the Dockerfiles, the compose files, `tests/test.sh`) is
the template's, and the rebuild writes `runtime_binding.upstream.json`.

- Not on the release list (the user, 2026-10-12: like Rotor, Cast ABC is made evaluable, not released).
  `mujoco_release.json` lists no entry for it. A temporary manifest makes the base, for local generation and for Harbor
  under its name, on a test-only commit of a temporary branch that is never merged: the interlocked_claws entry with
  `"tiers": []` (the released tiers are not made again) and this new base, then
  `release_rebuild.py <commit> --version <v> --families interlocked_claws` (with Harbor):

  ```json
  "interlocked_claws": {"tiers": [], "note": "core, claws_v6: no overrides",
    "new_bases": {"interlocked-claws-11-i01": {"template": "interlocked-claws-easy-i00",
      "variation": "interlocked_claws_cast_abc", "package_text": "scripts/robotangle/package_text/interlocked_claws",
      "design": "Hanayama's Cast ABC: an AB plate and a C ring, both held from the start, separated along the official route (eight groove crossings and one flip; the C ring's width and one groove's position set so that this route is the shortest)",
      "note": "..."}}}
  ```

- The template is easy: the same goal (whole-piece separation, terminal revision `claws_separation_v1_grasp_contact_2mm`),
  the same pregrasp opening and robot, and its package (Dockerfiles, compose files, `tests/test.sh`) is the one Harbor
  graded Cast ABC with on 2026-10-11 (oracle 1, nop 0, both modes). medium's and hard's package files are the same;
  very_easy's and easier's differ (their goal is a placement).
- Each file is the template's released -i00 file (`scripts/robotangle/packages/interlocked-claws-easy-i00-<mode>/`) with only
  Cast ABC's lines replaced, each value taken from the Cast ABC instance (birth e6f9667c9f67c760) or from what the tooling
  generates for it:
  - the names: `topogym-interlocked-claws-easy-i00-<mode>` becomes `topogym-interlocked-claws-11-i01-<mode>` (the task
    name, the README title, the twin);
  - the task sentence (the instruction's, the description, the README's task line and goal): the instance's
    `public_goal.summary`, "Completely separate the C ring from the AB plate and briefly hold both still.";
  - the variation id in the privileged `task.toml` (description and instance comment) and the privileged README's task line;
  - the grasp regions (`ab_plate`, `c_ring`, 30 mm) and the parts in `task.toml`'s grasp line;
  - the scene objects in `s.objects()` order on MuJoCo (`ab_plate`, `c_ring`, `goal_region`) and the standard "Objects here";
  - the privileged instruction's implementation reference: `InterlockedClawsCastABCTask` and its module. Its
    `evaluate_state` is `InterlockedClawsTask`'s, so the excerpt is unchanged;
  - the grasping rule (privileged "- Grippers:", standard "Grasping:"): `Session.grasp_rules()` for the Cast ABC instance
    (`rcb_topogym.py` at 4a3bc8ae9, with 1cbd0283d's MuJoCo weld rule: a hold is also lost when it "has to push or pull its
    part with more than 60 N averaged over 20 ms");
  - no goal-square and no separator line (2026-10-13; until then the texts kept both, with "The AB plate is free to move
    (its speed is not checked) ...", which contradicted the requirement that both pieces are held still): the separation
    predicate checks neither (`InterlockedClawsTask.evaluate_goal_region`, `terminal.separation_candidate`). In their place
    stands the separation line `topogym_rules.claws_separation_words()` writes for every separation-only claws instance
    (what the gap is measured along), as in the released easy / medium / hard texts this tree holds;
  - `task.toml`: no difficulty line (the variation record has no tier); the oracle_evidence line reads as medium's and
    hard's ("No graded positive yet.") instead of naming easy's model trial. `mk_tasks.py` replaces it for a base with a
    reference (`oracle = "full"`);
  - the per-step wall clock, in both modes (the standard "Time per call" line that `topogym_rules.limited_timing` derives
    from it keeps its wording): 0.053 s, MuJoCo, from Harbor's verifier replays of the -i01 reference in the image
    `eai-robotangle:0.3.0-dev-cast-abc` (2026-10-11, AMD Ryzen 9 7950X, 16 cores / 32 threads, load average about 35):
    standard 316.6 s for 5 979 control steps, privileged two replays in 627.4 s. The same reference elsewhere: 0.039 s in
    that Harbor standard run's live episode (235.1 s); 0.033 s on an AMD EPYC 9354 host (128 threads, load average
    60-67; two strict replays in 396.9 s, one run 197.0 s); `describe`'s three hold steps 0.053 s (dev image, the 7950X
    host, load about 36). The released 0.075 s is easy's Genesis measurement.
  - Every other line is the released one (base image, tooling version, protocol name, oracle evidence wording, ...), so
    `mk_tasks.py` rewrites it as for every family.
- Checks (2026-10-12, at 4a3bc8ae9; `describe`'s fields computed in-process for the instance, since `describe` itself
  needs the release image's runtime identity):
  - against `topogym_mk_task.instruction()` for the Cast ABC instance the privileged instruction differs in two lines:
    the separator (above); the tool-frame line, which keeps the released axes (finger-closing axis local
    [0.0, 1.0, 0.004]; MuJoCo's robot spec gives [-0.0, 1.0, -0.001]). The standard grader details differ in the
    separator line. The contact-penetration line follows the tooling since 2026-10-12 ("deeper than 2 mm for at least
    0.25 s": CONTACT_RULE sustained_m 0.002 since cade53224; it first kept the released "1 mm");
  - the grasping line is `grasp_rules()` word for word in both modes; the `evaluate_state` excerpt is the predicate the
    instance runs;
  - no word of easy's scene remains (claw piece, piece_0 / piece_1, InterlockedClawsEasyTask, interlocked_claws_easy,
    314.5 mm, 0.075 s, easy's model trial, a difficulty line);
  - made into the new base by `release_rebuild.write_release` under the temporary manifest (a scratch copy of the tree, the
    instance made in the release image) and rendered by `mk_tasks.py`: 2 task dirs rendered, `mk_tasks.py --check` passes;
    the rendered tasks are named `robotangle-interlocked-claws-11-i01-<mode>`, keep no word of easy's scene, carry
    no difficulty line and `oracle = "full"`, state the 60 N rule in the grasping line, and their change log is the one
    new-task line with the design above (it names the template, `robotangle-interlocked-claws-easy-i00-<mode>`).
- The reference: `scripts/robotangle/oracles/interlocked-claws-11-i01/` (6d7ee21e4; 5 979 rows, final state
  4701f99e7860e8c9). Harbor graded it on 2026-10-11 under the template's name with an earlier version of these texts
  (before the 60 N rule entered `grasp_rules()`), and on 2026-10-12 under the new base's name with these texts: test-only
  commit 21b3e083d (dev 62d8d0d5c plus the entry above), `release_rebuild.py --families interlocked_claws` PASS, oracle 1
  and nop 0 in both modes (`docs/rcb/families/interlocked_claws_cast_abc.md`).

## interlocked-claws-12-i01 (Rotor, `interlocked_claws_rotor`)

Hanayama's Cast Rotor (`interlocked_claws_rotor`, `very_hard` in its variation record;
`docs/rcb/families/interlocked_claws_rotor.md`): two identical interlocked rotor pieces, both held from the start, separated
along the official 13-move route. It has no released base either; its own base, `interlocked-claws-12-i01`, is made from
the released `interlocked-claws-easy-i00` as its template, as Cast ABC's is.

- Not on the release list (the user, 2026-10-11: Rotor is made evaluable first; its release is a separate decision).
  `mujoco_release.json` lists no entry for it. A temporary manifest makes the base, for local generation and for Harbor
  under its name, on a test-only commit of a temporary branch that is never merged: the interlocked_claws entry with
  `"tiers": []` and this new base (its `tier` is the variation record's difficulty), then
  `release_rebuild.py <commit> --version <v> --families interlocked_claws` (with Harbor):

  ```json
  "interlocked_claws": {"tiers": [], "note": "core, claws_v6: no overrides",
    "new_bases": {"interlocked-claws-12-i01": {"template": "interlocked-claws-easy-i00",
      "variation": "interlocked_claws_rotor", "tier": "very_hard",
      "package_text": "scripts/robotangle/package_text/interlocked_claws",
      "design": "Hanayama's Cast Rotor: two identical interlocked rotor pieces (100 mm outer radius, 0.5 mm clearance), both held from the start, separated along the official 13-move route, the shortest one",
      "note": "..."}}}
  ```

- The template is easy, as for Cast ABC: the same goal (whole-piece separation, terminal revision
  `claws_separation_v1_grasp_contact_2mm`), the same pregrasp opening and robot, and its package is the one Harbor graded
  Rotor with on 2026-10-12 (below).
- Each file is the template's released -i00 file (`scripts/robotangle/packages/interlocked-claws-easy-i00-<mode>/`) with only
  Rotor's lines replaced, each value taken from the Rotor instance (birth 28cecf9c01aadf5e) or from what the tooling
  generates for it:
  - the names: `topogym-interlocked-claws-easy-i00-<mode>` becomes `topogym-interlocked-claws-12-i01-<mode>` (the task
    name, the README title, the twin);
  - the task sentence (the instruction's, the description, the README's task line and goal): the instance's
    `public_goal.summary`, "Completely separate the two interlocked pieces and briefly hold both still.";
  - the variation id in the privileged `task.toml` (description and instance comment) and the privileged README's task line;
  - the grasp regions, `piece_1` and `piece_0`, 115 mm each, in the order of `simulator.grasp_assist.targets`: each is
    centred on its piece and covers it whole (a horn tip, the farthest point, is 111.3 mm from the centre). The parts keep
    easy's names, so `task.toml`'s grasp line ("the grippers start closed on piece_0, piece_1") stays as released;
  - the scene objects in `s.objects()` order on MuJoCo (`goal_region`, `piece_0`, `piece_1`) and the standard "Objects here";
  - the privileged instruction's implementation reference: `InterlockedClawsRotorTask` and its module. Its `evaluate_state`
    is `InterlockedClawsTask`'s, so the excerpt is unchanged;
  - the grasping rule (privileged "- Grippers:", standard "Grasping:"): `Session.grasp_rules()` for the Rotor instance
    (`rcb_topogym.py` at 7f5516d02): the tool frame within 15 cm of the grasp region's centre (the task's
    `grasp_max_tcp_distance_m`; the reference regrasps 5-10 cm from the opening grasps) and 1cbd0283d's 60 N rule;
  - no goal-square and no separator line (2026-10-13; until then the texts kept both, with "The other piece (piece_1) is
    free to move (its speed is not checked) ...", which contradicted the requirement that both pieces are held still): the
    separation predicate checks neither (`InterlockedClawsTask.evaluate_goal_region`, `terminal.separation_candidate`). In
    their place stands the separation line `topogym_rules.claws_separation_words()` writes for every separation-only claws
    instance (what the gap is measured along);
  - the contact review line says "deeper than 2 mm for at least 0.25 s", the tooling's rule (`shared/rcb_topogym.py`
    `CONTACT_RULE`, since cade53224); the released text says 1 mm;
  - `task.toml`: `difficulty = "very_hard"`, the variation record's difficulty (the released TopoGym tasks use very_easy
    to hard); the oracle_evidence line reads as medium's and hard's ("No graded positive yet.") instead of naming easy's model
    trial. `mk_tasks.py` replaces it for a base with a reference (`oracle = "full"`);
  - the per-step wall clock, in both modes (the standard "Time per call" line keeps its wording): 0.15 s, MuJoCo, from
    Harbor's runs of the -i01 reference in the image `eai-robotangle:0.3.0-dev-rotor` (2026-10-12, AMD Ryzen 9 7950X,
    16 cores / 32 threads, load average about 30): the standard live episode 3170.6 s for 21 120 control steps (0.150 s),
    its fresh replay 3624.8 s (0.172 s), the privileged two replays 6432.7 s (0.152 s); `describe`'s three hold steps
    0.149 s. The reference's recording on another host (a cloud host.ai, under heavy load) took 0.13 s a step; the released
    0.075 s is easy's Genesis measurement.
  - Every other line is the released one, so `mk_tasks.py` rewrites it as for every family.
- Checks (2026-10-12, at 7f5516d02; `describe`'s fields computed in-process for the instance, in the image
  `eai-robotangle:0.3.0-dev-rotor` with this tree's code):
  - against `topogym_mk_task.instruction()` for the Rotor instance the privileged instruction differs in three lines:
    the separator (above); the contact-penetration line, which keeps its released wording ("it replaces the "at most 1 mm
    / 2 mm" wording above") with the 2 mm rule; the tool-frame line, which keeps the released axes (finger-closing axis
    local [0.0, 1.0, 0.004]; MuJoCo's robot spec gives [-0.0, 1.0, -0.001]). The standard grader details differ in the
    same two rule lines;
  - the grasping line is `grasp_rules()` word for word in both modes; the `evaluate_state` excerpt is the predicate the
    instance runs;
  - no word of easy's scene remains (claw piece, InterlockedClawsEasyTask, interlocked_claws_easy, 314.5 mm, 0.075 s,
    radius 30 mm, within 6 cm, easy's model trial and difficulty, the 1 mm review rule);
  - made into the new base by `release_rebuild.write_release` under the temporary manifest (a scratch copy of the tree; the
    instance made in the release image of d27c622d7, whose `topogym/` differs from 7f5516d02's only in the placement
    scene note of the owl, which Rotor does not use) and rendered by `mk_tasks.py`: 2 task dirs rendered,
    `mk_tasks.py --check` passes; the rendered tasks are named `robotangle-interlocked-claws-12-i01-<mode>`, keep no
    word of easy's scene, carry `difficulty = "very_hard"` and `oracle = "full"`, state the 15 cm gate and the 60 N rule
    in the grasping line and the 2 mm review rule, and their change log is the one new-task line with the design above.
- The reference: `scripts/robotangle/oracles/interlocked-claws-12-i01/` (0e35ed038; 21 116 rows, final state
  87c1e785d178acfc). Harbor graded it on 2026-10-12 under the template's name (image `eai-robotangle:0.3.0-dev-rotor` from
  d27c622d7, rendering off): oracle 1 and nop 0 in both modes. The privileged oracle's two replays end in
  87c1e785d178acfc, the state of the recording on the other host. The standard oracle's live episode and replay both end
  in 5671cc01e7914e63, four steps longer: the standard service splits each of the reference's four rows that move a
  right-arm joint 0.055-0.070 rad (over its 0.04 rad a step) into two steps. Those texts were an earlier version of these:
  the review rule said 1 mm, the objects were in easy's order, a step took "about 0.13 s", and the grasping line had no
  60 N rule. Under the new base's name, with these texts, it was graded later on 2026-10-12: test-only commit d2abf8f9d
  (dev 1ee40ae14 plus the entry above), `release_rebuild.py --families interlocked_claws` PASS, oracle 1 and nop 0 in
  both modes, the same final states (`docs/rcb/families/interlocked_claws_rotor.md`).
