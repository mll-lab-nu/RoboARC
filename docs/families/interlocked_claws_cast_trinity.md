# interlocked_claws_cast_trinity on MuJoCo (Hanayama Cast Trinity)

Jianwen, 2026-10-13. An experimental setting of the interlocked_claws family beside its five tiers (variation id
`interlocked_claws_cast_trinity`, pool experimental, no tier); the five tiers are unchanged. Design record, sources and
every measurement: `runs/cast_trinity_design_20261011/` (README.md, PROGRESS.md; not in git). Released as
`interlocked-claws-13-i01` under the RoboARC names (2026-10-16); the base name `interlocked-claws-cast-trinity-i01` below
is the one these checks ran under (the same instance and reference bytes).

## The puzzle and the setting

Hanayama Cast Trinity (level 6 of 6, designed by Kyoo Wong): three Lambda-shaped pieces, each with a hooked tip at one end
and two heads with a narrow gap between them at the other. At the start the three tips hook into each other and every
pair is linked. The official solution moves the pieces round each other, then three times pushes one piece's thin part
through another piece's head gap, separating the pairs one at a time (B from T, then T from H, then B from H). No 3D model
of the puzzle exists; the user decided on 2026-10-12 that the pieces may be rebuilt from photos with small adjustments, as
long as the solution stays as involved as the original's, is feasible in the new model and has no shortcut.

- Geometry (`topogym/assets/geometry/cast_trinity.py`, model `cast_trinity_v12`): rods, a flat band and block heads, 150 mm
  long (about 2.7 times the real puzzle), density 600 as every claws piece. Each piece's band has a 4 mm thick neck by the
  heads, the only part thinner than the 5 mm head gap; B's two heads interlock as two L shapes, so its gap is crosswise and
  nothing passes it. Pieces `trinity_t` (gold, engraved TRINITY), `trinity_h` (H, T's mirror) and `trinity_b` (copper).
- Collision: profile `claws_v6_cast_trinity` = claws_v6 plus `split`: each piece collides as its exact convex decomposition
  (`t_parts`, `h_parts`, `b_parts`), so the windows, slots and head gaps stay open.
- Opening: T in the left hand, B in the right (pregrasp_weld, contact_weld_v1; both hands acquire at the first control
  step), H hanging on both; the first state of the reference plan (all three pairs linked).
- Assisted grasp: every piece is a target of both hands, as a whole piece (centred on its bounding box, radius 9 cm; a hand
  may acquire with its TCP up to 12 cm from that centre; `InterlockedClawsCastTrinityTask.pregrasp_targets`). The family's
  default (the two opening pieces, 3 cm round the opening grasps) would leave H impossible to hold.
- Goal (`trinity_terminal.pairwise_separation_candidate`, terminal revision `trinity_pairwise_separation_v1`): every two of
  the three pieces separated by a plane with at least 10 mm (all visible and collision vertices), all three at most
  1 mm/s and 0.02 rad/s, held for 0.5 s; holding allowed. The public goal states the rules for three pieces
  (`separation_rules`; `goal_description.describe_goal` uses a setting's own rules when it has them). Since 2026-10-19
  the texts' grader details say how the grader measures the separation (every two pieces, ten directions only) and that
  neither the green square nor a separator is checked (`scripts/topogym_rules.py`; until then they carried the placement
  families' goal-square and separator lines; `scripts/robotangle/package_text/interlocked_claws/README.md`).

## Reference (-i01)

`scripts/robotangle/oracles/interlocked-claws-cast-trinity-i01/privileged.npz` (`standard.npz` is the same file), 9596 rows
(239.9 s), native gripper space, on the instance `rcb_topogym.py make-instance --task interlocked_claws --variation
interlocked_claws_cast_trinity --scene-seed 0 --gripper-space native` (birth 195513489176c313). sha256 6dfe3a698467f487.
No row moves an arm joint more than 0.035 rad (the standard service splits rows above 0.04 rad), so the standard mode runs
the same rows. Recorded with

```bash
python scripts/robotangle/shared/rcb_topogym.py run-oracle --task interlocked_claws --instance <instance.json> \
    --code scripts/robotangle/references/interlocked_claws/cast_trinity/oracle.py --out <dir>/privileged.npz
```

The oracle (`scripts/robotangle/references/interlocked_claws/cast_trinity/`: `oracle.py`, its plan `plan.json` and
`plan.npz`, and the two-piece planner modules it imports, the design record's code as it ran) commands joint and gripper
targets only; it never sets a pose. Each arm joint gets its target plus feedforwards of its drive's velocity lag and
Coulomb friction, a slow integral of its joint error, and (for the arm that moves a piece) an integral of that piece's
pose error relative to the other held piece through the arm's Jacobian.

- The first segment follows the floating plan (`plan.npz`: both arms' joints per control step and B's pose in T's frame):
  B along its two-piece route while the left hand holds T, both hands turning the puzzle, B's neck pushed through T's gap.
- Everything after it is planned from the measured state, since where a hanging piece comes to rest cannot be predicted
  closely enough:
  - `clear`: the right hand moves B 40 mm away from T (after the push B is 0.4 mm from T and would slip back into T's gap),
    along the straight move in T's frame that leaves the largest gap and keeps T and B unlinked;
  - `take`: a hand takes a piece where it is: the first of the piece's valid grasps that the arm reaches (the robot's own
    bounded least-squares IK) and whose open hand, straight approach and joint-space transit touch nothing (checked on a
    scratch copy of the MuJoCo data, never the live one), and from which the next separation can be planned; of a grasp
    and its 180-degree twin, the one with the forearm roll and wrist rotation further from their +-180 degree limits;
  - `release`: the left hand opens only 2 mm on each side and leaves along its contacts' normals (H rests on its fingers);
  - `pairplan`: the separation from the measured relative pose (RRT-Connect with both hands at their actual grasps as
    obstacles, at least 0.5 mm between the pieces, ending at the lying passage), then a straight push along the passage.
    Both arms share the route (one takes a fraction of the motion about the pieces' midpoint, the other the rest; the
    relative route is unchanged) and the pair may first be moved and turned about the vertical; the share and placement are
    the first whose whole route the robot's IK reaches without a joint jump and without an arm touching anything else.
    The joints along the route are computed beforehand and followed with the servo;
  - `place`: T on along the push, up, to the table and released 3 cm above it.
- In this recording: segment 3 (T out of H) turns the pair 45 degrees about the vertical, the left arm taking a quarter of
  the relative motion; segment 4 (B out of H) is moved by the right arm alone. The pieces' relative pose stays within
  1.1 mm of the planned routes; the three takes end 0.19, 0.10 and 0.09 mm from their grasps.
- The planner's budgets are wall-clock (RRT time limits); every plan in the recording was found well inside them.

Checks of the reference (2026-10-13, a cloud host, the RoboTangle runtime python, mujoco 3.14; `runs/cast_trinity_design_20261011/`):
- One run of the oracle: success at row 9596, final state 66be99300a3ceebd, 527 s of wall clock.
- Two fresh-process strict replays (`rcb_topogym.py replay --repeat 1 --strict`, one process each): both success at row
  9596, final state 66be99300a3ceebd, 315 s each.
- The stress observer (`scripts/robotangle/stress/replay.py --hold 0`, criteria v2.10): no event; no instability, review or
  forcing; deepest task contact 0.76 mm (H and B), deepest held-finger contact 0.42 mm; largest press 96.8 N at 58 % of the
  drive cap (no forcing); fastest task piece 0.73 m/s and 15.8 rad/s (T dropped onto the table); energy created 0.0014 J.
- The equal-length nop (9596 rows of the birth targets, under the observer): no success, no event.
- The instance rebuilt after the public goal's rules for three pieces: birth 195513489176c313 as before (only the text and
  the public goal in instance.json differ); the reference replays to success on it.

## Harbor under this base (2026-10-13)

`release_rebuild.py` on the test-only commit cd17b901b (never merged: interlocked_claws in `mujoco_release.json` lists
only this new base, under its earlier name `interlocked-claws-cast-trinity-i01`), image `eai-robotangle:0.3.0-dryrun-trinity`,
on a lab RTX 4090 host (AMD Ryzen 9 7950X, no GPU), rendering off: **PASS**.
- The instance made in the release image: birth 195513489176c313, the reference's.
- privileged and standard: oracle 1 (the reference, trajectory sha256 6dfe3a698467f487, deterministic), nop 0; the
  oracle's end 66be99300a3ceebd.
- The standard live episode took 692 s for 9,596 control steps (0.072 s at a load average of 50-70); the texts state that
  step time (a7ad58b501).
- Evidence: the rebuild's record, commit 9f1f2d24c3 (branch release-rebuild/cd17b901b-T, `scripts/robotangle/release.json`
  `bases["interlocked-claws-cast-trinity-i01"]`: evidence, nop, host); the commit message of 6fa095f775. The rebuild's work
  directory was a session scratchpad and no longer exists.

## Stress acceptance (r1, 2026-10-13 to 10-16)

Criteria v2.10, search protocol 3, seed0 109010 (seeds 109011-109019), 1,000 episodes over random, directed, mutate, cem,
prefix (named four times) and periodic, 4 search and 2 replay workers; a self-check (seed0 109001) before it.
- Started on a cloud host (source 42bda423c) on 2026-10-13. The user stopped it at 2026-10-14T03:22Z (STOPPED.txt), with random,
  directed, mutate and cem (167 episodes each), the four prefix workers, the reference replays and the nop done and
  periodic at 51 of 166. Resumed on a lab RTX 4090 host on 2026-10-16 (source a7ad58b501, a text-only difference) with the same
  round.py command: the reference replayed there first to 66be99300a3ceebd, as on a cloud host; periodic started over (same seed);
  the paths in plan.json and the candidates rewritten (a copy keeps the original plan; a note file records the move).
- The round: **PASS**. 584,448 controls; 0 confirmed BREAKDOWN under every reading v2 to v2.10. One candidate went to
  confirmation (mutate e98: a finger-to-finger row with A_ii 0.091, under 0.1); its fresh-process replay broke nothing,
  only the right hand's grasp was lost (relative_pose_error). 43 grasp slips; 3 pops, the fastest 1.99 m/s. P(miss) 0.368
  at a 0.1 % breakdown rate and 0.0067 at 0.5 %.
- The reference: success, CLEAN, the same final state (12.4 min under the observer). The nop: no success, CLEAN.
- In-search VIOLENT: random 53, directed 1, mutate 67, cem 88, prefix 17, periodic 4.
- How the search's episodes ended (their own records, read in the metric review of 2026-10-19,
  `runs/robotangle_mujoco/metric_review_20261019/bed_bolt_trinity_zh.md`):
  - two successes, both prefix episodes cut after the reference had separated every pair: prefix2 e38 (cut at row 9,367,
    success at control step 9,545) and prefix3 e5 (cut at row 9,568, success at 9,595). The fastest piece in both is
    0.73 m/s and 15.8 rad/s, the reference's own (T dropped onto the table), and neither has a pop;
  - 88 episodes ended with a piece more than 100 mm below the tabletop (random 27, mutate 30, cem 22, prefix 9): the
    puzzle thrown off the table at 2-5 m/s;
  - every other episode ran to its length. The three pops (random e1 0.80 m/s, prefix4 e13 0.67 m/s, mutate e25 1.99 m/s)
    ended in no success.
- Evidence: a lab RTX 4090 host <host work dir>/topogym_stress/work/trinity_r1/r1/ (search/, the reports); a copy of the reports in
  `runs/cast_trinity_design_20261011/stress_r1/` (report.md, report.json, the move note, STATUS, STOPPED.txt);
  PROGRESS.md there, sections of 10-13 and 10-16.
