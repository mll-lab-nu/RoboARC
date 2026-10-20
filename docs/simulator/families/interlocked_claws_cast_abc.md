# interlocked_claws_cast_abc on MuJoCo (Hanayama Cast ABC)

Jianwen, 2026-10-11. An experimental setting of the interlocked_claws family beside its five tiers (variation id
`interlocked_claws_cast_abc`, pool experimental, tier open until the model evaluation); the five tiers are unchanged.
Design record, sources and every measurement: `runs/cast_abc_design_20261011/` (README.md, PROGRESS.md; not in git).

## The puzzle and the setting

Hanayama Cast ABC (HZ 1-03, level 1; "design supervised by Nob. Yoshigahara", after a puzzle from England, 1911): an AB
plate (the letters A and B cast as one plate) and a C ring with a pointed gap. The plate's bars carry eleven cross
grooves; six are deep enough for the C's gap to pass the bar, five are dummies. The official solution (Hanayama's own
clip, Puzzle Master, Crux Puzzles, The Puzzler, Puzzle Time: the same route): the gap passes eight real grooves and the C
is turned over once, then the C comes out of the B's lower window.

- Geometry (`topogym/assets/geometry/cast_abc.py`, model v4): both pieces from a top-down video frame, 10 mm per bar width
  (plate 233 x 137 x 19 mm, 101 g; C ring 70 x 127 x 15 mm, 36 g; density 600 as every claws piece). Two dimensions differ
  from the measurements so that the official route is the shortest one in this geometry (the user accepted them on
  2026-10-11): the C 7.0 instead of 7.2 bar widths wide, and the A cross-bar groove 0.4 bar widths toward the A's right
  leg. With the measured values a two-groove shortcut opens (A cross bar, then the bottom bar).
- Collision: profile `claws_v6_cast_abc` = claws_v6 plus `split`: each piece collides as its exact convex decomposition
  (plate 201 pieces, ring 101; thinnest 1.04 mm / 2.29 mm), so the windows, the notches and the C's gap stay open. The
  scene builder's check: plate volume 2.9e-7, part outside the mesh at most 7.1e-8 m, mesh uncovered at most 4.7e-8 m.
- Opening: both pieces in the hands (pregrasp_weld, contact_weld_v1; both hands acquire at the first control step), at the
  reference route's first sample: the C round the A's left leg. The left hand pinches the plate's bottom bar at its left end
  across the plate; the right hand pinches the C's upper end across its depth.
- Goal: the family's whole-piece separation (a plane separating all geometry with at least 10 mm, both pieces at most
  1 mm/s and 0.02 rad/s, held for 0.5 s; holding allowed). While the C encircles a bar no such plane exists.
  Task text: "Completely separate the C ring from the AB plate and briefly hold both still."

## Reference (-i01)

`scripts/robotangle/oracles/interlocked-claws-cast-abc-i01/privileged.npz` (`standard.npz` is the same file), 5979 rows
(149.5 s), native gripper space, on the instance made on this branch (`rcb_topogym.py make-instance --task
interlocked_claws --variation interlocked_claws_cast_abc --scene-seed 0 --gripper-space native`; birth
e6f9667c9f67c760). Recorded with

```bash
python scripts/robotangle/shared/rcb_topogym.py run-oracle --task interlocked_claws --instance <instance.json> \
    --code scripts/robotangle/references/interlocked_claws/cast_abc_oracle.py --out <dir>/privileged.npz
```

The oracle code follows `scripts/robotangle/references/interlocked_claws/cast_abc_plan.npz`: per 40 Hz control step the
planned joints of both arms (the reference route's inverse kinematics) and the planned pose of the C in the plate's frame.
Each arm joint gets its planned target plus feedforwards of its drive's velocity lag and Coulomb friction, and a slow
correction: the left arm integrates its joint error (the plate's weight makes the low-gain drives sag), the right arm
integrates the C's pose error relative to the measured plate through its Jacobian. Grippers stay closed. The episode
ends in success at row 5979: final state 4701f99e7860e8c9.

Measured on the recording (fresh-process replay, every control step): the two pieces never touch (no contact row); their
closest distance is 1.69 mm (the floating route's own minimum is 1.88 mm); the C's pose relative to the plate stays within
0.85 mm and 0.47 deg of the plan after the first 2 s (1.49 mm while the pieces settle into the hands at birth); the grasp
welds carry at most 5.1 N (slip at 60 N).

Checks of the reference on this branch's code (2026-10-11), on the RoboTangle 0.2.22 runtime python (mujoco 3.14.0,
trimesh 5.1.0, shapely 2.1.2, scipy 1.17.1, numpy 2.3.5; the image's libraries): make-instance there gives the same birth
e6f9667c9f67c760, instance.json and asset bytes as the image `topogym-coil-v2:20261009-02` (the recording's).
- Two fresh-process strict replays (`rcb_topogym.py replay --repeat 2 --strict`): success at row 5979, both
  4701f99e7860e8c9 (the recording's own final state).
- The stress observer (`scripts/robotangle/stress/replay.py --hold 0`, criteria v2.10): CLEAN; no fail, trigger or pop;
  the fastest task link 0.044 m/s, task contact force at most 9.5 N (the holding fingers), task contact depth at most
  0.41 mm (finger pads on their pieces), energy created 0.0003 J, no grasp slip, no speed or torque clamp.
- The equal-length nop (5979 rows of the birth targets, under the observer): no success, CLEAN.
- One run without the observer: 197 s of wall clock (33 ms per control step) on a loaded 128-core host.

A video of the reference's physics run (nyx_direct): `runs/cast_abc_design_20261011/phys/video/cast_abc_i01_nyx.mp4`.

Rechecked on dev 7f5516d02 (2026-10-12; `runs/cast_abc_design_20261011/phys/head_7f5516d02/`). The commits since
6d7ee21e4 leave Cast ABC's physics and scene as they were: `grasp_rules` text (1cbd0283d); placement-only goal text;
`scene._family_parts`' winding-number recheck of part vertices more than 1e-7 m outside the mesh (Cast ABC's worst is
7.06e-8 m); the claws grasp TCP distance default (0.06, unchanged); new profile keys of other settings. No commit touched
nyx_direct, the render bridge, the MuJoCo world, grasp or robot, or Cast ABC's own code.
- make-instance there: birth e6f9667c9f67c760, instance.json and all 305 asset files byte for byte as recorded.
- At rest (`stress/replay.py --actions nop:400`, 40 hold steps, criteria v2.10): CLEAN, no event; energy created
  0.0003 J; fastest task link 0.044 m/s and 1.4 rad/s (the pieces settling into the hands at birth); deepest task contact
  0.41 mm (finger pads).
- Collision geoms: 201 and 101 convex meshes; their volume is 168.5 and 59.9 cm^3 against the pieces' own convex hulls'
  493.6 and 117.5 cm^3, so no hull closes a window or notch; thinnest 1.038 mm and 2.289 mm, none under 0.5 mm.
- Nyx (nyx_direct): one frame per camera, both pieces in their hands, colours and occlusion as expected (`nyx/`).
- Two fresh-process strict replays (one process each): success at row 5979, both 4701f99e7860e8c9.

## Harbor (2026-10-11)

Image `eai-robotangle:0.3.0-dev-cast-abc` (sha256:ae1cd42c1435edcddced02f24c2be22723081b62a591e99aded79ca423bc5f88),
built with `images/robotangle/Dockerfile.mujoco` from 6d7ee21e4. make-instance inside it gives the birth e6f9667c9f67c760
with the same asset bytes; its instance.json adds only `topogym_source`. `harbor_check.py --tasks --lanes 1`, rendering
off (`RCB_RENDER=off`). Expected oracle 1, nop 0; all four trials as expected:

| mode | agent | reward | wall | evidence |
|---|---|---|---|---|
| privileged | oracle | 1 | 689 s | final state 4701f99e7860e8c9 twice; deterministic, instance exact, no review or forcing event |
| privileged | nop | 0 | 46 s | no trajectory submitted |
| standard | oracle | 1 | 633 s | live and replay both success, both 4701f99e7860e8c9; no review or forcing event |
| standard | nop | 0 | 78 s | no action, ends on the timeout in the birth state |

New settings had no release base yet, so this check rendered the package of the released base `interlocked-claws-easy-i00`
and replaced its scene-describing lines with Cast ABC's (15 lines in privileged, 10 in standard; no easy-scene word
left). The texts that were graded: `runs/cast_abc_design_20261011/out/task_texts/`. The grading holds; the package text is
not a release text. The setting's own base followed on 2026-10-12 (next section).

## Harbor under its own base interlocked-claws-cast-abc-i01 (2026-10-12)

Cast ABC is made evaluable, not released (the user, 2026-10-12, as for Rotor): `mujoco_release.json` lists nothing for
it. Its texts are `scripts/robotangle/package_text/interlocked_claws/interlocked-claws-cast-abc-i01-<mode>/` (7f5516d02;
the sustained contact rule corrected to 2 mm in 62d8d0d5c), laid over the released `interlocked-claws-easy-i00` by
release_rebuild's new_bases. The base is made on a test-only commit, 21b3e083d on the branch
`jianwen/cast-abc-harbor-test` (dev 62d8d0d5c plus the interlocked_claws entry reduced to `"tiers": []` and this new base;
never merged):
`release_rebuild.py 21b3e083d --version 0.3.0-dryrun-cast-abc-newbase --families interlocked_claws` (with Harbor): **PASS**.
- Image `ghcr.io/mll-lab-nu/eai-robotangle:0.3.0-dryrun-cast-abc-newbase` (sha256:f132c1c7e41c54f5…, local only). The
  instance made in its image without instances: birth e6f9667c9f67c760, the reference's (instance sha256
  79b3fd3e9b73e6e0, 306 files). `mk_tasks.py --check`: every base and the MuJoCo instance bytes ok.
- The texts Harbor graded: the release's `packages/interlocked-claws-cast-abc-i01-<mode>/` task.toml, README.md and
  instruction.md are the package_text files byte for byte; the rendered tasks are named
  `robotangle-interlocked-claws-cast-abc-i01-<mode>`, keep no word of easy's scene (the change log names the template,
  as every new base's does), have no difficulty line and `oracle = "full"`, state the 60 N grasp rule and 0.053 s per
  control step.
- Rendering off; expected oracle 1, nop 0; all four as expected:

| mode | agent | reward | wall | evidence |
|---|---|---|---|---|
| privileged | oracle | 1 | 688 s | final state 4701f99e7860e8c9 twice; deterministic, instance exact, no review or forcing event |
| privileged | nop | 0 | 47 s | no trajectory submitted |
| standard | oracle | 1 | 753 s | live and replay both success, both 4701f99e7860e8c9; instance and assets exact |
| standard | nop | 0 | 83 s | no action, ends on the timeout in the birth state |

Run: `runs/cast_abc_design_20261011/harbor_newbase/` (REPORT.md; `work/harbor/results.md`). The instance upload and the
image push that a release would need were not done.

## Stress acceptance (2026-10-12)

On 6d7ee21e4, criteria v2.10, 1,000 episodes over the six strategies, seeds 98001-98009: **PASS**. No confirmed BREAKDOWN;
the reference success, CLEAN, same final state (10.5 min under the observer); the nop no success, CLEAN; 110 VIOLENT in
the search (71 from cem; reported, not failed); 229 grasp slips. Report: `runs/stress_20261012/cast_abc_6d7ee21e4/`.

## Grasp slips under random actions

The self-check's search (68 episodes) counted 28 grasp slips, more than the 0-2 of the cast_coil v2, hedgehog and
lock_picking self-checks. Rerun of the four reproducible strategies (62 episodes, every episode equal to the self-check's)
with the weld's force split by source, per substep:

- All 28 are `grasp_slipped` (the weld's translational force, mean over 20 ms, above 60 N): the left hand (plate) 22, the
  right hand (C) 6. Fifteen more losses of the C: 12 relative-pose errors (the C's origin is 186 mm from the weld anchor,
  so 2.3-3.6 deg of turn while the C is stuck moves it the 10 mm limit) and 3 opened fingers.
- In each slip window the piece's own inertia and weight is 0.35-7.6 N (median 1.0 N, the plate's weight); the rest is
  contact: the C pressing on the plate (up to 145 N), the other arm pushing the plate, a piece pressed on the table. Both
  hands hold one of two interlocked pieces; random actions pull them apart and the force reaches the welds through the
  interlock. Not the grasp point or the plate's moment: at the joint speed limit the plate's own load stays at most 16 N.
- Same code, seeds, strategies and lengths on the other instances: claws easy 63 slips, medium 57, hard 9, Cast ABC 28.
  (cast_coil v2, hedgehog and lock_picking open with empty hands.)
- The reference loses nothing: the welds carry at most 5.1 N.

Nothing was changed in the grasp itself. The agent's task text now states MuJoCo's slip rule as well (the user's decision,
2026-10-12): the shared commit 1cbd0283d added it to `Session.grasp_rules()`, and Cast ABC's texts carry it (package_text
since 7f5516d02, so 62d8d0d5c and the test commit 21b3e083d): the privileged instruction.md's "Grippers:" line (line 102)
and the standard instruction.md's "Grasping:" line (line 116; line 115 of the rendered standard task) say a hold is also
lost when it "has to push or pull its part with more than 60 N averaged over 20 ms", with the example of two held parts
that hook into each other being pulled apart. Data: `runs/cast_abc_design_20261011/phys/slip/summary.json`.

## Still open

- Not on the release list (the user, 2026-10-12): evaluable through the test-only manifest above; listing it, with the
  instance upload and the image, would be the user's decision.
- The tier, after the model evaluation.
- Public release: Cast ABC is a Hanayama product design; confirm before publishing (as for the Rotor setting).
- The pipeline's step 4 (noise: other arm orientations, a release onto the table, small reorientations) is not done.
