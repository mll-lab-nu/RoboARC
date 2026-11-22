# cast_keyhole_double_maze on MuJoCo (the double maze)

Released later as the base `roboarc-cast-keyhole-06-i01`; this note records its development, from when it was an experimental
setting (statements below about its release status are of that time).

An experimental setting (2026-10-12) of the cast_keyhole family beside its five tiers (variation id
`cast_keyhole_double_maze`, pool experimental, tier open until the model evaluation); the five tiers are unchanged (their
instances built from this branch have the births their -i01 references recorded). The design record, searches and every
measurement were kept outside this repository.

## The puzzle and the setting

Two 9 mm plates, gold (119 x 278.5 mm) and silver (119 x 252 mm; the colours of appearance seed 0 and the object names,
cast_keyhole.md), each with a slot grid of five columns, threaded
through each other at right angles: the silver plate's left edge runs in the gold plate's grid, the gold plate's right
arm runs in the silver plate's grid. Both grids share one depth axis (how far the plates are pushed into each other), so a
move along a row moves both mazes, and only a door between two rows is crossed in one grid alone; the two mazes must be
solved interleaved. Slots are 2 mm wider than the plates.

- Geometry (`topogym/assets/geometry/cast_keyhole_double_maze.py`, resource
  `variations/cast_keyhole/double_maze/geometry.json`, design dm4b): the plates as the family's slab prisms
  (`cast_keyhole.polygon_slabs` / `extrude_slabs`), density 6600 kg/m^3 as every cast_keyhole plate (gold 1.087 kg,
  silver 1.036 kg). dm4b is design dm4 with its three round holes as 24-gons around the circles (only material removed):
  drawn as 192-gons the slabs left collision pieces down to 0.09 mm that the profile's prism_partition could not re-cut
  (dm4: 100 pieces under 0.5 mm; dm4b: none; gold 36 pieces, thinnest 4.76 mm; silver 57, thinnest 0.59 mm).
- The designed route: 59 moves, 37 door crossings (gold 12, silver 25). Lattice searches (5 DOF, 1 mm / 2 deg) at true
  size and at 0.5 mm shrink per face find the shortest solution door by door along it (37 of 37 in the same order); with
  every silver door closed there is no way out at either size, also with the plates tilted up to +-60 deg; the gold's left
  arm never passes a silver hole (exhaustive at both sizes).
- Collision: profile `keyhole_double_maze_v2`, its own key `("cast_keyhole", "cast_keyhole_double_maze")`: keyhole_v4's
  settings (`_KEYHOLE`: prism_partition fatten_m 0.5 mm, impratio 50, solimp .99) with MuJoCo's multi-point convex
  collision off (`collision.multiccd` False, `<flag multiccd="disable"/>`; below, "The first acceptance round"). v1 was
  keyhole_v4's settings alone. `_KEYHOLE` and the five tiers' profile are unchanged.
- Opening: both plates in the hands (pregrasp_weld, contact_weld_v1 with initial_intent "close": both hands bind during
  the first control steps), at the designed route's first sample: the silver plate's bottom bar in the gold plate's bore.
  The left hand holds the gold plate at its left prong, the right hand the silver plate at its head. Plate poses and grasp
  frames are stored in the ALOHA model frame (as the interlocked_claws in-hand settings); the arms' joints are solved for
  them. The mount is unchanged; the puzzle floats 6 cm above the old easy oracle's working box (both arms follow the whole
  route by inverse kinematics with every joint at least 0.15 rad from its limits). Before the welds bind (20 ms of
  bilateral contact), each 1 kg plate turns in its hand by about 0.3-0.5 deg.
- Goal: the family's placement (terminal revision `cast_keyhole_placement_v4_grasp_contact_2mm`): the entire silver plate
  inside the fixed goal volume (here 350 x 350 mm inside the dashed line, 342 mm high), the gold plate behind the separator
  (y <= 52.9 mm), the silver at most 1 mm/s and 0.02 rad/s for 0.5 s; holding allowed. Task text (since the appearance
  seeds v2): "Separate and place the entire right-hand plate (`keyhole_silver`) inside the green dashed goal region.
  Keep the left-hand plate (`keyhole_gold`) on the puzzle/system side (the names are identifiers, not colours)."; it
  named the plates silver and gold before.

## Reference (-i01)

`scripts/robotangle/oracles/cast-keyhole-double-maze-i01/privileged.npz` (not included in this repository; `standard.npz`
is the same file), 8082 rows
(202.0 s), native gripper space, on the instance made on this branch with profile keyhole_double_maze_v2
(`rcb_topogym.py make-instance --task cast_keyhole --variation cast_keyhole_double_maze --scene-seed 0 --gripper-space
native`, with the MuJoCo tasks' SIMD masking; birth c15ba78d2bfda87e). Recorded with

```bash
python scripts/roboarc/shared/rcb_topogym.py run-oracle --task cast_keyhole --instance <instance.json> \
    --code tools/references/cast_keyhole/double_maze_oracle.py --out <dir>/privileged.npz
```

The oracle code follows `tools/references/cast_keyhole/double_maze_plan.npz`: per 40 Hz control step the
planned joints of both arms, the planned centres of mass of both plates and the silver's pose in the gold's frame. The
plan: from the birth joints to the route's first sample (58 rows), the designed route (1254 samples, 6528 rows), the
silver 40 mm further out of the gold's bottom opening (189 rows), then the gold 4 cm toward -y (behind the separator)
while the silver moves in a straight line into the middle of the goal volume, orientations kept (1157 rows); both arms by
inverse kinematics on the ALOHA model, solved for the plates where they sit in the hands once the welds have bound;
time-scaled to at most 0.1 rad/s per joint, 10 mm/s and 0.1 rad/s of the silver relative to the gold, 20 mm/s of either
plate. Control as Cast ABC's oracle (planned joints plus feedforwards of the drives' velocity lag and Coulomb friction;
the left arm integrates its joint error; the right arm integrates the silver's pose error relative to the measured gold
through its Jacobian), plus a feedforward of each plate's weight (the static joint sag J^T W / kp: unfed, the 1 kg plates
sag the low-gain drives by up to 0.074 rad, and within a few control steps the plates turn against each other by
several degrees). The first 8 control steps hold the birth joints with only the birth state's weight feedforward
(ramped in over 3 steps), independent of the plan, so the plates settle into the hands the same way each time. Grippers
stay closed. The episode ends in success at row 8082: final state 5bc8f1e7136b3f59. (On profile v1 the same code and
plan ended on c18098f933020b0f, birth a804367eaf3bfdd3; the two recordings' actions differ by at most 1.3e-6 rad.)

Measured on the recording: the plate-plate contact force is at most 4.0 N over every physics tick (the observer's peak,
while the plates settle into the hands; depth below 0.01 mm); counted every MuJoCo substep, the plates touch in 12 of
323,272 substeps (the largest at birth); read at the end of every control step from step 8 on (the diagnostic run, the same
actions bit for bit), the plates never touch; along the route the silver's pose relative to the gold stays within
1.96 mm and 0.22 deg of the plan (the designed clearance is 1.94-2.00 mm, the plates move in slots 2 mm wider than they
are thick); each hand's pads touch their plate at 2 points on average (never none); the grasp welds carry at most
9.8 N (gold) and 11.1 N (silver) as 20 ms means (the plates' weight; slip at 60 N). All of these read the same on v1.

Checks of the reference on profile v2 (2026-10-12), in a local dev image on the AMD Ryzen 9
7950X host with this branch's code:
- The same oracle code and plan, run with the same physics (birth c15ba78d2bfda87e) on the RoboTangle runtime python on
  a cloud host (mujoco 3.14.0, the AMD EPYC 9354 host), record the same actions bit for bit.
- Two fresh-process strict replays (`rcb_topogym.py replay --repeat 2 --strict`): success at row 8082, both
  5bc8f1e7136b3f59 (the recording's own final state), the instance exact.
- The stress observer (`tools/stress/replay.py --hold 0`): CLEAN; no fail, trigger or pop; no grasp slip;
  no speed or torque clamp. Its peaks are the birth's first 20 ms (before the welds bind): the fastest task link
  0.096 m/s, a finger on the silver 45.1 N, finger-pad depth 0.55 mm; energy created 0.0045 J.
- The equal-length nop (8082 rows of the birth targets, under the observer): no success, CLEAN.
- One run without the observer: 249.0 s (0.031 s per control step, load average about 29). On v1 the same checks
  passed (two strict replays on c18098f933020b0f, CLEAN, nop CLEAN; 214.5 s on a cloud host, 293.9 s here at load about 36).

Kinematically (the plan's joints on the ALOHA model, every sample): no robot-robot or robot-table penetration; the gold
stays at least 66.4 mm from the right arm, the silver at least 19.9 mm from the left arm, both plates at least 52 mm above
the table.

Evidence (plans, runs, diagnostics, check outputs) was kept outside this repository.

A video of the v1 reference's physics run was made with nyx_direct (`render_reference_video.py --renderer direct`; 54 s
at 30 fps, 3.75x; not included; the replay reproduces that recording, c18098f933020b0f). The v2 recording's actions differ from it by at most 1.3e-6 rad.

## Harbor under the new base name (2026-10-12, profile v1)

This run graded the base on profile keyhole_double_maze_v1 (birth a804367eaf3bfdd3, the v1 reference); on v2 it ran
again on 2026-10-13 ("Harbor under the new base name on v2" below). `release_rebuild.py b38a7f6be --version <dry-run version> --families cast_keyhole` (with Harbor), on the
test-only commit b38a7f6be (167f4bbd8 + a cast_keyhole entry
in mujoco_release.json with `"tiers": []` and the new base `cast-keyhole-double-maze-i01`, template
`cast-keyhole-easy-i00`; never merged; the real manifest lists no cast_keyhole entry). A local dry-run image
(sha256:214444106259c90e2afeda800bda03db69be7a71a4fda1228118fed52be9fb0f), built
with `source/images/robotangle/Dockerfile.mujoco` from that commit; make-instance inside it gives the birth a804367eaf3bfdd3
(the reference's); `mk_tasks.py --check` passes for every base and for the MuJoCo instance bytes. Rendering off
(`RCB_RENDER=off`). Expected oracle 1, nop 0; all four trials as expected:

| mode | agent | reward | wall | evidence |
|---|---|---|---|---|
| privileged | oracle | 1 | 812 s | final state c18098f933020b0f in both replays; deterministic, instance exact, no review or forcing event |
| privileged | nop | 0 | 52 s | no trajectory submitted |
| standard | oracle | 1 | 867 s | live and replay both success, both c18098f933020b0f; no review or forcing event |
| standard | nop | 0 | 90 s | no action, ends on the timeout in the birth state |

The texts Harbor graded: the release's `packages/cast-keyhole-double-maze-i01-<mode>/{task.toml,README.md,instruction.md}`
are the files of `tools/release/package_text/cast_keyhole/` at 167f4bbd8, byte for byte; the task dirs Harbor ran are
the rendered ones, named `robotangle-cast-keyhole-double-maze-i01-<mode>`, with no word of easy's scene, no difficulty
line, `oracle = "full"`, the 60 N grasp rule, 0.036 s per control step and the 2 mm sustained contact rule, in both
modes. The report and the text check were kept outside this repository.

## The first acceptance round and keyhole_double_maze_v2 (2026-10-12)

The 1,000-episode acceptance on f2449335c (round keyhole_dm_r1, criteria v2.10, seeds 104001-104009) failed on one
confirmed BREAKDOWN, cem_e157: "energy_created(local): X rose 1.37 J in 0.1 s, contact (table, silver) gave 1.44 J"
(999 episodes otherwise; 424 VIOLENT, 599 grasp slips). Analysis only, then the decision below:

- The stream: from row 0 it commands the left gripper slightly open (0.0057 against the birth's 0.002), so the gold is
  never welded; at row 30 it opens the right gripper and lets the silver go (a release, not a slip). The arms had swung
  the plates up to about 1.6 m at the joint speed limit; the silver falls about 0.45 m (its 3.13 m/s), hits the gold
  (1795 N), spins up to 12-18 rad/s, and tumbles over the table's -y edge with its centre of mass beyond the edge.
- The energy: for 17 substeps the table box and three of the silver's convex pieces at the box's edge got contacts from
  MuJoCo's multi-point convex collision (multiccd) with wrong depths and normals: a 5.84 mm piece 1.44 mm deep where the
  exact separating-axis penetration was 0.33 mm, a 0.59 mm piece 1.56 mm for 0.003 mm, a 1.2 mm piece 0.85 mm for
  0.003 mm, the normals 84-90 deg off (sideways instead of up); 6-9.8 kN along them pushed the plate the wrong way.
  Recomputed on the same states, contacts without multiccd have the exact depths and normals (libccd is off by 43-84
  deg). The observer's small A_ii (0.027-0.067) are the elliptic cone's tangential rows (efc_diagA = the normal row's
  A_ii / impratio 50); their true J M^-1 J^T diagonal is 1.0-1.9: no degenerate geometry.
- Reproduction without the stream: the saved state just before the event, the silver moved up to 0.5 mm, turned up to
  0.5 deg, its velocities scaled by 0.9-1.1 (40 states, the table's net work on the silver per contact episode): 9 of 40
  create more than 0.25 J (up to 1.78 J), 40 of 40 have wrong contacts. 100 random spinning drops onto the edge create
  nothing: it takes this tumble over the edge.
- Candidates on the same 40 states, cem_e157 and the round's 148 table-related streams (each replayed under the observer
  with the wrong plate-table contacts counted every substep):

| | 40 states: over 0.25 J / with wrong contacts | cem_e157 | 148 streams: BREAKDOWN / wrong contacts / largest pair net |
|---|---|---|---|
| v1 as recorded | 9 / 40 | BREAKDOWN | 1 / 63 / 1.44 J (table, silver) |
| multiccd off (v2) | 0 / 0 | VIOLENT, 0.012 J | 0 / 0 / 0.19 J (a finger on the silver; 0.19 J on v1 too) |
| collision pieces re-cut to 2 mm (fatten_m; thinnest 1.05 mm) | 5 / 38 | VIOLENT, 0.012 J | 0 / 12 / 0.65 J (two fingers; the same on v1) |
| density 3300 kg/m^3 | 6 / 40 (energies halved) | VIOLENT | not run |

  The density test: at 3300 and 1650 kg/m^3
  cem_e157 creates no energy, but the plates fly faster and never reach the tumble; the v1 reference recorded at 6600
  does not succeed open loop on the lighter plates.
- The decision (2026-10-12): multiccd off in the double maze's own profile only (keyhole_double_maze_v2); the
  oracle unchanged, the reference recorded again; density, sizes, texts and difficulty unchanged. The scene builder takes
  the setting as an option no other profile uses (cba44aeba): instances of the five keyhole tiers, claws easy, Cast ABC,
  hedgehog easy, cast_coil easy, cast_coil_v2 easier and buzz easy built with it are byte for byte those built before,
  with their references' births.
- What multiccd off changes: each convex piece touches with one point instead of up to four (a plate lying flat on the
  table: 57 table points instead of 216 for the silver, 36 instead of 143 for the gold; both rest as still, 0.02 mm
  deep). The reference's contacts are single finger points and the brief contact at birth, so its tracking, contacts and
  weld forces read the same; with the same oracle code and plan it succeeds at row 8082 again.

## The acceptance round on v2 (2026-10-13)

On e9e29c39e (round keyhole_dm_r2, criteria v2.10, seeds 106001-106009): **PASS**.
- Self-check first: the reference success, CLEAN, 5bc8f1e7136b3f59 twice; the nop no success, CLEAN; cem_e157 as the
  reproducer DM1_keyhole_double_maze_silver_table_edge VIOLENT twice (BREAKDOWN on v1); 70 search episodes, 0 BREAKDOWN.
- The round: 1,000 episodes, 529,671 control steps, 0 confirmed BREAKDOWN; the reference success, CLEAN, the same final
  state (16.1 min under the observer on the loaded host); the nop no success, CLEAN; DM1 VIOLENT; 627 grasp slips.
- The merge of this branch (37d27b3f6) builds the same birth c15ba78d2bfda87e, so the round carries over.

## Harbor under the new base name on v2 (2026-10-13)

`release_rebuild.py 0956310dd --version <dry-run version> --families cast_keyhole --lanes 2` (with Harbor),
on the test-only commit 0956310dd (e9e29c39e + b38a7f6be's
cast_keyhole entry in mujoco_release.json, its note updated to v2; never merged). A local dry-run image
(sha256:86d3245e6d06d1063f2dec1a90b9cc5968d859c76ec7f44f31f90ecf71ab2b63);
make-instance inside it gives the birth c15ba78d2bfda87e (the reference's; instance sha256 5ebf97fa3b9142f3, 97
files); `mk_tasks.py --check` passes for every base and for the MuJoCo instance bytes. Rendering off. Expected oracle 1,
nop 0; all four trials as expected:

| mode | agent | reward | wall | evidence |
|---|---|---|---|---|
| privileged | oracle | 1 | 699 s | final state 5bc8f1e7136b3f59 in both replays; deterministic, instance exact, no review or forcing event |
| privileged | nop | 0 | 43 s | no trajectory submitted |
| standard | oracle | 1 | 766 s | live and replay both success, both 5bc8f1e7136b3f59; no review or forcing event |
| standard | nop | 0 | 70 s | no action, no success |

The texts Harbor graded: the release's `packages/cast-keyhole-double-maze-i01-<mode>/{task.toml,README.md,instruction.md}`
are the files of `tools/release/package_text/cast_keyhole/` at e9e29c39e, byte for byte; the task dirs Harbor ran
are the rendered ones, named `robotangle-cast-keyhole-double-maze-i01-<mode>`, with no word of easy's scene, no
difficulty line, `oracle = "full"`, the 60 N grasp rule, 0.031 s per control step and the 2 mm sustained contact rule,
in both modes. The report and the text check were kept outside this repository. The instance upload and the image push a
release would need were not done.

## Still open

- The tier, after the model evaluation; whether the base goes onto the release list is still to be decided (then the
  instance upload and the image push).
- Grasp slips under random actions: 627 in the v2 round (599 on v1), from the opening with both interlocked 1 kg plates
  in hand (the five tiers open with empty hands: 29-49 per round). Not a failure under the criteria; the reference slips
  0 times. Whether to change anything is still to be decided.
- The pipeline's step 4 (noise: other arm orientations, a release onto the table, small reorientations) is not done.
