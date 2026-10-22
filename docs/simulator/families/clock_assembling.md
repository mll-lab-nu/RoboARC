# clock_assembling on MuJoCo (a watch movement's gear train)

2026-10-16 (v2 2026-10-17; v3 2026-10-19/20: the movement drawn more like a real one, the metric review's rule
changes, the bridge solid to the wheels, the left hand withdrawing in the run: "Realism, rules and the solid bridge"
below). A new family, MuJoCo only: two ALOHA arms assemble the gear train of a watch movement enlarged 8 times, then wind
it and watch it run. First setting `clock_assembling_gear_train` (pool experimental, no tier; model revision
`clock_gear_train_v3`, terminal revision `clock_gear_train_v3`; both v2 until 2026-10-19/20). Design record, reference video and every measurement:
`<design record>/watch_assembly_design_20261014/` (README_v3.md, REGISTRATION_PROPOSAL.md, RUN_TEST_PROPOSAL.md; not in git).

## The task

The owner's reference is the gear-train part of BV14bAjzEEcA (0:31-1:12, an IWC calibre 81925): the barrel, then the large,
the small and the steel wheel lowered onto their pivots, the bridge laid over all of them while a finger holds it and
tweezers nudge the pivots into its holes, the screws. The video shows no run of the train (1:13-1:20 is another wheel slid
in tilted under a placed wheel's rim, on the dial side); the run test (wind, let go, watch the fourth wheel) follows the
written sources' "check that the train runs free" and the winding works the video shows later (a ratchet wheel screwed on
the barrel arbor, a click; 3:46-4:26). The scene (`topogym/assets/geometry/clock_gear_train.py`):

- a copper-gilt main plate 304 mm across (38 mm real) on a holder (drawn with a movement holder's ring round its edge,
  below its top), its top 10 cm above the table; four lower pivot holes, three pillars (two with screw holes), two steady
  pins;
- on a stand: the barrel (90 teeth; a drum and its arbor with the ratchet and the square), the center wheel (80 + a 12-leaf
  pinion), the third wheel (75 + 10) and the fourth wheel (70 + 10), axes vertical, each turned in its cup by scene
  seed 0's draw for every seed (a seed varies appearance only, the owner 2026-10-20: `movement_finish`, `stand_finish`,
  `screw_finish` re-colour visual geoms only); the
  train bridge (six 20 mm bars, 12 mm thick, 60 g; a click drawn by the barrel's boss) on four posts (the barrel boss's
  post 17 mm in radius, the others 8 mm); two screws in a tray; a 9 cm probe on its stand. The layout is the design's after its base placement search (the whole scene moved by (-0.03,
  -0.06) m).
- Each wheel's disc lies 1 mm over the disc (the barrel: its drum) it overlaps in plan (0.125 mm real): with the 4 mm lower
  pivots no wheel can be slid in under a wheel already seated, so the train goes together in one order only (the order
  check below). A wheel standing on its lower pivot leans up to 1.5 deg per axis.

Collision: one MJCF entity, `clock`, the movement, stands and every part (its links; the barrel's arbor is the link
`barrel_arbor`, a child of the barrel); one entity because its collision classes act only inside an entity (Genesis' MJCF
semantics, `mjcf_entity.py`) and its equalities join bodies of one entity. Ten classes keep two meshing wheels from touching
(discs are root circles, teeth visual only), let the pivots meet only holes, the shanks only screw holes. The bridge is
solid to the wheels (since 2026-10-19): no wheel passes through it; the barrel boss collides as a ring with a 12.5 mm
opening the ratchet and the square pass through, and the arbors collide up to 1.2 mm under the seated bridge (0.4 mm
for a wheel seated 0.8 mm high, as the seat rule allows), drawn up to 0.4 mm under it. Against the robot and the table
every geom collides. Drawn-only geoms (contype 0, no mass: the colours, the wheels' spokes, the rings, the holder) are
dropped from the MuJoCo scene (`mjcf_entity.py`); the renderer draws them.

The rules are the task's mechanism (`topogym/tasks/clock_assembling/mechanism.py`, `TaskDefinition.mujoco_mechanism`;
the interface is bed_bolt's, docs/simulator/MUJOCO_CORE.md "Task mechanisms"), once per 2.5 ms tick:
- seat: a wheel whose lower pivot is down in its hole (within 0.6 mm of the axis, at most 0.8 mm high, tilt < 3 deg) is
  welded to its carrier (two +-1.5 deg lean hinges and a spin hinge at the pivot; inactive weld at birth);
- gate: while a wheel's pinion meets the previous wheel's teeth tip on tip (< 0.2 of a pitch), a short cylinder at its hole
  mouth collides with the wheel (switched on only where nothing overlaps it);
- mesh: two seated wheels are coupled by a joint equality between their spin hinges (polycoef[1] = the ratio, -7.5, -8,
  -7.5; the constant puts the leaves in the gaps);
- screws: a screw whose head sits on the bridge over either screw hole (its underside 0.3 mm below to 0.8 mm above the
  bridge's top, its axis within 0.6 mm of the hole's) counts its clockwise turns relative to the bridge in that hole (one
  screw to a hole; moved to the other hole it starts again); at 2 it is welded to the bridge;
- fix: both screws tight, the four wheels seated, every upper pivot in its bridge hole (0.3 mm in) and the bridge seated
  (0.5 mm, 0.5 deg): the bridge welded to the world, the lean hinges locked,
  the pivot-hole contacts off; the barrel's arbor welded to its own carrier (the click's hinge first set to the arbor's
  turn) and the mainspring's lock released;
- click: the click hinge's lower limit follows the largest angle the arbor reached less one ratchet tooth (18 deg, 20
  teeth): the arbor winds counterclockwise (seen from above) only;
- run test: once the arbor has not advanced for 0.5 s after a wind of at least 10 deg, a 5 s window measures the train
  running on the mainspring: the fourth wheel's turns, the time it spends below 10 % of its free-run speed (the speed the
  mainspring's present windup gives against the train's damping and friction, from the model) and its turn per drum turn;
  winding again restarts the window. The run counts for the mainspring alone: from the last winding to the window's end
  no contact of the robot or the probe with the drum or a wheel (the arbor, wound by its square, is not watched), and
  the drum turning +z (the mainspring's way); otherwise it is void (`run_void`) until the arbor is wound again.

The mainspring is the hinge between the drum and the arbor (stiffness 0.002 N m/rad, damping 1e-5; held rigid by the
joint equality `spring_lock` until the fixing, so the barrel is carried as one part); the click hinge has damping 1e-3 (a
released arbor falls back onto the click at a few rad/s).

Goal (`ClockAssemblingTask.evaluate_state`, terminal revision `clock_gear_train_v3`): the four wheels seated (at most 1 mm
above their seats), every upper pivot in its bridge hole (at least 1 mm deep), the bridge seated (0.5 mm, 0.5 mm, 0.5 deg)
and fixed, the three meshes coupled, and the run test: wound at least 120 deg (as the click holds it when the window
opens), at least 3.5 turns of the fourth wheel in the window (5 until 2026-10-19), at most 0.2 s below 10 % of its free-run
speed, its turn per drum turn -450 within 2 %, not void; held for 0.5 s, holding allowed (but not on the train in the
run).

## The order check

`<design record>/watch_assembly_design_20261014/scripts/order/` (results `out/order_check_v2/`): part motion only, on this collision
model (the entity MJCF, its classes, MuJoCo's signed distance). Only neighbours in the train overlap in plan, so a step is
decided by the moving wheel and its seated neighbours; all 32 (wheel, seated set) steps the 24 orders meet:
- the 20 steps with no seated wheel above the moving one: a straight descent from 80 mm, checked at every sample (the
  clearance at least the displacement to the next sample): all feasible;
- the 3 steps sliding a wheel under its seated child (barrel under center, center under third, third under fourth): a flood
  fill over the poses with the lower pivot in its jewel (from seated to the capsule's lowest point at the ring's top; grid
  0.1 mm x 0.2 mm x 0.5 deg and 0.05 mm x 0.1 mm x 0.25 deg; neighbouring poses joined only by certified collision-free
  motion), exhausted without the pivot getting out: the highest reached is 0.75 / 0.55 / 2.65 mm under the ring's top (fine
  grid).
So only barrel, center, third, fourth assembles. (With the v1 gaps of 4 / 2 / 2 mm, 12 orders did: the barrel slid under the
center wheel with 0.025 mm to spare, the center wheel under the third within a 0.02 mm window; `out/order_check/`.)

The one order is a requirement of the design, and this check is its evidence: after any change to the movement's geometry
or collision model (`topogym/assets/geometry/clock_gear_train.py`: a wheel's heights or radii, the gaps, TILT_MAX, the
contact classes), re-run it and keep the result with the change: `order_search.py <out.json>` (the 20 descents and the
coarse exits), then `order_flood.py <out.json> <moving>:<seated>` for `barrel:center`, `center:third` and `third:fourth`
on both grids (`--dz 0.1 --dc 0.2 --dt 0.5` and `--dz 0.05 --dc 0.1 --dt 0.25`). A change that lets any other order
through goes back to the design. Re-run 2026-10-18 after the probe's contact change (<design record>/watch_assembly_design_20261014/out/
order_check_b2/): the 32 path steps and the six floods identical to out/order_check_v2/ in every field but the wall clock.
Re-run 2026-10-20 on the bridge solid to the wheels (2a2e34ab5f, out/order_check_realism/): identical to order_check_b2 in
every field but the wall clock (the bridge is not in this check); and on the arbors' lower collision tops (af683a9b21,
out/order_check_realism_af683a9b21/): every flood exhausted without the pivot getting out, its highest pose the same,
every step's feasibility the same; only the counts of reachable poses (26 and 18 fewer
of 2.1 million and 337 thousand) and the infeasible descents' worst clearances (by 0.2 microns) moved. One order still.

## Reference (-i01)

`scripts/roboarc/oracles/clock-assembling-01-i01/privileged.npz` (`standard.npz` is the same file), 8047 rows
(201.2 s), native gripper space, on the instance `rcb_topogym.py make-instance --task clock_assembling --variation
clock_assembling_gear_train --scene-seed 0 --gripper-space native` (birth ad4ee5041eeb19de), end state 9683cab251ece2c0.
sha256 3f5b8292181f2ecf. No row moves an arm joint more than 0.035 rad. Recorded 2026-10-20 on <test branch>
af683a9b21 (birth e479193c42dee4e0, end f9e93ddbf4a3bec4, sha256 7623382346b467e6), then re-stamped with the birth of the
model revision clock_gear_train_v3 (7698adb5e6; the revision is in the state's hash): `instance_sha` rewritten, the rows
and the tick ledger byte for byte the same; a fresh strict replay succeeds at row 8047 on 9683cab251ece2c0 and its
(qpos, qvel, act) equal the recording's at every control step (out/realism_v3model/). On af683a9b21 (the realism change, the rule changes, the bridge solid to the wheels, the left hand withdrawing in the run;
"Realism, rules and the solid bridge" below). Its rows equal the earlier reference's (8171 rows) up to row 1399 and
differ from the third wheel's placing on. Recorded with

```bash
python scripts/roboarc/shared/rcb_topogym.py run-oracle --task clock_assembling --instance <instance.json> \
    --code tools/references/clock_assembling/oracle.py --out <dir>/privileged.npz
```

The oracle (`tools/references/clock_assembling/`: `oracle.py`, `clock_oracle.py`; the design's L1 oracle and
L3 controller) plans every move from the measured state and commands joint and gripper targets only: TCP poses per control
step through damped least-squares IK on a scratch copy of the world's model, branch changes in joint space high above the
scene, each target plus feedforwards of the drive's velocity lag, Coulomb friction and the held part's weight; a move
holds its commanded pose while an arm's commands lag their IK solution by more than 0.05 rad.
- Steps: the arms unfold from home in joint space; the left hand places the barrel (by its square), center, third and
  fourth wheel (lower until seated, turn half a leaf when a gate stops it); the right hand pinches the bridge across the
  middle of bridge_bar2 (between the center and third wheel holes, 4.9 mm from the bridge's centre of mass; closing toward
  the bridge's -y keeps the right arm on one IK branch), lays it on the pivot tips and lets go; if a pivot misses its hole, the left hand takes the probe and nudges it (pressing the rim's top or pushing
  its edge; a pivot caught under its entry cone's edge is eased while the right hand lifts the bridge 1.2 mm), else the
  probe stays on its stand; the right hand's open fingertips press the bridge flat; screw 1 by the left hand, screw 2 by
  the right (240 deg strokes, released and re-taken, until the rules count the screw tight); the left hand winds the
  barrel by its square, three 90 deg strokes (released and re-taken), lets go, lifts and withdraws to its park pose while
  the mainspring runs the train (the run is the mainspring's alone, and the cameras see the train turn).
- In this recording every pivot enters its hole as the bridge comes down (no probe). The arms follow their commanded TCP
  within 0.4 / 1.6 / 7.9 mm (left) and 0.0 / 1.2 / 11.7 mm (right; median / 90 % / 99 %), 31 mm at most (the unfolding),
  9.4 mm (left) and 15.7 mm (right, carrying the bridge) while holding a part; IK never misses; two joint-space moves (the
  unfoldings). The bridge pressed flat sits 0.00 mm / 0.12 mm / 0.0002 deg off its seat (height, position, tilt).
- The bridge pinch: the fingers' collision meshes stand 0.24 / 0.38 mm off the bar's sides and the load is carried by the
  fingertip spheres (each 0.6 mm, about 0.4 mm proud of the inner face): ALOHA's shared finger collision geometry meets a
  flat side with those spheres first, wherever it pinches. The pinch is 5.1 mm from the bridge's centre of mass; while
  carried, the weld's torque about the grasp axis is 0.0029 N m at p95 (the bridge's weight 4.7 mm off it), 2 % of what
  the four contacts hold by friction (sum mu F_i d_i, d_i each contact's distance from the axis through the grasp point
  along the fingers' line; caliper_sorting's audit). Every grasp of the reference stays under 1 at p95 (wheels and
  screws 0.14-0.26, winding 0.44-0.77); <design record>/watch_assembly_design_20261014/GRASP_AUDIT.md and its stills. Re-measured
  on the probe's hard contacts (birth 62e5c06433a1de4f, out/grasp_audit_b2/): the same contacts, forces and angles in
  all 16 grasps, the ratios within 0.03 (wheels and screws 0.14-0.23, winding 0.44-0.77).
- The run: the click holds 205 deg of wind when the window opens; the fourth wheel turns 12.8 times in the 5 s
  window, never under 10 % of its free-run speed, at -449.97 per drum turn; its mean speed is 117 % of the free-run
  speed; the left hand leaves the square and withdraws during the window, touching neither the drum nor a wheel.

Checks of the reference (2026-10-20, this host, a development image (0.3.2-rc1), mujoco 3.14, rendering
off; <design record>/watch_assembly_design_20261014/out/realism_ref_af683a9b21/, before the re-stamp: the state hashes there are of
the v2 model revision, the physics the same):
- The oracle's run: success at row 8047, final state f9e93ddbf4a3bec4, 296 s of wall clock.
- Two fresh-process strict replays (`rcb_topogym.py replay --repeat 1 --strict`, one process each): both success at row
  8047, final state f9e93ddbf4a3bec4, 253 / 267 s.
- The stress observer (`tools/stress/replay.py --hold 0`): CLEAN; no event, no review, no forcing; deepest task
  contact 0.7 mm (a held finger on screw 2); largest robot contact force 23.9 N (the bridge pinch); fastest part 31 rad/s
  (0.93 m/s at its surface: the arbor falling back onto the click); the barrel's arbor meets the bridge's ring boss at
  4.2 N while the bridge comes down; energy created 0.0058 J; 64 ms per control step.
- The equal-length nop (8047 rows of the birth targets, under the observer): no success, CLEAN.
- The mechanism's switches (<design record>/watch_assembly_design_20261014/scripts/stress/mech_trace.py): the wheels seat and mesh
  at rows 625 / 1224 / 1919 / 2590, screw 1 tightens at 5588, screw 2 and the bridge's fixing at 7188, the run test's
  window opens three times (rewinding restarts it) and completes at 8026; no switch reverses, the click's limit only
  moves up (148 moves), the run is never void.
- `tools/tests/test_clock_assembling.py`: reset() restores the birth (the click open, nothing seated) and the first
  1100 rows end on the same state twice; the nop seats nothing. `tools/tests/test_clock_rules.py`: its nine checks
  hold ("Realism, rules and the solid bridge").
- The arms and the drawn-only solids: at least 28.1 mm from the holder's ring, 61.3 mm from the posts' board
  (`<record>/scripts/realism/drawn_clearance.py`); the barrel boss's 17 mm post: the arms at least 28.0 mm from it (the right
  fingers taking the bridge), the bridge 4.6 mm once it is off its stand (lifting straight up from it;
  `<record>/scripts/realism/post_clearance.py`).
- Harbor (`release_rebuild.py 97a4e50f36 --families clock_assembling --bench robotangle`, the branch's own commit: dev's
  `mujoco_release.json` already lists the base; image a development image (0.3.0-dryrun-clock-assembling-v5), rendering off,
  2026-10-20, this host): PASS. The base made in the image is born ad4ee5041eeb19de (= the reference's), instance
  sha256 4f4e7d042a3fe8ce; oracle 1 and nop 0 in both modes, the oracle's end state 9683cab251ece2c0 in both, the
  evidence claims hold. The standard live episode took 319.3 s (0.040 s per control step), the privileged grader's two
  replays 668.8 s (<design record>/watch_assembly_design_20261014/harbor_test/work_v5/REPORT.md).
- Earlier references: the checks of 2026-10-18 (the probe's hard contacts, birth 62e5c06433a1de4f, end 955e65d624a1ce41;
  out/v2_reference_b2/) and the Harbor runs below are of the reference before these changes.
- Harbor (`release_rebuild.py 7a88493802 --families clock_assembling --bench robotangle`, a test-only commit of the
  temporary branch <test branch>: 5f71fc7943 plus the manifest entry of the base
  `clock-assembling-01-i01`; rendering off; 2026-10-18, an RTX 4090 host): PASS. The base made in the image is born
  62e5c06433a1de4f (= the reference's); oracle 1 and nop 0 in both modes, the oracle's end state 955e65d624a1ce41 in
  both, the evidence claims hold. The standard live episode took 224.5 s (0.027 s per control step), the privileged
  grader's replays 474.2 s (<design record>/watch_assembly_design_20261014/harbor_test/work_v4/REPORT.md). Before the probe's
  change (003470adb9, birth 2e43006feab1833b): PASS as well, the standard live episode 447.2 s (0.055 s) on a loaded
  host (work_v3/REPORT.md).

Calibration of the run test (`<design record>/watch_assembly_design_20261014/scripts/port/calib_run.py`: from the reference's state at
the run's start, the 5 s window again with extra friction on one wheel's spin hinge):
| run (from the reference's state at the window) | fourth's turns in 5 s | time under 10 % of free-run | verdict |
|---|---|---|---|
| as recorded | 12.8 | 0 | pass |
| center spin friction 1e-4 / 1e-3 N m | 12.3 / 7.5 | 0 / 0 | pass / pass |
| third spin friction 1e-5 / 3e-5 / 1e-4 / 1e-3 N m | 12.4 / 11.5 / 8.6 / 2.1 | 0 / 0 / 0 / 4.1 s | pass x3 / fail |
| fourth spin friction 1e-6 / 5e-6 / 2e-5 N m | 12.5 / 11.2 / 6.5 | 0 / 0 / 0.7 s | pass x2 / fail |

(The v3 reference, out/realism_ref_af683a9b21/calib.json; at 3.5 turns the verdicts are those at 5: the two binding
trains fail on their stall time.)

The free-run speed (the model's: 450 x (stiffness x windup - friction) / damping, reflected to the drum) is within 1 % of
the recorded median speed (14.8 rad/s at 189 deg of windup). Wound exactly 120 deg, a free train turns the fourth wheel
about 6-7 times in the window (the speed scales with the windup), so 5 turns leaves a margin; a train that binds (the third
or fourth wheel's spin friction 1e-3 / 2e-5 N m: 60 / 9 mN m at the drum, more than the mainspring's 6.6 mN m) stalls for
1-4 s. Re-run on the probe's hard contacts (out/v2_reference_b2/calib.json): every row the same. A train wound from rest
spins up for seconds (the metric review: wound fast until the click holds 125 deg, 4.2 turns in the window), hence 3.5
turns since 2026-10-19.

## The probe's contacts (2026-10-18)

The acceptance round clock_r2 (d4b3287848, seeds 121301-121309) failed on one stream: cem_e155, from the birth, swings
the left upper arm over the probe's stand and squeezes the probe (a 20 g rod, radius 3 mm) against its supports; the soft
robot x probe contact (the probe on the task class, mixed with the robot's) sank 4.3 mm at 200 N, and that pair created
energy (net work 1.74 J; energy_created(local): X up 0.43 J in 0.1 s, X + S 0.73 J). Shrunk to 51 controls + 40 hold
(the corpus reproducer W1_clock_upper_arm_probe_energy) and confirmed in two fresh processes. Trial fixes, each on a copy
of d4b3287848 (<design record>/watch_assembly_design_20261014/out/stress_clock_r2/trials_summary.json):

| trial | birth | reference end | reproducer x2 | the candidate |
|---|---|---|---|---|
| none | 2e43006feab1833b | e6ac83c093d69ec6 | BREAKDOWN | BREAKDOWN (1.74 J) |
| b1: the probe's collision geoms at priority 1 | unchanged | unchanged | CLEAN (the pair still 1.29 J) | BREAKDOWN (1.35 J) |
| b2: b1 + the hard impedance (solimp 0.99 0.999) | 62e5c06433a1de4f | 955e65d624a1ce41 (success) | CLEAN | CLEAN (0.003 J, 0.67 mm) |
| a: the probe 60 g | 0b9202ad581d8950 | a98305f3e2bc75fc (success) | BREAKDOWN | BREAKDOWN |
| b3: b2 + the stand's supports at priority 2 | 6ec746c1f37db9ad | c2a3d970aabbe426 (success) | CLEAN | CLEAN, but the probe x stand pair 1.53 J |

Moving the stand out of the arm's reach (kinematics only, probe_sweep2.json): out of the upper arm's sweep takes about
113 mm away from the left arm's base, but no place the left hand can use is out of the forearm's. The owner chose b2
(2026-10-18): `_probe` gives the probe's collision geoms priority 1 and the HARD_LIMIT impedance; the shared profile is
unchanged, and only this family's geometry module changes. It also stiffens the probe's rest on its stand, hence the new
birth; the reference's rows still succeed on it (re-stamped above). On it W1 replays CLEAN in two fresh processes (final
8d54d632d21b19af) and cem_e155 CLEAN.

The acceptance round clock_r3 on the fix (5f71fc7943, criteria v2.13b, an RTX 4090 host; the self-check, seeds 121401-121406,
PASS): **PASS**. Seeds 121501-121509, 1,000 episodes, 515,644 controls. The reference success, CLEAN, the same end
955e65d624a1ce41 in two processes, 8.1 min; the nop CLEAN; W1 CLEAN in both runs. In search 15 VIOLENT and no BREAKDOWN:
12 a part thrown (surface over 3 m/s, or 4.2 m/s), 3 a robot force over 2 kN; those three confirmed, no BREAKDOWN under
any reading v2 to v2.13b. Two of them are the left upper arm striking the probe again (the probe leaves at the arm's
speed, 0.51 / 0.59 m/s, then the arm presses the stand at 2.3 kN) and create no energy; the third is the left arm's own
links. The mechanism (mech_round.json): prefix episodes from every phase, the 18 after the bridge's fixing all CLEAN;
the largest net work at a movement equality 0.0015 J (mesh_center). <design record>/watch_assembly_design_20261014/out/stress_clock_r3/.


## Realism, rules and the solid bridge (2026-10-19/20)

One change, decided by the owner piece by piece (the realism list, the rules and geometry change (b) on 2026-10-19; the
barrel boss's wider post and the arbors' lower collision tops on 2026-10-20), on <test branch> (from dev
a0856f6b32): 56ee994442, 5c195ffa69, 2a2e34ab5f, af683a9b21, and the model revision clock_gear_train_v3 (7698adb5e6:
the collision model changed). The reference above is recorded on af683a9b21, once, and re-stamped with the birth of the
model revision (its rows the same).

The movement drawn more like a real one (56ee994442; the realism list, <design record>/watch_assembly_design_20261014/REALISM_LIST.md:
the cam_high stills had shown a pink plastic disc). Drawn geoms only: a copper-gilt plate and a paler gilt bridge (one hue
had hidden the bridge on the plate) with polished bevels, the wheels crossed out (rim, spokes, hub) over their unchanged
collision discs, darker steel with polished rings on the barrel's cover and the ratchet, deep-red jewels in gilt settings,
a few of a real plate's jewels and screws, a grey movement holder's ring under the plate's edge, smooth cups on the wheel
stand, the bridge posts in one board, blued screws, a steel probe with a wooden handle; the red spin marks gone. Rings
drawn by boxes showed facet by facet; they are inline MJCF meshes. The MuJoCo scene drops drawn geoms, so the physics is
the same bit for bit: the reference of the time replayed on a0856f6b32 and on 56ee994442, every control step's (qpos,
qvel, act) and the rules' readout equal at all 8171 steps, the same birth and end; the instance differs only in its
asset identity (out/realism_bitwise/compare_visual.json). The arms pass the holder's ring at 28 mm at least.

The rules, after the metric review (5c195ffa69; <design record>/robotangle_mujoco/metric_review_20261019/clock_zh.md):
- the fixing waits for the four wheels seated and every upper pivot in its bridge hole: a bridge screwed down before the
  wheels used to fix, switch the pivot holes off and let the wheels be dropped through it within 0.6 mm (the review's
  finding 1);
- either screw counts in either hole (screw1 counted only in P1 and screw2 only in P2: swapped, they never tightened;
  finding 2);
- a head counts from 0.3 mm below the bridge's top (a head resting or pressed on it sinks microns into the soft contact
  and counted nothing; the reference only tightened because the assisted grasp lifted its screws by tens of microns;
  finding 3);
- the run needs 3.5 turns, not 5 (finding 5);
- the run counts for the mainspring alone: no robot or probe contact with the drum or a wheel from the last winding to the
  window's end, the drum turning the mainspring's way (a jammed train turned by hand passed; finding 4).
The goal's words say each (`goal_words`, terminal revision clock_gear_train_v3), and that the square is let go after
winding (finding 11); the package texts are regenerated from them. The oracle's left hand withdraws to its park pose in
the run.

The bridge solid to the wheels (2a2e34ab5f, af683a9b21; the review's geometry change (b)). With the rules alone a wheel
released within 0.6 mm of its hole over a bridge laid (not yet fixed) on its pillars fell through it, seated, the bridge
then fixed and the run passed (test_clock_rules.py on 5c195ffa69, out/realism_rules/quick1/: 0, 0.2, 0.4, 0.6 mm seated,
1 and 2 mm stopped by the hole's pad; all three wheels over their holes: fixed, success at step 241). Now:
- the bridge and the wheels collide (BRIDGE and WHEEL in each other's affinity);
- the barrel boss collides as a ring (16 boxes) with the drawn 12.5 mm opening, and the two bars to it from the opening
  on: the ratchet (9.5 mm) and the square pass through as the bridge comes down;
- the barrel boss's post on the bridge stand is 17 mm in radius, drawn as it collides (the owner's decision of 2026-10-20):
  with the ring the 8 mm post went
  up through the opening, the bridge lay on three posts, and the birth and everything after changed (the earlier
  reference failed on it); on the wider post the ring rests as the solid boss did;
- the arbors collide up to ARBOR_TOP, 1.2 mm under the seated bridge (0.4 mm before; the owner's decision of 2026-10-20):
  the seat rule seats a wheel up to
  0.8 mm high and its weld keeps it there, and the oracle's wheels seat at 0.78-0.79 mm; with the 0.4 mm end shake the
  center and third arbors reached the bridge's bars (0.01-0.09 N), the bridge sat 0.32 mm high and tilted 0.24 deg, and
  their friction stopped the train (the fourth wheel 0.006 turns in the window; the mainspring gives it about 1.5e-5 N m;
  out/realism_ref_2a2e34ab5f/run_contacts.json). The arbors are drawn up to 0.4 mm under the bridge as before, and the
  upper pivots are unchanged.

`tools/tests/test_clock_rules.py` (parts put in place by their free joints, the arms at their birth pose) checks the
change, on af683a9b21 (out/realism_ref_af683a9b21/rules_test.json):
1. bridge first, both screws tight: not fixed;
2. a screw resting on the bridge, turned 120 deg in place: 0.333 turns counted;
3. the screws swapped: both tight (2.03 turns each);
4. the right order: fixed;
5. from rest, wound fast until the click holds 125.5 deg: 4.16 turns, success;
6. the probe laid against the drum after the winding: void, no success; the probe away and wound again: success;
7. a binding train (the third wheel's spin friction 1e-3 N m) with the drum driven backwards: void, no success (the drum
   driven forwards without a contact is not caught);
8. the center wheel released just above the unfixed bridge (its lower pivot 2 mm over the bridge's top), 0, 0.2, 0.4 or
   0.6 mm off its hole: it rests on the bridge (59.2 mm up), not seated; all three wheels over their holes: none seated,
   the bridge not fixed;
9. every wheel seated 0.74 mm high, the bridge on its pillars and screwed: it sits flat (0.001 mm, 0.0002 deg), the run
   passes (5.3 turns, no stall).

The acceptance round clock_r4 on the final commit (97a4e50f36, criteria v2.13b, an RTX 4090 host; the self-check, seeds
121601-121606, PASS): **PASS**. Seeds 121701-121709, 1,000 episodes, 506,855 controls. The reference success, CLEAN, the
same end in two processes, 7.3 min; the nop CLEAN; W1 CLEAN. In search 26 VIOLENT and no BREAKDOWN: 21 a part thrown
(surface over 3 m/s), 3 a robot force over 2 kN, 2 a wrist over its speed; the two confirmed (prefix2_e39: the left
upper arm on a stand at 2.35 kN; prefix4_e9: the right arm's own links at 2.34 kN, the right fingers knocking screw 2 off
at 1.1 m/s) have no BREAKDOWN under any reading v2 to v2.13b. The mechanism (mech_round.json): prefix episodes from every
phase, the 26 started after the bridge's fixing 23 CLEAN and 3 VIOLENT; in one (prefix2_e36, cut in the run) the left
fingers press the drum at 167 N and drive the train, the fourth wheel at 549 rad/s (VIOLENT for its surface speed; the
fourth wheel's seat weld 0.024 J of net work, energy created 0.0058 J). <design record>/watch_assembly_design_20261014/out/
stress_clock_r4/.
