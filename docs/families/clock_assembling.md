# clock_assembling on MuJoCo (a watch movement's gear train)

Jianwen, 2026-10-16 (v2 2026-10-17). A new family, MuJoCo only: two ALOHA arms assemble the gear train of a watch movement
enlarged 8 times, then wind it and watch it run. First setting `clock_assembling_gear_train` (pool experimental, no tier;
model revision `clock_gear_train_v2`). Design record, reference video and every measurement:
`runs/watch_assembly_design_20261014/` (README_v3.md, REGISTRATION_PROPOSAL.md, RUN_TEST_PROPOSAL.md; not in git).

## The task

The user's reference is the gear-train part of BV14bAjzEEcA (0:31-1:12, an IWC calibre 81925): the barrel, then the large,
the small and the steel wheel lowered onto their pivots, the bridge laid over all of them while a finger holds it and
tweezers nudge the pivots into its holes, the screws. The video shows no run of the train (1:13-1:20 is another wheel slid
in tilted under a placed wheel's rim, on the dial side); the run test (wind, let go, watch the fourth wheel) follows the
written sources' "check that the train runs free" and the winding works the video shows later (a ratchet wheel screwed on
the barrel arbor, a click; 3:46-4:26). The scene (`topogym/assets/geometry/clock_gear_train.py`):

- a brass main plate 304 mm across (38 mm real) on a holder, its top 10 cm above the table; four lower pivot holes, three
  pillars (two with screw holes), two steady pins;
- on a stand: the barrel (90 teeth; a drum and its arbor with the ratchet and the square), the center wheel (80 + a 12-leaf
  pinion), the third wheel (75 + 10) and the fourth wheel (70 + 10), axes vertical, each turned in its cup by scene
  seed 0's draw for every seed (a seed varies appearance only, the user 2026-10-20: `movement_finish`, `stand_finish`,
  `screw_finish` re-colour visual geoms only); the
  train bridge (six 20 mm bars, 12 mm thick, 60 g; a click drawn by the barrel's boss) on four posts; two screws in a tray; a
  9 cm probe on its stand. The layout is the design's after its base placement search (the whole scene moved by (-0.03,
  -0.06) m).
- Each wheel's disc lies 1 mm over the disc (the barrel: its drum) it overlaps in plan (0.125 mm real): with the 4 mm lower
  pivots no wheel can be slid in under a wheel already seated, so the train goes together in one order only (the order
  check below). A wheel standing on its lower pivot leans up to 1.5 deg per axis.

Collision: one MJCF entity, `clock`, the movement, stands and every part (its links; the barrel's arbor is the link
`barrel_arbor`, a child of the barrel); one entity because its collision classes act only inside an entity (Genesis' MJCF
semantics, `mjcf_entity.py`) and its equalities join bodies of one entity. Ten classes keep two meshing wheels from touching
(discs are root circles, teeth visual only), let the pivots meet only holes, the shanks only screw holes. Against the robot
and the table every geom collides.

The rules are the task's mechanism (`topogym/tasks/clock_assembling/mechanism.py`, `TaskDefinition.mujoco_mechanism`;
the interface is bed_bolt's, docs/rcb/MUJOCO_CORE.md "Task mechanisms"), once per 2.5 ms tick:
- seat: a wheel whose lower pivot is down in its hole (within 0.6 mm of the axis, at most 0.8 mm high, tilt < 3 deg) is
  welded to its carrier (two +-1.5 deg lean hinges and a spin hinge at the pivot; inactive weld at birth);
- gate: while a wheel's pinion meets the previous wheel's teeth tip on tip (< 0.2 of a pitch), a short cylinder at its hole
  mouth collides with the wheel (switched on only where nothing overlaps it);
- mesh: two seated wheels are coupled by a joint equality between their spin hinges (polycoef[1] = the ratio, -7.5, -8,
  -7.5; the constant puts the leaves in the gaps);
- screws: a screw seated on the bridge counts its clockwise turns relative to the bridge; at 2 it is welded to it;
- fix: both screws tight and the bridge seated (0.5 mm, 0.5 deg): the bridge welded to the world, the lean hinges locked,
  the pivot-hole contacts off; the barrel's arbor welded to its own carrier (the click's hinge first set to the arbor's
  turn) and the mainspring's lock released;
- click: the click hinge's lower limit follows the largest angle the arbor reached less one ratchet tooth (18 deg, 20
  teeth): the arbor winds counterclockwise (seen from above) only;
- run test: once the arbor has not advanced for 0.5 s after a wind of at least 10 deg, a 5 s window measures the train
  running on the mainspring: the fourth wheel's turns, the time it spends below 10 % of its free-run speed (the speed the
  mainspring's present windup gives against the train's damping and friction, from the model) and its turn per drum turn;
  winding again restarts the window.

The mainspring is the hinge between the drum and the arbor (stiffness 0.002 N m/rad, damping 1e-5; held rigid by the
joint equality `spring_lock` until the fixing, so the barrel is carried as one part); the click hinge has damping 1e-3 (a
released arbor falls back onto the click at a few rad/s).

Goal (`ClockAssemblingTask.evaluate_state`, terminal revision `clock_gear_train_v2`): the four wheels seated (within 1 mm),
every upper pivot in its bridge hole (at least 1 mm deep), the bridge seated (0.5 mm, 0.5 mm, 0.5 deg) and fixed, the three
meshes coupled, and the run test: wound at least 120 deg (as the click holds it when the window opens), at least 5 turns of
the fourth wheel in the window, at most 0.2 s below 10 % of its free-run speed, its turn per drum turn -450 within 2 %;
held for 0.5 s, holding allowed.

## The order check

`runs/watch_assembly_design_20261014/scripts/order/` (results `out/order_check_v2/`): part motion only, on this collision
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
through goes back to the design. Re-run 2026-10-18 after the probe's contact change (runs/watch_assembly_design_20261014/out/
order_check_b2/): the 32 path steps and the six floods identical to out/order_check_v2/ in every field but the wall clock.

## Reference (-i01)

`scripts/robotangle/oracles/clock-assembling-01-i01/privileged.npz` (`standard.npz` is the same file), 8171 rows
(204.3 s), native gripper space, on the instance `rcb_topogym.py make-instance --task clock_assembling --variation
clock_assembling_gear_train --scene-seed 0 --gripper-space native` (birth 62e5c06433a1de4f). sha256 411d2db636bc396c. No row moves an arm
joint more than 0.035 rad. Re-recorded 2026-10-18 with the bridge pinched across the middle of its center-third bar (the
grasp audit, runs/watch_assembly_design_20261014/GRASP_AUDIT.md: the earlier pinch held the bridge by the end of one bar,
57 mm from its centre of mass), on the birth 2e43006feab1833b (end e6ac83c093d69ec6); then re-stamped the same day with
the birth of the probe's hard contacts (below, "The probe's contacts"): `instance_sha` rewritten, the 8171 rows and the
tick ledger byte for byte the same (runs/watch_assembly_design_20261014/scripts/restamp.py, as bed_bolt's). Recorded with

```bash
python scripts/robotangle/shared/rcb_topogym.py run-oracle --task clock_assembling --instance <instance.json> \
    --code scripts/robotangle/references/clock_assembling/oracle.py --out <dir>/privileged.npz
```

The oracle (`scripts/robotangle/references/clock_assembling/`: `oracle.py`, `clock_oracle.py`; the design's L1 oracle and
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
  the right (240 deg strokes, released and re-taken); the left hand winds the barrel by its square, three 90 deg strokes
  (released and re-taken), lets go and lifts; the mainspring runs the train.
- In this recording every pivot enters its hole as the bridge comes down (no probe). The arms follow their commanded TCP
  within 0.4 / 1.5 / 7.9 mm (left) and 0.0 / 1.2 / 11.7 mm (right; median / 90 % / 99 %), 31 mm at most (the unfolding),
  9.4 mm (left) and 15.7 mm (right, carrying the bridge) while holding a part; IK never misses; two joint-space moves (the
  unfoldings).
- The bridge pinch: the fingers' collision meshes stand 0.24 / 0.38 mm off the bar's sides and the load is carried by the
  fingertip spheres (each 0.6 mm, about 0.4 mm proud of the inner face): ALOHA's shared finger collision geometry meets a
  flat side with those spheres first, wherever it pinches. The pinch is 5.1 mm from the bridge's centre of mass; while
  carried, the weld's torque about the grasp axis is 0.0029 N m at p95 (the bridge's weight 4.7 mm off it), 2 % of what
  the four contacts hold by friction (sum mu F_i d_i, d_i each contact's distance from the axis through the grasp point
  along the fingers' line; caliper_sorting's audit). Every grasp of the reference stays under 1 at p95 (wheels and
  screws 0.14-0.26, winding 0.44-0.77); runs/watch_assembly_design_20261014/GRASP_AUDIT.md and its stills. Re-measured
  on the probe's hard contacts (birth 62e5c06433a1de4f, out/grasp_audit_b2/): the same contacts, forces and angles in
  all 16 grasps, the ratios within 0.03 (wheels and screws 0.14-0.23, winding 0.44-0.77).
- The run: the click holds 208 deg of wind when the window opens; the fourth wheel turns 12.3 times in the 5 s
  window, never under 10 % of its free-run speed, at -449.97 per drum turn; its mean speed is 111 % of the free-run
  speed.

Checks of the reference on the probe's hard contacts (2026-10-18, a lab RTX 4090 host, ghcr.io/mll-lab-nu/eai-robotangle:0.3.2-rc1,
mujoco 3.14, rendering off; runs/watch_assembly_design_20261014/out/v2_reference_b2/):
- The oracle's run (on the earlier birth): success at row 8171, final state e6ac83c093d69ec6, 300 s of wall clock.
- Two fresh-process strict replays of the re-stamped reference (`rcb_topogym.py replay --repeat 1 --strict`, one process
  each): both success at row 8171, final state 955e65d624a1ce41, 216 s each.
- The stress observer (`scripts/robotangle/stress/replay.py --hold 0`): CLEAN; no event, no review, no forcing; deepest task
  contact 0.7 mm (a held finger); largest contact force 23.9 N; fastest part 31 rad/s (0.93 m/s at its surface: the
  arbor falling back onto the click; without the click's damping it was 113 rad/s and VIOLENT); energy created 0.0058 J;
  57 ms per control step.
- The equal-length nop (8171 rows of the birth targets, under the observer): no success, CLEAN.
- The mechanism's switches along the reference (runs/watch_assembly_design_20261014/scripts/stress/mech_trace.py): the
  wheels seat and mesh at rows 625 / 1224 / 1919 / 2590, screw 1 tightens at 5649, screw 2 and the bridge's fixing at 7297,
  the run test's window opens three times (rewinding restarts it) and completes at 8150; no switch reverses, and the
  click's limit only moves up (146 moves).
- `scripts/robotangle/test_clock_assembling.py`: reset() restores the birth (the click open, nothing seated) and the first
  1100 rows end on the same state twice; the nop seats nothing.
- Harbor (`release_rebuild.py 7a88493802 --families clock_assembling --bench robotangle`, a test-only commit of the
  temporary branch jianwen/clock-assembling-harbor-test4: 5f71fc7943 plus the manifest entry of the base
  `clock-assembling-01-i01`; rendering off; 2026-10-18, a lab RTX 4090 host): PASS. The base made in the image is born
  62e5c06433a1de4f (= the reference's); oracle 1 and nop 0 in both modes, the oracle's end state 955e65d624a1ce41 in
  both, the evidence claims hold. The standard live episode took 224.5 s (0.027 s per control step), the privileged
  grader's replays 474.2 s (runs/watch_assembly_design_20261014/harbor_test/work_v4/REPORT.md). Before the probe's
  change (003470adb9, birth 2e43006feab1833b): PASS as well, the standard live episode 447.2 s (0.055 s) on a loaded
  host (work_v3/REPORT.md).

Calibration of the run test (`runs/watch_assembly_design_20261014/scripts/port/calib_run.py`: from the reference's state at
the run's start, the 5 s window again with extra friction on one wheel's spin hinge):
| run (from the reference's state at the window) | fourth's turns in 5 s | time under 10 % of free-run | verdict |
|---|---|---|---|
| as recorded | 12.3 | 0 | pass |
| center spin friction 1e-4 / 1e-3 N m | 11.8 / 7.0 | 0 / 0 | pass / pass (slowing to 1.8 rad/s at the window's end) |
| third spin friction 1e-5 / 3e-5 / 1e-4 / 1e-3 N m | 11.9 / 11.0 / 8.1 / 2.0 | 0 / 0 / 0 / 4.1 s | pass x3 / fail |
| fourth spin friction 1e-6 / 5e-6 / 2e-5 N m | 12.0 / 10.8 / 6.0 | 0 / 0 / 1.0 s | pass x2 / fail |

The free-run speed (the model's: 450 x (stiffness x windup - friction) / damping, reflected to the drum) is within 1 % of
the recorded median speed (14.8 rad/s at 189 deg of windup). Wound exactly 120 deg, a free train turns the fourth wheel
about 6-7 times in the window (the speed scales with the windup), so 5 turns leaves a margin; a train that binds (the third
or fourth wheel's spin friction 1e-3 / 2e-5 N m: 60 / 9 mN m at the drum, more than the mainspring's 6.6 mN m) stalls for
1-4 s. Re-run on the probe's hard contacts (out/v2_reference_b2/calib.json): every row the same.

## The probe's contacts (2026-10-18)

The acceptance round clock_r2 (d4b3287848, seeds 121301-121309) failed on one stream: cem_e155, from the birth, swings
the left upper arm over the probe's stand and squeezes the probe (a 20 g rod, radius 3 mm) against its supports; the soft
robot x probe contact (the probe on the task class, mixed with the robot's) sank 4.3 mm at 200 N, and that pair created
energy (net work 1.74 J; energy_created(local): X up 0.43 J in 0.1 s, X + S 0.73 J). Shrunk to 51 controls + 40 hold
(the corpus reproducer W1_clock_upper_arm_probe_energy) and confirmed in two fresh processes. Trial fixes, each on a copy
of d4b3287848 (runs/watch_assembly_design_20261014/out/stress_clock_r2/trials_summary.json):

| trial | birth | reference end | reproducer x2 | the candidate |
|---|---|---|---|---|
| none | 2e43006feab1833b | e6ac83c093d69ec6 | BREAKDOWN | BREAKDOWN (1.74 J) |
| b1: the probe's collision geoms at priority 1 | unchanged | unchanged | CLEAN (the pair still 1.29 J) | BREAKDOWN (1.35 J) |
| b2: b1 + the hard impedance (solimp 0.99 0.999) | 62e5c06433a1de4f | 955e65d624a1ce41 (success) | CLEAN | CLEAN (0.003 J, 0.67 mm) |
| a: the probe 60 g | 0b9202ad581d8950 | a98305f3e2bc75fc (success) | BREAKDOWN | BREAKDOWN |
| b3: b2 + the stand's supports at priority 2 | 6ec746c1f37db9ad | c2a3d970aabbe426 (success) | CLEAN | CLEAN, but the probe x stand pair 1.53 J |

Moving the stand out of the arm's reach (kinematics only, probe_sweep2.json): out of the upper arm's sweep takes about
113 mm away from the left arm's base, but no place the left hand can use is out of the forearm's. The user chose b2
(2026-10-18): `_probe` gives the probe's collision geoms priority 1 and the HARD_LIMIT impedance; the shared profile is
unchanged, and only this family's geometry module changes. It also stiffens the probe's rest on its stand, hence the new
birth; the reference's rows still succeed on it (re-stamped above). On it W1 replays CLEAN in two fresh processes (final
8d54d632d21b19af) and cem_e155 CLEAN.

The acceptance round clock_r3 on the fix (5f71fc7943, criteria v2.13b, a lab RTX 4090 host; the self-check, seeds 121401-121406,
PASS): **PASS**. Seeds 121501-121509, 1,000 episodes, 515,644 controls. The reference success, CLEAN, the same end
955e65d624a1ce41 in two processes, 8.1 min; the nop CLEAN; W1 CLEAN in both runs. In search 15 VIOLENT and no BREAKDOWN:
12 a part thrown (surface over 3 m/s, or 4.2 m/s), 3 a robot force over 2 kN; those three confirmed, no BREAKDOWN under
any reading v2 to v2.13b. Two of them are the left upper arm striking the probe again (the probe leaves at the arm's
speed, 0.51 / 0.59 m/s, then the arm presses the stand at 2.3 kN) and create no energy; the third is the left arm's own
links. The mechanism (mech_round.json): prefix episodes from every phase, the 18 after the bridge's fixing all CLEAN;
the largest net work at a movement equality 0.0015 J (mesh_center). runs/watch_assembly_design_20261014/out/stress_clock_r3/.

