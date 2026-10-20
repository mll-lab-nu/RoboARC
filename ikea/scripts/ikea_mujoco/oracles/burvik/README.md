# burvik closed-loop reference

`burvik.npz` is the one reference for both modes of `ikea-burvik-i00` (privileged `solution/trajectory.npz`, standard
`solution/oracle_actions.npz`). Its rows are the EXECUTED targets (`data.ctrl` after `rcb_ikea.Session.applied()`): every
row is within the arm joint-speed limits and the Franka Hand slew (0.1 m/s of finger width: 182.142857 command units/s
in this scene, 0.4554 per tick) (+ the 1e-9 tolerance) of the previous row, from the keyframe ctrl, so the clamp never
changes a row on replay. Keys: `ctrl` (162534, 16) float64, `substeps` 4, `timestep` 0.000625, `actuators`.

| | |
|---|---|
| rows | 162,534 (406.3 s of simulated time) |
| sha256 | `6445eb0f7dc19ca29bc093fd629419d1694e7201802a9c511ac27244fffb44cf` |
| rows over the speed limits (arms and hands) | 0; `slewed_ticks` 0 |
| initial state | `d563d833bd219a22…` = bindings.json (the frozen birth, loaded unchanged) |
| end / settled state sha | `60878a5edc5d187a…` / `02732bf5d33e824b…` in every replay below |
| verdict | success: legs 0-3 and the handle welded and on their assembled poses, the top upright; released, at rest, no breakdown (max part contact depth 0.631 mm); the same after the 1.0 s hold |

Only the robot's actions differ from the author's demo; the scene, `success.py`, the mechanism, the gates and the physics
(prism rule, the joint-speed clamp and the Franka Hand slew, MuJoCo 3.14.0 + the EPA byte patch, SIMD masking) are
untouched.

**Under the hand slew** (rcb_ikea.GRIPPER_WIDTH_SPEED_M_S; the 2026-10-18 rebuild; the earlier reference without it was
145,134 rows, `9a3335c7…`): `orc.Orc.step` holds the arms at their last targets while a hand's command is more than
`BURVIK_GRIP_WAIT` = 2 ticks of its slew from the row's (a grasp or a release completes before the arm goes on; a demo
ramp within two ticks is not held), and P6 ends differently: the table is set down level on its feet before the hands
open, and the hands leave it without dragging it (below). Everything else is the same builder.

## Why the earlier attempt stopped at leg_3, and the fix
The base rows (`burvik_p2cl.npz`, sha256 `71f97f3b…`: the standard agent's retimed demo, within the speed limits)
release the top about 25 mm above the jig, its rim behind the guards' front faces. It lands on the guards' tops and
leans 6 deg forward onto the posts. Then the four legs slope down, and the hand at leg_3's far end (the only place it can
hold the leg) sweeps into `handle_rack_b_base` while spinning: about 90 N, the hand shoved 14 mm, the leg out of the nut.

The jig has a proper seat: the slot between the guards (behind) and the posts (in front), rim on the two pads. A top set
there settles at rest, leaning 0.5 deg (`probe/seat2.py` in the dev folder). `seat.py` puts it there: from base row
64500 both arms carry the top as one rigid body (the top's pose error -> its twist -> each TCP's rigid twist -> damped
least squares), align over the slot, lower at 2 mm/s, open, and the base retreat follows, shifted by the arms' offset and
ramped out. The top ends at (-0.1594, 0, 0.3919), leaning 0.51 deg. With the legs level, leg_3 threads with the
earlier controller unchanged.

## Phases (`build2.py` = `build.py` with P1 and P6 replaced)
1. **P0** base rows 0-64000: legs 0/1 threaded into the upside-down top, the top turned on its side.
2. **P1** `seat.py` (above).
3. **P2** base rows 70000-78500 (the right hand takes the handle).
4. **P3** the handle: right-arm servo in the top's frame, 30 mm out on the axis, in at 2 mm/s; seated (gap 0.006 mm).
5. **leg_2, leg_3** `build.leg_force`: the leg's frame servoed onto the op axis, then the HAND servoed (TCP on its line,
   the j7 axis on its direction), j7 spun at -2.2 rad/s with a bounded 0.2 mm axial lead, zero torque at the seat
   until the weld; the right arm's handle hold lead-bounded to 4 mm. leg_2 welds after 13,019 ticks, leg_3 after
   12,962; the handle's weld follows.
6. **P6** the upright turn, `upright4.py`. The demo turns the table with the left hand on leg_2 next to the top and the
   right hand on the handle's crossbar, both hands moving as ONE rigid body (the right hand's pose in the left hand's
   frame is constant to 0.0 mm). As joint rows it fails here: the left grasp is 18 mm off leg_2's centre (one finger
   pushes the leg) and the right arm holds the handle in another IK branch (joint offsets up to 1.4 rad from the
   demo's) with its elbow on the top: the table does not lift and both grasps slip. So:
   - `reconf_right`: the right hand opens, backs out 60 mm, moves in joint space to the demo-branch IK of the same
     point with the demo's hand orientation there, goes in and closes;
   - `upright3.grasp`: the left hand by IK onto leg_2 at world x -0.058, centred, pointing down; closed (71 / 71 N);
   - `upright_move_rigid`: the demo's rigid motion of base rows 121000-130400 (forward kinematics of its left TCP),
     applied to OUR two hand poses, joints by joint-limited IK every 25 rows seeded with the demo's joints, linearly
     interpolated, the take's load offset kept; no feedback (`BURVIK_UP4_KC=0`);
   - `place_ik`: the table, hanging about 45 mm over the bench and tilted 9 deg after the turn (the grasps creep), is
     levelled while it moves horizontally to where its feet are clear of the stand's posts (`feet_shift`: -60, -30 mm),
     then lowered at 5 mm/s until a leg touches the bench; both arms re-solved by IK every 25 ticks for that path
     composed with the grasps as they are then (so the table, not the hands, follows it): it stands at 0.3 deg;
   - `release_home`: hold, open both hands (while the hands open slowly the standing table is pushed about 40 mm sideways
     and its centre lifted 12 mm; it settles back on its feet); the right hand, whose open fingers still hook the
     handle's crossbar, moves 40 mm down first (`BURVIK_UP4_RDIR=down`, `RTICKS=400`: up, sideways or straight back lift or drag the table);
     each hand backs out 100 mm along its approach axis; the left hand goes out radially to 0.36 m from the table's
     axis and round to its home azimuth (`polar_retreat`); joint blend to the base rows' last row.

The reference is ONE builder run from the birth (a from-scratch rebuild is bit-identical: sha256 `6445eb0f…` both
times), base `burvik_p2cl.npz`:

    BURVIK_GRIP_WAIT=2 BURVIK_HOLD_LEAD_ALL=0.004 BURVIK_HOLD_LEAD=0.004 BURVIK_HOLD_LEAD3=0.004 BURVIK_LIFT=0 \
    BURVIK_SPIN=-2.2 BURVIK_PRESS=0.0002 BURVIK_OUT_DX=0.05 BURVIK_TRANSIT_RATE=1.0 \
    BURVIK_UP=4 BURVIK_UP4_JOINT=0 BURVIK_UP4_KC=0 BURVIK_UP4_STRIDE=1.0 BURVIK_UP_XL=-0.058 \
    BURVIK_UP4_PLACE=ik BURVIK_UP4_RDIR=down BURVIK_UP4_RTICKS=400 BURVIK_UP4_RETREAT=polar \
    python build2.py --code <repo> --scenes <scenes> --base burvik_p2cl.npz --out DIR --to p6     # -> DIR/ref.npz

(`move_from_take.py` and `release_from_placed.py` rerun P6 from a `snap_p6_take.pkl` / its release from a
`snap_p6_placed.pkl`, for experiments.)

## What did not work for P6 (kept for the record)
- `upright.py`: a Cartesian rigid servo of both hands; with two top-down grasps on legs 0 and 2 it lifts within
  0.3 mm, but halfway through the turn the left arm reaches its joint limits (q2 -1.76, q4 -3.04).
- `upright3.py` + `plan_up.py`: joint-limited, collision-checked IK plans of grasp pairs on legs 0/2 (top-down, tilted,
  side grasps). The plans are feasible, but the table pivots in the grasps (the gravity torque acts about the pinch
  axis of both hands) and creeps 30-130 mm off the plan; integral feedback winds the arms into collisions.

## Gates (the hand-slew reference)
1. Frozen birth: every replay builds to the bound `initial_state_sha256` (`d563d833…`).
2. Fresh replays (`check_ref.py`: `rcb_ikea` Session with the hand slew, clamp in, `final_check()`), all success with the
   hashes above, 0 rows over any arm or hand limit, `slewed_ticks` 0:
   - AMD Zen 4: host A (Ryzen 9 7950X, image 0.3.0, Docker) x2: 1684 / 1679 s;
   - Intel AVX2: the quiet i9-14900HX host (MuJoCo 3.14.0 from PyPI with the EPA byte patch, under the workstation lock): 1694 s;
   - Intel AVX-512: the cluster Xeon Gold 6230R (SIF 0.3.0) x2: 3781 / 3783 s, and the stress checkpoint run.
   A from-scratch rebuild (the command above) gives the same file (`6445eb0f…`).
3. Harbor on the final dirs (host A, image 0.3.0 / -sim 0.4.0 Nyx): privileged oracle 1 (deterministic, both replays
   `60878a5e…`, replay_wall 1646 s, video), nop 0; standard oracle 1 (live settled success at control step 160,001 of
   162,534, 40 calls, live end `ebe8cde2…` on both runs, service time 1824-1986 s, video), nop 0.
4. Timing (one execution against the 4320 s bar): 3781-3783 s on a dedicated the cluster Xeon Gold 6230R core (87.6 % of the
   bar, 35 % of the budget); 1694 s (15.7 %) on the quiet i9-14900HX; 1679-1684 s on host A.
5. Leak check: `environment/` and `tests/` hold only the tooling; neither the rows nor their sha appear in them or in the
   instructions; no upstream pointer in the scene texts. `check-video`: both dirs pass.
6. Stress: 203 episodes on the cluster (Xeon Gold 6230R, SIF 0.3.0, the IKEA harness with RoboARC's observer v2.13b, the
   hand-slew rcb_ikea), 29 per strategy, seeds 990000-990202, 3-10 s each, from 21 checkpoints of THIS reference (every
   5 %; the checkpoint run itself ended on `60878a5e…`, success).

   | strategy | n | CLEAN | VIOLENT | BREAKDOWN |
   |---|---|---|---|---|
   | random | 29 | 22 | 7 | 0 |
   | prefix_random | 29 | 5 | 22 | 2 |
   | prefix_jerk | 29 | 2 | 27 | 0 |
   | mutate | 29 | 27 | 2 | 0 |
   | fast | 29 | 29 | 0 | 0 |
   | grip | 29 | 29 | 0 | 0 |
   | push | 29 | 29 | 0 | 0 |
   | total | 203 | 143 | 58 | 2 |

   - Non-jerk BREAKDOWN 2 / 174 = 1.15 %, prefix_jerk 0 / 29 (without the slew: 7 / 174 and 3 / 29, 68 CLEAN; the slew
     removes the finger slam, the 11.6 kN finger-finger force that made almost every gripper episode VIOLENT).
   - The two: energy created at the left finger joint's limit (prefix_random s990134) and an "uncaused" leg speed jump
     (s990183, a leg turning in its SDF thread), both under random arm targets from a checkpoint;
     both reproduced from the birth in a fresh process (same verdict, same end hash). No physics was changed.

## Files
- `build.py`, `orc.py`, `servo.py`: the earlier builder (phases, leg threading, handle servo; unchanged).
- `build2.py`: build.py with `seat.py`'s P1 and `upright4.py`'s P6 (`BURVIK_UP=3`: `upright3.py`).
- `seat.py`, `upright4.py`, `move_from_take.py` (P6 from a `snap_p6_take.pkl`): the reference's new phases.
- `upright.py`, `upright3.py`, `plan_up.py`: helpers used by upright4 and the P6 attempts above.
- `check_ref.py`: fresh-process replay and `final_check`. `snap2ref.py`: a snapshot's rows as an npz.
- `clearance.py`, `exp_*.py`, `probe_handle.py`: the earlier probes (leg_3 clearance, rack, handle).
