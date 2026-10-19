# interlocked_claws redesign (5 tiers, MuJoCo): design note

Agent "claws", 2026-10-10/11 (rigid redesign, steps 3-4; brief the work notes). Working
folder the work notes (STATUS.md, review packets in `review/<tier>/`). This note goes with the
code; the per-tier evidence is in the review packets.

## Why the released tiers are easy (strict references, measured)

| released tier (-i01) | rows | relative motion B in A | rotation axes | gripper opens | what it takes |
|---|---|---|---|---|---|
| very_easy (L hooks) | 4370 | 0.29 m, 0.5 deg (pure translation) | 0 | 0 | one straight unhook + a long carry |
| easier (XYZ cross head) | 20112 | (pending) | | 0 | a long route, no regrasp |
| easy (helical claws) | 812 | 0.12 m, 159 deg about 1 axis | 1 | 0 | one screw turn, 20 s |
| medium (orthogonal gate) | 1442 | (pending) | | 0 | short, 36 s |
| hard (claw weave) | 16225 | (pending) | | 0 | long by route length |

No released oracle ever opens a gripper: both hands are welded to their piece from the birth to the end. Every
separation is one or two motions about one axis; the longer tiers are long by route length, not by what the
separation needs.

## What the new tiers add

1. **A sequence of disengagement moves about several axes.** Every tier is a key (B, brown, the target) inside a lock
   (A, teal) whose openings let the key through only in one pose each: the key must pull back, swing, roll, tilt and
   push in that order. Every move is needed (removing any one move leaves no free path: `move_necessity`).
2. **No straight route.** From the start, 79 straight escape directions (26 lattice + 40 random + the designed axes,
   0.5 m each) all collide: 0 / 79 escape at every tier. The pieces never separate before the last move.
3. **Clearances of 3-4 mm** (design; certified free path >= 5 mm at very_easy, 3 mm design clearance at hard) on
   pieces of 0.15-0.30 m: the ALOHA's own tracking error under load (5-10 mm at the TCP, measured) is larger than
   the clearance, so an open-loop plan jams (very_easy v5 did: A sagged 37 mm, the 60 N grasp slipped).
4. **Regrasps and hand-overs that are forced by the geometry** (easier and up): the key leaves *forward* through an
   exit gate GX on the far side, handle last; the hand on the handle cannot follow it through the entry gate G1, so it
   must let go and take the key again by its head grip behind GX. At medium and hard the key leaves on the left, where
   the left hand holds the lock: the right hand moves from the key to the lock's second grip, then the left hand lets go
   of the lock and takes the key (a hand-over through the lock).
5. **Bimanual counter-rotation** (easy): a 180 deg roll of the claw about the shaft, more than one wrist can turn its
   piece with its grip in place; the two hands share it (90 deg each, opposite senses).
6. **A table start** (hard): the assembly rests on the table after a physics settle, both hands start empty at home.

## Method: inverse generation by carving (code: `topogym/tasks/interlocked_claws/redesign/`)

```
 free pose of B (goal side)                    engaged pose (start)
      |   insertion moves (exact, paths.py)          |
      +--------------------------------------------->+
                                                     |  disengagement = the exact reverse
 lock A = gate plates (one per opening) carved from B's swept volume along that path + clearance (carve.py)
        + struts joining the plates outside the swept volume + handle arms toward each arm's shoulder
```

1. The key B is a compound of exact boxes (`shapes.py`): a crossbar claw with an asymmetric prong (one keyhole
   orientation), a shaft, a grip block at its end (and a head grip on a stub ahead of the claw, easier and up).
2. The separated pose and the insertion moves (slide / turn about a pivot / turn about the key's own axis) give the
   engaged pose; the disengagement is their exact reverse (`paths.insertion_to_disengagement`).
3. Each gate is a plate in which the cells B's swept volume crosses are cut out, grown by the clearance
   (`carve.carve_gate`, 1.5 mm cells). A plate therefore lets B through only along the designed path.
4. Struts join the plates' corners where they clear the swept volume by clearance + 2 mm (rejected otherwise).
5. Handles (`tiers.auto_handle`): for each hand, the first plate corner / edge root whose bar, grip block and the
   gripper body around it clear the swept volume, the hand that holds B along the whole path, and the other hands.
6. The scene: A is placed in the robot's workspace (placement search, below); the task keeps the family's robot,
   pregrasp birth (both hands closed on their grips; welds form in the first rows), goal (the entire brown piece in
   the green dashed area, the other piece outside it, quiet 0.5 s), materials and MuJoCo profile (claws_v6).

## The ladder

| tier | design | disengagement (lock frame) | rotation axes | clearance | hand changes | start |
|---|---|---|---|---|---|---|
| very_easy | swivel key: entry gate G1 + side pocket G2 + back wall | pull back out of G2; swing 50 about z (pivot G1); roll 90 about the shaft; pull out of G1 | 2 | 4 mm | none (L holds the lock, R moves the key) | pregrasped, lock held up |
| easier | through-wall key: + exit gate GX; G1 never lets the claw back; the lock stands on legs | pull back; swing 50 z; roll 90; push forward through GX; **R lets go, L leaves the lock and takes the key's head grip behind GX**; pull out | 2 | 4 mm | 2 releases + 1 regrasp | pregrasped, lock on legs |
| easy | flip key: GX takes the claw upside down; the lock is held up | pull back; swing 50 z; **roll 180** (the axial grip's wrist turn); push through GX; **hand-over**: R key -> lock's 2nd grip, L lock -> key's head grip; pull out | 2 | 4 mm | hand-over (2 regrasps) | pregrasped, lock held up |
| medium | hand-over key with a tilted exit (25 deg up) | pull back; swing 50 z; **tilt 25 about y**; roll 90; push through GX; hand-over; pull out | 3 | 3.5 mm | hand-over | pregrasped, lock held up |
| hard | table key: claw in a notch behind G2; steeper exit | **pick up both pieces from the table, lift 6 cm**; lift the claw 15 mm out of the notch; pull back; swing 50 z; tilt 25 y; roll 90; push through GX; hand-over; pull out | 3 | 3 mm | 2 first grasps + hand-over | lock on legs, hands empty at home |

Why each step up is harder for the robot:
- very_easy -> easier: the exit is on the far side of the lock, out of the key hand's reach and behind a gate its
  handle cannot pass: a hand change is forced (the right arm cannot reach the exit side; the placement search confirms
  it for every layout tried).
- easier -> easy: 180 deg of roll (one wrist's full useful range with the grip in place: the birth pad orientation
  has to be chosen for it) and a true hand-over in the air (the lock must be held by one hand at every instant).
- easy -> medium: a third rotation axis (the exit gate is tilted 25 deg), 3.5 mm clearance.
- medium -> hard: nothing is in hand at the start (pick up two pieces resting on the table, lift them together), a
  15 mm notch to lift out of first, a steeper exit, 3 mm clearance.

Changes made while fitting the ladder to the robot (all tiers re-certified at the object level after each):
- easier: the lock stands on legs and the left hand takes the head (with the lock held up by the left hand, the
  right arm would have to reach the exit side: no layout allows it).
- easy: the 180 deg roll was first shared by both hands (counter-rotation); no layout made the left hand's 90 deg
  turn of the lock feasible, so the key hand turns it alone; easy got the hand-over instead.
- medium / hard: the swing turns the same way as very_easy's (the layouts that work for the ALOHA); the exit tilt
  went 35 -> 25 deg (at 35 the head ends too low or too close to the left arm's base in every layout, and the
  downward pull-out reaches the table); hard's 60 deg swing / 45 deg tilt became 50 / 25.

Storyboards (lock frame, each move with its swept poses): `review/<tier>/previews/<tier>_storyboard.png`; robot views
of start, path and goal: `review/<tier>/previews/<tier>_sheet.png`.

## Object-level certificate (`topogym/planning/interlocked_claws/redesign/check_design.py`)

| tier | boxes A / B | path min distance | start: breakers escaped | every move needed | end separated |
|---|---|---|---|---|---|
| very_easy | 23 / 4 | 7.4 mm | 0 / 79 | yes (4/4) | yes (gap 39 mm) |
| easier | 34 / 6 | 7.4 mm | 0 / 79 | yes (5/5) | yes (13 mm) |
| easy | 33 / 6 | 7.4 mm | 0 / 79 | yes (5/5) | yes (13 mm) |
| medium | 43 / 6 | 5.5 mm | 0 / 79 | yes (6/6) | yes (30 mm) |
| hard | 47 / 6 | 5.0 mm | 0 / 79 | yes (7/7) | yes (42 mm) |

## Piece frames at the composite COM (validator, 10-11)
Each piece is a multi-link URDF (body + one fixed-joint link per grip block, so that every grip is its own grasp
region). Its frame used to be the design frame (the claw / lock origin), off the composite COM: M then couples
translation and rotation, and at an impact that jumps the spin (GPU candidate very_easy e585: 5.6 -> 18 rad/s) the
semi-implicit step is not energy-consistent (+6.2 mJ against the 1 mJ rule). Since bd4066c9e every piece's frame is
its composite COM (all links' geometry shifted, the grip links welded at the identity); the body's pose is the design
pose @ trans(com); the oracle, previews and IK convert measured poses back to design frames. All oracles and evidence
were re-recorded after it (instances on the bound tree rt3).

## The oracle (`scripts/robotangle/redesign/interlocked_claws/oracle.py`)

A program of phases planned from measured states and recorded as rows (an open-loop replay reproduces them):
acquire (the welds form) -> for each designed move: hand changes first (open, back off, over the lock at a safe
height, down onto the new grip, close until the weld forms), then the move -> transport of B into the goal area ->
settle. Each move is executed **closed-loop in chunks of 4 path poses**: per chunk, the load bias per arm (commanded
minus measured joints, clamped at 0.06 rad) is added to the next targets, and the hand on B aims at the designed
relative pose in A's *measured* frame. Hand changes: open (the weld lets go), back off and rise, a joint-space hop to the point over the pre-grasp, down
and in, then a **servo onto the grip** (measured TCP vs the grip; joint target IK + load bias; the pads have ~4.5 mm
per side on a grip block: without it the close finds no bilateral candidate), close until the weld forms. Transport:
lift above the lock, a joint-space hop over the goal area, down; the key's distance to the lock checked along the way.
IK: the scene's own MuJoCo ALOHA, IK branches chosen by a dynamic programme
over anchors (`robot_plan.solve_path`), wrist_angle in [-1.6, 1.9] rad, |wrist_rotate| <= 2.8 rad (core).

## Status per tier (after the COM fix; live: STATUS.md)

| tier | oracle | rows | birth / final state | fresh replays | nop | Session breakers (old rows, straight pull, one arm) |
|---|---|---|---|---|---|---|
| very_easy | **success** | 1801 | 90bed6fbea180192 / 9a36adc68cb9ff91 | 2 / 2 same hash | fails | all fail |
| easier | **success** | 3304 | b8ee36f64f29dfc7 / 504c972da5c13d2a | 2 / 2 same hash | fails | all fail |
| easy | **success** | 4277 | 1fc784208b95adfc / 64196635f343edbf | 2 / 2 same hash | fails | all fail |
| medium | **success** | 4780 | 2a58a514f9b6b76a / 3a54532aa3162cf5 | 2 / 2 same hash | fails | all fail |
| hard | **success** | 6221 | ff4b68725467157f / 2fb8e555a4e4f8d3 | 2 / 2 same hash | fails | all fail |

Frozen instances: a cloud host `out/instances2/claws-rd-<tier>/` on the bound tree `rt3` (SOURCE 1d53ef50).
**Stress (criteria v2.9, protocol 3): PASS on all five** (Quest job 9478819, `runs/claws_stress_q1/out/<tier>`): the 35
claws corpus streams remapped onto each instance 0 BREAKDOWN, the reference twice CLEAN with the same hash, nop CLEAN,
~21-22.6k self-play controls per tier with 0 confirmed BREAKDOWN. GPU screens (gpusearch `runs/kd_<tier>`, 4,096
episodes per tier) calm; all 400 exported candidates replayed on CPU with the observer (Quest job 9481187): **0
BREAKDOWN** (228 violent but causal, 172 clean).

## Anti-hack summary (post-fix references vs the strict references, both recorded in fresh Sessions)

| tier | rows old -> new | relative rotation net (old -> new) | rotation path | rotation axes | releases / grasps old -> new | TCP path old -> new |
|---|---|---|---|---|---|---|
| very_easy | 4370 -> 1801 | 0.5 -> 106.6 deg | 43 -> 175 deg | 0 -> 2 | 0 / 2 -> 0 / 2 | 0.57 -> 1.14 m |
| easier | 20112 -> 3304 | 0.5 -> 132.1 deg | 310 -> 245 deg | 3 -> 3 | 0 / 2 -> 2 / 3 | 1.75 -> 2.57 m |
| easy | 812 -> 4277 | 158.8 -> 178.7 deg | 211 -> 418 deg | 1 -> 5 | 0 / 2 -> 3 / 4 | 0.33 -> 2.92 m |
| medium | 1442 -> 4779 | 60.2 -> 108.5 deg | 128 -> 275 deg | 3 -> 3 | 0 / 2 -> 2 / 4 | 0.64 -> 2.98 m |
| hard | 16225 -> 6220 | 4.3 -> 108.4 deg | 1277 -> 269 deg | 11 -> 4 | 0 / 2 -> 2 / 4 | 4.92 -> 4.12 m |

The old references never open a gripper; the new ones from easier up cannot succeed without one. The old very_easy
and easier are long by their route (stops and small corrections: 9 and 82 motion phases), not by what the separation
needs; the trajectory module's phase counts are therefore higher for them, the relative-motion and grasp rows are what
the tasks need. **Flag:** the released easier and hard twist back and forth more than the new ones (rotation path 310 vs 245 deg
and 1277 vs 269 deg, for a net 0.5 / 4 deg); the new ones need more net rotation, grasp changes (and for hard a
table start and 3 mm). Option for hard: a second chamber with a reverse roll and a second tilt (see its packet).

## Limits and open items
- Stress: one confirmed BREAKDOWN (very_easy, GPU candidate kc top039_b0_e585): `jump_uncaused` on the lock's grip
  link as the thrown lock hits the table's side (the piece's angular speed 5.6 -> 18 rad/s, its kinetic energy -0.26 J;
  the link has no partner of its own). The pieces are multi-link (each grip block a fixed-joint link: one grasp region
  per grip); rule 5 judges the link alone. With the validator (main, 10-11 02:45); the pieces stay multi-link.
- Placement: the first layouts of easier/easy/medium failed IK on the swing (one arm sweeps 21 cm and 50 deg); a
  whole-program placement search (`placement2.py`: every move, every hand change with both pad orientations, gripper
  body clearance) is choosing them.
- Hard's table start: grasping the settled assembly is untested; fallback = hard with the pregrasp birth (the rest of
  hard's geometry unchanged).
