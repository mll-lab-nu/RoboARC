# fiber_bundling redesign (5 tiers, MuJoCo, Panda + Robotiq 2F-85): design note

Step 3 at the proposal stage. The fiber agent wrote this on 2026-10-10. The physics is the released MuJoCo port: the
cable as iron wire (`fiber_v4`, elastoplastic joints at 0.4 deg), the exact constraint diagonal (b1ec43997) and F1
(e9e54207d). The review packet is the work notes.

## Why the released tiers are easy
- The five released tiers differ only in their strand count: 2, 3, 4, 8 and 16.
- The ring is the same in every tier, 13.5 mm inner radius. That gives 1.5 mm of clearance around the 12 mm sheath.
- The ring stands coaxial with the cradle, 0.11 m beyond the strand roots.
- One oracle shape solves all of them:
  - grasp the sheath from the cradle;
  - press each strand against the ring's outer face (the comb loop, one press per strand);
  - feed straight along the cradle's own axis for 0.14 m.
- The goal is only "everything passed". No position of the bundle is specified, and the cable never has to be shaped.
- Hard (16 strands) has no MuJoCo reference. The released Genesis demo combed to a 16 mm envelope.

## What the new tiers add (all Panda + Robotiq physical, all within what the wire model holds)
1. **16 strands in every tier.** That is the generator's maximum, so the comb work is the old hard's in every tier.
2. **A closer-fitting ring:**
   - 12.8 mm inner radius (0.8 mm of clearance) from easy on;
   - 12.6 mm (0.6 mm) in hard.

   The strands must be combed further in, and the sheath has to stay centred to a fraction of a millimetre.
3. **The ring turned 30-40 deg about the vertical, and moved.** The feed axis is no longer the cradle's axis. The
   robot must bend the sheath, and the wire keeps the bend: plastic beyond 0.4 deg.
4. **Formboard pegs.** These are vertical pegs (r 12 mm) whose top stands 45 mm above the cable, with a 5 mm head
   like a nail's. The sheath must pass each peg on a given side, beside it (below its top) and wrapped on it (a
   surface gap ≤ 20 mm): this is how wire harnesses are built on a formboard.
   - The head keeps a wrapped cable on the peg. On a bare shaft, the comb presses ratcheted the cable up the peg and
     the lay roll threw it over the top (a lab showed it, below).
   - The feed pulls the cable 0.14-0.19 m through the cradle, so a bend formed anywhere else is dragged off. Only a peg
     that deflects the taut cable holds a bend, and the routes are derived taut (below).
   - A peg on its required side at the start would be wrapped by any pull toward the ring. Each formboard tier therefore
     has a peg that starts on the wrong side: the sheath must be lifted over it (45 mm) and laid down on its far side,
     the formboard lay-over.
   - medium: one lay-over and a 67 deg detour around the peg.
   - hard: an S-route, a 52 deg right turn on peg 0, then a 98 deg left turn on peg 1, which takes the lay-over.
     Peg 0 is a tall post (top 85 mm above the cable; peg 1's top is 35 mm). The open cradle lets a lifted cable rise
     along its whole length, so a lift that clears peg 1 would also clear a short peg 0 (a lab showed it, below). The
     post keeps the S's first bend. It is no taller because the Panda's wrist passes over it at the comb and lay poses
     (hard v2, below).
5. **A lay specification, from easier on.** The red fiber (strand 0) must end at a stated angle about the ring axis,
   measured from the ring's top: 12-20 deg tolerance, like a keyed connector. The bundle's roll is specified. It takes
   a deliberate roll of the held sheath of 40-60 deg. Each target needs a roll the arm can hold through the whole feed
   at that tier's ring (the oracle's IK probes), and lies at least 40 deg from where an unrolled feed leaves the fiber,
   twice the tolerance or more.
6. **A longer service length in medium and hard.** The sheath is 0.36 m (9 links of 40 mm) instead of 0.24 m (6 links),
   and the cradle sits 0.12 m further back. The front face starts where it did, so the route has room between the
   cradle and the hand.

The family's predicate is kept in every tier: every strand station and the sheath front through the ring, tip first;
every strand bend ≤ 60 deg; no gripper-strand contact; a 0.5 s stable window. The new terms come on top of it
(`topogym/tasks/fiber_bundling/redesign/tasks.py`: `lay_term`, `route_terms`).

## The ladder
| tier | strands | ring inner radius / clearance | ring pose (centre, world) | sheath | route | lay (red fiber) |
|---|---|---|---|---|---|---|
| very_easy | 16 | 13.5 / 1.5 mm | released: (0.25, -0.25), coaxial | 0.24 m, 6 links | straight | - |
| easier | 16 | 13.5 / 1.5 mm | released | 0.24 m | straight | -30 ± 20 deg (11 o'clock) |
| easy | 16 | 12.8 / 0.8 mm | (0.221, -0.141), turned 30 deg | 0.24 m | one free bend, 30 deg | -130 ± 20 deg |
| medium | 16 | 12.8 / 0.8 mm | (0.179, -0.147), turned 40 deg | 0.36 m, 9 links | lay-over, then a 67 deg detour around peg 0 | -140 ± 15 deg |
| hard | 16 | 12.6 / 0.6 mm | (0.197, -0.162), turned 40 deg | 0.36 m | S-route: -52 deg on a tall post, lay-over, +98 deg on peg 1 | -130 ± 12 deg |

The very_easy tier is the old hard (16 strands, the released ring and pose). It is the floor the coordinator asked
for. Each higher tier keeps everything below it and adds at least one new kind of work.

Lay angles are right-handed about the feed direction, from the ring's top. Seen from the entry side, -30 deg is
11 o'clock and -130 deg about 8 o'clock. The red fiber starts at -90 deg, so the targets need rolls of +60 (easier),
-40 (easy), -50 (medium) and -40 deg (hard). Each target was checked against the oracle's two IK probes at that
tier's ring, after the route and the combs:

| ring | roll reachable at the feed pose | feed completes with the roll held |
|---|---|---|
| released (very_easy, easier) | about ±60 deg | - |
| easy's (turned 30 deg) | -48..+68 deg | -45..+30 deg (+45 stops at the ring plane, the wrist near a singularity) |
| medium's (turned 40 deg) | -70..+46 deg | -60..+30 deg (+45 stops 47 mm short) |
| hard's (turned 40 deg) | -52..+42 deg | -60..+15 deg (+30 reaches 25 of 30 mm) |

At the turned rings only negative rolls reach 40 deg or more, so the targets there sit at -130..-140 deg. The first
targets (-45 deg for easy, -40 deg for hard: positive rolls) failed in the labs and were re-derived.

The agent is told the new terms in the public goal description (`RedesignFiberTask.public_instruction`), without
coordinates:
- each peg by its order from the cradle and its height, the side it must end on (looking along the sheath from the
  terminal toward the ring), beside it and within 20 mm;
- the lay as an angle and a clock position seen from the entry side, with its tolerance. For easy: "130 deg
  counterclockwise from the top of the ring (about 7:40 on a clock face), within ±20 deg".

## The tiers

### very_easy: the 16-strand bundle (the old hard)
- **Start.** The released scene: 16 strands fanned 28-50 deg (their tips 25-35 mm off the axis) and the released ring.
- **Goal.** The family's predicate.
- **Derivation.** The released hard, unchanged.
- **Why harder than the old very_easy** (2 strands): about 8× the comb work. The comb loop has to bring 16 strands
  within the ring's entry envelope, using sector presses that take several strands each, and every press can disturb
  the strands already combed.

### easier: the keyed bundle
- **Start.** As very_easy.
- **Goal.** The family's predicate, and the red fiber at -30 ± 20 deg.
- **Derivation.** The oracle is very_easy's plus a +60 deg roll of the held sheath before the feed. The closed-loop
  feed keeps the roll.
  - The target is 60 deg from the red fiber's start, so a feed that never rolls misses the tolerance by 40 deg.
  - The comb presses also roll the sheath (their ±70 deg grasp rolls), so the lay has to be set after the last comb,
    not left to chance. Where the old oracles leave the red fiber is measured in their replays on the new scenes
    (anti-hack, below).
- **Why harder than the old easier:** 16 strands instead of 3, plus a specified final roll.

### easy: the turned grommet
- **Start.** 16 strands. The ring is 12.8 mm (0.8 mm clearance) and turned 30 deg toward the robot's side.
- **Goal.** The family's predicate, and the red fiber at -130 ± 20 deg.
- **Derivation.** The oracle route bends 30 deg 35 mm beyond the cradle (radius 0.05 m) and pulls the released 0.14 m.
  The ring sits on the route 30 mm before its end, with its axis on the route's tangent there. The robot's hand ends
  0.787 m from the shoulder (the released references reach 0.834 m).
- **Why harder than the old easier and easy:**
  - the sheath must be bent and set: the feed axis is not the cradle's axis;
  - the alignment tolerance is half the released one;
  - a specified roll (-40 deg) at a pose where the arm can roll only -48..+68 deg and feed with the roll held only
    for -45..+30 deg;
  - 16 strands.

### medium: the lay-over detour
- **Start.** 16 strands on a 0.36 m sheath; the cradle is 0.12 m further back, the front where it was. The ring is
  12.8 mm, turned 40 deg toward the robot. A formboard peg stands 6 mm to the right of the start sheath, 0.115 m
  beyond the cradle.
- **Goal.** The family's predicate; the peg on the sheath's left, wrapped (within 20 mm); the red fiber at
  -140 ± 15 deg.
- **Derivation.** The oracle lifts the sheath 0.15 m, carries it 0.13 m across the peg, lays it down on the peg's far
  side, and swings it about the peg onto the ring axis. Then the comb, the roll (-50 deg) and the feed follow, and the
  feed pulls the cable taut: from the cradle it dips under the peg and turns 67 deg around it to the ring.
  - The taut cable with no peg (an unrouted feed's) passes the peg on its other side, 32 mm away.
  - The hand ends 0.751 m from the shoulder.
- **Why harder than the old medium** (8 strands, straight): a 3D lay-over and a seated detour; a -50 deg roll where the
  arm reaches -68..+64; 0.8 mm clearance; twice the strands.

### hard: the S-route
- **Start.** 16 strands on a 0.36 m sheath; the ring is 12.6 mm (0.6 mm clearance), turned 40 deg.
  - Peg 0, a tall post (top 85 mm above the cable), stands 6 mm to the right of the start sheath, 0.035 m beyond
    the cradle.
  - Peg 1 (top 35 mm) stands 36 mm to its right, 0.08 m further on.
- **Goal.** The family's predicate; peg 0 on the sheath's right and peg 1 on its left, both wrapped; the red fiber at
  -130 ± 12 deg.
- **Derivation.** The taut route turns 52 deg right on peg 0, crosses to peg 1, and turns 98 deg left on it, onto the
  ring axis; 0.19 m of pull.
  - Peg 1 starts on the wrong side. The oracle lifts the held link 0.075 m, which lifts the cable over peg 1 (it rises
    ~0.068 m there) but keeps it below the post's top (~0.06 m there, the top at 0.085 m). It moves the link 0.15 m across and 0.08 m back, so the cable presses onto the post's
    far side and clears peg 1. Then it lays the cable down and swings it about peg 1 onto the ring axis.
  - The unrouted taut cable passes peg 1 on the wrong side.
  - The hand ends 0.771 m from the shoulder.
- **Why harder than the old hard** (16 strands, straight; no reference): two seated bends in opposite directions, one
  of them after a lay-over that must clear one peg and not the post beside it (a hand lift of about 0.06-0.09 m); the
  tightest ring; the tightest lay.

## Inverse generation
- **Oracle first.** `topogym/planning/fiber_bundling/redesign/derive.py` builds each tier from its oracle route: the
  sheath's top-view centre line from the cradle exit.
  - Free-bend tiers: straight runs and bends.
  - Formboard tiers: the cable pulled taut around its pegs (`taut_route`): straight runs tangent to each peg's circle
    (peg + sheath + 2 mm) and arcs around them, ending on the ring axis. The feed's pull leaves only a taut route
    standing; the first, untaut routes put pegs that a pulled cable left (a lab showed it, below).
- **Then the scene, backwards:**
  - the ring goes on the route 30 mm before its end (the feed's front_s), with its axis on the tangent there;
  - each formboard bend's peg is placed so that the taut cable wraps it on the required side, and a peg that starts on
    the other side marks a lay-over;
  - the lay target needs a roll the arm can hold through the feed (probed by IK in the oracle), at least 40 deg from
    the unrolled feed's result.
- **Then the checks, scripted** (derive.py prints them; `tiers.json` has the numbers):
  - every fixture keeps clear of the start state (the straight sheath and the strand fan): pegs by ≥ 2 mm, rings by
    ≥ 8 mm;
  - fixtures keep clear of the pickup (the open Robotiq around the grasped link: the pegs by ≥ 34 mm);
  - fixtures keep clear of the cradle base and of each other;
  - the hand's final point stays within the released references' reach (≤ 0.834 m from the shoulder);
  - the taut cable with no pegs (from the cradle exit to where it kinks into the ring axis) passes a peg on the wrong
    side, or through it.
- **Then the breakers below.** Each design is adjusted where one of them succeeds.

## Feasibility of the key mechanics
Each mechanic is tested once, in the cheapest setting that shows it: robot-level Sessions in the production physics.
- The redesign oracle is `topogym/planning/fiber_bundling/redesign/oracle.py`: the fiber re-planner's stages (the
  released pickup, the comb loop, the gated closed-loop feed), plus a route program, a lay roll, and IK probes of the
  reachable roll and of the feed with the roll held.
- The labs are `labs.py`: each tier's fixture and goal terms on the released three-strand bundle, so the comb is
  cheap. The route and lay terms do not depend on the strand count.
- The 16-strand tiers' own oracles are the last step (below).

| mechanic | setting | result |
|---|---|---|
| roll and lay | lab_lay: the released ring; the -i01 easier reference's pickup and combs, then the lay roll (+60 deg) and the gated feed | **success**, 3647 rows. The red fiber ends at -30.2 deg (target -30 ± 20); the roll stays set through the feed |
| reachable roll | the oracle's IK probes at each ring | see the lay paragraph above: easy's first target needed a roll the feed cannot hold, so it was re-derived (-130 deg) |
| turned, close ring: comb, roll, feed | lab_turned: 12.8 mm (0.8 mm clearance), turned 30 deg; easy's lay | **success**, 5973 rows (149 s). Three combs bring the strands to 8.0-10.2 mm; the -40 deg roll leaves the red fiber at -129.7 deg (target -130 ± 20); the feed passes the 0.8 mm clearance |
| lay-over and seated detour | lab_seated: medium's 0.36 m sheath, peg and ring; lift 0.15 m, move 0.13 m across the peg, lower, swing about the peg onto the ring axis; then the comb, roll and feed | **success**, 5966 rows. The route holds through the feed: the peg ends on the cable's left, the cable touching it and below its top. Four combs (±45 deg rolls) bring the strands inside the 16 mm guided envelope; the -50 deg roll leaves the red fiber at -139.7 deg (target -140 ± 15); the feed passes the 0.8 mm clearance |
| lay-over past a post (the S) | lab_layover: hard's pegs and ring | first with a 45 mm peg 0: lifting 0.15 m raised the whole cable (it leaves the open cradle; at peg 0 it rose 0.12 m), so it crossed peg 0 too and ended on its wrong side. With the tall post: lift 0.10 m, move (-0.08, -0.15) m, lower, swing: **both pegs routed, the cable touching each** (gap 0). Three combs (±45 deg rolls) bring the strands to 8.0-8.4 mm. The first lay target (-40 deg, a +50 deg roll) is out of reach at this ring (roll -52..+42, feed -60..+15), so it became -130 deg. The roll then reached -130.1 deg, but the combs had ratcheted the cable up peg 1's bare shaft (from 3 to 42 mm above the axis), and the roll threw it over the 45 mm top: the S unwrapped and the front ended 69 mm off the ring axis, so the feed could not start. Hence the heads on the pegs. With the heads (v8, v9) the cable stayed wrapped, but the standoff roll still threw the held link 66 mm off the axis: the Panda's wrist (link7) struck the post's head. The roll now comes late in the feed and the feed's gates allow the S's drag: **v12 success**, 6486 rows, red fiber -130.0 deg, both pegs routed. On hard v2 (post 85 mm, peg 1 35 mm, lift 0.075 m): **v13 success**, 10092 rows, red fiber -130.2 deg |
| 16-strand comb and feed | very_easy (the old hard), the tier's own oracle | **success**, 10045 rows (251 s simulated; 76 min of wall time on one a cloud host core): 11 direct combs bring all 16 strands to 11.4 mm or less, then the gated feed. The first MuJoCo oracle of the old hard. A press takes 4-7 min of wall time with 16 strands; forked candidates doubled that (the parent re-executes the chosen one), so the oracle combs directly, one executed press per comb |
| 16-strand oracles, every tier | each tier's own scene, `oracle.py` | **all five succeed**: very_easy 10045 rows, easier 10151 (red fiber -30.0 deg), easy 9277 (-129.8 deg), medium 14514 (-140.4 deg, peg routed), hard 12096 (-130.0 deg, both pegs routed). Recorded on springref 0; their rows replay to success on springref 2.62 (c64294012). Medium and hard comb with forked candidates (`--adopt --skip-strands`); hard's comb-roll rule is 45 deg from up (`--min-from-up 45`) |

Oracle fixes that came out of the labs:
- the lay roll measures the grasp offset at the current pose (a stale one undid the roll in the feed);
- the closed-loop feed plans only its alignment leg, because the planned straight feed's 1 mm clearance check cannot
  pass a ring with 0.8 or 0.6 mm of clearance;
- after a route, some comb presses at the 40 deg rings miss the planner's strict IK tolerance (0.1 mm) by ~0.15 mm,
  with every joint far from its limits. The planner's IK falls back to 0.5 mm there (counted in the report);
- the planner's comb rolls (0, ±70 deg) cannot be held at the turned rings; `--rolls 0,-45,45,-70,70` prefers the
  smaller rolls that the planner's own rule allows (the pressed strand at least 60 deg from up).
- hard's lay roll at the standoff drove the wrist into the tall post's head. link7 is rigid with the hand, so the arm's
  redundancy cannot move it. A kinematic sweep (`--lay-sweep`: IK continuation over front positions x rolls, the
  robot's clearance to every fixture) gave 41 mm of interference at the standoff, and clearance from 60 mm before the
  ring on. `--lay-at-mm -55` rolls there, inside the closed-loop feed, then measures the feed's frame afresh. The same
  sweep keeps hard's comb rolls at 0 and +40 deg: a negative roll at the comb pose puts link7 into the post;
- hard's feed pulls 0.17 m of cable through the S (150 deg of wrap on two pegs, and the sheath's rear link reaches
  peg 0). The closed loop stalled at its 3 mm lag gate with 15 N on the grasp; the Panda's position control sags under
  the drag. Gates of 15 mm lag and a 28 N hold (back-off at 32 N) let it complete, under the grasp weld's 32.5 N slip
  rule;
- candidate combs (`--adopt`) stopped when no press of the widest strand cut the excess. `--skip-strands` gives such a
  strand up, combs the next one, and retries the given-up strands after any accepted press.

## Anti-hack against the strict references
- **Strict references:** the previous successful oracle of each tier, that is the fiber -i01 references
  (`scripts/robotangle/oracles/fiber-bundling-<tier>-i01/`). very_easy … medium have their own. Hard has none, so it
  uses medium's.
- **Metrics:** the redesign tooling's (`antihack.trajectory`), computed by `metrics.py` from the Panda's commanded TCP
  path (forward kinematics of each row; one arm; holding from the commanded closure):

  | strict reference | rows | duration | motion phases | phases total | waypoints | TCP path |
  |---|---|---|---|---|---|---|
  | very_easy -i01 | 3722 | 93.1 s | 10 | 11 | 29 | 1.53 m |
  | easier -i01 | 3608 | 90.2 s | 8 | 9 | 22 | 1.43 m |
  | easy -i01 | 5009 | 125.2 s | 13 | 14 | 35 | 1.84 m |
  | medium -i01 (also hard's) | 8116 | 202.9 s | 22 | 23 | 53 | 2.55 m |

- **The old oracle on the new scene** (`oldreplay.py`: its rows in a MuJoCo Session on the new instance, 40 hold
  rows). Every one fails:

  | tier | old oracle | end | strands through | front through | red fiber (vs target) | routes |
  |---|---|---|---|---|---|---|
  | very_easy | very_easy -i01 | gripper touches a strand (row 3515) | - | - | - | - |
  | easier | easier -i01 | gripper touches a strand (row 3463) | - | - | - | - |
  | easy | easy -i01 | no success (5049 rows) | 0 / 16 | no | -89.9 deg (40 off) | - |
  | medium | medium -i01 | no success (8156 rows) | 0 / 16 | no | -89.8 deg (50 off) | peg 0 31 mm away: not routed |
  | hard | medium -i01 | no success (8156 rows) | 0 / 16 | no | -89.9 deg (50 off) | peg 0 routed; peg 1 on the wrong side |

  The replays ran on the first instances (lay targets -150 / -20 / -160 deg); the scenes are the same, and the red
  fiber's measured angle is compared with the re-derived targets.
- **The new oracles:** the 16-strand tier oracles (`oracle.py` on each tier, all five succeed). Their metrics go into
  `antihack_fiber.py`'s report per tier, with the old oracle's replay and the breakers. The verdict is PASS on all five:

  | tier | strict reference | phases (new / strict) | waypoints | TCP path (m) | duration (s) |
  |---|---|---|---|---|---|
  | medium | medium -i01 | 38 / 23 | 75 / 53 | 4.10 / 2.55 | 363 / 203 |
  | hard | medium -i01 | 32 / 23 | 76 / 53 | 4.09 / 2.55 | 302 / 203 |

  very_easy, easier and easy are in their packets.
  - The labs' metrics are not comparable, because they comb three strands. lab_turned (easy's mechanics) has 11
    phases, 22 waypoints and a 1.52 m TCP path in 149 s, against easy -i01's 14 / 35 / 1.84 m.
  - The 16-strand comb alone took the released hard 20 presses (its selection), each a guided press of several
    phases.

## Breakers
Each breaker is a cheaper way to the goal that skips one tier's new work. The task's own predicate decides, so a
breaker fails when the run does not succeed. Robot-level breakers run in Sessions (the labs where the term does not
depend on the strand count). Object-level breakers evaluate the route terms on sheath centre lines sampled like the
real one (link centres 40 mm apart), with no stepping (`route_check.py`).

| breaker | tier | how | result |
|---|---|---|---|
| no_comb | very_easy | the released pickup, then the closed-loop feed straight in | **fails**: the uncombed strands are pushed out to 48-60 mm and one touches the gripper (row 1214) |
| no_roll | easy (lab_turned) | the same pickup, combs and feed, no lay roll | **fails**: everything passes, but the red fiber ends at -89.9 deg, 40 deg from the target (tolerance 20) |
| shortcut | medium, hard | object level: the taut cable with no pegs, from the cradle exit to the ring axis | **fails**: medium's peg ends on the cable's wrong side 32 mm away; hard's peg 1 on the wrong side 55 mm away |
| over_top | medium, hard | object level: the derived route, but laid straight across the lay-over peg's top, 5 mm to the required side of its centre | **fails** on the below-top rule. A top-view-only rule would have counted it as routed |
| old oracle | every tier | the old -i01 oracle replayed on the new scene (anti-hack, above) | **fails** in every tier |
| comb_no_roll | easier | the very_easy oracle's rows on the easier scene (the same start) | pending the very_easy oracle |
| wrong_side | hard | robot level (lab_layover): swing about peg 1 onto the ring axis without the lay-over, then the same combs, lay and feed | running |

The object-level route checks also confirm the derived taut routes count as routed: medium's peg at -3.9 mm (the
cable on it), hard's pegs at -2.7 and -4.6 mm.

## Previews
The packet has, per tier:
- the start sheet (front, from behind the robot, top, ring close-up);
- the route plot (top view, the shortcut dashed);
- renders of the lab oracles' states: after the route (medium, hard) and at the end (the success state).

They are rendered with `topogym/planning/fiber_bundling/redesign/preview.py` (`--actions` replays a stream in the
production Session first), on the CPU with the redesign tooling's renderer.

## Code
- `topogym/tasks/fiber_bundling/redesign/tasks.py`: `RedesignFiberTask`. It holds the ring transform, the pegs, and
  the `lay` and `route` goal terms on top of the family's predicate. `TIERS` holds the derived tier data.
- `topogym/tasks/fiber_bundling/redesign/labs.py`: the mechanics labs (each tier's fixtures and terms, three strands).
- `topogym/planning/fiber_bundling/redesign/`:
  - `derive.py`: inverse generation and its checks;
  - `build.py`: frozen instances, with the variations registered in-process and the released grasp assistance on the
    second-to-last sheath link;
  - `run_registered.py`: runs any of these scripts with the redesign variations registered;
  - `oracle.py`: the redesign oracle (route program, lay roll, IK probes, direct or candidate combs);
  - `preview.py`: the preview sheets;
  - `oldreplay.py`, `passtrace.py`: replays of a stream on a redesign instance (final terms; per-strand passage);
  - `metrics.py`, `antihack_fiber.py`: trajectory metrics and the anti-hack report (the redesign tooling's schema);
  - `route_check.py`: the object-level route breakers.

## Limits and open items
- The planner's keyframes assume the 6-link sheath (`GRASP_LINK = "trunk_04"`). Medium's and hard's oracles need it
  set to trunk_07, in-process, without touching the shared planner.
- The lay is measured at the strand's root about the ring axis. A twisted bundle whose roots and tips disagree counts
  by its roots.
- The pegs' heights are tested at robot level: a 45 mm peg is cleared by a 0.10-0.15 m hand lift, and a taut cable
  stays on its side. The cradle is open, so a lifted cable rises along its whole length (at 0.035 m past the cradle
  ~0.8x the hand's lift): hence hard's tall post.
- **Hard v2 (2026-10-11): the post's top 0.12 -> 0.085 m, peg 1's 0.045 -> 0.035 m.** At the comb pose the Panda's wrist
  (link7) passes over the post and the gripper's spring link over peg 1's head. The 16-strand oracle needs negative
  comb rolls for the strands on one side. With the 0.12 m post's head every roll below -20 deg put link7 17-41 mm into
  it (the oracle's `--lay-sweep`), and peg 1's head took 40-50 N in roll-0 presses (hard16_v2/v3 lost the grasp
  there). With the lower heads the sweep clears rolls down to -30 deg. lab_layover_v13 on v2 (lift 0.075 m): route,
  combs, late roll and feed **succeed**, 10092 rows, red fiber -130.2 deg, both pegs routed.
- The comb's IK at the 40 deg rings is close to the Panda's limits: the planner's ±70 deg press rolls fail there, and
  some presses meet only a 0.5 mm IK tolerance. The oracle's choices are recorded in each run's report.
- 16 strands make the simulation ~6x slower than the 2-strand tier (~0.3 s per control step on one a cloud host core), so each
  16-strand tier oracle takes hours on the shared host.
