# rope_threading redesign: bulkhead lacing (5 tiers, MuJoCo)

Status: **design record** (10-11 morning). Robot-level oracles (step 4): very_easy, easier and easy
succeed live with deterministic fresh-process replays, a failing nop, a failing old -i01 oracle and failing breakers;
medium and hard are being re-run after the T redesign (see the packets). Agent "rope" (took the family over from the soft-body redesign
session on 10-10 21:15). Sandbox branch (never pushed). Evidence and packets:
the work notes.

## 1. The released tiers, and why they are too easy
The released tiers (dev v16 layout on MuJoCo) put 1-3 rings in the middle of an empty table. Both arms reach both sides of
every ring, the apertures are 69 / 50 mm, and the 1.1 m rope has ~0.5 m to spare. The MuJoCo strict references (the
family's feedback relay: easy6, 1882 rows; hard6s, 3326 rows) use 4 / 6 grasps and no regrasp, and no leg of them is
forced: any arm may feed or receive any ring.

## 2. The concept
A **bulkhead** (a 12 mm wall, 0.32 m tall, in the plane x = 0 between the two ViperX bases) splits the table into the
left arm's and the right arm's halves. The rope is laced, in a fixed signed order, through
- **grommets** in the wall (centre 0.15 m up): the feeding arm cannot reach past the wall, so the other arm must take the
  tip beyond it: a **forced hand-over**;
- **low rings** on one half (centre 7 cm up on a pedestal; v1; on medium / hard one of them, T, turned 10 deg off the
  y axis, facing neither arm), or a **ring on a post** (hard's last): the other arm cannot reach past the wall, so the arm that fed the ring must let go,
  walk around it and pick the tip beyond it: a **forced same-hand regrasp**;
- the last ring is never counted as forced: the tip may be pushed through and held (holding is allowed).

Every ring has its own axis (+-x through the wall, +-y on the halves, straight down for easier's eyelet, turned 10 deg
for T). Posts (r 10 mm) stand on every straight chord between consecutive rings that a single push could
thread. The rope length is the planned route plus
a short tail (0.45-0.48 m before the first grommet, of which ~0.2 m is the rise from the table) and a 10-12 cm tip.

The success predicate is unchanged: `RopeThreadingTask.evaluate_state` (ordered signed aperture crossings along the
material coordinate, thickness clear, unambiguous, quiet). A tier only declares its rings, their required signs, and
static fixtures (boxes and vertical cylinders). Nothing ends an episode early.

**Quiet gate.** The released rope_threading profile sets the terminal `maximum_speed_m_s` to **0.01 (10 mm/s)**
(`topogym/tasks/release_profile.py`; the frozen -i01 instances carry 0.01 and their goal text says 10 mm/s). The redesign
keeps the released gate. The review note R8 ("the demo's quiet gate is 1 mm/s") came from `common.py`'s base default
(0.001, which knot_untying uses); it is not rope_threading's released gate. The demos are judged by the task's own
10 mm/s gate; the final rope speed is reported as well.

**Grasp.** contact_weld_v1 on every rope link (the rope_threading MuJoCo strict references' grasp settings; the weld
slips at the ALOHA's 60 N / 20 ms mean rule). No change against the family's MuJoCo port.

## 3. Geometry v1 (why it changed since the 10-10 morning previews)
Session 1's object-level demos threaded both grommets reliably, but lost every same-hand ring at the release: the rings
stood 13-20 cm up and faced sideways, the near side of the rope hung down to the table and outweighed the ~8 cm
overhang, so the rope slid back out. v1 keeps the bulkhead lacing and makes **every ring a hand lets go of low** (centre
7 cm over the table, on a pedestal block that keeps a lying rope from sliding under it). After a feed the rope rests on
the table on both sides of the ring, so nothing pulls it back. The relay becomes **lay and pick**: every draw ends with
the tip laid on the table in front of the next ring, along its axis; every feeding grasp is a top-down pick of that laid
rope 14 cm behind the tip; every same-hand receive is a pick of the tip lying beyond the ring. The tilted ring T moved
inboard (task x 0.30, was 0.39-0.40: drawing to it folded the right arm over its base).

Further v1 changes during the night, each forced by a check or a robot-level run (iteration log, section 9):
- **Start = a gentle spiral** (minimum bend radius 0.19-0.21 m, tail inside, tip outside at (0.13, -0.40)) instead of
  the zigzag. The rope's rest shape is its start shape (the released profile: `rope_reference_shape` "initial"); the
  zigzag's 5 cm U-turns, once drawn straight, curled back and the laid rope crept away at ~1.7 cm/s before the next
  pick (robot level, very_easy and easier).
- **The tip is laid 12 cm before the next ring** (was 6 cm): the drawing hand's wrist camera hit the ring / grommet.
- **easier's eyelet V moved in to x 0.22** (was 0.32): the vertical feed with a cross grasp needed wrist_angle > 2.0 rad.
- **No shelf on medium / hard**: at 0.25 m over the right half it blocked the right arm's upper arm / forearm in every
  pose near T, and first the receive beyond G2 (so G2 was not forced).
- **T is a low pedestal ring turned 10 deg** (centre (0.28, 0.12, 0.07), travel (0.17, -0.98, 0); pedestal ring 40 mm on
  medium, 35 mm on hard): tilted 30 deg down it overhung the table pick beyond it; tilted 25 deg up (centre 0.12, then
  0.15 m) it lost the rope at every release on hard (the rope slid back down the incline); turned 25 deg the tip
  retracted 13 cm out of it after the same-hand release (every variant); turned -10 / 0 deg the picks beyond it failed;
  +10 deg solved medium and hard (the cluster fan-out, 10-11).
- **easy has no post on B -> G3** (P4): every robot-level route pressed it (65-146 N); the chord still needs a
  same-hand regrasp and a wall passage.
- **medium's A at x -0.22** (as hard's): at x -0.26 the laid tip swung ~70 deg off A's axis at the release (three runs).
- **The lay run**: the drawing hand runs straight along the lay line for the feeding station + 12 cm (the rope behind
  the tip cut the corner to the grommet), never starting beyond y 0.40.
**The start shapes, the ring heights and medium / hard's fixtures differ from the 10-10 morning previews.**

## 4. The five tiers
Task frame (m): x towards the right arm, y towards the back, z above the table; world = task + (-0.1357, 0, 0.9196).
Rope r 4 mm (rope_chain_v1, 5.5 mm links), ring tube r 6 mm. Start: a flat spiral on the right half in front of the right
arm (centre (0.40, -0.40), outer radius 0.27 m, 3.5 cm pitch), the tail inside, the tip outside at (0.13, -0.40).

| tier | rings, material order (sign) | inner radius mm | rope m (links) | forced events | fixtures | step up the ladder |
|---|---|---|---|---|---|---|
| very_easy | G1- A+ G2+ B- | 50 50 50 50 | 2.19 (398) | 3: G1, G2 hand-overs; A regrasp | bulkhead (2 windows), 2 pedestals, 3 posts | the concept: two wall transfers, one same-hand regrasp |
| easier | G1- A+ G2+ V+ | 45 45 45 45 | 2.51 (~456) | 3 | bulkhead, pedestal, 2 posts | 45 mm; the last ring is a vertical eyelet threaded downward (the held rope turned from lying to hanging) |
| easy | G1- A+ G2+ B- G3- | 45 45 45 40 40 | 2.73 (496) | 4: + B regrasp | bulkhead (3 windows), 2 pedestals, 3 posts | a fifth ring, a third wall passage, 40 mm |
| medium | G1- A+ G2+ T- G3- | 40 everywhere | see packet | 4: + T regrasp | bulkhead (3 windows), 2 pedestals, 2 posts | T turned 10 deg (facing neither arm), 80 mm everywhere, shorter tail, clockwise start |
| hard | G1- A+ G2+ T- G3- C- | 40 35 40 35 40 35 | see packet | 5: + G3 hand-over | as medium + C's post | six rings, a third forced hand-over, 35 mm on the same-hand rings |

Ring centres (task frame) and axes: G1 (0, -0.24 / -0.28, 0.15) travel -x; A (-0.26, 0.03, 0.07) +y; G2 (0, 0.28 / 0.34,
0.15) +x; B (0.26, 0.01, 0.07) -y (very_easy), (0.30, 0.12, 0.07) -y (easy); V (0.22, -0.04, 0.17) -z (easier; moved in from x 0.32, where the vertical feed needed wrist_angle > 2.0 rad); T (0.28,
0.12, 0.07) (0.17, -0.98, 0) (medium, hard); G3 (0, -0.07 / -0.12, 0.15) -x; C (-0.32, -0.12, 0.15) -x (hard).

Legs (feeder -> receiver): G1 right -> left (hand-over); A left -> left (regrasp); G2 left -> right (hand-over); B / T
right -> right (regrasp); G3 right -> left (hard: hand-over; easy / medium: last, right holds); C left (last, holds).

## 5. Why it is hard for the ALOHA
- **Transfers through a wall.** A parallel gripper and its wrist camera keep the hand >= ~5-6 cm from a wall or ring it
  feeds through; the hand never passes an aperture. A push ends with the tip ~8.5 cm past the plane, so someone must take
  it beyond: through the wall only the other arm can (checked over 48 approach directions x rolls with the arm's
  conservative capsules: the feeder has no collision-free pinch pose beyond the wall).
- **Same-hand regrasps around a ring.** The other arm cannot reach past the wall, so the feeder lets go, lifts over the
  ring and picks the 2-6 cm of tip that lies beyond it, without dragging the rope back out.
- **Lay and pick.** A rope can only be pushed like a wire for ~10-15 cm (the chain is floppy beyond that), so every feed
  starts from a fresh grasp ~14 cm behind the tip, lined up with the ring's axis: the arm lays the rope along the next
  ring's axis, lets go, and picks it again at the right station.
- **Orientation.** Axes change ring to ring (+-x, +-y, straight down, turned 25 deg): the held rope must be
  turned onto each axis (closed loop on the tip's measured direction).
- **Limited slack.** The tail before G1 is ~0.45 m, of which ~0.2 m is the rise from the table; pulling extra rope through
  a grommet early (the released relay draws 12-19 cm per hand-over) eats the rope needed for the last rings.
- **Clutter.** Posts block every straight chord a single push could thread (except hard's G3 -> C, where the chord is the
  route: the feeding hand stops at G3, 30 cm short of C).

## 6. Inverse generation (the recipe)
1. **Oracle first**: the hand-over plan (which arm feeds / receives each ring) is chosen first; each ring is then placed
   where those arms can do it (a grommet in the wall between the halves, a low ring on the receiving arm's half), its
   axis set by the route's tangent there.
2. **Scene from the oracle** (`scenes._make_v1`): curvature-checked Hermite connectors (bend radius >= 3.5 cm; easier's
   V arch 2.6 cm) join straight runs of 0.21 m before every ring (the feeder's push line) and 0.095 m after it; bulkhead
   windows cut where the route crosses x = 0; pedestals under the low rings; a post on every threadable chord, slid to the
   station farthest from the route; the rope length = the route length; the start = a flat spiral in front of the
   right arm, clear of everything by >= 4 cm (easier: 5 cm from V's base).
3. **Checks** (`scenes.check_design`): the route crosses exactly the required signed sequence (the task's own aperture
   predicate), clears every fixture by >= 38 mm, the start rope crosses nothing. The relay plan (`relay_plan.RelayPlanner`
   -> `oracle.timeline`) solves IK at every waypoint with capsule checks against the fixtures, the table and the other
   arm (vias / RRT where a segment collides), wrist_angle <= 2.0 rad.
4. **Break it, adjust, repeat**: the old relays (as-is and retargeted), straight-line tips, shortest routes and centre
   drags all fail on v0 (20 / 20, object level); the v1 re-run is in the packet. Iterations: v0 (session 1) -> v1 (low
   rings, lay and pick; T inboard).

## 7. Anti-hack evidence
See the packet (`review/<tier>/README.md`): the planned (and, when it exists, the recorded) oracle's trajectory metrics
against the strict reference (old easy for very_easy..easy, old hard for medium / hard), the object pose change, the
forced-event table, and the old -i01 oracle replayed on the new scenes.

## 8. Iteration log (v1, 10-10 night; every step forced by a check or a robot-level run)
| when | what broke | change |
|---|---|---|
| plan1 | the drawing hand's wrist camera hit the next ring (tip laid 6 cm before it); easier's vertical feed needed wrist_angle > 2.0 rad | tip laid 12 cm before the ring; V to x 0.22 |
| forced check | medium / hard: the shelf blocked the right arm's receive beyond G2 (G2 not forced) | shelf narrowed, then removed (plan2: it blocked the right arm everywhere near T) |
| plan2 / plan3 | T tilted 30 deg down overhung the table pick beyond it | T tilted 25 deg up, centre 0.12 m, then 0.15 m (plan5: from 20 cm behind the tip the rising feed line began at the table) |
| robot level | a joint-space descent swept the fingers along the lying rope; the home gripper (0.0084) is nearly closed | pre-grasp backed off along the approach axis with the final rotation, gripper opened, straight descent |
| robot level | the laid rope crept away before the next pick (the zigzag start's 5 cm U-turns are the rope's rest shape) | start = a gentle spiral (bend radius >= 0.19 m) |
| robot level | after the feeder let go, the tip beyond a same-hand ring retracted from 8.5 to 3-4 cm (too short to pick) | same-hand rings fed from 20 cm behind the tip |
| robot level | hard: an unaligned tip pushed into the wall jammed the simulation (~40 s per control) | damped, revert-on-worse alignment; no push when misaligned; push stopped when the tip lags; jam watchdog + rewind |
| plan4 | hard: the lay before A put the left fingers on C's base; a chord post sat on C's feeding station | A at x -0.22 on hard; posts keep 5.5 cm from the lay lines |
| robot level (10-11) | easier's oracle pressed V with 128 N for 4 s, easy's pressed post P4 with up to 146 N (the load-bias compensation integrated contact force) | the bias is not grown while a hand presses; straight lines and pushes stop when the Session reports forcing; the hand lets go at the end if holding does not settle the rope. easier re-recorded without forcing |
| robot level (10-11) | medium / hard: laid tips whipped back 130-160 deg at releases, the tip retracted 13 cm beyond T | the lacing route runs clockwise round the wall; the start spiral (= the rope's rest shape) was counter-clockwise and stored bending against it: medium / hard start from a clockwise spiral |
| robot level (10-11) | medium: a post on the G1 -> A chord hooked the rope drawn from G1 | no post there (no single push threads both: the feeder of G1 cannot pass the wall) |
| robot level (10-11) | medium (T +25): the tip retracted 13 cm out of T after the same-hand release in every variant | T turned +10 (the cluster fan-out of T variants 0 / +-10 x CW / CCW start) |
| robot level (10-11) | easy: post P4 pressed at 65-146 N on every route; hard: the left gripper hit the bulkhead strip at 3.2 kN drawing to C | no P4; C's lay run starts >= 12 cm from the wall |
| new head (10-11, ccd4ba7fa: SOFT_CONTACT .95/.95, rings keep class) | very_easy / easier references pressed post P2 at 80-133 N: a fast joint move to a pre-grasp left forearm_roll 0.79 rad behind its target; the lagging arm left the checked joint line | joint moves lag-paced (<= 0.10 rad); a forcing stop backs off at once; references re-recorded on the head |
| prism posts (10-12, b2b751580: robot-touchable cylinders built as 32-facet prisms) | the posts P1-P3 became prisms; very_easy / easier / medium's rows reached the full sequence but the rope was not yet quiet when the rows ended | instances rebuilt (inst5); those rows + a 10 s hold of the last row; easy's rows unchanged; fresh replays identical on Intel (the cluster) and AMD (a cloud host) |
| hard on the new head (10-11 / 10-12) | with the lay run before C 12 cm off the wall (plan9) every run lost G3 when the rope was laid in front of C (~60 rewinds); the joint move onto the pick beyond A was blocked by P1 and the hand jammed on it at 197-245 N | the lay run 8 cm off the wall (plan8) under the gentle executor; a blocked pre-grasp move goes over a lift up to 20 cm or is not made |
| evidence (10-12) | a live run's forcing summary lost an event across a rewind (hard: none live, 197 N in the fresh replays) | forcing is read from the fresh-process replays only |

## 9. Open items
- Robot-level oracle per tier on the head (step 4): see the packets for each tier's state (deterministic fresh-process
  replays, nop, old -i01 oracle and breakers on the new head, stress confirmations).
- Product registration: `topogym/tasks/rope_threading/redesign/registration.py` (variation records + per-tier profile
  entries with the ring keep on each tier's own ring entities); `patches/rope_threading_redesign_registration.patch`
  wires it into profiles.py / variation_records.py (outside this agent's write area).
