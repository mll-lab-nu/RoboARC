# battery_replacement on MuJoCo (swap a game controller's AA cells)

2026-10-17. A new family, MuJoCo only: two ALOHA arms open a game controller's battery cover, pry the two old AA
cells out with a pry stick, drop them into a recycling box, seat two new cells with the right polarity and latch the
cover. First setting `battery_replacement_xbox_aa` (pool experimental, no tier; model revision `xbox_aa_v3b`), base
`battery-replacement-01-i01`. Design record, sources and every measurement: `<design record>/controller_battery_design_20261014/`
(README.md v2 design, README_v3.md the v3b demo and section 10 the topogym port, REGISTRATION_PROPOSAL.md; not in git).
The registration (registry, variation record, profile, rcb_topogym TASKS, goal_description, topogym_rules, mk_tasks) was
approved by the owner on 2026-10-19.

Changes of 2026-10-19 (the owner's choice in the main session, after the metric review
`<design record>/robotangle_mujoco/metric_review_20261019/battery_zh.md`): rule C's 60 mm is one budget per cell and episode,
counted from where the cell leaves the compartment (finding 7; Grasping below); the public goal's cover line says what is
checked (finding 2; Goal below) and the stick may end on the table or in a hand (finding 5); the texts' scene says to
slide the cover back "until it stops (it latches there)" (finding 3). The success rule's thresholds and the reference's
actions are unchanged; the instance's birth moved with the grasp configuration and the goal text.

Changes of 2026-10-20 (the owner's choices from the cam_high realism list; visual only, the shared renderer and the asset
pipeline untouched): the pry stick drawn as a black nylon spudger, its holder dark grey, the cradle slate with a rim
band, the cover's pad a dark grey rubber mat, the tray dark, the recycling box a blue bin with a recycling label (the
scene list below). Every colliding geom is as before; each new visual lies within 0.92 mm of the collision it draws,
both ways (`<design record>/controller_battery_design_20261014/realism2/props_gap.py`: the rounded edges at most 0.88 mm inside a
corner (1.2 mm x (sqrt 3 - 1)) plus 0.04 mm of polygon, the bands and decals 0.1-0.2 mm proud). On dev 9115da07d8 and
on this tree the instance is born 764192d9a7e49a0f, and the reference's qpos and qvel are bit for bit the same at every
control step, the final state de1cb2ee33e369ed on both (`realism2/`, `port/i01r2/`); the reference is unchanged (no
re-stamp).

The same day the owner chose option C of the controller's look (`<design record>/controller_battery_design_20261014/xbox_visual/`): the
controller's outer body is an Xbox Wireless Controller at 1:1 in static collision, drawn by its visual twins (The task,
the controller's outer body). Until then the shell's collision was a 90 x 80 x 24.5 mm block over an open 15 mm slot
above the cradle, and the grips and the body's sides were drawn without collision, up to 47 mm off it (a hand or a cell
could pass through them, a cell laid on a grip would fall through).

## The task

An Xbox wireless controller's battery compartment at 1:1 (two AA cells, Energizer E91 sizes: 14.5 mm across, 50.5 mm
long with the 1 mm nub, 23 g). The scene (`topogym/assets/geometry/battery_replacement.py`, laid out in the ALOHA mount
frame; m, z = 0 the table top):

- the controller face down, fixed in its cradle, the compartment's centre at (0, 0.12), its floor 34 mm above the table:
  a cavity 56 x 34.6 mm, 18.5 mm deep, two slots split by a 1.6 mm rib; each slot has a + contact plate at one end (slot
  A's + end toward +x, slot B's toward -x) and a spring contact at the other (a slide joint, 1.2 N/mm, 2 mm compressed by
  a seated cell, at most 1.5 mm more); the rim stands 4 mm above a seated cell, with 1 mm of side clearance around it;
  beside each + end a pry notch (4 x 6.55 mm, 2 mm into the end wall and the long wall, a 1 mm floor groove);
- the controller's outer body (option C, 2026-10-19; `BODY_PIECES`, `body_pieces()`): an Xbox Wireless Controller at 1:1,
  152.9 x 101.9 x 61 mm, 280 g (dimensions.com, "Xbox One Controller"; retailers list the Series model at 153 x 102 x 61 mm,
  for example Kaufland; Microsoft's pages give no dimensions), face down: its face 0.1 mm above the cradle's top, 18.9 mm
  below the compartment floor, the back of its centre the shell's back (20.5 mm above the floor), the grips toward -y (the
  cameras), the top edge with the bumpers toward +y, the triggers standing up from the back at the top edge (to 27 mm),
  overall x +-76.5, y -55 to +46.5, z -18.9 to +27 (with the sticks, hidden in the cradle, 60 mm). The pieces are static
  collision geoms of the shell body (the same contact and friction as the shell): a box filling the slot under the shell
  down to the face, and twelve convex hulls of rounded discs (`_rounded_hull`): the shell's sides, the top edge, the
  bottom edge with its chin, two shoulders, two grips, two bumpers, two triggers. Each is drawn by its visual twin, so
  what is drawn is what collides (`xbox_visual/gap_check.py`: every visual geom of the controller that is not a collision
  twin, the decals, the face's sticks and buttons, the headset jack, lies within 0.21 mm of the collision; the polarity
  marks on the rim moved from 13 to 21 mm off centre, where slot B's had lain 3 mm above the cover's hook slots). The
  colours: Carbon Black (the shell's), the bumpers and triggers darker. The face's two sticks, D-pad, A/B/X/Y, Xbox, view,
  menu and share buttons are visual geoms below the face, inside the cradle: face down they are not seen. The cradle stays
  140 x 120 x 15 mm: the shell is welded to the world, so the grips' 6.5 mm overhang on each side needs no support, and a
  stand narrower than the controller, its grips over the edges, is how such a holder looks; a wider one would only add
  collision where no part goes;
- the cover (59.6 x 38.6 x 2 mm, a tongue, three hooks, a 3 mm ridge, 5.5 g), closed and latched: it unlatches by sliding
  5 mm toward the controller's top edge (+y) past a detent plunger (2 N preload, 0.5 N/mm; measured 5.0 N to open, 5.9 N
  to close), then lifts off;
- the old cells old_A, old_B seated (slot A + end toward +x, slot B toward -x);
- the pry stick standing blade down in its holder at (0, 0.225): a 2 x 1.2 x 30 mm blade with a 2.0 x 1.0 x 1.2 mm foot
  at its tip toward the cell's axis, a 6 x 6 x 40 mm shaft, a 16 x 16 x 50 mm handle, 30 g; the holder's 6.8 mm square
  hole has a 2 mm x 45 deg lead-in at its mouth (`STAND_FUNNEL`: the hole's 0.4 mm play is less than the hands' tracking
  error); drawn as a black nylon spudger (since 2026-10-20: the shaft and the handle as boxes with their edges rounded by
  1.2 mm inside the colliding ones, five grip bands 0.15 mm proud on the handle), in a dark grey holder;
- the new cells new_1, new_2 in the tray at (0.065, -0.025); the recycling box (inner 70 x 50 x 30 mm) at
  (-0.065, -0.025), empty; a pad for the cover at (-0.12, 0.22) (display only). Since 2026-10-20 the tray is dark
  plastic, the recycling box a blue bin with a recycling label on its front (a white panel and a green triangle, 0.1-0.2
  mm proud), the pad a dark grey rubber mat, and the cradle slate with its edges rounded by 1 mm inside its box and a
  lighter rim band 0.1 mm proud on its top.

The new cells lie in the tray as `+-` (new_1's + end toward +x, new_2's toward -x) for every seed: until 2026-10-20 the
scene seed chose the polarity (seed % 4: `+-`, `-+`, `++`, `--`; seed 0 `+-`); since then a seed varies appearance only
(the owner). The appearance parameters `controller_finish` (the Xbox outer body with its bumpers and triggers, the
compartment's walls, which share the body's colour as on the real controller, and the cover), `tray_finish`,
`box_finish` (the bin; its label keeps its colours) and `props_finish` (the cradle and its rim band, the stick's holder,
the cover's pad, and in the stick entity alone the pry stick's blade, shaft, handle and grip bands) re-colour visual
geoms only (`scene_variation.appearance`: appearance seed 0 is the recorded look); the cells and the red "+" marks keep
the colours the texts name. Since 2026-10-21 the defaults are the props' look of 2026-10-20 (tray `dark`, bin `navy`,
props `grey` with the black nylon stick; seed 0 writes the entity files of <test branch> byte for byte);
seeds 1 and 2 (tray grey / white, bin blue / green, props wood with an orange nylon stick / black with a blue nylon
stick) differ from it in every one of these parts, and the texts describe the stick without its colours. Since
2026-10-21 (the owner, after the migration session's colour table, `<host work dir>/colour_table_20261021/`): the stick is
always a nylon colour (black, orange, blue; the "wood" props had drawn it in wood's tan, beside a text that says "a nylon
spudger"), and the coloured controller finishes are the real coloured controllers (Pulse Red, Shock Blue: a coloured
front and a white back, battery door included), so face down a white back with coloured bumpers and triggers, on which
the rim's red "+" marks read (seed 2's red body had made them red on red); the rim's white "-" marks turn dark on a white
back. Seed 0 writes the entity files of f6471bc717 byte for byte. Everything else is fixed.

Every mechanism is physical: the controller is one MJCF entity whose shell body has no joint (MuJoCo welds it to the
world) with three slide joints (the two spring contacts, the plunger; their stiffness and spring zero in the MJCF); the
cover, the four cells and the stick are free bodies. No per-tick task rule. Contacts: profile
`("battery_replacement", None)` = `battery_replacement_v1`, claws_v6's values (the design's sandbox ran the ALOHA on
them), the parts' own contact parameters in their MJCF (solref 1.25 ms / damping ratio 2, solimp 0.9 0.99, condim 3,
friction 0.3). MuJoCo's multi-point convex collision (`multiccd`) stays on: with it off a cell lying on the box floor
touches at one point and rocks without end (0.3 rad/s at rest). Every collision geom is in group 3 with a coloured
visual twin (nyx_direct draws colliding geoms of groups 0-2 grey); the controller entity is not declared fixed, so
nyx_direct draws it.

Grasping: contact_weld_v1 on the cover, the stick's handle and the four cells (each region the whole part plus 5 mm; the
stick's around its handle; the tool frame within 6 cm). The cells follow rule C (the owner's choice of 2026-10-18, the
design's GRASP_PROPOSAL.md: a cell must look and be held, not hang from a fingertip touch). In this family's words:
- a pinch's **central angle**: the angle around the cell's axis between the two fingers' contact points (in the plane
  square to the axis, seen from the axis), 180 deg for two pads pressing on opposite sides of a diameter. The rules test
  it as two-finger force closure (`force_closure_pair`): a contact of each finger on a line within atan(mu) of both
  contact normals; across a cylinder that is a central angle of at least 180 - 2 atan(mu), 90 deg at mu 1 and 150 deg at
  mu 0.268 (= tan 15 deg). Decided once, when the hold is taken;
- a **firm** hold: its contacts close force at mu 0.268 (central angle 150 deg or more); a **weak** hold: at mu 1 only;
- the **old cells** are `WeakHoldGraspTarget(friction coefficient 1, firm 0.268, 60 mm)` (`topogym/utils/grasp.py`; an
  opt-in subclass of `ForceClosureGraspTarget`, its runtime in `topogym/world/mujoco/grasp.py`): any force-closure pinch
  welds (a seated cell's top is all the fingertips reach, 31-35 deg above its mid-plane at best with 1 mm of side
  clearance and the rim 4 mm over it, so a seated cell is never taken; pried up by the stick it can be plucked), but the
  weak holds of a cell share **60 mm for the whole episode** (the owner's choice of 2026-10-19, after the metric review
  `<design record>/robotangle_mujoco/metric_review_20261019/battery_zh.md`, finding 7; until then each weak hold had its own 60 mm
  from where it was taken, so pluck, carry 55 mm, put down and pluck again moved a cell anywhere): each weak hold spends
  the straight-line distance the cell's centre moves from where its count starts to where it is let go, and its count
  starts where it is taken or, when it is taken with the cell still **in the compartment** (its centre inside the cavity
  between the compartment's walls, up to the rim: |x| <= 28, |y| <= 17.3, 0 <= z <= 18.5 mm in the compartment frame,
  `HOME_BOX_M`, the target's `home_link` / `home_box_m` on the controller's shell body), where the centre first leaves
  it (checked after every World tick); a hold that never leaves the compartment spends nothing. A weak hold is lost
  (reason `weak_hold_travel`, as a lost grasp: the gripper opens before it takes hold again) once what the cell's earlier
  weak holds spent plus its own distance so far exceed 60 mm; with the budget overspent a later weak hold of that cell is
  lost at its first check. The spent budget is part of the world's checkpoints (the grasp backend's slip state); a firm
  hold carries a cell anywhere and spends nothing;
- the **new cells** are `ForceClosureGraspTarget(friction coefficient 0.268)`: taken only by a firm pinch.
The reference also holds every carried cell near its middle (the two contacts' midpoint within 8 mm of the cell's centre
along it, the middle third of its 50.5 mm) with the gripper closed on it (the opening the cell's width at the contacts,
14.2-14.5 mm); the rules do not test that.

## Goal

`BatteryReplacementTask.evaluate_state` (terminal revision `battery_replacement_swap_v1`), all held for 0.5 s, holding
allowed, the stick anywhere:
- each slot holds a new cell, seated: its axis within 5 deg of the slot's, its centre within 1.2 mm sideways, 1 mm
  vertically and 1.5 mm along the axis of the seated pose, the slot's spring compressed at least 1 mm (a seated cell
  compresses it 1.78 mm; a pried one 0 or less); polarity right (the cell's + end toward the slot's + plate);
- the cover closed and latched: within 0.5 mm of its closed pose along and across, between 0.2 mm below and 0.6 mm above
  it (latched: +0.24 mm; slid open 5.2-5.5 mm; lying on top unlatched +3.7 mm), tilted at most 1 deg, the plunger's
  joint at least -0.25 mm. What makes the cover latched is the 0.5 mm along the slide: there the plunger (2.4 mm wide at
  the rim) sits at least 79 % in the cover's 2.4 mm groove. The plunger's own condition fails only while something holds
  the plunger down: its 2 N preload carries it past the joint's soft upper limit, so at rest it reads +0.36 to +0.42 mm
  with the cover latched, short of its place or absent (metric review 2026-10-19, finding 2; until that day this line
  said "in its groove", and the public goal "its detent plunger in the groove");
- both old cells' centres inside the recycling box (within its inner walls, below its rim);
- the four cells and the cover at most 5 mm/s and 0.1 rad/s.

The sideways tolerance is 1.2 mm: the owner's choice of 2026-10-17 (the proposal had 1.0
mm). The slot's free play is 1 mm each side, but a cell resting against a slot wall measures 1.03 mm off centre (its
contact sinks slightly), so 1.0 mm would fail a seated cell that only leans on the wall.

The design ran the shortcuts it could think of (design README.md section 5, `out/success_cases.txt`): the nop, the old
cells put back, one cell swapped, a reversed cell, the cover lying on top or left slid open or missing, a new cell held
half way, fingertips on a seated cell, a fingertip or the cover or a new cell as the lever: none succeeds.

## Reference (-i01)

`scripts/roboarc/oracles/battery-replacement-01-i01/privileged.npz` (`standard.npz` is the same file), 7029 rows
(175.7 s), native gripper space, on the instance `rcb_topogym.py make-instance --task battery_replacement --variation
battery_replacement_xbox_aa --scene-seed 0 --gripper-space native` (birth 764192d9a7e49a0f). sha256 472848140a225398.
No row moves an arm joint more than 0.035 rad. Recorded with

```bash
python scripts/roboarc/shared/rcb_topogym.py run-oracle --task battery_replacement --instance <instance.json> \
    --code tools/references/battery_replacement/oracle.py --out <dir>/privileged.npz
```

Rule C (2026-10-18) replaced the reference of 2026-10-17 (6453 rows, birth 20a67a185905da92, end 1b3de25c9df39f07): its
old cells were carried 16 cm hanging from a fingertip pinch (central angle 109 / 117 deg) and its new cells welded by
nearly closed fingertips on their top (45-60 deg, new cell 1 carried with no finger on it). The birth hash moved with
the grasp configuration frozen in the instance (the cells' targets); the scene is the same.

The weak-hold budget per episode (2026-10-19) re-recorded it on the new instance (birth 15a12bc37ada9259, before
ebd436acde9a62a1): the oracle wrote the same 7029 rows of actions and the same tick ledger, the physics ran the same (the
observer's peaks identical), and the final state moved from 7ac8b8d39269e320 to 3b286ba853cb929c with the frozen
grasp configuration and the goal text in the record. Each old cell has one weak hold (the pluck to the shell's back); it
leaves the compartment 1.3 / 2.7 mm (A / B) after it is taken (the pried cell's centre is then 17.4 / 16.0 mm above the
floor, the rim at 18.5) and spends 41.6 / 39.6 mm of its 60 (measured on the run r4 of the same actions,
`grasp_proposal/v4/r4/br_log.npz`).

The oracle (`tools/references/battery_replacement/`: `oracle.py`, its plan `plan.npz` / `plan.json`
exported by `export_plan.py` from the design's kinematic planner `<design record>/controller_battery_design_20261014/scripts/
d3_oracle.py` (v4), both arms' joints per control step) follows the plan and adapts every approach and every held part's
path to the parts' measured poses (IK near the plan's joints), with feedforwards of the drives' inverse dynamics, Coulomb
friction and the held part's weight, a slow integral of the joint error, the cover pushes as joint torques; the plan
waits at gates for the physics (a grip welded, the cover slid, a cell pried up 8 mm, a pinching hand settled within 0.5
mm, a new cell pressed down). Only joint and gripper targets are commanded. Steps (simulated time):
- 0-13 s, the cover: the left hand pinches its ridge, slides it +5.5 mm (the detent lets go), tilts it 14 deg about the
  lip, pulls the tongue out and parks the cover on its pad;
- 13-56 s, old cell A: the right hand takes the stick from its holder, puts the blade into A's notch and levers 60 deg
  about the shell's edge; the foot lifts the cell about 8 mm; at 30 s the left hand plucks it with a fingertip pinch (a
  weak hold: central angle 109 deg), lifts it 20 mm, moves it 8 mm toward -x (away from the right hand, whose gripper
  base it touches about 130 mm above the cell) and lays it on the shell's back outside the cover's recess (-y strip, 31
  mm off the compartment's centre line; the farthest it gets from where it was taken: 44.3 mm; it spends 41.6 mm of
  the 60 from where its centre leaves the compartment, at 18.6 mm above the floor);
  the hand lets go, rises 6 mm and at 36 s pinches the cell's middle from both sides (171 deg, the contacts 1.1 mm from
  the centre, opening 13.9 mm), carries it to the box and drops it there; the stick goes back into its holder;
- 57-102 s, old cell B: the same with the hands swapped, laid on the +y strip (pluck 118 deg, 43.2 mm; middle 176 deg);
- 107-131 s, a new cell into A: the left hand takes new cell 2 in the tray from both sides at its middle (169 deg, the
  contacts 7.1 mm toward the + end, opening 14.3 mm; the hand comes down tilted 23 deg, the grasp it keeps until the cell
  leans in its slot at about 22 deg), turns it to the slot's polarity and lets its - end down against the spring contact
  (the fingertips 1.18 mm over the rim at the release); the right hand's nearly closed fingertips push the + face toward
  the spring and press the + end down;
- 137-160 s, new cell 1 into B: the same with the hands swapped (168 deg; 1.39 mm). The left hand takes new cell 2 and
  the right hand new cell 1 because new cell 1's + end points +x in the tray (seed 0): with it the left hand's tilted
  grasp would turn its fingertips back toward its own base (no IK); the goal takes either new cell in either slot;
- 165-176 s, the cover: the left hand puts the tongue under the lip, lays the cover flat, lets go of the ridge and pushes
  it -5.5 mm with a fingertip until the plunger snaps into its groove, then lets go and lifts 6 cm while the goal holds.

Option C (2026-10-19) re-stamped it (`<design record>/controller_battery_design_20261014/scripts/restamp.py`: `instance_sha`
rewritten, the 7029 rows and the tick ledger byte for byte the same) with the birth of the instance with the outer body,
764192d9a7e49a0f, after the reference's qpos and qvel matched bit for bit at every control step on the instance before C
(15a12bc37ada9259, the source of e85fe67bd7) and on C's (`xbox_visual/qdump.py`, `qcompare.py`; 7030 x 61 qpos, 7030 x 55
qvel). Nothing comes within 2.0 mm of the new pieces at any step (`qdump.py`, MuJoCo's geom distances from every robot
and part geom: the right hand's finger 2.05 mm from the top edge, a left finger 2.26 mm from the bottom edge, the stick's
blade 3.0 mm from the fill (through the floor), its shaft 4.7 mm from the side, 11.1 mm from a grip). Two fresh-process
strict replays: success at row 7029, final state de1cb2ee33e369ed (the end's record carries the birth's), 131 s and
117 s; the observer CLEAN with the same peaks (`port/i01c/`).

Checks of the reference under the per-episode budget (2026-10-19, an RTX 4090 host, the same image, rendering off, one process
with one thread at a time; `<design record>/controller_battery_design_20261014/port/i01v5/`): the opt-in tests pass in the image
(18); run-oracle success at row 7029, final state 3b286ba853cb929c, 133 s; two fresh-process strict replays success at
row 7029, final state 3b286ba853cb929c, 126 s and 140 s; the observer CLEAN with the same peaks as on 10-18 (no event, no
grasp lost, deepest task contact 0.82 mm, largest contact force 62.5 N, 44 ms per control step).

Checks of the reference (2026-10-18, an RTX 4090 host, a development image (0.3.2-rc1), mujoco 3.14, rendering off,
one process with one thread at a time; `<design record>/controller_battery_design_20261014/port/i01v4/`; birth ebd436acde9a62a1):
- One run of the oracle: success at row 7029, final state 7ac8b8d39269e320, 132 s of wall clock.
- Two fresh-process strict replays (`rcb_topogym.py replay --repeat 1 --strict`): both success at row 7029, final state
  7ac8b8d39269e320, 122 s and 125 s.
- The stress observer (`tools/stress/replay.py --hold 0`): CLEAN; no event, no grasp lost, no review, no
  forcing; deepest task contact 0.82 mm; largest contact force 62.5 N; fastest part 1.13 m/s (an old cell dropped into
  the box); energy created 0.0027 J; 41 ms per control step.
- The equal-length nop (7029 rows of the birth targets, under the observer): no success, CLEAN.
- reset() (`port/reset_check.py`): three times the first 2400 rows, each ending on 00e52d5419a95eac, each reset back to
  the birth.
- Robustness: uniform noise of +-1e-3 rad on every arm joint command (`BR_NOISE_RAD`), seeds 1-4: all four succeed
  (7004-7039 rows).
- The grasps measured on its replay (`<design record>/controller_battery_design_20261014/grasp_proposal/v4_check.py`, the run r4 of
  the same trajectory): the two plucks 108.9 / 117.7 deg (35 / 31 deg above the mid-plane, 5.7 / 4.3 mm toward the +
  end); the cells lying on the shell's back move 0.56 / 0.71 mm before the regrasp; no finger touches the controller
  while a new cell goes in.

## Rule C and the other families

`WeakHoldGraspTarget` is an opt-in subclass: a configuration that does not use it keeps every dataclass field (so every
instance's grasp state and state hash) and never reaches its code. Checked by replay (2026-10-18, an RTX 4090 host, the same
image, rendering off; `<design record>/controller_battery_design_20261014/ab/`): on one -i01 reference per MuJoCo family that uses
the contact-weld grasp backend (the shortest of each), the instance made and the reference replayed strictly on the tree
without the change (6d67001cf7) and with it: the same birth and the same end state for all ten that build from a
checkout (slotted-post-ring-01, interlocked-claws-03, cast-coil-01, hedgehog-in-cage-02, orbits-01, fiber-bundling-02,
lock-picking-01, cast-keyhole-01, horseshoe-ring-02, buzz-wire-3d-04; every one a success). knot-untying-03 and
rope-threading-01 do not build from a checkout (their MuJoCo ropes need the released instance's frozen birth) and were
not replayed; the rope-chain and grasp paths they use run in slotted, orbits and fiber above.

## Stress (docs/simulator/STRESS_HOWTO.md)

The -i01 of 2026-10-19 (the per-episode weak-hold budget and the outer body; birth 764192d9a7e49a0f) on a1027c0665
(an RTX 4090 host, assigned by the migration session; a git archive of the commit, its id checked; a development image (0.3.2-rc1),
one container jwbat-stress-<stage> of 6 CPUs, 4 search and 2 replay workers, one thread each, started below 80 % load and
memory; `<host work dir>/bat_stress/out/battery_r3/` there), criteria v2.13b:
- self-check (seeds 123401-123406, 2026-10-20 06:59-07:17 UTC): PASS. The instance made from the checkout is born
  764192d9a7e49a0f (= the reference's); the reference success, CLEAN, the same hash in two fresh processes (9.2 min); the
  nop no success, CLEAN; 62 episodes, 12,520 controls, 0 confirmed BREAKDOWN, 0 grasp slips.
- acceptance round (seeds 123501-123509, 2026-10-20 07:50-10:38 UTC): PASS. 1,000 episodes, 465,750 controls, every
  strategy (random 41,178, directed 23,103, mutate 18,573, cem 23,268, periodic 86,049, prefix 273,579 controls); the
  reference success, CLEAN, final state de1cb2ee33e369ed in both fresh processes (7.6 min); the nop no success, CLEAN;
  0 confirmed BREAKDOWN under every reading (v2 ... v2.13b); in-search VIOLENT 7 (all prefix); 1 pop (0.60 m/s); lost
  holds over the 1,000 episodes: 15 grasp slips, 4 relative-pose errors, 2 fingers opened and 2 by rule C's
  weak_hold_travel (prefix2 episodes 0 and 7, both CLEAN). The host's load rose to 98 during the round (other users);
  memory stayed at 18-45 %.

Rule C's -i01 on the test-only commit 2186292da1 (an RTX 4090 host, a development image (0.3.2-rc1), 4 search and 2 replay
workers, one thread each; `<host work dir>/bat_stress/out/battery_r2/` there), criteria v2.13b:
- self-check (seeds 123201-123206): PASS. The reference success, CLEAN, the same hash in two fresh processes (5.1 min);
  the nop no success, CLEAN; 61 episodes, 17,110 controls, 0 confirmed BREAKDOWN.
- acceptance round (seeds 123301-123309, 2026-10-19 01:20-04:37 UTC): PASS. 1,000 episodes, 479,948 controls, every
  strategy; 0 confirmed BREAKDOWN under every reading (v2 ... v2.13b); in-search VIOLENT 14 (random 1, cem 3, prefix
  10); 1 pop (0.88 m/s); 15 grasp slips; among the saved episodes one hold lost by rule C's weak_hold_travel.
The round before rule C (battery_r1 on 780d3a669e, seeds 123001-123006 and 123101-123109, PASS) no longer stands for the
base; its outputs are kept. battery_r2 ran before the per-episode budget and the outer body of 2026-10-19; the base's
birth 764192d9a7e49a0f has its own above (battery_r3; a self-check of the budget alone, queued on 10-19, was withdrawn
before it started when the owner chose option C, and the local self-check of C was withdrawn for an RTX 4090 host).

## Harbor

The props' look (2026-10-21, physics unchanged): `release_rebuild.py 1965935f90 --version 0.3.0-dryrun-battery-replacement-4 --families
battery_replacement --bench robotangle --jobs 1 --lanes 1` on the branch's own commit (<test branch>; rendering off;
an RTX 4090 host, load average 57 when it started; `<design record>/controller_battery_design_20261014/harbor_test/work4/`): PASS. Born
764192d9a7e49a0f (= the reference's); `mk_tasks.py --check` ok, all bases and the MuJoCo instance bytes; oracle 1 and nop 0
in both modes, the oracle's end state de1cb2ee33e369ed in both, the evidence claims hold (the standard live episode
292.4 s on the loaded host). Nothing uploaded. New cam_high stills (birth, row 3514, end: `realism2/media/`) and the demo
(`out/topogym_demo_r2/`, the oracle live: success, de1cb2ee33e369ed).

The per-episode budget and the outer body (2026-10-20): `release_rebuild.py a1027c0665 --version
0.3.0-dryrun-battery-replacement-3 --families battery_replacement --bench robotangle --jobs 1 --lanes 1` on the branch's own
commit (the registration and the manifest's new base are in dev below it, so no test-only commit; rendering off;
an RTX 4090 host, load average 6.8; `<design record>/controller_battery_design_20261014/harbor_test/work3/`): PASS. The base made in the
image is born 764192d9a7e49a0f (= the reference's; instance 40 files, 323,567 bytes, the outer body's meshes); `mk_tasks.py
--check` ok, all bases and the MuJoCo instance bytes; oracle 1 and nop 0 in both modes, the oracle's end state
de1cb2ee33e369ed in both, the evidence claims hold. The standard live episode took 132.6 s (0.0189 s per control step),
its collect hook's replay 167.9 s; the privileged oracle trial 334 s (its grader's two replays). The texts keep 0.04 s per
step.

Rule C's -i01 (2026-10-18): `release_rebuild.py 2186292da1 --families battery_replacement --bench robotangle --jobs 1
--lanes 1` (the test-only branch <test branch>: this family's 38a4a0a023 with the shared
a83a53579d below it, the registration and the manifest entry as test-only commits; rendering off; an RTX 4090 host, load average
10.7, image a development image (0.3.0-dryrun-battery-replacement-2)): PASS. The base made in the image is born
ebd436acde9a62a1 (= the reference's); `mk_tasks.py --check` ok; oracle 1 and nop 0 in both modes, the oracle's end state
7ac8b8d39269e320 in both, the evidence claims hold. The standard live episode took 137.0 s (0.0195 s per control step),
its collect hook's replay 197.6 s; the privileged grader's two replays 342 s. The texts keep 0.04 s per step (the run
below, at load average 34.6).

Before rule C (the reference of 2026-10-17):
`release_rebuild.py 780d3a669e --families battery_replacement --bench robotangle --jobs 1 --lanes 1` (a test-only
commit of the temporary branch <test branch>: this family's commit 861930b5bf, the registration
and a manifest entry that lists the base `battery-replacement-01-i01`, template `interlocked-claws-easy-i00`; rendering
off; 2026-10-17, an RTX 4090 host, image a development image (0.3.0-dryrun-battery-replacement-1)): PASS. The base made in the image
is born 20a67a185905da92 (= the reference's); `mk_tasks.py --check` ok; oracle 1 and nop 0 in both modes, the oracle's
end state 1b3de25c9df39f07 in both, the evidence claims hold. The standard live episode took 255.6 s (0.040 s per
control step; the trial 625.7 s with the collect hook's fresh replay, 293.9 s), the privileged grader's two replays
636 s. The texts state 0.04 s per step (`package_text/battery_replacement/README.md`). The HF upload of the instance and
the image push are gated, not done.
