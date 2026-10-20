# battery_replacement on MuJoCo (swap a game controller's AA cells)

Jianwen, 2026-10-17. A new family, MuJoCo only: two ALOHA arms open a game controller's battery cover, pry the two old AA
cells out with a pry stick, drop them into a recycling box, seat two new cells with the right polarity and latch the
cover. First setting `battery_replacement_xbox_aa` (pool experimental, no tier; model revision `xbox_aa_v3b`), base
`battery-replacement-01-i01`. Design record, sources and every measurement: `runs/controller_battery_design_20261014/`
(README.md v2 design, README_v3.md the v3b demo and section 10 the topogym port, REGISTRATION_PROPOSAL.md; not in git).
The registration (registry, variation record, profile, rcb_topogym TASKS, goal_description, topogym_rules, mk_tasks) waits
for the user's approval of the proposal.

## The task

An Xbox wireless controller's battery compartment at 1:1 (two AA cells, Energizer E91 sizes: 14.5 mm across, 50.5 mm
long with the 1 mm nub, 23 g). The scene (`topogym/assets/geometry/battery_replacement.py`, laid out in the ALOHA mount
frame; m, z = 0 the table top):

- the controller face down, fixed in its cradle, the compartment's centre at (0, 0.12), its floor 34 mm above the table:
  a cavity 56 x 34.6 mm, 18.5 mm deep, two slots split by a 1.6 mm rib; each slot has a + contact plate at one end (slot
  A's + end toward +x, slot B's toward -x) and a spring contact at the other (a slide joint, 1.2 N/mm, 2 mm compressed by
  a seated cell, at most 1.5 mm more); the rim stands 4 mm above a seated cell, with 1 mm of side clearance around it;
  beside each + end a pry notch (4 x 6.55 mm, 2 mm into the end wall and the long wall, a 1 mm floor groove);
- the cover (59.6 x 38.6 x 2 mm, a tongue, three hooks, a 3 mm ridge, 5.5 g), closed and latched: it unlatches by sliding
  5 mm toward the controller's top edge (+y) past a detent plunger (2 N preload, 0.5 N/mm; measured 5.0 N to open, 5.9 N
  to close), then lifts off;
- the old cells old_A, old_B seated (slot A + end toward +x, slot B toward -x);
- the pry stick standing blade down in its holder at (0, 0.225): a 2 x 1.2 x 30 mm blade with a 2.0 x 1.0 x 1.2 mm foot
  at its tip toward the cell's axis, a 6 x 6 x 40 mm shaft, a 16 x 16 x 50 mm handle, 30 g; the holder's 6.8 mm square
  hole has a 2 mm x 45 deg lead-in at its mouth (`STAND_FUNNEL`: the hole's 0.4 mm play is less than the hands' tracking
  error);
- the new cells new_1, new_2 in the tray at (0.065, -0.025); the recycling box (inner 70 x 50 x 30 mm) at
  (-0.065, -0.025), empty; a pad for the cover at (-0.12, 0.22) (display only).

The new cells lie in the tray as `+-` (new_1's + end toward +x, new_2's toward -x) for every seed: until 2026-10-20 the
scene seed chose the polarity (seed % 4: `+-`, `-+`, `++`, `--`; seed 0 `+-`); since then a seed varies appearance only
(the user). The appearance parameters `controller_finish` (outer body and cover), `tray_finish`, `box_finish` and
`props_finish` (cradle, stick holder, cover pad) re-colour visual geoms only (`scene_variation.appearance`: appearance
seed 0 is the recorded look); the cells, the red "+" marks and the stick keep the colours the texts name. Everything else
is fixed.

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
stick's around its handle; the tool frame within 6 cm). The cells follow rule C (the user's choice of 2026-10-18, the
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
  clearance and the rim 4 mm over it, so a seated cell is never taken; pried up by the stick it can be plucked), but a
  weak hold is lost (reason `weak_hold_travel`, as a lost grasp: the gripper opens before it takes hold again) once the
  cell's centre is more than 60 mm from where it was taken; a firm hold carries it anywhere;
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
  it (latched: +0.24 mm; slid open 5.2-5.5 mm; lying on top unlatched +3.7 mm), tilted at most 1 deg, the plunger at
  least -0.25 mm (in its groove);
- both old cells' centres inside the recycling box (within its inner walls, below its rim);
- the four cells and the cover at most 5 mm/s and 0.1 rad/s.

The sideways tolerance is 1.2 mm: the user's choice of 2026-10-17 (confirmed in the main session; the proposal had 1.0
mm). The slot's free play is 1 mm each side, but a cell resting against a slot wall measures 1.03 mm off centre (its
contact sinks slightly), so 1.0 mm would fail a seated cell that only leans on the wall.

The design ran the shortcuts it could think of (design README.md section 5, `out/success_cases.txt`): the nop, the old
cells put back, one cell swapped, a reversed cell, the cover lying on top or left slid open or missing, a new cell held
half way, fingertips on a seated cell, a fingertip or the cover or a new cell as the lever: none succeeds.

## Reference (-i01)

`scripts/robotangle/oracles/battery-replacement-01-i01/privileged.npz` (`standard.npz` is the same file), 7029 rows
(175.7 s), native gripper space, on the instance `rcb_topogym.py make-instance --task battery_replacement --variation
battery_replacement_xbox_aa --scene-seed 0 --gripper-space native` (birth ebd436acde9a62a1). sha256 005f718ce4b0e0fa.
No row moves an arm joint more than 0.035 rad. Recorded with

```bash
python scripts/robotangle/shared/rcb_topogym.py run-oracle --task battery_replacement --instance <instance.json> \
    --code scripts/robotangle/references/battery_replacement/oracle.py --out <dir>/privileged.npz
```

Rule C (2026-10-18) replaced the reference of 2026-10-17 (6453 rows, birth 20a67a185905da92, end 1b3de25c9df39f07): its
old cells were carried 16 cm hanging from a fingertip pinch (central angle 109 / 117 deg) and its new cells welded by
nearly closed fingertips on their top (45-60 deg, new cell 1 carried with no finger on it). The birth hash moved with
the grasp configuration frozen in the instance (the cells' targets); the scene is the same.

The oracle (`scripts/robotangle/references/battery_replacement/`: `oracle.py`, its plan `plan.npz` / `plan.json`
exported by `export_plan.py` from the design's kinematic planner `runs/controller_battery_design_20261014/scripts/
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
  mm off the compartment's centre line; the farthest it gets from where it was taken: 44.3 mm, the rule's limit 60);
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

Checks of the reference (2026-10-18, a lab RTX 4090 host, ghcr.io/mll-lab-nu/eai-robotangle:0.3.2-rc1, mujoco 3.14, rendering off,
one process with one thread at a time; `runs/controller_battery_design_20261014/port/i01v4/`):
- One run of the oracle: success at row 7029, final state 7ac8b8d39269e320, 132 s of wall clock.
- Two fresh-process strict replays (`rcb_topogym.py replay --repeat 1 --strict`): both success at row 7029, final state
  7ac8b8d39269e320, 122 s and 125 s.
- The stress observer (`scripts/robotangle/stress/replay.py --hold 0`): CLEAN; no event, no grasp lost, no review, no
  forcing; deepest task contact 0.82 mm; largest contact force 62.5 N; fastest part 1.13 m/s (an old cell dropped into
  the box); energy created 0.0027 J; 41 ms per control step.
- The equal-length nop (7029 rows of the birth targets, under the observer): no success, CLEAN.
- reset() (`port/reset_check.py`): three times the first 2400 rows, each ending on 00e52d5419a95eac, each reset back to
  the birth.
- Robustness: uniform noise of +-1e-3 rad on every arm joint command (`BR_NOISE_RAD`), seeds 1-4: all four succeed
  (7004-7039 rows).
- The grasps measured on its replay (`runs/controller_battery_design_20261014/grasp_proposal/v4_check.py`, the run r4 of
  the same trajectory): the two plucks 108.9 / 117.7 deg (35 / 31 deg above the mid-plane, 5.7 / 4.3 mm toward the +
  end); the cells lying on the shell's back move 0.56 / 0.71 mm before the regrasp; no finger touches the controller
  while a new cell goes in.

## Rule C and the other families

`WeakHoldGraspTarget` is an opt-in subclass: a configuration that does not use it keeps every dataclass field (so every
instance's grasp state and state hash) and never reaches its code. Checked by replay (2026-10-18, a lab RTX 4090 host, the same
image, rendering off; `runs/controller_battery_design_20261014/ab/`): on one -i01 reference per MuJoCo family that uses
the contact-weld grasp backend (the shortest of each), the instance made and the reference replayed strictly on the tree
without the change (6d67001cf7) and with it: the same birth and the same end state for all ten that build from a
checkout (slotted-post-ring-01, interlocked-claws-03, cast-coil-01, hedgehog-in-cage-02, orbits-01, fiber-bundling-02,
lock-picking-01, cast-keyhole-01, horseshoe-ring-02, buzz-wire-3d-04; every one a success). knot-untying-03 and
rope-threading-01 do not build from a checkout (their MuJoCo ropes need the released instance's frozen birth) and were
not replayed; the rope-chain and grasp paths they use run in slotted, orbits and fiber above.

## Stress (docs/rcb/STRESS_HOWTO.md)

Rule C's -i01 on the test-only commit 2186292da1 (a lab RTX 4090 host, `eai-robotangle:0.3.2-rc1`, 4 search and 2 replay
workers, one thread each; `<host work dir>/bat_stress/out/battery_r2/` there), criteria v2.13b:
- self-check (seeds 123201-123206): PASS. The reference success, CLEAN, the same hash in two fresh processes (5.1 min);
  the nop no success, CLEAN; 61 episodes, 17,110 controls, 0 confirmed BREAKDOWN.
- acceptance round (seeds 123301-123309, 2026-10-19 01:20-04:37 UTC): PASS. 1,000 episodes, 479,948 controls, every
  strategy; 0 confirmed BREAKDOWN under every reading (v2 ... v2.13b); in-search VIOLENT 14 (random 1, cem 3, prefix
  10); 1 pop (0.88 m/s); 15 grasp slips; among the saved episodes one hold lost by rule C's weak_hold_travel.
The round before rule C (battery_r1 on 780d3a669e, seeds 123001-123006 and 123101-123109, PASS) no longer stands for the
base; its outputs are kept.

## Harbor

Rule C's -i01 (2026-10-18): `release_rebuild.py 2186292da1 --families battery_replacement --bench robotangle --jobs 1
--lanes 1` (the test-only branch jianwen/battery-replacement-harbor-test2: this family's 38a4a0a023 with the shared
a83a53579d below it, the registration and the manifest entry as test-only commits; rendering off; a lab RTX 4090 host, load average
10.7, image `eai-robotangle:0.3.0-dryrun-battery-replacement-2`): PASS. The base made in the image is born
ebd436acde9a62a1 (= the reference's); `mk_tasks.py --check` ok; oracle 1 and nop 0 in both modes, the oracle's end state
7ac8b8d39269e320 in both, the evidence claims hold. The standard live episode took 137.0 s (0.0195 s per control step),
its collect hook's replay 197.6 s; the privileged grader's two replays 342 s. The texts keep 0.04 s per step (the run
below, at load average 34.6).

Before rule C (the reference of 2026-10-17):
`release_rebuild.py 780d3a669e --families battery_replacement --bench robotangle --jobs 1 --lanes 1` (a test-only
commit of the temporary branch jianwen/battery-replacement-harbor-test: this family's commit 861930b5bf, the registration
and a manifest entry that lists the base `battery-replacement-01-i01`, template `interlocked-claws-easy-i00`; rendering
off; 2026-10-17, a lab RTX 4090 host, image `eai-robotangle:0.3.0-dryrun-battery-replacement-1`): PASS. The base made in the image
is born 20a67a185905da92 (= the reference's); `mk_tasks.py --check` ok; oracle 1 and nop 0 in both modes, the oracle's
end state 1b3de25c9df39f07 in both, the evidence claims hold. The standard live episode took 255.6 s (0.040 s per
control step; the trial 625.7 s with the collect hook's fresh replay, 293.9 s), the privileged grader's two replays
636 s. The texts state 0.04 s per step (`package_text/battery_replacement/README.md`). The HF upload of the instance and
the image push are gated, not done.
