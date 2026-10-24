# Caliper sorting

Family key `caliper_sorting` · 1 task base · robot `aloha-2` · physics MuJoCo 3.14.0

Four sleeves stand in a rack. They look alike and differ only in their inside and outside diameters. Each of four bins
carries a label with two readings of a vernier caliper ("ID r1 / OD r2": the inside jaws spread in a sleeve's bore, the
outside jaws closed on its outside). The caliper is fixed face up on a stand, its jaws out over the table, its slider
locked; it reads in its own unit, not in millimetres, and its zero is its own. The robot must loosen the slider's
locking screw, bring each sleeve to the jaws, set the jaws, read the vernier and put every sleeve into the bin whose
label gives its two readings. The grippers cannot gauge the sleeves, and a size known in millimetres any other way
cannot be matched to a label: the readings must come from the scale. The family was designed by Jianwen; its base
came to RoboARC as an experimental base in 1.2.0 and was released in 1.4.0.

## Robot and grasping

Two 6-DoF ALOHA 2 (ViperX) arms with parallel-jaw grippers on a fixed base (`aloha-2`). Actions are 14-D joint-position
targets (6 arm joints and one gripper target per arm) at 40 Hz; standard mode also accepts end-effector pose targets.

Assisted grasping (`contact_weld_v1`) of the four sleeves and of the caliper's slider (the moving jaws): a closing
gripper takes hold of a part once both fingers touch it inside its grasp region, with the tool frame within 15 cm of it
and the penetration at most 2 mm, for 0.02 s; the part then follows the gripper rigidly. Any opening command releases
it. A hold is lost if the fingers open more than 15 %, the part drifts 1 cm / 0.15 rad from its held pose, or the hold
has to push or pull its part with more than 60 N (averaged over 20 ms); the gripper must then open before it can take
hold again. Two grippers never hold the same part, and other objects are
grasped physically (friction). The locking screw's knob is not a grasp target: it is turned by pinching it and rotating
the gripper about its axis, by friction only.

## Success rule

What the grader checks, in plain words (each task's `instruction.md` states the exact rule and its limits):

- **Sorted.** Each sleeve belongs in the bin whose two readings both match its own (within 0.1 of a reading). Every
  sleeve's centre ends inside the bin of its class: within 40 mm of the bin's centre in both horizontal directions (the
  bin's 80 mm x 80 mm cavity) and 3 mm to 40 mm above the tabletop. A bin may stay empty; each bin holds two sleeves
  standing.
- **The caliper.** Its scale unit is one main-scale division (the vernier reads tenths of it); the size of that unit and
  the reading with the jaws closed are this caliper's own and are not given. The slider is locked at the start by its
  locking screw, a knurled knob on the slider's front face that turns between two stops 45 degrees apart, clockwise
  (seen from the scale side) tightening: within 10 degrees of the loose stop the slider moves freely (about 1.5 N of
  friction); from 20 degrees on it is locked where it stands. Whether it is locked at the end does not matter.
- **Released.** No sleeve is still held by an assisted grasp, and no part of the robot presses on a sleeve with more than
  0.05 N.
- **Still.** Every sleeve moves at most 1 mm/s and 0.02 rad/s.
- All final conditions must hold together, continuously, for 0.5 s of simulated time.

A sleeve left in a bin that is not its own (its centre inside that bin as above, released and still as above) for 0.5 s
ends the episode in failure: a sleeve set down to be taken again goes on the table, not into another bin. A task piece
falling more than 100 mm below the tabletop ends the episode in failure. Contact penetration does not end the episode
by itself. A success with deep interpenetration (beyond the limits stated in the task text) is held for human review
instead of being credited automatically, and a task contact deeper than 10 mm counts as a simulator breakdown.

Every task has a wall-clock budget of 10800 s (3 h) in both modes. In privileged mode the delivered trajectory is
replayed open-loop twice in a fresh process: both replays must succeed, and the strict goal must be reached within the
trajectory's own rows. In standard mode the simulator service grades the live episode and a fresh-process replay
confirms it. The details are on the [benchmark page](../benchmarks/roboarc.md) (section "Grading replays").

## Bases

1 base (hard 1). It is one frozen scene instance and ships as two Harbor task directories, `-privileged` and
`-standard`, with the same scene, goal and grader. The difficulty is `[metadata].difficulty` in each `task.toml`.

| Base | Difficulty | Variation (task text) | Notes | Task dirs |
|---|---|---|---|---|
| `roboarc-caliper-sorting-01-i01` | hard | `caliper_sorting_vernier` | four sleeves of two classes, four labelled bins, a vernier caliper fixed on a stand | [privileged](../../tasks/roboarc-caliper-sorting-01-i01-privileged/) / [standard](../../tasks/roboarc-caliper-sorting-01-i01-standard/) |

## Links

- [RoboARC benchmark page](../benchmarks/roboarc.md): the rules shared by all families (grading replays, safety readings in standard mode, the standard-mode service, the time budget).
- [Standard mode](../STANDARD_MODE.md) and [protocol](../PROTOCOL.md).
- [All families](README.md).
- [caliper_sorting.md](../simulator/families/caliper_sorting.md): development notes (the caliper on its stand, its lock, the sleeve classes, the reference).
