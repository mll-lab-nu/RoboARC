# Bed bolt

Family key `bed_bolt` · 5 task bases · robot `aloha-2` · physics MuJoCo 3.14.0

A small bed frame stands on the table. A threaded blind hole runs into the corner block in the far corner of its right
end board; in front of the frame the long bolt lies in a V cradle and an L-shaped key stands in its stand. The robot
must carry the bolt to the hole, push its tip in until the thread catches, then drive it home with the key until its
head seats on the block. A cross board in front of the block face closes a bay round the hole, too short for the key's
long arm along the bolt: the key's short arm goes into the bolt's head and the long arm stands up as a lever, swung
60 degrees at a time. The bases differ in the drive: a hex or Torx key that comes out of the head between swings, or a
ratcheting key (square, Phillips or slotted tip) whose tip stays in the head while its handle swings back. Since 2.1.0
a white card lies flat on the table between the key's stand and the bolt's cradle, with the bolt's head and its
recess drawn 4.5 times their size, so the five drives tell apart at a glance in the cameras; it is drawn only (no
collision, no mass) and white at every appearance seed (its drawing unchanged). The family
was designed by Jianwen; its bases came to RoboARC as experimental bases in 1.2.0 and were released in 1.4.0.

## Robot and grasping

Two 6-DoF ALOHA 2 (ViperX) arms with parallel-jaw grippers on a fixed base (`aloha-2`). Actions are 14-D joint-position
targets (6 arm joints and one gripper target per arm) at 40 Hz; standard mode also accepts end-effector pose targets.

Assisted grasping (`contact_weld_v1`) of the bolt and the key: a closing gripper takes hold of one once both fingers
touch it inside its grasp region, with the tool frame within 15 cm of it and the penetration at most 2 mm, for
0.02 s; the part then follows the gripper rigidly. Any opening command releases it. A hold is lost if the fingers open
more than 15 %, the part drifts 1 cm / 0.15 rad from its held pose, or the hold has to push or pull its part with more
than 60 N (averaged over 20 ms); the gripper must then open before it can take hold again. Once the thread has caught,
the bolt is no longer a graspable part: a hold on it is lost at that moment, and fingers on it touch it by friction
only. Two grippers never hold the same part, and other objects are grasped physically (friction). The grippers start
empty and nearly closed: open a gripper, then close it on a part to take hold.

## Success rule

What the grader checks, in plain words (each task's `instruction.md` states the exact rule and its limits):

- **Seated.** The bolt is driven in until its head is within 1.5 mm of the block face, its tip at least 83.5 mm deep in
  the hole, with the thread caught, and its axis within 2 degrees of the hole's axis.
- **The thread.** It catches when the bolt's tip is 2.5 to 8.2 mm inside the hole, within 1 mm of the hole's axis and
  2 degrees of its direction, with the bolt's thread lined up with the hole's; each turn then drives the bolt 5.7 mm in.
  A bolt the thread does not hold cannot be pushed past the thread: not past 8.2 mm before it has ever caught, and
  afterwards not deeper than it has already been turned in, where it catches again in the same thread. Once the thread has engaged, pushing or pulling the bolt along its
  axis does not advance it, and fingers cannot turn it (they slip): only the key does. With fewer than 2 turns engaged
  the thread cannot carry the bolt: when nothing holds it, it droops, and past 3 degrees it lets go and falls out.
- **Ratchet** (bases 03-05). Swung in the tightening direction the key turns its tip and the bolt; swung back, the
  handle turns freely (with a slight drag) while the tip stays in the head, so the tip need not come out between swings. In base 05 the slot runs across the whole head, open at both
  ends, so the blade can slide out along it.
- **Still.** The bolt and the key move at most 1 mm/s and 0.02 rad/s. Holding is allowed; the key may stay in the head
  or anywhere else.
- All final conditions must hold together, continuously, for 0.5 s of simulated time.

A task piece falling more than 100 mm below the tabletop ends the episode in failure. Contact penetration does not end
the episode by itself. A success with deep interpenetration (beyond the limits stated in the task text) is held for
human review instead of being credited automatically, and a task contact deeper than 10 mm counts as a simulator
breakdown.

Every task has a wall-clock budget of 10800 s (3 h) in both modes. In privileged mode the delivered trajectory is
replayed open-loop twice in a fresh process: both replays must succeed, and the strict goal must be reached within the
trajectory's own rows. In standard mode the simulator service grades the live episode and a fresh-process replay
confirms it. The details are on the [benchmark page](../benchmarks/roboarc.md) (section "Grading replays").

## Bases

5 bases (very_easy 1, easier 1, easy 1, medium 1, hard 1). Each base is one frozen scene instance and ships as two
Harbor task directories, `-privileged` and `-standard`, with the same scene, goal and grader. The difficulty is
`[metadata].difficulty` in each `task.toml`.

| Base | Difficulty | Variation (task text) | Notes | Task dirs |
|---|---|---|---|---|
| `roboarc-bed-bolt-01-i01` | medium | `bed_bolt_crossbar` | L-shaped hex key in a hex socket | [privileged](../../tasks/roboarc-bed-bolt-01-i01-privileged/) / [standard](../../tasks/roboarc-bed-bolt-01-i01-standard/) |
| `roboarc-bed-bolt-02-i01` | hard | `bed_bolt_torx` | L-shaped Torx key (a six-lobe star tip) | [privileged](../../tasks/roboarc-bed-bolt-02-i01-privileged/) / [standard](../../tasks/roboarc-bed-bolt-02-i01-standard/) |
| `roboarc-bed-bolt-03-i01` | very_easy | `bed_bolt_square` | ratcheting L-shaped square key | [privileged](../../tasks/roboarc-bed-bolt-03-i01-privileged/) / [standard](../../tasks/roboarc-bed-bolt-03-i01-standard/) |
| `roboarc-bed-bolt-04-i01` | easier | `bed_bolt_phillips` | ratcheting L-shaped Phillips key (a cross tip) | [privileged](../../tasks/roboarc-bed-bolt-04-i01-privileged/) / [standard](../../tasks/roboarc-bed-bolt-04-i01-standard/) |
| `roboarc-bed-bolt-05-i01` | easy | `bed_bolt_slotted` | ratcheting L-shaped slotted key (a flat blade); the slot is open at both ends | [privileged](../../tasks/roboarc-bed-bolt-05-i01-privileged/) / [standard](../../tasks/roboarc-bed-bolt-05-i01-standard/) |

## Links

- [RoboARC benchmark page](../benchmarks/roboarc.md): the rules shared by all families (grading replays, safety readings in standard mode, the standard-mode service, the time budget).
- [Standard mode](../STANDARD_MODE.md) and [protocol](../PROTOCOL.md).
- [All families](README.md).
- [bed_bolt.md](../simulator/families/bed_bolt.md): development notes (the scene, the thread mechanism, the reference, stress acceptance, the drives of bases 02-05).
