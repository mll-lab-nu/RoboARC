# Clock assembling

Family key `clock_assembling` · 1 task base · robot `aloha-2` · physics MuJoCo 3.14.0

The gear train of a watch movement, enlarged 8 times, lies in parts on the table: a main plate on a holder, the barrel
(a drum with its arbor, ratchet and square, on a mainspring), the center, third and fourth wheels on a stand, the train
bridge, two screws in a tray and a probe on its stand. The robot must seat the four wheels on their pivots in the main
plate, each pinion meshing with the previous wheel's teeth, lay the bridge over them on its pillars with every upper
pivot in its hole, turn the two bridge screws tight, then wind the barrel through its ratchet, let go and let the
mainspring run the train. Each wheel lies over the one before it, so the train goes together in one order only. The
scene is dynamic: the wound mainspring drives the gear train on its own (`[metadata] dynamics = "dynamic"`). The family
was designed by Jianwen; its base came to RoboARC as an experimental base in 1.2.0 and was released in 1.4.0.

## Robot and grasping

Two 6-DoF ALOHA 2 (ViperX) arms with parallel-jaw grippers on a fixed base (`aloha-2`). Actions are 14-D joint-position
targets (6 arm joints and one gripper target per arm) at 40 Hz; standard mode also accepts end-effector pose targets.

Assisted grasping (`contact_weld_v1`) of every part (the barrel, its arbor with the ratchet and the square, the center,
third and fourth wheels, the bridge, the two screws and the probe): a closing gripper takes hold of a part once both
fingers touch it inside its grasp region, with the tool frame within 15 cm of it and the penetration at most 2 mm, for
0.02 s; the part then follows the gripper rigidly. Any opening command releases it. A hold is lost if the fingers open
more than 15 %, the part drifts 1 cm / 0.15 rad from its held pose, or the hold has to push or pull its part with more
than 60 N (averaged over 20 ms); the gripper must then open before it can take hold again. Two grippers never hold the
same part, and other objects are grasped physically (friction). The grippers start empty and nearly closed: open a
gripper, then close it on a part to take hold.

## Success rule

What the grader checks, in plain words (each task's `instruction.md` states the exact rule and its limits):

- **Wheels seated and meshed.** The barrel, the center wheel, the third wheel and the fourth wheel sit on their lower
  pivots in the main plate, each no more than 1 mm above its seat. A wheel seats once its lower pivot is down in its
  hole: at most 0.8 mm above its seat, within 0.6 mm of the hole's axis and tilted less than 3 degrees. Each wheel's
  pinion meshes with the previous wheel's teeth (barrel, center, third, fourth); a wheel lowered with its pinion's teeth
  tip to tip on the other wheel's teeth stops until it is turned a little.
- **Upper pivots.** Every upper pivot (the barrel's: its arbor) is inside its hole in the bridge, at least 1 mm deep and
  within the hole's radius of its axis.
- **Bridge seated.** The bridge sits on its pillars within 0.5 mm of its seat height, 0.5 mm of its seat position and
  tilted less than 0.5 degrees.
- **Screws tight.** Both screws are in the bridge's two screw holes (either screw in either hole, one to a hole) and
  turned clockwise by 2 full turns there. No thread is simulated: the turns count only while the screw's head sits on
  the bridge over its hole (its underside between 0.3 mm below and 0.8 mm above the bridge's top, its axis within
  0.6 mm of the hole's); counterclockwise turns count back, and a screw moved to the other hole starts again. Once both
  are tight, every wheel is seated, every upper pivot is in its hole and the bridge is seated, the bridge is fixed: the
  bridge is screwed down after the wheels, not before.
- **Wound and run.** After the bridge is fixed, the barrel is wound by turning its square counterclockwise seen from
  above; the click lets it turn that way only and holds it one ratchet tooth (18 degrees) behind the furthest point
  reached. It must be wound at least 120 degrees (as held by the click), then the square let go; any further advance
  of the square counts as winding again. Half a second after
  the last winding a 5 s run is timed: the fourth wheel must turn at least 3.5 times in it, spend at most 0.2 s slower
  than a tenth of its free-run speed, and turn 450 times as much as the barrel in the opposite direction, within 2 %.
  The run counts only for the mainspring alone: from the last winding to the end of the run neither the robot nor the
  probe may touch the barrel or a wheel, and the barrel must turn the way the mainspring drives it; otherwise the run
  does not count until the barrel is wound again. Winding again restarts the run.
- Holding parts is allowed, apart from the run.
- All final conditions must hold together, continuously, for 0.5 s of simulated time.

A task piece falling more than 100 mm below the tabletop ends the episode in failure. Contact penetration does not end
the episode by itself. A success with deep interpenetration (beyond the limits stated in the task text) is held for
human review instead of being credited automatically, and a task contact deeper than 10 mm counts as a simulator
breakdown.

Every task has a wall-clock budget of 11400 s (3 h 10 min) in both modes. In privileged mode the delivered trajectory is
replayed open-loop twice in a fresh process: both replays must succeed, and the strict goal must be reached within the
trajectory's own rows. In standard mode the simulator service grades the live episode and a fresh-process replay
confirms it. The details are on the [benchmark page](../benchmarks/roboarc.md) (section "Grading replays").

## Bases

1 base (hard 1). It is one frozen scene instance and ships as two Harbor task directories, `-privileged` and
`-standard`, with the same scene, goal and grader. The difficulty is `[metadata].difficulty` in each `task.toml`.

| Base | Difficulty | Variation (task text) | Notes | Task dirs |
|---|---|---|---|---|
| `roboarc-clock-assembling-01-i01` | hard | `clock_assembling_gear_train` | barrel, center, third and fourth wheel, the train bridge, two screws; wound and run (train ratio 450) | [privileged](../../tasks/roboarc-clock-assembling-01-i01-privileged/) / [standard](../../tasks/roboarc-clock-assembling-01-i01-standard/) |

## Links

- [RoboARC benchmark page](../benchmarks/roboarc.md): the rules shared by all families (grading replays, safety readings in standard mode, the standard-mode service, the time budget).
- [Standard mode](../STANDARD_MODE.md) and [protocol](../PROTOCOL.md).
- [All families](README.md).
- [clock_assembling.md](../simulator/families/clock_assembling.md): development notes (the movement, the order check, the reference, the probe, the rules).
