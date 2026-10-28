# Cast keyhole

Family key `cast_keyhole` · 5 task bases · robot `aloha-2` · physics MuJoCo 3.14.0

Two cast plates are interlocked: the keyhole plate (`keyhole_silver`, a closed, stepped hole through it) and the fork plate (`keyhole_gold`, its stepped slot open at one end; in base 06 the right-hand and the left-hand plate). The robot must free the keyhole plate completely and place all of it inside the green dashed goal square on the table, while the fork plate stays on the puzzle side. The object names are identifiers: the plates are gold and silver at appearance seed 0 and change colour and material at seeds 1 and 2.

## Robot and grasping

Two 6-DoF ALOHA 2 (ViperX) arms with parallel-jaw grippers on a fixed base (`aloha-2`). Actions are 14-D joint-position targets (6 arm joints and one gripper target per arm) at 40 Hz; standard mode also accepts end-effector pose targets.

Assisted grasping of rigid parts (`contact_weld_v1`): when a closing gripper has both fingers on the same graspable part, inside that part's declared grasp region, the part is attached to the gripper and follows it rigidly. Any opening command releases it. The hold is lost if the fingers open too far, the part slips away from the gripper, or the hold has to push or pull the part with more than 60 N (averaged over 20 ms); the gripper must then open before it can take hold again. Two grippers never hold the same part. Parts touched outside their grasp region, and all other objects, are held by friction only. In base 06 the grippers start closed on the two pieces.

## Success rule

What the grader checks, in plain words (each task's `instruction.md` states the exact rule and its limits):

- **Containment.** Every point of the keyhole plate, not only its centre, lies inside the goal square, at least 1 mm inside the inner edge of the dashed green line, and within the goal volume's height above the table; it may not sink more than 1 mm below the tabletop. It need not touch the table.
- **Separation from the rest.** The fork plate, and every other puzzle component, stay entirely on the puzzle side of an invisible fixed vertical plane between the puzzle and the goal square. Moving the whole puzzle into the goal does not count.
- **At rest.** The keyhole plate moves at most 1 mm/s and 0.02 rad/s. Holding it is allowed; release is not required.
- All final conditions must hold together, continuously, for 0.5 s of simulated time.

A piece falling more than 100 mm below the tabletop ends the episode in failure. Contact penetration does not end the episode by itself. A success with deep interpenetration (beyond the limits stated in the task text) is held for human review instead of being credited automatically, and a task contact deeper than 10 mm counts as a simulator breakdown.

Every task has a wall-clock budget of 10800 s (3 h) in both modes. In privileged mode the delivered trajectory is replayed open-loop twice in a fresh process: both replays must succeed, and the strict goal must be reached within the trajectory's own rows. In standard mode the simulator service grades the live episode and a fresh-process replay confirms it. The details are on the [benchmark page](../benchmarks/roboarc.md) (section "Grading replays").

## Bases

5 bases (easier 1, easy 1, medium 1, hard 2). Each base is one frozen scene instance and ships as two Harbor task directories,
`-privileged` and `-standard`, with the same scene, goal and grader. The difficulty is `[metadata].difficulty` in
each `task.toml`. Numbers missing from the table are not used: base numbers are never reused, so a removed base leaves a gap.

| Base | Difficulty | Variation (task text) | Notes | Task dirs |
|---|---|---|---|---|
| `roboarc-cast-keyhole-02-i01` | easier | `easier` |  | [privileged](../../tasks/roboarc-cast-keyhole-02-i01-privileged/) / [standard](../../tasks/roboarc-cast-keyhole-02-i01-standard/) |
| `roboarc-cast-keyhole-03-i01` | easy | `cast_keyhole_easy` |  | [privileged](../../tasks/roboarc-cast-keyhole-03-i01-privileged/) / [standard](../../tasks/roboarc-cast-keyhole-03-i01-standard/) |
| `roboarc-cast-keyhole-04-i01` | medium | `cast_keyhole_medium` |  | [privileged](../../tasks/roboarc-cast-keyhole-04-i01-privileged/) / [standard](../../tasks/roboarc-cast-keyhole-04-i01-standard/) |
| `roboarc-cast-keyhole-05-i01` | hard | `cast_keyhole_hard` |  | [privileged](../../tasks/roboarc-cast-keyhole-05-i01-privileged/) / [standard](../../tasks/roboarc-cast-keyhole-05-i01-standard/) |
| `roboarc-cast-keyhole-06-i01` | hard | `cast_keyhole_double_maze` | double maze: two plates, each with a slot grid, threaded through each other at right angles | [privileged](../../tasks/roboarc-cast-keyhole-06-i01-privileged/) / [standard](../../tasks/roboarc-cast-keyhole-06-i01-standard/) |

## Links

- [RoboARC benchmark page](../benchmarks/roboarc.md): the rules shared by all families (grading replays, safety readings in standard mode, the standard-mode service, the time budget).
- [Standard mode](../STANDARD_MODE.md) and [protocol](../PROTOCOL.md).
- [All families](README.md).
- [cast_keyhole.md](../simulator/families/cast_keyhole.md): development notes (physics, references, stress acceptance).
- [cast_keyhole_double_maze.md](../simulator/families/cast_keyhole_double_maze.md): development notes of the double-maze base (06).
