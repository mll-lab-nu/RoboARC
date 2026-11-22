# Interlocked claws

Family key `interlocked_claws` · 8 task bases · robot `aloha-2` · physics MuJoCo 3.14.0

Two rigid puzzle pieces (three in base 13) are hooked into each other, and the grippers start closed on them. Depending on the base, the robot must either free one piece and place it entirely in the green dashed goal square while the other stays on the puzzle side, or separate the pieces completely and hold them still. Besides the claw pair, the family includes a redesigned key-and-lock pair and three of Hanayama's cast puzzles (Cast ABC, Cast Rotor and Cast Trinity).

## Robot and grasping

Two 6-DoF ALOHA 2 (ViperX) arms with parallel-jaw grippers on a fixed base (`aloha-2`). Actions are 14-D joint-position targets (6 arm joints and one gripper target per arm) at 40 Hz; standard mode also accepts end-effector pose targets.

Assisted grasping of rigid parts (`contact_weld_v1`): when a closing gripper has both fingers on the same graspable part, inside that part's declared grasp region, the part is attached to the gripper and follows it rigidly. Any opening command releases it. The hold is lost if the fingers open too far, the part slips away from the gripper, or the hold has to push or pull the part with more than 60 N (averaged over 20 ms); the gripper must then open before it can take hold again. Two grippers never hold the same part. Parts touched outside their grasp region, and all other objects, are held by friction only. In base 13 (Cast Trinity) the grippers start closed on two of the three pieces (`trinity_t`, `trinity_b`) while the third (`trinity_h`) hangs on both; each piece is a graspable part as a whole, and a hand takes hold with its tool frame within 12 cm of it.

## Success rule

What the grader checks, in plain words (each task's `instruction.md` states the exact rule and its limits):

Placement bases (02, 08-10):

- **Containment.** Every point of the target piece (base 02: the cross piece; bases 08-10: the key piece), not only its centre, lies inside the goal square, at least 1 mm inside the inner edge of the dashed green line, and within the goal volume's height above the table; it may not sink more than 1 mm below the tabletop. It need not touch the table.
- **Separation from the rest.** The other piece, and every other puzzle component, stay entirely on the puzzle side of an invisible fixed vertical plane between the puzzle and the goal square. Moving the whole puzzle into the goal does not count.
- **At rest.** The placed piece moves at most 1 mm/s and 0.02 rad/s. Holding it is allowed.

Separation bases (05, 11-13):

- **Separated.** The two complete pieces (in base 13: every two of the three pieces) are at least 10 mm apart along at least one of ten directions: the three world axes, each of the two pieces' three body axes, and the line between their centres. A gap inside the interlock does not count. The goal square plays no role.
- **At rest.** All pieces move at most 1 mm/s and 0.02 rad/s. Holding them is allowed.

In both kinds, all final conditions must hold together, continuously, for 0.5 s of simulated time.

A piece falling more than 100 mm below the tabletop ends the episode in failure. Contact penetration does not end the episode by itself. A success with deep interpenetration (beyond the limits stated in the task text) is held for human review instead of being credited automatically, and a task contact deeper than 10 mm counts as a simulator breakdown.

Every task has a wall-clock budget of 11400 s (3 h 10 min) in both modes. In privileged mode the delivered trajectory is replayed open-loop twice in a fresh process: both replays must succeed, and the strict goal must be reached within the trajectory's own rows. In standard mode the simulator service grades the live episode and a fresh-process replay confirms it. The details are on the [benchmark page](../benchmarks/roboarc.md) (section "Grading replays").

## Bases

8 bases (easier 1, easy 1, medium 2, hard 4). Each base is one frozen scene instance and ships as two Harbor task directories,
`-privileged` and `-standard`, with the same scene, goal and grader. The difficulty is `[metadata].difficulty` in
each `task.toml`. Numbers missing from the table are not used: base numbers are never reused, so a removed base leaves a gap.

| Base | Difficulty | Variation (task text) | Notes | Task dirs |
|---|---|---|---|---|
| `roboarc-interlocked-claws-02-i01` | easier | `easier` | place the cross piece | [privileged](../../tasks/roboarc-interlocked-claws-02-i01-privileged/) / [standard](../../tasks/roboarc-interlocked-claws-02-i01-standard/) |
| `roboarc-interlocked-claws-05-i01` | hard | `interlocked_claws_hard` | separate both pieces | [privileged](../../tasks/roboarc-interlocked-claws-05-i01-privileged/) / [standard](../../tasks/roboarc-interlocked-claws-05-i01-standard/) |
| `roboarc-interlocked-claws-08-i01` | easy | `interlocked_claws_rd_easy` | redesigned key and lock; place the key | [privileged](../../tasks/roboarc-interlocked-claws-08-i01-privileged/) / [standard](../../tasks/roboarc-interlocked-claws-08-i01-standard/) |
| `roboarc-interlocked-claws-09-i01` | medium | `interlocked_claws_rd_medium` | redesigned key and lock; place the key | [privileged](../../tasks/roboarc-interlocked-claws-09-i01-privileged/) / [standard](../../tasks/roboarc-interlocked-claws-09-i01-standard/) |
| `roboarc-interlocked-claws-10-i01` | hard | `interlocked_claws_rd_hard` | redesigned key and lock; place the key | [privileged](../../tasks/roboarc-interlocked-claws-10-i01-privileged/) / [standard](../../tasks/roboarc-interlocked-claws-10-i01-standard/) |
| `roboarc-interlocked-claws-11-i01` | medium | `interlocked_claws_cast_abc` | Cast ABC: separate the C ring from the AB plate | [privileged](../../tasks/roboarc-interlocked-claws-11-i01-privileged/) / [standard](../../tasks/roboarc-interlocked-claws-11-i01-standard/) |
| `roboarc-interlocked-claws-12-i01` | hard | `interlocked_claws_rotor` | Cast Rotor: separate two identical interlocked rotor pieces | [privileged](../../tasks/roboarc-interlocked-claws-12-i01-privileged/) / [standard](../../tasks/roboarc-interlocked-claws-12-i01-standard/) |
| `roboarc-interlocked-claws-13-i01` | hard | `interlocked_claws_cast_trinity` | Cast Trinity: separate three interlocked pieces from one another | [privileged](../../tasks/roboarc-interlocked-claws-13-i01-privileged/) / [standard](../../tasks/roboarc-interlocked-claws-13-i01-standard/) |

## Links

- [RoboARC benchmark page](../benchmarks/roboarc.md): the rules shared by all families (grading replays, safety readings in standard mode, the standard-mode service, the time budget).
- [Standard mode](../STANDARD_MODE.md) and [protocol](../PROTOCOL.md).
- [All families](README.md).
- [interlocked_claws.md](../simulator/families/interlocked_claws.md): development notes (physics, references, stress acceptance).
- [interlocked_claws_cast_abc.md](../simulator/families/interlocked_claws_cast_abc.md): development notes of the Cast ABC base (11).
- [interlocked_claws_rotor.md](../simulator/families/interlocked_claws_rotor.md): development notes of the Cast Rotor base (12).
- [interlocked_claws_cast_trinity.md](../simulator/families/interlocked_claws_cast_trinity.md): development notes of the Cast Trinity base (13).
