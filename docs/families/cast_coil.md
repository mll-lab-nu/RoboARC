# Cast coil

Family key `cast_coil` · 5 task bases · robot `aloha-2` · physics MuJoCo 3.14.0

Two rigid puzzle pieces (`coil_gold` and `coil_silver`: gold and silver at appearance seed 0, other colours and materials at seeds 1 and 2) are on the table. The robot must assemble them into their matching interlocking configuration, then release both. The bases use different piece designs (see the table).

## Robot and grasping

Two 6-DoF ALOHA 2 (ViperX) arms with parallel-jaw grippers on a fixed base (`aloha-2`). Actions are 14-D joint-position targets (6 arm joints and one gripper target per arm) at 40 Hz; standard mode also accepts end-effector pose targets.

Assisted grasping of rigid parts (`contact_weld_v1`): when a closing gripper has both fingers on the same graspable part, inside that part's declared grasp region, the part is attached to the gripper and follows it rigidly. Any opening command releases it. The hold is lost if the fingers open too far, the part slips away from the gripper, or the hold has to push or pull the part with more than 60 N (averaged over 20 ms); the gripper must then open before it can take hold again. Two grippers never hold the same part. Parts touched outside their grasp region, and all other objects, are held by friction only.

## Success rule

What the grader checks, in plain words (each task's `instruction.md` states the exact rule and its limits):

- **Assembled.** The pieces' matching assembly frames agree to within 1 mm and 0.02 rad.
- **On the table.** The bottom of at least one piece is within 3 mm of the tabletop, neither piece extends more than 1 mm below it, and both overlap the tabletop horizontally.
- **Released and at rest.** Both pieces are released, the robot has withdrawn to at least 5 mm of clearance, and both pieces move at most 1 mm/s and 0.02 rad/s (at most 1 mm and 0.02 rad of drift).
- All final conditions must hold together, continuously, for 0.5 s of simulated time.

A piece falling more than 100 mm below the tabletop ends the episode in failure. Contact penetration does not end the episode by itself. A success with deep interpenetration (beyond the limits stated in the task text) is held for human review instead of being credited automatically, and a task contact deeper than 10 mm counts as a simulator breakdown.

Every task has a wall-clock budget of 10800 s (3 h) in both modes. In privileged mode the delivered trajectory is replayed open-loop twice in a fresh process: both replays must succeed, and the strict goal must be reached within the trajectory's own rows. In standard mode the simulator service grades the live episode and a fresh-process replay confirms it. The details are on the [benchmark page](../benchmarks/roboarc.md) (section "Grading replays").

## Bases

5 bases (very_easy 1, easier 1, easy 1, medium 1, hard 1). Each base is one frozen scene instance and ships as two Harbor task directories,
`-privileged` and `-standard`, with the same scene, goal and grader. The difficulty is `[metadata].difficulty` in
each `task.toml`. Numbers missing from the table are not used: base numbers are never reused, so a removed base leaves a gap.

| Base | Difficulty | Variation (task text) | Notes | Task dirs |
|---|---|---|---|---|
| `roboarc-cast-coil-04-i01` | medium | `cast_coil_medium` | Cast Coil pieces | [privileged](../../tasks/roboarc-cast-coil-04-i01-privileged/) / [standard](../../tasks/roboarc-cast-coil-04-i01-standard/) |
| `roboarc-cast-coil-05-i01` | hard | `cast_coil_hard` | Cast Coil pieces | [privileged](../../tasks/roboarc-cast-coil-05-i01-privileged/) / [standard](../../tasks/roboarc-cast-coil-05-i01-standard/) |
| `roboarc-cast-coil-06-i01` | very_easy | `cast_coil_v2_very_easy` | piece design modelled on the official Cast Coil puzzle | [privileged](../../tasks/roboarc-cast-coil-06-i01-privileged/) / [standard](../../tasks/roboarc-cast-coil-06-i01-standard/) |
| `roboarc-cast-coil-07-i01` | easier | `cast_coil_v2_easier` | two interlocking lattice pieces | [privileged](../../tasks/roboarc-cast-coil-07-i01-privileged/) / [standard](../../tasks/roboarc-cast-coil-07-i01-standard/) |
| `roboarc-cast-coil-08-i01` | easy | `cast_coil_v2_easy` | two interlocking lattice pieces (a second lattice design) | [privileged](../../tasks/roboarc-cast-coil-08-i01-privileged/) / [standard](../../tasks/roboarc-cast-coil-08-i01-standard/) |

## Links

- [RoboARC benchmark page](../benchmarks/roboarc.md): the rules shared by all families (grading replays, safety readings in standard mode, the standard-mode service, the time budget).
- [Standard mode](../STANDARD_MODE.md) and [protocol](../PROTOCOL.md).
- [All families](README.md).
- [cast_coil.md](../simulator/families/cast_coil.md): development notes (physics, references, stress acceptance).
