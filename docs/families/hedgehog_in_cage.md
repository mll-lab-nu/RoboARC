# Hedgehog in cage

Family key `hedgehog_in_cage` · 5 task bases · robot `franka-robotiq-2f85` · physics MuJoCo 3.14.0

A spiked ball (the hedgehog) is trapped in a fixed cage on a mount. The robot must get the whole ball, every spike included, out of the cage and into the green dashed goal square on the table; the cage stays where it is. Bases 06 and 07 use other caged objects (a six-spiked star, a wooden owl) and add a thin rod with a box handle on the table that may be used as a tool.

## Robot and grasping

One 7-DoF Franka Panda arm with a Robotiq 2F-85 parallel gripper on a fixed base (`franka-robotiq-2f85`). Actions are 8-D joint-position targets (7 arm joints and the gripper) at 40 Hz; standard mode also accepts end-effector pose targets.

Assisted grasping of rigid parts (`contact_weld_v1`): when a closing gripper has both fingers on the same graspable part, inside that part's declared grasp region, the part is attached to the gripper and follows it rigidly. Any opening command releases it. The hold is lost if the fingers open too far, the part slips away from the gripper, or the hold has to push or pull the part with more than 65 N (averaged over 20 ms), or no finger has touched the part for 0.2 s; the gripper must then open before it can take hold again. Two grippers never hold the same part. Parts touched outside their grasp region, and all other objects, are held by friction only.

## Success rule

What the grader checks, in plain words (each task's `instruction.md` states the exact rule and its limits):

- **Containment.** Every point of the caged object (the ball, star or owl), not only its centre, lies inside the goal square, at least 1 mm inside the inner edge of the dashed green line, and within the goal volume's height above the table; it may not sink more than 1 mm below the tabletop. It need not touch the table.
- **Separation from the rest.** The cage, its mount, and every other puzzle component, stay entirely on the puzzle side of an invisible fixed vertical plane between the puzzle and the goal square. Moving the whole puzzle into the goal does not count.
- **At rest.** The object moves at most 1 mm/s and 0.02 rad/s. Holding it is allowed; release is not required.
- All final conditions must hold together, continuously, for 2 s of simulated time.

A task piece falling more than 100 mm below the tabletop ends the episode in failure; in bases 06 and 07 this includes the rod. Contact penetration does not end the episode by itself. A success with deep interpenetration (beyond the limits stated in the task text) is held for human review instead of being credited automatically, and a task contact deeper than 10 mm counts as a simulator breakdown.

Every task has a wall-clock budget of 11400 s (3 h 10 min) in both modes. In privileged mode the delivered trajectory is replayed open-loop twice in a fresh process: both replays must succeed, and the strict goal must be reached within the trajectory's own rows. In standard mode the simulator service grades the live episode and a fresh-process replay confirms it. The details are on the [benchmark page](../benchmarks/roboarc.md) (section "Grading replays").

## Bases

5 bases (easier 2, easy 2, hard 1). Each base is one frozen scene instance and ships as two Harbor task directories,
`-privileged` and `-standard`, with the same scene, goal and grader. The difficulty is `[metadata].difficulty` in
each `task.toml`. Numbers missing from the table are not used: base numbers are never reused, so a removed base leaves a gap.

| Base | Difficulty | Variation (task text) | Notes | Task dirs |
|---|---|---|---|---|
| `roboarc-hedgehog-in-cage-02-i01` | easier | `easier` |  | [privileged](../../tasks/roboarc-hedgehog-in-cage-02-i01-privileged/) / [standard](../../tasks/roboarc-hedgehog-in-cage-02-i01-standard/) |
| `roboarc-hedgehog-in-cage-03-i01` | easy | `hedgehog_in_cage_easy` |  | [privileged](../../tasks/roboarc-hedgehog-in-cage-03-i01-privileged/) / [standard](../../tasks/roboarc-hedgehog-in-cage-03-i01-standard/) |
| `roboarc-hedgehog-in-cage-06-i01` | hard | `hedgehog_in_cage_cast_cage` | Cast Cage: a six-spiked star in a closed barrel with four side windows (Hanayama's Cast Cage at four times its real size); a rod tool on the table | [privileged](../../tasks/roboarc-hedgehog-in-cage-06-i01-privileged/) / [standard](../../tasks/roboarc-hedgehog-in-cage-06-i01-standard/) |
| `roboarc-hedgehog-in-cage-07-i01` | easy | `hedgehog_in_cage_owl` | Bird in Cage: a wooden owl in a cage; a rod tool on the table | [privileged](../../tasks/roboarc-hedgehog-in-cage-07-i01-privileged/) / [standard](../../tasks/roboarc-hedgehog-in-cage-07-i01-standard/) |
| `roboarc-hedgehog-in-cage-08-i01` | easier | `hedgehog_in_cage_window_change` | the very_easy cage with a narrower start window | [privileged](../../tasks/roboarc-hedgehog-in-cage-08-i01-privileged/) / [standard](../../tasks/roboarc-hedgehog-in-cage-08-i01-standard/) |

## Links

- [RoboARC benchmark page](../benchmarks/roboarc.md): the rules shared by all families (grading replays, safety readings in standard mode, the standard-mode service, the time budget).
- [Standard mode](../STANDARD_MODE.md) and [protocol](../PROTOCOL.md).
- [All families](README.md).
- [hedgehog_in_cage.md](../simulator/families/hedgehog_in_cage.md): development notes (physics, references, stress acceptance).
