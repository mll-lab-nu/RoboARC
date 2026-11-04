# Orbits

Family key `orbits` · 2 task bases · robot `aloha-2` · physics MuJoCo 3.14.0

A closed cord loop on a wooden handle is entangled with a fixed assembly: a post with a stopper ball (base 01), or a ring between two posts with stopper balls (base 02). The robot must free the cord and handle and place them together, entirely, inside the green dashed goal square on the table, while the fixed assembly stays on the puzzle side. The cord is a chain of rigid capsule links whose two halves are tied together at the middle.

## Robot and grasping

Two 6-DoF ALOHA 2 (ViperX) arms with parallel-jaw grippers on a fixed base (`aloha-2`). Actions are 14-D joint-position targets (6 arm joints and one gripper target per arm) at 40 Hz; standard mode also accepts end-effector pose targets.

Assisted grasping of rigid parts (`contact_weld_v1`): when a closing gripper has both fingers on the same graspable part, inside that part's declared grasp region, the part is attached to the gripper and follows it rigidly. Any opening command releases it. The hold is lost if the fingers open too far, the part slips away from the gripper, or the hold has to push or pull the part with more than 60 N (averaged over 20 ms); the gripper must then open before it can take hold again. Two grippers never hold the same part. Parts touched outside their grasp region, and all other objects, are held by friction only. The cord's grasp regions cover it along its length.

## Success rule

What the grader checks, in plain words (each task's `instruction.md` states the exact rule and its limits):

- **Containment.** Every point of the cord and handle, not only its centre, lies inside the goal square, at least 1 mm inside the inner edge of the dashed green line, and within the goal volume's height above the table; it may not sink more than 1 mm below the tabletop. It need not touch the table.
- **Separation from the rest.** The fixed assembly (posts, stopper balls, ring, base), and every other puzzle component, stay entirely on the puzzle side of an invisible fixed vertical plane between the puzzle and the goal square. Moving the whole puzzle into the goal does not count.
- **Loop closed.** The cord's two halves stay joined at the middle within 3 mm.
- **At rest.** The handle moves at most 10 mm/s and 0.02 rad/s, and every cord link at most 10 mm/s. Holding is allowed; release is not required.
- All final conditions must hold together, continuously, for 0.5 s of simulated time.

Breaking the cord loop at its middle tie ends the episode in failure.

Every task has a wall-clock budget of 11400 s (3 h 10 min) in both modes. In privileged mode the delivered trajectory is replayed open-loop twice in a fresh process: both replays must succeed, and the strict goal must be reached within the trajectory's own rows. In standard mode the simulator service grades the live episode and a fresh-process replay confirms it. The details are on the [benchmark page](../benchmarks/roboarc.md) (section "Grading replays").

## Bases

2 bases (very_easy 1, easier 1). Each base is one frozen scene instance and ships as two Harbor task directories,
`-privileged` and `-standard`, with the same scene, goal and grader. The difficulty is `[metadata].difficulty` in
each `task.toml`.

| Base | Difficulty | Variation (task text) | Notes | Task dirs |
|---|---|---|---|---|
| `roboarc-orbits-01-i01` | very_easy | `orbits_very_easy` | one post with a stopper ball | [privileged](../../tasks/roboarc-orbits-01-i01-privileged/) / [standard](../../tasks/roboarc-orbits-01-i01-standard/) |
| `roboarc-orbits-02-i01` | easier | `orbits_easier` | a ring between two posts with stopper balls | [privileged](../../tasks/roboarc-orbits-02-i01-privileged/) / [standard](../../tasks/roboarc-orbits-02-i01-standard/) |

## Links

- [RoboARC benchmark page](../benchmarks/roboarc.md): the rules shared by all families (grading replays, safety readings in standard mode, the standard-mode service, the time budget).
- [Standard mode](../STANDARD_MODE.md) and [protocol](../PROTOCOL.md).
- [All families](README.md).
- [orbits.md](../simulator/families/orbits.md): development notes (physics, references, stress acceptance).
