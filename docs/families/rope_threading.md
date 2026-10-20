# Rope threading

Family key `rope_threading` · 8 task bases · robot `aloha-2` · physics MuJoCo 3.14.0

The scene has a rope and a set of fixed rings on stands (bases 01-03), or a wall with grommets plus further rings (bases 04-08). The robot must thread the rope through the named holes in a given order along the rope (from its material start to its material end), crossing each hole in a required direction. The rope is a chain of rigid capsule links, 8 mm thick.

## Robot and grasping

Two 6-DoF ALOHA 2 (ViperX) arms with parallel-jaw grippers on a fixed base (`aloha-2`). Actions are 14-D joint-position targets (6 arm joints and one gripper target per arm) at 25 Hz; standard mode also accepts end-effector pose targets.

Assisted grasping of rigid parts (`contact_weld_v1`): when a closing gripper has both fingers on the same graspable part, inside that part's declared grasp region, the part is attached to the gripper and follows it rigidly. Any opening command releases it. The hold is lost if the fingers open too far, the part slips away from the gripper, or the hold has to push or pull the part with more than 60 N (averaged over 20 ms); the gripper must then open before it can take hold again. Two grippers never hold the same part. Parts touched outside their grasp region, and all other objects, are held by friction only. The rope's grasp regions cover it along its length.

## Success rule

What the grader checks, in plain words (each task's `instruction.md` states the exact rule and its limits):

- **Order and direction.** The rope passes through the required holes in the stated order along the rope, each in its stated direction.
- **Clean passages.** It passes fully through each required hole exactly once, with no extra crossings or ambiguous tangencies; the full rope thickness clears the rim with an extra 0.1 mm margin, on both sides of the ring's thickness.
- **Intact and at rest.** The rope is intact and every rope point moves at most 10 mm/s. Holding is allowed; release is not required.
- All final conditions must hold together, continuously, for 0.5 s of simulated time.

Every task has a wall-clock budget of 10800 s (3 h) in both modes. In privileged mode the delivered trajectory is replayed open-loop twice in a fresh process: both replays must succeed, and the strict goal must be reached within the trajectory's own rows. In standard mode the simulator service grades the live episode and a fresh-process replay confirms it. The details are on the [benchmark page](../benchmarks/roboarc.md) (section "Grading replays").

## Bases

8 bases (very_easy 1, easier 1, easy 2, medium 2, hard 2). Each base is one frozen scene instance and ships as two Harbor task directories,
`-privileged` and `-standard`, with the same scene, goal and grader. The difficulty is `[metadata].difficulty` in
each `task.toml`.

| Base | Difficulty | Variation (task text) | Notes | Task dirs |
|---|---|---|---|---|
| `roboarc-rope-threading-01-i01` | easy | `rope_threading_easy` | two rings | [privileged](../../tasks/roboarc-rope-threading-01-i01-privileged/) / [standard](../../tasks/roboarc-rope-threading-01-i01-standard/) |
| `roboarc-rope-threading-02-i01` | medium | `rope_threading_medium` | three rings | [privileged](../../tasks/roboarc-rope-threading-02-i01-privileged/) / [standard](../../tasks/roboarc-rope-threading-02-i01-standard/) |
| `roboarc-rope-threading-03-i01` | hard | `rope_threading_hard` | three smaller rings | [privileged](../../tasks/roboarc-rope-threading-03-i01-privileged/) / [standard](../../tasks/roboarc-rope-threading-03-i01-standard/) |
| `roboarc-rope-threading-04-i01` | very_easy | `rope_threading_rd_very_easy` | wall with grommets: 4 holes | [privileged](../../tasks/roboarc-rope-threading-04-i01-privileged/) / [standard](../../tasks/roboarc-rope-threading-04-i01-standard/) |
| `roboarc-rope-threading-05-i01` | easier | `rope_threading_rd_easier` | wall with grommets: 4 holes (smaller) | [privileged](../../tasks/roboarc-rope-threading-05-i01-privileged/) / [standard](../../tasks/roboarc-rope-threading-05-i01-standard/) |
| `roboarc-rope-threading-06-i01` | easy | `rope_threading_rd_easy` | wall with grommets: 5 holes | [privileged](../../tasks/roboarc-rope-threading-06-i01-privileged/) / [standard](../../tasks/roboarc-rope-threading-06-i01-standard/) |
| `roboarc-rope-threading-07-i01` | medium | `rope_threading_rd_medium` | wall with grommets: 5 holes (smaller) | [privileged](../../tasks/roboarc-rope-threading-07-i01-privileged/) / [standard](../../tasks/roboarc-rope-threading-07-i01-standard/) |
| `roboarc-rope-threading-08-i01` | hard | `rope_threading_rd_hard` | wall with grommets: 6 holes | [privileged](../../tasks/roboarc-rope-threading-08-i01-privileged/) / [standard](../../tasks/roboarc-rope-threading-08-i01-standard/) |

## Links

- [RoboARC benchmark page](../benchmarks/roboarc.md): the rules shared by all families (grading replays, safety readings in standard mode, the standard-mode service, the time budget).
- [Standard mode](../STANDARD_MODE.md) and [protocol](../PROTOCOL.md).
- [All families](README.md).
- [rope_threading.md](../simulator/families/rope_threading.md): development notes (physics, references, stress acceptance).
