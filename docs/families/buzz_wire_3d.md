# 3D buzz wire

Family key `buzz_wire_3d` · 10 task bases · robot `franka-robotiq-2f85` · physics MuJoCo 3.14.0

A fixed metal wire is bent in three dimensions above a base. The tool is a wand: a metal loop on a wooden handle. The robot must guide the loop along the entire wire, with the wire inside the loop, from the start end to the finish end without the metal ever touching the wire, then set the wand down on the finish support and let go.

## Robot and grasping

One 7-DoF Franka Panda arm with a Robotiq 2F-85 parallel gripper on a fixed base (`franka-robotiq-2f85`). Actions are 8-D joint-position targets (7 arm joints and the gripper) at 40 Hz; standard mode also accepts end-effector pose targets.

Assisted grasping of rigid parts (`contact_weld_v1`): when a closing gripper has both fingers on the same graspable part, inside that part's declared grasp region, the part is attached to the gripper and follows it rigidly. Any opening command releases it. The hold is lost if the fingers open too far, the part slips away from the gripper, or the hold has to push or pull the part with more than 65 N (averaged over 20 ms), or no finger has touched the part for 0.2 s; the gripper must then open before it can take hold again. Two grippers never hold the same part. Parts touched outside their grasp region, and all other objects, are held by friction only.

## Success rule

What the grader checks, in plain words (each task's `instruction.md` states the exact rule and its limits):

- **Traversal.** The loop starts within 10 mm (along the wire) of the start end and travels continuously to within 10 mm of the finish end, with exactly one wire crossing inside the loop. Skipping parts of the route or taking the loop off the wire before the end does not count.
- **Finish.** After the traversal, the wand rests on the finish support in the finish resting pose (within 5 mm and 10 degrees; a half turn about the handle axis is equivalent) and actually touches the support.
- **Released.** No robot contact remains: the grasp is released and the robot has withdrawn to at least 5 mm of clearance.
- **At rest.** The wand moves at most 1 mm/s and 0.02 rad/s, with at most 1 mm and 0.02 rad of drift.
- All final conditions must hold together, continuously, for 1 s of simulated time.

Any contact between a metal part of the wand and the metal wire ends the episode in failure, including brief or zero-force touches (contact of the wooden handle does not count). The wand falling more than 100 mm below the tabletop also ends the episode in failure.

Every task has a wall-clock budget of 10800 s (3 h) in both modes. In privileged mode the delivered trajectory is replayed open-loop twice in a fresh process: both replays must succeed, and the strict goal must be reached within the trajectory's own rows. In standard mode the simulator service grades the live episode and a fresh-process replay confirms it. The details are on the [benchmark page](../benchmarks/roboarc.md) (section "Grading replays").

## Bases

10 bases (very_easy 2, easier 2, easy 2, medium 2, hard 2). Each base is one frozen scene instance and ships as two Harbor task directories,
`-privileged` and `-standard`, with the same scene, goal and grader. The difficulty is `[metadata].difficulty` in
each `task.toml`.

| Base | Difficulty | Variation (task text) | Notes | Task dirs |
|---|---|---|---|---|
| `roboarc-buzz-wire-3d-01-i01` | very_easy | `very_easy` |  | [privileged](../../tasks/roboarc-buzz-wire-3d-01-i01-privileged/) / [standard](../../tasks/roboarc-buzz-wire-3d-01-i01-standard/) |
| `roboarc-buzz-wire-3d-02-i01` | easier | `easier` |  | [privileged](../../tasks/roboarc-buzz-wire-3d-02-i01-privileged/) / [standard](../../tasks/roboarc-buzz-wire-3d-02-i01-standard/) |
| `roboarc-buzz-wire-3d-03-i01` | easy | `buzz_wire_3d_easy` |  | [privileged](../../tasks/roboarc-buzz-wire-3d-03-i01-privileged/) / [standard](../../tasks/roboarc-buzz-wire-3d-03-i01-standard/) |
| `roboarc-buzz-wire-3d-04-i01` | medium | `buzz_wire_3d_medium` |  | [privileged](../../tasks/roboarc-buzz-wire-3d-04-i01-privileged/) / [standard](../../tasks/roboarc-buzz-wire-3d-04-i01-standard/) |
| `roboarc-buzz-wire-3d-05-i01` | hard | `buzz_wire_3d_hard` |  | [privileged](../../tasks/roboarc-buzz-wire-3d-05-i01-privileged/) / [standard](../../tasks/roboarc-buzz-wire-3d-05-i01-standard/) |
| `roboarc-buzz-wire-3d-06-i01` | very_easy | `buzz_wire_3d_rd_very_easy` | redesigned wire: insulated black rubber sleeves the loop may rest on; insulated final bend and finish post | [privileged](../../tasks/roboarc-buzz-wire-3d-06-i01-privileged/) / [standard](../../tasks/roboarc-buzz-wire-3d-06-i01-standard/) |
| `roboarc-buzz-wire-3d-07-i01` | easier | `buzz_wire_3d_rd_easier` | redesigned wire: insulated black rubber sleeves the loop may rest on; insulated final bend and finish post | [privileged](../../tasks/roboarc-buzz-wire-3d-07-i01-privileged/) / [standard](../../tasks/roboarc-buzz-wire-3d-07-i01-standard/) |
| `roboarc-buzz-wire-3d-08-i01` | easy | `buzz_wire_3d_rd_easy` | redesigned wire: insulated black rubber sleeves the loop may rest on; insulated final bend and finish post | [privileged](../../tasks/roboarc-buzz-wire-3d-08-i01-privileged/) / [standard](../../tasks/roboarc-buzz-wire-3d-08-i01-standard/) |
| `roboarc-buzz-wire-3d-09-i01` | medium | `buzz_wire_3d_rd_medium` | redesigned wire: insulated black rubber sleeves the loop may rest on; insulated final bend and finish post | [privileged](../../tasks/roboarc-buzz-wire-3d-09-i01-privileged/) / [standard](../../tasks/roboarc-buzz-wire-3d-09-i01-standard/) |
| `roboarc-buzz-wire-3d-10-i01` | hard | `buzz_wire_3d_rd_hard` | redesigned wire: insulated black rubber sleeves the loop may rest on; insulated final bend and finish post | [privileged](../../tasks/roboarc-buzz-wire-3d-10-i01-privileged/) / [standard](../../tasks/roboarc-buzz-wire-3d-10-i01-standard/) |

## Links

- [RoboARC benchmark page](../benchmarks/roboarc.md): the rules shared by all families (grading replays, safety readings in standard mode, the standard-mode service, the time budget).
- [Standard mode](../STANDARD_MODE.md) and [protocol](../PROTOCOL.md).
- [All families](README.md).
- [buzz_wire_3d.md](../simulator/families/buzz_wire_3d.md): development notes (physics, references, stress acceptance).
