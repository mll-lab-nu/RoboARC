# Blanket flatten

Family key `blanket_flatten` · 5 task bases · robot `aloha-2` · physics MuJoCo 3.14.0

A cloth blanket lies on the table, folded over on itself. The robot must unfold it and spread it flat on the table without overlaps or excessive stretching, then let go and move the grippers away from it. The start state of the cloth differs from base to base.

## Robot and grasping

Two 6-DoF ALOHA 2 (ViperX) arms with parallel-jaw grippers on a fixed base (`aloha-2`). Actions are 14-D joint-position targets (6 arm joints and one gripper target per arm) at 40 Hz; standard mode also accepts end-effector pose targets.

Assisted grasping of soft material (`pbd_local_patch_v1`): a pinch with both fingers on an exposed patch of the blanket attaches that local patch to the gripper, while the rest of the cloth stays deformable. Opening the gripper releases it; the hold is also lost on excessive opening, excessive material strain or a follow error above 10 mm. The two hands may hold separate, non-overlapping patches of the blanket. In bases 03, 04 and 06 a safety floor keeps the grippers from pressing the cloth into the table: apart from the finger pads, no gripper part is driven below the top of the cloth under it.

## Success rule

What the grader checks, in plain words (each task's `instruction.md` states the exact rule and its limits):

- **Coverage.** The blanket covers at least 95 % of its unfolded area in the top-down view (overlapping layers count once, holes do not count).
- **Flat on the table.** Every cloth point is at most 30 mm above the tabletop and no more than 8 mm below it. In base 06, cloth resting on the fence does not count as lying flat.
- **No stretching.** No material edge is stretched beyond 1.15 times its original length.
- **Released.** All grasps are released and the robot's moving parts (arm links, wrists, grippers, fingers) have withdrawn to at least 5 mm of clearance from the cloth. The two arm bases are fixed to the world and are not counted (since 2.1.0): the cloth may lie against them.
- **At rest.** Every cloth point moves at most 30 mm/s and drifts at most 3 mm during the confirmation.
- All final conditions must hold together, continuously, for 0.5 s of simulated time.

The episode fails if any cloth point falls more than 100 mm below the tabletop. The cloth is simulated on the GPU with a deterministic solver, so a replay on the same host reproduces the agent's run.

Every task has a wall-clock budget of 11400 s (3 h 10 min) in both modes. In privileged mode the delivered trajectory is replayed open-loop twice in a fresh process: both replays must succeed, and the strict goal must be reached within the trajectory's own rows. In standard mode the simulator service grades the live episode and a fresh-process replay confirms it. The details are on the [benchmark page](../benchmarks/roboarc.md) (section "Grading replays").

## Bases

5 bases (easier 1, easy 2, medium 1, hard 1). Each base is one frozen scene instance and ships as two Harbor task directories,
`-privileged` and `-standard`, with the same scene, goal and grader. The difficulty is `[metadata].difficulty` in
each `task.toml`. Numbers missing from the table are not used: base numbers are never reused, so a removed base leaves a gap.

| Base | Difficulty | Variation (task text) | Notes | Task dirs |
|---|---|---|---|---|
| `roboarc-blanket-flatten-03-i01` | easier | `easier` | redesigned start state | [privileged](../../tasks/roboarc-blanket-flatten-03-i01-privileged/) / [standard](../../tasks/roboarc-blanket-flatten-03-i01-standard/) |
| `roboarc-blanket-flatten-04-i01` | easy | `easy` | redesigned start state | [privileged](../../tasks/roboarc-blanket-flatten-04-i01-privileged/) / [standard](../../tasks/roboarc-blanket-flatten-04-i01-standard/) |
| `roboarc-blanket-flatten-06-i01` | hard | `hard` | redesigned start state; a fixed fence stands beyond the blanket's far side | [privileged](../../tasks/roboarc-blanket-flatten-06-i01-privileged/) / [standard](../../tasks/roboarc-blanket-flatten-06-i01-standard/) |
| `roboarc-blanket-flatten-07-i01` | easy | `blanket_flatten_easy` |  | [privileged](../../tasks/roboarc-blanket-flatten-07-i01-privileged/) / [standard](../../tasks/roboarc-blanket-flatten-07-i01-standard/) |
| `roboarc-blanket-flatten-08-i01` | medium | `blanket_flatten_medium` |  | [privileged](../../tasks/roboarc-blanket-flatten-08-i01-privileged/) / [standard](../../tasks/roboarc-blanket-flatten-08-i01-standard/) |

## Links

- [RoboARC benchmark page](../benchmarks/roboarc.md): the rules shared by all families (grading replays, safety readings in standard mode, the standard-mode service, the time budget).
- [Standard mode](../STANDARD_MODE.md) and [protocol](../PROTOCOL.md).
- [All families](README.md).
- [blanket_flatten.md](../simulator/families/blanket_flatten.md): development notes (physics, references, stress acceptance).
- [blanket_cloth_interface.md](../simulator/families/blanket_cloth_interface.md): development notes on the cloth interface of the blanket tasks.
