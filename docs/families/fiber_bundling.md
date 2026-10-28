# Fiber bundling

Family key `fiber_bundling` · 5 task bases · robot `franka-robotiq-2f85` · physics MuJoCo 3.14.0

A cable lies in a fixed cradle: a thick sheath whose rear terminal is anchored, and at its front a bundle of fine fibers branching from the sheath. The robot must pass every fine fiber, and then the front sheath, through a fixed ring, without touching the fine fibers with the gripper. In bases 02, 04 and 05 the number of fibers grows with the difficulty (3, 8 and 16). Bases 08 and 09 use a formboard design with 16 fibers, a tighter ring and a required end position of one of the two red fibers; base 09 adds a peg that the sheath must be routed past.

## Robot and grasping

One 7-DoF Franka Panda arm with a Robotiq 2F-85 parallel gripper on a fixed base (`franka-robotiq-2f85`). Actions are 8-D joint-position targets (7 arm joints and the gripper) at 40 Hz; standard mode also accepts end-effector pose targets.

Assisted grasping of rigid parts (`contact_weld_v1`): when a closing gripper has both fingers on the same graspable part, inside that part's declared grasp region, the part is attached to the gripper and follows it rigidly. Any opening command releases it. The hold is lost if the fingers open too far, the part slips away from the gripper, or the hold has to push or pull the part with more than 65 N (averaged over 20 ms), or no finger has touched the part for 0.2 s; the gripper must then open before it can take hold again. Two grippers never hold the same part. Parts touched outside their grasp region, and all other objects, are held by friction only. Only the thick sheath may be touched and grasped.

## Success rule

What the grader checks, in plain words (each task's `instruction.md` states the exact rule and its limits):

- **Passage.** Every fiber passes through the ring tip-first, followed by the rest of the fiber up to its root; the crossing history is tracked, so placing fibers on the far side or passing only the tips does not count.
- **Through and clear.** All fiber points end beyond the exit side of the ring tube (allowing for fiber thickness), and the sheath's branching face also passes through and clears the tube. The rear terminal stays anchored.
- **No kinks.** The angle between adjacent fiber segments is at most 60 degrees.
- **Formboard (bases 08 and 09).** The root of one of the two red fibers ends at a stated clock position around the ring, within a stated tolerance; in base 09 the sheath also passes a peg on a stated side, lying beside the peg below its top.
- Holding is allowed; there is no separate release or speed condition.
- All final conditions must hold together, continuously, for 0.5 s of simulated time.

Any gripper touch of a fine fiber, at any time, ends the episode in failure, including brief or zero-force touches. A fiber falling more than 100 mm below the tabletop also ends the episode in failure.

Every task has a wall-clock budget of 10800 s (3 h) in both modes. In privileged mode the delivered trajectory is replayed open-loop twice in a fresh process: both replays must succeed, and the strict goal must be reached within the trajectory's own rows. In standard mode the simulator service grades the live episode and a fresh-process replay confirms it. The details are on the [benchmark page](../benchmarks/roboarc.md) (section "Grading replays").

## Bases

5 bases (easier 1, easy 1, medium 2, hard 1). Each base is one frozen scene instance and ships as two Harbor task directories,
`-privileged` and `-standard`, with the same scene, goal and grader. The difficulty is `[metadata].difficulty` in
each `task.toml`. Numbers missing from the table are not used: base numbers are never reused, so a removed base leaves a gap.

| Base | Difficulty | Variation (task text) | Notes | Task dirs |
|---|---|---|---|---|
| `roboarc-fiber-bundling-02-i01` | easier | `easier` | 3 fibers | [privileged](../../tasks/roboarc-fiber-bundling-02-i01-privileged/) / [standard](../../tasks/roboarc-fiber-bundling-02-i01-standard/) |
| `roboarc-fiber-bundling-04-i01` | medium | `fiber_bundling_medium` | 8 fibers | [privileged](../../tasks/roboarc-fiber-bundling-04-i01-privileged/) / [standard](../../tasks/roboarc-fiber-bundling-04-i01-standard/) |
| `roboarc-fiber-bundling-05-i01` | hard | `fiber_bundling_hard` | 16 fibers | [privileged](../../tasks/roboarc-fiber-bundling-05-i01-privileged/) / [standard](../../tasks/roboarc-fiber-bundling-05-i01-standard/) |
| `roboarc-fiber-bundling-08-i01` | easy | `fiber_bundling_rd_easy` | formboard, 16 fibers, tighter ring; red-fiber end position | [privileged](../../tasks/roboarc-fiber-bundling-08-i01-privileged/) / [standard](../../tasks/roboarc-fiber-bundling-08-i01-standard/) |
| `roboarc-fiber-bundling-09-i01` | medium | `fiber_bundling_rd_medium` | formboard, 16 fibers, tighter ring; one peg; red-fiber end position | [privileged](../../tasks/roboarc-fiber-bundling-09-i01-privileged/) / [standard](../../tasks/roboarc-fiber-bundling-09-i01-standard/) |

## Links

- [RoboARC benchmark page](../benchmarks/roboarc.md): the rules shared by all families (grading replays, safety readings in standard mode, the standard-mode service, the time budget).
- [Standard mode](../STANDARD_MODE.md) and [protocol](../PROTOCOL.md).
- [All families](README.md).
- [fiber_bundling.md](../simulator/families/fiber_bundling.md): development notes (physics, references, stress acceptance).
