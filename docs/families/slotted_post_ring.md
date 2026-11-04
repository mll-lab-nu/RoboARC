# Slotted post ring

Family key `slotted_post_ring` · 5 task bases · robot `aloha-2` · physics MuJoCo 3.14.0

A removable metal ring is caught on a cord that runs through a slotted post; the cord carries beads and ends in discs. The robot must free the ring and place all of it inside the green dashed goal square on the table, while the post, cord, beads and end discs stay on the puzzle side. The cord is a chain of rigid capsule links.

## Robot and grasping

Two 6-DoF ALOHA 2 (ViperX) arms with parallel-jaw grippers on a fixed base (`aloha-2`). Actions are 14-D joint-position targets (6 arm joints and one gripper target per arm) at 40 Hz; standard mode also accepts end-effector pose targets.

Assisted grasping of rigid parts (`contact_weld_v1`): when a closing gripper has both fingers on the same graspable part, inside that part's declared grasp region, the part is attached to the gripper and follows it rigidly. Any opening command releases it. The hold is lost if the fingers open too far, the part slips away from the gripper, or the hold has to push or pull the part with more than 60 N (averaged over 20 ms); the gripper must then open before it can take hold again. Two grippers never hold the same part. Parts touched outside their grasp region, and all other objects, are held by friction only.

## Success rule

What the grader checks, in plain words (each task's `instruction.md` states the exact rule and its limits):

- **Containment.** Every point of the ring, not only its centre, lies inside the goal square, at least 1 mm inside the inner edge of the dashed green line, and within the goal volume's height above the table; it may not sink more than 1 mm below the tabletop. It need not touch the table.
- **Separation from the rest.** The post, cord, beads and end discs, and every other puzzle component, stay entirely on the puzzle side of an invisible fixed vertical plane between the puzzle and the goal square. Moving the whole puzzle into the goal does not count.
- **At rest.** The ring moves at most 1 mm/s and 0.02 rad/s. Holding it is allowed; release is not required.
- All final conditions must hold together, continuously, for 0.5 s of simulated time.

Every task has a wall-clock budget of 11400 s (3 h 10 min) in both modes. In privileged mode the delivered trajectory is replayed open-loop twice in a fresh process: both replays must succeed, and the strict goal must be reached within the trajectory's own rows. In standard mode the simulator service grades the live episode and a fresh-process replay confirms it. The details are on the [benchmark page](../benchmarks/roboarc.md) (section "Grading replays").

## Bases

5 bases (very_easy 1, easier 1, easy 1, medium 1, hard 1). Each base is one frozen scene instance and ships as two Harbor task directories,
`-privileged` and `-standard`, with the same scene, goal and grader. The difficulty is `[metadata].difficulty` in
each `task.toml`.

| Base | Difficulty | Variation (task text) | Notes | Task dirs |
|---|---|---|---|---|
| `roboarc-slotted-post-ring-01-i01` | very_easy | `slotted_post_ring_very_easy` |  | [privileged](../../tasks/roboarc-slotted-post-ring-01-i01-privileged/) / [standard](../../tasks/roboarc-slotted-post-ring-01-i01-standard/) |
| `roboarc-slotted-post-ring-02-i01` | easier | `slotted_post_ring_easier` |  | [privileged](../../tasks/roboarc-slotted-post-ring-02-i01-privileged/) / [standard](../../tasks/roboarc-slotted-post-ring-02-i01-standard/) |
| `roboarc-slotted-post-ring-03-i01` | easy | `slotted_post_ring_easy_chain` |  | [privileged](../../tasks/roboarc-slotted-post-ring-03-i01-privileged/) / [standard](../../tasks/roboarc-slotted-post-ring-03-i01-standard/) |
| `roboarc-slotted-post-ring-04-i01` | medium | `slotted_post_ring_medium_chain` |  | [privileged](../../tasks/roboarc-slotted-post-ring-04-i01-privileged/) / [standard](../../tasks/roboarc-slotted-post-ring-04-i01-standard/) |
| `roboarc-slotted-post-ring-05-i01` | hard | `slotted_post_ring_hard_chain` |  | [privileged](../../tasks/roboarc-slotted-post-ring-05-i01-privileged/) / [standard](../../tasks/roboarc-slotted-post-ring-05-i01-standard/) |

## Links

- [RoboARC benchmark page](../benchmarks/roboarc.md): the rules shared by all families (grading replays, safety readings in standard mode, the standard-mode service, the time budget).
- [Standard mode](../STANDARD_MODE.md) and [protocol](../PROTOCOL.md).
- [All families](README.md).
- [slotted_post_ring.md](../simulator/families/slotted_post_ring.md): development notes (physics, references, stress acceptance).
