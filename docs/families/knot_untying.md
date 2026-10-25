# Knot untying

Family key `knot_untying` · 6 task bases · robot `aloha-2` · physics MuJoCo 3.14.0

A rope with a knot in it lies on the table. The robot must untie it and spread it out on the tabletop so that, seen from above, the rope no longer crosses or overlaps itself, then let go. The rope is a chain of rigid capsule links, 8 mm thick. In bases 06, 07 and 09 the knot is a loose figure-eight whose standing end stays clamped to the table under a staple, and in 07 and 09 the rope must also end inside a goal region that is not drawn in the scene but shown in region images, for example base 07: [region_cam_high.png](../../tasks/roboarc-knot-untying-07-i01-standard/environment/goal/region_cam_high.png), [region_cam_low.png](../../tasks/roboarc-knot-untying-07-i01-standard/environment/goal/region_cam_low.png).

## Robot and grasping

Two 6-DoF ALOHA 2 (ViperX) arms with parallel-jaw grippers on a fixed base (`aloha-2`). Actions are 14-D joint-position targets (6 arm joints and one gripper target per arm) at 40 Hz; standard mode also accepts end-effector pose targets.

Bases 02, 04 and 05: physical grasping only. Closing the gripper squeezes the rope with the gripper's drive force, and friction holds it. Bases 06, 07 and 09: assisted grasping of rigid parts (`contact_weld_v1`): when a closing gripper has both fingers on the same graspable part, inside that part's declared grasp region, the part is attached to the gripper and follows it rigidly. Any opening command releases it. The hold is lost if the fingers open too far, the part slips away from the gripper, or the hold has to push or pull the part with more than 60 N (averaged over 20 ms); the gripper must then open before it can take hold again. Two grippers never hold the same part. Parts touched outside their grasp region, and all other objects, are held by friction only. The rope's grasp regions cover it along its whole length.

## Success rule

What the grader checks, in plain words (each task's `instruction.md` states the exact rule and its limits):

- **Untangled.** No crossings, overlaps, collapsed segments or exact foldbacks in the top-down projection; segments that are not neighbours along the rope are at least one rope diameter apart in the top view. Only changing the over/under order does not count.
- **On the table.** No rope point more than 1 mm below the tabletop or more than one rope diameter plus 3 mm above it.
- **Released.** All grasps released, with at least 1 mm of clearance between the rope and every movable robot part.
- **At rest.** Every rope point moves at most 1 mm/s and drifts at most 1 mm during the confirmation.
- **Goal region (07, 09).** The whole centre line of the rope lies inside the goal rectangle.
- All final conditions must hold together, continuously, for 0.5 s of simulated time.

Every task has a wall-clock budget of 10800 s (3 h) in both modes. In privileged mode the delivered trajectory is replayed open-loop twice in a fresh process: both replays must succeed, and the strict goal must be reached within the trajectory's own rows. In standard mode the simulator service grades the live episode and a fresh-process replay confirms it. The details are on the [benchmark page](../benchmarks/roboarc.md) (section "Grading replays").

## Bases

6 bases (very_easy 1, easier 2, medium 2, hard 1). Each base is one frozen scene instance and ships as two Harbor task directories,
`-privileged` and `-standard`, with the same scene, goal and grader. The difficulty is `[metadata].difficulty` in
each `task.toml`. Numbers missing from the table are not used: base numbers are never reused, so a removed base leaves a gap.

| Base | Difficulty | Variation (task text) | Notes | Task dirs |
|---|---|---|---|---|
| `roboarc-knot-untying-02-i01` | easier | `easier` |  | [privileged](../../tasks/roboarc-knot-untying-02-i01-privileged/) / [standard](../../tasks/roboarc-knot-untying-02-i01-standard/) |
| `roboarc-knot-untying-04-i01` | medium | `knot_untying_medium` |  | [privileged](../../tasks/roboarc-knot-untying-04-i01-privileged/) / [standard](../../tasks/roboarc-knot-untying-04-i01-standard/) |
| `roboarc-knot-untying-05-i01` | hard | `knot_untying_hard` |  | [privileged](../../tasks/roboarc-knot-untying-05-i01-privileged/) / [standard](../../tasks/roboarc-knot-untying-05-i01-standard/) |
| `roboarc-knot-untying-06-i01` | very_easy | `knot_untying_rd_very_easy` | loose figure-eight, standing end clamped under a staple, long free tail | [privileged](../../tasks/roboarc-knot-untying-06-i01-privileged/) / [standard](../../tasks/roboarc-knot-untying-06-i01-standard/) |
| `roboarc-knot-untying-07-i01` | easier | `knot_untying_rd_easier` | loose figure-eight, end clamped; two posts and a curb; goal region | [privileged](../../tasks/roboarc-knot-untying-07-i01-privileged/) / [standard](../../tasks/roboarc-knot-untying-07-i01-standard/) |
| `roboarc-knot-untying-09-i01` | medium | `knot_untying_rd_medium` | loose figure-eight round an eye post, end clamped; a second post and a curb; goal region | [privileged](../../tasks/roboarc-knot-untying-09-i01-privileged/) / [standard](../../tasks/roboarc-knot-untying-09-i01-standard/) |

## Links

- [RoboARC benchmark page](../benchmarks/roboarc.md): the rules shared by all families (grading replays, safety readings in standard mode, the standard-mode service, the time budget).
- [Standard mode](../STANDARD_MODE.md) and [protocol](../PROTOCOL.md).
- [All families](README.md).
- [knot_untying.md](../simulator/families/knot_untying.md): development notes (physics, references, stress acceptance).
