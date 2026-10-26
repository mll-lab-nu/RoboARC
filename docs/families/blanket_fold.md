# Blanket fold

Family key `blanket_fold` · 5 task bases · robot `aloha-2` · physics MuJoCo 3.14.0

A 600 mm x 480 mm cloth blanket lies on the table, either flat or already folded over on itself. The robot must fold it into a given target shape at a given place on the table, then let go and move away. In every base the target is the half fold: the far half is turned over onto the near half about the middle line parallel to the long edges, giving a two-layer 600 mm x 240 mm stack. In some bases the top layer's corner must then also be dog-eared (folded over at about 45 degrees). Near and far are as seen in the overhead camera `cam_high`.

Each task ships target images of the required final state, seen from the overhead and the low camera, for example base 02: [target_cam_high.png](../../tasks/roboarc-blanket-fold-02-i01-standard/environment/goal/target_cam_high.png), [target_cam_low.png](../../tasks/roboarc-blanket-fold-02-i01-standard/environment/goal/target_cam_low.png).

## Robot and grasping

Two 6-DoF ALOHA 2 (ViperX) arms with parallel-jaw grippers on a fixed base (`aloha-2`). Actions are 14-D joint-position targets (6 arm joints and one gripper target per arm) at 40 Hz; standard mode also accepts end-effector pose targets.

Assisted grasping of soft material (`pbd_local_patch_v1`): a pinch with both fingers on an exposed patch of the blanket attaches that local patch to the gripper, while the rest of the cloth stays deformable. Opening the gripper releases it; the hold is also lost on excessive opening, excessive material strain or a follow error above 10 mm. The two hands may hold separate, non-overlapping patches of the blanket. A safety floor keeps the grippers from pressing the cloth into the table: apart from the finger pads, no gripper part is driven below the top of the cloth under it.

## Success rule

What the grader checks, in plain words (each task's `instruction.md` states the exact rule and its limits):

- **Corners.** Each of the four corner points lies within 55 mm (top view) of its place in the target fold.
- **Layer order.** For at least 88 % of the blanket's points (and 80 % of those that have material beneath them in the target), the panel directly beneath is the same as in the target fold.
- **Outline.** After the best rigid top-view fit, the covered area overlaps the target's with an intersection over union of at least 0.85.
- **Flat and placed.** The folded cloth lies flat on the table (each point within 35 mm of its height in the target fold) and where the target images show it: within 40 mm and 5 degrees. In base 05, cloth resting on the block does not count.
- **Material.** No point more than 8 mm below the tabletop; no edge stretched beyond 1.15 times its length.
- **Released and at rest.** All grasps released, the robot's moving parts withdrawn to at least 5 mm of clearance (the fixed arm bases are not counted, since 2.1.0), every cloth point at most 30 mm/s and at most 3 mm of drift during the confirmation.
- All final conditions must hold together, continuously, for 0.5 s of simulated time.

The episode fails if any cloth point falls more than 100 mm below the tabletop. The cloth is simulated on the GPU with a deterministic solver.

Every task has a wall-clock budget of 10800 s (3 h) in both modes. In privileged mode the delivered trajectory is replayed open-loop twice in a fresh process: both replays must succeed, and the strict goal must be reached within the trajectory's own rows. In standard mode the simulator service grades the live episode and a fresh-process replay confirms it. The details are on the [benchmark page](../benchmarks/roboarc.md) (section "Grading replays").

## Bases

5 bases (very_easy 1, easier 1, easy 1, medium 1, hard 1). Each base is one frozen scene instance and ships as two Harbor task directories,
`-privileged` and `-standard`, with the same scene, goal and grader. The difficulty is `[metadata].difficulty` in
each `task.toml`.

| Base | Difficulty | Variation (task text) | Notes | Task dirs |
|---|---|---|---|---|
| `roboarc-blanket-fold-01-i01` | very_easy | `blanket_fold_very_easy` | starts flat; target: half fold | [privileged](../../tasks/roboarc-blanket-fold-01-i01-privileged/) / [standard](../../tasks/roboarc-blanket-fold-01-i01-standard/) |
| `roboarc-blanket-fold-02-i01` | easier | `blanket_fold_easier` | starts flat; target: half fold with the top layer's corner dog-eared | [privileged](../../tasks/roboarc-blanket-fold-02-i01-privileged/) / [standard](../../tasks/roboarc-blanket-fold-02-i01-standard/) |
| `roboarc-blanket-fold-03-i01` | easy | `blanket_fold_easy` | starts folded over; target: half fold | [privileged](../../tasks/roboarc-blanket-fold-03-i01-privileged/) / [standard](../../tasks/roboarc-blanket-fold-03-i01-standard/) |
| `roboarc-blanket-fold-04-i01` | medium | `blanket_fold_medium` | starts folded over; target: half fold with dog-ear | [privileged](../../tasks/roboarc-blanket-fold-04-i01-privileged/) / [standard](../../tasks/roboarc-blanket-fold-04-i01-standard/) |
| `roboarc-blanket-fold-05-i01` | hard | `blanket_fold_hard` | as 04, with a fixed block standing beyond the blanket's far side | [privileged](../../tasks/roboarc-blanket-fold-05-i01-privileged/) / [standard](../../tasks/roboarc-blanket-fold-05-i01-standard/) |

## Links

- [RoboARC benchmark page](../benchmarks/roboarc.md): the rules shared by all families (grading replays, safety readings in standard mode, the standard-mode service, the time budget).
- [Standard mode](../STANDARD_MODE.md) and [protocol](../PROTOCOL.md).
- [All families](README.md).
- [blanket_cloth_interface.md](../simulator/families/blanket_cloth_interface.md): development notes on the cloth interface shared by the blanket tasks (there is no separate note for this family).
