# Lock picking

Family key `lock_picking` · 3 task bases · robot `franka-robotiq-2f85` · physics MuJoCo 3.14.0

A two-tab key lies on a rest next to a lock: a row of rotor plates, each with a bore and slots and able to turn only between stops, followed by a final END plate. The robot must pick up the key and insert it through all rotor plates and the END plate, respecting the plates' slots and stops. The key's leading tab is longer than its second tab, which sits one plate spacing behind it.

## Robot and grasping

One 7-DoF Franka Panda arm with a Robotiq 2F-85 parallel gripper on a fixed base (`franka-robotiq-2f85`). Actions are 8-D joint-position targets (7 arm joints and the gripper) at 40 Hz; standard mode also accepts end-effector pose targets.

Assisted grasping of rigid parts (`contact_weld_v1`): when a closing gripper has both fingers on the same graspable part, inside that part's declared grasp region, the part is attached to the gripper and follows it rigidly. Any opening command releases it. The hold is lost if the fingers open too far, the part slips away from the gripper, or the hold has to push or pull the part with more than 65 N (averaged over 20 ms), or no finger has touched the part for 0.2 s; the gripper must then open before it can take hold again. Two grippers never hold the same part. Parts touched outside their grasp region, and all other objects, are held by friction only.

## Success rule

What the grader checks, in plain words (each task's `instruction.md` states the exact rule and its limits):

- **Inserted.** The key's shaft passes through the full thickness of every plate at once, facing the insertion direction, and the rear edge of the leading tab clears the outer face of the END plate by at least 1 mm. Reaching END with only the tip does not count.
- **Fit.** The shaft fits through all bores (normal contact with the bore wall and a small tilt are allowed; the task text states bore and shaft radii), and the tabs never overlap solid plate material.
- **Rotors.** Every limited-travel rotor stays within its slot bounds (0.5 degree tolerance).
- Holding the key is allowed; release and zero velocity are not required.
- All final conditions must hold together, continuously, for 0.5 s of simulated time.

A rotor turned beyond its allowed travel ends the episode in failure, as does the key falling more than 100 mm below the tabletop.

Every task has a wall-clock budget of 11400 s (3 h 10 min) in both modes. In privileged mode the delivered trajectory is replayed open-loop twice in a fresh process: both replays must succeed, and the strict goal must be reached within the trajectory's own rows. In standard mode the simulator service grades the live episode and a fresh-process replay confirms it. The details are on the [benchmark page](../benchmarks/roboarc.md) (section "Grading replays").

## Bases

3 bases (easy 1, medium 1, hard 1). Each base is one frozen scene instance and ships as two Harbor task directories,
`-privileged` and `-standard`, with the same scene, goal and grader. The difficulty is `[metadata].difficulty` in
each `task.toml`. Numbers missing from the table are not used: base numbers are never reused, so a removed base leaves a gap.

| Base | Difficulty | Variation (task text) | Notes | Task dirs |
|---|---|---|---|---|
| `roboarc-lock-picking-03-i01` | easy | `lock_picking_easy` | three rotor plates, different rotor stops | [privileged](../../tasks/roboarc-lock-picking-03-i01-privileged/) / [standard](../../tasks/roboarc-lock-picking-03-i01-standard/) |
| `roboarc-lock-picking-04-i01` | medium | `lock_picking_medium` | three rotor plates, different rotor stops | [privileged](../../tasks/roboarc-lock-picking-04-i01-privileged/) / [standard](../../tasks/roboarc-lock-picking-04-i01-standard/) |
| `roboarc-lock-picking-05-i01` | hard | `lock_picking_hard` | four rotor plates, tighter shaft fit | [privileged](../../tasks/roboarc-lock-picking-05-i01-privileged/) / [standard](../../tasks/roboarc-lock-picking-05-i01-standard/) |

## Links

- [RoboARC benchmark page](../benchmarks/roboarc.md): the rules shared by all families (grading replays, safety readings in standard mode, the standard-mode service, the time budget).
- [Standard mode](../STANDARD_MODE.md) and [protocol](../PROTOCOL.md).
- [All families](README.md).
