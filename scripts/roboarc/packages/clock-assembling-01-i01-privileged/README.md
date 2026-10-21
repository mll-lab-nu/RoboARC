# topogym-clock-assembling-01-i01-privileged

| | |
|---|---|
| Benchmark | TopoGym |
| Mode | privileged |
| Twin | `topogym-clock-assembling-01-i01-standard` |
| Simulator / robot | `topogym` / `aloha` |
| Family / instance | `clock_assembling` / seed 0 |
| Oracle | `none` — No reference solution (no solution/). The frozen instance was validated by the build: it builds in a fresh process (seed < 2^31) and holds still under hold checks in a fresh container, and a verifier replay grades without a grader error. No graded positive yet. Human review per family (protocol §2.7). |
| Agent time budget | 21600 s; tokens: protocol default (PROTOCOL §5) |
| Base image | `labv18b/rcb-topogym:0.2.16-dev-cu128` |

## Task
TopoGym clock_assembling (aloha, Genesis): produce one replayable trajectory. Assemble the watch movement's gear train: the four wheels on their pivots in the main plate, the train bridge over them on its pillars with every upper pivot in its hole, the two bridge screws tight; then wind the barrel through its ratchet, let go and watch the fourth wheel run. Frozen scene instance (variation clock_assembling_gear_train, seed 0).

## Difficulty
Two 6-dof aloha (viperx) arms with parallel-jaw grippers on a fixed base with a 14-D joint-target action space at 40 Hz; success needs the task's predicate (above) held for 0.5 s. Grasping: assisted grasping (contact_weld_v1).

## Reference solution
None yet (`oracle = "none"`, no `solution/`). A reference solution must be written for this instance (assisted
grasping, if the family enables it, is the same for the oracle and the agent).

## Verification
Open-loop replay of `/app/output/trajectory.npz` from `tests/instance.json` in a fresh process, twice; pass iff the
task's predicate holds after every replay (`tests/grade.py`). `/logs/verifier/replay.mp4` is rendered during the second
replay (two passes, no separate video replay). The physics runs on the CPU and is bit-deterministic on one host; the final-state hashes
(`deterministic`) are still diagnostic, as for the GPU families; the rebuilt frozen scene must match (`instance_ok`)
(robot_coding_bench's GPU-backend rule).

## Details
| | |
|---|---|
| Grasping | assisted grasping (contact_weld_v1) |
| Success rule | task-native success (goal_complete True; frozen parameters; terminal revision clock_gear_train_v3) |

### Goal
Assemble the watch movement's gear train: the four wheels on their pivots in the main plate, the train bridge over them on its pillars with every upper pivot in its hole, the two bridge screws tight; then wind the barrel through its ratchet, let go and watch the fourth wheel run.

### Success requirements

- Seat the barrel, the center wheel, the third wheel and the fourth wheel on their lower pivots in the main plate (each wheel no more than 1 mm above its seat); a wheel seats once its lower pivot is down in its hole: the wheel at most 0.8 mm above its seat, within 0.6 mm of the hole's axis and tilted less than 3 degrees. Each wheel's pinion must mesh with the previous wheel's teeth (barrel, center, third, fourth); a wheel lowered with its pinion's teeth tip to tip on the other wheel's teeth stops until it is turned a little.
- Every upper pivot (the barrel: its arbor) inside its hole in the bridge, at least 1 mm deep and within the hole's radius of its axis.
- The bridge seated on its pillars: within 0.5 mm of its seat height, 0.5 mm of its seat position and tilted less than 0.5 degrees.
- Both screws in the bridge's two screw holes (either screw in either hole, one to a hole) and turned clockwise by 2 full turns there (no thread is simulated: the turns are counted only while the screw's head sits on the bridge over its hole, its underside between 0.3 mm below and 0.8 mm above the bridge's top and its axis within 0.6 mm of the hole's; counterclockwise turns count back; a screw moved to the other hole starts again). Once both are tight, every wheel is seated, every upper pivot is in its bridge hole and the bridge is seated (height and tilt), the bridge is fixed: screw the bridge down after the wheels, not before.
- After the bridge is fixed, wind the barrel: turn its square (the ratchet under it turns with it) counterclockwise seen from above; the click lets it turn that way only and holds it one ratchet tooth (18 degrees) behind the furthest point reached. Wind it at least 120 degrees (as held by the click), then stop winding and let go of the square: the mainspring drives the barrel and the train, and any further advance of the square counts as winding again. Half a second after the last winding the rules time a 5 s run: the fourth wheel must turn at least 3.5 times in it, spend at most 0.2 s slower than a tenth of its free-run speed (the speed the mainspring's present windup gives the train against its own damping and friction), and turn 450 times as much as the barrel in the opposite direction, within 2 %. The run counts only for the mainspring alone: from the last winding to the end of the run neither the hands (any part of the robot) nor the probe may touch the barrel or a wheel, and the barrel must turn the way the mainspring drives it; otherwise the run does not count until the barrel is wound again. Winding again restarts the run.
- Holding parts is allowed, apart from the run above.
- Keep all final conditions satisfied together for at least 0.5 s of simulation time. If any final condition stops holding, the confirmation interval starts again. A brief qualifying pose is not enough; let the simulation advance through the full interval.

## Task failure conditions

- A task piece falling more than 100 mm below the tabletop ends the episode in failure. Finger contact with a held movable piece may penetrate at most 2 mm; holding requires both fingers of the same hand to contact the same piece, or an active assisted grasp by that hand. All other task contacts, including piece-piece, table, arm and gripper-housing contacts, may penetrate at most 1 mm. Exceeding either limit ends the episode in failure.

## Grader details (in addition to the requirements above)

- Contact penetration as this benchmark grades it (it replaces the "at most 1 mm / 2 mm" wording above): penetration is recorded, it never ends the episode by itself. A success with deep interpenetration (a task contact deeper than 3 mm, or deeper than 2 mm for at least 0.25 s: 100 consecutive physics ticks; for a finger contact with a held piece 4 mm, or 2 mm for 0.25 s) is not credited automatically but reviewed, with the recorded evidence and frames; a clean success is credited automatically. A task contact deeper than 10 mm is a simulator breakdown (below).
- Contact penetration is read from the contacts at the end of every physics tick (2.5 ms; 10 ticks per control step). "Held" is decided anew at every tick: a zero-force touch counts, and if one finger loses contact (with no active assisted grasp), the other finger is back at the non-held limits in the same tick.
- Robot safety (this benchmark's tooling): no gripper target goes lower than 15 mm below the tabletop; a commanded target that would bring any part of a gripper lower is stopped there, on the commanded path. The fingertips reach about 16 mm beyond the tool frame (TCP) along its approach axis, open or closed. Arm drive torques are capped (the effort limits in robot_spec), so an arm pushing against an obstacle stalls at that torque. An arm with any part more than 5 mm deep in a fixed object (a fixture, the table) is stopped where it is: its joint targets become its measured joints for that control step, so it stops pushing (the safety readings report these stops as static_guard).
- Forcing is recorded, not judged: an arm drive (not a gripper drive) at 95% of its torque cap or more while that arm's gripper, or a part it holds, presses on anything else with 20 N or more in total is forcing; 0.5 s of it on every physics tick is a forcing event in the evidence (`s.safety()`, `s.status()["forcing"]`), shown to the reviewer when a success is reviewed. Forcing does not end the episode and does not change the grade by itself.
- Simulator breakdown (a non-finite state or a task contact deeper than 10 mm): the episode ends as an infrastructure error, not your failure, unless an arm was forcing (even for one tick) within the 0.25 s before it: then it is your failure (forced_breakdown: what broke and what forced it).
- The rules (seating, the tooth-on-tooth gates, meshing, the screws' turns, the bridge's fixing, the click and the run test's timing) are applied after every physics tick (2.5 ms); the success conditions are checked at the end of every tick. The mainspring is a spring between the barrel's arbor and drum; the click is a limit on the arbor's turn that the rules move up as it is wound.
- Grasp regions (assisted grasping, see Grippers above; one sphere per graspable part): barrel: radius 65.8 mm; center: radius 42 mm; third: radius 33.8 mm; fourth: radius 33.8 mm; bridge: radius 120.6 mm; screw1: radius 20.9 mm; screw2: radius 20.9 mm; probe: radius 56.1 mm; barrel_arbor: radius 30.6 mm. Both finger contacts must lie inside the part's sphere; a part touched elsewhere is grasped physically. Each region's centre (in its link's frame) is in `/app/instance.json` under `simulator.grasp_assist.targets`.
- A failure is reported in `s.status()["episode_end"]` as "A task body fell below the table" or "forced_breakdown: <what>" (<what>: the readable reason); `s.safety()` reports forcing, contact depths, volume ratios and table-guard stops against these limits, and `s.status()["review"]` the deep task contacts that would hold a success for review.
