# roboarc-bed-bolt-04-i01-privileged

| | |
|---|---|
| Benchmark | RoboARC |
| Mode | privileged |
| Twin | `roboarc-bed-bolt-04-i01-standard` |
| Simulator / robot | `topogym` / `aloha` |
| Family / instance | `bed_bolt` / seed 0 |
| Oracle | `none` — No reference solution (no solution/). The frozen instance was validated by the build: it builds in a fresh process (seed < 2^31) and holds still under hold checks in a fresh container, and a verifier replay grades without a grader error. No graded positive yet. Human review per family (protocol §2.7). |
| Agent time budget | 21600 s; tokens: protocol default (PROTOCOL §5) |
| Base image | `{{IMAGE}}` |

## Task
RoboARC bed_bolt (ALOHA 2, MuJoCo): produce one replayable trajectory. Drive the long bolt into the blind hole in the bed frame's corner block with the ratcheting L-shaped Phillips key (a cross tip on a ratchet) until its head is within 1.5 mm of the block face. Frozen scene instance (variation bed_bolt_phillips, seed 0).

## Difficulty
Two 6-dof aloha (viperx) arms with parallel-jaw grippers on a fixed base with a 14-D joint-target action space at 40 Hz; success needs the task's predicate (above) held for 0.5 s. Grasping: assisted grasping (contact_weld_v1).

## Reference solution
{{REFERENCE_SOLUTION}}

## Verification
Open-loop replay of `/app/output/trajectory.npz` from the frozen instance (`/tests/instance.json`, staged from the image and checked against `tests/runtime_binding.json`) in a fresh process, twice; pass iff the
task's predicate holds after every replay (`tests/grade.py`). `/logs/verifier/replay.mp4` is rendered during the second
replay (two passes, no separate video replay). The physics runs on the CPU and is bit-deterministic on one host; the final-state hashes
(`deterministic`) are still diagnostic, as for the GPU families; the rebuilt frozen scene must match (`instance_ok`).
A trajectory may have at most {{ROW_CAP}} rows (`RCB_MAX_STEPS`: two replays must fit the verifier's timeout; a longer one is
`bad_trajectory`).

{{NOP}}

## Details
| | |
|---|---|
| Grasping | assisted grasping (contact_weld_v1) |
| Success rule | task-native success (goal_complete True; frozen parameters; terminal revision bed_bolt_seated_v1) |

### Goal
Drive the long bolt into the blind hole in the bed frame's corner block with the ratcheting L-shaped Phillips key (a cross tip on a ratchet) until its head is within 1.5 mm of the block face.

### Success requirements
- Drive the bolt in until its head is within 1.5 mm of the block face, its tip at least 83.5 mm deep in the hole, with the thread caught, and its axis within 2 degrees of the hole's axis.
- The hole is threaded. The thread catches when the bolt's tip is 2.5 to 8.2 mm inside the hole, within 1 mm of the hole's axis and 2 degrees of its direction, with the bolt's thread lined up with the hole's (a straight push lines up somewhere in that range; turning the bolt about its axis moves where); each turn then drives the bolt 5.7 mm in. A bolt the thread does not hold cannot be pushed in past the thread: not past 8.2 mm before it has ever caught, and afterwards not deeper than it has already been turned in; it catches again only there, in the same thread. With fewer than 2 turns engaged the thread cannot carry the bolt: when no finger and no key end holds it, it droops, and past 3 degrees it lets go and falls out.
- Once the thread has engaged, pushing or pulling the bolt along its axis does not advance it. Fingers cannot turn the bolt (they slip); use the key.
- The key has a ratchet at its bend: swung in the tightening direction it turns its tip, and with it the bolt; swung back, the handle turns freely (with a slight drag) while the tip stays still in the socket, so the tip need not come out between swings.
- Keep the bolt's and the key's linear speed at most 1 mm/s and angular speed at most 0.02 rad/s. Holding is allowed; the key may stay in the socket or anywhere else.
- Keep all final conditions satisfied together for at least 0.5 s of simulation time. If any final condition stops holding, the confirmation interval starts again. A brief qualifying pose is not enough; let the simulation advance through the full interval.

### Task failure conditions
- A task piece falling more than 100 mm below the tabletop ends the episode in failure. Finger contact with a held movable piece may penetrate at most 2 mm; holding requires both fingers of the same hand to contact the same piece, or an active assisted grasp by that hand. All other task contacts, including piece-piece, table, arm and gripper-housing contacts, may penetrate at most 1 mm. Exceeding either limit ends the episode in failure.

### Grader details (in addition to the requirements above)
- Contact penetration as this benchmark grades it (it replaces the "at most 1 mm / 2 mm" wording above): penetration is recorded, it never ends the episode by itself. A success with deep interpenetration (a task contact deeper than 3 mm, or deeper than 2 mm for at least 0.25 s: 100 consecutive physics ticks; for a finger contact with a held piece 4 mm, or 2 mm for 0.25 s) is not credited automatically but reviewed, with the recorded evidence and frames; a clean success is credited automatically. A task contact deeper than 10 mm is a simulator breakdown (below).
- Contact penetration is read from the contacts at the end of every physics tick (2.5 ms; 10 ticks per control step). "Held" is decided anew at every tick: a zero-force touch counts, and if one finger loses contact (with no active assisted grasp), the other finger is back at the non-held limits in the same tick.
- Robot safety (this benchmark's tooling): no gripper target goes lower than 15 mm below the tabletop; a commanded target that would bring any part of a gripper lower is stopped there, on the commanded path. The fingertips reach about 16 mm beyond the tool frame (TCP) along its approach axis, open or closed. Arm drive torques are capped (the effort limits in robot_spec), so an arm pushing against an obstacle stalls at that torque. An arm with any part more than 5 mm deep in a fixed object (a fixture, the table) is stopped where it is: its joint targets become its measured joints for that control step, so it stops pushing (the safety readings report these stops as static_guard).
- Forcing is recorded, not judged: an arm drive (not a gripper drive) at 95% of its torque cap or more while that arm's gripper, or a part it holds, presses on anything else with 20 N or more in total is forcing; 0.5 s of it on every physics tick is a forcing event in the evidence (`s.safety()`, `s.status()["forcing"]`), shown to the reviewer when a success is reviewed. Forcing does not end the episode and does not change the grade by itself.
- Simulator breakdown (a non-finite state or a task contact deeper than 10 mm): the episode ends as an infrastructure error, not your failure, unless an arm was forcing (even for one tick) within the 0.25 s before it: then it is your failure (forced_breakdown: what broke and what forced it).
- Grasp regions (assisted grasping, see Grippers above; one sphere per graspable part): bolt: radius 53.2 mm; key: radius 107.9 mm. Both finger contacts must lie inside the part's sphere; a part touched elsewhere is grasped physically. Each region's centre (in its link's frame) is in `/app/instance.json` under `simulator.grasp_assist.targets`.
- A failure is reported in `s.status()["episode_end"]` as "A task body fell below the table" or "forced_breakdown: <what>" (<what>: the readable reason); `s.safety()` reports forcing, contact depths, volume ratios and table-guard stops against these limits, and `s.status()["review"]` the deep task contacts that would hold a success for review.

### Provenance
{{PROVENANCE}}

### Change Log
{{CHANGE_LOG}}
