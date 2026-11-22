# roboarc-wire-splicing-01-i01-standard

| | |
|---|---|
| Benchmark | RoboARC |
| Mode | standard (`eai-standard/2.4`) |
| Twin | `roboarc-wire-splicing-01-i01-privileged` |
| Simulator / robot | `topogym` / `aloha` |
| Family / instance | `wire_splicing` / seed 0 |
| Oracle | `none` — No reference solution (no solution/). The frozen instance was validated by the build: it builds in a fresh process (seed < 2^31) and holds still under hold checks in a fresh container, and a verifier replay grades without a grader error. No graded positive yet. Human review per family (protocol §2.7). |
| Agent time budget | 21600 s; tokens: protocol default (PROTOCOL §5) |
| Base image | `{{IMAGE}}` |

## Task
Standard-mode twin of `roboarc-wire-splicing-01-i01-privileged` (same frozen instance): the agent gets only the `eai-standard/2.4` client
(docs/STANDARD_MODE.md); the simulator runs in the `sim` sidecar (`environment/docker-compose.yaml`, GPU on the
sidecar), which owns the scene and records the episode as a (T, 14) trajectory; its collect hook
(`environment/sim/finalize.sh`) replays the trajectory once in a fresh process for success, rendering the video during
that replay. The verifier runs in a separate sandbox (`tests/Dockerfile`) over the sidecar's artifacts. Compare against
the privileged task to measure the cost of losing resets, privileged state and the tooling's primitives.

## Difficulty
Same scene, instance and success rule as `roboarc-wire-splicing-01-i01-privileged` (see its Difficulty). In standard mode the agent loses resets, privileged state and planning primitives: it sees camera images and proprioception only, acts through the client, and gets one episode, so it has to estimate the scene from images and recover from its own mistakes without starting over.

## Reference solution
{{REFERENCE_SOLUTION}}

## Verification
See `tests/grade.py`.

{{NOP}}

## Details

### Goal
Splice the red and the blue wire: bend the red wire's bare end into an eye round the yellow rod of the bending post, thread the blue wire's bare end through the eye, wind the blue wire's bare tail round the red wire's doubled end at least 1.5 turns, at least 1 full turn of them round both strands of the doubled end together, fold the red wire's leftover end back over the coil, then let go.

### Success requirements
- The red wire's bare end is turned back on itself by more than 150 degrees into an eye that is closed at its neck: the two legs touch where they first meet below the bend.
- The blue wire passes through the closed eye exactly once (its linking number with the eye is +1 or -1).
- The blue wire's bare tail winds at least 1.5 turns round the red wire's doubled section. The turns are counted about the doubled section's axis: the line fitted to the red wire's stem (its bare part from the strip to the neck) and the return-leg points that lie beside it, within 8.9 mm (2.5 core diameters; the core is 3.56 mm thick) of the stem's own line. The counted coil is the longest run of the blue wire's bare points within 8.9 mm of that axis and between the neck and the red wire's strip, each widened by one core diameter.
- At least 1 full turn of the coil goes round both strands of the red wire's doubled end together, not only round the line between them. The strands: the stem side, from the strip to the eye's apex (its point nearest the eye end of the axis), and the return side, from the apex to the return leg's point farthest toward the strip (a leftover end folded back from there is not one of them). For each point of the counted coil, each strand's point at the same station along the axis is taken, where the strand crosses the plane through the coil point perpendicular to the axis, and the coil point's angle about it in that plane; along the coil these angles accumulate into turns about each strand, a step counting only where both strands have their points. Some contiguous part of the coil must wind at least 1 turn about each strand, both the same way as the coil winds about the axis: a loop round one strand only, or one passing between the strands, does not count.
- The coil is tight: its points lie on average at most 7.1 mm (2 core diameters) from the axis.
- The red wire's leftover end is folded back: the direction from its fourth-last to its last point points back toward the eye (its dot product with the axis, which runs from the eye toward the strip, is at most -0.5), and its tip is at least 4.3 mm (1.2 core diameters) from the axis.
- Let go: no hand holds a wire. The blue wire's turns may drop by at most 0.25 and the threading may not change while the conditions are held, and the splice must be still: every bare point's speed relative to the bare points' mean motion at most 5 mm/s.
- Keep all final conditions satisfied together for at least 2 s of simulation time. If any final condition stops holding, the confirmation interval starts again. A brief qualifying pose is not enough; let the simulation advance through the full interval.

### Task failure conditions
- A task piece falling more than 100 mm below the tabletop ends the episode in failure. Finger contact with a held movable piece may penetrate at most 2 mm; holding requires both fingers of the same hand to contact the same piece, or an active assisted grasp by that hand. All other task contacts, including piece-piece, table, arm and gripper-housing contacts, may penetrate at most 1 mm. Exceeding either limit ends the episode in failure.

### Grader details (in addition to the requirements above)
- Contact penetration as this benchmark grades it: penetration is recorded, it never ends the episode by itself. A success with deep interpenetration (a task contact deeper than 3 mm, or deeper than 2 mm for at least 0.25 s: 100 consecutive physics ticks; for a finger contact with a held piece 4 mm, or 2 mm for 0.25 s) is not credited automatically but reviewed, with the recorded evidence and frames; a clean success is credited automatically. A task contact deeper than 10 mm is a simulator breakdown (below).
- Contact penetration is read from the contacts at the end of every physics tick (2.5 ms; 10 ticks per control step). "Held" is decided anew at every tick: a zero-force touch counts, and if one finger loses contact (with no active assisted grasp), the other finger is back at the non-held limits in the same tick.
- Robot safety (this benchmark's tooling): no gripper target goes lower than 15 mm below the tabletop; a commanded target that would bring any part of a gripper lower is stopped there, on the commanded path. The fingertips reach about 16 mm beyond the tool frame (TCP) along its approach axis, open or closed. Arm drive torques are capped (the effort limits in robot_spec), so an arm pushing against an obstacle stalls at that torque. An arm with any part more than 5 mm deep in a fixed object (a fixture, the table) is stopped where it is: its joint targets become its measured joints for that control step, so it stops pushing (the safety readings report these stops as static_guard).
- Forcing is recorded, not judged: an arm drive (not a gripper drive) at 95% of its torque cap or more while that arm's gripper, or a part it holds, presses on anything else with 20 N or more in total is forcing; 0.5 s of it on every physics tick is a forcing event in the evidence (`meta["safety"]`), shown to the reviewer when a success is reviewed. Forcing does not end the episode and does not change the grade by itself.
- Simulator breakdown (a non-finite state or a task contact deeper than 10 mm): the episode ends as an infrastructure error, not your failure, unless an arm was forcing (even for one tick) within the 0.25 s before it: then it is your failure (forced_breakdown: what broke and what forced it).
- Grasp regions (assisted grasping, see Grippers above; one or more spheres per graspable part): wire_a: 63 spheres of radii 26.1 mm, 5.1 mm; wire_b: 97 spheres of radii 25.1 mm, 5.5 mm. Both finger contacts must lie inside one of the part's spheres; a part touched elsewhere is grasped physically.

| | |
|---|---|
| mode | standard (RGB, optional depth, proprioception, calibrated cameras, no reset) |
| action space | `eai-standard/2.4`: `submit` chunks of `joint_position` (14, streamed at the 40 Hz control rate with the agent's `dt`) or `ee_pose` (16, IK per row); `observe` frames or a watch window; binary RCB1 responses |
| oracle | `full` (`solution/`) |
| verification | `success` = live success ∧ replay success: the live episode, then one fresh-process replay in the sidecar's collect hook, after the live service has exited; the state hashes are reported, not required (MuJoCo on the CPU replays bit for bit on one host; the rule does not rely on it); `/logs/verifier/replay.mp4` is rendered during that replay |
| crash recovery | `eai-standard/2.4` §13: journal + per-call checkpoints, `restart: on-failure:6`, resume class `approx` |

### Provenance
{{PROVENANCE}}

### Change Log
{{CHANGE_LOG}}
