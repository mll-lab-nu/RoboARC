# roboarc-slotted-post-ring-03-i01-standard

| | |
|---|---|
| Benchmark | RoboARC |
| Mode | standard (`eai-standard/2.4`) |
| Twin | `roboarc-slotted-post-ring-03-i01-privileged` |
| Simulator / robot | `topogym` / `aloha` |
| Family / instance | `slotted_post_ring` / seed 0 |
| Oracle | `none` — No reference solution (no solution/). The frozen instance was validated by the build: it builds in a fresh process (seed < 2^31) and holds still under hold checks in a fresh container, and a verifier replay grades without a grader error. No graded positive yet. Human review per family (protocol §2.7). |
| Agent time budget | 21600 s; tokens: protocol default (PROTOCOL §5) |
| Base image | `{{IMAGE}}` |

## Task
Standard-mode twin of `roboarc-slotted-post-ring-03-i01-privileged` (same frozen instance): the agent gets only the `eai-standard/2.4` client
(docs/STANDARD_MODE.md); the simulator runs in the `sim` sidecar (`environment/docker-compose.yaml`, GPU on the
sidecar), which owns the scene and records the episode as a (T, 14) trajectory; its collect hook
(`environment/sim/finalize.sh`) replays the trajectory once in a fresh process for success, rendering the video during
that replay. The verifier runs in a separate sandbox (`tests/Dockerfile`) over the sidecar's artifacts. Compare against
the privileged task to measure the cost of losing resets, privileged state and the tooling's primitives.

## Difficulty
Same scene, instance and success rule as `roboarc-slotted-post-ring-03-i01-privileged` (see its Difficulty). In standard mode the agent loses resets, privileged state and planning primitives: it sees camera images and proprioception only, acts through the client, and gets one episode, so it has to estimate the scene from images and recover from its own mistakes without starting over.

## Reference solution
{{REFERENCE_SOLUTION}}

## Verification
See `tests/grade.py`.

{{NOP}}

## Details

### Goal
Separate and place the entire removable metal ring inside the green dashed goal region. Keep the post, cord, beads and end discs on the puzzle/system side.

### Success requirements
- Put every part of the entire removable metal ring inside the fixed goal volume, leaving at least 1 mm inside its horizontal border. The centre alone is insufficient; no tip, spike, ring segment or cord segment may remain outside.
- Keep the target above the tabletop (contact tolerance 1 mm below the surface) and within the goal volume's height. It need not touch the table. The goal volume extends 230.257 mm above the tabletop.
- Keep the post, cord, beads and end discs, and every other remaining puzzle component, entirely on the system side of the fixed separator. Moving the whole puzzle into the goal is not completion.
- Keep each target rigid part's linear speed at most 1 mm/s and angular speed at most 0.02 rad/s. Holding the target with the gripper is allowed; release is not required.
- Keep all final conditions satisfied together for at least 0.5 s of simulation time. If any final condition stops holding, the confirmation interval starts again. A brief qualifying pose is not enough; let the simulation advance through the full interval.

### Grader details (in addition to the requirements above)
- Goal square in this instance: the inner edge of the 5 mm wide dashed green line is the containment limit (the region's border runs along the middle of the line); it encloses a 238.3 mm x 238.3 mm square. Every point counts: for a rigid part every vertex of its visual and collision meshes; for the cord every vertex of its links (capsules).
- The separator is an invisible, fixed vertical plane parallel to the goal square's near edge, 5 mm beyond the point of all remaining components that was closest to the goal square at the start; the square's near edge is at least 30 mm beyond it. The cord, beads and end discs are free to move (their speed is not checked), but no point of them may cross that plane.
- The cord is a chain of 87 rigid capsule links (16.47 mm long, 7 mm thick) on ball joints that bend and twist freely (no spring), with damping. The links do not deform: nothing about the cord's shape or volume ends the episode. A part of the chain (a link, bead, end disc or the ring) moving faster than 3 m/s or spinning faster than 120 rad/s is slowed by extra damping (a speed clamp); this is not reported and does not end the episode.
- Robot safety (this benchmark's tooling): no gripper target goes lower than 15 mm below the tabletop; a commanded target that would bring any part of a gripper lower is stopped there, on the commanded path. The fingertips reach about 16 mm beyond the tool frame (TCP) along its approach axis, open or closed. Arm drive torques are capped (the effort limits in robot_spec), so an arm pushing against an obstacle stalls at that torque. An arm with any part more than 5 mm deep in a fixed object (a fixture, the table) is stopped where it is: its joint targets become its measured joints for that control step, so it stops pushing (reported in clipped[]).
- Forcing is recorded, not judged: an arm drive (not a gripper drive) at 95% of its torque cap or more while that arm's gripper, or a part it holds, presses on anything else with 20 N or more in total is forcing; 0.5 s of it on every physics tick is a forcing event in the evidence (`meta["safety"]`), shown to the reviewer when a success is reviewed. Forcing does not end the episode and does not change the grade by itself.
- Simulator breakdown (a MuJoCo instability: a non-finite or diverging acceleration, position or velocity): the episode ends as an infrastructure error, not your failure, unless an arm was forcing (even for one tick) within the 0.25 s before it: then it is your failure (forced_breakdown: what broke and what forced it).
- Grasp regions (assisted grasping, see Grippers above; one sphere per graspable part): slotted_chain: radius 79 mm. Both finger contacts must lie inside the part's sphere; a part touched elsewhere is grasped physically.

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
