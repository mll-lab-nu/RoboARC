# topogym-caliper-sorting-01-i01-standard

| | |
|---|---|
| Benchmark | TopoGym |
| Mode | standard (`eai-standard/2.3`) |
| Twin | `topogym-caliper-sorting-01-i01-privileged` |
| Simulator / robot | `topogym` / `aloha` |
| Family / instance | `caliper_sorting` / seed 0 |
| Oracle | `none` — No reference solution (no solution/). The frozen instance was validated by the build: it builds in a fresh process (seed < 2^31) and holds still under hold checks in a fresh container, and a verifier replay grades without a grader error. No graded positive yet. Human review per family (protocol §2.7). |
| Agent time budget | 21600 s; tokens: protocol default (PROTOCOL §5) |
| Base image | `labv18b/rcb-topogym:0.2.16-dev-cu128` |

## Task
Standard-mode twin of `topogym-caliper-sorting-01-i01-privileged` (same frozen instance): the agent gets only the `eai-standard/2.3` client
(docs/rcb/STANDARD_MODE_2_2.md); the simulator runs in the `sim` sidecar (`environment/docker-compose.yaml`, GPU on the
sidecar), which owns the scene and records the episode as a (T, 14) trajectory; its collect hook
(`environment/sim/finalize.sh`) replays the trajectory once in a fresh process for success, rendering the video during
that replay. The verifier runs in a separate sandbox (`tests/Dockerfile`) over the sidecar's artifacts. Compare against
the privileged task to measure the cost of losing resets, privileged state and the tooling's primitives.

## Difficulty
Same scene, instance and success rule as `topogym-caliper-sorting-01-i01-privileged` (see its Difficulty). In standard mode the agent loses resets, privileged state and planning primitives: it sees camera images and proprioception only, acts through the client, and gets one episode, so it has to estimate the scene from images and recover from its own mistakes without starting over.

## Reference solution
See `solution/`.

## Verification
See `tests/grade.py`.

## Details

### Goal
Measure the sleeves with the caliper and put every sleeve into the bin whose label gives its two caliper readings, then briefly hold everything still.

### Success requirements
- Four sleeves stand in the rack. They look alike and differ only in their inside and outside diameters. Each of the four bins carries a label 'ID r1 / OD r2' with two readings of this caliper: r1 with its knife-edge inside jaws spread in a sleeve's bore, r2 with its outside jaws closed on the sleeve's outside. A sleeve belongs in the bin whose two readings both match its own (within 0.1). A bin may stay empty; each bin holds two sleeves standing.
- The caliper is fixed face up on a stand, its jaws out over the table: it cannot be picked up. Bring each sleeve to its jaws, set the jaws with the slider and read the vernier.
- The caliper is not graduated in millimetres. Its scale is in its own unit (one main-scale division; the vernier reads tenths of it); the size of that unit and the reading with the jaws closed differ from scene to scene and are not given, so a size measured in millimetres any other way cannot be matched to a label: the readings must come from the scale.
- The slider is locked at the start. Its locking screw is the knurled knob on the slider's front face, left of the vernier on the jaws' side; a white line on its top shows its angle. It turns between two stops 45 degrees apart, and clockwise (seen from the scale side) tightens. Within 10 degrees of the loose stop the slider moves freely (about 1.5 N of friction); from 20 degrees on it is locked where it stands and does not move however hard it is pushed. Turn the knob by pinching it and rotating the gripper about its axis: it turns by friction only and is not a grasp target.
- Every sleeve's centre must end inside the bin of its class: within 40 mm of the bin's centre in both horizontal directions (the bin's 80 mm x 80 mm cavity) and 3 mm to 40 mm above the tabletop.
- Every sleeve must be released: no sleeve may still be held by an assisted grasp, and no part of the robot (arm, gripper or finger) may press on a sleeve with more than 0.05 N. Whether the caliper is locked at the end does not matter.
- Keep every sleeve's linear speed at most 1 mm/s and angular speed at most 0.02 rad/s.
- Keep all final conditions satisfied together for at least 0.5 s of simulation time. If any final condition stops holding, the confirmation interval starts again. A brief qualifying pose is not enough; let the simulation advance through the full interval.

### Task failure conditions
- A task piece falling more than 100 mm below the tabletop ends the episode in failure. Finger contact with a held movable piece may penetrate at most 2 mm; holding requires both fingers of the same hand to contact the same piece, or an active assisted grasp by that hand. All other task contacts, including piece-piece, table, arm and gripper-housing contacts, may penetrate at most 1 mm. Exceeding either limit ends the episode in failure.
- A sleeve left in a bin that is not its own (its centre inside that bin as above, released as above and still at the speeds above) for 0.5 s ends the episode in failure. When you set a sleeve down to take it again, set it on the table, not in another bin.

### Grader details (in addition to the requirements above)
- Contact penetration as this benchmark grades it (it replaces the "at most 1 mm / 2 mm" wording above): penetration is recorded, it never ends the episode by itself. A success with deep interpenetration (a task contact deeper than 3 mm, or deeper than 2 mm for at least 0.25 s: 100 consecutive physics ticks; for a finger contact with a held piece 4 mm, or 2 mm for 0.25 s) is not credited automatically but reviewed, with the recorded evidence and frames; a clean success is credited automatically. A task contact deeper than 10 mm is a simulator breakdown (below).
- Contact penetration is read from the contacts at the end of every physics tick (2.5 ms; 10 ticks per control step). "Held" is decided anew at every tick: a zero-force touch counts, and if one finger loses contact (with no active assisted grasp), the other finger is back at the non-held limits in the same tick.
- Robot safety (this benchmark's tooling): no gripper target goes lower than 15 mm below the tabletop; a commanded target that would bring any part of a gripper lower is stopped there, on the commanded path. The fingertips reach about 16 mm beyond the tool frame (TCP) along its approach axis, open or closed. Arm drive torques are capped (the effort limits in robot_spec), so an arm pushing against an obstacle stalls at that torque. An arm with any part more than 5 mm deep in a fixed object (a fixture, the table) is stopped where it is: its joint targets become its measured joints for that control step, so it stops pushing (the safety readings report these stops as static_guard).
- Forcing is recorded, not judged: an arm drive (not a gripper drive) at 95% of its torque cap or more while that arm's gripper, or a part it holds, presses on anything else with 20 N or more in total is forcing; 0.5 s of it on every physics tick is a forcing event in the evidence (`meta["safety"]`), shown to the reviewer when a success is reviewed. Forcing does not end the episode and does not change the grade by itself.
- Simulator breakdown (a non-finite state or a task contact deeper than 10 mm): the episode ends as an infrastructure error, not your failure, unless an arm was forcing (even for one tick) within the 0.25 s before it: then it is your failure (forced_breakdown: what broke and what forced it).
- Grasp regions (assisted grasping, see Grippers above; one sphere per graspable part): caliper: radius 20 mm; sleeve_0: radius 40 mm; sleeve_1: radius 40 mm; sleeve_2: radius 40 mm; sleeve_3: radius 40 mm. Both finger contacts must lie inside the part's sphere; a part touched elsewhere is grasped physically.

| | |
|---|---|
| mode | standard (RGB, optional depth, proprioception, calibrated cameras, no reset) |
| action space | `eai-standard/2.3`: `submit` chunks of `joint_position` (14, streamed at the 40 Hz control rate with the agent's `dt`) or `ee_pose` (16, IK per row); `observe` frames or a watch window; binary RCB1 responses |
| oracle | none yet (`oracle = "none"`, no `solution/`) |
| verification | `success` = live success ∧ replay success: the live episode, then one fresh-process replay in the sidecar's collect hook, after the live service has exited; the state hashes are reported, not required (GPU physics is not bit-deterministic; the CPU rigid families' is, on one host); `/logs/verifier/replay.mp4` is rendered during that replay |
| crash recovery | `eai-standard/2.3` §13: journal + per-call checkpoints, `restart: on-failure:6`, resume class `approx` |
