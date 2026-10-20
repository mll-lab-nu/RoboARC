# topogym-clock-assembling-01-i01-standard

| | |
|---|---|
| Benchmark | TopoGym |
| Mode | standard (`eai-standard/2.3`) |
| Twin | `topogym-clock-assembling-01-i01-privileged` |
| Simulator / robot | `topogym` / `aloha` |
| Family / instance | `clock_assembling` / seed 0 |
| Oracle | `none` — No reference solution (no solution/). The frozen instance was validated by the build: it builds in a fresh process (seed < 2^31) and holds still under hold checks in a fresh container, and a verifier replay grades without a grader error. No graded positive yet. Human review per family (protocol §2.7). |
| Agent time budget | 21600 s; tokens: protocol default (PROTOCOL §5) |
| Base image | `labv18b/rcb-topogym:0.2.16-dev-cu128` |

## Task
Standard-mode twin of `topogym-clock-assembling-01-i01-privileged` (same frozen instance): the agent gets only the `eai-standard/2.3` client
(docs/rcb/STANDARD_MODE_2_2.md); the simulator runs in the `sim` sidecar (`environment/docker-compose.yaml`, GPU on the
sidecar), which owns the scene and records the episode as a (T, 14) trajectory; its collect hook
(`environment/sim/finalize.sh`) replays the trajectory once in a fresh process for success, rendering the video during
that replay. The verifier runs in a separate sandbox (`tests/Dockerfile`) over the sidecar's artifacts. Compare against
the privileged task to measure the cost of losing resets, privileged state and the tooling's primitives.

## Difficulty
Same scene, instance and success rule as `topogym-clock-assembling-01-i01-privileged` (see its Difficulty). In standard mode the agent loses resets, privileged state and planning primitives: it sees camera images and proprioception only, acts through the client, and gets one episode, so it has to estimate the scene from images and recover from its own mistakes without starting over.

## Reference solution
See `solution/`.

## Verification
See `tests/grade.py`.

## Details

### Goal
Assemble the watch movement's gear train: the four wheels on their pivots in the main plate, the train bridge over them on its pillars with every upper pivot in its hole, the two bridge screws tight; then wind the barrel through its ratchet, let go and watch the fourth wheel run.

### Success requirements

- Seat the barrel, the center wheel, the third wheel and the fourth wheel on their lower pivots in the main plate (each wheel within 1 mm of its seat); a wheel seats once its lower pivot is down in its hole, within 0.6 mm of the hole's axis and tilted less than 3 degrees. Each wheel's pinion must mesh with the previous wheel's teeth (barrel, center, third, fourth); a wheel lowered with its pinion's teeth tip to tip on the other wheel's teeth stops until it is turned a little.
- Every upper pivot (the barrel: its arbor) inside its hole in the bridge, at least 1 mm deep and within the hole's radius of its axis.
- The bridge seated on its pillars: within 0.5 mm of its seat height, 0.5 mm of its seat position and tilted less than 0.5 degrees.
- Both screws in the bridge's screw holes and turned clockwise by 2 full turns with their heads on the bridge (no thread is simulated: the turns are counted while a head sits on the bridge); with both tight and the bridge seated, the bridge is fixed.
- After the bridge is fixed, wind the barrel: turn its square (the ratchet under it turns with it) counterclockwise seen from above; the click lets it turn that way only and holds it one ratchet tooth (18 degrees) behind the furthest point reached. Wind it at least 120 degrees (as held by the click), then stop winding: the mainspring drives the barrel and the train. Half a second after the last winding the rules time a 5 s run: the fourth wheel must turn at least 5 times in it, spend at most 0.2 s slower than a tenth of its free-run speed (the speed the mainspring's present windup gives the train against its own damping and friction), and turn 450 times as much as the barrel in the opposite direction, within 2 %. Winding again restarts the run.
- Holding parts is allowed; the probe and the hands may stay anywhere.
- Keep all final conditions satisfied together for at least 0.5 s of simulation time. If any final condition stops holding, the confirmation interval starts again. A brief qualifying pose is not enough; let the simulation advance through the full interval.

## Task failure conditions

- A task piece falling more than 100 mm below the tabletop ends the episode in failure. Finger contact with a held movable piece may penetrate at most 2 mm; holding requires both fingers of the same hand to contact the same piece, or an active assisted grasp by that hand. All other task contacts, including piece-piece, table, arm and gripper-housing contacts, may penetrate at most 1 mm. Exceeding either limit ends the episode in failure.

## Grader details (in addition to the requirements above)

- Contact penetration as this benchmark grades it (it replaces the "at most 1 mm / 2 mm" wording above): penetration is recorded, it never ends the episode by itself. A success with deep interpenetration (a task contact deeper than 3 mm, or deeper than 2 mm for at least 0.25 s: 100 consecutive physics ticks; for a finger contact with a held piece 4 mm, or 2 mm for 0.25 s) is not credited automatically but reviewed, with the recorded evidence and frames; a clean success is credited automatically. A task contact deeper than 10 mm is a simulator breakdown (below).
- Contact penetration is read from the contacts at the end of every physics tick (2.5 ms; 10 ticks per control step). "Held" is decided anew at every tick: a zero-force touch counts, and if one finger loses contact (with no active assisted grasp), the other finger is back at the non-held limits in the same tick.
- Robot safety (this benchmark's tooling): no gripper target goes lower than 15 mm below the tabletop; a commanded target that would bring any part of a gripper lower is stopped there, on the commanded path. The fingertips reach about 16 mm beyond the tool frame (TCP) along its approach axis, open or closed. Arm drive torques are capped (the effort limits in robot_spec), so an arm pushing against an obstacle stalls at that torque. An arm with any part more than 5 mm deep in a fixed object (a fixture, the table) is stopped where it is: its joint targets become its measured joints for that control step, so it stops pushing (the safety readings report these stops as static_guard).
- Forcing is recorded, not judged: an arm drive (not a gripper drive) at 95% of its torque cap or more while that arm's gripper, or a part it holds, presses on anything else with 20 N or more in total is forcing; 0.5 s of it on every physics tick is a forcing event in the evidence (`meta["safety"]`), shown to the reviewer when a success is reviewed. Forcing does not end the episode and does not change the grade by itself.
- Simulator breakdown (a non-finite state or a task contact deeper than 10 mm): the episode ends as an infrastructure error, not your failure, unless an arm was forcing (even for one tick) within the 0.25 s before it: then it is your failure (forced_breakdown: what broke and what forced it).
- The rules (seating, the tooth-on-tooth gates, meshing, the screws' turns, the bridge's fixing, the click and the run test's timing) are applied after every physics tick (2.5 ms); the success conditions are checked at the end of every tick. The mainspring is a spring between the barrel's arbor and drum; the click is a limit on the arbor's turn that the rules move up as it is wound.
- Grasp regions (assisted grasping, see Grippers above; one sphere per graspable part): barrel: radius 65.8 mm; center: radius 42 mm; third: radius 33.8 mm; fourth: radius 33.8 mm; bridge: radius 120.6 mm; screw1: radius 20.9 mm; screw2: radius 20.9 mm; probe: radius 56.1 mm; barrel_arbor: radius 30.6 mm. Both finger contacts must lie inside the part's sphere; a part touched elsewhere is grasped physically.

| | |
|---|---|
| mode | standard (RGB, optional depth, proprioception, calibrated cameras, no reset) |
| action space | `eai-standard/2.3`: `submit` chunks of `joint_position` (14, streamed at the 40 Hz control rate with the agent's `dt`) or `ee_pose` (16, IK per row); `observe` frames or a watch window; binary RCB1 responses |
| oracle | none yet (`oracle = "none"`, no `solution/`) |
| verification | `success` = live success ∧ replay success: the live episode, then one fresh-process replay in the sidecar's collect hook, after the live service has exited; the state hashes are reported, not required (GPU physics is not bit-deterministic; the CPU rigid families' is, on one host); `/logs/verifier/replay.mp4` is rendered during that replay |
| crash recovery | `eai-standard/2.3` §13: journal + per-call checkpoints, `restart: on-failure:6`, resume class `approx` |
