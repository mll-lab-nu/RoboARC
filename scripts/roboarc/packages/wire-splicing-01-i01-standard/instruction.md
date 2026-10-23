# RoboARC (simulator package `topogym`) · wire_splicing — STANDARD MODE (robot as a service, eai-standard/2.4)

You control a robot (**two 6-DoF ALOHA (ViperX) arms with parallel-jaw grippers on a fixed base**) **as if it were real hardware**: the simulator runs in a service you cannot open. You get one
client and one episode. **Task:** Splice the red and the blue wire: bend the red wire's bare end into an eye round the yellow rod of the bending post, thread the blue wire's bare end through the eye, wind the blue wire's bare tail round the red wire's doubled end at least 1.5 turns, at least 1 full turn of them round both strands of the doubled end together, fold the red wire's leftover end back over the coil, then let go.

## Success requirements

- The red wire's bare end is turned back on itself by more than 150 degrees into an eye that is closed at its neck: the two legs touch where they first meet below the bend.
- The blue wire passes through the closed eye exactly once (its linking number with the eye is +1 or -1).
- The blue wire's bare tail winds at least 1.5 turns round the red wire's doubled section. The turns are counted about the doubled section's axis: the line fitted to the red wire's stem (its bare part from the strip to the neck) and the return-leg points that lie beside it, within 8.9 mm (2.5 core diameters; the core is 3.56 mm thick) of the stem's own line. The counted coil is the longest run of the blue wire's bare points within 8.9 mm of that axis and between the neck and the red wire's strip, each widened by one core diameter.
- At least 1 full turn of the coil goes round both strands of the red wire's doubled end together, not only round the line between them. The strands: the stem side, from the strip to the eye's apex (its point nearest the eye end of the axis), and the return side, from the apex to the return leg's point farthest toward the strip (a leftover end folded back from there is not one of them). For each point of the counted coil, each strand's point at the same station along the axis is taken, where the strand crosses the plane through the coil point perpendicular to the axis, and the coil point's angle about it in that plane; along the coil these angles accumulate into turns about each strand, a step counting only where both strands have their points. Some contiguous part of the coil must wind at least 1 turn about each strand, both the same way as the coil winds about the axis: a loop round one strand only, or one passing between the strands, does not count.
- The coil is tight: its points lie on average at most 7.1 mm (2 core diameters) from the axis.
- The red wire's leftover end is folded back: the direction from its fourth-last to its last point points back toward the eye (its dot product with the axis, which runs from the eye toward the strip, is at most -0.5), and its tip is at least 4.3 mm (1.2 core diameters) from the axis.
- Let go: no hand holds a wire. The blue wire's turns may drop by at most 0.25 and the threading may not change while the conditions are held, and the splice must be still: every bare point's speed relative to the bare points' mean motion at most 5 mm/s.
- Keep all final conditions satisfied together for at least 2 s of simulation time. If any final condition stops holding, the confirmation interval starts again. A brief qualifying pose is not enough; let the simulation advance through the full interval.

## Task failure conditions

- A task piece falling more than 100 mm below the tabletop ends the episode in failure. Finger contact with a held movable piece may penetrate at most 2 mm; holding requires both fingers of the same hand to contact the same piece, or an active assisted grasp by that hand. All other task contacts, including piece-piece, table, arm and gripper-housing contacts, may penetrate at most 1 mm. Exceeding either limit ends the episode in failure.

## Grader details (in addition to the requirements above)

- Contact penetration as this benchmark grades it: penetration is recorded, it never ends the episode by itself. A success with deep interpenetration (a task contact deeper than 3 mm, or deeper than 2 mm for at least 0.25 s: 100 consecutive physics ticks; for a finger contact with a held piece 4 mm, or 2 mm for 0.25 s) is not credited automatically but reviewed, with the recorded evidence and frames; a clean success is credited automatically. A task contact deeper than 10 mm is a simulator breakdown (below).
- Contact penetration is read from the contacts at the end of every physics tick (2.5 ms; 10 ticks per control step). "Held" is decided anew at every tick: a zero-force touch counts, and if one finger loses contact (with no active assisted grasp), the other finger is back at the non-held limits in the same tick.
- Robot safety (this benchmark's tooling): no gripper target goes lower than 15 mm below the tabletop; a commanded target that would bring any part of a gripper lower is stopped there, on the commanded path. The fingertips reach about 16 mm beyond the tool frame (TCP) along its approach axis, open or closed. Arm drive torques are capped (the effort limits in robot_spec), so an arm pushing against an obstacle stalls at that torque. An arm with any part more than 5 mm deep in a fixed object (a fixture, the table) is stopped where it is: its joint targets become its measured joints for that control step, so it stops pushing (the safety readings report these stops as static_guard).
- Forcing is recorded, not judged: an arm drive (not a gripper drive) at 95% of its torque cap or more while that arm's gripper, or a part it holds, presses on anything else with 20 N or more in total is forcing; 0.5 s of it on every physics tick is a forcing event in the evidence (`meta["safety"]`), shown to the reviewer when a success is reviewed. Forcing does not end the episode and does not change the grade by itself.
- Simulator breakdown (a non-finite state or a task contact deeper than 10 mm): the episode ends as an infrastructure error, not your failure, unless an arm was forcing (even for one tick) within the 0.25 s before it: then it is your failure (forced_breakdown: what broke and what forced it).
- Grasp regions (assisted grasping, see Grippers above; one or more spheres per graspable part): wire_a: 63 spheres of radii 26.1 mm, 5.1 mm; wire_b: 97 spheres of radii 25.1 mm, 5.5 mm. Both finger contacts must lie inside one of the part's spheres; a part touched elsewhere is grasped physically.

## Rules (real-robot rules)
- **No reset.** The episode ends on success (all final conditions satisfied for
  0.5 s), when your time is up, on a simulator error, or when the task's own failure predicate fires (then `meta["reason"]` is `task_failed`). Knocked-over objects stay
  knocked over.
- **No privileged state.** You see what the cameras see (RGB, and depth when you ask for it) plus the robot's own state:
  joint positions, velocities and drive efforts, gripper openings, end-effector poses. Nothing tells you where the
  objects are (no object poses or positions). The success implementation is not exposed, and of the simulator's internal
  measurements only the safety readings of `meta["safety"]` are (listed under Action types, below); the task
  requirements above still apply in full.
- **Your only budget is the time limit (`## Time limit`)**; there is no step budget, but the
  episode ends after 200000 control steps, watches included (83 min of simulated time; a technical cap,
  `spec["max_control_steps"]`: `meta["reason"]` is `step_guard`, and the episode is graded as it stands). Actions and simulated time are reported as cost.
  This service also ends the episode itself (`meta["reason"]` is `timeout`; it is graded as it stands) when its own
  count of that limit, `meta["time_left_s"]`, reaches 0. That count starts at your first request to the service, so it
  never reaches 0 before your time is up.
- There is nothing to hand in: the service records your episode as the trajectory and grades it. Leave your script
  at `/app/output/gen.py` for the record.
- **Nothing is put into your context automatically.** Images and videos come back as arrays/bytes in a Python object;
  what you look at and where you store it is your code's decision.

## Startup and long-running commands

- Scene initialization and kernel compilation can take minutes. Actual time depends on the task, hardware, and load.
  Quiet logs during initialization do not by themselves mean failure. Use `python -u` for unbuffered script output.
- A command tool's wait/yield interval (for example, `yield_time_ms=30000`) is not the command's lifetime.
  When a command is still running, retain its process/session identifier and continue waiting for that same command.
  With Codex's `exec_command`, expose the complete result, including `session_id` and `exit_code`, rather than only
  `output` (for example, `text(r)` instead of only `text(r.output)`). Continue it with `write_stdin` using the returned
  `session_id` and no input. An outer JavaScript wrapper saying `Script completed` does not prove Python has exited.
  Use the corresponding wait/status operation if another command tool is provided.
- Do not rerun a command just because its first wait returned partial logs. Confirm the original process has exited
  before starting a replacement; if cancellation is necessary, stop only your own identified process and confirm exit.
- Keep the task's simulator and sensor configuration. Waiting does not extend the actual agent wall-clock budget.
- Connect with `Robot()` and let its retry loop wait for the existing service (default connection budget: 3600 seconds,
  still subject to the actual agent deadline). Do not try to start or rebuild a simulator yourself.
- Reconnecting a client accesses the same episode; it does not reset the world. While an action request is still
  running, wait for its original command rather than resubmitting the same actions, which could execute them twice.

{{TIME_PER_CALL}}
## Client (`/opt/tools/rcb_client.py`, protocol `eai-standard/2.4`)
```python
from rcb_client import Robot
r = Robot()                                   # connects to the existing service; retries until ready (default timeout 3600 s)
spec = r.spec()                               # actions, cameras (intrinsics, mounting), robot_spec, world, kinematics, budget

obs = r.get_obs()                             # current frames from 4 cameras: obs.last[cam] -> uint8 (480,848,3)
obs.meta["proprio"]                           # {"layout": [72 names], "now": [...]}: groups in spec["proprio_groups"]
obs.meta["cameras"]                           # per camera: intrinsic_cv, extrinsic_cv, cam2world_gl of the LAST frame
obs = r.get_obs(depth=True)                   # + obs.last_depth[cam]: float32 (480,848) z-depth in metres, 0 = no reading
obs = r.get_obs(seconds=2.0, video="mp4")     # hold the last command and watch for 2 s (max 30): obs.video[cam] = mp4 bytes

res = r.submit("joint_position", A, dt=0.1)   # A: (T, 14) rows dt seconds apart; streamed at 40 Hz with linear interpolation
res = r.submit("ee_pose", P)                  # P: (T, 16) tool-frame targets; inverse kinematics per row
res.meta["done"], res.meta["reason"], res.meta["plan"], res.meta["clipped"], res.meta["control_steps_chunk"]
res.video["cam_high"]                          # mp4 bytes of the execution (video="mp4" | "frames" | "both" | "none")
res.frames["cam_high"]                         # uint8 (N,480,848,3) at video_fps if requested
res.last["cam_right_wrist"]; res.proprio              # last frame per camera; float32 (N, 72) proprio, one row per frame
res.frame_step, res.frame_time_s              # per frame: control steps since the episode start, simulated time (s)
res.extrinsic_cv["cam_right_wrist"]                    # (N,3,4) world -> camera at every frame
res = r.submit("joint_position", A, dt=0.1, depth="frames")   # + res.depth[cam]: (N,480,848) z-depth per frame (1 mm steps)
res.group("tcp_pose"); res.group("joint_effort", now=True)   # proprio columns by group name
r.fk(q)["poses"]["left_tcp"]                   # forward kinematics of the 12 arm joints (spec["kinematics"]); moves nothing
r.ik({"left_tcp": pose}, seed=q)             # inverse kinematics -> {"ok", "q", "error"}; moves nothing
res.save("/app/runs/003")                     # meta.json, <cam>.mp4, last/, frames/, proprio.npy, depth, frame_meta.npz
```
Cameras: `cam_high` (static); `cam_low` (static); `cam_left_wrist` (on the robot link `left/gripper_base`; it moves, so use the pose that comes with each frame); `cam_right_wrist` (on the robot link `right/gripper_base`; it moves, so use the pose that comes with each frame). Resolution 848×480. `spec["cameras"][cam]` gives the
intrinsics and the mounting: a static pose, or the parent link and the fixed offset (world_T_cam = FK(link) · `link_T_cam_cv`).
A point (X, Y, Z) in a camera's OpenCV frame (x right, y down, z forward) appears at u = fx·X/Z + cx, v = fy·Y/Z + cy;
depth is that Z. `spec["robot_spec"]` lists the joints with their position, velocity and effort limits, the grippers,
the PD gains and the tool frames; `spec["world"]` the frame conventions (the world frame: origin at the robot's root link on its mounting, +z up; no positions of the scene are given).

### Action types
- `joint_position` (14): `left/waist`, `left/shoulder`, `left/elbow`, `left/forearm_roll`, `left/wrist_angle`, `left/wrist_rotate`, `left_native_gripper`, `right/waist`, `right/shoulder`, `right/elbow`, `right/forearm_roll`, `right/wrist_angle`, `right/wrist_rotate`, `right_native_gripper`. arm `left`: 6 joint targets (rad), then `left_native_gripper` = native drive target (m, limits 0.002 .. 0.037; open/closed targets in robot_spec); arm `right`: 6 joint targets (rad), then `right_native_gripper` = native drive target (m, limits 0.002 .. 0.037; open/closed targets in robot_spec). Absolute targets tracked by PD controllers.
  Rows are `dt` apart (`1/40 ≤ dt ≤ 0.5` s, a multiple of the control period); a chunk may cover at most 10 s;
  out-of-limit values are clipped and listed in `meta["clipped"]` (limits in the spec). The gripper switches at the
  row. Use `obs.meta["proprio"]["now"][:14]` as the starting row to move smoothly. An arm joint target moves at most
  0.04 rad per control step (1.6 rad/s, the same speed as `ee_pose` rows): a row that asks for more takes
  the control steps it needs, and `meta["clipped"]` says so.
  The proprio group `gripper_opening` (`res.group("gripper_opening")`) is the measured gap between each gripper's finger
  pads in metres (0 = the pads touch); the gripper entries of `joint_pos` are in the unit of the gripper command above.
- `ee_pose` (16): per arm (in the order ['left', 'right']) `x y z qw qx qy qz gripper`; world frame (origin at the robot's root link on its mounting, +z up; the frame of the proprio TCP pose and the camera extrinsics); quaternions wxyz. Each row is solved by inverse kinematics (no collision checking) and driven to in joint
  space; an arm whose IK fails holds and is reported in `meta["plan"]`. The pose is each gripper's **tool frame** (TCP;
  its approach and finger-closing axes are in `spec["robot_spec"]["tcp"]`), the same convention as the proprio TCP
  poses, so an observed pose is a valid target. At most 50 rows per submit. A row whose IK solution lies more than
  0.5 rad from the current joints on any joint is refused the same way as a failed IK (for a small pose change that is
  another IK branch, which would swing the arm): split long moves into rows a few cm and degrees apart.
- One call returns at most 2000 images (`spec["max_frames_per_call"]`: cameras x frames; a frame with
  `depth="frames"` counts 4). A request that would exceed it is answered 400 before anything moves: ask for
  fewer cameras, a lower `video_fps` or a shorter call. Rows that take longer than planned (the speed limit, `ee_pose`)
  stop adding frames at that number; the motion is executed in full and `meta["frames_skipped"]` counts them.
- Both action types: no gripper target goes lower than 15 mm below the tabletop; a target that would bring any part of
  a gripper lower is stopped there and listed in `meta["clipped"]`. Every response's `meta["safety"]` reports, since the
  previous call, forcing (the arms' pressing loads and forcing events), the deepest task contact, the longest time above
  its sustained limit and the review events (rigid families), the lowest rope / cord volume ratio (soft families) and the
  table-guard stops, against the limits above.

Grasping: assisted grasping (contact_weld_v1): a gripper whose command moves towards closed takes hold of a wire (wire_a, wire_b) once both fingers touch it inside its links' grasp regions, the tool frame within 15 cm of it and the penetration at most 2 mm, for 0.02 s: of each wire it touches, the touched link nearest the tool frame and every other link whose centre lies between the two pads then follow the gripper rigidly. Both fingers on the wires with every contact between the pads also holds, the same way (the red wire's two legs side by side, one pad on each; the eye's two sides with the blue wire beside them). Any command that opens the gripper (by more than 0.2 % of its travel) lets it go. A hold that breaks (the fingers open more than 15 %, a held link drifts 1 cm / 0.15 rad from its held pose, the hold has to push or pull with more than 60 N averaged over 20 ms, or a weld has fed more than 0.1 J of work into the wire) is lost, and the gripper must open before it can take hold again; that force is reached when, for example, a held wire is pulled hard against the other hand or a fixture. The jig is fixed; the grippers start empty and nearly closed: open a gripper, then close it on a wire to take hold.

Objects here: `wire_a` (red) and `wire_b` (blue), each fixed at its terminal, and `jig` (fixed: the bending post with its yellow rod). As everywhere in this mode, nothing reports their poses: what you know of them comes from the cameras.
