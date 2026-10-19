# kyrre: the build history (shipped 2026-10-19 with native cylinders; the reference and its tools: scripts/ikea_mujoco/oracles/kyrre/)

## Native-cylinder exception (2026-10-18 candidate; approved by the owner 2026-10-19, image 0.3.2)
**The change (kyrre only).** `images/ikea-mujoco/scenes/kyrre/scene.py` keeps MuJoCo's native cylinders: its `load()`
builds the scene with the 32-facet prism rule off (`NATIVE_CYLINDERS = True`; the rule is restored after the build). The
shared runtime and the other eleven scenes are byte-identical; the welded-part contact switch-off (0.3.1) is on.
assets.json re-pins kyrre only (digest a485e4d945b7). Birth unchanged: ecd23f8f42cd96e0 = bindings.json. The 11 shipped
references end and settle on their known hashes in the candidate image (host B local build sha256:d444a931171a;
the work notes).

**The reference** (`scripts/ikea_mujoco/oracles/kyrre/kyrre_ref.npz`, sha256 0808ce78..., 733,558 rows, 1833.9 s simulated; `scripts/ikea_mujoco/oracles/kyrre/rebuild.sh` rebuilds it
bit for bit):
1. the demo (every arm row already within the Panda limits) retimed for the Franka Hand slew (182.14 units/s, branch
   dev/qineng-ikea-hand-slew): `std_reference.py make --method clamp --despike 0 --grip wait` (73,806 wait ticks);
   alone it latches 7 of 9 welds: the hand's slower release nudges screw 3-1 5.9 mm along its axis on the staging spot
   and the second grasp misses it (`--grip inplace` is worse: 2 welds);
2. `oracles/kyrre/patch_grasp.py`: screw 3-1's second grasp re-aimed (rows 501,000-506,500): the right TCP keeps the demo's pose
   relative to the screw (damped-least-squares IK on the target configuration, 6-D, blended in and out); then all 9
   welds latch (screw 3-2 needs nothing);
3. the hex key's set-down moved 12 mm along its rest (-y) and 3 mm up (rows 729,800-733,400): the demo puts the key
   back on its knife-edge pose (one contact, the arm hanging in the slot), where it chatters at 0.0993 rad/s from the
   birth on, in both the native and the prism scene (so it never meets the 0.05 rad/s rest rule); 12 mm along, it lies
   on a flat and rests (`oracles/kyrre/keyrest.py`). Moving it off the rest instead (table spots) makes the forearm touch leg 2.
Checks: 0 rows over the arm limits, 0 gripper steps over the slew, 0 slewed ticks; final_check success (goal, released,
at rest, no breakdown, now and after the 1.0 s hold); the stool's rigidity 0.0002 deg (bar 1 deg), hold 12.4 s.

**Determinism (fresh replays, end 0a9f1a9e6d77 / settled e242979c6712 on every host).** AMD Ryzen 9 7950X (host B Docker,
candidate image, with the hand-slew rcb_ikea and with dev's: same), Intel i9-14900HX (AVX2, laptop venv with the
EPA-patched wheel), Intel Xeon Gold 6338 (AVX-512, Quest SIF 0.3.0 + the candidate scene dir).

**Timing.** 2740 s on the i9-14900HX (one core under the shared lock: 63 % of the 4320 s bar, 25 % of the 10800 s
budget); 2694 s on host B's 7950X (load 5-10); Harbor's verifier replays 2746 s on host B; 8283 s on a Quest Xeon Gold 6230R core and 7953 s on a Xeon Gold 6338
core (74 % of the budget, over the 40 % bar on that slow class, as gladom / variera / stenerik / uppsta there).

**Stress (Quest, v2.13b observer, 203 episodes = 7 strategies x 29, the same seeds and checkpoints for an A/B).**
- native cylinders: CLEAN 132, VIOLENT 60, BREAKDOWN 11 (5.4 %); non-jerk 2 / 174 = 1.15 % (shipped set: 1.75 %);
  prefix_jerk 9 / 29. All 11 reproduced from the birth on Quest Intel (the reference prefix on that CPU matched the
  Zen 4 checkpoint hashes every time).
- the same episodes with the prism rule on (the control; the reference's states loaded into the prism scene):
  BREAKDOWN 12 (5.9 %); non-jerk 4 / 174 = 2.3 %; prefix_jerk 8 / 29.
- The classes are the same in both and none involves a cylinder geom: uncaused speed jumps of the idle press-in spring
  jaws (`jaw_s2_press_in_*`, 6 native / 7 control), the seat jumping under a jerk (1 / 2), energy at robot joint limits
  (right_joint5, left_finger_joint2; 2 / 3), a forearm pressing a leg (1 / 1), integrator residual (1 / 1). REVIEW.md
  item 2's cylinder-caused breakdowns (cap-rim normal flips against the robot's convex hulls) do not show here: the
  prism rule converts 12 geoms in kyrre, the six screws' `head_base` discs and `thr_core` cores (the hands carry the
  screws by them), and no BREAKDOWN in either run names a screw.

**Harbor (host B, Harbor 0.23.0, harbor_run_safe.sh, local images 0.3.1-kyrre-cyl / sim 0.4.1-kyrre-cyl).** Privileged
oracle 1 (deterministic, instance_exact 1, video rendered, replay 2746 s) / nop 0 (missing_trajectory); standard oracle
1 (live settled success, live sha = replay sha, 184 calls, service 3073 s) / nop 0. The task dirs on the branch render
kyrre out of WIP with a per-task image override (bindings.json `image` / `sim_image`, mk_standard_tasks.py reads
`sim_image`); evidence.json is not touched (no row cap yet).


## Status 2026-10-18 (V3, kyrre agent): blocked
Physics now: the prism rule, the arm clamp, MuJoCo 3.14.0 with the EPA patch, and the **welded-part contact switch-off**
(new, shared: `ikea_runtime.RetentionMechanism`, from a scene's `sw_cutover.json`; in kyrre a welded leg no longer
collides with the seat, a seated screw and its insert's thread with anything). The nine shipped references end and
settle on the same hashes with and without it (it is off in their scenes): the work notes.

**What passes.** A guided closed-loop build (`ky_guide.py` + `ky_search.py`, a port of oracles/uppsta's controller):
the demo as the phase guide, its run without the prism rule as the guide run (`track_np.py`, `mkguide.py`), per arm
approach / carry loops on the parts' guide poses, segment by segment with a parameter search and a deviation gate, and a
**finish** where the guide run's screw weld has latched and ours has not (`ky_lib.ratchet`: the key turned about the
true screw axis inside the free sector of the hand's turn, the arm included, `ky_lib.arm_sector`). It gets through
leg 1, screws 1-1 and 1-2 (both finished by the ratchet: the demo's strokes leave 1-2 at a 0.75 mm gap, its key turns it
by friction on the head top) and leg 2, and places screw 2-1 upright in its hole (demo row 265,488; 285,137 executed
rows; 54 of 148 segments).

**Where it stops (screw 2-1's drive, segment 55).**
- The demo drives 2-1 with the left hand's key fully in the socket. Our key enters it about 8 degrees off the flats and
  wedges; when the demo lifts the key for its return stroke the key stays in the socket (the fingers slide on it) and
  turns the screw back: 0 net advance over the whole 74k-row window.
- Lining the key up on the true socket needs the key's guide pose turned about the screw (the hex phase), which moves
  a lever-grip hand up to 45 mm along its arc; the carry loop does not follow that, and the ratchet finish from there
  pushes the stool 70-490 mm in every candidate (the forearm or the key against a leg).
- Screws 2-2, 3-1 and 3-2 follow the same pattern (carry upright, drop from 28 mm, finger twirls, lever strokes) and are
  not reached yet.

**Timing (measured).** The 285,137-row prefix replays in 1,814 s on a quiet a cloud host EPYC 9354 (6.4 ms per row). At that
rate the whole build (about 710k rows: the demo's 660k + the finishes) is about 4,500 s on that Zen 4 CPU, about 105 %
of the 4,320 s bar (an i9-14900HX is about 1.2x faster: about 3,800 s), and about 7,700 s on a Quest Xeon core. The
demo's own pace is the floor: its strokes cannot go faster (a key turned above about 1 rad/s cams out of the socket),
and the rows where no hand touches anything (where a tempo profile could skip rows safely) are under 1 % of the demo.

**What would unblock it.** Either a scene-side change (a shorter task, e.g. fewer screws per episode, or a key the hand
can drive without strokes) or a dedicated hex-key drive per screw (engage on the flats, strokes inside the measured free
sector, the forearm kept off the legs) for the four remaining screws; the latter is a multi-day build and the timing
would still sit at about the bar on Zen 4.

Files (this directory): `ky_lib.py` (rig, key measurements, KeyServo, ratchet, arm_sector, hooks), `ky_guide.py` /
`ky_search.py` / `ky_finalize.py` (the guided build), `mkguide.py` / `track_np.py` / `scan_demo.py` (the guide),
`build2.py` + `cycles.json` (the earlier demo-strokes builder), `t_seg.py` / `t_seg2.py` / `teleport.py` (probes).
Runs: a cloud host `/workspace/qn_ikea_v3/kyrre/` (`ref_g2/` the guide, `runs/g0/seg*.pkl` the kept segments, `runs/g0/log*`).

## Earlier findings (2026-10-16)

There is no reference: `ikea-kyrre-i00` stays WIP. Two independent blockers, each enough on its own:
1. **The author's demo fails under the final physics** (4 of 9 welds; screw 2-1 is lost at its hole).
2. **Timing: one replay of the demo takes 12,316 s on a dedicated Quest core, 85.5 % of the 14,400 s budget** (the
   ceiling is 30 %). A passing reference would need about 3x fewer rows than the demo, and every speed-up of the demo
   we tried breaks it at the first or second screw.

Only robot actions were tried; the scene, `rcb_ikea.py`, `grade.py`, `success.py` and the physics are untouched.
Status 2026-10-16 evening CDT.

## The demo under the final physics (rung 1)
- `solution/trajectory.npz` of the WIP privileged dir (`kyrre_vlim` as sent): 659,752 rows (1649.4 s simulated), sha256
  `465163457683ff8def208ecd50ada1f5f5cd0af4511b887b29c3b34cb71fad27`.
- Speed: 0 rows over the limit (row to row, the first row against the keyframe ctrl: step 0); largest step = the limit
  (ratio 1.00000000000004); `slewed_ticks` 0. So rung 2 (std_reference retiming) has nothing to do.
- Birth: `ecd23f8f42cd96e0…` = bindings.json. 12 cylinders (the six screws' head discs and thread cores) are prisms.
  sw_cutover.json `{"from_op": 1}`: the seated-screw switch-off is active from the first operation; SW_FROM_OP /
  SW_FROM_T unset in every run (each probe records its environment).
- Verdict: **fail**. Welds s1-leg-1, s1-screw-1-1, s1-screw-1-2, s1-leg-2 engage; s1-screw-2-1 never does (the screw
  ends 47.7 mm off its hole, 79.7° tilted), so nothing after it. End also not at rest (the hex key turns at 0.099 rad/s).
- End / settled sha `4eee1ebebb30c049…` / `ab284bcaf688ba5d…`, the same in 2 processes on a cloud host EPYC 9354 (Zen 4), 1 more
  there (the profile run) and 4 on Quest Xeon 6230R (Intel): the failure is deterministic across CPU classes.
- Where screw 2-1 is lost (`profile.py` track, `track_part.py`): picked at row ~231k, held at its hole rows 237k-255k
  while the hand works it, released, and at row ~262k it drops onto the seat (-18, 38, 126 mm); the hex key then drives
  an empty hole (rows 270k-339k).
- Diagnostic (NOT the benchmark physics): `diag_noprism.py` replays the demo with the prism rule switched off, to see
  whether the rule causes the loss. Without the rule all 9 welds engage (screw 2-1 by row 360k) and success.py holds at the end (goal True, released), but the end is still not at rest, so even there the terminal rules fail (9121 s on Quest). So the loss of screw 2-1 comes from the prism rule (the demo was recorded without it): the screw's thread core and head disc are prisms now, and the open-loop demo's insertion no longer catches.

## Timing
| host | one replay | % of 14,400 s |
|---|---|---|
| Quest Xeon 6230R, dedicated Slurm cores (SIF 0.3.0; 2 jobs x 2 processes) | 12,316-13,974 s | 85.5-97.0 % |
| a cloud host EPYC 9354 (box load 30 → 90 during the run) | 8,562 s | 59.5 % |
| host C 7950X standard live episode (the standard agent's dry run) | 7,045 s | 48.9 % |

The cost per row grows along the episode: 224 contacts at the birth (the seat, legs and screws resting on the table:
60 + 3 x 40 + ...), 393 by row 640k (welded legs keep their box contacts with the seat); the profile's wall per
400-row block rises from 1.3 s to 15 s. Rows of the second half cost about 3x those of the first.

To reach the 30 % ceiling on Quest (4,320 s at the demo's mean 18.7 ms/row), a reference needs <= ~230k rows, about 2.9x
fewer than the demo; the late rows are the expensive ones, so a uniform cut has to be larger still. Where the demo's
time goes: each screw takes ~90-120k rows (pick from the tray ~35k, then ~11 hex-key ratchet strokes over ~55k rows,
140 s, mostly well below the joint-speed limits); six screws are ~600k of the 660k rows.

## Speed-ups tried (all on the final physics, the clamp in; rows recorded as executed)
- `warp_ref.py`: per 400-row segment, the demo resampled at k = 4 / 3 / 2 / 1.5 / 1 demo rows per tick in forked
  children, the fastest whose state matches the demo's own state at the segment end (parts 1 mm / 1°, held parts in
  the hand frame, screws by their axis, welds) taken, with backtracking. It gets 2.3x through the first 40k demo rows
  (leg 1 and the screw pick), then no factor, k = 1 included, reproduces the demo: the screw's in-hand pose differs by
  0.7 mm / 1.4° after its pick, the hex key by 10 mm by row 39.6k. The demo's grasps depend on sub-millimetre state.
- `warp_ops.py`: per operation (cut a little after each demo weld), free blocks (no robot-part contact) at kF and
  contact blocks at kC, the phase gate = the operation's weld latched by the mechanism + untouched parts within 3 mm.
  Leg 1 passes at (4, 2). Screw 1-1: (4,2), (4,1.5), (3,1.5), (4,1) lose the screw or the hex key (55-91 mm off);
  (2,1), only the free blocks faster, misses the weld (screw 3.8 mm off); only (1,1) passes. Then screw 1-2 fails even
  at (1,1), the demo's own rows, because leg 1 was compressed: the open-loop demo does not tolerate any speed-up.
- Mix test (warp_ref's warped first 40k demo rows, 17,311 rows, + the demo's rest from row 40k; 637,063 rows): only
  leg 1 engages; screw 1-1 never does (15,762 s on the loaded a cloud host box). The segment-level match was not enough either.

## What a passing reference needs
A new closed-loop controller for most of the task, not a retimed demo: picking each screw, seating it in its hole and
driving it with the hex key under true-state feedback (key-in-socket alignment, strokes near the joint-speed limit,
feedback on the screw's advance), about 25-30 s per screw instead of the demo's ~140 s, plus a fix for screw 2-1. We
did not attempt it tonight. It is a burvik-scale build, and each test of a late operation replays hours of scene.

## What the scene side could change (for the owner; not done here)
- Per-row cost: the seated-screw switch-off (sw_cutover) already removes the seated screws' contacts; the welded legs'
  contacts with the seat (and the resting parts' many box contacts with the table) remain and dominate. A similar
  switch-off for welded legs would cut the late rows' cost.
- Task size: six hex-key screws plus three legs in one episode is about 3x the time budget at the demo's pace.
- A demo recorded under the prism rule (the rule applies to the screws' cylinders).

## SW_FROM_OP / SW_FROM_T
`kyrre_mechanism.sw_cutover()` reads these env vars before sw_cutover.json. Today nothing sets them:
- privileged grader: the verifier runs in its own container (`environment_mode = "separate"`), whose environment is
  the image's ENV (no SW_*) plus nothing from task.toml or test.sh; the agent's container environment never reaches it;
- standard service: the sidecar's compose `environment:` lists only RCB_* / MUJOCO_GL / PYOPENGL_PLATFORM /
  OMP_NUM_THREADS / NVIDIA_DRIVER_CAPABILITIES; the agent cannot set the sidecar's environment;
- our runs: every probe records SW_FROM_OP / SW_FROM_T = None.
The latent risk: `grade.py` passes its own `os.environ` to the replay processes, and `instance_exact` pins
sw_cutover.json but not the environment, so a verifier or sidecar environment that ever set SW_FROM_OP would change the
physics with every check still green. Options: the scene drops the env override, or grade.py / the service scrub
SW_FROM_* before replaying (a grader change, the integration agent's call). In privileged mode an agent can set them
in its own container, which changes only its own local runs.

## Files
- `profile.py`: a replay with per-block wall time, contacts, robot-part contact, part speeds, welds, and the full
  qpos / qvel / eq_active track at every block end (`--block 400`).
- `warp_ref.py`, `warp_ops.py`: the two time-warp searches above.
- `track_part.py`: one part's pose along a track. `diag_noprism.py`: the diagnostic replay without the prism rule.
- The fresh-replay probe is `scripts/ikea_mujoco/oracles/stegesund/probe.py`.
- Run dirs: Quest `ikea_b2_20261016/sk/runs/` (kyrre_demo*, diag), a cloud host `/workspace/qn_ikea_b2/sk/runs/` (kyrre_demo,
  prof, warp_test*, wops1, kyrre_mix1).
