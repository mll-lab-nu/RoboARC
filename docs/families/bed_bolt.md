# bed_bolt on MuJoCo (a bed-frame corner bolt driven with an L-shaped hex key)

Jianwen, 2026-10-16. A new family with one experimental setting, `bed_bolt_crossbar` (pool experimental, no tier), on
MuJoCo only. Design record, sources and every measurement of the design: `runs/bed_bolt_design_20261012/` (README.md
sections 7-9, PROGRESS.md; not in git).

## The task

A small bed frame stands on the table (60 x 44 cm, four 2.2 cm boards 12 cm high on four round legs, slats on ledges). A
corner block sits in the far corner of the right end board; a blind hole (radius 5.4 mm, 86 mm deep) runs from its face
along +x into the end board, its mouth 13.5 cm above the table. A cross board 16 cm in front of the block face closes a
bay round the hole. In front of the frame the long bolt (shank 10 x 85 mm, head 16 x 10 mm, a hex socket 8.15 mm across
flats and 6 mm deep with a 1 mm countersink) lies in a V cradle; the L-shaped hex key (8 mm, long arm 200 mm, short arm
45 mm, both ends hex prisms with 1.5 mm chamfers) stands short arm down in its stand. Two ALOHA arms.

The robot carries the bolt to the hole and pushes its tip in until the thread catches, then turns it with the key until
the head seats on the block. The bay is too short for the key's long arm along the bolt: the short arm goes into the
socket and the long arm stands up as a lever, swung 60 degrees at a time (the key comes out, turns back and goes in
again).

- Geometry: `topogym/assets/geometry/bed_bolt.py` (the design's v2), three MJCF entities in the ALOHA mount frame:
  `bed_frame` (static: the frame with its bore, the cradle, the key's stand), `bolt` (the bolt and the thread carriage),
  `key`.
- Profile `bed_bolt_v2` = claws_v6, the values the design's sandbox ran the ALOHA with, and (2026-10-17, v2) the
  thread's stop plate keeping its own nearly hard contact (static_contact.keep for the bolt entity's one static body). The bolt's and key's contacts keep
  their own parameters (steel: 3 ms, oiled steel friction 0.1 on the head, socket and key).
- Assisted grasp: contact_weld_v1 on the whole bolt and the whole key, a hand acquiring with its TCP up to 15 cm from the
  part's centre (the key is 20 cm long).
- The threads are drawn (2026-10-17, visual only: group-1 meshes with no contacts or mass): the bolt's shank as a
  single-start right-hand thread of the 5.7 mm lead (crest at the shank's 5.0 mm, root 4.3 mm), the bore as the matching
  internal thread (crest 4.4 mm, root at the bore's 5.4 mm), half a turn from the bolt's crest when lined up, so the two
  interleave once caught; the dark hole mark sits at the bore's bottom. The contacts are still the plain shank and bore
  and the thread mechanism. Before the catch a bolt whose roll does not match the hole's thread angle shows its crests
  through the hole's in the first pitch (the drawn threads overlap there; accepted, 2026-10-17).
- Colours: the key is black-oxide steel, rgba (.22, .23, .25) (c114c59ca1, 2026-10-17, the user's request; it was
  yellow, (.80, .70, .35)). Colour only: the instance rebuilt on it has birth 5304a7d2b2771443, and two fresh strict
  replays of the reference end on 3da1814664852f27 (`runs/robotangle_mujoco/bed_bolt/20261017_metal_key/colour_check/`).
  Nyx shades the MJCF rgba as linear albedo, so the key renders a neutral mid grey: about (132, 133, 134) of 255 where
  it lies in its stand in cam_high, against (189, 181, 146) before, and darker than the bolt's steel (.55, .57, .60),
  about 167. Superseded by the look below (2026-10-19).
- The look (2026-10-19; the user approved items 1-8 of `runs/robotangle_mujoco/bed_bolt/20261018_variants/
  VISUAL_AUDIT_zh.md`; visual only, `_dress` in the geometry):
  - edges: Nyx drew MJCF boxes and cylinders as meshes whose faces share vertices and average their normals, so every
    edge looked rounded (the boards like cushions); their looks are now vertex-only meshes, drawn with MuJoCo's normals,
    chamfered inside the solid (boards and slats 1.5 mm, ledges and the key rest 1 mm, the legs' tops 2 mm, the cradle
    walls 0.4 mm);
  - colours: pine boards each a shade apart (base (.70, .50, .30)), birch slats, a hardwood corner block and holders;
  - the bolt: black oxide (.16, .16, .17), its head drawn round (the recess's rim pieces cut to the head's 48-sided
    prism, its top rim chamfered 0.8 mm, the underside 0.5 mm);
  - the tools: the hex and Torx keys satin chrome (.66, .68, .71), bent round 8 mm on the centre line instead of the
    ball at the corner; the ratcheting driver's handle dark, its housing and bit bright, a ring on the housing;
  - additions: a zinc threaded insert's flange in the block face (the hole's drawn thread in zinc), zinc screw heads on
    the slats, the cradle walls drawn as solid V-blocks down to the table, the key's stand drawn as one block with its
    hole's mouth dark on top.
  The MuJoCo scene drops the geoms with no contact and no mass and the meshes only they use
  (`topogym.world.mujoco.mjcf_entity.merge_mjcf_entity`), so each instance's physics `scene.xml` is byte for byte dev
  a0856f6b32's; only `instance.json`'s asset identity (the entity files' sha256) differs. The births are unchanged and
  fresh strict replays of the five references end on their recorded states. The fingers as drawn (the robot MJCF's
  visual finger meshes at their bodies' poses) never come inside a V-block, the stand's hole or the bend's added inner
  fillet at any row of the five references. Harbor on 9b2e427a79: PASS (oracle 1, nop 0, both modes, all five bases;
  mk_tasks checks and the instance bytes ok). Raw-frame demos of the five references in the new look (every camera,
  stride 5, 30 fps; each replay at its recorded end) and before / after cam_high stills of 01 and 05:
  `runs/robotangle_mujoco/bed_bolt/20261019_visual/` (`demos_9b2e427a79/`, `before_after_0{1,5}_cam_high.png`).

## The thread (the task's mechanism)

MuJoCo has no screw joint. A carriage at the hole mouth, aligned with the bore, carries the caught bolt: two tilt hinges
(pivot at the mouth), a slide along the bore and a spin hinge about it. A joint equality couples the slide to the spin
(slide = 5.7 mm / (2 pi) x spin), the slide stops at the seat (85 mm, the head on the block face), and a weld from the
spin body to the bolt (inactive at birth) carries the bolt once the thread has caught. The task's mechanism
(`topogym/tasks/bed_bolt/thread.py`, `MujocoWorld.set_mechanism`, docs/rcb/MUJOCO_CORE.md "Task mechanisms") runs once per
2.5 ms tick:

| | rule (user 2026-10-12) |
|---|---|
| catch (window and thread user 2026-10-17) | the bolt's tip at least 2.5 mm inside the bore and no deeper than the stop's limit (below) plus 0.8 mm, within 1 mm of its axis, its axis within 2 degrees, lined up with the hole's thread (the bolt's roll about the bore axis within 30 degrees of phase_rad + 2 pi depth / 5.7 mm; phase_rad 0.90238 rad is the -i01 reference's own at its catch, so it catches on the same tick); 0.5 s after a let-go: the weld switched on at the present relative pose, the bore's contacts and the stop off. A straight push sweeps a whole pitch inside the 2.5-8.2 mm window, so it lines up somewhere in it |
| stop (user 2026-10-17) | a loose bolt cannot be pushed past the thread: a collision-only plate across the bore (`thr_stop_plate`, group 3, never drawn, a body of its own after the carriage; 10 mm thick, the socket's contact 1.25 ms / .99 .999, kept by the profile so it wins its pairs) whose face the tip meets, 0.3 mm past the limit: the first thread, 8.2 mm, until the thread has caught, then the deepest the thread has carried the bolt (s_max) if deeper. Off while caught; moved only while off; on again only on a tick where nothing overlaps it. A bolt that let go therefore catches again only where it has been turned in, in the same groove. Before this the window ran to 20 mm (a sandbox literal): a 2.5-degree tilted push or a let-go and a push caught 19 mm deep, 3.33 turns against the reference's 0.44 |
| hold | the tilt hinges' stiffness is 2e-4 N m / rad, always; from 2 engaged turns both tilt hinges are held rigid by joint equalities (`thread_tilt_y`, `thread_tilt_z`), set at their angles on that tick, each target eased to 0 at 2 degrees/s while the actual tilt is within 0.2 degrees of it (the bolt re-centres; held tilted, the target waits), released below 2 turns or at a let-go. This replaces the sandbox's 3 N m / rad from 2 turns, whose stiffness switch stored 0.5 k tilt^2 (16.7 mJ at the reference's 6.7-degree crossing) and let a 100 N axial push buckle the bolt to 16 degrees |
| let go | below 2 turns, drooping more than 3 degrees with no robot finger on the bolt and no key end in its socket: the weld off, the bore's contacts on |
| slip (removed, user 2026-10-18) | until 2026-10-18 a key end in the socket while the bolt turned faster than 3 rad/s (over the last 25 ms) lost its grip: the key-socket contacts off until that end had left the socket. The user removed the rule from the family. The mechanism's readings keep its two flags, `slipped` (always false) and `socket_on` (always true), and the rate window still follows the key about the bore axis while the thread is not caught, so the state hashes are unchanged (below) |
| lock (user 2026-10-16) | caught: the spin held at its angle by a joint equality (`thread_lock`, set at that tick's angle); a push or pull along the axis or a finger's twist does not turn the bolt. No key end in the socket: locked, always. A key end in it: static friction of 1 N m, the lock breaks away on the tick the spin carries more than 1 N m (the lock row and the spin hinge's friction row together), the 1 N m friction row acts from then, and the spin locks again once 25 ms have passed since the break-away and its rate over the last 25 ms is below 0.05 rad/s |
| friction (user 2026-10-16) | 1 N m on the spin hinge (about 1.1 kN along the axis to back-drive the 5.7 mm lead; a two-finger pinch on the head gives about 0.56 N m; the key's 17 cm lever turns it with about 6 N at the hand) |
| grasp (user 2026-10-16) | while caught the bolt is no assisted-grasp target (the mechanism's `grasp_allowed`; a binding already on it is released, `target_not_graspable`); the fingers' contacts stay |

Contacts the mechanism switched off come back only on a tick where nothing overlaps them (MuJoCo's convex distance), and
the thread's equalities (lead, weld, lock, tilt holds) are at impedance .9999 with 2-dt time constants: they act on an almost massless
carriage, so on the exact constraint diagonal their regulariser is large, and at .99 / .999 a 60 N push moved the bolt
0.7 mm with the spin still. A friction row creeps below its bound in proportion to the load (measured on 9f8d718c12: about
9e-5 turn per second per newton of push, 60 N for 180 s: 1.04 turns; on eb21a8481c the key left unheld in the socket
turned the bolt 3.9 degrees in 5 s by its own weight, 0.035 N m), which is why the bolt is held by the lock and the
friction row acts only while it slides.

The design's sandbox projected the slide onto the lead after every MuJoCo step; here the lead is a constraint inside
MuJoCo's solver, so no state is written between steps and the stress observer's books see its work.

## Goal

Terminal revision `bed_bolt_seated_v1`: the thread caught and the slide within 1.5 mm of the seat (the head within 1.5 mm
of the block face), the bolt's axis within 2 degrees of the bore's, the bolt and the key still (1 mm/s, 0.02 rad/s), for
0.5 s; holding allowed.

## Reference (-i01)

The base is `bed-bolt-01-i01` (the family's number 01 under the RoboARC names `<family>-<NN>-i01`; named
`bed-bolt-crossbar-i01` until 2026-10-17, the evidence below up to that day carries the earlier name).

The threaded hole (2026-10-17: the catch window, the stop, the thread angle, the drawn threads) left the reference's
physics unchanged: along all 25,325 rows its (qpos, qvel) is bit for bit the old tree's (847096c4d1) at every control
step (end 7423cfdd92cef89c), on two hosts. The mechanism's new state (s_max, the stop's switch) is in the state hash, so
the records are new: birth 0bfc2739be27db3a, end fa027063506f37b9; the reference re-stamped with that birth, its rows
and tick ledger byte for byte the same (sha256 bd3033a4...; before: 2455d3bd..., birth 5304a7d2b2771443, end
3da1814664852f27). Evidence: `runs/robotangle_mujoco/bed_bolt/20261017_thread/` (chain3r: the physics hashes, two strict
replays, the observer, nop, reset, the thread probes and the push checks on 0832f63253).

`scripts/robotangle/oracles/bed-bolt-01-i01/privileged.npz` (`standard.npz` is the same file; sha256 2455d3bd... until
the re-stamp above),
25,325 rows (633.1 s), native gripper space, on the instance made on 5587125540 (`rcb_topogym.py make-instance --task
bed_bolt --variation bed_bolt_crossbar --scene-seed 0 --gripper-space native`; birth 5304a7d2b2771443, the same on the
local host and on a lab RTX 4090 host). Recorded on a lab RTX 4090 host with

```bash
python scripts/robotangle/shared/rcb_topogym.py run-oracle --task bed_bolt --instance <instance.json> \
    --code scripts/robotangle/references/bed_bolt/bed_bolt_crossbar/oracle.py --out <dir>/privileged.npz
```

The oracle code (`scripts/robotangle/references/bed_bolt/bed_bolt_crossbar/oracle.py`) is the design's closed-loop replay
(`runs/bed_bolt_design_20261012/scripts/phys3_replay.py`) through the Session, on the design's kinematic plan
(`plan.npz`, `plan.json` next to it); its header lists what it does per control step. The right hand carries the bolt
in, the thread catches at 12.2 s (0.44 turns; the right hand's assisted grasp is released there, `target_not_graspable`),
the right hand pinches the shank until 174.0 s (3.27 turns), the left hand takes the key, puts its short arm into the
socket and swings the lever 60 degrees per stroke (86 strokes, 7 push-ins backed off and repeated, all before 125 s).
The thread reaches 2 turns at the tenth stroke with the bolt tilted 3.8 degrees; the tilt holds bring it to 0 within
two strokes. The episode ends in success at row 25,325: 14.77 turns, the tip 84.21 mm deep, the axis 0.0 degrees off,
no let-go, slip, forcing, review or instability; final state 3da1814664852f27. The largest arm command step is
0.0350 rad (the first row 0.030 rad from the birth targets).

Checks of the reference on 5587125540, in the release image's libraries (`eai-robotangle:0.3.2-rc1`, mujoco 3.14.0),
on a lab RTX 4090 host (load average 25-32 on 32 threads); evidence `runs/robotangle_mujoco/bed_bolt/20261016/tilt_5587125540/`:
- Two fresh-process strict replays (`rcb_topogym.py replay --repeat 1 --strict`, one process each): success at row
  25,325, both 3da1814664852f27, 443 s and 463 s of wall clock (0.018 s per control step).
- The stress observer (`stress/replay.py --hold 0`): CLEAN; energy created 0.0073 J; the fastest task link 0.24 m/s and
  2.7 rad/s (the key); task contact depth at most 0.27 mm; one grasp event, the right hand's hold on the bolt released
  at the catch (`target_not_graspable`, the veto). The tilt hold's net work, 0.028 J (`thread_tilt_y`, lifting the
  drooping bolt and the key to the axis), is booked as constraint work. 56.6 ms per control step under the observer.
- The equal-length nop (25,325 rows of the birth targets, under the observer): no success, CLEAN.
- One run of the oracle: 542 s of wall clock.

The thread's rules under load, at rows of this reference (`runs/robotangle_mujoco/bed_bolt/20261016/tools/`):
- A push or pull along the bore axis (row 8038: 3.94 turns, key out, the right hand gone; 180 s each): 60 N moves the
  slide 0.0008 turns, 200 N 0.0026 turns, both ways; constant from the first second on (no creep); the tilt stays 0.0
  degrees and the spin locked. After a 2 s push of 200 N the slide is back to 0.
- A wrist twist of 60 degrees with the fingers fully closed on the shank, with and without assisted grasping: no hold
  forms; the spin turns 0.000004 degrees at row 560 (0.44 turns, key out) and at row 6581 (3.11 turns, key out); with the
  key in the socket but not held (row 6690, the left hand opened first) 0.000002 degrees. The key left unheld in the
  socket for 5 s: 0 degrees (the lock carries the key's weight, 0.041 N m).
- A slow key swing with the left hand (row 10270, 8 s): at 0.05, 0.075 and 0.1 rad/s the lock breaks away once and the
  bolt follows the key continuously (19.9, 31.6 and 43.2 degrees for the hand's 22.9, 34.4 and 45.8 degrees); slower than
  the re-lock rate it sticks and slips: at 0.04 rad/s two steps of 7.3-7.5 degrees, at 0.025 rad/s one step of 6.5
  degrees after 6 s of winding up (the hand and the key wind up about 8 degrees before the spin carries 1 N m).
- An independent check of the push (jianwen-2d's probe, on this reference): at row 8038 60 N moves the slide 0.00078
  turns and 200 N 0.00261 turns, both ways, the tilt 0.0 degrees, locked, constant from 1 s to 180 s; at row 10270 (the
  key in the socket, the left hand holding it) 200 N either way moves it 0.0026 turns and the spin stays locked: the
  push's torque on the spin through the lead, about 0.18 N m, is under the 1 N m break-away.
- The mechanism's switches in this reference (a replay under the observer with its per-tick trace, every control step
  in which a switch happened against all steps; `tools/switch_trace.py`): the spin lock switches 172 times (86 break-aways
  at a swing's start and 86 re-locks at its end, all with the key in the socket; the lock stays on as the key leaves and
  enters), the tilt holds once (at 2.0003 turns, tilted 5.86 degrees). The observer's energy excess falls over every one
  of those steps (by 2.3 mJ or more; the largest rise over any step of the run is +0.34 mJ); the lock holds its angle
  within 1.5e-5 rad and the eased tilt targets stay within 0.0073 degrees of the tilt.

## The key's slip removed (2026-10-18)

The user removed the slip rule (a key end in the socket lost its grip while the bolt turned faster than 3 rad/s) from
the whole family (a4377047bb); the public goal drops its sentence and the texts were regenerated (7eed8ef8e6). The slip
never fired in the -i01 reference, and the state hash keeps the rule's two flags (constant), so the base is unchanged:
- the instance made on a4377047bb: birth 0bfc2739be27db3a, scene.xml sha256 9b65dc102aed2964... (as before), on the local
  worktree and on a lab RTX 4090 host (a clean tree of the commit, `<host work dir>/bedbolt_variants/src-a4377047bb`); the
  instance.json differs from dev's only in the goal text;
- three fresh-process strict replays of the reference (`rcb_topogym.py replay --repeat 1 --strict`): success at row
  25,325, end fa027063506f37b9 (two local, 366 s and 360 s; one on a lab RTX 4090 host, 362 s).
Evidence: `runs/robotangle_mujoco/bed_bolt/20261018_noslip/` (hash_local/, hash_m3/, tools/hash01.sh).

## Stress acceptance without the key's slip (r4, 2026-10-18)

On a lab RTX 4090 host, on 6d71d4f7c0 (the slip removed; jianwen/bed-bolt-noslip), criteria v2.13b; containers jwbn-nsl_r4-*.
Evidence: a lab RTX 4090 host `<host work dir>/bedbolt_variants/noslip/stress/nsl_r4/`; the reports and the switch watch in
`runs/robotangle_mujoco/bed_bolt/20261018_noslip/stress_r4/`.
- Self-check: PASS. The reference success, CLEAN, the same hash (21.2 min under the observer); the nop no success,
  CLEAN; 69 search episodes, 22,581 controls, 0 confirmed BREAKDOWN under every reading v2 to v2.13b.
- The acceptance round: **PASS**. seed0 120020 (seeds 120021-120029), 1,000 episodes over random, directed, mutate, cem,
  periodic and prefix (named four times), 4 search and 2 replay workers, STRESS_KEEP_ALL=1.
  - 1,271,341 controls; 0 confirmed BREAKDOWN under every reading v2 to v2.13b. 24 grasp slips, 0 pops. P(miss) 0.37 at a
    0.1 % breakdown rate and 0.0067 at 0.5 %.
  - The reference: success, CLEAN, the same final state (26.9 min under the observer). The nop: no success, CLEAN.
  - In-search VIOLENT: random 11, directed 0, mutate 0, cem 4, periodic 0, prefix 9.
- The switch watch (`tools/watch.py`; `20261018_noslip/tools/watch_rules.py`, in which any change of `slipped` or
  `socket_on` is unexplained): all 166 prefix episodes and the flagged from-birth episodes (333 traced), every replay
  reproducing the search's own readings.
  - From birth the thread never caught. Prefix tails: 23 let-goes, 39 lock break-aways and 67 re-locks with the key in
    the socket, 8 locks with no key end, one catch; no slip and no socket switch. None unexplained by its rule.
  - Energy: the observer's energy excess falls over every switch step (by 0.14 mJ or more); the largest rise over any
    step is +105 mJ, at no switch.
  - 20 findings coincide with a switch (within 4 control steps): 5 energy peaks of 19-40 mJ (4 at let-goes, one at a
    break-away with the key in the socket; the excess falls 62-486 mJ over those steps); 9 grasp losses (8 the left
    hand's hold on the key at a swing's start or end, where the lock breaks away or re-locks, one at a let-go); in prefix2
    e12 (cut at row 3,325) a VIOLENT trigger, a 2.5 kN contact of the left upper arm with the bed frame 3 steps after a
    re-lock with the key in the socket, with a grasp slip and two degenerate rows (no BREAKDOWN on confirmation); and two
    causal-jump events at break-aways in prefix3 e27.
- Harbor under bed-bolt-01-i01 on 6d71d4f7c0 (dev's `mujoco_release.json` lists the base; no test-only commit), a lab RTX 4090 host,
  rendering off: **PASS**. The instance made in the release image (`eai-robotangle:0.3.0-dryrun-bed-bolt-01-noslip-1`)
  has the reference's birth 0bfc2739be27db3a; mk_tasks checks ok; oracle 1 and nop 0 in both modes, the end
  fa027063506f37b9 in both replays (privileged 1228 s, standard 1322 s). Evidence: `20261018_noslip/harbor_test/`.

## Stress acceptance on the threaded hole (r3, 2026-10-18)

This round and the one below ran with the key's slip rule, removed since (2026-10-18).

On a lab RTX 4090 host, on 0832f63253 (the threaded hole and the stop's profile), criteria v2.13b. Evidence:
a lab RTX 4090 host `<host work dir>/bedbolt/stress/r3/`; the reports and the switch watch in
`runs/robotangle_mujoco/bed_bolt/20261017_thread/stress_r3/`.
- Self-check: PASS. The reference success, CLEAN, the same hash (21.7 min under the observer); the nop no success,
  CLEAN; 69 search episodes, 22,581 controls, 0 confirmed BREAKDOWN under every reading v2 to v2.13b.
- The acceptance round: **PASS**. seed0 120010 (seeds 120011-120019), 1,000 episodes over random, directed, mutate, cem,
  periodic and prefix (named four times), 4 search and 2 replay workers, STRESS_KEEP_ALL=1.
  - 1,132,336 controls; 47 candidates replayed in fresh processes; 0 confirmed BREAKDOWN under every reading v2 to
    v2.13b. 24 grasp slips, 0 pops. P(miss) 0.37 at a 0.1 % breakdown rate and 0.0067 at 0.5 %.
  - The reference: success, CLEAN, the same final state (35.6 min under the observer). The nop: no success, CLEAN.
  - In-search VIOLENT: random 6, directed 0, mutate 1, cem 43, periodic 0, prefix 11.
  - The round's reference copy carries the stamp from before the re-stamp (the same rows).
- The switch watch (`tools/watch.py`, with the stop's switch and s_max; `tools/watch_rules.py` with the catch-depth and
  stop rules): all 166 prefix episodes and the 204 flagged from-birth episodes, every replay reproducing the search's own
  readings.
  - From birth the thread never caught: no switch.
  - Prefix tails: 13 let-goes (each with the stop on again once clear), 39 lock break-aways and 57 re-locks with the key
    in the socket, 14 locks with no key end, 7 slips, no re-catch. None unexplained by its rule.
  - Energy: the observer's energy excess falls over every switch step but one, a lock break-away with the key in the
    socket at 4.27 turns (prefix4 e6, row 8688: +1.4 mJ, under the 10 mJ finding level); the largest rise over any
    step is +125 mJ, at no switch.
  - Findings: 590 (292 energy peaks of at least 0.01 J, 65 VIOLENT triggers, 233 observer events). No VIOLENT finding
    coincides with a switch. One energy finding does: prefix4 e8 (cut at row 3,830, 1.44 turns): the thread let go
    drooping 14 degrees, the stop came on two steps later and the bore's contacts two after that; 20 mJ created there
    while the excess fell 81 mJ over that step (CLEAN). The other 10 coincidences are grasp events (the left hand's
    hold on the key slipping at a swing's start or end; one hand's fingers opening).

## Stress acceptance (2026-10-17)

This round ran on the thread before the threaded hole (the 20 mm catch window, no stop, no thread angle); r3 above is
the round on the threaded hole.

On a lab RTX 4090 host, on b2ed1b464e (the code of 5587125540), criteria v2.13b, under the base's earlier name. Evidence:
a lab RTX 4090 host `<host work dir>/jianwen-bedbolt/stress/r2/` (steps.txt, round_1.log, out/); the reports and the switch watch in
`runs/robotangle_mujoco/bed_bolt/20261016/stress_r2/`.
- Self-check: PASS. The reference: success, CLEAN, 3da1814664852f27 in two fresh processes (23.0 min under the
  observer). The nop: no success, CLEAN. 69 search episodes, 22,581 controls, 2 VIOLENT (random), 0 confirmed BREAKDOWN
  under every reading v2 to v2.13b.
- The acceptance round: **PASS**. seed0 120000 (seeds 120001-120009), 1,000 episodes over random, directed, mutate, cem,
  periodic and prefix (prefix named four times, for the 25k-row reference), 5 search and 3 replay workers,
  STRESS_KEEP_ALL=1 (every episode's rows kept for the switch watch).
  - 1,103,991 controls; 39 candidates replayed in fresh processes; 0 confirmed BREAKDOWN under every reading v2 to
    v2.13b. 27 grasp slips, 0 pops. P(miss) 0.37 at a 0.1 % breakdown rate and 0.0067 at 0.5 %.
  - The reference: success, CLEAN, the same final state (29.9 min under the observer). The nop: no success, CLEAN.
  - In-search VIOLENT: random 7, directed 1, mutate 0, cem 43, periodic 0, prefix 10.
  - The first container was killed at 05:54 (exit 137, no traceback). The round resumed on the same output at 06:44: the
    finished workers were kept and the four prefix workers restarted from their first episode. It ended at 10:15.

### The mechanism's switches in the round

The rcb reviewer asked for the mechanism's switches to be watched in the first acceptance round: the spin lock on and
off, stick-slip, the tilt holds and the bolt's grasp veto. Any energy or VIOLENT finding that coincides with a switch is
to be reported.
- How (`runs/robotangle_mujoco/bed_bolt/20261016/tools/watch.py`):
  - each episode is replayed as the search ran it, from the rows the round kept: a fresh Session, strict, the stress
    observer with its per-tick trace;
  - the mechanism is read after every control step;
  - a finding coincides with a switch when a switch happened within 4 control steps of the finding's tick (0.1 s, the
    local energy rule's window).
- Every replay reproduced the search's own observer readings (verdict, largest force, speed, energy excess).
- From birth (random, directed, mutate, cem, periodic): the 170 flagged episodes of 834 (VIOLENT, or an energy peak of
  at least 0.01 J). The thread never caught: no switch, no coincidence. These were traced on those strategies' finished
  files; their records are identical in the final round.
- Prefix (the reference's first k rows, then a search tail; the only episodes in which the thread is caught): all 166.
  - Switches over the whole episodes, the reference's part included: 166 catches; 6,390 lock break-aways and 6,380
    re-locks with the key in the socket; 174 locks with no key end in the socket (at a catch, or as the key leaves after
    a break-away); 125 tilt holds switched on; 163 grasp vetoes.
  - In the tails: 25 let-goes (0.44-1.77 turns, drooping 3.5-16.2 degrees), 3 re-catches, 7 slips, 1 tilt hold switched
    on. A slip is a key end in the socket while the bolt turns faster than 3 rad/s.
  - Each tail switch was checked against its rule (`tools/watch_rules.py`): a catch inside the 2.5-20 mm window, a
    let-go below 2 turns, a tilt hold only from 2 turns, the lock off with no key end only at a let-go, the bore's
    contacts back only while not caught. None is unexplained.
  - One re-catch moved the turn count forward. In prefix e17 (cut at row 4,139) the thread let go at 1.61 turns, and 22
    control steps later it caught again at 2.99 turns. The loose bolt had been pushed 8 mm deeper, and a catch reads the
    bolt's actual depth. The catch window ends at 20 mm, so a push gains at most about 3.5 turns.
  - Energy: the observer's energy excess falls over every control step in which a switch happened, in all 166 episodes
    (the smallest fall is 0.1 mJ). The largest rise over any control step is +39 mJ, at no switch.
- Findings in the prefix episodes: 115 (58 energy peaks of at least 0.01 J, 17 VIOLENT triggers, 40 observer events).
  - No energy finding coincides with a switch.
  - One VIOLENT finding does, in prefix4 e18 (cut at row 3,516, 1.44 turns). The tail moved the right hand off the shank
    and the thread let go, the bolt tilted 16.2 degrees. The bore's contacts came back 21 steps later. 4 steps after that,
    the loose bolt dropping out of the bore turned at 61 rad/s with a surface point above 3 m/s. The energy excess fell
    over both switch steps (61 and 75 mJ), the episode's energy excess is 0, and the round's confirmation found no
    BREAKDOWN.
  - The other 8 coincidences are grasp events. 7 times the left hand's hold on the key slipped at a swing's start or end,
    where the lock breaks away or re-locks. Once the right hand's fingers opened, at the let-go above.
- Evidence: `stress_r2/watch_prefix/` (watch.jsonl, summary.json, rules.json), `stress_r2/watch_birth/`,
  `stress_r2/watch_periodic/`.

## Known limitation: the left hand's hold on the key

The left hand holds the key's long arm by the assisted grasp (contact_weld_v1), and the weld carries more twisting
torque than the two pads' friction could. The grasp audit of the -i01 reference (`runs/robotangle_mujoco/bed_bolt/
20261018_noslip/tools/grasp_check.py`, rendering off, on a4377047bb, a lab RTX 4090 host; `20261018_noslip/grasp01/out/
audit.json`; the replay ends on fa027063506f37b9 with success) compares, on each of the 24,683 rows the left hand holds
the key, the weld's torque on the key about the fingers' closing axis with the torque Coulomb friction can hold there:
- at the fingers' own contacts (their friction coefficients, measured normal forces of about 8.5 N per finger, and
  distances from the closing axis): the weld carries more on 6,871 rows, p95 10.8 times as much;
- a line-contact estimate (each pad presses the bar along a line across the finger's 19 mm width, uniform pressure):
  p95 10.3 times with the measured forces; 1.8 times (5,051 rows above 1) if each finger pressed with 50 N.
The same audit of the drives' references (`runs/robotangle_mujoco/bed_bolt/20261018_variants/<drive>/grasp/out/
audit.json`, with the wrist and close-view stills; each replay ends on its reference's state with success): the Torx key
(01's grip): p95 9.9 at the fingers' contacts; the ratcheting key, whose swings start with the tip held in the recess
and the bit carrying the thread's 1 N m from the first degree, more: square p95 17.8 at the fingers' contacts and 3.1 by
the line-contact estimate at 50 N, Phillips 21.9 and 3.9.
The user accepted this as a known limitation of the family (2026-10-18): the grip is not changed. The grasps still look
real: on every drive both pads close on the bar from opposite sides on every held row (their push directions at least
174.5 degrees apart, every contact on the inner pad faces).

## Harbor under bed-bolt-01-i01 (2026-10-17)

On the threaded hole (2026-10-18): `release_rebuild.py` on the test-only commit 66c5b3d3a9 (branch
jianwen/harbor-bed-bolt-01-test3, never merged: fd53276335, the reference re-stamped, with this base in
`mujoco_release.json` `new_bases`), on a lab RTX 4090 host, rendering off: **PASS**.
- The instance made in the release image (`eai-robotangle:0.3.0-dryrun-bed-bolt-01-4`): birth 0bfc2739be27db3a, the
  re-stamped reference's. mk_tasks checks ok.
- privileged: oracle 1, nop 0, the oracle's end fa027063506f37b9 in both replays, 1954 s.
- standard: oracle 1, nop 0, the end fa027063506f37b9 live and in the replay, 2021 s; the live episode 768 s for 25,325
  control steps (0.030 s at load average about 80).
- The first run (66c5b3d3a9 too) graded privileged the same but lost both standard trials to a host file permission:
  under `<host work dir>` (a default ACL that gives "other" nothing) Harbor could not read the reward.json the verifier
  container wrote as root; a default ACL for the user on the work directory fixed it.
- Evidence: `runs/robotangle_mujoco/bed_bolt/20261017_thread/harbor_test/` (work/REPORT.md, work/harbor/results.md).

Before the threaded hole: `release_rebuild.py` on the test-only commit 9c040f7816 (branch jianwen/harbor-bed-bolt-01-test, never merged:
f45025639c with this base in `mujoco_release.json` `new_bases`), on a lab RTX 4090 host, rendering off: **PASS**.
- The instance made in the release image (`eai-robotangle:0.3.0-dryrun-bed-bolt-01-1`): birth 5304a7d2b2771443, the
  reference's. mk_tasks checks ok.
- privileged: oracle 1, nop 0, the oracle's end 3da1814664852f27 in both replays, 1110 s.
- standard: oracle 1, nop 0, the end 3da1814664852f27 live and in the replay, 1159 s. The live episode took 417.5 s for
  25,325 control steps (0.0165 s at load average about 17).
- Evidence: `runs/robotangle_mujoco/bed_bolt/20261017_metal_key/harbor_test/` (work/REPORT.md, work/harbor/results.md).

The demo of the reference with the black-oxide key (nyx_direct, one MP4 per camera, stills at rows 0, 489, 1641, 13767
and 25325, each checked against MuJoCo's own depth and segmentation render of the same state):
`runs/robotangle_mujoco/bed_bolt/20261017_metal_key/demo/` (manifest.json, checks/still_checks_summary.json).

The demo on the threaded hole (2026-10-18; `runs/robotangle_mujoco/bed_bolt/20261017_thread/demo/`, made with
`tools/demo_thread.py` in the release image above, on its instance): the reference replays to fa027063506f37b9 with
success; nyx_direct draws `bolt_thread` and `hole_thread` and not the stop plate (`drawn_check`, from the entity sources
the instance records); every part's Nyx depth agrees with MuJoCo's own render within 5 mm on at least 99.25 % of its
pixels in every camera at every still; once caught, the bolt's drawn crests sit within 1.3 degrees of the angle the hole's
drawn thread leaves for them (the two interleave). cam_high_upload.mp4 is 8.6 MB.

## The drives: bases 02-05 (2026-10-18)

The user's variants differ from 01 only in the tool's tip and the matching recess in the bolt's head; the bed frame,
the hole and its thread, the bolt's shank and head, the birth layout, the goal and every rule are 01's. Design and
measurements: `runs/robotangle_mujoco/bed_bolt/20261018_variants/DESIGN.md` (Chinese). The user's decisions:
- 02 Torx keeps 01's L-shaped key and its 60-degree strokes (drawn out, turned back, pushed in again).
- 03 square, 04 Phillips and 05 slotted use one ratcheting L-shaped offset driver: their recesses match themselves
  only every 90 or 180 degrees, more than the bay lets the lever sweep (88-96 degrees measured). The L and the bit at
  its bend are two bodies on a hinge about the short arm (`RATCHET` in the geometry); the mechanism raises the hinge's
  lower limit to its angle whenever the angle rises, so the bit turns only one way against the L (the L swung back
  free-wheels with 0.03 N m of drag, the L swung forward carries the bit and the bolt). The key goes into the recess
  once; every stroke is a 60-degree swing and a turn back with the tip left in the recess. The limit is in
  capture() / restore() and read() (`ratchet_lock_rad`, `ratchet_rad`), for the ratcheting drives only.
- No speed slip and no Phillips cam-out (removed from the family); the slotted blade's slide along the open slot is
  contact geometry, not a rule.

The recesses and tips (`DRIVES` in `topogym/assets/geometry/bed_bolt.py`; 0.075 mm clearance per side as the hex's):

| base | variation | recess (matches itself every) | tool |
|---|---|---|---|
| 02 | `bed_bolt_torx` | six-lobe, about T50 (lobe tips 8.95 mm, inner 6.4 mm), 6 mm deep (60 deg) | 01's L-shaped key with six-lobe tips |
| 03 | `bed_bolt_square` | 6.50 mm square, 6 mm deep (90 deg) | ratcheting key, 6.35 mm (1/4 in) square tip |
| 04 | `bed_bolt_phillips` | cross of about PH4's size: four 2.2 mm slots whose ends slope from 4.6 mm at the face to 2.4 mm at 6 mm deep (90 deg) | ratcheting key, cross tip of four 2.05 mm wings |
| 05 | `bed_bolt_slotted` | straight slot 2.5 mm wide, 7.5 mm deep, across the whole head (180 deg) | ratcheting key, a blade 15 mm wide and 2.35 mm thick (below) |

On every drive's tree 01's instance and reference stay bit for bit as they were (birth 0bfc2739be27db3a, a fresh strict
replay ending on fa027063506f37b9, 01's texts regenerating to the committed ones).

| | 02 Torx | 03 square | 04 Phillips |
|---|---|---|---|
| reference | 068c149950: 25,277 rows (631.9 s), 86 strokes, 7 push-ins repeated as 01's | 4ae8eef795: 15,540 rows (388.5 s), 82 swings | be116dec44: 15,535 rows (388.4 s), 82 swings |
| oracle code | 01's, on its plan | `bed_bolt_square/oracle.py` (the key in the recess once) | the square's as it is |
| birth / end | 88e0c83081d83d55 / 9065337a7d7b18f5 | 83a09f9d160239c0 / 478283fbcfd6f603 | 76964c8ef3746bfd / 496e553fc76d49fb |
| engaged turns, tip depth | 14.77, 84.21 mm | 14.67, 83.6 mm | 14.81, 84.4 mm |
| right hand's pinch until | 169.7 s (3.27 turns) | 109.9 s (3.39 turns) | 109.8 s (3.41 turns) |
| largest tilt at a stroke's start (square and cross) | - | 3.3 deg | 3.3 deg |
| second fresh strict replay | same end | same end | same end |
| observer (`stress/replay.py --hold 0`) | CLEAN (1126 s) | CLEAN (859 s) | CLEAN (725 s) |
| equal-length nop | no success, CLEAN | no success, CLEAN | no success, CLEAN |
| push / pull 60 N, 200 N for 180 s (key in, right hand gone) | row 8100: 0.0008, 0.0026 turns, locked | row 4500: 0.0008, 0.0026 turns, locked | row 4500: 0.0008, 0.0026 turns, locked |
| demo (nyx_direct, one MP4 per camera, close-ups of the tip in the recess) | `torx/demo/` | `square/demo/` | `phillips/demo/` |
| Nyx depth against MuJoCo's, within 5 mm (worst part with 50 pixels or more, all stills) | 98.6 % | 98.6 % | 98.6 % |
| Harbor (oracle 1, nop 0 in both modes; birth the reference's; mk_tasks checks ok) | PASS, test-only fe59e40f74 | PASS, test-only 9464ba6908 | PASS, test-only 9464ba6908 |
| acceptance round (1,000 episodes, criteria v2.13b; below) | PASS (r2) | PASS (r1) | PASS (r1) |

The slip's removal (a4377047bb, merged into the variants at 7143b286aa) left the Torx and square references as they
were: strict fresh replays on 7143b286aa end on 9065337a7d7b18f5 and 478283fbcfd6f603 with success. Their texts were
regenerated without the slip sentence (c9b65a129b). The thread probes (a straight push at six rolls catches in the
first thread, 2.5-6.8 mm deep; a 2.5-degree tilted push; a let-go and re-catch in the same groove; loose pushes onto the
stop) catch where 01's do (chain3r), within 0.02 mm, on every drive; the deep let-go probe lets go and catches again
as 01's on the Torx (on c9b65a129b byte for byte its result on 068c149950) and finds no let-go on the ratcheting drives,
whose tip never leaves the recess.
Evidence: `runs/robotangle_mujoco/bed_bolt/20261018_variants/` (`torx/`, `square/`, `phillips/`, `noslip_replay_7143b286aa/`).

Torx's acceptance round (r2, 2026-10-18; on this host (a lab RTX 4090 host), on c9b65a129b; seed0 120110, 4 search and 2 replay
workers; `runs/robotangle_mujoco/bed_bolt/20261018_variants/torx/stress_r2/`): **PASS**. Self-check PASS (the reference
CLEAN, the same hash, 16.4 min); the round 1,039,333 controls, 0 confirmed BREAKDOWN under every reading v2 to v2.13b, 26
grasp slips, 0 pops, P(miss) 0.37 / 0.0067; the reference success, CLEAN, the same final state (16.6 min), the nop no
success, CLEAN; in-search VIOLENT random 9, directed 1, cem 2, prefix 25. The switch watch (349 episodes traced, every
replay reproducing the search's readings): from birth the thread never caught; prefix tails 19 let-goes, 41 break-aways
and 55 re-locks with the key in the socket, 22 locks with no key end, no slip; none unexplained. The energy excess
falls over every switch step but one, a re-lock with the key in the socket at 2.14 turns (prefix e37, row 4,948: +2.5
mJ, under the 10 mJ finding level); the largest rise over any step is +68 mJ, at no switch. 26 findings coincide with a
switch: grasp losses at a swing's start or end, energy peaks of 11-36 mJ at let-goes and locks (the excess falling over
those steps), and findings in VIOLENT episodes (prefix2 e6: a 2.0 kN contact of an arm with the bed frame; prefix4 e11: a
surface above 3 m/s; prefix4 e18: 36 mJ at a let-go), none a BREAKDOWN on confirmation.

The square's and the cross's acceptance rounds (r1, 2026-10-18/19; criteria v2.13b; 1,000 episodes over the six
strategies, prefix named four times, 4 search and 2 replay workers; `square/stress_r1/`, `phillips/stress_r1/` in the
variants' runs folder): both **PASS**.

| | 03 square (a lab RTX 4090 host, c9b65a129b, seed0 120120) | 04 Phillips (a lab RTX 4090 host, 5d8483254f, seed0 120130) |
|---|---|---|
| self-check | PASS (the reference CLEAN, the same hash, 14.8 min) | PASS (11.0 min) |
| search | 778,880 controls, 0 confirmed BREAKDOWN, 38 grasp slips, 0 pops | 801,238 controls, 0 confirmed BREAKDOWN, 38 grasp slips, 0 pops |
| reference, nop | success, CLEAN, the same final state (18.7 min); no success, CLEAN | success, CLEAN, the same final state (12.4 min); no success, CLEAN |
| in-search VIOLENT | random 10, cem 13, prefix 20 | random 13, cem 51, prefix 12 |
| switch watch: traced, unexplained | 334, none | 368, none |
| prefix tails | 22 let-goes, 46 break-aways and 82 re-locks with the key in the socket, 18 locks with no key end, 1 tilt hold on; no slip | 19 let-goes, 37 break-aways and 79 re-locks with the key in, 24 locks with no key end; no slip |
| energy over switch steps | falls over all but one: a re-lock with the key in at 13.34 turns (prefix4 e6, row 13,791), +4.7 mJ, where a 13.7 mJ energy peak was recorded (the episode CLEAN) | falls over all but one: a re-lock with the key in at 9.42 turns (prefix e14, row 9,997), +2.3 mJ |
| largest rise over any step | +83 mJ, at no switch | +64 mJ, at no switch |
| findings at a switch | 26: grasp losses at a swing's start or end; energy peaks of 13-60 mJ at let-goes and locks (the 60 mJ one in a VIOLENT episode, prefix2 e24, the excess falling 268 mJ over that step) | 17: grasp losses; two energy rises of 11-16 mJ at re-locks; in VIOLENT prefix3 e28 the bit spinning with a surface above 3 m/s just after a let-go (the excess falling 70 mJ) |

None of these findings is a BREAKDOWN on confirmation.

A known small effect (accepted 2026-10-19, no mechanism change): at a re-lock with the key in the socket (the spin
lock switched on again once the spin has slowed below 0.05 rad/s) the observer's energy excess can rise by a few
millijoules over that control step, under the 10 mJ finding level; everywhere else it falls over every switch step.
One step in each of three rounds: the square, +4.7 mJ at 13.34 turns (prefix4 episode 6, row 13,791; a 13.7 mJ energy
peak was recorded on that step, the episode CLEAN); the Torx, +2.5 mJ at 2.14 turns (prefix episode 37, row 4,948); the
cross, +2.3 mJ at 9.42 turns (prefix episode 14, row 9,997). The slotted's round has two, both in the reference's own
rows (+0.02 and +0.06 mJ). 01's r4 has none (its largest change over a switch step is -0.14 mJ); 01's r3 had +1.4 mJ at
a break-away (prefix4 e6, row 8,688).

The Harbor runs used test-only commits that list the drives' bases in `mujoco_release.json` `new_bases` (branches
jianwen/harbor-bed-bolt-02-test1 and jianwen/harbor-bed-bolt-03-04-test1, never merged); evidence `torx/harbor_test/`
and `harbor0304_test/` in the variants' runs folder.

05 slotted (2026-10-18/19). Below 2 turns the thread does not carry the bolt and the blade in the open slot gives it no
support along the slot; in each 60-degree swing the left hand's key tip leaves its target by 4-6 mm (the hand's tracking
error at the end of the 17 cm lever), which the closed recesses take up and the slot does not. With 01's catch (0.44
turns) and a 9 mm blade in a 4.5 mm deep slot the blade walked 4-8 mm along the slot, rode up out of it and left the head
before the bolt reached 2 turns (about 9 swings; the best of 13 oracle variants reached 1.53 turns). The settled scene is
the user's choice A (2026-10-19) on top of B (2026-10-18), slotted only:
- B, a deeper catch: the bolt lies in its cradle turned -10 deg against the others' 30 deg, so its thread lines up with
  the hole's near the end of the first thread (caught at 7.1 mm, 1.25 turns), and the slot is turned in the head
  (rotate_deg 20 against the square's 60) so that, caught there, it lies where the reference's first insertion meets it;
- A, a blade 15 mm wide, nearly the head's 16 mm (a real M10 slotted driver's), the key's stand with a 17 mm hole along
  the blade;
- the slot 7.5 mm deep (pre-approved up to that), the smallest that held: with the final oracle a 4.5 mm slot loses the
  blade at 1.58 turns and a 6.5 mm slot at 6.74 turns.
The oracle (`bed_bolt_slotted/oracle.py`, the square's with the blade kept on the slot): the planned bolt poses turned with
the cradle, the push 4.8 mm deeper; the bit lined up against the slot; from the first swing on, through the right hand's
leaving, the key turned about the bolt's own axis and corrected every control step from the blade's measured end, along
the slot by 0.8 of its offset plus an integral of 0.03 per step, along the axis to 6.5 mm below the head face with 0.6 of
the error. It is fragile: of the 10 settings of these gains and the depth target tried with the depth held (o43-o49,
r1-r4), 2 seat the bolt (o47 and r1); the others stop between 4.8 and 13.7 turns when the blade leaves the slot at a turn
back, so a reference recorded on another commit may need the sweep again. 52 development runs: `runs/robotangle_mujoco/bed_bolt/20261018_variants/slotted/dev/RUNS.md`.

| 05 slotted | |
|---|---|
| reference | fce8be8813: 15,310 rows (382.7 s), 81 swings, caught at 1.25 turns, the right hand's pinch until 108.3 s (4.19 turns), 14.70 turns, the tip 83.8 mm deep, the axis 0.0 deg off, the largest tilt at a stroke's start 3.9 deg; the largest arm command step 0.035 rad |
| birth / end | b920c4a58cb335e4 / 9ff4432d309ee102 (the same in a fresh strict replay and in the development runs o47, o52) |
| second fresh strict replay (a lab RTX 4090 host) | same end |
| observer | CLEAN (1049 s) |
| equal-length nop | no success, CLEAN |
| reset | back to the birth three times |
| push / pull 60 N, 200 N for 180 s (key in, right hand gone) | row 4610: 0.0008, 0.0026 turns, locked (at row 4500, mid-swing with the lock broken away, 200 N carried the bolt on 1.44 turns with the key's swing: not a locked row) |
| thread probes | CLEAN; a straight push catches 2.5-7.1 mm deep depending on the bolt's roll, the reference's roll at 7.1 mm (1.25 turns); a tilted push and a loose push onto the stop at this roll stop at the stop without lining up |
| grasp audit | both pads on the bar from opposite sides on all but 4 of 14,647 held rows (at least 174.7 deg apart, contacts on the inner faces); the weld's twist p95 28.9 at the fingers' contacts and 4.9 at 50 N (the known limitation) |
| demo | `slotted/demo/` (Nyx depth within 5 mm on at least 98.6 % of each part's pixels in every still) |
| texts | 5acdc08607; one more line than the square's: the blade can slide out along the open slot |
| Harbor | PASS on the test-only 3c3d64bf11 (oracle 1, nop 0 in both modes; birth the reference's; mk_tasks checks ok) |
| acceptance round (1,000 episodes, criteria v2.13b; below) | PASS (r1) |

The slotted's acceptance round (r1, 2026-10-19; on a lab RTX 4090 host, on 5acdc08607; seed0 120140, 4 search and 2 replay
workers; `slotted/round_sl_r1/` in the variants' runs folder): **PASS**. Self-check PASS (the reference CLEAN, the same
hash, 30.1 min); the round 754,814 controls, 0 confirmed BREAKDOWN under every reading v2 to v2.13b, 38 grasp slips, 0
pops, P(miss) 0.37 / 0.0067; the reference success, CLEAN, the same final state (12.1 min), the nop no success, CLEAN;
in-search VIOLENT random 17, directed 1, cem 1, prefix 26. The switch watch (340 episodes traced, every replay
reproducing the search's readings): prefix tails 16 let-goes, 31 break-aways and 48 re-locks with the key in the slot,
36 locks with no key end; no slip; none unexplained. The energy excess falls over every switch step but two re-locks
with the key in, inside the reference's own rows before row 5,408 (+0.02 and +0.06 mJ); the largest rise over any step
is +68 mJ, at no switch. 23 findings coincide with a switch: 14 grasp losses at a swing's start or end; energy peaks of
12-51 mJ at break-aways and locks in the tails (the excess falling 48-621 mJ over those steps); and two in the
reference's own rows, 10.8 and 11.0 mJ at break-aways with the key in (rows 8,756 and 9,744, the excess falling 28 and 27
mJ), recorded in 9 of the 67 and 40 of the 53 prefix episodes that replay past them. None is a BREAKDOWN on
confirmation.
