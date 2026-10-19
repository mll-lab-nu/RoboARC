# Review of xiaoyu/robotangle_mujoco: IKEA MuJoCo, 12 tasks × 2 modes (all 12 ship)

By Qineng's side, 2026-10-15/16. Base: dev/robotangle_mujoco 425df7da26 plus xiaoyu/robotangle_mujoco 7b73015a7a, merged on
`v2-ikea` with no conflicts. Your commits are unchanged. Our changes are separate commits on top, listed below.

## What we need from you
1. **Further batches:** the scene sources and meshes as before: scene code in git under `images/ikea-mujoco/scenes/<task>/`,
   meshes on the HF dataset `eai-v2-internal/helper-uploads` under `ikea_mujoco/<task>/assets/`.
2. **The SOP:** how you make and check a task.
   - How a scene and its demo are produced.
   - How `success.py` is ported from the source assembly grader.
   - Which checks you run before handing a task over.
   - The branch mentions `docs/rcb/ikea_mujoco_sop.md`, but that file is not on the branch.

Everything else (grading, the standard twins, the v2.0 port, stress, timing) is ours. The list below is for your
information.

## Second batch (2026-10-16): stenerik, stegesund, kyrre, uppsta

Your 12 commits (7b73015a7a..02ab929f85) were merged onto dev/robotangle_mujoco 5c7e4308fb on `v2-ikea-b2`. Your commits
are unchanged; ours are separate commits on top. The meshes (HF `eai-v2-internal/helper-uploads` ee3908c127) match
`assets.json` file by file. The SOP (`docs/rcb/ikea_mujoco_sop.md`) is still missing.

**Conflicts, resolved as in the first batch**
- Image: ours, `ghcr.io/mll-lab-nu/eai-ikea-mujoco:0.3.0` (+ `-sim:0.3.0`) with all 12 scenes. Your local 0.2.0 / 0.3.0
  used another registry name; our 0.2.0 (98fa2d354818) is a different image. One image for every task: the per-task image
  overrides are gone, the per-task hooks of `mk_tasks.py` (image, demo, keyframe, demo_substeps) are kept.
- `mk_standard_tasks.py`, the evidence files, eket / vesken / variera: ours (our closed-loop references and our Harbor
  evidence; your eket_vlim / vesken_vlim demos and variera's 0920 result are superseded by them).

**What we changed and why**
- The 4 new scenes carried the old `ikea_runtime.py` (no cylinder->prism rule, the slow cross product). They now carry
  ours, the same file as the other 8: the prism rule applies to every scene.
- H.2: task.json "source" and the docstrings naming the upstream paths moved to `provenance.json`.
- stegesund and uppsta name the arms `left/` / `right/`. The speed table's lookup split on `_` only, so those arms had
  no joint-speed clamp at all. Both separators now; the standard service and generator take the scene's own separator.
- stegesund's text said "no release or rest test": the terminal rules apply to every task. Its and uppsta's action text
  now state joint 7 is unlimited, as the other Panda scenes.
- kyrre's mechanism reads `SW_FROM_OP` / `SW_FROM_T` from the environment to override `sw_cutover.json`. A frozen scene
  must not: `rcb_ikea` drops them on import (the tooling, the grader's replays, the standard service).
- Every privileged verifier now renders a replay video (main's PROTOCOL §4 rule, board 0136; `shared/render_video.py`,
  CPU OSMesa, evidence only).
- Standard twins and camera rigs for the 4 new scenes (`standard/rigs.json`).

**Results per task** (final physics: prisms, the arm clamp, EPA-patched MuJoCo, SIMD masking)
- **stegesund ships, both modes.** Your `stegesund_vlim` demo unchanged is the reference: 0 rows over the limits, it
  ends released and at rest. Same hashes on AMD Zen 4 and Intel; Harbor privileged oracle 1 / nop 0, standard oracle 1
  (live settled success, live == replay) / nop 0; stress 203 episodes, one genuine non-jerk BREAKDOWN (0.57 %, the known
  finger-limit class). One replay takes 3245-3518 s on a quiet Xeon core.
  Note for you: the floor plane does not collide (class `deco`), so a stand pushed off the table falls forever. No
  verdict depends on it; a colliding floor would be cleaner.
- **burvik ships, both modes (2026-10-18).** No scene change: the earlier "leg_3 needs a fixture change" was a
  mis-seated top. The demo (and the retimed base rows) release the top about 25 mm above the jig with its rim behind the
  guards; it lands on the guards' tops and leans 6° onto the posts, so the legs slope and the hand spinning leg_3 meets
  the handle rack. The jig's own slot (rim on the pads, between guards and posts) holds it at 0.5°. Our closed-loop
  reference (`oracles/burvik/`) seats it there with a two-hand servo; then leg_3 threads with the earlier controller.
  For the upright turn the demo's arm motion is kept as one rigid two-hand motion, re-executed on centred grasps (the
  right hand re-takes the handle in the demo's arm configuration). Only the robot's actions changed. Same hashes on
  AMD Zen 4, Intel AVX2 and Intel AVX-512; Harbor privileged oracle 1 / nop 0, standard oracle 1 (live settled success)
  / nop 0. Under the Franka Hand slew (rebuilt 2026-10-18: each row held while the hand is more than two ticks from its
  command; the table set down level on its feet before the hands open): 162,534 rows, one replay 3781-3783 s on a Xeon
  6230R core (88 % of the bar), 1694 s on the i9; stress 203 episodes: non-jerk BREAKDOWN 2 / 174 = 1.15 %,
  prefix_jerk 0 / 29 (4.0 % and 3 / 29 without the slew).
  Note for you: the demo releases the top before it is in the jig's slot; a demo recorded on the final physics would
  avoid it.
- **kyrre ships, both modes (2026-10-19), the one exception to the prism rule (your decision).** kyrre's scene.py
  builds with the prism rule off, so its six screws keep MuJoCo's native cylinders (image 0.3.2; the other eleven scenes
  are byte-identical to 0.3.1, and every shipped reference ends and settles on its recorded hashes in 0.3.2).
  The reason, a stress A/B on the same 203 episodes (same seeds, same reference checkpoints):
  - non-jerk BREAKDOWN 2 / 174 = 1.15 % with native cylinders, 4 / 174 = 2.3 % with the prisms (the shipped set: 1.75 %);
    all strategies 5.4 % against 5.9 %; all 11 native BREAKDOWNs reproduced from the birth;
  - the same classes in both (idle press-in jaws' speed jumps, energy at joint limits, a forearm on a leg), none on a
    cylinder geom; item 2's cap-rim normal flips do not appear in kyrre;
  - the author's demo assembles the stool only without the rule (with prisms, screw 2-1 is lost at its hole).
  Shared change made for it earlier: the **welded-part contact switch-off** (`ikea_runtime.RetentionMechanism`, from a
  scene's `sw_cutover.json`; kyrre's seated-screw switch-off on top). The reference (`oracles/kyrre/rebuild.sh`):
  the demo retimed for the Franka Hand slew (grip waits), screw 3-1's second grasp re-aimed at where the screw lies, and
  the hex key set down 12 mm along its rest. Note for you: the scene's own rest pose of the key is a knife edge (one
  contact, the arm hanging in the slot), where the key vibrates at 0.099 rad/s from the birth on, with prisms or not, so
  a key returned exactly there never meets the rest rule; a flat-down rest pose would be cleaner. Same hashes on AMD
  Zen 4, Intel AVX2 and Intel AVX-512; 2,740 s on an i9-14900HX core (63 % of the bar), 8,283 s on a Quest Xeon 6230R
  core. Harbor privileged oracle 1 / nop 0, standard oracle 1 (live settled success) / nop 0 on 0.3.2 / -sim 0.4.2.
- **stenerik ships, both modes (2026-10-17).** The demo commands up to 55x the joint-speed limit, and on our scene it
  seats only the first screw (the screwdriver arrives 0.5-2.5 mm off the screw axis on the prisms). Our closed-loop
  reference (`oracles/stenerik/`) steers and drives the screws on the true screw axis, and presses each pad closed
  loop: the demo row held, the hand servoed on the true pad-support force until the scene's bond latches (the demo's
  presses peaked at a 0.99 N mean, under the 1 N / 0.3 s gate; for three pads its press passes just above the pad).
  Only the robot's actions changed. Note for you: the welded screws keep their thread contacts, so the late ticks cost
  about 5x the early ones; a contact switch-off for seated screws would make this scene much cheaper.
- **uppsta ships, both modes (2026-10-18).** On our scene the demo's first dowel grasp squeezes out of the fingers (the
  32-facet dowel; the round one rolls into the grasp), so the reference is a closed-loop one (`oracles/uppsta/`, 430,225
  rows): the demo run in the scene without the prism rule is the phase guide. The first such reference passed but took
  12,680 s on a Xeon core (~6,000 s on a fast host). A profile showed the cost is not the thread: each end block has 114
  convex collision pieces, ~900 contacts with the board once seated, and it kept them after its screw welded it to the
  board. Two changes, none to the task: (1) the **welded-part contact switch-off** for seat welds (`seat_weld.py`, opt-in
  per scene, on in uppsta, off in stegesund): a clamped part welded to its host stops colliding with it, every other
  contact stays exactly as before; (2) the reference follows the demo at every 4th row away from the grasps and releases
  (every 2nd or the demo's pace where a segment's gate needs it), every row within the arm limits and the pending
  Franka Hand slew. One replay: 3,199 s on the quiet i9 laptop (74 % of the bar), 3,519 s on a Ryzen 7950X, 8,425 s on
  a Xeon core (over the bar on that slow CPU, as gladom / variera / stenerik). Same hashes on Zen 4, Intel AVX-512 and
  AVX2. **What would help from you (optional, uppsta ships without it):** a coarser end-block collision model (15-30
  convex pieces). Measured at screw b's drive (3000 rows from one state, laptop): 23-26 ms per row as is; 15.5 with
  every 2nd end-block piece; 10.5 with every 4th; 8.2 with the block's board contacts off. A coarser hole thread saves
  almost nothing (25.1 with every 2nd thread piece; 25.5 with the screw's thread contacts off). The board's 640 pieces
  (336 contacts lying on the table) are the next largest cost.

**Budget and texts (2026-10-16, owner):** the agent budget is 10800 s (3 h) of wall clock in both modes, as RoboARC;
the speed bar is 40 % of it (one reference execution <= 4320 s on the trial host). An instruction audit also brought:
- the privileged texts now state the joint-speed clamp;
- a privileged row cap per task from its verifier timeout (the grader refuses a longer file), and standard verifier
  timeouts that cover the longest episode the wall clock allows;
- the External material and Time limit sections in the privileged instructions;
- `r.operation_status()` in the standard texts, and gladom's mean-force requirement.

**Nyx for the standard cameras (owner 2026-10-16, in this wave).** IKEA's agent-facing images are rendered by Nyx, as
RoboARC's (`-sim:0.4.0`, `standard/ikea_nyx.py`):
- RGB from Nyx with RoboARC's look (path tracer, 32 spp, denoiser, studio HDRI, its point lights placed relative to each
  scene's work surface; colours converted to linear light); depth stays MuJoCo's raster, as in RoboARC; the same camera
  rig and calibration; the IKEA scenes keep their own geometry.
- Nyx runs in its own process and venv inside the sim sidecar (gs-nyx 0.1.3 needs numpy < 2.4 and torch), fed the
  compiled scene once and the poses per frame; the physics process stays python 3.12 + numpy 2.5.4 + the patched
  MuJoCo 3.14.0.
- Gates passed: the live state hashes with Nyx are bit-identical to EGL over the first 2000 ticks of all 8 references;
  a 4-camera observe call costs about what EGL's did (0.08-0.22 s); Nyx draws only what MuJoCo's raster drew (no sites,
  labels or debug geoms); EGL-vs-Nyx contact sheets were reviewed; Harbor standard oracle 1 / nop 0 on `-sim:0.4.0` with
  the live end hashes of the EGL runs (variera's verifier was still running at the push; its live run matched).
- The privileged replay video stays MuJoCo's renderer on CPU OSMesa: it is human evidence only and the verifier has no
  GPU.


## What we changed and why (first batch)

**Image and registry**
- The image is now `ghcr.io/mll-lab-nu/eai-ikea-mujoco:0.2.0` (and `-sim:0.2.0`), following the benchmark registry rule.
- The build context is checked against `assets.json` as before.
- `bindings.json` holds our build's id.

**No upstream pointers (H.2)**
The scene files the agent reads (`task.json` "source", `layout.json` sources / `source_usd`, the docstrings) named:
- the upstream repositories and their paths;
- their grader, controller and operation scripts;
- the IKEA manual URLs and the USD asset paths.

All of it moved to `scripts/ikea_mujoco/provenance.json`. `stage_context.py` now refuses a scene text with such a
pointer. Physics is unchanged: every scene builds to its bound initial-state hash.

**SIMD masking (cross-CPU determinism)**
- Without numpy / OpenBLAS SIMD masking, eket's reference diverged before row 20000 between Zen 4 (AVX-512) and Intel
  AVX2 hosts.
- On Zen 4, MuJoCo then segfaulted at row 29396: Harbor oracle 0 on our hosts, while it passed on yours.
- The image now sets `NPY_DISABLE_CPU_FEATURES` (AVX-512 off) and `OPENBLAS_CORETYPE=Haswell`, as the RoboARC image does.
  With them the hashes agree across both CPU classes.

**Grading: the benchmark's terminal rules, all 8 tasks, both modes** (`rcb_ikea.Session.status()` / `final_check()`,
`grade.py`)
- Success = `success.py` + released + at rest + no breakdown.
  - Released: no robot link touches a part.
  - At rest: every part ≤ 0.01 m/s and ≤ 0.05 rad/s.
  - No breakdown: no non-finite state, no BADQACC / BADQPOS / BADQVEL, no part contact deeper than 10 mm.
- It must hold at the end of the agent's rows and again after the grader holds the last row for 1.0 s.
- burvik's "the top may still be held by a robot" is gone.
- A replay that crashes in every process is a breakdown (0), not a grader error.
- Why: these are our rules for every benchmark (a momentary or held pose is not an assembly).

**Verifier isolation (H.1)**
- `test.sh` deletes symlinks, FIFOs, sockets and devices under `/app/output` before grading.
- `grade.py` refuses a symlinked trajectory.

**Speed (bit-identical)**
- `ikea_runtime.rotate()` used `np.cross`, whose axis handling dominated the success monitors' Python cost.
- A plain 3-vector cross is bit-identical and 6.5x faster: variera's reference runs 2.1x faster, the others 10-40 %.
- The end-of-reference hashes are unchanged.

**Texts**
- The terminal rules paragraph is in every instruction.
- The Panda action text says that joint 7 has no joint limit in these scenes. The real Panda's has one.

**Main-side port (v2.0)**
- `scripts/ikea_mujoco/port_main.py` and `scripts/ikea_mujoco/main_side/` render the tasks onto main with DIGEST, the
  egress allowlist, seeds, the External material / Time limit sections, resources and the monitor.
- `docs/benchmarks/ikea_mujoco.md` is the benchmark document.
- Tasks declare `dynamics = "static"`.

**Standard twins.** They are aligned to our standard rules:
- the same wall-clock cutoff as the privileged twin, 14400 s (H.7); they had 7200 s of agent time excluding service time;
- robot frame only, so no world coordinates or world base poses in anything the agent reads;
- joint-speed rule: arm targets at the robots' real velocity limits (Panda 2.175 / 2.61, ViperX 3.14159 rad/s; clamp, not fail: a row that asks for more takes the control steps it needs), the same table as the tooling;
- the live episode ends on the full verdict, not on `success.py` alone;
- standard references re-made within the speed rule. The privileged references jump up to 0.63 rad per 2.5 ms tick.

**References**
- variera's demo ended with the left hand closed on the screwdriver, which fails the release rule. We appended a 240-row
  release tail: both grippers open and hold for 0.6 s (`scripts/ikea_mujoco/tails/variera.npy`; `mk_tasks.py --demos`
  appends it).
- Before the physics fixes, the other 7 demos passed the terminal rules unchanged. With the masked image they scored
  Harbor oracle 1 on host B for lack, eket, svallet, ikea365, vesken, burvik and gladom, and nop scored 0 on all 8.
- That variera tail is superseded: the shipped references are the ones under "References." below.

**Cross-CPU**
The reference end hashes are identical on Intel AVX2 (i9-14900HX), Intel AVX-512 (Xeon Gold 6230R, Quest) and AMD
Zen 4 (Ryzen 9 7950X, EPYC 9354), all with the masking.

**MuJoCo 3.14.0 EPA crash**
- MuJoCo 3.14.0's EPA keeps the horizon in two fixed 24-entry arrays, and `addEdge` writes past them without a check.
  Upstream fixed this after 3.14.0.
- In eket, a dowel (cylinder) pressed deep into a thin convex piece of a side panel makes the horizon reach 25+ edges.
  The result is SIGSEGV or silent memory corruption.
- We see it in about 1 of 400 random eket episodes and in references slowed to the real joint-speed limits. We found
  none in the other 7 scenes.
- Fixed: the image byte-patches the wheel's own `libmujoco.so`. 14 bytes change, and only the horizon arrays get room
  (`images/ikea-mujoco/mujoco-3.14.0-epa/`, upstream 6d662aed; sha256 52e2a24e…).
  - Gates: bit-identical to the stock wheel on all 8 references and on 52 stress episodes; no crash on the 9
    reproducers.
  - A from-source build of 3.14.0 is not bit-identical to the wheel, so it is not used.
  - The grader still scores any replay crash as a breakdown (0).
- Changing the CCD options (tolerance, multiccd, nativeccd) avoids particular poses, but every such change breaks
  eket's open-loop reference.

**Timing** (the final references, against the 14400 s budget; the bar is ~20 %)

| task | one replay, by host | standard live episode, host B (loaded) |
|---|---|---|
| eket | 518 s Quest Xeon 6230R (3.6 %) | 289 s |
| svallet | 394 s Quest (old reference); Harbor verifier 355 s for 2 replays | 293 s |
| ikea365 | 293 s a cloud host EPYC, 690 s Quest (4.8 %) | 341 s |
| lack | Harbor verifier 619 s for 2 replays | 474 s |
| vesken | 630 s a cloud host (4.4 %), 1135 s loaded host C | 810 s |
| gladom | 2882 s quiet i9-14900HX laptop (20.0 %) | 3361 s (23.3 %); 4567 s (31.7 %) in an earlier, more loaded run |
| variera | 3413 s quiet i9-14900HX laptop (23.7 %) | 4160 s (28.9 %); 5689 s (39.5 %) in an earlier, more loaded run |

- MuJoCo's own step is ~95 % of the time after the bit-identical speed-up (SDF threads).
- gladom is at the bar and variera is above it on a quiet host. On loaded host B (load 50-85) both are well above.
- This is the owner's decision, in line with the knot exception (≤ 30 %).

**Physics fixes after the stress test** (RoboARC policy: one rule for all scenes, clamp not fail)

The stress test: 1,696 episodes, 293 BREAKDOWN, all confirmed from the birth. 238 of them came from instant
(unslewed) jerk commands; every other strategy combined gave 3.9 %.

1. **Joint-speed clamp, both modes:** `Session.applied()`. Every arm joint target moves at most the robot's real
   velocity limit per tick (Franka Panda 2.175 / 2.61 rad/s, ViperX 3.14159 rad/s).
   - Rows within the limit + 1e-9 are applied unchanged.
   - The Franka Hand slew (0.1 m/s of finger width): on since 2026-10-18 (below, "Franka Hand slew").
2. **Cylinders:** RoboARC's cylinder→prism rule (kyrre excepted since 2026-10-19, above). Every cylinder a robot geom can touch is a 32-facet prism at scene
   build: 64 geoms. This removed every cylinder-caused breakdown.
3. **eket dowels:** the soft dowel shells touch only the side panels; a stiff twin meets everything else.
   - The Franka finger slam is handled by the hand slew in (1).
   - vesken's spinning jaws are not fixed.

**References.**
- The original demos command joint targets faster than the robots' real limits and fail under the clamp.
- Each shipped task has ONE reference for both modes. The privileged `solution/trajectory.npz` and the standard
  `solution/oracle_actions.npz` are the same rows, all within the limits:
  - lack: the demo passed through the clamp;
  - svallet, gladom, variera: the demo time-dilated or centred to the limits (`standard/std_reference.py`);
  - eket, ikea365, vesken: new closed-loop references built on the final physics (`scripts/ikea_mujoco/oracles/<task>/`).
    The demo is the phase guide, with true-state feedback (Jacobian-IK servos, gates, local searches), recorded as the
    executed rows.
- **Shipped (7 tasks × 2 modes):** lack, svallet, gladom, variera, eket, ikea365, vesken.
  - Harbor on the final image: privileged oracle 1 / nop 0, standard oracle 1 / nop 0 (live == replay).
  - Each passes final_check in two fresh replays with the same hashes across CPUs.

**burvik: shipped 2026-10-18** (its bullet under "Results per task"). The earlier WIP finding here, that leg_3 needs a
scene or fixture change, was wrong: the top was mis-seated, not the jig.

**Live verdict, standard mode.** As in RoboWits #243 / BEHAVIOR #254, the service ends an episode only at a SETTLED
success: `status()`, then `final_check()` on a snapshot with the last row held 1.0 s. If the settled check fails, the
episode goes on, and `meta["settle_failed"]` counts it. The live and replay verdicts use one rule.

**Combined stress re-run, fixes 1 + 2 + 3b, the 4 shipped scenes then: lack, svallet, gladom, variera**
- Non-jerk BREAKDOWN fell to 1.75 %, from 3.9 % before.
- prefix_jerk is now slewed by the clamp, and its BREAKDOWN rate fell from 89 % to 7 %.
- Residuals. None of them blocks shipping:
  - gladom: finger slams under grip, before the Franka Hand slew (then off);
  - gladom: one real table penetration on prefix_jerk;
  - svallet: ViperX self-contact on jerk.

## Franka Hand slew (2026-10-18)

The Franka Hand's command is clamped at its real finger speed, 50 mm/s per finger (Franka Hand product manual
220010/1.2/EN, 5.1), i.e. 0.1 m/s of commanded finger width: 318.75 units/s in lack and stenerik, 182.142857 units/s in
the other Panda scenes; the ViperX (svallet) is not limited. Clamp, not fail, in the tooling, the privileged replay and
the standard service alike (`rcb_ikea.GRIPPER_WIDTH_SPEED_M_S`, `gripper_speed_limits()`, `Session.step_limit`), and
both modes' instructions state it with the scene's numbers (instruction-audit V1). It stops the stress test's
instant-close finger slam (~2.5 m/s fingers, 11.6 kN pad contacts, eket__mutate__s420015).

Only the robot's actions changed: every Panda reference was retimed or rebuilt for it, each passing the full gates
(fresh replays with equal hashes on Zen 4 and Intel, Harbor oracle 1 / nop 0 in both modes, check-video, fast-host
timing inside the 4320 s bar). lack: each over-fast hand row held until the hand reaches it; eket, ikea365, vesken,
gladom, variera, stegesund, stenerik, burvik: closed-loop rebuilds that hold a demo row until the slewed hand reaches
it; uppsta's reference was built within the slew and is unchanged. The scenes and images are unchanged (the tooling is
copied into the task dirs; image 0.3.1 / -sim 0.4.1). The old references fail under the new rule, so a result graded
before it is not comparable (privileged: re-replay with the new `tests/rcb_ikea.py`; standard: re-run).

## Found and still open
- **Harbor + BuildKit:** BuildKit can swap same-size, same-mtime files between tasks' build contexts (`environment/`,
  `tests/`), even with `--no-cache`. Our `instance_exact` check caught it once (lack's verifier ran with eket's binding).
  - Run Harbor through `harbor_run_safe.sh`, which gives each task's files distinct mtimes.
  - A maintainer issue is drafted.
- **Timing:** gladom is at 20 % and variera at 24 % on a quiet host, above the bar on a loaded one. The owner decides.
- **Residuals from the stress tests (none blocks shipping):**
  - Finger slams: closed by the Franka Hand slew (2026-10-18); the stress rates above were measured before it.
  - Energy created at robot joint limits under random arm targets (vesken, eket).
  - One table penetration (gladom, prefix_jerk).
  - ViperX self-contact on jerk (svallet).
