# stenerik closed-loop reference

`stenerik.npz` is the one reference for both modes of `ikea-stenerik-i00` (privileged `solution/trajectory.npz`,
standard `solution/oracle_actions.npz`). Its rows are the EXECUTED targets: `data.ctrl` after
`rcb_ikea.Session.applied()`. Every row is within the speed limits + 1e-9 of the previous row, starting from the
keyframe ctrl: the arm joints at the Panda's limits and, since 2026-10-18, the Franka Hand command (`actuator8`) at
0.1 m/s of finger width (318.75 units/s here, 0.797 units per tick; `rcb_ikea.GRIPPER_WIDTH_SPEED_M_S`), so the clamp
never changes a row on replay. Keys: `ctrl` (222600, 16) float64, `substeps` 4, `timestep` 0.000625, `actuators`.

| | |
|---|---|
| rows | 222,600 (556.5 s of simulated time) |
| sha256 | `7a1e6e502d633bfe1c563fefbde51814e2c07cde96ab619d349e9a8eb12a3039` |
| rows over the speed limit (arms and hands) | 0 (largest step 1.000000000000001 x the limit); `slewed_ticks` 0 |
| initial state | `3dc8f1b6f99efce3…` = bindings.json (the frozen birth, loaded unchanged) |
| end / settled state sha | `a690aa1b973b3f10…` / `20a59284ab83ebe8…` in every replay below |
| verdict | success: six screws seated and welded, both supports joined, four pads bonded on their sites; released, at rest, no breakdown (max part contact depth 0.73 mm); the same after the 1.0 s hold |

The physics is the final one: the prism rule, the arm clamp at the real Panda limits, the Franka Hand slew, MuJoCo
3.14.0 with the EPA byte patch, and numpy SIMD masking. Only the robot's actions differ from the author's demo; the scene,
`success.py`, the gates and the thresholds are untouched. The reference before the hand rule (215,782 rows, sha256
`e0097d6689ea0f84…`) closes the hands faster than the Franka Hand can; re-timed for the hand it seats no screw and bonds
no pad, so it was rebuilt with the closed-loop builder under the rule (2026-10-18, below).

## How it was made
The author's demo (253,729 rows) is the phase guide. It commands up to 55x the joint-speed limit and, on our scene,
seats only the first screw (`trace.py`, `scan.py`; the history is below). The reference executes it within the limits
(`Rig.move` reaches an over-limit row over the ticks it needs) and closes the loop on the true state:

0. **The Franka Hand (`st_lib.Rig.move`).** The hands slew toward each row's command at the hand's speed, and the row
   is HELD until both hand commands have reached it (`GRIP_WAIT`), so every grasp and release completes before the demo
   goes on (as with the instant command the demo was made with). 2,229 ticks of the pad phase are such waits.
1. **Screws (`build.py`, `st_lib.py`).** Per screw cycle (`CYCLES`, demo rows from the author-scene scan):
   - **place:** the carried screw is steered onto its insert axis (lateral error and tilt, damped-least-squares IK);
   - **screws 1, 2 and 4:** the demo's own spin, with the screwdriver tip kept on the true screw axis (`tip_hook`);
   - **screws 3, 5 and 6:** a closed-loop drive (`drive`): centre the tip on the true axis, turn j7 until the wings meet the
     slots, lower to 0.8 N, spin at 2.5 rad/s with a 2-6 N axial push, a 0.5 N m snugging torque near the seat until
     the weld latches, then zero torque;
   - **standing a leaning screw up (`straighten`, in `drive`):** the demo grips the third cycle's screw (screw_4, for insert
     2) 4 mm deeper in the hand than the others and releases it 10.3 mm above its seat, with or without the slew; in the
     old reference it landed meshed on the insert's thread start, under the slew it lands perched on it and leans
     7.3 deg (the thread phase at landing). Cycle 3 is therefore driven closed loop: once the tip is in the head's
     recess, a screw leaning more than 2 deg is pulled upright about its own tip (the tip moved toward where the head
     of an upright screw would be, the driver axis turned toward the insert axis, small IK steps) before the spin:
     7.26 -> 0.07 deg, then seated and welded. (Tried and dropped: lowering before the release, as the fingers meet
     support_1 6.5 mm above the seat and the assembly shifts; a +21 deg turn of the held screw about its axis, which
     meshes but the hand's touch shoves the crossbar 14 mm; a shallower grasp, the screw slips out.) The sixth
     cycle's screw leans 3.8 deg and is stood up the same way (0.68 deg).
   - **`untorque` before every lift:** otherwise the wound-up wings wedge in the recess and the lift yanks the crossbar.
2. **Pads (`pads.py`).** From the checkpoint after the sixth screw cycle (demo row 206,220):
   - **the grasp of a pad on the backing, centred (`centre_hook`):** in the 600 demo rows before the demo closes a
     hand on a pad still on the backing strip, the hand is steered in its own x-y plane so the pad's centre is on the
     TCP's approach axis (true state). Under the slew the demo's grasp of pad_1 arrives 2.8 mm off centre along the
     closing direction (the old reference: 1.5 mm): a finger lands on the 38 mm pad's top edge instead of beside it,
     presses the pad into the backing (6-10 N) and the closing fingers squeeze it out, so no pad leaves the backing.
     Centred, the fingers close on the pad's sides (37.0 mm) and the demo's own pull peels each pad (the backing's
     1.5 N / 0.25 s release gate);
   - the demo's peel and carry rows, each carried pad steered onto the nearest free site (`pad_hook`);
   - **the press, closed loop:** in each pad's press window (from the end of the demo's re-close of the fingers after
     the release on the site, 2000 rows), when a hand touches the pad on its site, the demo row is held and the hand
     is servoed along the site normal on the TRUE pad-support contact force (`M.force`, the scene's own reading) toward
     1.8 N (`PP_KP` 3e-6 m per N per tick, at most 4 um per tick; it backs off if the gap passes -0.06 mm), until the
     scene's adhesive weld latches; held 40 more ticks; the push removed over 200 ticks with the row still held; the
     demo resumes. If no hand touches the pad 1000 rows into its window (the demo's press passes just above it), the
     demo's pressing hand for that pad (left for pads 1-2, right for 3-4) is lowered onto it by the same servo (at most
     6 mm; it stops if the hand meets anything else);
   - **pace:** outside each pad's placement and press (from 800 rows before the demo's release on the site to the end
     of the press window) the demo is followed every 2nd row (`PP_STRIDE=2`; `Rig.move` still keeps every tick within
     the limits). The welded screws keep their thread contacts (27 ms per tick on a cloud host at the end against 5.5 ms at
     the start), so this phase is the most expensive per row. The demo's own pace (stride 1) passes too (230,929 rows,
     4097-4130 s per replay on a cloud host EPYC 9354, 95 % of the 4320 s bar); stride 3 loses pad_2 in the carry.
   - The four presses in the reference (each pad lowered onto by its pressing hand, `touch0` false): pad_1 bonded 425
     ticks after the take-over (max 1.60 N), pad_2 238 (1.80 N), pad_3 530 (1.34 N), pad_4 300 (1.57 N). Every pad ends
     within 0.1 mm / 0.1 deg of its site.

Resuming is exact: a checkpoint's state hash equals a fresh replay of its rows (`t_resume.py`), each checkpoint carries
all its rows from the birth, and the builder is deterministic across CPUs (the end-of-cycle-2 checkpoints of the same
run on Zen 4 (host B) and Intel (Quest) are byte-identical). The reference is `build.py --upto 206220` from the birth, then
`pads.py --ck ck_0206220.pkl` with their defaults. It was made as `build.py` resumed from the end of cycle 2 plus
`pads.py`; the same two commands from the birth reproduce it byte for byte (`7a1e6e50…`). The fresh replays below are the
proof of the rows.

## Gates (the hand-slew reference, 2026-10-18)
1. Frozen birth: every replay builds to the bound `initial_state_sha256` (`3dc8f1b6…`, bindings.json); the scene files
   are assets.json's pins (image 0.3.0). The scene, `success.py`, the gates, the row cap (431,000) and the verifier
   timeout are unchanged.
2. Leak check: `environment/` holds only the tooling (privileged: the Dockerfile and `rcb_ikea.py`; standard: the client
   and the sim service); neither the rows (no `.npz`, no row bytes) nor their sha nor the row count appear in
   `environment/`, `tests/` or the instructions; the scene texts are unchanged (no upstream pointer, no world
   coordinates, table height or base pose).
3. Fresh replays (`check_ref.py`: `rcb_ikea` Session, clamp in, `final_check()`), all success, all ending on the hashes
   above, 0 rows over the arm or hand limits, `slewed_ticks` 0:
   - AMD Zen 4: host B Ryzen 9 7950X in image 0.3.0 x2 (3292 / 3313 s, two at once, host load ~15) and the Harbor
     verifier's two replays (host B);
   - Intel AVX-512: Quest Xeon Gold 6230R in the 0.3.0 SIF x2 (6944 / 7082 s) and the stress checkpoint run (7881 s);
   - Intel AVX2: the i9-14900HX laptop (MuJoCo 3.14.0 from PyPI with the EPA byte patch, numpy 2.5.4), 2649 s.
   - The builder itself is deterministic across Zen 4 and Intel (byte-identical checkpoints), and the final `build.py` +
     `pads.py` from the birth reproduce the reference's sha.
4. Harbor (host B, Harbor 0.23.0, image 0.3.0 / -sim 0.4.0 Nyx; the task dirs of this commit):
   - privileged: oracle reward 1 (deterministic 1, instance_exact 1, both replays `a690aa1b…` / `20a59284…`,
     replay_wall 3324 s, video 750 frames), nop reward 0 (missing trajectory);
   - standard, graded on the live episode (#305): oracle reward 1 (live settled success at control step 222,600 in 56
     calls, live end hash `a690aa1b…` = the reference's, settle 1.0 s success, no resume, live service time 3672 s =
     34.0 % of the budget, video 5566 frames), nop reward 0.
5. Stress: 203 episodes on Quest (Xeon Gold 6230R, SIF 0.3.0; our stress harness with RoboARC's observer v2.13b), 29
   per strategy, seeds 970000-970202, 3-10 s each, from 21 checkpoints of THIS reference (every 5 % of its rows; the
   checkpoint run itself ended on `a690aa1b…`, success).

   | strategy | n | CLEAN | VIOLENT | BREAKDOWN |
   |---|---|---|---|---|
   | random | 29 | 19 | 10 | 0 |
   | prefix_random | 29 | 5 | 24 | 0 |
   | prefix_jerk | 29 | 2 | 26 | 1 |
   | mutate | 29 | 23 | 6 | 0 |
   | fast | 29 | 21 | 8 | 0 |
   | grip | 29 | 29 | 0 | 0 |
   | push | 29 | 27 | 2 | 0 |
   | total | 203 | 126 | 76 | 1 |

   - Non-jerk BREAKDOWN 0 / 174 (the reference before the hand rule: 1 / 174); prefix_jerk 1 / 29 (before: 2 / 29).
   - prefix_jerk s970107 (step targets to random values, no slew): "energy created (local)", 0.62 J within 0.1 s, the
     support_2-screw_5 thread contact giving 0.35 J under a 22.6 kN contact force spike. Confirmed from the birth in a fresh
     process (the reference prefix to its checkpoint, then the episode's rows): the same end state hash `9ea1d1a6…` and
     verdict. The thread-contact energy class of the jerk strategy; a candidate for the observer rules; no physics was
     changed.
   - VIOLENT: force triggers of the random / jerk / mutated targets (arm or finger contact forces), some surface speeds
     over 3 m/s; no breakdown follows.
6. Timing (one reference execution against the 4320 s bar, 40 % of 10800 s): 2649 s on the quiet i9-14900HX laptop
   (61.3 % of the bar, 24.5 % of the budget); 3292-3313 s on host B's Ryzen 9 7950X (two at once, load ~15; 76.2-76.7 % of
   the bar); 6944-7082 s on a dedicated Quest Xeon Gold 6230R core (64.3-65.6 % of the budget: over the bar on that slow
   CPU, as before the hand rule (6253-6286 s) and as gladom and variera); 3672 s for the standard live episode on host B.
7. `check-video` (main's `ci_checks/check-video.py`): both dirs render and collect a replay video.

The gates of the reference before the hand rule (215,782 rows, `e0097d66…`, end `a02c9a18…` / settled `1a8e27ec…`):
the same set, passed on 2026-10-17 (a cloud host EPYC 9354, host A / host B Ryzen 9 7950X, Quest Xeon 6230R / 6338, the i9 laptop;
Harbor oracle 1 / nop 0 in both modes; stress 203 episodes, non-jerk BREAKDOWN 1 / 174), in this file's history.

## Files
- `st_lib.py`: rig (`Rig.move`: the speed limits and the hold for the hand), measurements, IK, `centre`,
  `align_wings`, `straighten`, `drive`, `untorque`, `Follower`, `place_hook`, `tip_hook`, `pad_hook`, `pad_view`.
- `build.py`: the screw builder (`--out DIR [--from-ck CK --start ROW]`; checkpoints at each cycle's s0 / s1 / end;
  `ST_STOP_S0=k` stops at cycle k's s0).
- `pads.py`: the pad phase with the centred grasps (`centre_hook`) and the closed-loop presses (`--ck CK --out DIR [--upto ROW]`; writes `ref.npz` with all
  rows from the birth, `pre_<pad>.pkl` / `bond_<pad>.pkl` checkpoints, and the grader's `final_check()` at the end).
- `check_ref.py REF.npz OUT.json`: the fresh-process gate (rows over the limit, open-loop replay through `rcb_ikea`,
  `final_check()`, wall time).
- `scan.py`, `trace.py`: the phase scan and the demo replays (`NOCLAMP=1` for the study on the author's scene).
  `t_drive3.py`, `t_pad.py`, `probe_pad.py`: drive / pad probes from a checkpoint. `prof.py`: per-tick cost and the
  contact census at checkpoints. `env.sh`: the a cloud host environment.

## History: why the demo does not pass
- 10,824 of the demo's 253,729 rows are over the joint-speed limits (the right wrist j7 at 1.01x, single-row jumps up
  to 55x, one screw carry at 42x).
- On the author's scene (before the prism rule) with no clamp it passes all 17 operations; on the final scene it
  seats only the first screw, through the clamp, `std_reference` centred / track / dilate, and even with no clamp.
  The screwdriver arrives 0.5-2.5 mm off the true screw axis with its wings up to 37 deg off the slots and rides on
  the head; clamped screw carries land off the insert; the re-grip ratchet for insert 4 advances its screw ~1 mm in 47k
  rows.
- The first closed-loop build (2026-10-16) got through the six screws, both joins and the four peels, but the demo's
  pad presses peaked at a 0.99 N windowed mean, under the scene's 1 N / 0.3 s bond gate: hence the closed-loop press.
