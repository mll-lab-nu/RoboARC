# STEGESUND closed-loop reference (Franka Hand slew)

`stegesund.npz` is the one reference for both modes of `roboarc-ikea-stegesund-i00` (privileged `solution/trajectory.npz`,
standard `solution/oracle_actions.npz`). It is a closed-loop build on the final physics (prism rule, the arm joint-speed
clamp, the Franka Hand slew `GRIPPER_WIDTH_SPEED_M_S` = 0.1 m/s of finger width = 0.455357 units per tick here, MuJoCo
3.14.0 with the EPA byte patch, numpy SIMD masking). Its rows are the EXECUTED targets (`data.ctrl` after
`rcb_ikea.Session.applied()`), so a replay applies every row unchanged. Only the robot's actions differ from the author's
demo; the scene, `success.py`, the tasks' grading and budgets are untouched (stegesund does not opt in to the welded-group
contact switch-off).

| | |
|---|---|
| rows | 379,529 (948.8 s of simulated time) |
| sha256 | `0bfcaf1ce04871feff7577ac6db8207b0887f235f49eb351131f5f12fb334cf1` |
| speed | 0 rows over any arm or hand limit (from the keyframe ctrl; largest step 1.0000000003 x the limit, inside the 1e-9 tolerance), 0 outside the control range, `slewed_ticks` 0 on replay |
| initial state | `917b2bc63ab1ec62...` = bindings.json (frozen birth, loaded unchanged) |
| end / settled state sha | `71688676098e9765...` / `a7da23fa39888317...`, identical in every replay below |
| verdict | success: half_B 0.25 mm / 0.03 deg, shelf 0.28 mm / 0.03 deg, rod 0.32 mm / 0.14 deg, bolts 1-4 seated and welded, upright 0.015 deg; released, at rest, no breakdown, max part contact depth 0.92 mm; the same after the 1.0 s hold |

## Why the old reference fails under the hand slew
The author's demo (sha `2181d216...`, 357,221 rows) ramps the hand at 0.638 units per tick. Under the slew every re-time
(hold for the hand, holds with lag, uniform slow-down) fails at the first step, the half-lap press of half_B onto half_A
(rows ~47,500-56,000); nothing after it can assemble. Diagnosis (`rec_guide.py`: the demo replayed with the hand slew off,
which reproduces the shipped hashes):
- The press is chaotic in the demo itself. From the guide's own state at row 47,500, a 1e-6 rad shift of one joint target
  decides it: 2 of 5 such shifts pass. The left hand holds half_B 200 mm from the crossings; under the push half_B either
  slides in evenly (the guide) or pivots in the fingers about the closing axis and jams 16.6 mm / 3.9 deg off. Any re-time
  lands on a random side; the shipped demo passed by luck.
- After the press, the slower grasps leave sub-millimetre differences (a bolt 0.6-0.9 mm along its axis in the hand, the key
  dropped 8-14 mm off on the table, the frame 0.3 deg in its press fit). The demo's open-loop bolt screwing, key drives and
  handovers do not tolerate them: bolts cross-thread or the key cams out and pushes the frame along the table.

## How it was made (`build.py`, the chain in `recipe/rebuild.sh`)
The demo is the phase guide; its rows run through the clamp plus a per-arm joint offset left by the corrections:
- **grip wait:** a demo row is held until both hands' applied command equals the row's.
- **press:** searched, not steered (integral tracking of half_B winds up against the jam and throws the frame): a tiny
  constant target shift (1e-6 rad on left joint 2 from row 47,500) puts the press on the guide's side; the frame ends within
  0.05 mm of the guide. The carry before it tracks half_B relative to half_A (`recipe/t1_carry.json`).
- **approach windows** (`mk_approach.py`): for 1,200 rows before every grasp after row 120,300, the TCP is steered in the
  target part's frame onto the guide's.
- **bolts:** the carried bolt is held on its guide pose relative to the shelf before it touches the leg, then only its lateral
  position and tilt are kept (spin and advance free, gain 0.005) while the hand screws it in.
- **hex-key drives** (`fixed:<bolt>`): from the key's hover above the socket, the guide's hand path is moved by one constant
  rigid correction (the bolt's actual pose against the guide's; the key's grasp against the guide's), so the key turns on the
  true bolt axis. A drive that stops short would get the demo's last turn repeated (`drive_fix`); none needed it.
- **frame turn** (rows 185,000-221,000): the grasped frame is steered along the guide's world path (gain 0.01).
- **rod:** the carried rod is steered onto its guide pose relative to half_A (gain 0.01) before it is dropped.
- offsets are ramped out after each release, 600 ticks after the hand has opened (stages after bolt 2).
Each stage ends with a checkpoint; the chain of resumed runs and their settings is `recipe/rebuild.sh`. Rebuilt from
scratch with it on the cluster (Xeon, SIF 0.3.0, 2026-10-18): bit-identical to `stegesund.npz` (sha256 `0bfcaf1c...`).

## Gates (2026-10-18)
1. Speed: 0 rows over the arm or hand limits, 0 clamped ticks on replay (`probe.py`).
2. Frozen birth: every replay builds to `917b2bc6...` (bindings.json).
3. Fresh replays (`probe.py`: rcb_ikea Session, clamp in, `final_check()`), all success with the hashes above:
   AMD Zen 4 host B Ryzen 9 7950X (image 0.3.0) x2 (1546 / 1554 s, host load ~22); Intel AVX-512 cluster Xeon Gold 6338
   (SIF 0.3.0) x2 (2450 / 2473 s); Intel AVX2 i9-14900HX host (PyPI mujoco 3.14.0 + EPA patch) x1, 1164 s quiet.
   The builder's own end state (the cluster Xeon) is the same.
4. Harbor on host B (Harbor 0.23.0, image 0.3.0, -sim 0.4.0, `harbor_run_safe.sh`): privileged oracle 1 (deterministic 1,
   instance_exact 1, both replays `71688676...` / `a7da23fa...`, replay_wall 1553 s, video 751 frames whose end state is the
   verdict's) / nop 0; standard oracle 1 (graded live: live settled success at control step 376,000 in 94 calls,
   live end `802cd7d6...`, service time 1938 s, video) / nop 0.
5. check-video (`ci_checks/check-video.py`): both task dirs pass.
6. Timing: one execution 1164 s on the quiet i9 (10.8 % of the 10,800 s budget, under the 4320 s bar); 2450-2473 s on a
   Xeon 6338 core. The verifier's row cap (714,000) and time limit (10,080 s) are unchanged (evidence.json `budget_basis`).

The previous reference (the author's demo, unchanged), its stress record and that record's forensic tools are in git
history (this file before 2026-10-18).
