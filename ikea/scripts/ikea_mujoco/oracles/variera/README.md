# variera closed-loop reference (Franka Hand slew)

`variera_ref.npz` is the one reference for both modes of `ikea-variera-i00` (privileged `solution/trajectory.npz`,
standard `solution/oracle_actions.npz`). Its rows are the EXECUTED targets: `data.ctrl` after
`rcb_ikea.Session.applied()`. Every row is within the arm joint-speed limits and the Franka Hand slew (0.455357 command
units per 2.5 ms tick, 182.142857 units/s) + 1e-9 of the previous row, starting from the keyframe ctrl, so the clamp never
changes a row on replay. Keys: `ctrl` (243159, 16) float64, `substeps` 4, `timestep` 0.000625, `actuators`.

| | |
|---|---|
| rows | 243,159 (607.9 s of simulated time) |
| sha256 | `2e821fcfb52beacb9a9b357aae59448b4e5df9a3be63523628fbd50ae872f529` |
| rows over a limit (arm or hand) | 0; `slewed_ticks` 0 on replay |
| initial state | `067105ad8f8329ec…` = bindings.json (the frozen birth, loaded unchanged) |
| end / settled state sha | `95134e1fbd5cf0f9…` / `327c16c5f0071724…` |
| verdict | success: both frames under the shelf (lateral 0.29 / 0.32 mm, 0.05 / 0.01 deg), four screws seated in four distinct receivers (13.05-13.10 turns, seat gap <= 0.015 mm), released, at rest, no breakdown; the same after the 1.0 s hold |

Only the robot's actions differ from the old reference; the scene, `success.py`, the gates and the thresholds are
untouched.

## Why the old reference fails under the hand slew
The old reference (the author's demo time-dilated to the joint-speed limits, with the release tail; sha256 `a2653e57…`)
steps its hand commands up to 2.96 units per tick. Clamped in place it fails at once (the frames end at the wrong side).
Re-timed with `grip_wait` (each row held until the slewed hand reaches it) the frames, the shelf and screws 1-2 pass, but
screw 3 never threads and screw 4 falls over. Root cause: frame_2 lands 2.2 mm off after its release, so its receivers sit
2.2 mm off the shelf holes. A side frame (18 x 280 x 157 mm, 180 g) stands on an 18 mm base; the slow opening (0.4 s
instead of 0.1 s) kicks it into rocking (+-4 mm at its origin) and it settles at a random spot.

## How it is made (`build.py`, `vlib.py`)
The old reference is the phase source; the guide (`rec_guide.py --mode noslew --upto 25000`) is the same rows replayed
without the hand slew, where they pass: per source row, the parts' and both TCPs' world poses.
1. **grip_wait:** a row whose hand command the slewed hand has not reached is held (arm targets unchanged) until it has.
2. **Held tracks:** while a hand carries a side frame (source rows 1900-4646 left / frame_1, 8300-10734 right /
   frame_2), the frame's true pose is servoed toward the guide's (world) by damped-least-squares IK through the TCP's
   lever arm (gain 0.02 per tick, capped at half the joint limit); once the frame rests on the bench, only its x, y and
   yaw (no push into the bench). The increments stay in the arm's offset and are ramped out over 400 ticks after the
   release.
3. **Soft releases** of both frames: the hand opens to 5 units below the zero-force command (the servo command whose
   target is the fingers' present opening, 136.2), holds 200 ticks, opens 0.5 mm per finger beyond it and holds 400
   ticks (the fingers stay as stops while the frame settles), then opens to the demo's value. Both frames then land
   within 0.13-0.15 mm of the guide.
4. **Finish drives:** where the demo's screwdriver spin ends with the head not seated (screw 3: 0.47 mm, screw 4: 0.98 mm
   above the seat; screws 1-2 are seated), left joint 7 keeps spinning at 0.0065 rad per tick while the TCP follows the
   screw down by the pitch (IK on joints 1-6, no rotation) and the tip is kept on the true screw axis, with a light
   extra push while the descent stalls, until the seat gap is below 0.03 mm (screw 3: 920 ticks, screw 4: 1900); then
   joint 7's target is set to the joint (zero torque) for 100 ticks before the demo lifts. Joint 7's extra turns are
   ramped to a whole quarter turn (the cross recess) during the transfer.
5. Tried and dropped: release compensation (moving a held frame on the bench by the measured release error before a
   repeated release, `release_search`, `comp=1`) makes the frame rock: off.

Rebuild (the 0.3.0 image, `rcb_ikea.py` on `PYTHONPATH` with this directory):
`rec_guide.py --traj OLD.npz --mode noslew --out G --upto 25000 --ck 0` then
`build.py --guide G/guide.npz --src OLD.npz --out DIR` (DIR/ref.npz; one run, about 5700 s on a loaded Zen 4 core).
`check_ref.py REF.npz OUT.json` is the fresh-process gate (rows over the limits, open-loop replay through `rcb_ikea`,
`final_check()`, wall time).

## Gates (2026-10-18)
1. Frozen birth: every replay builds to the bound `initial_state_sha256` (`067105ad…`, bindings.json); image 0.3.0's
   scene files (`instance_exact` 1 in the Harbor verifier).
2. Limits: 0 rows over any arm joint or hand limit (+1e-9, from the keyframe ctrl), `slewed_ticks` 0 in every replay;
   the standard live run reports `n_clipped` 0 on all 61 calls.
3. Fresh replays (`check_ref.py`: `rcb_ikea` Session, clamp in, `final_check()`), all success, all ending
   `95134e1fbd5cf0f9…` / settled `327c16c5f0071724…`:
   - AMD Zen 4: host B Ryzen 9 7950X in image 0.3.0 x2 (3646 / 3688 s, host load 50-60), the Harbor verifier's two replays
     (3709 / 3731 s);
   - Intel AVX-512: the cluster Xeon Gold 6230R in the 0.3.0 SIF x2 (8024 / 8696 s);
   - Intel AVX2: the i9-14900HX host (MuJoCo 3.14.0 from PyPI with the EPA byte patch, numpy 2.5.4), 2664 s.
4. Harbor on host B (image 0.3.0 / -sim 0.4.0 Nyx, harbor_run_safe.sh):
   - privileged: oracle reward 1 (deterministic 1, instance_exact 1, both replays `95134e1f…` / `327c16c5…`, replay
     wall 3731 s, video 751 frames ending in the verdict's state), nop reward 0 (missing trajectory);
   - standard (graded live, #305): oracle reward 1 (live settled success at control step 243,159 in 61 calls, live end
     hash `95134e1f…`, service time 4274 s on the loaded host, video 1), nop reward 0.
   The runs used the dirs rendered before this evidence was recorded; the re-render changes only the row cap (463,000
   -> 486,000) and the privileged verifier timeout (21,120 -> 26,700 s), both derived from these measurements.
5. `check-video` (main's `ci_checks/check-video.py`): both dirs render and collect a replay video.
6. Timing (one reference execution against the 4320 s bar, 40 % of 10800 s): 2664 s on the quiet i9-14900HX host
   (61.7 % of the bar, 24.7 % of the budget); 3646-3688 s on loaded host B Zen 4 (85 %); 8024-8696 s on a dedicated the cluster
   Xeon Gold 6230R core (over the bar on that slow CPU, as the old reference's 6839 s was).
