# gladom closed-loop reference (Franka Hand slew)

`gladom.npz` is the one reference for both modes of `roboarc-ikea-gladom-i00` (privileged `solution/trajectory.npz`, standard
`solution/oracle_actions.npz`). Its rows are the EXECUTED targets: `data.ctrl` after `rcb_ikea.Session.applied()`, so every
row is within the arm joint-speed limits and the Franka Hand's 0.455357 command units per tick (182.142857 units/s, 50 mm/s
per finger) of the previous row, from the keyframe ctrl, and the clamp never changes a row on replay. Keys: `ctrl`
(186611, 16) float64, `substeps` 4, `timestep` 0.000625, `actuators`.

Only the robot's actions differ from the 0.3.0 reference; the scene, `success.py`, the gates and the thresholds are untouched.

## Why the old reference fails under the hand slew
The 0.3.0 reference (the author's demo time-dilated to the arm limits) closes and opens the hand in 0.3-0.4 s; the slewed
hand needs up to 1.4 s. Holding each row until the hand reaches it (`std_reference.py --grip wait`) is not enough
(3 screws stop 3.6-3.9 mm short). The diagnosis (`diag.py`, `probe_grasp.py`, `probe_part.py`):
- the screws are turned twice: by the fingers on the head (2.5 turns), then with the hex driver from the stand (6-7 turns);
- the slower close leaves the driver 1.1-2.7 mm off the hand axis (the demo's grasp has it centred): its tip orbits the
  socket while j7 spins, and dropped back into the stand off centre it topples, so the later screws are "driven" by an
  empty hand;
- a screw dropped into the stand's regrasp slot by slowly opening fingers catches on them and tips over;
- the demo's drive ends at the gate (gap 0.77-0.82 mm against 0.8 mm), so a small loss of turns leaves a screw unlatched.

## How it is made (`build.py`, one run from the birth)
The demo rows are executed through the clamp plus a left-arm offset that the closed loop leaves behind, all on the TRUE state:
1. **Grip wait:** a row whose hand command the slewed hand has not reached yet is held.
2. **Slot place (`place`, rows 10471 / 37237 / 77587 / 132090):** before the hand lets go of a screw over the slot, the
   screw is steered onto the slot's axis and upright (damped-least-squares IK, pivot inside the slot), then lowered until a
   finger touches the stand (4.8-7.0 mm), so it is set down instead of dropped.
3. **In-hand compensation (`control`):** while the left hand holds the driver, the tip's offset in the TCP frame is measured
   whenever the driver touches nothing but the fingers (frozen while it touches the stand or a screw) and the TCP is moved
   by minus that offset (IK, no rotation), also while j7 spins. The offset is kept until the hand is open again plus 200
   ticks, then ramped out over 400 ticks.
4. **Drive to seat (`seat`, at the end of each demo spin: rows 28908 / 57880 / 103008 / 153438):** if the screw is not
   latched, the row is held and j7 keeps turning at 0.8 x its limit until the true gap is below 0.4 mm, then holds still
   until the retention weld latches; the extra turn goes back to the nearest whole turn once the driver is out of the socket.

Rebuild: `python build.py --out DIR` (the 0.3.0 reference at `GLADOM_DEMO`, `rcb_ikea.py` and this directory on
`PYTHONPATH`, inside the IKEA image 0.3.0); `DIR/ref.npz` is bit-identical to `gladom.npz`
(host B: run `b8`, resumed from a wait-only diag checkpoint at demo row 10000, before any correction acts, and run `b9`
from the birth both give sha256 `35190e10...`; ~5200 s on the loaded host). Its log in the build: all four screws latched (screw_long 9.53 turns after a
489-tick seat drive; screw_bottom 8.21 and screw_ring_l by the demo's own spin; screw_ring_r 8.70 after 137 ticks), frame B
and the ring joined, the tray on the ring (2.45 N).

## Gates
| | |
|---|---|
| rows | 186,611 (466.5 s of simulated time), sha256 `35190e106aeb5c08bd3af3e0875def267ae020531343e33b416442540e8af54c` |
| limits | 0 rows over any arm or hand limit (largest step 1.0000000000016 x the limit, within the 1e-9 tolerance); `slewed_ticks` 0 |
| birth | `700c53dd5fd9fe2b...` = bindings.json, in every replay and in the standard sidecar's startup binding (instance_exact) |
| end / settled | `88048a4ff7757b51...` / `5c2edffa330cf6a3...` in every replay: host B Ryzen 9 7950X (Zen 4) x2, the cluster Xeon Gold 6230R (Intel AVX-512, SIF 0.3.0) x2, the i9-14900HX host (Intel AVX2, PyPI MuJoCo 3.14.0 + EPA byte patch), both Harbor verifier replays |
| verdict | success: four screws latched, frame B and the ring joined, the tray on the ring (2.45 N); released, at rest, no breakdown (max part depth 0.82 mm); the same after the 1.0 s hold |
| Harbor privileged | oracle 1 (deterministic, instance_exact, replay_wall 5091 s, video 750 frames), nop 0 (host B, image 0.3.0) |
| Harbor standard | oracle 1 (live settled success at control step 184,000 in 46 calls, graded by the live episode, video), nop 0 (host B, -sim 0.4.0 Nyx) |
| check-video | `ci_checks/check-video.py`: both dirs render and collect a video |
| timing | 2630 s on the quiet i9-14900HX host (61 % of the 4320 s bar, 24.3 % of 10800 s); 8069 / 8082 s on Xeon 6230R cores (74.8 %, over the bar on that slow CPU, as before: the 0.3.0 reference took 5636 s there); 4550 s on the loaded host B |

## Files
- `gl_lib.py`: the rig (executed rows, driver / screw / hand measurements, TCP Jacobian).
- `build.py`: the builder; `check_ref.py REF.npz OUT.json`: the fresh-process gate (rows over the limits, open-loop replay
  through `rcb_ikea`, `final_check()`, wall time).
- `diag.py`, `probe_grasp.py`, `probe_part.py`: the demo replay with logs and the grasp / release probes.
