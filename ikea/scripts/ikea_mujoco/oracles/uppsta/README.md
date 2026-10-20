# UPPSTÅ closed-loop reference

`uppsta.npz` is the one reference for both modes of `ikea-uppsta-i00` (privileged `solution/trajectory.npz`, standard
`solution/oracle_actions.npz`). The physics is the final one: the prism rule (17 cylinders become 32-facet prisms: 4
dowels, 2 x 2 on the screws, 8 pegs, the hammer), the arm joint-speed clamp at the real Panda limits, MuJoCo 3.14.0 with
the EPA byte patch, numpy SIMD masking, and (image 0.3.1) the welded-group contact switch-off of `seat_weld.py`. It is a
normal `rcb_ikea` trajectory file (`ctrl`, `substeps` 4, `timestep` 0.0003125, `actuators`).

| | |
|---|---|
| rows | 430,225 (537.8 s of simulated time; the demo has 548,882 rows) |
| sha256 | `69cb5194277c444ee28c4efa5f27e95a2e71bde1e8bbfa3915549db37013e6d9` |
| speed rule | 0 rows over the limit + 1e-9 from the previous row (counted from the keyframe ctrl), arm joints AND the Franka Hand (182.14 command units/s in this scene, the pending hand-slew rule), so the clamp never changes a row under either rule; every row within the control range |
| initial state | `3b69cfd5…` = bindings.json (the frozen birth, loaded unchanged) |
| end / settled state sha | `a748a42dc36d7d2a…` / `a2d8a234abc35166…` in every replay below |
| verdict | success: both screws seated and welded, the four dowels, both end blocks and the eight pegs in place; released, at rest, no breakdown (max part contact depth 2.70 mm); the same after the 1.0 s hold |

Only the robot's actions differ from the author's demo; the scene, `success.py`, the gates and the thresholds are the
author's. The one scene-side change is the welded-group contact switch-off (below), which removes only contacts between
two bodies already welded together.

## What made it fit the speed bar (2026-10-17/18)
The previous reference (549,754 rows) took 12,680 s on a Xeon 6230R core and about 6,000 s on a fast host. A profile
from its segment checkpoints showed that the thread is not the cost:
- each end block has 114 convex collision pieces. Seated on the board it makes ~900 contacts with it, before its screw
  is welded and, until now, for the rest of the episode after it. At screw b's drive (workstation, 3000 rows from one state):
  23-26 ms per row as is, 8.2 with the end block's board contacts off, 25.5 with the screw's thread contacts off;
- the Python hooks (mechanism, monitor, validity) are under 1 % of a row; all the time is `mj_step`.

Two changes:
1. **Welded-group contact switch-off** (`images/ikea-mujoco/scenes/uppsta/seat_weld.py`, opt-in per scene, on in uppsta,
   off in stegesund). When a screw welds, its end block is welded to the board at the same moment (the existing clamped
   weld); from then on the bodies joined by latched welds form one rigid group and contacts between two bodies of the
   same group are off. Every other contact stays exactly as before: a check from 4 checkpoints gives the contacts with the
   switch on = the contacts without it minus the in-group pairs, geom pair for geom pair and depth for depth, with equal
   state hashes. Effect after the welds: 1,683 -> 955 and 441 -> 261 contacts; 2-4x cheaper rows. The bit scheme is the
   one of the kyrre switch-off in `ikea_runtime.py`; stegesund's reference keeps its end and settled hashes with the
   new file.
2. **Tempo** (`ref/tempo.npz`, `mk_tempo.py`). Away from every gripper command change (400 rows before, 600 after), the
   controller follows every 4th demo row (`Runner.emit` still splits any step over the joint-speed or hand limit into
   the ticks it needs). Each segment is gated as before; where the tempo try fails its gate the segment falls back to
   every 2nd row, then to the demo's pace (`params.json`: `tempo_max`).

Rows and time (one replay):

| reference | rows | i9-14900HX host | Ryzen 9 7950X (host B) | Xeon Gold 6230R core |
|---|---|---|---|---|
| 2026-10-16 (no switch-off, demo pace) | 549,754 | ~6,000 (est.) | 7,900 (loaded) | 12,680 |
| switch-off + every 2nd row (intermediate) | 458,709 | 4,856 (load 8-10) | 7,954 (load ~50) | 10,461 / 10,525 |
| **this reference** (switch-off + every 4th row + hand slew) | 430,225 | **3,199** | **3,519** (load 12-17) | 8,414 / 8,425 |

## Why the demo does not work as sent
The demo rows (each 2.5 ms tick repeated twice) are exactly 2.0× the per-row limit on 47,544 rows. Two versions are
within the limits:
- `build_ref.py --method clamp`: the demo as the clamp applies it;
- `build_ref.py --method interp`: each tick's target reached over its two rows. This is the guide's row file.

Neither one passes, and the clamp is not the reason. All three fail at the first dowel, at row ~13,100: the demo
unchanged and unclamped, the clamp version, and the interp version.

The right hand closes on dowel_3 lying on the table near its end. `probe_grasp.py` shows what happens next:
- the round dowel rolls into the grasp;
- the 32-facet dowel squeezes out along its axis, and the demo carries an empty hand from then on.

The same interp rows in the scene *without* the prism rule assemble the block completely and pass `final_check()`
(`diag.py --noprism`, 5,911 s on host B). That run is the guide.

## The controller (`uppsta_oracle.py`, `search.py`)
The pattern is vesken's.

**Guide.** Every free body's pose every 8 rows of the no-prism run (`diag.py --noprism --track 8`, not committed:
230 MB). `mksched.py` uses it to build:
- per arm, the demo's grasp intervals (gripper command below 100), each with the part it carries in the guide run
  (`ref/sched.json`);
- the screws' arm and weld events (`ref/events.json`).

**Segments.** Segment k is demo rows [B_{k-1}, B_k). The boundaries are the 30 release rows; there are 31 segments.

**Per arm and row, from the true state:**
- **approach.** For up to 1,500 rows before a grasp, the TCP is offset by where the part really is versus the guide
  (translation, plus rotation when it is under 14°).
- **carry.** After 300 rows of grip, an integral loop holds the part on its guide pose. The pose is relative to the
  board when the part is near it (pegs: within 160 mm). Two hands on one part share the gain.
- **idle.** The offset decays to exactly zero.

**Hex key at a screw.**
- The key is aligned to the true screw's spin modulo 60° (the hexagon). The offset is held once the key is in the
  socket.
- Its tip is servoed onto the hole's axis, which keeps the screw upright while it turns.

**Screws.** A screw's spin error counts modulo 60°.

**Press.** Optionally, a part is pushed further along its insertion direction before its release.

**IK and speed.** Damped-least-squares IK on `left/tcp` / `right/tcp`. Each emitted row moves at most 0.999 × the
limit from the previous applied target (the arm joints, and the Franka Hand with the hand-slew `rcb_ikea.py`), and a
longer step is split. So `Session.applied()` never changes a recorded row.

**Tempo.** With `ref/tempo.npz` the controller takes demo rows i, i+S, ... (S up to 4 from the profile, capped per
segment by `tempo_max`), the guide compared at row i; a step over the limits is split as above.

**Gate (`search.py`).** For each segment, from the previous segment's checkpoint, the default parameters run first.
The 7 alternatives run in parallel only if the default fails. A segment passes when:
- the screws' welds are as in the guide run;
- there is no breakdown;
- every part is within 3 mm of its guide pose. A part released at the segment end is judged 800 rows later (it must
  stay where it was put). Any other part counts the smaller of its end and look-ahead deviations. A part success.py
  already accepts in place counts 0. The hex key and the hammer count a tenth;
- a screw released at the segment end is within 8° of the guide's spin, modulo 60°.

**Kept parameters (`params.json`).** Per segment, from the build (`search.py --params` reproduces it): `tempo_max` 1 or
2 where the 4th-row tempo failed its gate; `press`, gain and rotation-tracking variants where the default failed, as
in the earlier build (dowels that tip off their hole rim after release, screw_a dropping into the thread half a hex step
off the guide, the hex key). Segment 8 (screw b into its hole) passes only with rotation tracking off at the demo's
pace, and its screw lands 21.5° off the guide's spin modulo 60, over the 8° segment gate; the key still meets the socket
in segment 10 (the key is aligned to the true screw's spin), so the build ran segment 8 with `--spin-max 25`.

**Build.** Every segment from the birth, chained through checkpoints, with the hand-slew `rcb_ikea.py` (branch
dev/qineng-ikea-hand-slew: the Franka Hand at 0.1 m/s of finger width, 182.14 command units/s here) so that the rows
also satisfy that rule; `finish.py` writes the file and checks the step rule and the verdict. The same build on an
Intel Xeon (the cluster) and on Zen 4 (host B) wrote bit-identical rows (checked on the intermediate build).

## Gates (2026-10-18)
1. **Frozen birth.** Every replay builds to the bound `initial_state_sha256` (`3b69cfd5…`, bindings.json), loaded
   unchanged; the scene files are assets.json's pins (image 0.3.1).
2. **Leak check.** `environment/` holds only the tooling (privileged: the Dockerfile and `rcb_ikea.py`; standard: the
   client and the sim service); neither the rows nor their sha appear in `environment/`, `tests/` or the instructions;
   `stage_context.pointers()` finds nothing in the scene; the texts give no world coordinates, table height or base pose.
3. **Fresh replays** (`rcb_ikea.py replay`, the grader's path, from the birth), all success with the hashes above:
   - Intel AVX2: the i9-14900HX host (PyPI MuJoCo 3.14.0 + the EPA byte patch, numpy 2.5.4, the 0.3.1 scene files, the
     hand-slew `rcb_ikea.py`), 3,199 s, one process under the shared workstation lock;
   - AMD Zen 4: host B Ryzen 9 7950X in image 0.3.1 (the current `rcb_ikea.py`, no hand slew), 3,519 s at host load 12-17;
     the Harbor verifiers (two replays each, below);
   - Intel AVX-512: the cluster Xeon Gold 6230R, two dedicated cores (SIF 0.3.0 with the 0.3.1 uppsta scene files, the
     hand-slew `rcb_ikea.py`), 8,414 / 8,425 s; the stress checkpoint run (9,405 s).
   So the rows give the same states with and without the hand slew (no row is ever clamped by either rule).
4. **Harbor** (host B, Harbor 0.23.0, image 0.3.1 `b9552184c2de` / -sim 0.4.1 `54694704aa75`):
   - privileged, final dir (row cap 860,000, verifier 25,920 s): oracle reward 1 (deterministic 1, instance_exact 1,
     both replays `a748a42d…` / settled `a2d8a234…`, replay_wall 4157 s, video rendered), nop reward 0 (missing
     trajectory). The row cap comes from the first run (replay_wall 3827 s: oracle 1, nop 0);
   - standard (PR #305 grading, the live episode), final dir: oracle reward 1 (live settled success at control step
     430,225 in 54 calls, live end hash `a748a42d…` = the reference's, settle_failed 0, video), nop reward 0; the first
     run gave the same.
5. **Stress** (the cluster, SIF 0.3.0 + the 0.3.1 uppsta scene files, RoboARC's observer v2.13b): 203 episodes, 29 per
   strategy, seeds 990000-990202, 3-10 s each, from 22 checkpoints of this reference (every 5 % of its rows).

   | strategy | n | CLEAN | VIOLENT | BREAKDOWN |
   |---|---|---|---|---|
   | random | 29 | 20 | 9 | 0 |
   | prefix_random | 29 | 9 | 20 | 0 |
   | prefix_jerk | 29 | 3 | 25 | 1 |
   | mutate | 29 | 16 | 13 | 0 |
   | fast | 29 | 20 | 9 | 0 |
   | grip | 29 | 27 | 2 | 0 |
   | push | 29 | 28 | 1 | 0 |
   | total | 203 | 123 | 79 | 1 |

   - Non-jerk BREAKDOWN 0 / 174; prefix_jerk 1 / 29: s990184, energy created at the left hand's finger-joint limit
     under step targets (0.58 J in 0.1 s; contact force 11.9 kN), the known finger-limit class; end_block_b, knocked off,
     then falls below the table top (the floor does not collide, class `deco`, as in stegesund).
     Confirmed from the birth in a fresh process (the reference prefix to the end checkpoint, then the episode's rows):
     the same state hash `501eefdc…` and the same verdict. No MuJoCo warning, no `rcb_ikea` Validity breakdown; a
     candidate for the observer's limit-energy rule, as stenerik's; no physics was changed.
   - VIOLENT: a force trigger (38), surface speeds over 3 m/s (28) or angular speeds (13) of random / jerk targets; no
     breakdown follows.
6. **check-video** (main's `ci_checks/check-video.py`): both dirs render and collect a replay video; the privileged
   verifier's video and the standard live video were rendered in the Harbor runs.

## Timing
One reference execution against the 4320 s bar (40 % of 10800 s): 3,199 s on the quiet i9-14900HX host (74 % of the
bar, 29.6 % of the budget), 3,519 s on host B's Ryzen 9 7950X (81 %); 8,414-8,425 s on a dedicated Xeon Gold 6230R core
(78.0 % of the budget: over the bar on that slow CPU, as gladom, variera and stenerik). The standard live episode took 4,291 s and 4,685 s of service time on host B at load 15-30 (39.7 / 43.4 % of the
budget), the privileged verifier's two replays running alongside.

## Residuals seen
- Robot self-contact up to 3.8 kN (left/link1 on left/link5, row 93,468), from the demo's own arm motion; not repaired.
- The hex key ends 4.3 mm from its guide pose: it falls when a screw's weld turns off its contacts. success.py does not
  place the key, and it is at rest at the end.

## Files
- `build_ref.py`: the clamp and interp row files of the demo (the guide's rows).
- `diag.py`: a replay with a phase log and the guide track (`--noprism` is a study switch, not the physics).
- `mksched.py`: the guide's schedule. `mk_tempo.py`: the tempo profile `ref/tempo.npz`.
- `uppsta_oracle.py`: the controller. `search.py`: the per-segment search and build (`--params` reproduces it).
- `finish.py`: the reference file, the step rule and the verdict.
- `probe_grasp.py`: the first-dowel study. `tempo.py`: the earlier tempo study (contact-based; superseded by `mk_tempo.py`).
- `ref/`, `params.json`: the schedule, events, tempo profile and kept parameters used. The guide's track (`track.npz`,
  230 MB) is not committed.
