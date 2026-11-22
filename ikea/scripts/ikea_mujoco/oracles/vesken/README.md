# VESKEN closed-loop reference

`vesken.npz` is the reference of `roboarc-ikea-vesken-i00-{privileged,standard}` under the final physics (prism rule, eket's
fix 3b, MuJoCo 3.14.0 with the EPA byte patch, numpy SIMD masking) and the shared speed rule: the arm joint-speed
clamp and the Franka Hand slew (`actuator8` at most 0.455357 units per 2.5 ms tick, 50 mm/s per finger; Franka Hand
manual 220010/1.2/EN 5.1). It is a normal `rcb_ikea` trajectory file (`ctrl`, `substeps` 4, `timestep` 0.000625,
`actuators`). Rebuilt 2026-10-18 for the hand slew (the 2026-10-16 reference, `32040cc6…`, fails 0/23 under it).

| | |
|---|---|
| rows | 389,504 (973.8 s of simulated time; the demo has 344,359 rows) |
| sha256 | `20c14565036f2ea361047d6b9c8a11afc39928758aaa1c0120280fac798e3fef` |
| rows over the speed limits (arm and hand) | 0 against the previous row, starting from the keyframe ctrl; 0 outside the control range; `slewed_ticks` 0 |
| end / settled state sha | `943aa32ee13e5b97…` / `29685e37f6feb2b9…` (fresh processes on Zen 4 and Intel, identical) |
| verdict | success: 23 / 23 operations passed, released, at rest, no breakdown; max part contact depth 2.071 mm |
| robot–robot contact | at most 6.4 N (finger on finger) |

## The controller (`vesken_oracle.py`)

The demo is used as a phase guide. Its unclamped run is the reference run (`ref/`):
- `plan.json`: per operation, the release row H, which the mechanism's own weld event gives.
- `sched.json`: per arm, the gripper intervals of the demo ctrl, and the part each interval carries in the reference run.
- `eref_free.npz`: per operation, the connector error of the reference run.
- `pos_free.npz`: every part's pose every 20 rows. It is 16 MB and not committed; `tools/posrec.py` regenerates it.

Operation k is demo rows [H_{k-1}, H_k). The controller reads the true state on every row, and each arm does one of
four things:
- **approach**: up to 1,500 rows before a grasp, the TCP is offset by where the part really is compared with the
  reference run. The offset covers translation, plus the part's rotation when that is under about 14°. The hand
  then closes on the part the same way it did in the reference run.
- **carry**: during a grasp interval, an integral loop moves the TCP's position and orientation offset so that the
  carried part follows its reference pose. Near its host, the error used is the connector error relative to the host
  (the mechanism's own geometry) minus the reference run's error. When two arms hold the same part, each gets half
  the gain.
- **idle**: the offset decays to exactly zero, and the demo row is then used unchanged.
- **weld**: before the release row, the controller keeps tracking on the last row until the operation's weld is on
  (at most 1.5 s). If the part is seated but its contact force is below 0.1 N, the hand presses 0.1 mm deeper every
  20 rows (`seek`, at most 3 mm). This is the "press until the snap is seated" step: a part that is exactly seated
  but floating misses the gate's 0.03 N contact force.

Offsets become joint targets through damped least squares IK on `left_tcp` / `right_tcp` (`mj_jacSite`). Each emitted
row moves at most 0.999 × the speed limits (arm joints and the hand) from the previous applied target. A longer step is split into the ticks
it needs, so `Session.applied()` never changes a recorded row.

One demo motion is replaced (`DEMO_PATCHES`). On its way from home to cap_3 (rows 324,600-325,600), the left arm folds
j4 to -3.03 rad with the wrist bent, and the hand hits left_link2: 4.9 kN in the unclamped demo, 6.9 kN in the
original reference. That motion is replaced by a smooth straight joint-space path between the same two end
configurations.

### Additions for the hand slew (2026-10-18)
The slower hand changes how the parts box empties: some parts land turned or shifted (foot_1 23 mm off, caps leaning
the other way), and the full-rotation tracking and the approach rotation then swing the arms into parts the other arm
holds. Every addition below is a parameter; its default keeps the old behaviour unless stated.
- `rot_mode` (default `tilt`): a one-connector part (foot, column, cap; square about its outward axis) tracks only the
  tilt of that axis: towards the host's axis near the host (what the snap gate measures), the reference run's
  elsewhere. `sym`: the full rotation modulo quarter turns about the part axis (`sym_rotvec`), also used by the
  approach feed-forward (`ff_sym`, default on).
- `rot_stage` / `stage_decay`: a staging carry (the part is put down again before its own operation) tracks the
  position only and gives the approach rotation back.
- `HOLD_LIFTS`: the right arm lifts tray_0 by (0, 3, 3) cm over demo rows 34,600-36,800 (the part's reference moves
  with it): in the demo the left wrist (left_link7) passes 1.0 mm from foot_0, which hangs under that tray
  (found by a free-run scan for arm links within 8 mm of anything but their own arm).
- `tweaks`: per-operation TCP offset/rotation windows: op 1 rolls the left wrist (-0.35, +0.17, 0) rad over rows
  20,500-21,900 so cap_1 is released leaning the reference run's way (else it leans the mirrored way and its staging
  regrasp in op 20 fails); op 19 rolls it -0.17 rad about x over rows 279,800-282,200 so left_link7 clears cap_1.
- `ff_rot_max`, `ff_rot_rows`, `max_rot`: the approach rotation's size limit, the rows before the grasp in which it
  acts (a hand turned early moves the whole approach path), and the rotation offset's limit (default 0.3 rad).
- `lat0`: near the host the loop targets zero lateral connector error (the reference run's axial progress only).
- `retry()`: only when the weld is still off after the wait: back the part off 3 mm along its axis, let the loop
  re-centre it, press again (twice at most). `ff_yaw` / `ik_warm` exist but are not used by the kept parameters.

## Gates and search (`search.py`)

For each operation, from the checkpoint of the operation before it, `search.py` runs the candidates (16 by default,
`CANDIDATES`; per-op lists with `--op-cands` / `--cands`, recorded in `search_cands.json`) in parallel. A candidate
passes if the operation's weld is on and, at the release row:
- lateral error < `--lat-max` and axis error < `--ax-max` (the welds are soft: a 0.72 mm lateral drifted to 0.82 mm
  by the end; a 1.3 deg column moved its top under the next tray);
- every part not yet placed is within `--dev-max` of its reference-run pose (feet and caps: `--box-dev-max`, and their
  rotation modulo quarter turns within `--box-rot-max`);
- it executed at most `--trial-rows` rows.
The passing candidate with the smallest lateral + axis/2 + deviation/10 is kept. The 2026-10-18 build used
`--dev-max 25 --box-dev-max 60 --box-rot-max 60`, and `--lat-max 0.6 --ax-max 0.5` from op 8 on (op 18: 0.7, best
0.63). Rack columns drift on their own in this op range (the reference run's column_2_3 moves 10 mm too), which is
why the deviation bound is 25 mm. The kept parameters are `params.json`.

## Checks (2026-10-18)
**Rebuild.** `rebuild.py` (a fresh process from the birth, every operation from a restored checkpoint as in the
search) reaches the search's end state exactly: end `943aa32e…`, settled `29685e37…` (2400 s on a cluster Xeon).

**Fresh replays** (`verify.py`, image 0.3.0 `ed950d079b63`): birth state `66aaaacb…` = bindings.json
`initial_state_sha256`; 0 rows over the arm or hand limit, 0 clipped, `slewed_ticks` 0; success 23/23; end `943aa32e…`,
settled `29685e37…` on every host:

| host | CPU | wall per replay |
|---|---|---|
| host B, Docker, 2 processes (loaded host, 4 Harbor trials alongside) | AMD Ryzen 9 7950X (Zen 4) | 748 s, 761 s |
| the cluster, SIF, 2 processes | Intel Xeon Platinum 8592+ | 983 s, 983 s |
| the cluster, SIF, 1 process | Intel Xeon Gold 6230R (2.1 GHz, the slow trial class) | 1822 s |

Against the 4320 s bar (40 % of the 10,800 s budget): 761 s on loaded Zen 4 (17.6 %), 1822 s on the Gold 6230R (42 %).

**Harbor on host B** (Harbor 0.23.0, `harbor_run_safe.sh`, image 0.3.0 `ed950d079b63`, `-sim:0.4.0` a466a594c463):

| Mode | Run | Result |
|---|---|---|
| privileged | oracle | 1: deterministic, 2 replays (792 s), end `943aa32e…`, video rendered (751 frames) |
| privileged | nop | 0 (missing trajectory) |
| standard | oracle | 1, graded by the live episode: live success at control step 388,000, 97 calls, `n_clipped` 0 on every call, service time 1889 s |
| standard | nop | 0 |

`check-video` (`ci_checks/check-video.py`): both dirs render and collect a video.

## Reproduce

All commands run with `PYTHONPATH=<rcb_ikea dir>:<scene dir>`, `IKEA_SCENES` set, and the numpy masking env.

1. Build the reference-run data from the demo (`vesken.npz` of the IKEA tree, the demo's ctrl rows):
   - `tools/annotate.py demo free ann_free` and `tools/zone_rows.py free demo zone_free.pkl`, then `tools/mkplan.py`,
     give `plan.json`;
   - `tools/posrec.py` and `tools/eref.py`, then `tools/mksched.py`, give `sched.json` and `eref_free.npz`.
2. Either:
   - `search.py --dir ck` (with the options above), then `finish.py demo ref ck/op22.pkl out.npz params.json`; or
   - `rebuild.py demo ref params.json out.npz`, which reruns the kept parameters op by op and gives the same rows.
3. Check with `verify.py out.npz` in a fresh process: rows over the limit, then `final_check()`.
