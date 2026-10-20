# buzz_wire_3d on MuJoCo

The family was ported on 2026-10-09 / 10, and its references were re-planned on the exact constraint diagonal on
2026-10-10. Shared mechanisms (the Panda command-lead bound, the Robotiq's frictionless spring links and
drivers, the static contacts, the exact constraint diagonal) are in [MUJOCO_CORE.md](../MUJOCO_CORE.md).

## References on the Robotiq's springref 2.62 (buzz_v2 profile, from 2026-10-11)

The Robotiq's spring links now keep the MJCF's springref 2.62 and a Robotiq grasp lets go 0.2 s after its fingers leave
the held body (c64294012; [MUJOCO_CORE.md](../MUJOCO_CORE.md), "The Robotiq's spring links"). The references below (b1ec43997)
held the wand at the curled fingertips: on 2.62 none of them succeeds as recorded (very_easy / easier finish 5.25 mm /
7.0 deg and 8.3 mm / 9.1 deg off; easy and hard touch the wire). very_easy and easier are re-planned from the same cut;
they were committed as `scripts/robotangle/oracles/buzz-wire-3d-<tier>-i01/` (development names; the tier-named -i01
directories are not included in this repository).

The finish tool (`buzz_finish.py`) grasps and releases as the buzz redesign's oracle does on 2.62 (`STAGED`,
default on; `BUZZ_STAGED=0`: the earlier behaviour): wide open, the pads hang ~9.7 mm below the TCP and land on the deck
before the wand, so a grasp's transfer stops at the pre-grasp pose, the gripper closes to 0.30 in the air, the hand
descends at 2 cm/s, closes, and lifts the wand 8 mm (`BUZZ_LIFT`); the final approach's check ignores the deck slab
(role `finish_support`) and keeps everything else. A release opens over 1.2 s, holds, backs off at 1.5 cm/s and
settles 8 s (a fast open set the parallel-held wand turning: 11-12 deg after the first release).

```bash
python source/scripts/robotangle/references/buzz_wire_3d/make_ref.py <instance.json> \
    source/scripts/robotangle/oracles/buzz-wire-3d-<tier>-i00/privileged.npz --cut <cut> --finish adaptive --out ref.npz
```

| tier | cut | rows (success row) | final error | quiet window | state hash | file sha256 | observer | fresh replays |
|---|---|---|---|---|---|---|---|---|
| very_easy | 7000 | 10465 | 2.22 mm / 2.55 deg (2 rounds) | 10425-10465 | 21b22f0b895cce9e | f3f858f734e6d910 | CLEAN, no weld without finger contact | same hash (head 9eb1043c5) |
| easier | 7850 | 13094 | 2.04 mm / 7.21 deg (4 rounds) | 13054-13094 | 179f94c738aa3a03 | d5ea04307c2c2c67 | CLEAN, no weld without finger contact | same hash, 2 processes |
| easy | | no reference on 2.62: the released route's release at row 5331 drops the wand onto the wire; from the held wand at row 5320 the tool's partial release touches the wire and its level release finds no collision-free path | | | | | | |
| hard | | no reference on 2.62: the wand touches the wire in the route's traversal grasp (row 3831, both pads at 23-24 N) | | | | | | |

Instances: the gpusearch b0f births (`buzz-wire-3d-<tier>-i00`, birth f5e748b7 for very_easy; a fresh make-instance on
c64294012 gives the same birth). easy and hard need their route re-planned around the parallel grasp (low priority).

**Release (2026-10-11):** easy and hard are out of `source/scripts/robotangle/mujoco_release.json`'s buzz tiers: the released
routes fail on springref 2.62; superseded by the step-3 redesign. Their `-i01` files stayed in the oracles directory
as history (they succeed only before c64294012).

## References before the springref change (buzz v2, on b1ec43997)

`scripts/robotangle/oracles/buzz-wire-3d-<tier>-i01/privileged.npz`; `standard.npz` is the same file (the release
rebuild's references, `source/scripts/robotangle/release_rebuild.py`). A reference here is the released Genesis route's rows up
to a cut, then the family's feedback finish planned live on MuJoCo from the measured state
(`source/scripts/robotangle/references/buzz_wire_3d/make_ref.py`, finish "adaptive"), then hold rows until the episode ends.
Each was planned in a fresh Session on the physics of b1ec43997, on the instances made on 0f8654a49 (buzz births run no
physics: the same birth hashes), and re-made with the repository's tool (the same rows and state hashes):

```bash
python source/scripts/robotangle/references/buzz_wire_3d/make_ref.py <instance.json> \
    source/scripts/robotangle/oracles/buzz-wire-3d-<tier>-i00/privileged.npz --cut <cut> --finish adaptive --out ref.npz
# easy: BUZZ_REACH_DEG=9 in the environment (the final release from up to 9 deg of remaining turn)
```

The tool holds after a finish that reports a failure as well (the earlier version did not): hard's live finish
left the wand at 0.05 mm / 0.4 deg, then gave up looking for another grasp before its 1 s stability window closed; 20
hold rows complete it.

Why re-planned, not re-recorded: on the exact diagonal the wand lands a few millimetres differently at the last regrasp
(very_easy row 7591), so the old finish rows rest it outside the task's tolerance (5 mm / 10 deg): very_easy 8.4 mm,
easier 7.7 mm, easy 32.9 deg. Hold rows cannot fix that; the live finish plans from where the wand is.

Success: the traversal complete, then the wand released on the finish support within 5 mm / 10 deg and quiet for 1 s.
The success row closes that window.

| tier | cut | rows (success row) | final error | quiet window (rows, by the 1 s rule) | state hash | file sha256 | observer | 2 fresh replays |
|---|---|---|---|---|---|---|---|---|
| very_easy | 7000 | 8825 | 3.79 mm / 2.49 deg | 8785-8825 | 890c0796aa564d80 | 1d350a081bb98afd | CLEAN | same hash |
| easier | 7850 | 9671 | 3.80 mm / 2.46 deg | 9631-9671 | 4dfe96c347f17ed4 | a10365a44df0daed | CLEAN | same hash |
| easy | 5600 | 9905 | 3.60 mm / 8.0 deg | 9865-9905 | 0c21d28055e32e81 | d741fdf5e0e64154 | CLEAN | same hash |
| medium | | no reference (blocked: its finish needs a wider grasp family than exists after the traversal; re-tried on the exact diagonal, below) | | | | | | |
| hard | 16800 | 18677 | 0.05 mm / 0.41 deg | 18637-18677 | f41a190a9e421dfc | 5a4446113a2f919c | CLEAN | same hash |

- **easy has 2 deg of margin** (8.0 deg against <= 10 deg; the released reference had 6.9 deg). Its finish makes the
  final release from up to 9 deg of remaining turn (`BUZZ_REACH_DEG=9`): on MuJoCo the turns do not bring the heading
  much closer, so the default finish (4 deg reach, 3 rounds) fails (best 5.4-5.7 mm / 8-18 deg; its last partial release
  touched the wire), and a 10 deg reach passes only at the tolerance (10.0 deg). The tier is probably harder for agents
  than designed: flagged for step 3.
- **medium stays blocked on the exact diagonal** (2026-10-11). `make_ref.py` from the released route's row 7340,
  the traversal end, with the ring around the finish post and the wand 51.9 deg off. Five finish variants (about 2 h CPU on a cloud
  host) all fail:
  - held turn: none exists; every candidate's wrist meets the arches;
  - lower release: none exists;
  - adaptive rounds without a release first: no grasp with a turn;
  - centred release, then adaptive rounds (reach 12 or 15 deg, 5 or 6 rounds): the release leaves the wand 5.22 mm /
    36.6 deg off (5.3 mm / 35 deg before the exact diagonal). Round 0 finds no grasp with a turn: the collision-free
    grid grasps (yaw 210 deg, tilt 60-75 deg) close on air, and 210 / 60 touches the wire.

  Best state: 5.2 mm / 36.6 deg, against 5 mm / 10 deg. The reason is unchanged: after the traversal no grasp family
  can turn the wand the 25-35 deg it lacks, so medium stays flagged for step 3.

## References of the ported easy / medium / hard on springref 2.62 (RoboARC 0.1.0, 2026-10-14)

The released 0.1.0 shipped roboarc-buzz-wire-3d-03 / 04 / 05 (easy / medium / hard) without a reference. Why the
released routes fail on 2.62: their wand path clears the wire kinematically (the commanded joints with the measured
grasp: >= 1-2 mm everywhere), but the 2.62 gripper's pads and followers run into the wire (live contact logs: the
right finger / left follower on the wire just before each touch); slowing the route down does not change the touch row.

Method (floating solution -> grasp schedule; the robot base is the frozen instance's, so there is no base
search), tools in `source/scripts/robotangle/references/buzz_wire_3d/ported/`:
1. Floating solution and grasp schedule: the released route's own pick and deck regrasp (`oracles/buzz-wire-3d-<tier>-i00`),
   with the traversal grip taken 14 mm further out along the handle (the approach rows after the deck release
   re-solved by IK, seeded by the route's own rows); one grip for the whole wire, as in the route.
2. `nplan.py prep|fwd`: per 2 mm route node, the wand's pose from the route joints with the route's grasp; a forward
   search over a hand turn about the wire tangent (through the ring centre) and a ring tilt about the bending axis,
   with the closed gripper >= 2 mm off the wire (mj_collision with a 4 mm margin on a model copy), the deck not
   penetrated, the wand off the wire metal, joint steps <= 0.3 rad per node and a null-space prior toward the route's
   arm configuration; the plan starts from the arm as the prefix leaves it.
3. `ported_oracle.py --plan`: the prefix rows, the plan's joints (<= 0.01 rad per row), a hold, then 1 mm steps down the
   post until the task records the traversal complete; a slow drop; then the very_easy / easier references' finish
   (`buzz_finish.run_finish`, `--adaptive`). Every row goes through the standard joint-step limiter
   (`standard_rate.RateLimiter`, spread mode), so the privileged reference is the standard one too.

| base | rows | end state | file sha256 | final error | fresh replays (eai-roboarc:0.1.0) | stress observer | Harbor (host A) |
|---|---|---|---|---|---|---|---|
| buzz-wire-3d-03-i01 (easy) | 9501 | 0ddfb562ca575b00 | 565180eae29f6f63 | 2.14 mm / 9.81 deg | 2 x success, same hash, 296-300 s on host C | CLEAN (depth 1.74 mm, E 0.015 J) | oracle 1 / nop 0, both modes |
| buzz-wire-3d-04-i01 (medium) | 9219 | bedce75e61d2be24 | e092a8ab59fd604c | 1.88 mm / 8.16 deg | 2 x success, same hash, 322-330 s on host C | CLEAN (depth 1.26 mm, E 0.018 J) | oracle 1 / nop 0, both modes (1025 / 1105 s) |

easy recipe: `nplan.py prep <instance> oracles/buzz-wire-3d-easy-i00/privileged.npz 1746 0.014 <prep.npz> 0.006 0.0`,
`nplan.py fwd <prep.npz> <plan.npz> --dmax 90 --dstep 10 --lags=-60,...,60 --prior-w 0.5` (the -i01 plan A), then
`ported_oracle.py <instance> <out> --plan <plan.npz> --drop-back 0.008 --clear 0.001 --drop-open-s 0.6` and the finish
from the dropped wand, `BUZZ_REACH_DEG=9 BUZZ_ROUNDS=5 ported_oracle.py ... --resume <actions.npy> --resume-rows 5067
--adaptive` (one continuous recording: the resumed rows are the first run's own rows). The easy prep was made before the
prefix IK was seeded by the route's rows (it then continued from the previous row's solution); its prep / plan and the
first run's rows are kept in the development work notes (not included). The angle's margin is small
(9.81 of 10 deg), as the released easy's was.

Status of hard (2026-10-14, no reference yet). medium is solved (above).
- The traversal completes live (traversal_complete, no touch).
- The finish does not reach the target:
  - The drop at the post's foot lands 43-62 deg off (8.5-9 mm); variants tried: open speed, soft open, the plan's
    end states.
  - At the post's foot the hand is under the handle (the ring's normal follows the wire down), so the traversal grip
    cannot set the wand down (held_place_probe.py).
  - Closed-finger pushes about the post turn the wand to about -24 deg and no further: the next push collides with the
    wire's last loop.
  - Top grasps exist only within ~-19..+21 deg of the finish heading. So a landing in that band, or a way past the
    -24 deg barrier, is still missing.
- Second time box (2026-10-14, ~4 h, ideas 1-4), all negative:
  - Wider grasp search, deck-checked (probe_wide.py: side / end / topb, roll, fraction, pitch, depth), at the
    -24 deg rest and at the -49 deg landing: no grasp. Every candidate puts the pads 11-76 mm into the deck.
  - Wider push grid (probe_push2.py / probe_push3.py) at -24 deg, logging why each push fails: of about 62k
    planning attempts, 50k fail on start IK and 12k start inside the wire's metal. The only plannable push (8 deg,
    grip fraction 0.3, 60 deg lean, fingertips 20 mm off the handle) never touches the handle, so the wand stays at
    -23.7 deg (ported_oracle.py PUSH_WIDE=1).
  - Landing sweep (lift the held wand 0-60 mm along the wire, turn it +-60 deg about the wire, then drop): none of
    the turns can be planned; the drops land at -45..-47 deg.
  - Longer extensions past the wire's end (nplan fwd, step 0.06 / 0.10 / 0.15 m): dead ends at s = 2.03-2.12 m.
  - Adaptive finish with release variants (centre / level / lower / held turn): no grasp with a turn.
  Best state so far: 8.5 mm / 23.7 deg (05S_h2), against 5 mm / 10 deg.
