# cast_coil on MuJoCo (the five released tiers)

Shared mechanisms are in [MUJOCO_CORE.md](../MUJOCO_CORE.md). This note (2026-10-10) covers the five released
Genesis tiers (very_easy .. hard) on MuJoCo and their -i01 references. The official two-axis redesign
(`cast_coil_v2_very_easy`, `cast_coil_v2_easier`, `cast_coil_v2_easy`, `cast_coil_v2_design`, profile
`cast_coil_official_v3`, its own `OfficialCoilPolicy`; task texts in `tools/release/package_text/cast_coil/`) is
separate work and is not covered or released here.

## The task

Two cast pieces, gold and silver, lie apart on the table (ALOHA, assisted grasp `contact_weld_v1`). The robot assembles
them: the silver in the gold's frame within 1 mm and 0.02 rad of the assembled pose, resting near the table, held for
the stable time. Gold and silver are the pieces' colours at appearance seed 0 and their object names (`coil_gold`,
`coil_silver`); since the appearance seeds v2 (CHANGELOG, "Unreleased") the texts say "the two Cast Coil pieces", and
seeds 1 and 2 re-finish both (`piece_finish`, as cast_keyhole's).
- very_easy (straight_comb): one alignment, then a straight insertion of complementary large teeth;
- easier (offset_baffle): an offset insertion, then the front bar slid into the root pocket;
- easy (baseline_assembly), medium (axial_screw_turn), hard (split_turn_offset): the full coil pieces, with turning and offset
  insertions.

Goals, rules, geometry and difficulty are the released ones.

## Profile coil_legacy_v3 (`topogym/world/mujoco/profiles.py`, `_COIL_LEGACY`, keys `("cast_coil", "cast_coil_<tier>")`)

- The shared rigid contacts (`_RIGID`, the claws v2 class): 2 dt (1.25 ms) overdamped, solimp dmax .99, converged CCD,
  ALOHA hulls of at most 64 vertices. The task asks for Genesis' 1 ms constraint time constant; MuJoCo's floor is 2 dt.
  Genesis' `max_collision_pairs` and `noslip_iterations` have no MuJoCo meaning.
- **The ALOHA finger coupling** (`robot.coupling = ALOHA_COUPLING`, 8cdf9a228): the follower finger on claws v5's stiff
  coupling, the E1 fix every ALOHA family takes (porting guide trap 10). MuJoCo's default 20 ms finger coupling creates
  energy under fast gripper oscillation and lets the follower finger lag 2-4 mm. coil_legacy_v1 lacked it; the
  birth hashes are unchanged (8cdf9a228), the references below are recorded on v2.
- **coil_legacy_v3 (2026-10-11): robot-robot contacts on the soft class** (`robot.self_contact = SOFT_CONTACT`, 5 ms
  overdamped, as claws v4 and keyhole_v4). v2 kept `_RIGID`'s 20 ms class, and the acceptance round on df00ee35b drove
  an ALOHA link more than 10 mm into another robot link: easier random_e79 (a left finger 13.9 mm into the right
  forearm, 283 N), easier random_e141 (a right finger 10.7 mm into its own upper arm) and hard cem_e45 (the left
  hand's two fingers, 12.4 mm), each BREAKDOWN in a fresh process (e79 in two). On v3 the same streams read 3.3, 2.5
  and 2.9 mm, CLEAN, each in two fresh processes with one final state hash. The references touch no robot-robot pair:
  rebuilt on v3, every tier keeps its birth hash, and every
  reference below replays to its recorded final state hash. The shrunk stream is
  `tools/stress/corpus/reproducers/J2_coil_easier_arm_arm_depth.{json,npz}`.
- Collision geometry: each piece is declared as exact disjoint collision boxes (`topogym.assets.geometry.cast_coil`:
  orthogonal CSG partitioned on the coordinate planes), which MuJoCo collides as box primitives: no hull, no CoACD, no
  slivers. The assembled clearances (0.3 mm comb / baffle, 0.75 mm between layers) stay exact.
- The exact constraint diagonal (b1ec43997) applies, as to every family.
- cast_coil renders with nyx_direct (`TASKS["cast_coil"]["renderer"] = "direct"`).

## Instances

Made as the release makes them (`tools/release/mujoco_release.json`):
`rcb_topogym.py make-instance --task cast_coil --variation cast_coil_<tier> --scene-seed 0 --gripper-space native`.
No simulator overrides or task parameters. (The released Genesis instances are native too.)

Scene acceptance on coil_legacy_v2 (nop 400 + 40 hold rows under the stress observer;
`references/cast_coil/geom_check.py` for the geometry, the assembled clearance and the rest):

| tier | birth sha | collision boxes | thinnest | assembled clearance in MuJoCo | nop 400 + 40 | deepest contact | drift after the birth drop |
|---|---|---|---|---|---|---|---|
| very_easy | 1d701378ad18f372 | 10 | 9.9 mm | 0.30 mm, no contact | no fault, no success | 0.025 mm | 0 |
| easier | b12a895ceb892a09 | 9 | 7.5 mm | 0.30 mm, no contact | no fault, no success | 0.095 mm | 0 |
| easy | 7873b4b3907b0d39 | 28 | 0.75 mm | 0.75 mm, no contact | no fault, no success | 0.019 mm | 0 |
| medium | f11ed465d7059125 | 137 | 0.75 mm | 0.75 mm, no contact | no fault, no success | 0.020 mm | 0 |
| hard | 4babdba86c1af954 | 38 | 0.75 mm | 0.75 mm, no contact | no fault, no success | 0.020 mm | 0 |

- The pieces are born 0.5 mm above the table (`initial_floor_clearance_m`) and drop onto it in the first rows; after
  that (from row 40) neither moves. On easier the silver also tips 7.7 mm at birth and then rests.
- At most 2e-4 J created. The 0.75 mm boxes are the bridges across a layer gap; no box is thinner than 0.5 mm.

## References (-i01)

The family's own closed-loop policy, unchanged, run live in the recording Session (`run-oracle`):
`topogym.planning.cast_coil.public_session.run_feedback_policy` (SimplifiedCoilPolicy on very_easy / easier,
CastCoilPolicy on easy / medium / hard; measured piece poses in, EE_ABS out).
`references/cast_coil/oracle_cast_coil.py` would hold the last targets after the policy returns until the episode ends;
no tier needed it (0 hold rows).

Recipe (the RoboTangle image's Python; `RCB_RENDER=off` for the oracle and replays):
```
rcb_topogym.py make-instance --task cast_coil --variation cast_coil_<tier> --scene-seed 0 --gripper-space native \
    --out <dir>/instance.json
rcb_topogym.py run-oracle --task cast_coil --instance <dir>/instance.json \
    --code tools/references/cast_coil/oracle_cast_coil.py --out <dir>/privileged.npz
rcb_topogym.py replay --task cast_coil --instance <dir>/instance.json --traj <dir>/privileged.npz --repeat 2 --strict
tools/stress/replay.py --instance <dir>/instance.json --actions nop:<rows> --out <nop.json>
```

Recorded on 8cdf9a228 (coil_legacy_v2) on a cloud host, as `scripts/robotangle/oracles/cast-coil-<tier>-i01/`
(`privileged.npz`; `standard.npz` is the same file; development names, the tier-named -i01 directories are not
included in this repository). Every oracle succeeds without a review or forcing event.

| tier | rows, released -> i01 | sim s | final state sha | npz sha256 |
|---|---|---|---|---|
| very_easy | 1758 -> 1595 | 39.9 | d4520d1d14b98cc7 | 7512518fd3a3af4a23865c2533199b937e003c08204e52493298798eb9afcd7d |
| easier | 2008 -> 2136 | 53.4 | e2a38dc1a50dce61 | 19f433d80ba831e79e697bd2838dba518deca61d6eceddf953ebcb6b9461cd9a |
| easy | 6502 -> 8764 | 219.1 | b38cbb0dfc841cab | 041d5a8c9f3504b2cf15edd6462944eb2c0d1cbd7954907357cddf22d6ebea23 |
| medium | 7436 -> 8972 | 224.3 | 86ae706d00ac28db | 6140f0eb306c0847bab720602c4abc174e6fbd2f793a34b5af1f0272fcc6996a |
| hard | 6698 -> 8832 | 220.8 | 2b7798806c064723 | 93ef41e06d0bf77ae38f2baa4c31ae2f35026c25b3926f49935e673e180c58ad |

Checks on this tree (fd50b2d21; a cloud host, the RoboTangle runtime, `RCB_RENDER=off`): each instance
re-made with the release recipe; the reference replayed in a fresh process (`rcb_topogym.py replay --repeat 2
--strict`); a nop of the same length through `stress/replay.py`.

| tier | birth sha (= reference's instance_sha) | 2 strict replays | deterministic | nop (same length) |
|---|---|---|---|---|
| very_easy | 1d701378ad18f372 | success at row 1595, d4520d1d14b98cc7 both | yes | no success |
| easier | b12a895ceb892a09 | success at row 2136, e2a38dc1a50dce61 both | yes | no success |
| easy | 7873b4b3907b0d39 | success at row 8764, b38cbb0dfc841cab both | yes | no success |
| medium | f11ed465d7059125 | success at row 8972, 86ae706d00ac28db both (re-checked 2026-10-13, below) | yes | no success |
| hard | 4babdba86c1af954 | success at row 8832, 2b7798806c064723 both (re-checked 2026-10-13, below) | yes | no success |

**Joint-speed rule (2026-10-13).** The standard service moves an arm joint target at most
`JOINT_STEP_LIMIT_RAD` (0.04 rad) per control step (rules 2026-10-03); a row asking for more takes the control steps it
needs. The medium and hard references each had one such row (medium row 8905, hard row 8765, a right-arm joint, about
0.05 rad), so the standard-mode oracle ran one control step longer than the privileged replay and ended in another state
(both successes, in a Harbor run). The references are now the control steps the standard service executed: that row is
split into its two steps (medium 8971 -> 8972 rows, hard 8831 -> 8832), every other row unchanged, no arm step above
0.04 rad. Replayed twice in the privileged grader (a release-candidate image, fresh processes): success
at the last row, medium 86ae706d00ac28db, hard 2b7798806c064723, the standard oracle's end states; both modes now run the
same rows. `meta.speed_rule` in each file records the change; the previous files are in git history.

Evidence (oracle logs, replays, nops, nyx_direct videos, the checks on fd50b2d21) was kept outside this repository.
