# rope_threading on MuJoCo

The released Genesis family is the ALOHA threading an IPC FEM rope through rings in a set order and direction. It has
three released tiers: easy, medium and hard. very_easy and easier are defined but not released, so they have no
frozen birth. On MuJoCo the rope is the same rigid capsule chain as knot_untying's
(`topogym/world/mujoco/rope_chain.py`; see `knot_untying.md` for the model), using profile `rope_chain_v3`
(`topogym/world/mujoco/profiles.py`). It was ported on 2026-10-10; the working evidence was kept
outside this repository.

## The rope and the scene

- **Rope.** The released rope's material: E 5e6, 100x knot's, with its initial U shape as the elastic rest (a
  "shaped" rod). Links are 5.5 mm, the mesh spacing, giving 200 links and a chord error of 8e-6 mm at birth. The
  rope carries the chain's bend stops, stiff connects and 0.2 mm contact margin.
- **Robot contact: fingers only,** as the release's `rope_robot_contact` "fingers" (IPC robot_collision_links).
  - Collision bits: the rope is contype 2 / conaffinity 3; the robot's geoms outside the finger links move to 4 / 5;
    everything else keeps 1 / 1.
  - Rope-rope, rope-scene, rope-finger, robot-robot and robot-scene contacts stay.
- **Rings R1-R3.** MuJoCo collides a mesh as its convex hull, which would fill the aperture.
  `topogym/assets/geometry/rope_threading.py` `ring_hulls` splits each ring into its exact convex tube segments: 64
  per closed ring and 32 for the open cradle, checked by the scene builder's split check. The stands and bases keep
  their own convex meshes.

## The assisted grasp

contact_weld_v1 on rope links. Every link is a target, and the hand welds the link nearest its TCP; the released
tolerances are kept. Consequences:
- **The pinch must close on the rope.** The released ipc_soft_band_v1 bound the rope with the pads up to 6 mm away;
  a weld needs both fingers on a link.
- **A weld on a rope link is lost to its pose-error limit (25 mm / 0.15 rad) at about 8 N of rope tension.** The
  60 N slip force is never reached.

## References (step 1)

The released route fails open-loop: its pinches leave the pads 6 mm off the rope. The references are the family's
own feedback relay (contact_relaxed_v16, `topogym.planning.rope_threading.public_session.run_feedback_policy`) run
live in the frozen MuJoCo Session and recorded. Two changes apply, both outside the planner, in a development
wrapper (`rt_relay.py`, not included in this repository):
- **The pinch closes the fingers fully,** to the ALOHA's closed position.
- **A measured-keeper regrasp measures the rope only once it is quiet:** ≤ 1 cm/s for 0.2 s, waiting at most 10 s.
  On hard, the keeper measured a rope still swinging at 0.1 m/s after a release, and the hand closed 22-27 mm off it.
- Before the exact diagonal, medium also needed its measured keepers' approach and pinch 6 mm deeper along the
  pinch-to-rope line: MuJoCo's ALOHA grips a rope between its pads only up to ~20 mm ahead of the fingers' midpoint,
  and medium's keeper grasp at s = 1.0 put the rope at 20.5 mm. On the exact diagonal the plain relay succeeds, and
  the deeper variant stops at R2 (below).

The run_feedback_policy check of the simulator options against the Genesis release is skipped: the MuJoCo instance's
options are its own.

The references are committed as `scripts/robotangle/oracles/rope-threading-<tier>-i01/privileged.npz` (development
names; the tier-named -i01 directories are not included in this repository); `standard.npz` is the same file. They were recorded on the exact constraint diagonal (b1ec43997), on the development
tree at 1b0a8960b's code, with instances made on that code. The relays recorded before the exact diagonal (easy 1882
rows, hard 3326) no longer succeed on it, so both were run again; medium, parked before, now succeeds with the plain
relay. All three replay to the same state in two fresh processes.

| tier | rows | relay min | final hash @ success row (two fresh replays) | file sha256, first 16 (privileged = standard) | nops | observer (v2.8) | node gap / robot depth |
|---|---|---|---|---|---|---|---|
| easy | 2675 | 10.3 | 519d64757c5b8c40 @2675 | c589ff0a979785c8 | fail / fail | CLEAN (142 causal whips <= 3 m/s) | 0.021 / 0.93 mm |
| hard | 3345 | 16.8 | a394709f1e029f37 @3345 | 2295d6fb2173a4f0 | fail / fail | CLEAN (154 causal whips <= 3 m/s) | 0.043 / 0.93 mm |
| medium | 3386 | 28.2 | f47a92608d0e31d3 @3386 | 7a736bb517f4c9ba | fail / fail | CLEAN (155 causal whips <= 3 m/s) | 0.042 / 0.93 mm |

- **On ccd4ba7fa** (the soft contact class at a constant impedance; the rings keep the task's contact against the
  robot; rope_chain_v3), the same rows on the same births (re-bound to the development tree), two fresh processes
  each (the cluster, 2026-10-11): easy 5ba000a6bd8b327b @2675 and medium 740d4048045ab81d @3366 succeed; hard no
  longer succeeded with its rows (ends 5c06d5cc4bf31e89 after its 3345 rows) and is re-planned:
  - **hard, re-planned on ccd4ba7fa** (2026-10-11): the same recipe, the plain contact_relaxed_v16 relay through
    the wrapper (`rt_relay.py <instance> <out> rope_threading_hard`: fingers closed fully, the quiet wait
    before measured keepers), run live on the hard birth on the development tree plus ccd4ba7fa's three files (the cluster). 3345 rows, success
    at row 3345; `scripts/robotangle/oracles/rope-threading-hard-i01/privileged.npz` (= `standard.npz`), file sha256
    6c429cfd38608b63. Two fresh replays on the canonical tree at ccd4ba7fa: success at row
    3345, 41632c4dace19971 both; the stress observer (criteria v2.10) CLEAN (144 causal whips <= 3 m/s, robot depth
    0.33 mm, task speed 2.64 m/s). The --station-shift .02 variant gives the same rows; --keeper-deeper .006 fails
    (no collision-checked measured keeper path). The step-2 round on it (RT4 hard, the cluster, 300 episodes)
    replays it twice more to 41632c4dace19971 @3345, CLEAN, and PASSes (0 confirmed BREAKDOWN, criteria v2.10).
- Quiet waits before the measured keepers: easy 3.0 s and 10 s (the cap; the rope still moved at 1.4 cm/s); hard
  4.4, 10 (6.2 cm/s), 2.9 and 1.0 s; medium 3.0, 10 (1.4 cm/s), 2.4 and 2.2 s.
- The nops are a missing trajectory and 400 rows holding the birth command.
- The minutes are the relay's wall time on a cloud host at load 90-140.

- **medium, before the exact diagonal:** with the quiet wait and the deeper keeper pinch the relay got past its first
  failure (the left keeper pinch at s = 1.0) and stopped at the right hand's measured-keeper receive at R2
  (s = 1.08): every grasp candidate collided with R2 in the planner's checks. On the exact diagonal the rope lies
  where the plan expects and the plain relay passes; the deeper-keeper variant still stops at R2 (1487 rows).

- The angular trigger of v2.6 flagged the earlier references VIOLENT through rope links at 112 rad/s; v2.7 counts a
  spin only on a link whose surface moves faster than 3 m/s, and v2.8 a whip only when its link ends above 3 m/s.
- Frames: rope_threading renders with nyx_direct since 11c886249, which exports the glTF rings. The render
  parity check on easy and hard against the Genesis host is bitwise in every buffer.

## Speed clamp and tension guard (rope_chain_v2 / v3, 2026-10-11)

The chain carries knot_untying's speed clamp (`knot_untying.md`, The rope): extra damping on a link above 6 m/s
centre-of-mass speed or 300 rad/s spin, on the excess only, booked as dissipation. The thresholds are about 2.5x the
three references' maxima (2.64 m/s, 113 rad/s), and no reference reaches them: easy, medium and hard replay to their
rope_chain_v1 hashes with 0 activations. It bounds the node connects' energy pumping on a fast-spinning link, found
on knot_untying.

rope_chain_v3 adds knot_untying's tension guard (`knot_untying.md`, The rope) at the same 30 E I / l², which is
~1000 N on this 100x stiffer rope. A weld that slips at 8 N never reaches it, so it guards only against a future pinch;
the references (peak connect tension 3.1 N) replay to their hashes with 0 activations.

## Known stress items

- **Weld-loss snaps.** A rope under ~8 N whose weld is lost while the other hand still holds it whips back. v2.6
  reads rt_pull10 and the wrist twist as redistribution, so causal.
  - rt_pull12 stays BREAKDOWN: about 1 mJ of the snap comes from the stretched node connects' soft-constraint
    energy, which never counts.
  - If the formal search finds such weld losses often, the fix is on the rope side: stiffer connects, or a gentler
    weld release.
- **The angular trigger.** At 60 rad/s it fires on 5.5 mm links turning at a few tenths of a metre per second. A
  link-size-aware form was requested for the stress criteria.

## Rebuilding the instances

An instance is the knot procedure plus the grasp override: `grasp_assist` contact_weld_v1 with a target on every
link, `{"entity": "rope", "link": "000".."199", "center_m": [L/n/2, 0, 0], "radius_m": 0.008}`, and the released
tolerances. The port script (not included in this repository) does this with `--weld-links 0.0055`. The instance assets built on
818debf8a, f92d9f6dc and 1b0a8960b's code are byte-identical. The instances keep the released `gripper_space`
"native", and their births are easy 7c294d48661292a8, medium c45012f92e572106 and hard f5bbde9d87c3b05d.

rope_threading is not in `tools/release/mujoco_release.json` yet. Adding it needs knot_untying's items except the
gripper space (`mujoco_birth`, the frozen rope volume and the render host), plus the `grasp_assist` override.
