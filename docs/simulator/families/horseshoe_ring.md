# horseshoe_ring on MuJoCo

Shared mechanisms (the ALOHA, its finger coupling, the contact classes, the exact constraint diagonal) are in
[MUJOCO_CORE.md](../MUJOCO_CORE.md); this note (2026-10-10) records the five released tiers' references on MuJoCo.

## The task

Two steel horseshoes joined by two chain links, with a removable ring threaded through them, lie on the table. The
ALOHA must free the ring and place it inside the green dashed goal region, keeping the horseshoes and chains entirely
on the system side of a fixed separator (the goal and the separator are frozen with the instance). Assisted grasps
(`contact_weld_v1`) on both shoes and on the ring. Success (`placement_region_v3_grasp_contact_2mm`): the whole ring
inside the goal volume, every remainder point on the system side, the ring quiet, every permanent chain/eye
interlock intact, held for 0.5 s.

Profile `horseshoe_v1` (`topogym/world/mujoco/profiles.py`, `_HORSESHOE`): the claws contacts (claws_v5) and exact
capsule loops. Renderer: nyx_direct (TASKS renderer "direct").

## The oracle

The family's own staged feedback oracle, run live in a MuJoCo Session: the tier's Harbor callback
(`harbor_bridge.catalog.session_callback`: very_easy / easier through `below_easy`, easy / medium / hard through
`horseshoe_ring.public_session`), i.e. `stages.gravity_release_table_pickup_stages` with the tier's hash-bound inputs in
`topogym/planning/harbor_bridge/reference_data/horseshoe_ring/<tier>/` (grasps, approach, reference, fold, support and
the recipe's stage parameters). very_easy, medium and hard (`stages.gravity_release_table_pickup_stages`):
1. grasp both shoes at their outer bends, lift, flip and fold them (object-space fold, closed loop on the measured
   shoes);
2. turn the folded pair for the handoff and let the upper shoe go;
3. the ring falls free (very_easy, easier, easy, medium: then picked from the table and carried into the goal along a
   searched route) or stays hanging on the right hand's fingers (hard: the hand swap below carries it into the goal).

easier and easy (`stages.hang_release_table_pickup_stages`, the recipe's `"stages"`): the one-hand hang below, no fold.

Every stage checks the task's contact limits and loop interlocks on every tick.

## References

`scripts/robotangle/oracles/horseshoe-ring-<tier>-i01/privileged.npz` (development names; the tier-named -i01
directories are not included in this repository); `standard.npz` is the same file. Each was recorded
by `rcb_topogym.py run-oracle` with the Harbor callback (`tools/references/horseshoe_ring/` holds the tools),
in a fresh Session on an instance made on this head (`make-instance --variation horseshoe_ring_<tier> --scene-seed 0
--gripper-space native`; the state hash is the same as in width space). Runs end at the success row (the oracle stops
there). Session-mode stages need a runtime binding next to the instance; for the recordings a dev binding was written
(the evidence tool `bind_dev.py`, not included in this repository).

| tier | instance birth | change on MuJoCo | rows | state hash | file sha256 | 2 fresh replays | nop |
|---|---|---|---|---|---|---|---|
| very_easy | 74238ec9c7fe951a | none | 11519 | 0210823bd203f436 | 7c6cda3e8d4515c8 | same hash, success; observer CLEAN | no success, CLEAN |
| easier | 939581d626de4e1b | the one-hand hang (below) | 5285 | 51d3e2e89e867c11 | 8f425e241376e19c | same hash, success; observer CLEAN | no success, CLEAN |
| easy | b8ca9e310fe2be28 | the one-hand hang with the early shed tilt (below) | 5790 | 9d6f8521ab21a267 | d632bc4ddc74a14d | same hash, success; observer CLEAN | no success, CLEAN |
| medium | 176e999d8e6ce00d | right grasp 180 -> 200 deg; the flip takes 30 s (below) | 13030 | af4b662ab4cd08b6 | ca55e95e24d17f39 | same hash, success; observer CLEAN | no success, CLEAN |
| hard | f416cbe0b5c79222 | hand swap instead of the release in the air (below) | 7817 | 11ee018da5421bd5 | 57aabe42b8b7bcdc | same hash, success; observer CLEAN | no success, CLEAN |

"2 fresh replays": `rcb_topogym.py replay --repeat 2 --strict` in a fresh process; the stress observer's replay
(`tools/stress/replay.py`, criteria v2.10) is a third process and reaches the same state hash. "nop": as
many rows holding the birth targets, under the observer.

**On the 32-facet cylinder prisms (2026-10-12, 64820cfbf).** The instance build makes every cylinder a robot geom
can touch a convex prism of 32 facets ([MUJOCO_CORE.md](../MUJOCO_CORE.md)). horseshoe_ring's wires are chains of short cylinders
(radius 2.8 mm) and spheres (`topogym/assets/geometry/horseshoe_ring.py`, `cylinder_between`): every tier's
`scene.json` lists 152 `cylinder_prisms`, and every birth changed (the old births, in the same order: 144382069257e898,
326271cbacebbb2d, edbf7c52f6b8d465, e037cf2101a49695, 859fbc7b07bae463; the bodies' birth poses in instance.json are
the same to 0.01 mm). The old references no longer reach the goal there (very_easy's, 11569 rows plus 40 hold rows, ends without
success at 865fa5d73ff9cdf2 while the observer reads CLEAN: the run goes off its route). All five were recorded again
by the family oracle. very_easy, easier, medium and hard succeeded with their committed `reference_data` unchanged.
easy did not: its one-hand hang now ends with the ring around the held shoe's eye end, and the recipe adds the early
shed tilt (below). A re-run of easier's callback on the final head gives the same rows and state hash.

Earlier, on the soft contact class at a constant impedance (151874501), the births and very_easy's reference were
unchanged; medium's, hard's, easier's and easy's references were recorded on that physics (hard on head 564748029,
medium on 305767514, easier and easy on 91be8e64c, 2026-10-11/12).

Wall time of a recording on one CPU core: 10-45 min. On the prisms, on a shared host at load about 30 on 32 cores:
very_easy 43 min, easier 11 min, easy 10 min, medium 26 min, hard 15 min. The two strict replays took 28 / 12 / 8 /
32 / 18 min and the observer's replay 22 / 10 / 8 / 20 / 16 min. The ring route search has a 900 s wall-clock
budget, which a CPU-starved host exceeds: very_easy timed out on a box at load 150 / 128 cores.

### medium: the right shoe grasp

The released right grasp (the band's outer bend, 180 deg, flip 1, 16.5 mm opening on the 16 mm band) fails on the
MuJoCo birth: the right finger lands on top of the band and the descent stops 4 mm high (TCP tracking failed in
`descend_to_band`); the planning check now also finds the ring inside the right fingers' clearance at 180 deg. 200 deg
(flip 1, same opening) closes on the band, and the recorded fold passes the offset screen (`grasp_probe.py`).
`grasps.json` and `approach.json` were re-made on the MuJoCo birth (`regrasp_inputs.py`); fold, reference, support and
every stage parameter are the released ones.

On the constant soft impedance the 20 s flip no longer works: the ring, free on the chains, tumbles during the flip to
the other side of the folding pair (from row 1400 on, the two physics' runs part) and is pinched between the closing
shoes (73 N; the right shoe 6-11 mm behind its reference, the fold stops at reference index 2820-2890). Backing the
fold off and repeating it four times stopped at the same place. With the flip over 30 s (`prefix` parameter
`flip_s`) the ring stays clear, and the released hand-off and table pickup follow (the largest link speed 35.4 rad/s).

### hard: the hand swap

After the fold, the tilt and the left hand's release of the upper shoe, the right hand holds the lower shoe by one
leg, the upper shoe hangs from the chain links, and the ring hangs on the right hand's two fingers, already free of the
puzzle: it cannot leave the hand without the hand opening, and opening drops the shoes. The released oracle (left hand
grasps the ring, both hands move, the right hand opens 0.15-0.4 m above the table) let the remainder fall: on MuJoCo a
chain link then spun at 83-124 rad/s with 3.5 m/s at its surface (observer VIOLENT), and lowering both hands first never
gave a fall that was both CLEAN and on the system side (6 full runs, 11 probes, 2026-10-11; the fall is chaotic). The
hand-off is therefore a hand swap (recipe: `support_release` with `hand_swap`, step `drop_ring_by_hand_swap`;
`topogym/planning/horseshoe_ring/swap.py`, `HorseshoePolicy.drop_ring_by_hand_swap`):
1. the left hand takes the lower shoe by its other leg (the band pinched across its thickness from outside the U;
   candidates by ellipse angle, native IK and contacts checked at the measured poses, joint transit planned);
2. the right hand opens; the grasp assist's weld, refused to the left hand while the right one held the shoe
   (`ownership_conflict`), passes to the left hand 0.02 s later. The right fingers come off the band (5 cm along
   -y and -z, 5 cm in +x);
3. the left hand moves the remainder 4 cm in +y: it is then 4.8-6.3 cm inside the system side;
4. the right hand carries the ring on its open fingers over the goal region: the ring's measured box is moved a
   fraction of the way to the region's centre (the full centring is out of the right arm's reach) with its lowest
   point 6 cm above the table, keeping the box at least 2 cm inside the region's border (planned: 3.7 cm on
   2026-10-11, 3.65 cm on the prisms); then the fingers would pitch down 35 deg.

The task's goal is met at the end of step 4, before the pitch, and the episode ends there (row 7817 on the prisms;
the numbers that follow were measured on the 2026-10-11 recording, row 7748): the ring hangs on the open, level right
fingers over the goal, its lowest point 6.1 cm above the table, its box 3.7 cm inside the region's border, its speed
0.98 mm/s; the remainder is in the left hand, 4.8 cm inside the system side. The goal allows this:
`PlacementGoalMixin.evaluate_placement` -> `evaluate_goal_region` -> `topogym.evaluation.task_metrics.placement_candidate`, `reached = all_target_inside and remainder_on_system_side and
quiet and integrity` (quiet: the ring under 1 mm/s and 0.02 rad/s), with `holding_allowed=True` (the task's
`goal_measure`: "whole ring in fixed region; remainder behind separator; holding allowed"), plus the loop interlocks
(`HorseshoeRingTask.evaluate_goal_region`). Nothing asks for the hands to be open or for the remainder to be on the
table. After the episode's end a Session does not step the scene any more (`Session.step` returns True unchanged,
and the stress replay's hold tail only runs when the stream did not end the episode), so no hold follows this state.

Nothing falls in this reference. Observer: CLEAN; on the prisms the largest task-body angular speed 29.9 rad/s and
linear speed 0.75 m/s, the largest task force 112.0 N, the deepest task contact 0.51 mm (2026-10-11: 53.9 rad/s at
0.73 m/s, a chain link in the fold; 108.5 N; 0.54 mm).

### easier and easy: the one-hand hang

The released fold does not work for either tier on MuJoCo (2026-10-10 / 11): in easier the ring's wire lies inside
the open left jaw at the left shoe's outer bend, so the left shoe cannot be grasped there, and the recorded fold fails
from every other grasp; in easy the fold and tilt leave the ring hooked through both shoes' bends, and no release
turn, other tilt or active pickup separated it. Both are solved without the fold, as the tiers' own design describes
(`topogym/tasks/horseshoe_ring/below_easy.py`: with one shoe held up the chains hang along one axis, and the ring
comes off over the lower shoe where its bore holds that shoe seen along the axis; easier 146 x 136 mm, easy
142.6 x 135.9 mm, the shoe a bar of about 138 x 25 mm):
1. the right hand takes the right shoe with the recorded grasp and lifts it 0.32 m (`stages.HangPolicy`); the left
   hand stays at home. The chain links and the left shoe hang from it, and the ring, which at the birth lies around
   both links and a shoe's eye end (easy: the right shoe's, below), hangs on the hanging shoe (it does not slide off
   by itself). The policy waits until the ring and the hanging shoe are quiet. easy, on the cylinder prisms: the
   early shed tilt during the lift (below);
2. the left hand grasps the ring (`ManipulationPlanner.ring_grasp` with a screen: first a grasp from which a direct
   way to a descent start exists, else one from which the descent alone is reachable);
3. it brings the ring level around the links just above the hanging shoe, its long axis along the shoe's width
   (turns of 0, +-15, +-30 deg about the hanging axis are candidates), then lowers it along the hanging axis past the
   shoe's bend (`hang.plan_ring_pull_over`: the links are not screened, they pass through the ring; the hanging shoe is;
   the way to the descent's start is the direct swing, the screw motion, else a joint-space search; one continuous
   motion at 1.5 cm/s). The ring must then be 1 cm clear of every other piece;
4. the ring is carried into the goal along a searched route (`place_ring`, as in the other tiers).

The right hand never lets go and nothing falls (observer on the prisms: easier's largest task-body angular speed
5.7 rad/s; easy's 13.2 rad/s at 0.56 m/s, when the ring slides off the held shoe's eye end in the shed tilt; on
2026-10-11/12 6.0 and 10.4 rad/s). The planning of steps 2 and 3 uses the exact IK: the below-easy IK scope's
near-pose fallback (`oracle_ik.suspended_scope`) made easier's grasp search take over an hour.

### easy on the cylinder prisms: the early shed tilt

At easy's birth the ring lies flat on the table around the right shoe's eye end (the open end of the U, where the
chain links pass through the eyes) and both links: those bodies' vertices lie inside its bore. On the cylinders the
rising eye end left the ring behind: it slid off back to the table and then hung from the links beside the left shoe,
its centre 1.10 m high (the earlier reference, between its rows 600 and 700; evidence tool `diag_lift.py`). On the
prisms the rising end carries the ring up with it: the ring ends hanging around the held shoe's eye end, its top arc
7 mm above the shoe, its centre 1.195 m high, 9.5 cm above where it hung before. The oracle then grasps the ring
there, and no way of the pull-over leaves that pose: every direct swing or screw presses the ring 0.6-0.9 mm into the
held shoe within its first millimetre (the plan admits 0.5 mm), and the joint-space search ends without a route
within its budget (5000 iterations, 50,000 queries). A lift of 0.26 m or a 30 deg tilt at the full height leaves the
ring where it is.

The recipe's hang therefore makes the shed tilt early (`HangPolicy`, prefix `hang` options `shed_tilt_deg`,
`shed_yaw_deg`, `shed_lift_m`, `shed_hold_s`; off unless a recipe sets them; the hard tier's development had the same
tilt, removed in f36ee0f73): at 0.12 m of lift, while the ring still lies on the rising end, the right hand turns the
shoe -90 deg about the vertical through its grasp at 10 deg/s, pitches its eye end 30 deg down about the grasp point
at 5 deg/s, holds 4 s, levels it, turns it back, and lifts the remaining 0.20 m. The ring slides off the end onto the
links and hangs as on the cylinders (`hang_probe.py`: its box 0.107-0.247 m over the table, before 0.109-0.249 m; the
held shoe no longer passes through it; observer CLEAN). The pitch alone fails the IK at about 10 deg (the wrist), and
a +90 deg turn runs out of reach partway; with 40 deg at 0.16 m the ring falls clear of everything onto the table,
which the pull-over recipe does not expect. The rest is unchanged: the ring grasp, the swing over the hanging shoe and
the placement (this run: the direct swing).

## Tools (`tools/references/horseshoe_ring/`)

```bash
R=tools/references/horseshoe_ring
# try shoe grasps from the birth (oracle's own opening sequence), then screen the recorded fold with each
python $R/grasp_probe.py <instance.json> <tier> probe.json --side right --angles 200,160 --openings 0.0165
# write a passing trial into a recipe directory (grasps.json, approach.json re-made; recipe sha256 updated)
python $R/regrasp_inputs.py <instance.json> <recipe dir> probe.json <trial index>
# hard: the hand swap from the 2026-10-10 reference's row 6440 (upper shoe let go, left hand away) under the stress observer
python $R/swap_probe.py <instance.json> <2026-10-10 privileged.npz> 6440 out.json '[{"remainder_shift_m": [0, 0.04, 0]}]'
# the one-hand hang (easier, easy): the right hand lifts its shoe, the other shoe hangs on the chains; end state pickled
python $R/hang_probe.py <instance.json> <tier> out.json --lift 0.32 --views out
# easy on the prisms: with the early shed tilt (recipe hang shed_tilt_deg / shed_yaw_deg / shed_lift_m)
python $R/hang_probe.py <instance.json> easy out.json --lift 0.32 --shed 30 --yaw -90 --shed-lift 0.12
# the staged oracle with a recipe of one's own (HS_RECIPE, HS_ORACLE_OUTPUT), before it goes into reference_data
python scripts/roboarc/shared/rcb_topogym.py run-oracle --task horseshoe_ring --instance <bound instance.json> \
    --code $R/dev_oracle.py --out ref.npz
```

The references themselves run the Harbor callback (`session_callback`) with the committed reference_data.
