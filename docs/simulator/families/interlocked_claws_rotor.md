# interlocked_claws_rotor on MuJoCo (Hanayama Cast Rotor)

Released later as the base `roboarc-interlocked-claws-12-i01`; this note records its development, from when it was an experimental
setting (statements below about its release status are of that time).

An experimental setting (2026-10-11) of the interlocked_claws family beside its five tiers and cast_abc
(variation id `interlocked_claws_rotor`, difficulty very_hard, pool experimental); the five tiers and cast_abc are
unchanged. The design record (the floating solution, the grasp schedule and the arm placement) and the working files
are not included in this repository. Not in `mujoco_release.json`: Rotor is a commercial Hanayama design, released
only after explicit approval.

## The puzzle and the setting

Hanayama Cast Rotor (level 6, by Kyoo Wong): two identical flat pieces, each threaded through the other. Each piece is a
hub with three spokes; each spoke ends in a T-shaped head whose two horns run along the outer rim, with a slit between
the tips of neighbouring horns and a window between neighbouring spokes. One spoke has a groove along its middle, one is
thin throughout, and the two tips beside one slit carry a notch on one face. A spoke leaves the window it passes through
only through that window's slit, which only the groove and the thin spoke fit. From the official start the pieces come
apart after 13 moves (9 twists, 3 slides over a groove, 1 over the thin spoke), both grooves then passing the slits at
once. The official solution is the shortest one in the open-source state graph of the puzzle (72 states;
github.com/mitkel/hanyama-rotor, its state definitions only) and in the model.

- Geometry (`topogym/assets/geometry/claws_rotor.py`, model v12f): the outline traced from the official top view, the
  thickness rule fitted to close-up photographs (the horns 9 mm thin blades, the T-heads tapered, the groove 10 mm, the
  thin spoke 13 mm, the notches on one face), 100 mm outer radius (the real piece 28 mm), density 600 as every claws
  piece (150 g each). The pieces are written as 4809 exact convex pieces each (strictly coplanar prisms of a triangulated
  outline, re-cut by `topogym.world.mujoco.prisms.fatten`; four slivers under 0.45 mm removed, which only takes material
  away), the thinnest 0.459 mm.
- Collision: profile `claws_v6_rotor` = claws_v6 plus `split`: each piece collides as these convex pieces, so the windows,
  slits, groove and notches stay open. The scene builder's check (with the winding-number recheck of 4d3c3ac28): volume
  9.8e-14, part outside the mesh 0 m, mesh uncovered at most 7.5e-13 m.
- Entities: `piece_0` (the target, the family's brown; in the right hand at the opening; the design record's silver
  piece) and `piece_1` (teal; the left hand's; the design record's gold piece). The names carry no colour words: the
  pieces are painted in the family's colours, not Hanayama's gold and silver. Brown and teal are appearance seed 0's;
  seeds 1 and 2 re-finish both pieces as the family's (`InterlockedClawsTask.piece_finishes`).
- Opening: both pieces in the hands (pregrasp_weld, contact_weld_v1), interlocked at the official start (state −00−00),
  at the reference route's first sample: the gold piece's centre 0.28 m above the table midway between the arms. Each
  hand pinches its piece's rim across the 9.0 mm thickness. The grasp regions cover the whole piece (radius 0.115 m about
  its centre, a hand acquiring with its TCP up to 0.15 m from that centre), because the reference regrasps three times
  at other places on the pieces.
- Goal: the family's whole-piece separation (a plane separating all geometry with at least 10 mm, both pieces at most
  1 mm/s and 0.02 rad/s, held for 0.5 s; holding allowed). Task text: "Completely separate the two interlocked pieces and
  briefly hold both still."

## Reference (-i01)

`scripts/robotangle/oracles/interlocked-claws-rotor-i01/privileged.npz` (not included in this repository; `standard.npz`
is the same file), 21116 rows
(527.9 s), native gripper space, on the instance made on this branch (`rcb_topogym.py make-instance --task
interlocked_claws --variation interlocked_claws_rotor --scene-seed 0 --gripper-space native`; birth 28cecf9c01aadf5e).
Recorded with

```bash
python scripts/roboarc/shared/rcb_topogym.py run-oracle --task interlocked_claws --instance <instance.json> \
    --code tools/references/interlocked_claws/rotor_oracle.py --out <dir>/privileged.npz
```

The episode ends in success at row 21116: final state 87c1e785d178acfc; 2431-2814 s of wall clock on a loaded 128-core
host (two recordings: before and after the pieces were renamed piece_0 / piece_1, the same 21116 actions bit for bit).
No robot speed clamp and no forcing event (an arm at its torque cap pressing what it holds on something). The silver's
pose relative to the gold stays within 6.3 mm of the plan (the largest while the gold is carried back at the first
regrasp). The three regrasps (route sample, hand, where the released piece came to rest, how close the carry back put
it to the route pose): 1125, left, 15.1 mm, 0.26 mm; 1209, right, 4.7 mm, 1.4 mm; 2199, right, 7.6 mm, 0.31 mm.

The oracle code (`tools/references/interlocked_claws/rotor_oracle.py`) follows `rotor_plan.npz`: per 40 Hz
control step the planned joints of both arms (the reference route's two-arm trajectory, time-scaled to at most 0.2 rad/s
per joint, 20 mm/s and 0.2 rad/s per TCP) and the planned pose of the silver piece in the gold piece's frame. Each arm
joint gets its planned target plus feedforwards of its drive's velocity lag and Coulomb friction, and a slow
correction: an arm that only holds integrates its joint error; while both hands hold, the right arm servos the silver's
pose relative to the measured gold through its Jacobian. Three regrasps (the left hand at route sample 1125, the right at
1209 and 2199): both hands first turn the puzzle (96, 81 and 54 deg) so that the released piece slides at most 10 mm and
stops (MuJoCo release scans of the floating solution), the hand opens to the grasp thickness + 20 mm and both arms hold
while the piece settles; the hand retreats and moves along the plan; then, from the piece's measured pose, it approaches
the new grasp along a straight TCP line from 50 mm in front, closes (the assisted grasp acquires on bilateral finger
contact), and carries the piece back the way it slid (its pose relative to the other piece, noted every control step
from the release to the grasp, in reverse) onto the route pose. A straight carry back from where the gold came to rest
(15 mm from the route pose, the release scan's 9.3 mm) pressed it on the silver at the arms' torque caps.

## Checks

Checks of the reference on this branch's code (2026-10-12), on the RoboTangle 0.2.22 runtime python (mujoco 3.14.0;
the image's libraries), on a loaded 128-core host, four at once:
- Two fresh-process strict replays (`rcb_topogym.py replay --repeat 1 --strict`, two processes): success at row 21116,
  both 87c1e785d178acfc (the recording's own final state); 2928 s and 3116 s.
- The stress observer (`tools/stress/replay.py --hold 0`, criteria v2.10): CLEAN; no fail, trigger or pop;
  the fastest task link 0.48 m/s and 2.25 rad/s (the gold piece dropping onto the silver after the first release), task
  contact force with the robot at most 14.6 N and depth at most 0.11 mm (finger pads on their pieces), energy created
  0.0008 J; no forcing event (the largest press 28.3 N at 6.5 % of a drive's cap), no speed clamp, no grasp slip.
- The equal-length nop (21116 rows of the birth targets, under the observer): no success, CLEAN.
- One run without the observer: the recording itself, 2814 s (47 min); the replays above, 49 and 52 min.

The collision geometry, the opening and the reference were checked in the design record first: the floating route
(FCL and MuJoCo at 0.25 mm / 0.125 deg: no contact, closest 0.52 mm), the two-arm trajectory (inverse kinematics within
0.1 mm, every joint at least 0.17 rad from its limits, no arm within 2.2 mm of a piece it does not hold, nothing within 30
mm of the table), and MuJoCo runs with the hands as rigid bodies following the plan and with the ALOHA's own drives.

## Harbor under the template's name (2026-10-12)

A local dev image (sha256:c781c87d3570ba58186faba33983a8db92a1136de42c51ab7cf1163081b9095f), built
with `source/images/robotangle/Dockerfile.mujoco` from d27c622d7 (after this setting's merge d897b2cda). The graded instance
is the one the reference was recorded on (birth 28cecf9c01aadf5e); make-instance inside the image gives the same birth.
`harbor_check.py --tasks --lanes 1`, rendering off (`RCB_RENDER=off`), on a 32-thread host at a load average of about
30. Expected oracle 1, nop 0; all four trials as expected:

| mode | agent | reward | wall | evidence |
|---|---|---|---|---|
| privileged | oracle | 1 | 6509 s | final state 87c1e785d178acfc twice, the recording's on the other host; deterministic, instance exact, no review or forcing event |
| privileged | nop | 0 | 43 s | no trajectory submitted |
| standard | oracle | 1 | 6891 s | live and replay both success, both 5671cc01e7914e63; no review or forcing event |
| standard | nop | 0 | 71 s | no action, ends on the timeout in the birth state |

The standard oracle runs 21 120 control steps, four more than the reference: the standard service moves a joint at most
0.04 rad a step and splits a row that moves one further into two steps. Four rows move one right-arm joint 0.055-0.070
rad (rows 9960-9961 and 15522-15523), about 1.5 s after the right hand opened at a regrasp, where the oracle goes from
holding in place back to following the plan and drops in one step the servo correction it had accumulated. The reference
is kept (the acceptance round below runs it); a re-recording for another reason would ramp that step.

Rotor had no base of its own then: this check rendered the package of the released base `interlocked-claws-easy-i00`
with Rotor's lines replaced (the graded texts were kept outside this repository). Its own base's texts (next
section) differ from them in four places that do not change the grade: the review rule 2 mm (was 1 mm), the objects in
MuJoCo's `s.objects()` order, 0.15 s a step (was the recording host's 0.13 s), and the 60 N rule in the grasping line.

## Harbor under its own base interlocked-claws-rotor-i01 (2026-10-12)

Rotor is made evaluable, not released (2026-10-11): `mujoco_release.json` lists nothing for it. Its texts are
`package_text/interlocked_claws/interlocked-claws-rotor-i01-<mode>/` (1ae6a9d5a, merged as 85b984148; released later as
`tools/release/package_text/interlocked_claws/interlocked-claws-12-i01-<mode>/`),
laid over the released `interlocked-claws-easy-i00` by release_rebuild's new_bases. The base is made on a test-only
commit, d2abf8f9d (1ee40ae14 plus the interlocked_claws entry reduced
to `"tiers": []` and this new base, the entry the package_text README gives; never merged):
`release_rebuild.py d2abf8f9d --version <dry-run version> --families interlocked_claws --lanes 2` (with Harbor,
at most two trials at a time): **PASS**.
- A local dry-run image (sha256:a684b7347f686609…). The
  instance made in its image without instances: birth 28cecf9c01aadf5e, the reference's (instance sha256
  161fa24f1a355324, 4813 files). `mk_tasks.py --check`: every base and the MuJoCo instance bytes ok.
- The texts Harbor graded: the release's `packages/interlocked-claws-rotor-i01-<mode>/` task.toml, README.md and
  instruction.md are the package_text files byte for byte; the rendered tasks are named
  `robotangle-interlocked-claws-rotor-i01-<mode>`, keep no word of easy's scene (the change log names the template, as
  every new base's does), carry `difficulty = "very_hard"` and `oracle = "full"`, and state the 15 cm gate and the 60 N
  rule in the grasping line, the 2 mm review rule and 0.15 s per control step.
- Rendering off; expected oracle 1, nop 0; all four as expected, with the final states of the run under the template's
  name:

| mode | agent | reward | wall | evidence |
|---|---|---|---|---|
| privileged | oracle | 1 | 6314 s | final state 87c1e785d178acfc twice; deterministic, instance exact, no review or forcing event |
| privileged | nop | 0 | 36 s | no trajectory submitted |
| standard | oracle | 1 | 6347 s | live and replay both success, both 5671cc01e7914e63 (21 120 control steps: four rows split, as above); instance and assets exact; no review or forcing event |
| standard | nop | 0 | 59 s | no action, ends on the timeout in the birth state |

The run report was kept outside this repository. The instance
upload and the image push that a release would need were not done.

## Stress acceptance (2026-10-12)

Round `claws_rotor_r1` on d27c622d7 (a lab RTX 4090 host), criteria v2.10, 1,000 episodes over the six strategies, seeds
101001-101017: **PASS**. No confirmed BREAKDOWN under any criteria version (v2 to v2.10); the reference success, CLEAN,
final state 87c1e785d178acfc in both fresh processes (89.2 min under the observer on a loaded host); the nop no success,
CLEAN; 984 695 control steps searched, 126 episodes VIOLENT (cem 78, random 38, prefix 9, periodic 1; reported, not
failed), 359 grasp slips.

## Grasp slips under random actions

A grasp slip is a hold that the assisted grasp releases because its weld pushes or pulls the part with more than 60 N
averaged over 20 ms (the rule the grasping line states, 1cbd0283d). The round's 359 slips are the most of any acceptance
round reported on 2026-10-11 and 2026-10-12, and its search also ran the most control
steps: 0.36 slips per 1,000 control steps. Cast ABC, which also opens with two interlocked pieces in two hands, has 0.53
(229 in 428 832); every other round there at most 0.27 (cast_coil easier, round r1: 83 in 305 158). Cast ABC's family
doc measured where such forces come from ("Grasp slips under random actions": each hand holds one of two interlocked
pieces, and random actions press one piece on the other through the interlock, so the force reaches the welds); Rotor
holds its pieces the same way, with 0.5 mm clearance, and its slips were not split by source. The reference loses no
grasp (the observer replay under Checks: no grasp slip). Nothing was changed.

## Still open

- Not on the release list (2026-10-11): evaluable through the test-only manifest above; listing it, with the
  instance upload and the image, is still to be decided.
- Public release: Rotor is a Hanayama product design; confirm before publishing.
- The pipeline's step 4 (noise: other arm orientations, a release onto the table, small reorientations) is not done.
