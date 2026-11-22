# hedgehog_in_cage on MuJoCo

Shared mechanisms are in [MUJOCO_CORE.md](../MUJOCO_CORE.md); the port (profiles `hedgehog_v2` and medium's
`hedgehog_v1_waisted`, the exact cage partitions) is described in an internal port note, not included in this
repository. This note (2026-10-10) records the instances and the references on the exact constraint diagonal.

**Profiles now (2026-10-11): `hedgehog_v4`, medium `hedgehog_v3_waisted`.** v3 / v2_waisted came with the Robotiq's
springref 2.62 (c64294012). v4 / v3_waisted put robot-robot contacts on the soft class (`robot.self_contact =
SOFT_CONTACT`, as claws, keyhole and coil): a random arm target in the stress round folded the wrist
until link5 pressed the open Robotiq's right coupler, and on the springref 2.62 that contact grew to 166 kN in 9 ticks
(R1); with the soft class it stays VIOLENT (2.6 kN, 1.2 mm),
not BREAKDOWN. The births and every reference below are unchanged on v4: each reference's two fresh-process strict
replays end at the same state hash with success, the stress observer (`replay.py --hold 0`) is CLEAN with success, and a
nop of the same length is CLEAN without success. Their hashes are identical, so none of them touches robot to robot.

## The task

A spiked ball sits in a cage: core 54 mm across, dowel spikes 10.8 mm thick; the cage 135 mm across and 240 mm tall,
with 30 mm collars and five vertical windows between them. The Panda + Robotiq 2F-85 takes the ball out through a
window, then holds or places it inside the goal volume: every spike included, at least 1 mm inside a 249 mm square next
to the cage, below 237 mm above the table, still for 2 s. Holding is allowed (assisted grasp `contact_weld_v1`).

## Instances

Made as the release makes them (`tools/release/mujoco_release.json`):
`rcb_topogym.py make-instance --task hedgehog_in_cage --variation <variation> --scene-seed 0 --gripper-space native`.
very_easy and easier keep their released variation ids (`very_easy`, `easier`); the others are
`hedgehog_in_cage_<tier>`.

Checks:
- **Collision geometry.** Each cage's MuJoCo pieces come from the build's exact partition and its own checks (scene.json
  `hulls`). The thinnest piece is measured on the compiled model.
- **Birth.** The ball's pose in the cage frame equals the declaration on every tier (0 mm, 0 deg). Its clearance to the
  cage is measured twice: by MuJoCo's own distance over the compiled geoms, and exactly (FCL) between the declared
  collision meshes.
- **Rest.** 400 nop rows and 40 hold rows through the stress observer (`stress/replay.py`, `verdict.py`), plus the
  ball's drift over the same 440 rows.

| tier | birth | cage pieces (thinnest) | ball-cage clearance at birth, MuJoCo / declared | nop 400 + 40 | ball over 440 rows |
|---|---|---|---|---|---|
| very_easy | c6f1313410bdf22c | 488, prism partition (1.259 mm) | 0.30008 / 0.30009 mm | CLEAN, no success | 0.31 mm, then at rest |
| easier | ea2b672276d6b731 | 528, prism partition (1.259 mm) | 0.30008 / 0.30009 mm | CLEAN, no success | 0.31 mm, then at rest |
| easy | 93051f105efef2f5 | 594, prism partition (1.257 mm) | 2.60326 / 2.60327 mm | CLEAN, no success | 15.0 mm and 17.7 deg, then at rest |
| medium | a1343ca52df1bf24 | 426, the family's split: volume within 9e-8, uncovered at most 7 nm (2.126 mm) | 0.2 / 0.2 mm | CLEAN, no success | 11.4 mm and 16.9 deg, then at rest |
| hard | e1b0676facb80f8c | 428, prism partition, and the post cylinder (0.325 mm) | 0.49999 / 0.5 mm | CLEAN, no success | 0.50 mm, then at rest |

- **No energy created.** The observer measures at most 5e-5 J of created energy on any tier. The deepest task contact
  is 0.32 mm (easy).
- **hard's 0.325 mm piece.** It is the width of a slab island that hard's top rim declares; no exact partition can be
  thicker.
- **easy and medium start in motion** (known since the port). The ball rolls at 0.2–0.4 m/s in the first rows; by
  the end of the 440 rows it moves at most 3e-6 m/s.

## References (on the Robotiq spring references, 2026-10-11)

`scripts/robotangle/oracles/hedgehog-in-cage-<tier>-i01/privileged.npz`; `standard.npz` is the same file. (Development
names: the tier-named and experimental -i01 directories in this note are not included in this repository.)

**The physics change.** c64294012 (merged as 2f576d681) sets the Robotiq's spring links to the MJCF's
springref 2.62 and lets a grasp go once no finger has touched the welded body for 0.2 s (`robot.SPRING_REFS`,
`grasp.WELD_CONTACT_LOST_S`; MUJOCO_CORE.md, "The Robotiq's spring links"). The births are unchanged. On it the earlier
references (below) still succeed on very_easy and easier, with new state hashes and the same files; easy, medium and
hard do not succeed, with the observer CLEAN: the routes failed, not the physics.

**Re-recorded on the Robotiq spring references.** Only the solution changed; the task, its rules, geometry and tiers did
not.
- **The fingers.** At a given command the pads now close parallel and reach 18 mm farther from the hand than the curled
  fingertips the grasps were made with. In the hand frame, the inner pad faces and the fingertip reach are ±17.4 mm and
  97 mm at the old approach command .6 (springs about 0); now .6 gives ±12.0 mm and 115 mm, and .5 gives ±17.5 mm and
  115 mm. So `recipes.py` approaches and lets go at .5 (was .6), and holds every grasp pose 18 mm farther back along the
  hand's approach axis (`APPROACH_GRIP`, `HAND_BACKOFF_M`); the fingertips end where the grasp design had them.
- **Two errors of the Genesis -> MuJoCo port, in the oracle's own geometry** (`kinematics.py`, `held_oracle.py`). Both
  used Genesis' link indices (link0 = 0). MuJoCo's robot entity has its root body first (link0 = 1). So the base link,
  which stands on the table, was never excluded, and every robot clearance read -1; and the held planner took link6
  for link7 (its wrist envelope check failed). Both now find the links by name. The oracle had not run closed loop on
  MuJoCo before: the earlier references replayed the released rows.
- **medium and hard:** the family's oracle, re-run closed loop (`oracle_code.py`).
- **easy:** the oracle's first grasp (S8) up to its release (row 7300), then the held exit (below) from that state
  (`easy_held_exit.py --release-row 7300`). The oracle's own regrasp of S6 does acquire on this physics, but the held
  transport after it touches the mount (its route keeps 0.1 mm, and the hand lags), and the guard stops the run.

```bash
R=tools/references/hedgehog_in_cage
# medium, hard: the family's oracle, closed loop on the frozen instance
python scripts/roboarc/shared/rcb_topogym.py run-oracle --task hedgehog_in_cage --instance <instance.json> \
    --code $R/oracle_code.py --out ref.npz
# easy: the same run gives the first grasp (it stops at its regrasp's transport); its first 7300 rows, then the held exit
python scripts/roboarc/shared/rcb_topogym.py run-oracle --task hedgehog_in_cage --instance <instance.json> \
    --code $R/oracle_code.py --out stage1.npz
python $R/easy_held_exit.py <instance.json> stage1.npz ref.npz --release-row 7300
```

Run them with this checkout first on the import path, as the stress tools do (`tools/stress/_env.py`;
in a RoboTangle image, `PYTHONPATH=<checkout>` and `TOPOGYM_SOURCE` pointing elsewhere). Each closed-loop run was made
twice, with identical rows. Every reference passes two fresh-process replays as the graders replay
(`rcb_topogym.py replay --repeat 1 --strict`, one process each), the stress observer (`replay.py --hold 0`) and a nop
of the same length:

| tier | rows | state hash | file sha256 | 2 fresh replays | observer | nop (same length) |
|---|---|---|---|---|---|---|
| very_easy | 1923 | 65295842c0910fad (was 7989dc1560d60b24) | c3711b2f3f382cdb (unchanged) | same hash, success | CLEAN, success | CLEAN, no success |
| easier | 1882 | 4c443ab219eeb740 (was 4490b2bdd5cb8dbf) | fa0fa4c45427439a (unchanged) | same hash, success | CLEAN, success | CLEAN, no success |
| easy | 8255 | aa3edd84fa14f563 | a3f4aecf7397e7e2 | same hash, success | CLEAN, success | CLEAN, no success |
| medium | 2600 | 6a39e39054194172 | a6656cb8c6b8e07a | same hash, success | CLEAN, success | CLEAN, no success |
| hard | 6239 | b03a22cba8259202 (was b348d3467de4fd78, see below) | 265c70c01f61e104 (unchanged) | same hash, success | CLEAN, success | CLEAN, no success |

hard's end state on the 32-facet prism rule (2026-10-13): hard's cage carries one cylinder geom, the fixed internal
post the star stands on at the start (r 8 mm, length 125.5 mm; the other 428 cage pieces are meshes). Since the rule
(`b2b751580`, also as `64820cfbf` on 2026-10-12) builds a cylinder the robot can collide with as a 32-facet
convex prism, the post is a prism in every instance made since. The birth is unchanged (e1b0676facb80f8c: the star
settles on the post the same way), but the star's contacts with the post differ while it moves, so the unchanged rows
now end on b03a22cba8259202. Checked on 126cf1b6c (the merge of the remote): two fresh replays on that hash with
success, the stress observer CLEAN, the nop CLEAN and not successful,
and Harbor under the release rebuild, both modes, oracle 1 and nop 0, both ending on that hash (a local run,
2026-10-13). The 1,000-episode acceptance round on the prism instance (a cloud host, round m126_hedgehog_hard_round on 126cf1b6c,
seeds from 108001): **PASS**, 559,018 controls, 0 confirmed BREAKDOWN under every reading v2-v2.12 (466 candidates
replayed in fresh processes: 293 VIOLENT, 173 CLEAN), 29 grasp slips, 0 pops; the reference CLEAN on b03a22cba8259202.

### Before: on the exact constraint diagonal, springs about 0 (superseded)

Recorded in a fresh Session on the same instances (a development branch merged with e75390aab),
with `rerecord.py` (very_easy, easier, medium, hard: the released oracle's rows, the -i00 references, then hold rows of
the last row) and `easy_held_exit.py` (easy: the released rows to their release, row 7126, then the held exit).
An independent check of the port's references on that physics ended at the same four hashes.

| tier | rows, released -> i01 | state hash | file sha256 |
|---|---|---|---|
| very_easy | 1923 -> 1923 | 7989dc1560d60b24 | c3711b2f3f382cdb |
| easier | 1882 -> 1882 | 4490b2bdd5cb8dbf | fa0fa4c45427439a |
| easy | 9564 -> 8067 | 6cc88aeef56550b6 | 4ea4aef02bc7d302 |
| medium | 2261 -> 2261 | 573e9c3732413776 | 8a2fd2a7bad2e0c3 |
| hard | 6029 -> 6029 | 0bfc9bfada7c3097 | b5e9ad4e8c7db54b |

## easy: the held exit

**Why the released route fails.** It opens the gripper at row 7126 with the ball 88 mm from the cage axis and three
spikes still in the 0 deg window, 25–37 mm deep. That window is a 60 mm slot between parallel walls, 180 mm tall. Two of
the three spikes point 30 deg to either side, and their tips reach the slot walls. A horizontal pull jams: it presses one
side spike against its wall (10–40 mm into the pull, the overlap reaches 9.8 mm). Lowering is free: the clearance stays
0.37 mm down to the mount. So the released route drops the ball 0.1 m down the slot, and the ball comes out only by
rolling on the mount. On the exact diagonal it lands 31 mm from where the released regrasp (row 7615) closes. The hand's
approach then pushes the ball off the mount, and the regrasp closes on air.

**The route.** `easy_held_exit.py` keeps the grasp instead. From the measured state at row 7125 it takes the ball pose,
the radial direction u (from the cage axis through the ball centre) and the grasp (the ball in the TCP frame), then:
1. **Shift** the ball 3 mm toward the cage axis and 2 mm sideways. Its clearance to the slot grows from 0.37 to 2.6 mm.
2. **Roll** it -85 deg about u, through its centre. The side spikes' sideways spread turns into a vertical one, which
   the slot's 180 mm height admits. The shift of step 1 is kept to 30 deg and returned to zero by 60 deg.
3. **Pull** it 70 mm out along u.
4. **Carry** it past the cage on its -y side, then over the goal volume's centre, then lower the ball centre to 1.04 m.
5. **Hold** until the episode ends.

Each step is sampled densely (at most 1 mm or 1 deg per sample). The hand follows by IK, seeded from the previous
sample, and the arm moves at most 0.006 rad per control step. Rolling the other way (+u) also frees the ball, but the
hand then swings into the mount and the table.

The route, checked on the row-7125 state (`--check`; kinematics only):

| segment | ball clearance (cage, mount, table) | robot clearance | largest joint step between samples | joint-limit margin |
|---|---|---|---|---|
| shift | 0.78 mm | 30 mm or more | 0.002 rad | 0.94 rad |
| roll | 2.61 mm | 30 mm or more | 0.042 rad | 0.95 rad |
| pull | 7.0 mm | 30 mm or more | 0.005 rad | 1.15 rad |
| carry | 20.2 mm | 30 mm or more | 0.003 rad | 0.93 rad |

IK residual: at most 0.03 mm. Recorded: 8067 rows (the released route needs 9564), success; no review event, no forcing
event.

On the Robotiq spring references (2026-10-11) the route starts from the family's oracle instead of the released rows:
its first grasp of S8, extracted to the same intermediate pose, up to its release at row 7300 (the released rows open
at 7126). The same five steps from that state: 596 samples, IK residual at most 0.029 mm, largest joint step between
samples 0.056 rad. Recorded: 8255 rows, success; no review event, no forcing event.

## window_change (experimental; its own base, hedgehog-in-cage-window-change-i01)

`hedgehog_in_cage_window_change` has its own oracle (`topogym/planning/hedgehog_in_cage/window_change.py`): three grasps
and two set-downs. On the exact constraint diagonal with the springs about 0 it took 2683 rows, ended at
56eff074da9fcdc6, and passed two fresh replays; a nop of the same length did not succeed. The design note is
internal and not included in this repository.

On the Robotiq spring references (2026-10-11) that reference no longer succeeded (the stress validation's replay ended at
20497bfcbfc054e5, observer CLEAN), and the plan (`window_change_plan.json`) was re-fitted; the oracle's code is
unchanged. The set-downs' landings are this design's sensitive point (the design note says so): with the new fingers
the first set-down landed 19 mm and 33 deg from where the plan's second stage starts (S10 then lay inside the cage,
out of reach), and the second one now rolls off the mount onto the table.
- **All stages:** the tiers' two finger changes: the approach at the pad opening the old command had (old native .649
  and .6 -> new .5375 and .5018), and each grasp 18 mm farther back along the hand's approach.
- **Stage 1 (S8):** the held route goes on 20 % of the way from its old release toward the plan's old landing (6.9 mm
  down, 5.5 deg) before the hand opens (holding the ball all the way down is out of the arm's reach), and the hand
  then backs off 40 mm straight up (along its approach the arm reaches only 2 mm).
- **Stage 2 (S10):** re-fitted to that measured landing: the offset to the planned route blended out over its first
  10 poses (the ball keeps 0.80 mm from the cage, mount and table once off its rest), the grip turned 180 deg about the
  hand's approach axis (the same pad contact; the original turn ran joint 7 into its limit), new rows to the pregrasp
  (HedgehogIK.plan_free_tcp, 2 mm clearance) and approach, and a 40 mm straight back-off after the release.
- **Stage 3 (S5):** the ball now lies on the table, its centre inside the goal square but not all its spikes (no
  success there). The stage grips S5 there (turned 165 deg about the spike), lifts the ball 30 mm, carries it over the
  goal region's centre and lowers its centre to 110 mm above the table, then holds (the old stage pulled it off the
  mount).

The instance is made in the width gripper space (the oracle commands widths), and the reference is run-oracle code:

```bash
python scripts/roboarc/shared/rcb_topogym.py make-instance --task hedgehog_in_cage \
    --variation hedgehog_in_cage_window_change --scene-seed 0 --gripper-space width --out <dir>/instance.json
python scripts/roboarc/shared/rcb_topogym.py run-oracle --task hedgehog_in_cage --instance <dir>/instance.json \
    --code tools/references/hedgehog_in_cage/window_change_code.py --out <dir>/privileged.npz
```

`scripts/robotangle/oracles/hedgehog-in-cage-window-change-i01/` (`mujoco_release.json` new_bases: template
`hedgehog-in-cage-very-easy-i00`, gripper space width, texts in `tools/release/package_text/hedgehog_in_cage/`): birth
2ecb7e00fc766fb8, 2961 rows, state hash 45d1fdb91dfcf298, file sha256 005916f1b7f43f78 (`standard.npz` is the same
file). Two fresh-process strict replays end at that hash with success; the stress observer (`replay.py --hold 0`) is
CLEAN with success (its peaks, 1.2 m/s and 22 rad/s, are the ball rolling off the mount onto the table after the
second set-down); a nop of the same length is CLEAN without success. The offline tools of the re-fit (wc_plan_adapt.py, wc_place_extend.py,
wc_refit.py) are not in the repository, like the design's.

## cast_cage (experimental; its own base, hedgehog-in-cage-cast-cage-i01)

`hedgehog_in_cage_cast_cage` is Hanayama's Cast Cage at four times its real size (the design record is not in the
repository): a closed barrel with four side windows,
a six-spiked star inside, and a push rod lying on the table as a tool. The star starts at rest with spike E 2.7 mm out of
W0, too little to hold; nothing else sticks out. The scene: `topogym/tasks/hedgehog_in_cage/cast_cage.py` (cage turned
120 deg on a 0.24 m mount at (0.05, 0.05) m; the placement study and why the mount, in the design record), profile
`hedgehog_cast_cage_v1`.

Its oracle (`topogym/planning/hedgehog_in_cage/cast_cage.py`, the designed route in `cast_cage_route.json`):
- **Rod:** the hand over the handle, pointing down (the TCP 13.5 mm above the handle's centre: closed, the fingertips
  reach about 19 mm past the TCP and end 3 mm above the table), closed until the assisted weld holds, lifted; the rod's
  pose in the hand as the weld took it is measured and used from then on.
- **Push:** the rod's tip from 60 mm outside W90 straight in to 27.3 mm (real size) past the window's entry along
  (0.256, -0.873, 0.415) in the cage frame, held 1 s, straight back out, at 40 mm/s. The star tips onto the grasp rest
  T, spike U about 34 mm out of W0. In the design's standalone scene with this task's contact model the push reached T
  13 of 13 times from this rest, start moved up to 0.3 mm and turned up to 1 deg (6.6-10.7 N on the rod).
- **Rod back** where it was taken, 1 mm up, then released.
- **Star:** checked at T (within 8 mm / 10 deg of the planned rest, U at least 20 mm out); U held (approach turned 60
  deg about U, the TCP 74 mm from the star's centre); straight on to the route state between 54 and 72 deg whose move
  clears the cage most (64.2 deg here, 0.67 mm), then the designed route to its end (258 deg), every pose through IK
  with the hand-in-star pose the weld took; out along the exit direction, over the goal's centre, down until the star's
  lowest point is 20 mm above the table; held 3 s.
- Free motions are `HedgehogIK.plan_free_tcp` joint paths (4 mm clearance) with the rod lying on the table as an
  obstacle, and while held as part of the hand.

The instance is made in the width gripper space (the default), and the reference is run-oracle code:

```bash
python scripts/roboarc/shared/rcb_topogym.py make-instance --task hedgehog_in_cage \
    --variation hedgehog_in_cage_cast_cage --scene-seed 0 --out <dir>/instance.json
python scripts/roboarc/shared/rcb_topogym.py run-oracle --task hedgehog_in_cage --instance <dir>/instance.json \
    --code tools/references/hedgehog_in_cage/cast_cage_code.py --out <dir>/privileged.npz
```

`scripts/robotangle/oracles/hedgehog-in-cage-cast-cage-i01/` (`mujoco_release.json` new_bases: template
`hedgehog-in-cage-hard-i00`, gripper space width, texts in `tools/release/package_text/hedgehog_in_cage/`): birth ad1fcefad2565341,
2864 rows (71.6 s), state hash 72fac357df0221a0, file sha256 523c1406649f6d0f (`standard.npz` is the same file). Two
fresh-process strict replays end at that hash with success (69 s each); the stress observer (`replay.py --hold 0`) is
CLEAN with success (deepest task contact 0.14 mm, held finger 0.64 mm, no forcing, no review event); a nop of the same
length is CLEAN without success.

## owl (Bird in Cage; experimental; its own base, hedgehog-in-cage-owl-i01)

`hedgehog_in_cage_owl` is the wooden Bird in Cage puzzle at 2.5 times its real size (the design record is not in the
repository): a cage of bars between two discs with a rung across
its widest gap, a wooden owl standing inside with its tail sticking out through that gap, two wing plates on one hinge
that turn together, and a thin rod with a box handle lying on the table as a tool. The owl leaves only through the widest
gap, below the rung, lying on its side; laying it down inside the cage works only with the wings between about 205 and
225 deg, and lying on its side it is narrower than the gap only with the wings between about 250 and 290 deg, so the
wings are set twice (the design's searches: with one wing setting there is no way out). The scene: `topogym/tasks/hedgehog_in_cage/owl.py` (geometry
`topogym/assets/geometry/hedgehog_owl.py`; cage turned 315 deg on the family's mount), profile `hedgehog_owl_v1`. The
task text says the rod is a tool, that where it ends up does not matter, and that it falling off the table ends the
episode in failure (it is a task piece; `placement_scene_note`).

The reference was built piece by piece against the Session by the design's offline builder
(`v7_build.py` with `v8_session.py`; not in the repository), each piece
planned from the state the simulation had actually reached: rod strokes from a physics search checked for the arm,
carried routes from a push search with the real gripper's hulls on the tail. One arm, the rod and the tail in turns:
- **Wings 1:** the rod picked from the table, two strokes turn the wings from 270 to 221 deg with the owl standing free,
  the rod put back.
- **Lay down:** the tail held at its edges, the owl laid on its side on the cage floor, let go.
- **Wings 2:** the rod again, three strokes turn the wings to 264 deg, the rod put back.
- **Out:** the tail held, the owl pulled out under the rung, carried over the goal region and let go.
Grasps close slowly onto the part (0.3 mm per control step to 2 mm below its width) so the pads do not bounce off it.

The instance is made in the native gripper space (the rows command the Robotiq drive); the repository's run-oracle code
steps the reference's rows:

```bash
python scripts/roboarc/shared/rcb_topogym.py make-instance --task hedgehog_in_cage \
    --variation hedgehog_in_cage_owl --scene-seed 0 --gripper-space native --out <dir>/instance.json
python scripts/roboarc/shared/rcb_topogym.py run-oracle --task hedgehog_in_cage --instance <dir>/instance.json \
    --code tools/references/hedgehog_in_cage/owl_code.py --out <dir>/privileged.npz
```

`scripts/robotangle/oracles/hedgehog-in-cage-owl-i01/` (`mujoco_release.json` new_bases: template
`hedgehog-in-cage-very-easy-i00`, gripper space native, texts in `tools/release/package_text/hedgehog_in_cage/`): birth
e22fac98a9c3807a, 7224 rows (180.6 s), state hash f587cfd6835c9308, file sha256 23c2871db1b55c6b (`standard.npz` is the
same file). Two fresh-process strict replays end at that hash with success (about 175 s each, MuJoCo 3.14); the stress
observer (`replay.py --hold 0`) is CLEAN with success (deepest task contact 0.38 mm, held finger 0.66 mm, no forcing, no
review event); a nop of the same length does not succeed. Acceptance round bird_owl_r1 on d27c622d7: 1,000 episodes,
criteria v2.10, PASS (no confirmed breakdown, 45 grasp slips). Harbor under its own name (release_rebuild dry run on
3d8436413, 2026-10-12, rendering off): the instance made in the image at birth e22fac98a9c3807a, oracle 1 and nop 0 in
both modes, the oracle's end state f587cfd6835c9308 in both.
