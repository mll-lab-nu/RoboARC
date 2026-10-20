# caliper_sorting on MuJoCo (measure sleeves with a vernier caliper on a stand, sort them into labelled bins)

Jianwen, 2026-10-16 (version 2: 2026-10-17; version 3: 2026-10-18). A new family, MuJoCo only (variation
`caliper_sorting_vernier`, pool experimental, no tier). Design record, sources and every measurement:
`runs/caliper_sorting_design_20261014/` (STAND_FEASIBILITY.md and stand_check/: the caliper on its stand; GRASP_AUDIT.md:
why the hand-held caliper went; README_v3.md: the hand-held caliper; README.md: the sleeve classes, the vernier reading
test, why the gripper cannot gauge; REGISTRATION_PROPOSAL.md; not in git).

## The task

Version 3 (model `caliper_sorting_v3`, 2026-10-18; the user: the caliper on a stand). The grasp audit of version 2 found
the left hand's pinch on the beam's tail could not carry the hand-held caliper (the weld carried 4.6 times the torque
the pinch's friction could give; design record GRASP_AUDIT.md); a two-finger pinch that could (the face pinch, an upright
caliper of 120 g or 45 g) was either out of reach or failed in physics, so the caliper is now fixed on a stand and the
sleeves are brought to it. Four sleeves stand in a rack. They look alike (200 g, 20 mm long) and are of two classes, two
sleeves each, that share one diameter's class: the classes differ in one diameter by three vernier divisions. Four bins
carry labels with two caliper readings each ("ID 38.8 / OD 96.2": the inside jaws spread in the bore, the outside jaws
closed on the outside). A 150 mm vernier caliper is fixed face up on a stand, its jaws out over the table, closed and
locked. Its scale is in its own unit: one main-scale division is k mm (0.95-1.25 per scene) and the closed jaws read b
(10-60 units); the vernier reads tenths. The robot loosens the slider's locking screw and measures every sleeve: the
right hand brings it from the rack to the jaws, the left hand sets the slider (open, close onto the sleeve or open in its
bore), tightens the screw, the right hand takes the sleeve off the locked jaws, the left wrist camera reads the vernier,
the left hand loosens the screw; then the sleeve goes into the bin whose label gives its two readings. The gripper cannot
gauge the sleeves (66-74 mm outside exceeds its opening, a 7.5-9.4 mm bore admits no finger); a size known in
millimetres (forward kinematics, object sizes) cannot be matched to a label without reading the scale, and because the
episode's two classes share one diameter's class, sorting by relative size alone is right about half the time: 52-57 %
over 4001 scenes of the final generator (the metric review of 2026-10-19,
`runs/robotangle_mujoco/metric_review_20261019/caliper_zh.md`, finding 2; the design record's 50.6 % used older size
ranges). The floor is kept (the user, 2026-10-17 and 2026-10-19). On the one released scene, -i01, a deterministic guess
is not a coin toss: the natural rule "the shared class is the smaller label" is always right there.

- Scene (`topogym/assets/geometry/caliper_sorting.py`), placed in the ALOHA mount frame (table top z = 0) relative to the
  work origin W = (0.16, 0.02) m: the caliper's frame C (origin on the closed outside measuring faces, x along the beam
  toward its tail, y toward the inside jaws, z out of the scale face) at W + (-0.18, 0.07) m, 90 mm above the table, half
  a turn about z (the beam toward -x, the outside jaws toward +y); the stand `caliper_stand`, a block 44 x 24 mm under
  the beam's tail (C x 165-209 mm: the slider reaches C x 155 mm at 80 mm open) from the table to the beam's underside; a
  rack of six slots (pitch 44 mm) at x = W + 0.17 m, y = W + 0.04 ... 0.26 m, four of them holding sleeves; four bins
  (cavity 80 x 80 x 40 mm, walls 3 mm) at W + (0.06 or 0.15, -0.14 or -0.23) m, each with a label card (drawn only,
  no collision). Per scene the caliper and its stand shift by +-15 mm. The label cards (2026-10-19, the user: readable
  in cam_high, 848 x 480, about 1.5 mm per pixel at the bins): 84 x 40 mm, tilted 25 deg up from the horizontal toward
  cam_high (18-26 deg off its line of sight; 35-44 deg before), black text of cap height 14 mm with strokes 0.17 cap and
  0.15 cap more space between glyphs on white (before: 76 x 40 mm at 50 deg, 10 mm, 0.13), texture 24 px/mm. A
  front-row bin's card hangs from its front wall; a back-row bin's stands behind its back wall on a post, 50 mm up, so
  that cam_high sees it over the bin's own sleeves and it is over no other bin (before, the back row's cards lay over the
  front-row bins and the sleeves put there hid them).
- Caliper (Mitutoyo 530 series 0-150 mm, catalogue D-10): URDF `caliper` with links `mount` (the stand's clamp, the root,
  fixed to the world), `beam` (on the prismatic joint `mount` along C z, its limits +-1 um, damping 10 N s/m), `slider`
  (prismatic joint `slider`, 0-152 mm = the jaw opening s; friction loss 1.5 N, damping 2 N s/m, armature 1e-3 and a
  stiff friction constraint from the profile) and `screw`: the locking screw's knob on the slider's face, left of the
  vernier window on the jaws' side (slider x 12, y -8.85 mm), a knurled cylinder 7 mm across and 8 mm high with a white
  index line, on the revolute joint `screw` (axis -C z: clockwise seen from the scale side tightens; stops 0 and 45 deg,
  held nearly hard by the profile; friction loss 0.02 N m with a hard friction constraint; armature 1e-5). Why the
  clamp: MuJoCo filters a parent's contacts with its child only when the parent is not welded to the world; welded to
  the world, the beam collided with the slider that wraps it (2-4 mm deep at birth). With the beam on a joint whose
  limits hold it within 1 um the filter applies again; the limit does next to no work (in the reference it peaks at 2.2 N
  and does 1.3e-4 J of work in all; the stress observer's largest net constraint work there is under 1.8e-8 J). Collision: boxes, the roller and the knob as cylinders, the
  inside jaws' 0.3 mm knife edges as exact capsules (profile option `capsules`). Masses 130 g / 50 g / 2 g.
- The lock (`topogym/tasks/caliper_sorting/lock.py`, the task's MuJoCo mechanism): the joint equality
  `caliper::slider_lock` (slider = polycoef[0], solref 1.25 ms / 1, solimp .99 .999), declared with the URDF in
  `morph_options["joint_equalities"]` (shared commit 0ea61073f9). The mechanism writes the slider's position into
  polycoef[0] and switches the equality on once the knob is 20 deg or more off its loose stop, and off at 10 deg or less;
  it writes nothing else. The caliper is born locked (the equality authored active at the closed jaws, the knob on its
  45 deg stop). capture / restore / read carry the switch, the lock position and a switch count; read() adds the knob's
  angle (state_sha hashes it).
- Sleeves: URDF `sleeve_<k>` (link `sleeve`, axis local z), a ring of 192 boxes (outer faces on the outside diameter,
  inner on the inside one: 0.009 mm short of the circle at the corners); drawn as a turned tube with 0.5 mm chamfers.
- Textures (main scale and vernier in the scene's unit and zero, labels) are drawn by the generator with numpy and a
  stroke font of its own (no system font). Every link has ONE textured GLB visual whose material, texture and image carry
  names of their own (Nyx draws unnamed materials of different GLB files of one scene with the first file's image).
- Per scene seed: seed 0 is the reference's scene (k 1.08 mm per unit, b 31.4; ID 38.8 / 39.1, OD 96.2 / 96.5 = 7.99 /
  8.32 and 69.98 / 70.31 mm; classes A and B, both OD 96.2, in rack slots 0-3 as A B B A; labels C A D B; no shift). Other
  seeds draw k, b, the class readings (sizes 66-74 mm outside, 7.5-9.4 mm inside), which diameter the two classes share
  and its class, the four rack slots and the sleeves' order, the label order and the caliper's shift. Every sleeve's true
  size is its class's +-0.02 mm.
- Profile `caliper_sorting_v3` = claws_v6 (contacts 1.25 ms / dampratio 2 / .9 .99; welds 2.5 ms) plus the rules of the
  joints `caliper::mount` (armature 1e-5, solreflimit 1.25 ms / 2, solimplimit .99 .999: a hard limit, as the knob's),
  `caliper::slider` and `caliper::screw`, and `capsules`.
- Assisted grasp (contact_weld_v1): the slider (a 20 mm sphere about slider (44, -2.6, 0) mm, 32 mm from the knob, so a
  pinch on the knob never takes hold of it) and every sleeve (radius 40 mm), for both hands; the beam is fixed and no
  target; the knob is no target (it turns by friction); a hand may acquire with its TCP up to 0.15 m from a centre.
- Goal (terminal revision `sleeves_in_labelled_bins_v2`, 2026-10-19; v1 before): every sleeve's centre within 40 mm of
  the centre of the bin labelled with its class in x and y and 3-40 mm above the table; every sleeve at most 1 mm/s and
  0.02 rad/s; no assisted grasp on a sleeve and no robot contact with one above 0.05 N (`released`, as cast_coil's; new
  in v2); all for 0.5 s. Whether the caliper is locked at the end is not judged. Failures: the family's rigid ones (a
  part 100 mm below the table; contact penetration through the tooling's review rules), and (new in v2) a sleeve left in
  a bin that is not its class's (its centre inside that bin's box as above, no assisted grasp on it, no robot contact,
  still at the speeds above) for 0.5 s ends the episode in failure (`evaluate_transition`; the history holds, per
  sleeve, since when it has stood so). Why (the metric review of 2026-10-19, findings 1 and 3; the user, 2026-10-19):
  the standard service ends an episode at its first success, so without the failure a sorter that never reads the
  caliper could learn the right bins by trying placements one after the other; and a sleeve still pinched by the
  fingers after the weld let go counted as put down. The reference succeeds unchanged (its last robot contact with a
  sleeve is at row 22911; success at row 22962); the state hash, which covers the task's record, changes with the rules
  (birth f0c67c958f7901e8): the world part is unchanged.

## Reference (-i01)

Version 3 (2026-10-19). `scripts/robotangle/oracles/caliper-sorting-01-i01/privileged.npz` (`standard.npz` the same
trajectory), 22962 rows (574.1 s), native gripper space, on the instance `rcb_topogym.py make-instance --task
caliper_sorting --variation caliper_sorting_vernier --scene-seed 0 --gripper-space native` (birth f0c67c958f7901e8
under the rules v2 of 2026-10-19, re-stamped; 8388295a68647c29 under v1, see "The rules v2 and the labels" below).
No row moves an arm joint command more than 0.035 rad. Recorded on a tree of the test commit 11d97de9a7
(jianwen/caliper-sorting-harbor-test: this family with its registration; the tree of fe3c7f0e0d, dev ed8664d107 merged,
has the same birth and replays it to the same final state) with

```bash
python scripts/robotangle/shared/rcb_topogym.py run-oracle --task caliper_sorting --instance <instance.json> \
    --code scripts/robotangle/references/caliper_sorting/oracle.py --out <dir>/privileged.npz
```

The oracle (`scripts/robotangle/references/caliper_sorting/`: `oracle.py`, the physics replay on the Session, and the
planner it imports, `oracle4.py`, with `plan3.py` (the grasps and caliper frames), `scenario3.py`, `aloha_kin.py` and the
scene numbers of `geom.py`, which takes them from the task's geometry module; `check4.py` checks a plan's clearances)
commands joint and gripper targets only.

- The kinematic planner (`oracle4.Episode.plan`, a generator): the left hand loosens the locking screw (stage 21: the
  caliper is born locked); per sleeve the left hand opens the jaws to 80 mm (it pinches the slider's two side faces, the
  approach tilted 20 deg toward the tail; 3), the right hand takes the sleeve from the rack by its two end faces (2) and
  holds it until it is in its bin; it lowers the sleeve between the outside jaws 2 mm off the fixed jaw, waits for it to
  settle, and sets its rim 0.3 mm into the fixed jaw (4); the left hand closes the slider onto it (guarded: until the
  slider's jaw presses the sleeve with more than 1 N for 0.25 s; 5), lets go and tightens the screw (17); the right hand
  sets the sleeve's centre in the middle between the locked jaws (the preload off) and lifts it straight out of them at
  9 mm/s (6); the left wrist camera reads the vernier (7) and the
  left hand loosens the screw (18), closes the jaws to 5 mm (8); the right hand puts the bore onto the knife edges along
  its axis and sets its wall 0.3 mm onto the fixed edge (9); the left hand opens the slider onto the bore (10), tightens
  the screw (19); the right hand pulls the sleeve off the edges along its axis at 1 cm/s (11); the second reading (12),
  the screw loosened (20); the sleeve into the bin (13).
- The knob (`turn_knob`): the left hand from above, its body leaning 40 deg toward the tail (at 30 or 25 deg the hands
  met at the bore's lock), the fingers across the beam on the outside diameter and along it in the bore (across, the
  turn swung a finger's root into the sleeve's face); it pinches the knob (no assisted grasp), turns it about its axis at
  15 deg/s (+30 locks, again by the shortfall + 3 deg while it stands under 22 deg; loosening goes back by the knob's angle + 5
  deg), lets go and backs out along its approach line.
- The reading: the left wrist camera 70 mm from the scale over the coinciding vernier line, 55 deg from the face's normal
  on the inside jaws' side, the hand half a turn about the line of sight (stand_check/read_design.py: of the camera's
  distances, view angles and rolls tried, the one with the most clearance); the way in planned backward from the reading pose along the approach. A reading is
  `vernier_reading(s)` of the slider joint's position s in the middle of the 1 s pause there, in the caliper's unit (the
  scene's k and b). A reading more than 0.12 units from every label value is measured again (at most twice); the bin is
  the label nearest to both readings.
- The physics replay: the plan's 20 ms frames resampled to the 25 ms control steps; each arm follows by IK near the
  plan's joints, with inverse-dynamics feed-forward, a slow integral of the tracking error (frozen for the hand that
  holds the slider or pinches the knob, and for the right hand while its sleeve waits on the fixed jaw, stages 5 and 17),
  and the slider's friction as feed-forward while the left hand moves it; the
  left hand's path follows the slider as it actually is, the right hand's path to a sleeve in the rack that sleeve.
- Holding the slider while the screw locks (2026-10-18; the coordinator's question why every reading came out one
  division off its class): the left hand's pinch and turn of the knob pushed the free slider along the beam. In the bore
  the fingers closing on the knob pushed it 0.08-0.14 mm at once (the pinch centred 0.1 mm or so off the knob's axis);
  on the outside the pinched knob felt +0.4 to +1.3 N along the beam from the arm (more as the knob turned), under the
  slider's 1.5 N friction loss, but the friction constraint is soft and the slider crept at about 0.045 mm/s, 0.087 mm by
  the time the lock engaged. Toward the sleeve the sleeve stops it, so it always moved away: the readings came out +0.1
  on the outside and -0.1 in the bore (design record grasp_audit_v3/scripts/probe_oracle.py: s at the touch, the
  release, the pinch, the lock and the reading). With the smallest unit (0.95 mm) and a sleeve 0.02 mm off its class,
  0.14 mm would have been two divisions, nearer the other class. Now, while the fingers hold the knob and the lock is not
  yet on, the same hand holds the slider in place: its target moves along the beam by 15 times the slider's offset from
  its place plus a slow integral of it (0.05 per step, together at most 2 mm), and the turn waits while the offset is over
  0.010 mm (at most 5 s). The place: where the hand let go of the slider on the sleeve; in a bore 0.010 mm inside that,
  which leaves the knife edges about 0.02 mm clear of the wall: locked tight against the wall (0.4 N), the edges wedged
  the sleeve as the right hand pulled it off (the forces rose to 100 N, the grasp let go); with 0.02 mm the moving edge
  felt nothing during the pull. The lock then holds the slider within 0.006 mm of its place. Worst case over all scenes:
  a reading is (the sleeve's offset from its class, +-0.02 mm, plus the lock's offset from the true size) / k off its
  class value: on the outside at most 0.026 mm, in the bore at most 0.041 mm, at k = 0.95 0.027 and 0.043 divisions, both
  under half a division: the vernier shows the class value.
- The sleeve's lift out of the locked jaws (2026-10-19; the grasp audit): while the right hand waited with the sleeve
  on the fixed jaw for the closing and the lock, its arm's integral kept growing and pressed it there with 4-11 N by the
  lift; lifted so, the fixed jaw's friction, 35 mm off the sleeve's axis, twisted the sleeve in its end-face pinch about
  that axis for about 2 s, up to 2.2 times the torque the pinch's friction could hold (about 1 % of a sleeve's held
  steps). Now the right arm's integral is frozen in stages 5 and 17 (the press stays about 0.2 N) and the sleeve is
  centred between the locked jaws before it is lifted; the readings did not change.
- In this recording: the screw loosened to 0.3 deg at the start; locks at 23.4-28.7 deg, frees at 0.0 deg, each
  switching as planned; the knob's turn waited 3.5-3.7 s on the outside and 5 s (its limit) in the bore for the slider;
  the slider locked at the true size +0.005 mm on the outside and -0.019 to -0.021 mm in the bore; readings OD / ID
  96.2 / 38.8, 96.2 / 39.1, 96.2 / 39.1, 96.2 / 38.8, every one its class's value (A B B A: OD 96.2, ID 38.8 / 39.1);
  bins 2, 4, 4, 2 as the classes; no re-measure; success at row 22962; 627 s of wall clock on a lab RTX 4090 host.

Checks of the reference (2026-10-19, a lab RTX 4090 host, on the tree of fe3c7f0e0d: dev ed8664d107 merged; the image
eai-robotangle:0.3.0-dryrun-caliper-v2 (on 0.3.2-rc1), mujoco 3.14.0, one thread, numpy / OpenBLAS SIMD masking):
- The instance made on that tree: birth 8388295a68647c29, the instance file byte for byte the recording tree's.
- Two fresh-process strict replays (`rcb_topogym.py replay --repeat 1 --strict`): both success at row 22962, final state
  3e9024ee5d6b1eda, 557 s and 597 s (load about 22).
- The stress observer (`scripts/robotangle/stress/replay.py --hold 0`): no event (CLEAN); success; energy created
  0.00072 J; fastest task part 0.23 m/s and 2.4 rad/s; largest press 57.4 N; deepest task contact 0.13 mm; the largest
  net constraint work 1.8e-8 J (the slider's grasp weld; the limit of `caliper::mount` less).
- The equal-length nop (22962 rows of the birth targets, under the observer): no success, no event.
- The lock in the state hash and through a reset (the design record's mech_check.py): the first 1200 rows (the loosening
  switches the lock once) end at 415c9730057f37e6; reset, the birth hash 8388295a68647c29 again; the same rows, the same
  hash.
- The demo film's recording (the oracle again, rendering with nyx_direct): the same actions row for row, the same final
  state.
- Grasp audit (design record GRASP_AUDIT_v3.md): 20 welds (the left hand on the slider 16 times, the right hand on a
  sleeve 4 times) and 17 knob pinches; every weld's torque demand over what its pinch's friction could carry has a 95th
  percentile under 1 (the sleeves 0.30-0.36, at most 1.15-1.39 on 3-4 single control steps as the sleeve leaves the
  knife edges); the limit of `caliper::mount` peaks at 2.2 N and does 1.3e-4 J of work in all.
- Harbor under the base caliper-sorting-01-i01 (2026-10-19; the test commit 53d4d0a96e of the never-merged branch
  jianwen/caliper-sorting-harbor-test; image ghcr.io/mll-lab-nu/eai-robotangle:0.3.0-dryrun-caliper-v3, local only;
  rendering off): the instance made in the image has the reference's birth 8388295a68647c29; privileged oracle 1 / nop
  0, standard oracle 1 / nop 0, both oracles ending in 3e9024ee5d6b1eda. The standard sidecar's fresh-process replay took
  608.1 s (0.027 s per control step, the texts' figure).

The rules v2 and the labels (2026-10-19; branch jianwen/caliper-label-legibility):
- The state hash covers the task's record, and the rules v2 add a history (per sleeve, since when it has stood in
  another class's bin) and metrics: the birth is f0c67c958f7901e8 (was 8388295a68647c29) and the reference ends in
  ec9ade23fc0a034c (was 3e9024ee5d6b1eda). Without the task's record the hash is the same on both trees, at the birth
  055cfa28ddd7503e and after the last row 6908f5c45d545709 (the design record's label_legibility/world_sha.py): the
  world, the robot and their targets did not change, nor did the label cards' change (drawn only) move them.
- The reference re-stamped (instance_sha only; the rows and the tick ledger byte for byte the same; as bed_bolt's and
  clock's). A fresh-process strict replay on the tree of this branch: success at row 22962, final state
  ec9ade23fc0a034c, 522 s. Its last robot contact with a sleeve above 0.05 N is at row 22911, so `released` holds well
  before the success (the metric review's probe).
- `scripts/robotangle/test_caliper_sorting.py`: a sleeve put in another class's bin ends the episode after 0.525 s
  ("A sleeve was left in a bin that is not its own"); in its own bin nothing ends; a nop ends nothing.
- The labels in cam_high (label_legibility/: the task's camera at the birth and the end, the labels cut out at its own
  pixels): the text 8-9 pixels high (was about 6), all four read at the start and at the end (before, bin_1's was hidden
  by the sleeves of the bin in front of it). Along the reference no arm and no sleeve passes through a card: the nearest
  is a right finger 8.8 mm from bin_1's card as the last sleeve goes in (48 control steps within 20 mm), bin_3's
  17.2 mm, the other two 43.8 mm and more; the sleeves 13.8 mm and more.
- Harbor under the base caliper-sorting-01-i01 (2026-10-19; this branch at d0a35aff89; image
  ghcr.io/mll-lab-nu/eai-robotangle:0.3.0-dryrun-caliper-v4, local only; rendering off): `mk_tasks.py --check` (all
  bases, and the MuJoCo instance bytes) ok; the instance made in the image has the reference's birth f0c67c958f7901e8;
  privileged oracle 1 / nop 0, standard oracle 1 / nop 0, both oracles ending in ec9ade23fc0a034c. The standard
  sidecar's fresh-process replay took 678.3 s (0.030 s per control step, the texts' figure since 2621ebcf4f).
- The demo film (the design record's out/topogym_demo_v3/) closes on the task's cam_high at the end at its own pixels,
  with the four labels cut out of it (the same actions row for row, final state ec9ade23fc0a034c).
- The physics did not change, so no new acceptance round (r3 above stands; the coordinator, 2026-10-19).

Version 2 (2026-10-17: the hand-held caliper; reference 33391 rows, final state 588d58f1fbaaab36, Harbor both modes
oracle 1 / nop 0, acceptance round r2 PASS) and version 1 (six sleeves of four classes in millimetres, no lock;
acceptance round r1) are superseded; their records: the design record (README_v3.md, GRASP_AUDIT.md, v4dev/).
