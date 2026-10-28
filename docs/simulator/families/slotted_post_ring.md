# slotted_post_ring on MuJoCo

Shared mechanisms are in [MUJOCO_CORE.md](../MUJOCO_CORE.md); this note (2026-10-10 / 11) covers what is specific to
slotted_post_ring and why. The evidence was kept outside this repository.

## The task

A wooden post with a through-slot stands on a base. A cord runs through the slot; each end carries a bead and ends in
a disc (110 mm) that cannot pass the ring (inner diameter 102-108 mm). The ALOHA frees the ring and places the whole
ring in the goal area. Nine settings:

| settings | cord | assisted grasp |
|---|---|---|
| `very_easy`, `easier` | the rigid chain file (`topogym/assets/geometry/slotted_chain.py`, 88 links) on the easy geometry (28 mm slot), since c15011129 | contact weld on the ring only |
| `easy_chain`, `medium_chain`, `hard_chain` | the same chain file, 34 mm slot, ring radius 64 / 62.5 / 61 mm | contact weld on the ring only |
| `easy`, `medium`, `hard`, `clearance36` (FEM on Genesis) | on MuJoCo the cord is a rigid chain (task parameter `cord_model: chain`, set by `rcb_topogym.simulator_config`): the 10-07 chain of `topogym/tasks/slotted_post_ring/fem_chain.py` and `assets/geometry/slotted_post_chain.py` along the setting's authored cord | none (unassisted, as the FEM contract) |

Geometry, goals and rules are the task files'; the port changes none of them. A chain cannot stretch or squeeze as the
FEM cord (E 50 kPa) did, and the FEM-only integrity rules (tet inversion, attachment stretch) have no chain counterpart.

## Profiles (`topogym/world/mujoco/profiles.py`)

- `_CHAIN = _CLAWS` (claws_v5: Newton, implicitfast, elliptic cones, converged CCD, ALOHA hulls <= 64 vertices, the
  stiff follower-finger coupling, the soft class on robot-robot and robot-static contacts), 0.625 ms steps (2.5 ms ticks
  of 4 substeps).
- Every geom of the chain entities takes the claws contact class through the profile option `mjcf` (`_CHAIN_GEOMS`:
  solref 1.25 ms / dampratio 2, solimp .9 .99 .001): `slotted_chain` (the chain file; `_SLOTTED_108`, also very_easy /
  easier), `cord` and `target_ring` (the FEM settings' chains; `_SLOTTED_FEM_CHAIN`).
- `_SLOTTED_FEM_CHAIN` splits the post (one extrusion with the slot as a hole) into exact fat convex prisms
  (`slotted_post_chain.post_parts`, 82 pieces, thinnest 0.86 mm); the chain file's fixture is exact boxes in `<entity>::world`.
- MJCF entities with their own collision bits are remapped so that MuJoCo reproduces Genesis' per-entity masks
  (`mjcf_entity.py`): the ALOHA collides with the beads, discs, ring and links.

## Settling (frozen with the instance)

A chain draped over the post and ring keeps sliding slowly after its fall. `rcb_topogym.MUJOCO_CHAIN_SETTLE_S` gives each
setting the shortest settling after which every 10 s window moves every movable body at most 1 mm (measured from the
authored pose, one sample per second), rounded up. `make-instance` freezes the settled state
(`instance_assets/mujoco_settled.npz`, `simulator.mujoco_settled_state`), so only building an instance pays it; a Session
restores it bit for bit (checked on all nine: the restored birth hash equals the settling Session's).

| settings | needs | set |
|---|---|---|
| very_easy | 54 s | 70 s |
| easier | 310 s (bursts where links slip over the post and ring) | 330 s |
| easy / medium / hard_chain | 71 / 70 / 57 s | 80 s |
| easy, medium, hard, clearance36 | 5-6 s | 15 s |

## Acceptance of the scenes (seed 0, width gripper space)

nop 400 + hold 40 under the stress observer (`stress/replay.py`), verdict v2.8; drift = the largest displacement of a
movable body over the first 10 s after frame 0; thinnest = the thinnest colliding task geom.

| setting | state_sha | verdict | deepest contact | 10 s drift | thinnest geom |
|---|---|---|---|---|---|
| very_easy | 1267a54bfe4ae141 | CLEAN | 0.14 mm | 0.89 mm | 7.0 mm |
| easier | abb49f48c87259c0 | CLEAN | 0.07 mm | 0.69 mm | 7.0 mm |
| easy_chain | f971d3280130a466 | CLEAN | 0.01 mm | 0.47 mm | 7.0 mm |
| medium_chain | 9c9f9818d73ab420 | CLEAN | 0.01 mm | 0.46 mm | 7.0 mm |
| hard_chain | cbf0bc9bb23adc5a | CLEAN | 0.01 mm | 0.35 mm | 7.0 mm |
| easy | 5a951102e1c43c4f | CLEAN | 0.01 mm | 0.24 mm | 0.86 mm |
| medium | 31d6f7c413e6c969 | CLEAN | 0.01 mm | 0.17 mm | 0.86 mm |
| hard | 7917e65a3f165a89 | CLEAN | 0.01 mm | 0.17 mm | 0.86 mm |
| clearance36 | 87e801350c47968a | CLEAN | 0.01 mm | 0.19 mm | 0.94 mm |

On every setting: nop is not a success, no energy created, the slot is open (probe points inside it lie in no fixture
geom), no geom under 0.5 mm, and beads, bead bores, discs, ring and links all collide with the ALOHA.

## Rendering

nyx_direct (`TASKS` renderer "direct"). The FEM settings are drawn as the FEM setting (hemp cord, wooden discs, at
appearance seed 0; seeds 1 and 2 re-finish the cord, post, beads, discs and ring, `cord_finish`,
`slotted_post_ring_hard_chain.PUZZLE_FINISHES`): the FEM
cord is skinned on the chain's links and the rigid parts ride the MuJoCo bodies that hold them
(`nyx_direct.chain_cord_host`, as the render bridge's `_chain_cord_host`). Against the bridge's Genesis host
(`test_nyx_direct.py`, birth and after a 40-row nop): input buffers bitwise but rounding at birth, RGB within the host's
own run-to-run spread. Known asset issue, not fixed here: the chain file's fixture authors a collision box and a visual twin of
the same size; the two coincident surfaces flicker between grey and wood from run to run on either renderer (up to
~10,000 pixels per camera).

## References (`scripts/robotangle/oracles/<base>-i01/`, `standard.npz` = `privileged.npz`)

Recorded live on MuJoCo by the family's feedback oracles through `rcb_topogym.py run-oracle`
(`tools/references/slotted_post_ring/record_reference.sh`; oracle code in `tools/references/oracle_runners/`, not included in this repository), on
an instance built in **native** gripper space (the oracles command native gripper targets; same scene and settling as the
width instances above). Each replays in two fresh processes to the recording's hash. (The base names below are development names; these
-i01 directories are not included in this repository under these names.)

| base | instance state_sha | rows | final state_sha | oracle |
|---|---|---|---|---|
| slotted-post-ring-very-easy-i01 | 3cb1151a6c3f5afb | 437 | 5a730ec14ba1b22c | `planning/slotted_post_ring/below_easy_oracle.py` (BelowEasyPolicy) |
| slotted-post-ring-easier-i01 | 88c14ec362546ea4 | 969 | 69b0d12f924a8ad0 | the same (EasierPolicy) |
| slotted-post-ring-easy-chain-i01 | 146a3e5a43839f37 | 9193 | 53166fda8ebc81ba | `planning/slotted_post_ring/chain/` (the chain settings, below) |
| slotted-post-ring-medium-chain-i01 | 77ec761119dc157a | 10580 | cf9b8cc743473af3 | the same |
| slotted-post-ring-hard-chain-i01 | 48302a49de756a78 | 9407 | fd92819c3925a0cb | the same |

Armature 1e-5 (2026-10-13): the chain file's
ball joints at 1e-5 (was 1e-6), the fix for the stress breakdowns of a link pressed into the fixture (S1, prefix4_e20,
corpus S3 / S4). Every setting is born anew (table above, two fresh strict replays each). very_easy and easier keep their
recordings, which still succeed on the new births; the three chain settings are re-recorded by the chain oracle with
53fc82bf70 (M1's from-home IK branch) and 988cc270b6 (three fallbacks). The paragraph below describes the earlier
armature-1e-6 instances.

Final states on instances rebuilt at 9b5c7c28b, two fresh strict replays each. ccd4ba7fa made the soft contact class
(robot-robot and robot-static contacts: the ALOHA against itself, the table, the ground and the fixture) a constant
impedance, solimp .95 .95 instead of .9 .95; `scene.migrate_contacts` moves frozen scenes to it too, and the slotted
profiles keep no static piece (no `static_contact.keep`), so an instance frozen earlier and a rebuilt one load the same
model (easier's instance frozen at ac8dafe6a replays to the rebuilt one's final state). very_easy makes no robot-robot
or robot-static contact at all and replays to its old final state; easier makes 906 robot-robot (left finger on the right
gripper base, from tick 5709) and 42 robot-static (right finger on the fixture) contact steps, and its final state
changed (the recording is unchanged and still succeeds). The three chain-setting recordings no longer succeed on this head (the stress
validation's head check: CLEAN, final states dc0f919fefc35253 / d2e3a4fb21f24aa0 / b854e5a4d391e701 for hard / medium /
easy) and are re-recorded.

Recipe: `record_reference.sh slotted_post_ring slotted_post_ring_<tier> native
tools/references/oracle_runners/slotted_below_easy_session_oracle.py <out> <work>`; for the chain settings
`... slotted_post_ring_<tier>_chain native tools/references/oracle_runners/slotted_chain_session_oracle.py <out> <work>`.

Oracle changes for MuJoCo (plans only; goals, rules, geometry unchanged):
- easier: the left hand's straight lift after letting bead2 go is 10 cm (was 12): the hand ends the carry ~4 mm lower
  and 12 cm above it is just outside the left arm's reach.
- easier / very_easy: after the ring is set down and released, the hand backs off along its approach as far as the arm
  reaches (6 cm at most): on the 330 s settled instance the full 6 cm was 1.9 mm outside the IK tolerance.

The FEM settings (easy, medium, hard, clearance36) never had an oracle on any engine.

All references also pass the stress observer as single replays (`stress/replay.py --hold 0`; verdict v2.9, easy_chain
v2.10): CLEAN (easy / medium / hard_chain: peak task speed 1.61 / 1.59 / 1.58 m/s, spin 45 / 45 / 55 rad/s, three / two
/ one chain whips under 3 m/s).

### The chain settings (easy / medium / hard_chain)

`planning/slotted_post_ring/chain/` (M1 disc1 through the slot, M2 ring over the post, M3 ring back through the slot and
off the bight, M4b discs and cord to the system side, M4a ring into the goal) through
`tools/references/oracle_runners/slotted_chain_session_oracle.py`, native gripper space. The original plan stopped at the end
of M1 on MuJoCo; contacts are harder (the hand stops where Genesis let it press on) and the chain does not stretch.
Plan changes so far, each from one observed failure (goals, rules, geometry unchanged):

| stage | observed on MuJoCo | change |
|---|---|---|
| M1 | A's push of disc1 into the slot stops ~26 mm short against the post; B's 12 pull steps left disc1 5 mm in the slot | B pulls in at most 20 steps |
| M2 | the carry draws disc1 in behind the post after the set-down spot was computed; the ring was let go 10 mm onto it | the spot follows disc1 during the descent |
| M2 | closed fingertips pushing disc1 touched the ring, the grasp assist welded it and the push carried it off | disc1 is pushed with open fingers while the ring is within 5 mm |
| M2 | 3 mm over the table the fingertips came down on the links round disc1; over the links they hold only the disc's top 2.5 mm | fingertips 1.5 mm over a link's top, up to 3 passes |
| M2 (medium) | the push hand's camera mount could point at the post (a tie in the choice) and the gripper base met the fixture | the mount prefers A's side |
| M3 (easy) | after M2 the 64 mm ring lies on links (its centre 22 mm up; 14 mm on medium and hard); at the 30 deg take point A's pads closed on the tube and a link (26.2 mm for a 19.3 mm tube), the ring sat 27 mm off in the grip and B's rear take met A's finger | A takes the first tube point of 30 / 15 / 45 / 0 / 60 deg with no chain point within 3 cm (easy: 60 deg), and the next one when the closed pads hold more than the tube |
| M3 (easy) | carried as taken at 60 deg, the ring's centre was 3 cm nearer the post at the swing and its front arc met the post's back face | taken elsewhere than at 30 deg, A first turns about its pads to its 30 deg stance (the ring turns about its own axis); the rest of M3 is unchanged |
| M4b | the first hand and closing sign for bead1 was 328 mm out of reach | chosen by IK among both hands and signs |
| M4b | a corrective reach commanded 7.5 mm under the real target was blocked by a fingertip on the table and failed | `motion.reach()` keeps the first arrival |
| M4b | a disc pinched 3 cm from its eye stood on its rim and held up the set-down | let go where it is held up |

All three settings succeed (easy / medium / hard_chain: 9327 / 9230 / 9117 rows). M1, M2, M4b and M4a hold all their
stage checks; M3 frees the ring and lays it at its spot leaning, so its own `ring_flat` check is false, and M4a then
takes it from there into the goal. IK steps to another branch, reported by `move_ee`: easy one full
turn of the right forearm roll in M2 (6.28 rad, between the ring's push and disc1's, the hand empty), medium one and
hard three of 0.5 rad on the right forearm roll or wrist rotation (the ring stays in the grip). Each attempt takes ~1.2-2 h on the shared box. The medium and hard
references were recorded before the M3 change above; replayed to the end of M2, both leave the 30 deg take point clear
(nearest chain point 43.4 / 32.2 mm), so the changed M3 takes the ring there as recorded.
