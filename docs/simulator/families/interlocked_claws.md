# interlocked_claws on MuJoCo

Shared mechanisms (the ALOHA, its finger coupling, the contact classes, the exact constraint diagonal) are in
[MUJOCO_CORE.md](../MUJOCO_CORE.md); this note (2026-10-10) records the references.

## References (claws v6, on b1ec43997; grasp welds at 2.5 ms since the claws_v6 profile)

`scripts/robotangle/oracles/interlocked-claws-<tier>-i01/privileged.npz` (development names; the tier-named -i01
directories are not included in this repository); `standard.npz` is the same file (the release
rebuild's references, `source/scripts/robotangle/release_rebuild.py`). Each was recorded in a fresh Session on b1ec43997 (the
exact constraint diagonal), on an instance made on that commit (`rcb_topogym.py make-instance`, scene seed 0, native
gripper space):
- the rows: claws v5's (the released Genesis rows, re-recorded on MuJoCo), with the wrist dither removed (below);
- then hold rows of the last row until the episode ends.

The tools are in `source/scripts/robotangle/references/interlocked_claws/` (run them like the stress tools). Claws v5's rows
are the released oracle's rows padded with copies of their last row, so the references are reproduced from the tree:

```bash
R=source/scripts/robotangle/references/interlocked_claws; O=source/scripts/robotangle/oracles
# very_easy, easier: the released oracle, re-recorded on the current physics
python $R/rerecord.py <instance.json> $O/interlocked-claws-<tier>-i00/privileged.npz ref.npz
# easy, medium, hard: first the dither notched out over claws v5's length (792 / 1416 / 16202 rows)
python $R/notch_ref.py $O/interlocked-claws-<tier>-i00/privileged.npz route.npz \
    --cols left/wrist_rotate,right/wrist_rotate --length <rows>
python $R/rerecord.py <instance.json> route.npz ref.npz
```

Success: every target inside and quiet for 0.5 s. The success row closes that window: the goal holds over its last
21 rows.

| tier | instance birth | rows v5 -> v6 (success row) | quiet window (rows, by the 0.5 s rule) | state hash (claws_v6) | file sha256 | observer | 2 fresh replays |
|---|---|---|---|---|---|---|---|
| very_easy | e9f9c48121060aa1 | 4363 -> 4370 | 4350-4370 | 0252d799b7c4afec | 8a52c984f247de9a | CLEAN | same hash |
| easier | 0c6e61da08f46aa7 | 20111 -> 20112 | 20092-20112 | fefe0fbf9c972c03 | cc36baffc5e72ee7 | CLEAN | same hash |
| easy | e1fded021d662254 (new) | 792 -> 812 | 792-812 | 30545aa2bfa43760 | 9b0b72ab379d7de5 | CLEAN | same hash |
| medium | d47e72299f96f4a6 (new) | 1416 -> 1442 | 1422-1442 | 094d2865820e6292 | d9ccaa4056d5ffa2 | CLEAN | same hash |
| hard | 515b1b04e332c221 | 16202 -> 16225 | 16205-16225 | f71af286b7e9b8a7 | 32c9c70f239325e2 | CLEAN | same hash |

- Instance births: easy and medium settle their pieces with physics at birth, so their birth hashes are new on the exact
  diagonal; the other tiers' births run no physics and keep theirs.
- On the exact diagonal every tier reaches its goal a little later than v5 did (very_easy +7 rows, easier +1); easy,
  medium and hard also lost their dither (+20, +26, +23 rows).
- The grasp welds at 2.5 ms (profile claws_v6, W4): the same files succeed at the same rows, CLEAN, with the state
  hashes above (each the same in a plain replay, under the observer and with the welds set at runtime). On the
  1.25 ms welds of claws_v5 they ended on a331ef1c / eca56d89 / 4d369001 / be9885f2 / 2d99141a, the validator's CL6
  record.
- Prisms (2026-10-12, the cylinder -> prism rule b2b751580): only medium has robot-touchable cylinders (12 bar
  geoms of r 2.8-5.0 mm). Re-made on the head (f9db156d16) its birth stays d47e72299f96f4a6 and its reference still
  succeeds at row 1442, but it now ends on 1d991d7010559803 (2 fresh replays on the cluster).
  The same head without the rule (`treeNP`) ends on 094d2865820e6292, so the change is the prisms'. The file is
  unchanged (a re-recording stops at the same first success, row 1442).

## The grasp welds at 4 dt (claws_v6)

On the exact constraint diagonal a weld runs at its nominal time constant. At 2 dt (1.25 ms, MuJoCo's floor), beside
the 2 dt contacts and finger coupling, the claws welds ratcheted a jam's load: in the validator's CL6 (claws easy,
periodic_e38, now corpus C1) a fast wrist motion loaded the left weld from 22 to 149 N in 20 ms; its slip then released
the squeezed piece into the finger hull, 11.3 mm deep (simulator_instability). At 2.5 ms:
- e38 stays CLEAN, 0.49 mm, in 2 fresh processes; also with the change switched on in mid-jam (row 84: 1.1 mm).
- The claws corpus (35 streams) and reproducers (18): 0 BREAKDOWN. Four VIOLENT streams turn CLEAN (M5, P e98, two
  corpus) and M3c turns VIOLENT (a piece thrown at 4.9 m/s, no energy created). The deepest task-robot contact is
  0.86 mm (0.82 before).
- Rejected: re-anchoring a slipping weld (the piece still squirts, 11.1 mm) and twice the substeps (+33 % per step,
  and it fails when switched in mid-jam).
- Scope: claws only. The same change on the other families' welds breaks references (buzz hard, keyhole easy), and on
  blanket's 5 ms patch pins a 4 dt time constant breaks very_easy at a release.

## The wrist dither (removed in v6)

The released Genesis references (the -i00 oracles, source "harbor_bridge") command both `wrist_rotate` joints
with a 1.6 Hz sinusoid on top of the motion; claws v5 kept those rows. A time-lapse video samples about 3.1 frames per
simulated second and aliases it into a frame-to-frame tremor (the claws hard video: centroid |d2|/|d1| 1.7-1.9 at the
frame rows, 0.25 at the control rate). Every other source was flat on the same run: no arm drive at its cap, the fingers
and their coupling force steady, the gripper-claw contacts steady.

| tier | joints | amplitude | span |
|---|---|---|---|
| hard | left and right wrist_rotate | +-0.060 rad | 5-405 s |
| medium | left and right wrist_rotate | +-0.05 rad | the whole episode (0-35 s) |
| easy | right wrist_rotate | +-0.028 rad | 0-15 s |
| very_easy, easier | none | | |

Removed with a zero-phase notch, 1.35-1.9 Hz, on both `wrist_rotate` columns (`notch_ref.py`: the line through the
end values is kept, the ends move by at most 3 mrad; the largest change is 0.06 rad). No tier needs the dither: each
succeeds without it. The scan (`cmd_scan.py`: 10 s windows, a spectral peak above 0.5 Hz with more than 5 mrad RMS and
at least 90 % of the power above 0.5 Hz) finds no such tone in any other family's references.
