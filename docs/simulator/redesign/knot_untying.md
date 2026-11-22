# knot_untying redesign: anchored ropes, clutter, goal regions (5 tiers, MuJoCo)

Status (10-13): design accepted by the owner with the 10-12 region decision (`_r2`) and the 10-13 back strip (`_r3`); robot-level routes on the current
head for very_easy (solved) and for the untying of easier, easy and hard (placement into the goal region still open);
medium not untied. Nothing here is a frozen reference: every reference is recorded once after the physics freeze
(oracle on the frozen head, fresh replays on two hosts, acceptance, videos). Agent "knot" (from the soft-body redesign
session, 10-10 21:15), then knot-robot. Code: `topogym/tasks/knot_untying/redesign/`,
`topogym/planning/knot_untying/redesign/`, `tools/redesign/knot_untying/`.

## 1. Why the released tiers were too easy
The released tiers are loose single knots (1-4 crossings) on an empty table with both ends free, 0.18 m tails and
1.0-1.35 m ropes. Their oracle is one pull of an end with the other end pinned, then an end spread; the naive breaker
"pull both ends apart" solves the released easy.

## 2. The concept
- **An anchored end.** Every rope's standing end is clamped to the table (a staple over the last ~9 mm; the clamp is
  `tasks.base.AttachmentSpec`, an existing protocol concept, now also on the MuJoCo world). Pulling the free end only
  cinches a figure-eight or an overhand: every crossing must be undone from the free end.
- **Long tails, short strokes.** Tails of 0.5-1 m against strokes of <= ~18 cm inside an arm's reach: each deletion takes
  several strokes and regrasps, the other arm pinning the over strand.
- **Clutter** (easier and up): 16 cm posts on the straight shortcut lines, a 3 cm curb between the knot and the goal
  region (a rope resting on it is not on the table).
- **A goal region** (easier and up): the untied rope must lie inside a region beyond the curb; on hard two ropes, each in
  its own strip, not crossing each other.

| tier | knot(s) | crossings | rope(s) | clutter | goal beyond the family's terms |
|---|---|---|---|---|---|
| very_easy | figure-eight, loose, 0.55 m free tail | 4 | 1.63 m | none | none |
| easier | figure-eight, loose, 0.5 m tail (`easier_v8`; fallback `easier_loose`, a 32 mm figure-eight) | 4 | 1.79 m (loose 1.97 m) | 2 posts, curb | the rope inside the back strip |
| easy | two overhands in series (`easy_v8`, 0.3 m tail; fallback `easy_loose`) | 6 | 2.29 m (loose 2.46 m) | 2 posts, curb | the rope inside the back strip |
| medium | figure-eight round a post + an overhand stopper (`medium_v81`, `medium_b81`, fallback `medium_loose`, robot-access `medium_loose_ra`) | 7 | 1.76 m | eye post, second post, curb | the rope off the post, inside the front-left region |
| hard | two ropes: an overhand on A, B threaded through A's knot (`hard`; fallback `hard_nostop` without B's stopper, robot-access `hard_nostop_ra`) | 10 | 1.54 + 1.28 m | 2 posts, curb | A in the front strip, B in the back strip, A and B apart |

The goal (`goal.py`, `task_rd.KnotUntyingRedesignTask.evaluate_state`) uses goal terms only and ends nothing: per rope
the family's terms (no projected crossings or overlaps at 2 r on the material centre line, on the table within the
family's z band, quiet at 1 mm/s, intact volume), each rope's centre line inside its region, two ropes 2 r apart in plan,
and the family's release settling.

## 3. Inverse generation
The goal first (the rope lying untied from its clamp), then the reverse of untying: the free end tucked back through the
loops in a fixed order gives the knotted start (`tiers*.py`, `compose.py`, `geometry.py`: a parametric knot with routed
tails, lifted to a non-intersecting centre line). Clutter from the oracle's paths (posts on the shortcut lines but
>= 3 cm off the oracle's strokes; the curb between knot and region; the clamp in one arm's exclusive zone). Births on the
chain in the object-level lab (`mjlab.py`, `birth.py`: settle, weld the clamp, release, settle), loose rather than
cinched (a cinched knot on this chain is a ball no parallel gripper can work). The lab starts are kept with the code
(`tools/redesign/knot_untying/births/<name>/`); `robot/build_instance.py` freezes a robot-level MuJoCo
instance from one.

## 4. Physics and grasp
MuJoCo 3.14, CPU float64, the exact constraint diagonal; the family's profile **knot_chain_v7** (the rope chain with the
speed clamp and the tension guard). Grasp **contact_weld_v1** on every rope link (as rope_threading on MuJoCo): it holds
~0.75-2 N when it drags a strand (it lets go when the link lags 25 mm behind the hand) and up to ~8.5 N when the rope is
taut. R3 (can an agent fault the chain?): the chain faults only past ~32 N at the hand; pulls with 8.5 N hands (one hand,
two hands apart, ratchets of regrasps) on the snuggest knots reach 8.3-10.7 N in the chain, no fault.

## 5. The oracle on the robot
The lab oracle (`untie.Untie.run_oracle2`: anchored node deletions with a pin, clear_tail, lift_over_post, reach-aware
draws) runs unchanged on the ALOHA through a robot backend (`robot_lab.RobotLab`: the arms as hands, Cartesian TCP moves
one row per control step, vertical pinches, the weld's grasp state; `robot_untie.RobotUntie`: the hand primitives). The
ALOHA's grippers cannot work close together (a finger bar and camera mount keep two vertical grippers >= 10-15 cm apart),
so the robot pins with closed fingers pressing the over strand (`--press-pins`) and far along it (`--pin-order far`).
Opt-in variants (each only changes behaviour when given): `--real-fingers` (fixture distances with the fingers at their
real opening), `--retry-room`, `--tight` (lower planned margins), `--stall-adapt [--stall-adapt-grasp]`, `--endgame`,
`--fast`, `--no-pins`, finishes `--flatten`, `--fixup-if-untied`, `--layout-stop-untied`. A route that branched is
re-made as a chain: stage 1 `<flags A> --stop-after-decisions N`, stage 2 `--resume <stage 1 rows> <flags B> ...`
(rows and tick ledger byte-equal to the original, checked for every chain below).

**Placement** (after the untying; `robot_place.py`, `robot/place_lab.py`): one process replays the untie rows once,
checkpoints, and serves placement tries from that untied state (rewind-exact; a successful try's rows are a complete
recording). Strategies: far-side-first serpentine plans laid by two-hand lays (`lay2`), pinned alternating lays (`lay3`)
or the old layout (`lay`). Measured limits on the weld: the first pinch >= 24 cm from the clamp, no wrist turn while
holding, carries at 3 cm/s at curb height.

## 6. Goal regions: the owner's decision 10-12 (`_r2`)
Goal regions only (births, curb, posts, clamps and ropes unchanged; the same rows give identical world and robot hashes on
the old and the `_r2` instances): easier's back strip ends at y 0.31 (was 0.33; the far layout level at 0.28 is where
both arms reach a rope-along-x pinch around x 0, and 0.31 is the shallowest strip that still takes three full legs at
the 4 cm finger-room spacing); hard's strip for rope a 1 cm wider outward (y -0.34; `hard_nostop_ra` -0.35); easy and
medium unchanged (`tiers_v8.py`: `easier_r2`, `easier_loose_r2`, `hard_nostop_r2`, `hard_nostop_ra_r2`). The longer
ropes (easier_loose 1.97 m, easy_v8 2.29 m, easy_loose 2.46 m) need four legs in that band.

**Design decision 10-13 (`_r3`, option A):** the back strip of easier and easy runs x -0.45..0.42 (y as `_r2`); curb,
fixtures, ropes and births unchanged (`tiers_v8.py`: `easier_r3`, `easier_loose_r3`, `easy_r3`, `easy_loose_r3`;
births `<name>_r3` = the base births with only the region and the note changed). The serpentine needs at least 5.5 cm
between legs (4 cm crossed its own legs, 5.5 cm did not); with x -0.45..0.42 the long ropes get 3 legs 5.5 cm apart,
and every leg stays inside the arms' vertical-pinch reach (the oracle's IK reaches 0.586 / 0.592 m from the bases at the
8.5 cm carry height). The `_r3` strip contains the `_r2` strip, so a route placed in `_r2` also meets `_r3`.

## 7. Routes on the current head (10-13)
`tools/redesign/knot_untying/routes.json` (commands, flags, rows, hashes):
- very_easy: **solved**, route b45fast, 7701 rows (192.5 s); two hosts byte-identical, fresh replay = live, success;
  nop, the old -i01 oracle and five breakers fail.
- easier_v8 (chain e9b, row 13344), easier_loose (one stage, row 23768 / 19800), easy_v8 (one stage, row 28002),
  hard_nostop (chains hnnp, row 10350, and hnc, row 14492), hard_nostop_ra (one stage, row 13901): **untied**, every
  chain verified; the placement into the region is still open (best hard_nostop: rope b meets all its terms, rope a one
  crossing short).
- medium (all variants), easy_loose, easy (v7) and hard: not untied (fewest crossings 1-6).

Stress on the inst10 instances (search per name, CPU confirmations, criteria v2.11): 0 BREAKDOWN, 0 exploits on all 11.
