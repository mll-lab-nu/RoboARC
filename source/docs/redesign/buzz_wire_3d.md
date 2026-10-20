# buzz_wire_3d: a harder 5-tier set (design note; agent buzz, 2026-10-10/11)

Status: DRAFT for the user's review. Code in the sandbox clone the work notes, branch
`qineng-rt-redesign-buzz_wire_3d` (new files only):
- `topogym/tasks/buzz_wire_3d/redesign/`: `design.py` (a tier as data -> the family's own BuzzWireLayout), `motifs.py`
  (a 3D wire-bender with exact bend radii), `task.py` (the redesign task class: the family's BuzzWireTask with its
  predicate unchanged, plus insulated rest sleeves and wooden obstacles), `tiers.py` (the generators);
- `topogym/planning/buzz_wire_3d/redesign/`: `kin.py` (Panda + Robotiq kinematics on the Session's own model),
  `wandpath.py` (the wand on the wire, the grasp family), `feasibility.py` (single-grasp explorer), `oracle.py` (the
  feedback oracle), `io.py`;
- `tools/redesign/buzz_wire_3d/`: `segplan.py` (one held segment, one grasp), `run_oracle.py`,
  `record.py`, `replay_probe.py`, `preview.py`, `kinpath.py`, `antihack/metrics.py`.

The predicate, the tolerances (5 mm / 10 deg, quiet 1 s), the contact rule (wand metal on wire metal ends the episode)
and the robot are the family's, unchanged.

## 1. What makes this family hard for a robot (and what did not)

The released tiers are long, but a single grasp can follow most of each route: the difficulty is route length. The new
tiers make the robot **change its grip**, because the wire's shape makes every grip fail somewhere:

- **The deck start needs a grasp from above** (the wand lies flat on the deck, the ring around the start post).
- **A U-dip needs a grasp on the grip's narrow edge** ("side"): the hand must stay beside the dip's plane while the ring
  turns 180 deg in it; a hand holding the broad face ("top") runs into the wire as it turns down.
- **A spiral staircase around a wooden post winds the hand around the post**: the handle cannot point inwards (the
  post), so the hand orbits the axis, and the wrist runs out of range within a turn.
- **Tight ring clearance** (the ring's radial clearance around the wire, 8.8 mm in very_easy down to ~6.8 mm).

A grip change needs a place to put the wand down. Mid-route, the only places are **insulated rest sleeves**: short
black rubber sleeves on the wire (as on real buzz-wire toys), on which the ring may hang (the game's rule is metal on
metal; a sleeve is not metal). The wand hangs handle-down; the robot regrasps it there. At the end, an **insulated
finish post**: the ring slides down it onto the deck, and a last grasp from above sets the final heading.

Tried and dropped:
- **Full vertical loops / corkscrews**: no grasp of the family completes one, and they cannot be split by hanging rests
  (the held handle's direction relative to gravity flips every vertical half-turn).
- **Walled channels** (walls on both sides of the wire, meant to forbid a hand beside it): the walls block the arm
  itself; no grasp could start.

## 2. How a tier is generated and checked (inverse generation)

1. **Pieces with a known solution.** Each section type is first solved alone: a held segment between two rests, for
   every grasp of the family, by the explorer (`feasibility.py`): a frontier search over arc length x handle angle x
   joint branch, every node IK-converged, collision-free (robot vs wire / sleeves / deck / posts, 2 mm), the ring
   centred on the wire, consecutive nodes close in joint space.
2. **The route is assembled from solved pieces** (`tiers.course`: horizontal legs through corners at a cruise height,
   features per leg), with a rest between pieces that need different grasps.
3. **The anti-hack argument is computed, not assumed** (`work/mkpairs.py`): every pair of rests (i, j), every grasp
   (13 of them: top / side / their flipped twins topb / sideb at two grip positions, end), a held segment from i
   (hanging, or the deck start) to j (hanging, or the drop). The minimal number of held segments is the shortest path
   start -> drop over the feasible pairs. A tier is accepted only if that minimum equals the number of pieces: no rest
   can be skipped by any grasp.
4. **The oracle replays the chain live** in a MuJoCo Session (`run_oracle.py`): every plan from the measured state.
5. **Break it**: the old oracle's rows on the new instance, a whole-route single grasp, skipped rests, straight moves.

## 3. The ladder

Every tier: the wand starts on the deck around the start post and ends on the deck around the insulated finish post,
within 5 mm / 10 deg, quiet for 1 s (the family's predicate). The Robotiq runs at springref 2.62 (parallel pads; core
10-11); a grasp is a plain two-pad grip on the 7 x 9.8 mm handle.

"Held segments" is the minimum over the all-pairs analysis (every rest pair x 13 grasps): no grasp of the family can
carry the wand across more than one segment, so a solution needs at least that many grips (plus the final placement).

| tier | wire (what the robot must get through) | rests | held segments (min, all-pairs) | grips in order (oracle chain) | ring radius / radial clearance | robot mount |
|---|---|---|---|---|---|---|
| very_easy | start run towards the robot, a corner, a 15 cm U-dip, the finish run (1.19 m) | 1 | **2** | top (from the deck) / side | 11.5 / 8.77 mm | default |
| easier (v9) | easy's route without the U-dip: start run (rest S1), corner, a 3/4-turn descending spiral staircase round a 12 mm wooden post, a 10 cm run to the insulated finish post (1.25 m) | 5 | **3** | top (deck to S1) / side (corner, into the spiral) / sideb (round the spiral to the drop) | 11.0 / 8.27 mm | default |
| easy (v8) | a 14 cm U-dip on the start run, a corner, a rest, the spiral staircase (1.49 m) | 5 | **3** | top / side (dip + corner) / sideb | 10.5 / 7.77 mm | default |
| medium (v8) | easy's route with a 16 cm U-dip (1.53 m) | 5 | **3** | top / side / sideb | 10.0 / 7.27 mm | default |
| hard (v8) | easy's route with three rests round the spiral (1.49 m) | 6 | **3** | top / side / sideb | 9.5 / 6.77 mm | (-0.54, 0, 0) |

(v8: the spiral's post 12 mm in radius, so that the parallel pads at springref 2.62 clear it; with it the sideb grip
winds from the first spiral rest to the drop and the forced count is 3 for easier..hard. The ladder above very_easy is
carried by the dip (none / 14 / 16 / 14 cm), the ring clearance (8.27 / 7.77 / 7.27 / 6.77 mm) and hard's spiral
rests, the oracle length (rows 14874 / 14309 / 15019 / 21462) and the grasps (4 / 5 / 5 / 7), not by the forced count.
Head: ccd4ba7fa.)

Design history (each fix came from a live run):
- v1 easy/medium/hard: the first rest hung 1 cm from the U-dip's descending leg; v2 lengthened the start run;
- v2: the rest's sleeve reached into the start post's top bend; the hung wand leaned 27 deg (also friction on the
  rubber sleeve holds a hung wand tilted 20-35 deg: regrasps plan on the measured pose or on the ideal one);
- v3: the dip moved to the run's end made every drop on the finish post infeasible; v4: rest 4.3 cm past the bend,
  dip at 0.75 of the run;
- easy v4 live run: the ring touched another branch of the wire just past the second rest (planned clearance to that
  branch 0.3 mm, execution error 1-2.5 mm) -> the planner now keeps the wand 2 mm off every other wire branch (0.3 mm to
  its own stretch, where the ring is centred); v5/v6 tried longer runs to the finish (every drop infeasible) -> v7;
- the head's finger-contact-lost release: deck grasps 2 mm below the grip's centre (above it the pads slid off).

A 5-segment hard was searched (a second dip on the tangential run; a tighter spiral round a thicker post; four spiral
rests); none had a feasible chain. Hard differs from medium by the ring (6.8 vs 7.3 mm) and three spiral rests that the
oracle must choose among.

Every chain also ends with the final placement on the deck (a grasp from above of the wand lying flipped around the
finish post, turned about the post and set on the finish pose within 5 mm / 10 deg).

The live oracle walks the **rest graph** of the analysis (`TierOracle.run_graph`): from each rest the furthest
feasible next rest first; if the measured grip (it varies by a few mm / deg from the nominal one) cannot be planned to
that rest, the next option; at the start, if no rest is reachable with the measured grip, the wand is put back on the
deck and picked again with another variant.

Why each forced change exists (from the analysis, not assumed):
- **top -> side** (all tiers with a dip): the deck start can only be grasped from above (a side grip's hand would go
  into the deck); a hand holding the broad face cannot follow the ring down a U-dip (the wire comes back up into the
  hand), every top grip stops 8-10 cm into the dip.
- **side -> sideb** (easy): holding the narrow edge, the hand is on the robot's side of the ring; round the spiral the
  ring turns away from the robot and that hand would have to pass through the post. Only the other edge (the hand on
  the far side) winds round it, and that grip cannot be taken at the start run's rest (the arm crosses the wire).
- **sideb -> sideb** (easy): the same grip, re-taken at the spiral's rest, with the wrist reset; without the rest no
  grip gets from the run before the spiral to the finish (every one stops 3/4 round, where the spiral is lowest and
  nearest the robot).

Hanging rests: the wand is lowered onto the sleeve (the ring's inner top edge 0.3 mm above it) with the handle within
90 deg of down and released; it swings down. At springref 2.62 it keeps swinging +-3-4 deg for many seconds on the
low-friction sleeve and may hang yawed on it; the regrasp plans on the measured pose, or on the ideal hanging pose
(the closing pads square the grip up).

### Robot mount (the user's method, 10-11)
1 floating solution = the wire path; 2 grasp schedule = the all-pairs chain (one arm, regrasps on the rest sleeves);
3 base search: a tier's `Design.robot_mount` (x, y, yaw) sets `scene_config.env.robot_mount` in
`BuzzWireRedesignTask.finalize_spec` before the scene is built (the existing scene mechanism; the frozen instance keeps
it, so record, replay and the agent's task share it). Legal: link0's footprint inside the table with the scene's 1 cm
edge margin, 2 cm from the deck, the home pose collision-free (5 mm). Stage A (`basescreen.py`, 124 legal mounts x 4
tiers on the cluster): every held segment, the rest regrasps with their approach lines, the deck pick, the drop heading and
depth, finish-grasp coverage in 20 deg steps. Stage B: the all-pairs analysis at the candidates (`segplan.py --mount`):
the forced count must not drop (it stayed 3 everywhere). Result: passing mounts cluster at the default; hard needs
(-0.54, 0, 0) (its rest-3 sideb approach); easier v3 had no deck pick at any legal mount (-> v9). 4 noise: the oracle's
IK seed and release variants (below); arm-orientation variety and inspection turns are not in the references yet.

### Final placement
After the drop the wand lies flipped around the finish post. Grasps from above reach it only when its handle is
0..140 deg from the finish heading (dyaw); on the other side the finish run arches over the handle (no grasp, and a
push collides with the wire or the spiral's post). Measured landings (checkpoint fan-outs on the cluster): easy 113..120 deg
(24/24 in reach), hard 104..121 (18/18), medium 132, very_easy 26; easier 175..177 with the release aimed 90 deg off the
finish heading (in reach after one closed-finger push of 60 deg about the post; 7/7 successes), -91..-119 without.
The oracle's place: grasp-turn-release rounds, a closed-finger push about the post when no grasp reaches, free moves
that hold at each waypoint, start from the measured arm and fall back to RRT-Connect (the old lift-over move jammed the
arm on the wire).

**The -100 deg landing basin (prism instances, 2026-10-12): partly a dead end.** On the head-built medium instance about
90 closed-loop runs landed the dropped wand at dyaw -88..-115: the wand's natural rest when the ring slides down the
post, so an agent will land there too. A planning-only probe at two such landings (the cluster
`rigid_integrate_20261012/runs/q1/outb`, `probe_place.py`) searched:
- 720 top-face grasps (top / topb x roll 0/90/180/270 x fraction 0.55-1.0 x pitch -40..40 x depth 0 / 2 mm);
- turns of +-20..140 deg about the post with each grasp found;
- 576 closed-finger pushes (+-15..60 deg x fraction 0.4-1.0 x lean 0-1.5 rad x pad height 1.5-12 mm).

Results:
- **-102.4 deg (recoverable in the planning model):**
  - 122 grasp configurations, all topb with roll 0;
  - turns of -20 / -40 / -70 deg towards the finish heading, so two rounds reach the window;
  - one 15 deg push.
  The live oracle still failed there: its grasp list tried roll-180 / fraction 0.7-0.9 first, and those grips stopped
  36-38 mm short of the handle.
- **-92.3 deg (dead end):** no grasp, turn or push at all: the finish run's arch and a wood stop sit over the handle.

A wand that comes to rest there stays there for good, which is unfair to an agent. The current references avoid it by
releasing higher up the post (medium: `--drop-backs 0.16`, landing +136). Two possible design fixes, for the user to choose:
- **Smallest: a low wooden deflector on the deck** under the finish run, in the -60..-120 deg sector about the post. The
  handle can then no longer settle under the arch.
  - It touches neither the wire nor the routes before the drop.
  - The five references land at 26 (very_easy), +108..+136 and 174 deg (easier), so they should not touch it. Their
    fresh replays on the re-made instances must confirm that, and the probe must be re-run at the new landings.
- **Larger:** raise or turn the finish run's last leg so that the arch clears a top grasp over that sector. This changes
  the wire, so every tier's route must be re-run.

### Oracle status (head ccd4ba7fa; references re-frozen and fresh-replayed there, hashes unchanged since c64294012)
| tier | rows | final state | plan | made |
|---|---|---|---|---|
| very_easy | 10424 | a28ee3880f00a88a | `plans/very_easy.json` | a cloud host traversal + the cluster final place from the drop checkpoint |
| easier | 14874 | 822041b44f971c30 | `plans/easier.json` (drop_offset_deg -90) | the cluster, full run |
| easy | 14309 | 3c0936b332b53fa6 | `plans/easy.json` | a cloud host traversal + the cluster from the last-segment checkpoint |
| medium | 15019 | 38c3c3fa8ce1bb1b | `plans/medium.json` | a cloud host traversal + the cluster from the last-segment checkpoint |
| hard | 21462 | 1f21175b0e8c9dbd | `plans/hard.json`, mount (-0.54, 0, 0) | the cluster, full run + last-segment checkpoint |
Rows are one continuous recording from birth (the checkpoint's rows are part of it); every reference was replayed open
loop from birth in a freshly frozen instance with the same final hash.

### Step 4: noise (primary references since 10-12 00:30; the table above = the previous references, kept)
| tier | rows | final state | noise | stress (GPU screen / CPU confirmations) |
|---|---|---|---|---|
| easier | 22390 | d44afab498503c77 | other arm branch at the deck pick (wrist flip) + +-10 deg inspection turns at each segment start | calm / 91 clean |
| easy | 14794 | 343dcc0702337d3a | +-10 deg inspection turn at the last segment | 1 GPU-only flag (CPU calm) / 95 clean |
| medium | 15958 | da732c0028db67fd | same | calm / 98 clean |
| hard | 17299 | 3fc2d023b5148208 | same | calm / 88 clean |
Oracle options: `--noise-branch [--noise-branch-n N | --noise-branch-at i,j]` (take the most different arm configuration
among the collision-free grasp configurations) and `--inspect-deg D` (a checked turn of the held wand about the wire's
tangent at a segment start; skipped where not clear). For easy / medium / hard every full noisy run (120 runs: branch
noise at any grasp, inspection turns, drop offsets) landed the wand on the unreachable side (-66..-138 deg), so their
noise is applied from the reference's last-segment checkpoint; the other-branch option found no second configuration
at their place grasps. Hard's noisy reference is shorter than the previous one (which carried failed place rounds) and
slightly below the old released hard on rows / TCP path / waypoints (see its packet). All instances re-frozen with the
final instruction text (patch 01); births and reference hashes unchanged.

Oracle machinery (`TierOracle`), each piece prompted by a live failure: staged deck grasps (the pads close in the air
before the last 12 mm: at springref 2.62 a wide-open hand jams its pads on the deck), node tracking with holds and
IK-continuation densification (the drives lag ~0.1 s), slow releases and 20 s settles at rests (a hung wand swings for
seconds), re-planned grasps when the wand moved, collision-checked free moves through via points, a rest-graph walk
with fallbacks, re-picks at the start, grasp-turn-release rounds about the finish post.

## 4. Breakers (all must fail)
- **the old route retargeted**: each old oracle's rows (`scripts/roboarc/oracles/buzz-wire-3d-<tier>-i01`) replayed on
  the new instance of the same tier (medium: old easy's rows; old medium has no reference) -> a cloud host `out/brk/<tier>/`;
- **nop** (2000 hold rows);
- **one grip for the whole route / for skipped rests**: the all-pairs analysis (every infeasible pair is a breaker
  that fails, for every grasp of the family);
- **straight-line carry**: the ring encircles the wire, so lifting it off is a wire touch (the episode ends).

## 5. Anti-hack: the strict references (old oracles) and the new tiers

Old oracles (`scripts/roboarc/oracles/buzz-wire-3d-<tier>-i01/privileged.npz`), replayed in a fresh Session on
their own instances (a cloud host `out/old/<tier>/rec.npz`), metrics by `antihack/metrics.py` (soft-body definitions v2, one
arm):

| tier | rows | grasps | phases total | waypoints | TCP path (m) | joint path (rad) | wand translation / rotation | wand path / accumulated rotation |
|---|---|---|---|---|---|---|---|---|
| very_easy | 8825 | 3 | 22 | 33 | 2.98 | 26.3 | 0.353 m / 155 deg | 0.78 m / 779 deg |
| easier | 9671 | 3 | 24 | 39 | 3.01 | 31.1 | 0.353 m / 155 deg | 0.82 m / 1014 deg |
| easy | 9905 | 5 | 42 | 74 | 4.02 | 40.5 | 0.353 m / 149 deg | 1.15 m / 1133 deg |
| medium | (no reference: blocked) | | | | | | | |
| hard | 18677 | 5 | 33 | 92 | 6.36 | 57.8 | 0.448 m / 150 deg | 2.34 m / 1508 deg |

New oracles (head references, record.py replays on the frozen instances):

| tier | rows | grasps | phases total | waypoints | TCP path (m) | joint path (rad) | wand translation / rotation | wand path / accumulated rotation | forced grips |
|---|---|---|---|---|---|---|---|---|---|
| very_easy | 10424 | 3 | 23 | 60 | 3.742 | 44.886 | 0.2996 m / 173.43 deg | 1.381 m / 1966.5 deg | 2 held segments |
| easier | 14874 | 4 | 30 | 79 | 6.868 | 66.961 | 0.2048 m / 75.22 deg | 1.567 m / 2790.9 deg | 3 held segments |
| easy | 14309 | 5 | 33 | 87 | 5.585 | 56.707 | 0.2046 m / 75.13 deg | 1.82 m / 3404.2 deg | 3 held segments |
| medium | 15019 | 5 | 34 | 84 | 5.784 | 58.851 | 0.2052 m / 75.71 deg | 2.197 m / 5703.3 deg | 3 held segments |
| hard | 21462 | 7 | 51 | 105 | 13.1 | 80.953 | 0.2047 m / 75.08 deg | 1.865 m / 3552.1 deg | 3 held segments |

Against the old reference of the same tier (medium has none): every new reference is longer in rows, waypoints, TCP
path and joint path, and its wand turns more (accumulated rotation 2.4-3.0x); the wand's path is longer for very_easy,
easier and easy and shorter for hard (1.87 vs 2.34 m: the old hard wire was longer). Grasps: very_easy 3 vs 3, easier 4 vs 3, easy 5 vs 5,
hard 7 vs 5. Motion phases are fewer only for easy (33 vs 42: the old easy route had many short moves). The new
tiers' grips are forced (all-pairs analysis); the old routes' were a choice.

## 6. Open questions for main / the user
- The instruction text should mention the insulated rest sleeves and the insulated finish post (the TASKS text is
  shared: a patch for main).
- The deck is 0.62 x 0.45 m (released: 0.52 x 0.37) to make room for the routes.
- Demo videos: rendered on host A (nyx_direct) from the head references; each replay reproduces its hash on host A.
- very_easy is "more" than the old very_easy on paths and rotations, equal on grasps: should very_easy carry a second
  forced grip (e.g. easier's spiral after the dip), making the whole ladder one step harder?
