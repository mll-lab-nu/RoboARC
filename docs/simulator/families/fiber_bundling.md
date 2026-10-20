# fiber_bundling on MuJoCo

Ported 2026-10-09 / 10. Shared mechanisms are in [MUJOCO_CORE.md](../MUJOCO_CORE.md); this note covers what is specific
to fiber and why.

## The task

A fiber bundle runs from a fixed terminal box through a cable cradle. It has a sheath (lead and trunk links) and fine
strands fanning out of the sheath front. The Panda + Robotiq 2F-85 grasps the sheath at `trunk_04` (assisted grasp
`contact_weld_v1`), combs every strand toward the sheath axis by pressing it against the bundling ring's outer surface,
then feeds the bundle through the ring.

Success (`terminal_v4_fiber_ring_plane`), held for 0.5 s:
- every strand station and the sheath front have passed the ring plane inside the aperture;
- every bend is at most 60 deg;
- the gripper has never touched a strand (that fails the task at once).

Tiers very_easy .. medium carry references. Hard is held to robust physics only (no reference, no speed target).

## Model

The authored articulated chain (`bundle.xml`), as is.
- Strands: 10 capsule links of 8 mm, radius 1.2 mm, about 45 mg each.
- Sheath: 40 mm links, trunk radius 12 mm.
- Every cable joint is a ball joint.

The sheath front (12 mm radius) must pass a 13.5 mm aperture: 1.5 mm of clearance.

## Profile `fiber_v4` (`topogym/world/mujoco/profiles.py`, `_FIBER`)

**Solver.** Newton, 50 iterations, implicitfast, pyramidal cone. Converged convex collision (500 / 1e-10), shared with
every family.

**Contacts**
- Every contact: 2.5 ms (4 dt), dampratio 1; robot solimp .95 / .99, the Robotiq pads included.
  - Contact time constant vs fiber reproducers broken (of the validator's 14): 10 ms broke 10, 2.5 ms broke 2.
- The Robotiq's collision hulls have at most 128 vertices; the Panda arm keeps its full hulls.
- Every static body is soft (`static_contact.soft`, `robot.self_contact`). Results on 16 reproducers and 20 corpus
  streams, under the validator's observer:

  | static contacts | breakdowns | deepest robot-static contact |
  |---|---|---|
  | soft for all | 0 | 3.8 mm (terminal box); 2.8 mm on a 10 mm cradle plate (half of it is 5 mm) |
  | cradle, ring and post kept exact | 2 | |
  | no `static_contact` | 3 | |

- **The cable's own contacts: 1.25 ms, dampratio 2** (`mjcf`). That is the same stiffness as 2.5 ms at dampratio 1,
  with twice the damping.
  - The finding on the reproducer S1 cem_e52: the arm pressing a trunk link at ~530 N stored ~0.5 J in the soft
    penetration, and the contact released it into the light cable.
  - Pair net work +0.47 J → ≤ 0; X rise 0.19 → 0.04 J.

**Air drag** 5e-5 N s^2/m^2 on the fine strands (`drag`). Whipped strands otherwise reach speeds that no contact
resolves at 0.625 ms steps. Its effect at normal speeds:

| strand link speed | drag force |
|---|---|
| 0.1 m/s | 5e-7 N |
| 1 m/s | 5e-5 N (0.17 % of the link's speed per tick) |

**The wire** (`joints`, `plastic`). The requirement: it bends like everyday iron wire under force, stays bent, and nothing
bends before contact.
- Every cable joint is elastoplastic: a spring, elastic up to 0.4 deg. Past its yield torque the joint's rest
  orientation follows it, so a bent joint stays bent and nothing creeps below the yield.
- Yield torques:
  - strand: 1e-3 N m (a strand tip starts to bend at about 14 mN);
  - sheath: 0.1 N m.
- Stiffness = yield torque / 0.4 deg, critically damped: strand 7.8e-5, sheath 0.024 N m s.
- **Why 0.4 deg.** Annealed iron yields at a strain of about 0.1 % (200 MPa / 200 GPa). On this 1.2 mm-radius wire that
  allows an elastic curvature of 0.83 /m: 0.38 deg over an 8 mm segment.
  - At 1 deg (2.6x springier), a combed strand sprang back 4.5 mm.
  - At 1 deg, very_easy's second strand could not be combed within the target at any press depth.
- **Why not dry friction (v3: frictionloss, friction impedance .995).** MuJoCo's soft friction rows creep under any
  held load:
  - after a reference's success, with the robot still, the bundle settled 4-5 mm in ~30 s, to the frictionless
    equilibrium;
  - a sustained 1 mN on a strand tip crept it 123 mm in 3 min.
  - No friction setting held the load without breaking reproducers:

    | setting | settling after success | reproducer breakdowns |
    |---|---|---|
    | friction time constant 1 dt (refsafe off) | 4.1 mm | 1 |
    | contact impratio 10 | 3.9 mm | 3 |
    | noslip | 45 mm | |
    | friction impedance .9999 | 2.3 mm, still moving | violent inputs explode |

**The grasp.** Weld solref 1.25 ms. The Robotiq's slip force is 65 N (shared, `grasp.WELD_SLIP_FORCE_N`), 2 * mu
0.65 * 50 N of grip; the references' largest 20 ms mean is 19.4 N.
- On the exact constraint diagonal (b1ec43997, `MUJOCO_CORE.md`) the weld's rows are regularised by their true A_ii.
  At fiber's grasp the weld now gives along its translation and holds its rotation:

  | weld rows | before | exact diagonal |
  |---|---|---|
  | translation | 0.016 mm at 12.6 N, about 800 N/mm | 0.26 mm at 9.2 N, about 30 N/mm |
  | rotation | 0.0060 rad at 0.58 N m, about 95 N m/rad | 0.00057 rad at 0.41 N m, about 750 N m/rad |

  The sheath no longer droops in the hand.

## Wire checks (very_easy instance, a probe pushing a strand / the sheath; development probes `wire_probe.py`, `wire_long.py`, not included)

| check | strand (fiber_00_05) | sheath (lead_10) |
|---|---|---|
| (a) no contact, 10 s / 10 min | 0.0 mm / 0.0 mm | 0.0 mm |
| (b) force onset / motion onset | -0.009 / -0.039 mm; 0 N before touch | -0.033 / -0.064 mm; 0 N before touch |
| (c) bend kept after release | root joint 5.1 → 4.7 deg | below the sheath's yield: elastic, 4.6 of 10.8 mm back |
| (c) 60 s after release | 0.09 mm | 0.84 mm |
| (c) steady tip force, 1 s: 0.1 / 1 / 10 mN | 0.02 / 0.19 / 2.6 mm (v3: 0.63 / 5.6 / 38) | |
| (c) 1 mN held for 3 min | 0.39 mm of elastic give, then still (v3: 123 mm) | |
| (c) resting contact 5 mm past touch, 3 min | holds 17.7 mN, 0.04 mm | |
| (d) bend shape | smooth, 0 kinks | smooth, 0 kinks |

## References

The released (Genesis) references comb a floppy cable. On the wire they fail: combs spring back, and in the feed the
strands fold outward. They are **re-planned, seeded by the released ones**; goal, instances and texts are unchanged.
1. The released pickup rows (up to the held, settled sheath).
2. The planner's own comb loop (`topogym/planning/fiber_bundling/stages.py`: `plan_comb`, the strand farthest from the
   sheath axis first, from the measured state).
   - Presses are deeper, d_press .020-.030 m instead of .032, because a wire strand springs back by its elastic range.
   - Candidates are tried in forked copies of the process; a strand is accepted at 1.5 mm inside the 11.5 mm target.
   - The planner's guided 16 mm target is the fallback.
3. The planner's feed stage, along the measured front link (the planner assumed it collinear with the grasped link).
   Where open-loop rows drift, the feed runs **closed loop**, as the planner's own policy runs it (it waits on a pose
   gate):
   - a planned, collision-checked leg aligns the front link with the ring axis;
   - every control step, the measured front point's lateral offset is integrated into the commanded pose;
   - the front advances only while it is within 1 mm of the axis.
   - Why: a wire strand whose tip meets the ring carries compression up to its buckling load (~18 N), where the floppy
     strands gave way at a few mN. Open-loop rows let the held sheath drift 5-9 mm off its 1.5 mm clearance and land on
     the rim.
4. On the exact diagonal, very_easy's and easier's closed-loop rows, replayed, still carried their 8 mm lateral
   correction for the old weld's droop. The front reached the ring 5-7 mm off its axis, the front link met the rim, and
   the hand pushed on: the weld stretched 1.8 mm and held 62.7 N (20 ms mean) until the link slipped through. No strand
   sank into the rim (at most 0.07 mm, 0.3 N; a development trace, `jamtrace.py`). Those two feeds are re-planned on the
   exact diagonal from the end of their combs (the same pickup and comb rows), with more gates:
   - a slow zone: 4 mm/s from 15 mm before to 10 mm past the ring plane, the front within 0.5 mm of the axis;
   - no advance while the commanded front leads the measured one by more than 3 mm;
   - no advance while the weld holds more than 20 N, and a step back above 26 N (neither was reached).

   Easy and medium keep their rows (open-loop feeds).

The tools are in `source/scripts/robotangle/references/fiber_bundling/`: `replan.py` (the re-planner: pickup, comb loop, gated
closed-loop feed) and `weldforce.py` (the weld's force and the slip rule's 20 ms mean, the final strand radii). The
recording is `source/scripts/robotangle/references/interlocked_claws/rerecord.py` (any MuJoCo family). Run them with the
RoboTangle image's Python, as the stress tools. The recipe of the -i01 references, on the exact diagonal:
- The instance: `rcb_topogym.py make-instance --task fiber_bundling --variation fiber_bundling_<tier> --scene-seed 0
  --gripper-space native --simulator-overrides <file>`, the file holding the released instance's `grasp_assist`
  (`source/scripts/robotangle/mujoco_release.json`).
- very_easy and easier. Their pickup and combs, rows 0-2940 and 0-2840, are the v4 references' (replan.py from the
  released rows, `--presses .032,.030,.028,.026,.024,.022,.020`, on bbdbae3ad + fiber v4). The feed is re-planned from
  them:

  ```
  replan.py very_easy <dir> --instance <instance.json> \
      --from-rows scripts/robotangle/oracles/fiber-bundling-very-easy-i01/privileged.npz --prefix-rows 2941 \
      --near-mm 15 --near-past-mm 10 --near-speed .004 --near-gate .0005 --lag-max .003 --force-hold 20 --force-back 26
  ```

  easier: the same with its oracle and `--prefix-rows 2841`. The run writes `<dir>/rows.npz`.
- Recording, every tier: `rerecord.py <instance.json> <rows.npz> <privileged.npz> --tail 40`. For very_easy and easier
  the rows are the re-planned ones; easy and medium keep the v4 references' rows, which are their -i01 rows.
- The check: `weldforce.py <instance.json> <privileged.npz> <out.json>`.

Run from the repo on e9e54207d (F1 included), these files reproduce very_easy's and easier's references row for row,
with the same tick ledgers and final state hashes.

**Re-recorded on c64294012 (the Robotiq's spring links at springref 2.62, the pads parallel; a Robotiq grasp lets go
0.2 s after its fingers leave), FB2/FB3 included.** The current files in
`scripts/robotangle/oracles/fiber-bundling-<tier>-i01/` (`privileged.npz`; `standard.npz` is the same file; development
names, the tier-named -i01 directories are not included in this repository) are these recordings: the same rows, executed live until the first success (`retag.sh`: `validate_tier.sh`, then two fresh
observer replays, the weld force and the 60 s hold). The births are unchanged; the instance bytes change with the robot
MJCF.

| tier | rows | final state sha | strands at success (mm) | weld 20 ms mean / max |
|---|---|---|---|---|
| very_easy | 3722 | c223ae53c4a1062b | 9.25, 9.82 | 18.8 / 18.8 N |
| easier | 3607 | 43689f73645291c1 | 8.0, 10.19, 9.43 | 18.8 / 18.8 N |
| easy | 5009 | 06063247f02ded66 | 8.53, 9.37, 11.66, 8.0 | 18.4 / 18.4 N |
| medium | 8116 | 44a467ecde888924 | 10.6, 8.78, 9.82, 8.0, 9.46, 10.27, 9.99, 9.93 | 19.5 / 19.5 N |

easier now succeeds one row earlier (3607 rows, its first 3607 rows unchanged). All four: grade success (det 1,
review 0); nop hold and nop missing 0; standard live success with a finalize replay match; leak 0 / 18; the stress
observer (criteria v2.9) CLEAN, the same sha in 2 fresh processes; held 60 s after success, complete. Weld time
without finger contact on springref 0 was 0.01-0.03 % (only the distal pad boxes touched).

**Re-recorded on prisms (2026-10-12, the cylinder -> prism rule b2b751580, universal).** Re-made on the head
(f9db156d16, release_rebuild's MAKE_ONE: the released `grasp_assist`), each tier builds 71 robot-touchable cylinders as
32-facet prisms (the terminal gland, the six trunk shells, the 64 ring tube segments), and its birth changes. The same
rows were recorded again on those instances (`rerecord.py --tail 40`, the recipe above) and replayed in 2 fresh
processes each (the cluster):

| tier | birth (prisms) | rows | final state sha | 2 fresh replays | weld 20 ms mean |
|---|---|---|---|---|---|
| very_easy | f138fe7ea8d7de37 | 3722 | c2d5a617ceddf9ef | same hash | 18.95 N |
| easier | 44971f84830ab741 | 3607 | d979d1959ae56652 | same hash | 18.95 N |
| easy | 88a8f2192311bf0c | 5008 (one row earlier) | 2588ea321f2a69f4 | same hash | 20.1 N |
| medium | e3cc9a212b1a924f | 8116 | 0a12dc62f5a7cd5f | same hash | 21.2 N |

The earlier recording, on b1ec43997 (the exact constraint diagonal) + fiber v4:

| tier | rows | final state sha | strands at success (mm) | feed | weld 20 ms mean / max |
|---|---|---|---|---|---|
| very_easy | 3722 | c61bfa62a37b39da | 9.13, 9.73 | closed loop, gated | 19.2 / 19.2 N |
| easier | 3608 | 5d5b14dafbd4c467 | 8.0, 9.98, 9.57 | closed loop, gated | 19.2 / 19.2 N |
| easy | 5009 | f298ab9a8a0d3630 | 8.23, 9.45, 10.41, 8.0 | open loop | 19.0 / 19.0 N |
| medium | 8116 | a9efd7c46e2b2896 | 10.54, 8.78, 9.64, 8.0, 9.42, 10.31, 10.33, 10.03 | open loop | 19.4 / 19.5 N |

The largest 20 ms means come in a comb press (rows 748-1488); the feeds stay at or below 17.2 N.

All four on that build:
- the same final state as with the flag set on the 0f8654a49 model;
- grade: success, det 1, review 0;
- nop hold and nop missing: 0;
- standard: live success, finalize replay match;
- leak: 0 / 18;
- the stress observer (criteria v2.8): CLEAN, the same sha in 2 fresh processes;
- held 60 s after success: complete. The bundle moves 1.3-1.6 mm within 10 s as the arm settles to its last command
  and the weld relaxes; from 10 s to 60 s nothing moves.

Before the exact diagonal (bbdbae3ad + fiber v4; bit-identical on a0b4dddef and 0f8654a49, the Robotiq's frictionless
spring links and drivers): very_easy 3556 rows, 7af6d2d3fc24a031; easier 3459, 8b8a6e91fa1aa0cc; easy 5009,
991b03208bfd3873; medium 8116, 75377edd4df10081; weld 20 ms means 19.9-27.0 N.

## Stress

On the exact diagonal: the validator's round (FB1).

**FB2: the sheath x ring contact pair class (13faed450).** The validator's FB2 round on F1 (criteria v2.9) found one
BREAKDOWN on easier, prefix2_e7 (the easier reference's rows 0-3411 plus a 9-row tail; final state 76c7a66be9452c3d):
- the sheath's front link trunk_05 (its shell cylinder, r 12 mm) sank 15.1 mm into a ring geom (a cylinder of r 4 mm,
  half-length 0.86 mm), and the depth jumped 4 -> 14.8 mm in one tick;
- the force peaked at 1132 N, and 11.9 J was created;
- the sheath's own contact parameters had won the contact (solref .00125 2, solimp .95 .99).

The fix gives the sheath and the ring their own contact class, an explicit pair for every `fiber_bundle::trunk_*`
geom x `bundling_ring::ring` geom (1536 per scene; the profile field `pairs`, e8fd2c34a):
- condim 3, friction 1 1 .005 .0001 .0001;
- solref .00125 2, solimp **.999 .9999** .001 .5 2;
- margin and gap 0.

Strand-ring contacts keep their parameters. Two impedances were tested with a copy of the validator's v2.9 tools, on
p2e7, the 21 fiber corpus streams and the 7 fiber reproducers:

| build | p2e7 | deepest task contact | energy created (max) | BREAKDOWN |
|---|---|---|---|---|
| F1 (before) | BREAKDOWN | 15.1 mm (p2e7) | 11.9 J (p2e7) | 1 / 29 |
| pairs at .99 / .999 | BREAKDOWN | 11.9 mm, 2407 N (p2e7) | 0.057 J | 1 / 29 |
| **pairs at .999 / .9999** | VIOLENT, causal (speed 3.9 m/s, whips) | 1.48 mm (F4); p2e7 0.81 mm | 0.057 J | **0 / 29** |

The stiffer setting is the one built: the stiffest without energy creation.

The four -i01 references are bit-identical on it. In fresh replays they reach the same final states, with no
sheath-ring contact, as recorded. The instances' births are unchanged (very_easy ce6d0969c5b7d5fe, easier
6ad3cc09347376ed, easy 39a9eaef93ddaad8, medium 0468dd15bfc64f44, hard 2139031d1a6b5961). Only the scene bytes change:
the scene now carries the pairs. The validator's FB2 re-round (easier, plus a short targeted round on very_easy, easy
and medium) runs on this build.

**FB3: the sheath meets only the ring tube's spheres (ca3afe9dc).** That re-round (FB3_1k on 13faed450) found
prefix15_e6: trunk_05 "13.7 mm" into the ring at 115.7 kN, +3.57 J. Forensics showed the depth was not real:
- the sheath's flat front cap pressed up against the ring's underside, 0.1-0.37 mm deep on the tube's cylinder segments;
- for single MuJoCo steps, the general convex collider (GJK/EPA) returned one short cylinder's witness (1.7 mm long)
  5-14 mm deep along its own axis (the ring's tangent);
- the near-hard pair then pushed 20-115 kN for one step.

An offline test reproduced it: 11.5-14.6 % wrong witnesses in jittered poses. The result depended on the argument
order.

The pairs now take only the tube's 64 spheres. They are 1.72 mm apart, so the dents are at most 0.093 mm. Sphere x
cylinder and sphere x sphere are analytic contacts, and the same test gave 0 / 4000 wrong witnesses with them. The scene
gains no geom, and the strands keep the whole tube. A pairs entry can now name `geom_types` per selector. On this build:
- the four -i01 references are bit-identical (fresh replays, the same final states, no sheath-ring contact);
- the births are unchanged;
- p15e6 reads CLEAN, and p2e7 reads VIOLENT, causal (0.87 mm);
- the 21 corpus streams and the 7 reproducers: 0 BREAKDOWN, energy created at most 0.057 J.

The same shape class remains in two places: the sheath's cylinder against the fixed terminal's cylinder and, in the
redesign, against the formboard pegs' caps. Neither has been flagged so far.

Before it, a self-check with criteria v2.7 on 0f8654a49: PASS on the four reference tiers (references CLEAN, the same
sha in 2 fresh processes; nops CLEAN); the corpus and the fiber reproducers without BREAKDOWN; a search of 74.6k
controls (hard included) with 0 confirmed.

Criteria v2.5 on bbdbae3ad + fiber v4, on all five instances:
- the 21 corpus streams: no BREAKDOWN;
- the 16 fiber reproducers (repo F1-F5x and the S1 set): no BREAKDOWN;
- nops: CLEAN;
- a self-check search of 70k controls: 0 confirmed BREAKDOWN.

Hard: physics only, its search finds none.

Known:
- **easier / easy cem_e12** (the v2.4 flags): a flung sheath stopped by a still robot link. Each strand, held stiff,
  pivots about its root, and the tips speed up 10-16 % while the bundle loses energy. v2.5 rule 6 reads this as a
  chain's redistribution (VIOLENT).
- **S1 cem_e52:** Panda link7 against a trunk link at ~530 N. It was the last energy creation; the cable's overdamped
  contact removes it.
