# knot_untying on MuJoCo

The released Genesis family is the ALOHA untying a knot in an IPC FEM rope. It has five tiers: very_easy, easier,
easy, medium and hard. The double_overhand variant has no released instance, so it is not ported. On MuJoCo the rope
is a rigid capsule chain (`topogym/world/mujoco/rope_chain.py`), using profile `knot_chain_v7`
(`topogym/world/mujoco/profiles.py`). The rope agent ported it on 2026-10-09/10; its working evidence is in
the work notes

## The rope

- **Links.** Capsules of the rope's radius (4 mm) between nodes at equal material spacing of about 4.5 mm: 220 to 301
  links. Each link is a free body; a connect equality joins it to the next.
  - Maximal coordinates keep every Jacobian row local, so the sparse Newton solve stays cheap.
  - A cable in generalized coordinates gave dense Hessians: 224-525 ms per step at 100 links.
- **Node connects.** Impedance .9999 (solref 1.25 ms). MuJoCo's softness is mass-normalised and a link weighs
  0.1-0.2 g. Softer connects (.95-.99) opened about 0.1 mm per newton of tension.
- **Bend and twist.** A discrete elastic rod: U = ½ θᵀKθ, with K from the rope's E and ν (the released material).
  - The couples are exact (−J_r⁻ᵀ K θ), applied through qfrc_applied before every MuJoCo step at the current pose.
  - Each link's rotational inertia is raised to at least 24 k dt² per axis, and the rod damps the relative angular
    velocity with 0.15 I / dt (knot_chain_v5; v4 used 12 / 0.15, v3 8 / 0.3, v2 4 / 0.3).
    - Four blow-ups drove the floor up. Each was an exponentially growing zig-zag of the explicit rod's relative
      rotations in a stretch of rope under tension:
      - e0 (found on v2): the tail pinched between the fingers;
      - e1 (v2): mid-rope sliding on the table;
      - e2 (v3, also on the exact constraint diagonal): the rope pulled taut between the hands at ~6 m/s;
      - K3 (v4, the GPU search): a flung rope's free end whip-cracking at 13-18 m/s and curling past the bend stops.
    - For e0-e2, the stability bound of explicit couples with explicit damping is
      1 + k_T / k < inertia_rule × (1 − 2 damping_rule), with k_T the tension's stiffness through the node connects.
      It ordered the fixes, but K3 shows its limit: switching the damping off (the formula's 12 on v4) makes K3 worse.
      The inertia floor is the lever. The sweeps are under Stress.
    - Cost: bending waves run 6.4 % slower at 0.2 m wavelength and 20 % at 0.1 m. The floor is 25x the links'
      physical rotational inertia; the rod damping is 3.6 k dt.
- **Bend stops.** A sphere per link touches only its neighbours' spheres, so a joint stops at 0.9 rad: a bend radius
  of 5 mm, 1.25 rope radii.
- **Speed clamp (knot_chain_v6, 2026-10-11).** Extra damping on a link only while it is far faster than any reference:
  above 2 m/s centre-of-mass speed a force −0.5 m/dt (1 − 2/|v|) v at its centre of mass, and above 150 rad/s spin a
  couple −0.5 I/dt (1 − 150/|ω|) ω (`RopeChain._clamp`, per MuJoCo step, in qfrc_applied).
  - Why: a whipped free tail spinning at hundreds of rad/s. Each explicit step moves a node connect's two anchors
    along their tangents, opening a gap of about ω² l dt² that the soft row closes into kinetic energy, so the rows'
    net work grows roughly as ω⁴. The GPU search hit it twice on v5:
    - easier kn5_22 top000_b1_e79: the connects' net work rose from +0.04 to +200 mJ per step within 30 steps, as a
      wave running out to the tail, then a bad qacc at 2560 m/s;
    - easy top029_b1_e151: +0.004 → +0.9 mJ per step at 40 → 700 rad/s, then an uncaused 14.0 → 15.4 m/s jump.
    Every other force class balances the kinetic energy there (a per-step energy trace by class, work/dev/tail_trace2.py
    in the rope agent's evidence).
  - The thresholds are 3x the references' maxima (0.63 m/s, 48 rad/s). It only removes energy, and the stress
    observer books it as dissipation (W). Below both thresholds it writes nothing, so a run that never reaches them is
    bit-identical to v5: the five -i01 references replay to their v5 hashes with 0 activations.
  - It is read from the live profile at load, frozen scenes included, so the v5 births stand. The observer reports
    its activations per episode (summary `rope_clamp`), to show how often an agent's actions reach it.
- **Tension guard (knot_chain_v7, 2026-10-11).** A link's rotational inertia is raised while its node connects
  carry more than 30 E I / l² (14.9 N on this rope): to I0 · T / (30 E I / l²), relaxing by 1 % per MuJoCo step once
  the tension drops (`RopeChain._tension_guard`, before every MuJoCo step, from the previous step's connect forces).
  - Why: the explicit rod is stable only below a critical chain tension that grows with the links' rotational inertia
    (the cinch test: ~30 N locally at the v5 floor). The GPU search found the robot pinching the rope between the
    left wrist_link and gripper_base (easier kn5_22 top054_b1_e229, kn6_22 top004_b0_e84): the chain tension reached
    ~38 N, and a mode grew ~3x per MuJoCo step through the rod and the connects, to a bad qacc at 1317 m/s. Every
    link stayed slower than the speed clamp's thresholds until the last few steps. Per-step books: the robot's
    contacts fed 1-5 mJ per step into the stretched chain for 10 ticks; then the rod damping's work turned positive
    and grew 3x per step.
  - Energy: raising the inertia keeps the link's angular momentum (per body axis) and its centre-of-mass velocity,
    and relaxing keeps its spin, so the kinetic energy only drops (the observer reports it as
    `tension_guard_loss_j`). The guard writes nothing while every link stays at or below the threshold. The -i01
    references peak at 1.1 N (2.2 E I / l²), so they replay bit-identically.
  - The alternatives tried on e229, on copies of the same tree:
    - a stronger speed clamp (rate 1): still a bad qacc;
    - implicit damping of the links' spin (MuJoCo dof damping, which implicitfast integrates outside the constraint
      solve): 55 J created through the connects;
    - a soft contact class for the robot outside its fingers: the rope sank 10+ mm into the wrist;
    - finger-only robot contact: CLEAN, but the arm passes through the rope, and the easier reference (its rope
      resting on right/base_link) no longer succeeds.
    With the guard, e229 is CLEAN and e84 VIOLENT (it leaves the workspace). The guard raised the pinched links'
    inertia up to 2.5x and removed 2.5 mJ.
- **Contacts.** Every link touches everything except its 3 neighbours on each side.
  - Friction 0.3 (the material's friction_mu); the ALOHA's fingers keep μ 1, and MuJoCo takes the larger.
  - **Contact margin 0.2 mm,** the IPC rope's d_hat (`materials.rope_ipc_parameters`). Contacts act within that
    distance, as IPC's barrier does, so a resting contact keeps a positive gap: +0.14 to +0.16 mm, instead of
    MuJoCo's 0.04-0.06 mm penetration.
  - The release rule needs that margin. It lets the released rope touch the robot's fixed parts but never intersect
    them (exact check at 1 nm). Without the margin, easier's goal held from row 1100, but the rope lay against
    `right/base_link`, robot_clear stayed false, and the Session never confirmed the success.
- **Robot.** The ALOHA as interlocked_claws imports it, with claws_v5's stiff follower-finger coupling
  (robot.coupling). All of the robot touches the rope (robot_contact "full"). There is no assisted grasp: the task
  is a friction pinch.

## Birth, predicates and failures

- **Birth.** The rope is born from the released instance's frozen IPC settled state (`SimulatorConfig.mujoco_birth`).
  Link frames are the rope's material frames. The chord error at birth is ≤ 0.081 mm.
- **Predicates.** They read a FEM view of the chain, unchanged: the frozen volume's vertices, skinned linearly on
  material arc length as the render bridge's ChainSkin does.
- **rope_volume_invalid never fires.** That failure is an inverted or collapsed FEM volume persisting 0.25 s under the
  Session's rule R2. Its cause, a crushed tetrahedral volume, does not exist on a chain of rigid links, and the chain
  reports volume ratio 1. Nothing replaces the rule; the protocol is unchanged.

## Energy books (stress observer)

The rod is booked as stored energy: E includes ½ θᵀKθ, and W leaves out the rod's elastic couples
(`rope_energy` / `rope_elastic_qfrc`; checked by `scripts/robotangle/test_rope_energy.py`). A rope link's stored energy
for the pop clause includes its rod joints. Old and new books agree within 2.2e-7 J. A free swing under the rod
alone shows X ≈ 0: the largest rise above the running minimum is 4e-5 J.

## Chain integrity under the robot's real loads

In-Session rows at the servo caps (π rad/s, the ViperX torques), with the node gap measured after every MuJoCo step
(on knot_chain_v2):

| load | node gap |
|---|---|
| two-arm pulls | 0.005-0.014 mm |
| ±1.5 rad wrist twists | 0.007-0.071 mm |
| whips at link speeds of 2.6 m/s | 0.031-0.093 mm |

The gap stays flat under a steady load. No real grasp passes a rope more than about 8 N:
- the friction pinch slips below 1 N;
- a knot's loop slides through its crossing at about 5 N, measured with a test-only weld;
- a test-only stiff weld held a 14.2 N two-arm pull at 0.030 mm, and pulling further leaves the arms' reach.

## References (step 1)

The references are committed as `scripts/robotangle/oracles/knot-untying-<tier>-i01/privileged.npz`; `standard.npz`
is the same file. Each is the released route plus 200 hold rows of its last row, executed open-loop in a fresh process
with knot_chain_v5 on the exact constraint diagonal. The rope agent's tree was at 1eb2a37ed's code, with instances
made on that code. All five tiers succeed and replay to the same state in two fresh processes.

| tier | rows | min | final hash @ success row (two fresh replays) | file sha256, first 16 (privileged = standard) | nops | observer (v2.8) | node gap / robot depth |
|---|---|---|---|---|---|---|---|
| very_easy | 1208 | 9.4 | 3681a9d8565fcdac @1208 | f6accf43f06cf14b | fail / fail | CLEAN, X −15.0 J | 0.006 / 0.37 mm |
| easier | 1073 | 10.6 | 0f05e05b290994bf @1073 | c5833a2a4d4ca1c2 | fail / fail | CLEAN, X −15.3 J | 0.004 / 0.36 mm |
| easy | 753 | 5.3 | 191d932660caa264 @753 | 366acc4f8e8a794c | fail / fail | CLEAN, X −10.0 J | 0.000 / 0.21 mm |
| medium | 1944 | 13.4 | 98a806f32bcb1b85 @1944 | 2b690038b73e0bc9 | fail / fail | CLEAN, X −16.8 J | 0.005 / 0.42 mm |
| hard | 2391 | 21.4 | 46c94c1ebe3d8458 @2391 | 8bf03b10223e6d0e | fail / fail | CLEAN, X −19.8 J | 0.004 / 0.97 mm |

**On ccd4ba7fa** (the soft contact class at a constant impedance, solimp .95 .95; knot_chain_v7 with the tension guard;
the same rows and the same births, re-bound to the rope agent's tree kn13), the five references replay in two fresh
processes each (validate, the cluster, 2026-10-11) to: very_easy cb0e1ae48af8ffc0 @1208, easier 2b9088531bc1a524 @1072,
easy 191d932660caa264 @753 (unchanged), medium 25da90d051da4a35 @1941, hard 14ed4da2f8b16ff4 @2391. All succeed; easier
and medium succeed one and three rows earlier than above.

The knot_chain_v4 references (pushed in a2933d0bb) were, in the same order: c74c1389 @1207, e1b29996 @1076,
9f634774 @751, 658b8dca @1929, e3398dbb @2362.

- **very_easy's two-arm pull is one continuous ramp.** The released route drives it (rows 208-717, 5-18 s) as 13
  one-second segments. Each starts with a one-row command step of 15-33 mrad on the elbows and wrist_angles, and
  1-4 mrad on the right waist, then rises along a bell and stops. The reference replaces those rows of the arm
  columns with a clamped cubic spline through the step midpoints (`work/dev/smooth_phase.py` of the rope agent), so
  the phase starts and ends where it did, at rest.
  - The peak joint-space step falls from 57.5 to 8.3 mrad per row.
  - The largest departure from the released commands is 44 mrad, on the left elbow, where the first segment's step
    is spread over the first second.
  - On knot_chain_v4 the released route, unsmoothed, also succeeded (1208 rows, d14b030a6e2661df, CLEAN).
- Minutes are wall time on the a cloud host box at load 70-110; every tier is within the 72-minute rule.
- The nops are a missing trajectory and 400 rows holding the birth command.
- The observer reads criteria v2.8 with the rope books.
- Frames are birth and final, rendered by the Session's own route (nyx_direct).
- The instances keep the released `gripper_space` "width", and the references' gripper columns are widths.
- Earlier rounds:
  - On knot_chain_v3 with the exact diagonal, the v3 references also replayed to success at their rows.
  - On knot_chain_v2 all five tiers passed step 1 before the self-check found the rod zig-zag.

## Stress (step 2)

- **Four blow-ups,** all the rod zig-zag above, kept as reproducers (on the rope agent's box in
  /workspace/mj/rope/stress/knot_repro):
  - K0 and K1, found on knot_chain_v2 by the self-check's prefix search (e0, e1). In their reports the "robot-robot
    contact" (left finger against right wrist, 35.6 mm / 2242 N) is the arms overlapping after MuJoCo's reset to
    qpos0. It is not a cause.
  - K2, found on knot_chain_v3 by the same search (e2, 728 rows). It also blows up on the exact constraint diagonal
    (b1ec43997), at row 728.
  - K3, found on knot_chain_v4 by the GPU knot search, then confirmed on CPU (easy, gpusearch's b0_e99 cut at 268 rows,
    sha256 42629f72).
    - Rows 0-212 are the easy reference. From row 213 the arms drive into their joint-limit corners, then hold there
      from row 229.
    - The rope, flung, flies free. At row 267 its end whip-cracks: link 0 goes 7.6 -> 18 m/s in 4 ms, and node gaps
      grow 0.2 -> 2.8 mm.
    - The end curls past the 0.9 rad stops to 1.2 rad (stop penetration 1.6 mm / 37 N). Links 13-25 then grow
      17 -> 700 m/s in 6 MuJoCo steps until BADQACC.
    - No robot-robot contact precedes it. The solver used at most 10 of 100 iterations per step. The 7.4 kN
      robot-robot contact in its report is MuJoCo's reset to qpos0, as core confirmed.
- **The headroom sweep.** In-memory variants of the rod from the recorded state 10 rows before each fault (e2: rows
  718-735). The headroom is inertia_rule × (1 − 2 damping_rule).

  | headroom | inertia_rule / damping_rule | e0, e1 (v2's state) | e2 (v3's state), exact diagonal | e2, without it |
  |---|---|---|---|---|
  | 1.6 | 4 / 0.3 (v2) | blow up | | |
  | 2.0 | 5 / 0.3 | stable | | |
  | 2.1, 2.4 | 5.33, 6 / 0.3 | stable | | |
  | 3.2 | 8 / 0.3 (v3) | stable | blows up | blows up |
  | 3.7 | 5.33 / 0.15 | stable | | blows up |
  | 4.2 | 6 / 0.15 | | | blows up |
  | 4.8 | 12 / 0.3 | stable | stable | stable |
  | 5.6 | 8 / 0.15 | | stable | stable |
  | 6.4 | 16 / 0.3 | | stable | stable |
  | 8.4 | 12 / 0.15 (v4) | | stable | |

  - Tripling the damping (4 / 0.9: negative headroom) blows e0 and e1 up at once.
  - At headroom 3.2, clamping the rod couple at 0.9 rad, the bend stops off, or condim 3 on the fingers all leave e2
    blowing up.
  - v4 holds at 1.5x its effective stiffness (headroom 5.6) in every case.
- **The K3 sweep** (v5's basis). In-memory variants from the state 10 rows before the fault (rows 257-285):

  | variant | result |
  |---|---|
  | v4: inertia_rule 12, damping_rule 0.15 | blows up (row 268) |
  | robot-robot pairs filtered out; solver iterations x5 | the same fault, bit for bit |
  | rope contact solref x0.5 | no change (MuJoCo's 2 dt floor); the same fault |
  | rod damping off | blows up earlier (row 267) |
  | inertia_rule 15, 16, 18, 27 | stable (whip 13-17 m/s, no fault) |
  | inertia_rule 18, damping_rule 0.10 | stable |
  | 16 MuJoCo steps per tick (dt 0.3125 ms), at the v4 floor and at rules 27 / 18 / 16 for that step | stable |
  | node connects at impedance .99; rope contact margin 1 mm | stable |

  - v4 fell short by under 25 %: the threshold lies between rules 12 and 15.
  - v5 (rule 24) still holds at 1.5x the effective stiffness (rule 16).
  - The other two stable rows change the chain (softer connects) or the contact timing (margin), not the rod's margin.
  - Run under v5 from birth: e99's full stream and K0-K2 run without a fault.
  - Rejected option: 16 MuJoCo steps per tick with rule 24 at that step. It holds K3 with the same margin and halves
    the floor (bending waves 1.7 % / 6.6 % slower), but doubles the CPU of every knot run.
  - Side-by-side 1x videos of v4 and v5 from the same state showed no sluggish or stiff rope on v5. They used
    nyx_direct on easy's pull (rows 290-410, cam_high) and e99's fling (rows 225-267, cam_high and cam_low). The
    fling's peak link speed is 11.5 m/s on v5 against 11.7 on v4.
- **The cost in bending-wave speed** (against a rod without rotational inertia):

  | profile | inertia floor / the links' physical inertia | 0.2 m waves | 0.1 m waves |
  |---|---|---|---|
  | knot_chain_v2 | 4.2x | 1.2 % slower | 4.5 % slower |
  | knot_chain_v3 | 8.4x | 2.3 % slower | 8.6 % slower |
  | knot_chain_v4 | 12.6x | 3.4 % slower | 12 % slower |
  | knot_chain_v5 | 25.2x | 6.4 % slower | 20 % slower |

- **On knot_chain_v5** (the exact diagonal on), K0-K3 replay without a fault or BREAKDOWN (+40 hold rows):

  | reproducer | rows | verdict (criteria v2.8) | end state |
  |---|---|---|---|
  | K0 (e0) | 514 | VIOLENT: task speed 4.9 m/s, causal whips | f1cd84bb8649c21b |
  | K1 (e1) | 1430 | CLEAN | 5ed44c2a5bc461a3 |
  | K2 (e2) | 728 | VIOLENT: task speed 8.0 m/s, causal whips | 643906ded6cfae43 |
  | K3 (whip) | 268 | VIOLENT: task speed 12.1 m/s, a link at 873 rad/s, causal whips, the rope leaves the workspace | bd5d16a8b8612331 |

  The energy created stays ≤ 1.7 mJ in every run.
- **On knot_chain_v4** (the exact diagonal on), K0-K2 replayed without BREAKDOWN, twice each in fresh processes, to
  the same end state:

  | reproducer | rows | verdict (criteria v2.8) | end state |
  |---|---|---|---|
  | K0 (e0) | 514 | VIOLENT: task speed 3.1 m/s, causal whips | b9f2aed03c802f18 |
  | K1 (e1) | 1430 | CLEAN | a1808ffef40994b1 |
  | K2 (e2) | 728 | VIOLENT: task speed 6.1 m/s, causal whips | a434d2402c57edc8 |

  The energy created stays ≤ 1.4 mJ in every run. On knot_chain_v3 with the exact diagonal, K0 ran without a fault,
  K1 was CLEAN and K2 blew up at row 728.
- **The step-2 self-check on knot_chain_v4** (before K3; the formal round on v5 is the validator's), with criteria
  v2.8, search protocol 3, 12,000 controls per tier, seeds
  72001-72030 as on v3, the exact diagonal on, the i01 references): every tier PASSes.
  - The search ran 49-60 episodes per tier across random, directed, mutate, CEM, prefix and periodic: 0 BREAKDOWN.
  - Six episodes read VIOLENT, all through causal triggers: one contact above 2000 N (easier, random), and task
    speeds of 3.7-8.6 m/s with causal whips (prefix and mutate on easy, medium and hard).
  - Each reference replays CLEAN to its hash twice, each nop fails CLEAN, very_easy's 3 corpus streams read no
    BREAKDOWN, and K0 / K1 / K2 read VIOLENT / CLEAN / VIOLENT.
  - On v3 the same search found K2 (medium FAIL); on v2, K0 and K1.
- **The cinch fault.** Cinching a knot past a critical tension also blows the chain up (found by the soft-body
  redesign), and the threshold rises with the headroom. Test (rope agent, `work/dev/cinch_test.py`): the instance's
  knotted chain, its first link welded to the world and its last welded to a hand that pulls away at 2 cm/s, both
  welds as stiff as the node connects, the exact diagonal on. The knot cinches, the tension ramps, and the chain blows
  up within ~20 ms once the steady tension reaches:

  | instance | knot_chain_v3 | knot_chain_v4 | knot_chain_v5 |
  |---|---|---|---|
  | easy | 4.6 N | 15.1 N | 31.9 N |
  | medium | 4.5 N | 14.6 N (27.6 N inside the knot) | 16.2 N (29.2 N inside the knot) |

  On medium the fault comes at about the same tension inside the tightly cinched knot on v4 and v5 (28-29 N), so
  there the floor buys little. A contact_weld_v1 rope grasp lets go at 8.2-8.5 N, so from v4 on the weld slips before
  a cinch faults the chain. The friction pinch of the released tasks slips below 1 N.
- **Hooks and pins with the real robot** (main's question after the redesign's kinematic two-hook test reached
  22.6 N). Test: `work/dev/hook_check.py`, live in the Session on v4 at the servo caps and the ViperX torques.
  - The closed grippers hook the knot's main loop and pull apart, or pin (press 6 or 15 mm) the two tails 4 cm
    outside the knot and drag them apart, or hook one side and pin the other. Pulls are 0.3 m in 3 s or 0.6 s, all
    five tiers.
  - The largest chain tension of the 35 runs is 7.3 N (medium, as the two hooks land inside the knot), then 5.9 N
    (very_easy, a 15 mm press). Every peak comes when a finger lands or presses, not during the pulls.
  - The pins slip off the round rope within the first centimetres. The loose released knots feed rope from their free
    tails: two hooks in easy's loop reach 0.07 N.
  - No run faults, and the node gaps stay ≤ 0.07 mm. Some runs untie the knot.
- **Angular trigger.** At 60 rad/s it flagged the v2 references VIOLENT, through rope links turning at 80-112 rad/s
  (0.4 m/s across a 4.5 mm link). Criteria v2.7 counts a spin only on a link whose surface moves faster than 3 m/s.

## Rebuilding the instances

An instance is make-instance on MuJoCo with these overrides:
- `mujoco_birth`, the released instance's IPC settled-state manifest;
- the released instance's frozen rope volume, installed as a frozen asset;
- the released Genesis instance copied to `instance_assets/render_host/`, where the Genesis render bridge and
  nyx_direct read the rope's birth.

The rope agent's script is the work notes The rope agent's dev
bindings also name nyx_direct.py (`nyx_direct_sha256`), as `mk_tasks.dev_binding` does for families that render
"direct". The instances on knot_chain_v5 (1eb2a37ed's code) have the births very_easy 46dd7e5fda9ecfdf, easier
ab6b75ab24a5771b, easy 3fe175d34a4dbf8c, medium 3564452a7a4a8503 and hard 03de1494d1e7453e (v4's: 6a72fc56, 3e7e1540,
7f4c65ea, 9693ada1, 2a2426a3).

knot_untying is not in `scripts/robotangle/mujoco_release.json` yet. Adding it needs three things the release rebuild
does not do today:
- the make-instance `gripper_space` "width" (the file names "native" for every base);
- `mujoco_birth` pointing at the released instance's `instance_assets/ipc_state/manifest.json` (its
  "@released_instance" stands for the instance.json only);
- the frozen rope volume and the render host copied in as above.

## rd easier, re-timed: knot-untying-07-i01 (V3, 2026-10-18)
The knot_untying rd easier tier (a loose figure-eight, its standing end clamped under a staple, two posts beside it and a
curb) ships as knot-untying-07-i01 in RoboARC 0.2.1. Its 2026-10-14 route et1 (14172 rows) took ~39 % of the 3 h budget;
the reference is the same route re-timed, not re-planned (`redesign/knot_untying/robot/retime_search.py`): original waits
x3, moves while the rope was quiet in the original (every node < 2 mm/s) x3, other moves x1.75, gripper changes as
recorded, every arm joint step at most 0.0385 rad, then the final targets held to the Session's success. No physics,
substep or task change.
- Instance: premade (`redesign/knot_untying/premade/make_premade.py` in eai-roboarc:0.2.0-noinst, from inst10/easier_v8_r2
  with the 2026-10-13 back strip, `premade/region_source.py`), birth a3c58ab73130b9b3; two makes byte-identical, births
  exact.
- Reference g333: 7419 rows, end 8ae062ea4c733dbc, sha256 82c65a96cbaca655...; strict double replay (host A), the same end on
  host B and the cluster (Gold 6230R, 6338, Platinum 8592+); max arm joint step 0.0385 rad.
- Timing: 3037 s = 28.1 % of 3 h on host B (Ryzen 9 7950X, a quiet control in the same window), within the knot bar (<= 30 %,
  user 2026-10-14). The texts state 0.41 s per control step.
- Harbor oracle 1 / nop 0 in both modes; leak probe = knot-09's findings; stress (the cluster, 300 episodes, v2.13b): PASS.
- The rd hard tier (hu1) stays out: its re-timed untie never stays untied (the work notes).
- The user approved the review on 2026-10-18.
