# Stress validation for RoboTangle on MuJoCo

This is step 2 of porting a family to MuJoCo (`MUJOCO_PORTING_GUIDE.md`): **show that no input an agent can send makes
the simulator break.** The tools are in `tools/stress/`.

- They need only the standard library, numpy and mujoco, plus the repository's own `topogym` and `rcb_topogym`.
- They judge the *simulator*, not the agent. They add no episode-ending rule; the protocol stays as it is.
- Robot behaviour is limited by clamps (joint speed, torque, grasp slip), never failed.

Criteria **v2.13b** (adopted 2026-10-13) are in force for rounds from 2026-10-13 on; rounds PASSed before keep
their v2.12 verdicts. Any change needs three things: a traced false positive, a regression run that keeps every
reproducer flagged, and the coordinator's OK.

## In plain words

A run **fails** (BREAKDOWN) when the simulator does something no physical scene can do. These guards are absolute:

- a non-finite state, a MuJoCo instability or buffer warning, or the Session's own breakdown end;
- any contact deeper than 10 mm, or a task body sinking into the table;
- the scene gaining energy overall: X above 0.1 J, where X is the mechanical energy minus the work the robot did;
- a task body jumping in speed with nothing that could have caused it;
- a stored-energy "pop" that kicks a body by more than 3 m/s (its velocity changes by more than 3 m/s within one
  tick), when the energy was stored in a contact, a weld or connect, or a joint limit. A rope's own bend and twist
  energy springing back is physics (v2.10): it is reported as violent above 3 m/s, not failed. A pop fails only when
  the constraint-stored energy released into the link could alone move it that fast (v2.11).
- a VBD cloth gaining energy on its own: its balance X (its energy, contact storage included, minus the robot's work)
  rising more than 0.25 J above its lowest point; with two-way coupling, the cloth and the coupling together staying more
  than 0.25 J above their lowest point for 10 ms; a
  cloth particle kicked faster than 3 m/s only when the contact storage released into it could pay for it (v2.12).

Violent motion the robot caused is **reported, not failed** (VIOLENT): thrown pieces, whipped cables, kN impacts. A
local energy gain fails only when three independent signals all show it.

**The blind spot.** If a contact creates energy at the same moment as an impact nearby destroys more, the energy books
show no gain, and the criteria do not see it. Nothing visible comes of it, but it is not proven absent. The absolute guards
still hold in that case.

## Criteria v2.13b

v2.12's rules (below) on books taken as MuJoCo's integrator applies the forces, plus one fail class; every threshold is
unchanged.
- Contact work is booked at the step's start kinematics with its mean generalized velocity, J(q0)' f . (v0 + v1) / 2,
  the exact discrete identity of the velocity update (v2.12 took the mean of the start and end kinematics, which drifts
  on a long hold: orbits easier prefix14_e1 booked +22.7 mJ on a passive finger x cord pair the integrator books at
  +1.1 mJ, a false BREAKDOWN by rule 6).
- A task link's kinetic energy includes its joints' armature (rotor) energy, as MuJoCo's mass matrix does.
- energy_created(integrator): in one local event (the 0.1 s window of the local rule) the excess and the integrator's own
  residual (the excess less every contact / equality / limit row's work) both rose 0.25 J, and no other energy rule
  failed: the step, not a contact, made the energy. It fills the gap the integrator-side books open where v2.12 blamed a
  contact (S4, armature 1e-6: a real step-created 0.30 J).
- Regression (validate, 2026-10-13): the 30 corpus reproducers whose found-on builds exist (B1, B1b, B2, B3, C1, E1,
  E1m, E1p, F1-F5, F3b, F5x, G1-G3, G11, G12, G46, M1c easy / medium, M3, M3c easier / hard, M4c, M5, M6, M6full) stay
  BREAKDOWN with equal end states (F5 now names energy_created(integrator) 5.50 J where v2.12 blamed a 0.34 J contact;
  F4 adds it to uncaused_jump); G5-G10, J1, J2, K0-K3, O1, R1 and V1 are not covered (their builds are gone; chains2's
  S1 / S3-S5 and prefix7_e2/old on h315 stand for those classes); the CPU-profile blanket regression (59 streams) keeps
  every verdict and end state; chains2's A/B (143 streams) changes only prefix14_e1 (BREAKDOWN -> VIOLENT). The
  GPU-profile blanket set (111 streams) gates blanket rounds under v2.13b and runs when a GPU is free.

## Criteria v2.12

### Inputs

- An input is any sequence of action rows an agent can send to `Session.step`, privileged mode. Any values are allowed:
  the Session clips, clamps and guards them, exactly as for an agent.
- An input ends where the episode ends: a success, a failure predicate, or the Session's breakdown rule.
- Corpus streams get 40 hold controls after their last row, because a breakdown can come after the input stops.

### Measurements (`observer.py`, read-only)

A run with the observer ends on the same state hash as a run without it.

- **After every MuJoCo step** (0.625 ms), not only at the tick's end as the Session's own readout does:
  - non-finite state, any MuJoCo warning, solver iterations at the cap;
  - every contact's depth and full force, by pair category: task-robot, task-task, task-static, robot-static,
    robot-robot;
  - task-link speeds and spins. A link's surface speed is ω·r_link, where r_link is the farthest point of its geoms
    from its centre of mass;
  - the work of every non-conservative force (with implicitfast's qDeriv term) and of the constraint forces;
  - each contact pair's work, and (v2.9) each equality's and each limited joint's or tendon's work.
- **Fingers** (the robot's links named *finger*): the finger-finger contacts' peak force and depth, and the smallest
  A_ii (efc_diagA, the constraint diagonal) of the finger-finger contact rows, of every finger contact row, and of the
  finger joints' limit and equality rows (peaks `ff_row_inv_A`, `finger_row_inv_A`, `finger_eqlim_inv_A`, as 1 / A_ii).
  A degenerate row has A_ii near 0 and carries kN at about 1 mm: BL1 e30 before rule A, a left pad driven into the
  other finger's hull, reads 40.7 kN at 1.2 mm with A_ii 2.8e-4; with rule A the same stream reads 47 N. A low A_ii
  alone is not the mechanism: a finger pressed on its own folded arm's base reads 0.03 (about 33 kg effective) at
  278 N. So the signature is the combination: any contact row with A_ii below 0.1 carrying more than 2 kN, whatever
  its bodies (event `degenerate_row`, peak `degenerate_force`). These are measurements, never a verdict.
- **After MuJoCo's auto-reset** (v2.9): on a bad acceleration MuJoCo warns (mjWARN_BADQACC) and resets the state to
  qpos0. The warning fails the run (F1). From that step on the observer records no peaks and no events, so the run's
  numbers describe the physics up to the fault, not the jump to qpos0 (the summary's `post_reset` gives the step).
- **After every tick** (2.5 ms):
  - the causal check of every speed jump;
  - tunnelling and the workspace;
  - grasp losses by reason (e.g. `grasp_slipped`);
  - the energy balance.
- **The energy balance.**
  - E is kinetic + potential energy. W is the work of drives, gravity compensation, damping, fluid and applied forces.
    The potential includes a 2D flex's membrane and bending energy, which MuJoCo's own energy leaves out
    (`topogym/world/mujoco/energy.py`, `flex_energy`; checked by `tools/tests/test_flex_energy.py`).
    A rope chain's rod is stored energy: its elastic energy ½θᵀKθ is in E, and its elastic couples (applied forces)
    are left out of W (`topogym/world/mujoco/rope_chain.py`, `rope_energy` / `rope_elastic_qfrc`; checked by
    `tools/tests/test_rope_energy.py`).
    X = E − E0 − W.
  - A passive scene has X ≤ 0. In every run X equals the work of all constraint forces to within 0.1 J, so the books
    balance.
  - S = Σ max(0, −½ f·pos) over the active constraint rows estimates the soft constraints' stored energy.
- **Cloth (a 2D flex).** Its vertex bodies belong to no entity's links, so the observer makes each flex one
  articulated task entity. Every task check then covers the cloth: speeds, jumps (with rules 3 and 6, the flex's
  elastic energy as its exact potential), tunnelling per vertex, the workspace.
  - A flex contact carries geom −1 on the flex side. It is attributed to its vertex body (`con.vert`), or, for an
    element contact, to the element's vertex nearest the contact point.
  - A cloth patch grasp's pins count as welds, hand to vertex.
  - The per-contact books are vectorised for the 2-4k contacts a cloth makes per step. No-flex outputs and traces are
    byte-identical to the per-contact loops' (claws E1p, buzz B2, hedgehog H1, fiber F3 and F4). On the very_easy
    blanket reference the observer adds ~24 % to the run (1.21 against 0.98 s per control on a loaded host), ends on
    the same state, and its books close: X − Wc 5 mJ with the robot grasping and moving the cloth.
- **The observer's own errors.** An exception in the observer is recorded (`observer_error`, its traceback) and the
  run continues. It is never reported as a world fault. verdict.py lists it as a failure that names it a harness
  bug.

### Breakdown: any one fails the run

| | Breakdown | Threshold |
|---|---|---|
| F1 | Non-finite state, a MuJoCo instability or buffer warning, the Session's `simulator_instability` / `forced_breakdown`, a world fault | any |
| F2 | A contact deeper than 10 mm, task or robot, at any MuJoCo step | 10 mm |
| F3 | Tunnelling: a task link's centre of mass more than 10 mm below the tabletop, inside the table (2 cm inset) | 10 mm |
| F4 | An uncaused speed jump, or a stored-energy pop whose kick exceeds 3 m/s | rule; 3 m/s |
| F5 | Energy created, globally or locally | 0.1 J; 0.25 J |

**F4, the causal rule.** A rise of a task link's centre-of-mass speed by more than 0.5 m/s within one tick needs a cause.
It is causal when any of these holds:

1. **A partner moving fast enough.** A contact or weld partner, in this tick or the previous one, has a point speed
   v_partner with `|v_after| ≤ max(√(|v_before|² + ω·Iω/m), v_peak_50ms) + 2·v_partner + 0.1 m/s + |g|·Δt_tick`.
   - The 2·v_partner term covers an elastic hit by a much heavier body.
   - |g|·Δt_tick (v2.6) is gravity's own speed change within one tick: 0.025 m/s on 2.5 ms ticks, 0.39 m/s on
     rope_threading's 40 ms ticks.
   - The first term covers the body's own rotational energy turning into translation, and a rebound up to its own
     recent peak speed.
2. **A chain.** The task links connected to this one by task-task contacts (this tick or the previous one) have a
   non-task partner (a robot link, fixture or weld) that satisfies the same bound. An impulse can pass through several
   pieces within one step, while the observer samples speeds at the step's start.
3. **A cable whip.** For a link of an articulated entity, another entity hit the entity within the last 0.25 s at point
   speed u, and `|v_after| ≤ 6·(2u + 0.1)`. Whips are reported as violent but causal when the link ends above
   3 m/s (v2.8).
4. **A pop: stored-energy release.** The link's kinetic-energy gain in the tick is at most the drop of the stored energy
   acting on it (contacts, welds, joint limits; a rope link's rod joints). That is a squeeze, a weld or a bend being
   released. A pop whose kick, the change of the link's velocity vector within the tick |v_after − v_before|, exceeds
   3 m/s is still a breakdown (v2.9). Up to v2.8 the guard bounded the after-speed, so a link of a rope whipped to
   6-9.5 m/s that a 0.3-0.6 mJ release nudged by 0.6-2.0 m/s failed. For a body at rest the kick is its after-speed,
   as before.

   **The guard's scope (v2.10).** The guard is for energy stored in constraint violations, which have no physical
   counterpart: squeezes, penetrations, a weld's or connect's stretch, a loaded limit. A fast pop whose kinetic-energy
   gain the drop of the link's own rod-joint elastic energy (a rope's bend and twist potential, a modelled physical
   energy) pays for alone is no breakdown. It still reads VIOLENT through the speed trigger. The case is
   rope_threading medium kr3_4 top035_b2_e52 (rope_chain_v2): a release wave runs along links 186-193 toward the free
   end, and rope::193 goes 0.22 → 3.36 m/s with a kick of 3.22 m/s. Its 1.687 mJ stored drop was all rod-joint energy
   (the node connects held 0.3 µJ; there was no contact); its kinetic gain was 1.60 mJ. The event is
   `pop_rod_v210_fast`, `_kick` or `_after`, the suffix naming the class it had up to v2.9, so v2.9 and earlier read exactly.

   **The guard's energy share (v2.11).** A fast pop that the rod does not pay for alone is a breakdown only if the
   constraint-stored energy released into that link in the tick, E_c (its contact, equality and limit rows; the rod's
   elastic storage excluded), could alone move it at the guard's speed: v_c = sqrt(2 E_c / m_link) > 3 m/s. A mixed
   release whose constraint share is too small to make the pop is no breakdown; it reads VIOLENT through the speed
   trigger. The case is rope_threading medium KN4-r2 kt1_4 top035_b1_e144 on rope_chain_v3 with the constant soft
   impedance (ccd4ba7fa): the finger lets go of the pinched tail, and rope::198 goes 0.008 → 3.24 m/s with a kick of
   3.23 m/s. Before the tick it held 0.98 mJ in its rod joints, 0.23 mJ in the finger's contact rows and 0.05 mJ against
   R2_stand; the rod alone could not pay the 1.23 mJ gain, and the constraint share (0.29 mJ) gives v_c = 1.80 m/s.
   Every fast pop records `constraint_drop_j` and `vc_m_s`; the event is `pop_share_v211_fast`, `_kick` or `_after`.

   **The VBD cloth (v2.12).** A VBD cloth (blanket) keeps its own book (`vbd_books.VBDBook`, after every cloth step):
   X += (E(X1) − E(X0)) − W_robot, where E is the cloth's kinetic, gravity, membrane, bending, self-contact barrier and
   body-contact penalty energy on the step's frozen contact sets, and W_robot is the robot contacts' work on it. The
   pins' writes (W_pin) and contact-set changes between steps (W_set) are booked outside X. Three rules:
   - **The cloth's own guard:** X may not rise more than 0.25 J above its running minimum, judged apart from the system
     sum (MuJoCo-side dissipation cannot hide a cloth creation). The case is blanket_flatten easier4 BLF3b cem_e3 on
     blanket_vbd_gpu (694e790b5): two steps after a pin's strain release, one cloth step gains +2.34 J with KE ≤ 6 mJ
     (the solver's inertial initial guess put self-contact pairs deep inside the barrier); v2.11 read it CLEAN because the
     system's X was 17 J below its start and no contact had a net gain. X counts the storage on purpose: storage that
     booked robot work charged comes back on release (BLF2c random_e9: 1.05 J without it, 0.5 mJ with it).
   - **Pressed pile (user 2026-10-13).** Cloth steps taken while a gripper is held on its cloth floor are booked apart
     (X_press); the rest is X_free. A step counts as pressed when the Session's table guard stopped a gripper on the
     floor "cloth" within the last 0.25 s. A pressed pile's stored barrier and penalty energy is commanded by the agent, so
     a rise of X_press above 0.25 J is not a breakdown:
     - **WARNING** ("pressed-pile storage, agent-caused") when the cloth stays calm under the press: kinetic energy ≤ 0.05 J,
       no particle kick above 3 m/s, no pop;
     - **VIOLENT** (agent-caused) when it throws cloth;
     - X_free keeps the cloth's own guard (BREAKDOWN above 0.25 J).
     A round reports its warning rate per tier (in-search warnings / episodes, and how many were thrown) and fails above
     1 %. The case is blanket_flatten easy4 directed_e6 (W2b on b36, and on GPU): the right gripper is held on its cloth
     floor and the pile's energy swings ±0.1-0.5 J per step with KE 1-2 mJ (X +0.78 J GPU, +0.60 J twin). Books from
     before the split (no X_free_max_rise) read by X as before.
   - **A cloth particle's pop:** a particle kicked faster than 3 m/s in one cloth step fails only if the contact storage
     released into it (a self-contact pair's energy split over its four particles, a body contact's over its corners)
     alone gives v_c = sqrt(2 S / m) > 3 m/s (`vbd.<cloth>.pop12`), as v2.11 for rope links.
   - **Two-way coupling** (the cloth pushing back on the robot, user 2026-10-12): the coupling's work is booked apart
     from both balances, on the robot inside the MuJoCo side's xfrc work and on the cloth as W_robot. Its own balance
     C = W_couple + W_react is what the lagged exchange itself creates. The cloth and the coupling together, X + C, may
     not stay more than 0.25 J above their running minimum for 4 consecutive cloth steps (10 ms). C alone is not judged:
     at a contact's onset the cloth books the end-of-step force times the step's displacement while the robot pays the
     reaction one step later, a one-step spike the next steps repay (blanket_flatten very_easy BLF2a cem_e1 on
     blanket_vbd_gpu: C +0.43 J in one step at tick 1024, ending at −16.6 J, the cloth's X 3 mJ; two-way lane's per-step
     counters). Books recorded before this window (C_max_rise only) pass when X_max_rise + C_max_rise ≤ 0.25 J and
     otherwise fail as unresolved, to be re-run.
5. **The robot paid for it (v2.4).** For a rigid task body, in that tick, all of these hold:
   - the static partners and the other task bodies did ≤ 0 work on it;
   - its kinetic-energy gain is at most the work its robot partners did on it (contacts, plus the weld of a grasp);
   - every robot contact that did positive work is passive itself: its pair net work over the contact episode is at
     most 0.01 J. Otherwise the "robot work" may have come from a contact that creates energy.

   The case is a wedge ejection: an arm presses a body against a fixture and squeezes it out faster than twice the
   arm's own speed, and pays for it. F5 still judges the energy.
6. **Redistribution within a chain (v2.5, extended in v2.6).** For a link of an articulated entity (a cable, a fibre
   bundle, a rope), in that tick, all of these hold:
   - the link itself took no positive work from outside its entity (its speed came through the chain's joints), and
     no active equality ties the link itself to another body;
   - the entity's kinetic energy rose by at most the work done on it from outside, plus gravity's, plus the drop of its
     exact stored potentials, within 1 mJ. Outside work: contacts with other entities, the robot and statics, and
     (v2.6) the equalities that tie the entity to another body, a grasp weld or a connect, booked on the entity's own
     dofs. Stored potentials (v2.6): its joint springs and a rope's rod (rope_energy). A soft constraint's estimated
     stored energy (S) never counts, so a soft constraint's own creation can never pay for itself;
   - every outside contact that did positive work is passive (pair net over its contact episode ≤ 0.01 J), and so
     is every tie that did positive work (its own net over its active episode ≤ 0.01 J): a weld or coupling that
     creates energy keeps the jump uncaused.

   The case is a pinned-rod whip: a flung bundle is stopped by a nearly still robot link, and its strands, held stiff
   by their dry friction, pivot about their roots. The tips get 10-16 % faster while the bundle's kinetic energy
   drops. It is reported as a whip (a VIOLENT trigger when the link ends above 3 m/s, v2.8). The v2.6 case: a rope held by both hands loses one weld at ~8 N, and the rope's
   rod and the other hand's weld send the freed end off at 1.1 m/s while the rope's kinetic energy rises by less than
   the bend energy it releases.

   The v2.6 parts (|g|·Δt_tick, the ties, the stored potentials) are checked after every v2.5 rule. A report can
   therefore still read v2.5 exactly: the events `jump_causal_gravity`, `jump_causal_gravity_pop` and
   `whip_redistribution_v26` are the jumps that only v2.6 excuses.

   **Over a 3-tick window (v2.9).** When the tick's own balance fails, the same balance over the last 3 ticks (7.5 ms)
   is checked, with each tick's link condition holding in all three: an impulse that crosses a tick boundary inside
   the chain. The case is knot_untying's falling rope (knot_chain_v4, round KN1): at tick 5275 the rope's energy
   wobbles by +1.85 mJ in one tick and −2.95 mJ in the next, and its end link goes 8.9 → 11.6 m/s; over the 3 ticks
   the kinetic gain (11.29 mJ) is within gravity's work, the outside work and the stored potentials' drop (11.51 mJ
   allowed). The event is `whip_redistribution_v29`; v2.8 reads it as an uncaused jump.

   **Contact round-off (v2.10).** The link clause ("no positive work from outside its entity") tolerates 1 µJ
   (LINK_OUT_EPS_J). A rope link lying on the table takes ~1e-8 J from it, contact-solver round-off, while the chain
   kicks it. The case is rope_threading hard kr2_1 top075_b3_e169 and top092_b0_e250 (rope_chain_v2, and v1): rope::070
   0.02 → 0.84 m/s and rope::152 0.06 → 2.1 m/s with the table's +4.1e-9 J and +2.0e-8 J; the rope's kinetic energy fell
   1.5 mJ within its outside work, gravity and stored drop, and the rope agent's per-step books close to 1e-5 mJ. The
   event is `whip_redistribution_v210`, checked after every earlier form; earlier readings count it as an uncaused jump.

**F5, energy created.**

- **Global, rigorous:** X > 0.1 J at any tick.
- **Local:** creation after earlier dissipation. All three signals must exceed 0.25 J:
  - X rises within 0.1 s: the exact books show a gain;
  - X + S rises above its running minimum: the gain is not stored energy being released;
  - one contact pair's net work over its contact episode is positive, or (v2.9) one equality's or one limited
    joint's or tendon's net work over its active episode: there is a culprit. The equality and limit rows are booked
    like the contact pairs (f·(J·v)·dt per row, a new episode after 25 ms inactive), so a gain that a gripper's
    closed loop (a four-bar's connects) or a limit produced is named too.

  Each signal alone has a known false-positive mode, so none of them decides alone:
  - the X rise fires on releases (a weld squeezed by a slam lets go);
  - S is inflated by the damping force at impacts and flickers in static squeezes;
  - a contact inside a constraint loop (a weld, a gripper's linkage, two fingers) can circulate work.

  Why S stays −½ f·pos (evaluated for v2.9). At an impact a loaded soft limit's force chatters from tick to tick
  (buzz G12 on b1ec43997: Panda joint6 20-25 mrad past its limit under the motor, its force 25-168 N·m-equivalent, so
  −½ f·pos jumps 0.25 → 2.10 J in 3 ticks while the limit's own work is negative), and X + S fires without a gain.
  Four other estimates were tried on G12, where 0f8654a49 is a real breakdown (the coupler on wire_base, +1.90 J)
  and b1ec43997 is not:

  | S | X + S rise on 0f8654a49 (must fire) | on b1ec43997 (should not) |
  |---|---|---|
  | −½ f·pos (in force) | 0.84 J | 1.04 J |
  | the springs only, ½·K·imp·r²/(A+R) | 0.66 J | 0.68 J: it undercounts a motor-loaded limit, so the later release shows |
  | −½ f·r with the damping term taken out of f | 0.87 J | 1.10 J |
  | −½ f·pos capped by each group's absorbed energy | 0.84 J | 1.04 J: the limit had absorbed plenty |
  | windowed: S's 20 ms minimum now, its maximum for the running floor | **0.02 J: the real breakdown is lost** | 0.03 J |

  At an impact a chattering S and a real creation look alike to X + S, so any filter that removes the false fire also
  hides creations that coincide with an impact. The false fire is harmless as long as the culprit signal stays
  independent: on b1ec43997 no pair, equality or limit gained, and G12 reads VIOLENT. The springs-only and the
  windowed X + S are recorded as forensic readings (energy_created_spring_j, energy_created_win_j; `--reading v2.9s`
  / `v2.9w`), never as a verdict.

### Violent but causal: reported, not failed

These are forensic triggers. They are listed with their causal partner and are not failures when F1-F5 hold:

- a task link above 3 m/s;
- a task link spinning above 60 rad/s while its surface moves faster than 3 m/s (ω·r_link > 3 m/s; v2.7). For a link
  of 5 cm or more this is the same as 60 rad/s alone. A 6 mm rope link twisting at 112 rad/s moves its surface at
  0.7 m/s and is not flagged;
- a contact force above 2000 N;
- a task link leaving the workspace (table + 10 cm, up to 1 m above it);
- a causal whip, or a chain's redistribution (rule 6), that leaves its link above 3 m/s (v2.8). Slower ones are
  listed ("whip <= 3 m/s"), as pops with a kick at or below 3 m/s are: a cloth corner snapping back at 1.2 m/s on a grasp's
  release is not flagged, a fibre strand whipped to 4.5-34 m/s is.

### Legitimate motion, determinism, speed

- **References** succeed on the tested build. Their envelope is CLEAN, and two fresh processes end on the same state
  hash. One reference execution, with the observer, takes at most **180 min**: it finishes within the 3 h agent budget (user 2026-10-13; 120 min under the 2 h budget from 2026-10-12, 72 min, 20 % of the former 6 h budget, before).
- **nop** (400 hold controls) does not succeed and stays CLEAN.
- **Reproducers** of mechanisms fixed earlier replay without BREAKDOWN, deterministically.

### Acceptance of an instance

An instance PASSES when all of these hold on the same commit:

- the corpus has no BREAKDOWN;
- references, nop and reproducers are as above;
- a fresh self-play search of a declared size, with recorded seeds, finds no BREAKDOWN after fresh-process
  confirmation.

**The size of an acceptance round (decided 2026-10-09):** about **1,000 episodes per tier**, so about 5k
for a 5-tier family and 2k for a 2-tier one.
- All six strategies share the episodes equally: `round.py run --episodes 1000` gives each about 167 per instance.
- Every worker gets a fresh seed, and the seeds are contiguous. An extension is a new round from the next seed
  (`--seed0` = the old seed0 + its number of workers; the report prints it), so the seed range simply continues.
- `--budget B` (B controls per worker) remains for development runs.

### The search (search protocol 3, from 2026-10-10)

`search.py` runs one strategy per worker. Every episode is deterministic per seed, and a smaller size (episodes or
budget) gives a prefix of the same episodes.

| strategy | episodes |
|---|---|
| random | piecewise-constant random joint and gripper targets, and oscillations, from the frozen birth |
| directed | IK moves on every arm (pull, twist, shake, ram parts and fixtures, slam towards the table, throw, regrasp; the family's mode list comes from the corpus manifest) |
| mutate | the instance's corpus streams and incidents, mutated: compression, amplification, reversal, noise, splicing |
| cem | the cross-entropy method over 5-waypoint target sequences, maximising the breakdown score |
| **prefix** | **mid-task states:** the instance's reference, replayed from birth up to a random row k, then a perturbation from that exact state (random, directed, or a mutated stretch of the reference's continuation) |
| **periodic** | **sustained periodic inputs:** 1-3 joints or grippers (a gripper half the time), each a sinusoid of its own frequency (0.5, 1, 2, 5, 10 Hz, or the 20 Hz Nyquist square wave at 40 Hz control), amplitude (5-50 % of the half range) and phase, for 4-20 s |
| emptyclose | targeted (claws W4 + rule A, round CL8t): grippers commanded shut on nothing while the arm drives the hand into its own arm's base, the table or the other hand, then held there pressing, sometimes with the gripper chattering (BL1 e30's pattern) |

Two options reach every worker of a round from its environment, and the search summary records them:
- `STRESS_CEM_SEEDED=1`: each CEM restart starts from a corpus or incident stream (5 of its rows as the waypoints),
  so CEM searches around known near-misses.
- `STRESS_OBJECTIVE=fingers`: the search score also rewards finger-finger force (/ 2 kN), depth (/ 10 mm) and a small
  A_ii on a finger-finger row. It steers CEM and which VIOLENT episodes are kept; it is never a verdict.

Besides BREAKDOWN episodes and the 5 highest-scoring VIOLENT ones, a worker keeps every episode with a finger-finger
force above 500 N, a finger row's A_ii below 0.1 or a degenerate-row signature. The round confirms in fresh processes
those above 2 kN finger-finger, with a finger-finger or finger-joint row below A_ii 0.1, or with the signature.

**How prefix works.**
- It explores the states agents actually reach mid-task: mid-traversal in buzz wire, mid-extraction in the puzzles,
  a held piece, a cable in the cradle.
- Each pass replays the reference once. At 2-5 random cut points it forks the process, and the child runs the
  perturbation from the exact state.
- The budget is charged with the controls actually simulated: the pass's prefix once, plus every perturbation.
- An episode's saved stream is `reference[:k] + perturbation`, so `replay.py` reproduces it from birth. (Checked: a
  fork-run episode and its fresh-process replay agree field for field.)
- Without a reference the worker falls back to random episodes from birth, labelled `prefix-nr`.
- **More workers for a slow strategy.** Long references make each pass expensive: on a 10k-row route a prefix episode
  costs about 2k controls of reference, against 150-500 for the others. A round can name a strategy several times,
  and each occurrence gets its own worker and seed.
  - With `--episodes`, the strategy's equal share is split over its workers, so they finish with the others:
    `--strategies random,directed,mutate,cem,periodic,prefix,prefix,prefix,prefix` runs prefix's ~167 episodes as
    4 workers of ~42.
  - With `--budget`, each occurrence gets its own budget, i.e. a larger share of the controls.
  - The report sums them in one column.

**Why periodic (protocol 3).** Some creations are slow and steady, and need a sustained periodic input to show.
- E1, claws_v4: the ALOHA's finger-coupling joint equality (MJCF default solref 0.02/1), pumped by a 20 Hz half-stroke
  gripper oscillation, creates ~0.03 J/s. The global books cross 0.1 J after about 4 s.
- Random and CEM search on CPU did not find it in ~1.5M controls: their inputs change too often, or settle too soon.
  The coordinator's GPU batch search found it.
- The periodic strategy drives exactly such inputs, at several frequencies and amplitudes per joint.
- **Checked on CPU** (claws_v4 easier, 0cc192ac2, seeds 97001-97004, periodic only, 4 × 25k controls): the strategy
  found E1 again in 1 of its 194 episodes. This is E1p in `corpus/reproducers/`: the left gripper alone as a 20 Hz
  square wave (0.0185 / 0.0290) for 11.9 s, global X 0.102 J at 11.9 s; two fresh processes give the same verdict
  and hash.
- The same input rebuilt on each tier's birth targets breaks down on medium too (E1m, constructed: X 0.105 J). On
  very_easy and easy it creates ~0.16 J but X peaks at 0.03 J; on hard it creates 0.01 J.

**Protocol changes.** Protocols 2 and 3 change the search only, not the criteria.
- Protocol 2 added prefix, protocol 3 added periodic.
- The rounds before 2026-10-10 (A1, A2_400k, A3) ran protocol 1, the first four strategies. A3', A3'' and A4_500k
  ran protocol 2. Every acceptance round from claws_v5 and buzz wire on runs protocol 3.
- `round.py --strategies random,directed,mutate,cem[,prefix]` reproduces protocols 1 and 2. Every report has a column
  per strategy.

## Version history

| Version | Date | Change | Why |
|---|---|---|---|
| v1 | 2026-10-09 | F1-F5 and the kinematic causal rule `|v_after| ≤ |v_before| + 2·v_partner + 0.1` | the coordinator's rule: a spike fails only without a causal partner or with energy beyond the robot's work |
| v1.1 | 2026-10-09 | rotational energy and the 50 ms rebound peak in the bound | a rebound at 5 % of the incoming energy was flagged |
| v2 | 2026-10-09 | the pop clause; pops above 3 m/s fail | squeeze and weld releases (0.74-1.10 m/s; kinetic-energy gain ≤ the stored energy released) are the robot's own work |
| v2.1 | 2026-10-09 | chains | a finger at 1.5 m/s hits claw A, which pushes claw B in the same step; B's direct partner read 0 m/s |
| v2.2 | 2026-10-09 | F5-local: X rise within 0.1 s and pair net (instead of X + S and pair net) | S was inflated (24.8 J at a 5.6 m/s impact whose claw carried 1.4 J) and flickered |
| v2.3 | 2026-10-09 | F5-local needs all three signals | v2.2 flagged a legitimate release: a slammed weld let go 0.72 J while X rose 0.38 J |
| v2.4 | 2026-10-10 | a robot-powered jump is causal (rule 5 above) | hedgehog H1: link5 at 0.80 m/s wedges the ball out of the cage at 4.03 m/s; link5 did +1.02 J, the cage −0.51 J, the gain was +0.56 J, X rose 0.04 J |
| v2.5 | 2026-10-10 | a chain's redistribution is causal (rule 6 above) | fiber (7dac7abde) easier and easy cem_e12: a flung bundle (sheath 3.5 m/s) is stopped by a still robot link; strand tips go 4.20 → 4.86 m/s while the bundle's kinetic energy drops 0.98 J; X 0.019 J |
| v2.6 | 2026-10-10 | rule 6 for tied chains (the ties' work booked, with passivity) and with the drop of exact stored potentials; |g|·Δt_tick in rules 1-2 | rope_threading easy (rope_chain_v1) rt_pull10: the right weld is lost at 8.1 N while the left holds; rope links 142-144 go 0.03 → 1.1-1.2 m/s while the rope's kinetic energy rises 7 mJ and its rod releases 22 mJ; X 0.000 J |
| v2.7 | 2026-10-10 | the angular trigger (VIOLENT only) counts a spin above 60 rad/s only on a link whose surface moves faster than 3 m/s (ω·r_link) | knot_untying (rope chain, 6 mm links): the references very_easy, medium and hard succeed with X 0.000 J but read VIOLENT on rope links twisting at 80-112 rad/s, whose surfaces move at 0.50-0.70 m/s |
| v2.8 | 2026-10-10 | a causal whip or a chain's redistribution (rule 6) is a VIOLENT trigger only when its link ends above 3 m/s (the speed trigger's threshold; slower ones are listed, as pops are) | blanket_flatten very_easy (settled birth, ve_short2): at a grasp's release about 11 corner vertices go from rest to 0.6-1.2 m/s in one tick, paid by the cloth's own stored elastic energy (+3.1 mJ); the reference succeeds with X 0.000 J but read VIOLENT (whip x6) |
| v2.9 | 2026-10-10 | (a) F5-local's culprit is also an equality's or a limit's net work; (c) rule 6 also balances over a 3-tick window; (d) a pop fails on its kick \|v_after − v_before\| above 3 m/s, not on its after-speed; (e) no peaks or events after MuJoCo's auto-reset (mjWARN_BADQACC) | (c) knot_untying KN1 prefix11 e1 and prefix16 e0: a falling rope's energy wobbles within one tick, and its end link's jump balances over 3 ticks; (d) rope_threading easy b1_e15 (GPU search): 0.3-0.6 mJ releases nudge links of a rope whipped at 7 m/s past 3 m/s with kicks of at most 2.04 m/s; (a) a gain that a gripper's closed loop or a limit makes had no culprit; (e) K3's 35.6 mm robot depth was MuJoCo's reset to qpos0, not physics |
| **v2.10** | 2026-10-11 | rule 6's link clause tolerates 1 µJ of positive outside work (contact round-off); the 3 m/s pop guard covers energy stored in contacts, equalities and limits only, so a fast pop the rod's own elastic energy pays for is no breakdown (VIOLENT through the speed trigger) | rope_threading hard kr2_1 top075 / top092: the table did +4.1e-9 / +2.0e-8 J on a resting rope link the chain kicked, every other clause held; rope_threading medium kr3_4 e52: rope::193's 1.687 mJ was rod-joint energy (contacts 0), released toward the free end with a kick of 3.22 m/s |
| v2.11 | 2026-10-11 | a fast pop the rod does not pay for is a breakdown only if its constraint-stored release alone gives v_c = sqrt(2 E_c / m_link) > 3 m/s | rope_threading medium KN4-r2 e144 (ccd4ba7fa): a finger releases the pinched tail; rope::198's 3.23 m/s kick took 0.98 mJ from its rod and 0.29 mJ from contact rows, v_c 1.80 m/s |
| v2.12 | 2026-10-12 | VBD cloth: its own balance X (storage included) may not rise > 0.25 J apart from the system sum, outside a guarded press (a pressed pile's rise is a WARNING when calm, VIOLENT when thrown, ≤ 1 % of a tier's episodes); a particle pop by its energy share; with two-way coupling, X + C may not stay > 0.25 J above its running minimum for 10 ms (4 cloth steps) | blanket_flatten easier4 BLF3b cem_e3 (blanket_vbd_gpu, 694e790b5): +2.34 J in one cloth step after a pin release, CLEAN under v2.11 (the system's X was 17 J below its start) |
| **v2.13b** | 2026-10-13 | integrator-side books (contact work J(q0)' f . v_mean, armature energy in link KE) and energy_created(integrator): the excess and the step's own residual both rise 0.25 J in one local event with no other energy rule failing | orbits easier prefix14_e1 (c871): a passive finger x cord contact booked +22.7 mJ over a 19,000-tick hold (integrator: +1.1 mJ), BREAKDOWN by rule 6 under v2.12; S4 (armature 1e-6) keeps its real 0.30 J step-created breakdown through the new class |

- **Regression set.** Every version must keep these reproducers BREAKDOWN on the build where each was found:
  M1c easy and medium, V1, M3, M3c easier and hard, M4c, M5, F1, F2, F3, F3b, F4, F5, F5x, plus M6 since 593ae66ac,
  E1, E1p and E1m since 0cc192ac2, B1 and B1b since 15579a3b7, B2, G1-G3 and G46 since 19bee571e, B3, G5-G7
  and G10 since b1d1706a4, G8 and G9 since bbdbae3ad, G11 since a0b4dddef, G12 since 0f8654a49, and C1 since
  b1ec43997 (an ALOHA finger driven 11.3 mm through a claws piece on the exact diagonal; claws v6, round CL6), K0 and
  K1 since 93b176b09 (knot_chain_v2), K2 since 78e0c03a9 (knot_chain_v3) and K3 since c4ebbcb3d (knot_chain_v4): the
  explicit rope rod's zig-zag blow-ups and, on v4, a middle-link blow-up under fast arm motion (mjWARN_BADQACC). G10 (a Robotiq
  driver on the wire) breaks down through bbdbae3ad; on a0b4dddef its stream takes another path and the mechanism
  shows only path-matched; it is VIOLENT from 0f8654a49 on. G11 (a frictionless spring link on the wire, +1.54 J)
  breaks down on a0b4dddef and 0f8654a49, and G12 (a coupler on the static wire_base, +1.90 J) on 0f8654a49; both are
  VIOLENT from b1ec43997 (the exact constraint diagonal) on.
  They are in `corpus/reproducers/`, each with its mechanism, build and sha256.
- **Checks that must not read BREAKDOWN (v2.9):** rope_threading easy b1_e15, b1_e132, b1_e22 and b1_e52 (GPU search
  on knot_chain_v4's rope: pops that kick links of a whipped rope by at most 2.04 m/s; VIOLENT on the whip), knot
  KN1 prefix11 e1 and prefix16 e0 (VIOLENT), and the rope agent's rt_pull12 (CLEAN).
- **Checks that must not read BREAKDOWN (v2.10):** rope_threading hard kr2_1 top075_b3_e169 (c030746442c56e46) and
  top092_b0_e250 (7cd06af7bc5999a8), and rope_threading medium kr3_4 top035_b2_e52 (3bf2366b25f3d3f0), on
  rope_chain_v2 and v3 (VIOLENT).
- **Checks that must not read BREAKDOWN (v2.11):** rope_threading medium KN4-r2 kt1_4 top035_b1_e144 on ccd4ba7fa
  (79365b669b0d3c56, VIOLENT).
- **Checks that must not read BREAKDOWN (v2.12):** blanket_flatten very_easy's GPU reference (0a8b914f39cc8988, x2,
  CLEAN) and BLF2c random_e9 (b2a4b2520e901cee, CLEAN) on blanket_vbd_gpu 694e790b5.
- **Must read BREAKDOWN (v2.12):** on blanket_vbd_gpu 694e790b5, BLF3b cem_e3 (bac12fb3e78cf29e, cloth X +2.34 J),
  BLF4a very_easy4 periodic_e2 (1d8cea585bd0bd02, +10.29 J) and random_e8 (f5c0396beb93990d, +5.27 J). With the cloth
  floor's finger links in the stack (2f5ee7f3eb) all three read CLEAN (X ≤ 2.2 mJ).
- **v2.4's regression run:** 109 runs with the work clause off and on. They were the regression set, M6, M6full, B1,
  B1b, H1, and all 89 historical runs with an uncaused jump.
  - Every reproducer stays BREAKDOWN with its jump counts unchanged.
  - M3 and M3c are not excused: their gripper-base pairs give +1.06 J and +2.00 J.
  - Only one historical flag drops: S1 claws very_easy random_e50, a passive wrist push (pair −0.20 J).
  - The clause as first proposed, without the passivity condition, was rejected. It excused S1 cem_e155, whose
    gripper-base contact itself created +0.31 J, below F5's threshold.
- **v2.5's regression run** (`wr_eval.py` in the validator's space): 136 runs, each off, with fiber's proposed form and
  with the adopted energy form, on its own build and options. They were the regression set, M6, M6full, B1, B1b, B2,
  H1, E1, E1p, E1m, all 89 historical runs with an uncaused jump, the two fiber candidates, and fiber's 21 corpus
  streams on 7dac7abde. State hashes are identical in all three settings (the observer stays read-only).
  - Every reproducer stays BREAKDOWN. F5 keeps all 28 uncaused jumps and F4 all 7: in their ticks the bundle gained
    kinetic energy that no outside work paid for (F5 +1.62 J against +0.28 J; F4 +0.031 J against gravity's +0.019 J).
  - F3 keeps its BREAKDOWN (130 mm deep) and one uncaused jump; 39 of its 40 are excused. Their links were yanked out
    of the penetration through the chain, paid for by contacts that absorbed more than they returned (pair nets
    down to −1.73 J).
  - Fiber's corpus is unchanged: 0 BREAKDOWN.
  - Historical flags that drop (all fibre whips, X ≤ 0.019 J): the two candidates, very_easy random_e76 (b1 and b2,
    6 jumps), medium cem_e240 (b1, 118) and easier mix_e7 (b3, 81). In every excused tick the bundle's kinetic energy
    fell, by 0.009-0.98 J, and its tips went at most 1.23 times faster. The balance had a margin of at least
    0.0086 J beyond the 1 mJ tolerance.
- **v2.6's regression run** (`v26_eval.py` in the validator's space): 145 runs under v2.5 and v2.6, on each run's own
  build and options: v2.5's 136 and the buzz mechanisms since (B3, G1-G3, G5-G7, G46 and B2 on two builds). State
  hashes are identical. Every reproducer stays BREAKDOWN with its jump counts unchanged (F4 7, F5 28, F3 1); fiber's
  21 corpus streams are unchanged; no historical verdict changes.
  - The package's v2.6 and its v2.5 reading reproduce separate v2.6 and v2.5 runs on FIB easier and easy, F4, F5,
    mix_e7, rt_pull10, rt_pull12, rt_twist and rt_whip, case for case.
  - The rope agent's 16 release and weld-loss streams (rope_threading easy and knot very_easy, both rope chains):

    | stream | v2.5 | v2.6 |
    |---|---|---|
    | rt_pull10 (a weld lost, the other holds) | BREAKDOWN, 6 jumps | VIOLENT: 6 redistribution |
    | rt_twist (both welds opened after a twist) | BREAKDOWN, 5 jumps | VIOLENT: 5 redistribution |
    | rt_pull12 (a weld lost, the other holds) | BREAKDOWN, 3 jumps | BREAKDOWN, 3 jumps |
    | rt_whip | VIOLENT (61 redistribution) | VIOLENT (the same) |
    | the other 12 (rt_pull10f, rt_pull15/30/45, knot welds and pinch) | CLEAN or VIOLENT | unchanged |

    rt_pull12 stays flagged on purpose: its rope gained 8.5 mJ against −5.2 mJ of outside work and an 11.6 mJ rod
    drop. The missing ~1 mJ was stored in the rope's stretched node connects, and soft-constraint stored energy
    is not counted. v2.9 reads it CLEAN, as a balanced release: over the 3-tick window the rope's kinetic gain
    (8.5 mJ) is within its rod's 8.9 mJ stored drop, gravity's −0.6 mJ and the outside work's +1.0 mJ.
- **v2.7's regression run** (`tip_eval.py` and `tip_bd.py` in the validator's space), fresh replays with the v2.7
  observer:
  - v2.6's regression set on each run's own build and options (142 of its 145 runs; the other three need a
    harness-only option), G8 and G9 on bbdbae3ad, and knot medium's two open prefix BREAKDOWNs: 146 runs, 75 of them
    BREAKDOWN. All 75 stay BREAKDOWN. The package's v2.6 reading equals the v2.6 runs case for case, final state
    hashes included (142 of 142).
  - The fixed checks of the latest rounds: claws v5 (63 runs), buzz bbdbae3ad (59), hedgehog 27a3bbf4d (9), knot (13)
    and fiber 7dac7abde (27). Every final hash compared equals the round's own (130 of 130).
  - The angular trigger feeds VIOLENT only, so no BREAKDOWN can move; the run confirms it.
  - Rigid families cannot change: their links measure 5.5-27 cm (buzz wand 5.5 cm, hedgehog ball 6.7 cm, claws
    11.5-26.9 cm), and there a spin above 60 rad/s always moves the surface faster than 3 m/s.
  - What changes: knot's references very_easy, medium and hard and its corpus stream afb86e8b, VIOLENT → CLEAN.
    Fiber's verdicts do not change. In 9 runs a strand link spun at 86-523 rad/s with its surface under 3 m/s, and
    each of them stays flagged by its other triggers (strands at 4.5-29 m/s, whips, leaving the workspace) or stays
    BREAKDOWN (F1, F2, F3b).
- **Not adopted: the surface speed alone** (ω·r_link > 3 m/s as the trigger). It flags claws pieces turning at
  14-51 rad/s, whose surfaces move at 3.3-13 m/s: 15 rigid runs go CLEAN → VIOLENT.
- **v2.8's regression run** (`v28_eval.py` and `v28_run.py` in the validator's space), fresh replays with the v2.8
  observer, each read under v2.7 and v2.8:
  - The set: v2.7's 146 BREAKDOWN-set runs on their own builds, plus G10 on bbdbae3ad. Also the fixed checks of claws
    v5, buzz bbdbae3ad, hedgehog, knot and fiber 7dac7abde, the rope agent's 16 release and weld-loss streams, and
    blanket's settled references easy_short1 and ve_short2. 336 runs in all. Final hashes equal v2.7's run (317 of 317
    compared).
  - All 77 BREAKDOWN stay BREAKDOWN, among them B1-B3, G1-G10, E1, the M and F series, knot medium's prefix_e0/e1 and
    rope's rt_pull12. No rigid verdict changes (claws 151, buzz 74, hedgehog 10 runs).
  - Every whip that ends above 3 m/s stays a trigger: fibre in 28 runs (3.0-55.8 m/s; FIB cem_e12 11-15 m/s,
    random_e76 29 m/s, mix_e7 35 m/s), knot in 1 (12.8 m/s).
  - Nine streams change, all VIOLENT → CLEAN, each with every whip at or below 3 m/s and no other trigger:

    | stream | family | peak link speed | peak whip end speed | whips |
    |---|---|---|---|---|
    | ve_short2 (settled very_easy reference) | blanket | 1.31 m/s | 0.91 m/s | 6 |
    | fiber corpus easier 4aa288cdbb5a | fiber | 2.62 m/s | 1.39 m/s | 3 |
    | fiber corpus easier cf6e8077f19e | fiber | 1.07 m/s | 0.54 m/s | 3 |
    | knot corpus 6454eec02de7945b | knot | 1.83 m/s | 1.33 m/s | 3 |
    | knw_twist (knot weld stream) | knot | 1.43 m/s | 1.43 m/s | 3 |
    | rt_pull10 | rope | 1.55 m/s | 1.22 m/s | 110 |
    | rt_pull10f | rope | 1.50 m/s | 1.50 m/s | 108 |
    | rt_twist | rope | 1.04 m/s | 0.63 m/s | 34 |
    | rt_whip | rope | 2.47 m/s | 1.82 m/s | 72 |
- **v2.9's regression run** (`v29_eval.py` and `v29_run.py` in the validator's space), fresh replays with the v2.9
  observer, each read under v2.8 and v2.9:
  - The set: v2.8's 336 runs, G11 on a0b4dddef, 0f8654a49 and b1ec43997, G12 on the last two, the K series on the runtime where
    each was found (K0 and K1 on knot_chain_v2, K2 on v3, K3 on v4), KN1's prefix11 e1 and prefix16 e0, C1 on
    b1ec43997 and the four rope pops: 352 runs.
  - 91 runs are BREAKDOWN under v2.8, and 84 stay BREAKDOWN, among them A1, C1, G8-G12, K0-K3, E1 / E1m / E1p, the
    F and M series, ROLL, S1 and S2. No rigid verdict changes (claws 152, buzz 79, hedgehog 10 runs).
  - (e): only K0-K3 have an auto-reset (ticks 2570, 7146, 3636, 1337). All four stay BREAKDOWN (the fault, the
    warning, uncaused jumps before it).
  - Seven change, each explained by its books:

    | stream | family | part | v2.8 | v2.9 | numbers |
    |---|---|---|---|---|---|
    | b1_e15 (rope_threading easy, GPU search) | rope | (d) | BREAKDOWN: 13 pops above 3 m/s (3.7 m/s) | VIOLENT (7.4 m/s, whips) | largest kick 2.04 m/s; each pop releases 0.3-0.65 mJ, its kinetic gain at most that |
    | b1_e132 | rope | (d) | BREAKDOWN: 8 pops (4.2 m/s) | VIOLENT (7.0 m/s) | largest kick 1.42 m/s |
    | b1_e22 | rope | (d) | BREAKDOWN: 1 pop (3.3 m/s) | VIOLENT (9.5 m/s) | kick 0.61 m/s |
    | b1_e52 | rope | (d) | BREAKDOWN: 4 pops (3.4 m/s) | VIOLENT (6.0 m/s) | largest kick 1.11 m/s |
    | KN1 prefix11 e1 (knot easier) | knot | (c) | BREAKDOWN: 1 uncaused jump | VIOLENT (11.6 m/s) | tick 5275, rope::000 8.9 → 11.6 m/s; window gain 11.29 mJ, allowed 11.51 mJ (gravity 10.79, outside −0.09, stored +0.19, 1 mJ tolerance) |
    | KN1 prefix16 e0 (knot very_easy) | knot | (c) | BREAKDOWN: 1 uncaused jump | VIOLENT (12.6 m/s) | tick 3449, rope::000 7.6 → 12.5 m/s; window gain 1.51 mJ, allowed 2.30 mJ (gravity 3.01, outside −0.56, stored +1.15, 1 mJ tolerance) |
    | rt_pull12 (rope agent, rope_chain_v1) | rope | (c) | BREAKDOWN: 3 uncaused jumps | CLEAN | tick 186, rope::171-173 0.04 → 1.5 m/s; window gain 8.5 mJ against the rod's 8.9 mJ stored drop, gravity −0.6 mJ, outside +1.0 mJ |

    The four rope hashes equal the GPU search's own CPU replays on another host (06b5d32f, 35b0a6d5, c27e0435,
    33039afd), and KN1's equal the round's (a41e6440, 96c4fcc4).
- **v2.10's regression run** (`v210_eval.py`, `v210_run.py`, `v210_sub.py`, `v210_pops.py` in the validator's space),
  fresh replays with the v2.10 observer, each read under v2.9 and v2.10:
  - The set: v2.9's 352 runs where their runtimes survive, the 10-10 / 10-11 knot, rope and fiber cases (knot_chain_v5
    and v6, rope_chain_v1 and v2, fiber FB2 p2e7 and FB3 p15e6), K0-K3, knot's corpus and the 16 rope load streams on
    knot_chain_v4 and v6, and corpus/pops: 411 runs. The 34 runs whose runtime trees are gone (knot self-check, the
    gknot set, K0-K2 on their own chains, the load streams on obsdev) were bounded offline from their stored v2.9
    records: none has an uncaused jump or a fast pop that v2.10 could excuse without a fault.
  - Final-state hashes equal the v2.9 runs' everywhere (0 differences). 92 runs are BREAKDOWN under v2.9; 86 stay
    BREAKDOWN, with their uncaused-jump counts unchanged. The 5 corpus/pops stay non-BREAKDOWN.
  - Six change, all BREAKDOWN → VIOLENT, all in the two classes:

    | stream | runtime | part | v2.9 | v2.10 | numbers |
    |---|---|---|---|---|---|
    | kr2_1 hard top075_b3_e169 | rope_chain_v2 (kn11) | round-off | 1 uncaused jump | VIOLENT (5.1 m/s, whips) | rope::070 0.02 → 0.84 m/s, table +4.1e-9 J; rope ΔKE −1.505 mJ vs outside −0.785, gravity +0.833, ΔU +0.994 mJ |
    | kr2_1 hard top092_b0_e250 | rope_chain_v2 (kn11) | round-off | 1 uncaused jump | VIOLENT (5.1 m/s) | rope::152 0.06 → 2.1 m/s, table +2.0e-8 J; rope ΔKE −1.536 mJ vs outside −0.385, gravity −2.033, ΔU −1.976 mJ |
    | kr2_1 hard top075_b3_e169 | rope_chain_v1 (kn9) | round-off | 1 uncaused jump | VIOLENT | the same stream and hash on the earlier rope |
    | kr2_1 hard top092_b0_e250 | rope_chain_v1 (kn9) | round-off | 1 uncaused jump | VIOLENT | the same stream and hash on the earlier rope |
    | kr3_4 medium top035_b2_e52 | rope_chain_v2 (kn11) | rod release | 1 pop kick 3.22 m/s | VIOLENT (5.2 m/s) | rope::193 0.22 → 3.36 m/s, KE +1.60 mJ, rod drop 1.68 mJ (= the whole stored drop) |
    | kr2_1 hard top008_b0_e198 | rope_chain_v1 (kn9) | rod release | 6 pops, kicks 3.13-3.93 m/s | VIOLENT (12.0 m/s) | tick 1135, rope::048-053 3.5-4.4 → 6.2-6.4 m/s, each KE gain (1.40-2.29 mJ) ≤ its rod drop (2.61-2.98 mJ) = its whole stored drop |

    On rope_chain_v2 and v3 the last stream reads VIOLENT under both readings.
- **v2.12's regression run** (blanket_flatten on blanket_vbd_gpu 694e790b5, the validator's `v212gpu_read2.py`): the
  109 candidates of the stopped CPU-profile rounds BLF2-4 (very_easy -i01, very_easy4, easier4) replayed on the GPU
  profile, plus the very_easy GPU reference x2, each read under v2.11 and v2.12: **111 runs**. 107 CLEAN / CLEAN,
  1 VIOLENT / VIOLENT, 2 CLEAN → BREAKDOWN (cem_e3, periodic_e2: the cloth's own guard), 1 BREAKDOWN / BREAKDOWN
  (random_e8). No particle pop fails (largest kick 2.79 m/s). The CPU profile's breakdowns (easier4 directed_e0, the
  eq:1 periodic streams) read CLEAN on the GPU profile.
- **v2.11's regression run** (`v211e_run.py`, `v211e_eval.py`, `popscan.py` in the validator's space), fresh replays with
  the v2.11 observer, each read under v2.10 and v2.11: v2.10's set, the kn12 streams (rope_chain_v3 / knot_chain_v7),
  the ccd4ba7fa blocker streams (rope e144, e85, e116, e8, e216, e138, e211, e28, e52 on the rope agent's tree kn13;
  fiber e131), K0-K3, knot's corpus and the 16 load streams on knot_chain_v4 and v6, and corpus/pops: **481 runs**.
  - Final-state hashes equal the v2.10 runs' everywhere. 90 runs are BREAKDOWN under v2.10; 89 stay. **One changes**:
    e144, BREAKDOWN → VIOLENT (v_c 1.80 m/s).
  - The only other constraint-paid fast pop in the set is e211's rope::199 (v_c 6.74 m/s; its after-speed was fast,
    its kick 2.92 m/s, so it was no breakdown under v2.10 either). Of the 14,324 stored results, 4 runs fail on fast
    pops alone under v2.10, all rope releases (e144, e52, kr2_1 top008 on rope_chain_v1, b1_e15); v2.10's rod scope
    already clears the other three.
- **Not adopted: fiber's form, u = the relative speed at the hit** (rule 3).
  - On the same 136 runs it also keeps every reproducer, and it drops the candidates and random_e76.
  - But it lets a cable's own speed justify its own jump: after any non-static hit the bound becomes 12u + 0.6
    (44 m/s for the candidates' 3.6 m/s), with no check of where the energy came from. A 0.5 g tip jumping from
    4 to 10 m/s gains only 0.02 J, below every energy guard, so for gram-scale links the jump rule is the only guard.
  - It also moves flags without cause (F4 7 → 6, F5 28 → 33), because the relative speed can be below the
    hitter's own.
- **Not adopted: static partners as causes for cable whips.**
  - With the whip gain 6 it was rejected. It removes all 28 uncaused jumps of F5 (a 34 m/s whip), and for gram-scale
    cable links the jump rule is the only guard.
  - With the literal `2u + 0.1` bound, nothing historical changes. It is held until a real candidate build shows such
    a flag.

## How to run

The tools need the RoboTangle image's Python, with mujoco 3.14.0, Genesis and the asset roots, because the frozen
instances are read through `rcb_topogym`.

- `_env.py` puts this checkout's `topogym` and `scripts/roboarc/shared` first on `sys.path`.
- It also sets the task containers' numpy / OpenBLAS SIMD masking, so AVX-512 and AVX2 hosts give the same state hash,
  and one thread per library.
- MuJoCo is single-threaded, so parallelise across processes: `--procs` search workers and `--replay-workers` replays.
- A search worker runs many episodes in one process (a median of 3-11 per worker, up to 167). Each starts from
  `Search.reset()`: the frozen birth plus MuJoCo's warning counters as the fresh process had them. The rewind keeps the
  counters, and after one fault MuJoCo's auto-reset left BADQACC at 1, so the world missed every later blow-up in that
  process. After a faulted episode the worker also builds a new Session.

```bash
S=tools/stress
# one stream (a reference, a corpus input, a candidate), with the observer:
python $S/replay.py --instance <instance.json> --actions <stream.npz | nop:400> --out out/one.json --hold 40 --trace out/one.trace.json.gz

# a self-check: does the harness run, and are your references clean? (2,000 controls per worker; never an acceptance)
python $S/round.py run --self-check --instances <dir>/<base>/instance.json ... \
    --refs <refs dir: <base>/privileged.npz> --corpus $S/corpus/manifest.json --reproducers $S/corpus/reproducers \
    --out out/selfcheck

# an acceptance round: ~1,000 episodes per instance over all six strategies; declare new seeds
python $S/round.py run --instances ... --refs ... --corpus $S/corpus/manifest.json --reproducers $S/corpus/reproducers \
    --seed0 50000 --episodes 1000 --procs 20 --replay-workers 8 --out out/round_A
# its extension, if one is decided: the same command from the next seed (the report prints it), into a new directory

python $S/round.py report out/round_A              # rebuild the report
python $S/verdict.py out/round_A/results/corpus    # per-stream tables
python $S/verdict.py out/round_A/results/confirm --reading v2.2    # an earlier reading, for comparison
```

**Running a family that is not in the corpus yet.**

- Add its streams to a manifest in the same format: `families.<task>.directed_modes` and `cases[]` with `base`,
  `input`, `incident` and `actions_sha256`.
- The search works without a corpus: random, directed and cem need none.

**When a run fails:**

1. Shrink it to a reproducer: `shrink.py --instance … --actions <cand.npz> --match "<fail substring>" --out repro.npz`.
2. Confirm the reproducer twice with `replay.py`: same verdict, same state hash.
3. Trace the mechanism, with the `--trace` per-tick energy books and the pair categories.
4. Add it to `corpus/reproducers/` with a `.json`: base, family, mechanism, the commit it was found on, sha256.

A fixed build must replay every reproducer without BREAKDOWN.

## How to read the verdict

`report.md` has one row per instance:

| column | what it says |
|---|---|
| corpus | streams replayed and how many broke down (any BREAKDOWN fails the instance) |
| reference | success, CLEAN, two fresh processes with the same hash, and the wall time (≤ 180 min, reported per tier) |
| nop | 400 hold controls: no success, CLEAN |
| reproducers | each earlier mechanism's verdict now; BREAKDOWN or nondeterminism fails |
| search | controls, episodes and grasp slips of the fixed-size search; a second table splits them by strategy |
| confirmed BREAKDOWN v2 / v2.2 / v2.3 / v2.4 / v2.5 / v2.6 / v2.7 / v2.8 / v2.9 / v2.10 / v2.11 / v2.12 | the search candidates replayed in fresh processes, under the eleven earlier readings and v2.12; **v2.12 decides** (v2.7 and v2.8 change only VIOLENT triggers, so their BREAKDOWN counts equal v2.6's; a result recorded before v2.9 reads as v2.8, one recorded before v2.10 as v2.9, one recorded before v2.11 as v2.10, one without a VBD cloth's v2.12 books as v2.11) |
| pops | stored-energy releases (count, max speed): legitimate while their kick is at most 3 m/s; their distribution is reported |
| P(miss) | the detection power below |

- **BREAKDOWN** means the simulator broke: shrink it, trace it, fix it.
- **VIOLENT** means violent but causal. Look at the trigger and its causal partner. It is not a failure, but it is
  worth knowing: for example, pieces thrown at 10-12 m/s by an uncapped arm.
- **CLEAN** means neither.

## Note: robot-static jams and the soft class (2026-10-11)

Three energy_created (local) breakdowns came from one physics class, not from the criteria: an arm driven into a static
fixture holding several coupled rows of the soft robot-static class (rope_threading hard kt1_3 e116, fiber easier
FB5b e131, knot easy_v8 e245). The class now has a constant impedance and rope_threading's rings keep the task's
contact against the robot (`docs/simulator/MUJOCO_CORE.md`, "The soft class's impedance"); with it the three read VIOLENT
or CLEAN, and of the four other flagged rope streams of that search (pops, a jump) three do; e144 keeps a 3.2 m/s
pop kick of the rope, which v2.11's energy share reads VIOLENT (its constraint share gives 1.80 m/s). Rounds run before the
change replay on it as built: re-confirm their candidates on the current head before quoting them.

## Detection power

A search of N independent episodes on a tier misses a mechanism that occurs in a fraction p of its episodes with
probability

  P(miss) = (1 − p)^N,

so it catches with 95 % probability any mechanism with p ≥ 1 − 0.05^(1/N) ≈ 3 / N. At the acceptance size, ~1,000
episodes per tier, a mechanism with p = 0.3 % is missed with ~5 % probability.

| p (per episode) | N for a 95 % catch | N for a 99 % catch |
|---|---|---|
| 2 % | 149 | 228 |
| 1 % | 299 | 459 |
| 0.1 % | 2,995 | 4,603 |

- **p is the rate under the round's whole strategy mix.** A mechanism that only one of k strategies can trigger is
  diluted by k. E1 needs a sustained gripper oscillation, which only periodic sends, and periodic hit it in 1 of 194
  of its own episodes (≈ 0.5 %). In a six-strategy round p is then ≈ 0.09 %, and 1,000 episodes catch it with only
  ~58 % probability. That is why every mechanism found once becomes a reproducer, replayed in every later round.
- Episodes are not exactly independent: CEM concentrates on what it has found, and mutate re-uses the corpus. Read
  the numbers as the power of the random-like share. Every report quotes N per instance and per strategy.

## The corpus (`tools/stress/corpus/`, about 2.0 MB, `SHA256SUMS`)

- `manifest.json`: 56 streams for interlocked_claws (all 5 tiers) and fiber_bundling (very_easy to medium).
  - Sources: L1, L8, the 10-08 incidents and L12.
  - Each stream records its `actions_sha256` (of the float32 array) and its sources.
  - The manifest also holds each family's directed-move modes.
- `reproducers/`: 41 failure reproducers, each with its mechanism and the commit it was found on (M1-M6, V1, F1-F5x;
  E1, E1p and E1m: the finger-coupling pump found by the GPU search, again by periodic on CPU, and rebuilt on medium;
  buzz wire's B1 and B1b (a finger into the static wire), B2 (the pad tunnels the wire's terminal post), the GPU
  search's G1-G3 and G46, the tracking catch-up under the TCP clamp: B3 and the GPU search's G5-G7, G8 and G9,
  the Robotiq spring-link press on static geometry under the lead bound, G10, a Robotiq driver on the wire, and
  G11, a frictionless spring link on the wire, G12, a coupler on the static wire_base, C1, an ALOHA finger
  driven through a claws piece, and K0-K3, knot rope blow-ups).
- `pops/`: 5 legitimate pops (0.74-1.10 m/s). They must never become BREAKDOWN.

## Verified

The packaged tools reproduce the validator's round A3 on claws very_easy (build 593ae66ac, seeds 40017-40020) exactly:

- the reference, nop, reproducer and corpus state hashes;
- all 555 search episodes, record for record (101,414 controls);
- the verdict, PASS.

Sizing a round in episodes changes no episode (claws very_easy, b10 = 0cc192ac2, seed 40017):

- with a 3,000-control budget, random, cem, prefix and mutate give the same 56 records as before the change;
- `--episodes n` gives the first n of them (random 10, cem 20, prefix 3).
