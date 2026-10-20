# blanket_flatten on MuJoCo (2026-10-10)

blanket_flatten's PBD cloth (Genesis) runs on MuJoCo 3.14 as a 2D flex over contact-sphere vertices. The shared code,
the profile and the books are in the development history. Step 1 (references) is in progress on b1ec43997 (the exact
constraint diagonal): the released Genesis references do not acquire on MuJoCo (why: below), so each tier's reference
comes from the family's own feedback policy run on MuJoCo. Step 2 waits for the observer's cloth support (the
validator's open items, below).

## Commits (development history)
| commit | what | kind |
|---|---|---|
| 24c3faceb | `flex_energy`: a 2D flex's membrane / bending potential, which mujoco 3.14's own energy leaves out (`topogym/world/mujoco/energy.py`, `tools/tests/test_flex_energy.py`) | shared |
| e50b9f1cd | the stress observer books the flex energy | stress |
| 323a223fc | MuJoCo PBD cloth: a `pbd_cloth` declaration becomes a 2D flex over contact-sphere vertices (`cloth.py`; scene dispatch next to the rope branch; world readout, checkpoints, `restore_cloth_births`) | shared |
| 7c2c1fe3a | `pbd_local_patch_v1` on MuJoCo (`grasp_cloth.py`): the soft grasp's rules on connect pins, the 20 ms slip window | shared |
| 72b944fe9 | profile `blanket_v1`; the visual-skin audit on Genesis only | family |
| 3059873e7 | rcb_topogym: blanket_flatten builds MuJoCo instances; `bind_flex` in `_render_bridge` | shared |
| 2cba0798f | the observer books the drives as the `discrete` integrator applies them | stress |
| 71676effb | the observer books the `discrete` integrator's own loss on flex elasticity (a gain stays in X) | stress |
| d57613087 | on MuJoCo the route oracle records the cloth's birth deviation instead of refusing it | family |
| 0a879b946 | the cloth grasp pins every flex vertex that carries the patch | shared |

The pending series on b1ec43997 (for review; references follow when they pass):
| subject | what | kind |
|---|---|---|
| shared: MuJoCo cloth keeps a settled birth's evidence | `cloth.py` passes `mujoco_cloth_settle` from the birth source into the cloth record | shared |
| settled MuJoCo births | the settle tool (since replaced by make-instance's `mujoco_birth_settle`, below) | family |
| very_easy declares the measured fold layout | up to two stages, as easier | family |
| measured layouts for exposed flaps and edge strips | fold_stage's exposed-flap rule; `edge_stage`, `crease_stage` | family |
| very_easy declares the base clearance stage | the exact diagonal left its flat sheet 6 mm in the right base's cover | family |
| oracle: measured route stages, the slide, the drop-then-lay profile | below | family |

## The model (`topogym/world/mujoco/cloth.py`, profile `blanket_v1` in `profiles.py`)
- **Vertices.** One body per flex vertex: three slides, a point mass (rho x area, shared equally, as PBD shares it per
  particle) and one sphere geom, r 4.5 mm. The spheres are the cloth's contact with the table and with its own layers,
  as PBD's particles are.
  - Why spheres: MuJoCo 3.6-3.15 cap a flex's own contacts at 50 per geom and body (a cloth on one table box falls
    through), and flex self-collision costs 130-500 ms per step (and segfaulted).
  - r 4.5 mm is above half the grid's 8.49 mm quad diagonal, so the sphere net has no holes (at 4 mm a very_easy flap
    passed through the sheet).
- **Flex.** A 2D flex over the vertex bodies on the authored 51 x 41 grid (its 2091 nodes are PBD particles: the PBD
  mesh is that grid with every edge split once, reproduced with pymeshlab, same order, 1.7e-16 m). Elasticity
  "s3e7": young 3e7, thickness 1.1e-4, poisson .3 (membrane 3.3 kN/m, bending 3.3e-6 N m). The flex is the robot's
  contact surface (radius 4 mm). Contact bits: table 2/3, spheres 4/6, flex 8/1.
- **Particles.** Every PBD particle is a fixed barycentric combination of its flex triangle's corners, so the task's
  predicate reads `state["particles"]` unchanged. Velocity: the end-of-tick displacement (as GenesisWorld since tooling 19).
- **Integrator.** `discrete` (mujoco 3.14 integrates flex elasticity only there), CG up to 200 iterations, tolerance
  1e-6, one 2.5 ms step per tick. The step's own M+K solve runs under the same cap: at 50 it stopped short every substep
  (mjWARN_INERTIA on 100 % of substeps, up to 17 N of force error on the robot); at 200 it converges (+4 % per control).
- **Robot.** The ALOHA as claws v5 imports it (`_CLAWS`, so `ALOHA_COUPLING` is in).
- **Fine flex.** `_CLOTH_FINE` (the 8181-particle PBD mesh as the flex, spheres r 2.9 mm) is one mapping away: exact
  for every particle at birth, about 6x the cost per step.
- **Speed** (open loop, robot + cloth): very_easy 637 rows 4.6 min, easy 3.6 min, medium 9.3 min, hard about 22 min.

## Birth
- The birth comes from the released instance's frozen PBD particles (`SimulatorConfig.mujoco_birth`). Grid nodes are
  exact (`bind_flex` 2.4e-16 m). Sub-grid crumples (finer than the 12 mm grid) are smoothed: the particles lie at most
  1.9 mm (very_easy) to 6.3 mm (gathered tiers) off the coarse surface.
- **Relaxation.** In the first 0.1 s the flex relaxes the PBD state's strain: very_easy at most 7.4 mm, gathered tiers
  19-38 mm, coverage unchanged within 0.003. For comparison, Genesis itself: very_easy 0, hard 3.4 mm, then the PBD's
  own creep, 44 mm in 4 s.
- In energy terms the gathered tiers' relaxation releases about 0.45-0.50 J of strain within 0.1 s, all of it taken by
  the integrator's numerical loss (books below).
- **Settled births (the -i01 instances).** A cloth born straight from a gathered PBD state snaps out of it (a vertex at
  3.3 m/s, 46-71 causal whips: a nop read VIOLENT on easy and medium). The instances are therefore born calm, in one
  make-instance call, as the release rebuild makes every instance: with the simulator override `mujoco_birth_settle`
  (`{"quiet_m_s": 0.05, "quiet_s": 0.1, "max_s": 10}`) make-instance builds the cloth from the released particles
  (`mujoco_birth`: the released instance; the task parameters default to its own, its frozen PBD archive included),
  lets it relax under a nop (the robot holding home, the instance's own profile) until every flex vertex has stayed
  under 5 cm/s for 0.1 s, writes the settled birth source (the released instance with the settled particles and zero
  velocity) into the instance's assets (`instance_assets/mujoco_birth/settled_birth.json`) and builds the instance from
  it. The settle evidence (settle time, speeds, the displacement against the released particles and the unsettled
  birth, coverage / height / stretch before and after) is kept in the cloth record (`scene.json`,
  `entities.blanket.cloth.birth.settle`). For the release spec: `simulator_overrides` = {`mujoco_birth`:
  `@released_instance`, `mujoco_birth_settle`: as above}, `released_simulator_keys` = [`grasp_assist`], variations
  {very_easy: `very_easy`} (the released very_easy variation id), scene seed 0, gripper space native.
- The settle runs physics, so the births were made again on b1ec43997 (the exact constraint diagonal), as claws'
  were (first in two steps: an unsettled instance, the settle, make-instance from the settled
  source). The one-step make inside the release image (0.3.0-rc runtime, on host A) gives the same four births: state_sha
  equal, scene.xml and cloth_blanket.npz byte-identical; scene.json differs only in the settle evidence's source path.

  | tier | birth (state_sha) | settle | peak vertex speed | moved vs released (max / mean) | coverage before -> after | height | stretch |
  |---|---|---|---|---|---|---|---|
  | very_easy | 21441c0569cfbef9 | 0.15 s | 0.24 m/s | 7.4 / 1.2 mm | 0.8594 -> 0.8557 | 13.6 -> 12.7 mm | 1.002 -> 1.001 |
  | easy | f146812a31afc5c5 | 0.35 s | 0.64 m/s | 19.7 / 5.1 mm | 0.6462 -> 0.6631 | 36.9 -> 36.8 mm | 1.168 -> 1.023 |
  | medium | 68c784b1288dc881 | 0.40 s | 0.93 m/s | 32.8 / 6.8 mm | 0.4762 -> 0.4868 | 36.1 -> 38.0 mm | 1.113 -> 1.034 |
  | hard | 3aacbcd368c4d449 | 0.375 s | 1.00 m/s | 24.3 / 4.2 mm | 0.2769 -> 0.2751 | 57.8 -> 61.0 mm | 1.178 -> 1.039 |

  The validator's nop (400 controls, criteria v2.8) reads CLEAN on all four (max vertex speed 0.0 / 0.23 / 0.51 / 0.25
  m/s, lowest vertex 3.5-4.4 mm above the table). medium's 0.51 m/s is a late slip, not a snap: a pleat layer at the
  birth's near-left corner (flex vertex 83, x -0.40, y 0.035) creeps 4 mm down over 5.8 s and then drops the last 7 mm
  onto the table (0.16 m/s over a control step). It is left as it is: CLEAN, and a longer settle could still miss a
  later slip of the same kind (`work/vertex_track.py`; the `work/` tools named in this note are development scripts, not
  included in this repository).

  **Known easing:** the snap itself spreads the gathered cloth, so easy starts +0.017 and medium +0.011 closer to the
  0.95 coverage goal than the released PBD birth, and the PBD birth's stretch (up to 1.18) relaxes to about 1.02-1.04.
  Settling to full rest (1 mm/s for 0.5 s: 0.6 / 2.8 / 7.9 / 17.7 s on the earlier physics) would add only medium's
  other +0.008.
- The route oracle on MuJoCo records the deviation of the live particles from the frozen archive (max / mean) in its
  report instead of the bitwise check (very_easy: 1.9 mm / 0.03 mm).

## Grasp (`pbd_local_patch_v1`, `grasp_cloth.py`)
- The selection and health rules are SoftGraspBackend's, unchanged: both finger surfaces within the declared
  `contact_distance_m` (6 mm) of the patch, the strain / exposure / obstacle guards, the 10 mm follow limit.
- The native constraint: a pool of 24 inactive connect equalities per hand, re-targeted at acquisition to every flex
  vertex that carries the patch (the corners of each patch particle's triangle; an 18 mm patch: at most 14 vertices).
  Every patch particle is then a combination of pinned vertices and follows the hand, as Genesis attaches the patch's
  particles themselves. With only the patch's vertex particles pinned, very_easy's corner lift lagged 10.1 mm (lost at
  the 10 mm limit); now 0.51 mm.
- Slip (clamp, never fail): once the 20 ms mean of a hand's pin-force sum exceeds the ALOHA's 60 N, the patch releases
  ("grasp_slipped").

## Why the released references do not acquire on MuJoCo (very_easy, measured)
- Same released rows on both engines (`work/grasp_geom.py`): the right finger pads' lowest point agrees within
  0.4-0.7 mm at every probed row (205-280). The robot is not the difference.
- Genesis acquires at a distance: at rows ~270-280 both pads come within the declared 6 mm (5.8 / 6.0 mm). The pads
  hover about 5.5 mm above the cloth top; the particles then follow at those offsets.
- Under the pads the PBD birth has a ridge 10.8 mm above the table (particles 7942/7944/7945). The coarse flex smooths
  it (sub-grid) and relaxes it to 7.3 mm, so on MuJoCo the pads stay 9.5 mm away at Genesis' acquisition rows.
- Not the cloth's contact thickness: the fingers never touch the MuJoCo cloth there, and flex radii of 1, 2 and 4 mm
  give identical distances. The resting heights differ by 0.5 mm only (spheres 4.5 mm, PBD 4.0 mm).
- The family's feedback policy (`public_session.run_feedback_policy`) plans its descent from the live cloth and
  acquires on MuJoCo under the same rule.

## Energy books (stress observer)
- The `discrete` integrator applies the drives implicitly, in its effective metric:
  M(q0) qacc = qfrc_smooth + qfrc_constraint + efm_c + efm_ca - H qacc, with H = efm_diag (h D + h^2 K) + the actuators'
  rank-one terms (efm_as) + the flex's K. qDeriv stays zero. An actuator's position stiffness acts as extra damping
  h kp (shoulder: 0.66 N m s/rad). The observer books it (2cba0798f).
- The integrator also loses energy on the stiff flex: 0.44-0.48 J within the first 0.1 s of a gathered birth (very_easy
  0.001 J), about 3 mJ/s during reference-like handling, 0.15-0.16 J/s while shaking a held cloth at 4 Hz or chattering
  the gripper on it. A gain (numerical creation) never exceeded 6e-5 J. The observer books the loss only, from elastic
  terms only (71676effb).
- With both booked: max |X - Wc| <= 7.6 mJ on 8 streams (nop, both Genesis references, shake, chatter, births). The
  committed books before them were off by up to 5.2 J.

## Route oracle on MuJoCo (`topogym/planning/blanket_flatten/`)
The references come from the family's own feedback policy (`public_session.run_feedback_policy`, the route oracle the
released references came from), recorded through Session.step (`work/ref_blanket.py`). It plans every phase from the
live cloth, so its grasps acquire on MuJoCo under the task's own rule. Changes, all engine-agnostic and acting only while
the task is not yet successful (goals and rules unchanged):
- On MuJoCo the oracle records the birth's deviation from the frozen archive instead of its bitwise check (d57613087).
- **very_easy** declares the measured fold layout (up to two stages), as easier does: the laid corner left the near
  edge's strip turned over (coverage 0.934).
- **Exposed flaps.** fold_stage's over / under vote rested on a few crease particles when a flap barely overlaps the
  sheet (very_easy on the settled birth: 5 particles, median -0.15 mm); it chose "under" and the drags crumpled the
  cloth (coverage 0.56). A flap with fewer than 20 particles over or under the sheet is now an exposed flap and is
  turned back like a page.
- **Edge strips.** A strip folded along a whole edge is not a corner flap. `edge_stage` (on fold_stage's rigid flat fit)
  and `crease_stage` (no fit: every triangle's material orientation marks the turned-over strips; the crease is where
  turned-over and face-up triangles meet; the flat direction is the face-up sheet's local uv -> xy map) lay it out
  with both hands, dragging the crease outward past the flat position (tucked) or turning the strip back (on top).
  medium (the far edge's 5 cm strip tucked under along its whole length after the route) and hard (both long edges
  tucked after the page turns: a tri-fold, a third of the cloth turned over, where no rigid fit places the sheet)
  declare up to three such layouts.
- **Measured route stages.** A route stage may be `measured`: planned on the live cloth when it starts (`crease`,
  `slide` or `fold`, with the stage's own parameters). None found, or out of reach: skipped, and the fold layouts and the
  base clearance still follow when it was the last stage. A crease stage may name its edge ((material axis, sign)).
- **Drop, then lay.** Laying a hanging strip down while moving the hand back works only below the strip's own width:
  above it the strip still hangs free, and on the limp MuJoCo cloth a hand that moved back 5-7 cm from 15 cm rolled the
  strip up (medium, coverage 0.78). `crease_stage(lay="drop")`: the strip hangs free (the crease at least 3 cm above
  the strip's width), is carried out past its flat place (`beyond_m`), comes straight down until its free edge is 1 cm
  above the table, then the hand lowers to 3 cm while moving back the strip's width less 2 cm.
- **Reach.** The ALOHA reaches top-down to y 0.36-0.44 over the middle of the table (both arms) and to y -0.40 on the
  robot's side (`work/reach_map.py`); the left arm works x <= -0.05, the right x >= -0.15. A strip tucked along the far
  edge at y 0.45-0.5 cannot be laid out where it lies: `slide_stage` slides the sheet toward the robot from just inside
  its near boundary (both hands, 5 cm up, 3 cm/s) until its far side is within `far_y_m`.
- **Live stages never end the episode.** A stage planned on the live cloth (measured, or a layout after the route) whose
  pinch the assist does not admit, or whose grasp is lost on the way, opens the hands and the oracle goes on (hard
  crashed there once: "Assist did not admit both hands").
- **Base clearance on very_easy.** Under the exact diagonal the very_easy sheet came to rest flatter (0.9999) and its
  box 6 mm inside the right arm base's conservative cover (x 0.237 against 0.231; `work/cover_gap.py`): every cloth gate
  true, the robot never clear. very_easy now declares hard's base clearance stage (a 3 cm slide away from the cover).

## Why medium and hard need more than their routes on MuJoCo: PBD creep
- The released medium and hard routes finish on PBD creep: after the last release the Genesis cloth keeps spreading by
  itself for tens of seconds with nothing touching it (the virtual-hand witness of each route, `work/genesis_witness.py`):
  - medium: coverage 0.882 at the release, 0.909 / 0.931 / 0.947 / 0.968 / 0.975 at +1.5 / 5.5 / 9.5 / 13.5 / 15 s.
  - hard: 0.604 after stage A, 0.671 after its 4 s pause; 0.693 after stage B, 0.717 / 0.739 / 0.761 / 0.776 at +5 / 9
    / 13 / 15 s, still rising. Genesis' hard also ends stage A with both long edges tucked under (25.6 % of the
    triangles turned over; the -u end 0.115 m wide), and the creep opens them (14.2 % turned, the -u end 0.26 m wide,
    15 s after stage B).
  - The route comments call it relaxing ("the sheet relaxes flat over ~40 s after release"). A 4 kg/m^2 cloth on a
    mu 0.5 table does not crawl; it comes from PBD's position-level friction and contact.
  - Not bending spring-back: the gathered tiers declare `bending_relaxation` 0, and Genesis' PBD bending projection
    (`pbd_solver._kernel_solve_bending`) scales every correction by it, so their PBD cloth has no bending stiffness at
    all. The MuJoCo cloth's small bending (3.3e-6 N m, a 4 mm bending length) is the nearer of the two to that.
- MuJoCo's cloth stays where the hands leave it: the routes alone end at medium 0.888 (the far edge's 5 cm strip tucked
  under along its length) and hard 0.42 (both long edges tucked, a third of the cloth turned over).
- So on MuJoCo the oracle must lay out with its hands what the creep laid out on Genesis. The goals and rules are
  unchanged.

## hard: blocked, flagged for step 3 (2026-10-10; on the flex. Ported on the GPU step in V3, below)
The released hard route needs Genesis' PBD creep, and no measured re-plan within the time limit replaces it on MuJoCo.
- The route's page turns (stage A) remove the twist but leave the pleats. The sheet lies diagonally, both long edges'
  strips are tucked, and its far side is beyond reach. Coverage on MuJoCo: 0.40 after stage A, 0.405 after stage B.
- Genesis gets 0.604 after stage A, 0.671 after its 4 s pause, and 0.693 -> 0.776+ in the 15 s after stage B, with
  nothing touching the cloth.
- The measured re-plan, all on b1ec43997 (route overrides in the development work notes):
  - Slide the near side toward the robot to y -0.15 from just inside its lower envelope. The farther hand pulls longer,
    so the sheet turns toward the x axis and into reach: 0.641.
  - Then a far-side slide to y +0.28, putting tension on the pleats: 0.732.
  - Crease layouts after either slide add nothing (0.63 / 0.59). The near edge's rolled strip and the far strip's crease
    are not exposed to a pinch: the assist never admits the hand, or no exposed crease point is left.
  - Corner layouts (fold_stage): the first gives 0.794, the second takes it back to 0.753. On a sheet that is still
    compressed, the rigid flat fit is poor (about 3100 particles off it), so it picks corners that are not folded.
- Length: the best run had already used about 4000 rows (40 min per run unloaded) at 0.75. A run reaching 0.95, if one
  exists, would be well over the 50 min per observed run.
- The planner options explored for it (slide to a target y on either side, the largest crease strip only, exposed
  crease grasps) were not committed. The redesign's hard4b (four double folds under a half fold, a fence; 0.965 under
  the exact diagonal at object level) replaces this tier.

## What the flex gets wrong (for the cloth solver that replaces it, 2026-10-10)
The flex is not developed further (2026-10-10: only necessary bugs get fixed; a VBD cloth solver is the likely
successor under the same designs). Two findings for that solver.

**Small flaps stay folded: the collision surface.** In the videos a small turned-over piece at one side is never turned
back (the near-right ear of the redesign's flatten finals; medium's reference also ends with 0.3 % turned there).
Diagnosis on two finals' ears in the redesign's FlexSim lab (blanket_v1 scene, exact diagonal, no robot, the assist's
patch rule):
- Not the patch rule: the ear's corner is exposed and admitted (10 particles, 4 pinned vertices).
- The contact spheres (r 4.5 mm on the 12 mm grid) leave holes. A small flap's spheres sink into the holes of the layer
  under it: 2.8 mm above it instead of 9 mm, contact normals 18 deg from horizontal, so sliding out takes 5.7x the load
  at the spheres' friction 0.15. (A half fold's two full layers sit on edge saddles instead: 6.7 mm apart, 1.2x.)
- A turn-back from the corner (the oracle's lift / over / down) stretches the cloth 23-36 % (5 N on 4 pinned vertices,
  70x the ear's weight) instead of turning the flat ear. A curled ear opens while held and re-curls within 0.25 s of
  the release. Slower, a hinge-centred arc, or holding 3 s first change nothing.
- One change at a time from the same state: friction 0 frees the flat ear but not the curl; 10x bending keeps the curl
  open but not the flat ear; friction 0.075 or 0.3, bending 0 and bending damping change nothing. Overlapping spheres
  (r 6.5 mm, the 12 mm neighbours excluded) free both: the curl opens by itself as the spheres grow. r 6.0 does not
  free the flat ear.
- So the sphere net interlocks stacked layers, and the cloth is too limp to keep a flap open once it is released.
  Overlapping spheres would cost: a single layer rests at 6.4 mm (now 4.4); a two-layer fold's top layer at 17.9 mm
  (now 11.1) and its peak at 30.8 mm (the 30 mm flatness gate fails); +60-80 % per step where layers stack (cloth only);
  and every birth and reference re-made. Not done.

| property | the flex | a real blanket |
|---|---|---|
| areal mass | 4.0 kg/m^2 (1.15 kg sheet; the released value) | 0.3-0.8 kg/m^2 |
| bending | 3.3e-6 N m: bending length 4.5 mm | 1.5-5 cm |
| membrane | 3.3 kN/m | woven fabric 1-20 kN/m |
| cloth-cloth / cloth-table friction | 0.15 / 0.5 | 0.3-0.6 / 0.3-0.6 |
| collision surface | spheres r 4.5 mm on a 12 mm grid: layers 2.8 / 6.7 / 9 mm apart, by how their grids register | smooth, 3-10 mm per layer |

**BL1: a solver starved elsewhere blows up the cloth.** Under the `discrete` integrator the flex's stiffness is solved in
the same CG system as every contact. In BL1 (medium, random stream e30) the left fingers locked at 40.7 kN on degenerate
pad-in-hull contacts, CG hit its 200-iteration cap in every substep, and the cloth blew up (24 m/s) with nothing
touching it. Fixed at the source by the pad rule (08964a3bad); the three references are unchanged under it. A cap of
1000 also held e30 (+4 % per control, but every birth and reference moves) and stays unapplied. A cloth solver of its
own would not share the contacts' solve.

## The VBD cloth backend (profile `blanket_vbd`, 2026-10-11; the flex stays the default until acceptance)
The cloth is Newton's VBD (`topogym/world/mujoco/cloth_vbd.py`) on the authored 51 x 41 grid, stepped after every
`mj_step` with the deterministic CPU step (`vbd_fast.py`, frozen 2dbf9fce...ae2e on 10-11, with the coarse
correction; first 54e8df91...40f8). The interface it keeps:
`blanket_cloth_interface.md`. Selected by `SimulatorConfig.mujoco_profile = "blanket_vbd"`.

**On the GPU (profile `blanket_vbd_gpu`, 2026-10-12; decision: the blanket references are recorded on GPU, within
the 72 min budget).** The same cloth stepped by the deterministic CUDA step (`vbd_gpu.GpuDetSolverVBD`: the frozen
CPU step's kernels and kwargs on one device, self-contact incidence in a canonical order, exact-min truncation,
fixed-order coarse assembly, CUDA graphs, FMA contraction off in the process). Bit-identical across fresh processes and
GPU classes (sm_80 / sm_89 / sm_90 / sm_120), not bit-identical to the CPU step: a reference replays on the profile it
was recorded on. A frozen scene built on `blanket_vbd` runs on the GPU under this profile (`cloth_vbd.PLACEMENT_KEYS`:
device, solver and its kwargs from the profile; the material from the scene). A GPU process steps no CPU cloth.
Two more changes ride on this profile only:
- `mujoco_substeps` 4: MuJoCo steps at 0.625 ms, 4 steps per 2.5 ms cloth step, as every other ALOHA family steps it.
  At 2.5 ms, MuJoCo's refsafe clamped the finger coupling's 1.25 ms time constant to 5 ms, and the coupling returned
  energy (validator BLF2c / 3c / 4a, eq:1). The cloth's colliders and pins take the fresh kinematics after the 4th step,
  so they move over the cloth step from the tick's start pose to its end pose. Before, they lagged one step and never
  moved within a cloth step.
- `cloth_floor`: the table guard keeps non-pad gripper geometry above the local cloth stack. The cloth is
  one-way coupled, so without it a gripper base could press a stack flat into the table (directed_e0: 227 N over 18 mm).
- `adaptive` iterations (adopted 2026-10-13; on in blanket_vbd_gpu and blanket_vbd_cpu_twin, off in
  blanket_vbd). After the step's 20 Gauss-Seidel iterations, blocks of 20 more run (up to 80) while the last iteration's
  largest vertex update is above 0.1 mm. The test is an exact max, so it stays deterministic, and a step under the
  threshold is the plain step.
  - Why: 20 iterations do not converge in a crumpled stack at rest, and the unconverged steps ratchet membrane energy up
    with nothing driving it (validator wave 2, hard4 mutate_e3 on the twin: X rise 0.268 J). With 80 iterations the same
    window is dissipative; plastic flow and the coarse step are not the cause.
  - Evidence: 8 perturbed copies per stream, verdicts at v2.12:

    | stream, arm | breakdowns | X rise median / max | iterations (mean) |
    |---|---|---|---|
    | mutate_e3 (twin), off | 1/8 | 0.067 / 0.268 J | 20 |
    | mutate_e3 (twin), adaptive 0.1 mm | 0/8 | 0.056 / 0.133 J | 22.4 |
    | directed_e6 (GPU), off | 7/8 | 0.386 / 2.28 J | 20 |
    | directed_e6 (GPU), adaptive 0.1 mm | 6/8 | 0.354 / 0.92 J | 25.6 |

  - The pressed-pile class (directed_e6: a held gripper on a stack) is not solved by it, nor by any other solver option
    tried; it stays a criteria / scope question.
  - Cost: on the very_easy, very_easy4, easier4 and fold very_easy routes 20.004-20.04 iterations per step on average
    (under 0.1 % of steps extended); in a pressed pile +28 %. With it off, every run is bit-identical to before.

**The family's material (chosen from the plastic-bending study):**
| quantity | value | why |
|---|---|---|
| areal mass | 0.5 kg/m^2 | a real blanket (0.3-0.8) |
| membrane | Lame mu 1270 / lambda 1090 N/m (E t 3.3 kN/m, nu .3) | the flex's |
| bending | edge_ke 6.5e-3 (7.6e-5 N m, bending length 2.5 cm) | at 1.41e-3 a curled corner partly unrolls by itself |
| plastic bending | yield .6 rad, rate .02 per 2.5 ms step (8 /s), max rest angle 2.7 rad | the only studied setting where a dog-ear and a curl both hold alone and both stay open after the turn-back; at yield .3 a pinched tip stays set |
| contact | table distance 3 mm; layer offset 7 mm (self margin), +1.4 mm detection gap; body gap 4 mm | 7 mm > the assist's 6 mm exposure clearance |
| friction pairs | cloth-cloth .4, cloth-table .5, cloth-robot 1.0 | the task's table friction; fabric |
| solver | 20 iterations per 2.5 ms step, 4 threads; coarse correction every 2nd iteration (patches of radius 4 hops, 100 sweeps) | 5 iterations leave folds creeping 2-8 mm/s (spike); without the coarse step a drag reads 37 N for a 0.71 N load (below) |
| friction smoothing | friction_eps 1e-3 m/s | half the fold creep of 1e-2 |

**State.** Under plastic bending the bending rest angles are state: they are in the checkpoint, in the instance
(`cloth_vbd_blanket.npz` `birth_rest_angle`; the settled source `vbd_cloth_rest_angle`) and in the particle readout, so
the state hash covers them. The plastic dissipation (`solver.plastic.loss_j`) is in the checkpoint too.

**Birth.** make-instance's settle on VBD: every grid vertex under 5 cm/s for 0.1 s, and every rest angle moving less than
1e-3 rad per control (`quiet_rest_angle_rad`). Without the second rule a very_easy birth settled in 0.15 s while its rest
angles still moved .62 rad in the next second (1.2 mm of motion). very_easy -i01 on the frozen step (f3bf61560):
settle 1.3 s (52 controls), coverage 0.8593 -> 0.8625, state_sha ed36dbbfece3ca48, the same on the cluster (Intel Xeon
6230R) and a cloud host (AMD Zen 4).

**very_easy reference on VBD (f3bf61560, the cluster CPU).** The route oracle unchanged succeeds in 1341 rows (the flex's:
1897): coverage 0.99995, height 3.9 mm, the stretch peak 1.02 at the second corner's lay (1.24 on the step without the
coarse correction); final state_sha 8e3cb95c197621ee; 53 min at 2.4 s per control on a busy cluster node.
privileged.npz sha256 a20e39aba5519453bdf3f56be8bd0de88c2c9522d78a9d15bd179a1f5c3baf86.
- Energy books (the stress observer, ve_F1 rows 1-260): the same end state with and without the observer; the cloth's
  X -35 mJ (dissipative), its largest rise 4.3 uJ; the pins' work +0.67 J; plastic loss 2.2 uJ; observer errors 0.
- Determinism: fresh-process replays of the recorded rows end on 8e3cb95c197621ee on the cluster (Intel Xeon 6230R) and on
  a cloud host (AMD Zen 4), with every 200-row hash equal (row 600 7edfab8a3bb05851, row 1200 2aa8418e5b658a5a).
- Restore: rewind at row 260 of ve_F1 with the right hand's pins active, 40 rows again: every hash equal. A restore test
  must also reset the Session's previous targets (`Session._targets`, from which the speed limits and the table guard
  step), as `Session.reset` does after its rewind: a bare `env.privileged.rewind_to` leaves the last row's targets, and
  the first row after it is slewed differently once the arm has moved (rewind at row 600 of the reference, the first
  try).

**Measured earlier (a cloud host, very_easy, the step without the coarse correction, the flex-era rows of ve_F1):**
- Resting heights (particle centres above the table): one layer 3.0 mm, two 10.0 mm, three 17.0 mm; the gap to the
  layer below 7.0 mm (p05 6.97). A half fold's crease peaks at 19.8 mm, a letter fold's at 33.7 mm. So blanket_fold's
  1 mm "below" bound and the 6 mm exposure clearance hold unchanged.
- Fresh-process determinism (elastic step): two 300-row replays end on 1e5ab88aa84fb471.
- Exact restore (final material): rewind at row 260 with the right hand's 9 pinned vertices, 40 rows again: all 40
  hashes equal. 4 threads = 1 thread bit for bit.
- Hold from a birth settled on speed only, 8 s: rest angles set in the first second, then 0.39-0.45 mm/s of creep
  (coverage +0.002 over 8 s).
- Grasp: pin forces 0.5-7 N while the right hand lifts its corner (the 60 N slip rule never fires).
- Render: nyx_direct shows the fabric; MuJoCo's depth raster has the cloth surface 3.4-3.7 mm in front of the table.
- Cost: 1.6-2.5 s per control on a cloud host at nice 19 and load ~120 (0.49 s/control on an idle host A).

**Drag forces (2026-10-11).** At 20 Gauss-Seidel iterations, table friction kept the solver from carrying smooth tension
across the sheet. Dragging a lifted corner 25 cm at 10 cm/s read 37 N at the pins (physical: at most mu m g = 0.71 N),
with peak strain 1.39 next to the hand and 19 N still there after the hand stopped.
- More iterations only scale it down: 40 gave 18 N, 60 gave 12 N.
- Speed barely matters: 5 / 20 cm/s gave 30 / 42 N.
- Damping changes nothing, and the residual stopping rule makes it worse.
- friction_eps 1 m/s gives 10 N but loses static friction: a laid fold creeps 21 mm in 3 s.
- Table mu 0 gives 0.54 N, which is the diagnosis.
The fix is the coarse correction in `vbd_fast.py`: an exact Newton step on the free vertices' common translation
every few iterations, with table friction entering it as a Coulomb capacity. The same drag then reads about 1 N.

**Tiers on VBD.** very_easy is the released tier migrated to VBD. easy and medium are superseded by the redesign's VBD
tiers (2026-10-11): their PBD gathered starts are not a VBD equilibrium. easy carries 17 % membrane strain, and
PBD's pleat layers nest far closer than the 7 mm offset. Settled on VBD, with or without rest angles set to the shape
and kinetic damping, 40-120 hinges keep ratcheting plastically and slips recur after 10 s.

**Exposure rule on VBD (2026-10-11).** The assist rejects a patch under a covering layer: a particle within 18 mm
horizontally, more than 6 mm higher, and more than 36 mm away in material. On the flex this distance was measured from
the covered patch particle. On VBD it is measured from the WHOLE patch (`exposed_patch(remote_from_patch=True)`, set by
`MujocoVBDClothGraspBackend`; the flex keeps the old rule, byte-unchanged).
- Why: a patch centred on a fold's bend reaches 18 mm down the bottom layer. VBD's top layer lies 7 mm above that,
  over the clearance, and a few cm past the bend it is more than 36 mm in material from those bottom particles. So
  every bend grip read "occluded", including the apex. On the flex the layers were 3 mm apart and this never fired.
- The top layer there continues the patch's own surface: it is within 36 mm of the patch's top part. A layer folded
  over a patch is at least twice its distance to the crease away in material, so it still covers.
- `tools/tests/test_exposure_rule.py` checks the cases:
  - bend grips pass;
  - the bottom layer under a half fold fails 1.5-15 cm from the crease;
  - the sheet under a dog-ear's tip fails;
  - a corner tucked under the sheet fails;
  - a letter fold's lower two layers fail;
  - the flex rule equals the previous code.
- The clearance stays 6 mm. At 8 mm, both rules would pass the bottom layer under a 7 mm fold.

**Known simplification (accepted for acceptance, 2026-10-11).** The robot links feel no reaction from cloth
contact, only from the pins: the pins' reaction goes into the hand's `xfrc_applied` one step later, while the robot's
other contacts act on the cloth only. The sheet weighs 0.144 kg (1.4 N in total) and the pins carry the real loads.
The proper fix is the two-way coupling below (off by default).

**Two-way coupling (profile `cloth.two_way`: ON in `blanket_vbd_gpu` (decided 2026-10-12, the cloth floor kept as a second line of defence), off in `blanket_vbd` (the frozen CPU references); `cloth_coupling.py`).** The cloth's
contact forces on every robot body go back to MuJoCo, so a gripper pressing a stack is held up by it.
- Per cloth step (after the solver, at its end positions against its start): per robot contact (particle and edge /
  face), Newton's force on the cloth's free corners (pinned vertices are the pins'), the rigid contact point, and an
  effective stiffness: the penalty in series with the particle's own support (its vertex-solve Hessian without this
  contact: inertia, membrane, bending, table and other contacts, self-contacts), so a hanging corner gives ~12 N/m per
  vertex and a stack at most ke. A Warp kernel, one thread per contact, FMA contraction off; the host sorts the robot
  contacts by (shape, corners) (the GPU step's key order) and sums per body in float64, in that order.
- The wrench for the next cloth step's MuJoCo steps (held over them: 4 x 0.625 ms on `blanket_vbd_gpu`), linearly
  implicit in the robot's motion: (I + dt (c dt K + D) W) G = F - (dt K + D) v - dt (c dt K + D) a, W the bodies'
  mobility at implicitfast's metric (M + h D_joint), a the last MuJoCo step's non-cloth acceleration, c = (k + 1) / (2 k).
  At rest G = F. Why implicit: the ALOHA's gripper base has 1.8 kg along z and a finger 0.18 kg along its slide; with
  50-100 contacts at ke 2e4, k dt^2 / m passes 4 and the lagged explicit force would oscillate.
- Limits (clamp, never fail): no pull along the body's stiffness-weighted contact normal; friction's tangential wrench at
  most max(|F_t|, 0.4 G_n) (0.4: the weakest pair friction, where a dragged stack slips; friction's Hessian holds within
  a 2.5 um slip only); the cloth's torque on any actuated joint at most 2x its actuator's limit (per arm).
- Books (the coupling's work is booked apart, never as the cloth's own excess): `cloth.coupling` exposes
  `work_on_robot_j` / `step_work_on_robot_j` (the wrench's work on the robot over a cloth step's MuJoCo steps),
  `work_on_cloth_j` / `step_work_on_cloth_j` (the robot contacts' work on the cloth's free vertices over the cloth step)
  and `applied` (the wrench per body, held in `xfrc_applied` beside the pins' reaction). The cloth book reports W_react,
  W_couple and their balance C = W_couple + W_react (> 0: created by the lagged exchange), peak `vbd_coupling_rise_j`;
  the cloth's X keeps subtracting the robot's work. The validator books them in criteria v2.12.
- The colliders and pins read the current qpos's kinematics (a private MjData). On `blanket_vbd` (one MuJoCo step per cloth
  step) this also removes the one-step pose lag: there one-way books W_robot = 0 (colliders at rest within a cloth step),
  and the robot's push lands in X as creation (e0: 12.22 J one-way stale, 7.43 J with fresh poses, W_robot 15.66 J).

Measured (validator easier4 directed_e0; `cloth_floor` off so the coupling acts alone):
| | one-way | two-way |
|---|---|---|
| right gripper base | on the table, z 0.9503, 44-52 N (peak 220-227 N) | held by the stack, z 0.9617, no table contact |
| layer gaps under it | min 6.3-6.75 mm, stack squeezed 18 -> 14 mm thick | min 6.80-6.90 mm (p05 7.0), stack 7.8-8.5 mm |
| cloth book excess rise | 12.22 J (CPU), 4.88 J (GPU) | 0.3-0.5 mJ, energy_created 0.0, C -0.02 J (CPU -0.01 J) |
| force on the gripper base | - | impact 52 N, then F = G 8.0 +- 0.3 N (steps 700-769), step-to-step change p99 0.065 N |
| determinism | | GPU: f389e44046e5c633 x2 on RTX PRO 6000 (sm_120) = RTX 4090 (sm_89); CPU: Intel 6230R = AMD Zen 4; rewinds at rows 54 / 62 / 70 exact on both (wrenches pending) |
| cost per cloth step | | GPU 3.5-24 ms (shared GPUs; the cloth step 160-520 ms), CPU 0.6-1.3 ms (308 ms) |
Routes, two-way vs one-way (the cluster CPU, blanket_vbd, the route oracles closed-loop):
| route | one-way | two-way |
|---|---|---|
| flatten very_easy -i01 | success, 1279 rows, 7274ae93 (= the recorded reference), pins p50 / p95 / max 0.51 / 1.05 / 1.12 N | success, 1294 rows, coverage 0.99993, pins 0.50 / 1.04 / 1.14 N |
| fold very_easy -i01 | success, 929 rows, pins 1.65 / 7.5 / 11.8 N | success, 885 rows, every term, pins 1.67 / 10.1 / 13.0 N |
On `blanket_vbd_gpu` (4 substeps, cloth_floor on) with two-way: flatten very_easy success in 1298 rows (coverage
0.99994, pins 0.51 / 1.02 / 1.13 N), fold very_easy success in 881 rows (every term, pins 1.64 / 10.2 / 15.4 N).
No pin force near the 60 N slip rule in any. The MuJoCo-actuated accept and fold-8 s test programs give the same verdicts
(dog-ear freed and open; fold 600 turned triangles, gaps 7.0 mm, creep 3.2 / 2.7 mm over 3 s), finger wrenches <= 7 N;
drag, stack crease / corner and press-release use kinematic hands only, which two-way does not touch.
Rejected on the way: G . F as the no-pull test and unbounded friction stiffness (impact G_x -63 N under F_x -1 N, G_z
-16 N); friction explicit (chatter: G +-1 N per step, 10 um jitter, the coupling created +0.31 J).
With `cloth_floor` on (`blanket_vbd_gpu`) the guard keeps the gripper above the stack: contacts, no force, nothing to do.

## The redesigned tiers on VBD (2026-10-11)

The blanket redesign replaces the tiers with five flatten starts
(`very_easy4`, `easier4`, `easy4`, `medium4`, `hard4`) and adds the family `blanket_fold` (ladder B). The MuJoCo side:

- **Births.** Each start is a VBD lab state: 8181 particles, plus the bending rest angles (`rest_angle`) in Newton's
  edge order (`edge_indices`, checked equal to the product cloth's). The birth source is the released instance of
  the tier's difficulty, with the cloth replaced by the start's particles at zero velocity. The rest angles go to
  `vbd_cloth_rest_angle`; `minimum_coverage_ratio` is 0.95 and the assist's strain guard 1.5.
  - easier has no release of its own. It takes very_easy's, dropping its fixed `corner_fold_profile`.
  - make-instance settles the birth as for every VBD birth, in 2-3.4 s. Coverage changes by at most 0.003 from the
    lab's.
  - Without the rest angles, the very_easy4 start never got quiet in 10 s, because the folds kept yielding.
  - Tools: `work/redesign_birth_source.py` and a cluster batch script.
- **Fence (hard4).** New task parameter `obstacles` (default `()`, in `hash_when_changed`, so existing instance ids
  are unchanged). Each entry is `dict(name, center_xy_m, half_size_m, yaw_deg=0, friction=.5)` and becomes a fixed box
  standing on the table top. MuJoCo collides it with the robot; the VBD cloth takes it as a static collider, like the
  table. hard4's fence: centre (-0.1356786, 0.322), half sizes (.24, .02, .03).
- **Route planner hook.** `routes.route_for(variation)` returns a declared route. For a redesigned tier it returns a
  fresh `redesign_routes()` entry: a planner holds one episode's state.
  - `run_route_oracle` calls `route["planner"](cloth, uv, table, target, task=task)`. It is called from the start when
    the route declares no stages, and after each last stage once the sheet is quiet: a Quiet settle capped at
    max(stage `settle_s`, `planner_settle_s` = 6 s).
  - Calls continue while the task is not successful, at most `planner_rounds` times. The returned route-format stages
    (`[]` = done) are appended.
  - Planned stages are planned live: an unreachable or unadmitted one is skipped and the planner is called again.
  - A hand that a segment's `to` / `by` does not name holds.
  - The report lists `planner_calls`. `public_session` accepts the lab tier names (`very_easy4c`, ..., `hard4b`)
    against their difficulty's variation.
- **blanket_fold.** rcb_topogym's `TASKS["blanket_fold"]` uses blanket_flatten's simulator; its goal text is
  provisional. The profile `blanket_vbd` lists the task.
  - make-instance packages `fold_target_path` into `instance_assets/fold_target/` (relative; Session resolves it), and
    a birth taken unsettled into `instance_assets/mujoco_birth/`.
  - Births (`work/fold_birth.py`): easy / medium from easier4's settled flatten birth, hard from hard4's (with the
    fence), very_easy / easier from a flat settled birth. The flat birth is the product grid's particle view, centred
    at (-0.1356786, 0.05), at table + 3 mm.
  - The archive's acceptance comes from the flatten tier's own Task (the hard archive is `scene_layout`).
- **GPU search** (exploration only; records are CPU): on host C, in the record
  image with the bound tree. Its `cloth_vbd.py` is patched to the deterministic `GpuDetSolverVBD` (`VBD_DEVICE=cuda:0`).
  It runs at about 0.6-1 s per control on an RTX 4090; the CPU step on the cluster takes about 2.7-7 s, depending on the
  node's load.
- **Fingertip footing (3975f4c68).** A lab hand is a point; a robot pinch is two fingers.
  - medium4e and hard4b's first grasp (`far_half_back`) sits 12 mm inside the turned-over layer's edge. At yaw 0/180 the
    jaws close across that edge, one finger stands beyond it (finger gaps 7 / 16 mm), and the assist never admitted it.
  - Routes with `footing` (all ten redesign routes) therefore try yaw 0/90/180/270. They prefer the yaws whose two
    fingertips (1 cm either side of the grasp point along the jaw axis, measured from the finger surfaces) stand over
    cloth tops within 3 mm of each other; joint margin breaks the tie. After this, the GPU search had no unadmitted stage.
  - A planner that takes `last` gets the previous planned stage's outcome. redesign_routes re-plans a skipped, unadmitted
    or lost stage once (4 cm grasp search) and otherwise reports it to the lab program as not done.
- **Fast calls (2026-10-11 night).** Each planner call used to cost about 6000 physics steps on medium4e,
  about 4300 of them robot overhead: a 3 s approach, a 1.5 s descent, 0.3 s holds, a 10-control contact dwell, a 0.5 s
  release and a 1.5 s, 8 cm retreat. Routes with `fast` (all ten redesign routes) now use:
  - an approach at the standard joint-step limit (0.04 rad per control, x0.975, smoothstep peak 1.5x the mean, at least
    0.6 s);
  - a descent at 0.12 m/s;
  - 0.1 s holds, a 0.3 s close, a 5-control contact dwell and a 0.3 s release;
  - between the stages of one planner call, a 3 cm lift in 0.3 s; after a call's last stage, 8 cm in 0.6 s.
  The cloth's own moves (the route segments) and the quiet settles are unchanged. A planner call may now return several
  stages (`programs.run_stages`: one measurement fixes them, e.g. a round of corner corrections). Its later stages keep
  the lab's grasp-relative `by` targets. `last["stages"]` reports every stage of the previous call. Routes without
  `fast` (very_easy and the released tiers) are unchanged.

## Route validation on the GPU step (2026-10-12; references wait for the physics freeze)

The family ships on `blanket_vbd_gpu` (the deterministic CUDA step, 4 MuJoCo substeps per cloth step, the table
guard's cloth floor). Per the process of 10-12, references are recorded once, after the physics freeze. Until then
each tier's route is checked on the current head by one closed-loop run, below.

- Runs: host C RTX 4090, the record image plus the bound tree, with the masks. The births are the rec1 instances
  (the CPU `blanket_vbd` births) re-pointed to `mujoco_profile: blanket_vbd_gpu` (same states).
- Tools: run scripts for host C, a cloud host and the cluster (development work notes).

| tier | route | head | rows | end | coverage |
|---|---|---|---|---|---|
| very_easy | very_easy | 694e790b5 (+ 7213ede04) | 1298 | 0a8b914f39cc8988 | 1.0 |
| very_easy4 | very_easy4c | 2a5f6adf0 + 7213ede04 | 2568 | 1df47e579397ca01 | 0.99 |
| easier4 | easier4 | 2a5f6adf0 + 7213ede04 | 1965 | 68d07992fc4ce210 | 0.95 |
| easy4 | easy4f | 2a5f6adf0 | 3520 | bdd39d02f94d2de5 | 0.98 |
| medium4 | medium4e | 2a5f6adf0 | 5961 | be2925336ea4f407 | ~0.95 |
| hard4 | hard4b | 2a5f6adf0 | 5214 | 22725a5bfb5b31e2 | 0.96 |

- Before 2a5f6adf0, the fast routes' full jaw opening at release stretched the cloth 2-3.6x on this step. After it,
  release stretch is mostly <= 1.2, peaking at 1.7-1.8 on medium4e / hard4b.
- very_easy: deterministic x2 in fresh processes (fdabb6f25: 6989b1861d7d7109 twice).
- CPU records on `blanket_vbd` (ccd4ba7fa SIF, masked, x2 fresh replays), kept as data:
  - very_easy: 1279 rows, 7274ae938f8ba75f;
  - very_easy4c: 3217 rows, a5f0b9fc7a1fb907;
  - easier4: 2372 rows, 84c00969a60773ff;
  - medium4e (fast): 3745 rows, 71a6eed5d0e05de3;
  - hard4b (fast): 4641 rows, 92f314ec5d02d715.
- Masking: runs outside the task containers must set NPY_DISABLE_CPU_FEATURES / OPENBLAS_CORETYPE (MUJOCO_CORE.md).
  Unmasked Zen 4 runs took other routes (medium4e: 4026 rows vs 3745 masked).

## The ported easy / medium on the GPU step (wave 2, 2026-10-14 / 15)
Both births were made on 10-14 with the release tool's make path (`release_rebuild.make_instances` in
eai-roboarc:0.1.0-noinst, host A RTX 4090, the birth settle up to 60 s). Both references were recorded by the route oracle in
eai-roboarc:0.1.0 with the branch's planning bound in, and both ship PREMADE. standard.npz is the same file as
privileged.npz: every arm joint step is within the standard service's 0.04 rad.

| base | birth | reference | rows | end | privileged.npz sha256 | max joint step | strict fresh replay (host A) |
|---|---|---|---|---|---|---|---|
| blanket-flatten-07-i01 (easy) | 12fe5f2d5f7374a7 | EASY: the far-edge drag, then the near-edge pull | 1581 | a6f78e4255ff5adf | dfb55159b9ae2794... | 0.0275 rad | 838.9 s = 7.8 % of 3 h |
| blanket-flatten-08-i01 (medium) | 49ee5e7e16239616 | MEDIUM: the far-edge drag, then the hang-and-lay stage | 1480 | d539d4d85d80ab0e | 3f9d713fc87c9bc8... | 0.0303 rad | 843.0 s = 7.8 % of 3 h |

**medium: why it needs the hang-and-lay stage.**
- The birth carries a standing pleat across the whole width, at material v -0.11..-0.01. It is about 14 cm of cloth
  packed into 3-6 cm of y and 2.5 cm tall: an S of three stacked layers whose creases the plastic bending holds.
- The far-edge drag carries the pleat along. After it every u column has 7.4 cm of slack, and coverage is 0.78.
- On the table nothing opens it. Each of these moves the whole sheet together with the pleat:
  - the route's measured slide: coverage 0.814 at the end of the 10-14 route, 2987 rows;
  - easy's near-edge pull at 10 cm;
  - a pin-and-pull (one hand holds the sheet 2 cm beyond the pleat while the other pulls the near part 9 cm). It
    opened the pleat by only 0.7 cm, in the middle or on either side.
- In the air it opens. Both hands pinch the near edge at u +-0.15, lift it 12 cm, then rise to 20 cm while moving 3 cm
  toward the robot. The pleat hangs off the table under the near part's weight. The hands hold 2 s, then lay the near
  part down 9 cm toward the robot, 3 cm above the table.
- Result: slack 0 in every column except u +0.30 (1.9 cm), coverage 0.954, and success at the retreat. The route's
  measured slide / crease stages after it are not reached.
- 25 cm / 10 cm instead also opens it, but stretches the cloth to 1.20 at the release.

**medium: checks, all in eai-roboarc:0.1.0 with the MuJoCo task masks.**
- Birth probe on host B: instance_exact and assets_exact, state 49ee5e7e16239616.
- Strict fresh replay on host A: success at row 1480, end d539d4d85d80ab0e, 843.0 s. The same GPU was also running the
  re-record and another job.
- Observer replay on host B (host-local cuda-compat override): success, the same end, CLEAN under v2.13b.
- nop 400 on host B: no success, CLEAN.
- A re-record from the committed route (no override) gives byte-identical rows and tick ledger.

## hard ported on the GPU step (V3, 2026-10-17 / 18): blanket-flatten-09-i01
The birth is the 10-14 make (`release_rebuild.make_instances` in eai-roboarc:0.1.0-noinst on host A, the same run as the
easy / medium ports): c7e8fe0051099141, settle 20.6 s, coverage 0.277 -> 0.453 (the goal: 0.95). It ships PREMADE as
07 / 08 do. The released HARD route reaches 0.52 on it (10-14), so the reference has a new route, `hard5`
(`topogym/planning/blanket_flatten/hard_port.py`, `routes.route_for("hard5")`). The route is a sequence of three
parts (`robot_oracle.run_sequence`, new: each part is a route of its own, run from the state the previous part left).

| part | what | measured after it |
|---|---|---|
| 1 | the released page turns (HARD stage A) with the redesign's fast approaches and footing, then a 3 s settle | coverage 0.571; the far long edge's strip (~6 cm) tucked under along the whole length, a standing pleat across the width at v ~ -0.06 (4-7 cm tall), the near-right corner folded |
| 2 | hang and lay, as medium's: pinch the near side at (-0.20, -0.22) / (0.05, -0.17) (low, single layer), lift 12 cm, rise to 20 cm moving 3 cm toward the robot, hold 2 s, lay it 9 cm toward the robot; then the measured re-centring (reverse.recenter, 2 rounds on the planner hook) | the pleat opens (0.659); near pair to its flat places 0.884, far pair 0.905; the sheet flat and close to axis-aligned, only the far-left strip still tucked |
| 3 | the measured slide toward the robot (slide_stage, far side to y 0.16: 13.1 cm), then re-centring, fold layouts and finishing relays while the goal is not met | the tucked strip rolls out (its bottom layer stays on the table while the sheet above moves away from the crease); success at the first settle: coverage 0.958, 28.2 mm, stretch 1.014, robot clearance 24 mm |

What did not work (branching lab on host C):
- stage B variants after the page turns (the released couple, its reverse, a left-half drag, a far-edge drag): pinches
  over the standing pleat are not admitted, or the sheet is dragged with its folds (0.58-0.61);
- the measured crease layouts on the far strip (drop-then-lay, or lift-drag-lower at its crease): the plastic crease keeps
  the strip shut when it hangs, and the sheet is dragged 5 cm (0.905 -> 0.906);
- the family's fold layout on the far-left corner before the slide: the corner drag turns the sheet ~20 deg into the left
  arm base's cover (goal true at 0.976, the release clearance never).

**The reference** (`scripts/robotangle/oracles/blanket-flatten-09-i01/`, standard.npz = privileged.npz):
- 2813 rows, end 6a9c4796da46cf47, sha256 4fbd7aba306e9720ab49ae45e3b2536e1f9902c9d6d909a2fb39ceb8637f5253; recorded by
  the route oracle in eai-roboarc:0.2.0 with the branch's planning bound in (host C RTX 4090);
- a re-record from the committed code: the same rows and tick ledger (only the file's wall-time meta differs);
- a re-record on 8beebf0b71 (the rate-limiter clamp 48409e39d6 included): the same rows and tick
  ledger, 734 s for the whole oracle run on a quiet host C RTX 4090 (6.8 % of 3 h);
- max arm joint step 0.0328 rad (every row one control step under the standard service's 0.04 rad);
- birth probe in eai-roboarc:0.2.0: instance_exact, assets_exact, c7e8fe0051099141;
- strict fresh replays (rcb_topogym replay --strict, two separate processes, eai-roboarc:0.2.0): success at row 2813 with
  the live thresholds, end 6a9c4796da46cf47 both; 3549 / 3555 s on a host C shared by ~15 other GPU processes. A paired
  07 replay in the same window took 2983 s (838.9 s on host A in wave 2): normalised ~17 min; at 07's per-row time on host A,
  2813 rows take ~25 min (13.8 % of the 3 h budget);
- nop 400 (stress observer, v2.13b, GPU): no success, CLEAN.
- the observer's replay of the reference (GPU, v2.13b): success, the same end, CLEAN;
- Harbor oracle 1 / nop 0 in both modes (rendering off, the GPU kept) on a local test image (eai-roboarc:0.2.0 plus this
  instance): privileged success x2 deterministic, standard live = replay, end 6a9c4796da46cf47;
- leak probe (the V2 audit): no finding (the generic `keyframes` runtime hit, as 07 / 08);
- stress (plan B, CPU 300): the cluster, the CPU twin (blanket_vbd_cpu_twin; the twin instance's scene re-pointed to the CPU
  solver), stress tools of 11371a74f1, eai-roboarc:0.2.0 SIF, seeds 91901-91934: 300 episodes (random, directed,
  mutate, cem, periodic, 60 each; 71,070 controls; prefix left out: a twin prefix episode replays ~1,400 GPU rows at
  ~9 s per control), nop CLEAN, 0 confirmed BREAKDOWN under v2.13b (and every earlier reading), 0 pops: PASS.

## References in the repository
(The flex references of 2026-10-10. For easy / medium they are superseded by the wave-2 ports above:
`oracles/blanket-flatten-easy-i01` became `blanket-flatten-07-i01`, and `oracles/blanket-flatten-medium-i01` became
`blanket-flatten-08-i01`, each with the new reference. The tier-named -i01 directories below are development names, not
included in this repository.)

| tier | `scripts/robotangle/oracles/` | rows | birth (state_sha) | end (state_sha) | privileged.npz sha256 |
|---|---|---|---|---|---|
| very_easy | `blanket-flatten-very-easy-i01/` | 1897 | 21441c0569cfbef9 | 2f3ed0fa25ffd525 | ffe810d4c9aec47c72a848f1fecaf2427ef192750d2715d41b5191c0c9e0a40b |
| easy | `blanket-flatten-easy-i01/` | 753 | f146812a31afc5c5 | 3ad1a55ce1dbc26a | 16ebf45e4e4e0bbb9bd7d57198ffc76aff390616dda581a0ab5cbb5600199ad2 |
| medium | `blanket-flatten-medium-i01/` | 1511 | 68c784b1288dc881 | b39cd3f0fb5ee97c | 89ca72c566eaf4b0b471097a2d9203a226a7456cda3e88c9ba007f7df04b3b9f |
| hard | (on the flex: blocked; on the GPU step: `blanket-flatten-09-i01/`, above) | | 3aacbcd368c4d449 | | |

standard.npz is the same file as privileged.npz (a MuJoCo standard oracle streams the reference's rows). Each passed
two fresh replays with the stress observer (criteria v2.8, b1ec43997): CLEAN, the same end state. The -i00 dirs are
the released Genesis references, untouched.

## Per tier (the -i01 births made on b1ec43997; references on b1ec43997 + the series)
| tier | birth (state_sha) | reference | status |
|---|---|---|---|
| very_easy | 21441c0569cfbef9 | `oracles/blanket-flatten-very-easy-i01` (run ve_F1): 1897 rows, 47.4 s | success: coverage 0.975, height 18.3 mm, stretch 1.015, robot clearance 10 mm; final 2f3ed0fa25ffd525. The corner route, two measured fold layouts, the base clearance slide (the sheet had come to rest 5.5 mm inside the right base's cover). Validator: 2 fresh replays CLEAN, equal; X - Wc -4.2 mJ; 2.0-2.1 s per control on a host loaded to 150/128 (65-67 min per observed run, near the 72 min limit) |
| easy | f146812a31afc5c5 | `oracles/blanket-flatten-easy-i01` (run easy_F1): 753 rows, 18.8 s | success: coverage 0.960, height 15.7 mm, stretch 1.013, robot clearance 43 mm; final 3ad1a55ce1dbc26a. The easy route unchanged. Validator: 2 fresh replays CLEAN, equal; X - Wc +3.0 mJ. Cross-CPU: Intel i9 workstation = AMD Zen 4 cloud host (76 checkpoints) |
| medium | 68c784b1288dc881 | `oracles/blanket-flatten-medium-i01` (run medium_F2): 1511 rows, 37.8 s | success: coverage 0.964, height 23.7 mm, stretch 1.015, robot clearance 40 mm; final b39cd3f0fb5ee97c. The route unchanged (coverage 0.871, the far strip tucked at y 0.45-0.52), then the measured slide (11.6 cm toward the robot: the tucked strip rolls out, 0.964); the measured crease finds nothing left. Validator: 2 fresh replays CLEAN, equal; X - Wc +0.4 mJ |
| hard | 3aacbcd368c4d449 | none: BLOCKED, flagged for step 3 | best measured re-plan 0.79 (below); the redesign's hard4b replaces this tier |

Before the switch to the exact diagonal (old physics; superseded): very_easy `ve_short2` (1348
rows, final 2d64ec9dadf010a9) and easy `easy_short1` (798 rows, 65dff236fc53b421) succeeded, two fresh replays
equal each. On the new physics ve_short2's rows still succeed when replayed (final 9f8493bff2ac2b11, twice).

## Open items
1. hard: ported on the GPU step as blanket-flatten-09-i01 (V3, above); its review was approved (2026-10-18), shipped
   in RoboARC 0.2.1.
2. Done: blanket in `mujoco_release.json` (45790f509; the entry under Birth).
3. The observer's cloth support (the validator's, before blanket's formal round): flex contacts through
   con.vert / con.elem (a flex contact's geom is -1); the cloth vertices as an articulated task entity (speeds, jumps,
   rule 6, a per-vertex tunnel check); the cost at 2-4k contacts per substep.
4. Cross-CPU on the other final references. easy on b1ec43997 passed: the standalone replay of its export (the compiled
   model with the exact diagonal, the start state, every step's controls and pin changes; `work/xcpu_export.py`,
   `work/xcpu_play.py`, mujoco 3.14.0 + numpy only) ends on the same state on the workstation's Intel i9 (AVX2) as the
   Session run on a cloud host's Zen 4: 76 checkpoints equal, final 4edcdf9f410da15c. very_easy's full Session replay on the
   workstation (no observer) also ends on its recorded state: 52.7 min.
5. The flex's interlocking collision surface and limp bending (above): not fixed; the replacing cloth solver should not
   inherit them.

## Development working files (not included in this repository)
- Instance sets: the -i01 births on b1ec43997 (with the settled birth sources and the unsettled instances), the released
  Genesis instances, the -i01 births on the old physics. Per reference run (`ve_F1`, ...): privileged.npz, summary.json,
  final.npz, oracle/oracle_report.json; newer runs also phases.jsonl and snaps.npz.
- Tools (`work/`): `settle_tier.sh <tier> <root>`,
  `make_mj_instance.sh <tier> <root> [birth source]`, `ref_blanket.py` (ROUTE_OVERRIDE=<route.json> for experiments),
  `replay_ref.py`, `reach_map.py`, `cover_gap.py`, `plot_cloth.py` / `plot_material.py` / `plot_section.py`,
  `books_check.py` + `books_table.py`, `grasp_geom.py`, `discrete_books.py`, `diag_cost.py`.
