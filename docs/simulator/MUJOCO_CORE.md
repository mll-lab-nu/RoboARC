# RoboTangle on MuJoCo: the shared core

RoboTangle's physics moves from Genesis to MuJoCo family by family. A family runs on MuJoCo when its instances say
`simulator.physics_engine = "mujoco"`. `scripts/roboarc/shared/rcb_topogym.py` (`TASKS`) chooses the engine
for new instances. Released Genesis instances keep their own frozen configuration and still load on Genesis.
Rendering does not change: agents keep seeing Nyx images (the render bridge, below).

## Where the engine boundary is

Everything above TopoGym's `Env` stays engine-free: the Task (its predicates read the world-state dict only), the
GraspManager, the Session and its rules (`rcb_topogym.py`), the standard service and both graders. `Env.reset` builds one
**World** and one **Robot**; for MuJoCo these are `topogym.world.mujoco.MujocoWorld` and `MujocoRobot`.

| Layer | Genesis | MuJoCo |
|---|---|---|
| World | `topogym/world/genesis_world.py` (removed in 1.1.0) | `topogym/world/mujoco/world.py` |
| Scene | `topogym/world/tabletop.py` (removed in 1.1.0) + `World.add_entities` | `topogym/world/mujoco/scene.py`: MJCF from the same declarations |
| Robot | `topogym/robot/joint_position.py` | `topogym/world/mujoco/robot.py`: the same class, on the scene's PD drives |
| Assisted grasp (`contact_weld_v1`) | `world/physics/grasp_rigid.py` | `topogym/world/mujoco/grasp.py`: one inactive weld per hand and target link |
| Family settings | the task's `physics_options` | `topogym/world/mujoco/profiles.py` |
| Entities seen by the tooling | Genesis `RigidEntity` | `topogym/world/mujoco/views.py`: the same attribute names |

### The World interface (what Env, Session and the grasp manager call)

- `World.from_tabletop(env_cfg, robot_cfg, *, physics_options, compute_backend, precision, seed, cameras, **engine)`
  - MuJoCo `engine` keywords: `scene` (a frozen scene directory, or None to build one), `work_dir`, `grasp_assist`,
    `profile`.
- `add_entities(specs)`, then `build()`, `reset_clock()`, `configure_grasp(robot, config)`.
- `advance()`: one World tick (`physics_dt_s`) = `solver_substeps` MuJoCo steps.
- `read_state()`: the per-tick world-state dict. Its schema is the Genesis one:
  - `bodies[entity]`: the entity's base link pose and velocities, plus `link_indices`;
  - `articulations[entity]`: `joints` (position / velocity) and `links` (pose / velocities);
  - `contacts`: `link_a`, `link_b`, `force`, `penetration`, `position`, `geom_a`, `geom_b`;
  - `grasp`, `assist_active`, `physics_step`, `sim_time_s`;
  - `particles`, `fem` and `ipc` are empty on MuJoCo.
- `capture_state()` / `restore_state(state)`: exact. They use `mj_getState(mjSTATE_INTEGRATION)`, plus the weld anchors
  (`model.eq_data`), the welds' slip windows, the grasp manager's state and the tick's readout cache.
- `physics_step`, `sim_time_s`, `physics_dt_s`, `entities`, `cameras` (empty until the render bridge attaches), `close()`.
- `contact_events` in `read_state()` (tasks whose entities declare `contact_roles`, `contacts.py`): GenesisWorld's
  schema-1 per-substep contact evidence, sampled after every MuJoCo step of the tick; part of the checkpoints.

### Readout timing (one tick)

- Kinematics (poses, velocities) come from the **end** of the tick.
- Contacts, contact forces and drive forces come from the tick's **last substep**, as Genesis reports them.
- A restored state reads exactly as it did when it was captured.

### Entity views (`views.py`)

The tooling and the tasks use this subset of the Genesis entity API, and the views provide it.

- **Entity:**
  - `name`, `links`, `joints`, `n_dofs`, `n_qs`, `base_link`, `get_link`, `get_joint`;
  - `get_qpos`, `get_dofs_velocity`, `get_dofs_control_force` (the drive's PD output after its clamp: the joint
    effort, a motor-current analogue);
  - `set_dofs_position`, `set_dofs_kp` / `kv` / `force_range`, `control_dofs_position`;
  - `get_AABB` / `get_vAABB`, `morph.fixed`;
  - `chain_fk()`: stateless forward kinematics on a private `MjData`;
  - `forward_kinematics(q)` / `inverse_kinematics(link, pos, quat, init_qpos, ...)`: Genesis' entity queries on that
    private `MjData`, so `topogym.solvers.genesis_kinematics` (GenesisFKSolver, GenesisIKSolver) and EE_ABS actions
    work on MuJoCo. The IK is Genesis 1.2.1's kernel (damped least squares, per-dof step clamp, joint ranges, float32
    parameters), one sample (no random restarts): on the ALOHA and the Panda it reproduces Genesis' solutions to
    1e-9 rad and picks the same branch (checked against Genesis by `tools/tests/test_mujoco_kinematics.py`, in RoboARC 1.0.0).
- **Link:** `name`, `idx` (the MuJoCo body id; contacts use it), `parent_idx`, `is_fixed`, `geoms` (collision),
  `vgeoms`, `get_pos` / `get_quat` / `get_vel` / `get_ang`.
- **Joint:** `name`, `qs_idx_local`, `dofs_idx_local`.
- **Geom:** `init_verts`, `init_faces`, `init_pos`, `init_quat`, `get_verts()`, `mesh.trimesh`.
- **Planning tools** read a few Genesis internals and tell the engines apart by `hasattr(env._world, "scene")`
  (`topogym.planning.oracle_path`, `horseshoe_ring.kinematics`): `MujocoWorld` must never define `scene`. On MuJoCo
  they take a geom's pose from `GeomView.world_pose()` and the geom pairs MuJoCo's collision filter admits from
  `views.collision_pairs(world, geoms)` (without the pairs touching in `qpos0`, as Genesis filters a robot's
  self-collision; both from the planning tools' swept-collision check, a84f4f509).

### Naming in the MJCF

- The robot keeps its own MJCF names (`left/waist`, `left/gripper_link`, ...). Its mount body is `robot`.
- Task links are bodies named `<entity>::<link>`. Their joints are `<entity>::<joint>`; a movable entity's free joint
  is `<entity>::root`.
- The drives are `<arm>/drive/<joint>`, and the grasp welds are `grasp/<hand>/<entity>/<link>`.
- `scene.json` maps every entity to its bodies and joints.

## The scene of an instance (`scene.py`)

The builder takes the same inputs as the Genesis scene:

- the tabletop configuration (table, ground plane, robot mount, gravity, timing);
- the robot's MJCF (`topogym.assets.robot.robot_asset`, the file Genesis loads);
- the robot description (controlled joints, `kp`, `kv`, force limits);
- the task's `EntitySpec`s after `finalize_spec`;
- the episode's grasp assistance.

It writes:

- the robot under its mount, as Genesis imports it, with:
  - `gravcomp` on every robot body (Genesis' `gravity_compensation=1`);
  - one `general` PD actuator per controlled DoF, with Genesis' `kp`, `kv` and force range;
  - the profile's joint armature;
  - the robot's internal linkage parts frictionless (`robot.py` `FRICTIONLESS_LINKS`: the Robotiq 2F-85's spring links
    and drivers): condim 1 with contact priority 1, so their own contact wins against any other geom. They are bars of
    the four-bar, not gripping surfaces. With Genesis' armature (0.1 kg m^2) on the four-bar's passive joints, the
    friction rows of such a link's contact were regularised by an approximate constraint diagonal far from the true one
    and returned energy (buzz G8 / G9, a spring link pressed on the static wire base: +0.67 / +3.68 J; G10, a driver on
    the wire: +3.27 J; BREAKDOWN). Frictionless: G8 / G9 / G10 violent but causal (X rise <= 0.12 J; G10 path-matched,
    the drivers switched frictionless after the spring links' last touch), every buzz and fiber reproducer
    non-BREAKDOWN, and no Panda reference changes. The followers carry the pads and the couplers hold what is gripped
    (buzz hard's wand rests on the right coupler, 0.6 N; a frictionless coupler fails that reference): both keep their
    friction. Rejected: armature 0 on the passive joints (clears them, but the four-bar then folds differently on
    contact and 9 of the 13 Panda references lose their grasp) and MuJoCo's exact constraint diagonal alone (G9 and G10
    still break down; since 2026-10-10 it is on as well, see "The exact constraint diagonal" under Contacts). Their
    priority makes their solref / solimp win too, so they keep the robot's task-contact class and never take
    `robot.self_contact`'s soft one (2026-10-10, F1): in fiber, which has that soft class, a spring link pinched into
    the cable at about 100 N sank 19.2 mm on 5 ms / .95 (the validator's FB1, BREAKDOWN). On the robot class (2.5 ms
    dr 1, .95 / .99) it stays at 1.0 mm. F1 changes only fiber's scenes: buzz and hedgehog build byte-identical
    scenes. Over the FB1 round's 40 fiber streams (21 corpus, 16 reproducers, 3 references) it gives 0 BREAKDOWN with
    every verdict unchanged; 30 end on the same state, and the 10 that move are those where a linkage part touches.
    The 4 fiber references keep their states. Frozen scenes keep their geoms: a family adopts this by building new
    instances;
  - where the profile's `robot` asks for it (Panda+Robotiq, fiber): the MJCF's own per-class contact parameters kept,
    contact priority dropped, the Robotiq pad collision boxes ending at the visual pad face (the spring links keep the
    MJCF's springref 2.62 since 2026-10-11: below, "The Robotiq's spring links");
  - the robot's meshes referenced through a token, never an absolute path: `@robot-assets@/` (TOPOGYM_ASSET_ROOT) or,
    for the Panda's meshes, `@genesis-assets@/` (the genesis package's asset layout). At load the Panda's paths resolve to
    its vendored copy, `robots/franka_emika_panda/` in the robot asset root (the same bytes; the genesis package only
    where that copy is absent; the image has no genesis package since 1.1.0), so frozen scenes keep their bytes;
  - where the profile's `robot.maxhullvert` asks for it, the robot's collision meshes capped at that many convex-hull
    vertices (below);
- task entities: `urdf` (any tree of fixed, revolute and prismatic joints), `mesh`, `box`, `sphere`, `cylinder`, and
  `mjcf` (`mjcf_entity.merge_mjcf_entity`: the authored articulated tree under `<entity>::` names, visual-only geoms
  dropped, every parent-child body pair excluded from contact as in Genesis).
  - URDF inertials are copied exactly.
  - Collision primitives stay primitives.
  - A collision mesh collides as its convex hull, or as CoACD hulls when the profile sets `coacd`. The hulls are cached
    by mesh bytes and settings in `$XDG_CACHE_HOME/topogym/mujoco-hulls`.
  - With the profile's `prism_partition`, a link whose collision meshes are convex prisms of one extrusion (the slab
    prisms of a cast plate) collides as an exact re-partition of the same solid into fat convex prisms
    (`prisms.py`): constrained Delaunay triangles of the slabs' union, merged greedily while the union stays convex.
    Scanline slabs are as thin as the closest two outline vertex heights (cast_keyhole hard: 0.002 mm), the sliver
    hulls of the contact policy's evidence. A link of several slab groups and other convex pieces (hedgehog's cages)
    re-partitions each group of prisms sharing a normal and both planes and keeps the other pieces as declared; needles
    the snap leaves at T-junctions join a neighbour. `prism_partition: {"fatten_m": w}` then re-cuts every piece thinner
    than w with a neighbour, exactly: the plane through an edge of their shared face is turned about it, and a split
    counts only when both halves are convex (their volumes add up within a 20 nm layer, the meshes' own rounding), so a
    wall facet sampled 0.16 mm from the next becomes two fat pieces.
  - Visual-only entities (the goal region) are static bodies without collision.
- the inactive grasp welds;
- `scene.json`, with sha256 of every file.

`make-instance` freezes the built directory as `instance_assets/mujoco/` and sets `simulator.mujoco_scene`. Every later
session loads those bytes and checks them against `scene.json`. The CPU-generated geometry (URDF primitive sizes, CoACD
hulls) is therefore never regenerated at run time. A frozen scene must match the declarations, with the same entities at
the same authored poses (to 1e-9), or the build refuses it.

## Contacts and collision detection

MuJoCo collides two convex shapes with GJK / EPA (MuJoCo's native CCD), and, since 3.x by default, with several contact
points per pair (`multiccd`). The validate agent's breakdown search (2026-10-09) found three failure classes, all
deep contacts with wrong depths or normals, each injecting energy. The settings that remove them:

- **The exact constraint diagonal, every model** (`MujocoWorld.build` sets `mjENBL_DIAGEXACT` on the loaded model,
  frozen scenes included; decided 2026-10-10). MuJoCo regularises each constraint row by
  R = (1 - d) / d x A_ii (d: the row's impedance) and by default estimates A_ii from the bodies' and dofs' inverse
  weights at qpos0. On our scenes that estimate was 0.0023-435x off per row (gpusearch, CPU): contact normals about 4x
  too stiff, grasp welds and connects up to 435x too soft. On the Robotiq's passive four-bar the mis-regularised rows
  returned energy: buzz G8-G10 (cleared by the frictionless links above), then G11 on a spring link's normal row
  (X rise 0.39 J, BREAKDOWN) and 96 milder cases on couplers and followers. With the flag each row is regularised by its
  true A_ii at the current state. What it changes:
  - Grasp welds hold closer and transmit more: claws very_easy's weld residual 0.87 -> 0.16 mm; at fiber's feed jam
    the weld's 20 ms mean goes from 19.9 to 62.7 N (fiber re-plans that feed below 32.5 N).
  - Contact dynamics change everywhere: references recorded before do not replay to the same states. Most reach their
    goal a few rows later than their recorded last row (re-record them: the rows, then hold rows until the episode
    ends); buzz's finish lands the wand elsewhere (re-planned). Instances whose birth runs physics (claws easy,
    medium) are re-made: their birth hashes change.
  - MJWarp 3.14 rejects the flag (`put_model`: NotImplementedError); the GPU screen ports it (gpusearch).
  - Evidence (the physics of 441b884e7 with the flag on and off; a cloud host `<work dir>/core/results/diag/`):
    - Stress: G1-G11, gpusearch's 23 known streams and B1 / B1b / B2 / B3 (26 streams), each in 2 fresh processes,
      read with criteria v2.8: 0 BREAKDOWN (24 VIOLENT, 2 CLEAN), every stream deterministic. G11's X rise goes from
      0.39 J (BREAKDOWN) to 0.036 J; the largest X rise is 0.065 J and the largest pair net work 0.017 J.
    - gpusearch's 96 coupler / follower streams (each pair's largest net work within one contact episode, fresh
      processes): with the flag every one stays below 0.05 J (largest 0.020 J); without it 3 exceed it (0.695, 0.151,
      0.084 J).
    - Cost per control step, the flag off and on run side by side:

      | scene | ms per control, off -> on |
      |---|---|
      | claws hard (3000 rows) | 95.0 -> 99.9 (+5 %) |
      | buzz hard (3000 rows) | 51.8 -> 57.5 (+11 %) |
      | fiber medium (whole reference, no observer) | 191.7 -> 214.2 (+11.8 %) |
      | knot hard (301 links, first 500 rows) | 953 -> 891 (-6 %) |
      | rope_threading hard (200 links, first 200 rows) | 182 -> 159 (-13 %) |
      | blanket very_easy / hard (80 holds at birth) | 520.6 -> 599.6 (+15 %) / 659.7 -> 678.3 (+3 %); whole runs equal |

- **Degenerate rows: on the exact diagonal nothing bounds them** (the general lesson, from blanket BL1, 2026-10-10).
  R_ii = (1 - d) / d x A_ii shrinks with A_ii. A contact whose normal the two bodies can barely move along
  (A_ii -> 0) gets R -> 0, and its force, aref / R, is bounded by nothing:
  - not by the contact class: every solref / solimp gives a force proportional to 1 / A_ii;
  - not by the drives' force caps: such rows push against each other, so the force is internal;
  - not by the solver: CG at 1000 iterations carried the same force.

  The approximate diagonal bounded them by the bodies' inverse weights (BL1 with the flag off: 42 N). **The fix pattern
  is to remove the geometry that makes the row, not to soften a class.** The clearest case is two bodies with almost no
  relative freedom, like the two fingers of one gripper on one slide axis. A finger and its own gripper are
  parent-filtered.
  - The case, BL1 (validator, blanket medium e30):
    - An arm folded onto its own base pressed one ALOHA finger closed: 50 N on its slide, 190 N at impact.
    - At 0.19 m/s per finger the menagerie pads crossed in one 2.5 ms step. The pads are three r 0.6 mm spheres per
      finger, 0.39 mm proud of the finger's hull.
    - A pad centre inside the other finger's hull took a side face's normal, almost perpendicular to the slides:
      A_ii 0.003-0.007 against 3.3, so R 1.5-3.4e-4.
    - Three such rows carried 15, 27 and 41 kN and locked the gripper (+104, +77, -172 N on the slides) for 80 ticks.
      CG hit its cap, and the cloth blew up.
    - The same rows are reachable in claws: in the validator's CL7 round, claws easy random2_e30 put 26.8 kN on one
      gripper's two fingers.
  - The rule (`robot.FINGER_PADS`, `robot.separate_finger_pads`; set by `MujocoWorld.build` on the loaded model, frozen
    scenes included):
    - Each gripper's pads and finger hulls get their own collision bits. A pad meets the other finger's pads, and every
      non-finger geom as before, but never the other finger's hull; hull meets hull.
    - At load the plan is checked against the original bits for every finger geom against every geom and flex. It is
      refused unless only those pad / hull pairs change (12 per two-gripper scene).
    - An empty close rests pad on pad (blanket 0.17 mm, claws 0.36 mm), short of any pad-hull contact (0.39 mm), so
      references stay bit-identical.
  - Path-matched alternatives on e30 (from row 105) all cleared the crossing by under 0.4 mm:

    | alternative | finger-finger peak | margin |
    |---|---|---|
    | exact diagonal off | 42 N | 0.36 mm short of crossing |
    | coupling at 5 ms | 35 N | 0.24 mm short |
    | robot-robot impedance .99 | 37 N | 0.24 mm short |
    | finger joint stop at 8.0 mm | 50 N | crossed; avoided the flip by a hair |
    | 1.25 ms timestep (blanket's run) | 23 N | 20 µm short |

    The rule needs no margin: with it e30 ends clean (47 N peak), and the pads may cross while the 27 mm hulls stop the
    fingers.
  - Evidence. Physics: the head (956c17ab8's code) with the rule. Knot and rope_threading ran on the rope agent's bound
    runtimes with the rule and re-bound instance copies. Results on a cloud host in `<work dir>/core/results/padA/`.

    | check | result |
    |---|---|
    | BL1 e30, blanket medium | BREAKDOWN at 40.7 kN -> CLEAN, finger peak 47 N, no degenerate rows |
    | references, each at its recorded success row and state | blanket 3 (2f3ed0fa, 3ad1a55c, b39cd3f0); claws 5 (0252d799, fefe0fbf, 30545aa2, 094d2865, f71af286); knot 5 (c74c1389, e1b29996, 9f634774, 658b8dca, e3398dbb); rope_threading 2 (519d6475, a394709f); keyhole 4 (601d09f4, 6f6214b3, 57ec3d2f, 92df286d), and hard unchanged (no success, 5d4e4f9f) |
    | claws corpus (35) and reproducers (19, C1 included), against CL7's records | 53 bit-identical. fb6d317d moves: an empty close reaches 0.47 mm pad on pad. CLEAN either way |
    | CL7's 38 stored candidates past 0.39 mm robot-robot depth, base vs rule | 29 bit-identical, 9 move. No verdict worsens. random2_e30 VIOLENT (26.8 kN) -> CLEAN; random2_e9 VIOLENT -> CLEAN |
    | blanket BL1_300's candidates, nop400 x 4, and its random job's first 8 episodes (easy, medium) | e30 as above; the four VIOLENT ones keep their gentle finger contacts (2.4-5.9 N) and verdicts; nop states and the 16 episodes identical |

  - The second case, the ALOHA's wrist (validator CL8t, 2026-10-10): each arm's `lower_forearm_link` against its own
    `gripper_base`, two joints apart (wrist angle, wrist rotate).
    - Their hulls, the forearm and the gripper's wrist mount, overlap only past wrist_angle 116.5 deg with
      wrist_rotate within about 15 deg of 180 deg: by up to 11.9 mm at the ViperX's 123 deg and 18.4 mm at the model's
      128 deg limit. Below 114.6 deg nothing touches at any rotation.
    - With only those two joints between them, the contact's normal can lie almost perpendicular to every motion they
      have. CL8t (the claws round on rule A, which passed) flagged it in two VIOLENT streams, a search candidate on
      medium and a corpus stream on easy: 2.2-2.4 kN on 0.09-0.12 mm, A_ii 0.005-0.008, in one substep.
    - The rule (`robot.SELF_CONTACT_EXCLUDES`): the pair is excluded from contact. `load_scene` adds the `<exclude>` at
      every load (`MujocoWorld.build`), frozen scenes included; the scene's files are unchanged. In that corner the
      wrist mount may now pass into the forearm.
    - A joint-range guard (wrist_angle <= 2.0 rad) instead would cut keyhole's, horseshoe_ring's and rope_threading's
      references: they command wrist_angle up to 127 deg, never with the wrist rotated near 180 deg.
    - Evidence (a cloud host `<work dir>/core/results/wrist/`; fresh replays, the exclusion on CL8t's tree, on the head and
      on the rope agent's bound runtimes with re-bound instance copies):

      | check | result |
      |---|---|
      | CL8t claws medium mutate2_e2 | VIOLENT (the wrist row, 2,383 N) -> CLEAN, no degenerate row |
      | CL8t claws corpus fdd7cf1e | VIOLENT (the wrist row, 2,188 N; a piece had left the workspace at tick 7825) -> VIOLENT for that exit only, no degenerate row |
      | references, each at its recorded success row and state | all 25 unchanged, the same success row and state hash: claws 5, blanket 3, knot 5 on the rope agent's bound runtime and 5 on the head (rule A included), rope_threading easy and hard, keyhole 5 (hard ends without success, as recorded). The Panda families have no such pair; horseshoe_ring has no MuJoCo reference yet |

- **A cylinder the robot can touch is a convex prism** (`SceneBuilder._cylinder_prisms`, at every instance build,
  2026-10-12): every cylinder geom outside the robot that a robot geom can collide with (static or task; collision
  bits with default classes resolved) is built as a convex prism of 32 facets, its vertices on the radius (vertex k at
  angle 2 pi k / 32 from the geom's local x, the axis its local z; a `fromto` cylinder takes MuJoCo's own frame,
  `mju_quatZ2Vec`), the same pose and the cylinder's mass. The scene's manifest lists them (`cylinder_prisms`).
  - Why: MuJoCo has no analytic cylinder test against a convex hull or another cylinder. Its GJK / EPA (with
    multiccd) returned, near a flat cap's rim, the right depth with a sign-flipped or non-minimal normal, and the row
    pushed nothing: knot easy_v8 w1_e10 (a finger 11 mm into a post, 0 N for 15 ms) and fiber FB3 (the sheath's cap
    against a ring segment, 5-14 mm witnesses). At w1_e10's poses, jittered (knot-stress): as built 1-31 % wrong
    normals by depth, the 32-facet prism 0-0.3 %, a capsule 0, the cylinder with multiccd off 0.
  - Frozen scenes keep their cylinders (a compiled model cannot change a geom's type): the released fiber -i01
    references and their step-2 pass stand as recorded; a re-made instance gets prisms.
  - Audit on the head (core `work/fb3/cyl_audit.py`, a cloud host `results/cylaudit`): robot-touchable cylinders exist only in
    fiber (the fixed terminal's gland r 18 mm, the six trunk shells r 12 mm, the ring's 64 tube segments r 4 mm) and in
    the knot redesign's posts; claws, keyhole, hedgehog, buzz, blanket, rope_threading and knot -i01 have none, and their
    built scenes are bit-identical with and without the rule (very_easy of claws, keyhole, hedgehog, buzz: the same
    scene.xml and birth). Random contacts against fiber's cylinders gave 0.05-0.2 % wrong normals (0-3 and 3-10 mm),
    but random poses under-detect this failure: on a synthetic 12 mm post the ALOHA's finger meshes flipped 0-1.2 % at
    random against 5-31 % around w1_e10's real approach, so a low random rate is no proof; the rule removes the class.
- **Converged convex collision, every family** (`profiles._COLLISION`, MuJoCo `<option>`): `ccd_iterations` 500,
  `ccd_tolerance` 1e-10. At the defaults (35, 1e-6) thin and curved shapes got wrong depths: a 7.56 mm claws bar passed
  through another, a fiber contact read 130 mm. No measurable cost per control step.
- **Small robot collision hulls** (`robot.maxhullvert`, claws 64): the ALOHA's gripper hulls had 1911 and 7003
  vertices, its fingers 346 / 348; on them multi-point CCD reported bogus deep contacts (robot self-collision +65 J, a
  gripper driven 21 mm into a claw). Capped at 64 the hulls stay inside the full ones within 0.1-2.0 mm (fingers 0.6 /
  1.0 mm); at 128 the failures come back. The visual meshes, and so the frames, are untouched. Disabling `multiccd`
  instead also removes them on the full hulls (61 -> 4 of 80 reproducers), but leaves single-point contacts on flat faces.
  The Panda's arm hulls (102-152 vertices) are not a known source; capping them at 64 moves their surface by up to
  21 mm (link1), so a family that caps a Panda+Robotiq lists the arm meshes in `robot.maxhullvert_skip`.
- **Robot-robot contacts** (`robot.self_contact`; claws: `profiles.SOFT_CONTACT`, below): the robot's collision
  geoms carry the class with `solmix` 0, so a contact between two robot geoms (self-collision, the gripper against its
  own base or shoulder, the two arms) uses it, while a robot geom touching anything else takes the other geom's
  parameters. Robot links need no task precision; with the task's nearly hard contact, an arm folded onto itself at its
  torque caps created energy (the validator's M6: +22 J at 60-114 kN, 0.03-3.6 mm deep). The grasp pads
  (`own_contact` geoms, the Robotiq's) stay out of it, with their own class and `solmix` 1, so their grasps of task
  bodies keep the mean of both classes, unless `robot.self_contact_pads`.
- **The ALOHA's finger coupling** (`robot.coupling`; claws: `profiles.ALOHA_COUPLING`, `solref` 1.25 ms, dampratio 2,
  `solimp` .9 .95): the gripper drive moves one finger and the MJCF's joint equality makes the other follow, at
  MuJoCo's default 20 ms. That soft coupling created energy against the driven finger: a gripper command switched
  between 0.021 and 0.037 m every control made the left coupling deliver +0.21 J (+0.32 J with the hand in free
  space), and claws easier / medium broke down (the validator's E1, 0.13 J). The real ViperX gripper moves both
  fingers from one servo through a rigid linkage; at 2 dt, overdamped, the coupling only dissipates (-0.02 / -0.03 J
  on the same rows). Every ALOHA family has the same coupling: cast_keyhole and horseshoe_ring adopt it by adding
  `coupling=ALOHA_COUPLING` to their robot settings.
- **Static contacts** (`static_contact`, opt-in per family) of static bodies (no joint between them and the world: the
  table, the ground, task links fixed to the world):
  - `soft`: a class (`profiles.SOFT_CONTACT`) the static bodies carry with `solmix` 0, so a robot geom with
    `robot.self_contact` touching them gets it (the mean of the two classes; the class itself when both are the same),
    while a grasp pad keeps its own class and a task body its own parameters: claws' pieces on the table stay nearly
    hard. Claws opts in (the table and the ground).
  - `keep: {entity: reason}`: static bodies that keep their own (task) parameters and win every contact with them
    (`solmix` 1e9): thin static geometry. A key is an fnmatch pattern on entity names (rope_threading: `R[0-9]*`, its
    rings, stands and bases, two or three per tier). `soft: None`: every static body keeps its parameters and wins (buzz).
  - The rule: a contact must keep its penetration well below half the thinner body's thickness, where the normal flips
    and the contact starts creating energy. Thick bodies may be soft; thin ones keep a stiff contact. The validator's
    B1 (buzz: a 3.5 g Robotiq pad struck into the 1.47 mm-radius fixed wire, 121 kN) came from the mean of the pad's
    4 ms class and the wire's: soft enough for the pad's box face to pass the wire's axis (2.0 mm deep). With the
    wire winning, every robot hull stays within 0.4 mm and B1 / B1b are violent only, with no energy created; with any
    soft class on the wire a hull reaches 2-12 mm and they break down.
- **Who wins** (MuJoCo mixes `solref` / `solimp` by `solmix`: a `solmix`-0 geom takes the other's, two `solmix`-0
  geoms their mean, 1e9 outweighs 1; friction and condim are always the larger of the two):

  | contact | robot geom (`self_contact`, solmix 0) | grasp pad (`own_contact`, 1) | task body (1) |
  |---|---|---|---|
  | robot geom (solmix 0) | the robot class | the pad's class | the task body's |
  | grasp pad (1) | the pad's class | the pad's class | the mean of pad and task |
  | soft static (0) | the soft class (with the robot's: their mean) | the pad's class | the task body's |
  | kept / winning static (1e9) | the static's | the static's | the static's (the task's own, when both are the family's) |
  | static without `static_contact` (1) | the static's | the mean of pad and static | the mean of task and static |
  | task body (1) | the task body's | the mean of pad and task | the mean of the two |

  Without `robot.self_contact` the robot's geoms have `solmix` 1 and take the mean with everything (Genesis' rule).
- **The soft class** (`profiles.SOFT_CONTACT`: `solref` 5 ms, dampratio 2, `solimp` .95 .95 since 2026-10-11, .9 .95 before; 2026-10-10 sweep, one class
  on robot-robot and robot-static contacts alike, the pads included, the clamp on; BRK: a breakdown by the validator's
  criteria; fiber and buzz on their own profiles plus the class):

  | class | claws, 97 reproducers: BRK / violent / robot depth / largest force | claws M6 | fiber F1 / F2 | buzz B1 / B1b |
  |---|---|---|---|---|
  | none (robot-robot 20 ms on claws) | 0 / 38 / 5.1 mm / 4.9 kN | clean | BRK 14.6 J / BRK 854 J | BRK 214 J / BRK 13 J |
  | 20 ms, dampratio 1 (MuJoCo's default) | 0 / 25 / 8.5 mm / 1.3 kN | clean | BRK, 10.8 mm deep / violent | BRK, 10.4 mm deep / BRK 8 J |
  | 10 ms, 1 | | clean | BRK 7.5 J / violent | BRK 675 J / clean |
  | 5 ms, 1 | 2 / 25 / 3.0 mm / 5.0 kN | BRK 3.3 J | violent / BRK 296 J | BRK 115 J / violent |
  | **5 ms, 2** | **0 / 26 / 2.75 mm / 1.4 kN** | clean | violent / violent | BRK 51 J / violent |
  | 2.5 ms, 2 | | BRK 5.0 J | violent / violent | BRK 38 J / violent |

  MuJoCo's default lets a robot sink 10-12 mm into thin fixed geometry (a wire, a cradle plate); at dampratio 1 the
  stiffer classes break down again. On claws' 42-stream corpus the 5 ms / dampratio-2 class keeps 0 flags (deepest
  contact 2.11 -> 1.11 mm, largest force 1789 -> 1074 N) and the references keep their state hashes. Buzz's wire is
  thin: it keeps its own contact (B1 above). With the pads kept out of the robot class (the default) and the statics'
  rule as above: fiber's F1 / F2 violent (robot contacts <= 1.75 mm deep); buzz with `soft: None` builds the same model
  as its `static_wins` build (every geom parameter equal, 5 tiers), B1 / B1b violent at 10.2 / 3.2 kN, 0.43 / 0.33 mm
  deep, no energy created.
- **The soft class's impedance** (2026-10-11): `solimp` .95 .95, a constant impedance, instead of .9 rising to .95 over
  the first 1 mm (`profiles.SOFT_CONTACT`; `SOFT_CONTACT_V1` the old class). Frozen scenes carry the old class;
  `MujocoWorld.build` migrates them at load (`scene.migrate_contacts`, with `static_contact.keep` below), so every
  instance runs the current class.
  - The case: an arm driven into a static fixture holds several coupled soft rows at once (a jam), and the rows give back
    more than they took: gpusearch's rope_threading hard kt1_3 e116 (the left wrist on ring R3 at 1.15 m/s with the
    gripper base on it from another side, 7 rows, normal-row A_ii 0.24 -> 0.16, 2.4 mm deep; the pair +2.20 J, X +1.44 J
    in 0.1 s), fiber easier FB5b e131 (the Robotiq's spring link on the cable cradle, +0.25 J) and knot easy_v8 e245 (a
    finger rammed into a post by the shoulder at its cap, 14 mm deep, +0.39 J pair, X +0.99 J). Not a witness flip, not
    the exact diagonal (the approximate diagonal leaves e116 a breakdown); a single hit stays dissipative in every class
    (a passive arm swung into a post: 60 hits, all lose energy).
  - With a depth-dependent impedance the coupled rows' forces are not those of one potential, so a jam's loading and
    unloading paths differ; a constant impedance removes that term. It does not remove a jam's stored energy where the
    effective inertia is large (e116: the wrist rows still return 4 J of 5.8 J): rope_threading's rings, thin static
    tubes, keep the task's nearly hard contact against the robot (`static_contact.keep` `R[0-9]*`), as buzz's wire.
    The rope meets them exactly as before: kept geoms win with `solmix` 2^60 (`scene.KEEP_SOLMIX`), whose weight against
    the rope's `solmix` 1 rounds to exactly 1, and the rings' own parameters are the rope's (rope_threading easy and hard
    -i01 end on the same states with and without the rule).
  - Candidates, each on every soft geom from row 0 (a cloud host `<work dir>/core/results/softsw*`, the cluster runs `core_sw_*`,
    `core_s_*`; tools `core/work/springref/sr_replay.py --soft`, `core/work/softsw/soft_wrap.py`):

    | stream | as built | constant impedance (.95 .95) | 2.5 ms | constant + rings kept |
    |---|---|---|---|---|
    | rope hard e116 | BREAKDOWN | BREAKDOWN | VIOLENT | VIOLENT (6.4 kN, causal) |
    | fiber easier FB5b e131 | BREAKDOWN | CLEAN | VIOLENT | CLEAN (this commit) |
    | knot easy_v8 e245 | BREAKDOWN | CLEAN (4.4 mm) | BREAKDOWN, worse (+10.3 J, 14.3 mm) | (no rings) |
    | rope e8 / e216 / e144 / e138 (pops, a jump) | BREAKDOWN x4 | VIOLENT, VIOLENT, BREAKDOWN, CLEAN | BREAKDOWN, VIOLENT, VIOLENT, BREAKDOWN | VIOLENT, VIOLENT, BREAKDOWN (a 3.2 m/s pop kick, the rope's), CLEAN |
    | claws CL6 periodic_e38 | CLEAN | CLEAN | CLEAN | |
    | short searches, 80-120 episodes per family (buzz, claws, claws redesign, fiber, hedgehog, keyhole, knot, rope) | 0 BREAKDOWN | 0 BREAKDOWN, VIOLENT counts unchanged (fiber 19 -> 23) | 0 BREAKDOWN, VIOLENT up (fiber 19 -> 47, rope 0 -> 6) | |

  - The references (fresh replays against the same replay as built): bit-identical: claws -i01 x5, keyhole x5, fiber
    -i01 x4, hedgehog x2, buzz very_easy, knot easy, claws redesign easier / easy / medium. Moved, success kept (to
    re-record): knot -i01 very_easy, easier, medium, hard, the knot redesign v7 very_easy, rope_threading easy and
    medium, blanket very_easy / easy / medium, claws redesign very_easy and hard. rope_threading hard -i01 loses its
    success (to re-plan from its route).
- **Claws contacts** (`profiles._RIGID`): overdamped (`solref` 1.25 ms, dampratio 2) and nearly hard (`solimp` dmax .99).
  A light piece pressed by kN drive forces sinks about (1 - d) / d * timeconst^2 * F / m: five times less at .99 than at
  .95. The time constant stays 2 dt: the claws are thin (medium bars 7.56 mm, hard tubes in 1.7 mm CoACD slices) and a
  softer contact let abuse sink them 7.5 mm (2.5 ms: 4 of 95 reproducers broke down; 1.25 ms: 0 of 95).
- **Air drag on light links** (profile `drag`: `{links: <fnmatch pattern on body names>, k}`, `MujocoWorld._apply_drag`):
  once per tick, before its substeps, every named moving body gets f = -k |v| v at its centre of mass in `xfrc_applied`
  (numpy, a function of the tick's start state, so replays stay deterministic). Quadratic air drag is k = 0.5 rho Cd A;
  for a 2.4 mm x 8 mm fibre link in air (rho 1.204 kg/m3, Cd 1.2) k = 1.4e-5 N s^2/m^2. It matters only for gram-scale
  links whipped to tens of m/s, which otherwise reach speeds no contact resolves at 0.625 ms steps; at legitimate speeds it
  is negligible. MuJoCo's own fluid model (`opt.density` / `viscosity`, or `fluidshape` per geom) was 4-7x slower per
  control step: implicitfast adds its velocity derivatives, which span the whole cable tree.
- **Two-way VBD cloth coupling** (the blanket profiles' `cloth.two_way`: on in `blanket_vbd_gpu`, off in `blanket_vbd`; `cloth_coupling.py`, 2026-10-12):
  the outside cloth's contact wrench on each robot body goes into `xfrc_applied` before each cloth step's MuJoCo steps
  and is held over them, linearly implicit in the robot's motion (the bodies' mobility J (M + h D)^-1 J^T at the MuJoCo
  step h, the cloth step dt), capped at 2x each actuated joint's torque limit. It and the pins' reaction share
  `ClothVBD._xfrc` (subtracted and re-added every cloth step), so `capture_state` carries them through
  `mjSTATE_INTEGRATION`'s `xfrc_applied` and the cloth's checkpoint. mj_step leaves mjData's kinematics at the step's
  start: the coupling reads its poses, Jacobians and mass matrix from a private MjData (mj_kinematics, mj_comPos, mj_crb
  of the current qpos). Details and measurements: `families/blanket_flatten.md`, "Two-way coupling".
- **Joint attribute rules** (profile `joints`: `[{match: <fnmatch pattern on joint names>, <MJCF joint attribute>:
  value}]`, `SceneBuilder._joint_rules`): applied to the scene's joints in order; a rule matching no joint is an error.
- **An MJCF entity's contact attributes** (profile `mjcf`: `{<entity>: {contact: {<geom attribute>: value}}}`,
  `mjcf_entity.merge_mjcf_entity`): set on the entity's default classes and on its geoms that set them themselves, so a
  family can change, for example, an articulated cable's own `solref` without editing its asset.
- **Explicit contact pairs** (profile `pairs`, `SceneBuilder._pairs`): one contact between two named groups of task
  bodies with its own parameters, where per-geom classes and `solmix` can only mix. An entry is
  `{bodies: [[<fnmatch patterns on body names>], [<patterns>]], condim, friction (5 values), solref, solimp, margin,
  gap}` (optional `solreffriction`, `reason`), all of them required: a MuJoCo `<pair>` left at its defaults would get
  20 ms. Fiber's use: the sheath against the ring, stiffer than the mixed contact (`fiber_bundle::trunk_*` x
  `bundling_ring::ring`).
  - Every geom of a body the first list matches meets every geom of a body the second matches through one `<pair>`
    with exactly these parameters (no mixing, no priority), and one `<exclude>` per body pair drops the automatic
    contacts it replaces.
  - Only contacts MuJoCo would make become pairs: body pairs its filter skips (one weld body, weld parent and child)
    or already excluded, and geom pairs whose `contype` / `conaffinity` (default classes resolved) never meet, stay as
    they are. On fiber easier (b19 scene) the 1,536 pairs of trunk x ring, and the 4,096 with the lead added (its first
    link, welded to the world like the ring, filtered), equal MuJoCo's own filter on the unpaired model geom for geom,
    and the geoms' sizes, poses, bits and contact classes, the masses and the joints are unchanged.
  - A matched body's unnamed geoms are named `<body>#<index among its geoms>` (a `<pair>` names its geoms; the ring's
    128 were unnamed). A pattern that matches no task body, a body in both lists, or an entry without any pair is an
    error. Robot bodies are never matched: their contacts are the robot classes' and the collision bit plans applied at
    load (`separate_finger_pads`), which a `<pair>` would bypass.
  - The pairs are written into `scene.xml` at build, so a frozen scene carries them; `scene.json` `pairs` counts them
    per entry. A scene without the field builds byte-identical.
- **Speed clamp on MJCF chain links** (profile `speed_clamp`: `{links: <fnmatch pattern on body names>, v_m_s,
  w_rad_s, rate}`, `MujocoWorld._clamp_apply`): a rope's clamp (`rope_chain.RopeChain._clamp`) for links that hang on
  the joints of one MJCF tree instead of being free bodies.
  - Before every MuJoCo step, a named link whose centre-of-mass speed exceeds `v_m_s` gets the force
    -rate m / dt (1 - v_m_s / |v|) v at its centre of mass, and one whose spin exceeds `w_rad_s` the couple
    -rate I / dt (1 - w_rad_s / |w|) w. Both oppose the link's motion, so they only remove energy. They go through the
    tree's Jacobian (`mj_applyFT`) into `qfrc_applied`, where the stress observer books their work as dissipation.
  - While no link exceeds a threshold it writes nothing, so a run that never reaches them is bit-identical to one
    without the field; `MujocoWorld.clamp_stats` counts the activations (the keys of `RopeChain.clamp_stats`).
  - It is read from the family's live profile at load, frozen scenes included (as a rope's clamp), so frozen births
    stand. First user: orbits (`families/orbits.md`).
- **Elastoplastic joints** (profile `plastic`: `{joints: <fnmatch pattern on joint names>, yield_angle_deg}`,
  `MujocoWorld._plastic_return`): each named ball joint has a stiffness (set with `joints`) whose spring pulls toward
  `model.qpos_spring`.
  - Up to the yield angle the joint is elastic. Past it, the return mapping moves the rest orientation along, to exactly
    the yield angle behind the joint about the same axis. A joint bent past its yield torque (stiffness x yield angle)
    therefore stays bent, with at most the yield angle of elastic give, and nothing creeps below the yield.
  - It runs before every MuJoCo step, not once per tick. At a whip's 100 rad/s a joint turns 3.6 deg in a step (14 deg in
    a tick), and that is how far the spring can overshoot its yield torque before the next return.
  - The arithmetic is quaternion products, sums and square roots only, so every host computes the same bits.
  - The rest orientations start at the birth pose and are state: `capture_state` / `restore_state` carry them, and
    `read_state` reports them (`joints[...]["rest"]`), so the Session's state hash covers them. A restore that misses
    them gives a different trajectory (checked).
  - Cost: 11-13 us per step when nothing yields, 0.2-0.8 % of a control step.
  - A family without the option builds and steps as before.
- **Fiber** (`profiles._FIBER`, `fiber_v5`; `fiber_v4` before the Robotiq springref, 2026-10-11): the cable as iron wire (elastoplastic joints, 0.4 deg), its contacts
  (the cable's own overdamped) and its re-planned references: `docs/simulator/families/fiber_bundling.md`.

Evidence: the validate agent's 95 claws reproducers (its S1 / S2 searches and corpus), each in a fresh Session under its
observer (energy balance, per-substep depth, causality): shipped 61 breakdowns, this policy 0; the deepest task contact
under abuse 21.5 -> 3.9 mm; per-control time unchanged.

The MuJoCo version is pinned to the image's own, **mujoco 3.14.0** (pinned in `source/images/roboarc/runtime.lock`): every instance,
reference, grade and sweep runs it. A frozen scene records the version it was built with (`scene.json`).

## Robot limits (clamp, never fail)

`topogym/world/mujoco/robot.py` `JOINT_SPEED_LIMITS_RAD_S` holds a robot's arm joint speed limits (Franka Panda
datasheet: 2.175 rad/s on joints 1-4, 2.61 rad/s on joints 5-7; ALOHA: the Interbotix ViperX URDF's pi rad/s on all six
arm joints). `rcb_topogym.Session.step` then limits how fast the commanded arm joint targets move: a longer step is
shortened along its own direction, before the table guard, rounded to float32 like every recorded row (a replayed row
never triggers it again). Each limited joint is reported in
`Session.last_step_meta["limits"]` (`{"entity": joint, "limit": "joint_velocity", "cap": rad/s}`), which the standard
service turns into its robot-only `clipped[]` notice. Torque limits are the drives' force range from the robot
description (Panda: 87 / 12 N m, the datasheet values of Menagerie's `panda.xml`); the Session's torque caps
(`ARM_TORQUE_CAPS_NM`) still lower them, and so does the robot's own `JOINT_TORQUE_LIMITS_NM` on MuJoCo (the smaller of
the two wins). ALOHA: its ViperX 300 S servos at their 14.8 V stall torque, 12.9 / 25.8 / 25.8 / 12.9 / 12.9 / 4.8 N m
(Dynamixel XM540-W270, doubled on the dual-servo shoulder and elbow; XM430-W350 on the wrist rotate). The Session's caps
alone let an ALOHA hand push about 3x a real one (520 vs 180 N along the reach); the claws references need at most
6.5 N m on any joint and replay unchanged. The cap is MuJoCo's only: Genesis' ALOHA families keep the Session's caps,
which their rope pinch needs. A drive at its cap shows in the existing forcing evidence; `robot_spec`'s `effort_limit`
states the cap.

**Cartesian speed** (`robot.py` `CARTESIAN_SPEED_LIMITS`, Panda only): after the joint speed limit, `Session.step`
shortens each arm's commanded joint step the same way until its TCP (the description's grasp centre) moves at most
1.7 m/s and turns at most 2.5 rad/s over the control step: the chord between the commanded TCP poses before and after the
step (forward kinematics on a private `MjData`), re-checked after shortening, with a 1e-4 tolerance so a float32-rounded
row never re-triggers. These are Franka's published limits on the commanded end-effector pose (Franka Control
Interface documentation, "Control Interface Specification and Robot Limits", Limits for Franka Emika Robot (FER),
Cartesian-space limits: https://frankarobotics.github.io/docs/robot_specifications.html). The joint limits alone let a
Robotiq pad reach 3.3 m/s; a pad crosses buzz's 1.47 mm wire's axis within one 0.625 ms substep above 2.35 m/s (the
validator's B2, 8.75 J). Bounding the TCP keeps the pads, 5-6 cm from it, near 1.85 m/s; bounding the flange (~0.26 m
from the pads) would not. Notices: `{"entity": joint, "limit": "tcp_velocity", "cap": m/s}` for each joint of the
shortened step (the standard service shows "<joint> command limited"). ALOHA: no published Cartesian limit, no clamp.

**Command lead** (`robot.py` `COMMAND_LEAD_BOUND`, Panda only; `Session._lead_bound`): the speed limits bound the
commanded targets, not the motion. A Panda drive (kv / kp = 0.1 s, four control periods) that is blocked or lags keeps
the command's offset and releases it as a catch-up well past the commanded speed: buzz B3 and the GPU search's G5-G7
reached 2.5-4.8 m/s at the TCP behind a 1.7 m/s command and drove a finger through the 1.47 mm wire (61-121 kN normal,
BREAKDOWN). After both speed limits `Session.step` therefore shortens each arm's step further, along its own direction,
until
- every arm joint's target leads the measured joint by at most cap / kp + v_max (kv / kp + T): the offset at which its
  drive still presses with the full torque cap through a whole control period while moving at its speed limit (0.283 /
  0.286 / 0.332 rad on joints 1-2 / 3-4 / 5-7). A lead of cap / kp (6-14 mrad) already saturates a Panda drive, so no
  push or grip loses force, and a pinned arm still presses at its caps;
- the commanded TCP pose is within v_max (kv / kp + T / 2) = 0.191 m and w_max (kv / kp + T / 2) = 0.281 rad of the
  measured one: the steady lag of a Panda moving at its Cartesian limit behind a held (zero-order) target, so a released
  arm catches up no faster than that limit. At a steady commanded 1.7 m/s the TCP lags 0.190 m and its speed ripples up
  to 1.9 % above the command between control steps (B2: 1.73 m/s, the pad still 23 % under the wire's tunnelling speed):
  the drives' own response to a held target, kept.

A target already further ahead (a knocked arm) is never pulled back, only kept from moving further out, so a limited row
replays unchanged. Notices: those of the limits the bounds enforce (`joint_velocity`, `tcp_velocity`). Evidence
(2026-10-10, 217a21717, CPU, stress criteria v2.5; the clamp alone -> with the bound; normal wire contact force):

| stream | verdict | reached TCP m/s | wire contact |
|---|---|---|---|
| B3 (validator round B14) | BREAKDOWN -> VIOLENT | 2.14 -> 1.61 | 11.3 kN at 1.06 mm -> 2.5 kN at 0.61 mm |
| G5 (GPU search top004) | BREAKDOWN -> VIOLENT | 2.52 -> 1.03 | 108 kN at 1.48 mm -> none |
| G6 (top000) | BREAKDOWN -> VIOLENT | 3.09 -> 0.96 | 121 kN at 1.92 mm -> none |
| G7 (top023) | BREAKDOWN -> VIOLENT | 4.78 -> 1.35 | 61 kN at 1.51 mm -> none |
| G46, its full 295 rows | VIOLENT -> CLEAN | 3.00 -> 1.52 | 1.9 -> 0.4 kN |
| G2, G3 | VIOLENT -> VIOLENT | 4.37, 3.60 -> 1.67, 1.16 | |
| B2, G1 | VIOLENT -> VIOLENT | 1.73, 1.57 (unchanged) | |

The VIOLENT that remain are robot-static impacts at commanded speeds within the limits (a finger on the wire base plate
early in the swing, 4.1-9.0 kN; a finger on the wire at up to 1.73 m/s). Every limited stream replays bit-identically
in a fresh process, and the bound never fires on a Panda reference (all 13 bit-identical: buzz very_easy / easier /
easy / hard, hedgehog's five, fiber very_easy / easier / easy / medium). Rejected forms: a lead of v_max T (the drives lag four control periods, so
free motion stops near a quarter of the limit: 0.39-0.41 m/s on B2, G1-G3 and G46), and projecting the target back
toward the measured state (a recorded row is projected again from another point on replay: live != replay). Neither a
gripper speed limit nor smaller substeps is needed: the gripper command is not involved (relative to the hand the
fingers move at most 0.035 m/s in the references and 0.12 m/s in the streams above with the bound, within the Robotiq
2F-85's finger speed of 20-150 mm/s, Instruction Manual 6.2), and 8 substeps leave B3 at 2.58 m/s. ALOHA: its command
path is unchanged; its servos cap their own catch-up. Over the 448 VIOLENT candidates of claws' final acceptance round
(C5) 374 take a joint past the URDF's pi rad/s, the median candidate to 1.125 x, the fastest to 1.285 x (4.04 rad/s);
every one past 1.1 x is a wrist rotate (313, 305 of them after a catch-up lead), at about its cap over its damping
(4.8 / 1.2 = 4.0 rad/s), below the XM430-W350's no-load speed (57 rpm = 5.97 rad/s at 14.8 V, ROBOTIS e-manual). The
other five joints never reach pi (at most 0.94 x, the wrist angle; the XM540-W270's no-load speed is 3.87 rad/s), and
the claws references stay at or below 0.19 x pi (all five keep their hashes).

**The grasp slips** (`grasp.WELD_SLIP_FORCE_N`, per robot): an assisted grasp whose weld has to pull or push its piece
with more than the robot's slip force, on average over 20 ms (`grasp.WELD_SLIP_WINDOW_S`), releases like any lost
grasp, reason `grasp_slipped`; the hand grips again after opening and closing. It never ends the episode, and the agent
sees it in the existing grasp state (`object_in_hand` turns None). A real two-finger grasp holds about the friction its
grip allows, 2 * mu * grip, and each value stays at least twice what its references need:
- ALOHA 60 N: the model's finger drive presses with 35 N at mu 1 (70 N); the claws references need at most 29 N.
- Robotiq 2F-85 (Panda) 65 N: the real gripper grips with 20-235 N; 2 * mu 0.65 (its pads) * 50 N, in the lower part of
  that range. The Panda references need at most 31.1 N as a 20 ms mean:
  - fiber, with the cable as wire, on the exact diagonal: very_easy 19.2, easier 19.2, easy 19.0, medium 19.4.
    The peak comes in a comb press; the feeds stay at or below 17.2.
  - hedgehog: 16.7.
  The force was 40 N until 2026-10-09, when fiber's wire sheath left it only 1.3x its need.
An arm can no longer drive a welded piece into another with hundreds of newtons (the validator's 436 N squeeze).

The window: the weld's translational constraint force (its first three constraint rows) after every substep, the last
20 ms of them (32 at claws' 0.625 ms substeps), zeros before the grasp began. The grasp slips once their mean exceeds
the slip force, checked after every substep and released at the end of the tick. A force held above the slip force for
20 ms always slips; a brief impact slips only when its impulse over the window exceeds slip force x 20 ms (1.2 N s on
the ALOHA). The windows are deterministic and part of the world's checkpoints; like the solver's warm start they are not
part of the state hash. Why a window: judged on one substep, the slip dropped cast_keyhole's assembly at its regrasp
(1.07 kg: the left hand holds the gold with the silver interlocked). When the right hand opens, the silver settles a
few mm into its slot, and the left weld goes from 11.7 N to 25-57 N for ~30 ms, with one 2.5 ms tick at 70.3 N (54.5 N
on keyhole_v2); its mean over any 20 ms stays under ~40 N. A real grip rides out such a brief load and slips under a
sustained one.

**The Robotiq's spring links** (`robot.SPRING_REFS`, set by `MujocoWorld.build` on every loaded model, frozen scenes
included; 2026-10-11). The 2F-85's spring links keep the MJCF's own springref, 2.62 rad. The scenes had imported them
about 0, as Genesis does (it reads a joint's stiffness, not its springref). Then the 0.05 N m/rad springs no longer hold
the followers back: the fingertips curl inward, the pads close in a V and touch only at their tips.
- buzz_wire_3d's references (springref 0) carried the wand with no finger on it for 41-65 % of their weld time, in the
  traversal grasp for 102-154 s at a stretch: the tips rolled over the handle's end and the weld held the wand alone.
  At the final placement one pad pushed 14-16 N while the other stood 17 mm away. fiber's sheath was touched by the
  distal pad boxes only.
- At 2.62 the pads close parallel (closed fraction 0.85-0.87 at acquisition, against 0.80), both touch the held body
  for the whole weld at 5-9 N each, and the wand's pose in the hand at acquisition is the same within 0.2 mm.
- Robotiq references on springref 2.62 with the contact rule below (fresh replays on this commit; a cloud host
  `<work dir>/core/results/springref2`, tool `core/work/springref/sr_replay.py`; every one CLEAN under the stress
  observer, none released by the contact rule; the same states as the springs set at runtime on the commit before):

  | references | result | next |
  |---|---|---|
  | fiber -i01 very_easy, easier, easy, medium | success at the recorded rows (easier one row earlier), new state hashes c223ae53, 43689f73, 06063247, 44a467ec | fiber re-records them |
  | buzz -i01 very_easy, easier | the traversal completes; the released wand rests 5.25 mm / 7.0 deg and 8.3 mm / 9.1 deg from the finish (tolerance 5 mm / 10 deg) | the finish re-planned from the same cut |
  | buzz -i01 easy, hard | the wand touches the wire: easy 3 rows after its first regrasp release (row 5334), hard in the traversal grasp (row 3831) | re-planned from the released route (low priority) |
  | hedgehog (the hedgehog agent's MuJoCo references) very_easy, easier | success at the recorded rows, new state hashes | |
  | hedgehog medium, hard | medium's grasp slips at row 1560; hard's grasp never acquires (easy failed before) | not re-planned: those tiers are not used |

**The grasp lets go when the fingers leave** (`grasp.WELD_CONTACT_LOST_S`, per robot; Robotiq 0.2 s; 2026-10-11): a
weld whose hand has had no finger contact on the welded link for that long releases like any lost grasp, reason
`finger_contact_lost` (clamp, never fail; the hand grips again after opening and closing). The time counts MuJoCo
steps whose contacts include no row between the hand's finger links (with their child bodies) and the welded link; any
row restarts it. It is part of the world's checkpoints, like the slip windows. Why 0.2 s: on springref 2.62 the Panda
references keep a finger on their held body all the time, except fiber's 55-70 ms right at acquisition (buzz hard's
120 ms comes after its route has already failed); 0.2 s is 2.9x that. A weld can no longer carry a body its fingers have
left, as buzz's did for minutes on springref 0, where the rule would have released every buzz reference's traversal
grasp after 25-500 ms. The ALOHA has no entry (not measured): its grasps never let go this way.

**Known limitation (no change yet): the weld's anchor.** The grasp weld anchors at the hand link's origin, as Genesis'
`add_weld_constraint` does, about 0.2 m from the grasped part. At the weld's `solimp` .9 / .95 its rotational rows yield
about 1 degree under small torques, which that lever turns into millimetres at the part: buzz_wire_3d's 6 g wand under
0.2 N m drifts 2-3 mm (Genesis: 0.33 mm). Buzz stiffens its own weld (`solimp` .99 / .999). The shared fix (anchor at
the grasp point, or a stiffer weld rotation) comes later, with evidence.

The standard service already moves `joint_position` targets at most `JOINT_STEP_LIMIT_RAD` (0.04 rad) per control step:
1.6 rad/s at a 40 Hz control rate (claws), below both robots' limits. There the robot limit acts on privileged
trajectories (the agent's own `Session.step`, and the grader's replays of them) and on anything else that reaches
`Session.step` directly.

**No general robot clamp** (2026-10-10). Nothing beyond the limits above acts on the robot's contact forces. The
breakdowns a force clamp would target build within one control step, or momentum carries them after the drives stop
pushing, so a clamp in the command path comes too late: a prototype reflex that held an arm at its measured joints for
the next control step once a robot-task contact passed 60 N (robot-static: 300 N) left all three of its cases in
BREAKDOWN (fiber FB2 14.0 mm into the ring, the reflex firing a step late at 70-128 N; FB1 before F1 14.4 mm; C1 before
W4 unchanged at 11.3 mm, its force jumping from under 60 N to 515 N inside one 25 ms step). And any threshold low enough
to matter trips on legitimate work: keyhole's references push with 88 N (23 N as a 20 ms mean) and press a held body
onward with 340-483 N, while the other families' references need at most 41 N robot-task and 11 N robot-static (buzz
hard's coupler: one tick at 175-247 N on the wire). Lower torque caps would break buzz's references, which press at up
to 72 % of today's caps, and the ALOHA already runs at its real servos; at today's caps a hand pushes 164-203 N (ALOHA)
or 269-427 N (Panda) in its strongest direction. Each such breakdown is fixed at the contact that makes it (F1, W4, rule
A, the wrist pair, fiber's sheath x ring pair). Tools and results: a cloud host `<work dir>/core/results/clamp/` and
`<work dir>/core/work/` (`reflex.py`, `reflex_replay.py`, `press_trace.py`, `push_ceiling.py`).

## Porting a family

1. Add the family to `profiles.PROFILES`:
   - `solver`: integrator, solver, iterations, cone, impratio; `collision=_COLLISION` (every family);
   - `contact`: `solref`, `solimp`, friction, condim;
   - `weld` (assisted grasp), `robot.armature` and `robot.maxhullvert` (small robot collision hulls);
   - `robot.self_contact` and `static_contact` (`profiles.SOFT_CONTACT`): the soft class of robot-robot and
     robot-static contacts, with the static entities that keep precise contacts and why;
   - `coacd` for non-convex collision meshes;
   - optional: `drag` (air drag on light links), `joints` (joint attribute rules), `mjcf` (an MJCF entity's contact
     attributes), `plastic` (elastoplastic ball joints), `speed_clamp` (damping on MJCF chain links above a speed).
2. If the family declares something the builder does not convert yet, extend `SceneBuilder.add_entity` in `scene.py`.
   Not converted yet: soft bodies, IPC attachments, and the `pbd_local_patch` / `ipc_soft_band` grasp models.
3. Add the family to the MuJoCo loop in `rcb_topogym.TASKS`, then make its instances.
4. Write new references. Old Genesis references usually still do the motion, but their confirmation window ends a few
   ticks later on MuJoCo: record them again with a held tail (claws: `work/make_reference.py`).

Run time: the image's own mujoco (3.14.0, a Genesis dependency) and coacd. Nothing new is installed.

## Runtime environment of a MuJoCo task (cross-CPU)

MuJoCo has no run-time instruction-set dispatch, and the frozen scene carries every CPU-generated number its physics
uses. What a session still generates on the CPU is the task's own assets (the predicates' vertices, the instance's
`asset_identity`), through numpy and OpenBLAS, whose kernels differ between AVX-512 and AVX2 hosts. MuJoCo task containers
(agent, standard `sim`, verifier) and `make-instance` therefore run with the numpy / OpenBLAS SIMD masking, so every CPU
class generates the bytes the instance was built with:

```
NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR
OPENBLAS_CORETYPE=Haswell
```

Measured on the claws instances (Zen 4): masked and unmasked builds differ in some generated files (easy: the pieces'
URDFs and `scene.xml`; hard: the claw URDFs and `scene.xml`; others: reports and a visual GLB only). Genesis families keep
their released environment (the masking changes their frozen asset hashes).

## Rendering

Rendering keeps today's Nyx look ([RENDER_BRIDGE.md](RENDER_BRIDGE.md)). blanket_flatten, buzz_wire_3d, cast_coil,
cast_keyhole, fiber_bundling, hedgehog_in_cage, horseshoe_ring, interlocked_claws, knot_untying, lock_picking, orbits,
rope_threading and slotted_post_ring render with nyx_direct (`scripts/roboarc/shared/nyx_direct.py`, their TASKS `renderer` "direct"),
which feeds Nyx from the MuJoCo world without a Genesis host. The other MuJoCo families render through the Nyx render bridge
(`scripts/roboarc/shared/render_bridge.py`). It builds today's Genesis scene of the instance as a Nyx host, never
steps it, and copies the MuJoCo state into it on every frame. Either renderer attaches on the first frame; until then
`MujocoWorld.cameras` is empty: the session is headless. `RCB_RENDER=off` attaches the same camera rig without pixels
instead (black frames, no GPU; RENDER_BRIDGE.md, "Rendering off"), for the Harbor oracle / nop checks.
`RCB_RENDER=nyx` or `direct` chooses either renderer for any task.

- **VBD cloth: adaptive iterations** (lane S, 2026-10-13). The blanket's VBD steps (`vbd_fast.py`, `vbd_gpu.py`) take
  further blocks of iterations while the last iteration's largest vertex update exceeds 0.1 mm (on in blanket_vbd_gpu and
  blanket_vbd_cpu_twin). The lesson: a fixed iteration count that converges in motion can leave a stiff crumpled stack
  unconverged at rest, and the residual ratchets stored energy up. Why, evidence and cost: families/blanket_flatten.md.

