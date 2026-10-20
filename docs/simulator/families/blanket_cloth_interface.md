<!-- Paths under /workspace/mj/ and work-note paths name the authors' work hosts; those artifacts are not part of this repository. Everything needed to rebuild instances is in source/. -->

# The blanket tasks and their cloth solver: the interface (2026-10-10, blanket agent)

This is every place the blanket family touches its cloth, with the contract a replacement cloth backend must keep. It
covers blanket_flatten's tiers and the redesign's flatten and fold tiers.
- Today the backend is the MuJoCo 2D flex (`topogym/world/mujoco/cloth.py`, profile `blanket_v1`); before it, Genesis'
  PBD. The likely successor is a VBD cloth (Newton).
- Line numbers are at dev/robotangle_mujoco c8db7f4e6. Redesign files are at their sandbox commit 37932c647
  (the work notes, "R:" below).
- Nothing here changes physics. What the flex gets wrong is in `blanket_flatten.md`, "What the flex gets wrong".

The thread through everything: **the 8181 PBD particles, in Genesis' order, are the cloth's public face.** The task,
grasp, instance, hash, render and fold detectors all read them and nothing else. A backend that produces them every
tick, takes hand attachments, collides correctly and checkpoints exactly can be plugged in under the unchanged designs.

## 1. Particle and triangle layout
| where | what | contract |
|---|---|---|
| `topogym/tasks/blanket_flatten/common.py:26`, `:41` | the sheet 0.60 x 0.48 m, mesh spacing 12 mm, `particle_size` 8 mm; material `{rho 4.0 kg/m^2, stretch_compliance 1e-7, bending_compliance 1e-5}` (PBD) | the backend maps this declaration to its own parameters and states the mapping (the flex's is `topogym/world/mujoco/profiles.py:267` `_CLOTH`, used at `:288`: young 3e7, thickness 1.1e-4, i.e. membrane 3.3 kN/m, bending 3.3e-6 N m) |
| `common.py:130`, `:167` | `declare()` writes `blanket.obj` (the 51 x 41 authored grid with uv) and the `EntitySpec("blanket", physics_model="pbd_cloth")` | build from this mesh. The world refuses a frozen cloth built from another mesh (asset sha256, `topogym/world/mujoco/world.py:290-294`) |
| `topogym/world/mujoco/cloth.py:65` `particle_mesh` | the particles: the authored grid with every edge split once (pymeshlab isotropic remesh at 8 mm), 101 x 81 = 8181, Genesis' order (checked to 1.7e-16 m) | **the particle identity**: 8181 particles, this order, float64 world positions, available at any time |
| `cloth.py:89` `barycentric_binding`, `:238` `particles_of` | the flex is the 2091-node grid; each node is a particle (`node`), and every particle is a fixed barycentric combination of its flex triangle's corners (`corners`, `weights`) | simulate the particles, or bind them to the backend's nodes by a fixed map. With a map, state its birth deviation from the PBD particles (the flex: up to 6.3 mm) |
| `cloth.py:80` `orient_up` | the flex faces are counter-clockwise seen from +z on the flat rest | the backend's own element order is free; the task's triangles come from uv (next row) |
| `common.py:202-249` `bind` | `get_particles_pos()` at the flat rest -> material uv per particle (`topogym/assets/geometry/blanket_flatten.py:188`, within `particle_size` of the authored cloth, else refused); `particle_faces` = Delaunay(uv) (`common.py:217`); `particle_edges` and uv rest lengths (`common.py:249`) | at `bind` the particles lie on the authored flat sheet. The birth is applied only after `bind` (section 3) |
| `topogym/world/physics/pbd.py:19` `bind_material_map` (called at `world.py:259`) | the grasp's material coordinates from the frozen material map, at the flat rest | the same, at the same moment |

## 2. The pin-grasp assist (`pbd_local_patch_v1`)
**The engine-free rules** (`SoftGraspBackend`, `topogym/world/physics/grasp_soft.py:40`). The values are in the
instance's `simulator.grasp_assist`.
| where | rule |
|---|---|
| `grasp_soft.py:89` `candidates` | admission when a hand closes. Candidates are particles within `max_tcp_distance_m` 6 cm of the TCP (`:97`). No particle may sit more than `max_penetration_m` 2 mm inside a finger (`:121`). Each finger's convex surface must be within `contact_distance_m` 6 mm of the patch (`:125`). The patch is the particles within 18 mm of a centre in material and in space (`:133`; `topogym/utils/grasp.py:222` `material_patch`). The cloth's strain over the guard edges must be at most `max_strain_ratio` 1.3 |
| `grasp_soft.py:147`; `topogym/utils/grasp.py:263` `exposed_patch` | a patch is occluded when any particle within 18 mm horizontally, more than 36 mm away in material, lies more than 6 mm (`contact_distance_m`) higher |
| `grasp_soft.py:187` `acquire`, `:205` `targets` | the patch particles' offsets in the hand frame at acquisition. The targets are hand pose x offsets |
| `grasp_soft.py:234` `check_health` | every tick: follow error at most `max_translation_error_m` 10 mm, strain at most 1.3, fingers opened by at most 15 %, no rigid-obstacle penetration; else release with the reason |

**The backend adapter** (`MujocoClothGraspBackend`, `topogym/world/mujoco/grasp_cloth.py:26`), built by
`world.py:304` `configure_grasp`:
| where | hook | contract |
|---|---|---|
| `grasp_cloth.py:54` | `positions(name)` | the 8181 particles now |
| `grasp_cloth.py:57`, `:61` | `attach_native(binding)`: the backend nodes that carry the patch particles, pinned to the hand link at their current offsets in the hand frame (MuJoCo: a pool of 24 connect equalities per hand, re-targeted via `eq_obj2id` / `eq_data`; pin solref [.005 1], solimp [.95 .99 .001 .5 2], `cloth.py:187`) | attach any set of cloth points (at most 14 nodes per patch on the flex) to a moving rigid body. Its pose comes from the robot simulation every substep. Stiff enough that the patch follows within 10 mm. Two hands at once |
| `grasp_cloth.py:83` / `:90` / `:96` | `detach_native`, `exists`, `verify_native_ownership` | release, and an exact report of which attachments are active |
| `grasp_cloth.py:125` `sample` (called after every substep, `world.py:386`) | the vector sum of each hand's pin forces. A 20 ms mean above 60 N (`topogym/world/mujoco/grasp.py:38-39`, ALOHA) marks the hand slipped | the attachment force per hand after every substep: the slip rule (clamp, never fail), and what the hand feels |
| `grasp_cloth.py:143` | `check_health` adds `grasp_slipped` | unchanged |
| `grasp_cloth.py:103` | `penetration_reason`: patch particles inside a convex rigid geom | the rigid geoms' vertices (the robot's world provides them) |
| `grasp_cloth.py:149` / `:152` | `capture_slip` / `restore_slip`: the slip windows in checkpoints | checkpoint the attachments and any state of their force rule |

The redesign's FlexSim lab reproduces these rules with mocap hands: R:`topogym/tasks/blanket_flatten/redesign/flexsim.py:237`
`patch`, `:242` `exposed`, `:247` `pins_for`, `:253` `grasp`.

## 3. Birth, instance and hash fields
| where | what | contract |
|---|---|---|
| `scripts/robotangle/mujoco_release.json:19-24` | blanket's release entry: `simulator_overrides {mujoco_birth: "@released_instance", mujoco_birth_settle: {quiet_m_s .05, quiet_s .1, max_s 10}}`, `released_simulator_keys [grasp_assist]`, tiers, variations | the release rebuild makes every instance from this entry, or from its successor's entry |
| the released instance (a cloud host `instances_rel/blanket-flatten-<tier>-i00/instance.json`) | `soft_state.blanket.position_m` (8181 x 3, the frozen PBD particles); `simulator.task_parameters` (with the PBD archive `initial_state_path` + `initial_state_sha256`); `grasp_assist` | the birth source |
| `cloth.py:100` `birth_from_instance` | reads those particles, the archive's sha256 and any settle evidence | build the cloth from the released particles |
| `scripts/robotangle/shared/rcb_topogym.py:4322` `_settle_cloth_birth`, `:4386-4441` (make-instance) | relax at rest, robot holding home (`Session.step`, hold action), until every vertex has stayed under 5 cm/s for 0.1 s (at most 10 s). Writes `instance_assets/mujoco_birth/settled_birth.json` (the released instance with the settled particles, zero velocity, evidence `mujoco_cloth_settle`) and builds the instance from it | the same settle rule (or a documented equivalent), its evidence kept. Deterministic: the same release, profile and arguments make the same birth |
| `rcb_topogym.py:4414` | `instance.json`: `soft_state` (the settled particles, `_soft_state` `:3304`), `objects` (the cloth as "particle": centroid, AABB, mean velocity, `n_points` 8181; `:3817`), `state_sha` | the same fields from the particles |
| `rcb_topogym.py:4441`, `:4450`; `rcb_topogym.py:378` | simulator: `physics_engine` mujoco, `mujoco_birth`, `mujoco_scene` `instance_assets/mujoco`; `compute_backend` cpu, precision 64, `solver_substeps` 1, `physics_dt_s` 0.0025, `control_decimation` 10; `topogym/simulator/config.py:70` (`mujoco_birth`), `:148` | the backend's own engine and frozen-scene keys, validated like these |
| `cloth.py:191-199` | `instance_assets/mujoco/cloth_blanket.npz` (`rest`, `faces`, `birth`, `node`, `particle_rest`, `particle_faces`, `corners`, `weights`); the cloth record in `scene.json` (`cloth.py:193`: format, mesh, counts, masses, sphere radius, settings, pins, asset sha256, birth {source / archive sha256, state_sha, settle evidence, particles sha256, off_surface_max_m}) | a frozen, versioned cloth description inside `instance_assets` (solver parameters, mesh, particle map, birth) from which the instance rebuilds bit for bit |
| `world.py:224-227` (load) | the frozen scene is loaded as it is; only the exact diagonal and the pad rule are set at load | the backend's frozen description is loaded unchanged |
| `world.py:327` `restore_cloth_births` | after the task's `bind`: every cloth at its frozen birth, at rest (`cloth.py:241` `apply_birth`). Refuses a birth made from another PBD archive (sha256) | apply the birth after `bind`; check the archive identity |
| `rcb_topogym.py:3312` `state_sha` (over `topogym/simulator/env.py:120` `tick_record`) | sha256 of the tick record (the world readout incl. the 8181 particle positions and velocities, the robot, the task) at 12 decimals; contacts sorted | the particle view is deterministic. Two fresh-process replays of a reference end on the recorded hash, also across CPU classes (the flex: workstation i9 AVX2 = a cloud host Zen 4) |
| `rcb_topogym.py:3268` `_poses_match` | an instance reopens only if its `soft_state` matches within 2 mm | the readout at birth equals the stored soft state |

## 4. What the predicates read
| where | reads | contract |
|---|---|---|
| `world.py:605` (read_state), `cloth.py:246` `particle_state` | `read_state()["particles"]["blanket"] = {position_m, velocity_m_s}`, both (8181, 3), at each tick's end. The velocity is the displacement over the tick (start taken at `world.py:375`) divided by 2.5 ms | positions and tick velocities as defined here (not instantaneous). The cloth never appears in `bodies`, and its contacts never appear in `world.contacts()` (`world.py:542`) |
| `common.py:255-311` `evaluate_state` | particles + table height only. covered (`common.py:294`): the top-down union of the uv-Delaunay triangles / 0.288 m^2 >= `minimum_coverage_ratio` (0.95 in the tiers), with `topogym/evaluation/cloth_metrics.py:44`; GEOS `cloth_metrics.py:33` within 1e-3. flat (`common.py:277`): the highest particle centre <= 30 mm above the table top. above_table (`common.py:278`): >= -8 mm; below -0.1 m: "Cloth fell below the table". material_intact (`common.py:279`): edge stretch <= 1.15. settled (`common.py:280`): speed <= 0.03 m/s | nothing beyond particles. **The heights are particle centres**, so the backend must state the resting height of one, two and three layers. On the flex one layer rests at 4.4 mm, and a two-layer fold's top layer at 11 mm |
| `common.py:314` `initialize` | a frozen start must not already meet the goal | unchanged |
| `common.py:57` `final_settling_spec`; `topogym/evaluation/terminal/settling.py:15`, `:113-120` | terminal (`release_settling_v1`): assist released; every particle <= 0.03 m/s and <= 3 mm drift over 0.5 s; the robot's collision geometry >= 5 mm clear of the particles' AABB grown by 8 mm (`particle_radius_m` = `particle_size`) | the particle AABB is the cloth's extent: the 8 mm padding must cover the backend's half-thickness |
| `rcb_topogym.py:2410`, `:2618` `_guard_particle_speeds` (installed at `:3176`) | any particle faster than 20 m/s, or non-finite, ends the episode as SIMULATOR_INSTABILITY (infrastructure, never graded) | the blow-up signal stays the particle speed |
| `topogym/planning/blanket_flatten/robot_oracle.py:211` `fold_stage`, `:290` `edge_stage`, `:383` `crease_stage` (`:411` the orientation test), `:31` `MIN_OVERLAP_VOTES` | the oracles' fold detector: a triangle is turned over when its uv -> xy orientation is opposite the sheet's majority. Corner flaps come from a rigid flat fit; over / under from height votes within one particle size | particles + uv only |
| R:`topogym/tasks/blanket_fold/redesign/predicate.py:57` `below_material`, `:32` `layer_map`, `:71` `stacking`, `:96` `evaluate`, `:21` thresholds | blanket_fold v3.1: per particle the material directly below (within 8 mm horizontally, **1-20 mm lower**, more than 3 cm away in material); layer counts per 2 cm cell; named points; outline IoU; heights against the target's (dz 20 mm, over 10 mm) | particles + uv only. The 1 mm lower bound was set for the flex's nested layers (3.0 mm apart); re-measure it, and the 6 mm exposure clearance (section 2), against the backend's layer gap |
| R:`flexsim.py:434` `evaluate` (`:457` `turned_over_fraction`) | the lab's coverage / turned-over / layered fractions | the same |

## 5. What the renderer reads
| where | what | contract |
|---|---|---|
| `rcb_topogym.py:384` | blanket renders with `nyx_direct` (TASKS renderer "direct") | unchanged |
| `scripts/robotangle/shared/nyx_direct.py:1564` -> `:1680` `bind_cloth` | each frame, the particle view (`Cloth.particles_of(Cloth.vertices(data))`) | **the 8181 particles per rendered frame** |
| `nyx_direct.py:1306` `pbd_visual_vertices`, `:1280` `frozen_pbd_birth`, `:776`, `:869-888` | Genesis' visual vertices as weighted sums of their supporting particles (12 each), from the released PBD archive (`PBDSolver/vverts_info`, the visual mesh's UVs and faces). One double-sided Nyx dynamic mesh (`:881`) with the fabric texture | nothing new: the skin, UVs and texture stay the archive's. A backend with its own finer surface may drive a mesh with the same UVs instead |
| `nyx_direct.py:1786` | depth: MuJoCo's own raster of the model (`mj_flex` for the flex) | a cloth surface in the depth renderer's scene, or depth from Nyx |
| `scripts/robotangle/shared/render_bridge.py:338` `bind_flex` (`:355`, `:370`); bound at `rcb_topogym.py:3065` | the Genesis-host path: every particle skinned from the nearest flex triangle at birth, written into the host's PBD entity each frame | the same particles |

## 6. Stress observer hooks
| where | hook | contract |
|---|---|---|
| `scripts/robotangle/stress/observer.py:498` `_install` | wraps `mujoco.mj_step` (after every substep: `_after_substep`) and `world.advance` (after every tick: `_after_tick`). Read-only: a run with it ends on the same hash | a callback after every solver step and after every tick, with read access to the state below |
| `observer.py:145-156` | the flex's vertex bodies form one articulated task entity, `flex:blanket`: speeds, speed jumps and their causes, whips, pops, tunnelling | node positions, velocities and masses per step |
| `observer.py:403-420` | a flex contact's sides (`con.flex`, `con.vert`, `con.elem`) mapped to vertex bodies; pair classes task-task (cloth self-contact; `:743`, not an outside cause), task-static (table, fence), task-robot | contact points, depths, normals and forces by pair class (self, static, robot) |
| `observer.py:886-889`, `:1013-1027`; `topogym/world/mujoco/energy.py:28` `flex_energy` | E = kinetic + gravity + the flex's membrane + bending energy (exact against the elastic forces' work) | the cloth's elastic potential, exact enough to book |
| `observer.py:236`, `:564` `_flex_vertex_dofs`, `:577` `_flex_loss`, `:1045` | the discrete integrator's own loss on the flex's elasticity, booked as L (loss only; a gain stays in X) | the integrator's numerical dissipation or creation per step |
| `observer.py:1098` | each grasp pin as a cloth-hand tie (its work) | the attachment forces |
| (summary) | solver iterations at the cap, MuJoCo warnings; the verdicts CLEAN / VIOLENT / BREAKDOWN (`scripts/robotangle/stress/verdict.py`, criteria v2.9) | solver health: iterations, convergence, failure flags. BL1 showed why: a starved shared solve blew up a cloth that nothing touched |

## 7. Save / restore
| where | what | contract |
|---|---|---|
| `world.py:625` `capture_state` (`:637`, `:643`) | the full integration state (`mj_getState` INTEGRATION), the pin targets (`eq_data`, `eq_obj2id`), the grasp manager's bindings, the slip windows, the cloth's tick start (velocity readout), the contact evidence | capture everything the next tick depends on: node state, attachments and their force-rule state, the velocity readout's tick start |
| `world.py:646` `restore_state` (`:671`) | restores all of the above and re-checks every binding against the active pins (`grasp_cloth.py:96`) | restore bit for bit. A restored run continues on the same hashes as an unbroken one |
| `world.py:357` `advance` (`:365`, `:375`, `:377-386`, `:392`, `:394`) | per tick: the grasp's `before_tick`, the cloth's tick start, the substeps (one `mj_step` per tick for blanket) with `grasp.sample()` after each, `_finish_tick` (`mj_flex`), the grasp's `after_tick` (health) | the same order of hooks around the backend's step |
| R:`flexsim.py:476` `save_state`, `:485` `load_state` | the lab's states: `vertices` (2091 x 3, the flex) and `particles` (8181 x 3), no velocities | the redesign's starts and targets are particle sets. A new backend loads them by particles and settles them (or re-generates them) |

## 8. Collisions
| where | what | contract |
|---|---|---|
| `cloth.py:40-42`, `:45-56`, `:161`, `:180` | vertex spheres r 4.5 mm (contype 4 / conaffinity 6) meet the table and ground (2 / 3) at friction 0.5 (the task's `table_friction` static mu) and each other at 0.15 (PBD's particle friction); solref [.005 1], solimp [.9 .99 .001 .5 2], condim 3, elliptic cone, impratio 10. The flex surface (radius 4 mm, 8 / 1) meets the robot's geoms (fingers, links, arm bases), friction 0.5, two-way. No flex self-collision | cloth-table (mu 0.5), cloth-self (fabric 0.3-0.6), cloth-robot (all robot collision geoms, two-way) |
| R:`flexsim.py:91` obstacles, `:106` weights | the redesign's fence (hard4b): static boxes with the table's collision bits, friction 0.5, solref [.005 2], solimp [.9 .95 .001 .5 2]; weights: free boxes on the cloth | cloth against static and free rigid boxes |
| `blanket_flatten.md`, "What the flex gets wrong" | the sphere net nests layers 2.8 mm apart and interlocks them; the cloth is ~100x too limp for its weight | a smooth self-contact surface and a stated thickness. The 6 mm exposure clearance needs covering layers more than 6 mm apart to see them |

## 9. What else moves with a new backend
- The flatten oracles (`robot_oracle.py`, `routes.py`, `below_easy_routes.py`) read only particles, uv and the assist.
  Their stages were tuned on the flex: the measured slide, the crease drop-then-lay, the base clearance and the fold
  layouts. Re-validate them and re-record the references.
- `robot_oracle.py:618` records the birth's deviation from the frozen archive when `physics_engine == "mujoco"`, instead
  of checking it bitwise. A new engine needs the same switch.
- Cost today: about 1.7-2.0 s per control step on a loaded a cloud host. very_easy's 1897-row reference took 52.7 min on the
  workstation. The limit is 72 min per observed run.

## 10. The VBD backend's answers (profile `blanket_vbd`, 2026-10-11)
`topogym/world/mujoco/cloth_vbd.py` (+ `vbd_fast.py`, `grasp_cloth.MujocoVBDClothGraspBackend`). Material and
measurements: `blanket_flatten.md`, "The VBD cloth backend".
| contract | the VBD backend |
|---|---|
| particles (sections 1, 4) | the 2091 grid vertices simulated; the 8181 particles by the flex's barycentric map (same birth deviation rule) |
| exposure (section 2) | a covering particle must be more than 36 mm in material from every patch particle, not just from the one it covers (`exposed_patch(remote_from_patch=True)`): with layers 7 mm apart, the top layer of a fold covered the bottom part of a bend grip's own patch. Clearance still 6 mm. `scripts/robotangle/test_exposure_rule.py` |
| assist (section 2) | the patch's grid vertices become kinematic (mass 0) at their hand-frame offsets, written every step; release restores the masses and zeroes their velocities. Hand force = minus the pinned vertices' membrane forces and weight, after every step, into the 60 N slip rule; its reaction goes to the hand body's `xfrc_applied` one step later |
| birth (section 3) | `instance_assets/mujoco/cloth_vbd_blanket.npz` (`rest`, `faces`, `birth`, `node`, `particle_rest`, `particle_faces`, `corners`, `weights`, `birth_rest_angle`); scene.json record format `robotangle-vbd-cloth/1` with the settings, versions (newton, warp, the solver module's sha256) and birth evidence. The settle also waits for the plastic rest angles (`quiet_rest_angle_rad`) |
| **new state: bending rest angles** (sections 3, 7) | plastic bending changes `model.edge_rest_angle`: it is in the checkpoint, the instance and the readout (`particles.blanket.rest_angle_rad`, so the state hash covers it). Arrays are assigned in place, never rebound (the threaded step holds their pointers) |
| heights (section 4) | 1 / 2 / 3 layers rest at 3.0 / 10.0 / 17.0 mm; layers 7.0 mm apart. The 8 mm AABB padding covers the 3 mm half-thickness |
| render (section 5) | nyx_direct unchanged (the particle view); MuJoCo depth sees the cloth through the deformable visual mesh |
| save / restore (section 7) | positions, velocities, rest angles, pins, pin forces, the lagged reaction, the tick start, the plastic loss; the self-contact BVH is rebuilt at every tick start, so every tick boundary restores exactly |
| collisions (section 8) | every robot / scene collision geom as a kinematic Newton shape (box, sphere, capsule, cylinder, the convex hull of a mesh), the static ones fixed; smooth self-contact at a 7 mm offset. Robot contacts act on the cloth only (two-way is the pins' reaction only), unless the profile's `cloth.two_way` is on (`blanket_vbd_gpu`; off in `blanket_vbd`): then every robot contact's force goes back to MuJoCo too (`cloth_coupling.py`; `blanket_flatten.md`, "Two-way coupling") |
| observer hooks (section 6) | `ClothVBD.step_hooks`: callables (cloth, dt) after every cloth step (lane D's energy books). Two-way: the coupling's work on the robot (`ClothVBD.coupling.work_on_robot_j`) and the book's W_react, C = W_robot + W_react |
