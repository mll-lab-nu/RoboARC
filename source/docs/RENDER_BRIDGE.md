# Render bridge: Nyx frames for MuJoCo states

RoboTangle's physics moves from Genesis to MuJoCo; the images agents see do not change.
`scripts/roboarc/shared/render_bridge.py` renders MuJoCo states with today's (0.2.22) Nyx pipeline.

## How it works (route A)

- **Host scene.** `render_bridge.build(instance.json)` builds the frozen instance's scene with the same TopoGym code
  as the 0.2.22 runtime: `Env(SimulatorConfig(**instance["simulator"])).reset(seed)`, RGB deferred. That gives the
  same entities, generated visual assets (checked against the instance's `asset_identity`), materials, lights, HDRI,
  Nyx camera rig and Nyx settings (FastPathTracer, 32 spp, denoiser, Reinhard, SMAA). It never integrates physics.
- **State in.** `bridge.update(mj_model, mj_data)` copies MuJoCo's joint coordinates into the host's rigid solver with
  one `set_qpos`, bit for bit. Genesis forward kinematics then yields the same link and visual-geometry poses that
  Genesis physics would hold in that state; wrist cameras follow their links. Velocities, contacts and constraints
  are not transferred; nothing reads the host's physics.
- **Frames out.** `render()` runs the unchanged sensor path (`SensorSuite.sample`): one shared Nyx pass for the whole
  rig. `depth()` is the unchanged Genesis raster depth (`SensorSuite.depth_frame`). `camera_metadata()` is the same
  calibration dict as `SensorFrame.camera_metadata`.
- **The one subtle step.** The host's clock never advances, but Genesis refreshes visual geometry, the raster scene
  and Nyx frames only when `scene.t` changes. After each update the bridge marks those clocks outdated, as a physics
  tick or `scene.reset()` does. Without that, a pose push renders the old pose (depth was 1.8 m off in a probe).

```python
import render_bridge                          # scripts/roboarc/shared/
bridge = render_bridge.build(instance_json)   # 21-26 s warm Genesis cache, ~60 s cold (as today's startup)
bridge.update(mj_model, mj_data)              # ~1 ms
rgb = bridge.render()                         # {camera: (H, W, 3) uint8}
depth = bridge.depth()                        # {camera: (H, W) float32 m, 0 = no reading}
meta = bridge.camera_metadata()               # {camera: intrinsics + world pose in this state}
bridge.close()
```

`bridge.bind(model).report()` shows how every joint matched. `bridge.check_fk(model, data)` compares host link poses
with MuJoCo body poses; a frame-convention or naming error shows up there, not in pixels. Ropes and cloth: see
"Deformables" below.

## Naming contract for MuJoCo models

A Genesis joint matches the first of these:

1. `overrides["<entity>/<joint>"]`;
2. the MuJoCo joint `<entity>::<joint>`, `<entity>/<joint>` or `<entity>_<joint>`;
3. the bare joint name, if no other host entity has a joint of that name (robot: `left/waist`, `joint1`);
4. the k-th joint of the MuJoCo body `<entity>::<link>`, `<entity>/<link>`, `<entity>_<link>`, `<link>`, or (base
   link) `<entity>`.

`aliases={"piece_0": "claw_0"}` renames an entity. Type (hinge, slide, ball, free) and size must agree. A MuJoCo
joint may not be used twice. Unmatched joints raise `MappingError` unless listed in `keep` (they stay at the birth
value). The MuJoCo core's scene builder (`<entity>::<link>`, free joint `<entity>::root`, robot names unchanged)
matches without options.

## Process model

One Python process (the image's Python 3.11): Genesis 1.2.1 + gs-nyx 0.1.3 + the MuJoCo physics. Genesis imports
`mujoco` itself (MJCF parsing), so both use one module. The image ships mujoco 3.14.0; a 3.15.0 overlay (the core
agent's) also works for the host (evidence below). The bridge reads only `qpos` (and `xpos`/`xquat` in `check_fk`).

The host leaves the process as it found it: `Env.reset` reseeds Python/NumPy/Torch RNGs and registers the process's one
active Env; the bridge restores both, so the MuJoCo Env stays the active one and its random streams are untouched.

## Parity evidence (2026-10-09, a cloud host RTX PRO 6000, image 0.2.22 runtime)

Harness: `tools/tests/test_render_bridge.py`. **Today** = `rcb_topogym.Session` (release tooling, Genesis
physics on CPU float64) replaying the release reference trajectory and rendering through `env.sensor_frame()`.
**Bridge** = a fresh process that builds the host, never steps it, turns each recorded state into MuJoCo `qpos`,
runs `mj_kinematics`, and calls `bridge.update`, `render`, `depth` and `camera_metadata`. Every state is rendered
once and in order, as a stream: each frame's Nyx history is the previous state, as in standard-mode streaming.

Sequences:

- **claws** (`interlocked-claws-easy-i00`): control steps 0-200 at every step (moving), then every 10th to 780.
  That is 259 states with 4 cameras at 848x480.
- **fiber** (`fiber-bundling-easy-i00`): steps 0-200 at every step, then every 20th to 1990. That is 291 states with
  3 cameras.

MuJoCo models:

- a kinematic mirror of the host (test fixture) under mujoco 3.14.0 (image) and 3.15.0 (overlay);
- the core agent's claws scene `instance_assets/mujoco` (also with the bridge built from the core's MuJoCo
  `instance.json`);
- the fiber agent's `scenes/fiber-easy/mjcf-source-passive/scene.xml`.

### Exact, every frame

| Run | Frames | qpos round trip | Host vs MuJoCo link poses | Nyx inputs | Depth | Calibration | Static Nyx scene |
|---|---|---|---|---|---|---|---|
| claws, mirror, mujoco 3.14 | 259 | 259/259 | 5.0e-16 m | 259/259 x2 refs | 259/259 | 259/259 | not recorded (run predates the check) |
| claws, mirror, mujoco 3.15 | 259 | 259/259 | 5.0e-16 m | 259/259 x2 | 259/259 | 259/259 | not recorded |
| claws, core's scene (18 joints: 16 by name, 2 by body) | 259 | 259/259 | 5.0e-16 m, 7.3e-16 rad, 26 links | 259/259 x2 | 259/259 | 259/259 | = today |
| claws, core's scene + core's instance.json | 259 | 259/259 | same | 259/259 x2 | 259/259 | 259/259 | = today; assets 0 differences |
| fiber, mirror | 291 | 291/291 | 1.2e-15 m, 1.7e-15 rad, 92 links | 291/291 x2 | 291/291 | 291/291 | = today |
| fiber, fiber agent's scene (81 joints by name, incl. 66 ball) | 291 | 291/291 | 1.2e-15 m, 1.7e-15 rad, 100 links | 291/291 x2 | 291/291 | 291/291 | = today |

- *Nyx inputs* are the buffers the Nyx pass read: every visual geometry's position and quaternion and every camera
  pose, as float32.
- *Static Nyx scene* is the exported scene file (instances, materials, textures, lights, HDRI, camera settings) plus
  the meshes, with random UUIDs and paths normalized. Claws is `bb6349b8…` in every run that recorded it (today R3,
  R4, both Forward today runs; bridge core, core-instance and Forward runs). Fiber is `d8a0fd3c…` in today R3 and its
  signature run, and in bridge B1, B2 and the fiber agent's run.
- The same inputs over the same scene is the same rendering problem. What remains is the renderer itself.

### RGB: Nyx is not reproducible run to run, and the bridge sits inside that noise

Today's pipeline rendered twice (same instance, actions and call order) gives different RGB:

- claws: 3-26 of 259 frames are identical per camera;
- fiber: no frame is identical.

The cause is the renderer's temporal history plus GPU nondeterminism. Flips are tile-shaped and reach ~65k pixels;
values move by up to ~65 levels on single pixels. Grading never uses RGB.

The test therefore treats the bridge as another run. Below is the mean over frames of the per-frame mean |difference|
in 8-bit levels, for every pair of runs, by pair type. Claws has 3 today runs and 4 bridge runs; fiber has 3 and 3.

| Camera | today vs today | bridge vs today | bridge vs bridge |
|---|---|---|---|
| claws cam_high | 5.01e-3 - 6.59e-3 | 3.44e-3 - 6.19e-3 | 3.62e-3 - 5.57e-3 |
| claws cam_low | 1.19e-3 - 2.07e-3 | 1.74e-3 - 2.44e-3 | 1.57e-3 - 2.98e-3 |
| claws cam_left_wrist | 4.02e-5 - 5.62e-5 | 5.03e-5 - 7.82e-5 | 3.76e-5 - 5.83e-5 |
| claws cam_right_wrist | 7.71e-4 - 1.14e-3 | 4.79e-4 - 1.56e-3 | 1.06e-3 - 1.54e-3 |
| fiber agentview | 1.23e-2 - 1.86e-2 | 0.70e-2 - 2.32e-2 | 1.26e-2 - 1.98e-2 |
| fiber front_oblique | 3.79e-2 - 4.88e-2 | 2.83e-2 - 4.71e-2 | 2.82e-2 - 3.77e-2 |
| fiber wrist | 6.95e-4 - 7.92e-4 | 5.16e-4 - 1.05e-3 | 3.03e-4 - 1.02e-3 |

- Typical frames differ in 20-100 pixels.
- Claws `cam_left_wrist` is the one camera where bridge-vs-today runs above today-vs-today (7.8e-5 vs 5.6e-5 levels).
  That excess comes from frame cadence, not from the bridge. Nyx RGB depends on the wall-clock spacing of renders: a
  today run with 0.25 s added between captures (nothing else changed) differs from the normal today runs by
  9.1e-5 - 1.0e-4 there, more than the bridge does. On `cam_right_wrist` that gap is 3.1e-3 - 3.8e-3 against the
  bridge's 0.7e-3 - 1.3e-3. Today's frames therefore already vary with machine load and physics speed.
- Review images are in the work notes. Each shows today | bridge |
  |today - bridge| x8 | |today - today| x8.

**Forward (raster) mode, test only.** Both pipelines were forced to `ERenderMode.Forward`, 121 claws frames:

- today vs today: 58-98 % of frames are pixel-identical, the rest differ in at most 27 px;
- bridge vs today: 32-91 % identical, at most 28 px;
- bridge vs bridge: 31-66 % identical, at most 37 px.

Forward mode is only a sharper probe. Production keeps the path tracer: the look must not change.

### Large state jumps (ghosting)

The test renders state X three times (claws at step 400, fiber at step 1500), then jumps to the birth state and
renders it 6 times. Today's jump is `Session.reset()`; the bridge pushes the two states.

- The first frame after the jump differs from the settled 6th frame only in anti-aliasing along the current edges.
  Claws `cam_high`: 6191 px, max 67 levels. There is no outline of the old pose: where X and the birth state differ,
  the mean change is 0.04 levels, with at most 7 px above 20 levels.
- From the second frame on, the image equals the settled one, apart from Nyx's usual random flips (one 4019-px flip
  in a second today run).
- Today's pipeline shows the same numbers (6192 px, max 67; `cam_left_wrist` 2136, `cam_low` 3870, `cam_right_wrist`
  0, in both).
- In fiber, the path tracer's noise flips dominate in both pipelines; neither shows a ghost.

## Speed (shared box, GPU 80-99 % busy with other jobs)

| | Today (Session) | Bridge |
|---|---|---|
| Startup | claws 59-60 s cold cache, 23-25 s warm; fiber 64-65 s cold, 27 s warm | build 57-59 s cold, 21-26 s warm (the Genesis part of today's startup) |
| State push per frame | - | 0.9-1.0 ms (`update`: one `set_qpos` + clock invalidation) |
| RGB, whole rig, one Nyx pass | claws 59 ms, fiber 40 ms (includes the observation's robot/task reads) | claws 42-52 ms, fiber 33-38 ms |
| Depth, whole rig | claws 15 ms, fiber 19 ms | claws 12 ms, fiber 15 ms |

Standard mode streams frames through `env.sensor_frame()` at its video rate. With the bridge each streamed frame costs
the same Nyx pass plus about 1 ms.

## Deformables: ropes and cloth (vertex streaming)

**What carries a deformable to Nyx today.** The plugin streams every deformable's vertices every frame. In
`NyxPyRenderer.update_scene`, FEM vertices come from `fem_solver.get_state_render(cur_substep)` (all of them; the
surface triangles index into them). PBD cloth visual vertices come from `pbd_solver.get_state_render()`, which
re-skins them from the particles. Genesis raster depth reads the same solver state (`RasterizerContext.update_fem` and
`update_pbd`).

So the never-stepped host takes deformable shapes the same way it takes joint coordinates: write the solver state,
then mark the scene clocks stale.
- FEM: `entity.set_pos(cur_substep, vertices)`.
- PBD: `entity.set_particles_pos(particles)`.

Measured (2026-10-09, release instances):

| | knot_untying very_easy, IPC FEM rope | blanket_flatten very_easy, PBD cloth |
|---|---|---|
| deformable in the host | 29,806 vertices, 34,608 surface triangles | 8,181 particles carry 2,091 visual vertices |
| one vertex write | 0.2–0.35 ms | 0.7–1.2 ms |
| birth vertices written back | Nyx inputs and depth bit-identical | the same |
| a +5 cm lift written | Nyx's vertex buffer moves +0.050 m; RGB and depth follow | the same |

### API

`bridge.update(model, data)` writes the registered deformables after the joint coordinates. Register each one once,
while the host still shows its birth shape and `data` holds MuJoCo's birth state:

```python
bridge.bind_chain("rope", model, data)              # rigid capsule chain -> the FEM rope's mesh
bridge.bind_flex("blanket", model, data)            # 2D flex -> the PBD cloth's particles
bridge.update(model, data)                          # joints + every bound deformable, then render as usual
```

**`bind_chain(entity, model, data, bodies=None, link_s=None)`** binds a rope.
- The chain is an ordered list of MuJoCo bodies. By default these are the bodies named `<entity>::*` or `<entity>/*`,
  in id order.
- It drives the FEM entity's vertex set and topology unchanged.
- Binding is linear-blend skinning on **material arc length**. A vertex at material s blends the two links whose
  centres bracket s, keeping its birth offset in each link's frame.
- Material s, not distance, picks the links, so a knot's crossing strands never borrow each other's vertices.
- `link_s` gives each link's material start and the last link's end. The default is equal parts of the rope's
  material range, checked against the chain's own joint spacing.

**`bind_flex(entity, model, data, flex=None)`** binds a cloth.
- The flex is a MuJoCo 2D flex. By default it is the flex named `entity`, else the only one.
- Each PBD particle is bound to the nearest flex triangle: barycentric weights of the nearest point, plus the
  particle's offset in the triangle's own frame.
- A flex built on the particle mesh carries the particles vertex for vertex.
- A coarser flex, for example one built on the visual mesh, carries them on its surface.
- In both cases the binding is exact at birth.

Two other changes:
- `build()` now takes IPC and frozen-asset instances (knot, rope) the way the Session does: the libuipc build of the
  GPU, the frozen rope volume, and realized identities for a bound instance.
- `test_render_bridge.py selftest` checks both skins on CPU (image Python): exact at birth, rigid motions carried
  rigidly, and a bent chain staying closed at the joint.

### Evidence (prototypes)

Kinematic MuJoCo models were built from the release instances. The model holds the robot mirror plus:
- **rope**: a chain of 120 capsule links, 8.2 mm long and 4.0 mm in radius, along the rope's birth centreline;
- **cloth**: flex A built on the 8,181 particles, and flex B built on the 2,091-vertex visual mesh.

The links and flex vertices were moved programmatically; no physics port was involved. Scripts are on a cloud host:
`/workspace/mj/render/deform/{rope_proto,cloth_proto}.py`, with outputs in `results/deform/`.

| | rope (knot very_easy) | cloth (blanket very_easy) |
|---|---|---|
| birth vertex error after binding | 3.9e-17 m | flex A 2.8e-17 m, flex B 2.7e-16 m (visual vertices) |
| birth frame vs the host's own birth frame | Nyx inputs and depth bit-identical; RGB inside Nyx's repeat noise | the same |
| update per frame (rigid + deformable + invalidation) | median 3.3–3.4 ms, max 5.6–8.9 ms (two runs) | median 2.7 ms, max 7.0–7.3 ms (two runs) |
| motions checked | rope lifted 10 cm and straightened, travelling wave ±20°/link, coil to 30°/link, torsion to 45°/link | lift 5 cm, a 150° fold about a line, travelling wave 3 cm |

Rope skin quality, as ring radius and ring spacing relative to birth:
- straightening: radius ≥ 0.998, spacing ≤ 1.004;
- wave: radius ≥ 0.987, spacing ≤ 1.04;
- coil: radius ≥ 0.975, spacing ≤ 1.08;
- torsion: radius ≥ 0.92, spacing ≤ 1.004. This thinning is linear-blend skinning's known "candy wrapper": a ring
  midway between two links twisted θ apart shrinks to cos(θ/2). That is 0.924 at 45° per link, as measured, and 0.996
  at 10°.

Neither skin shows tearing, holes or spikes. The cloth keeps its texture through the fold and the wave.

Caveats:
- No collisions in these kinematic tests. The links pass through the table and the robot, so the motions were run in
  the air. Contact belongs to the MuJoCo models.
- The real rope and cloth models (the rigid-chain work, a blanket flex) bind the same way.

## Rendering off (`RCB_RENDER=off`)

For runs whose frames nobody looks at and that have no GPU, such as the Harbor oracle / nop checks, `RCB_RENDER=off`
in the environment of every container that runs a Session (agent, standard `sim`, verifier) replaces the bridge with
`render_bridge.headless(instance)`:

- the same cameras, resolutions and calibration, from the instance's scene configuration (the Session's camera
  contract);
- wrist cameras follow their MuJoCo bodies (link pose @ mount offset, as Genesis' `Camera.move_to_attach`);
- RGB is black, depth has no reading (0); no Genesis, no Nyx, no GPU. Videos and review frames are black.

Checked on claws easy in the dev image without a GPU (2026-10-09): the fixed cameras' poses and the wrist mounts equal
Genesis' `pos_lookat_up_to_T` bit for bit, the wrist poses equal the MuJoCo body pose @ the contract mount bit for bit
along the reference, and the state hash is the same with and without frames (and with a video). Unset, the task's
renderer applies ("In the Session" below): the bridge (`nyx`) or nyx_direct (`direct`). `RCB_RENDER=nyx` or `direct`
chooses one of them for any task; any other value is refused.

Who sets it:
- only infrastructure runs: Harbor oracle / nop checks (`scripts/roboarc/harbor_check.py`) and CI;
- standard mode: the agent cannot reach the `sim` service's environment, so it cannot turn rendering off there;
- privileged mode: an agent that sets it in its own container only blinds its own Sessions; the verifier's container
  sets its own environment, and grading never uses pixels.

A MuJoCo task's binding names the bridge it ships (`render_bridge_sha256`, written by `mk_tasks.py`). For a family that
renders with nyx_direct, it also names `nyx_direct.py` (`nyx_direct_sha256`). A Session refuses a different file, as it
refuses different tooling.

## Nyx without Genesis (`nyx_direct.py`)

Step 2 of the no-Genesis road map (MUJOCO_MIGRATION.md): gs-nyx itself needs only numpy, so
`scripts/roboarc/shared/nyx_direct.py` feeds it straight from the MuJoCo world. It imports no genesis-world and
builds no Genesis host scene.

```python
import nyx_direct                                            # scripts/roboarc/shared/
nyx = nyx_direct.NyxDirect.from_world(world, instance=path)  # the Session's MujocoWorld and its instance.json; ~3.5 s
                                                             # (Genesis host: 24-25 s); rope chains bound at birth
nyx.update(model, data)                                      # this state's rigid-slot, camera poses, rope vertices
rgb = nyx.render()                                           # {camera: (H, W, 3) uint8}: one Nyx pass for the rig
depth = nyx.depth()                                          # {camera: (H, W) float32 m, 0 = none}: MuJoCo raster
nyx.close()
```

**What it builds.** Everything comes from declarations the MuJoCo world already holds: env_cfg, robot_cfg, the task's
EntitySpecs with their generated assets, and the MuJoCo scene manifest that maps entity links to bodies. From these
it writes the same `SceneAsset` that gs-nyx-plugin 0.1.4 exports for the Genesis scene those declarations build, using
gs_nyx's own SDK (`nps.export_scene_file`). It also hands Nyx the same per-frame buffers, in the plugin's native call
order: `startup`, `load_scene_from_file`, `allocate_scene_buffers`, then each frame `update`, `update_scene_buffers`,
`set_env_map(0)` and `render_frame` per camera.

- **Entities**, in Genesis' order:
  - ground plane, table, the table frame's eight bars, stage, robot, task entities, the lab dressing;
  - appearance rules applied as `apply_finishes` applies them (colour, metallic, roughness, wood maps).
- **Primitives and URDFs** become entity-level instances: boxes and the plane are Nyx primitives; URDFs are
  sub-scenes that Nyx loads itself, driven per link.
- **glTF mesh entities** (rope_threading's rings, stands and bases) are entity-level `Mesh` instances of the file,
  which Nyx loads with its own nodes and materials, as the exporter exports them:
  - at the morph pose, with `convertAxis` off (Genesis imports a glTF Y-up) and the morph's scale;
  - a surface with a colour or a texture overrides the file's materials (Genesis' `is_visual_overwritten`), so the
    file's textures are unset first;
  - one reference per mesh Genesis makes of the file (per mesh node and material, read from the file's JSON:
    `gltf_mesh_groups`), each at the link pose: the morph pose composed with the z-up turn that
    `file_meshes_are_zup` records as the morph's `offset_quat`;
  - fixed entities only for now. Depth draws the file's meshes turned Y-up to Z-up, as Genesis holds them.
- **MJCF entities** (the robot; fiber's bundle) get one instance per visual geometry, named `<link>_<k>`.
  - The mesh is compiled from the MJCF with MuJoCo exactly as Genesis' MJCF loader reads it: vertices split per
    (vertex, normal) pair, and per texture coordinate too where the mesh has them (an untextured material then
    drops the UVs); vertex colours from the material or rgba.
  - It is written with trimesh, as the exporter writes it, with the vertex normals Genesis has computed (primitives
    included); the claws robot's 24 OBJ files and fiber's 258 (robot and bundle) come out byte-identical.
- **FEM deformables** (ropes) get one `DynamicMesh` instance each, after every rigid instance (the exporter's
  solver order), with the exporter's complete material.
  - Static data, uploaded once: zero UVs and the boundary triangles of the tetrahedra, in Genesis' FEMEntity order.
  - Birth vertices: the instance's frozen IPC settled state where it has one (`settled_fem_birth`, numpy only),
    else the volume's spawn as Genesis instantiates it.
  - Per frame: the vertices that `bind_chain` skins from a MuJoCo body chain (`render_bridge.ChainSkin`, the same
    skin the host streams), float32.
  - Depth: the surface as a MuJoCo mesh whose vertices are re-uploaded every frame.
- **PBD cloths** (blanket) get one `DynamicMesh` instance after the FEM ones (the exporter's solver order), with the
  exporter's complete material. A cloth's surface is double-sided unless it says otherwise, as Genesis' `add_entity`
  sets it for cloth materials.
  - Topology, from the PBD solver's state that the instance freezes (`frozen_pbd_birth`, numpy only): each visual
    vertex's 12 supporting particles and their weights, the UVs and the faces.
  - Particles, from MuJoCo: the cloth's particle view of its flex (`Cloth.particles_of`, as the task, grasp and
    evaluation see the particles), bound by `bind_cloth`.
  - Per frame: each visual vertex is the sum, in support order, of particle × weight in float64, as Genesis'
    `_kernel_update_render_fields` computes it; then float32.
- **Per frame:**
  - every rigid slot gets MuJoCo body pose × its local pose, as float32, Z-up, wxyz;
  - static cameras are fixed;
  - wrist cameras get body pose × mount offset, then Genesis' float32 look-at arithmetic.
- **CUDA buffers** go through the driver API with ctypes on libcuda: one page-locked upload per frame, and frames
  copied on the device then fetched in one download. Neither torch nor genesis is needed for this part.

Three details decide the last bit.

- **Camera arithmetic runs on the host's backend.** MuJoCo instances freeze `compute_backend: cpu`, so today's host
  computes camera quaternions with torch float32 on the CPU, and torch's vectorised CPU sqrt is not IEEE-rounded.
  - When torch is importable (it is in the image), the adapter runs Genesis' two scripted kernels unchanged
    (`_tc_z_up_to_R`, `_tc_R_to_quat`, Apache-2.0) on the same device (`camera_device`), and the buffers match bit
    for bit: on the CPU for MuJoCo instances, on CUDA for the knot release instance (Genesis on `cuda`).
  - Otherwise a numpy replica is used. It agrees in about 83 % of random cases and is at most an ulp off.
- **Fixed entities' root rotation.** Genesis composes a root link with its morph quaternion by its numpy product
  (an 8-multiplication form, normalised): once for every root, twice for a scene file such as a URDF. A quarter
  turn then keeps a 2.2e-16 residue in a zero component, which float32 keeps (fiber's ring). The adapter composes
  them the same way (`genesis_quat_mul`).
- **Depth sample position.** Genesis' raster depth camera draws into a 4× multisampled buffer and resolves it by
  its first sample. NVIDIA places that sample at (+1/8, −3/8) px from the pixel centre.
  - The MuJoCo raster moves its principal point there (`GENESIS_DEPTH_SAMPLE`).
  - That alone takes the median depth difference from 0.41 mm to 0.004 mm.

### Acceptance: claws easy

Claws easy, 8 states along the reference, a cloud host RTX PRO 6000 with the GPU shared with other jobs. Reproduce with
`tools/tests/test_nyx_direct.py` (stages host, direct, spread, bench, ab).

Three host recordings (H1–H3) and two adapter runs (D1, D2), on the same MuJoCo states.

| check | result |
|---|---|
| genesis imported by the adapter's process | no |
| static scene (nyx_scene.json) | identical: 83 instances, materials, cameras, lights, HDRI. The robot's 24 OBJ files are byte-identical. The generated URDF/OBJ assets differ only in their per-session temporary paths; their bytes equal the instance's asset identity. |
| reference table (83 slots: UUID pattern, node names) | identical |
| per-frame input buffers, D1 and D2 vs H1 | rigid positions and rotations 664 / 664 bitwise; camera positions and rotations 32 / 32 bitwise |
| RGB | direct–host pairs sit in today's run-to-run spread. Mean over the 8 frames of the per-frame mean \|difference\|, in 8-bit levels: see the table below |
| depth (MuJoCo raster vs Genesis raster, per camera and frame) | the same valid pixels (every pixel hits the lab shell). Median \|Δ\| 0.001–0.009 mm, p99 0.03–0.62 mm. 99.1–99.6 % of pixels within 1 mm and 99.4–99.7 % within 5 mm; the rest are silhouette pixels. |
| build (scene export + Nyx startup) | 3.5–4.0 s, against 24–25 s for the Genesis host |
| per-frame cost (state in, RGB of all 4 cameras on the host), 3 rounds back to back, 80 frames each | median 82–85 ms, against 109–115 ms for the host path |

| camera | host–host (3 pairs) | direct–host (6 pairs) | direct–direct (1 pair) |
|---|---|---|---|
| cam_high | 0.54–1.44e-3 (≤ 9 px > 8 levels) | 0.47–1.88e-3 (≤ 10 px) | 1.45e-3 (10 px) |
| cam_low | 0–0.35e-3 (≤ 2 px) | 0–0.67e-3 (≤ 2 px) | 0.67e-3 (2 px) |
| cam_left_wrist | ≤ 1e-5 (0 px) | ≤ 1e-5 (0 px) | 1e-5 (0 px) |
| cam_right_wrist | 0–0.64e-3 (≤ 2 px) | 0–0.64e-3 (≤ 2 px) | 1e-5 (0 px) |

- **RGB.** The adapter's runs differ from each other as much as from the host's. What remains is Nyx's own
  frame-to-frame noise (render cadence, GPU), with no systematic difference.
- **Per-frame cost.** The GPU was shared with other jobs throughout. The adapter does no forward kinematics on
  Genesis, makes one input upload, copies frames on the device and does one download.
- **Re-run** on the head with the FEM path and the rope route (two host recordings, two adapter runs): the same
  result. Static scene and reference table identical, 664 / 664 and 32 / 32 bitwise, RGB and depth in the ranges
  above. Per-frame cost with both paths alternating frame by frame in one process: median 82 ms against 118 ms,
  0.70 of the host's per frame (p10–p90 0.63–0.77).
- **How the cost is measured.** On a shared GPU, separate processes land in different contention regimes within
  minutes (the same path measured 20–28 ms in one round and 120–200 ms in the next), so rounds of host and adapter
  runs compare regimes as much as paths. The figures for fiber and knot (and the claws re-run) therefore come from
  one process that builds both, today's host and the adapter, and alternates them frame by frame on a host
  recording's states (`test_nyx_direct.py ab`).

### Acceptance: fiber easy

Fiber easy on the wire profile (the r2 reference, 7,613 controls), 8 states along it; two host recordings and two
adapter runs on the same MuJoCo states (and three adapter runs before the rope route, with the same result).

| check | result |
|---|---|
| genesis imported by the adapter's process | no |
| static scene (nyx_scene.json) | identical: 322 instances. The robot's and the bundle's 258 MJCF visual meshes are byte-identical (117 distinct OBJ files); the ring is a URDF sub-scene. |
| reference table (322 slots) | identical |
| per-frame input buffers | bitwise at every state after birth: rigid positions and rotations 322 / 322, cameras 3 / 3. At birth 27 rotations, 2 positions and the wrist camera's rotation differ, in components below 1e-9, by at most 2e-16 (see below). |
| RGB | see the table below |
| depth (MuJoCo raster vs Genesis raster) | the same valid pixels. Median \|Δ\| 0.0001–0.006 mm, p99 ≤ 0.54 mm; 99.1–99.97 % of pixels within 1 mm, ≥ 99.4 % within 5 mm. |
| build (scene export + Nyx startup) | 5–8 s, against 25–41 s for the Genesis host |
| per-frame cost (state in, RGB of all 3 cameras on the host), both paths alternating frame by frame in one process, 120 frames each | median 62 ms against 92 ms for the host path: per frame 0.69 of the host's (p10–p90 0.62–0.76). Separate-process rounds in a quiet spell: 12.2–12.8 ms against 18.6–20.5 ms. |

| camera | host–host | direct–host (4 pairs) | direct–direct |
|---|---|---|---|
| agentview | 0.6e-3 (8 px > 8 levels) | 1.4–4.9e-3 (≤ 19 px) | 4.9e-3 (25 px) |
| front_oblique | 9.7e-3 (62 px) | 32–58e-3 (67–836 px) | 65e-3 (898 px) |
| wrist | 0.5e-3 (7 px) | 0.1–1.7e-3 (≤ 8 px) | 1.3e-3 (2 px) |

- **Birth.** The bundle's bodies are almost axis-aligned at birth: some rotation and position components are around
  1e-13. Genesis' and MuJoCo's forward kinematics round those in a different order (differences of 1e-17 to 2e-16),
  and float32 keeps the relative precision of such small numbers, so those rows are not bitwise. No pixel's
  geometry changes. After birth every row is bitwise.
- **front_oblique.** An occasional frame differs from the others by about 820–930 pixels, all on the terminal's
  display panel (an emissive surface; mean 0.1–0.3 levels there). It is Nyx's own: two adapter runs on bit-identical
  buffers differ by that much on such a frame, and today's host renders it too. A cadence test rendered the
  recording's 8 states 160 times per setting and counted frames more than 300 pixels off the recording on the panel:
  the host at its own pace (19 ms) 1, the host paced to 80 ms 0, the adapter unpaced 1, paced to 35 ms 2, paced to
  80 ms 3. The rate does not follow the cadence, so the adapter renders unpaced.

### Acceptance: knot very_easy (a FEM rope driven by a MuJoCo chain)

The rope agent's MuJoCo instance of knot_untying very_easy (the rope a rigid capsule chain, rope_chain.py), its r1
reference (1,234 controls), 8 states along it. Today's host is the Session's: the released Genesis instance in
`instance_assets/render_host/` (CUDA, IPC settled birth), whose FEM rope the chain drives through `bind_chain`. The
adapter takes the same route (`from_world(world, instance=...)`). The instance's runtime binding pins the tooling it
was made with, so the test runs on a copy bound to the code under test (as `mk_tasks.py --dev` binds a local
instance).

| check | result |
|---|---|
| genesis imported by the adapter's process | no |
| static scene | identical: 81 instances, the rope a `DynamicMesh` (29,806 vertices, 34,608 triangles) with the exporter's complete material; the ALOHA's 24 MJCF meshes byte-identical |
| reference table | identical: 80 rigid slots and the rope |
| static deformable data (UVs, triangle indices) | identical |
| per-frame input buffers | bitwise at every state: rigid positions and rotations 640 / 640, cameras 32 / 32 (camera arithmetic on CUDA, as the render host), rope vertices 238,448 / 238,448 |
| RGB | see the table below |
| depth | the same valid pixels. Median \|Δ\| 0.0001–0.008 mm, p99 ≤ 0.51 mm; ≥ 99.17 % of pixels within 1 mm, ≥ 99.42 % within 5 mm (the rope's own pixels included). |
| build | 6.9 s, against 53 s for the Genesis host (IPC) |
| per-frame cost (all 4 cameras), both paths alternating frame by frame in one process, 120 frames each | median 119 ms against 208 ms for the host path: per frame 0.57 of the host's (p10–p90 0.53–0.61) |

| camera | host–host | direct–host (4 pairs) | direct–direct |
|---|---|---|---|
| cam_high | 1.6e-3 (3 px > 8 levels) | 1.2–2.4e-3 (≤ 6 px) | 3.6e-3 (6 px) |
| cam_low | 9.7e-3 (14 px) | 3.1–9.1e-3 (≤ 16 px) | 2.2e-3 (6 px) |
| cam_left_wrist | 2.0e-3 (1 px) | 1.1–2.5e-3 (≤ 1 px) | 1.3e-3 (1 px) |
| cam_right_wrist | 2.1e-3 (1 px) | 0.7–2.4e-3 (≤ 2 px) | 1.7e-3 (2 px) |

### Acceptance: rope_threading easy and hard (glTF rings)

The rope agent's MuJoCo instances (rtinst7) and references (refs/rt6: easy 1,882 controls, hard 3,326), 8 states
along each. The host is the released Genesis instance in `instance_assets/render_host/`; its FEM rope is driven
through `bind_chain`, and its rings, stands and bases are glTF entities. Each instance is a copy bound to the code
under test.

| check | easy | hard |
|---|---|---|
| genesis imported by the adapter's process | no | no |
| static scene, reference table, static deformable data | identical | identical |
| per-frame input buffers | bitwise but one: rigid positions 704 / 704, rotations 703 / 704, cameras 32 / 32, rope vertices 39,512 / 39,512 | bitwise: rigid 728 / 728 both, cameras 32 / 32, rope vertices 39,512 / 39,512 |
| RGB, pixels more than 8 levels apart per camera and state | at most 22; host against host: up to 22 too | at most 6 |
| depth | the same valid pixels; median \|Δ\| ≤ 0.015 mm, p99 ≤ 0.69 mm; ≥ 99.08 % within 1 mm | median ≤ 0.013 mm, p99 ≤ 0.74 mm; ≥ 99.08 % within 1 mm |
| build | 9.6 s, against 188 s for the Genesis host | 10.3 s, against 163 s |

The one rotation that differs (easy, the left wrist link at the second state) has two components that are zero up to
rounding: 4.6e-16 and −3.2e-16 on the host, 4.0e-16 and −4.5e-16 here. The two forward kinematics round zero
differently.

### Acceptance: blanket easy (a PBD cloth driven by a MuJoCo flex)

The blanket agent's MuJoCo instances of blanket_flatten easy, 8 states along each reference:
- short-settled: instances_short easy -i01, its cloth born settled on MuJoCo (up to 20.4 mm from the released PBD
  particles), with refs/easy_short1 (798 controls);
- the earlier birth: instances_it200 easy -i01, with refs/easy_try1 (754 controls).

A Genesis host renders these instances only in this test:
- The MuJoCo instance's own configuration cannot restore the PBD archive. render_bridge.build also keeps
  `simulator.mujoco_birth`, which a Genesis configuration refuses. So the test copy carries the released Genesis
  instance as `instance_assets/render_host/`, as rope instances do.
- That archive restores only on its own GPU profile (sm89). On a cloud host the test skips that backend check: the host never
  steps, and the restore only seeds the PBD solver's tables.
- The host's particles are the MuJoCo particle view, as nyx_direct's. render_bridge.bind_flex would skin the released
  particles onto the flex and keep a settled birth's displacement (up to 20 mm) as offsets.

| check | short-settled | earlier birth |
|---|---|---|
| genesis imported by the adapter's process | no | no |
| static scene | identical: the cloth a `DynamicMesh` (2,091 vertices, 4,000 triangles), double-sided, its diffuse map the generated asset of the instance's identity | identical |
| reference table, static UVs and triangle indices | identical | identical |
| per-frame input buffers | bitwise: rigid 640 / 640 both, cameras 32 / 32 (on CUDA, as the released host), cloth vertices 16,728 / 16,728 | bitwise, the same counts |
| RGB, pixels more than 8 levels apart per camera and state | at most 16 (left wrist), 5 elsewhere | at most 24 (left wrist), 8 elsewhere |
| depth | the same valid pixels; median \|Δ\| ≤ 0.009 mm, p99 ≤ 0.65 mm; ≥ 99.11 % within 1 mm | median ≤ 0.009 mm, p99 ≤ 0.69 mm; ≥ 99.10 % within 1 mm |
| build | 17 s, against 285 s for the Genesis host | 14 s, against 249 s |

Before the cloth's surface was made double-sided, single-sided folds showed 11–31 pixels apart on the overview
cameras.

### In the Session: the renderer per family

A family's renderer is a field of its `rcb_topogym.TASKS` entry, `renderer`:
- `"direct"`, nyx_direct: blanket_flatten, buzz_wire_3d, cast_coil, cast_keyhole, fiber_bundling, hedgehog_in_cage,
  horseshoe_ring, interlocked_claws, knot_untying, lock_picking, orbits, rope_threading and slotted_post_ring
  (2026-10-10).
- Absent, `"nyx"`: the Genesis host, for every other family.

The Session hands the field to `_render_bridge`, and `RCB_RENDER`, when set, overrides it (`nyx`, `direct` or `off`).
For these families, `mk_tasks.py` (`NYX_DIRECT`):
- ships `nyx_direct.py` next to every copy of the tooling: the privileged `environment/` and `tests/` and the standard
  `environment/sim/`, with their Dockerfile `COPY` lines;
- writes `nyx_direct_sha256` into the binding (`binding()`, `dev_binding()`).

`_check_render_bridge` refuses a different `nyx_direct.py` when the binding names one. The other families' generated
tasks are unchanged, byte for byte.

With `direct`, `_render_bridge` returns `nyx_direct.DirectRig`, built by
`nyx_direct.rig(instance, world, render_bridge.headless(source))`:
- **Calibration:** the cameras of the headless rig of the same scene (as `off`): res, fov, near, far, transform, wrist
  mounts.
- **Pixels:** one `NyxDirect` pass for the whole rig per state, and depth from MuJoCo's raster.
- **Inputs:** both read the headless rig's forward kinematics on a private MjData; the physics data is untouched.
- **Route:** `NyxDirect.from_world(world, instance=...)`, the render host's.

Checked with one standard-mode episode per task under each renderer: the shared standard service natively on a cloud host, the
oracle client streaming the reference, one fresh-process replay with its video, the standard grade.

| episode | live / fresh replay | final state hash, both renderers | sensor frame, all cameras, median (concurrent load) |
|---|---|---|---|
| claws easy v5 (refs_v5, 792 rows) | success / success | f56b201c1f464455 | direct 24 ms, nyx 129 ms |
| buzz very_easy (b1d1706a4, r21, 9,133 rows) | success / success | 78911c221025e646 | direct 28 ms, nyx 86 ms |

The frames the agent received match within Nyx's noise, with one exception: buzz's wand. That difference is the Genesis
host's error, below.

### A Genesis-host error: free bodies aligned to their centre of mass

Genesis anchors a free root body without articulated descendants at its centre of mass and principal axes (`align`,
on by default). A free URDF task body whose centre of mass is not at its origin is one example. MuJoCo keeps that body's
frame at the URDF origin, with the centre of mass as the inertial offset.

The bridge copies MuJoCo's free-joint coordinates into the aligned Genesis joint unchanged, so the host draws the body
at the MuJoCo pose composed with minus its centre-of-mass offset. nyx_direct draws it at MuJoCo's pose.

- **Buzz (confirmed).** The wand is drawn 8.15 mm along its handle axis from its physical place. In the wand's frame,
  the visual ring spans x −16.7 to +32.6 mm around the origin, where MuJoCo's metal collision capsules lie (centroid at
  0.4 mm), and its centre of mass is at x 8.15 mm. With the reference tables matched, the host's and the adapter's
  buffers agree bit for bit except the wand's rows, which differ by that offset at every state.
- **Hedgehog (confirmed).** The ball's offset is (−0.41, −0.89, −2.11) mm in its frame, 2.32 mm long: the host draws
  the ball, spikes included, 2.3 mm from where MuJoCo has it (medium).
- **Claws, fiber and keyhole.** Not affected. Claws and fiber pass bitwise parity, and keyhole's plates have zero offset
  (very_easy).

`check_fk` cannot see this error. It compares the host's link frames with MuJoCo's bodies, and the bridge copies those
frames exactly; only the meshes sit off. The offsets above are what Genesis records per root link
(`RigidEntity._links_offset_pos` / `_links_offset_quat`), read from a never-stepped host of each instance (2026-10-10).

Buzz renders with nyx_direct (above). Hedgehog's ball declares `align: False` (a6d8fdb23,
`topogym/tasks/hedgehog_in_cage/common.py`), so both Genesis paths keep its frame at its URDF origin, and hedgehog
renders with nyx_direct as well (Acceptance: hedgehog_in_cage, below).

### Acceptance: claws and fiber (the switch)

Their shipping instances, two tiers each, 8 states along each reference, host twice:
- claws: the v5 instances (the -i01 release re-binds the same scenes) with refs_v5;
- fiber: the v4 instances (tag 19, on the shared head) with tag 17's references (unchanged by the spring-link fix).

| check | claws very_easy | claws hard | fiber very_easy | fiber medium |
|---|---|---|---|---|
| genesis imported by the adapter's process | no | no | no | no |
| static scene, reference table | identical | identical | identical | identical |
| input buffers | bitwise but 2 rotations at birth | bitwise: rigid 664 / 664 both, cameras 32 / 32 | bitwise but birth and one state (below) | bitwise but birth (below) |
| RGB: pixels > 8 levels apart per camera and state, direct–host (host–host) | cam_high 18 (13); others ≤ the host's | ≤ the host's on every camera | front_oblique 836 (68), below; others ≤ the host's | wrist 18 (9); front_oblique 957 (940), below; others ≤ the host's |
| depth median / p99 | ≤ 0.007 / 0.37 mm | ≤ 0.008 / 0.83 mm | ≤ 0.007 / 0.51 mm | ≤ 0.007 / 0.62 mm |
| build (direct vs host) | 11 s vs 116 s | 11 s vs 175 s | 13 s vs 129 s | 13 s vs 114 s |
| per-frame cost, both paths alternating in one process | 0.63 of the host's | 0.61 | 0.65 | 0.64 |

The buffer differences are all rounding of components at zero, 1e-17 to 3e-16:
- the birth FK residue (a pad, the wrist camera by one ulp);
- fiber's bundle links (trunk and fiber rotations, one trunk position): 35 rows at medium's birth, and 8 rows each
  at very_easy's birth and second state.

Fiber's front_oblique outliers are its display panel, all of the pixels, and the host shows them as well: medium's two host recordings differ by 940 pixels there. Rendering very_easy's eight states six times on each path gives no frame over 300 pixels on either (median 49 on both).

One standard-mode episode per family under the switch (RCB_RENDER unset, so nyx_direct): live and fresh replay succeed, live = replay, at the reference hash:
- claws easy v5, 792 rows, f56b201c1f464455, the same as under the host;
- fiber very_easy, 3,556 rows, 7af6d2d3fc24a031.

Licence: the gs-nyx 0.1.3 and gs-nyx-plugin 0.1.4 wheels carry no licence file and no licence metadata. Their
documentation repository (Genesis-Embodied-AI/nyx-for-genesis) is Apache-2.0. The adapter copies no plugin code: it
calls gs_nyx's public API as its `.pyi` stubs document it, and reproduces the plugin's data, checked against
recordings.

Not covered yet by `nyx_direct`:
- more than one PBD entity in a scene (the frozen archive is the solver's);
- free glTF mesh entities (a fixed one is covered above);
- MJCF materials with textures, and appearance rules on MJCF entities;
- task spheres and cylinders.

Each raises `NotImplementedError` rather than render differently.

### Acceptance: lock_picking (the switch)

The five released tiers on MuJoCo (very_easy .. hard, seed 0, built on 10e993813; the switch also covers the chain-lock
tiers lock_picking_v2_*, which were not part of this check). Per tier a 40-row nop run, states at rows 0, 20 and 40, the
Genesis host recorded twice, nyx_direct once (`test_nyx_direct.py host / direct / spread`); and the birth frame of the
same tier's Genesis instance (built by the same code, physics_engine genesis) against nyx_direct.

| check | very_easy, easy, medium, hard | easier |
|---|---|---|
| genesis imported by the adapter's process | no | no |
| static scene, reference table | identical | identical |
| input buffers | bitwise but rounding at zero (<= 9e-16) | the same, except 3 rotation rows of rotor 0 (below) |
| RGB: pixels > 8 levels apart per camera and state, direct-host (host-host) | <= 1 (<= 5) | agentview 569, front_oblique 3,274 (0); wrist 1 |
| depth median / p99 | <= 0.006 / 0.36 mm | <= 0.006 / 1.14 mm |
| the key at birth, MuJoCo against the Genesis instance | 0.0 mm, 0.0 deg | 0.0 mm, 0.0 deg |
| birth frame against the Genesis instance's own (240-row grid) | 0 pixels > 8 levels (very_easy: max 2) | 955 pixels, all on rotor 0 |

- **The key.** Declared `align: False` (topogym/tasks/lock_picking/common.py), so its frame is its URDF origin on both
  engines: the Genesis host draws it where MuJoCo has it, and its buffer rows are bitwise.
- **easier's rotor 0 (a Genesis-side error).** easier starts that rotor at 20 deg (`initial_disc_deg [20, 0]`). MuJoCo's
  hinge, the Genesis host's link (after the bridge's `set_qpos`) and the Genesis instance's own link all hold 20 deg
  (link quaternion (0.9848, -0.1736, 0, 0)), yet both Genesis paths draw the rotor's visual at 0 deg, and they do so at
  every state of the nop run. nyx_direct draws it at 20 deg, where its MuJoCo collision body is. The three differing
  buffer rows differ by exactly that 20 deg (sin 10 deg = 0.1736 in a quaternion component); the depth p99 of 1.14 mm
  is the same rotor. No other tier starts a rotor away from 0.

### Acceptance: hedgehog_in_cage (the switch)

The five released tiers on MuJoCo (very_easy .. hard, seed 0, native gripper space: the -i01 references' instances,
docs/rcb/families/hedgehog_in_cage.md) and the experimental hedgehog_in_cage_window_change, on jianwen/mj2-hedgehog
(jianwen/mj-integrate merged with e75390aab). Per tier a 40-row nop run, states at rows 0, 20 and 40, the Genesis host
recorded twice, nyx_direct once (`test_nyx_direct.py host / direct / spread`); and the birth frame of the same tier's
Genesis instance (built by the same code, physics_engine genesis) against nyx_direct.

| check | all six |
|---|---|
| genesis imported by the adapter's process | no |
| static scene, reference table | identical |
| input buffers | bitwise but rounding at zero (<= 7e-15), the ball's rows included |
| RGB: pixels > 8 levels apart per camera and state, direct-host (host-host) | 0 (0) on agentview, front_oblique and wrist; the largest difference 8 levels (hard's birth, front_oblique) |
| depth median / p99 | <= 0.007 / 0.42 mm |
| the ball at birth, MuJoCo against the Genesis instance | 0.0 mm, 0.0 deg |
| birth frame against the Genesis instance's own | 0 pixels > 8 levels (hard: max 2) |

- **The ball.** It declares `align: False` (a6d8fdb23), so its frame is its URDF origin on both engines: the Genesis
  host draws it where MuJoCo has it (before that change, 2.3 mm away), and its buffer rows agree to rounding.
- **What nyx_direct draws here.** The cage and the ball are URDFs whose glTF visuals carry baked wood textures (finish
  "wood" on every tier), the mount is a box, the goal region a fixed mesh: all covered, nothing raises
  `NotImplementedError`, and no pixel of the textured wood is more than 8 levels from the host's.
- Evidence: `runs/robotangle_mujoco/v2/hedgehog_in_cage/render/`.

## Reference videos

`tools/release/render_reference_video.py` renders a reference run for review (the step-6 videos):
- A Session replays the recorded actions with their recorded ticks.
- Every stride-th control step, the state is rendered with the family's renderer: `--renderer direct` (nyx_direct)
  or `host` (the bridge).
- The replay must end at the reference's hash, or the video is marked failed.

**No temporal aliasing.** A sped-up video samples the motion. A motion faster than half the sampling rate shows up as
a slower, false one. Claws hard v5 was rendered at a stride of 13 control steps, 3.08 states per simulated second;
the reference's 1.62 Hz wrist dither then turned into frame-to-frame alternation that looked like a tremor
(2026-10-10). The rule:
- At least 8 rendered states per simulated second (`--frames-per-sim-second`, refused below 6): the stride is at most
  floor(control_hz / 8) control steps, so motions up to 4 Hz stay below Nyquist.
- The stride still aims at a video of at most 45 s (`--max-seconds`), so short references keep their slower playback.
- No frame blending, and no extra skipping to shorten a long reference: it makes a longer video. Claws hard is 405 s of
  simulation, so 3,242 frames, 108 s at 30 fps (3.75×).
- For a close look at a window, a 1x clip renders every control step at the control rate.

Cost, for the 18 step-6 videos of claws, buzz, knot and fiber: about 97 minutes of rendering instead of 69. The
replays' physics stays the larger part, 232 minutes.

## Not covered yet

- Instances with frozen generated assets other than the IPC rope volumes (none known).
- CUDA-physics instances (lock_picking, float32): the host follows the instance's backend; untested.
- MPM/SPH deformables: the plugin streams them too, but no RoboTangle family uses them.
