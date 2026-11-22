"""PBD cloth stepped by VBD next to MuJoCo: a "pbd_cloth" declaration whose profile "cloth" has backend "vbd" is simulated
by Newton's Vertex Block Descent solver (newton.solvers.SolverVBD, Apache-2.0, on NVIDIA Warp, Apache-2.0) on the
cloth's own grid, while MuJoCo steps the robot and the rigid scene. The contract with the blanket tasks:
docs/rcb/families/blanket_cloth_interface.md. The coupling follows the solver spike (tmp/2026-10-10_cloth_solver_spike,
REPORT.md).

  - the cloth: the authored grid (its nodes are PBD particles, as cloth.py's "visual" flex) as a Newton cloth mesh,
    stable Neo-Hookean membrane and dihedral bending, mass lumped from the profile's areal density. Positions live in a
    local frame (origin: the flat sheet's centre on its rest plane) to keep float32 magnitudes small;
  - MuJoCo never sees the cloth's dofs. After every mj_step (world.advance) the cloth takes one step of the same dt: the
    colliders' poses before and after the MuJoCo step (kinematic Newton bodies: every moving collision geom of the
    robot and scene, as box / sphere / capsule / cylinder / the convex hull of a mesh), the static geoms (the table, a
    fence) as fixed shapes, particle-shape contacts from newton.CollisionPipeline, then the solver's step;
  - the particle view: every PBD particle a fixed barycentric combination of its grid triangle's vertices (cloth.py's
    binding), so the task, the grasp, the instance, the hash and the render read the same 8181 particles;
  - the assisted grasp (grasp_cloth.MujocoVBDClothGraspBackend): the patch's vertices become kinematic (mass 0) at
    their offsets in the hand frame and follow the hand every step; the force the hand puts on the cloth (minus the
    pinned vertices' membrane forces and weight) feeds the 60 N slip rule and, one step later, MuJoCo as the hand
    body's xfrc_applied (two-way for the pins; without `two_way` the robot's other contacts act on the cloth only);
  - two-way coupling (the world profile's cloth key `two_way`: True or a dict of cloth_coupling.DEFAULTS; on in blanket_vbd_gpu):
    the cloth's contact forces on every robot body go back to MuJoCo as xfrc_applied over the next cloth step's MuJoCo
    steps, linearly implicit in the robot's motion and capped by the arms' actuator limits (cloth_coupling.py). Its
    colliders and pins take the kinematics of the current qpos (a private MjData), also on a profile without
    mujoco_substeps, where mjData's are the last MuJoCo step's start;
  - checkpoints: positions, velocities, pins and the lagged reaction; the self-contact BVH is rebuilt at every tick
    start, so every tick boundary is an exact restore point (the spike: without the rebuild a restore diverges).
The solver class is the profile's `solver` ("newton": SolverVBD, or "<module>:<Class>": a SolverVBD subclass with the
same constructor, e.g. a multi-threaded CPU step), built with the profile's settings; its step is the one seam.
The profile's `device` places the Newton model, the collision pipeline and the solver: "cpu" (vbd_fast.FastSolverVBD,
4 threads) or "cuda" / "cuda:<n>" (vbd_gpu.GpuDetSolverVBD, deterministic across GPU classes). A GPU process turns FMA
contraction off in every Newton module before the model is built (vbd_gpu.fuse_fp_off): a process steps its cloth on
one device only. Host code (the particle view, pins, checkpoints, books) reads and writes the device arrays in place."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

VBD_FORMAT = "robotangle-vbd-cloth/1"

# Profile "cloth" keys of the VBD backend and their defaults (the spike's "realistic" material; the layer offset 7 mm so
# that the assist's 6 mm exposure clearance sees a covering layer).
VBD_DEFAULTS = dict(
    backend="vbd",
    mesh="visual",                 # the authored grid (its nodes are PBD particles)
    particle_size_m=.008,          # PBD's particle size (the task's pbd.particle_size): the particle mesh's edge target
    density_kg_m2=.5,              # areal density (the released PBD declares 4.0; a blanket is 0.3-0.8)
    tri_ke=1270., tri_ka=1090., tri_kd=1e-4,   # Lame mu / lambda [N/m] (E t 3.3 kN/m, nu .3), membrane damping
    edge_ke=6.5e-3, edge_kd=1e-5,  # dihedral bending (7.6e-5 N m: a 2.5 cm bending length at .5 kg/m^2)
    radius_m=.003,                 # cloth-to-shape contact distance (the sheet rests 3 mm above the table)
    self_margin_m=.007,            # cloth-cloth offset: the layer spacing
    self_gap_m=.0014,              # extra self-contact detection reach (query 8.4 mm, under the grid's 8.49 mm diagonal)
    body_gap_m=.004,               # extra particle-shape detection reach
    contact_ke=2e4, contact_kd=1e-4,
    self_mu=.4, table_mu=.5, robot_mu=1.,      # pair friction: cloth-cloth, cloth-static, cloth-robot
    friction_eps=1e-2,
    topo_filter=1, rest_exclusion_m=0., vt_buffer=32, ee_buffer=48,
    iterations=20,
    solver="newton",               # "newton" (SolverVBD) or "<module>:<Class>" (a SolverVBD subclass)
    solver_kwargs={},
)
# A profile may add device: "cpu" (absent: the default) or "cuda" / "cuda:<n>" (the Newton model, the collision
# pipeline, the solver). It is not a default above, so the CPU profiles' recorded settings stay as they were.
# The step: device, solver and solver_kwargs follow the world's profile when it names a device (a frozen scene recorded
# on the CPU profile runs on the GPU step under blanket_vbd_gpu); the material and everything else come from the
# scene's record.
PLACEMENT_KEYS = ("device", "solver", "solver_kwargs")


def _gpu(device) -> bool:
    return str(device).startswith("cuda")


def settings_of(profile) -> dict:
    return {**VBD_DEFAULTS, **dict(profile.get("cloth") or {})}


def is_vbd(profile) -> bool:
    return (profile.get("cloth") or {}).get("backend") == "vbd"


def versions(settings=None) -> dict:
    """newton's and warp's versions; with settings naming a solver module ("<module>:<Class>"), that file's sha256."""
    from topogym.world.mujoco.warp_cpu import init
    init()
    import newton
    import warp
    out = dict(newton=newton.__version__, warp=warp.config.version)
    spec = (settings or {}).get("solver")
    if spec not in (None, "newton"):
        module = importlib.import_module(spec.partition(":")[0])
        out["solver"] = dict(spec=spec, sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
    return out


# ----------------------------------------------------------------------------------------------- scene building
def add_cloth_vbd(builder, spec, birth) -> None:
    """Write cloth `spec` for the VBD backend: its data file cloth_vbd_<name>.npz (the grid, the particle binding, the
    birth) and the manifest record (kind "pbd_cloth", cloth format robotangle-vbd-cloth/1). Nothing enters the MJCF."""
    import trimesh
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    from .cloth import particle_mesh, orient_up, barycentric_binding
    if birth is None:
        raise ValueError(f"{spec.name}: building a VBD cloth needs its frozen birth (SimulatorConfig.mujoco_birth)")
    s = settings_of(builder.profile)
    if s["mesh"] != "visual":
        raise ValueError(f"{spec.name}: the VBD cloth runs on the authored grid (mesh 'visual')")
    mesh = trimesh.load(spec.asset_path, force="mesh", process=False)
    R = Rotation.from_quat(np.asarray(spec.quaternion_wxyz, dtype=float)[[1, 2, 3, 0]]).as_matrix()
    V = np.asarray(mesh.vertices, dtype=np.float64) @ R.T + np.asarray(spec.position_m, dtype=float)
    F = np.asarray(mesh.faces, dtype=np.int64)
    P, PF = particle_mesh(V, F, s["particle_size_m"])
    born = np.asarray(birth["particles"], dtype=np.float64)
    if born.shape != P.shape:
        raise ValueError(f"{spec.name}: the birth has {born.shape} particles, the PBD particle mesh {P.shape}")
    dist, node = cKDTree(P).query(V)
    if dist.max() > 1e-9:
        raise ValueError(f"{spec.name}: the cloth grid's nodes are not PBD particles ({dist.max():.3g} m)")
    rest, faces = V, orient_up(V, F)
    corners, weights, gap = barycentric_binding(P, rest, faces)
    if gap.max() > 1e-9:
        raise ValueError(f"{spec.name}: a PBD particle lies off the flat cloth ({gap.max():.3g} m)")
    area = float(trimesh.Trimesh(vertices=V, faces=F, process=False).area)
    data = f"cloth_vbd_{spec.name}.npz"
    extra = {} if birth.get("rest_angle") is None else dict(birth_rest_angle=np.asarray(birth["rest_angle"], np.float32))
    np.savez(builder.out / data, rest=rest, faces=faces, birth=born[node], node=node, particle_rest=P,
             particle_faces=PF, corners=corners, weights=weights, **extra)
    builder.entities[spec.name] = dict(
        kind="pbd_cloth", root=None, links=[], joints=[], free_joint=None, collision=True,
        cloth=dict(format=VBD_FORMAT, file=data, mesh="visual", vertices=int(len(rest)), particles=int(len(P)),
                   area_m2=area, mass_kg=float(s["density_kg_m2"]) * area, settings=json.loads(json.dumps({k: v for k, v in s.items() if k != "two_way"})),
                   versions=versions(s),
                   asset_sha256=hashlib.sha256(Path(spec.asset_path).read_bytes()).hexdigest(),
                   birth=dict(source_sha256=birth.get("source_sha256"), archive_sha256=birth.get("archive_sha256"),
                              state_sha=birth.get("state_sha"), settle=birth.get("settle"),
                              particles_sha256=hashlib.sha256(born.tobytes()).hexdigest(),
                              off_surface_max_m=float(np.linalg.norm(
                                  np.einsum("nk,nki->ni", weights, born[node][corners]) - born, axis=1).max()))))


# ----------------------------------------------------------------------------------------------- runtime
_PLANE, _HFIELD, _SPHERE, _CAPSULE, _ELLIPSOID, _CYLINDER, _BOX, _MESH = 0, 1, 2, 3, 4, 5, 6, 7


def _xyzw(q_wxyz):
    q = np.asarray(q_wxyz, dtype=float)
    return (float(q[1]), float(q[2]), float(q[3]), float(q[0]))


def _rotate(q_wxyz, v):
    w, x, y, z = (float(c) for c in q_wxyz)
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=float)
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def _static_body(m, b) -> bool:
    """A body that cannot move: it and its ancestors have no dofs, and none is a mocap body."""
    while b > 0:
        if int(m.body_dofnum[b]) or int(m.body_mocapid[b]) >= 0:
            return False
        b = int(m.body_parentid[b])
    return True


class ClothVBD:
    """One VBD cloth of a built MuJoCo world: the Newton model, its colliders, the particle view, the pins."""

    steps_outside = True        # world.advance steps it after each mj_step (tick_begin / before_step / after_step)

    def __init__(self, world, name: str, record: dict, scene_dir):
        from topogym.world.mujoco.warp_cpu import init
        init()                  # the pinned compile target and the image's kernel cache, before newton loads
        import warp as wp
        import newton
        self.world, self.name, self.record = world, name, record
        info = record["cloth"]
        self.settings = s = {**VBD_DEFAULTS, **info["settings"]}
        placement = dict((getattr(world, "profile", None) or {}).get("cloth") or {})
        if "device" in placement:
            s.update({k: placement[k] for k in PLACEMENT_KEYS if k in placement})
        self.device = device = str(s.get("device", "cpu"))
        if _gpu(device):
            if not wp.get_cuda_device_count():
                raise RuntimeError(f"VBD cloth {name}: its profile runs on {device}, but Warp sees no CUDA device")
            from topogym.world.mujoco.vbd_gpu import fuse_fp_off
            fuse_fp_off()        # before any Newton kernel of this process is loaded
        with np.load(Path(scene_dir) / info["file"], allow_pickle=False) as data:
            self.rest = data["rest"].copy()
            self.faces = data["faces"].copy()
            self.birth = data["birth"].copy()
            self.corners, self.weights = data["corners"].copy(), data["weights"].copy()
            self.particle_rest = data["particle_rest"].copy()
            self.node_particles = data["node"].copy()
            # the birth's bending rest angles (a plastic bending model changes them: they are state), else the flat rest
            self.birth_rest_angle = data["birth_rest_angle"].copy() if "birth_rest_angle" in data.files else None
        self.flex = None
        self.bodies = np.zeros(0, dtype=np.int64)       # no MuJoCo bodies carry the cloth
        self.body_set = frozenset()
        self.tick_start = None
        m, d = world.model, world.data
        self.origin = np.array([self.rest[:, 0].mean(), self.rest[:, 1].mean(), self.rest[:, 2].min()])
        b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0., 0., float(m.opt.gravity[2])))
        b.add_cloth_mesh(pos=(0., 0., 0.), rot=wp.quat_identity(), scale=1., vel=(0., 0., 0.),
                         vertices=[wp.vec3(*v) for v in (self.rest - self.origin)],
                         indices=np.asarray(self.faces).reshape(-1).tolist(), density=float(s["density_kg_m2"]),
                         tri_ke=float(s["tri_ke"]), tri_ka=float(s["tri_ka"]), tri_kd=float(s["tri_kd"]),
                         edge_ke=float(s["edge_ke"]), edge_kd=float(s["edge_kd"]),
                         particle_radius=float(s["radius_m"]))
        # friction: VBD mixes the cloth's and a shape's coefficients by their geometric mean
        self_mu = float(s["self_mu"])
        shape_mu = lambda pair: float(pair) ** 2 / self_mu        # noqa: E731

        def cfg(mu):
            return newton.ModelBuilder.ShapeConfig(mu=shape_mu(mu), ke=float(s["contact_ke"]), kd=float(s["contact_kd"]),
                                                   has_shape_collision=False, has_particle_collision=True)
        import mujoco
        mujoco.mj_kinematics(m, d)
        self.collider_bodies = []            # MuJoCo body ids of the kinematic Newton bodies, in Newton order
        self._newton_body = {}
        self.colliders = []                  # (MuJoCo geom, kind, Newton body or -1)
        for g in range(int(m.ngeom)):
            if not (int(m.geom_contype[g]) or int(m.geom_conaffinity[g])):
                continue
            gt, mb = int(m.geom_type[g]), int(m.geom_bodyid[g])
            if gt in (_PLANE, _HFIELD):     # the ground lies under the table: a sheet that falls off ends the episode
                continue
            static = _static_body(m, mb)
            if static:
                pos, q = np.asarray(d.geom_xpos[g]) - self.origin, np.zeros(4)
                mujoco.mju_mat2Quat(q, np.asarray(d.geom_xmat[g]))
                body, mu = -1, float(s["table_mu"])
            else:
                if mb not in self._newton_body:
                    self._newton_body[mb] = b.add_body(xform=wp.transform((0., 0., 0.), wp.quat_identity()), mass=0.,
                                                       is_kinematic=True,
                                                       label=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, mb) or f"body{mb}")
                    self.collider_bodies.append(mb)
                pos, q = np.asarray(m.geom_pos[g]), np.asarray(m.geom_quat[g])
                body, mu = self._newton_body[mb], float(s["robot_mu"])
            xf = wp.transform(wp.vec3(*pos), wp.quat(*_xyzw(q)))
            size = np.asarray(m.geom_size[g], dtype=float)
            if gt == _SPHERE:
                b.add_shape_sphere(body, xform=xf, radius=float(size[0]), cfg=cfg(mu))
            elif gt == _CAPSULE:
                b.add_shape_capsule(body, xform=xf, radius=float(size[0]), half_height=float(size[1]), cfg=cfg(mu))
            elif gt == _CYLINDER:
                b.add_shape_cylinder(body, xform=xf, radius=float(size[0]), half_height=float(size[1]), cfg=cfg(mu))
            elif gt == _BOX:
                b.add_shape_box(body, xform=xf, hx=float(size[0]), hy=float(size[1]), hz=float(size[2]), cfg=cfg(mu))
            elif gt == _MESH:
                from scipy.spatial import ConvexHull
                k = int(m.geom_dataid[g])
                a, n = int(m.mesh_vertadr[k]), int(m.mesh_vertnum[k])
                verts = np.asarray(m.mesh_vert[a:a + n], dtype=np.float64)
                hull = ConvexHull(verts)
                tri = hull.simplices.copy()
                c = verts[hull.vertices].mean(axis=0)            # outward winding
                nrm = np.cross(verts[tri[:, 1]] - verts[tri[:, 0]], verts[tri[:, 2]] - verts[tri[:, 0]])
                flip = np.einsum("ij,ij->i", nrm, verts[tri[:, 0]] - c) < 0
                tri[flip] = tri[flip][:, [0, 2, 1]]
                used, inv = np.unique(tri.reshape(-1), return_inverse=True)
                mesh = newton.Mesh(verts[used].astype(np.float32), inv.astype(np.int32), compute_inertia=False)
                b.add_shape_convex_hull(body, xform=xf, mesh=mesh, cfg=cfg(mu))
            else:
                raise NotImplementedError(f"VBD cloth: collider geom type {gt} is not supported")
            self.colliders.append((g, gt, body))
        b.color()
        self.model = model = b.finalize(device=device)
        model.soft_contact_ke = float(s["contact_ke"])
        model.soft_contact_kd = float(s["contact_kd"])
        model.soft_contact_mu = self_mu
        self.pipe = newton.CollisionPipeline(model, soft_contact_gap=float(s["body_gap_m"]))
        if _gpu(device):
            fuse_fp_off()        # the modules the pipeline imported
        self.contacts = self.pipe.contacts()
        self.solver = self._solver_class(s["solver"])(
            model, iterations=int(s["iterations"]), integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=True, particle_self_contact_margin=float(s["self_margin_m"]),
            particle_self_contact_gap=float(s["self_gap_m"]),
            particle_topological_contact_filter_threshold=int(s["topo_filter"]),
            particle_rest_shape_contact_exclusion_radius=float(s["rest_exclusion_m"]),
            particle_vertex_contact_buffer_size=int(s["vt_buffer"]),
            particle_edge_contact_buffer_size=int(s["ee_buffer"]), friction_epsilon=float(s["friction_eps"]),
            **dict(s.get("solver_kwargs") or {}))
        self.step_fn = self.solver.step                  # the one seam (lane S: a faster SolverVBD subclass)
        self.s0, self.s1 = model.state(), model.state()
        self.ctrl = model.control()
        self.mass0 = model.particle_mass.numpy().copy()
        self.inv_mass0 = model.particle_inv_mass.numpy().copy()
        self.rest_angle0 = model.edge_rest_angle.numpy().copy()     # the flat sheet's
        self.n = int(len(self.rest))
        # the pin force: the membrane forces on the pinned vertices (stable Neo-Hookean, as Newton evaluates them)
        self._tri = model.tri_indices.numpy().astype(np.int64)
        self._tri_pose = model.tri_poses.numpy().astype(np.float64)
        self._tri_area = model.tri_areas.numpy().astype(np.float64)
        self._tri_mat = model.tri_materials.numpy().astype(np.float64)
        self._vertex_tris = [[] for _ in range(self.n)]
        for t, (i, j, k) in enumerate(self._tri):
            for order, v in enumerate((i, j, k)):
                self._vertex_tris[int(v)].append((t, order))
        self._g = np.asarray(m.opt.gravity, dtype=np.float64)
        self.pins = {}                 # hand -> dict(body, idx (vertex ids), offsets (hand frame))
        self.pin_force = {}            # hand -> the force the hand puts on the cloth after the last step [N]
        self._xfrc = {}                # MuJoCo body -> the reaction applied to it now (6,)
        self._poses0 = None
        self.step_hooks = []           # callables (cloth, dt) after every cloth step (the stress observer's books)
        self._version = 0              # bumped by every change of the cloth state (the readout caches below)
        self._vertex_cache = self._rest_cache = None
        self._particle_cache = []
        self.set_vertices(self.rest, np.zeros_like(self.rest))
        self.rebuild_bvh()
        # two-way coupling: a coupling setting of the world's profile (not the scene's material), default off
        # (the world profile's only: a scene records its material, never the coupling, so a scene built on the GPU
        # profile runs one-way on blanket_vbd)
        two_way = dict((getattr(world, "profile", None) or {}).get("cloth") or {}).get("two_way")
        two_way = two_way if isinstance(two_way, dict) else ({} if two_way is True else None)   # a dict (even {}) = on
        self.coupling = None
        if two_way is not None:
            from .cloth_coupling import TwoWayCoupling
            self.coupling = TwoWayCoupling(self, two_way)

    @staticmethod
    def _solver_class(spec: str):
        if spec in (None, "newton"):
            from newton.solvers import SolverVBD
            return SolverVBD
        module, _, cls = spec.partition(":")
        return getattr(importlib.import_module(module), cls)

    # -------------------------------------------------------------- the particle view
    def vertices(self, d=None) -> np.ndarray:
        """The grid vertices now (world, float64). One device read per cloth state (_version): every state change goes
        through this class (set_vertices, after_step, restore, unpin), and each call returns its own copy."""
        if self._vertex_cache is None or self._vertex_cache[0] != self._version:
            self._vertex_cache = (self._version, self.s0.particle_q.numpy().astype(np.float64) + self.origin)
        return self._vertex_cache[1].copy()

    def velocities(self) -> np.ndarray:
        return self.s0.particle_qd.numpy().astype(np.float64)

    def max_vertex_speed(self, d=None) -> float:
        return float(np.linalg.norm(self.velocities(), axis=1).max())

    def particles_of(self, vertices) -> np.ndarray:
        """The particles of grid vertices (n, 3). The last two float64 inputs' results are kept by their bytes (a tick's
        readout skins its end vertices and its start vertices, the previous tick's end); each call returns its own copy."""
        x = np.asarray(vertices)
        key = x.tobytes() if x.dtype == np.float64 and x.shape == (self.n, 3) else None
        if key is not None:
            for k, p in self._particle_cache:
                if k == key:
                    return p.copy()
        p = np.einsum("nk,nki->ni", self.weights, x[self.corners])
        if key is not None:
            self._particle_cache = [(key, p.copy()), *self._particle_cache[:1]]
        return p

    def set_vertices(self, X, V=None) -> None:
        self._version += 1
        self.s0.particle_q.assign((np.asarray(X, dtype=np.float64) - self.origin).astype(np.float32))
        self.s0.particle_qd.assign(np.zeros((self.n, 3), np.float32) if V is None
                                   else np.asarray(V, dtype=np.float32))

    def apply_birth(self, d=None) -> None:
        """Every vertex at its frozen birth position, at rest, with the birth's bending rest angles."""
        self.set_vertices(self.birth)
        self.model.edge_rest_angle.assign(self.rest_angle0 if self.birth_rest_angle is None else self.birth_rest_angle)
        self._version += 1
        self.rebuild_bvh()

    def rest_angles(self) -> np.ndarray:
        """The bending rest angles now; one device read per cloth state (_version), as vertices."""
        if self._rest_cache is None or self._rest_cache[0] != self._version:
            self._rest_cache = (self._version, self.model.edge_rest_angle.numpy().copy())
        return self._rest_cache[1].copy()

    def top_z_at(self, xy, radius_m: float = .012) -> np.ndarray:
        """The cloth's local stack top for the Session's table guard: for each world xy point (n, 2), the highest grid
        vertex centre z (world) within radius_m of it in xy (default: one grid spacing); -inf where no vertex is that
        near. The surface lies contact_radius_m (the profile's radius_m) above a vertex centre. Exact maxima over the
        step's current state (one device read), so it is deterministic and the same on the CPU and GPU steps."""
        q = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        X = self.vertices()
        out = np.full(len(q), -np.inf)
        r2 = float(radius_m) ** 2
        for a in range(0, len(q), 256):            # (256, n_vertices) blocks
            d2 = ((q[a:a + 256, None, :] - X[None, :, :2]) ** 2).sum(axis=2)
            z = np.where(d2 <= r2, X[None, :, 2], -np.inf)
            out[a:a + 256] = z.max(axis=1)
        return out

    @property
    def contact_radius_m(self) -> float:
        return float(self.settings["radius_m"])

    def particle_state(self, d, dt) -> dict:
        """The particles (positions, tick velocities) and the bending rest angles (state under plastic bending, so the
        state hash covers them)."""
        x = self.vertices()
        p = self.particles_of(x)
        start = x if self.tick_start is None else self.tick_start
        v = (p - self.particles_of(start)) / float(dt)
        return dict(position_m=p, velocity_m_s=v, rest_angle_rad=self.rest_angles().astype(np.float64))

    # -------------------------------------------------------------- stepping (world.advance)
    def rebuild_bvh(self) -> None:
        self.solver.rebuild_bvh(self.s0)

    def tick_begin(self) -> None:
        """At every tick start: the self-contact BVH rebuilt (every tick boundary is an exact restore point)."""
        self.rebuild_bvh()

    def _poses(self, d) -> np.ndarray:
        q = np.zeros((int(self.model.body_count), 7), np.float32)
        q[:, 6] = 1.
        for k, mb in enumerate(self.collider_bodies):
            q[k, :3] = np.asarray(d.xpos[mb]) - self.origin
            w = np.asarray(d.xquat[mb])
            q[k, 3:] = (w[1], w[2], w[3], w[0])
        return q

    def before_step(self, m, d) -> None:
        """Before a cloth step's MuJoCo steps: the colliders' start poses; the lagged pin reactions (and, two-way, the
        cloth's contact wrenches) into xfrc_applied, held over those steps."""
        dk = d if self.coupling is None else self.coupling.fresh(d)
        self._poses0 = self._poses(dk)
        for mb, f in list(self._xfrc.items()):
            d.xfrc_applied[mb] -= f
        self._xfrc = {}
        for hand, pin in self.pins.items():
            f = self.pin_force.get(hand)
            if f is None:
                continue
            mb = pin["body"]
            x = self.vertices()[pin["idx"]]
            com = np.asarray(d.xipos[mb])
            # the cloth pulls the hand with -f (f: the hand on the cloth), split over the pinned vertices
            share = -np.asarray(f) / len(pin["idx"])
            wrench = np.concatenate([-np.asarray(f), np.cross(x - com, share).sum(axis=0)])
            d.xfrc_applied[mb] += wrench
            self._xfrc[mb] = self._xfrc.get(mb, 0.) + wrench
        if self.coupling is not None:
            for mb, wrench in self.coupling.begin_step(dk).items():
                d.xfrc_applied[mb] += wrench
                self._xfrc[mb] = self._xfrc.get(mb, 0.) + wrench

    def after_step(self, m, d, dt) -> None:
        """After an mj_step: one cloth step over dt (colliders from their start to their end pose; pins at the hand's
        end pose), then the pin forces (two-way: and the cloth's wrenches for the next cloth step)."""
        self._version += 1
        dk = d
        if self.coupling is not None:
            dk = self.coupling.fresh(d)
            self.coupling.end_step(dk)
        if self.model.body_count:
            self.s0.body_q.assign(self._poses0)
            self.s1.body_q.assign(self._poses(dk))
        if self.pins:
            q = self.s0.particle_q.numpy()
            for pin in self.pins.values():
                pos, quat = np.asarray(dk.xpos[pin["body"]]), np.asarray(dk.xquat[pin["body"]])
                q[pin["idx"]] = (pos + np.array([_rotate(quat, o) for o in pin["offsets"]]) - self.origin).astype(np.float32)
            self.s0.particle_q.assign(q)
        self.s0.clear_forces()
        self.pipe.collide(self.s0, self.contacts)
        self.step_fn(self.s0, self.s1, self.ctrl, self.contacts, float(dt))
        self.s0, self.s1 = self.s1, self.s0
        self._version += 1
        if self.pins:
            X = self.vertices()
            for hand, pin in self.pins.items():
                self.pin_force[hand] = self._hand_force(X, pin["idx"])
        if self.coupling is not None:
            self.coupling.update(d, dk, float(dt))
        for hook in self.step_hooks:
            hook(self, float(dt))

    def _hand_force(self, X, idx) -> np.ndarray:
        """The force the hand puts on the cloth: minus the membrane forces and the weight of the pinned vertices."""
        f = np.zeros(3)
        for v in idx:
            for t, order in self._vertex_tris[int(v)]:
                f += self._membrane_force(X, t, order)
            f += self.mass0[int(v)] * self._g
        return -f

    def _membrane_force(self, X, t, order) -> np.ndarray:
        """Newton's stable Neo-Hookean membrane force on vertex `order` of triangle t (particle_vbd_kernels.
        evaluate_neo_hookean_membrane_force_hessian, without damping)."""
        i, j, k = self._tri[t]
        x01, x02 = X[j] - X[i], X[k] - X[i]
        Dm = self._tri_pose[t]
        f0 = x01 * Dm[0, 0] + x02 * Dm[1, 0]
        f1 = x01 * Dm[0, 1] + x02 * Dm[1, 1]
        a, bb, c = f0 @ f0, f1 @ f1, f0 @ f1
        J = np.sqrt(max(a * bb - c * c, 1e-20))
        mu, lam = self._tri_mat[t, 0], self._tri_mat[t, 1] + self._tri_mat[t, 0]
        lam = np.sign(lam) * max(abs(lam), 1e-6)
        g0, g1 = (bb * f0 - c * f1) / J, (a * f1 - c * f0) / J
        sv = lam * (J - (1. + mu / lam))
        P0, P1 = mu * f0 + sv * g0, mu * f1 + sv * g1
        m0, m1, m2 = float(order == 0), float(order == 1), float(order == 2)
        d0 = Dm[0, 0] * (m1 - m0) + Dm[1, 0] * (m2 - m0)
        d1 = Dm[0, 1] * (m1 - m0) + Dm[1, 1] * (m2 - m0)
        return -(P0 * d0 + P1 * d1) * self._tri_area[t]

    # -------------------------------------------------------------- pins (the assisted grasp)
    def pin(self, hand: str, body: int, idx) -> None:
        """Pin grid vertices idx to MuJoCo body `body` (the hand link) at their current offsets in its frame."""
        d = self.world.data
        idx = np.asarray(sorted(int(v) for v in idx), dtype=np.int64)
        pos, quat = np.asarray(d.xpos[body]), np.asarray(d.xquat[body])
        inv = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
        X = self.vertices()
        offsets = np.array([_rotate(inv, X[v] - pos) for v in idx])
        self.pins[hand] = dict(body=int(body), idx=idx, offsets=offsets)
        self.pin_force[hand] = np.zeros(3)
        self._apply_masses()

    def unpin(self, hand: str) -> None:
        pin = self.pins.pop(hand)
        self._version += 1
        self.pin_force.pop(hand, None)
        self._apply_masses()
        qd = self.s0.particle_qd.numpy()
        qd[pin["idx"]] = 0.
        self.s0.particle_qd.assign(qd)

    def _apply_masses(self) -> None:
        m, im = self.mass0.copy(), self.inv_mass0.copy()
        for pin in self.pins.values():
            m[pin["idx"]], im[pin["idx"]] = 0., 0.
        self.model.particle_mass.assign(m)
        self.model.particle_inv_mass.assign(im)

    # -------------------------------------------------------------- checkpoints
    def capture(self) -> dict:
        return dict(q=self.s0.particle_q.numpy().copy(), qd=self.s0.particle_qd.numpy().copy(),
                    rest_angle=self.rest_angles(),
                    pins={h: dict(body=p["body"], idx=p["idx"].copy(), offsets=p["offsets"].copy())
                          for h, p in self.pins.items()},
                    pin_force={h: np.array(f) for h, f in self.pin_force.items()},
                    xfrc={int(b): np.array(f) for b, f in self._xfrc.items()},
                    tick_start=None if self.tick_start is None else self.tick_start.copy(),
                    plastic_loss_j=self._plastic_loss(),
                    two_way=None if self.coupling is None else self.coupling.capture())

    def _plastic_loss(self):
        """A plastic bending step's cumulative dissipation [J], else None."""
        plastic = getattr(self.solver, "plastic", None)
        return None if plastic is None else float(plastic.loss_j)

    def restore(self, state: dict) -> None:
        self._version += 1
        self.s0.particle_q.assign(state["q"])
        self.s0.particle_qd.assign(state["qd"])
        self.model.edge_rest_angle.assign(state["rest_angle"])
        self.pins = {h: dict(body=p["body"], idx=p["idx"].copy(), offsets=p["offsets"].copy())
                     for h, p in state["pins"].items()}
        self.pin_force = {h: np.array(f) for h, f in state["pin_force"].items()}
        self._xfrc = {int(b): np.array(f) for b, f in state["xfrc"].items()}
        self.tick_start = None if state["tick_start"] is None else state["tick_start"].copy()
        if state.get("plastic_loss_j") is not None:
            self.solver.plastic.loss_j = float(state["plastic_loss_j"])
        if self.coupling is not None:
            self.coupling.restore(state["two_way"])
        self._apply_masses()
        self.rebuild_bvh()
