"""Image build (Dockerfile.mujoco): compile the VBD cloth's Warp CPU kernels into the image's cache (WARP_CACHE_PATH), with
the pinned compile target of topogym/world/mujoco/warp_cpu.py. Warp compiles a module the first time one of its kernels
runs, so this steps a cloth through every path the cloth uses: the blanket grid (51 x 41 vertices, 12 mm, one split
diagonal) with self-contact, a static table box, kinematic bodies carrying every collider kind of
topogym/world/mujoco/cloth_vbd.py (box, sphere, capsule, cylinder, a mesh's convex hull) moving through the sheet,
pinned (mass 0) vertices, the collision pipeline, VBD steps and a self-contact BVH rebuild, with Newton's SolverVBD and
with our multi-threaded step (topogym/world/mujoco/vbd_fast.py) at 1 and 4 threads with plastic bending. Warp names a
module after its Python module, so the fast step is imported from that path, as the tasks import it. A new Warp module
must be stepped here too, or each process compiles it again (into a per-user cache: warp_cpu.init()).
"""
import numpy as np

from topogym.world.mujoco import warp_cpu

warp_cpu.init()

import warp as wp  # noqa: E402

import newton  # noqa: E402
from newton.solvers import SolverVBD  # noqa: E402

from topogym.world.mujoco.vbd_fast import FastSolverVBD  # noqa: E402

NX, NY, H = 51, 41, 0.012
rest = np.array([(i * H, j * H, 0.) for i in range(NX) for j in range(NY)], float)
rest[:, :2] -= rest[:, :2].mean(axis=0)
faces = []
for i in range(NX - 1):
    for j in range(NY - 1):
        a, b, c, d = i * NY + j, (i + 1) * NY + j, (i + 1) * NY + j + 1, i * NY + j + 1
        faces += [a, b, c, a, c, d]
CUBE = np.array([(x, y, z) for x in (-.01, .01) for y in (-.01, .01) for z in (-.01, .01)], np.float32)
CUBE_TRI = np.array([0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5,
                     3, 7, 5], np.int32)
KW = dict(iterations=5, integrate_with_external_rigid_solver=True, particle_enable_self_contact=True,
          particle_self_contact_margin=0.007, particle_self_contact_gap=0.0014,
          particle_topological_contact_filter_threshold=1, particle_rest_shape_contact_exclusion_radius=0.,
          particle_vertex_contact_buffer_size=32, particle_edge_contact_buffer_size=48, friction_epsilon=1e-2)


def build():
    """A fresh model per solver (a solver run leaves state in the model and the pipeline that would carry over)."""
    b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0., 0., -9.81))
    b.add_cloth_mesh(pos=(0., 0., 0.003), rot=wp.quat_identity(), scale=1., vel=(0., 0., 0.),
                     vertices=[wp.vec3(*v) for v in rest], indices=faces, density=0.5, tri_ke=1270., tri_ka=1090.,
                     tri_kd=1e-4, edge_ke=6.5e-3, edge_kd=1e-5, particle_radius=0.003)
    cfg = newton.ModelBuilder.ShapeConfig(mu=6.667, ke=2e4, kd=1e-4, has_shape_collision=False,
                                          has_particle_collision=True)
    I = wp.quat_identity()
    hand = b.add_body(xform=wp.transform((0., 0., 0.05), I), mass=0., is_kinematic=True, label="hand")
    b.add_shape_box(hand, xform=wp.transform((0., 0., 0.), I), hx=0.035, hy=0.012, hz=0.01, cfg=cfg)
    arm = b.add_body(xform=wp.transform((0.2, 0., 0.05), I), mass=0., is_kinematic=True, label="arm")
    b.add_shape_sphere(arm, xform=wp.transform((0., 0., 0.), I), radius=0.01, cfg=cfg)
    b.add_shape_capsule(arm, xform=wp.transform((0.03, 0., 0.), I), radius=0.008, half_height=0.02, cfg=cfg)
    b.add_shape_cylinder(arm, xform=wp.transform((-0.03, 0., 0.), I), radius=0.008, half_height=0.02, cfg=cfg)
    b.add_shape_convex_hull(arm, xform=wp.transform((0., 0.03, 0.), I),       # compiles newton._src.geometry.inertia
                            mesh=newton.Mesh(CUBE, CUBE_TRI, compute_inertia=False), cfg=cfg)
    b.add_shape_box(-1, xform=wp.transform((0., 0., -0.02), I), hx=0.5, hy=0.4, hz=0.02,
                    cfg=newton.ModelBuilder.ShapeConfig(mu=1.6667, ke=2e4, kd=1e-4, has_shape_collision=False,
                                                        has_particle_collision=True))
    b.color()
    model = b.finalize(device="cpu")
    model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = 2e4, 1e-4, 0.4
    return model


def drive(make_solver):
    """Six steps: the bodies move down into the sheet, a patch is pinned at step 2, the BVH is rebuilt at step 3."""
    model = build()
    pipe = newton.CollisionPipeline(model, soft_contact_gap=0.004)
    contacts = pipe.contacts()
    solver = make_solver(model)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    mass = model.particle_mass.numpy()
    x0 = np.array([0., 0.2], np.float32)
    for k in range(6):
        if k == 2:                               # pin a patch (kinematic vertices), as a grasp does
            pinned = mass.copy()
            pinned[:20] = 0.
            model.particle_mass.assign(pinned)
            model.particle_inv_mass.assign(np.where(pinned > 0, 1. / np.maximum(pinned, 1e-12), 0.))
        q = np.zeros((model.body_count, 7), np.float32)
        q[:, 0], q[:, 2], q[:, 6] = x0, 0.03 - 0.005 * k, 1.      # the bodies move down into the sheet
        s0.body_q.assign(q)
        q[:, 2] -= 0.005
        s1.body_q.assign(q)
        s0.clear_forces()
        pipe.collide(s0, contacts)
        solver.step(s0, s1, ctrl, contacts, 0.0025)
        s0, s1 = s1, s0
        if k == 3:
            solver.rebuild_bvh(s0)
    if hasattr(solver, "close"):
        solver.close()
    x = s0.particle_q.numpy()
    assert np.isfinite(x).all()
    return x


drive(lambda model: SolverVBD(model, **KW))
plastic = dict(yield_angle=0.6, rate=0.02)          # the blanket's fast step, with crease memory (plastic bending) on
fast = [drive(lambda model, n=n: FastSolverVBD(model, threads=n, plastic=plastic, **KW)) for n in (1, 4)]
assert np.array_equal(fast[0], fast[1]), "FastSolverVBD: 1 and 4 threads differ"
print("warp kernels compiled into", wp.config.kernel_cache_dir)
