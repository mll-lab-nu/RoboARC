"""Energy books of a Newton VBD cloth (newton.solvers.SolverVBD, Apache-2.0) for the stress observer, the way the rope
chain's rod energy is booked (topogym/world/mujoco/rope_chain.py rope_energy): numpy, float64, from Newton's own
formulas (newton 1.6.1, newton/_src/solvers/vbd/particle_vbd_kernels.py and rigid_vbd_kernels.py), so that minus the
gradient of every potential is the elastic force VBD applies (checked by finite differences: vbd_fd_check).

Potentials (each a function of the particle positions, with the contact sets frozen as VBD freezes them for a step):
  kinetic      1/2 sum m |v|^2 over the simulated particles (a pinned vertex has mass 0: it belongs to its pin)
  gravity      -sum m g . x
  membrane     per triangle, area * psi, the stable Neo-Hookean membrane of evaluate_neo_hookean_membrane_force_hessian:
               psi = mu/2 (I_c - 2) + lambda'/2 (J_s - alpha)^2, I_c = tr(F^T F), J_s = sqrt(det(F^T F)),
               lambda' = lambda + mu, alpha = 1 + mu / lambda' (mu, lambda = tri_materials[:, 0:2]), less its rest value
               mu^2 / (2 lambda') (a constant: 98.4 J over the blanket_v1 sheet), so that it is 0 at rest
  bending      per interior edge, 1/2 k (theta - theta_rest)^2, k = edge_ke * rest length, theta the signed dihedral angle
               of evaluate_dihedral_angle_based_bending_force_hessian
  self         per vertex-triangle pair and per (unordered) edge-edge pair of VBD's collision buffers closer than the
               interaction distance r (particle_self_contact_margin), the barrier of evaluate_self_contact_force_norm:
               1/2 k (r - d)^2 for d >= r/2, k tau^2 (1/2 - ln(d / tau)) for 1e-5 < d < r/2 (tau = r/2), and its
               quadratic extension below 1e-5 (k = soft_contact_ke)
  shape        per particle-shape contact of the collision pipeline, 1/2 ke p^2 with the penetration
               p = r_i + margin - n . (x_i - b) > 0 (b the contact's body point, n its normal: _eval_body_particle_contact),
               split into static shapes (body -1: the table, the fence) and robot shapes (kinematic bodies)
Not potentials (VBD's dissipation, booked as loss by the observer): the membrane, bending and contact damping (tri_kd,
edge_kd, soft_contact_kd), the smoothed Coulomb friction of every contact, and the implicit integrator's own loss."""
from __future__ import annotations

import numpy as np

D_MIN = 1.0e-5          # evaluate_self_contact_force_norm's quadratic-extension threshold


# --------------------------------------------------------------------------------------------- elementary energies
def kinetic(mass, vel):
    return 0.5 * float(np.sum(mass * np.einsum("ij,ij->i", vel, vel)))


def gravity(mass, pos, g=(0., 0., -9.81)):
    return -float(np.sum(mass * (pos @ np.asarray(g, float))))


def membrane(pos, tri, dm_inv, area, mu, lmbd, per_element=False):
    x0 = pos[tri[:, 0]]
    x01 = pos[tri[:, 1]] - x0
    x02 = pos[tri[:, 2]] - x0
    f0 = x01 * dm_inv[:, 0, 0, None] + x02 * dm_inv[:, 1, 0, None]
    f1 = x01 * dm_inv[:, 0, 1, None] + x02 * dm_inv[:, 1, 1, None]
    a, b, c = (np.einsum("ij,ij->i", f0, f0), np.einsum("ij,ij->i", f1, f1), np.einsum("ij,ij->i", f0, f1))
    js = np.sqrt(np.maximum(a * b - c * c, 1.0e-20))
    lam = lmbd + mu
    lam_safe = np.sign(lam) * np.maximum(np.abs(lam), 1.0e-6)
    alpha = 1.0 + mu / lam_safe
    # minus the rest value mu^2 / (2 lambda'): the energy is 0 at rest (where the stress vanishes) and >= 0 near it
    psi = 0.5 * mu * (a + b - 2.0) + 0.5 * lam * (js - alpha) ** 2 - 0.5 * lam * (1.0 - alpha) ** 2
    e = area * psi
    return e if per_element else float(e.sum())


def dihedral_angles(pos, edges):
    """theta per edge (edges: (k, 4) = opposite 0, opposite 1, edge start, edge end; Newton's sign convention);
    nan where the edge is a boundary edge or degenerate (no force there)."""
    ok = (edges[:, 0] >= 0) & (edges[:, 1] >= 0)
    e4 = np.where(ok[:, None], edges, 0)
    x0, x1, x2, x3 = (pos[e4[:, i]] for i in range(4))
    n1 = np.cross(x2 - x0, x3 - x0)
    n2 = np.cross(x3 - x1, x2 - x1)
    e = x3 - x2
    l1, l2, le = (np.linalg.norm(v, axis=1) for v in (n1, n2, e))
    ok &= (l1 >= 1e-6) & (l2 >= 1e-6) & (le >= 1e-6)
    with np.errstate(invalid="ignore", divide="ignore"):
        n1h, n2h, eh = n1 / l1[:, None], n2 / l2[:, None], e / le[:, None]
        sin = np.einsum("ij,ij->i", np.cross(n1h, n2h), eh)
        cos = np.einsum("ij,ij->i", n1h, n2h)
        theta = np.arctan2(sin, cos)
    return np.where(ok, theta, np.nan)


def bending(pos, edges, rest_angle, rest_length, stiffness, per_element=False):
    theta = dihedral_angles(pos, edges)
    k = stiffness * rest_length
    e = np.where(np.isfinite(theta) & (stiffness > 0), 0.5 * k * (np.nan_to_num(theta) - rest_angle) ** 2, 0.0)
    return e if per_element else float(e.sum())


def self_contact_potential(d, r, k):
    """The barrier of evaluate_self_contact_force_norm as an energy of the distance d (zero at d >= r)."""
    d = np.asarray(d, float)
    tau = 0.5 * r
    k2 = k * tau * tau
    quad = 0.5 * k * (r - d) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        log = k2 * (0.5 - np.log(np.maximum(d, D_MIN) / tau))
    ext = k2 * (0.5 * d * d - 2.0 * D_MIN * d) / (D_MIN * D_MIN) + k2 * (2.0 - np.log(D_MIN / tau))
    return np.where(d >= r, 0.0, np.where(d >= tau, quad, np.where(d > D_MIN, log, ext)))


# --------------------------------------------------------------------------------------------- distances
def point_triangle_distance(p, a, b, c):
    """Distance from each point p to its triangle (a, b, c), all (n, 3) (Ericson's closest-point regions)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    q = np.empty_like(p)
    done = np.zeros(len(p), bool)

    def put(mask, val):
        m = mask & ~done
        q[m] = val[m]
        done[m] = True
    put((d1 <= 0) & (d2 <= 0), a)
    put((d3 >= 0) & (d4 <= d3), b)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = d1 / (d1 - d3)
        put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v[:, None] * ab)
        put((d6 >= 0) & (d5 <= d6), c)
        w = d2 / (d2 - d6)
        put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w[:, None] * ac)
        w2 = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + w2[:, None] * (c - b))
        denom = 1.0 / (va + vb + vc)
        vv, ww = vb * denom, vc * denom
        put(np.ones(len(p), bool), a + ab * vv[:, None] + ac * ww[:, None])
    return np.linalg.norm(p - q, axis=1)


def segment_segment_distance(p1, q1, p2, q2, eps=1e-12):
    """Closest distance between segments p1q1 and p2q2 (each (n, 3)), with clamping (Ericson 5.1.9). The parallel
    fallback (s = 0) only for exactly parallel edges, as Warp's closest_point_edge_edge (denom != 0): an absolute
    threshold sent near-parallel cloth edges (sin < ~0.01 at 8 mm: denom ~ a e sin^2 < 1e-12) to s = 0, a non-closest
    pair whose gradient puts the pair's force on the wrong vertices (lane D FD, steps 1900 / 3500). eps stays the
    degenerate-edge tolerance."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = np.einsum("ij,ij->i", d1, d1), np.einsum("ij,ij->i", d2, d2), np.einsum("ij,ij->i", d2, r)
    c, b = np.einsum("ij,ij->i", d1, r), np.einsum("ij,ij->i", d1, d2)
    denom = a * e - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(denom > 0, np.clip((b * f - c * e) / np.where(denom > 0, denom, 1.0), 0, 1), 0.0)
        t = (b * s + f) / np.where(e > eps, e, 1.0)
        s = np.where(t < 0, np.clip(-c / np.where(a > eps, a, 1.0), 0, 1), np.where(t > 1, np.clip((b - c) / np.where(a > eps, a, 1.0), 0, 1), s))
        t = np.clip(t, 0, 1)
    c1, c2 = p1 + d1 * s[:, None], p2 + d2 * t[:, None]
    return np.linalg.norm(c1 - c2, axis=1)


# --------------------------------------------------------------------------------------------- the books
class VBDBooks:
    """The cloth's potentials from a finalized newton Model (numpy copies; the contact sets come per call)."""

    def __init__(self, model, solver=None):
        self.tri = model.tri_indices.numpy().astype(np.int64)
        self.dm_inv = model.tri_poses.numpy().astype(np.float64)
        self.area = model.tri_areas.numpy().astype(np.float64)
        mat = model.tri_materials.numpy().astype(np.float64)
        self.mu, self.lmbd = mat[:, 0], mat[:, 1]
        self.edges = model.edge_indices.numpy().astype(np.int64)
        self.rest_angle = model.edge_rest_angle.numpy().astype(np.float64)
        self.rest_length = model.edge_rest_length.numpy().astype(np.float64)
        self.edge_ke = model.edge_bending_properties.numpy()[:, 0].astype(np.float64)
        self.radius = model.particle_radius.numpy().astype(np.float64)
        self.gravity_vec = np.asarray(model.gravity.numpy()[0] if hasattr(model.gravity, "numpy") else model.gravity,
                                      float).reshape(-1)[:3]
        self.contact_ke = float(model.soft_contact_ke)
        self.shape_body = model.shape_body.numpy()
        sm = getattr(model, "shape_margin", None)
        self.shape_margin = sm.numpy().astype(np.float64) if sm is not None and sm.shape[0] else np.zeros(len(self.shape_body))
        self.self_r = float(solver.particle_self_contact_margin) if solver is not None and getattr(solver, "particle_enable_self_contact", False) else None
        self.solver = solver

    # ----- contact sets (as VBD holds them for a step)
    def self_pairs(self):
        """(vertex-triangle pairs (n, 2), unordered edge-edge pairs (m, 2)) from the solver's collision buffers."""
        info = self.solver.trimesh_collision_detector.collision_info

        def rows(flat, offsets, sizes, counts):
            flat, off, cap, cnt = (np.asarray(x.numpy()) for x in (flat, offsets, sizes, counts))
            out = []
            for i in range(len(cnt)):
                n = min(int(cnt[i]), int(cap[i]))
                if n:
                    seg = flat[2 * off[i]:2 * (off[i] + n)].reshape(-1, 2)
                    out.append(seg)
            return np.concatenate(out) if out else np.zeros((0, 2), np.int64)
        vt = rows(info.vertex_colliding_triangles, info.vertex_colliding_triangles_offsets,
                  info.vertex_colliding_triangles_buffer_sizes, info.vertex_colliding_triangles_count)
        ee = rows(info.edge_colliding_edges, info.edge_colliding_edges_offsets, info.edge_colliding_edges_buffer_sizes,
                  info.edge_colliding_edges_count)
        vt = vt[(vt[:, 0] >= 0) & (vt[:, 1] >= 0)]
        ee = ee[(ee[:, 0] >= 0) & (ee[:, 1] >= 0)]
        ee = np.unique(np.sort(ee, axis=1), axis=0) if len(ee) else ee
        return vt.astype(np.int64), ee.astype(np.int64)

    def shape_contacts(self, contacts):
        """The collision pipeline's particle-shape contacts: (particle, shape, body point b (body frame), normal)."""
        n = int(contacts.soft_contact_count.numpy()[0])
        return (contacts.soft_contact_particle.numpy()[:n].astype(np.int64), contacts.soft_contact_shape.numpy()[:n].astype(np.int64),
                contacts.soft_contact_body_pos.numpy()[:n].astype(np.float64), contacts.soft_contact_normal.numpy()[:n].astype(np.float64))

    # ----- potentials
    def self_energy(self, pos, vt, ee, per_pair=False):
        r, k = self.self_r, self.contact_ke
        if r is None:
            return (np.zeros(0), np.zeros(0)) if per_pair else 0.0
        if len(vt):
            t = self.tri[vt[:, 1]]
            dvt = point_triangle_distance(pos[vt[:, 0]], pos[t[:, 0]], pos[t[:, 1]], pos[t[:, 2]])
            evt = self_contact_potential(dvt, r, k)
        else:
            evt = np.zeros(0)
        if len(ee):
            e1, e2 = self.edges[ee[:, 0]], self.edges[ee[:, 1]]
            dee = segment_segment_distance(pos[e1[:, 2]], pos[e1[:, 3]], pos[e2[:, 2]], pos[e2[:, 3]])
            eee = self_contact_potential(dee, r, k)
        else:
            eee = np.zeros(0)
        return (evt, eee) if per_pair else float(evt.sum() + eee.sum())

    def shape_energy(self, pos, contacts_np, body_q, per_contact=False):
        """1/2 ke p^2 per particle-shape contact; body_q (bodies, 7) = (pos, quat xyzw) as Newton's transforms."""
        part, shape, bpos, normal = contacts_np
        if not len(part):
            return (np.zeros(0), np.zeros(0, bool)) if per_contact else (0.0, 0.0)
        body = self.shape_body[shape]
        bx = bpos.copy()
        moving = body >= 0
        if moving.any():
            q = body_q[body[moving]]
            bx[moving] = q[:, :3] + quat_rotate_xyzw(q[:, 3:7], bpos[moving])
        p = self.radius[part] + self.shape_margin[shape] - np.einsum("ij,ij->i", normal, pos[part] - bx)
        e = np.where(p > 0, 0.5 * self.contact_ke * p * p, 0.0)
        if per_contact:
            return e, moving
        return float(e[~moving].sum()), float(e[moving].sum())

    def potentials(self, pos, *, vt=None, ee=None, contacts_np=None, body_q=None):
        out = dict(membrane=membrane(pos, self.tri, self.dm_inv, self.area, self.mu, self.lmbd),
                   bending=bending(pos, self.edges, self.rest_angle, self.rest_length, self.edge_ke))
        if vt is not None:
            out["self"] = self.self_energy(pos, vt, ee)
        if contacts_np is not None:
            out["static"], out["robot"] = self.shape_energy(pos, contacts_np, body_q)
        return out


def quat_rotate_xyzw(q, v):
    """Rotate v (n, 3) by unit quaternions q (n, 4) in (x, y, z, w) order (Newton / Warp transforms)."""
    u, w = q[:, :3], q[:, 3:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


# --------------------------------------------------------------------------------------------- the observer's book
def _contact_kernel():
    """A Warp kernel: each body contact's force on the cloth, as FastSolverVBD's solve_color evaluates it (Newton's
    _eval_body_particle_contact for a particle contact, _eval_soft_ef_contact for an edge / face contact; the total
    over its corners), at the step's end positions against the step's start (the friction anchor)."""
    import warp as wp
    from newton._src.solvers.vbd.rigid_vbd_kernels import _eval_body_particle_contact, _eval_soft_ef_contact

    @wp.kernel
    def k_body_contact_force(n: int, corners_a: wp.array(dtype=wp.vec3i), pk: wp.array(dtype=float),
                             kd: wp.array(dtype=float), mu: wp.array(dtype=float), feps: float,
                             pos: wp.array(dtype=wp.vec3), anchor: wp.array(dtype=wp.vec3), radius: wp.array(dtype=float),
                             shape_body: wp.array(dtype=int), body_q: wp.array(dtype=wp.transform),
                             body_q_prev: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
                             body_com: wp.array(dtype=wp.vec3), c_shape: wp.array(dtype=int),
                             c_bpos: wp.array(dtype=wp.vec3), c_bvel: wp.array(dtype=wp.vec3),
                             c_normal: wp.array(dtype=wp.vec3), margin: wp.array(dtype=float),
                             bary: wp.array(dtype=wp.vec3), dt: float, out: wp.array(dtype=wp.vec3)):
        i = wp.tid()
        if i < n:
            c = corners_a[i]
            if c[1] < 0:
                p = c[0]
                cf, ch = _eval_body_particle_contact(p, pos[p], anchor[p], i, pk[i], kd[i], mu[i], feps, radius,
                                                     shape_body, body_q, body_q_prev, body_qd, body_com, c_shape,
                                                     c_bpos, c_bvel, c_normal, margin, dt)
                out[i] = cf
            else:
                cf, ch, cp = _eval_soft_ef_contact(i, c, bary[i], pos, anchor, radius, pk[i], kd[i], mu[i], feps,
                                                   shape_body, body_q, body_q_prev, body_qd, body_com, c_shape, c_bpos,
                                                   c_bvel, c_normal, margin, dt)
                out[i] = cf
    return k_body_contact_force


class VBDBook:
    """The energy book of one VBD cloth (topogym/world/mujoco/cloth_vbd.ClothVBD) for the stress observer, kept from
    the cloth's step hook (ClothVBD.step_hooks, after every cloth step). Read-only: it never writes the cloth.

    Per cloth step (start state X0 = the solver's particle_q_prev, after the pins were written; end state X1, V1):
      E      the cloth's own energy: kinetic + gravity + membrane + bending (at the step's bending rest angles: under
             plastic bending they are state) + the self-contact barrier (the step's collision buffers, frozen) + the
             body contacts' penalty, 1/2 ke p^2 per contact (static and robot shapes, particle and edge / face
             contacts; the step's contact set, frozen);
      W_robot the robot shapes' work on the cloth: VBD's own contact force (normal + friction + damping, at the
             step's end against its start) times the displacement of the contact's body point over the step. A
             static shape does no work (its point does not move), so its friction and damping are dissipation;
             robot contacts act on the cloth only (the robot does not feel them);
      W_pin  the pins' work: the energy change from writing the pinned vertices to the hand's pose before the step
             (they have mass 0: their weight and membrane pull are the hand's) and from pinning / releasing;
      X      += (E(X1) - E(X0)) - W_robot: the cloth's balance. VBD's damping, friction, implicit integrator,
             penetration truncation and plastic bending all dissipate, so X never rises in a sound run; a rise is
             energy created inside the cloth. plastic_loss_j (the solver's own book) is reported for attribution.
    The pins' reaction on the hand (cloth_vbd's lagged xfrc_applied) is booked on the MuJoCo side by the observer.
    Two-way coupling (cloth_coupling.py): the cloth's contact wrenches on the robot are xfrc_applied too, so the
    observer's MuJoCo work W already holds their work on the robot. The book reports the coupling's own counts
    (cloth_coupling.TwoWayCoupling.work_on_robot_j / work_on_cloth_j since the book began): W_react, the coupling
    wrench's work on the robot, and W_couple, the robot contacts' work on the cloth's free vertices; and the balance
      C      = W_couple + W_react: the energy the lagged, implicit exchange itself creates (> 0) or dissipates (< 0);
             exact action-reaction without lag would give 0. C_max_rise is its largest rise over its running minimum.
      XC     = X + C, the cloth and the coupling together; XC_sustained_rise is the largest rise of XC above its running
             minimum that holds for 4 consecutive cloth steps (10 ms): a contact's onset books end-of-step force x the
             step's displacement on the cloth while the robot pays the reaction one step later (two-way lane, 10-12),
             a one-step spike of C that the next steps pay back.
    X keeps subtracting W_robot (the robot's work on the cloth), so the coupling's work is never the cloth's excess."""

    def __init__(self, cloth):
        self.cloth = cloth
        self.books = VBDBooks(cloth.model, cloth.solver)
        self._kernel = None
        self.reset()

    def reset(self):
        self.prev = None
        self.X = self.X_min = self.X_max_rise = 0.
        self.W_robot = self.W_pin = 0.
        self.W_react = self.W_couple = self.C = self.C_min = self.C_max_rise = 0.
        self.XC_min = self.XC_max_rise = self.XC_sustained = 0.
        self._xc_hist = []
        self._react0 = None
        self.steps = 0
        self.max_step_gain = (0., None)        # the largest single-step rise of X and the step it happened in
        self.plastic0 = None
        self.E_last = None
        import os
        # criteria v2.12: a particle kicked faster than 3 m/s in one step is a pop, a breakdown only if the contact
        # storage (self-contact barrier + body-contact penalties) released into that particle alone gives
        # v_c = sqrt(2 S_released / m) > 3 m/s (v2.11's energy share). Contact-set changes between steps (the energy of
        # the last state on the new sets less on the old) are booked in W_set (outside X, as W_pin).
        self.W_set = 0.
        # v2.12 pressed pile (user 2026-10-13): each cloth step's dX goes to X_press when a gripper was held on its cloth
        # floor within the observer's window (pressed, set before the tick), else to X_free; their rises are judged apart.
        # KE_press_max / kick_press_max: the cloth's kinetic energy and a particle's largest velocity change in pressed steps.
        self.pressed = False
        self.X_free = self.X_free_min = self.X_free_max_rise = 0.
        self.X_press = self.X_press_min = self.X_press_max_rise = 0.
        self.KE_press_max = self.kick_press_max = 0.
        self.press_steps = 0
        self.W_set_max = 0.
        self.pop12 = dict(fast=0, share=0, max_kick=0., max_kick_step=None, max_vc=0., events=[])
        self.trace = [] if os.environ.get("VBD_BOOK_TRACE") else None   # per step: (step, dE,
        # edge / face contacts, W_robot, W_pin step, dX, X, pins, body contacts, VT pairs, EE pairs, kinetic)

    # -- energies
    def _contacts(self):
        """The step's body contacts (frozen): corners (n, 3) (-1 padded), barycentric weights, shape, body point (body
        frame), normal; and whether each one's shape moves."""
        c = self.cloth.contacts
        n = int(c.soft_contact_count.numpy()[0])
        shape = c.soft_contact_shape.numpy()[:n].astype(np.int64)
        return dict(n=n, corners=c.soft_contact_indices.numpy()[:n].astype(np.int64),
                    bary=c.soft_contact_barycentric.numpy()[:n].astype(np.float64), shape=shape,
                    bpos=c.soft_contact_body_pos.numpy()[:n].astype(np.float64),
                    normal=c.soft_contact_normal.numpy()[:n].astype(np.float64),
                    ke=self.cloth.solver.body_particle_contact_penalty_k.numpy()[:n].astype(np.float64),
                    moving=self.books.shape_body[shape] >= 0)

    def _contact_points(self, X, C):
        """Each contact's cloth point (the barycentric blend of its corners) and radius (the corners' largest)."""
        k = C["corners"]
        ok = k >= 0
        kk = np.maximum(k, 0)
        w = np.where(ok, C["bary"], 0.)
        x = np.einsum("nk,nkj->nj", w, X[kk])
        r = np.where(ok, self.books.radius[kk], -np.inf).max(axis=1)
        return x, r

    def _body_points(self, C, body_q):
        """Each contact's body point in the world (Newton frame) at body poses body_q."""
        bx = C["bpos"].copy()
        mv = C["moving"]
        if mv.any():
            q = body_q[self.books.shape_body[C["shape"][mv]]]
            bx[mv] = q[:, :3] + quat_rotate_xyzw(q[:, 3:7], C["bpos"][mv])
        return bx

    def _shape_energy(self, X, C, body_q):
        """1/2 ke p^2 per body contact (Newton's penalty: p = radius + margin - n . (x - b) > 0), static + robot."""
        if not C["n"]:
            return 0.
        x, r = self._contact_points(X, C)
        p = r + self.books.shape_margin[C["shape"]] - np.einsum("ij,ij->i", C["normal"], x - self._body_points(C, body_q))
        return float(np.sum(np.where(p > 0, 0.5 * C["ke"] * p * p, 0.)))

    def _internal(self, X, V, m, rest_angle, vt, ee, C=None, body_q=None):
        b = self.books
        e = kinetic(m, V) + float(-np.sum(m * (X @ b.gravity_vec)))
        e += membrane(X, b.tri, b.dm_inv, b.area, b.mu, b.lmbd)
        e += bending(X, b.edges, rest_angle, b.rest_length, b.edge_ke)
        if vt is not None:
            e += b.self_energy(X, vt, ee)
        if C is not None:
            e += self._shape_energy(X, C, body_q)
        return e

    def _storage(self, X, vt, ee, C, body_q, n):
        """v2.12 (pop share): the stored contact energy (self-contact barrier + body-contact penalties) on the frozen sets: total and
        per particle (a vertex-triangle or edge-edge pair's energy split equally over its four particles; a body
        contact's over its corners by their barycentric weights)."""
        b = self.books
        per = np.zeros(n)
        if vt is not None and (len(vt) or len(ee)):
            evt, eee = b.self_energy(X, vt, ee, per_pair=True)
            if len(vt):
                t = b.tri[vt[:, 1]]
                for col in (vt[:, 0], t[:, 0], t[:, 1], t[:, 2]):
                    np.add.at(per, col, 0.25 * evt)
            if len(ee):
                e1, e2 = b.edges[ee[:, 0]], b.edges[ee[:, 1]]
                for col in (e1[:, 2], e1[:, 3], e2[:, 2], e2[:, 3]):
                    np.add.at(per, col, 0.25 * eee)
        if C is not None and C["n"]:
            x, r = self._contact_points(X, C)
            p = r + b.shape_margin[C["shape"]] - np.einsum("ij,ij->i", C["normal"], x - self._body_points(C, body_q))
            e = np.where(p > 0, 0.5 * C["ke"] * p * p, 0.)
            k = C["corners"]
            w = np.where(k >= 0, C["bary"], 0.)
            for j in range(k.shape[1]):
                ok = k[:, j] >= 0
                np.add.at(per, k[ok, j], w[ok, j] * e[ok])
        return float(per.sum()), per

    def _body_work(self, X0, X1, dt):
        """The robot shapes' work on the cloth over the step: each contact's force on the cloth (VBD's own, at the
        step's end against its start) times its body point's displacement (a static shape's is 0: it does no work)."""
        import warp as wp
        c, s, m = self.cloth.contacts, self.cloth.solver, self.cloth.model
        n = int(c.soft_contact_count.numpy()[0])
        if not n:
            return 0.
        if self._kernel is None:
            self._kernel = _contact_kernel()
        dev = m.device
        out = wp.zeros(n, dtype=wp.vec3, device=dev)
        pos = wp.array(X1.astype(np.float32), dtype=wp.vec3, device=dev)
        anchor = wp.array(X0.astype(np.float32), dtype=wp.vec3, device=dev)
        cl = self.cloth
        margin = m.shape_margin if getattr(m, "shape_margin", None) is not None and m.shape_margin.shape[0] \
            else wp.zeros(max(1, int(m.shape_count)), dtype=float, device=dev)
        body_q, body_q_prev = cl.s0.body_q, cl.s1.body_q          # after the swap: s0 = the step's end, s1 its start
        wp.launch(self._kernel, dim=n, inputs=[n, c.soft_contact_indices, s.body_particle_contact_penalty_k,
                                                s.body_particle_contact_material_kd, s.body_particle_contact_material_mu,
                                                float(s.friction_epsilon), pos, anchor, m.particle_radius, m.shape_body,
                                                body_q, body_q_prev, cl.s0.body_qd, m.body_com, c.soft_contact_shape,
                                                c.soft_contact_body_pos, c.soft_contact_body_vel, c.soft_contact_normal,
                                                margin, c.soft_contact_barycentric, float(dt), out], device=dev)
        f = out.numpy().astype(np.float64)
        C = self._contacts()
        q0 = cl.s1.body_q.numpy().astype(np.float64)
        q1 = cl.s0.body_q.numpy().astype(np.float64)
        db = self._body_points(C, q1) - self._body_points(C, q0)     # 0 on static shapes
        w = np.einsum("ij,ij->i", f, db)
        return float(w[C["moving"]].sum())

    # -- the hook
    def hook(self, cloth, dt):
        m = cloth.model.particle_mass.numpy().astype(np.float64)
        X1 = cloth.s0.particle_q.numpy().astype(np.float64)
        V1 = cloth.s0.particle_qd.numpy().astype(np.float64)
        ra1 = cloth.model.edge_rest_angle.numpy().astype(np.float64)
        X0 = cloth.solver.particle_q_prev.numpy().astype(np.float64)
        vt, ee = self.books.self_pairs() if self.books.self_r is not None else (None, None)
        plastic = getattr(cloth.solver, "plastic", None)
        if self.prev is None:
            self.prev = (X1, V1, m, ra1)
            self.plastic0 = None if plastic is None else float(plastic.loss_j)
            self.E_last = self._internal(X1, V1, m, ra1, vt, ee, self._contacts(), cloth.s0.body_q.numpy().astype(np.float64))
            return
        Xp, Vp, mp, ra0 = self.prev
        V0 = cloth.s1.particle_qd.numpy().astype(np.float64)       # the step's input state (after a release's zeroing)
        C = self._contacts()
        q0 = cloth.s1.body_q.numpy().astype(np.float64)               # the step's start poses (s1 after the swap)
        q1 = cloth.s0.body_q.numpy().astype(np.float64)
        e_prev = self._internal(Xp, Vp, mp, ra0, vt, ee, C, q0)      # the last step's end, this step's contact sets
        e0 = self._internal(X0, V0, m, ra0, vt, ee, C, q0)           # after the pins were written (and pin changes)
        e1 = self._internal(X1, V1, m, ra1, vt, ee, C, q1)
        self.W_pin += e0 - e_prev
        ws = 0.
        wr = self._body_work(X0, X1, dt)
        self.W_robot += wr
        dx = (e1 - e0) - wr
        # v2.12: the contact-set changes booked; a fast particle judged by the storage released into it
        if self.E_last is not None:
            dset = e_prev - self.E_last
            self.W_set += dset
            self.W_set_max = max(self.W_set_max, abs(dset))
        live = m > 0
        if live.any():
            kick = np.linalg.norm(V1 - V0, axis=1)
            kick[~live] = 0.
            k_max = int(np.argmax(kick))
            if kick[k_max] > self.pop12["max_kick"]:
                self.pop12.update(max_kick=float(kick[k_max]), max_kick_step=self.steps + 1)
            fast = np.flatnonzero(kick > 3.0).tolist()
            if fast:
                _, per0 = self._storage(X0, vt, ee, C, q0, len(m))
                _, per1 = self._storage(X1, vt, ee, C, q1, len(m))
            for i in fast:
                rel = max(0., float(per0[i] - per1[i]))
                vc = (2. * rel / float(m[i])) ** .5
                self.pop12["max_vc"] = max(self.pop12["max_vc"], vc)
                key = "fast" if vc > 3.0 else "share"
                self.pop12[key] += 1
                if len(self.pop12["events"]) < 12:
                    self.pop12["events"].append(dict(step=self.steps + 1, particle=int(i), kick=round(float(kick[i]), 4),
                                                     released_j=round(rel, 9), vc=round(vc, 4), kind=key))
        if self.pressed:
            self.press_steps += 1
            self.X_press += dx
            self.X_press_min = min(self.X_press_min, self.X_press)
            self.X_press_max_rise = max(self.X_press_max_rise, self.X_press - self.X_press_min)
            self.KE_press_max = max(self.KE_press_max, float(kinetic(m, V1)))
            if live.any():
                self.kick_press_max = max(self.kick_press_max, float(kick.max()))
        else:
            self.X_free += dx
            self.X_free_min = min(self.X_free_min, self.X_free)
            self.X_free_max_rise = max(self.X_free_max_rise, self.X_free - self.X_free_min)
        self.X += dx
        self.steps += 1
        if dx > self.max_step_gain[0]:
            self.max_step_gain = (dx, self.steps)
        self.X_min = min(self.X_min, self.X)
        self.X_max_rise = max(self.X_max_rise, self.X - self.X_min)
        tw = getattr(cloth, "coupling", None)
        if tw is not None:                       # the coupling's balance (its work on the robot since the book began)
            if self._react0 is None:
                self._react0 = (float(tw.work_on_robot_j), float(tw.work_on_cloth_j))
            self.W_react = float(tw.work_on_robot_j) - self._react0[0]
            self.W_couple = float(tw.work_on_cloth_j) - self._react0[1]
            self.C = self.W_couple + self.W_react
            self.C_min = min(self.C_min, self.C)
            self.C_max_rise = max(self.C_max_rise, self.C - self.C_min)
            xc = self.X + self.C
            self.XC_min = min(self.XC_min, xc)
            self.XC_max_rise = max(self.XC_max_rise, xc - self.XC_min)
            self._xc_hist = (self._xc_hist + [xc - self.XC_min])[-4:]
            if len(self._xc_hist) == 4:
                self.XC_sustained = max(self.XC_sustained, min(self._xc_hist))
        self.prev = (X1, V1, m, ra1)
        self.E_last = e1
        if self.trace is not None:
            self.trace.append((self.steps, e1 - e0, int((C["corners"][:, 1] >= 0).sum()), wr, e0 - e_prev, dx, self.X, len(cloth.pins),
                               int(cloth.contacts.soft_contact_count.numpy()[0]), 0 if vt is None else len(vt),
                               0 if ee is None else len(ee), float(kinetic(m, V1))))

    def summary(self):
        if self.trace is not None:
            import os
            np.save(os.environ["VBD_BOOK_TRACE"], np.asarray(self.trace, dtype=np.float64))
        plastic = getattr(self.cloth.solver, "plastic", None)
        pl = None if plastic is None or self.plastic0 is None else float(plastic.loss_j) - self.plastic0
        return dict(steps=self.steps, X=self.X, X_max_rise=self.X_max_rise, max_step_gain=self.max_step_gain[0],
                    max_step_gain_step=self.max_step_gain[1], W_robot=self.W_robot, W_pin=self.W_pin,
                    plastic_loss_j=pl, E=self.E_last,
                    W_set=self.W_set, W_set_max_step=self.W_set_max, pop12=self.pop12,
                    X_free_max_rise=self.X_free_max_rise, X_press_max_rise=self.X_press_max_rise,
                    KE_press_max=self.KE_press_max, kick_press_max=self.kick_press_max, press_steps=self.press_steps,
                    **({} if getattr(self.cloth, "coupling", None) is None else
                       dict(W_react=self.W_react, W_couple=self.W_couple, C=self.C, C_max_rise=self.C_max_rise,
                            XC_max_rise=self.XC_max_rise, XC_sustained_rise=self.XC_sustained)))
