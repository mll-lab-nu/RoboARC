# SPDX-License-Identifier: Apache-2.0
# The kernel evaluates Newton's own contact and elasticity functions (newton 1.6.1, newton/_src/solvers/vbd,
# Copyright (c) 2025 The Newton Developers, Apache-2.0).
"""Two-way cloth-robot coupling for the VBD cloth (cloth_vbd.ClothVBD, profile cloth key `two_way`): the force the cloth
puts on each robot body is applied back to MuJoCo, so a gripper pressing a stack is held up by it instead of crushing
it against the table (blanket easier4 directed_e0: 227 N on a one-way coupled stack, layer gaps 5 -> 1.3 mm, a 2.9 kN
barrier). Default off: with two_way False nothing here runs and no existing run changes.

Per cloth step (after the solver's step, at its end positions against its start, as the solver's last iteration
evaluates them):
  1. Per body contact (the collision pipeline's particle and edge / face contacts on a robot shape): Newton's force on
     the cloth f (penalty + damping + friction), its Hessian H, the rigid contact point b. Only the share acting on free
     vertices counts (barycentric weights of the corners with mass > 0): a pinned vertex is the hand's (its reaction is
     the pin force, cloth_vbd._hand_force). Plus the contact's effective stiffness (below).
  2. Per MuJoCo body, a fixed-order sum (contacts sorted by (shape, corners), as lane S's GPU step orders body contacts;
     np.add.at sequential, float64): the wrench F at the body's centre of mass and the 6x6 stiffness K.
  3. The wrench for the next MuJoCo step G: linearly implicit in the robot's motion over that step (below), capped so
     that no arm joint feels more than cap_factor x its actuator's torque limit from the cloth (clamp, never fail).
Deterministic: the kernel has no atomics and no cross-thread sums (one thread per contact; a particle's own body contacts
in key order), FMA contraction off in this module (vbd_gpu's rule), the host sums in a fixed order. CPU and GPU give
the same algorithm; their bits differ as their cloth steps do.

The effective stiffness (why not the penalty's ke). The explicit lagged force is unstable once k dt^2 / m > ~4: ke 2e4 N/m
per contact, 50-100 contacts under a gripper base (1.8 kg along z) or a finger (0.18 kg along its slide) at dt 2.5 ms
cross it. But the penalty alone over-predicts: a hanging sheet gives way, a stack is springs in series. Each contact's
normal stiffness is therefore the penalty in series with the cloth's own support at the contact point, from the
particle's Hessian of the step energy without this contact (inertia m/dt^2, membrane, bending, its other body and table
contacts, self-contacts: exactly what the solver's vertex solve assembles), one level deep (the next layer counts as
fixed: an upper bound, which only makes the implicit step more damped). A hanging corner then has ~ m/dt^2 (12 N/m per
vertex: no false wall), a sheet on the table ~ ke/2, a pinched layer ~ ke/2. Friction enters K too (`tangential`),
but its linearization (the smoothed Coulomb friction's Hessian, mu f_n / (friction_eps dt), friction_eps 1e-3 m/s) holds
only within a 2.5 um slip: unbounded, it predicted lateral wrenches far beyond what the stack can resist (e0's first
impact: G_x -63 N under F_x -1 N while the arm moved sideways). So the tangential part of G is capped at
max(|F_t|, mu_cap G_n), mu_cap the weakest pair friction the cloth has (default min(robot, self, table) = 0.4: a stack
dragged sideways slips at its weakest interface), the correction's torque scaled alike. Without friction in K (explicit
friction) the pressed gripper chattered: on e0 the coupling created +0.31 J, G changed ~1 N step to step, 10 um jitter.

The implicit wrench. G is held over the next cloth step's k MuJoCo steps of h = dt / k (profile mujoco_substeps; dt the
cloth step, 2.5 ms; k = 4 on blanket_vbd_gpu, 1 on blanket_vbd). With F the cloth's wrench (at the body's pose where the
cloth step ended), K its stiffness, W = J Mt^-1 J^T the bodies' mobility (J: centre of mass Jacobians; Mt = M + h D_j,
implicitfast's metric with the joint damping and actuator kv) and a = J Mt^-1 (M qacc - J^T G1) the last MuJoCo step's
non-cloth acceleration (lagged), semi-implicit Euler over the k steps moves a body by dt v + c dt^2 (W G + a) with
c = (k + 1) / (2 k), and ends at velocity v + dt (W G + a). G is the cloth's force there (F - K dx - D v_end):
    (I + dt (c dt K + D) W) G = F - (dt K + D) v - dt (c dt K + D) a.
At rest (v = 0, qacc = 0, G1 = G) it gives G = F: the statics are the cloth's own. D = damping_s * K (a Rayleigh-like
contact damping, default 0). The force never turns into a pull along the body's contact normal n (the stiffness-weighted
mean of its contacts' normals): if G . n < 0 <= F . n the correction is shortened until G . n = 0 (the cloth can only
push, plus friction; e0's first impact, G . F alone: G_z down to -9 N under F_z +44 N, the friction's sideways part
keeping G . F > 0).

Books (cumulative, and per cloth step as step_*): `work_on_robot_j`, the coupling wrench's work on the robot over a
cloth step's MuJoCo steps (G . dx_com + tau . dtheta, from end_step); `work_on_cloth_j`, the robot contacts' work on the
cloth's free vertices over the cloth step (force . body-point displacement, from update). Their sum is the exchange's own
energy error (the lag), to be booked apart from the cloth's own balance (stress/vbd_books.py, the observer)."""
from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.geometry import ParticleFlags
from newton._src.solvers.vbd.particle_vbd_kernels import (
    evaluate_dihedral_angle_based_bending_force_hessian, evaluate_edge_edge_contact,
    evaluate_neo_hookean_membrane_force_hessian, evaluate_vertex_triangle_collision_force_hessian)
from newton._src.solvers.vbd.rigid_vbd_kernels import _eval_body_particle_contact, _eval_soft_ef_contact
from newton._src.utils.mesh import (MeshAdjacencyData, get_vertex_adjacent_edge_id_order,
                                    get_vertex_adjacent_face_id_order, get_vertex_num_adjacent_edges,
                                    get_vertex_num_adjacent_faces)

wp.set_module_options({"enable_backward": False, "fuse_fp": False})

DEFAULTS = dict(
    two_way=False,      # the cloth's contact forces on robot bodies applied back to MuJoCo
    cap_factor=2.0,     # cloth torque on any actuated arm joint <= cap_factor x its actuator's torque limit
    damping_s=0.0,      # contact damping D = damping_s * K [s]
    tangential=True,    # friction in the implicit stiffness too, its correction within the friction cap (below)
    friction_cap=None,  # |G_t| <= max(|F_t|, cap * G_n); None: the weakest of the cloth's pair frictions (robot, self, table)
)


@wp.func
def _contact_key(c: int, contact_shape: wp.array[int], corners_arr: wp.array[wp.vec3i]):
    cr = corners_arr[c]
    return ((wp.int64(contact_shape[c]) + wp.int64(1)) << wp.int64(48)) | (wp.int64(cr[0] + 1) << wp.int64(32)) | (
        wp.int64(cr[1] + 1) << wp.int64(16)) | wp.int64(cr[2] + 1)


@wp.func
def particle_hessian(
    p: int, dt: float, pos_anchor: wp.array[wp.vec3], pos: wp.array[wp.vec3], mass: wp.array[float],
    tri_indices: wp.array2d[wp.int32], tri_poses: wp.array[wp.mat22], tri_materials: wp.array2d[float],
    tri_areas: wp.array[float], edge_indices: wp.array2d[wp.int32], edge_rest_angles: wp.array[float],
    edge_rest_length: wp.array[float], edge_bending_properties: wp.array2d[float], adjacency: MeshAdjacencyData,
    friction_epsilon: float, particle_radius: wp.array[float], body_particle_contact_indices: wp.array[wp.vec3i],
    particle_contact_head: wp.array[int], particle_contact_next: wp.array[int],
    body_particle_contact_penalty_k: wp.array[float], body_particle_contact_material_kd: wp.array[float],
    body_particle_contact_material_mu: wp.array[float], shape_body: wp.array[int], body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
    contact_shape: wp.array[int], contact_body_pos: wp.array[wp.vec3], contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3], shape_margin: wp.array[float], contact_barycentric: wp.array[wp.vec3],
    inc_offsets: wp.array[wp.int32], inc_entries: wp.array[wp.vec4i], collision_radius: float,
    soft_contact_ke: float, soft_contact_kd: float, friction_mu: float, edge_edge_parallel_epsilon: float,
):
    """The vertex solve's Hessian of particle p (vbd_fast.solve_color's h: inertia, membrane, bending, body contacts in
    (shape, corners) order, gathered self-contacts) at the step's end positions."""
    h = mass[p] / (dt * dt) * wp.identity(n=3, dtype=float)
    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, p)):
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, p, i_adj_tri)
        if tri_materials[tri_index, 0] > 0.0 or tri_materials[tri_index, 1] > 0.0:
            f_tri, h_tri = evaluate_neo_hookean_membrane_force_hessian(
                tri_index, vertex_order, pos, pos_anchor, tri_indices, tri_poses[tri_index], tri_areas[tri_index],
                tri_materials[tri_index, 0], tri_materials[tri_index, 1], tri_materials[tri_index, 2], dt)
            h = h + h_tri
    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, p)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, p, i_adj_edge)
        if edge_bending_properties[nei_edge_index, 0] > 0.0:
            f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                nei_edge_index, vertex_order_on_edge, pos, pos_anchor, edge_indices, edge_rest_angles,
                edge_rest_length, edge_bending_properties[nei_edge_index, 0],
                edge_bending_properties[nei_edge_index, 1], dt)
            h = h + h_edge
    head = particle_contact_head[p]
    lastk = wp.int64(-1)
    while True:
        node = int(-1)
        bestk = wp.int64(9223372036854775807)
        cursor = head
        while cursor >= 0:
            kc = _contact_key(cursor // 3, contact_shape, body_particle_contact_indices)
            if kc > lastk and kc < bestk:
                bestk = kc
                node = cursor
            cursor = particle_contact_next[cursor]
        if node < 0:
            break
        lastk = bestk
        ci = node // 3
        corner = node - 3 * ci
        corners = body_particle_contact_indices[ci]
        if corners[1] < 0:
            cf, ch = _eval_body_particle_contact(
                p, pos[p], pos_anchor[p], ci, body_particle_contact_penalty_k[ci], body_particle_contact_material_kd[ci],
                body_particle_contact_material_mu[ci], friction_epsilon, particle_radius, shape_body, body_q,
                body_q_prev, body_qd, body_com, contact_shape, contact_body_pos, contact_body_vel, contact_normal,
                shape_margin, dt)
            h = h + ch
        else:
            bary = contact_barycentric[ci]
            cf, ch, _cp = _eval_soft_ef_contact(
                ci, corners, bary, pos, pos_anchor, particle_radius, body_particle_contact_penalty_k[ci],
                body_particle_contact_material_kd[ci], body_particle_contact_material_mu[ci], friction_epsilon,
                shape_body, body_q, body_q_prev, body_qd, body_com, contact_shape, contact_body_pos, contact_body_vel,
                contact_normal, shape_margin, dt)
            w = bary[corner]
            h = h + (w * w) * ch
    for k in range(inc_offsets[p], inc_offsets[p + 1]):
        e = inc_entries[k]
        if e[0] == 0:
            cf, ch = evaluate_vertex_triangle_collision_force_hessian(
                e[1], e[3], e[2], pos, pos_anchor, tri_indices, collision_radius, soft_contact_ke, soft_contact_kd,
                friction_mu, friction_epsilon, dt)
        else:
            cf, ch = evaluate_edge_edge_contact(
                p, e[3], e[1], e[2], pos, pos_anchor, edge_indices, collision_radius, soft_contact_ke,
                soft_contact_kd, friction_mu, friction_epsilon, dt, edge_edge_parallel_epsilon)
        h = h + ch
    return h


@wp.func
def _series(k: float, s: float):
    """Two springs in series (k: the contact, s: the cloth's support); 0 if either is not positive."""
    out = float(0.0)
    if k > 0.0 and s > 0.0:
        out = k * s / (k + s)
    return out


@wp.kernel
def reaction_terms(
    n_contacts: int, dt: float,
    pos_anchor: wp.array[wp.vec3], pos: wp.array[wp.vec3], mass: wp.array[float], particle_flags: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32], tri_poses: wp.array[wp.mat22], tri_materials: wp.array2d[float],
    tri_areas: wp.array[float], edge_indices: wp.array2d[wp.int32], edge_rest_angles: wp.array[float],
    edge_rest_length: wp.array[float], edge_bending_properties: wp.array2d[float], adjacency: MeshAdjacencyData,
    friction_epsilon: float, particle_radius: wp.array[float], body_particle_contact_indices: wp.array[wp.vec3i],
    particle_contact_head: wp.array[int], particle_contact_next: wp.array[int],
    body_particle_contact_penalty_k: wp.array[float], body_particle_contact_material_kd: wp.array[float],
    body_particle_contact_material_mu: wp.array[float], shape_body: wp.array[int], body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
    contact_shape: wp.array[int], contact_body_pos: wp.array[wp.vec3], contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3], shape_margin: wp.array[float], contact_barycentric: wp.array[wp.vec3],
    inc_offsets: wp.array[wp.int32], inc_entries: wp.array[wp.vec4i], collision_radius: float,
    soft_contact_ke: float, soft_contact_kd: float, friction_mu: float, edge_edge_parallel_epsilon: float,
    out_body: wp.array[wp.int32], out_point: wp.array[wp.vec3], out_force: wp.array[wp.vec3],
    out_normal: wp.array[wp.vec3], out_kn: wp.array[float], out_kt: wp.array[float], out_disp: wp.array[wp.vec3],
):
    """Per body contact i on a moving (robot) shape: the force the cloth puts on the body (minus Newton's force on the
    free corners), the rigid contact point (Newton frame), the normal, and the effective normal / tangential stiffness
    (the contact in series with the free corners' support). Static shapes and contacts on pinned vertices: body -1."""
    i = wp.tid()
    if i >= n_contacts:
        return
    out_body[i] = -1
    out_force[i] = wp.vec3(0.0)
    out_kn[i] = 0.0
    out_kt[i] = 0.0
    shape = contact_shape[i]
    body = shape_body[shape]
    if body < 0:
        return
    corners = body_particle_contact_indices[i]
    n = contact_normal[i]
    X_wb = body_q[body]
    bx = wp.transform_point(X_wb, contact_body_pos[i])
    out_disp[i] = bx - wp.transform_point(body_q_prev[body], contact_body_pos[i])   # the body point over the step
    cf = wp.vec3(0.0)
    ch = wp.mat33(0.0)
    bary = wp.vec3(1.0, 0.0, 0.0)
    if corners[1] < 0:
        p = corners[0]
        cf, ch = _eval_body_particle_contact(
            p, pos[p], pos_anchor[p], i, body_particle_contact_penalty_k[i], body_particle_contact_material_kd[i],
            body_particle_contact_material_mu[i], friction_epsilon, particle_radius, shape_body, body_q, body_q_prev,
            body_qd, body_com, contact_shape, contact_body_pos, contact_body_vel, contact_normal, shape_margin, dt)
    else:
        bary = contact_barycentric[i]
        cf, ch, bx = _eval_soft_ef_contact(
            i, corners, bary, pos, pos_anchor, particle_radius, body_particle_contact_penalty_k[i],
            body_particle_contact_material_kd[i], body_particle_contact_material_mu[i], friction_epsilon, shape_body,
            body_q, body_q_prev, body_qd, body_com, contact_shape, contact_body_pos, contact_body_vel, contact_normal,
            shape_margin, dt)
    # the free corners' share, and their support at the contact point: compliance sum_k w_k^2 / s_k along n and in the
    # tangent plane (s_k: the corner's Hessian without this contact's own term, projected; at least its inertia)
    wfree = float(0.0)
    cn = float(0.0)
    ct = float(0.0)
    for k in range(3):
        v = corners[k]
        w = bary[k]
        if v >= 0 and w != 0.0:
            if (particle_flags[v] & ParticleFlags.ACTIVE) and mass[v] > 0.0:
                wfree += w
                h = particle_hessian(
                    v, dt, pos_anchor, pos, mass, tri_indices, tri_poses, tri_materials, tri_areas, edge_indices,
                    edge_rest_angles, edge_rest_length, edge_bending_properties, adjacency, friction_epsilon,
                    particle_radius, body_particle_contact_indices, particle_contact_head, particle_contact_next,
                    body_particle_contact_penalty_k, body_particle_contact_material_kd,
                    body_particle_contact_material_mu, shape_body, body_q, body_q_prev, body_qd, body_com,
                    contact_shape, contact_body_pos, contact_body_vel, contact_normal, shape_margin,
                    contact_barycentric, inc_offsets, inc_entries, collision_radius, soft_contact_ke, soft_contact_kd,
                    friction_mu, edge_edge_parallel_epsilon)
                hs = h - (w * w) * ch
                m_in = mass[v] / (dt * dt)
                sn = wp.max(wp.dot(n, hs * n), m_in)
                st = wp.max(0.5 * (hs[0, 0] + hs[1, 1] + hs[2, 2] - wp.dot(n, hs * n)), m_in)
                cn += w * w / sn
                ct += w * w / st
    if wfree <= 0.0:
        return
    kcn = wfree * wp.dot(n, ch * n)
    kct = wfree * 0.5 * (ch[0, 0] + ch[1, 1] + ch[2, 2] - wp.dot(n, ch * n))
    out_body[i] = body
    out_point[i] = bx
    out_force[i] = -wfree * cf
    out_normal[i] = n
    out_kn[i] = _series(kcn, 1.0 / cn)
    out_kt[i] = _series(kct, 1.0 / ct)


def _skew(r):
    """(n, 3) -> (n, 3, 3): skew(r) x = r x x."""
    S = np.zeros((len(r), 3, 3))
    S[:, 0, 1], S[:, 0, 2] = -r[:, 2], r[:, 1]
    S[:, 1, 0], S[:, 1, 2] = r[:, 2], -r[:, 0]
    S[:, 2, 0], S[:, 2, 1] = -r[:, 1], r[:, 0]
    return S


def _rotvec(R0, R1):
    """The rotation vector of R1 R0^T (world frame), for small rotations exact to second order."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R1 @ R0.T).as_rotvec()


class TwoWayCoupling:
    """The reaction path of one ClothVBD: per cloth step the wrenches for the next MuJoCo step (pending), the work they
    do on the robot (work_on_robot_j), diagnostics (last)."""

    def __init__(self, cloth, settings: dict):
        import mujoco
        self.cloth = cloth
        self.s = {**DEFAULTS, **settings}
        solver = cloth.solver
        for attr in ("inc_offsets", "inc_entries", "_particle_contact_head", "_particle_contact_next", "particle_q_prev"):
            if getattr(solver, attr, None) is None:
                raise ValueError(f"two-way cloth coupling needs the deterministic VBD step (vbd_fast / vbd_gpu): the "
                                 f"solver has no {attr}")
        m = cloth.world.model
        self.mujoco = mujoco
        self.body_of = np.asarray(cloth.collider_bodies, dtype=np.int64)      # Newton body -> MuJoCo body
        # actuator torque limits per dof (joint transmissions; a force-limited actuator: |gear| x max |forcerange|)
        tau = np.full(m.nv, np.inf)
        for a in range(m.nu):
            if int(m.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                continue
            j = int(m.actuator_trnid[a, 0])
            dof = int(m.jnt_dofadr[j])
            if not int(m.actuator_forcelimited[a]):
                continue
            lim = abs(float(m.actuator_gear[a, 0])) * float(np.max(np.abs(m.actuator_forcerange[a])))
            tau[dof] = lim if not np.isfinite(tau[dof]) else tau[dof] + lim
        self.tau_max = tau
        self.dof_tree = np.asarray(m.body_rootid)[np.asarray(m.dof_bodyid)]
        # implicitfast's velocity metric: M + dt (joint damping + position actuators' kv)
        D = np.asarray(m.dof_damping, dtype=np.float64).copy()
        for a in range(m.nu):
            if int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT) and int(m.actuator_biastype[a]) == 1:
                D[int(m.jnt_dofadr[int(m.actuator_trnid[a, 0])])] += -float(m.actuator_biasprm[a, 2]) * float(m.actuator_gear[a, 0]) ** 2
        self.D = D
        self.dk = mujoco.MjData(m)   # private: the kinematics of the current qpos
        self.pending = {}            # MuJoCo body -> (6,) wrench at its COM for the next MuJoCo step
        self.applied = {}            # MuJoCo body -> (6,) wrench applied in the MuJoCo step under way
        self._pose0 = {}             # MuJoCo body -> (COM, R) at that step's start
        # the coupling's books, for the observer to keep apart from the cloth's own balance (cumulative and per cloth
        # step): the coupling wrench's work on the robot over a cloth step's MuJoCo steps (end_step) and the robot
        # contacts' work on the cloth over the cloth step (update); their sum is the lagged exchange's own energy error
        self.work_on_robot_j = 0.
        self.work_on_cloth_j = 0.
        self.step_work_on_robot_j = 0.
        self.step_work_on_cloth_j = 0.
        self.last = {}               # diagnostics of the last cloth step
        self._out = None

    def fresh(self, d):
        """A private MjData with d's qpos / qvel / mocap and their kinematics and centre-of-mass frames (mj_kinematics,
        mj_comPos): after mj_step, d's own are that step's start's unless the world refreshed them."""
        mujoco, m, dk = self.mujoco, self.cloth.world.model, self.dk
        dk.qpos[:] = d.qpos
        dk.qvel[:] = d.qvel
        if m.nmocap:
            dk.mocap_pos[:] = d.mocap_pos
            dk.mocap_quat[:] = d.mocap_quat
        mujoco.mj_kinematics(m, dk)
        mujoco.mj_comPos(m, dk)
        return dk

    # ------------------------------------------------------------------ device evaluation
    def _arrays(self, n):
        dev = self.cloth.model.device
        if self._out is None or self._out[0].shape[0] < n:
            cap = max(64, 1 << int(np.ceil(np.log2(max(n, 1)))))
            self._out = (wp.zeros(cap, dtype=wp.int32, device=dev), wp.zeros(cap, dtype=wp.vec3, device=dev),
                         wp.zeros(cap, dtype=wp.vec3, device=dev), wp.zeros(cap, dtype=wp.vec3, device=dev),
                         wp.zeros(cap, dtype=float, device=dev), wp.zeros(cap, dtype=float, device=dev),
                         wp.zeros(cap, dtype=wp.vec3, device=dev))
        return self._out

    def contact_terms(self, dt):
        """Per robot contact (host, float64; in (shape, corners) order): MuJoCo body, point (world), force on the body,
        normal, kn, kt."""
        cl, c, s, m = self.cloth, self.cloth.contacts, self.cloth.solver, self.cloth.model
        n = int(c.soft_contact_count.numpy()[0])
        if n == 0:
            return None
        ob, op, of, on, okn, okt, od = self._arrays(n)
        margin = m.shape_margin if getattr(m, "shape_margin", None) is not None and m.shape_margin.shape[0] \
            else wp.zeros(max(1, int(m.shape_count)), dtype=float, device=m.device)
        # after ClothVBD's swap: s0 = the step's end (positions, body poses), s1 = its start; the anchor is the
        # solver's particle_q_prev
        wp.launch(reaction_terms, dim=n, device=m.device, inputs=[
            n, float(dt), s.particle_q_prev, cl.s0.particle_q, m.particle_mass, m.particle_flags, m.tri_indices,
            m.tri_poses, m.tri_materials, m.tri_areas, m.edge_indices, m.edge_rest_angle, m.edge_rest_length,
            m.edge_bending_properties, s.particle_adjacency, float(s.friction_epsilon), m.particle_radius,
            c.soft_contact_indices, s._particle_contact_head, s._particle_contact_next,
            s.body_particle_contact_penalty_k, s.body_particle_contact_material_kd,
            s.body_particle_contact_material_mu, m.shape_body, cl.s0.body_q, cl.s1.body_q, cl.s0.body_qd, m.body_com,
            c.soft_contact_shape, c.soft_contact_body_pos, c.soft_contact_body_vel, c.soft_contact_normal, margin,
            c.soft_contact_barycentric, s.inc_offsets, s.inc_entries, float(s.particle_self_contact_margin),
            float(m.soft_contact_ke), float(m.soft_contact_kd), float(m.soft_contact_mu),
            float(s._self_contact_edge_edge_parallel_epsilon)],
            outputs=[ob, op, of, on, okn, okt, od])
        body = ob.numpy()[:n].astype(np.int64)
        sel = np.flatnonzero(body >= 0)
        if not len(sel):
            return None
        shape = c.soft_contact_shape.numpy()[:n].astype(np.int64)[sel]
        corners = c.soft_contact_indices.numpy()[:n].astype(np.int64)[sel]
        order = np.lexsort((corners[:, 2], corners[:, 1], corners[:, 0], shape))
        sel = sel[order]
        return dict(body=self.body_of[body[sel]], point=op.numpy()[sel].astype(np.float64) + cl.origin,
                    force=of.numpy()[sel].astype(np.float64), normal=on.numpy()[sel].astype(np.float64),
                    kn=okn.numpy()[sel].astype(np.float64), kt=okt.numpy()[sel].astype(np.float64),
                    disp=od.numpy()[sel].astype(np.float64))

    # ------------------------------------------------------------------ host: wrenches and the implicit step
    def update(self, d, dk, dt):
        """After a cloth step: the next MuJoCo step's wrenches. d: the world's data after mj_step (its qvel, qacc); dk:
        fresh(d), the pose the cloth step ended on; dt: the cloth step."""
        mujoco, m = self.mujoco, self.cloth.world.model
        T = self.contact_terms(dt)
        self.pending = {}
        self.last = dict(contacts=0, bodies={})
        self.step_work_on_cloth_j = 0.
        if T is None:
            return
        # the robot contacts' work on the cloth over this cloth step: the force on the free corners times the contact's
        # body-point displacement, summed in the contacts' fixed order (float64, sequential)
        self.step_work_on_cloth_j = float(np.cumsum(-np.einsum("ij,ij->i", T["force"], T["disp"]))[-1])
        self.work_on_cloth_j += self.step_work_on_cloth_j
        bodies = np.unique(T["body"])                       # sorted MuJoCo ids
        slot = np.searchsorted(bodies, T["body"])
        com = np.asarray(dk.xipos)[bodies]
        r = T["point"] - com[slot]
        S = _skew(r)
        f = T["force"]
        Fw = np.zeros((len(bodies), 6))
        np.add.at(Fw, slot, np.concatenate([f, np.einsum("nij,nj->ni", S, f)], axis=1))
        nn = np.einsum("ni,nj->nij", T["normal"], T["normal"])
        kt = T["kt"] if self.s["tangential"] else np.zeros_like(T["kt"])
        Kc = T["kn"][:, None, None] * nn + kt[:, None, None] * (np.eye(3)[None] - nn)
        G = np.concatenate([np.repeat(np.eye(3)[None], len(r), axis=0), S], axis=1)        # (n, 6, 3): f -> wrench
        K6 = np.einsum("nai,nij,nbj->nab", G, Kc, G)
        Kb = np.zeros((len(bodies), 6, 6))
        np.add.at(Kb, slot, K6)
        # the robot's mobility at the bodies' centres of mass
        nb, nv = len(bodies), m.nv
        J = np.zeros((6 * nb, nv))
        for k, b in enumerate(bodies):
            mujoco.mj_jacBodyCom(m, dk, J[6 * k:6 * k + 3], J[6 * k + 3:6 * k + 6], int(b))
        mujoco.mj_crb(m, dk)                                # the mass matrix at the step's end pose
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(m, dk, M)
        h = float(m.opt.timestep)                            # the MuJoCo step; k of them per cloth step dt
        k = max(1, int(round(dt / h)))
        c = (k + 1) / (2. * k)
        Mt = M + h * np.diag(self.D)
        MtiJT = np.linalg.solve(Mt, J.T)
        W = J @ MtiJT
        Gprev = np.concatenate([self.applied.get(int(b), np.zeros(6)) for b in bodies])
        JTGprev = np.zeros(nv)
        for b, w in self.applied.items():
            jp, jr = np.zeros((3, nv)), np.zeros((3, nv))
            mujoco.mj_jacBodyCom(m, dk, jp, jr, int(b))
            JTGprev += jp.T @ w[:3] + jr.T @ w[3:]
        fq = M @ np.asarray(d.qacc) - JTGprev                # the last step's non-cloth forces (lagged)
        v0 = J @ np.asarray(d.qvel)
        a = J @ np.linalg.solve(Mt, fq)
        K = np.zeros((6 * nb, 6 * nb))
        for k in range(nb):
            K[6 * k:6 * k + 6, 6 * k:6 * k + 6] = Kb[k]
        D = float(self.s["damping_s"]) * K
        C1, C2 = dt * K + D, c * dt * K + D
        F = Fw.reshape(-1)
        Gw = np.linalg.solve(np.eye(6 * nb) + dt * C2 @ W, F - C1 @ v0 - dt * C2 @ a)
        # never a pull along the body's contact normal: shorten the correction until G . n = 0
        Gw = Gw.reshape(nb, 6)
        nsum = np.zeros((nb, 3))
        np.add.at(nsum, slot, T["kn"][:, None] * T["normal"])
        for k in range(nb):
            ln = float(np.sqrt(nsum[k] @ nsum[k]))
            if ln <= 0.:
                continue
            n_b = -nsum[k] / ln                              # Newton's normal points at the cloth: the push is -n
            fn, gn = float(Fw[k, :3] @ n_b), float(Gw[k, :3] @ n_b)
            if gn < 0. <= fn:
                a = fn / (fn - gn)                           # in [0, 1): G . n = 0
                Gw[k] = Fw[k] + a * (Gw[k] - Fw[k])
        # friction within the cap: |G_t| <= max(|F_t|, mu_cap G_n) about the body's push direction
        mu_cap = self.s["friction_cap"]
        if mu_cap is None:
            cs = self.cloth.settings
            mu_cap = min(float(cs["robot_mu"]), float(cs["self_mu"]), float(cs["table_mu"]))
        for k in range(nb):
            ln = float(np.sqrt(nsum[k] @ nsum[k]))
            if ln <= 0.:
                continue
            n_b = -nsum[k] / ln
            g, f = Gw[k, :3], Fw[k, :3]
            gt, ft = g - (g @ n_b) * n_b, f - (f @ n_b) * n_b
            bound = max(float(np.sqrt(ft @ ft)), float(mu_cap) * max(float(g @ n_b), 0.))
            lt = float(np.sqrt(gt @ gt))
            if lt > bound:
                sc = bound / lt
                Gw[k, :3] = (g @ n_b) * n_b + sc * gt
                Gw[k, 3:] = Fw[k, 3:] + sc * (Gw[k, 3:] - Fw[k, 3:])
        # the cap: the cloth's torque on each actuated joint within cap_factor x its actuator limit, per kinematic tree
        tau = np.zeros(nv)
        for k in range(nb):
            tau += J[6 * k:6 * k + 6].T @ Gw[k]
        scale_of_tree = {}
        lim = float(self.s["cap_factor"]) * self.tau_max
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.isfinite(lim) & (np.abs(tau) > lim), lim / np.abs(tau), 1.)
        for tree in np.unique(self.dof_tree):
            scale_of_tree[int(tree)] = float(np.min(ratio[self.dof_tree == tree]))
        roots = np.asarray(m.body_rootid)[bodies]
        for k, b in enumerate(bodies):
            sc = scale_of_tree.get(int(roots[k]), 1.)
            self.pending[int(b)] = Gw[k] * sc
            self.last["bodies"][int(b)] = dict(F=Fw[k].copy(), G=self.pending[int(b)].copy(), scale=sc,
                                               kn=float(np.trace(Kb[k][:3, :3])), contacts=int((slot == k).sum()))
        self.last["contacts"] = int(len(slot))

    # ------------------------------------------------------------------ MuJoCo side
    def begin_step(self, d):
        """Before a cloth step's MuJoCo steps (d: fresh kinematics): the wrenches to hold over them (the pending ones)
        and the bodies' start poses for the work."""
        self.applied = {b: w.copy() for b, w in self.pending.items()}
        self._pose0 = {b: (np.asarray(d.xipos[b]).copy(), np.asarray(d.ximat[b]).reshape(3, 3).copy())
                       for b in self.applied}
        return self.applied

    def end_step(self, d):
        """After a cloth step's MuJoCo steps (d: fresh kinematics of their end): the work the held wrenches did on the
        robot over them (COM displacement, rotation; the wrench is constant over the steps)."""
        w = 0.
        for b, (x0, R0) in self._pose0.items():
            G = self.applied[b]
            x1, R1 = np.asarray(d.xipos[b]), np.asarray(d.ximat[b]).reshape(3, 3)
            w += float(G[:3] @ (x1 - x0)) + float(G[3:] @ _rotvec(R0, R1))
        self.work_on_robot_j += w
        self.step_work_on_robot_j = w
        self._pose0 = {}
        return w

    # ------------------------------------------------------------------ checkpoints
    def capture(self):
        return dict(pending={b: w.copy() for b, w in self.pending.items()},
                    applied={b: w.copy() for b, w in self.applied.items()}, work_on_robot_j=self.work_on_robot_j,
                    work_on_cloth_j=self.work_on_cloth_j, step_work_on_robot_j=self.step_work_on_robot_j,
                    step_work_on_cloth_j=self.step_work_on_cloth_j)

    def restore(self, st):
        self.pending = {int(b): np.array(w) for b, w in st["pending"].items()}
        self.applied = {int(b): np.array(w) for b, w in st["applied"].items()}
        self.work_on_robot_j = float(st["work_on_robot_j"])
        self.work_on_cloth_j = float(st["work_on_cloth_j"])
        self.step_work_on_robot_j = float(st["step_work_on_robot_j"])
        self.step_work_on_cloth_j = float(st["step_work_on_cloth_j"])
        self._pose0 = {}
