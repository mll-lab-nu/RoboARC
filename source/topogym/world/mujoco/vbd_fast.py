# SPDX-License-Identifier: Apache-2.0
# Kernels adapted from newton/_src/solvers/vbd/particle_vbd_kernels.py (newton 1.6.1,
# Copyright (c) 2025 The Newton Developers, Apache-2.0).
"""A multi-threaded, deterministic CPU step for Newton's VBD cloth.

`FastSolverVBD` is a drop-in `newton.solvers.SolverVBD`: the same constructor (plus `threads`), the same
`step(state_in, state_out, control, contacts, dt)`, for our configuration (triangle cloth, self-contact on, rigid
bodies integrated outside: `integrate_with_external_rigid_solver=True`, no body-particle penalty ramp).

What runs on the threads (Warp's CPU launch is a ctypes CFUNCTYPE call, which releases the GIL; each launch here is a
pre-packed direct call into the compiled kernel):
  the whole iteration loop of a step, as one program per thread with a barrier between phases; per color:
    A  `solve_color`: per vertex of the color, inertia + membrane + bending + body contacts + self-contacts, the 3x3
       solve, the displacement update. Self-contacts are GATHERED per vertex from an incidence list built after each
       collision detection (each vertex sums its own terms in a fixed order): no atomics, no shared writes.
    B1 `truncation_strided`: Newton's planar truncation per collision primitive, each thread min-reducing into its own
       buffer;
    B2 `truncation_apply_strided`: per vertex, the min over the thread buffers (exact, order-free), Newton's isotropic
       clamp, the position update.
  Threads take interleaved shares (index mod threads) so a localized fold spreads over all of them.
Determinism: every value depends only on fixed list orders and on exact mins, never on the partition or on thread
timing, so any thread count gives the same bits.

Numerics differ from SolverVBD only in the order in which a vertex's self-contact terms are summed (gathered per
vertex rather than scattered per collision entry). Elasticity, body contacts, truncation and integration are Newton's
own functions.

Plastic bending (crease memory), `plastic=dict(yield_angle, rate, max_rest=2.7, skip_pinned=True)`: after each step, per bending edge,
when the dihedral angle is farther than yield_angle [rad] from the edge's rest angle (model.edge_rest_angle), the rest
angle moves toward it by rate * (the excess beyond the yield) [rate per step; 0.02 at dt 2.5 ms is 8 /s], clamped to
+-max_rest. The elastic energy this releases is booked as plastic dissipation: solver.plastic.loss_j [J, cumulative,
summed in index order]. One write per edge, so it is deterministic and thread-count independent. The rest angles are
state: save and restore model.edge_rest_angle (and loss_j for the books) with checkpoints.

Coarse correction (`coarse_every`, default 2; 0 = off): Gauss-Seidel cannot carry the smooth, whole-sheet sliding
mode of a light, stiff sheet against table friction within 20 iterations (a dragged corner needed 35-40 N instead of
~0.7 N, with 40-60 % strain at the pins). Before iterations coarse_every, 2 coarse_every, ... (never on the raw
prediction before iteration 0) one Newton step of the step's energy restricted to a common translation of all free
particles is taken: residual = the sum of the particles' forces (internal and self-contact terms cancel); Hessian =
inertia + body-contact normal blocks + the membrane blocks of triangles that touch fixed (pinned) particles (exact
Galerkin: a triangle's Hessian sums to zero over a rigid translation). Friction against static shapes enters as a
Coulomb capacity sum mu |N|: the sheet slides by the tangential force beyond it only. Single-thread solve, index-order
sums: deterministic. No state across steps.

Patch-wise coarse step (`coarse_patch_hops`, default 4; 0 = the single common translation above): the free particles
are split into patches = mesh-graph balls of that hop radius (greedy in index order, so each patch is a region of
material and the layers of a fold are separate patches). Per coarse step: the exact global step above seeds every
patch, then projected block Gauss-Seidel (`coarse_sweeps`, default 100) over the patch graph solves for per-patch
translations: inertia + contact normal blocks per patch, the exact membrane coupling between patches (a triangle's
cross blocks follow from its diagonal blocks by translation invariance), self-contact normal parts in the residual, and
friction as Coulomb capacities against static shapes and against the patch each patch rests on (its first other-patch
self-contact partner). This lets a top layer slide over the layer below while the bottom stays.
Measured (2.5 ms x 20 iterations, 4 threads): single-corner drag 0.6-0.7 N (35-40 N without), a top-layer corner
dragged over the bottom layer 0.65 / 1.1 N median / peak, a two-layer stack pulled at its crease 2.3 / 2.6 N
(42 / 51 N without); cost about +35 % per step at coarse_every 2.

Adaptive iterations: dict(tol [m], block, max_iterations). After
the step's `iterations`, blocks of `block` more iterations (the same schedule: coarse cycles continue every
coarse_every) run while the last iteration's largest vertex update exceeds tol, up to max_iterations; the test is an
exact max, so it is deterministic, and a step under tol is the plain step. 20 iterations do not converge in a stiff,
crumpled or pressed stack, and the unconverged steps ratchet stored energy up at rest (validator wave 2: easy4
directed_e6 GPU X +0.78 J, hard4 mutate_e3 twin membrane +0.27 J; 80 iterations: dissipative).

Plastic `skip_pinned` (default True): hinges touching a pinned particle (mass 0) do not yield, so a grasp's rigid patch
leaves no crease memory.

Usage rules: every model/state/contacts array is read through pointers recorded once per array object, so update
arrays in place (.assign / copy), never rebind them. close() stops the worker threads (also done at garbage
collection). Options tol / min_iterations (residual stop), truncate_per="iteration" and dedup are experiments that
were rejected (fold collapse, instability, no gain): keep their defaults.
"""
from __future__ import annotations

import ctypes
import os
import threading
import time
import weakref
from collections import defaultdict

import numpy as np
import warp as wp

from newton.solvers import SolverVBD
from newton._src.geometry import ParticleFlags
from newton._src.geometry.tri_mesh_collision import (TriMeshCollisionInfo, get_edge_colliding_edges_count,
                                                     get_vertex_colliding_triangles_count)
from newton._src.solvers.solver import SolverBase
from newton._src.solvers.vbd.particle_vbd_kernels import (
    create_edge_edge_division_plane_closest_pt, create_vertex_triangle_division_plane_closest_pt,
    evaluate_dihedral_angle_based_bending_force_hessian, evaluate_edge_edge_contact,
    evaluate_neo_hookean_membrane_force_hessian, evaluate_vertex_triangle_collision_force_hessian, planar_truncation_t)
from newton._src.solvers.vbd.rigid_vbd_kernels import _eval_body_particle_contact, _eval_soft_ef_contact
from newton._src.geometry.broad_phase_common import binary_search
from newton._src.geometry.kernels import (EDGE_COLLISION_BUFFER_OVERFLOW_INDEX, VERTEX_COLLISION_BUFFER_OVERFLOW_INDEX,
                                          triangle_closest_point, vertex_adjacent_to_triangle)
from newton._src.utils.mesh import (MeshAdjacencyData, get_vertex_adjacent_edge_id_order,
                                    get_vertex_adjacent_face_id_order, get_vertex_num_adjacent_edges,
                                    get_vertex_num_adjacent_faces)

wp.set_module_options({"enable_backward": False})

PROFILE = os.environ.get("FASTVBD_PROFILE") == "1"
TIMES = defaultdict(float)


# ------------------------------------------------------------------------------------------------ kernels
@wp.kernel
def solve_color(
    dt: float,
    ids: wp.array[wp.int32],
    pos_anchor: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    mass: wp.array[float],
    inertia: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    tri_poses: wp.array[wp.mat22],
    tri_materials: wp.array2d[float],
    tri_areas: wp.array[float],
    edge_indices: wp.array2d[wp.int32],
    edge_rest_angles: wp.array[float],
    edge_rest_length: wp.array[float],
    edge_bending_properties: wp.array2d[float],
    adjacency: MeshAdjacencyData,
    friction_epsilon: float,
    particle_radius: wp.array[float],
    body_particle_contact_indices: wp.array[wp.vec3i],
    particle_contact_head: wp.array[int],
    particle_contact_next: wp.array[int],
    body_particle_contact_penalty_k: wp.array[float],
    body_particle_contact_material_kd: wp.array[float],
    body_particle_contact_material_mu: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    contact_barycentric: wp.array[wp.vec3],
    inc_offsets: wp.array[wp.int32],
    inc_entries: wp.array[wp.vec4i],
    collision_radius: float,
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    edge_edge_parallel_epsilon: float,
    particle_displacements: wp.array[wp.vec3],
    update_norm: wp.array[float],
):
    """newton's solve_elasticity + gather_particle_body_contact_force_and_hessian for one vertex, plus its gathered
    self-contact terms. inc_entries: (kind, a, b, slot); kind 0 = vertex-triangle (a = the vertex, b = the triangle,
    slot 0..2 = a triangle corner, 3 = the vertex), kind 1 = edge-edge (a = e1, b = e2, slot 0/1 = e1's endpoints)."""
    particle_index = ids[wp.tid()]
    update_norm[particle_index] = 0.0
    if not particle_flags[particle_index] & ParticleFlags.ACTIVE or mass[particle_index] == 0:
        particle_displacements[particle_index] = wp.vec3(0.0)
        return

    dt_sqr_reciprocal = 1.0 / (dt * dt)
    f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)
        if tri_materials[tri_index, 0] > 0.0 or tri_materials[tri_index, 1] > 0.0:
            f_tri, h_tri = evaluate_neo_hookean_membrane_force_hessian(
                tri_index, vertex_order, pos, pos_anchor, tri_indices, tri_poses[tri_index], tri_areas[tri_index],
                tri_materials[tri_index, 0], tri_materials[tri_index, 1], tri_materials[tri_index, 2], dt)
            f = f + f_tri
            h = h + h_tri

    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index, i_adj_edge)
        if edge_bending_properties[nei_edge_index, 0] > 0.0:
            f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                nei_edge_index, vertex_order_on_edge, pos, pos_anchor, edge_indices, edge_rest_angles,
                edge_rest_length, edge_bending_properties[nei_edge_index, 0],
                edge_bending_properties[nei_edge_index, 1], dt)
            f = f + f_edge
            h = h + h_edge

    fb = wp.vec3(0.0)
    hb = wp.mat33(0.0)
    head = particle_contact_head[particle_index]
    last = int(-1)
    while True:
        node = int(2147483647)
        cursor = head
        while cursor >= 0:
            if cursor > last and cursor < node:
                node = cursor
            cursor = particle_contact_next[cursor]
        if node == 2147483647:
            break
        last = node
        contact_index = node // 3
        corner = node - 3 * contact_index
        corners = body_particle_contact_indices[contact_index]
        contact_ke = body_particle_contact_penalty_k[contact_index]
        contact_kd = body_particle_contact_material_kd[contact_index]
        contact_mu = body_particle_contact_material_mu[contact_index]
        if corners[1] < 0:
            cf, ch = _eval_body_particle_contact(
                particle_index, pos[particle_index], pos_anchor[particle_index], contact_index, contact_ke, contact_kd,
                contact_mu, friction_epsilon, particle_radius, shape_body, body_q, body_q_prev, body_qd, body_com,
                contact_shape, contact_body_pos, contact_body_vel, contact_normal, shape_margin, dt)
            fb += cf
            hb += ch
        else:
            bary = contact_barycentric[contact_index]
            cf, ch, _cp_world = _eval_soft_ef_contact(
                contact_index, corners, bary, pos, pos_anchor, particle_radius, contact_ke, contact_kd, contact_mu,
                friction_epsilon, shape_body, body_q, body_q_prev, body_qd, body_com, contact_shape, contact_body_pos,
                contact_body_vel, contact_normal, shape_margin, dt)
            weight = bary[corner]
            fb += weight * cf
            hb += (weight * weight) * ch

    fs = wp.vec3(0.0)
    hs = wp.mat33(0.0)
    for k in range(inc_offsets[particle_index], inc_offsets[particle_index + 1]):
        e = inc_entries[k]
        if e[0] == 0:
            cf, ch = evaluate_vertex_triangle_collision_force_hessian(
                e[1], e[3], e[2], pos, pos_anchor, tri_indices, collision_radius, soft_contact_ke, soft_contact_kd,
                friction_mu, friction_epsilon, dt)
        else:
            cf, ch = evaluate_edge_edge_contact(
                particle_index, e[3], e[1], e[2], pos, pos_anchor, edge_indices, collision_radius, soft_contact_ke,
                soft_contact_kd, friction_mu, friction_epsilon, dt, edge_edge_parallel_epsilon)
        fs += cf
        hs += ch

    h = h + (hb + hs)
    f = f + (fb + fs)
    if abs(wp.determinant(h)) > 1e-8:
        h_inv = wp.inverse(h)
        dx = h_inv * f
        particle_displacements[particle_index] = particle_displacements[particle_index] + dx
        update_norm[particle_index] = wp.length(dx)


@wp.kernel
def eval_force_strided(
    phase: int,
    stride: int,
    n_particles: int,
    dt: float,
    pos_anchor: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    mass: wp.array[float],
    inertia: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    tri_poses: wp.array[wp.mat22],
    tri_materials: wp.array2d[float],
    tri_areas: wp.array[float],
    edge_indices: wp.array2d[wp.int32],
    edge_rest_angles: wp.array[float],
    edge_rest_length: wp.array[float],
    edge_bending_properties: wp.array2d[float],
    adjacency: MeshAdjacencyData,
    friction_epsilon: float,
    particle_radius: wp.array[float],
    body_particle_contact_indices: wp.array[wp.vec3i],
    particle_contact_head: wp.array[int],
    particle_contact_next: wp.array[int],
    body_particle_contact_penalty_k: wp.array[float],
    body_particle_contact_material_kd: wp.array[float],
    body_particle_contact_material_mu: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    contact_barycentric: wp.array[wp.vec3],
    inc_offsets: wp.array[wp.int32],
    inc_entries: wp.array[wp.vec4i],
    collision_radius: float,
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    edge_edge_parallel_epsilon: float,
    f_out: wp.array[wp.vec3],
    ha_out: wp.array[wp.mat33],
    cap_out: wp.array[float],
    nrm_out: wp.array[wp.vec3],
):
    """The coarse correction's inputs for particle tid * stride + phase: its total force f (solve_color's f, all terms)
    and its anchoring Hessian (inertia + body contacts; zero for a fixed particle)."""
    particle_index = wp.tid() * stride + phase
    if particle_index >= n_particles:
        return
    if not particle_flags[particle_index] & ParticleFlags.ACTIVE or mass[particle_index] == 0:
        f_out[particle_index] = wp.vec3(0.0)
        ha_out[particle_index] = wp.mat33(0.0)
        cap_out[particle_index] = 0.0
        nrm_out[particle_index] = wp.vec3(0.0)
        return

    dt_sqr_reciprocal = 1.0 / (dt * dt)
    f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)
        if tri_materials[tri_index, 0] > 0.0 or tri_materials[tri_index, 1] > 0.0:
            f_tri, h_tri = evaluate_neo_hookean_membrane_force_hessian(
                tri_index, vertex_order, pos, pos_anchor, tri_indices, tri_poses[tri_index], tri_areas[tri_index],
                tri_materials[tri_index, 0], tri_materials[tri_index, 1], tri_materials[tri_index, 2], dt)
            f = f + f_tri
            h = h + h_tri

    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index, i_adj_edge)
        if edge_bending_properties[nei_edge_index, 0] > 0.0:
            f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                nei_edge_index, vertex_order_on_edge, pos, pos_anchor, edge_indices, edge_rest_angles,
                edge_rest_length, edge_bending_properties[nei_edge_index, 0],
                edge_bending_properties[nei_edge_index, 1], dt)
            f = f + f_edge
            h = h + h_edge

    fb = wp.vec3(0.0)
    hb = wp.mat33(0.0)
    cap = float(0.0)
    nrm = wp.vec3(0.0)
    head = particle_contact_head[particle_index]
    last = int(-1)
    while True:
        node = int(2147483647)
        cursor = head
        while cursor >= 0:
            if cursor > last and cursor < node:
                node = cursor
            cursor = particle_contact_next[cursor]
        if node == 2147483647:
            break
        last = node
        contact_index = node // 3
        corner = node - 3 * contact_index
        corners = body_particle_contact_indices[contact_index]
        contact_ke = body_particle_contact_penalty_k[contact_index]
        contact_kd = body_particle_contact_material_kd[contact_index]
        contact_mu = body_particle_contact_material_mu[contact_index]
        if corners[1] < 0 and shape_body[contact_shape[contact_index]] < 0:
            # a static shape (table, fence): only its normal part here; its friction enters the coarse step as a
            # Coulomb capacity mu |normal force| (the coarse level treats the sheet as one body sliding on it)
            cf, ch = _eval_body_particle_contact(
                particle_index, pos[particle_index], pos_anchor[particle_index], contact_index, contact_ke, contact_kd,
                0.0, friction_epsilon, particle_radius, shape_body, body_q, body_q_prev, body_qd, body_com,
                contact_shape, contact_body_pos, contact_body_vel, contact_normal, shape_margin, dt)
            fb += cf
            hb += ch
            cap += contact_mu * wp.length(cf)
            nrm += cf
        elif corners[1] < 0:
            cf, ch = _eval_body_particle_contact(
                particle_index, pos[particle_index], pos_anchor[particle_index], contact_index, contact_ke, contact_kd,
                contact_mu, friction_epsilon, particle_radius, shape_body, body_q, body_q_prev, body_qd, body_com,
                contact_shape, contact_body_pos, contact_body_vel, contact_normal, shape_margin, dt)
            fb += cf
            hb += ch
        else:
            bary = contact_barycentric[contact_index]
            cf, ch, _cp_world = _eval_soft_ef_contact(
                contact_index, corners, bary, pos, pos_anchor, particle_radius, contact_ke, contact_kd, contact_mu,
                friction_epsilon, shape_body, body_q, body_q_prev, body_qd, body_com, contact_shape, contact_body_pos,
                contact_body_vel, contact_normal, shape_margin, dt)
            weight = bary[corner]
            fb += weight * cf
            hb += (weight * weight) * ch

    fs = wp.vec3(0.0)
    hs = wp.mat33(0.0)
    for k in range(inc_offsets[particle_index], inc_offsets[particle_index + 1]):
        e = inc_entries[k]
        if e[0] == 0:
            cf, ch = evaluate_vertex_triangle_collision_force_hessian(
                e[1], e[3], e[2], pos, pos_anchor, tri_indices, collision_radius, soft_contact_ke, soft_contact_kd,
                friction_mu, friction_epsilon, dt)
        else:
            cf, ch = evaluate_edge_edge_contact(
                particle_index, e[3], e[1], e[2], pos, pos_anchor, edge_indices, collision_radius, soft_contact_ke,
                soft_contact_kd, friction_mu, friction_epsilon, dt, edge_edge_parallel_epsilon)
        fs += cf
        hs += ch

    f_out[particle_index] = f + (fb + fs)
    ha_out[particle_index] = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float) + hb
    cap_out[particle_index] = cap
    nrm_out[particle_index] = nrm


@wp.kernel
def eval_patch_strided(
    phase: int,
    stride: int,
    n_particles: int,
    dt: float,
    pos_anchor: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    mass: wp.array[float],
    inertia: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    tri_poses: wp.array[wp.mat22],
    tri_materials: wp.array2d[float],
    tri_areas: wp.array[float],
    edge_indices: wp.array2d[wp.int32],
    edge_rest_angles: wp.array[float],
    edge_rest_length: wp.array[float],
    edge_bending_properties: wp.array2d[float],
    adjacency: MeshAdjacencyData,
    friction_epsilon: float,
    particle_radius: wp.array[float],
    body_particle_contact_indices: wp.array[wp.vec3i],
    particle_contact_head: wp.array[int],
    particle_contact_next: wp.array[int],
    body_particle_contact_penalty_k: wp.array[float],
    body_particle_contact_material_kd: wp.array[float],
    body_particle_contact_material_mu: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    contact_barycentric: wp.array[wp.vec3],
    inc_offsets: wp.array[wp.int32],
    inc_entries: wp.array[wp.vec4i],
    collision_radius: float,
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    edge_edge_parallel_epsilon: float,
    pid: wp.array[wp.int32],
    f_out: wp.array[wp.vec3],
    ha_out: wp.array[wp.mat33],
    cap_out: wp.array[float],
    nrm_out: wp.array[wp.vec3],
    caps_out: wp.array[float],
    partner_out: wp.array[wp.int32],
):
    """The patch coarse correction's inputs (eval_force_strided with the self-contact friction split off) for particle tid * stride + phase: its total force f (solve_color's f, all terms)
    and its anchoring Hessian (inertia + body contacts; zero for a fixed particle)."""
    particle_index = wp.tid() * stride + phase
    if particle_index >= n_particles:
        return
    if not particle_flags[particle_index] & ParticleFlags.ACTIVE or mass[particle_index] == 0:
        f_out[particle_index] = wp.vec3(0.0)
        ha_out[particle_index] = wp.mat33(0.0)
        cap_out[particle_index] = 0.0
        nrm_out[particle_index] = wp.vec3(0.0)
        caps_out[particle_index] = 0.0
        partner_out[particle_index] = -1
        return

    dt_sqr_reciprocal = 1.0 / (dt * dt)
    f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)
        if tri_materials[tri_index, 0] > 0.0 or tri_materials[tri_index, 1] > 0.0:
            f_tri, h_tri = evaluate_neo_hookean_membrane_force_hessian(
                tri_index, vertex_order, pos, pos_anchor, tri_indices, tri_poses[tri_index], tri_areas[tri_index],
                tri_materials[tri_index, 0], tri_materials[tri_index, 1], tri_materials[tri_index, 2], dt)
            f = f + f_tri
            h = h + h_tri

    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index, i_adj_edge)
        if edge_bending_properties[nei_edge_index, 0] > 0.0:
            f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                nei_edge_index, vertex_order_on_edge, pos, pos_anchor, edge_indices, edge_rest_angles,
                edge_rest_length, edge_bending_properties[nei_edge_index, 0],
                edge_bending_properties[nei_edge_index, 1], dt)
            f = f + f_edge
            h = h + h_edge

    fb = wp.vec3(0.0)
    hb = wp.mat33(0.0)
    cap = float(0.0)
    nrm = wp.vec3(0.0)
    head = particle_contact_head[particle_index]
    last = int(-1)
    while True:
        node = int(2147483647)
        cursor = head
        while cursor >= 0:
            if cursor > last and cursor < node:
                node = cursor
            cursor = particle_contact_next[cursor]
        if node == 2147483647:
            break
        last = node
        contact_index = node // 3
        corner = node - 3 * contact_index
        corners = body_particle_contact_indices[contact_index]
        contact_ke = body_particle_contact_penalty_k[contact_index]
        contact_kd = body_particle_contact_material_kd[contact_index]
        contact_mu = body_particle_contact_material_mu[contact_index]
        if corners[1] < 0 and shape_body[contact_shape[contact_index]] < 0:
            # a static shape (table, fence): only its normal part here; its friction enters the coarse step as a
            # Coulomb capacity mu |normal force| (the coarse level treats the sheet as one body sliding on it)
            cf, ch = _eval_body_particle_contact(
                particle_index, pos[particle_index], pos_anchor[particle_index], contact_index, contact_ke, contact_kd,
                0.0, friction_epsilon, particle_radius, shape_body, body_q, body_q_prev, body_qd, body_com,
                contact_shape, contact_body_pos, contact_body_vel, contact_normal, shape_margin, dt)
            fb += cf
            hb += ch
            cap += contact_mu * wp.length(cf)
            nrm += cf
        elif corners[1] < 0:
            cf, ch = _eval_body_particle_contact(
                particle_index, pos[particle_index], pos_anchor[particle_index], contact_index, contact_ke, contact_kd,
                contact_mu, friction_epsilon, particle_radius, shape_body, body_q, body_q_prev, body_qd, body_com,
                contact_shape, contact_body_pos, contact_body_vel, contact_normal, shape_margin, dt)
            fb += cf
            hb += ch
        else:
            bary = contact_barycentric[contact_index]
            cf, ch, _cp_world = _eval_soft_ef_contact(
                contact_index, corners, bary, pos, pos_anchor, particle_radius, contact_ke, contact_kd, contact_mu,
                friction_epsilon, shape_body, body_q, body_q_prev, body_qd, body_com, contact_shape, contact_body_pos,
                contact_body_vel, contact_normal, shape_margin, dt)
            weight = bary[corner]
            fb += weight * cf
            hb += (weight * weight) * ch

    # self-contact: normal part only (friction 0); its friction enters as a capacity mu |N| against the partner patch
    fs = wp.vec3(0.0)
    hs = wp.mat33(0.0)
    caps = float(0.0)
    partner = int(-1)
    me = pid[particle_index]
    for k in range(inc_offsets[particle_index], inc_offsets[particle_index + 1]):
        e = inc_entries[k]
        other = int(-1)
        if e[0] == 0:
            cf, ch = evaluate_vertex_triangle_collision_force_hessian(
                e[1], e[3], e[2], pos, pos_anchor, tri_indices, collision_radius, soft_contact_ke, soft_contact_kd,
                0.0, friction_epsilon, dt)
            if e[3] == 3:
                other = pid[tri_indices[e[2], 0]]
            else:
                other = pid[e[1]]
        else:
            cf, ch = evaluate_edge_edge_contact(
                particle_index, e[3], e[1], e[2], pos, pos_anchor, edge_indices, collision_radius, soft_contact_ke,
                soft_contact_kd, 0.0, friction_epsilon, dt, edge_edge_parallel_epsilon)
            other = pid[edge_indices[e[2], 2]]
        fs += cf
        hs += ch
        nrm += cf
        if other != me:
            caps += friction_mu * wp.length(cf)
            if partner < 0:
                partner = other

    f_out[particle_index] = f + (fb + fs)
    ha_out[particle_index] = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float) + hb + hs
    cap_out[particle_index] = cap
    nrm_out[particle_index] = nrm
    caps_out[particle_index] = caps
    partner_out[particle_index] = partner


@wp.func
def _tri_block(t: int, c: int, pos: wp.array[wp.vec3], pos_anchor: wp.array[wp.vec3], tri_indices: wp.array2d[wp.int32],
               tri_poses: wp.array[wp.mat22], tri_materials: wp.array2d[float], tri_areas: wp.array[float], dt: float):
    f_tri, h_tri = evaluate_neo_hookean_membrane_force_hessian(
        t, c, pos, pos_anchor, tri_indices, tri_poses[t], tri_areas[t], tri_materials[t, 0], tri_materials[t, 1],
        tri_materials[t, 2], dt)
    return h_tri


@wp.kernel
def coarse_patch_solve(
    n_particles: int,
    K: int,
    pid: wp.array[wp.int32],
    f: wp.array[wp.vec3],
    ha: wp.array[wp.mat33],
    capg: wp.array[float],
    nrm: wp.array[wp.vec3],
    caps: wp.array[float],
    partner: wp.array[wp.int32],
    mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    pos_anchor: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    tri_poses: wp.array[wp.mat22],
    tri_materials: wp.array2d[float],
    tri_areas: wp.array[float],
    n_tris: int,
    dt: float,
    nb_off: wp.array[wp.int32],
    nb_idx: wp.array[wp.int32],
    sweeps: int,
    A: wp.array[wp.mat33],
    r: wp.array[wp.vec3],
    cg: wp.array[float],
    nsum: wp.array[wp.vec3],
    cp: wp.array[float],
    cpart: wp.array[wp.int32],
    delta: wp.array[wp.vec3],
):
    """One thread, fixed order: a Newton step of the step's energy over per-patch translations (patches: graph balls of
    the mesh, i.e. material regions, so the layers of a fold are separate patches). Galerkin blocks: inertia + contact
    normal blocks per patch, the membrane coupling exactly (a triangle's cross blocks follow from its diagonal blocks
    by translation invariance); friction as Coulomb capacities against the ground (static shapes) and against the
    patch each patch rests on. Solved by projected block Gauss-Seidel over the patch graph (A[I, slot] for the static
    neighbour slots of nb_off/nb_idx; slot nb_off[I+1]-nb_off[I] - 1 is I itself)."""
    nnz = nb_off[K]
    for q in range(nnz):
        A[q] = wp.mat33(0.0)
    for I in range(K):
        r[I] = wp.vec3(0.0)
        cg[I] = 0.0
        nsum[I] = wp.vec3(0.0)
        cp[I] = 0.0
        cpart[I] = -1
        delta[I] = wp.vec3(0.0)
    for i in range(n_particles):
        if (particle_flags[i] & ParticleFlags.ACTIVE) != 0 and mass[i] > 0.0:
            I = pid[i]
            r[I] += f[i]
            q = nb_off[I + 1] - 1
            A[q] += ha[i]
            cg[I] += capg[i]
            nsum[I] += nrm[i]
            if caps[i] > 0.0 and partner[i] >= 0:
                if cpart[I] < 0:
                    cpart[I] = partner[i]
                if cpart[I] == partner[i]:
                    cp[I] += caps[i]
                else:
                    cg[I] += caps[i]      # a second partner: treated as ground
    for t in range(n_tris):
        a = tri_indices[t, 0]
        b = tri_indices[t, 1]
        c = tri_indices[t, 2]
        fa = (particle_flags[a] & ParticleFlags.ACTIVE) != 0 and mass[a] > 0.0
        fb = (particle_flags[b] & ParticleFlags.ACTIVE) != 0 and mass[b] > 0.0
        fc = (particle_flags[c] & ParticleFlags.ACTIVE) != 0 and mass[c] > 0.0
        Pa = pid[a]
        Pb = pid[b]
        Pc = pid[c]
        if fa and fb and fc and Pa == Pb and Pb == Pc:
            continue
        if not (fa or fb or fc):
            continue
        if tri_materials[t, 0] <= 0.0 and tri_materials[t, 1] <= 0.0:
            continue
        Haa = _tri_block(t, 0, pos, pos_anchor, tri_indices, tri_poses, tri_materials, tri_areas, dt)
        Hbb = _tri_block(t, 1, pos, pos_anchor, tri_indices, tri_poses, tri_materials, tri_areas, dt)
        Hcc = _tri_block(t, 2, pos, pos_anchor, tri_indices, tri_poses, tri_materials, tri_areas, dt)
        Hab = 0.5 * (Hcc - Haa - Hbb)
        Hac = 0.5 * (Hbb - Haa - Hcc)
        Hbc = 0.5 * (Haa - Hbb - Hcc)
        for u in range(3):
            for v in range(3):
                vu = tri_indices[t, u]
                vv = tri_indices[t, v]
                fu = (particle_flags[vu] & ParticleFlags.ACTIVE) != 0 and mass[vu] > 0.0
                fv = (particle_flags[vv] & ParticleFlags.ACTIVE) != 0 and mass[vv] > 0.0
                if not (fu and fv):
                    continue
                H = Haa
                if u == v:
                    if u == 1:
                        H = Hbb
                    elif u == 2:
                        H = Hcc
                else:
                    lo = wp.min(u, v)
                    hi = wp.max(u, v)
                    if lo == 0 and hi == 1:
                        H = Hab
                    elif lo == 0 and hi == 2:
                        H = Hac
                    else:
                        H = Hbc
                I = pid[vu]
                J = pid[vv]
                for q in range(nb_off[I], nb_off[I + 1]):
                    if nb_idx[q] == J:
                        A[q] += H
    # the common translation first (the exact global step, Coulomb against the ground), as the sweeps' start: the
    # sweeps then only resolve the patches' relative motion
    rg = wp.vec3(0.0)
    Ag = wp.mat33(0.0)
    cgt = float(0.0)
    ng = wp.vec3(0.0)
    for I in range(K):
        rg += r[I]
        cgt += cg[I]
        ng += nsum[I]
        for q in range(nb_off[I], nb_off[I + 1]):
            Ag += A[q]
    dg = wp.vec3(0.0)
    if wp.abs(wp.determinant(Ag)) > 1e-8:
        ngl = wp.length(ng)
        if ngl > 0.0 and cgt > 0.0:
            nhg = ng / ngl
            rgt = rg - wp.dot(rg, nhg) * nhg
            rgtl = wp.length(rgt)
            if rgtl <= cgt:
                anng = wp.dot(nhg, Ag * nhg)
                if anng > 0.0:
                    dg = nhg * (wp.dot(rg, nhg) / anng)
            else:
                dg = wp.inverse(Ag) * (rg - rgt * (cgt / rgtl))
        else:
            dg = wp.inverse(Ag) * rg
    for I in range(K):
        delta[I] = dg
    for sw in range(sweeps):
        for I in range(K):
            qd = nb_off[I + 1] - 1
            AII = A[qd]
            if wp.abs(wp.determinant(AII)) <= 1e-8:
                continue
            rho = r[I]
            for q in range(nb_off[I], qd):
                rho -= A[q] * delta[nb_idx[q]]
            anchor = wp.vec3(0.0)
            ctot = cg[I] + cp[I]
            if cpart[I] >= 0 and cp[I] > 0.0:
                anchor = delta[cpart[I]] * (cp[I] / ctot)
            nl = wp.length(nsum[I])
            d = wp.inverse(AII) * rho
            if nl > 0.0 and ctot > 0.0:
                nh = nsum[I] / nl
                at = anchor - wp.dot(anchor, nh) * nh
                ann = wp.dot(nh, AII * nh)
                if ann > 0.0:
                    s = (wp.dot(nh, rho) - wp.dot(nh, AII * at)) / ann
                    ds = at + s * nh
                    fh = rho - AII * ds
                    ft = fh - wp.dot(fh, nh) * nh
                    ftl = wp.length(ft)
                    if ftl <= ctot:
                        d = ds
                    else:
                        d = wp.inverse(AII) * (rho - ft * (ctot / ftl))
            delta[I] = d


@wp.kernel
def coarse_patch_apply_strided(phase: int, stride: int, n_particles: int, mass: wp.array[float],
                               particle_flags: wp.array[wp.int32], pid: wp.array[wp.int32], delta: wp.array[wp.vec3],
                               particle_displacements: wp.array[wp.vec3]):
    i = wp.tid() * stride + phase
    if i >= n_particles:
        return
    if (particle_flags[i] & ParticleFlags.ACTIVE) != 0 and mass[i] > 0.0:
        particle_displacements[i] = particle_displacements[i] + delta[pid[i]]


@wp.kernel
def coarse_solve(
    n_particles: int,
    f: wp.array[wp.vec3],
    ha: wp.array[wp.mat33],
    cap: wp.array[float],
    nrm: wp.array[wp.vec3],
    mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    pos_anchor: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    tri_poses: wp.array[wp.mat22],
    tri_materials: wp.array2d[float],
    tri_areas: wp.array[float],
    n_tris: int,
    dt: float,
    slot: int,
    delta: wp.array[wp.vec3],
):
    """One thread, in index order: the Newton step of the step's energy restricted to a common translation of all free
    particles (Galerkin: the elastic Hessian of a triangle sums to zero over a rigid translation, so only triangles
    with fixed corners couple; their block is the singleton side's diagonal block)."""
    r = wp.vec3(0.0)
    A = wp.mat33(0.0)
    ctot = float(0.0)
    nsum = wp.vec3(0.0)
    for i in range(n_particles):
        r += f[i]
        A += ha[i]
        ctot += cap[i]
        nsum += nrm[i]
    for t in range(n_tris):
        nfree = int(0)
        for c in range(3):
            v = tri_indices[t, c]
            if (particle_flags[v] & ParticleFlags.ACTIVE) != 0 and mass[v] > 0.0:
                nfree += 1
        if nfree == 1 or nfree == 2:
            for c in range(3):
                v = tri_indices[t, c]
                free = (particle_flags[v] & ParticleFlags.ACTIVE) != 0 and mass[v] > 0.0
                if (nfree == 1 and free) or (nfree == 2 and not free):
                    if tri_materials[t, 0] > 0.0 or tri_materials[t, 1] > 0.0:
                        f_tri, h_tri = evaluate_neo_hookean_membrane_force_hessian(
                            t, c, pos, pos_anchor, tri_indices, tri_poses[t], tri_areas[t], tri_materials[t, 0],
                            tri_materials[t, 1], tri_materials[t, 2], dt)
                        A += h_tri
    # static friction at the coarse level (Coulomb): the sheet slides only by the tangential force beyond the total
    # capacity ctot; below it the step moves along the mean contact normal only
    d = wp.vec3(0.0)
    if abs(wp.determinant(A)) > 1e-8:
        nl = wp.length(nsum)
        if nl > 0.0 and ctot > 0.0:
            nh = nsum / nl
            rt = r - wp.dot(r, nh) * nh
            rtl = wp.length(rt)
            if rtl <= ctot:
                ann = wp.dot(nh, A * nh)
                if ann > 0.0:
                    d = nh * (wp.dot(r, nh) / ann)
            else:
                d = wp.inverse(A) * (r - rt * (ctot / rtl))
        else:
            d = wp.inverse(A) * r
    delta[0] = d
    delta[1 + slot] = d


@wp.kernel
def coarse_apply_strided(phase: int, stride: int, n_particles: int, mass: wp.array[float],
                         particle_flags: wp.array[wp.int32], delta: wp.array[wp.vec3],
                         particle_displacements: wp.array[wp.vec3]):
    i = wp.tid() * stride + phase
    if i >= n_particles:
        return
    if (particle_flags[i] & ParticleFlags.ACTIVE) != 0 and mass[i] > 0.0:
        particle_displacements[i] = particle_displacements[i] + delta[0]


@wp.kernel
def max_strided(phase: int, stride: int, n: int, values: wp.array[float], out: wp.array[float]):
    """out[phase] = max of values[i] over i = phase, phase + stride, ... (one thread: a serial, exact max)."""
    m = float(0.0)
    i = phase
    while i < n:
        m = wp.max(m, values[i])
        i += stride
    out[phase] = m


@wp.func
def _tmin(buf: wp.array[float], i: int, t: float):
    buf[i] = wp.min(buf[i], t)


@wp.kernel
def truncation_strided(
    phase: int,
    stride: int,
    n_primitives: int,
    pos: wp.array[wp.vec3],
    displacement_in: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    collision_info_array: wp.array[TriMeshCollisionInfo],
    parallel_eps: float,
    gamma: float,
    truncation_t_out: wp.array[float],
):
    """newton's apply_planar_truncation_parallel_by_collision for primitives p = tid * stride + phase (one thread per
    primitive, all of its entries), min-reducing into this thread's own buffer."""
    primitive_id = wp.tid() * stride + phase
    if primitive_id >= n_primitives:
        return
    collision_info = collision_info_array[0]

    if primitive_id < collision_info.edge_colliding_edges_buffer_sizes.shape[0]:
        e1_idx = primitive_id
        collision_buffer_offset = collision_info.edge_colliding_edges_offsets[primitive_id]
        collision_count = get_edge_colliding_edges_count(collision_info, primitive_id)
        for c in range(collision_count):
            e2_idx = collision_info.edge_colliding_edges[2 * (collision_buffer_offset + c) + 1]
            if e1_idx != -1 and e2_idx != -1:
                e1_v1 = edge_indices[e1_idx, 2]
                e1_v2 = edge_indices[e1_idx, 3]
                e1_v1_pos = pos[e1_v1]
                e1_v2_pos = pos[e1_v2]
                delta_e1_v1 = displacement_in[e1_v1]
                delta_e1_v2 = displacement_in[e1_v2]
                e2_v1 = edge_indices[e2_idx, 2]
                e2_v2 = edge_indices[e2_idx, 3]
                e2_v1_pos = pos[e2_v1]
                e2_v2_pos = pos[e2_v2]
                delta_e2_v1 = displacement_in[e2_v1]
                delta_e2_v2 = displacement_in[e2_v2]
                is_dummy, n, d = create_edge_edge_division_plane_closest_pt(
                    e1_v1_pos, delta_e1_v1, e1_v2_pos, delta_e1_v2, e2_v1_pos, delta_e2_v1, e2_v2_pos, delta_e2_v2)
                if not is_dummy[0]:
                    _tmin(truncation_t_out, e1_v1, planar_truncation_t(e1_v1_pos, delta_e1_v1, n, d, parallel_eps, gamma))
                if not is_dummy[1]:
                    _tmin(truncation_t_out, e1_v2, planar_truncation_t(e1_v2_pos, delta_e1_v2, n, d, parallel_eps, gamma))
                if not is_dummy[2]:
                    _tmin(truncation_t_out, e2_v1, planar_truncation_t(e2_v1_pos, delta_e2_v1, n, d, parallel_eps, gamma))
                if not is_dummy[3]:
                    _tmin(truncation_t_out, e2_v2, planar_truncation_t(e2_v2_pos, delta_e2_v2, n, d, parallel_eps, gamma))

    if primitive_id < collision_info.vertex_colliding_triangles_buffer_sizes.shape[0]:
        particle_idx = primitive_id
        colliding_particle_pos = pos[particle_idx]
        colliding_particle_displacement = displacement_in[particle_idx]
        collision_buffer_offset = collision_info.vertex_colliding_triangles_offsets[primitive_id]
        collision_count = get_vertex_colliding_triangles_count(collision_info, primitive_id)
        for c in range(collision_count):
            tri_idx = collision_info.vertex_colliding_triangles[(collision_buffer_offset + c) * 2 + 1]
            if particle_idx != -1 and tri_idx != -1:
                tri_a = tri_indices[tri_idx, 0]
                tri_b = tri_indices[tri_idx, 1]
                tri_c = tri_indices[tri_idx, 2]
                t1 = pos[tri_a]
                t2 = pos[tri_b]
                t3 = pos[tri_c]
                delta_t1 = displacement_in[tri_a]
                delta_t2 = displacement_in[tri_b]
                delta_t3 = displacement_in[tri_c]
                is_dummy, n, d = create_vertex_triangle_division_plane_closest_pt(
                    colliding_particle_pos, colliding_particle_displacement, t1, delta_t1, t2, delta_t2, t3, delta_t3)
                if not is_dummy[0]:
                    _tmin(truncation_t_out, particle_idx,
                          planar_truncation_t(colliding_particle_pos, colliding_particle_displacement, n, d,
                                              parallel_eps, gamma))
                if not is_dummy[1]:
                    _tmin(truncation_t_out, tri_a, planar_truncation_t(t1, delta_t1, n, d, parallel_eps, gamma))
                if not is_dummy[2]:
                    _tmin(truncation_t_out, tri_b, planar_truncation_t(t2, delta_t2, n, d, parallel_eps, gamma))
                if not is_dummy[3]:
                    _tmin(truncation_t_out, tri_c, planar_truncation_t(t3, delta_t3, n, d, parallel_eps, gamma))


@wp.kernel
def truncation_pairs_strided(
    phase: int,
    stride: int,
    n_pairs: wp.array[wp.int32],
    pairs: wp.array[wp.vec3i],
    pos: wp.array[wp.vec3],
    displacement_in: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    parallel_eps: float,
    gamma: float,
    truncation_t_out: wp.array[float],
):
    """newton's planar truncation over a de-duplicated pair list (kind 0: vertex a / triangle b; kind 1: edges a < b,
    each unordered edge pair once), pairs k = tid * stride + phase, min-reducing into this thread's buffer."""
    k = wp.tid() * stride + phase
    if k >= n_pairs[0]:
        return
    pr = pairs[k]
    if pr[0] == 0:
        particle_idx = pr[1]
        tri_idx = pr[2]
        p0 = pos[particle_idx]
        d0 = displacement_in[particle_idx]
        tri_a = tri_indices[tri_idx, 0]
        tri_b = tri_indices[tri_idx, 1]
        tri_c = tri_indices[tri_idx, 2]
        t1 = pos[tri_a]
        t2 = pos[tri_b]
        t3 = pos[tri_c]
        delta_t1 = displacement_in[tri_a]
        delta_t2 = displacement_in[tri_b]
        delta_t3 = displacement_in[tri_c]
        is_dummy, n, d = create_vertex_triangle_division_plane_closest_pt(p0, d0, t1, delta_t1, t2, delta_t2, t3, delta_t3)
        if not is_dummy[0]:
            _tmin(truncation_t_out, particle_idx, planar_truncation_t(p0, d0, n, d, parallel_eps, gamma))
        if not is_dummy[1]:
            _tmin(truncation_t_out, tri_a, planar_truncation_t(t1, delta_t1, n, d, parallel_eps, gamma))
        if not is_dummy[2]:
            _tmin(truncation_t_out, tri_b, planar_truncation_t(t2, delta_t2, n, d, parallel_eps, gamma))
        if not is_dummy[3]:
            _tmin(truncation_t_out, tri_c, planar_truncation_t(t3, delta_t3, n, d, parallel_eps, gamma))
    else:
        e1_v1 = edge_indices[pr[1], 2]
        e1_v2 = edge_indices[pr[1], 3]
        e2_v1 = edge_indices[pr[2], 2]
        e2_v2 = edge_indices[pr[2], 3]
        a1 = pos[e1_v1]
        a2 = pos[e1_v2]
        b1 = pos[e2_v1]
        b2 = pos[e2_v2]
        da1 = displacement_in[e1_v1]
        da2 = displacement_in[e1_v2]
        db1 = displacement_in[e2_v1]
        db2 = displacement_in[e2_v2]
        is_dummy, n, d = create_edge_edge_division_plane_closest_pt(a1, da1, a2, da2, b1, db1, b2, db2)
        if not is_dummy[0]:
            _tmin(truncation_t_out, e1_v1, planar_truncation_t(a1, da1, n, d, parallel_eps, gamma))
        if not is_dummy[1]:
            _tmin(truncation_t_out, e1_v2, planar_truncation_t(a2, da2, n, d, parallel_eps, gamma))
        if not is_dummy[2]:
            _tmin(truncation_t_out, e2_v1, planar_truncation_t(b1, db1, n, d, parallel_eps, gamma))
        if not is_dummy[3]:
            _tmin(truncation_t_out, e2_v2, planar_truncation_t(b2, db2, n, d, parallel_eps, gamma))


@wp.kernel
def truncation_apply_strided(
    phase: int,
    stride: int,
    n_particles: int,
    nbuf: int,
    tbuf: wp.array2d[float],
    pos: wp.array[wp.vec3],
    max_displacement: float,
    displacement: wp.array[wp.vec3],
    pos_out: wp.array[wp.vec3],
):
    """newton's apply_truncation_ts for particles i = tid * stride + phase: t = the min over the thread buffers (each
    reset to 1 for the next use), the isotropic clamp, the position."""
    i = wp.tid() * stride + phase
    if i >= n_particles:
        return
    t = float(1.0)
    for b in range(nbuf):
        t = wp.min(t, tbuf[b, i])
        tbuf[b, i] = 1.0
    d = displacement[i] * t
    n = wp.length(d)
    if n > max_displacement:
        d = d * max_displacement / n
    displacement[i] = d
    pos_out[i] = pos[i] + d


@wp.kernel
def truncation_apply_range(
    offset: int,
    nbuf: int,
    tbuf: wp.array2d[float],
    pos: wp.array[wp.vec3],
    max_displacement: float,
    displacement: wp.array[wp.vec3],
    pos_out: wp.array[wp.vec3],
):
    """truncation_apply_strided for the contiguous particles [offset, offset + dim) (a thread's writes stay in its
    own cache lines)."""
    i = wp.tid() + offset
    t = float(1.0)
    for b in range(nbuf):
        t = wp.min(t, tbuf[b, i])
        tbuf[b, i] = 1.0
    d = displacement[i] * t
    n = wp.length(d)
    if n > max_displacement:
        d = d * max_displacement / n
    displacement[i] = d
    pos_out[i] = pos[i] + d


# ------------------------------------------------------------------------------------------------ detection
# newton's vertex_triangle_collision_detection_kernel / edge_colliding_edges_detection_kernel for the indices
# tid * stride + phase (each writes only its own primitive's buffer row; no triangle-side records, as SolverVBD
# configures its detector).
@wp.kernel
def vt_detect_strided(
    phase: int,
    stride: int,
    n_vertices: int,
    max_query_radius: float,
    min_query_radius: float,
    bvh_id: wp.uint64,
    bvh_group_roots: wp.array[wp.int32],
    pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    particle_world: wp.array[wp.int32],
    world_count: wp.int32,
    vertex_colliding_triangles_offsets: wp.array[wp.int32],
    vertex_triangle_filtering_list: wp.array[wp.int32],
    vertex_triangle_filtering_list_offsets: wp.array[wp.int32],
    min_distance_filtering_ref_pos: wp.array[wp.vec3],
    vertex_colliding_triangles: wp.array[wp.int32],
    vertex_colliding_triangles_count: wp.array[wp.int32],
    vertex_colliding_triangles_min_dist: wp.array[float],
    resize_flags: wp.array[wp.int32],
):
    v_index = wp.tid() * stride + phase
    if v_index >= n_vertices:
        return
    v = pos[v_index]
    vertex_buffer_offset = vertex_colliding_triangles_offsets[v_index]
    vertex_buffer_size = vertex_colliding_triangles_offsets[v_index + 1] - vertex_buffer_offset
    lower = wp.vec3(v[0] - max_query_radius, v[1] - max_query_radius, v[2] - max_query_radius)
    upper = wp.vec3(v[0] + max_query_radius, v[1] + max_query_radius, v[2] + max_query_radius)
    vertex_num_collisions = wp.int32(0)
    min_dis_to_tris = max_query_radius
    vertex_world = particle_world[v_index]
    for query_pass in range(2):
        run_query = bool(False)
        query_all = bool(False)
        group_root = wp.int32(-1)
        if vertex_world < 0:
            if query_pass == 0:
                run_query = True
                query_all = True
        else:
            if query_pass == 0:
                group_root = bvh_group_roots[vertex_world]
            else:
                group_root = bvh_group_roots[world_count]
            run_query = group_root >= 0
        if run_query:
            if query_all:
                query = wp.bvh_query_aabb(bvh_id, lower, upper)
            else:
                query = wp.bvh_query_aabb(bvh_id, lower, upper, group_root)
            tri_index = wp.int32(0)
            while wp.bvh_query_next(query, tri_index):
                t1 = tri_indices[tri_index, 0]
                t2 = tri_indices[tri_index, 1]
                t3 = tri_indices[tri_index, 2]
                if vertex_adjacent_to_triangle(v_index, t1, t2, t3):
                    continue
                if vertex_triangle_filtering_list:
                    fl_start = vertex_triangle_filtering_list_offsets[v_index]
                    fl_end = vertex_triangle_filtering_list_offsets[v_index + 1]
                    if fl_end > fl_start:
                        first_val = vertex_triangle_filtering_list[fl_start]
                        last_val = vertex_triangle_filtering_list[fl_end - 1]
                        if (tri_index >= first_val) and (tri_index <= last_val):
                            idx = binary_search(vertex_triangle_filtering_list, tri_index, fl_start, fl_end)
                            if idx > fl_start and vertex_triangle_filtering_list[idx - 1] == tri_index:
                                continue
                u1 = pos[t1]
                u2 = pos[t2]
                u3 = pos[t3]
                closest_p, _bary, _feature_type = triangle_closest_point(u1, u2, u3, v)
                dist = wp.length(closest_p - v)
                if min_distance_filtering_ref_pos and min_query_radius > 0.0:
                    closest_p_ref, _, __ = triangle_closest_point(
                        min_distance_filtering_ref_pos[t1], min_distance_filtering_ref_pos[t2],
                        min_distance_filtering_ref_pos[t3], min_distance_filtering_ref_pos[v_index])
                    dist_ref = wp.length(closest_p_ref - min_distance_filtering_ref_pos[v_index])
                    if dist_ref < min_query_radius:
                        continue
                if dist < max_query_radius:
                    min_dis_to_tris = wp.min(min_dis_to_tris, dist)
                    if vertex_num_collisions < vertex_buffer_size:
                        vertex_colliding_triangles[2 * (vertex_buffer_offset + vertex_num_collisions)] = v_index
                        vertex_colliding_triangles[2 * (vertex_buffer_offset + vertex_num_collisions) + 1] = tri_index
                    else:
                        resize_flags[VERTEX_COLLISION_BUFFER_OVERFLOW_INDEX] = 1
                    vertex_num_collisions = vertex_num_collisions + 1
    vertex_colliding_triangles_count[v_index] = vertex_num_collisions
    vertex_colliding_triangles_min_dist[v_index] = min_dis_to_tris


@wp.kernel
def ee_detect_strided(
    phase: int,
    stride: int,
    n_edges: int,
    max_query_radius: float,
    min_query_radius: float,
    bvh_id: wp.uint64,
    bvh_group_roots: wp.array[wp.int32],
    pos: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    particle_world: wp.array[wp.int32],
    world_count: wp.int32,
    edge_colliding_edges_offsets: wp.array[wp.int32],
    edge_edge_parallel_epsilon: float,
    edge_filtering_list: wp.array[wp.int32],
    edge_filtering_list_offsets: wp.array[wp.int32],
    min_distance_filtering_ref_pos: wp.array[wp.vec3],
    edge_colliding_edges: wp.array[wp.int32],
    edge_colliding_edges_count: wp.array[wp.int32],
    edge_colliding_edges_min_dist: wp.array[float],
    resize_flags: wp.array[wp.int32],
):
    e_index = wp.tid() * stride + phase
    if e_index >= n_edges:
        return
    e0_v0 = edge_indices[e_index, 2]
    e0_v1 = edge_indices[e_index, 3]
    e0_v0_pos = pos[e0_v0]
    e0_v1_pos = pos[e0_v1]
    lower = wp.min(e0_v0_pos, e0_v1_pos)
    upper = wp.max(e0_v0_pos, e0_v1_pos)
    lower = wp.vec3(lower[0] - max_query_radius, lower[1] - max_query_radius, lower[2] - max_query_radius)
    upper = wp.vec3(upper[0] + max_query_radius, upper[1] + max_query_radius, upper[2] + max_query_radius)
    edge_num_collisions = wp.int32(0)
    min_dis_to_edges = max_query_radius
    edge_world = particle_world[e0_v0]
    for query_pass in range(2):
        run_query = bool(False)
        query_all = bool(False)
        group_root = wp.int32(-1)
        if edge_world < 0:
            if query_pass == 0:
                run_query = True
                query_all = True
        else:
            if query_pass == 0:
                group_root = bvh_group_roots[edge_world]
            else:
                group_root = bvh_group_roots[world_count]
            run_query = group_root >= 0
        if run_query:
            if query_all:
                query = wp.bvh_query_aabb(bvh_id, lower, upper)
            else:
                query = wp.bvh_query_aabb(bvh_id, lower, upper, group_root)
            colliding_edge_index = wp.int32(0)
            while wp.bvh_query_next(query, colliding_edge_index):
                e1_v0 = edge_indices[colliding_edge_index, 2]
                e1_v1 = edge_indices[colliding_edge_index, 3]
                if e0_v0 == e1_v0 or e0_v0 == e1_v1 or e0_v1 == e1_v0 or e0_v1 == e1_v1:
                    continue
                if edge_filtering_list:
                    fl_start = edge_filtering_list_offsets[e_index]
                    fl_end = edge_filtering_list_offsets[e_index + 1]
                    if fl_end > fl_start:
                        first_val = edge_filtering_list[fl_start]
                        last_val = edge_filtering_list[fl_end - 1]
                        if (colliding_edge_index >= first_val) and (colliding_edge_index <= last_val):
                            idx = binary_search(edge_filtering_list, colliding_edge_index, fl_start, fl_end)
                            if idx > fl_start and edge_filtering_list[idx - 1] == colliding_edge_index:
                                continue
                e1_v0_pos = pos[e1_v0]
                e1_v1_pos = pos[e1_v1]
                std = wp.closest_point_edge_edge(e0_v0_pos, e0_v1_pos, e1_v0_pos, e1_v1_pos, edge_edge_parallel_epsilon)
                dist = std[2]
                if min_distance_filtering_ref_pos and min_query_radius > 0.0:
                    std_ref = wp.closest_point_edge_edge(
                        min_distance_filtering_ref_pos[e0_v0], min_distance_filtering_ref_pos[e0_v1],
                        min_distance_filtering_ref_pos[e1_v0], min_distance_filtering_ref_pos[e1_v1],
                        edge_edge_parallel_epsilon)
                    if std_ref[2] < min_query_radius:
                        continue
                if dist < max_query_radius:
                    edge_buffer_offset = edge_colliding_edges_offsets[e_index]
                    edge_buffer_size = edge_colliding_edges_offsets[e_index + 1] - edge_buffer_offset
                    min_dis_to_edges = wp.min(min_dis_to_edges, dist)
                    if edge_num_collisions < edge_buffer_size:
                        edge_colliding_edges[2 * (edge_buffer_offset + edge_num_collisions)] = e_index
                        edge_colliding_edges[2 * (edge_buffer_offset + edge_num_collisions) + 1] = colliding_edge_index
                    else:
                        resize_flags[EDGE_COLLISION_BUFFER_OVERFLOW_INDEX] = 1
                    edge_num_collisions = edge_num_collisions + 1
    edge_colliding_edges_count[e_index] = edge_num_collisions
    edge_colliding_edges_min_dist[e_index] = min_dis_to_edges


# ------------------------------------------------------------------------------------------------ plastic bending
@wp.kernel
def plastic_bending(
    phase: int,
    stride: int,
    n_edges: int,
    pos: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    edge_rest_length: wp.array[float],
    edge_bending_properties: wp.array2d[float],
    yield_angle: float,
    rate: float,
    max_rest: float,
    mass: wp.array[float],
    skip_pinned: int,
    rest_angle: wp.array[float],
    loss: wp.array[float],
):
    """Crease memory: per bending edge e = tid * stride + phase, when the dihedral angle theta (Newton's convention,
    atan2 of the hinge normals) is farther than yield_angle from the rest angle, the rest angle moves toward it by
    rate * (the excess beyond the yield) per step, clamped to +-max_rest. loss[e] = the elastic energy this releases
    (0.5 k L ((theta - r_old)^2 - (theta - r_new)^2) >= 0), booked as plastic dissipation. One write per edge."""
    e = wp.tid() * stride + phase
    if e >= n_edges:
        return
    loss[e] = 0.0
    if edge_indices[e, 0] == -1 or edge_indices[e, 1] == -1 or edge_bending_properties[e, 0] <= 0.0:
        return
    if skip_pinned != 0:       # a hinge touching a pinned (kinematic, mass 0) vertex: the grasp's rigid patch is not a
        for c in range(4):     # material deformation, so it leaves no crease memory
            if mass[edge_indices[e, c]] == 0.0:
                return
    x0 = pos[edge_indices[e, 0]]
    x1 = pos[edge_indices[e, 1]]
    x2 = pos[edge_indices[e, 2]]
    x3 = pos[edge_indices[e, 3]]
    n1 = wp.cross(x2 - x0, x3 - x0)
    n2 = wp.cross(x3 - x1, x2 - x1)
    ev = x3 - x2
    n1n = wp.length(n1)
    n2n = wp.length(n2)
    en = wp.length(ev)
    if n1n < 1.0e-6 or n2n < 1.0e-6 or en < 1.0e-6:
        return
    n1h = n1 / n1n
    n2h = n2 / n2n
    eh = ev / en
    theta = wp.atan2(wp.dot(wp.cross(n1h, n2h), eh), wp.dot(n1h, n2h))
    r = rest_angle[e]
    d = theta - r
    if wp.abs(d) > yield_angle:
        excess = d - wp.sign(d) * yield_angle
        r_new = wp.clamp(r + rate * excess, -max_rest, max_rest)
        k = edge_bending_properties[e, 0] * edge_rest_length[e]
        loss[e] = 0.5 * k * ((theta - r) * (theta - r) - (theta - r_new) * (theta - r_new))
        rest_angle[e] = r_new


@wp.kernel
def sum_serial(values: wp.array[float], n: int, out: wp.array[float]):
    """out[0] = values[0] + ... + values[n-1], in index order (one thread: the same bits on every host)."""
    acc = float(0.0)
    for i in range(n):
        acc += values[i]
    out[0] = acc


class PlasticBending:
    """Crease memory for a Newton cloth model (any device): call after each solver step with the new positions."""

    def __init__(self, model, yield_angle, rate, max_rest=2.7, skip_pinned=True):
        self.model = model
        self.yield_angle, self.rate, self.max_rest = float(yield_angle), float(rate), float(max_rest)
        self.skip_pinned = int(bool(skip_pinned))
        self.loss = wp.zeros(model.edge_count, dtype=float, device=model.device)
        self.total = wp.zeros(1, dtype=float, device=model.device)
        self.loss_j = 0.0           # cumulative plastic dissipation [J] (float64 on the host)

    def args(self, particle_q, phase=0, stride=1):
        m = self.model
        return [phase, stride, m.edge_count, particle_q, m.edge_indices, m.edge_rest_length, m.edge_bending_properties,
                self.yield_angle, self.rate, self.max_rest, m.particle_mass, self.skip_pinned, m.edge_rest_angle,
                self.loss]

    def apply(self, particle_q, book=True):
        wp.launch(plastic_bending, dim=self.model.edge_count, inputs=self.args(particle_q), device=self.model.device)
        if book:
            self.book()

    def book(self):
        wp.launch(sum_serial, dim=1, inputs=[self.loss, self.model.edge_count, self.total], device=self.model.device)
        self.loss_j += float(self.total.numpy()[0])


# ------------------------------------------------------------------------------------------------ direct calls
class _Call:
    """A recorded CPU launch packed once: calling it runs the kernel's compiled entry with the GIL released."""

    def __init__(self, launch):
        kernel, hooks, params = launch.kernel, launch.hooks, launch.params
        fields = [(kernel.adj.args[i].label, type(params[1 + i])) for i in range(len(kernel.adj.args))]
        Args = type("ArgsStruct", (ctypes.Structure,), {"_fields_": fields})
        self.args = Args()
        for i, (name, _) in enumerate(fields):
            setattr(self.args, name, params[1 + i])
        self.bounds = params[0]
        self.fn = hooks.forward
        self.pb, self.pa = ctypes.byref(self.bounds), ctypes.byref(self.args)
        self.launch = launch          # keeps the arrays alive

    def __call__(self):
        self.fn(self.pb, self.pa)


class _Seq:
    def __init__(self, *calls):
        self.calls = calls

    def __call__(self):
        for c in self.calls:
            c()


def _record(kernel, dim, inputs, outputs=()):
    return _Call(wp.launch(kernel, dim=dim, inputs=list(inputs), outputs=list(outputs), device="cpu",
                           record_cmd=True))


_libc = ctypes.CDLL("libc.so.6", use_errno=True)


class _PBarrier:
    """A pthread barrier called through ctypes: the GIL is released while a thread waits (about 4x cheaper per phase
    than threading.Barrier for 4 threads)."""

    def __init__(self, n):
        self.buf = ctypes.create_string_buffer(128)
        if _libc.pthread_barrier_init(self.buf, None, ctypes.c_uint(n)) != 0:
            raise OSError("pthread_barrier_init failed")
        self._wait = _libc.pthread_barrier_wait
        self.ref = self.buf

    def wait(self):
        self._wait(self.ref)


class _Team:
    """threads-1 persistent workers plus the caller. run(program) executes program[k] (a list of callables, one
    barrier after each) on thread k."""

    def __init__(self, n):
        self.n = n
        self.bar = _PBarrier(n)
        self.program = None
        self.closed = False
        self.workers = [threading.Thread(target=self._work, args=(k,), daemon=True) for k in range(1, n)]
        for w in self.workers:
            w.start()

    def _exec(self, k):
        wait = self.bar.wait
        for item in self.program[k]:
            if item is not None:
                item()
            wait()

    def _work(self, k):
        while True:
            self.bar.wait()
            if self.program is None:
                return
            self._exec(k)

    def close(self):
        """Stop the workers (they return at the next barrier) and free the barrier. Idempotent."""
        if self.closed:
            return
        self.closed = True
        if self.n > 1:
            self.program = None
            self.bar.wait()
            for w in self.workers:
                w.join(timeout=5.0)
            if not any(w.is_alive() for w in self.workers):
                _libc.pthread_barrier_destroy(self.bar.ref)

    def run(self, program):
        if self.closed:
            raise RuntimeError("FastSolverVBD: the thread team is closed")
        if self.n == 1:
            for item in program[0]:
                if item is not None:
                    item()
            return
        self.program = program
        self.bar.wait()
        self._exec(0)


def _expand(offsets, counts):
    """The buffer indices offsets[i] + 0 .. counts[i] - 1 for every i, in order."""
    counts = np.maximum(np.asarray(counts, np.int64), 0)
    total = int(counts.sum())
    if not total:
        return np.zeros(0, np.int64)
    starts = np.repeat(np.asarray(offsets, np.int64) - (np.cumsum(counts) - counts), counts)
    return starts + np.arange(total)


# ------------------------------------------------------------------------------------------------ the solver
class FastSolverVBD(SolverVBD):
    """SolverVBD whose particle iterations run on `threads` CPU threads (see the module notes)."""

    def __init__(self, model, *, threads=4, truncate_per="color", tol=0.0, min_iterations=1, dedup=False, plastic=None,
                 coarse_every=2, coarse_patch_hops=4, coarse_sweeps=100, adaptive=None, **kw):
        """tol > 0: a step stops iterating once an iteration's largest vertex update |H^-1 f| is below tol [m]
        (at least min_iterations, at most `iterations`); the decision is an exact max, so it is deterministic."""
        super().__init__(model, **kw)
        self.tol, self.min_iterations = float(tol), int(min_iterations)
        # adaptive iterations: dict(tol [m], block, max_iterations) or None. After the step's
        # `iterations`, further blocks of `block` iterations run while the last iteration's largest vertex update
        # exceeds tol, up to max_iterations (an exact max: deterministic). A step under tol is the plain step.
        self.adaptive = None if not adaptive else dict(tol=float(adaptive.get("tol", 1e-4)),
                                                         block=int(adaptive.get("block", 20)),
                                                         max_iterations=int(adaptive.get("max_iterations", 80)))
        self.dedup = bool(dedup)     # truncation over unique pairs (each unordered edge-edge pair once)
        # plastic bending (crease memory): dict(yield_angle [rad], rate [per step], max_rest [rad]) or None
        self.plastic = PlasticBending(model, **plastic) if plastic else None
        # coarse correction: a Newton step on the common translation of all free particles before iterations
        # coarse_every, 2 coarse_every, ... (0: off); coarse_delta[1 + c] keeps cycle c's step (diagnostics)
        self.coarse_every = int(coarse_every)
        n0 = model.particle_count
        self.coarse_f = wp.zeros(n0, dtype=wp.vec3, device="cpu")
        self.coarse_ha = wp.zeros(n0, dtype=wp.mat33, device="cpu")
        self.coarse_cap = wp.zeros(n0, dtype=float, device="cpu")
        self.coarse_nrm = wp.zeros(n0, dtype=wp.vec3, device="cpu")
        self.coarse_delta = wp.zeros(1 + 64, dtype=wp.vec3, device="cpu")       # [0]: current; [1 + c]: cycle c
        # patch-wise coarse step (coarse_patch_hops > 0): patches = graph balls of that hop radius, greedy in index order
        self.coarse_patch_hops, self.coarse_sweeps = int(coarse_patch_hops), int(coarse_sweeps)
        if self.coarse_patch_hops > 0:
            self._build_patches(model, self.coarse_patch_hops)
        self.iterations_used = []
        if truncate_per not in ("color", "iteration"):
            raise ValueError("truncate_per: 'color' (Newton's order) or 'iteration'")
        self.truncate_per = truncate_per
        if not self.particle_enable_self_contact or not model.device.is_cpu:
            raise ValueError("FastSolverVBD covers the CPU cloth configuration with self-contact")
        if self._integrates_rigid_bodies:
            raise ValueError("FastSolverVBD expects integrate_with_external_rigid_solver=True (or no rigid bodies)")
        if self.rigid_contact_k_start_value >= 0.0:
            raise ValueError("FastSolverVBD expects no body-particle penalty ramp (rigid_avbd_beta 0)")
        self.threads = int(threads)
        self.team = _Team(self.threads)
        # the workers reference only the team, so the solver can be collected; its finalizer stops them
        self._finalizer = weakref.finalize(self, self.team.close)
        n = model.particle_count
        T = self.threads
        self.tbuf = wp.full((T, n), 1.0, dtype=float, device="cpu")
        self.update_norm = wp.zeros(n, dtype=float, device="cpu")
        self.rmax = wp.zeros(T, dtype=float, device="cpu")
        self.inc_offsets = wp.zeros(n + 1, dtype=wp.int32, device="cpu")
        self.n_pairs = wp.zeros(1, dtype=wp.int32, device="cpu")
        self.pair_cap = 0
        self.pairs = None
        self.inc_cap = 0
        self.inc_entries = None
        self.inc_count = 0
        self._ensure_inc(16 * n)
        self._tri = model.tri_indices.numpy()
        self._edge = model.edge_indices.numpy()
        # each color's vertices dealt out to the threads (interleaved: a localized fold spreads over all of them)
        self._color_parts = []
        B = 32                      # blocks of 32 color entries dealt round-robin: balance and cache locality
        for g in model.particle_color_groups:
            ids = np.sort(g.numpy())
            blocks = [ids[i:i + B] for i in range(0, len(ids), B)]
            parts = [np.concatenate(blocks[k::T]) if blocks[k::T] else np.zeros(0, np.int32) for k in range(T)]
            self._color_parts.append([wp.array(pk.astype(np.int32), dtype=wp.int32, device="cpu") for pk in parts])
        self._programs = {}
        self._skip_incidence = False

    def _build_patches(self, model, hops):
        n = model.particle_count
        tri = model.tri_indices.numpy()
        nbr = [set() for _ in range(n)]
        for a, b, c in tri:
            nbr[a].update((b, c)); nbr[b].update((a, c)); nbr[c].update((a, b))
        nbr = [sorted(x) for x in nbr]
        pid = -np.ones(n, np.int32)
        K = 0
        for v0 in range(n):
            if pid[v0] >= 0:
                continue
            front = [v0]
            pid[v0] = K
            for _ in range(hops):
                nxt = []
                for u in front:
                    for w in nbr[u]:
                        if pid[w] < 0:
                            pid[w] = K
                            nxt.append(w)
                front = nxt
            K += 1
        adj = [set() for _ in range(K)]
        for t in tri:
            P = {int(pid[v]) for v in t}
            for I in P:
                adj[I].update(P)
        off = [0]
        idx = []
        for I in range(K):
            row = sorted(adj[I] - {I}) + [I]          # the diagonal block last
            idx += row
            off.append(len(idx))
        self.coarse_K = K
        self.coarse_pid = wp.array(pid, dtype=wp.int32, device="cpu")
        self.coarse_nb_off = wp.array(np.asarray(off, np.int32), dtype=wp.int32, device="cpu")
        self.coarse_nb_idx = wp.array(np.asarray(idx, np.int32), dtype=wp.int32, device="cpu")
        self.coarse_A = wp.zeros(len(idx), dtype=wp.mat33, device="cpu")
        self.coarse_r = wp.zeros(K, dtype=wp.vec3, device="cpu")
        self.coarse_cg = wp.zeros(K, dtype=float, device="cpu")
        self.coarse_ns = wp.zeros(K, dtype=wp.vec3, device="cpu")
        self.coarse_cp = wp.zeros(K, dtype=float, device="cpu")
        self.coarse_cpart = wp.zeros(K, dtype=wp.int32, device="cpu")
        self.coarse_pdelta = wp.zeros(K, dtype=wp.vec3, device="cpu")
        self.coarse_caps = wp.zeros(n, dtype=float, device="cpu")
        self.coarse_partner = wp.zeros(n, dtype=wp.int32, device="cpu")

    def close(self):
        """Stop this solver's worker threads now (also done when the solver is garbage-collected)."""
        self._finalizer()

    # ---------------------------------------------------------------- self-contact incidence lists
    def _ensure_inc(self, cap):
        if cap > self.inc_cap:
            self.inc_cap = int(cap * 1.5)
            self.inc_entries = wp.zeros(self.inc_cap, dtype=wp.vec4i, device="cpu")
            self._inc_np = np.zeros((self.inc_cap, 4), np.int32)

    def _build_incidence(self):
        """Per vertex, its self-contact terms in a fixed order: its vertex-triangle entries (as the vertex, then as
        corner a, b, c; each in buffer order), then its edge-edge entries (as e1's first, then second endpoint)."""
        d = self.trimesh_collision_detector
        n = self.model.particle_count
        vt = d.vertex_colliding_triangles.numpy().reshape(-1, 2)
        vt_off = d.vertex_colliding_triangles_offsets.numpy()[:n]
        vt_cnt = np.minimum(d.vertex_colliding_triangles_count.numpy(),
                            d.vertex_colliding_triangles_buffer_sizes.numpy())
        ee = d.edge_colliding_edges.numpy().reshape(-1, 2)
        ee_cnt = np.minimum(d.edge_colliding_edges_count.numpy(), d.edge_colliding_edges_buffer_sizes.numpy())
        ee_off = d.edge_colliding_edges_offsets.numpy()[:len(ee_cnt)]
        rows = []
        vt_pairs = ee_pairs = None
        idx = _expand(vt_off, vt_cnt)
        if len(idx):
            pv, pt = vt[idx, 0].astype(np.int64), vt[idx, 1].astype(np.int64)
            ok = (pv >= 0) & (pt >= 0)
            pv, pt = pv[ok], pt[ok]
            vt_pairs = (pv, pt)
            tv = self._tri[pt]
            for slot, recv in ((3, pv), (0, tv[:, 0]), (1, tv[:, 1]), (2, tv[:, 2])):
                rows.append(np.stack([recv, np.zeros_like(pv), pv, pt, np.full_like(pv, slot)], 1))
        idx = _expand(ee_off, ee_cnt)
        if len(idx):
            e1, e2 = ee[idx, 0].astype(np.int64), ee[idx, 1].astype(np.int64)
            ok = (e1 >= 0) & (e2 >= 0)
            e1, e2 = e1[ok], e2[ok]
            ee_pairs = (e1, e2)
            for slot in (0, 1):
                recv = self._edge[e1, 2 + slot].astype(np.int64)
                rows.append(np.stack([recv, np.ones_like(e1), e1, e2, np.full_like(e1, slot)], 1))
        if rows:
            R = np.concatenate(rows)
            R = R[np.argsort(R[:, 0], kind="stable")]
            counts = np.bincount(R[:, 0], minlength=n)
        else:
            R = np.zeros((0, 5), np.int64)
            counts = np.zeros(n, np.int64)
        offsets = np.zeros(n + 1, np.int32)
        offsets[1:] = np.cumsum(counts)
        if self.dedup:
            plist = []
            if vt_pairs is not None:
                plist.append(np.stack([np.zeros_like(vt_pairs[0]), vt_pairs[0], vt_pairs[1]], 1))
            if ee_pairs is not None:
                a, b = np.minimum(ee_pairs[0], ee_pairs[1]), np.maximum(ee_pairs[0], ee_pairs[1])
                u = np.unique(np.stack([a, b], 1), axis=0)
                plist.append(np.stack([np.ones(len(u), np.int64), u[:, 0], u[:, 1]], 1))
            PR = np.concatenate(plist) if plist else np.zeros((0, 3), np.int64)
            if len(PR) + 1 > self.pair_cap:
                self.pair_cap = int((len(PR) + 1) * 1.5)
                self.pairs = wp.zeros(self.pair_cap, dtype=wp.vec3i, device="cpu")
                self._pairs_np = np.zeros((self.pair_cap, 3), np.int32)
                self._programs.clear()
            self._pairs_np[:len(PR)] = PR
            self.pairs.assign(self._pairs_np)
            self.n_pairs.assign(np.array([len(PR)], np.int32))
        old = self.inc_entries
        self._ensure_inc(len(R) + 1)
        if self.inc_entries is not old:
            self._programs.clear()
        self.inc_offsets.assign(offsets)
        self._inc_np[:len(R)] = R[:, 1:]
        self.inc_entries.assign(self._inc_np)
        self.inc_count = len(R)

    def _detect_calls(self):
        det, m, T = self.trimesh_collision_detector, self.model, self.threads
        key = ("detect", id(det.vertex_positions), det.bvh_tris.id, det.bvh_edges.id, id(det.vertex_colliding_triangles),
               id(det.edge_colliding_edges))
        prog = self._programs.get(key)
        if prog is not None:
            return prog
        n, ne = m.particle_count, m.edge_count
        r, rmin = self._self_contact_query_radius, self.particle_rest_shape_contact_exclusion_radius
        ref = self.particle_q_rest if self.particle_q_rest is not None else det.vertex_positions
        VT = [_record(vt_detect_strided, (n - k + T - 1) // T,
                      [k, T, n, r, rmin, det.bvh_tris.id, det.bvh_tris_group_roots, det.vertex_positions,
                       m.tri_indices, m.particle_world, m.world_count, det.vertex_colliding_triangles_offsets,
                       det.vertex_triangle_filtering_list, det.vertex_triangle_filtering_list_offsets, ref,
                       det.vertex_colliding_triangles, det.vertex_colliding_triangles_count,
                       det.vertex_colliding_triangles_min_dist, det.resize_flags]) for k in range(T)]
        EE = [_record(ee_detect_strided, (ne - k + T - 1) // T,
                      [k, T, ne, r, rmin, det.bvh_edges.id, det.bvh_edges_group_roots, det.vertex_positions,
                       m.edge_indices, m.particle_world, m.world_count, det.edge_colliding_edges_offsets,
                       det.edge_edge_parallel_epsilon, det.edge_filtering_list, det.edge_filtering_list_offsets, ref,
                       det.edge_colliding_edges, det.edge_colliding_edges_count, det.edge_colliding_edges_min_dist,
                       det.resize_flags]) for k in range(T)]
        prog = [[_Seq(VT[k], EE[k])] for k in range(T)]
        self._programs[key] = prog
        return prog

    def _collision_detection_penetration_free(self, current_state):
        """Newton's detection (positions reset, BVH refit, vertex-triangle and edge-edge queries) with the queries on
        the threads, then the incidence lists."""
        t = time.perf_counter()
        self.pos_prev_collision_detection.assign(current_state.particle_q)
        self.particle_displacements.zero_()
        det = self.trimesh_collision_detector
        det.refit(current_state.particle_q)
        det.vertex_colliding_triangles.fill_(-1)
        det.triangle_colliding_vertices_min_dist.fill_(self._self_contact_query_radius)
        det.edge_colliding_edges.fill_(-1)
        self.team.run(self._detect_calls())
        t1 = time.perf_counter()
        if not self._skip_incidence:      # the iterations' lists; the forward step's truncation reads the buffers
            self._build_incidence()
        if PROFILE:
            TIMES["detect"] += t1 - t
            TIMES["incidence"] += time.perf_counter() - t1

    # ---------------------------------------------------------------- programs
    def _truncation_calls(self, particle_q_out):
        m, T = self.model, self.threads
        P = max(m.particle_count, m.edge_count)
        n = m.particle_count
        maxd = self._self_contact_query_radius * self.particle_conservative_bound_relaxation * 0.5
        if self.dedup:
            B1 = [_record(truncation_pairs_strided, (self.pair_cap - k + T - 1) // T,
                          [k, T, self.n_pairs, self.pairs, self.pos_prev_collision_detection,
                           self.particle_displacements, m.tri_indices, m.edge_indices,
                           self._self_contact_edge_edge_parallel_epsilon, self.particle_conservative_bound_relaxation,
                           self.tbuf[k]]) for k in range(T)]
        else:
            B1 = [_record(truncation_strided, (P - k + T - 1) // T,
                          [k, T, P, self.pos_prev_collision_detection, self.particle_displacements, m.tri_indices,
                           m.edge_indices, self.trimesh_collision_info, self._self_contact_edge_edge_parallel_epsilon,
                           self.particle_conservative_bound_relaxation, self.tbuf[k]]) for k in range(T)]
        cut = [round(i * n / T) for i in range(T + 1)]
        B2 = [_record(truncation_apply_range, cut[k + 1] - cut[k],
                      [cut[k], T, self.tbuf, self.pos_prev_collision_detection, maxd, self.particle_displacements,
                       particle_q_out]) if cut[k + 1] > cut[k] else None for k in range(T)]
        return B1, B2

    def _program(self, state_in, state_out, contacts, dt, it0=0, n_it=None):
        """The program of iterations it0 .. it0 + n_it - 1 (default: the step's 0 .. iterations - 1)."""
        n_it = self.iterations if n_it is None else int(n_it)
        key = (float(dt), id(state_in.particle_q), id(state_out.body_q), id(state_in.body_q), id(state_out.body_qd),
               id(contacts), id(self.body_particle_contact_penalty_k), id(self.inc_entries), self.iterations,
               self.truncate_per, self.tol, id(self.pairs), self.coarse_every, self.coarse_patch_hops, it0, n_it)
        prog = self._programs.get(key)
        if prog is not None:
            return prog
        m, T = self.model, self.threads
        body_q = state_out.body_q if state_out.body_q is not None else self._empty_body_q
        body_q_prev = state_in.body_q if state_in.body_q is not None else self._empty_body_q
        common = [self.particle_q_prev, state_in.particle_q, m.particle_mass, self.inertia, m.particle_flags,
                  m.tri_indices, m.tri_poses, m.tri_materials, m.tri_areas, m.edge_indices, m.edge_rest_angle,
                  m.edge_rest_length, m.edge_bending_properties, self.particle_adjacency, self.friction_epsilon,
                  m.particle_radius, contacts.soft_contact_indices, self._particle_contact_head,
                  self._particle_contact_next, self.body_particle_contact_penalty_k,
                  self.body_particle_contact_material_kd, self.body_particle_contact_material_mu, m.shape_body, body_q,
                  body_q_prev, state_out.body_qd, m.body_com, contacts.soft_contact_shape,
                  contacts.soft_contact_body_pos, contacts.soft_contact_body_vel, contacts.soft_contact_normal,
                  m.shape_margin, contacts.soft_contact_barycentric, self.inc_offsets, self.inc_entries,
                  self.particle_self_contact_margin, m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu,
                  self._self_contact_edge_edge_parallel_epsilon]
        A = [[_record(solve_color, parts[k].shape[0], [float(dt), parts[k], *common],
                      [self.particle_displacements, self.update_norm])
              if parts[k].shape[0] else None for k in range(T)] for parts in self._color_parts]
        B1, B2 = self._truncation_calls(state_in.particle_q)
        one = []
        for c in range(len(A)):
            one.append([A[c][k] for k in range(T)])
            if self.truncate_per == "color" or c == len(A) - 1:
                one.append(B1)
                one.append(B2)
        if self.tol > 0.0:           # one iteration per program run, plus the residual's per-thread max
            n = m.particle_count
            R = [_record(max_strided, 1, [k, T, n, self.update_norm, self.rmax]) for k in range(T)]
            prog = [[phase[k] for phase in one] + [R[k]] for k in range(T)]
        elif self.coarse_every > 0:
            n = m.particle_count
            E = [_record(eval_force_strided, (n - k + T - 1) // T, [k, T, n, float(dt), *common],
                         [self.coarse_f, self.coarse_ha, self.coarse_cap, self.coarse_nrm]) for k in range(T)]
            S = [[_record(coarse_solve, 1, [n, self.coarse_f, self.coarse_ha, self.coarse_cap, self.coarse_nrm,
                                            m.particle_mass, m.particle_flags,
                                            self.particle_q_prev, state_in.particle_q, m.tri_indices, m.tri_poses,
                                            m.tri_materials, m.tri_areas, m.tri_count, float(dt), min(c, 63),
                                            self.coarse_delta])] + [None] * (T - 1)
                 for c in range((it0 + n_it + self.coarse_every - 1) // self.coarse_every)]
            Pa = [_record(coarse_apply_strided, (n - k + T - 1) // T,
                          [k, T, n, m.particle_mass, m.particle_flags, self.coarse_delta, self.particle_displacements])
                  for k in range(T)]
            if self.coarse_patch_hops > 0:
                E = [_record(eval_patch_strided, (n - k + T - 1) // T, [k, T, n, float(dt), *common, self.coarse_pid],
                             [self.coarse_f, self.coarse_ha, self.coarse_cap, self.coarse_nrm, self.coarse_caps,
                              self.coarse_partner]) for k in range(T)]
                S = [[_record(coarse_patch_solve, 1, [
                    n, self.coarse_K, self.coarse_pid, self.coarse_f, self.coarse_ha, self.coarse_cap, self.coarse_nrm,
                    self.coarse_caps, self.coarse_partner, m.particle_mass, m.particle_flags, self.particle_q_prev,
                    state_in.particle_q, m.tri_indices, m.tri_poses, m.tri_materials, m.tri_areas, m.tri_count,
                    float(dt), self.coarse_nb_off, self.coarse_nb_idx, self.coarse_sweeps, self.coarse_A,
                    self.coarse_r, self.coarse_cg, self.coarse_ns, self.coarse_cp, self.coarse_cpart,
                    self.coarse_pdelta])] + [None] * (T - 1) for c in range(len(S))]
                Pa = [_record(coarse_patch_apply_strided, (n - k + T - 1) // T,
                              [k, T, n, m.particle_mass, m.particle_flags, self.coarse_pid, self.coarse_pdelta,
                               self.particle_displacements]) for k in range(T)]
            # pre-smoothing first: no coarse step on the raw prediction (before iteration 0), whose residual is not
            # smooth (a coarse step there popped freshly made flaps open)
            cyc = [[]] + [[E, S[c], Pa, B1, B2] for c in range(1, len(S))]
            prog = [[phase[k] for it in range(it0, it0 + n_it)
                     for phase in ((cyc[it // self.coarse_every] + one) if it % self.coarse_every == 0 else one)]
                    for k in range(T)]
        else:
            prog = [[phase[k] for _ in range(n_it) for phase in one] for k in range(T)]
        self._programs[key] = prog
        return prog

    # ---------------------------------------------------------------- the overridden steps
    def _penetration_free_truncation(self, particle_q_out=None):
        """Newton's truncation after the forward step (_initialize_particles), on the threads."""
        key = ("init", id(particle_q_out), id(self.inc_entries), id(self.pairs))
        prog = self._programs.get(key)
        if prog is None:
            B1, B2 = self._truncation_calls(particle_q_out)
            prog = [[B1[k], B2[k]] for k in range(self.threads)]
            self._programs[key] = prog
        self.team.run(prog)

    def step(self, state_in, state_out, control, contacts, dt):
        """SolverVBD.step for this configuration: the same prologue and epilogue (Newton's own methods); the particle
        iterations run as one threaded program."""
        self._apply_module_options()
        update_rigid = self._update_rigid_history
        self._update_rigid_history = True
        _F = SolverBase.CollisionFrequencyType
        self._sc_mode_this_step, self._sc_freq_this_step = self._resolve_self_contact_schedule()
        if self._sc_mode_this_step not in (_F.PRE_INIT, _F.PRE_POST_INIT):
            raise ValueError("FastSolverVBD supports self-contact detection PRE_INIT or PRE_POST_INIT (the default)")
        self._rigid_mode_this_step = _F.NONE
        self._rigid_freq_this_step = 1
        if self.collision_pipeline is not None:
            raise ValueError("FastSolverVBD expects the caller to run collide (no collision_pipeline)")
        if control is None:
            control = self.model.control(clone_variables=False)
        t0 = time.perf_counter()
        self._initialize_rigid_bodies(state_in, control, contacts, dt, update_rigid)
        self._skip_incidence = self._sc_mode_this_step == _F.PRE_POST_INIT
        try:
            self._initialize_particles(state_in, state_out, dt)
        finally:
            self._skip_incidence = False
        if self._sc_mode_this_step == _F.PRE_POST_INIT:      # Newton: again at the start of iteration 0
            self._collision_detection_penetration_free(state_in)
        t1 = time.perf_counter()
        prog = self._program(state_in, state_out, contacts, dt)
        if self.tol > 0.0:
            used = self.iterations
            for it in range(self.iterations):
                self.team.run(prog)
                if it + 1 >= self.min_iterations and float(self.rmax.numpy().max()) < self.tol:
                    used = it + 1
                    break
            self.iterations_used.append(used)
        else:
            self.team.run(prog)
            if self.adaptive is not None:
                a, done = self.adaptive, self.iterations
                rprog = self._programs.get(("rmax",))
                if rprog is None:
                    n, T = self.model.particle_count, self.threads
                    rprog = [[_record(max_strided, 1, [k, T, n, self.update_norm, self.rmax])] for k in range(T)]
                    self._programs[("rmax",)] = rprog
                while done < a["max_iterations"]:
                    self.team.run(rprog)
                    if float(self.rmax.numpy().max()) <= a["tol"]:
                        break
                    n_it = min(a["block"], a["max_iterations"] - done)
                    self.team.run(self._program(state_in, state_out, contacts, dt, it0=done, n_it=n_it))
                    done += n_it
                self.iterations_used.append(done)
        t2 = time.perf_counter()
        wp.copy(state_out.particle_q, state_in.particle_q)
        if self.plastic is not None:          # crease memory, after the solve: one write per edge
            key = ("plastic", id(state_out.particle_q))
            prog = self._programs.get(key)
            if prog is None:
                T, E = self.threads, self.model.edge_count
                prog = [[_record(plastic_bending, (E - k + T - 1) // T, self.plastic.args(state_out.particle_q, k, T))]
                        for k in range(T)]
                self._programs[key] = prog
            self.team.run(prog)
            self.plastic.book()
        self._snapshot_rigid_contact_history(contacts)
        self._finalize_rigid_bodies(state_in, state_out, dt)
        self._finalize_particles(state_out, dt)
        if PROFILE:
            TIMES["init"] += t1 - t0
            TIMES["iterations"] += t2 - t1
            TIMES["finish"] += time.perf_counter() - t2
