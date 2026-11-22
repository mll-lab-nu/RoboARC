# SPDX-License-Identifier: Apache-2.0
# Kernels adapted from newton/_src/solvers/vbd/particle_vbd_kernels.py (newton 1.6.1,
# Copyright (c) 2025 The Newton Developers, Apache-2.0).
"""A deterministic CUDA step for Newton's VBD cloth: GpuDetSolverVBD, the frozen CPU step
(topogym/world/mujoco/vbd_fast.py, FastSolverVBD 2dbf9fce) moved onto one CUDA device, bit-identical across fresh
processes and across GPU classes (validated on RTX 4090 sm_89, RTX PRO 6000 Blackwell sm_120, H100 sm_90, A100 sm_80).

A drop-in `newton.solvers.SolverVBD` with FastSolverVBD's constructor (threads is accepted and ignored) and the same
`step(state_in, state_out, control, contacts, dt)`, for the same configuration: triangle cloth, self-contact on, rigid
bodies integrated outside (`integrate_with_external_rigid_solver=True`), no body-particle penalty ramp, the caller runs
collide. The same algorithm as the CPU step: Gauss-Seidel per colour with per-vertex gathers, Newton's planar
truncation, the patch-wise coarse correction (`coarse_every`, `coarse_patch_hops`, `coarse_sweeps`) and plastic
bending (`plastic=dict(yield_angle, rate, max_rest, skip_pinned)`, loss booked in float64 as `solver.plastic.loss_j`).
See vbd_fast.py for what each part does, and for `adaptive` (default off): extra blocks of iterations while the last
iteration's largest vertex update (a float atomic max: exact) exceeds tol.

Nondeterminism sources on a GPU and how they are removed:
  - body contacts (Newton appends them by an atomic counter): each particle's contacts are summed in the order of an
    int64 key (shape, corners);
  - self-contact detection (the BVH's build and traversal order the buffers): the pairs are put in a canonical order
    before the per-vertex incidence lists are built, on the device (`inc_tri_buckets` + `inc_build`: order-free
    selections, an integer scan; exactly the host order). Only a buffer overflow would depend on the order: it is
    counted (`overflow`), and a bucket overflow falls back to the host build;
  - truncation: one float atomic_min per vertex (exact, order-free);
  - coarse step and plastic loss: parallel assembly in a fixed order over static entry lists, a one-block multicolour
    projected Gauss-Seidel (`coarse_pgs="tiled"`; "serial" = one thread, the CPU order), the loss on one thread;
  - FMA contraction: off (fuse_fp False) in this module and in every loaded Newton module (`fuse_fp_off`), so every
    architecture computes the same operations. It is process-wide: a process in GPU mode does not run the CPU step.
No float atomics enter any sum. The iterations replay as one CUDA graph per (state_in, state_out, contacts) triple
(`use_graph`).

Measured: a 2.5 ms x 20 iteration step on the 8 mm blanket mesh is about 13-23 ms on an idle GPU; a blanket robot run is
0.72-0.74 s/control end to end on an idle RTX 4090 (CPU step, 4 threads: 1.90). On a GPU shared with another process
every sync waits for time slices (the same run: 1.07 s/control), with the same bits.

Usage rules: every model/state/contacts array is read through pointers recorded once per array object, so update
arrays in place (.assign / copy), never rebind them. The FastSolverVBD experiments (tol, min_iterations,
truncate_per, dedup) are not carried over. VBD_GPU_FUSE=1 turns FMA contraction on (an A/B test only: not portable).
"""
from __future__ import annotations

import os
import sys
import time
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

# FMA contraction off by default (the same arithmetic on every GPU architecture); VBD_GPU_FUSE=1 turns it on for
# this module and leaves Newton's modules at their default (on) -- the A/B test of whether contraction differs by arch
GPU_FUSE = os.environ.get("VBD_GPU_FUSE", "0") == "1"
wp.set_module_options({"enable_backward": False, "fuse_fp": GPU_FUSE})

PROFILE = os.environ.get("FASTVBD_PROFILE") == "1"
# the Gauss-Seidel colour solve with one block of threads per vertex (solve_color_lanes, the same bits as solve_color);
# VBD_GPU_LANES=0 runs the one-thread-per-vertex kernel (A/B)
SOLVE_LANES = os.environ.get("VBD_GPU_LANES", "1") != "0"
TIMES = defaultdict(float)


@wp.func
def _contact_key(c: int, contact_shape: wp.array[int], corners_arr: wp.array[wp.vec3i]):
    """A body contact's order key: shape, then its particle corners (unique per contact)."""
    cr = corners_arr[c]
    return ((wp.int64(contact_shape[c]) + wp.int64(1)) << wp.int64(48)) | (wp.int64(cr[0] + 1) << wp.int64(32)) | (
        wp.int64(cr[1] + 1) << wp.int64(16)) | wp.int64(cr[2] + 1)


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
    lastk = wp.int64(-1)
    while True:
        # the particle's body contacts in a fixed order: by (shape, contact corners), independent of the order in
        # which the GPU collision pass appended them
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


@wp.func
def _rows3(r0: wp.vec3, r1: wp.vec3, r2: wp.vec3):
    return wp.mat33(r0[0], r0[1], r0[2], r1[0], r1[1], r1[2], r2[0], r2[1], r2[2])


LANES = wp.constant(32)


@wp.kernel
def solve_color_lanes(
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
    """solve_color with one block of LANES threads per vertex: the lanes evaluate the vertex's membrane, bending and
    self-contact terms in parallel (each term by exactly the same code), and lane 0 adds them in the original order
    (only the terms the loop would have added), runs the body contacts as before and writes the update. The same
    operations on the same values in the same order as solve_color, so the same bits (speed lane, 2026-10-13)."""
    v, lane = wp.tid()
    particle_index = ids[v]
    if not particle_flags[particle_index] & ParticleFlags.ACTIVE or mass[particle_index] == 0:
        if lane == 0:
            update_norm[particle_index] = 0.0
            particle_displacements[particle_index] = wp.vec3(0.0)
        return

    dt_sqr_reciprocal = 1.0 / (dt * dt)
    f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    n_tri = get_vertex_num_adjacent_faces(adjacency, particle_index)
    for base in range(0, n_tri, LANES):
        k = base + lane
        ok = int(0)
        ft = wp.vec3(0.0)
        h0 = wp.vec3(0.0)
        h1 = wp.vec3(0.0)
        h2 = wp.vec3(0.0)
        if k < n_tri:
            tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, k)
            if tri_materials[tri_index, 0] > 0.0 or tri_materials[tri_index, 1] > 0.0:
                f_tri, h_tri = evaluate_neo_hookean_membrane_force_hessian(
                    tri_index, vertex_order, pos, pos_anchor, tri_indices, tri_poses[tri_index], tri_areas[tri_index],
                    tri_materials[tri_index, 0], tri_materials[tri_index, 1], tri_materials[tri_index, 2], dt)
                ok = 1
                ft = f_tri
                h0 = h_tri[0]
                h1 = h_tri[1]
                h2 = h_tri[2]
        t_ok = wp.tile(ok)
        t_f = wp.tile(ft, preserve_type=True)
        t_h0 = wp.tile(h0, preserve_type=True)
        t_h1 = wp.tile(h1, preserve_type=True)
        t_h2 = wp.tile(h2, preserve_type=True)
        if lane == 0:
            for j in range(wp.min(LANES, n_tri - base)):
                if wp.tile_extract(t_ok, j) == 1:
                    f = f + wp.tile_extract(t_f, j)
                    h = h + _rows3(wp.tile_extract(t_h0, j), wp.tile_extract(t_h1, j), wp.tile_extract(t_h2, j))

    n_edge = get_vertex_num_adjacent_edges(adjacency, particle_index)
    for base in range(0, n_edge, LANES):
        k = base + lane
        ok = int(0)
        fe = wp.vec3(0.0)
        h0 = wp.vec3(0.0)
        h1 = wp.vec3(0.0)
        h2 = wp.vec3(0.0)
        if k < n_edge:
            nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index, k)
            if edge_bending_properties[nei_edge_index, 0] > 0.0:
                f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                    nei_edge_index, vertex_order_on_edge, pos, pos_anchor, edge_indices, edge_rest_angles,
                    edge_rest_length, edge_bending_properties[nei_edge_index, 0],
                    edge_bending_properties[nei_edge_index, 1], dt)
                ok = 1
                fe = f_edge
                h0 = h_edge[0]
                h1 = h_edge[1]
                h2 = h_edge[2]
        t_ok = wp.tile(ok)
        t_f = wp.tile(fe, preserve_type=True)
        t_h0 = wp.tile(h0, preserve_type=True)
        t_h1 = wp.tile(h1, preserve_type=True)
        t_h2 = wp.tile(h2, preserve_type=True)
        if lane == 0:
            for j in range(wp.min(LANES, n_edge - base)):
                if wp.tile_extract(t_ok, j) == 1:
                    f = f + wp.tile_extract(t_f, j)
                    h = h + _rows3(wp.tile_extract(t_h0, j), wp.tile_extract(t_h1, j), wp.tile_extract(t_h2, j))

    fs = wp.vec3(0.0)
    hs = wp.mat33(0.0)
    k0 = inc_offsets[particle_index]
    k1 = inc_offsets[particle_index + 1]
    for base in range(k0, k1, LANES):
        k = base + lane
        cf = wp.vec3(0.0)
        ch = wp.mat33(0.0)
        if k < k1:
            e = inc_entries[k]
            if e[0] == 0:
                cf, ch = evaluate_vertex_triangle_collision_force_hessian(
                    e[1], e[3], e[2], pos, pos_anchor, tri_indices, collision_radius, soft_contact_ke, soft_contact_kd,
                    friction_mu, friction_epsilon, dt)
            else:
                cf, ch = evaluate_edge_edge_contact(
                    particle_index, e[3], e[1], e[2], pos, pos_anchor, edge_indices, collision_radius, soft_contact_ke,
                    soft_contact_kd, friction_mu, friction_epsilon, dt, edge_edge_parallel_epsilon)
        t_f = wp.tile(cf, preserve_type=True)
        t_h0 = wp.tile(ch[0], preserve_type=True)
        t_h1 = wp.tile(ch[1], preserve_type=True)
        t_h2 = wp.tile(ch[2], preserve_type=True)
        if lane == 0:
            for j in range(wp.min(LANES, k1 - base)):
                fs += wp.tile_extract(t_f, j)
                hs += _rows3(wp.tile_extract(t_h0, j), wp.tile_extract(t_h1, j), wp.tile_extract(t_h2, j))

    if lane != 0:
        return
    update_norm[particle_index] = 0.0
    fb = wp.vec3(0.0)
    hb = wp.mat33(0.0)
    head = particle_contact_head[particle_index]
    lastk = wp.int64(-1)
    while True:
        # the particle's body contacts in a fixed order: by (shape, contact corners), independent of the order in
        # which the GPU collision pass appended them
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
    lastk = wp.int64(-1)
    while True:
        # the particle's body contacts in a fixed order: by (shape, contact corners), independent of the order in
        # which the GPU collision pass appended them
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
    lastk = wp.int64(-1)
    while True:
        # the particle's body contacts in a fixed order: by (shape, contact corners), independent of the order in
        # which the GPU collision pass appended them
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
def patch_reduce(
    pv_off: wp.array[wp.int32],
    pv_idx: wp.array[wp.int32],
    f: wp.array[wp.vec3],
    ha: wp.array[wp.mat33],
    capg: wp.array[float],
    nrm: wp.array[wp.vec3],
    caps: wp.array[float],
    partner: wp.array[wp.int32],
    mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    r: wp.array[wp.vec3],
    Ad: wp.array[wp.mat33],
    cg: wp.array[float],
    nsum: wp.array[wp.vec3],
    cp: wp.array[float],
    cpart: wp.array[wp.int32],
):
    """GPU coarse assembly, one thread per patch: its particles' terms summed in index order (as coarse_patch_solve)."""
    I = wp.tid()
    rr = wp.vec3(0.0)
    aa = wp.mat33(0.0)
    c0 = float(0.0)
    ns = wp.vec3(0.0)
    c1 = float(0.0)
    pt = int(-1)
    for k in range(pv_off[I], pv_off[I + 1]):
        i = pv_idx[k]
        if (particle_flags[i] & ParticleFlags.ACTIVE) != 0 and mass[i] > 0.0:
            rr += f[i]
            aa += ha[i]
            c0 += capg[i]
            ns += nrm[i]
            if caps[i] > 0.0 and partner[i] >= 0:
                if pt < 0:
                    pt = partner[i]
                if pt == partner[i]:
                    c1 += caps[i]
                else:
                    c0 += caps[i]
    r[I] = rr
    Ad[I] = aa
    cg[I] = c0
    nsum[I] = ns
    cp[I] = c1
    cpart[I] = pt


@wp.kernel
def tri_blocks(
    pid: wp.array[wp.int32],
    mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    pos_anchor: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    tri_poses: wp.array[wp.mat22],
    tri_materials: wp.array2d[float],
    tri_areas: wp.array[float],
    dt: float,
    TH: wp.array2d[wp.mat33],
    use: wp.array[wp.int32],
):
    """Per triangle (parallel): its three membrane diagonal blocks, when it couples patches or touches a fixed
    particle (coarse_patch_solve's test); use[t] = 0 otherwise."""
    t = wp.tid()
    a = tri_indices[t, 0]
    b = tri_indices[t, 1]
    c = tri_indices[t, 2]
    fa = (particle_flags[a] & ParticleFlags.ACTIVE) != 0 and mass[a] > 0.0
    fb = (particle_flags[b] & ParticleFlags.ACTIVE) != 0 and mass[b] > 0.0
    fc = (particle_flags[c] & ParticleFlags.ACTIVE) != 0 and mass[c] > 0.0
    use[t] = 0
    if fa and fb and fc and pid[a] == pid[b] and pid[b] == pid[c]:
        return
    if not (fa or fb or fc):
        return
    if tri_materials[t, 0] <= 0.0 and tri_materials[t, 1] <= 0.0:
        return
    use[t] = 1
    TH[t, 0] = _tri_block(t, 0, pos, pos_anchor, tri_indices, tri_poses, tri_materials, tri_areas, dt)
    TH[t, 1] = _tri_block(t, 1, pos, pos_anchor, tri_indices, tri_poses, tri_materials, tri_areas, dt)
    TH[t, 2] = _tri_block(t, 2, pos, pos_anchor, tri_indices, tri_poses, tri_materials, tri_areas, dt)


@wp.kernel
def slot_sum(
    K: int,
    nb_off: wp.array[wp.int32],
    se_off: wp.array[wp.int32],
    se: wp.array[wp.vec3i],
    slot_patch: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    TH: wp.array2d[wp.mat33],
    use: wp.array[wp.int32],
    Ad: wp.array[wp.mat33],
    A: wp.array[wp.mat33],
):
    """Per coarse block slot (parallel): the patch's own diagonal (inertia + contact normal blocks) on its diagonal
    slot, plus the membrane blocks H_uv of its static entry list (t, u, v) in a fixed order."""
    q = wp.tid()
    I = slot_patch[q]
    acc = wp.mat33(0.0)
    if q == nb_off[I + 1] - 1:
        acc = Ad[I]
    for k in range(se_off[q], se_off[q + 1]):
        e = se[k]
        t = e[0]
        u = e[1]
        v = e[2]
        if use[t] == 0:
            continue
        vu = tri_indices[t, u]
        vv = tri_indices[t, v]
        fu = (particle_flags[vu] & ParticleFlags.ACTIVE) != 0 and mass[vu] > 0.0
        fv = (particle_flags[vv] & ParticleFlags.ACTIVE) != 0 and mass[vv] > 0.0
        if not (fu and fv):
            continue
        H = TH[t, u]
        if u != v:
            lo = wp.min(u, v)
            hi = wp.max(u, v)
            H = 0.5 * (TH[t, 3 - lo - hi] - TH[t, lo] - TH[t, hi])
        acc += H
    A[q] = acc


@wp.kernel
def coarse_patch_pgs(
    K: int,
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
    """coarse_patch_solve's global step and projected block Gauss-Seidel sweeps (one thread), on assembled blocks."""
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
def coarse_patch_pgs_tiled(
    K: int,
    nb_off: wp.array[wp.int32],
    nb_idx: wp.array[wp.int32],
    color: wp.array[wp.int32],
    ncolors: int,
    sweeps: int,
    A: wp.array[wp.mat33],
    r: wp.array[wp.vec3],
    cg: wp.array[float],
    nsum: wp.array[wp.vec3],
    cp: wp.array[float],
    cpart: wp.array[wp.int32],
    delta: wp.array[wp.vec3],
):
    """coarse_patch_pgs in one thread block, one thread per patch: lane 0 takes the global step; the sweeps are a
    multicolour projected block Gauss-Seidel (patches of one colour share no block, so they update together); the
    current translations are exchanged through a shared tile between colours. Deterministic (no atomics)."""
    _blk, I = wp.tid()
    dg = wp.vec3(0.0)
    if I == 0:
        rg = wp.vec3(0.0)
        Ag = wp.mat33(0.0)
        cgt = float(0.0)
        ng = wp.vec3(0.0)
        for J in range(K):
            rg += r[J]
            cgt += cg[J]
            ng += nsum[J]
            for q in range(nb_off[J], nb_off[J + 1]):
                Ag += A[q]
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
    tg = wp.tile(dg, preserve_type=True)
    my = wp.tile_extract(tg, 0)
    # this thread's patch constants, read once (the sweeps below compute exactly what the per-sweep reads did: the same
    # values, the same operations in the same order; speed lane, 2026-10-13)
    mine = int(0)
    q0 = int(0)
    qd = int(0)
    AII = wp.mat33(0.0)
    Ainv = wp.mat33(0.0)
    rI = wp.vec3(0.0)
    cgI = float(0.0)
    cpI = float(0.0)
    ctot = float(0.0)
    cpart_I = int(-1)
    nl = float(0.0)
    nh = wp.vec3(0.0)
    ann = float(0.0)
    colI = int(-1)
    if I < K:
        qd = nb_off[I + 1] - 1
        AII = A[qd]
        if wp.abs(wp.determinant(AII)) > 1e-8:
            mine = 1
            Ainv = wp.inverse(AII)
        q0 = nb_off[I]
        rI = r[I]
        cgI = cg[I]
        cpI = cp[I]
        ctot = cgI + cpI
        cpart_I = cpart[I]
        nl = wp.length(nsum[I])
        if nl > 0.0 and ctot > 0.0:
            nh = nsum[I] / nl
            ann = wp.dot(nh, AII * nh)
        colI = color[I]
    for sw in range(sweeps):
        for c in range(ncolors):
            td = wp.tile(my, preserve_type=True)
            if mine == 1 and colI == c:
                rho = rI
                for q in range(q0, qd):
                    rho -= A[q] * wp.tile_extract(td, nb_idx[q])
                anchor = wp.vec3(0.0)
                if cpart_I >= 0 and cpI > 0.0:
                    anchor = wp.tile_extract(td, cpart_I) * (cpI / ctot)
                d = Ainv * rho
                if nl > 0.0 and ctot > 0.0:
                    at = anchor - wp.dot(anchor, nh) * nh
                    if ann > 0.0:
                        s = (wp.dot(nh, rho) - wp.dot(nh, AII * at)) / ann
                        ds = at + s * nh
                        fh = rho - AII * ds
                        ft = fh - wp.dot(fh, nh) * nh
                        ftl = wp.length(ft)
                        if ftl <= ctot:
                            d = ds
                        else:
                            d = Ainv * (rho - ft * (ctot / ftl))
                my = d
    if I < K:
        delta[I] = my


@wp.kernel
def inc_tri_buckets(
    n: int,
    vt_off: wp.array[wp.int32],
    vt_cnt: wp.array[wp.int32],
    vt_bs: wp.array[wp.int32],
    vt: wp.array[wp.int32],
    cap: int,
    tri_cnt: wp.array[wp.int32],
    tri_bucket: wp.array[wp.int32],
    flags: wp.array[wp.int32],
):
    """Per vertex: its vertex-triangle pairs scattered to their triangles' buckets (slot order varies run to run;
    the buckets are only read through order-free selections). flags[0] counts bucket overflows, flags[1] collision
    buffer overflows."""
    v = wp.tid()
    if v >= n:
        return
    c = vt_cnt[v]
    if c > vt_bs[v]:
        wp.atomic_add(flags, 1, 1)
        c = vt_bs[v]
    for k in range(c):
        e = vt_off[v] + k
        pv = vt[2 * e]
        pt = vt[2 * e + 1]
        if pv >= 0 and pt >= 0:
            sl = wp.atomic_add(tri_cnt, pt, 1)
            if sl < cap:
                tri_bucket[pt * cap + sl] = pv
            else:
                wp.atomic_add(flags, 0, 1)


@wp.func
def _next_row_min(base: int, cnt: int, vals: wp.array[wp.int32], stride: int, col: int, last: int):
    """The smallest value > last among vals[(base + k) * stride + col], k < cnt (-1 entries skipped); -1 if none."""
    best = int(2147483647)
    for k in range(cnt):
        x = vals[(base + k) * stride + col]
        if x > last and x < best:
            best = x
    if best == 2147483647:
        return -1
    return best


@wp.kernel
def inc_build(
    n: int,
    write: int,
    vt_off: wp.array[wp.int32],
    vt_cnt: wp.array[wp.int32],
    vt_bs: wp.array[wp.int32],
    vt: wp.array[wp.int32],
    ee_off: wp.array[wp.int32],
    ee_cnt: wp.array[wp.int32],
    ee_bs: wp.array[wp.int32],
    ee: wp.array[wp.int32],
    cap: int,
    tri_cnt: wp.array[wp.int32],
    tri_bucket: wp.array[wp.int32],
    vc_off: wp.array2d[wp.int32],
    vc_tri: wp.array[wp.int32],
    ve_off: wp.array2d[wp.int32],
    ve_edge: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    entries: wp.array[wp.vec4i],
):
    """Per receiver vertex v, its self-contact incidence entries in exactly the host order (_build_incidence):
    vertex-triangle as the vertex (by triangle), as triangle corner 0, 1, 2 (each by (vertex, triangle)), then
    edge-edge as edge endpoint 0, 1 (each by (e1, e2)). write = 0: count only; 1: write at offsets[v]."""
    v = wp.tid()
    if v >= n:
        return
    w = int(0)
    o = int(0)
    if write != 0:
        o = offsets[v]
    # as the vertex: its own row, by triangle
    c = wp.min(vt_cnt[v], vt_bs[v])
    last = int(-1)
    while True:
        pt = _next_row_min(vt_off[v], c, vt, 2, 1, last)
        if pt < 0:
            break
        last = pt
        if write != 0:
            entries[o + w] = wp.vec4i(0, v, pt, 3)
        w += 1
    # as a triangle corner: the pairs (pv, t) over v's corner-k triangles, by (pv, t)
    for kk in range(3):
        lastpv = int(-1)
        lastt = int(-1)
        while True:
            bpv = int(2147483647)
            bt = int(2147483647)
            for q in range(vc_off[v, kk], vc_off[v, kk + 1]):
                t = vc_tri[q]
                m = wp.min(tri_cnt[t], cap)
                for j in range(m):
                    pv = tri_bucket[t * cap + j]
                    after = pv > lastpv or (pv == lastpv and t > lastt)
                    better = pv < bpv or (pv == bpv and t < bt)
                    if after and better:
                        bpv = pv
                        bt = t
            if bpv == 2147483647:
                break
            lastpv = bpv
            lastt = bt
            if write != 0:
                entries[o + w] = wp.vec4i(0, bpv, bt, kk)
            w += 1
    # edge-edge as endpoint 0 / 1 of e1: e1 ascending (static list), each row by e2
    for kk in range(2):
        for q in range(ve_off[v, kk], ve_off[v, kk + 1]):
            e1 = ve_edge[q]
            ce = wp.min(ee_cnt[e1], ee_bs[e1])
            le = int(-1)
            while True:
                e2 = _next_row_min(ee_off[e1], ce, ee, 2, 1, le)
                if e2 < 0:
                    break
                le = e2
                if write != 0:
                    entries[o + w] = wp.vec4i(1, e1, e2, kk)
                w += 1
    if write == 0:
        counts[v] = w


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
    wp.atomic_min(buf, i, t)          # GPU: one buffer for all threads; a float min is exact and order-free


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
def _expand(offsets, counts):
    """The buffer indices offsets[i] + 0 .. counts[i] - 1 for every i, in order."""
    counts = np.maximum(np.asarray(counts, np.int64), 0)
    total = int(counts.sum())
    if not total:
        return np.zeros(0, np.int64)
    starts = np.repeat(np.asarray(offsets, np.int64) - (np.cumsum(counts) - counts), counts)
    return starts + np.arange(total)


# ------------------------------------------------------------------------------------------------ the solver

@wp.kernel
def max_update(update_norm: wp.array[float], out: wp.array[float]):
    """adaptive iterations: out[0] = the largest vertex update of the last iteration (a float max: exact, order-free)."""
    wp.atomic_max(out, 0, update_norm[wp.tid()])


@wp.kernel
def accumulate_f64(total: wp.array[float], acc: wp.array[wp.float64]):
    acc[0] = acc[0] + wp.float64(total[0])


class _GpuPlastic(PlasticBending):
    """PlasticBending whose loss is summed and accumulated on the device (one thread, index order, float64): no host
    sync per step. loss_j reads (and assigns) the device accumulator."""

    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        self._acc = wp.zeros(1, dtype=wp.float64, device=model.device)

    @property
    def loss_j(self):
        return float(self._acc.numpy()[0]) if hasattr(self, "_acc") else 0.0

    @loss_j.setter
    def loss_j(self, v):
        if hasattr(self, "_acc"):
            self._acc.assign(np.array([v], np.float64))

    def book(self):
        wp.launch(sum_serial, dim=1, inputs=[self.loss, self.model.edge_count, self.total], device=self.model.device)
        wp.launch(accumulate_f64, dim=1, inputs=[self.total, self._acc], device=self.model.device)


_NEWTON_FUSE_OFF = [-1]     # len(sys.modules) at the last scan


def fuse_fp_off():
    """No FMA contraction in Newton's own kernels either (forward step, contacts, finalize): the same arithmetic on
    every GPU architecture. Process-wide: do not run the CPU step (vbd_fast) in the same process. Idempotent and cheap:
    it scans the loaded modules again only when more were imported since its last call (the solver calls it before
    every step, so a Newton module imported late is covered before its first launch)."""
    if GPU_FUSE or len(sys.modules) == _NEWTON_FUSE_OFF[0]:
        return
    for name, mod in list(sys.modules.items()):
        if name.startswith("newton") and mod is not None and hasattr(mod, "__file__"):
            try:
                if wp.get_module_options(mod).get("fuse_fp", True) is not False:
                    wp.set_module_options({"fuse_fp": False}, module=mod)
            except Exception:
                pass
    _NEWTON_FUSE_OFF[0] = len(sys.modules)


_newton_fuse_fp_off = fuse_fp_off


class GpuDetSolverVBD(SolverVBD):
    """The frozen CPU step (vbd_fast 2dbf9fce) on one CUDA device, deterministic: per-vertex gathers (no float
    atomics in any sum), body contacts summed in (shape, corners) order, self-contact incidence in a canonical order
    (independent of the BVH), truncation by exact float mins, coarse steps and loss sums on one thread, no FMA
    contraction. Same constructor as FastSolverVBD (without threads)."""

    def __init__(self, model, *, plastic=None, coarse_every=2, coarse_patch_hops=4, coarse_sweeps=100, threads=None,
                 coarse_pgs="tiled", use_graph=True, gpu_incidence=True, adaptive=None, **kw):
        _newton_fuse_fp_off()
        super().__init__(model, **kw)
        d = model.device
        if d.is_cpu or not self.particle_enable_self_contact:
            raise ValueError("GpuDetSolverVBD covers the CUDA cloth configuration with self-contact")
        if self._integrates_rigid_bodies:
            raise ValueError("GpuDetSolverVBD expects integrate_with_external_rigid_solver=True (or no rigid bodies)")
        if self.rigid_contact_k_start_value >= 0.0:
            raise ValueError("GpuDetSolverVBD expects no body-particle penalty ramp (rigid_avbd_beta 0)")
        self.dev = d
        # adaptive iterations, as vbd_fast: dict(tol [m], block, max_iterations) or None (off)
        self.adaptive = None if not adaptive else dict(tol=float(adaptive.get("tol", 1e-4)),
                                                         block=int(adaptive.get("block", 20)),
                                                         max_iterations=int(adaptive.get("max_iterations", 80)))
        self.iterations_used = []
        self._umax = wp.zeros(1, dtype=float, device=d)
        self._programs = {}
        self.coarse_pgs = coarse_pgs      # "tiled" (one block, multicolour) or "serial" (one thread, the CPU order)
        self.use_graph = bool(use_graph)  # replay the iteration launches as a CUDA graph
        self.plastic = _GpuPlastic(model, **plastic) if plastic else None
        self.coarse_every = int(coarse_every)
        self.coarse_patch_hops, self.coarse_sweeps = int(coarse_patch_hops), int(coarse_sweeps)
        if self.coarse_every > 0 and self.coarse_patch_hops <= 0:
            raise ValueError("GpuDetSolverVBD implements the patch-wise coarse step (coarse_patch_hops > 0) or none")
        n = model.particle_count
        self.coarse_f = wp.zeros(n, dtype=wp.vec3, device=d)
        self.coarse_ha = wp.zeros(n, dtype=wp.mat33, device=d)
        self.coarse_cap = wp.zeros(n, dtype=float, device=d)
        self.coarse_nrm = wp.zeros(n, dtype=wp.vec3, device=d)
        if self.coarse_every > 0:
            self._build_patches(model, self.coarse_patch_hops)
        self.tbuf = wp.full((1, n), 1.0, dtype=float, device=d)
        self.inc_offsets = wp.zeros(n + 1, dtype=wp.int32, device=d)
        self.inc_cap = 0
        self.inc_entries = None
        self.inc_count = 0
        self._ensure_inc(16 * n)
        self._tri = model.tri_indices.numpy()
        self._edge = model.edge_indices.numpy()
        self._colors = [wp.array(np.sort(g.numpy()).astype(np.int32), dtype=wp.int32, device=d)
                        for g in model.particle_color_groups]
        self._programs = {}
        self._skip_incidence = False
        self.overflow = 0          # collision buffer overflows seen (the kept set would depend on the BVH order)
        self.gpu_incidence = bool(gpu_incidence)
        if self.gpu_incidence:
            self._init_gpu_incidence(model)

    def close(self):
        pass

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
        off, idx = [0], []
        for I in range(K):
            row = sorted(adj[I] - {I}) + [I]
            idx += row
            off.append(len(idx))
        d = self.dev
        self.coarse_K = K
        self.coarse_pid = wp.array(pid, dtype=wp.int32, device=d)
        self.coarse_nb_off = wp.array(np.asarray(off, np.int32), dtype=wp.int32, device=d)
        self.coarse_nb_idx = wp.array(np.asarray(idx, np.int32), dtype=wp.int32, device=d)
        self.coarse_A = wp.zeros(len(idx), dtype=wp.mat33, device=d)
        self.coarse_r = wp.zeros(K, dtype=wp.vec3, device=d)
        self.coarse_cg = wp.zeros(K, dtype=float, device=d)
        self.coarse_ns = wp.zeros(K, dtype=wp.vec3, device=d)
        self.coarse_cp = wp.zeros(K, dtype=float, device=d)
        self.coarse_cpart = wp.zeros(K, dtype=wp.int32, device=d)
        self.coarse_pdelta = wp.zeros(K, dtype=wp.vec3, device=d)
        self.coarse_caps = wp.zeros(n, dtype=float, device=d)
        self.coarse_partner = wp.zeros(n, dtype=wp.int32, device=d)
        # parallel assembly: each patch's particles (index order); each block slot's (triangle, u, v) entries
        pv_idx = np.argsort(pid, kind="stable").astype(np.int32)
        pv_off = np.zeros(K + 1, np.int32)
        pv_off[1:] = np.cumsum(np.bincount(pid, minlength=K))
        slot = {}
        slot_patch = np.zeros(len(idx), np.int32)
        for I in range(K):
            for q in range(off[I], off[I + 1]):
                slot[(I, idx[q])] = q
                slot_patch[q] = I
        ent = [[] for _ in range(len(idx))]
        for t, tv in enumerate(tri):
            for u in range(3):
                for v in range(3):
                    ent[slot[(int(pid[tv[u]]), int(pid[tv[v]]))]].append((t, u, v))
        se_off = np.zeros(len(idx) + 1, np.int32)
        se_off[1:] = np.cumsum([len(x) for x in ent])
        se = np.asarray([e for x in ent for e in x], np.int32).reshape(-1, 3)
        self.coarse_pv_off = wp.array(pv_off, dtype=wp.int32, device=d)
        self.coarse_pv_idx = wp.array(pv_idx, dtype=wp.int32, device=d)
        self.coarse_slot_patch = wp.array(slot_patch, dtype=wp.int32, device=d)
        self.coarse_se_off = wp.array(se_off, dtype=wp.int32, device=d)
        self.coarse_se = wp.array(se, dtype=wp.vec3i, device=d)
        self.coarse_Ad = wp.zeros(K, dtype=wp.mat33, device=d)
        # patch graph colouring (greedy, index order) for the multicolour sweeps; one thread block of BD >= K threads
        col = -np.ones(K, np.int32)
        for I in range(K):
            used = {int(col[J]) for J in idx[off[I]:off[I + 1]] if col[J] >= 0}
            c = 0
            while c in used:
                c += 1
            col[I] = c
        self.coarse_color = wp.array(col, dtype=wp.int32, device=d)
        self.coarse_ncolors = int(col.max()) + 1
        self.coarse_bd = max(32, 1 << int(np.ceil(np.log2(K))))
        if self.coarse_bd > 1024:
            raise ValueError(f"{K} patches: more than one thread block; raise coarse_patch_hops")
        self.coarse_TH = wp.zeros((len(tri), 3), dtype=wp.mat33, device=d)
        self.coarse_use = wp.zeros(len(tri), dtype=wp.int32, device=d)

    # ---------------------------------------------------------------- self-contact incidence lists (host, canonical)
    def _ensure_inc(self, cap):
        if cap > self.inc_cap:
            self.inc_cap = int(cap * 1.5)
            self.inc_entries = wp.zeros(self.inc_cap, dtype=wp.vec4i, device=self.dev)
            self._inc_np = np.zeros((self.inc_cap, 4), np.int32)
            self._programs.clear()

    def _init_gpu_incidence(self, model):
        n, d = model.particle_count, self.dev
        tri, edge = self._tri, self._edge
        vc = [[[] for _ in range(3)] for _ in range(n)]
        for t, tv in enumerate(tri):
            for k in range(3):
                vc[tv[k]][k].append(t)
        ve = [[[] for _ in range(2)] for _ in range(n)]
        for e, ev in enumerate(edge):
            for k in range(2):
                if ev[2 + k] >= 0:
                    ve[ev[2 + k]][k].append(e)
        def csr(lists, K):
            off = np.zeros((n, K + 1), np.int32)
            flat = []
            for v in range(n):
                off[v, 0] = len(flat)
                for k in range(K):
                    flat += sorted(lists[v][k])
                    off[v, k + 1] = len(flat)
            return off, np.asarray(flat if flat else [0], np.int32)
        vc_off, vc_tri = csr(vc, 3)
        ve_off, ve_edge = csr(ve, 2)
        self.ig_vc_off = wp.array(vc_off, dtype=wp.int32, device=d)
        self.ig_vc_tri = wp.array(vc_tri, dtype=wp.int32, device=d)
        self.ig_ve_off = wp.array(ve_off, dtype=wp.int32, device=d)
        self.ig_ve_edge = wp.array(ve_edge, dtype=wp.int32, device=d)
        self.ig_cap = 128
        self.ig_tri_cnt = wp.zeros(len(tri), dtype=wp.int32, device=d)
        self.ig_tri_bucket = wp.zeros(len(tri) * self.ig_cap, dtype=wp.int32, device=d)
        self.ig_flags = wp.zeros(2, dtype=wp.int32, device=d)
        self.ig_counts = wp.zeros(n, dtype=wp.int32, device=d)
        self._ensure_inc(64 * n)
        self.ig_calls = None

    def _build_incidence_gpu(self):
        """_build_incidence on the device, in exactly its order (see inc_build); one small host read of the
        overflow flags and the entry total per call."""
        det = self.trimesh_collision_detector
        n = self.model.particle_count
        if self.ig_calls is None or self.ig_calls[0] is not self.inc_entries:
            args = [det.vertex_colliding_triangles_offsets, det.vertex_colliding_triangles_count,
                    det.vertex_colliding_triangles_buffer_sizes, det.vertex_colliding_triangles]
            eargs = [det.edge_colliding_edges_offsets, det.edge_colliding_edges_count,
                     det.edge_colliding_edges_buffer_sizes, det.edge_colliding_edges]
            tb = self._rec(inc_tri_buckets, n, [n, *args, self.ig_cap, self.ig_tri_cnt, self.ig_tri_bucket,
                                                 self.ig_flags])
            common = [*args, *eargs, self.ig_cap, self.ig_tri_cnt, self.ig_tri_bucket, self.ig_vc_off, self.ig_vc_tri,
                      self.ig_ve_off, self.ig_ve_edge, self.ig_counts, self.inc_offsets, self.inc_entries]
            c0 = self._rec(inc_build, n, [n, 0, *common])
            c1 = self._rec(inc_build, n, [n, 1, *common])
            self.ig_calls = (self.inc_entries, tb, c0, c1)
        _, tb, c0, c1 = self.ig_calls
        self.ig_tri_cnt.zero_()
        self.ig_flags.zero_()
        tb.launch()
        c0.launch()
        self.inc_offsets[0:1].zero_()
        wp.utils.array_scan(self.ig_counts, self.inc_offsets[1:], inclusive=True)
        fl = self.ig_flags.numpy()
        tot = int(self.inc_offsets[n:n + 1].numpy()[0])
        if fl[0] > 0 or tot + 1 > self.inc_cap:
            # a triangle bucket overflowed (the kept pairs would depend on the GPU's scatter order) or the entry
            # buffer is too small: the host build for this call
            self._build_incidence()
            return
        self.overflow += int(fl[1])
        c1.launch()
        self.inc_count = tot

    def _build_incidence(self):
        """FastSolverVBD's per-vertex incidence lists, with the detected pairs first put in a canonical order (sorted
        by their indices), so the lists do not depend on the BVH's traversal order on the GPU."""
        det = self.trimesh_collision_detector
        n = self.model.particle_count
        vt = det.vertex_colliding_triangles.numpy().reshape(-1, 2)
        vt_off = det.vertex_colliding_triangles_offsets.numpy()[:n]
        vt_cnt_raw = det.vertex_colliding_triangles_count.numpy()
        vt_bs = det.vertex_colliding_triangles_buffer_sizes.numpy()
        vt_cnt = np.minimum(vt_cnt_raw, vt_bs)
        ee = det.edge_colliding_edges.numpy().reshape(-1, 2)
        ee_cnt_raw = det.edge_colliding_edges_count.numpy()
        ee_bs = det.edge_colliding_edges_buffer_sizes.numpy()
        ee_cnt = np.minimum(ee_cnt_raw, ee_bs)
        ee_off = det.edge_colliding_edges_offsets.numpy()[:len(ee_cnt)]
        self.overflow += int((vt_cnt_raw[:len(vt_bs)] > vt_bs).sum() + (ee_cnt_raw > ee_bs).sum())
        rows = []
        idx = _expand(vt_off, vt_cnt)
        if len(idx):
            pv, pt = vt[idx, 0].astype(np.int64), vt[idx, 1].astype(np.int64)
            ok = (pv >= 0) & (pt >= 0)
            pv, pt = pv[ok], pt[ok]
            o = np.lexsort((pt, pv))
            pv, pt = pv[o], pt[o]
            tv = self._tri[pt]
            for slot, recv in ((3, pv), (0, tv[:, 0]), (1, tv[:, 1]), (2, tv[:, 2])):
                rows.append(np.stack([recv, np.zeros_like(pv), pv, pt, np.full_like(pv, slot)], 1))
        idx = _expand(ee_off, ee_cnt)
        if len(idx):
            e1, e2 = ee[idx, 0].astype(np.int64), ee[idx, 1].astype(np.int64)
            ok = (e1 >= 0) & (e2 >= 0)
            e1, e2 = e1[ok], e2[ok]
            o = np.lexsort((e2, e1))
            e1, e2 = e1[o], e2[o]
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
        self._ensure_inc(len(R) + 1)
        self.inc_offsets.assign(offsets)
        self._inc_np[:len(R)] = R[:, 1:]
        self.inc_entries.assign(self._inc_np)
        self.inc_count = len(R)

    # ---------------------------------------------------------------- recorded launches
    def _rec(self, kernel, dim, inputs, outputs=(), block_dim=256):
        return wp.launch(kernel, dim=dim, inputs=list(inputs), outputs=list(outputs), device=self.dev, record_cmd=True,
                         block_dim=block_dim)

    def _detect_calls(self):
        det, m = self.trimesh_collision_detector, self.model
        key = ("detect", id(det.vertex_positions), det.bvh_tris.id, det.bvh_edges.id, id(det.vertex_colliding_triangles),
               id(det.edge_colliding_edges))
        prog = self._programs.get(key)
        if prog is not None:
            return prog
        n, ne = m.particle_count, m.edge_count
        r, rmin = self._self_contact_query_radius, self.particle_rest_shape_contact_exclusion_radius
        ref = self.particle_q_rest if self.particle_q_rest is not None else det.vertex_positions
        prog = [self._rec(vt_detect_strided, n,
                          [0, 1, n, r, rmin, det.bvh_tris.id, det.bvh_tris_group_roots, det.vertex_positions,
                           m.tri_indices, m.particle_world, m.world_count, det.vertex_colliding_triangles_offsets,
                           det.vertex_triangle_filtering_list, det.vertex_triangle_filtering_list_offsets, ref,
                           det.vertex_colliding_triangles, det.vertex_colliding_triangles_count,
                           det.vertex_colliding_triangles_min_dist, det.resize_flags]),
                self._rec(ee_detect_strided, ne,
                          [0, 1, ne, r, rmin, det.bvh_edges.id, det.bvh_edges_group_roots, det.vertex_positions,
                           m.edge_indices, m.particle_world, m.world_count, det.edge_colliding_edges_offsets,
                           det.edge_edge_parallel_epsilon, det.edge_filtering_list, det.edge_filtering_list_offsets, ref,
                           det.edge_colliding_edges, det.edge_colliding_edges_count, det.edge_colliding_edges_min_dist,
                           det.resize_flags])]
        self._programs[key] = prog
        return prog

    def _collision_detection_penetration_free(self, current_state):
        self.pos_prev_collision_detection.assign(current_state.particle_q)
        self.particle_displacements.zero_()
        det = self.trimesh_collision_detector
        det.refit(current_state.particle_q)
        det.vertex_colliding_triangles.fill_(-1)
        det.triangle_colliding_vertices_min_dist.fill_(self._self_contact_query_radius)
        det.edge_colliding_edges.fill_(-1)
        for c in self._detect_calls():
            c.launch()
        if not self._skip_incidence:
            if self.gpu_incidence:
                self._build_incidence_gpu()
            else:
                self._build_incidence()

    def _truncation_calls(self, particle_q_out):
        m = self.model
        P = max(m.particle_count, m.edge_count)
        n = m.particle_count
        maxd = self._self_contact_query_radius * self.particle_conservative_bound_relaxation * 0.5
        B1 = self._rec(truncation_strided, P,
                       [0, 1, P, self.pos_prev_collision_detection, self.particle_displacements, m.tri_indices,
                        m.edge_indices, self.trimesh_collision_info, self._self_contact_edge_edge_parallel_epsilon,
                        self.particle_conservative_bound_relaxation, self.tbuf[0]])
        B2 = self._rec(truncation_apply_range, n,
                       [0, 1, self.tbuf, self.pos_prev_collision_detection, maxd, self.particle_displacements,
                        particle_q_out])
        return B1, B2

    def _program(self, state_in, state_out, contacts, dt, it0=0, n_it=None):
        """The launches of iterations it0 .. it0 + n_it - 1 (default: the step's 0 .. iterations - 1)."""
        n_it = self.iterations if n_it is None else int(n_it)
        key = (float(dt), id(state_in.particle_q), id(state_out.body_q), id(state_in.body_q), id(state_out.body_qd),
               id(contacts), id(self.body_particle_contact_penalty_k), id(self.inc_entries), self.iterations,
               self.coarse_every, self.coarse_pgs, it0, n_it)
        prog = self._programs.get(key)
        if prog is not None:
            return prog
        m = self.model
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
        self._update_norm = getattr(self, "_update_norm", None) or wp.zeros(m.particle_count, dtype=float,
                                                                             device=self.dev)
        if SOLVE_LANES:     # one block of LANES threads per vertex (solve_color_lanes: the same bits)
            A = [self._rec(solve_color_lanes, (ids.shape[0], LANES), [float(dt), ids, *common],
                           [self.particle_displacements, self._update_norm], block_dim=LANES)
                 for ids in self._colors if ids.shape[0]]
        else:
            A = [self._rec(solve_color, ids.shape[0], [float(dt), ids, *common],
                           [self.particle_displacements, self._update_norm]) for ids in self._colors if ids.shape[0]]
        B1, B2 = self._truncation_calls(state_in.particle_q)
        one = []
        for a in A:
            one += [a, B1, B2]
        cyc = []
        if self.coarse_every > 0:
            n = m.particle_count
            E = self._rec(eval_patch_strided, n, [0, 1, n, float(dt), *common, self.coarse_pid],
                          [self.coarse_f, self.coarse_ha, self.coarse_cap, self.coarse_nrm, self.coarse_caps,
                           self.coarse_partner])
            K = self.coarse_K
            S = [self._rec(patch_reduce, K, [self.coarse_pv_off, self.coarse_pv_idx, self.coarse_f, self.coarse_ha,
                                             self.coarse_cap, self.coarse_nrm, self.coarse_caps, self.coarse_partner,
                                             m.particle_mass, m.particle_flags, self.coarse_r, self.coarse_Ad,
                                             self.coarse_cg, self.coarse_ns, self.coarse_cp, self.coarse_cpart]),
                 self._rec(tri_blocks, m.tri_count, [self.coarse_pid, m.particle_mass, m.particle_flags,
                                                     self.particle_q_prev, state_in.particle_q, m.tri_indices,
                                                     m.tri_poses, m.tri_materials, m.tri_areas, float(dt),
                                                     self.coarse_TH, self.coarse_use]),
                 self._rec(slot_sum, self.coarse_A.shape[0], [K, self.coarse_nb_off, self.coarse_se_off, self.coarse_se,
                                                              self.coarse_slot_patch, m.tri_indices, m.particle_mass,
                                                              m.particle_flags, self.coarse_TH, self.coarse_use,
                                                              self.coarse_Ad, self.coarse_A]),
                 (self._rec(coarse_patch_pgs_tiled, (1, self.coarse_bd),
                            [K, self.coarse_nb_off, self.coarse_nb_idx, self.coarse_color, self.coarse_ncolors,
                             self.coarse_sweeps, self.coarse_A, self.coarse_r, self.coarse_cg, self.coarse_ns,
                             self.coarse_cp, self.coarse_cpart, self.coarse_pdelta], block_dim=self.coarse_bd)
                  if self.coarse_pgs == "tiled" else
                  self._rec(coarse_patch_pgs, 1, [K, self.coarse_nb_off, self.coarse_nb_idx, self.coarse_sweeps,
                                                  self.coarse_A, self.coarse_r, self.coarse_cg, self.coarse_ns,
                                                  self.coarse_cp, self.coarse_cpart, self.coarse_pdelta]))]
            Pa = self._rec(coarse_patch_apply_strided, n,
                           [0, 1, n, m.particle_mass, m.particle_flags, self.coarse_pid, self.coarse_pdelta,
                            self.particle_displacements])
            cyc = [E, *S, Pa, B1, B2]
        prog = []
        for it in range(it0, it0 + n_it):
            if self.coarse_every > 0 and it > 0 and it % self.coarse_every == 0:
                prog += cyc
            prog += one
        self._programs[key] = prog
        return prog

    def _run_program(self, state_in, state_out, contacts, dt, it0=0, n_it=None):
        """The iterations: replayed as one CUDA graph (captured once per program) unless use_graph is off."""
        prog = self._program(state_in, state_out, contacts, dt, it0, n_it)
        if not self.use_graph:
            for c in prog:
                c.launch()
            return
        gk = ("graph", id(prog))
        g = self._programs.get(gk)
        if g is None:
            with wp.ScopedCapture(device=self.dev, force_module_load=False) as cap:
                for c in prog:
                    c.launch()
            g = cap.graph
            self._programs[gk] = g
            # the capture recorded the launches without running them: run this step's iterations now
        wp.capture_launch(g)

    def _penetration_free_truncation(self, particle_q_out=None):
        key = ("init", id(particle_q_out), id(self.inc_entries))
        prog = self._programs.get(key)
        if prog is None:
            prog = list(self._truncation_calls(particle_q_out))
            self._programs[key] = prog
        for c in prog:
            c.launch()

    def step(self, state_in, state_out, control, contacts, dt):
        fuse_fp_off()
        self._apply_module_options()
        update_rigid = self._update_rigid_history
        self._update_rigid_history = True
        _F = SolverBase.CollisionFrequencyType
        self._sc_mode_this_step, self._sc_freq_this_step = self._resolve_self_contact_schedule()
        if self._sc_mode_this_step not in (_F.PRE_INIT, _F.PRE_POST_INIT):
            raise ValueError("GpuDetSolverVBD supports self-contact detection PRE_INIT or PRE_POST_INIT (the default)")
        self._rigid_mode_this_step = _F.NONE
        self._rigid_freq_this_step = 1
        if self.collision_pipeline is not None:
            raise ValueError("GpuDetSolverVBD expects the caller to run collide (no collision_pipeline)")
        if control is None:
            control = self.model.control(clone_variables=False)
        if PROFILE:
            wp.synchronize_device(self.dev)
            t0 = time.perf_counter()
        self._initialize_rigid_bodies(state_in, control, contacts, dt, update_rigid)
        self._skip_incidence = self._sc_mode_this_step == _F.PRE_POST_INIT
        try:
            self._initialize_particles(state_in, state_out, dt)
        finally:
            self._skip_incidence = False
        if PROFILE:
            wp.synchronize_device(self.dev)
            t1 = time.perf_counter()
        if self._sc_mode_this_step == _F.PRE_POST_INIT:
            self._collision_detection_penetration_free(state_in)
        if PROFILE:
            wp.synchronize_device(self.dev)
            t2 = time.perf_counter()
        self._run_program(state_in, state_out, contacts, dt)
        if self.adaptive is not None:
            a, done = self.adaptive, self.iterations
            while done < a["max_iterations"]:
                self._umax.zero_()
                wp.launch(max_update, dim=self.model.particle_count, inputs=[self._update_norm, self._umax],
                          device=self.dev)
                if float(self._umax.numpy()[0]) <= a["tol"]:
                    break
                n_it = min(a["block"], a["max_iterations"] - done)
                self._run_program(state_in, state_out, contacts, dt, done, n_it)
                done += n_it
            self.iterations_used.append(done)
        if PROFILE:
            wp.synchronize_device(self.dev)
            t3 = time.perf_counter()
            TIMES["init"] += t1 - t0
            TIMES["detect2+incidence"] += t2 - t1
            TIMES["iterations"] += t3 - t2
        wp.copy(state_out.particle_q, state_in.particle_q)
        if self.plastic is not None:
            key = ("plastic", id(state_out.particle_q))
            prog = self._programs.get(key)
            if prog is None:
                prog = [self._rec(plastic_bending, self.model.edge_count, self.plastic.args(state_out.particle_q, 0, 1))]
                self._programs[key] = prog
            for c in prog:
                c.launch()
            self.plastic.book()
        self._snapshot_rigid_contact_history(contacts)
        self._finalize_rigid_bodies(state_in, state_out, dt)
        self._finalize_particles(state_out, dt)
