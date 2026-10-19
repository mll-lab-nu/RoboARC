"""Object pose change start -> goal (COMMON.md metric spec): rope chains and cloths. numpy only.

rope_pose_change(start, goal, rings=None)
    start / goal: {"pos": (n, 3) link origins, "quat": (n, 4) link quaternions wxyz (optional), "nodes": (n+1, 3)
    chain nodes (optional; default: the link origins)} matched by material index (link k = material order).
    -> centroid translation (m, and the vector), end-to-end chord rotation (deg; and its yaw about z), mean / max
       per-link orientation change (deg), mean / max link displacement (m), projected (top-view) crossing count at
       start / goal, ring passages at start / goal (rings: [{"name", "center", "normal", "radius"}]).
cloth_pose_change(P0, P1, faces=None, V0=None, V1=None)
    P0 / P1: (N, 3) particles (or flex vertices) matched by index; faces + V0 / V1: the flex mesh for areas.
    -> Procrustes rigid fit start -> goal (translation m, rotation deg, yaw deg, residual RMS m), mean / max particle
       displacement, layered (overlapped) area fraction and fold layers (top-view rasterisation of the mesh), flipped
       (turned-over) area fraction, covered area, at start and goal.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------------------------------------ rope
def _quat_angle_deg(q0, q1):
    q0 = np.asarray(q0, float); q1 = np.asarray(q1, float)
    q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    d = np.clip(np.abs(np.sum(q0 * q1, axis=-1)), 0, 1)
    return np.degrees(2 * np.arccos(d))


def _xaxis(q):
    """The x axis (the link axis of a rope chain link) of wxyz quaternions (n, 4) -> (n, 3)."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=1)


def _angle_deg_rows(U, V):
    U = np.asarray(U, float); V = np.asarray(V, float)
    c = np.sum(U * V, axis=1) / np.maximum(np.linalg.norm(U, axis=1) * np.linalg.norm(V, axis=1), 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _angle_deg(u, v):
    u = np.asarray(u, float); v = np.asarray(v, float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(u @ v / (nu * nv), -1, 1))))


def projected_crossings(polylines, skip_adjacent: int = 2, eps: float = 1e-12) -> dict:
    """Top-view (xy) crossings between non-adjacent segments of one or more polylines [(n_i, 3)].
    -> {"count", "pairs": [(line_a, seg_a, line_b, seg_b, over: 'a'|'b')] (first 64)}"""
    segs, owner, index = [], [], []
    for li, P in enumerate(polylines):
        P = np.asarray(P, float)
        for i in range(len(P) - 1):
            segs.append((P[i], P[i + 1])); owner.append(li); index.append(i)
    if len(segs) < 2:
        return dict(count=0, pairs=[])
    A = np.array([s[0] for s in segs]); B = np.array([s[1] for s in segs])
    owner = np.array(owner); index = np.array(index)
    n = len(segs)
    a2, b2 = A[:, :2], B[:, :2]
    lo = np.minimum(a2, b2); hi = np.maximum(a2, b2)
    count, pairs = 0, []
    for i in range(n - 1):
        j = np.arange(i + 1, n)
        same = owner[j] == owner[i]
        j = j[~(same & (np.abs(index[j] - index[i]) < skip_adjacent))]     # neighbours share a node: no crossing
        if not len(j):
            continue
        ok = (lo[j, 0] <= hi[i, 0]) & (hi[j, 0] >= lo[i, 0]) & (lo[j, 1] <= hi[i, 1]) & (hi[j, 1] >= lo[i, 1])
        j = j[ok]
        if not len(j):
            continue
        p, r = a2[i], b2[i] - a2[i]
        q, s = a2[j], b2[j] - a2[j]
        rxs = r[0] * s[:, 1] - r[1] * s[:, 0]
        qp = q - p
        t = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / np.where(np.abs(rxs) < eps, np.inf, rxs)
        u = (qp[:, 0] * r[1] - qp[:, 1] * r[0]) / np.where(np.abs(rxs) < eps, np.inf, rxs)
        hit = (np.abs(rxs) >= eps) & (t > 0) & (t < 1) & (u > 0) & (u < 1)
        for jj, tt, uu in zip(j[hit], t[hit], u[hit]):
            count += 1
            if len(pairs) < 64:
                za = A[i, 2] + tt * (B[i, 2] - A[i, 2]); zb = A[jj, 2] + uu * (B[jj, 2] - A[jj, 2])
                pairs.append((int(owner[i]), int(index[i]), int(owner[jj]), int(index[jj]), "a" if za > zb else "b"))
    return dict(count=int(count), pairs=pairs)


def ring_passages(nodes, rings) -> dict:
    """Per ring: the rope segments that pierce the ring's disk (plane crossing within the radius), with direction
    (+1 along the ring normal) and the material index (segment). rings: [{"name", "center", "normal", "radius"}]."""
    P = np.asarray(nodes, float)
    out, total = {}, 0
    for k, ring in enumerate(rings or []):
        c = np.asarray(ring["center"], float); nrm = np.asarray(ring["normal"], float)
        nrm = nrm / np.linalg.norm(nrm)
        r = float(ring["radius"])
        d = (P - c) @ nrm
        hits = []
        for i in range(len(P) - 1):
            if d[i] == 0 or d[i] * d[i + 1] < 0 or (d[i + 1] == 0 and d[i] != 0):
                if d[i] == d[i + 1]:
                    continue
                t = d[i] / (d[i] - d[i + 1])
                x = P[i] + t * (P[i + 1] - P[i])
                rad = np.linalg.norm((x - c) - ((x - c) @ nrm) * nrm)
                if rad <= r:
                    hits.append(dict(segment=i, direction=int(np.sign(d[i + 1] - d[i])), radius_m=round(float(rad), 4)))
        out[ring.get("name", f"ring{k}")] = dict(passages=len(hits), hits=hits[:16])
        total += len(hits)
    return dict(total=total, rings=out)


def rope_pose_change(start: dict, goal: dict, rings=None, other_start=None, other_goal=None) -> dict:
    """Pose change of one rope (matched by material index). other_*: further ropes' node polylines for crossings."""
    p0, p1 = np.asarray(start["pos"], float), np.asarray(goal["pos"], float)
    n0 = np.asarray(start.get("nodes", p0), float); n1 = np.asarray(goal.get("nodes", p1), float)
    c0, c1 = n0.mean(axis=0), n1.mean(axis=0)
    ch0, ch1 = n0[-1] - n0[0], n1[-1] - n1[0]
    yaw = float(np.degrees(np.arctan2(ch1[1], ch1[0]) - np.arctan2(ch0[1], ch0[0])))
    yaw = (yaw + 180) % 360 - 180
    disp = np.linalg.norm(p1 - p0, axis=1)
    out = dict(
        links=int(len(p0)), length_m=round(float(np.linalg.norm(np.diff(n0, axis=0), axis=1).sum()), 4),
        centroid_translation_m=round(float(np.linalg.norm(c1 - c0)), 4),
        centroid_translation_vec_m=[round(float(x), 4) for x in (c1 - c0)],
        chord_rotation_deg=round(_angle_deg(ch0, ch1), 2), chord_yaw_deg=round(yaw, 2),
        chord_length_m=[round(float(np.linalg.norm(ch0)), 4), round(float(np.linalg.norm(ch1)), 4)],
        link_displacement_mean_m=round(float(disp.mean()), 4), link_displacement_max_m=round(float(disp.max()), 4),
        link_displacement_max_index=int(np.argmax(disp)),
        tip_displacement_m=[round(float(disp[0]), 4), round(float(disp[-1]), 4)],
        height_range_m=[[round(float(n0[:, 2].min()), 4), round(float(n0[:, 2].max()), 4)],
                        [round(float(n1[:, 2].min()), 4), round(float(n1[:, 2].max()), 4)]])
    if start.get("quat") is not None and goal.get("quat") is not None:
        ang = _quat_angle_deg(start["quat"], goal["quat"])
        out.update(link_rotation_mean_deg=round(float(ang.mean()), 2), link_rotation_max_deg=round(float(ang.max()), 2))
        # the links' axis directions alone (bending / reorientation, without the twist about the link axis)
        ax = _angle_deg_rows(_xaxis(start["quat"]), _xaxis(goal["quat"]))
        out.update(link_axis_rotation_mean_deg=round(float(ax.mean()), 2),
                   link_axis_rotation_max_deg=round(float(ax.max()), 2))
    else:
        t0 = np.diff(n0, axis=0); t1 = np.diff(n1, axis=0)
        ax = _angle_deg_rows(t0, t1)
        out.update(link_axis_rotation_mean_deg=round(float(ax.mean()), 2),
                   link_axis_rotation_max_deg=round(float(ax.max()), 2))
    xs0 = projected_crossings([n0] + list(other_start or []))
    xs1 = projected_crossings([n1] + list(other_goal or []))
    out.update(crossings_start=xs0["count"], crossings_goal=xs1["count"])
    if rings:
        rp0, rp1 = ring_passages(n0, rings), ring_passages(n1, rings)
        out.update(ring_passages_start=rp0["total"], ring_passages_goal=rp1["total"],
                   ring_passages_detail=dict(start=rp0["rings"], goal=rp1["rings"]))
    return out


# ------------------------------------------------------------------------------------------------ cloth
def kabsch(P0, P1):
    """Rigid fit P1 ~ R P0 + t (least squares). -> (R, t, rms residual)."""
    P0 = np.asarray(P0, float); P1 = np.asarray(P1, float)
    c0, c1 = P0.mean(axis=0), P1.mean(axis=0)
    H = (P0 - c0).T @ (P1 - c1)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = c1 - R @ c0
    res = P1 - (P0 @ R.T + t)
    return R, t, float(np.sqrt((res ** 2).sum(axis=1).mean()))


def _rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def layers(V, faces, cell: float = 0.004, bounds=None) -> dict:
    """Top-view rasterisation of a triangle mesh: per cell the number of layers (triangles covering its centre).
    -> covered area (m^2), layered (>= 2 layers) area fraction of the covered area, max layers, flipped (normal down)
    area fraction (faces oriented up in the flat rest)."""
    V = np.asarray(V, float); F = np.asarray(faces, int)
    tri = V[F]
    lo = tri[:, :, :2].min(axis=(0, 1)) - cell if bounds is None else np.asarray(bounds[0], float)
    hi = tri[:, :, :2].max(axis=(0, 1)) + cell if bounds is None else np.asarray(bounds[1], float)
    nx, ny = int(np.ceil((hi[0] - lo[0]) / cell)), int(np.ceil((hi[1] - lo[1]) / cell))
    depth = np.zeros((nx, ny), np.int16)
    e1 = tri[:, 1] - tri[:, 0]; e2 = tri[:, 2] - tri[:, 0]
    nz = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    area3 = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    flipped_area = float(area3[nz < 0].sum())
    for k in range(len(F)):
        a, b, c = tri[k, :, :2]
        x0, y0 = np.floor((np.minimum(np.minimum(a, b), c) - lo) / cell).astype(int)
        x1, y1 = np.ceil((np.maximum(np.maximum(a, b), c) - lo) / cell).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, nx), min(y1, ny)
        if x1 <= x0 or y1 <= y0:
            continue
        # cell centres offset by an irrational fraction of a cell: a regular mesh's edges never pass through them
        # (an inclusive test would count a centre on a shared edge twice)
        gx = lo[0] + (np.arange(x0, x1) + 0.5 + 0.1180340) * cell
        gy = lo[1] + (np.arange(y0, y1) + 0.5 + 0.0729490) * cell
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-14:
            continue
        l1 = ((b[1] - c[1]) * (X - c[0]) + (c[0] - b[0]) * (Y - c[1])) / d
        l2 = ((c[1] - a[1]) * (X - c[0]) + (a[0] - c[0]) * (Y - c[1])) / d
        inside = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
        depth[x0:x1, y0:y1] += inside
    covered = depth >= 1
    cov_area = float(covered.sum() * cell * cell)
    layered = float((depth >= 2).sum() / max(1, covered.sum()))
    hist = np.bincount(depth.reshape(-1), minlength=4)[1:].tolist()
    return dict(covered_area_m2=round(cov_area, 5), layered_area_fraction=round(layered, 4),
                max_layers=int(depth.max()), layer_histogram_cells=hist[:8],
                flipped_area_fraction=round(flipped_area / max(1e-12, float(area3.sum())), 4),
                mesh_area_m2=round(float(area3.sum()), 5), cell_m=cell)


def cloth_pose_change(P0, P1, faces=None, V0=None, V1=None, rest=None, cell: float = 0.004) -> dict:
    P0 = np.asarray(P0, float); P1 = np.asarray(P1, float)
    R, t, rms = kabsch(P0, P1)
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    disp = np.linalg.norm(P1 - P0, axis=1)
    out = dict(points=int(len(P0)),
               procrustes_translation_m=round(float(np.linalg.norm(P1.mean(axis=0) - P0.mean(axis=0))), 4),
               procrustes_translation_vec_m=[round(float(x), 4) for x in (P1.mean(axis=0) - P0.mean(axis=0))],
               procrustes_rotation_deg=round(_rot_deg(R), 2), procrustes_yaw_deg=round(yaw, 2),
               procrustes_residual_rms_m=round(rms, 4),
               displacement_mean_m=round(float(disp.mean()), 4), displacement_max_m=round(float(disp.max()), 4),
               height_max_m=[round(float(P0[:, 2].max() - P0[:, 2].min()), 4), round(float(P1[:, 2].max() - P1[:, 2].min()), 4)])
    if faces is not None and V0 is not None and V1 is not None:
        L0, L1 = layers(V0, faces, cell), layers(V1, faces, cell)
        out.update(layers_start=L0, layers_goal=L1,
                   layered_area_fraction_start=L0["layered_area_fraction"],
                   layered_area_fraction_goal=L1["layered_area_fraction"],
                   fold_layers_start=L0["max_layers"], fold_layers_goal=L1["max_layers"],
                   flipped_area_fraction_start=L0["flipped_area_fraction"],
                   flipped_area_fraction_goal=L1["flipped_area_fraction"])
        if rest is not None:
            Lr = layers(rest, faces, cell)
            out["flat_area_m2"] = Lr["covered_area_m2"]
            out["coverage_proxy_start"] = round(L0["covered_area_m2"] / max(1e-12, Lr["covered_area_m2"]), 4)
            out["coverage_proxy_goal"] = round(L1["covered_area_m2"] / max(1e-12, Lr["covered_area_m2"]), 4)
    return out
