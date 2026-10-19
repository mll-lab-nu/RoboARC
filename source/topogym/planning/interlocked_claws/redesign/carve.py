"""Inverse generation of keyhole gates: a gate plate is carved out of the swept volume of the key along its designed
disengagement path, so that path is free by construction (with a clearance) while the plate blocks every motion that
crosses its slab anywhere else.

Method (all in the lock piece A's frame):
  1. the key B (compound boxes) is sampled as a point cloud: points on a grid of spacing <= h inside every box;
  2. the designed path (dense relative poses of B) moves the cloud; for a gate (a slab: centre c, normal n, in-plane
     axes u, v, thickness t), every point inside the slab grown by the clearance (|n.(x - c)| <= t/2 + clearance) is
     projected on the plate plane and marks its plate cell;
  3. the marked cells are grown by the clearance plus half the sampling spacing (a conservative dilation) and removed;
     the remaining cells of the plate rectangle are merged into rectangles (rows, then equal row spans) -> boxes.
A thin border ring of the plate is always kept (the frame), unless `open_sides` names a side to leave open.
The exact boxes are then certified against the exact key boxes along the path (geom.check_path).
"""
from __future__ import annotations

import numpy as np

from topogym.tasks.interlocked_claws.redesign.shapes import box


def key_points(boxes, spacing=.0015):
    """Points filling every box of the key (its surface and interior), spacing <= `spacing`."""
    pts = []
    for b in boxes:
        T = np.asarray(b["pose"], float)
        s = np.asarray(b["size"], float)
        n = np.maximum(2, np.ceil(s / spacing).astype(int) + 1)
        g = np.stack(np.meshgrid(*[np.linspace(-si / 2, si / 2, k) for si, k in zip(s, n)], indexing="ij"), -1).reshape(-1, 3)
        pts.append(g @ T[:3, :3].T + T[:3, 3])
    return np.concatenate(pts)


def plate_frame(center, normal, up):
    n = np.asarray(normal, float); n /= np.linalg.norm(n)
    u = np.asarray(up, float); u = u - (u @ n) * n; u /= np.linalg.norm(u)
    w = np.cross(u, n)
    return np.asarray(center, float), n, w, u          # plate axes: w (width), u (height)


def carve_gate(points, path, center, normal, up, width, height, thickness=.010, clearance=.004, cell=.002,
               border=.012, open_sides=(), spacing=.0015, min_rect=.004):
    """-> (boxes of the carved plate in A's frame, info). path: list of 4x4 relative poses of B in A's frame."""
    c, n, w, u = plate_frame(center, normal, up)
    nw, nu = int(np.ceil(width / cell)), int(np.ceil(height / cell))
    occupied = np.zeros((nw, nu), bool)
    reach = thickness / 2 + clearance
    for T in path:
        T = np.asarray(T, float)
        X = points @ T[:3, :3].T + T[:3, 3] - c
        dn = X @ n
        sel = np.abs(dn) <= reach + spacing
        if not sel.any():
            continue
        X = X[sel]
        iw = np.floor((X @ w + width / 2) / cell).astype(int)
        iu = np.floor((X @ u + height / 2) / cell).astype(int)
        ok = (iw >= 0) & (iw < nw) & (iu >= 0) & (iu < nu)
        occupied[iw[ok], iu[ok]] = True
    # dilation by the clearance plus the sampling and cell sizes (conservative)
    grow = int(np.ceil((clearance + spacing + cell) / cell))
    if occupied.any():
        from scipy.ndimage import binary_dilation
        # a square element: the carved outline stays rectilinear (fewer, fatter plate boxes), and it is at least as
        # conservative as the disc of the same radius
        carved = binary_dilation(occupied, structure=np.ones((2 * grow + 1, 2 * grow + 1), bool))
    else:
        carved = occupied
    solid = ~carved
    # the border ring stays (a frame) except open sides
    b = int(np.ceil(border / cell))
    ring = np.zeros_like(solid)
    if "w-" not in open_sides:
        ring[:b, :] = True
    if "w+" not in open_sides:
        ring[-b:, :] = True
    if "u-" not in open_sides:
        ring[:, :b] = True
    if "u+" not in open_sides:
        ring[:, -b:] = True
    lost_ring = bool((ring & carved).any())
    solid = solid | ring if not lost_ring else solid
    # keep only the solid cells connected to the outer border (islands inside the aperture would be loose parts)
    from scipy.ndimage import label
    lab, nlab = label(solid)
    edge = np.zeros_like(solid)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    keep = set(np.unique(lab[edge & solid]).tolist()) - {0}
    islands = int(nlab - len(keep))
    solid = np.isin(lab, sorted(keep))
    rects = merge_rectangles(solid)
    R = np.stack([n, w, u], 1)
    boxes = []
    for (i0, i1, j0, j1) in rects:
        sw, su = (i1 - i0) * cell, (j1 - j0) * cell
        cw = -width / 2 + (i0 + i1) / 2 * cell
        cu = -height / 2 + (j0 + j1) / 2 * cell
        # abutting rectangles overlap by 0.1 mm so that the union is one solid (exact float abutment can leave a gap)
        boxes.append(box(c + cw * w + cu * u, (thickness, sw + 1e-4, su + 1e-4), R))
    info = dict(islands_removed=islands, center=c.tolist(), normal=n.tolist(), up=u.tolist(), width=width, height=height, thickness=thickness,
                clearance=clearance, cell=cell, carved_cells=int(carved.sum()), cells=int(nw * nu),
                aperture_area_m2=float(carved.sum() * cell * cell), boxes=len(boxes), border_cut=lost_ring)
    return boxes, info, carved


def merge_rectangles(mask):
    """Cover a boolean grid with fat axis-aligned rectangles that may overlap (a cover, not a partition): from the
    first uncovered cell, the larger of (its maximal row run grown over the rows) and (its maximal column run grown
    over the columns), repeated until every cell is covered. -> [(i0, i1, j0, j1)] half-open index ranges."""
    mask = np.asarray(mask, bool)
    nw, nu = mask.shape
    left = mask.copy()
    out = []
    while left.any():
        i, j = map(int, np.argwhere(left)[0])
        # A: run along axis 1 in row i, then grow along axis 0
        j0 = j
        while j0 > 0 and mask[i, j0 - 1]:
            j0 -= 1
        j1 = j + 1
        while j1 < nu and mask[i, j1]:
            j1 += 1
        a0, a1 = i, i + 1
        while a0 > 0 and mask[a0 - 1, j0:j1].all():
            a0 -= 1
        while a1 < nw and mask[a1, j0:j1].all():
            a1 += 1
        A = (a0, a1, j0, j1)
        # B: run along axis 0 in column j, then grow along axis 1
        b0 = i
        while b0 > 0 and mask[b0 - 1, j]:
            b0 -= 1
        b1 = i + 1
        while b1 < nw and mask[b1, j]:
            b1 += 1
        c0, c1 = j, j + 1
        while c0 > 0 and mask[b0:b1, c0 - 1].all():
            c0 -= 1
        while c1 < nu and mask[b0:b1, c1].all():
            c1 += 1
        Bq = (b0, b1, c0, c1)
        area = lambda r: (r[1] - r[0]) * (r[3] - r[2])
        r = A if area(A) >= area(Bq) else Bq
        left[r[0]:r[1], r[2]:r[3]] = False
        out.append(r)
    return out
