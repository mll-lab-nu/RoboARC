"""An exact convex partition of a link's collision prisms (one extrusion of a polygon with holes) without slivers.

Some families declare a plate as convex slab prisms cut at the height of every outline vertex (cast_keyhole: scanline
trapezoids, the thinnest 2 um). MuJoCo collides each geom as its convex hull, so the solid stays exact, but a slab
thinner than a penetration lets the narrowphase report the slab's thin direction as the contact normal (sliver hulls,
docs/rcb/MUJOCO_CORE.md). profile["prism_partition"] re-partitions the same solid into fat convex prisms:
the slabs' 2D cells -> their union polygon -> constrained Delaunay triangles -> greedy exact merging (two pieces merge
only when their union is convex: the hull's area equals the sum of their areas), thinnest piece first, joined with the
neighbour that gives the thickest union, at most MAX_VERTICES polygon vertices per piece. Every output vertex is an
input vertex (up to the projection's rounding) and the area is conserved, so the collision solid does not change.
"""
from __future__ import annotations

import numpy as np

MAX_VERTICES = 32           # polygon vertices per piece: a prism of at most 64 vertices
GRID_M = 1e-7               # the cells' union is snap-rounded to this grid: the URDF meshes carry 10 nm rounding


def _frame(parts):
    """The extrusion: normal n with the shared bottom / top planes lo / hi, in-plane basis u, w."""
    from scipy.spatial import ConvexHull
    for n in ConvexHull(parts[0]).equations[:, :3]:
        spans = np.array([[(V @ n).min(), (V @ n).max()] for V in parts])
        if np.ptp(spans[:, 0]) < 1e-12 and np.ptp(spans[:, 1]) < 1e-12:
            u = np.cross(n, [1., 0., 0.] if abs(n[0]) < .9 else [0., 1., 0.])
            u /= np.linalg.norm(u)
            return n, u, np.cross(n, u), float(spans[0, 0]), float(spans[0, 1])
    raise ValueError("prism_partition: the collision meshes are not one extrusion (shared bottom and top planes)")


def _width(P):
    """(minimum width over the hull's edge normals, area) of a convex 2D point set."""
    from scipy.spatial import ConvexHull
    h = ConvexHull(P)
    p = P @ h.equations[:, :2].T
    return float(np.min(p.max(0) - p.min(0))), float(h.volume)


def _merge(polygons):
    from scipy.spatial import ConvexHull
    items = []
    for P in polygons:
        w, a = _width(P)
        items.append(dict(P=P, w=w, a=a, lo=P.min(0), hi=P.max(0)))
    while True:
        for i in sorted(range(len(items)), key=lambda k: items[k]["w"]):
            A, best = items[i], None
            for j, B in enumerate(items):
                if j == i or np.any(A["lo"] > B["hi"] + 1e-12) or np.any(B["lo"] > A["hi"] + 1e-12):
                    continue
                V = np.vstack([A["P"], B["P"]])
                h = ConvexHull(V)
                if h.volume <= (A["a"] + B["a"]) * (1 + 1e-9) and len(h.vertices) <= MAX_VERTICES:
                    W = V[h.vertices]
                    w, a = _width(W)
                    if best is None or w > best[0]:
                        best = (w, j, W, a)
            if best is not None:
                w, j, W, a = best
                items[i] = dict(P=W, w=w, a=a, lo=W.min(0), hi=W.max(0))
                del items[j]
                break
        else:
            return items


def _absorb_needles(items, width_m=1e-6, layer_m=GRID_M):
    """Pieces under width_m are needles of the snap-rounded union: vertices sub-micrometre apart at T-junctions of the
    declared slabs, which the triangulation joins to a far vertex. Each joins the neighbour whose union's hull adds the
    least area, if that area is at most a layer_m-thin strip along the needle (the snap grid's own scale). Nothing else
    changes."""
    from scipy.spatial import ConvexHull
    while True:
        thin = [k for k, it in enumerate(items) if it["w"] < width_m]
        if not thin:
            return items
        i, best = thin[0], None
        A = items[i]
        excess_m2 = layer_m * float(np.max(np.linalg.norm(A["P"][:, None] - A["P"][None], axis=2)))
        for j, B in enumerate(items):
            if j == i or np.any(A["lo"] > B["hi"] + 1e-9) or np.any(B["lo"] > A["hi"] + 1e-9):
                continue
            V = np.vstack([A["P"], B["P"]])
            h = ConvexHull(V)
            excess = h.volume - A["a"] - B["a"]
            if excess <= excess_m2 and len(h.vertices) <= MAX_VERTICES and (best is None or excess < best[0]):
                best = (excess, j, V[h.vertices])
        if best is None:
            raise ValueError("prism_partition: a needle piece has no neighbour to join")
        _, j, W = best
        w, a = _width(W)
        items[i] = dict(P=W, w=w, a=a, lo=W.min(0), hi=W.max(0))
        del items[j]


def _partition(parts):
    """The exact fat re-partition of prisms of one extrusion (see the module docstring)."""
    import shapely
    import trimesh
    from scipy.spatial import ConvexHull
    from shapely.geometry import Polygon
    n, u, w, lo, hi = _frame(parts)
    plane = np.c_[u, w]
    cells = []
    for V in parts:
        P = V @ plane
        cells.append(Polygon(P[ConvexHull(P).vertices]))
    # Snap-rounded union: neighbouring slabs share edges and T-junctions only up to the mesh files' 10 nm rounding.
    union = shapely.unary_union(cells, grid_size=GRID_M).simplify(GRID_M)
    triangles = [np.asarray(t.exterior.coords)[:3] for t in shapely.constrained_delaunay_triangles(union).geoms]
    items = _absorb_needles(_merge(triangles))
    area, declared = sum(it["a"] for it in items), sum(c.area for c in cells)
    if abs(area - declared) > 1e-5 * declared:
        raise ValueError(f"prism_partition: area {area} differs from the declared prisms' {declared}")
    pieces = []
    for it in items:
        P = it["P"]
        V = np.vstack([P @ plane.T + lo * n, P @ plane.T + hi * n])
        pieces.append((V, np.asarray(trimesh.convex.convex_hull(V).faces)))
    before = min(_width(V @ np.c_[u, w, n][:, :2])[0] for V in parts)
    audit = dict(prisms=len(parts), pieces=len(pieces), thinnest_prism_mm=round(1000 * before, 4),
                 thinnest_piece_mm=round(1000 * min(it["w"] for it in items), 4),
                 max_piece_vertices=2 * max(len(it["P"]) for it in items),
                 area_rel_error=float(abs(area - declared) / declared), thickness_mm=round(1000 * (hi - lo), 6))
    return pieces, audit


def _prism_frame(V):
    """(n, lo, hi) when V is a right prism along one of its hull's face normals (every vertex on one of two parallel
    planes, the two outlines equal up to the translation along n), else None; n's largest component is positive."""
    from scipy.spatial import ConvexHull
    for n in ConvexHull(V).equations[:, :3]:
        s = V @ n
        lo, hi = s.min(), s.max()
        bottom, top = np.abs(s - lo) < 1e-9, np.abs(s - hi) < 1e-9
        if hi - lo < 1e-9 or not (bottom | top).all() or bottom.sum() != top.sum():
            continue
        a, b = V[bottom] - np.outer(s[bottom], n), V[top] - np.outer(s[top], n)
        a, b = a[np.lexsort(np.round(a / 1e-8).T)], b[np.lexsort(np.round(b / 1e-8).T)]
        if np.abs(a - b).max() < 1e-8:
            n = n if n[np.argmax(np.abs(n))] > 0 else -n
            return tuple(np.round(n, 9)), round(float((V @ n).min()), 9), round(float((V @ n).max()), 9)
    return None


def convex_partition(parts, *, fatten_m=None):
    """parts: the link's collision prisms, (k, 3) vertex arrays in the link frame. Returns ([(vertices, faces)], audit).
    One extrusion: its exact fat re-partition. Several (a link of slab groups and other convex pieces, hedgehog's cages):
    every group of at least two prisms sharing a normal and both planes is re-partitioned on its own, any other piece
    is kept as declared. fatten_m: then thin pieces are re-cut with their neighbours (fatten)."""
    parts = [np.asarray(V, dtype=np.float64) for V in parts]
    try:
        _frame(parts)
        groups = [list(range(len(parts)))]
    except ValueError:
        groups = None
    if groups is not None:
        pieces, audit = _partition(parts)
    else:
        import trimesh
        by = {}
        for i, V in enumerate(parts):
            by.setdefault(_prism_frame(V), []).append(i)
        pieces, audit, kept = [], dict(groups=[]), []
        for frame, idx in by.items():
            if frame is None or len(idx) < 2:
                kept += idx
                continue
            got, a = _partition([parts[i] for i in idx])
            pieces += got
            audit["groups"].append(dict(normal=list(frame[0]), planes_m=[frame[1], frame[2]], **a))
        for i in sorted(kept):
            pieces.append((parts[i], np.asarray(trimesh.convex.convex_hull(parts[i]).faces)))
        audit.update(prisms=len(parts), pieces=len(pieces), kept_as_declared=len(kept),
                     thinnest_prism_mm=round(1000 * min(_width3(V) for V in parts), 4),
                     thinnest_piece_mm=round(1000 * min(_width3(V) for V, _ in pieces), 4))
    if fatten_m:
        pieces, audit["fatten"] = fatten(pieces, fatten_m)
        audit["thinnest_piece_mm"] = round(1000 * min(_width3(V) for V, _ in pieces), 4)
    return pieces, audit


def _width3(V):
    """Minimum width of a convex 3D point set over its hull's face normals."""
    from scipy.spatial import ConvexHull
    h = ConvexHull(V)
    p = V @ h.equations[:, :3].T
    return float(np.min(p.max(0) - p.min(0)))


def _clip(V, n, d):
    """conv(V) with n.x <= d, as its vertices, or None when (nearly) empty."""
    from scipy.spatial import ConvexHull
    s = V @ n - d
    if (s <= 1e-12).all():
        return V
    if (s >= -1e-12).all():
        return None
    h = ConvexHull(V)
    edges = {tuple(sorted((a, b))) for t in h.simplices for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    out = [V[s <= 1e-12]] + [(V[i] + s[i] / (s[i] - s[j]) * (V[j] - V[i]))[None] for i, j in edges
                             if min(s[i], s[j]) < -1e-12 and max(s[i], s[j]) > 1e-12]
    Q = np.unique(np.vstack(out), axis=0)
    try:
        h = ConvexHull(Q)
    except Exception:
        return None
    return Q[h.vertices] if h.volume > 1e-18 else None


def fatten(pieces, target_m, coarse_deg=np.linspace(-80., 80., 17), fine_deg=np.arange(-6., 6.5, 1.)):
    """Re-cut thin convex pieces with a neighbour, exactly: a piece P thinner than target_m and a neighbour Q sharing a
    face are re-split by a plane through an edge of that face, turned about the edge; a split counts only when both new
    pieces are convex (their volumes add up to P's and Q's: the solid is unchanged) and the thinner of the two is
    thicker than before. The plane through a reflex edge of the union keeps both sides convex over a wide range of turns,
    so a sliver between two close reflex edges (a wall facet sampled 0.16 mm from the next) becomes two fat pieces.
    Passes over the thin pieces (thinnest first) until one changes nothing. Tolerances follow the declared meshes' own
    rounding (about 10 nm): a vertex within 50 nm of a plane lies on it, and a split is exact when its two hulls exceed the
    two pieces' volume by at most a 20 nm layer over their surfaces (a real concavity, micrometres deep, does not pass).
    Returns (pieces, audit)."""
    import trimesh
    from scipy.spatial import ConvexHull
    V = [np.asarray(v, dtype=np.float64) for v, _ in pieces]
    vol = [ConvexHull(v).volume for v in V]
    area = [ConvexHull(v).area for v in V]
    wid = [_width3(v) for v in V]
    before, recuts, on_plane, layer = min(wid), 0, 5e-8, 2e-8

    def split(P, Q, m, dd, total, slack):
        sides = []
        for sgn in (1., -1.):
            got = [g for g in (_clip(P, sgn * m, sgn * dd), _clip(Q, sgn * m, sgn * dd)) if g is not None]
            if not got:
                return None
            X = np.unique(np.vstack(got), axis=0)
            if len(X) < 4:
                return None
            try:
                h = ConvexHull(X)
            except Exception:
                return None
            sides.append((X[h.vertices], h.volume, h.area))
        if abs(sides[0][1] + sides[1][1] - total) > slack:
            return None
        return min(_width3(sides[0][0]), _width3(sides[1][0])), sides

    def candidates(i, j):
        """(axis point, axis direction, face normal) for every edge of the face P = V[i] shares with Q = V[j]."""
        P, Q = V[i], V[j]
        out = []
        for e in np.unique(np.round(ConvexHull(P).equations, 12), axis=0):
            n, off = e[:3] / np.linalg.norm(e[:3]), e[3] / np.linalg.norm(e[:3])
            if np.sum(np.abs(Q @ n + off) < on_plane) < 3 or np.any(Q @ n + off < -on_plane):
                continue
            face = P[np.abs(P @ n + off) < on_plane]
            c = face.mean(0)
            x = face[0] - c
            x /= np.linalg.norm(x) + 1e-30
            ring = face[np.argsort(np.arctan2((face - c) @ np.cross(n, x), (face - c) @ x))]
            for k in range(len(ring)):
                a, b = ring[k], ring[(k + 1) % len(ring)]
                if np.linalg.norm(b - a) > 1e-9:
                    out.append((a, (b - a) / np.linalg.norm(b - a), n))
        return out

    def turned(n, t, ang):
        return n * np.cos(ang) + np.cross(t, n) * np.sin(ang) + t * (t @ n) * (1 - np.cos(ang))

    changed = True
    while changed:
        changed = False
        for i in sorted(range(len(V)), key=lambda k: wid[k]):
            if wid[i] >= target_m:
                break
            lo, hi = V[i].min(0) - 1e-9, V[i].max(0) + 1e-9
            best = None
            for j in range(len(V)):
                if j == i or np.any(V[j].min(0) > hi) or np.any(V[j].max(0) < lo):
                    continue
                total, slack = vol[i] + vol[j], layer * (area[i] + area[j])
                for a, t, n in candidates(i, j):
                    for deg in coarse_deg:
                        if deg == 0:
                            continue
                        m = turned(n, t, np.radians(deg))
                        got = split(V[i], V[j], m, m @ a, total, slack)
                        if got is None or got[0] <= min(wid[i], wid[j]) + 1e-7:
                            continue
                        for fine in fine_deg:          # refine around the coarse optimum
                            mm = turned(n, t, np.radians(deg + fine))
                            g2 = split(V[i], V[j], mm, mm @ a, total, slack) if fine else got
                            if g2 is not None and (best is None or g2[0] > best[0]):
                                best = (g2[0], j, g2[1])
            if best is not None:
                score, j, ((A, va, sa), (B, vb, sb)) = best
                V[i], V[j], vol[i], vol[j], area[i], area[j] = A, B, va, vb, sa, sb
                wid[i], wid[j] = _width3(A), _width3(B)
                recuts += 1
                changed = True
    out = [(v, np.asarray(trimesh.convex.convex_hull(v).faces)) for v in V]
    return out, dict(target_mm=round(1000 * target_m, 4), recuts=recuts, thinnest_before_mm=round(1000 * before, 4),
                     thinnest_after_mm=round(1000 * min(wid), 4), below_target=int(sum(w < target_m for w in wid)))
