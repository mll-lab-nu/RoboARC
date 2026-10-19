"""Knot centre lines for the redesigned knot_untying tiers (several ropes, anchored ends, compact knots), in metres.

Coordinates are task-local: x, y on the table relative to the scene origin, z above the table top. A rope is an (N, 3)
centre line sampled at ~1 mm. Knot diagrams are authored in the plane and lifted into 3D: every projected crossing
(self or between ropes) is assigned an over strand; the under strand stays on the table (z = r + clearance) and the
over strand rises on a compact C2 bump to 2 r + gap above it. The parametric cores (the family's own Fourier trefoil and
figure-eight curves, topogym/assets/geometry/knot_untying.py) carry their over / under in their z, which the lift keeps.

The authored lift is only the birth's topology; the start states are relaxed (and tightened by a simulated pull) on the
MuJoCo capsule chain (planning/knot_untying/redesign/mjlab.py), whose bend stops and contacts decide the real shape.
"""
from __future__ import annotations

import numpy as np

R_ROPE = .004              # the family's rope radius (KnotLayout.rope_radius_m)
TABLE_CLEARANCE = .001     # the family's authored centre-line clearance above the table (KnotLayout.table_clearance_m)


# --------------------------------------------------------------------------------------------- basic curve tools
def arclength(p):
    p = np.asarray(p, float)
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def resample(p, ds=.001):
    p = np.asarray(p, float)
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-12]
    p = p[keep]
    s = arclength(p)
    n = max(2, int(np.ceil(s[-1] / ds)) + 1)
    u = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(u, s, p[:, k]) for k in range(p.shape[1])])


def at(p, s_query):
    s = arclength(p)
    return np.column_stack([np.interp(s_query, s, p[:, k]) for k in range(p.shape[1])])


def tangent_at(p, s_query, h=.004):
    s = arclength(p)
    a = at(p, np.clip(np.asarray(s_query) - h, 0, s[-1]))
    b = at(p, np.clip(np.asarray(s_query) + h, 0, s[-1]))
    t = b - a
    return t / np.linalg.norm(t, axis=-1, keepdims=True)


def hermite(p0, t0, p1, t1, n=200, scale=None):
    """Cubic Hermite from p0 (tangent t0) to p1 (tangent t1); tangents scaled to `scale` (default the chord)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0) if scale is None else scale
    t0 = np.asarray(t0, float) / np.linalg.norm(t0) * L
    t1 = np.asarray(t1, float) / np.linalg.norm(t1) * L
    u = np.linspace(0, 1, n)[:, None]
    h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    return h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1


def straight(p, direction, length, ds=.001):
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    u = np.linspace(0, length, max(2, int(np.ceil(length / ds)) + 1))[:, None]
    return np.asarray(p, float) + u * d


def join(*parts):
    out = [np.asarray(parts[0], float)]
    for q in parts[1:]:
        q = np.asarray(q, float)
        if np.linalg.norm(q[0] - out[-1][-1]) < 1e-9:
            q = q[1:]
        out.append(q)
    return np.vstack(out)


def rot2(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def place(core, xy=(0., 0.), yaw_deg=0., mirror=False):
    """Rotate (about z) and translate a curve; mirror flips x first (the knot's handedness)."""
    p = np.array(core, float)
    if mirror:
        p[:, 0] *= -1
    p[:, :2] = p[:, :2] @ rot2(np.deg2rad(yaw_deg)).T + np.asarray(xy, float)
    return p


# --------------------------------------------------------------------------------------------- parametric cores
def figure_eight_core(a, h=1., cut=.5, n=4001):
    """The family's open figure-eight core (KnotLayout figure_eight): ((2 + cos 2t) sin 3t, -(2 + cos 2t) cos 3t) a,
    z = h sin 4t (over / under carrier), cut at the bottom outer arc. Ends near (+-, -3a), both leaving downwards."""
    t = np.linspace(cut, 2 * np.pi - cut, n)
    r = 2 + np.cos(2 * t)
    return np.column_stack((a * r * np.sin(3 * t), -a * r * np.cos(3 * t), h * np.sin(4 * t)))


def trefoil_core(a, h=1., cut=.5, n=2401):
    """The family's open overhand core (KnotLayout overhand_trefoil): a (sin t + 2 sin 2t, cos t - 2 cos 2t), z = h sin 3t,
    t in [pi + cut, 3 pi - cut]. Ends near the bottom (-y), leaving downwards."""
    t = np.linspace(np.pi + cut, 3 * np.pi - cut, n)
    return np.column_stack((a * (np.sin(t) + 2 * np.sin(2 * t)), a * (np.cos(t) - 2 * np.cos(2 * t)), h * np.sin(3 * t)))


def end_frame(curve, which):
    """(point, outward unit tangent) of a curve end ('start' or 'end'), in the plane."""
    p = np.asarray(curve, float)
    if which == "start":
        d = p[0, :2] - p[min(20, len(p) - 1), :2]
        return p[0], np.r_[d / np.linalg.norm(d), 0.]
    d = p[-1, :2] - p[max(-21, -len(p)), :2]
    return p[-1], np.r_[d / np.linalg.norm(d), 0.]


# --------------------------------------------------------------------------------------------- crossings
def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def projected_crossings(ropes):
    """Every transverse XY crossing between the segments of the given ropes (self crossings: non-adjacent segments).
    Returns dicts: a=(rope, s), b=(rope, s) with a's s < b's s for self crossings, za, zb (centre heights), angle_deg,
    xy. Material s from each rope's own arc length."""
    out = []
    S = [arclength(p) for p in ropes]
    for i, P in enumerate(ropes):
        for j in range(i, len(ropes)):
            Q = ropes[j]
            A, U = P[:-1], np.diff(P, axis=0)
            B, V = Q[:-1], np.diff(Q, axis=0)
            # bounding-box prefilter
            for k in range(len(A)):
                if i == j:
                    lo = k + 2
                    if lo >= len(B):
                        break
                    b, v = B[lo:], V[lo:]
                    off = lo
                else:
                    b, v = B, V
                    off = 0
                a, u = A[k], U[k]
                delta = b[:, :2] - a[:2]
                den = _cross2(u[:2][None], v[:, :2])
                ok = np.abs(den) > 1e-16
                s = np.where(ok, _cross2(delta, v[:, :2]) / np.where(ok, den, 1), -1)
                t = np.where(ok, _cross2(delta, u[:2][None]) / np.where(ok, den, 1), -1)
                hit = np.flatnonzero((s >= 0) & (s < 1) & (t >= 0) & (t < 1))
                for h in hit:
                    m = off + h
                    pa = a + s[h] * u
                    pb = B[m] + t[h] * V[m]
                    sa = S[i][k] + s[h] * (S[i][k + 1] - S[i][k])
                    sb = S[j][m] + t[h] * (S[j][m + 1] - S[j][m])
                    cosang = abs(np.dot(u[:2], V[m][:2])) / (np.linalg.norm(u[:2]) * np.linalg.norm(V[m][:2]) + 1e-18)
                    out.append(dict(a=(i, float(sa)), b=(j, float(sb)), za=float(pa[2]), zb=float(pb[2]),
                                    xy=pa[:2].tolist(), angle_deg=float(np.degrees(np.arccos(np.clip(cosang, 0, 1))))))
    return out


def gauss_code(ropes, crossings=None):
    """Per rope: [(s, crossing index, 'O' | 'U')] in material order (O: this strand is over there)."""
    crossings = projected_crossings(ropes) if crossings is None else crossings
    codes = [[] for _ in ropes]
    for idx, c in enumerate(crossings):
        a_over = c["za"] > c["zb"]
        codes[c["a"][0]].append((c["a"][1], idx, "O" if a_over else "U"))
        codes[c["b"][0]].append((c["b"][1], idx, "U" if a_over else "O"))
    return [sorted(c) for c in codes]


# --------------------------------------------------------------------------------------------- the lift
def bump(u):
    """C2 compact bump, 1 at 0, 0 for |u| >= 1 (the family's simplified-loop height profile)."""
    u = np.clip(np.abs(u), 0, 1)
    return (.5 + .5 * np.cos(np.pi * u)) ** 2


def lift(ropes, over=None, *, r=R_ROPE, gap=.0012, clearance=TABLE_CLEARANCE, w_max=.03, w_min=.012, margin=.004):
    """3D centre lines from (planar) ropes: under strands on the table, over strands lifted on bumps of height 2 r + gap.

    over: None -> decided by the input z (higher z goes over: the parametric cores' carrier); or a function
    crossing -> 'a' | 'b'. Each over-bump's half width is the material distance to the strand's nearest under-visit
    (minus `margin`), clipped to [w_min, w_max]."""
    # the carrier z (over / under) scaled down, so every arc length below is the planar one
    carrier = [np.column_stack([np.asarray(p, float)[:, :2],
                                1e-9 * (np.asarray(p, float)[:, 2] if np.asarray(p).shape[1] > 2 else np.zeros(len(p)))])
               for p in ropes]
    carrier = [resample(p) for p in carrier]
    flat = [np.column_stack([p[:, :2], np.zeros(len(p))]) for p in carrier]
    cr = projected_crossings(carrier)
    visits = [[] for _ in ropes]
    for c in cr:
        a_over = (c["za"] > c["zb"]) if over is None else (over(c) == "a")
        c["over"] = "a" if a_over else "b"
        visits[c["a"][0]].append((c["a"][1], a_over, c["angle_deg"]))
        visits[c["b"][0]].append((c["b"][1], not a_over, c["angle_deg"]))
    out = []
    height = 2 * r + gap
    for k, p in enumerate(flat):
        s = arclength(p)
        keep = np.ones(len(p))          # smooth union of the bumps: 1 - prod(1 - b_i) (no crease where two meet)
        unders = np.array([v for v, o, _ in visits[k] if not o]) if visits[k] else np.zeros(0)
        for sv, is_over, angle in visits[k]:
            if not is_over:
                continue
            # a shallow crossing needs a plateau: the strands separate laterally at d sin(angle) only
            plateau = min(.02, 2 * r / max(np.tan(np.deg2rad(max(angle, 5.))), 1e-6))
            extent = w_max + plateau if not len(unders) else float(np.clip(np.min(np.abs(unders - sv)) - margin,
                                                                          w_min, w_max + plateau))
            plateau = min(plateau, max(0., extent - w_min))
            w = extent - plateau
            u = np.maximum(np.abs(s - sv) - plateau, 0.) / w
            keep *= 1. - bump(u)
        q = p.copy()
        q[:, 2] = r + clearance + height * (1. - keep)
        out.append(q)
    return out, cr


# --------------------------------------------------------------------------------------------- validators
def min_nonlocal_distance(ropes, exclusion=None, r=R_ROPE, stride=1):
    """Smallest 3D centre-line distance between material points more than `exclusion` apart along the same rope (or on
    different ropes). Point-sampled at ~1 mm (conservative to ~0.5 mm). Returns (distance, (rope, s), (rope, s))."""
    from scipy.spatial import cKDTree
    exclusion = np.pi * r if exclusion is None else exclusion
    pts, owner, ss = [], [], []
    for k, p in enumerate(ropes):
        q = resample(p, .001)[::stride]
        pts.append(q)
        owner.append(np.full(len(q), k))
        ss.append(arclength(q) if stride == 1 else arclength(resample(p, .001))[::stride])
    P, O, S = np.vstack(pts), np.concatenate(owner), np.concatenate(ss)
    tree = cKDTree(P)
    best = (np.inf, None, None)
    pairs = tree.query_pairs(4 * r, output_type="ndarray")
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        far = (O[i] != O[j]) | (np.abs(S[i] - S[j]) > exclusion)
        i, j = i[far], j[far]
        if len(i):
            d = np.linalg.norm(P[i] - P[j], axis=1)
            m = int(np.argmin(d))
            best = (float(d[m]), (int(O[i[m]]), float(S[i[m]])), (int(O[j[m]]), float(S[j[m]])))
    return best


def bend_radius(p):
    p = resample(p, .001)
    a, b = np.diff(p, axis=0)[:-1], np.diff(p, axis=0)[1:]
    den = 2 * np.linalg.norm(np.cross(a, b), axis=1)
    num = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(a + b, axis=1)
    return np.divide(num, den, out=np.full(len(den), np.inf), where=den > 1e-15)


def max_link_angle(p, link=.0045):
    """Largest angle between consecutive chain links when the curve is cut into `link` m links (the chain's bend
    stops engage at 0.9 rad)."""
    q = resample(p, .0005)
    s = arclength(q)
    n = max(2, int(round(s[-1] / link)))
    nodes = np.column_stack([np.interp(np.linspace(0, s[-1], n + 1), s, q[:, k]) for k in range(3)])
    d = np.diff(nodes, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return float(np.max(np.arccos(np.clip(np.einsum("ij,ij->i", d[:-1], d[1:]), -1, 1))))


def report(ropes, r=R_ROPE):
    cr = projected_crossings(ropes)
    d, pa, pb = min_nonlocal_distance(ropes, r=r)
    return dict(lengths_m=[float(arclength(p)[-1]) for p in ropes], crossings=len(cr),
                self_crossings=[sum(1 for c in cr if c["a"][0] == c["b"][0] == k) for k in range(len(ropes))],
                mutual_crossings=sum(1 for c in cr if c["a"][0] != c["b"][0]),
                gauss=[[(round(s, 4), i, o) for s, i, o in g] for g in gauss_code(ropes, cr)],
                min_nonlocal_distance_m=d, min_surface_gap_m=d - 2 * r, closest=[pa, pb],
                min_bend_radius_m=float(min(bend_radius(p).min() for p in ropes)),
                max_link_angle_rad=float(max(max_link_angle(p) for p in ropes)),
                min_z_m=float(min(p[:, 2].min() for p in ropes)), max_z_m=float(max(p[:, 2].max() for p in ropes)))
