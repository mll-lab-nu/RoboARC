"""Dense wire polylines for the buzz_wire_3d redesign (toy coordinates: metres, relative to the deck base's bottom
centre, z up from the table top; the deck top is at base_elevation + base thickness = 0.1425).

Every motif returns an (N, 3) polyline sampled at ~1 mm. `chain` joins motifs with C1 cubic connectors, so the
family's SpatialWire (a chord-length cubic spline through the controls) follows them without kinks."""
from __future__ import annotations

import numpy as np

STEP = .001


def _n(length, step=STEP):
    return max(2, int(np.ceil(length / step)) + 1)


def line(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.linspace(a, b, _n(np.linalg.norm(b - a)))


def arc(center, u, v, radius, a0, a1):
    """Circle arc centre + r (cos t u + sin t v), t from a0 to a1 (radians)."""
    c, u, v = (np.asarray(x, float) for x in (center, u, v))
    t = np.linspace(a0, a1, _n(abs(a1 - a0) * radius))
    return c + radius * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)


def helix(center, axis, u, radius, pitch, turns, a0=0.0):
    """A helix about `axis` through `center`: starts at angle a0 (from u towards axis x u), advances `pitch` per turn."""
    c, w, u = (np.asarray(x, float) for x in (center, axis, u))
    w = w / np.linalg.norm(w)
    u = u - (u @ w) * w; u /= np.linalg.norm(u)
    v = np.cross(w, u)
    length = turns * np.hypot(2 * np.pi * radius, pitch)
    t = np.linspace(0, turns * 2 * np.pi, _n(length))
    return (c + radius * (np.cos(t + a0)[:, None] * u + np.sin(t + a0)[:, None] * v)
            + (pitch * t / (2 * np.pi))[:, None] * w)


def bezier(p0, d0, p1, d1, handle0, handle1=None):
    """C1 cubic from p0 (leaving along d0) to p1 (arriving along d1)."""
    handle1 = handle0 if handle1 is None else handle1
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d0 = np.asarray(d0, float) / np.linalg.norm(d0); d1 = np.asarray(d1, float) / np.linalg.norm(d1)
    c = [p0, p0 + handle0 * d0, p1 - handle1 * d1, p1]
    est = np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[1]) + np.linalg.norm(c[3] - c[2])
    t = np.linspace(0, 1, _n(est))[:, None]
    return (1 - t) ** 3 * c[0] + 3 * (1 - t) ** 2 * t * c[1] + 3 * (1 - t) * t ** 2 * c[2] + t ** 3 * c[3]


def direction(poly, end):
    p = np.asarray(poly)
    return (p[-1] - p[-4]) if end else (p[3] - p[0])


def chain(pieces, handle=0.03):
    """Concatenate motifs; consecutive pieces whose ends do not meet are joined by a C1 Bezier connector."""
    out = [np.asarray(pieces[0], float)]
    for piece in pieces[1:]:
        piece = np.asarray(piece, float)
        a, b = out[-1], piece
        if np.linalg.norm(a[-1] - b[0]) > 1e-6:
            h = handle if np.isscalar(handle) else handle
            j = bezier(a[-1], direction(a, 1), b[0], direction(b, 0), h)
            out.append(j[1:-1])
        out.append(b if np.linalg.norm(out[-1][-1] - b[0]) > 1e-9 else b[1:])
    poly = np.concatenate(out)
    keep = np.r_[True, np.linalg.norm(np.diff(poly, axis=0), axis=1) > 2e-4]
    return poly[keep]


def resample(poly, step=STEP):
    poly = np.asarray(poly, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
    t = np.linspace(0, s[-1], _n(s[-1], step))
    return np.stack([np.interp(t, s, poly[:, k]) for k in range(3)], 1)


def curvature_radius(poly):
    p = resample(poly, .002)
    d1 = np.gradient(p, axis=0); d2 = np.gradient(d1, axis=0)
    k = np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-15)
    return 1 / np.maximum(k, 1e-12)


def fillet(points, radius):
    """A bent-wire path through corner points: straight runs joined by circular arcs of `radius` (scalar or one per
    interior corner), as a wire bender makes them. Raises if two neighbouring fillets overlap."""
    P = np.asarray(points, float)
    n = len(P)
    rad = np.full(n, float(radius)) if np.isscalar(radius) else np.r_[0, np.asarray(radius, float), 0]
    T = np.zeros(n)
    dirs = [(P[i + 1] - P[i]) / np.linalg.norm(P[i + 1] - P[i]) for i in range(n - 1)]
    for i in range(1, n - 1):
        th = np.arccos(np.clip(dirs[i - 1] @ dirs[i], -1, 1))
        T[i] = rad[i] * np.tan(th / 2) if th > 1e-9 else 0.0
    for i in range(n - 1):
        if T[i] + T[i + 1] > np.linalg.norm(P[i + 1] - P[i]) + 1e-12:
            raise ValueError(f"fillets overlap on segment {i}: {T[i] + T[i + 1]:.4f} > {np.linalg.norm(P[i + 1] - P[i]):.4f}")
    out = [P[0][None]]
    for i in range(1, n - 1):
        a, b = dirs[i - 1], dirs[i]
        p0 = P[i] - T[i] * a
        out.append(line(out[-1][-1], p0)[1:])
        th = np.arccos(np.clip(a @ b, -1, 1))
        if th < 1e-9:
            continue
        nrm = np.cross(a, b); nrm /= np.linalg.norm(nrm)
        inward = np.cross(nrm, a)                      # from p0 towards the arc centre
        c = p0 + rad[i] * inward
        u = p0 - c
        v = np.cross(nrm, u)
        out.append(arc(c, u / np.linalg.norm(u), v / np.linalg.norm(v), rad[i], 0.0, th)[1:])
    out.append(line(out[-1][-1], P[-1])[1:])
    poly = np.concatenate(out)
    keep = np.r_[True, np.linalg.norm(np.diff(poly, axis=0), axis=1) > 2e-4]
    return poly[keep]


def tangent_in(poly):
    p = np.asarray(poly)
    v = p[min(5, len(p) - 1)] - p[0]
    return v / np.linalg.norm(v)


def tangent_out(poly):
    p = np.asarray(poly)
    v = p[-1] - p[max(0, len(p) - 6)]
    return v / np.linalg.norm(v)


def hermite(points, tangents, scale=1.0):
    """A C1 cubic Hermite curve through key points with unit tangents (None: Catmull-Rom estimate); each segment's
    tangent magnitude is scale x its chord."""
    P = np.asarray(points, float)
    n = len(P)
    Tg = []
    for i in range(n):
        t = tangents[i] if tangents is not None and tangents[i] is not None else None
        if t is None:
            a = P[max(i - 1, 0)]; b = P[min(i + 1, n - 1)]
            t = b - a
        t = np.asarray(t, float); Tg.append(t / np.linalg.norm(t))
    out = [P[0][None]]
    for i in range(n - 1):
        chord = np.linalg.norm(P[i + 1] - P[i])
        m0, m1 = scale * chord * Tg[i], scale * chord * Tg[i + 1]
        u = np.linspace(0, 1, _n(chord * 1.3))[1:, None]
        h00 = 2 * u ** 3 - 3 * u ** 2 + 1; h10 = u ** 3 - 2 * u ** 2 + u
        h01 = -2 * u ** 3 + 3 * u ** 2; h11 = u ** 3 - u ** 2
        out.append(h00 * P[i] + h10 * m0 + h01 * P[i + 1] + h11 * m1)
    return np.concatenate(out)


def min_bend(poly, step=0.002):
    return float(np.min(curvature_radius(poly)))


class Turtle:
    """A wire bender in 3D: a position and a frame (heading h, left l, up u as columns of R); every command is exact
    (constant body-frame angular rate per unit length: straight runs, circular bends, helices), so the wire is C1 with
    known bend radii."""

    def __init__(self, p, heading, up_hint=(0., 0., 1.)):
        h = np.asarray(heading, float); h /= np.linalg.norm(h)
        u = np.asarray(up_hint, float) - (np.asarray(up_hint, float) @ h) * h
        if np.linalg.norm(u) < 1e-6:
            u = np.array([1., 0, 0]) - h[0] * h
        u /= np.linalg.norm(u)
        self.p = np.asarray(p, float)
        self.R = np.column_stack([h, np.cross(u, h), u])
        self.pts = [self.p.copy()]
        self.marks = {}

    @property
    def h(self):
        return self.R[:, 0]

    def mark(self, name):
        self.marks[name] = len(self.pts) - 1
        return self

    def _segment(self, length, omega_world, step=STEP):
        """Advance `length` along the heading while the frame turns at omega_world (rad/m, world axis, constant)."""
        from scipy.spatial.transform import Rotation as Rot
        n = max(1, int(np.ceil(length / step)))
        ds = length / n
        w = np.asarray(omega_world, float)
        for _ in range(n):
            # exact for a constant world axis: integrate the heading's rotation over ds in two half steps
            half = Rot.from_rotvec(w * ds / 2)
            hm = half.apply(self.R[:, 0])
            self.p = self.p + ds * hm if np.linalg.norm(w) < 1e-12 else self.p + self._chord(w, ds)
            self.R = (Rot.from_rotvec(w * ds).as_matrix()) @ self.R
            self.pts.append(self.p.copy())
        return self

    def _chord(self, w, ds):
        # displacement of a point moving along heading h(s) = exp(s [w]x) h0 over ds (exact integral)
        h0 = self.R[:, 0]
        k = np.linalg.norm(w)
        a = w / k
        par = (h0 @ a) * a
        perp = h0 - par
        th = k * ds
        return par * ds + (np.sin(th) / k) * perp + ((1 - np.cos(th)) / k) * np.cross(a, perp)

    def straight(self, length):
        return self._segment(length, np.zeros(3))

    def turn_to(self, direction, radius):
        """A planar bend of `radius` from the current heading to `direction`."""
        d = np.asarray(direction, float); d /= np.linalg.norm(d)
        h = self.h
        c = np.clip(h @ d, -1, 1)
        ang = np.arccos(c)
        if ang < 1e-9:
            return self
        ax = np.cross(h, d)
        if np.linalg.norm(ax) < 1e-9:
            ax = self.R[:, 2]
        ax /= np.linalg.norm(ax)
        return self._segment(ang * radius, ax * (1 / radius))

    def helix(self, axis, radius, pitch, turns):
        """A helix about the world `axis` (through a centre beside the current point), entered with the current
        heading; the heading must already make the helix angle with the axis (use helix_heading)."""
        a = np.asarray(axis, float); a /= np.linalg.norm(a)
        L = turns * np.hypot(2 * np.pi * radius, pitch)
        rate = 2 * np.pi * turns / L          # rotation of the frame about the axis per unit length
        return self._segment(L, a * rate)

    @staticmethod
    def helix_heading(axis, tangent, radius, pitch):
        """The heading that enters a helix about `axis` going along `tangent` (perpendicular to the axis)."""
        a = np.asarray(axis, float); a /= np.linalg.norm(a)
        t = np.asarray(tangent, float); t = t - (t @ a) * a; t /= np.linalg.norm(t)
        al = np.arctan2(pitch, 2 * np.pi * radius)
        return np.cos(al) * t + np.sin(al) * a

    def polyline(self):
        return np.asarray(self.pts)


def self_clearance(poly, skip=0.06):
    """Smallest distance between two wire points more than `skip` apart along the wire (m), and the pair's arc lengths."""
    p = resample(poly, .003)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    best = (np.inf, 0., 0.)
    for i in range(0, len(p), 2):
        far = np.abs(s - s[i]) > skip
        if not far.any():
            continue
        d = np.linalg.norm(p[far] - p[i], axis=1)
        k = int(np.argmin(d))
        if d[k] < best[0]:
            best = (float(d[k]), float(s[i]), float(s[far][k]))
    return best


def post_conflict(poly, end, tcp_offset=0.052, radius=0.05, height=0.30):
    """Smallest horizontal distance (m) from the vertical line above the resting handle (whose
    handle points at the deck centre) to the wire points below `height` above that post's base,
    excluding the post itself (points within 15 mm of the post axis)."""
    p = np.asarray(poly)
    post = p[0] if end == 0 else p[-1]
    hdir = -post[:2] / np.linalg.norm(post[:2])
    axis = post[:2] + tcp_offset * hdir
    sel = (p[:, 2] < post[2] + height) & (np.linalg.norm(p[:, :2] - post[:2], axis=1) > 0.015)
    if not sel.any():
        return np.inf
    return float(np.min(np.linalg.norm(p[sel, :2] - axis, axis=1)))
