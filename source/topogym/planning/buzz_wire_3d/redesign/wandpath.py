"""The wand's pose along the wire, and the grasp family.

Wand frame (the family's wand.urdf): origin at the ring centre, the ring in the xy plane (normal z), the stem and the
rectangular grip along +x (grip box: x 30.52 mm long, y 9.8 mm wide, z 7 mm thick). A pose on the wire at rail arc
length s: the ring centre on the wire's centre line, the ring normal along the wire tangent, and the handle at angle psi
about the tangent, measured from a parallel-transported reference frame (`WirePath.ref`).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


class WirePath:
    def __init__(self, rail_world, step=0.001):
        rail = np.asarray(rail_world, float)
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(rail, axis=0), axis=1))]
        self.length = float(arc[-1])
        n = int(np.ceil(self.length / step)) + 1
        self.s = np.linspace(0, self.length, n)
        self.p = np.stack([np.interp(self.s, arc, rail[:, k]) for k in range(3)], 1)
        t = np.gradient(self.p, axis=0)
        # smooth the tangent over +-2 mm against the rail's 2.8 mm chords
        k = np.ones(5) / 5
        t = np.stack([np.convolve(np.pad(t[:, j], 2, mode="edge"), k, mode="valid") for j in range(3)], 1)
        self.t = t / np.linalg.norm(t, axis=1, keepdims=True)
        # parallel-transported reference: starts as world -y projected on the first normal plane
        x = np.array([0., -1., 0.]) - (np.array([0., -1., 0.]) @ self.t[0]) * self.t[0]
        if np.linalg.norm(x) < 1e-6:
            x = np.array([1., 0., 0.]) - self.t[0][0] * self.t[0]
        x /= np.linalg.norm(x)
        ref = [x]
        for i in range(1, n):
            a, b = self.t[i - 1], self.t[i]
            v = np.cross(a, b); c = float(a @ b); sn = np.linalg.norm(v)
            if sn > 1e-12:
                x = R.from_rotvec(v / sn * np.arctan2(sn, c)).apply(x)
            x = x - (x @ b) * b; x /= np.linalg.norm(x)
            ref.append(x)
        self.ref = np.asarray(ref)
        # bending axis (binormal) for lead / lag tilts: t x dt/ds where the wire bends, carried over straight parts
        dt = np.gradient(self.t, axis=0)
        b = np.cross(self.t, dt)
        nb = np.linalg.norm(b, axis=1)
        prev = np.cross(self.t[0], self.ref[0])
        out = []
        for i in range(n):
            v = b[i] / nb[i] if nb[i] > 1e-5 else prev
            v = v - (v @ self.t[i]) * self.t[i]
            if np.linalg.norm(v) < 1e-9:
                v = prev
            v /= np.linalg.norm(v)
            if v @ prev < 0 and nb[i] > 1e-5:
                v = -v
            out.append(v); prev = v
        self.b = np.asarray(out)

    def idx(self, s):
        return int(np.clip(np.round(s / (self.s[1] - self.s[0])), 0, len(self.s) - 1))

    def frame(self, s, psi, lag=0.0):
        """(ring centre, wand rotation) at arc length s with the handle at angle psi (rad) about the tangent; `lag`
        (rad) tilts the ring about the wire's bending axis (lead / lag in a bend)."""
        i = self.idx(s)
        t, x0 = self.t[i], self.ref[i]
        h = np.cos(psi) * x0 + np.sin(psi) * np.cross(t, x0)
        W = R.from_matrix(np.column_stack([h, np.cross(t, h), t]))
        if lag:
            W = R.from_rotvec(lag * self.b[i]) * W
        return self.p[i].copy(), W

    def psi_toward(self, s, direction):
        """psi pointing the handle as close as possible to `direction`."""
        i = self.idx(s)
        t, x0 = self.t[i], self.ref[i]
        d = np.asarray(direction, float) - (np.asarray(direction, float) @ t) * t
        if np.linalg.norm(d) < 1e-9:
            return 0.0
        return float(np.arctan2(d @ np.cross(t, x0), d @ x0))


# ------------------------------------------------------------------------- grasps
def handle_centre(layout, fraction=.55):
    return np.array([layout.loop_radius_m + layout.stem_length_m + fraction * layout.grip_length_m, 0., 0.])


def grasp(kind, layout, *, roll=0.0, fraction=.55, depth=0.0, pitch=0.0):
    """TCP pose in the wand frame (position, rotation).
    kind 'end':  approach along the handle towards the ring (hand z = -x_wand); fingers close across the grip's width
                 (hand y = y_wand) at roll 0, across its thickness at roll 90 deg; roll turns about the approach.
    kind 'top':  approach along -z_wand (onto the broad face; the start grasp from the deck); fingers across the width.
    kind 'side': approach along -y_wand (onto the narrow edge); fingers across the thickness.
    kinds 'topb' / 'sideb': the same onto the opposite face / edge (+z_wand / +y_wand): the ring is symmetric, so these
                 are the grasps of the wand flipped about its handle axis (the hand on the other side of the ring).
    depth: TCP offset along the approach (positive = deeper).
    pitch: tilt (rad) about the finger axis (hand y): for 'top', positive leans the hand body away from the ring."""
    c = handle_centre(layout, fraction)
    if kind == "end":
        z = np.array([-1., 0, 0]); y = np.array([0., 1, 0])
    elif kind == "top":
        z = np.array([0., 0, -1]); y = np.array([0., 1, 0])
    elif kind == "side":
        z = np.array([0., -1, 0]); y = np.array([0., 0, 1])
    elif kind == "topb":                  # onto the other broad face (the hand on the ring's other side)
        z = np.array([0., 0, 1]); y = np.array([0., 1, 0])
    elif kind == "sideb":                 # onto the other narrow edge
        z = np.array([0., 1, 0]); y = np.array([0., 0, 1])
    else:
        raise ValueError(kind)
    x = np.cross(y, z)
    Rg = R.from_matrix(np.column_stack([x, y, z])) * R.from_euler("z", roll) * R.from_euler("y", pitch)
    return c + depth * Rg.apply([0, 0, 1]), Rg


def parse_grasp(spec, layout):
    """'kind:roll_deg[:fraction[:pitch_deg[:depth_mm]]]' -> grasp pose."""
    f = spec.split(":")
    kind, roll = f[0], float(f[1]) if len(f) > 1 else 0.0
    frac = float(f[2]) if len(f) > 2 else .55
    pitch = float(f[3]) if len(f) > 3 else 0.0
    depth = float(f[4]) / 1000 if len(f) > 4 else 0.0
    return grasp(kind, layout, roll=np.deg2rad(roll), fraction=frac, pitch=np.deg2rad(pitch), depth=depth)


def tcp_from_wand(wand_p, wand_r, g):
    gp, gr = g
    return wand_p + wand_r.apply(gp), wand_r * gr


def wand_from_tcp(tcp_p, tcp_r, g):
    gp, gr = g
    wr = tcp_r * gr.inv()
    return tcp_p - wr.apply(gp), wr
