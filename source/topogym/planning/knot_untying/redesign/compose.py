"""Composition of redesigned knot ropes: parametric knot cores, smooth tails, connectors between stacked cores, second
ropes threaded through loops. Everything task-local (x, y on the table, z above it; the lift puts strands on the table).
"""
from __future__ import annotations

import numpy as np

from topogym.tasks.knot_untying.redesign import geometry as G


def tail_from(end_point, end_dir, side_dir, length, turn=.035, ds=.001):
    """From a core end (point, outward tangent) turn smoothly into a straight tail along side_dir (planar)."""
    p0 = np.r_[np.asarray(end_point, float)[:2], 0.]
    t0 = np.r_[np.asarray(end_dir, float)[:2], 0.]
    sd = np.r_[np.asarray(side_dir, float)[:2], 0.]
    sd /= np.linalg.norm(sd)
    p1 = p0 + turn * (t0 / np.linalg.norm(t0)) * .6 + turn * sd
    bend = G.hermite(p0, t0, p1, sd, n=200, scale=turn * 1.2)
    used = G.arclength(bend)[-1]
    rest = max(.0, length - used)
    return G.join(bend, G.straight(p1, sd, rest, ds)) if rest > 0 else bend


def knot_with_tails(core, *, left_dir=(-1, 0), right_dir=(1, 0), left_len=.3, right_len=.4, turn=.035):
    """A core (start end, ..., end end) with tails: returns the rope from the LEFT tail's free end through the core to
    the RIGHT tail's end, whichever core end is further left becoming the left one. Planar carrier z kept for lifting."""
    core = np.asarray(core, float)
    if core[0, 0] > core[-1, 0]:
        core = core[::-1].copy()
    p_s, d_s = G.end_frame(core, "start")
    p_e, d_e = G.end_frame(core, "end")
    zs, ze = core[0, 2], core[-1, 2]
    left = tail_from(p_s, d_s, left_dir, left_len, turn)
    right = tail_from(p_e, d_e, right_dir, right_len, turn)
    left[:, 2] = zs
    right[:, 2] = ze
    return G.join(left[::-1], core, right)


def connector(p0, d0, p1, d1, n=300):
    """Planar Hermite connector from point p0 leaving along d0 to point p1 arriving along d1."""
    p0, p1 = np.r_[np.asarray(p0, float)[:2], 0.], np.r_[np.asarray(p1, float)[:2], 0.]
    return G.hermite(p0, np.r_[d0[:2], 0.], p1, np.r_[d1[:2], 0.], n=n)


def spline_path(points, start_dir=None, end_dir=None, ds=.001):
    """Smooth planar C1 path through waypoints (Catmull-Rom tangents; optional end tangents), z = 0."""
    P = np.array([np.r_[np.asarray(p, float)[:2], 0.] for p in points])
    n = len(P)
    T = np.zeros_like(P)
    for i in range(n):
        if i == 0:
            T[i] = (P[1] - P[0]) if start_dir is None else np.r_[start_dir[:2], 0.]
        elif i == n - 1:
            T[i] = (P[-1] - P[-2]) if end_dir is None else np.r_[end_dir[:2], 0.]
        else:
            T[i] = .5 * (P[i + 1] - P[i - 1])
        T[i] /= np.linalg.norm(T[i])
    parts = []
    for i in range(n - 1):
        L = np.linalg.norm(P[i + 1] - P[i])
        parts.append(G.hermite(P[i], T[i], P[i + 1], T[i + 1], n=max(20, int(L / ds)), scale=L))
    return G.join(*parts)


def knot_with_paths(core, left_points, right_points, *, turn=.03):
    """A core with tails routed through waypoints. left_points / right_points: waypoints from the core's left / right
    end outwards (the first waypoint is reached after a smooth turn out of the core). Returns the rope from the left
    tail's far end to the right tail's far end (carrier z of the core kept at the ends)."""
    core = np.asarray(core, float)
    if core[0, 0] > core[-1, 0]:
        core = core[::-1].copy()
    p_s, d_s = G.end_frame(core, "start")
    p_e, d_e = G.end_frame(core, "end")
    left = spline_path([p_s] + [np.asarray(p, float) for p in left_points], start_dir=d_s)
    right = spline_path([p_e] + [np.asarray(p, float) for p in right_points], start_dir=d_e)
    left[:, 2] = core[0, 2]
    right[:, 2] = core[-1, 2]
    return G.join(left[::-1], core, right)


def rope_through(start_points, cores, links, end_points, *, start_z=0., end_z=0.):
    """One rope: from start_points[0] through the waypoints to core 0, then core 0 (start end -> end end), links[0]
    waypoints to core 1, ..., then end_points. Cores keep their carrier z; paths get the carrier z of the core end they
    touch (or start_z / end_z at the free ends). No flipping: core i is traversed from its first to its last point."""
    pieces = []
    p_s, d_s = G.end_frame(cores[0], "start")
    lead = spline_path([np.asarray(p, float) for p in start_points] + [p_s], end_dir=-d_s)
    lead[:, 2] = cores[0][0, 2]
    pieces.append(lead)
    for i, core in enumerate(cores):
        pieces.append(np.asarray(core, float))
        p_e, d_e = G.end_frame(core, "end")
        if i + 1 < len(cores):
            q_s, dq_s = G.end_frame(cores[i + 1], "start")
            link = spline_path([p_e] + [np.asarray(p, float) for p in links[i]] + [q_s], start_dir=d_e, end_dir=-dq_s)
            u = np.linspace(0, 1, len(link))
            link[:, 2] = (1 - u) * core[-1, 2] + u * cores[i + 1][0, 2]
            pieces.append(link)
    tail = spline_path([p_e] + [np.asarray(p, float) for p in end_points], start_dir=d_e)
    tail[:, 2] = cores[-1][-1, 2]
    pieces.append(tail)
    return G.join(*pieces)


def free_path(points, z_carrier):
    """A rope (or rope part) without a core: planar path through points; z_carrier(s, xy) -> carrier z per sample
    (decides over / under at crossings with other ropes, compared with their carriers)."""
    p = spline_path(points)
    s = G.arclength(p)
    p[:, 2] = [z_carrier(si, pi[:2]) for si, pi in zip(s, p)]
    return p
