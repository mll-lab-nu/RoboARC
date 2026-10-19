"""Trajectory complexity and object pose change of a buzz_wire_3d run (one Panda arm), from a record.py recording.
Definitions follow the soft-body redesign's antihack/trajectory.py (COMMON.md spec, definitions v2; commit 1f7b30cf1 of
the work notes), for a single arm:
  gripper events   commanded closedness (native / 0.8) crossing 0.6 upward = close, 0.4 downward = open
  holding          the grasp-assist weld (the recording's `held`); gaps <= 2 rows filled, runs < 3 rows dropped
  grasp / release  holding off -> on / on -> off; regrasp = a grasp after an earlier release
  dwell            TCP slower than 3 mm/s for >= 0.3 s
  motion phases    segments between gripper events, grasp events and dwell boundaries (merged within 0.2 s) in which the
                   TCP moves >= 5 mm; phases_total = motion phases + grasps + releases
  waypoints        interior Ramer-Douglas-Peucker vertices (5 mm) of the TCP path turning > 30 deg, plus halts (slower than
                   3 mm/s for >= 0.15 s between moves of >= 1 cm), merged within 0.2 s
  path lengths     TCP path (m), joint path (rad, per-row L2), duration = rows x 25 ms
Object (the wand): start -> end translation (m) and rotation (deg, with the ring's half-turn symmetry about the handle
axis), the wand's own path length (m) and accumulated rotation (deg) while moving."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

DT = 0.025


def _runs(mask):
    m = np.asarray(mask, bool).astype(np.int8)
    d = np.diff(np.r_[0, m, 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def clean_mask(mask, fill_gap=2, min_run=3):
    m = np.asarray(mask, bool).copy()
    for a, b in _runs(~m):
        if a > 0 and b < len(m) and (b - a) <= fill_gap:
            m[a:b] = True
    for a, b in _runs(m):
        if b - a < min_run:
            m[a:b] = False
    return m


def gripper_events(closed, close_th=0.6, open_th=0.4):
    ev, state = [], bool(closed[0] >= close_th) if len(closed) else False
    for i, v in enumerate(closed):
        if not state and v >= close_th:
            state = True; ev.append((i, "close"))
        elif state and v <= open_th:
            state = False; ev.append((i, "open"))
    return ev


def rdp(P, eps):
    P = np.asarray(P, float); n = len(P)
    if n < 3:
        return list(range(n))
    keep = np.zeros(n, bool); keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = P[b] - P[a]; L = np.linalg.norm(seg); Q = P[a + 1:b] - P[a]
        d = np.linalg.norm(Q, axis=1) if L < 1e-12 else np.linalg.norm(Q - np.outer(Q @ (seg / L), seg / L), axis=1)
        k = int(np.argmax(d))
        if d[k] > eps:
            keep[a + 1 + k] = True; stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return list(np.flatnonzero(keep))


def waypoints(P, dt=DT, rdp_eps=0.005, turn_deg=30.0, halt_speed=0.003, halt_s=0.15, stop_min_move=0.01, merge_s=0.2):
    P = np.asarray(P, float); T = len(P)
    if T < 3:
        return []
    idx = rdp(P, rdp_eps)
    corners = []
    for j in range(1, len(idx) - 1):
        u, v = P[idx[j]] - P[idx[j - 1]], P[idx[j + 1]] - P[idx[j]]
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu > 1e-9 and nv > 1e-9 and np.degrees(np.arccos(np.clip(u @ v / (nu * nv), -1, 1))) > turn_deg:
            corners.append(int(idx[j]))
    sp = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1) / dt]
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    halts = [(a, b) for a, b in _runs(sp < halt_speed) if (b - a) * dt >= halt_s]
    stops = []
    for a, b in halts:
        if a == 0 or b >= T:
            continue
        prev_end = max([hb for ha, hb in halts if hb <= a] + [0])
        next_start = min([ha for ha, hb in halts if ha >= b] + [T - 1])
        if arc[a] - arc[prev_end] >= stop_min_move and arc[next_start] - arc[min(b, T - 1)] >= stop_min_move:
            stops.append(int(a))
    rows = sorted(set(corners) | set(stops))
    merged = []
    for r in rows:
        if not merged or (r - merged[-1]) * dt > merge_s:
            merged.append(r)
    return merged


def trajectory(rec, dt=DT):
    """rec: a record.py npz (dict-like): tcp_p (T+1, 3), q (T+1, 7), held (T+1,), grip_cmd (T+1,)."""
    P = np.asarray(rec["tcp_p"], float)
    q = np.asarray(rec["q"], float)
    closed = np.clip(np.asarray(rec["grip_cmd"], float) / 0.8, 0, 1)
    held = clean_mask(np.asarray(rec["held"], bool))
    gev = gripper_events(closed)
    grasps, releases = [], []
    for a, b in _runs(held):
        grasps.append(int(a)); releases.append(int(b)) if b < len(held) else None
    regrasps = max(0, len(grasps) - 1)
    sp = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1) / dt]
    dwell = [(a, b) for a, b in _runs(sp < 0.003) if (b - a) * dt >= 0.3]
    bounds = sorted(set([0, len(P) - 1] + [r for r, _ in gev] + grasps + releases + [x for ab in dwell for x in ab]))
    merged = []
    for b in bounds:
        if not merged or (b - merged[-1]) * dt > 0.2:
            merged.append(b)
    phases = []
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    for a, b in zip(merged[:-1], merged[1:]):
        if arc[b] - arc[a] >= 0.005:
            hold_frac = held[a:b].mean() if b > a else 0
            label = "transport" if hold_frac > 0.5 else ("retreat" if a in releases else "approach")
            phases.append(dict(start=int(a), end=int(b), moved_m=round(float(arc[b] - arc[a]), 4), label=label))
    wp = waypoints(P, dt)
    return dict(rows=int(len(P) - 1), duration_s=round((len(P) - 1) * dt, 2), tcp_path_m_total=round(float(arc[-1]), 3),
                joint_path_rad_total=round(float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1))), 3),
                motion_phases=len(phases), grasp_count=len(grasps), release_count=len(releases),
                regrasp_count=regrasps, handover_count=0, arms_used=1,
                phases_total=len(phases) + len(grasps) + len(releases), waypoints_total=len(wp),
                gripper_events=len(gev), dwells=len(dwell), held_rows=int(held.sum()),
                phases=phases[:200], waypoint_rows=wp[:400], grasp_rows=grasps, release_rows=releases)


def wand_change(rec):
    """Object pose change of the wand: start -> end, plus its path length and accumulated rotation."""
    p = np.asarray(rec["wand_p"], float); qw = np.asarray(rec["wand_q"], float)
    r = R.from_quat(qw[:, [1, 2, 3, 0]])
    flip = R.from_euler("x", 180, degrees=True)
    d = r[-1] * r[0].inv()
    ang = min(d.magnitude(), (r[-1] * flip * r[0].inv()).magnitude())
    steps = np.linalg.norm(np.diff(p, axis=0), axis=1)
    rots = (r[1:] * r[:-1].inv()).magnitude()
    return dict(translation_m=round(float(np.linalg.norm(p[-1] - p[0])), 4), rotation_deg=round(float(np.degrees(ang)), 2),
                path_m=round(float(steps.sum()), 3), accumulated_rotation_deg=round(float(np.degrees(rots.sum())), 1),
                start=p[0].round(4).tolist(), end=p[-1].round(4).tolist())
