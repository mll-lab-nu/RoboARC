"""Trajectory complexity (COMMON.md metric spec), one definition for recorded rows and planned oracles.

Inputs (any of):
  (a) recorded action rows (T, 14) + the ALOHA action layout + a frozen scene for FK (antihack.aloha.AlohaFK), and
      optionally an antihack.record recording (measured TCP, grasp-assist holdings, finger contacts);
  (b) a planned oracle: TCP waypoints per arm with times (or joint waypoints + FK) and a gripper event list.

Definitions (all thresholds are keyword arguments; the defaults are the ones reported):
  gripper events   commanded gripper closedness (0 open .. 1 closed) crossing 0.6 upward = close, 0.4 downward = open
  holding          grasp-assist weld held (rope_threading, blanket_flatten), else a physical pinch: commanded closed
                   (> 0.5) and both fingers touching a task object (knot_untying); gaps <= 2 rows filled, runs < 3
                   rows dropped
  grasp / release  holding False -> True / True -> False per arm
  hand-over        a grasp while the other arm holds (the same object) and that arm releases before this one does
  regrasp          a grasp of an object that was already held and released before (not a hand-over)
  dwell            both TCPs slower than 3 mm/s for >= 0.3 s
  motion phases    the segments between consecutive gripper events / grasp events / dwell boundaries in which a TCP
                   moves >= 5 mm, labelled approach (ends in a close / grasp of a moving arm), transport (a moving arm
                   holds), retreat (starts with a release), reposition (else); plus the grasp / hand-over / regrasp /
                   release events themselves. phases_total = motion phases + grasp-type events + releases
  waypoints        per arm: interior vertices of the TCP path simplified by Ramer-Douglas-Peucker (5 mm) whose turn
                   angle exceeds 30 deg, plus halts (the TCP slower than 3 mm/s for >= 0.15 s between moves of >= 1 cm;
                   v2: speed dips inside a straight move do not count); merged within 0.2 s
  path lengths     TCP path per arm (m, sum of |dP| over rows); joint-space path per arm (rad, sum of the per-row L2
                   norm of the arm joint step); duration (s) = rows x control period
"""
from __future__ import annotations

import numpy as np

from .aloha import ARMS, gripper_closedness, layout_columns

DEFAULTS = dict(close_th=0.6, open_th=0.4, hold_closed=0.5, fill_gap_rows=2, min_hold_rows=3, dwell_speed=0.003,
                dwell_s=0.3, phase_min_move=0.005, rdp_eps=0.005, turn_deg=30.0, halt_speed=0.003, halt_s=0.15,
                stop_min_move=0.01, merge_s=0.2, definitions="v2")


# ------------------------------------------------------------------------------------------------ helpers
def _runs(mask):
    """[(start, end_exclusive)] of True runs."""
    m = np.asarray(mask, bool).astype(np.int8)
    d = np.diff(np.r_[0, m, 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def clean_mask(mask, fill_gap: int, min_run: int):
    m = np.asarray(mask, bool).copy()
    inv = _runs(~m)
    for a, b in inv:
        if a > 0 and b < len(m) and (b - a) <= fill_gap:
            m[a:b] = True
    for a, b in _runs(m):
        if b - a < min_run:
            m[a:b] = False
    return m


def gripper_events(closedness, close_th=0.6, open_th=0.4):
    """[(row, 'close' | 'open')] from a closedness series (hysteresis)."""
    c = np.asarray(closedness, float)
    ev, state = [], c[0] >= close_th if len(c) else False
    for i, v in enumerate(c):
        if not state and v >= close_th:
            state = True; ev.append((i, "close"))
        elif state and v <= open_th:
            state = False; ev.append((i, "open"))
    return ev


def rdp(P, eps):
    """Indices kept by Ramer-Douglas-Peucker on a polyline P (N, 3)."""
    P = np.asarray(P, float)
    n = len(P)
    if n < 3:
        return list(range(n))
    keep = np.zeros(n, bool); keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = P[b] - P[a]
        L = np.linalg.norm(seg)
        Q = P[a + 1:b] - P[a]
        if L < 1e-12:
            d = np.linalg.norm(Q, axis=1)
        else:
            u = seg / L
            d = np.linalg.norm(Q - np.outer(Q @ u, u), axis=1)
        k = int(np.argmax(d))
        if d[k] > eps:
            m = a + 1 + k
            keep[m] = True
            stack += [(a, m), (m, b)]
    return list(np.flatnonzero(keep))


def _smooth(x, w):
    if w <= 1:
        return np.asarray(x, float)
    k = np.ones(int(w)) / int(w)
    return np.convolve(np.asarray(x, float), k, mode="same")


def waypoints(P, dt, rdp_eps=0.005, turn_deg=30.0, halt_speed=0.003, halt_s=0.15, stop_min_move=0.01, merge_s=0.2,
              **_ignored):
    """Waypoint rows of one TCP path P (T, 3): RDP corners turning more than turn_deg, and halts (the TCP slower than
    halt_speed for >= halt_s, between moves of >= stop_min_move): "direction changes > 30 deg or speed minima between
    phases" (COMMON.md). Speed dips inside a straight move (a route of short segments, each accelerating and braking)
    are not waypoints (v2, 04:20: v1 counted minima below 20 % of the neighbouring peaks and inflated the released
    knot routes, e.g. knot medium 54 -> see metrics files). -> dict(rows, corners, stops)."""
    P = np.asarray(P, float)
    T = len(P)
    if T < 3:
        return dict(rows=[], corners=0, stops=0)
    idx = rdp(P, rdp_eps)
    corners = []
    for j in range(1, len(idx) - 1):
        a, b, c = P[idx[j - 1]], P[idx[j]], P[idx[j + 1]]
        u, v = b - a, c - b
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-9 or nv < 1e-9:
            continue
        ang = np.degrees(np.arccos(np.clip(u @ v / (nu * nv), -1, 1)))
        if ang > turn_deg:
            corners.append(int(idx[j]))
    sp = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1) / dt]
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    stops = []
    halts = [(a, b) for a, b in _runs(sp < halt_speed) if (b - a) * dt >= halt_s]
    for a, b in halts:
        if a == 0 or b >= T:
            continue                          # the start / end of the path are not waypoints
        prev_end = max([hb for ha, hb in halts if hb <= a] + [0])
        next_start = min([ha for ha, hb in halts if ha >= b] + [T - 1])
        if arc[a] - arc[prev_end] >= stop_min_move and arc[next_start] - arc[min(b, T - 1)] >= stop_min_move:
            stops.append(int(a))
    rows = sorted(set(corners) | set(stops))
    merged = []
    for r in rows:
        if merged and (r - merged[-1]) * dt < merge_s:
            continue
        merged.append(r)
    return dict(rows=merged, corners=len(corners), stops=len(stops))


def holding_from_record(rec: dict, closed: np.ndarray, *, hold_closed=0.5, fill_gap_rows=2, min_hold_rows=3):
    """(T, 2) bool holding per arm from an antihack.record recording: grasp-assist welds if any, else pinches."""
    held = np.asarray(rec.get("held"), bool) if rec.get("held") is not None else None
    T = len(closed)
    out = np.zeros((T, 2), bool)
    source = "none"
    if held is not None and held.any():
        out = held[:T].astype(bool)
        source = "weld"
    elif rec.get("finger_contacts") is not None:
        fc = np.asarray(rec["finger_contacts"])[:T]
        out = (closed[:T] > hold_closed) & (fc[:, :, 1] >= 2)
        source = "pinch"
    for k in range(2):
        out[:, k] = clean_mask(out[:, k], fill_gap_rows, min_hold_rows)
    return out, source


def classify_grasps(holding: np.ndarray, dt: float, objects=None):
    """Grasp events from per-arm holding masks. objects: optional (T, 2) object ids (same object if equal).
    -> list of {row, arm, event: grasp|release, kind: first|handover|regrasp}"""
    T = len(holding)
    ev = []
    for k, arm in enumerate(ARMS):
        for a, b in _runs(holding[:, k]):
            ev.append(dict(row=int(a), arm=arm, event="grasp", end=int(b)))
            if b < T:
                ev.append(dict(row=int(b), arm=arm, event="release"))
    ev.sort(key=lambda e: (e["row"], e["event"] != "release"))
    held_before = set()
    released_obj = set()
    for e in ev:
        k = ARMS.index(e["arm"])
        obj = "obj" if objects is None else str(objects[min(e["row"], T - 1)][k])
        e["object"] = obj
        if e["event"] == "release":
            released_obj.add(obj)
            continue
        other = 1 - k
        r = e["row"]
        handover = False
        if holding[r, other] and (objects is None or str(objects[r][other]) == obj):
            # the other arm releases before this arm does
            rest = holding[r:, other]
            off = np.flatnonzero(~rest)
            if len(off) and r + off[0] < e["end"]:
                handover = True
        if handover:
            e["kind"] = "handover"
        elif obj in held_before and obj in released_obj:
            e["kind"] = "regrasp"
        else:
            e["kind"] = "first"
        held_before.add(obj)
    return ev


def phases(P_arms, dt, grip_events, grasp_events, *, dwell_speed=0.003, dwell_s=0.3, phase_min_move=0.005,
           holding=None):
    """Motion phases between gripper / grasp events and dwell boundaries. P_arms: (T, 2, 3). -> dict."""
    P = np.asarray(P_arms, float)
    T = len(P)
    sp = np.zeros((T, 2))
    sp[1:] = np.linalg.norm(np.diff(P, axis=0), axis=2) / dt
    still = (sp < dwell_speed).all(axis=1)
    dwell_runs = [(a, b) for a, b in _runs(still) if (b - a) * dt >= dwell_s]
    cuts = {0, T}
    for r, _arm, _e in grip_events:
        cuts.add(int(r))
    for e in grasp_events:
        cuts.add(int(e["row"]))
    for a, b in dwell_runs:
        cuts.add(int(a)); cuts.add(int(b))
    cuts = sorted(c for c in cuts if 0 <= c <= T)
    # merge cuts closer than 0.2 s
    merged = [cuts[0]]
    for c in cuts[1:]:
        if (c - merged[-1]) * dt < 0.2 and c != T:
            continue
        merged.append(c)
    if merged[-1] != T:
        merged.append(T)
    ev_at = {}
    for r, arm, e in grip_events:
        ev_at.setdefault(int(r), []).append((arm, e))
    gr_at = {}
    for e in grasp_events:
        gr_at.setdefault(int(e["row"]), []).append(e)
    out = []
    for a, b in zip(merged[:-1], merged[1:]):
        if b <= a:
            continue
        moved = [float(np.linalg.norm(np.diff(P[a:b + 1 if b < T else b, k], axis=0), axis=1).sum()) for k in range(2)]
        if max(moved) < phase_min_move:
            continue
        movers = [ARMS[k] for k in range(2) if moved[k] >= phase_min_move]
        label = "reposition"
        holds = [bool(holding is not None and holding[a:b, ARMS.index(m)].mean() > 0.5) for m in movers]
        end_events = [x for x in ev_at.get(b, []) if x[0] in movers and x[1] == "close"] + \
                     [x for x in gr_at.get(b, []) if x["arm"] in movers and x["event"] == "grasp"]
        start_rel = [x for x in ev_at.get(a, []) if x[0] in movers and x[1] == "open"] + \
                    [x for x in gr_at.get(a, []) if x["arm"] in movers and x["event"] == "release"]
        if any(holds):
            label = "transport" if sum(holds) == 1 else "bimanual_transport"
        elif end_events:
            label = "approach"
        elif start_rel:
            label = "retreat"
        out.append(dict(start=int(a), end=int(b), t0=round(a * dt, 3), t1=round(b * dt, 3), label=label,
                        arms=movers, moved_m=[round(m, 4) for m in moved]))
    counts = {}
    for p in out:
        counts[p["label"]] = counts.get(p["label"], 0) + 1
    return dict(motion_phases=len(out), labels=counts, segments=out, dwells=len(dwell_runs),
                dwell_s=round(sum(b - a for a, b in dwell_runs) * dt, 3))


# ------------------------------------------------------------------------------------------------ main entry points
def metrics_from_paths(P_arms, dt, *, closedness=None, holding=None, q_arms=None, objects=None, grasp_source="",
                       first_success_row=None, **kw) -> dict:
    """The trajectory-complexity block from TCP paths (T, 2, 3) (+ commanded closedness (T, 2), holding (T, 2),
    arm joints (T, 2, 6))."""
    o = {**DEFAULTS, **kw}
    P = np.asarray(P_arms, float)
    T = len(P)
    grip = []
    if closedness is not None:
        for k, arm in enumerate(ARMS):
            grip += [(r, arm, e) for r, e in gripper_events(closedness[:, k], o["close_th"], o["open_th"])]
    grip.sort()
    if holding is None and closedness is not None:
        # no physics: a closed gripper holds (plans: the planner's own grasp events are used instead when given)
        holding = np.stack([clean_mask(closedness[:, k] > o["hold_closed"], o["fill_gap_rows"], o["min_hold_rows"])
                            for k in range(2)], axis=1)
        grasp_source = grasp_source or "commanded_closure"
    gev = classify_grasps(holding, dt, objects) if holding is not None else []
    ph = phases(P, dt, grip, gev, dwell_speed=o["dwell_speed"], dwell_s=o["dwell_s"],
                phase_min_move=o["phase_min_move"], holding=holding)
    per_arm = {}
    wp_total = 0
    wp_coarse = 0
    for k, arm in enumerate(ARMS):
        wc = waypoints(P[:, k], dt, 2 * o["rdp_eps"], o["turn_deg"], o["halt_speed"], o["halt_s"], o["stop_min_move"],
                       o["merge_s"])
        wp_coarse += len(wc["rows"])
        seg = np.linalg.norm(np.diff(P[:, k], axis=0), axis=1)
        w = waypoints(P[:, k], dt, o["rdp_eps"], o["turn_deg"], o["halt_speed"], o["halt_s"], o["stop_min_move"],
                      o["merge_s"])
        wp_total += len(w["rows"])
        rec = dict(tcp_path_m=round(float(seg.sum()), 4), waypoints=len(w["rows"]), waypoint_rows=w["rows"],
                   corners=w["corners"], stops=w["stops"],
                   peak_tcp_speed_m_s=round(float(seg.max() / dt) if len(seg) else 0.0, 4),
                   gripper_closes=sum(1 for g in grip if g[1] == arm and g[2] == "close"),
                   grasps=sum(1 for e in gev if e["arm"] == arm and e["event"] == "grasp"))
        if q_arms is not None:
            dq = np.diff(np.asarray(q_arms, float)[:, k], axis=0)
            rec["joint_path_rad"] = round(float(np.linalg.norm(dq, axis=1).sum()), 4)
            rec["joint_path_l1_rad"] = round(float(np.abs(dq).sum()), 4)
        per_arm[arm] = rec
    kinds = [e.get("kind") for e in gev if e["event"] == "grasp"]
    n_grasp = len(kinds)
    n_rel = sum(1 for e in gev if e["event"] == "release")
    out = dict(
        rows=int(T), control_dt_s=dt, duration_s=round(T * dt, 3),
        duration_to_success_s=None if first_success_row is None else round(first_success_row * dt, 3),
        motion_phases=ph["motion_phases"], phase_labels=ph["labels"],
        phases_total=ph["motion_phases"] + n_grasp + n_rel,
        waypoints_total=wp_total,
        waypoints_total_coarse=wp_coarse,       # the same with a 10 mm RDP tolerance (ignores ~5-10 mm wiggles)
        tcp_path_m_total=round(sum(a["tcp_path_m"] for a in per_arm.values()), 4),
        joint_path_rad_total=(round(sum(a.get("joint_path_rad", 0) for a in per_arm.values()), 4)
                              if q_arms is not None else None),
        grasp_count=n_grasp, release_count=n_rel,
        regrasp_count=sum(1 for k in kinds if k == "regrasp"),
        handover_count=sum(1 for k in kinds if k == "handover"),
        first_grasps=sum(1 for k in kinds if k == "first"),
        arms_used=sum(1 for a in per_arm.values() if a["tcp_path_m"] > 0.02),
        bimanual_overlap_s=round(float((holding.all(axis=1)).sum() * dt), 3) if holding is not None else None,
        gripper_events=len(grip), dwells=ph["dwells"], dwell_s=ph["dwell_s"],
        grasp_source=grasp_source, per_arm=per_arm, grasp_events=gev, phases=ph["segments"], thresholds=o)
    return out


def metrics_from_rows(rows, layout, dt, fk, *, record: dict | None = None, use_measured_tcp: bool = False, **kw) -> dict:
    """(a) recorded rows: FK of the commanded arm joints (or the recording's measured TCP), commanded gripper events,
    holdings from the recording (weld / pinch) when given, else from the commanded closure."""
    rows = np.asarray(rows, float)
    cols = layout_columns(layout)
    if record is not None and use_measured_tcp and record.get("tcp") is not None:
        P = np.asarray(record["tcp"], float)[:, :, :3]
        src = "measured_tcp"
    else:
        P, _ = fk.tcp_path(rows, layout)
        src = "fk_of_commanded_rows"
    T = min(len(P), len(rows))
    if record is not None and record.get("targets") is not None:
        T = min(T, len(record["targets"]))       # the episode may have ended before the last row
    P = P[:T]
    closed = np.stack([gripper_closedness(rows[:T, cols[arm][1]], cols[arm][2]) if cols[arm][1] is not None
                       else np.zeros(T) for arm in ARMS], axis=1)
    q = np.stack([rows[:T, cols[arm][0]] for arm in ARMS], axis=1)
    holding, source = (None, "")
    if record is not None:
        holding, source = holding_from_record(record, closed, hold_closed=kw.get("hold_closed", 0.5),
                                              fill_gap_rows=kw.get("fill_gap_rows", 2),
                                              min_hold_rows=kw.get("min_hold_rows", 3))
        if source == "none":
            holding = None
    m = metrics_from_paths(P, dt, closedness=closed, holding=holding, q_arms=q, grasp_source=source,
                           first_success_row=(record or {}).get("first_success_row"), **kw)
    m["tcp_source"] = src
    return m


def densify_plan(plan: dict, fk=None):
    """A planned oracle -> (P (T, 2, 3), closedness (T, 2), holding (T, 2) or None, q (T, 2, 6) or None, dt).

    plan = {"control_dt_s": 0.04,
            "arms": {"left": {"tcp": [[t, x, y, z], ...]} | {"joints": [[t, q1..q6], ...]}, "right": {...}},
            "events": [{"t": 1.2, "arm": "left", "event": "grasp" | "release" | "close" | "open",
                        "object": "rope"}, ...]}
    Waypoints are linearly interpolated at the control period (an arm missing from "arms" stays still); "close" /
    "open" set the commanded closure, "grasp" / "release" the holding (and imply close / open)."""
    dt = float(plan.get("control_dt_s", 0.04))
    arms = plan.get("arms", {})
    t_end = 0.0
    for a in arms.values():
        for key in ("tcp", "joints"):
            if a.get(key):
                t_end = max(t_end, float(np.asarray(a[key], float)[:, 0].max()))
    for e in plan.get("events", []):
        t_end = max(t_end, float(e["t"]))
    T = int(round(t_end / dt)) + 1
    ts = np.arange(T) * dt
    P = np.zeros((T, 2, 3))
    q = None
    for k, arm in enumerate(ARMS):
        a = arms.get(arm, {})
        if a.get("tcp"):
            W = np.asarray(a["tcp"], float)
            for j in range(3):
                P[:, k, j] = np.interp(ts, W[:, 0], W[:, 1 + j])
        elif a.get("joints"):
            if fk is None:
                raise ValueError("joint waypoints need an AlohaFK")
            W = np.asarray(a["joints"], float)
            qa = np.stack([np.interp(ts, W[:, 0], W[:, 1 + j]) for j in range(6)], axis=1)
            q = np.zeros((T, 2, 6)) if q is None else q
            q[:, k] = qa
            for t in range(T):
                P[t, k] = fk.tcp({arm: qa[t]})[arm][0]
    closed = np.zeros((T, 2))
    hold = np.zeros((T, 2), bool)
    objects = np.full((T, 2), "", dtype=object)
    have_hold = False
    for e in sorted(plan.get("events", []), key=lambda e: float(e["t"])):
        k = ARMS.index(e["arm"])
        r = int(round(float(e["t"]) / dt))
        if e["event"] in ("close", "grasp"):
            closed[r:, k] = 1.0
        if e["event"] in ("open", "release"):
            closed[r:, k] = 0.0
        if e["event"] == "grasp":
            hold[r:, k] = True; have_hold = True
            objects[r:, k] = str(e.get("object", "obj"))
        if e["event"] == "release":
            hold[r:, k] = False
    return P, closed, (hold if have_hold else None), q, dt, (objects if have_hold else None)


def metrics_from_plan(plan: dict, fk=None, **kw) -> dict:
    """(b) a planned oracle (see densify_plan)."""
    P, closed, hold, q, dt, objects = densify_plan(plan, fk)
    m = metrics_from_paths(P, dt, closedness=closed, holding=hold, q_arms=q, objects=objects,
                           grasp_source="plan_events" if hold is not None else "", **kw)
    m["tcp_source"] = "plan"
    return m


SUMMARY_KEYS = ("rows", "duration_s", "duration_to_success_s", "motion_phases", "phases_total", "waypoints_total",
                "waypoints_total_coarse",
                "tcp_path_m_total", "joint_path_rad_total", "grasp_count", "release_count", "regrasp_count",
                "handover_count", "first_grasps", "arms_used", "bimanual_overlap_s", "gripper_events", "dwells",
                "phase_labels", "grasp_source", "tcp_source")


def summary(m: dict) -> dict:
    out = {k: m.get(k) for k in SUMMARY_KEYS}
    out["per_arm"] = {arm: {k: v for k, v in a.items() if k != "waypoint_rows"} for arm, a in m.get("per_arm", {}).items()}
    return out
