"""The splice's measurements from the two wires' centreline nodes (the design's v5_criteria.py, README_v5 section 5).

Nodes: the origin of every link of a wire (from its terminal) and the end of its last link. Wire A's and B's strip
nodes are their first bare nodes (index n_ins).

eye_loop(P, i0)        A's eye on its bare nodes P[i0:]: the stem direction t_s (mean of the first 8 bare links), the
                       turning angle of every link against t_s; the eye runs from the last stem node with angle < 30 deg
                       before the turn (k1) to the first node with angle > 150 deg after it (k2); then closed across the
                       neck: the stem node i <= k1 and tail node j >= k2 (each within 12 nodes) that are closest (with
                       first_touch: of the pairs within 0.5 mm of the closest, the one nearest the bend), loop =
                       A[i..j] plus the segment j -> i. None when the bare wire never turns past 150 deg.
piercing(loop, Q)      signed number of times polyline Q passes through the loop: the linking of B with A's closed eye.
                       +-1: B threads the eye once.
winding(Q, a0, ahat)   turns of polyline Q about the line (a0, ahat): signed angle of each node about the line,
                       unwrapped, summed over consecutive nodes, / 2 pi.
around_both(...)       turns of B's coil round both of A's strands together (user decision 2026-10-22): about each
                       strand's own centreline, the best contiguous run of the coil, the lesser of its two counts.
meter_reading(m, checks)  the splice meter's reading (resistance model, needle position) from one measurement.
"""
import numpy as np

from topogym.assets.geometry import wire_splicing as geo

BACK, STEM = 150., 30.        # "turned back": more than 150 deg from the stem direction; stem links < 30 deg


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def eye_loop(P, i0, n_stem=8, first_touch=False, back=170., stem=10.):
    Q = np.asarray(P[i0:], float)
    t = _unit(np.diff(Q, axis=0))
    ts = _unit(t[:n_stem].mean(axis=0))
    ang = np.degrees(np.arccos(np.clip(t @ ts, -1, 1)))
    over = np.flatnonzero(ang > back)
    if not len(over):
        return None
    k2 = int(over[0]) + 1                      # node after the first link past `back` deg
    before = np.flatnonzero(ang[:over[0]] < stem)
    if not len(before):
        return None
    k1 = int(before[-1]) + 1                   # node after the last stem link
    # close the eye where the two legs come closest below the bend (the neck), within 12 nodes of the bend
    best, i_n, j_n = np.inf, k1, k2
    for di in range(0, 13):
        i = k1 - di
        if i < 1:
            break
        for dj in range(0, 13):
            j = k2 + dj
            if j >= len(Q):
                break
            dd = np.linalg.norm(Q[i] - Q[j])
            if dd < best - 1e-9:
                best, i_n, j_n = dd, i, j
    if first_touch:
        # once a coil squeezes the legs together along a stretch, "closest" is ambiguous: take the touching pair
        # (gap <= 0.5 mm over the closest one) nearest the bend
        best_sum = np.inf
        for di in range(0, 13):
            i = k1 - di
            if i < 1:
                break
            for dj in range(0, 13):
                j = k2 + dj
                if j >= len(Q):
                    break
                if np.linalg.norm(Q[i] - Q[j]) <= best + 0.0005 and di + dj < best_sum:
                    best_sum, i_n, j_n = di + dj, i, j
    loop = Q[i_n:j_n + 1]
    return np.vstack([loop, loop[:1]]), i0 + i_n, i0 + j_n


def piercing(loop, Q):
    """Signed number of times polyline Q passes through the closed loop: the loop's best-fit plane (centroid, normal by
    SVD); for every segment of Q that crosses the plane, the winding number of the projected loop around the crossing
    point (angle sum / 2 pi, rounded), signed by the crossing direction. The sign is fixed by the loop's node order and
    Q's direction (the in-plane axes and the normal form a right-handed frame)."""
    L = np.asarray(loop[:-1], float)
    o = L.mean(axis=0)
    _, _, vt = np.linalg.svd(L - o)
    e1, e2 = vt[0], vt[1]
    n = np.cross(e1, e2)
    dist = (np.asarray(Q) - o) @ n
    total = 0
    P2 = np.c_[(L - o) @ e1, (L - o) @ e2]
    for i in range(len(Q) - 1):
        a, b = dist[i], dist[i + 1]
        if not ((a < 0 <= b) or (b < 0 <= a)):       # half-open: a node on the plane counts once
            continue
        x = Q[i] + (a / (a - b)) * (Q[i + 1] - Q[i])
        p = np.array([(x - o) @ e1, (x - o) @ e2])
        v = P2 - p
        ang = np.arctan2(v[:, 1], v[:, 0])
        dth = np.diff(np.r_[ang, ang[:1]])
        dth = (dth + np.pi) % (2 * np.pi) - np.pi
        w = int(round(dth.sum() / (2 * np.pi)))
        total += w * (1 if b > a else -1)
    return total


def axis_frame(ahat):
    ahat = ahat / np.linalg.norm(ahat)
    e1 = np.cross(ahat, [0, 0, 1.]) if abs(ahat[2]) < 0.9 else np.cross(ahat, [1., 0, 0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(ahat, e1)


def cyl(Q, a0, ahat):
    """Axial coordinate, radius, angle of points Q about the line (a0, ahat)."""
    e1, e2 = axis_frame(ahat)
    r = Q - a0
    z = r @ ahat
    rp = r - np.outer(z, ahat)
    return z, np.linalg.norm(rp, axis=1), np.arctan2(rp @ e2, rp @ e1)


def winding(Q, a0, ahat):
    _, _, th = cyl(Q, a0, ahat)
    dth = np.diff(th)
    dth = (dth + np.pi) % (2 * np.pi) - np.pi
    return float(dth.sum() / (2 * np.pi))


BESIDE_D = 2.5                # a tail node lies beside the stem within this many core diameters of the stem's own line


def core_line(PA, iA, k1, k2):
    """A's doubled section: the principal line of the stem nodes (strip .. k1) together with the tail nodes from k2 on
    that lie beside the stem (within BESIDE_D core diameters of the stem nodes' own principal line: the run from k2 while
    they do, at most as many as the stem's), i.e. the line midway between the two legs where they lie together;
    direction from the eye toward the strip. (The tail past the coil is left out: with the 1.5-turn coil 25 mm of A's
    return leg stays outside it; opened into a V it tilted the line 24 deg and, folded back, took 9 of the 13 fitted tail
    nodes to the far side, the line tilted 17-45 deg and a 1.96-turn coil read 0.05-0.21 turns. With the return leg
    beside the stem along the whole coil, as at 6.5 turns, both fits give the same line.)"""
    stem = np.asarray(PA[iA:k1 + 1], float)
    cs = stem.mean(axis=0)
    _, _, vt = np.linalg.svd(stem - cs)
    a_s = vt[0]
    tail = []
    for k in range(k2, min(k2 + (k1 - iA) + 1, len(PA))):
        q = PA[k] - cs
        if np.linalg.norm(q - (q @ a_s) * a_s) > BESIDE_D * geo.CORE_D:
            break
        tail.append(k)
    pts = np.vstack([stem, PA[tail]]) if tail else stem
    c0 = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c0)
    ahat = vt[0] if np.dot(vt[0], PA[iA] - PA[k1]) > 0 else -vt[0]
    return c0, ahat


def around_both(PA, iA, k1, k2, c0, ahat, Q):
    """Turns of polyline Q (B's coil) round both of A's strands together (user decision 2026-10-22: the coil must go
    round the two strands of A's doubled end at least once, not only round the line between them). A's strands: the
    stem side, A[iA .. apex], and the return side, A[apex .. its node farthest toward the strip] (apex: the eye node
    nearest the eye end of the axis; a tail folded back from that far node is not one of the two strands). For every node
    of Q, each strand's point at the node's station along the axis (where the strand crosses the plane through the node
    perpendicular to the axis; of several, the one nearest the node; none where the strand does not reach), and the
    node's angle about that point in the plane; the angles accumulate along Q (each step wrapped into (-pi, pi]), a step
    counting only where both strands have their points at both ends. A loop round one strand only, or one that passes
    between the strands, gains on one count and not the other. Returns the largest, over the contiguous runs of Q, of
    the lesser of the run's two counts, taken in the sense Q winds about the axis, and that run's node indices in Q."""
    Q = np.asarray(Q, float)
    if len(Q) < 2:
        return 0., None
    e1, e2 = axis_frame(ahat)
    zA = (PA - c0) @ ahat
    ka = k1 + int(np.argmin(zA[k1:k2 + 1]))
    kf = k2 + int(np.argmax(zA[k2:]))
    zq = (Q - c0) @ ahat
    n = np.arange(len(Q))
    th = np.full((2, len(Q)), np.nan)
    for s, S in enumerate((PA[iA:ka + 1], PA[ka:kf + 1])):
        zs = (S - c0) @ ahat
        a, b = zs[:-1, None] - zq[None], zs[1:, None] - zq[None]               # (strand segments, nodes of Q)
        hit = (a != b) & (((a <= 0) & (b >= 0)) | ((b <= 0) & (a >= 0)))
        t = np.where(hit, a / np.where(hit, a - b, 1.), 0.)
        P = S[:-1, None] + t[..., None] * (S[1:] - S[:-1])[:, None]
        dist = np.where(hit, np.linalg.norm(P - Q[None], axis=2), np.inf)
        j = dist.argmin(axis=0)
        v = Q - P[j, n]
        th[s] = np.where(np.isfinite(dist[j, n]), np.arctan2(v @ e2, v @ e1), np.nan)
    step = (np.diff(th, axis=1) + np.pi) % (2 * np.pi) - np.pi
    both = np.isfinite(step).all(axis=0)
    W = np.c_[np.zeros(2), np.cumsum(np.where(both, step, 0.), axis=1) / (2 * np.pi)]
    sign = np.sign(winding(Q, c0, ahat)) or 1.
    lesser = np.triu(np.minimum(*(sign * (w[None, :] - w[:, None]) for w in W)), 1)     # [a, b]: run a .. b
    a, b = np.unravel_index(int(np.argmax(lesser)), lesser.shape)
    best = float(lesser[a, b])
    return (best, (int(a), int(b))) if best > 0. else (0., None)


def measure(PA, PB, iA, iB, D):
    """All the splice's quantities. PA, PB: node arrays of wires A and B; iA, iB: their strip node indices; D: core
    diameter."""
    out = dict(eye=False)
    e = eye_loop(PA, iA, first_touch=True, back=BACK, stem=STEM)
    if e is None:
        return out
    loop, k1, k2 = e
    out.update(eye=True, eye_k1=k1, eye_k2=k2,
               eye_inner_mm=float(1000 * (np.linalg.norm(PA[k1] - PA[k2]) - D)),
               neck_gap_mm=float(1000 * (np.linalg.norm(PA[k1 - 4] - PA[min(k2 + 4, len(PA) - 1)]) - D)))
    out['pierce'] = int(piercing(loop, PB[iB:]))
    c0, ahat = core_line(PA, iA, k1, k2)
    z_eye, z_strip = float((PA[k1] - c0) @ ahat), float((PA[iA] - c0) @ ahat)
    # B's coil: the longest run of B's bare nodes within 2.5 D of the axis and between the eye and the strip (+- 1 D)
    QB = PB[iB:]
    zb, rb, _ = cyl(QB, c0, ahat)
    inside = (rb <= 2.5 * D) & (zb >= z_eye - D) & (zb <= z_strip + D)
    runs, cur = [], []
    for i, ok in enumerate(inside):
        if ok:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    run = max(runs, key=len) if runs else []
    if len(run) >= 2:
        coil = QB[run]
        both, ab = around_both(PA, iA, k1, k2, c0, ahat, coil)
        out.update(wrap_turns=abs(winding(coil, c0, ahat)), coil_nodes=len(run), both_turns=both,
                   both_run=None if ab is None else (iB + run[ab[0]], iB + run[ab[1]]),
                   coil_mean_r_D=float(rb[run].mean() / D), coil_len_mm=float(1000 * (zb[run].max() - zb[run].min())),
                   coil_z_from_eye_mm=(float(1000 * (zb[run].min() - z_eye)), float(1000 * (zb[run].max() - z_eye))),
                   stem_len_mm=float(1000 * (z_strip - z_eye)))
    else:
        out.update(wrap_turns=0., coil_nodes=0, both_turns=0.)
    # A's tail end: direction of its last 3 links against the axis (+1 = toward the strip, -1 = folded back toward the
    # eye) and its radius from the axis
    tdir = _unit(PA[-1] - PA[-4])
    zt, rt, _ = cyl(PA[-1:], c0, ahat)
    out.update(tail_dir=float(tdir @ ahat), tail_r_D=float(rt[0] / D), tail_z_from_eye_mm=float(1000 * (zt[0] - z_eye)))
    return out


EDGE_GAP = .01                # a needle kept out of the band stops this far (dial fraction, 1.2 deg) below its edge


def meter_reading(m, checks):
    """The splice meter's reading (user decision 2026-10-19): a documented model, the splice's equivalent resistance
    computed from its configuration (m: measure(); checks: shape_checks() of it), not simulated physics.
      no path      unless A's eye is closed at the neck, B threads it once (linking number +-1) and B's coil lies on A's
                   doubled bare section (on_bare_core): resistance_mohm None, the needle at its rest stop;
      otherwise    R = R_TURN_MOHM / n, n = the coil's turns round both of A's strands together (both_turns: the
                   Round-both condition's count, user decision 2026-10-22); n = 0 reads as no path;
      in_band      every shape condition holds (eye, threading, wraps, round both, tight, on the bare core, tail
                   folded): the instantaneous success predicate less the quiet and hands-off conditions. Out of the
                   band the needle stops EDGE_GAP below the band's edge at most, so the needle is in the red band
                   exactly when in_band.
    dial_position: 0 at the dial's 100 mOhm end, 1 at its 1 mOhm end, clamped to the scale (a path above 100 mOhm reads 0;
    the rest stop is geo.METER rest_u); needle_rad: the needle hinge's position (geo.needle_q)."""
    conducting = bool(checks.get("eye") and checks.get("pierce") and checks.get("on_bare_core"))
    turns = float(m.get("both_turns", 0.)) if conducting else 0.
    resistance = geo.R_TURN_MOHM / turns if turns > 0. else None
    in_band = bool(checks) and all(checks.values())
    edge = geo.dial_position(geo.R_TURN_MOHM / geo.BAND_TURNS)
    if resistance is None:
        u = geo.METER["rest_u"]
    else:
        u = min(max(geo.dial_position(resistance), 0.), 1.)
        u = max(u, edge) if in_band else min(u, edge - EDGE_GAP)
    return dict(conducting=resistance is not None, resistance_mohm=resistance, turns=turns, in_band=in_band,
                dial_position=float(u), needle_rad=geo.needle_q(u))


def shape_checks(m, D, *, min_turns, min_both_turns, max_coil_r_D, tail_dot, min_tail_r_D):
    """The conditions on one measurement (judge's, without the hold, quiet and hands-off ones)."""
    if not m.get('eye'):
        return dict(eye=False)
    return dict(
        eye=True,
        pierce=abs(m.get('pierce', 0)) == 1,
        wrap=m.get('wrap_turns', 0) >= min_turns,
        round_both=m.get('both_turns', 0) >= min_both_turns,
        tight=m.get('coil_mean_r_D', 9) <= max_coil_r_D,
        on_bare_core=m.get('coil_nodes', 0) > 0 and m['coil_z_from_eye_mm'][0] >= -1000 * D
        and m['coil_z_from_eye_mm'][1] <= m['stem_len_mm'] + 1000 * D,
        tail_folded=m.get('tail_dir', 1) <= -tail_dot and m.get('tail_r_D', 0) >= min_tail_r_D)
