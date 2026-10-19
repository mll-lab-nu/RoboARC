"""Single-grasp reachability and path planning along the wire (the anti-hack argument and the oracle's held segments).

For one grasp (a fixed TCP pose in the wand frame), the held wand must follow the wire: its ring centre on the wire,
its normal along the tangent (or tilted by a lag about the bending axis), the handle at an angle psi about the tangent
(free, but continuous). A frontier search over arc length s keeps every (psi, lag, joint configuration) node that is
reachable from the start by small steps:
- IK converged (TCP within 0.2 mm / 2 mrad), joints within the Panda's limits;
- no robot geom within `clear_robot` of a static body (wire, sleeves, deck, obstacles), no robot self-collision;
- the wand's metal clear of the wire metal by `clear_wand`, the wand not penetrating anything else;
- consecutive nodes: psi and lag change by at most one grid step, every joint by at most `dq_max`.
The frontier is pruned to a few nodes per (psi, lag) bin (diverse joint branches). `explore` returns the furthest s
reached and the feasible (s, psi) map; with `keep=True` it keeps the layers and back-pointers, and `extract` returns a
path ending in a chosen psi. A sampling argument (labelled as such), not a proof.
"""
from __future__ import annotations

import numpy as np

from .wandpath import tcp_from_wand


def _is(name, *keys):
    return any(k in name for k in keys)


def classify(kin):
    """Cache geom classes by name: wire metal, wand metal (loop / stem), sleeves."""
    if getattr(kin, "_cls", None) is None:
        names = [kin.geom_name(g) for g in range(kin.m.ngeom)]
        kin._cls = dict(
            wire_metal=np.array([_is(n, "wire_metal", "wire::metal") for n in names]),
            wand_metal=np.array([_is(n, "wand_metal", "wand::loop", "wand::stem") for n in names]))
    return kin._cls


def wire_arc_map(kin, path):
    """Arc length (m) along `path` of every wire-metal geom (its centre's nearest rail point), cached on kin."""
    if getattr(kin, "_wire_s", None) is None:
        import mujoco
        c = classify(kin)
        mujoco.mj_kinematics(kin.m, kin.d)
        ids = np.flatnonzero(c["wire_metal"])
        out = {}
        for g in ids:
            out[int(g)] = float(path.s[int(np.argmin(np.linalg.norm(path.p - kin.d.geom_xpos[g], axis=1)))])
        kin._wire_s = out
    return kin._wire_s


def node_ok(kin, q, wand_pose, *, clear_robot=0.003, clear_wand=0.0003, clear_other=0.002, wand_soft=-0.0005,
            s_ring=None, local=0.04, deck_ok=False):
    """clear_wand: the ring's own stretch of wire (it is centred on it); clear_other: every other branch of the wire
    (more than `local` m of arc away, when s_ring and kin._wire_s are known): the live runs' execution error is 1-2.5 mm."""
    c = classify(kin)
    ws = getattr(kin, "_wire_s", None)
    for dist, kind, g1, g2 in kin.contacts(q, wand=wand_pose):
        if kind == "robot-static" and dist < clear_robot:
            nm = kin.geom_name(g1) + "|" + kin.geom_name(g2)
            if deck_ok and ("deck" in nm or "finish_support" in nm):
                continue                     # the pads on the deck at the deck pick (the deck pushes the hand up)
            return False, "robot-static " + nm
        if kind == "robot-robot" and dist < 0:
            return False, "robot-robot"
        if kind == "wand-static":
            metal = (c["wand_metal"][g1] and c["wire_metal"][g2]) or (c["wand_metal"][g2] and c["wire_metal"][g1])
            if metal:
                gw = g2 if c["wire_metal"][g2] else g1
                other = s_ring is not None and ws is not None and abs(ws.get(int(gw), s_ring) - s_ring) > local
                if dist < (clear_other if other else clear_wand):
                    return False, "wand-wire"
            if dist < wand_soft:
                return False, "wand-static"
    return True, ""


def explore(kin, path, g, *, s0=0.0, s1=None, ds=0.004, psi_n=24, seeds=(), dq_max=0.35, per_bin=3, max_nodes=400,
            clear_robot=0.003, log=None, start_psis=None, lags=(0.0,), start_lags=None, start_qs=None, keep=False,
            start_iters=300, prior_w=0.0):
    """Frontier search; a node is (psi index, lag index, q). `lags`: ring tilts (rad) about the bending axis.
    start_qs: optional explicit start configurations [(psi index, lag index, q)] (e.g. the measured arm)."""
    s1 = path.length if s1 is None else s1
    psis = np.arange(psi_n) * 2 * np.pi / psi_n
    lags = np.asarray(lags, float)
    n = max(2, int(round((s1 - s0) / ds)) + 1)
    S = np.linspace(s0, s1, n)
    feas = np.zeros((len(S), psi_n), np.int8)
    wire_arc_map(kin, path)
    whys = {}
    frontier = []
    if start_qs is not None:
        frontier = [(j, l, np.asarray(q, float), -1) for j, l, q in start_qs]
    else:
        for j in (range(psi_n) if start_psis is None else start_psis):
            for l in (range(len(lags)) if start_lags is None else start_lags):
                p, r = path.frame(S[0], psis[j], lags[l])
                tp, tr = tcp_from_wand(p, r, g)
                for seed in seeds:
                    q, ok, ep, er = kin.ik(tp, tr, seed, iters=start_iters)
                    if not ok:
                        continue
                    good, why = node_ok(kin, q, (p, r), clear_robot=clear_robot, s_ring=float(S[0]), deck_ok=S[0] < 0.02)
                    if good and not any(jj == j and ll == l and np.abs(q - qq).max() < 0.2 for jj, ll, qq, _ in frontier):
                        frontier.append((j, l, q, -1))
    layers = [frontier] if keep else None
    for j, l, _, _ in frontier:
        feas[0, j] = 1
    reached = 0 if frontier else -1
    for i in range(1, len(S)):
        new = []
        for pi, (j, l, q, _) in enumerate(frontier):
            for dj in (0, -1, 1):
                jj = (j + dj) % psi_n
                for dl in ((0, -1, 1) if len(lags) > 1 else (0,)):
                    ll = l + dl
                    if ll < 0 or ll >= len(lags):
                        continue
                    if sum(1 for a, b2, _, _ in new if a == jj and b2 == ll) >= per_bin:
                        continue
                    p, r = path.frame(S[i], psis[jj], lags[ll])
                    tp, tr = tcp_from_wand(p, r, g)
                    if prior_w:
                        q2, ok, ep, er = kin.ik(tp, tr, q, iters=80, q_prior=0.5 * (kin.lo + kin.hi), prior_w=prior_w)
                    else:
                        q2, ok, ep, er = kin.ik(tp, tr, q, iters=60)
                    if not ok or np.abs(q2 - q).max() > dq_max:
                        whys["ik" if not ok else "joint-jump"] = whys.get("ik" if not ok else "joint-jump", 0) + 1
                        continue
                    if any(a == jj and b2 == ll and np.abs(q2 - b).max() < 0.15 for a, b2, b, _ in new):
                        continue
                    good, why = node_ok(kin, q2, (p, r), clear_robot=clear_robot, s_ring=float(S[i]), deck_ok=S[i] < 0.02)
                    if not good:
                        whys[why] = whys.get(why, 0) + 1
                    if good:
                        new.append((jj, ll, q2, pi))
        if len(new) > max_nodes:
            new = new[:: int(np.ceil(len(new) / max_nodes))]
        if not new:
            break
        frontier = new
        if keep:
            layers.append(frontier)
        for j, l, _, _ in frontier:
            feas[i, j] = 1
        reached = i
        if log and i % 25 == 0:
            log(dict(s=round(float(S[i]), 4), nodes=len(frontier)))
    return dict(whys=whys, s=S, psis=psis, lags=lags, feas=feas, reached_s=float(S[reached]) if reached >= 0 else None,
                complete=bool(reached == len(S) - 1), final=frontier, layers=layers)


def extract(res, end_psi=None, end_lag=None, prefer_q=None, psi_tol=0):
    """A path (list of (s, psi, lag, q)) from the last layer back to the start, ending within psi_tol bins of psi
    index end_psi (None: any) and at lag index end_lag (None: any); the closest psi first."""
    layers = res["layers"]
    if not layers or not res["complete"]:
        return None
    last = layers[-1]
    n = len(res["psis"])

    def dpsi(j):
        return 0 if end_psi is None else min((j - end_psi) % n, (end_psi - j) % n)
    cand = [k for k, (j, l, q, _) in enumerate(last)
            if dpsi(j) <= psi_tol and (end_lag is None or l == end_lag)]
    if not cand:
        return None
    cand.sort(key=lambda k: (dpsi(last[k][0]), 0 if prefer_q is None else np.abs(last[k][2] - prefer_q).max()))
    k = cand[0]
    out = []
    for i in range(len(layers) - 1, -1, -1):
        j, l, q, parent = layers[i][k]
        out.append((float(res["s"][i]), float(res["psis"][j]), float(res["lags"][l]), q))
        k = parent
    return out[::-1]
