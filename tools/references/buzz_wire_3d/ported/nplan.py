"""Offline nudge planner (parallel). Same kinematics as ported_oracle.nudged_route, split in stages:
  prep  <instance> <route.npz> <cut> <shift_dx> <out.npz>        replay the route to the grasp (live), save g, qpos, nodes
  table <prep.npz> <k> <K> <out.npz> [--dmax 60 --dstep 5]        feasibility of nodes k::K for every delta
  dp    <prep.npz> <table_glob> <plan.npz> [--rclear 0.002 --jump 0.3 --wclear 0.0003]   merge + DP -> joint path
"""
import sys, os, json, glob, time
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))


def prep(inst, route, cut, dx, out, s_stop=0.10, node_mm=2.0, ext_back=None):
    from rcb_topogym import Session
    from topogym.planning.buzz_wire_3d.redesign.kin import PandaKin
    from topogym.planning.buzz_wire_3d.redesign.wandpath import WirePath, wand_from_tcp
    s = Session("buzz_wire_3d", instance=inst, strict=True)
    w = s.env._world; m, d = w.model, w.data
    kin = PandaKin(m)
    rows = np.load(route)["actions"].astype(np.float32)
    pre = rows[:cut].copy()
    rel = 600 + int(np.flatnonzero(pre[600:, 7] < 0.4)[0])
    wand, hand = m.body("wand::wand").id, m.body("hand").id
    qadr = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 8)]
    from topogym.planning.standard_rate import RateLimiter
    lim = RateLimiter(s, mode="spread")
    v = None
    for i in range(cut):
        if dx and i == rel + 40:
            Rw_ = R.from_quat(np.array(d.xquat[wand])[[1, 2, 3, 0]])
            v = Rw_.apply([dx, 0, 0]); q_seed = np.array(d.qpos[qadr])
        if v is not None:
            f = min(1.0, (i - rel - 40) / 60)
            tp_, tr_ = kin.fk(pre[i, :7].astype(float))
            q_, ok_, _, _ = kin.ik(tp_ + f * v, tr_, pre[i, :7].astype(float), iters=300)   # seeded by the route's own row
            if not ok_ or np.abs(q_ - q_seed).max() > 0.1:
                q_, ok_, _, _ = kin.ik(tp_ + f * v, tr_, q_seed, iters=300)
            assert ok_, i
            pre[i, :7] = q_; q_seed = q_
        lim.step(pre[i])
    Rh = R.from_matrix(d.xmat[hand].reshape(3, 3)); tcp = d.xpos[hand] + Rh.apply([0, 0, .1029])
    Rw = R.from_quat(np.array(d.xquat[wand])[[1, 2, 3, 0]]); pw = np.array(d.xpos[wand])
    g = (Rw.inv().apply(tcp - pw), Rw.inv() * Rh)
    g_route = (g[0] - np.array([dx, 0, 0]), g[1])
    path = WirePath(s.env._task.rail)
    nodes, last, s_prev = [], None, 0.0
    for k, row in enumerate(rows[cut:]):
        if row[7] < 0.4:                                     # the route lets go: its traversal ends here
            break
        qr = np.asarray(row[:7], float)
        tp, tr = kin.fk(qr)
        wp, wr = wand_from_tcp(tp, tr, g_route)
        lo, hi = max(0, path.idx(s_prev - 0.03)), min(len(path.s), path.idx(s_prev + 0.03) + 1)
        i = lo + int(np.argmin(np.linalg.norm(path.p[lo:hi] - wp, axis=1))); s_prev = float(path.s[i])
        if last is not None and np.linalg.norm(wp - last[0]) < 1e-3 * node_mm and (wr * last[1].inv()).magnitude() < np.deg2rad(3):
            continue
        nodes.append(np.r_[k, s_prev, qr, wp, wr.as_quat(), path.t[i], path.b[i]])
        last = (wp, wr)
        if s_prev >= path.length - s_stop:
            break
    # extension: from the route's last ring position down the wire to its end (ext_back m before it), the wand's
    # orientation carried along (its handle angle about the tangent and its tilt kept), the arm seeded by the last q
    if ext_back is not None and nodes:
        last_row = nodes[-1]; s0 = float(last_row[1]); wr0 = R.from_quat(last_row[12:16]); i0 = path.idx(s0)
        for sv in np.arange(s0 + 0.002, path.length - ext_back + 1e-9, 0.002):
            i = path.idx(sv)
            z0 = wr0.apply([0., 0., 1.])
            if z0 @ path.t[i] < 0:
                z0 = -z0
            a_ = np.cross(z0, path.t[i]); sa = np.linalg.norm(a_)       # the ring's normal onto the wire's tangent
            rot = R.from_rotvec(a_ / sa * np.arctan2(sa, z0 @ path.t[i])) if sa > 1e-9 else R.identity()
            nodes.append(np.r_[-1, sv, last_row[2:9], path.p[i], (rot * wr0).as_quat(), path.t[i], path.b[i]])
    np.savez(out, prefix=pre, qpos=np.array(d.qpos), g_p=g[0], g_q=g[1].as_quat(), nodes=np.array(nodes), inst=inst,
             route=route, cut=cut, dx=dx, held=bool((w.read_state().get("grasp") or {}).get("active")))
    print(json.dumps(dict(nodes=len(nodes), s_end=float(nodes[-1][1]), g=np.round(g[0], 4).tolist())))


def _kin(P):
    from rcb_topogym import Session
    from topogym.planning.buzz_wire_3d.redesign.kin import PandaKin
    from topogym.planning.buzz_wire_3d.redesign.feasibility import classify
    from nudge_lib import Dist
    s = Session("buzz_wire_3d", instance=str(P["inst"]), strict=True)
    kin = PandaKin(s.env._world.model); kin.d.qpos[:] = P["qpos"]
    return s, kin, Dist(kin, classify(kin))


def table(prep_p, k, K, out, dmax=60.0, dstep=5.0, lags=(0.0,)):
    from topogym.planning.buzz_wire_3d.redesign.wandpath import tcp_from_wand
    P = np.load(prep_p)
    s, kin, dist = _kin(P)
    g = (P["g_p"], R.from_quat(P["g_q"]))
    nodes = P["nodes"]
    D = np.deg2rad(np.arange(-dmax, dmax + 1e-9, dstep)); LG = np.deg2rad(np.asarray(sorted(lags), float))
    n_all = len(nodes); lo, hi = (n_all * k) // K, (n_all * (k + 1)) // K
    idx = np.arange(lo, hi)                                   # a contiguous chunk: IK seeded along it
    Q = np.zeros((len(idx), len(LG), len(D), 7)); ok_ = np.zeros((len(idx), len(LG), len(D)), bool)
    rs = np.full(ok_.shape, -1.0); dk = np.full_like(rs, -1.0); ww = np.full_like(rs, -1.0)
    for a_, n in enumerate(idx):
        row = nodes[n]; qr = row[2:9]; wp = row[9:12]; wr = R.from_quat(row[12:16]); tg = row[16:19]; bn = row[19:22]
        for li, lg in enumerate(LG):
            for j, dl in enumerate(D):
                r2 = R.from_rotvec(bn * lg) * R.from_rotvec(tg * dl) * wr
                tp2, tr2 = tcp_from_wand(wp, r2, g)
                q, ok = None, False
                if a_ and ok_[a_ - 1, li, j]:
                    q, ok, _, _ = kin.ik(tp2, tr2, Q[a_ - 1, li, j], iters=100)
                    ok = ok and np.abs(q - Q[a_ - 1, li, j]).max() < 0.3
                if not ok:
                    q, ok, _, _ = kin.ik(tp2, tr2, qr, iters=200)
                if not ok:
                    continue
                kin.set_q(q); kin.set_wand(wp, r2); mujoco.mj_kinematics(kin.m, kin.d)
                a1, _, a3 = dist.robot_static()
                Q[a_, li, j], ok_[a_, li, j], rs[a_, li, j], dk[a_, li, j], ww[a_, li, j] = q, True, a1, a3, dist.wand_wire()
    np.savez(out, idx=idx, D=D, LG=LG, Q=Q, ok=ok_, rs=rs, deck=dk, ww=ww)


def dp(prep_p, tglob, out, rclear=0.002, jump=0.3, wclear=0.0003, dstep_max=1, repair=True):
    from topogym.planning.buzz_wire_3d.redesign.wandpath import tcp_from_wand
    P = np.load(prep_p); nodes = P["nodes"]; N = len(nodes)
    parts = [np.load(f) for f in sorted(glob.glob(tglob))]
    D, LG = parts[0]["D"], parts[0]["LG"]; M, L = len(D), len(LG)
    Q = np.zeros((N, L, M, 7)); feas = np.zeros((N, L, M), bool); rs = np.zeros((N, L, M))
    for p in parts:
        Q[p["idx"]] = p["Q"]; rs[p["idx"]] = p["rs"]
        feas[p["idx"]] = p["ok"] & (p["rs"] >= rclear) & (p["deck"] >= -0.0005) & (p["ww"] >= wclear)
    kin = dist = None
    g = (P["g_p"], R.from_quat(P["g_q"]))
    INF = 1e18
    cost = np.full((N, L, M), INF); back = np.full((N, L, M, 2), -1, int)
    base = np.abs(np.rad2deg(D))[None, :] / 10 + np.abs(np.rad2deg(LG))[:, None] / 5
    cost[0] = np.where(feas[0], base, INF)
    repaired = 0
    for n in range(1, N):
        for li, j in zip(*np.nonzero(feas[n])):
            best, bk = INF, None
            preds = [(ll, jj) for ll in range(max(0, li - 1), min(L, li + 2)) for jj in range(max(0, j - dstep_max), min(M, j + dstep_max + 1))
                     if cost[n - 1, ll, jj] < INF]
            for ll, jj in preds:
                if np.abs(Q[n, li, j] - Q[n - 1, ll, jj]).max() <= jump:
                    c = cost[n - 1, ll, jj] + base[li, j] + 0.5 * (abs(jj - j) + abs(ll - li))
                    if c < best:
                        best, bk = c, (ll, jj)
            if bk is None and repair and preds:
                if kin is None:
                    s_, kin, dist = _kin(P)
                row = nodes[n]; wp = row[9:12]; wr = R.from_quat(row[12:16]); tg = row[16:19]; bn = row[19:22]
                r2 = R.from_rotvec(bn * LG[li]) * R.from_rotvec(tg * D[j]) * wr
                tp2, tr2 = tcp_from_wand(wp, r2, g)
                for ll, jj in sorted(preds, key=lambda x: cost[n - 1, x[0], x[1]]):
                    q, ok, _, _ = kin.ik(tp2, tr2, Q[n - 1, ll, jj], iters=150)
                    if not ok or np.abs(q - Q[n - 1, ll, jj]).max() > jump:
                        continue
                    kin.set_q(q); kin.set_wand(wp, r2); mujoco.mj_kinematics(kin.m, kin.d)
                    a1, _, a3 = dist.robot_static()
                    if a1 < rclear or a3 < -0.0005 or dist.wand_wire() < wclear:
                        continue
                    Q[n, li, j], rs[n, li, j] = q, a1
                    best, bk = cost[n - 1, ll, jj] + base[li, j] + 0.5 * (abs(jj - j) + abs(ll - li)), (ll, jj)
                    repaired += 1
                    break
            if bk is not None:
                cost[n, li, j], back[n, li, j] = best, bk
        if (cost[n] >= INF).all():
            reach = [int((cost[m_] < INF).sum()) for m_ in range(max(0, n - 60), n)]
            lastone = [(m_, [(int(LG[a]), int(round(np.rad2deg(D[b])))) for a, b in zip(*np.nonzero(cost[m_] < INF))][:6]) for m_ in range(max(0, n - 4), n)]
            info = dict(dead_end=n, s=float(nodes[n][1]), route_row=int(nodes[n][0]), feasible_now=int(feas[n].sum()),
                        reachable_prev=int((cost[n - 1] < INF).sum()), repaired=repaired, reach_hist=reach, last=lastone)
            print(json.dumps(info)); return info
    li, j = np.unravel_index(int(np.argmin(cost[-1])), cost[-1].shape); seq = []
    for n in range(N - 1, -1, -1):
        seq.append((li, j))
        if n:
            li, j = back[n, li, j]
    seq = seq[::-1]
    qs = np.array([Q[n, li, j] for n, (li, j) in enumerate(seq)])
    mg = np.array([rs[n, li, j] for n, (li, j) in enumerate(seq)])
    dl = np.array([D[j] for li, j in seq]); lg = np.array([LG[li] for li, j in seq])
    np.savez(out, qs=qs, margin=mg, delta=dl, lag=lg, s=nodes[:, 1])
    info = dict(ok=True, nodes=N, s_end=float(nodes[-1][1]), max_delta_deg=float(np.rad2deg(np.abs(dl).max())),
                max_lag_deg=float(np.rad2deg(np.abs(lg).max())), min_robot_clear_mm=round(1e3 * float(mg.min()), 2),
                max_node_jump=float(np.abs(np.diff(qs, axis=0)).max()), repaired=repaired)
    print(json.dumps(info)); return info


if __name__ == "__main__":
    import argparse
    cmd = sys.argv[1]
    if cmd == "prep":
        ext = float(sys.argv[7]) if len(sys.argv) > 7 else None
        stop = float(sys.argv[8]) if len(sys.argv) > 8 else (0.0 if ext is not None else 0.10)
        prep(sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), sys.argv[6], s_stop=stop, ext_back=ext)
    elif cmd == "table":
        ap = argparse.ArgumentParser(); ap.add_argument("c"); ap.add_argument("prep"); ap.add_argument("k", type=int)
        ap.add_argument("K", type=int); ap.add_argument("out"); ap.add_argument("--dmax", type=float, default=60)
        ap.add_argument("--dstep", type=float, default=5); ap.add_argument("--lags", default="0"); x = ap.parse_args()
        table(x.prep, x.k, x.K, x.out, x.dmax, x.dstep, [float(v) for v in x.lags.split(",")])
    elif cmd == "dp":
        ap = argparse.ArgumentParser(); ap.add_argument("c"); ap.add_argument("prep"); ap.add_argument("tglob"); ap.add_argument("out")
        ap.add_argument("--rclear", type=float, default=0.002); ap.add_argument("--jump", type=float, default=0.3)
        ap.add_argument("--wclear", type=float, default=0.0003); ap.add_argument("--dstep-max", type=int, default=1)
        x = ap.parse_args(); dp(x.prep, x.tglob, x.out, x.rclear, x.jump, x.wclear, x.dstep_max)


# ------------------------------------------------------------------------------------------------ forward search
def _fast_model(P):
    """An independent copy of the Session's model (mj_saveModel round trip) with a 4 mm contact margin on the robot's
    near geoms and the wand, so one mj_collision reports every pair within 4 mm with its distance."""
    import tempfile, os
    from nudge_lib import NEAR
    s, kin0, _ = _kin(P)
    m0 = kin0.m
    f = tempfile.NamedTemporaryFile(suffix=".mjb", delete=False).name
    mujoco.mj_saveModel(m0, f, None)
    m = mujoco.MjModel.from_binary_path(f); os.unlink(f)
    bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "" for g in range(m.ngeom)]
    for g in range(m.ngeom):
        if bn[g] in NEAR or bn[g] == "wand::wand":
            m.geom_margin[g] = max(m.geom_margin[g], 0.004)
    from topogym.planning.buzz_wire_3d.redesign.kin import PandaKin
    from topogym.planning.buzz_wire_3d.redesign.feasibility import classify
    kin = PandaKin(m); kin.d.qpos[:] = P["qpos"]
    cls = classify(kin)
    gn = [kin.geom_name(g) for g in range(m.ngeom)]
    deck = np.array(["finish_support" in n or "wire_base" in n for n in gn])
    return s, kin, cls, deck


def _check(kin, cls, deck, q, wand):
    """-> (robot-static min excl. deck, deck min, wand-wire metal min, robot self penetration) within the margin"""
    rs, dk, ww, selfp = 1.0, 1.0, 1.0, False
    for dist, kind, g1, g2 in kin.contacts(q, wand=wand):
        if kind == "robot-static":
            if deck[g1] or deck[g2]:
                dk = min(dk, dist)
            else:
                rs = min(rs, dist)
        elif kind == "robot-robot" and dist < 0:
            selfp = True
        elif kind == "wand-static":
            if (cls["wand_metal"][g1] and cls["wire_metal"][g2]) or (cls["wand_metal"][g2] and cls["wire_metal"][g1]):
                ww = min(ww, dist)
    return rs, dk, ww, selfp


def fwd(prep_p, out, dmax=60.0, dstep=5.0, lags=(0.0,), rclear=0.002, jump=0.3, wclear=0.0003, n_pred=4, log_every=50,
        route_seed=False, prior_w=0.0, end_heading=False, end_lag_w=0.5, all_ends=False):
    from topogym.planning.buzz_wire_3d.redesign.wandpath import tcp_from_wand
    P = np.load(prep_p); nodes = P["nodes"]; N = len(nodes)
    s, kin, cls, deck = _fast_model(P)
    g = (P["g_p"], R.from_quat(P["g_q"]))
    D = np.deg2rad(np.arange(-dmax, dmax + 1e-9, dstep)); LG = np.deg2rad(np.asarray(sorted(lags), float))
    M, L = len(D), len(LG)
    base = np.abs(np.rad2deg(D))[None, :] / 10 + np.abs(np.rad2deg(LG))[:, None] / 5
    t0 = time.time()

    def pose(n, li, j):
        row = nodes[n]; wp = row[9:12]; wr = R.from_quat(row[12:16]); tg = row[16:19]; bn = row[19:22]
        r2 = R.from_rotvec(bn * LG[li]) * R.from_rotvec(tg * D[j]) * wr
        return wp, r2

    def evaluate(n, li, j, seeds):
        wp, r2 = pose(n, li, j)
        tp2, tr2 = tcp_from_wand(wp, r2, g)
        for k_, sq in enumerate(seeds):
            q, ok, _, _ = kin.ik(tp2, tr2, sq, iters=100 if k_ < len(seeds) - 1 or not route_seed else 200,
                                 q_prior=nodes[n][2:9] if prior_w else None, prior_w=prior_w)
            if not ok or (np.abs(q - sq).max() > jump and not (route_seed and k_ == len(seeds) - 1)):
                continue
            rs, dk, ww, selfp = _check(kin, cls, deck, q, (wp, r2))
            if rs < rclear or dk < -0.0005 or ww < wclear or selfp:
                continue
            return q, rs
        return None, None
    # node 0: the route's own configuration at every state that solves from it
    cur = {}
    q_live = np.asarray(P["qpos"])[kin.qadr]           # the arm as the prefix leaves it (holding the wand on the deck)
    l0, j0 = int(np.argmin(np.abs(LG))), int(np.argmin(np.abs(D)))
    q0, rs0 = evaluate(0, l0, j0, [q_live, nodes[0][2:9]])
    if q0 is not None:
        cur[(l0, j0)] = (base[l0, j0], q0, rs0)
        print(json.dumps(dict(node0_jump_from_live=round(float(np.abs(q0 - q_live).max()), 4))), flush=True)
    else:
        wp, r2 = pose(0, l0, j0); tp2, tr2 = tcp_from_wand(wp, r2, g)
        q, ok, pe, re = kin.ik(tp2, tr2, q_live, iters=300)
        print(json.dumps(dict(node0_failed=True, ik_ok=bool(ok), pos_err=float(pe), rot_err=float(re),
                              check=[float(x) for x in _check(kin, cls, deck, q, (wp, r2))[:3]])), flush=True)
    hist = [dict(cur=cur)]
    backs = [{}]
    for n in range(1, N):
        cand = {}
        for (li, j), (c, q, rs) in cur.items():
            for ll in (li - 1, li, li + 1):
                for jj in (j - 1, j, j + 1):
                    if 0 <= ll < L and 0 <= jj < M:
                        cand.setdefault((ll, jj), []).append((c + 0.5 * (abs(ll - li) + abs(jj - j)), (li, j), q))
        if nodes[n][1] < 0.012:                  # still around the start mount: lift it as it lies
            cand = {k: v for k, v in cand.items() if k == (l0, j0)}
        nxt, bk = {}, {}
        for st, preds in cand.items():
            preds.sort(key=lambda x: x[0])
            q, rs = evaluate(n, st[0], st[1], [p[2] for p in preds[:n_pred]] + ([nodes[n][2:9]] if route_seed else []))
            if q is None:
                continue
            # the predecessor whose seed worked is not tracked separately: the cheapest one within the jump bound
            for c, pst, pq in preds:
                if np.abs(q - pq).max() <= jump:
                    nxt[st] = (c + base[st], q, rs); bk[st] = pst
                    break
        if not nxt:
            info = dict(dead_end=n, s=float(nodes[n][1]), route_row=int(nodes[n][0]), prev=len(cur),
                        prev_states=[(int(round(np.rad2deg(LG[a]))), int(round(np.rad2deg(D[b])))) for a, b in list(cur)[:12]],
                        wall_s=round(time.time() - t0, 1))
            print(json.dumps(info), flush=True)
            return info
        if os.environ.get("FWD_DEBUG") and int(os.environ["FWD_DEBUG"].split(":")[0]) <= n <= int(os.environ["FWD_DEBUG"].split(":")[1]):
            print(n, round(float(nodes[n][1]), 4), "cand", len(cand), "kept", sorted((int(round(np.rad2deg(LG[a]))), int(round(np.rad2deg(D[b])))) for a, b in nxt), flush=True)
        cur = nxt
        backs.append(bk)
        hist.append(dict(cur={k: (v[0], v[1], v[2]) for k, v in cur.items()}))
        if n % log_every == 0:
            print(json.dumps(dict(n=n, s=round(float(nodes[n][1]), 4), states=len(cur), wall_s=round(time.time() - t0, 1))), flush=True)
    from topogym.assets.geometry.buzz_wire_3d import endpoint_pose
    fx = endpoint_pose(s.env._task.route, 1)[1].apply([1., 0., 0.])
    fyaw = np.arctan2(fx[1], fx[0])

    def end_score(k):
        wp, r2 = pose(N - 1, k[0], k[1])
        h = r2.apply([1., 0., 0.])
        dyaw = abs((np.arctan2(h[1], h[0]) - fyaw + np.pi) % (2 * np.pi) - np.pi)
        return np.rad2deg(dyaw) + end_lag_w * abs(np.rad2deg(LG[k[0]])) + 0.01 * cur[k][0]
    st = min(cur, key=end_score) if end_heading else min(cur, key=lambda k: cur[k][0])
    end_info = dict(end_states=len(cur), end_score=round(float(end_score(st)), 1),
                    end_lag_deg=round(float(np.rad2deg(LG[st[0]])), 1), end_delta_deg=round(float(np.rad2deg(D[st[1]])), 1))
    def backtrack(st):
        seq = [st]
        for n in range(N - 1, 0, -1):
            st = backs[n][st]; seq.append(st)
        return seq[::-1]
    if all_ends:
        for k_, est in enumerate(sorted(cur, key=end_score)):
            sq = backtrack(est)
            np.savez(str(out).replace(".npz", f"_end{k_}.npz"), qs=np.array([hist[n]["cur"][s_][1] for n, s_ in enumerate(sq)]),
                     margin=np.array([hist[n]["cur"][s_][2] for n, s_ in enumerate(sq)]), delta=np.array([D[j] for li, j in sq]),
                     lag=np.array([LG[li] for li, j in sq]), s=nodes[:, 1])
            print(json.dumps(dict(end=k_, lag=round(float(np.rad2deg(LG[est[0]])), 1), delta=round(float(np.rad2deg(D[est[1]])), 1),
                                  score=round(float(end_score(est)), 1))), flush=True)
    seq = backtrack(st)
    qs = np.array([hist[n]["cur"][st][1] for n, st in enumerate(seq)])
    mg = np.array([hist[n]["cur"][st][2] for n, st in enumerate(seq)])
    dl = np.array([D[j] for li, j in seq]); lg = np.array([LG[li] for li, j in seq])
    np.savez(out, qs=qs, margin=mg, delta=dl, lag=lg, s=nodes[:, 1])
    info = dict(ok=True, nodes=N, s_end=float(nodes[-1][1]), max_delta_deg=float(np.rad2deg(np.abs(dl).max())),
                max_lag_deg=float(np.rad2deg(np.abs(lg).max())), min_robot_clear_mm=round(1e3 * float(mg.min()), 2),
                max_node_jump=float(np.abs(np.diff(qs, axis=0)).max()), wall_s=round(time.time() - t0, 1), **end_info)
    print(json.dumps(info), flush=True)
    return info


if __name__ == "__main__" and sys.argv[1] == "fwd":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("c"); ap.add_argument("prep"); ap.add_argument("out")
    ap.add_argument("--dmax", type=float, default=60); ap.add_argument("--dstep", type=float, default=5)
    ap.add_argument("--lags", default="0"); ap.add_argument("--rclear", type=float, default=0.002)
    ap.add_argument("--jump", type=float, default=0.3); ap.add_argument("--wclear", type=float, default=0.0003)
    ap.add_argument("--route-seed", action="store_true"); ap.add_argument("--prior-w", type=float, default=0.0)
    ap.add_argument("--end-heading", action="store_true"); ap.add_argument("--end-lag-w", type=float, default=0.5)
    ap.add_argument("--all-ends", action="store_true")
    x = ap.parse_args()
    fwd(x.prep, x.out, x.dmax, x.dstep, [float(v) for v in x.lags.split(",")], x.rclear, x.jump, x.wclear,
        route_seed=x.route_seed, prior_w=x.prior_w, end_heading=x.end_heading, end_lag_w=x.end_lag_w,
        all_ends=x.all_ends)
