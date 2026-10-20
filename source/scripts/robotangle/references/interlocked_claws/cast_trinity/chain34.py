"""带手的后半局（漂浮仿真，夹爪是拿着的零件身上的碰撞体）：接在第一段（seg12_combo.py 的输出）后面，
  2. 慢转到推缝方向、落稳 1 s、对准、推缝（B–T 分开）                         T、B 拿着（左手 T=GT1，右手 B=GB1）
  2b. 拿着两件转回 #21，松开 B（B 的夹爪不再碰东西），落稳 3 s                  T 拿着
  换手：右手抓 T（GT2）、左手松 T、抓 H（GH，H 在落稳的位置）；之后 H 不动
  3. 两件路线规划 T 相对 H（夹爪算进碰撞：每只手不碰另一件拿着的零件、两手不相碰），沿路线走、推缝（T–H 分开）；B 自由
  T 拿走（放到桌上），B 落稳 2 s，右手抓 B（GB2）
  4. 两件路线规划 B 相对 H，沿路线走、推缝（H–B 分开）
用法：chain34.py <combo12_*.pkl> <推缝前倾斜目标下标> <GT2> <GH> <GB2> <输出名> [seed]
输出：out/chain34_<输出名>.pkl：各阶段的状态（各自坐标系）、重力、手、成功与否。"""
import os, sys, pickle, time
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import mujoco, fcl, trimesh
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', 'scripts'), os.path.join(HERE, '..', '..', 'scripts')):   # vast 上 rerun/ 深一层
    if os.path.isfile(os.path.join(_p, 'rrt.py')):
        sys.path.insert(0, _p)
        break
import rrt
import geom12 as g
import puzzle as pz
import k1
from physpush import Sim
from physwalk import GDIRS
from grasp_scan12 import Hand, candidates, bvh
from tilt_follow import resample, run as tf_run
import slide_push

MM = 1e-3
G21 = GDIRS[21]
HAND = Hand()
FAR = np.eye(4)
FAR[:3, 3] = [5.0, 5.0, 5.0]


def hand_verts(kind, name):
    G, w = [(c[1], c[2]) for c in candidates(g, kind) if c[0] == name][0]
    return [V @ G[:3, :3].T + G[:3, 3] for V in HAND.hull_verts(w)]


def fcl_objs(vs):
    return [fcl.CollisionObject(bvh(trimesh.convex.convex_hull(V)), fcl.Transform()) for V in vs]


def hits(objs, other):
    return any(fcl.collide(o, other, fcl.CollisionRequest(), fcl.CollisionResult()) for o in objs)


def to_frame(s, k):
    iA = np.linalg.inv(s[k])
    return {kk: iA @ v for kk, v in s.items()}


def tilt_release(state_T, hands, g0, g1, tilt_s=4.0, settle_s=3.0):
    """state_T（T 系）：T、B 拿着，H 自由；拿着 T、B 把重力从 g0 转到 g1，松开 B，落稳。返回状态列表（T 系）和每个状态的重力。"""
    sim = Sim(g, gravcomp=True, hands=hands)
    sim.m.body_gravcomp[sim.bid['H']] = 0.0
    sim.set_poses(state_T)
    sim.hold('T', np.eye(4), solref=0.002)
    sim.hold('B', state_T['B'], solref=0.003)
    a0, a1 = g0 / np.linalg.norm(g0), g1 / np.linalg.norm(g1)
    ax = np.cross(a0, a1)
    ang = np.arctan2(np.linalg.norm(ax), a0 @ a1)
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    rec, gr = [sim.poses()], [a0]
    nt = int(tilt_s / 0.02)
    for i in range(1, nt + 1):
        gv = Rot.from_rotvec(ax * ang * i / nt).apply(a0)
        sim.m.opt.gravity[:] = 9.81 * gv
        for _ in range(40):
            mujoco.mj_step(sim.m, sim.d)
        rec.append(sim.poses())
        gr.append(gv)
    sim.m.body_gravcomp[sim.bid['B']] = 0.0
    sim.d.eq_active[sim.eid['B']] = 0
    sim.drop_hand('B')
    for _ in range(int(settle_s / 0.02)):
        for _ in range(40):
            mujoco.mj_step(sim.m, sim.d)
        rec.append(sim.poses())
        gr.append(a1)
    return rec, gr, nt


def settle_held(state, held, hands, gv, secs, far=()):
    """state 在 held[0] 系；held 拿着，far 挪到 5 m 外拿着，其余自由落稳。"""
    poses = dict(state)
    for k in far:
        poses[k] = FAR.copy()
    sim = Sim(g, gravcomp=True, hands=hands)
    for k in g.KINDS:
        if k not in held and k not in far:
            sim.m.body_gravcomp[sim.bid[k]] = 0.0
    sim.m.opt.gravity[:] = 9.81 * gv / np.linalg.norm(gv)
    sim.set_poses(poses)
    for k in tuple(held) + tuple(far):
        sim.hold(k, poses[k], solref=0.002)
    rec = [sim.poses()]
    for _ in range(int(secs / 0.02)):
        for _ in range(40):
            mujoco.mj_step(sim.m, sim.d)
        rec.append(sim.poses())
    return rec


def plan_pair(P, poses, X, Y, hX, hY, secs, seed, back=0.0, margin=0.0):
    """poses 在 Y 系。两件路线 X 相对 Y 到最好的“躺着过缝”位形；夹爪算进碰撞（X 的手不碰 Y，Y 的手不碰 X，两手不相碰）。
    back > 0：终点改成沿推缝方向往回退 back 米的位形（不碰时），之后由执行者沿推缝方向直线推进去。
    margin > 0：路线上两件间隙不小于 margin 米（真夹爪跟踪有 0.2–1 mm 误差，零间隙的路线会卡住）。"""
    Z = [k for k in 'THB' if k not in (X, Y)][0]
    key = ''.join(sorted(X + Y, key='THB'.index))
    sign = P.links(poses)[key]
    K, Dv, _ = k1.lying(g, P, X, Y, sign)[0]
    oX, oY = fcl_objs(hX), fcl_objs(hY)

    def extra(T):
        tf = fcl.Transform(T[:3, :3], T[:3, 3])
        for o in oX:
            o.setTransform(tf)
        P.p[X].obj.setTransform(tf)
        P.p[Y].obj.setTransform(fcl.Transform())
        if hits(oX, P.p[Y].obj) or any(hits(oX, o2) for o2 in oY) or hits(oY, P.p[X].obj):
            return False
        return True
    print('    hands at the start: %s' % ('clear' if extra(poses[X]) else 'TOUCHING the other held piece'), flush=True)
    pl = rrt.Planner(P, X, {Y: np.eye(4), Z: FAR}, ((Y, X),), center=[0.07, 0, 0.0], half=[0.16, 0.12, 0.16],
                     rng=np.random.default_rng(seed), extra=extra, margin=margin)
    A, pre = poses[X], []
    if not pl.free(A):                                   # 起点贴着（实测的挂着状态常常是靠在一起的）：先退到附近一个不碰的位姿
        esc = pl.escape(A)
        print('    start in contact: escape %s' % ('found, %d poses' % len(esc) if esc else 'NOT found'), flush=True)
        if esc is None:
            return None
        pre, A = esc[:-1], esc[-1]
    if back > 0:
        Kb = K.copy()
        Kb[:3, 3] -= back * Dv
        if pl.free(Kb):
            K = Kb
    path = pl.connect(A, K, max_time=secs)
    if path is None:
        return None
    return pre + pl.shortcut(path, 300)


if __name__ == '__main__':
    cpath, tilt, GT2, GH, GB2, name = sys.argv[1:7]
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    tilt = int(tilt)
    P = pz.Puzzle(g)
    C = pickle.load(open(cpath, 'rb'))
    GT1, GB1 = C['gT'], C['gB']
    out = dict(GT1=GT1, GB1=GB1, GT2=GT2, GH=GH, GB2=GB2, tilt=tilt, ok=False)
    t0 = time.time()

    def save(why):
        out['why'] = why
        pickle.dump(out, open(os.path.join(HERE, 'out/chain34_%s.pkl' % name), 'wb'))
        print('%s (%.0f s)' % (why, time.time() - t0), flush=True)
    # 2. 推缝
    os.environ['HANDS_SPEC'] = 'T=%s;B=%s' % (GT1, GB1)
    os.environ['SP_FROM'] = '1,0,0'
    os.environ['SP_TILT_S'] = '4'
    r2 = slide_push.run(g, P, C['path'][-1], 'B', 'T', GDIRS[tilt], 1.0)
    out['s2'] = r2
    if not r2['success']:
        save('push of B through T failed: %s' % r2['why'])
        sys.exit(0)
    # 2b. 转回 #21、松开 B
    hands12 = {'T': hand_verts('T', GT1), 'B': hand_verts('B', GB1)}
    rec, gr, nt = tilt_release(r2['path'][-1], hands12, GDIRS[tilt], G21)
    lk = P.links(rec[-1])
    out['s2b'] = dict(path=rec, gravity=gr, n_tilt=nt)
    print('released B after tilting back: links %s' % lk, flush=True)
    if lk['TB'] != 0:
        save('B fell back through T after the release')
        sys.exit(0)
    # 换手：右手 T=GT2、左手 H=GH，查两只新手在这一刻不碰别的件
    s = rec[-1]
    hT2, hH = hand_verts('T', GT2), hand_verts('H', GH)
    oT2, oH, oT1 = fcl_objs(hT2), fcl_objs(hH), fcl_objs(hands12['T'])
    P.place(s)
    for o in oT2 + oT1:
        o.setTransform(fcl.Transform(s['T'][:3, :3], s['T'][:3, 3]))
    for o in oH:
        o.setTransform(fcl.Transform(s['H'][:3, :3], s['H'][:3, 3]))
    bad = [n for n, objs, others in (('T hand (GT2) vs H, B', oT2, ('H', 'B')), ('H hand (GH) vs T, B', oH, ('T', 'B')))
           if any(hits(objs, P.p[k].obj) for k in others)]
    if any(hits(oT2, o) for o in oT1):
        bad.append('the two hands on T touch during the handover')
    if any(hits(oH, o) for o in oT2 + oT1):
        bad.append('the H hand touches a T hand')
    out['handover_conflicts'] = bad
    print('handover check: %s' % (bad or 'clear'), flush=True)
    # 3. T 相对 H
    sH = to_frame(s, 'H')
    gH = sH['T'][:3, :3] @ G21                 # T 不动时的重力（T 系 #21）换到 H 系
    out['gravity_H'] = gH
    path = plan_pair(P, sH, 'T', 'H', hT2, hH, 300, seed)
    if path is None:
        save('no T-H path with the hands')
        sys.exit(0)
    os.environ['HANDS_SPEC'] = 'T=%s;H=%s' % (GT2, GH)
    r3 = tf_run(g, P, sH, resample(path), 'T', 'H', None, gH)
    out['s3'] = r3
    out['s3_ref'] = path
    if not r3['success']:
        save('segment 3 failed: %s' % r3['why'])
        sys.exit(0)
    # T 拿走，B 落稳，右手抓 B
    s4 = settle_held(r3['path'][-1], 'H', {'H': hH}, gH, 2.0, far='T')
    out['s4_settle'] = s4
    sB = s4[-1]
    hB2 = hand_verts('B', GB2)
    oB2 = fcl_objs(hB2)
    P.place(dict(sB, T=FAR))
    for o in oB2:
        o.setTransform(fcl.Transform(sB['B'][:3, :3], sB['B'][:3, 3]))
    for o in oH:
        o.setTransform(fcl.Transform(sB['H'][:3, :3], sB['H'][:3, 3]))
    out['regrasp_B_conflict'] = hits(oB2, P.p['H'].obj) or any(hits(oB2, o) for o in oH)
    print('B regrasp check: %s' % ('CONFLICT' if out['regrasp_B_conflict'] else 'clear'), flush=True)
    path4 = plan_pair(P, dict(sB, T=FAR), 'B', 'H', hB2, hH, 300, seed)
    if path4 is None:
        save('no B-H path with the hands')
        sys.exit(0)
    os.environ['HANDS_SPEC'] = 'H=%s;B=%s' % (GH, GB2)
    r4 = tf_run(g, P, dict(sB, T=FAR), resample(path4), 'B', 'H', None, gH)
    out['s4'] = r4
    out['s4_ref'] = path4
    out['ok'] = bool(r4['success'])
    save('whole chain %s' % ('SUCCEEDED' if r4['success'] else 'failed in segment 4: %s' % r4['why']))
