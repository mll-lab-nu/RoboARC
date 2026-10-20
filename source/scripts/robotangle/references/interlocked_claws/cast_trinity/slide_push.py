"""“倾斜 → 第三件滑开 → 对准 → 推缝”：MuJoCo 里 X、Y 硬 weld（重力补偿），第三件 Z 自由。
1. 重力换成 g'（Y 坐标系里；相当于两只手把整个谜题转个朝向），X、Y 不动，Z 落稳 settle 秒；
2. X 沿 SE(3) 直线从当前位姿走到最近的“躺着过缝”位形 K（每步 ≤ 0.5 mm、≤ 0.5°，等 X 跟上；跟不上 = 被挡）；
3. 沿 K 的推缝方向推 32 mm（每 0.5 mm 一格，查位置、朝向和穿透，同 physwalk）。
用法：slide_push.py <geom> <state.pkl[:index]> <X> <Y> <gx,gy,gz> <settle 秒> <输出名>
输出：out/slidepush_<输出名>.pkl（path = 全程状态，phase = 每个状态属于哪一步，success，gravity）。"""
import os, sys, pickle, importlib
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot, Slerp

import puzzle as pz, k1
from physpush import Sim, quat
from physwalk import pdist

HERE = os.path.dirname(os.path.abspath(__file__))
MM = 1e-3


def run(g, P, poses, X, Y, gv, settle_s, verbose=True):
    Z = [k for k in 'THB' if k not in (X, Y)][0]
    key = ''.join(sorted(X + Y, key='THB'.index))
    sim = Sim(g, gravcomp=True)
    sim.m.opt.gravity[:] = 9.81 * gv / np.linalg.norm(gv)
    sim.m.body_gravcomp[sim.bid[Z]] = 0.0
    sim.set_poses(poses)
    sim.hold(Y, np.eye(4), solref=0.002)
    sim.hold(X, poses[X], solref=0.003)
    rec, phase = [sim.poses()], ['start']
    if os.environ.get('SP_FROM'):                       # 慢慢转：重力从 SP_FROM 沿大圆转到 g'，用 SP_TILT_S 秒（两只手一起转谜题）
        g0 = np.array([float(v) for v in os.environ['SP_FROM'].split(',')])
        g0, g1 = g0 / np.linalg.norm(g0), gv / np.linalg.norm(gv)
        ax = np.cross(g0, g1)
        ang = np.arctan2(np.linalg.norm(ax), g0 @ g1)
        ax = ax / (np.linalg.norm(ax) + 1e-12)
        nt = int(float(os.environ.get('SP_TILT_S', '4')) / 0.02)
        for i in range(1, nt + 1):
            sim.m.opt.gravity[:] = 9.81 * Rot.from_rotvec(ax * ang * i / nt).apply(g0)
            for _ in range(40):
                mujoco.mj_step(sim.m, sim.d)
            rec.append(sim.poses())
            phase.append('tilt')
    for _ in range(int(settle_s / 0.02)):
        for _ in range(40):
            mujoco.mj_step(sim.m, sim.d)
        rec.append(sim.poses())
        phase.append('settle')
    cur = rec[-1]
    Ks = k1.lying(g, P, X, Y, P.links(poses)[key])
    K, D, _ = min(Ks, key=lambda k: pdist(cur[X], k[0]))
    A = cur[X]
    dist_mm = np.linalg.norm(K[:3, 3] - A[:3, 3]) / MM
    ang = Rot.from_matrix(A[:3, :3].T @ K[:3, :3]).magnitude()
    n = max(2, int(np.ceil(max(dist_mm / 0.5, np.degrees(ang) / 0.5))))
    sl = Slerp([0, 1], Rot.from_matrix(np.stack([A[:3, :3], K[:3, :3]])))
    ok, why = True, ''
    for t in np.linspace(0, 1, n + 1)[1:]:
        T = np.eye(4)
        T[:3, :3] = sl(t).as_matrix()
        T[:3, 3] = (1 - t) * A[:3, 3] + t * K[:3, 3]
        sim.d.mocap_pos[sim.mid[X]] = T[:3, 3]
        sim.d.mocap_quat[sim.mid[X]] = quat(T[:3, :3])
        for _ in range(40):
            mujoco.mj_step(sim.m, sim.d)
        c = sim.poses()
        rec.append(c)
        phase.append('approach')
        if np.linalg.norm(c[X][:3, 3] - T[:3, 3]) > 0.6 * MM or sim.penetration() > 0.3 * MM:
            ok, why = False, 'approach blocked at %.0f%% (lag %.2f mm, penetration %.2f mm)' % (
                100 * t, np.linalg.norm(c[X][:3, 3] - T[:3, 3]) / MM, sim.penetration() / MM)
            break
    if ok:
        for i in range(1, 65):
            T = K.copy()
            T[:3, 3] += D * i * 0.5 * MM
            sim.d.mocap_pos[sim.mid[X]] = T[:3, 3]
            sim.d.mocap_quat[sim.mid[X]] = quat(T[:3, :3])
            for _ in range(80):
                mujoco.mj_step(sim.m, sim.d)
            c = sim.poses()
            rec.append(c)
            phase.append('push')
            if (np.linalg.norm(c[X][:3, 3] - T[:3, 3]) > 0.4 * MM or
                    np.linalg.norm(Rot.from_matrix(c[X][:3, :3].T @ T[:3, :3]).as_rotvec()) > np.radians(0.4) or
                    sim.penetration() > 0.3 * MM):
                ok, why = False, 'push blocked at %.1f mm' % (i * 0.5)
                break
    lk = P.links(rec[-1])
    success = ok and lk[key] == 0
    if verbose:
        print('gravity %s: Z moved %.1f mm while settling; approach %.1f mm / %.1f deg; %s; links %s -> success %s' % (
            np.round(gv / np.linalg.norm(gv), 3).tolist(), np.linalg.norm(cur[Z][:3, 3] - poses[Z][:3, 3]) / MM, dist_mm,
            np.degrees(ang), why or 'pushed through', lk, success), flush=True)
    return dict(path=rec, phase=phase, success=success, gravity=gv, X=X, Y=Y, why=why)


if __name__ == '__main__':
    gname, spath, X, Y, gvec, settle_s, name = sys.argv[1:8]
    g = importlib.import_module(gname)
    P = pz.Puzzle(g)
    spath, _, sidx = spath.partition(':')
    d = pickle.load(open(spath, 'rb'))
    poses = d[int(sidx or 0)][1] if isinstance(d, list) else (d['path'][int(sidx)] if sidx else d['path'][-1])
    iY = np.linalg.inv(poses[Y])
    poses = {k: iY @ v for k, v in poses.items()}
    if gvec == 'all':                                   # 26 个方向都试一遍，只打印
        from physwalk import GDIRS
        for i, gd in enumerate(GDIRS):
            print('#%d' % i, end=' ')
            run(g, P, poses, X, Y, gd, float(settle_s))
    else:
        gv = np.array([float(v) for v in gvec.split(',')])
        r = run(g, P, poses, X, Y, gv, float(settle_s))
        pickle.dump(r, open(os.path.join(HERE, 'out/slidepush_%s.pkl' % name), 'wb'))
