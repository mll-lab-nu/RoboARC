"""“倾斜 → 第三件挂开 → 沿两件路线走 → 推缝”，一次仿真做完：X、Y 硬 weld（重力补偿），第三件 Z 自由。
1. X、Y 都拿着不动，重力从 g0 沿大圆转到 g1（都在 Y 坐标系里；两只手一起转谜题），落稳 settle 秒；
2. X 沿两件参考路线（pairref 的 rel：X 相对 Y）走，目标超前 2 格，等 X 跟上（≤ 1.2 mm、≤ 2°），卡住超过 4 s 算失败；
3. 路线终点是“躺着过缝”位形 K：沿 K 的推缝方向推 32 mm（同 physwalk 的推缝检查）。
用法：tilt_follow.py <geom> <state.pkl[:index]> <pairref.pkl> <X> <Y> <g0 "gx,gy,gz"|-> <g1 "gx,gy,gz"|all> [输出名]
  g0、g1 在 Y 坐标系里；g0 = '-' 表示不转。g1 = all：26 个方向都试一遍只打印（g0 必须给）。"""
import os, sys, pickle, importlib
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot

import puzzle as pz, k1
from physpush import Sim, quat
from physwalk import pdist, GDIRS

HERE = os.path.dirname(os.path.abspath(__file__))
MM = 1e-3
TOL_P, TOL_R = 1.2 * MM, np.radians(2.0)


def step(sim, n=40):
    for _ in range(n):
        mujoco.mj_step(sim.m, sim.d)


def resample(ref, dp=0.4 * MM, da=np.radians(0.4)):
    """相邻两点之间插值到每步 ≤ dp、≤ da（目标超前几格、按下一格查是否跟上，格子太粗就永远“跟不上”）"""
    from scipy.spatial.transform import Slerp
    out = [ref[0]]
    for A, B in zip(ref[:-1], ref[1:]):
        n = max(1, int(np.ceil(max(np.linalg.norm(B[:3, 3] - A[:3, 3]) / dp,
                                   Rot.from_matrix(A[:3, :3].T @ B[:3, :3]).magnitude() / da))))
        sl = Slerp([0, 1], Rot.from_matrix(np.stack([A[:3, :3], B[:3, :3]])))
        for t in np.linspace(0, 1, n + 1)[1:]:
            T = np.eye(4)
            T[:3, :3] = sl(t).as_matrix()
            T[:3, 3] = (1 - t) * A[:3, 3] + t * B[:3, 3]
            out.append(T)
    return out


def run(g, P, poses, ref, X, Y, g0, g1, tilt_s=4.0, settle_s=2.0, verbose=True, share=0.0):
    """share：沿路线走的时候 Y 在世界里反向转掉 X 相对转动的这个比例（两只手分担转动）；Y 系里的重力随之变（逐状态记在 grav 里）。"""
    Z = [k for k in 'THB' if k not in (X, Y)][0]
    key = ''.join(sorted(X + Y, key='THB'.index))
    sim = Sim(g, gravcomp=True)
    sim.m.body_gravcomp[sim.bid[Z]] = 0.0
    g1 = g1 / np.linalg.norm(g1)
    sim.m.opt.gravity[:] = 9.81 * (g1 if g0 is None else g0 / np.linalg.norm(g0))
    sim.set_poses(poses)
    sim.hold(Y, np.eye(4), solref=0.002)
    sim.hold(X, poses[X], solref=0.003)
    rec, phase, maxpen = [sim.poses()], ['start'], 0.0
    grav = [np.array(sim.m.opt.gravity) / 9.81]
    if g0 is not None:
        a0 = g0 / np.linalg.norm(g0)
        ax = np.cross(a0, g1)
        ang = np.arctan2(np.linalg.norm(ax), a0 @ g1)
        ax = ax / (np.linalg.norm(ax) + 1e-12)
        nt = int(tilt_s / 0.02)
        for i in range(1, nt + 1):
            sim.m.opt.gravity[:] = 9.81 * Rot.from_rotvec(ax * ang * i / nt).apply(a0)
            step(sim)
            rec.append(sim.poses())
            phase.append('tilt')
            grav.append(np.array(sim.m.opt.gravity) / 9.81)
    for _ in range(int(settle_s / 0.02)):
        step(sim)
        rec.append(sim.poses())
        phase.append('settle')
        grav.append(np.array(sim.m.opt.gravity) / 9.81)
    c0 = rec[-1]
    zmove = np.linalg.norm(c0[Z][:3, 3] - poses[Z][:3, 3]) / MM
    clear0 = P.clearance(c0, ((X, Z),))[X + Z] / MM
    # 沿路线走
    k, stuck, ok, why = 0, 0, True, ''
    while k < len(ref) - 1:
        tgt = ref[min(k + 3, len(ref) - 1)]
        sim.d.mocap_pos[sim.mid[X]] = tgt[:3, 3]
        sim.d.mocap_quat[sim.mid[X]] = quat(tgt[:3, :3])
        if share:
            Dk = Rot.from_matrix(tgt[:3, :3] @ ref[0][:3, :3].T).as_rotvec()
            sim.m.opt.gravity[:] = 9.81 * Rot.from_rotvec(-share * Dk).as_matrix().T @ g1
        step(sim)
        maxpen = max(maxpen, sim.penetration())
        c = sim.poses()
        rec.append(c)
        phase.append('follow')
        grav.append(np.array(sim.m.opt.gravity) / 9.81)
        tk = ref[k + 1]
        if (np.linalg.norm(c[X][:3, 3] - tk[:3, 3]) < TOL_P and
                np.linalg.norm(Rot.from_matrix(c[X][:3, :3].T @ tk[:3, :3]).as_rotvec()) < TOL_R):
            k += 1
            stuck = 0
        else:
            stuck += 1
            if stuck > 200:
                ok, why = False, 'follow stuck at step %d of %d' % (k, len(ref) - 1)
                break
    if ok:
        cur = rec[-1]
        Ks = k1.lying(g, P, X, Y, P.links(poses)[key])
        K, D, _ = min(Ks, key=lambda kk: pdist(cur[X], kk[0]))
        for i in range(1, 65):
            T = K.copy()
            T[:3, 3] += D * i * 0.5 * MM
            sim.d.mocap_pos[sim.mid[X]] = T[:3, 3]
            sim.d.mocap_quat[sim.mid[X]] = quat(T[:3, :3])
            step(sim, 80)
            c = sim.poses()
            rec.append(c)
            phase.append('push')
            grav.append(np.array(sim.m.opt.gravity) / 9.81)
            if (np.linalg.norm(c[X][:3, 3] - T[:3, 3]) > 0.4 * MM or
                    np.linalg.norm(Rot.from_matrix(c[X][:3, :3].T @ T[:3, :3]).as_rotvec()) > np.radians(0.4) or
                    sim.penetration() > 0.3 * MM):
                ok, why = False, 'push blocked at %.1f mm' % (i * 0.5)
                break
    lk = P.links(rec[-1])
    success = ok and lk[key] == 0
    if verbose:
        print('g1 %s: Z moved %.1f mm while tilting/settling (X-Z clearance then %.1f mm); %s; links %s -> success %s' % (
            np.round(g1, 3).tolist(), zmove, clear0, why or 'followed and pushed through', lk, success), flush=True)
    return dict(path=rec, phase=phase, success=success, g0=g0, g1=g1, X=X, Y=Y, why=why, maxpen=maxpen, grav=grav, share=share)


if __name__ == '__main__':
    gname, spath, rpath, X, Y, g0s, g1s = sys.argv[1:8]
    name = sys.argv[8] if len(sys.argv) > 8 else None
    g = importlib.import_module(gname)
    P = pz.Puzzle(g)
    spath, _, sidx = spath.partition(':')
    d = pickle.load(open(spath, 'rb'))
    poses = d[int(sidx or 0)][1] if isinstance(d, list) else (d['path'][int(sidx)] if sidx else d['path'][-1])
    iY = np.linalg.inv(poses[Y])
    poses = {k: iY @ v for k, v in poses.items()}
    ref = resample(list(pickle.load(open(rpath, 'rb'))['rel']))
    g0 = None if g0s == '-' else np.array([float(v) for v in g0s.split(',')])
    if g1s == 'all':
        for i, gd in enumerate(GDIRS):
            print('#%d' % i, end=' ')
            run(g, P, poses, ref, X, Y, g0, gd)
    else:
        r = run(g, P, poses, ref, X, Y, g0, np.array([float(v) for v in g1s.split(',')]))
        if name:
            pickle.dump(r, open(os.path.join(HERE, 'out/tiltfollow_%s.pkl' % name), 'wb'))
