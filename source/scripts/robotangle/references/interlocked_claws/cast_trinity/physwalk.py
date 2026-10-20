"""物理随机走（像人手“每件一点点地动”）：MuJoCo 里 Y 用硬 weld 固定，X 用硬 weld 跟一个 mocap 目标，第三件 Z 自由、受重力。
每一步：X 的目标做一个随机小动作（平移 ≤ 1.5 mm、转 ≤ 2°，转动绕 X 上随机一点），仿真 30 ms；X 跟不上（被挡）就撤回。
每 2 s 有一定概率换重力方向（相当于两只手把整个谜题转个朝向，让 Z 自己滑）。
目标值 = -（X 相对 Y 到最近 K2 的位姿距离，mm 当量）；按模拟退火接受/撤回目标的动作。离 K2 不到 1.5 时试一次推缝（X 沿推缝方向 32 mm，
每 0.5 mm 一格，同时查位置和朝向），推过且 X–Y 环绕数归零就记下成功。
用法：physwalk.py <geom> <state.pkl[:index]> <X> <Y> <seconds> <seed>；输出 out/physwalk_<tag>_<X><Y>_<seed>.pkl。"""
import os, sys, time, pickle, importlib, itertools
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot

import puzzle as pz, k1
from physpush import Sim, quat

HERE = os.path.dirname(os.path.abspath(__file__))
MM = 1e-3
GDIRS = [np.array(v, float) / np.linalg.norm(v) for v in itertools.product((-1, 0, 1), repeat=3) if any(v)]
FIXED_G = int(os.environ['PW_GRAV']) if os.environ.get('PW_GRAV') else None      # 整段只用这一个重力方向（GDIRS 的下标；-1 = 无重力）
# 或者任意方向：PW_GVEC = "gx,gy,gz"（单位向量），PW_GFRAME = Y（默认，Y 不动）或 X（X 不动、Y 动：Y 坐标系里的重力跟着 X 的转动走）
GVEC = np.array([float(v) for v in os.environ['PW_GVEC'].split(',')]) if os.environ.get('PW_GVEC') else None
GFRAME = os.environ.get('PW_GFRAME', 'Y')


def pdist(A, B):
    return np.linalg.norm(A[:3, 3] - B[:3, 3]) / MM + 50 * np.linalg.norm(Rot.from_matrix(A[:3, :3].T @ B[:3, :3]).as_rotvec())


def T4(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


if __name__ == '__main__':
    gname, spath, X, Y, secs, seed = sys.argv[1:7]
    g = importlib.import_module(gname)
    P = pz.Puzzle(g)
    rng = np.random.default_rng(int(seed))
    spath, _, sidx = spath.partition(':')
    d = pickle.load(open(spath, 'rb'))
    if isinstance(d, list):
        poses = d[int(sidx or 0)][1]
    else:
        poses = d['path'][int(sidx)] if sidx else d['path'][-1]
    Z = [k for k in 'THB' if k not in (X, Y)][0]
    iY = np.linalg.inv(poses[Y])
    poses = {k: iY @ v for k, v in poses.items()}
    key = ''.join(sorted(X + Y, key='THB'.index))
    lk0 = P.links(poses)
    Ks = k1.lying(g, P, X, Y, lk0[key])
    sim = Sim(g)
    sim.m.opt.gravity[:] = 0.0 if FIXED_G is None or FIXED_G < 0 else 9.81 * GDIRS[FIXED_G]

    def set_gravity(RX):
        if GVEC is not None:
            gv = GVEC / np.linalg.norm(GVEC)
            sim.m.opt.gravity[:] = 9.81 * (RX @ gv if GFRAME == 'X' else gv)
    set_gravity(poses[X][:3, :3])
    sim.set_poses(poses)
    sim.hold(Y, np.eye(4), solref=0.002)
    sim.hold(X, poses[X], solref=0.003)
    tgt = poses[X].copy()
    cur = sim.poses()
    val = -min(pdist(cur[X], K[0]) for K in Ks)
    best = val
    rec = [cur]
    glog = [(0, np.array(sim.m.opt.gravity) / 9.81)]
    t0 = time.time()
    temp = 2.0
    it = 0
    success = None
    while time.time() - t0 < float(secs) and success is None:
        it += 1
        if GVEC is not None:
            set_gravity(tgt[:3, :3])
        elif FIXED_G is not None:
            pass
        elif it % 66 == 0 and rng.random() < 0.3:           # 约每 2 s 有 30% 的概率换重力方向
            gd = GDIRS[rng.integers(len(GDIRS))] if rng.random() < 0.8 else np.zeros(3)
            sim.m.opt.gravity[:] = 9.81 * gd
            glog.append((len(rec), gd.copy()))           # 从第几个记录状态起换成这个重力方向（Y 坐标系）
        # X 的目标做一个随机小动作
        L = g.centerline(X)
        piv = tgt[:3, :3] @ L[rng.integers(len(L))] + tgt[:3, 3]
        dR = Rot.from_rotvec(rng.normal(size=3) * np.radians(2.0) / np.sqrt(3)).as_matrix()
        dp = rng.normal(size=3) * 1.5 * MM / np.sqrt(3)
        new = T4(dR @ tgt[:3, :3], dR @ (tgt[:3, 3] - piv) + piv + dp)
        sim.d.mocap_pos[sim.mid[X]] = new[:3, 3]
        sim.d.mocap_quat[sim.mid[X]] = quat(new[:3, :3])
        state = (sim.d.qpos.copy(), sim.d.qvel.copy())
        for _ in range(60):
            mujoco.mj_step(sim.m, sim.d)
        c = sim.poses()
        lag = np.linalg.norm(c[X][:3, 3] - new[:3, 3])
        v = -min(pdist(c[X], K[0]) for K in Ks)
        if lag > 0.6 * MM or sim.penetration() > 0.3 * MM or not (v >= val or rng.random() < np.exp((v - val) / temp)):
            sim.d.qpos[:], sim.d.qvel[:] = state                      # 撤回（Z 也回到之前）
            sim.d.mocap_pos[sim.mid[X]] = tgt[:3, 3]
            sim.d.mocap_quat[sim.mid[X]] = quat(tgt[:3, :3])
            mujoco.mj_forward(sim.m, sim.d)
            continue
        tgt, val = new, v
        rec.append(c)
        best = max(best, val)
        if it % 2000 == 0:
            print('   %d its, %d accepted, distance to K2 %.1f (best %.1f), Z moved %.1f mm (%.0f s)' % (
                it, len(rec), -val, -best, np.linalg.norm(c[Z][:3, 3] - poses[Z][:3, 3]) / MM, time.time() - t0), flush=True)
        if -val < 1.5:
            # 试推缝
            K, D, _ = min(Ks, key=lambda k: pdist(c[X], k[0]))
            st = (sim.d.qpos.copy(), sim.d.qvel.copy())
            ok = True
            push = []
            for i in range(0, 65):
                T = K.copy()
                T[:3, 3] += D * i * 0.5 * MM
                sim.d.mocap_pos[sim.mid[X]] = T[:3, 3]
                sim.d.mocap_quat[sim.mid[X]] = quat(T[:3, :3])
                set_gravity(T[:3, :3])
                for _ in range(80):
                    mujoco.mj_step(sim.m, sim.d)
                cc = sim.poses()
                push.append(cc)
                if (np.linalg.norm(cc[X][:3, 3] - T[:3, 3]) > 0.4 * MM or
                        np.linalg.norm(Rot.from_matrix(cc[X][:3, :3].T @ T[:3, :3]).as_rotvec()) > np.radians(0.4) or
                        sim.penetration() > 0.3 * MM):
                    ok = False
                    break
            lk = P.links(sim.poses())
            print('   push attempt at its %d: %s, links %s' % (it, 'through' if ok else 'blocked at %.1f mm' % (i * 0.5), lk), flush=True)
            if ok and lk[key] == 0:
                success = rec + push
                break
            sim.d.qpos[:], sim.d.qvel[:] = st
            mujoco.mj_forward(sim.m, sim.d)
            val = -1e9                                   # 推不过就把当前值作废，逼它走开再回来
    tag = os.path.splitext(os.path.basename(spath))[0] + (sidx and '_' + sidx) + ('' if FIXED_G is None else '_g%d' % FIXED_G)
    if GVEC is not None:                         # 任意方向时文件名用 PW_TAG（起点名太长）
        tag = os.environ['PW_TAG']
    print('success %s; best distance %.1f (%.0f s)' % (success is not None, -best, time.time() - t0))
    pickle.dump(dict(path=success if success else rec, success=success is not None, gravity=glog, n_walk=len(rec), X=X, Y=Y,
                     gvec=GVEC, gframe=GFRAME), open(os.path.join(HERE, 'out/physwalk_%s_%s%s_%s.pkl' % (tag, X, Y, seed)), 'wb'))
