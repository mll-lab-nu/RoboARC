"""物理推挤：MuJoCo 里三件都是自由刚体（碰撞体 = 精确凸块），没有重力，带阻尼。
Y 用很硬的 weld 固定在原点；X 用一根软 weld 拉向一个移动的 mocap 目标，目标沿两件参考路线（X 相对 Y）一步步走；
Z 完全自由，被碰到就让开。目标每走一格，等 X 跟上（位置误差 < 0.6 mm、转角 < 1°）再走下一格；跟不上就给 Z 一点随机小推力（“晃一晃”）。
记下每个控制周期三件的位姿。最后检查：X 到了参考路线终点（K2）没有、三对环绕数、最大穿透。
用法：physpush.py <geom> <state.pkl[:index]> <pairref.pkl> <X> <Y> <seconds> <seed>；输出 out/physpush_<geom>_<tag>_<X><Y>_<seed>.pkl。"""
import os, sys, time, pickle, importlib
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as Rot

import puzzle as pz

HERE = os.path.dirname(os.path.abspath(__file__))
MM = 1e-3
LOOK = int(os.environ.get('PUSH_LOOK', '2'))                 # 目标比当前格超前几格
TOL_P = float(os.environ.get('PUSH_TOL_MM', '1.2')) * MM      # 跟上当前格的位置容差
TOL_R = np.radians(float(os.environ.get('PUSH_TOL_DEG', '2.0')))
GRAV = int(os.environ['PUSH_GRAV']) if os.environ.get('PUSH_GRAV') else None   # Y 坐标系里的固定重力方向（physwalk.GDIRS 的下标）
GVEC = os.environ.get('PUSH_GVEC')                # 或任意方向 "gx,gy,gz"（Y 坐标系），输出名里记 '_gv'


def quat(R):
    return Rot.from_matrix(R).as_quat()[[3, 0, 1, 2]]


def hands_from_env():
    """HANDS_SPEC = "T=<抓法名>;B=<抓法名>"（grasp_scan12.candidates 里的名字）-> {零件: [夹爪凸包顶点（零件系）]}"""
    spec = os.environ.get('HANDS_SPEC')
    if not spec:
        return None
    import geom12
    from grasp_scan12 import Hand, candidates
    hand, out = Hand(), {}
    for item in spec.split(';'):
        k, name = item.split('=', 1)
        G, w = [(c[1], c[2]) for c in candidates(geom12, k) if c[0] == name][0]
        out[k] = [V @ G[:3, :3].T + G[:3, 3] for V in hand.hull_verts(w)]
    return out


def build(g, density=600.0, margin=0.0, gravcomp=False, hands=None):
    """gravcomp：三件都编进重力补偿（MuJoCo 只在编译时数补偿的物体），之后把自由的那件的 body_gravcomp 改回 0。
    hands：{零件: [夹爪凸包顶点（零件系）]}，夹爪作为那件零件身上的几乎没有质量的碰撞体（拿着它的手），名字 hand_<零件>_<i>。"""
    assets, bodies = '', ''
    for k in g.KINDS:
        geoms = ''
        for i, V in enumerate(g.cells(k)):
            V = np.asarray(V)[ConvexHull(V).vertices]
            assets += '<mesh name="%s%d" vertex="%s"/>' % (k, i, ' '.join('%.7f' % x for x in V.ravel()))
            geoms += '<geom type="mesh" mesh="%s%d" density="%g" friction="0.2 0.005 0.0001" solref="0.002 1" solimp="0.95 0.99 0.0005" margin="%g" condim="3"/>' % (k, i, density, margin)
        for i, V in enumerate((hands or {}).get(k, [])):
            assets += '<mesh name="hand_%s_%d" vertex="%s"/>' % (k, i, ' '.join('%.7f' % x for x in np.asarray(V).ravel()))
            geoms += '<geom name="hand_%s_%d" type="mesh" mesh="hand_%s_%d" density="0.001" friction="0.2 0.005 0.0001" solref="0.002 1" solimp="0.95 0.99 0.0005" condim="3"/>' % (k, i, k, i)
        bodies += '<body name="%s"%s><freejoint name="j%s"/>%s</body><body name="m%s" mocap="true"/>' % (
            k, ' gravcomp="1"' if gravcomp else '', k, geoms, k)
    eqs = ''.join('<weld name="w%s" body1="m%s" body2="%s" active="false" solref="0.02 1"/>' % (k, k, k) for k in g.KINDS)
    xml = ('<mujoco><option timestep="0.0005" gravity="0 0 0" integrator="implicitfast" cone="elliptic" noslip_iterations="3"/>'
           '<default><joint damping="0.002"/></default>'
           '<asset>%s</asset><worldbody>%s</worldbody><equality>%s</equality></mujoco>') % (assets, bodies, eqs)
    m = mujoco.MjModel.from_xml_string(xml)
    return m


class Sim:
    def __init__(self, g, gravcomp=False, hands=None):
        self.g = g
        self.m = build(g, gravcomp=gravcomp, hands=hands if hands is not None else hands_from_env())
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.qadr = {k: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'j' + k)] for k in g.KINDS}
        self.vadr = {k: m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'j' + k)] for k in g.KINDS}
        self.bid = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, k) for k in g.KINDS}
        self.mid = {k: m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'm' + k)] for k in g.KINDS}
        self.eid = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, 'w' + k) for k in g.KINDS}
        for k in g.KINDS:
            m.eq_data[self.eid[k], :] = 0
            m.eq_data[self.eid[k], 6] = 1
            m.eq_data[self.eid[k], 10] = 1

    def set_poses(self, poses):
        for k, T in poses.items():
            a = self.qadr[k]
            self.d.qpos[a:a + 3] = T[:3, 3]
            self.d.qpos[a + 3:a + 7] = quat(T[:3, :3])
            self.d.qvel[self.vadr[k]:self.vadr[k] + 6] = 0
        mujoco.mj_forward(self.m, self.d)

    def poses(self):
        out = {}
        for k in self.g.KINDS:
            T = np.eye(4)
            T[:3, :3] = self.d.xmat[self.bid[k]].reshape(3, 3)
            T[:3, 3] = self.d.xpos[self.bid[k]]
            out[k] = T
        return out

    def hold(self, k, T, solref=0.02):
        self.d.mocap_pos[self.mid[k]] = T[:3, 3]
        self.d.mocap_quat[self.mid[k]] = quat(T[:3, :3])
        self.m.eq_solref[self.eid[k]] = [solref, 1.0]
        self.d.eq_active[self.eid[k]] = 1

    def penetration(self):
        return max([-c.dist for c in self.d.contact[:self.d.ncon]] + [0.0])

    def drop_hand(self, k):
        """松手：那件零件身上的夹爪不再碰任何东西（手张开退走）"""
        for gi in range(self.m.ngeom):
            n = self.m.geom(gi).name
            if n.startswith('hand_%s_' % k):
                self.m.geom_contype[gi] = 0
                self.m.geom_conaffinity[gi] = 0

    def hand_contacts(self):
        """当前和夹爪碰着的（夹爪名, 另一个物体名, 深度 m）"""
        out = []
        for c in self.d.contact[:self.d.ncon]:
            n1, n2 = self.m.geom(c.geom1).name, self.m.geom(c.geom2).name
            if n1.startswith('hand_') or n2.startswith('hand_'):
                b1, b2 = self.m.body(self.m.geom_bodyid[c.geom1]).name, self.m.body(self.m.geom_bodyid[c.geom2]).name
                out.append((n1 if n1.startswith('hand_') else n2, b2 if n1.startswith('hand_') else b1, -c.dist))
        return out


if __name__ == '__main__':
    gname, spath, rpath, X, Y, secs, seed = sys.argv[1:8]
    g = importlib.import_module(gname)
    P = pz.Puzzle(g)
    rng = np.random.default_rng(int(seed))
    spath, _, sidx = spath.partition(':')
    d = pickle.load(open(spath, 'rb'))
    poses = d[int(sidx or 0)][1] if isinstance(d, list) else (d['path'][int(sidx)] if sidx else d['path'][d['best'][1]])
    Z = [k for k in 'THB' if k not in (X, Y)][0]
    iY = np.linalg.inv(poses[Y])
    poses = {k: iY @ v for k, v in poses.items()}
    ref = list(pickle.load(open(rpath, 'rb'))['rel'])
    if os.environ.get('PUSH_THEN_SLIT'):           # 参考路线走完后接上“躺着过缝”的推缝（沿 K2 的推缝方向 60 mm）
        import k1
        sgn = P.links(poses)[''.join(sorted(X + Y, key='THB'.index))]
        K2 = min(k1.lying(g, P, X, Y, sgn), key=lambda c: np.linalg.norm(c[0][:3, 3] - ref[-1][:3, 3]) + np.linalg.norm(c[0][:3, :3] - ref[-1][:3, :3]))
        K, D = K2[0], K2[1]
        ref += [K]
        for i in range(1, 61):
            T = K.copy()
            T[:3, 3] += D * i * MM
            ref.append(T)
    lk0 = P.links(poses)
    sim = Sim(g, gravcomp=GRAV is not None or GVEC is not None)
    if GRAV is not None or GVEC is not None:      # 拿在手里的两件由手托着（重力补偿），只有自由的那件受重力
        from physwalk import GDIRS
        gv = GDIRS[GRAV] if GVEC is None else np.array([float(v) for v in GVEC.split(',')])
        sim.m.opt.gravity[:] = 9.81 * gv / np.linalg.norm(gv)
        sim.m.body_gravcomp[sim.bid[Z]] = 0.0
    sim.set_poses(poses)
    sim.hold(Y, np.eye(4), solref=0.002)
    k = 0
    sim.hold(X, ref[0], solref=float(os.environ.get('PUSH_SOLREF', '0.02')))   # 有重力时用硬的（0.003），软的会被第三件压歪
    if os.environ.get('PUSH_ZHOME'):                 # 给 Z 一根很软的弹簧拉回开局位置（被推开后自己回位）
        sim.hold(Z, poses[Z], solref=float(os.environ['PUSH_ZHOME']))
    rec = [poses]
    t0 = time.time()
    stuck = 0
    maxpen = 0.0
    while k < len(ref) - 1 and time.time() - t0 < float(secs):
        tgt = ref[min(k + 1 + LOOK, len(ref) - 1)]
        sim.d.mocap_pos[sim.mid[X]] = tgt[:3, 3]
        sim.d.mocap_quat[sim.mid[X]] = quat(tgt[:3, :3])
        for _ in range(40):                       # 20 ms
            mujoco.mj_step(sim.m, sim.d)
        maxpen = max(maxpen, sim.penetration())
        cur = sim.poses()
        e_p = np.linalg.norm(cur[X][:3, 3] - tgt[:3, 3])
        e_r = np.linalg.norm(Rot.from_matrix(cur[X][:3, :3].T @ tgt[:3, :3]).as_rotvec())
        rec.append(cur)
        tk = ref[k + 1]
        e_p = np.linalg.norm(cur[X][:3, 3] - tk[:3, 3])
        e_r = np.linalg.norm(Rot.from_matrix(cur[X][:3, :3].T @ tk[:3, :3]).as_rotvec())
        if e_p < TOL_P and e_r < TOL_R:
            k += 1
            stuck = 0
            if k % 20 == 0:
                print('   step %d/%d, Z moved %.1f mm, max penetration %.2f mm (%.0f s)' % (
                    k, len(ref) - 1, np.linalg.norm(cur[Z][:3, 3] - poses[Z][:3, 3]) / MM, maxpen / MM, time.time() - t0), flush=True)
        else:
            stuck += 1
            if stuck % 25 == 0:               # 晃一晃 Z：一个随机的小冲量
                a = sim.vadr[Z]
                sim.d.qvel[a:a + 3] += rng.normal(size=3) * 0.02
                sim.d.qvel[a + 3:a + 6] += rng.normal(size=3) * 0.4
            if stuck % 500 == 0:
                print('   stuck at step %d (error %.2f mm, %.1f deg), max penetration %.2f mm (%.0f s)' % (
                    k, e_p / MM, np.degrees(e_r), maxpen / MM, time.time() - t0), flush=True)
    end = sim.poses()
    lk1 = P.links(end)
    print('reached step %d of %d; links %s -> %s; max penetration %.2f mm; solid collision at the end: %s (%.0f s)' % (
        k, len(ref) - 1, lk0, lk1, maxpen / MM, P.collides(end), time.time() - t0))
    tag = os.path.splitext(os.path.basename(spath))[0] + (sidx and '_' + sidx) + ('' if GRAV is None else '_g%d' % GRAV) + ('' if GVEC is None else '_gv') + os.environ.get('OUT_SUFFIX', '')
    pickle.dump(dict(path=rec, k=k, n=len(ref) - 1, maxpen=maxpen, gravity=GRAV), open(os.path.join(HERE, 'out/physpush_%s_%s_%s%s_%s.pkl' % (gname, tag, X, Y, seed)), 'wb'))
