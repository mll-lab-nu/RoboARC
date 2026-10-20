"""v12 的抓取点扫描：每件零件上一组候选抓法（ALOHA 夹爪的工具系在零件系里：x = 接近方向，y = 两指合拢方向，z = 被夹那段杆的走向），
对一段或几段路线里的每个状态，算真实夹爪（gripper/finger 连杆的碰撞凸包，指距按被夹处的厚度张开）到另外两件的最近距离，
报每个候选在全程里的最小值（分别对另一件拿着的零件、对自由的那件）。

用法：grasp_scan12.py <piece> <every> <seg.pkl> [<seg.pkl> ...]
  piece：要扫的零件（T/H/B），必须在每段里都是拿着的（X 或 Y）；every：每隔几个状态取一个。
输出：out/grasps12_<piece>_<tag>.pkl（候选列表 + 每个候选的最小距离），并打印最好的 15 个。"""
import os, sys, pickle, importlib
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import mujoco, trimesh, fcl

import puzzle as pz

HERE = os.path.dirname(os.path.abspath(__file__))
ALOHA = os.environ.get('ALOHA_XML', '/home/jianwen/topogym/assets/robots/aloha/aloha.xml')
MM = 1e-3
FINGER_OFFSET = 0.0157        # 两指指垫间距 = 2 x 指关节值 - FINGER_OFFSET（rotor_oracle.py 的同名常数）
OPEN_MARGIN = 0.001           # 扫描时每边多张开 1 mm


def bvh(mesh):
    m = fcl.BVHModel()
    m.beginModel(len(mesh.vertices), len(mesh.faces))
    m.addSubModel(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int32))
    m.endModel()
    return m


class Hand:
    """左臂夹爪（两臂的夹爪一样）的碰撞凸包，表示在工具点（site left/gripper）坐标系里；按指距缓存。"""

    def __init__(self):
        spec = mujoco.MjSpec.from_file(ALOHA)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'left/gripper')
        self.fq = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'left/' + j)] for j in ('left_finger', 'right_finger')]
        self.geoms = [g for g in range(m.ngeom) if m.geom_group[g] == 3 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                      and m.body(m.geom_bodyid[g]).name.startswith('left/')
                      and any(s in m.body(m.geom_bodyid[g]).name for s in ('gripper', 'finger'))]
        self.cache = {}

    def hull_verts(self, width):
        """夹爪碰撞凸包的顶点（工具点坐标系），指距按 width 张开（同 hulls，不加扫描用的 1 mm 余量以外的东西）"""
        m, d = self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, 0)
        q = np.clip((width + FINGER_OFFSET) / 2 + OPEN_MARGIN, 0.002, 0.037)
        for a in self.fq:
            d.qpos[a] = q
        mujoco.mj_kinematics(m, d)
        Sp, SR = d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()
        out = []
        for g in self.geoms:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
            w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            out.append(np.asarray(trimesh.convex.convex_hull((w - Sp) @ SR).vertices))
        return out

    def hulls(self, width):
        key = round(width, 4)
        if key in self.cache:
            return self.cache[key]
        m, d = self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, 0)
        q = np.clip((width + FINGER_OFFSET) / 2 + OPEN_MARGIN, 0.002, 0.037)
        for a in self.fq:
            d.qpos[a] = q
        mujoco.mj_kinematics(m, d)
        Sp, SR = d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()
        out = []
        for g in self.geoms:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
            w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            hull = trimesh.convex.convex_hull((w - Sp) @ SR)
            out.append(fcl.CollisionObject(bvh(hull), fcl.Transform()))
        self.cache[key] = out
        return out


def T4(x, y, p):
    x = x / np.linalg.norm(x)
    y = y - x * (x @ y)
    y = y / np.linalg.norm(y)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, y, np.cross(x, y)]
    T[:3, 3] = p
    return T


def candidates(g, kind):
    """(名字, 工具系 4x4（零件系）, 指距 m) 的列表。T/B 在零件系里生成；H 是 T 的镜像（y 取反后重新正交化，保持右手系）。"""
    if kind == 'H':
        M = np.diag([1.0, -1.0, 1.0])
        out = []
        for name, G, w in candidates(g, 'T'):
            x, y, p = M @ G[:3, 0], M @ G[:3, 1], M @ G[:3, 3]
            out.append((name, T4(x, -y, p), w))            # 镜像后 y 取反，工具系仍是右手系（指的合拢方向正反等价）
        return out
    out = []
    z = np.array([0.0, 0, 1])
    # 圆腿：x 从 45 到 112 mm，接近方向绕杆一周每 30°，两指沿 t x a 合拢；杆走向正反两种（绕接近方向转半圈）
    for xm in np.linspace(45, 112, 12):
        x = xm * MM
        c = np.array([x, g.window(x) + g.r_round(x), 0.0])
        c2 = np.array([x + 1 * MM, g.window(x + 1 * MM) + g.r_round(x + 1 * MM), 0.0])
        t = (c2 - c) / np.linalg.norm(c2 - c)
        n2 = np.cross(t, z)
        for th in range(0, 360, 30):
            a = np.cos(np.radians(th)) * z + np.sin(np.radians(th)) * n2
            for s in (1, -1):
                out.append(('rod x%.0f a%d %s' % (xm, th, '+' if s > 0 else '-'), T4(a, s * np.cross(t, a), c), 2 * g.r_round(x)))
    # 扁带：x 从 30 到 112 mm；夹厚度（两指沿 ±z，接近在零件平面里从外侧或从窗里）或夹宽度（接近沿 ±z）
    for xm in np.linspace(30, 112, 10):
        x = xm * MM
        c = np.array([x, -(g.window(x) + g.band_half_y(x)), 0.0])
        c2 = np.array([x + 1 * MM, -(g.window(x + 1 * MM) + g.band_half_y(x + 1 * MM)), 0.0])
        t = (c2 - c) / np.linalg.norm(c2 - c)
        n2 = np.cross(t, z)                      # 零件平面里垂直于带的方向（T 上指向 -y 外侧）
        for a, name in ((-n2, 'out'), (n2, 'in')):   # 'out'：手在带外侧（接近方向指向窗里）；'in'：手在窗里
            for s in (1, -1):
                out.append(('band x%.0f thick %s %s' % (xm, name, '+' if s > 0 else '-'), T4(a, s * z, c), g.band_t(x)))
        for a, name in ((z, 'top'), (-z, 'bottom')):
            for s in (1, -1):
                out.append(('band x%.0f wide %s %s' % (xm, name, '+' if s > 0 else '-'), T4(a, s * n2, c), g.band_w(x)))
    # 头
    if kind == 'T':
        gs = g.SLIT / 2
        for sy, hname in ((1, 'round-side head'), (-1, 'band-side head')):
            c = np.array([(g.X_HEAD + g.LEN) / 2, sy * (gs + g.HEAD_W / 2), 0.0])
            ey = np.array([0.0, sy, 0])
            ex = np.array([1.0, 0, 0])
            for s in (1, -1):                     # 接近方向从手掌指向零件：从外侧 = -ey，从头端 = -x
                out.append(('%s thick from side %s' % (hname, '+' if s > 0 else '-'), T4(-ey, s * z, c), g.HEAD_T))
                out.append(('%s thick from end %s' % (hname, '+' if s > 0 else '-'), T4(-ex, s * z, c + np.array([g.LEN - c[0] - 12 * MM, 0, 0])), g.HEAD_T))
                out.append(('%s wide from top %s' % (hname, '+' if s > 0 else '-'), T4(z, s * ey, c), g.HEAD_W))
                out.append(('%s wide from bottom %s' % (hname, '+' if s > 0 else '-'), T4(-z, s * ey, c), g.HEAD_W))
                out.append(('%s long from top %s' % (hname, '+' if s > 0 else '-'), T4(z, s * ex, c), g.LEN - g.X_HEAD))
    else:                                         # B：上下两块板各一组（板厚 B_PLATE，夹板厚，接近在板平面里）
        # 注意：夹一块板时总有一个手指要伸进两块板之间 5 mm 的缝，会插进 B 里（own_penetration 会把它们筛掉）；
        # 整个头块（两块板加起来，z 方向 23 mm）才夹得住。
        gs = g.SLIT / 2
        hz = gs + g.B_PLATE
        for yc in (-30 * MM, -24 * MM, 0.0, 24 * MM, 28 * MM):
            c = np.array([(g.X_HEAD + g.LEN) / 2 - 3 * MM, yc, 0.0])
            for a, aname in ((np.array([-1.0, 0, 0]), 'end'), (np.array([0, -1.0, 0]), '+y'), (np.array([0, 1.0, 0]), '-y')):
                for s_ in (1, -1):
                    out.append(('head block y%.0f from %s %s' % (yc / MM, aname, '+' if s_ > 0 else '-'), T4(a, s_ * z, c), 2 * hz))
        gs = g.SLIT / 2
        for zc, hname in ((gs + g.B_PLATE / 2, 'upper plate'), (-gs - g.B_PLATE / 2, 'lower plate')):
            for yc in (-12 * MM, 0.0, 12 * MM):
                c = np.array([(g.X_HEAD + g.LEN) / 2 - 3 * MM, yc, zc])
                for a, aname in ((np.array([-1.0, 0, 0]), 'end'), (np.array([0, -1.0, 0]), '+y'), (np.array([0, 1.0, 0]), '-y')):
                    for s in (1, -1):
                        out.append(('%s y%.0f from %s %s' % (hname, yc / MM, aname, '+' if s > 0 else '-'), T4(a, s * z, c), g.B_PLATE))
    return out


def own_penetration(g, P, hand, kind, G, w, n=1500):
    """夹爪（指距按 w 张开）伸进自己拿的那件零件多深（m）：夹爪各凸包表面采样点对零件网格的有符号距离的最大值（正 = 在里面）。
    夹对的抓法两个指垫只贴着零件表面（≈ 0）；手指伸进缝里或插进相邻的块就是几毫米。"""
    worst = 0.0
    for V in hand.hull_verts(w):
        h = trimesh.convex.convex_hull(V @ G[:3, :3].T + G[:3, 3])
        pts = np.vstack([h.vertices, h.sample(n)])
        worst = max(worst, float(trimesh.proximity.signed_distance(P.p[kind].mesh, pts).max()))
    return worst


_CELL_EQ = {}


def own_penetration_fast(g, hand, kind, G, w, n=300):
    """own_penetration 的快速版：夹爪凸包表面采样点（含顶点）落在零件哪个凸块里、离那个凸块表面多深，和零件凸块顶点落在
    夹爪凸包里多深，两者取大（m）。点挨着两个凸块的公共面时会低估，用来筛几毫米的插入足够。"""
    from scipy.spatial import ConvexHull
    if kind not in _CELL_EQ:
        cs = [np.asarray(V, float) for V in g.cells(kind)]
        _CELL_EQ[kind] = ([ConvexHull(V).equations for V in cs], np.vstack(cs))
    eqs, cell_pts = _CELL_EQ[kind]
    worst = 0.0
    for V in hand.hull_verts(w):
        Vp = V @ G[:3, :3].T + G[:3, 3]
        h = trimesh.convex.convex_hull(Vp)
        pts = np.vstack([h.vertices, h.sample(n)])
        for E in eqs:
            s_ = pts @ E[:, :3].T + E[:, 3]                     # <= 0 在凸块里
            inside = (s_ <= 0).all(axis=1)
            if inside.any():
                worst = max(worst, float((-s_[inside]).min(axis=1).max()))
        He = ConvexHull(Vp).equations
        s_ = cell_pts @ He[:, :3].T + He[:, 3]
        inside = (s_ <= 0).all(axis=1)
        if inside.any():
            worst = max(worst, float((-s_[inside]).min(axis=1).max()))
    return worst


def dist(a, b):
    if fcl.collide(a, b, fcl.CollisionRequest(), fcl.CollisionResult()):
        return -1.0
    r = fcl.DistanceResult()
    fcl.distance(a, b, fcl.DistanceRequest(), r)
    return r.min_distance


if __name__ == '__main__':
    import geom12 as g
    kind, every = sys.argv[1], int(sys.argv[2])
    P = pz.Puzzle(g)
    states = []
    for f in sys.argv[3:]:
        d = pickle.load(open(f, 'rb'))
        X, Y = d.get('X'), d.get('Y')
        if X is None:                              # physpush 的输出没存 X、Y：文件名里有 _<X><Y>_<seed>
            X, Y = os.path.basename(f).rsplit('_', 2)[-2]
        Z = [k for k in 'THB' if k not in (X, Y)][0]
        assert kind in (X, Y), (f, X, Y)
        other = Y if kind == X else X
        for s in d['path'][::every]:
            states.append((s, other, Z))
    hand = Hand()
    cands = candidates(g, kind)
    print('%d candidates, %d states' % (len(cands), len(states)), flush=True)
    res = []
    for ci, (name, G, w) in enumerate(cands):
        hs = hand.hulls(w)
        m_other, m_free = np.inf, np.inf
        for s, other, Z in states:
            W = s[kind] @ G
            tf = fcl.Transform(W[:3, :3], W[:3, 3])
            for h in hs:
                h.setTransform(tf)
            P.place(s)
            m_other = min(m_other, min(dist(h, P.p[other].obj) for h in hs))
            if not os.environ.get('SCAN_NOFREE') and np.linalg.norm(s[Z][:3, 3] - s[kind][:3, 3]) < 0.5:   # 掉远了的（或已放到桌上的）自由件不算
                m_free = min(m_free, min(dist(h, P.p[Z].obj) for h in hs))
            if m_other < 0 and m_free < 0:
                break
        res.append((name, G, w, m_other, m_free))
        if ci % 50 == 0:
            print('   %d/%d' % (ci, len(cands)), flush=True)
    res.sort(key=lambda r: -min(r[3], r[4]))
    tag = '_'.join(os.path.splitext(os.path.basename(f))[0][-24:] for f in sys.argv[3:])
    pickle.dump(res, open(os.path.join(HERE, 'out/grasps12_%s_%s.pkl' % (kind, tag[:120])), 'wb'))
    for name, G, w, a, b in res[:15]:
        print('%-40s width %4.1f mm   to the other held piece %6.1f mm   to the free piece %6.1f mm' % (name, w / MM, a / MM, b / MM))
    print('candidates with both >= 5 mm: %d of %d' % (sum(1 for r in res if min(r[3], r[4]) >= 5 * MM), len(res)))
