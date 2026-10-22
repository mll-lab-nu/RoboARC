"""零件的精确凸块（给 MuJoCo 的 profile split 用，规划也用同一套）：凸块之间只在公共平面上相接、互不重叠，体积加起来等于整件。

tube(pts, r, ...)   多边形截面的杆，沿折线 pts（r 可逐点给）；相邻两段在角平分面上斜切拼接；两端可以给切平面（n, d：保留 n.x <= d 的一侧）。
band(pts, twist, w, t)  扭扁的带：每个截面是 w x t 的长方形，绕折线方向转 twist（度，0 = 宽度方向在 xy 面内）；
                    相邻截面之间取凸包（两个长方形的凸包是凸的），公共截面是同一个长方形，所以严丝合缝。
hull(points)        一个凸块（头）。
单位米。"""
import numpy as np
from scipy.spatial import ConvexHull


def _frame(d, up=(0.0, 0.0, 1.0)):
    """d 方向的单位向量和与之垂直的两个方向 (e1, e2)：e2 尽量靠近 up。"""
    d = d / np.linalg.norm(d)
    up = np.asarray(up, float)
    e1 = np.cross(up, d)
    if np.linalg.norm(e1) < 1e-9:
        e1 = np.cross([1.0, 0, 0], d)
    e1 /= np.linalg.norm(e1)
    return d, e1, np.cross(d, e1)


def _cut_polygon(c, d, e1, e2, r, m, n, p):
    """以 c 为中心、轴向 d、半径 r 的正 m 边形柱面，和过 p、法向 n 的平面的交线（m 个点）。"""
    ang = 2 * np.pi * (np.arange(m) + 0.5) / m
    out = []
    for a in ang:
        q = c + r * (np.cos(a) * e1 + np.sin(a) * e2)      # 柱面上的一条母线 q + s d
        s = np.dot(n, p - q) / np.dot(n, d)
        out.append(q + s * d)
    return np.array(out)


def tube(pts, r, m=12, start=None, end=None, up=(0.0, 0.0, 1.0)):
    """返回凸块列表。start / end：(法向 n, 平面上一点 p)，默认是垂直于首末段、过首末点的平面。
    多边形的顶点在半径 r / cos(pi/m) 上，各边和半径 r 的圆相切（杆不比标称细）。"""
    pts = [np.asarray(p, float) for p in pts]
    R = np.broadcast_to(np.asarray(r, float), (len(pts),)) / np.cos(np.pi / m)     # r 可以逐点给（变粗细的杆）
    cells = []
    nseg = len(pts) - 1
    dirs = [(pts[i + 1] - pts[i]) / np.linalg.norm(pts[i + 1] - pts[i]) for i in range(nseg)]
    planes = []
    for i in range(nseg + 1):
        if i == 0:
            planes.append(start if start is not None else (dirs[0], pts[0]))
        elif i == nseg:
            planes.append(end if end is not None else (dirs[-1], pts[-1]))
        else:
            nb = dirs[i - 1] + dirs[i]
            planes.append((nb / np.linalg.norm(nb), pts[i]))
    # 每个切面上只算一个多边形（用进入这个切面的那一段的柱面），相邻两块共用它，接缝严丝合缝
    polys = []
    for j in range(nseg + 1):
        i = max(j - 1, 0)
        d, e1, e2 = _frame(dirs[i], up)
        polys.append(_cut_polygon(pts[i], d, e1, e2, R[j], m, *planes[j]))
    for i in range(nseg):
        cells.append(np.vstack([polys[i], polys[i + 1]]))
    return cells


def _rect(c, d, twist_deg, w, t, up=(0.0, 0.0, 1.0)):
    d, e1, e2 = _frame(d, up)
    a = np.radians(twist_deg)
    u = np.cos(a) * e1 + np.sin(a) * e2          # 宽度方向
    v = np.cross(d, u)                            # 厚度方向
    return np.array([c + su * w / 2 * u + sv * t / 2 * v for su in (-1, 1) for sv in (-1, 1)])


def _on_plane(R, d, n, p):
    """把长方形 R 的四个角沿 d 方向投到过 p、法向 n 的平面上。"""
    return np.array([q + np.dot(n, p - q) / np.dot(n, d) * d for q in R])


def band(pts, twist, w, t, up=(0.0, 0.0, 1.0), start=None, end=None):
    """扭带：每个点一个截面（垂直于相邻两段的平均方向），相邻截面取凸包。
    start / end：(法向 n, 平面上一点 p)，给了就把首 / 末截面沿带的方向投到这个平面上（和相邻的杆共用切面）。"""
    pts = [np.asarray(p, float) for p in pts]
    n = len(pts)
    secs = []
    for i in range(n):
        d = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
        d = d / np.linalg.norm(d)
        R = _rect(pts[i], d, twist[i], w, t, up)
        if i == 0 and start is not None:
            R = _on_plane(R, d, *start)
        if i == n - 1 and end is not None:
            R = _on_plane(R, d, *end)
        secs.append(R)
    return [np.vstack([secs[i], secs[i + 1]]) for i in range(n - 1)]


def volume(V):
    return ConvexHull(V).volume


def _clip(V, n, d):
    """conv(V) 里 n.x <= d 的部分（顶点），空了返回 None。"""
    s = V @ n - d
    if (s <= 1e-12).all():
        return V
    if (s >= -1e-12).all():
        return None
    h = ConvexHull(V)
    edges = {tuple(sorted((a, b))) for t in h.simplices for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    out = [V[s <= 1e-12]] + [(V[i] + s[i] / (s[i] - s[j]) * (V[j] - V[i]))[None] for i, j in edges
                             if min(s[i], s[j]) < -1e-12 and max(s[i], s[j]) > 1e-12]
    Q = np.unique(np.vstack(out), axis=0)
    try:
        h = ConvexHull(Q)
    except Exception:
        return None
    return Q[h.vertices] if h.volume > 1e-18 else None


def overlap(A, B):
    """conv(A) 和 conv(B) 重叠部分的体积。"""
    eq = ConvexHull(B).equations
    Q = A
    for e in eq:
        Q = _clip(Q, e[:3], -e[3])
        if Q is None:
            return 0.0
    return ConvexHull(Q).volume


def max_overlap(cells):
    """所有凸块两两重叠体积的最大值和对应的一对（包围盒不相交的跳过）。"""
    lo = [V.min(0) for V in cells]
    hi = [V.max(0) for V in cells]
    worst = (0.0, None)
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            if (lo[i] > hi[j]).any() or (lo[j] > hi[i]).any():
                continue
            v = overlap(cells[i], cells[j])
            if v > worst[0]:
                worst = (v, (i, j))
    return worst
