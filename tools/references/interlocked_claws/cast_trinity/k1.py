"""缝过薄处的起点 K1：X（mover）垂直于 Y（base）站着，X 的头旁圆孔套住 Y 的两个头，X 的薄颈在 Y 的缝口（Y 的 x 约 108 mm）。
沿 Y 的 +x 推 SLIDE 就过缝脱开。两种：尖朝上（tip up）和尖朝下（tip down），环绕数正负相反；按当前这一对的环绕数挑。
poses(g, X, Y) -> [(位姿（Y 坐标系）, 环绕数)]。"""
import numpy as np

import puzzle as pz

MM = 1e-3
SLIDE = 70 * MM


def candidates(X):
    """X 在 Y 坐标系里的两种 K1（X 的扁带在 -y：T、B；在 +y：H）。"""
    cx, cz = 134 * MM, 105 * MM
    if X == 'H':
        up = np.c_[[0, 0, -1.0], [-1.0, 0, 0], [0, 1.0, 0]]
        down = np.c_[[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]]
    else:
        up = np.c_[[0, 0, -1.0], [1.0, 0, 0], [0, -1.0, 0]]
        down = np.c_[[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]]
    out = []
    for R, z in ((up, cz), (down, -cz)):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [cx, 0, z]
        out.append(T)
    return out


def pick(g, X, Y, sign):
    """环绕数为 sign 的那个 K1（X 相对 Y）。"""
    LX, LY = g.centerline(X), g.centerline(Y)
    for T in candidates(X):
        lk = pz.linking(LY, pz.tf(T, LX))
        if int(round(lk)) == sign:
            return T
    return None


def lying(g, P, X, Y, sign, x0s=(36, 40, 44), z0s=(0, -4, 4), n=60, reversed_too=True):
    """正向（X 和 Y 同向）和反向（X 和 Y 反向，X 的尖在 Y 头端外，X 下压时它的两个头穿过 Y 的头旁圆孔）两种，合在一起按间隙排序。"""
    out = _lying(g, P, X, Y, sign, 1.0, x0s, z0s, n)
    if reversed_too:
        out += _lying(g, P, X, Y, sign, -1.0, (228, 232, 236), (0, -4, 4), 70)
    return sorted(out, key=lambda o: -o[2])


def _lying(g, P, X, Y, sign, dirx, x0s, z0s, n):
    """“躺着过缝”K2：X 和 Y 同向（X 的 x 轴 = Y 的 x 轴），X 的平面 = Y 的 x-z 面，X 的尖在 Y 的尖前方 x0 mm，
    Y 的两个头在 X 的头旁圆孔里，X 的薄颈在 Y 的头上方或下方；X 沿 ±z 推 60 mm 过缝。
    返回 [(位姿（Y 坐标系）, 推的方向（Y 坐标系单位向量）, 推的过程中最小间隙)]，按间隙从大到小。"""
    far = np.eye(4)
    far[:3, 3] = [1, 1, 1]
    Z = [k for k in 'THB' if k not in (X, Y)][0]
    LX, LY = g.centerline(X), g.centerline(Y)
    out = []
    for side in (1, -1):
        bx, bz = np.array([dirx, 0, 0]), np.array([0, float(side), 0])
        R = np.c_[bx, np.cross(bz, bx), bz]
        for x0 in x0s:
            for z0 in z0s:
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = [x0 * MM, 0, z0 * MM]
                if int(round(pz.linking(LY, pz.tf(T, LX)))) != sign:
                    continue
                c0 = P.clearance({Y: np.eye(4), Z: far, X: T}, ((Y, X),) if 'THB'.index(Y) < 'THB'.index(X) else ((X, Y),))
                if min(c0.values()) <= 0:
                    continue
                for sgn in (1, -1):
                    ok, cmin = True, 1.0
                    for i in range(1, n + 1):
                        U = T.copy()
                        U[2, 3] += sgn * i * MM
                        c = P.clearance({Y: np.eye(4), Z: far, X: U}, ((Y, X),) if 'THB'.index(Y) < 'THB'.index(X) else ((X, Y),))
                        if min(c.values()) <= 0:
                            ok = False
                            break
                        cmin = min(cmin, min(c.values()))
                    if ok and abs(pz.linking(LY, pz.tf(U, LX))) < 0.5:
                        out.append((T, np.array([0, 0, float(sgn)]), cmin))
    return sorted(out, key=lambda o: -o[2])
