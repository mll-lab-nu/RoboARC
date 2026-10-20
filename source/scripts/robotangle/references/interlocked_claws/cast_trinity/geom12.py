"""Cast Trinity 零件 v12（v11 的微调：尖端窄槽放宽到 16 mm，三件钩在开局里有让路的余地）。
v11（v10 的微调：薄颈加长，从头根往前 48 mm 都是 4 mm 厚，可以顺着另一件躺着从上往下压过它的缝）。
v10（v9 的微调：尖端窄槽放宽到 13 mm；v9 时三件钩在 11 mm 的槽里几乎滑不动）。
v9（v8 的微调：尖端窄槽放宽到 11 mm、扁带减到 5.5 mm（仍比 5 mm 的缝厚），三件尖端才钩得进）。
v8（v7 的微调：B 的两个头改成上下咬合的两个 L 形，缝在中间是横的（转 90°），腿不再抬高/压低）。
v7（v6 的微调：扁带只有靠头根的薄颈 NECK_T 厚，其余 BAND_T 厚，比头间缝厚——只有薄颈过得了缝）。
v6：扭带腿全程平躺，靠尖端变窄；钩在窄槽里时不再被立着的扭带挡住。
v5 原注释：Cast Trinity 零件 v5：按官方解法的机关建（ref/official_sequence.md），精确凸块拼成（cells.py）。单位米。

机关：
  开局三件的尖端在中心互相钩住：每件的尖端是细圆杆弯成的钩，插在别件尖端窄槽里（窄槽只容细杆，圆腿、头都过不去）。
  中间几步只换穿的位置（窄槽 -> 头旁圆孔、头端穿过圆孔），套着关系不变。
  分开靠“缝过薄处”：一件扭带腿在头根处平躺成的薄颈（厚 NECK_T）滑过另一件两个头之间的缝（宽 SLIT > NECK_T）。
  T、H 的缝顺着零件长度方向竖着（缝面 = x-z 面）；B 的缝转 90°（缝面 = x-y 面，两个头上下叠）。

零件坐标系：x 从尖端（x = 0）指向头端；y 横向（圆腿在 +y，扭带腿在 -y；H 是 T 的镜像）；z 垂直于零件平面。
  尖钩    中心线半径 RHO = SLOT_HALF + R_TIP 的半圆，从 (RHO, +RHO) 经 (0, 0) 到 (RHO, -RHO)，杆半径 R_TIP。
  圆腿    杆半径从 R_TIP 渐粗到 R_LEG，内侧贴着窗边 y = window(x)。
  扭带腿  靠尖端一段是细杆（到 X_BAND0），之后是 BAND_W x NECK_T 的扁带：先立着（宽度沿 z）进窄槽段，再扭平（宽度在零件平面里），
          到头根一直平躺 = 薄颈。内侧贴着窗边 y = -window(x)。
  窗      window(x) = 半宽：x < X_SLOT_END 是窄槽 SLOT_HALF，之后张成头旁圆孔（最宽 ROUND_HALF），头端由两个头封住。
  头      x in [X_HEAD, LEN]。T、H：两个头沿 y 分开（缝面 x-z）；B：两个头沿 z 上下叠（缝面 x-y），扭带腿接上面的头、圆腿接下面的头。"""
import numpy as np

import cells as C

MM = 1e-3
LEN = 150 * MM
X_HEAD = 118 * MM
SLOT_HALF = 8.0 * MM
X_SLOT_END = 62 * MM
X_ROUND_MID = 96 * MM
ROUND_HALF = 20 * MM
R_TIP = 3.0 * MM
RHO = SLOT_HALF + R_TIP
R_LEG = 7.0 * MM
X_THICK = 40 * MM          # 圆腿到这里长到 R_LEG
X_BAND0 = 22 * MM          # 扭带腿从这里开始是扁带
BAND_W = 18 * MM
BAND_W0 = 8 * MM            # 扁带靠尖端的宽
NECK_T = 4.0 * MM          # 薄颈厚（< SLIT）
BAND_T = 5.5 * MM          # 扁带其余部分的厚（> SLIT）
X_NECK = 66 * MM           # 薄颈从这里到头根
X_FLAT = 88 * MM           # 扁带到这里扭平
SLIT = 5.0 * MM
HEAD_W = 28 * MM
HEAD_T = 20 * MM
B_LIFT = 0.0
B_PLATE = 9.0 * MM         # B 每块板的厚
Y_TAB = 18.0 * MM          # B 两块竖片的内侧离中线的距离
KINDS = ('T', 'H', 'B')
N_LEG = 16
N_ARC = 10


def window(x):
    if x <= X_SLOT_END:
        return SLOT_HALF
    if x <= X_ROUND_MID:
        u = (x - X_SLOT_END) / (X_ROUND_MID - X_SLOT_END)
        return SLOT_HALF + (ROUND_HALF - SLOT_HALF) * np.sin(u * np.pi / 2)
    u = (x - X_ROUND_MID) / (X_HEAD - X_ROUND_MID)
    return ROUND_HALF * (1 - 0.15 * u * u)


def r_round(x):
    u = np.clip((x - RHO) / (X_THICK - RHO), 0, 1)
    return R_TIP + (R_LEG - R_TIP) * (3 * u * u - 2 * u ** 3)


def twist(x):
    """扁带的扭角（度）：0 = 平躺（v6 全程平躺）。"""
    return 0.0


def band_t(x):
    """扁带厚：X_NECK 之前 BAND_T，之后 6 mm 内收到 NECK_T。"""
    u = np.clip((x - X_NECK) / (6 * MM), 0, 1)
    return BAND_T + (NECK_T - BAND_T) * u


def band_w(x):
    """扁带宽：靠尖端 BAND_W0，到 X_FLAT 长到 BAND_W。"""
    u = np.clip((x - X_BAND0) / (X_FLAT - X_BAND0), 0, 1)
    return BAND_W0 + (BAND_W - BAND_W0) * (3 * u * u - 2 * u ** 3)


def band_half_y(x):
    a = np.radians(twist(x))
    return band_w(x) / 2 * abs(np.cos(a)) + band_t(x) / 2 * abs(np.sin(a))


def lift(x, kind):
    return 0.0          # v8：B 的腿不再抬高/压低（头改成 L 形）
    if kind != 'B':
        return 0.0
    u = np.clip((x - (X_HEAD - 22 * MM)) / (22 * MM), 0, 1)
    return B_LIFT * (3 * u * u - 2 * u ** 3)


def _xs(x0, x1, n):
    return np.linspace(x0, x1, n + 1)


def chain_rod(kind):
    """圆腿（头根 -> 尖钩起点）+ 尖钩 + 扭带腿的细杆段（-> X_BAND0）：点和逐点半径。"""
    pts, rad = [], []
    for x in _xs(X_HEAD, RHO, N_LEG)[:-1]:
        pts.append(np.array([x, window(x) + r_round(x), -lift(x, kind)]))
        rad.append(r_round(x))
    for k in range(N_ARC + 1):
        f = np.pi * k / N_ARC
        pts.append(np.array([RHO - RHO * np.sin(f), RHO * np.cos(f), 0.0]))
        rad.append(R_TIP)
    for x in _xs(RHO, X_BAND0, 3)[1:]:
        u = (x - RHO) / (X_BAND0 - RHO)
        r = R_TIP + (BAND_T / 2 - R_TIP) * u
        pts.append(np.array([x, -(window(x) + r), 0.0]))
        rad.append(r)
    return pts, rad


def chain_band(kind):
    xs = _xs(X_BAND0, X_HEAD, N_LEG)
    pts = [np.array([x, -(window(x) + band_half_y(x)), lift(x, kind)]) for x in xs]
    return pts, [twist(x) for x in xs]


def _box(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def heads(kind):
    g = SLIT / 2
    if kind in ('T', 'H'):
        return [_box([X_HEAD, g, -HEAD_T / 2], [LEN, g + HEAD_W, HEAD_T / 2]),
                _box([X_HEAD, -g - HEAD_W, -HEAD_T / 2], [LEN, -g, HEAD_T / 2])]
    # B：两个 L 形头上下咬合。上头 = 上板（z in [g, g + B_PLATE]，y in [-yb, Y_TAB - SLIT]）+ 扁带一侧的竖片（y in [-yb, -Y_TAB]，
    # z 往下到 -g - B_PLATE）；下头 = 下板（z in [-g - B_PLATE, -g]，y in [-(Y_TAB - SLIT), yr]）+ 圆腿一侧的竖片（y in [Y_TAB, yr]，z 往上）。
    # 两头之间的缝处处宽 SLIT：中间一段横的（z in [-g, g]），两端各一段竖的（|y| in [Y_TAB - SLIT, Y_TAB]）。上头大一号（长到 LEN）。
    yb = window(X_HEAD) + band_w(X_HEAD) + 4 * MM
    yr = window(X_HEAD) + 2 * R_LEG + 4 * MM
    P = B_PLATE
    return [_box([X_HEAD, -yb, g], [LEN, Y_TAB - 2 * g, g + P]),
            _box([X_HEAD, -yb, -g - P], [LEN, -Y_TAB, g]),
            _box([X_HEAD, -(Y_TAB - 2 * g), -g - P], [LEN - 6 * MM, yr, -g]),
            _box([X_HEAD, Y_TAB, -g], [LEN - 6 * MM, yr, g + P])]


def band_var(pts, tw, widths, t, start=None, end=None):
    """宽度、厚度逐点变化的扁带（cells.band 的逐截面版）；t 是逐点厚度。"""
    pts = [np.asarray(p, float) for p in pts]
    n = len(pts)
    secs = []
    for i in range(n):
        d = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
        d = d / np.linalg.norm(d)
        R = C._rect(pts[i], d, tw[i], widths[i], t[i])
        if i == 0 and start is not None:
            R = C._on_plane(R, d, *start)
        if i == n - 1 and end is not None:
            R = C._on_plane(R, d, *end)
        secs.append(R)
    return [np.vstack([secs[i], secs[i + 1]]) for i in range(n - 1)]


def cells(kind):
    if kind == 'H':
        M = np.diag([1.0, -1.0, 1.0])
        return [V @ M for V in cells('T')]
    hcut = (np.array([-1.0, 0, 0]), np.array([X_HEAD, 0, 0]))      # 圆腿起点切面 x = X_HEAD（保留 x <= X_HEAD）
    pts, rad = chain_rod(kind)
    bp, tw = chain_band(kind)
    d = bp[1] - bp[0]
    joint = (d / np.linalg.norm(d), bp[0])                          # 细杆和扁带共用的切面
    out = C.tube(pts, rad, start=hcut, end=joint)
    out += band_var(bp, tw, [band_w(p[0]) for p in bp], [band_t(p[0]) for p in bp], start=joint, end=(np.array([1.0, 0, 0]), np.array([X_HEAD, 0, 0])))
    out += heads(kind)
    return out


def centerline(kind):
    """闭合中心线：圆腿 + 尖钩 + 扭带腿 + 跨头间缝的一段（T、H 横跨竖缝，B 竖跨横缝）。"""
    if kind == 'H':
        L = centerline('T')
        L[:, 1] *= -1
        return L
    pts, _ = chain_rod(kind)
    bp, _ = chain_band(kind)
    L = pts + bp[1:]
    xc = X_HEAD + 12 * MM
    a, b = L[-1], L[0]
    close = [np.array([xc, a[1], a[2]]), np.array([xc, 0.0, 0.5 * (a[2] + b[2])]), np.array([xc, b[1], b[2]])]
    return np.array(L + close)
