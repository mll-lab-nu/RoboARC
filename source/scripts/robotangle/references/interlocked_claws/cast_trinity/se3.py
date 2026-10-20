"""位姿小工具：4x4 齐次矩阵。"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp


def T4(R=None, p=(0, 0, 0)):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = p
    return T


def rotvec(v):
    return Rot.from_rotvec(v).as_matrix()


def rx(a):
    return rotvec([a, 0, 0])


def ry(a):
    return rotvec([0, a, 0])


def rz(a):
    return rotvec([0, 0, a])


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def dist(A, B, wrot=0.05):
    """两个位姿的距离（米）：平移 + wrot 米/弧度 × 转角。"""
    return np.linalg.norm(A[:3, 3] - B[:3, 3]) + wrot * angle(A, B)


def angle(A, B):
    return np.linalg.norm(Rot.from_matrix(A[:3, :3].T @ B[:3, :3]).as_rotvec())


def interp(A, B, s):
    """A 到 B 的插值：平移线性、转动走最短弧。"""
    r = Rot.from_matrix(A[:3, :3].T @ B[:3, :3]).as_rotvec()
    return T4(A[:3, :3] @ rotvec(s * r), (1 - s) * A[:3, 3] + s * B[:3, 3])


def steps(A, B, dpos=0.002, drot=np.radians(1.5)):
    n = int(np.ceil(max(np.linalg.norm(A[:3, 3] - B[:3, 3]) / dpos, angle(A, B) / drot, 1)))
    return [interp(A, B, i / n) for i in range(1, n + 1)]


def rand_rot(rng):
    return Rot.random(random_state=rng).as_matrix()
