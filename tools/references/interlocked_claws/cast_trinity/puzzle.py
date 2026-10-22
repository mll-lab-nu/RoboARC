"""三件的碰撞、间隙和环绕数（规划用）。每件 = 它的精确凸块并成的整件网格（FCL BVH），和 MuJoCo 里碰撞的是同一个实体。
geom 模块要提供 KINDS、cells(kind)、centerline(kind)。"""
import numpy as np
import fcl

import solid

PAIRS = (('T', 'H'), ('T', 'B'), ('H', 'B'))


class Piece:
    def __init__(self, kind, cells):
        self.kind = kind
        self.cells = cells
        self.mesh = solid.union(cells)
        m = fcl.BVHModel()
        m.beginModel(len(self.mesh.vertices), len(self.mesh.faces))
        m.addSubModel(np.asarray(self.mesh.vertices), np.asarray(self.mesh.faces))
        m.endModel()
        self.obj = fcl.CollisionObject(m, fcl.Transform())

    def place(self, T):
        self.obj.setTransform(fcl.Transform(T[:3, :3], T[:3, 3]))


def linking(A, B):
    a0, a1 = A, np.roll(A, -1, axis=0)
    b0, b1 = B, np.roll(B, -1, axis=0)
    r00, r01, r10, r11 = b0[None] - a0[:, None], b1[None] - a0[:, None], b0[None] - a1[:, None], b1[None] - a1[:, None]

    def un(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)
    n1, n2, n3, n4 = un(np.cross(r00, r01)), un(np.cross(r01, r11)), un(np.cross(r11, r10)), un(np.cross(r10, r00))
    s = sum(np.arcsin(np.clip(np.sum(p * q, -1), -1, 1)) for p, q in ((n1, n2), (n2, n3), (n3, n4), (n4, n1)))
    sign = np.sign(np.sum(np.cross((a1 - a0)[:, None], (b1 - b0)[None]) * r00, -1))
    return float(np.sum(s * sign) / (4 * np.pi))


def tf(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


class Puzzle:
    def __init__(self, geom):
        self.g = geom
        self.p = {k: Piece(k, geom.cells(k)) for k in geom.KINDS}
        self.lines = {k: geom.centerline(k) for k in geom.KINDS}

    def place(self, poses):
        for k, T in poses.items():
            self.p[k].place(T)

    def collides(self, poses, pairs=PAIRS):
        self.place(poses)
        for a, b in pairs:
            if fcl.collide(self.p[a].obj, self.p[b].obj, fcl.CollisionRequest(), fcl.CollisionResult()):
                return True
        return False

    def clearance(self, poses, pairs=PAIRS):
        self.place(poses)
        out = {}
        for a, b in pairs:
            if fcl.collide(self.p[a].obj, self.p[b].obj, fcl.CollisionRequest(), fcl.CollisionResult()):
                out[a + b] = -1.0
                continue
            res = fcl.DistanceResult()
            out[a + b] = fcl.distance(self.p[a].obj, self.p[b].obj, fcl.DistanceRequest(), res)
        return out

    def links(self, poses):
        return {a + b: int(round(linking(tf(poses[a], self.lines[a]), tf(poses[b], self.lines[b])))) for a, b in PAIRS}
