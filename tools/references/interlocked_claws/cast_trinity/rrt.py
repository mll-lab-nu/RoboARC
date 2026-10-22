"""一件零件相对其余（固定）零件运动的规划：双向 RRT（RRT-Connect，Kuffner & LaValle 2000）+ 抄近路。位姿是 4x4 矩阵。"""
import time
import numpy as np
import se3

DPOS, DROT = 0.0015, np.radians(1.0)      # 相邻两个检查点最多差 1.5 mm、1 度


class Planner:
    def __init__(self, puzzle, mover, fixed, pairs, center, half, rng=None, margin=0.0, extra=None):
        """fixed: 其余零件的位姿 dict；pairs: 要查碰撞的零件对；采样范围是 center ± half（米）的盒子。
        margin > 0 时要求零件间隙不小于 margin（米），否则只要求不相碰。"""
        self.margin = margin
        self.extra = extra          # 额外的可行性判断（例如握着零件的手不碰别的件），输入 mover 位姿，返回 True/False
        self.P, self.mover, self.fixed, self.pairs = puzzle, mover, dict(fixed), pairs
        self.center, self.half = np.asarray(center, float), np.asarray(half, float)
        self.rng = rng or np.random.default_rng(0)
        self.checks = 0
        self.near = None

    def free(self, T):
        self.checks += 1
        poses = dict(self.fixed)
        poses[self.mover] = T
        if self.margin > 0:
            ok = min(self.P.clearance(poses, self.pairs).values()) >= self.margin
        else:
            ok = not self.P.collides(poses, self.pairs)
        return ok and (self.extra is None or self.extra(T))

    def clearance(self, T):
        poses = dict(self.fixed)
        poses[self.mover] = T
        return min(self.P.clearance(poses, self.pairs).values())

    def escape(self, A, reach=0.012, tries=None):
        """起点贴着别的件（间隙不足 margin）时，先找一个附近的、间隙够的位姿，直线退到那里（途中不许相碰、间隙不许变小超过 0.2 mm）。"""
        if self.free(A):
            return [A]
        if tries is None:
            tries = 800 if self.extra is not None else 3000
        c0 = self.clearance(A)
        best = None
        for i in range(tries):
            sc = reach * (0.3 + 0.7 * i / tries)
            B = se3.T4(A[:3, :3] @ se3.rotvec(self.rng.normal(size=3) * sc * 8), A[:3, 3] + self.rng.normal(size=3) * sc)
            if not self.free(B):
                continue
            st = se3.steps(A, B, DPOS, DROT)
            cl = [self.clearance(T) for T in st]
            if min(cl) < min(c0, 0.0) - 2e-4 or min(cl) < 0 and c0 >= 0:
                continue
            d = se3.dist(A, B)
            if best is None or d < best[0]:
                best = (d, st)
                if i > tries // 3:
                    break
        return None if best is None else [A] + best[1]

    def seg_free(self, A, B):
        return all(self.free(T) for T in se3.steps(A, B, DPOS, DROT))

    def sample(self):
        if self.near is not None and self.rng.random() < 0.7:      # 多数样本取在起点与终点连线附近
            A, B, sp, sr = self.near
            M = se3.interp(A, B, self.rng.random())
            return se3.T4(M[:3, :3] @ se3.rotvec(self.rng.normal(size=3) * sr), M[:3, 3] + self.rng.normal(size=3) * sp)
        return se3.T4(se3.rand_rot(self.rng), self.center + self.rng.uniform(-1, 1, 3) * self.half)

    @staticmethod
    def _nearest(nodes, T):
        P = np.array([n[:3, 3] for n in nodes])
        d = np.linalg.norm(P - T[:3, 3], axis=1)
        idx = np.argsort(d)[:12]
        return min(idx, key=lambda i: se3.dist(nodes[i], T))

    def _extend(self, tree, parent, target, greedy):
        i = self._nearest(tree, target)
        cur = tree[i]
        reached = False
        for T in se3.steps(cur, target, DPOS, DROT)[:(10 ** 6 if greedy else 12)]:
            if not self.free(T):
                break
            tree.append(T)
            parent.append(i)
            i = len(tree) - 1
            reached = T is not None and se3.dist(T, target) < 1e-9
        return i, reached

    def connect(self, A, B, max_time=120.0, verbose=True):
        """A 到 B 的无碰路线（位姿列表），找不到返回 None。"""
        if self.seg_free(A, B):
            return [A] + se3.steps(A, B, DPOS, DROT)
        self.near = (A, B, 0.02, np.radians(25))
        ta, pa, tb, pb = [A], [-1], [B], [-1]
        t0 = time.time()
        swapped = False
        it = 0
        while time.time() - t0 < max_time:
            it += 1
            q = self.sample()
            i, _ = self._extend(ta, pa, q, False)
            j, hit = self._extend(tb, pb, ta[i], True)
            if hit:
                path_a = self._trace(ta, pa, i)
                path_b = self._trace(tb, pb, j)
                path = path_a + path_b[::-1]
                if swapped:
                    path = path[::-1]
                if verbose:
                    print('    rrt: %d iterations, %d + %d nodes, %d checks, %.1f s' % (it, len(ta), len(tb), self.checks, time.time() - t0))
                return path
            ta, pa, tb, pb = tb, pb, ta, pa
            swapped = not swapped
        if verbose:
            print('    rrt: no path in %.0f s (%d + %d nodes)' % (max_time, len(ta), len(tb)))
        return None

    @staticmethod
    def _trace(tree, parent, i):
        out = []
        while i != -1:
            out.append(tree[i])
            i = parent[i]
        return out[::-1]

    def shortcut(self, path, tries=400):
        if self.extra is not None:          # 带手的检查很贵，抄近路少试几次
            tries = min(tries, 300)
        path = list(path)
        for _ in range(tries):
            if len(path) < 3:
                break
            i, j = sorted(self.rng.integers(0, len(path), 2))
            if j - i < 2:
                continue
            if self.seg_free(path[i], path[j]):
                path = path[:i + 1] + se3.steps(path[i], path[j], DPOS, DROT) + path[j + 1:]
        return path


def path_stats(path):
    d = sum(np.linalg.norm(a[:3, 3] - b[:3, 3]) for a, b in zip(path[:-1], path[1:]))
    a = sum(se3.angle(a, b) for a, b in zip(path[:-1], path[1:]))
    return d, a
