"""The redesign oracle: held segments along the wire between rest sleeves, parks (the wand hung on a sleeve),
regrasps from the hanging pose, the start pick from the deck and the finish placement. Every plan is made from the
measured state (feedback): the arm's joints, the wand's pose, and after each grasp the measured hand-on-wand transform.

Plans are kinematic (PandaKin on a private MjData of the I/O's own model: the Session's frozen scene, exactly), checked
for robot / wand collisions with the scene's static bodies; rows are joint targets interpolated at 40 Hz under a
joint speed and a ring speed cap well inside the Panda's limits."""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from .kin import PandaKin, HOME
from .wandpath import WirePath, parse_grasp, tcp_from_wand, wand_from_tcp
from .feasibility import explore, extract, node_ok

CTRL_DT = 0.025
OPEN, CLOSE = 0.0, 0.8
DECK_PRE_CLOSE = 0.45        # (unused) a pre-closed descent landed a pad on the grip at springref 2.62
DECK_STAGE = (0.012, 0.30)   # deck grasps: stop 12 mm above, close to 0.30 (~4-5 cm gap) in the air, then descend


class Plan:
    def __init__(self):
        self.rows = []          # (q7, grip)
        self.notes = []


class Oracle:
    def __init__(self, io, design, *, log=None, psi_n=24, ds=0.003, w_max=0.6, v_ring=0.03, seeds=24, rng=0,
                 dq_max=0.35):
        self.io, self.design = io, design
        self.kin = PandaKin(io.model, margin=0.006)
        self.L = design.layout()
        self.route = design.wire()
        self.origin = design.origin()
        self.path = WirePath(self.route.rail + self.origin)
        self.psi_n, self.ds, self.w_max, self.v_ring, self.dq_max = psi_n, ds, w_max, v_ring, dq_max
        r = np.random.default_rng(rng)
        self.seeds = [HOME] + [r.uniform(self.kin.lo, self.kin.hi) for _ in range(seeds)]
        self.log = log or (lambda **kv: print(json.dumps(kv, default=str), flush=True))
        self.grip = OPEN
        self.g = None           # current grasp (TCP in the wand frame), measured
        self.events = []
        self.diag = True

    # ---------------------------------------------------------------- execution primitives
    def send(self, q, grip=None, n=1):
        grip = self.grip if grip is None else grip
        self.grip = grip
        for _ in range(n):
            if self.io.step(q, grip):
                return True
        return False

    def move_joints(self, waypoints, *, w_max=None, speeds=None, settle=None):
        """Joint-space interpolation through waypoints (starting from the last command), at most w_max rad/s per
        joint; `speeds` optional per-segment durations (s). settle=(tol rad, max s): after each waypoint hold it until
        the arm is within tol (the drives lag ~0.1 s: a free move's corners would otherwise be cut, close to the wire)."""
        w_max = w_max or self.w_max
        q0 = getattr(self, "_last_q", self.io.q())
        for k, q1 in enumerate(waypoints):
            q1 = np.asarray(q1, float)
            T = max(np.abs(q1 - q0).max() / w_max, CTRL_DT)
            if speeds is not None:
                T = max(T, speeds[k])
            n = int(np.ceil(T / CTRL_DT))
            for i in range(1, n + 1):
                q = q0 + (q1 - q0) * (i / n)
                if self.send(q):
                    self._last_q = q
                    return True
            q0 = q1
            self._last_q = q1
            if settle is not None:
                for _ in range(int(settle[1] / CTRL_DT)):
                    if np.abs(self.io.q() - q1).max() < settle[0]:
                        break
                    if self.send(q1):
                        return True
                err = float(np.abs(self.io.q() - q1).max())
                if err > 0.05:
                    self.note(event="move_stuck", waypoint=k, err_rad=round(err, 3))
        return False

    def wait(self, seconds, grip=None):
        q = getattr(self, "_last_q", self.io.q())
        return self.send(q, grip, n=int(np.ceil(seconds / CTRL_DT)))

    def set_grip(self, value, seconds=0.6):
        q = getattr(self, "_last_q", self.io.q())
        n = int(np.ceil(seconds / CTRL_DT))
        g0 = self.grip
        for i in range(1, n + 1):
            if self.send(q, g0 + (value - g0) * i / n):
                return True
        return False

    # ---------------------------------------------------------------- kinematics helpers
    def ik(self, p, rot, seed, **kw):
        return self.kin.ik(p, rot, seed, **kw)

    def ik_any(self, p, rot, *, prefer=None, wand=None, clear=0.003):
        """Best collision-free IK solution over the seeds (closest to `prefer`)."""
        best = None
        for seed in ([prefer] if prefer is not None else []) + self.seeds:
            q, ok, _, _ = self.kin.ik(p, rot, seed, iters=300)
            if not ok:
                continue
            good, _ = node_ok(self.kin, q, wand, clear_robot=clear) if wand is not None else (True, "")
            if not good:
                continue
            cost = 0 if prefer is None else np.abs(q - prefer).max()
            if best is None or cost < best[0]:
                best = (cost, q)
                if prefer is not None and cost < 0.3:
                    break
        return None if best is None else best[1]

    def cartesian_line(self, q0, p_from, r_from, p_to, r_to, *, step=0.004, wand_of_tcp=None, clear=0.002, robot_clear=None,
                       robot_wand=None, deck_ok=False):
        """Joint waypoints along a straight TCP line (orientation slerped), IK continued from q0. deck_ok: robot
        contact with the deck itself does not count (deck grasps: the pads reach below the grip's bottom face)."""
        ang = (r_to * r_from.inv()).magnitude()
        n = max(2, int(np.ceil(max(np.linalg.norm(np.asarray(p_to) - p_from) / step, ang / 0.02))) + 1)
        key = R.concatenate([r_from, r_to])
        from scipy.spatial.transform import Slerp
        sl = Slerp([0, 1], key)
        out, q = [], np.asarray(q0, float)
        for t in np.linspace(0, 1, n)[1:]:
            p = (1 - t) * np.asarray(p_from) + t * np.asarray(p_to)
            r = sl(t)
            q2, ok, ep, er = self.kin.ik(p, r, q, iters=100)
            if not ok or np.abs(q2 - q).max() > 0.3:
                return None
            if wand_of_tcp is not None:
                good, why = node_ok(self.kin, q2, wand_of_tcp(p, r), clear_robot=clear)
                if not good:
                    return None
            elif robot_clear is not None and not self.robot_clear(q2, robot_clear, wand=robot_wand, deck_ok=deck_ok):
                return None
            out.append(q2); q = q2
        return out

    def robot_clear(self, q, clear, wand=None, deck_ok=False):
        """No robot geom within `clear` of a static body, no robot self-penetration (the wand, if given, placed)."""
        for dist, kind, g1, g2 in self.kin.contacts(q, wand=wand):
            if kind == "robot-static" and dist < clear and not (deck_ok and self.is_deck(g1, g2)):
                return False
            if kind == "robot-robot" and dist < 0:
                return False
        return True

    def is_deck(self, g1, g2):
        nm = self.kin.geom_name(g1) + "|" + self.kin.geom_name(g2)
        return "wire_base::deck" in nm or "finish_support" in nm or ("wire_base" in nm and "terminal__" not in nm)

    def robot_margin(self, q, wand=None, deck_ok=False):
        c = [dist for dist, kind, g1, g2 in self.kin.contacts(q, wand=wand) if kind == "robot-static"
             and not (deck_ok and self.is_deck(g1, g2))]
        bad = any(dist < 0 for dist, kind, g1, g2 in self.kin.contacts(q, wand=wand) if kind == "robot-robot")
        return -1.0 if bad else min(c, default=0.05)

    def free_path_ok(self, qa, qb, *, wand=None, clear=0.002, step=0.03):
        n = max(2, int(np.ceil(np.abs(qb - qa).max() / step)))
        for t in np.linspace(0, 1, n + 1)[1:]:
            q = qa + (qb - qa) * t
            for dist, kind, g1, g2 in self.kin.contacts(q, wand=wand):
                if kind == "robot-static" and dist < clear:
                    return False
                if kind == "robot-robot" and dist < 0:
                    return False
                if kind == "wand-robot" and dist < 0 and wand is not None:
                    return False
        return True

    # ---------------------------------------------------------------- grasp bookkeeping
    def measure_grasp(self):
        ph, Rh = self.io.hand_pose()
        tcp = ph + Rh.apply([0, 0, .1029])
        pw, Rw = self.io.wand_pose()
        self.g = (Rw.inv().apply(tcp - pw), Rw.inv() * Rh)
        return self.g

    def held_wand(self, q):
        p, r = self.kin.fk(q)
        return wand_from_tcp(p, r, self.g)


# ==================================================================================================== the full oracle
from scipy.spatial.transform import Slerp
from topogym.assets.geometry.buzz_wire_3d import endpoint_pose


def gap_for(spec):
    kind, roll = spec.split(":")[0], float(spec.split(":")[1])
    return 0.0070 if (kind.startswith("side") or (kind == "end" and int(round(roll)) % 180 == 90)) else 0.0098


class TierOracle(Oracle):
    """segments: [dict(grasp=<spec>, s1=<arc length> | None (the sleeve centre / the drop point), end='park'|'drop')]"""

    def __init__(self, io, design, segments, *, pick_grasp="top:180:0.8:0:-2", place_grasp="top:180:0.8:0:-2", **kw):
        super().__init__(io, design, **kw)
        self.segments = segments
        self.pick_grasp, self.place_grasp = pick_grasp, place_grasp
        self.cuts = [0.5 * (a + b) for a, b in design.sleeves]
        self.sleeve_r = design.sleeve_radius_m
        self.t0 = time.time()

    # ------------------------------------------------------------------ small helpers
    def note(self, **kv):
        m = self.io.metrics()
        kv.update(row=self.io.rows, wall=round(time.time() - self.t0, 1), held=self.io.holding(),
                  cursor=m.get("route_cursor_m"), tracked=m.get("traversal_tracked"), complete=m.get("traversal_complete"),
                  dist_mm=None if m.get("distance_to_finish_m") is None else round(1e3 * m["distance_to_finish_m"], 2),
                  angle=None if m.get("finish_angle_deg") is None else round(m["finish_angle_deg"], 2),
                  touched=self.io.touched())
        self.events.append(kv)
        self.log(**kv)

    def tcp(self, q=None):
        return self.kin.fk(self.io.q() if q is None else q)

    def line_to(self, p_to, r_to, *, speed=0.03, wand=False, clear=0.002, step=0.003, robot_clear=0.0, robot_wand=None,
                deck_ok=False, checked_only=False):
        """Straight TCP move (orientation slerped) at `speed` m/s, IK-continued from the last command."""
        q0 = getattr(self, "_last_q", self.io.q())
        p0, r0 = self.kin.fk(q0)
        wf = (lambda p, r: wand_from_tcp(p, r, self.g)) if wand else None
        qs = self.cartesian_line(q0, p0, r0, p_to, r_to, step=step, wand_of_tcp=wf, clear=clear,
                                 robot_clear=robot_clear, robot_wand=robot_wand, deck_ok=deck_ok)
        if qs is None and checked_only:
            raise RuntimeError("cartesian line not clear")
        if qs is None and robot_clear is not None and not wand:
            qs = self.cartesian_line(q0, p0, r0, p_to, r_to, step=step)
            if qs is not None:
                self.note(event="line_unchecked", why="robot clearance")
        if qs is None:
            raise RuntimeError("cartesian line infeasible")
        dist = np.linalg.norm(np.asarray(p_to) - p0) + 0.05 * (r_to * r0.inv()).magnitude()
        T = max(dist / speed, 0.1)
        per = T / len(qs)
        return self.move_joints(qs, speeds=[per] * len(qs))

    def free_move(self, q_goal, *, wand_pose=None, lift=0.10):
        """Joint-space move to q_goal, collision-checked (robot vs the scene and the wand where it lies / hangs): the
        straight joint blend if clear, else through via points: backed off along the current hand axis, backed off
        from the goal along its hand axis, lifted, HOME, and random configurations near the two (a few hundred
        tries); the shortest clear chain wins. If none is found, the old lift-over move (logged)."""
        q0 = getattr(self, "_last_q", self.io.q())
        if np.abs(self.io.q() - q0).max() > 0.05:      # the arm is not where it was sent (blocked): plan from it
            q0 = self.io.q()
            self._last_q = q0
        if self.free_path_ok(q0, q_goal, wand=wand_pose):
            return self.move_joints([q_goal], w_max=0.4, settle=(0.01, 1.5))
        p0, r0 = self.kin.fk(q0)
        pg, rg = self.kin.fk(q_goal)
        a0, ag = r0.apply([0, 0, 1]), rg.apply([0, 0, 1])
        vias = []
        for d0 in (0.06, 0.12):
            for dg in (0.08, 0.15):
                b0 = self.ik(p0 - d0 * a0, r0, q0)
                bg = self.ik(pg - dg * ag, rg, q_goal)
                if b0[1] and bg[1]:
                    vias.append([b0[0], bg[0]])
        up0 = self.ik(p0 + [0, 0, lift], r0, q0); upg = self.ik(pg + [0, 0, lift], rg, q_goal)
        if up0[1] and upg[1]:
            vias.append([up0[0], upg[0]])
        vias += [[HOME], [np.asarray(v[0]) for v in vias[:1]] + [HOME] if vias else [HOME]]
        rng = np.random.default_rng(len(self.events))
        for _ in range(120):
            mid = 0.5 * (q0 + q_goal) + rng.normal(0, 0.5, 7)
            vias.append([np.clip(mid, self.kin.lo, self.kin.hi)])
        best = None
        for v in vias:
            chain = [q0] + [np.asarray(x) for x in v] + [q_goal]
            if all(self.free_path_ok(a, b, wand=wand_pose) for a, b in zip(chain[:-1], chain[1:])):
                cost = sum(np.abs(b - a).max() for a, b in zip(chain[:-1], chain[1:]))
                if best is None or cost < best[0]:
                    best = (cost, chain[1:])
        if best is None:
            path = self.rrt_connect(q0, np.asarray(q_goal, float), wand=wand_pose)
            if path is not None:
                self.note(event="free_move_rrt", waypoints=len(path))
                best = (0.0, path)
        if best is not None:
            return self.move_joints(best[1], w_max=0.4, settle=(0.01, 1.5))
        self.note(event="free_move_collision_risk")
        return self.move_joints([up0[0], upg[0], q_goal], w_max=0.4, settle=(0.01, 1.5))

    def rrt_connect(self, qa, qb, *, wand=None, iters=2500, step=0.25, clear=0.002, seed=0):
        """Joint-space RRT-Connect between qa and qb (both trees grown towards random samples within the joint limits,
        every edge checked by free_path_ok), then greedy shortcutting. -> [waypoints after qa ... qb] or None."""
        ok = lambda a, b: self.free_path_ok(a, b, wand=wand, clear=clear)
        if ok(qa, qb):
            return [qb]
        rng = np.random.default_rng(seed + len(self.events))
        A, B = [(qa, -1)], [(qb, -1)]

        def extend(T, q):
            i = int(np.argmin([np.abs(n[0] - q).max() for n in T]))
            near = T[i][0]
            d = q - near
            n = np.abs(d).max()
            new = q if n <= step else near + d * (step / n)
            if ok(near, new):
                T.append((new, i))
                return len(T) - 1
            return None

        def connect(T, q):
            last = None
            for _ in range(40):
                k = extend(T, q)
                if k is None:
                    return last, False
                last = k
                if np.abs(T[k][0] - q).max() < 1e-9:
                    return k, True
            return last, False

        def trace(T, k):
            out = []
            while k >= 0:
                out.append(T[k][0]); k = T[k][1]
            return out
        for it in range(iters):
            q = rng.uniform(self.kin.lo, self.kin.hi) if rng.random() > 0.1 else (B[-1][0] if len(B) else qb)
            k = extend(A, q)
            if k is not None:
                j, reached = connect(B, A[k][0])
                if reached:
                    pa, pb = trace(A, k), trace(B, j)
                    if A is not None and np.allclose(pa[-1], qa):
                        path = pa[::-1] + pb[1:]
                    else:
                        path = pb[::-1] + pa[1:]
                    if not np.allclose(path[0], qa):
                        path = path[::-1]
                    # shortcut
                    i = 0
                    short = [path[0]]
                    while i < len(path) - 1:
                        j2 = len(path) - 1
                        while j2 > i + 1 and not ok(path[i], path[j2]):
                            j2 -= 1
                        short.append(path[j2]); i = j2
                    return short[1:]
            A, B = B, A
        return None

    def wait_holding(self, timeout=1.5):
        n = int(timeout / CTRL_DT)
        for _ in range(n):
            if self.io.holding():
                return True
            if self.wait(CTRL_DT):
                return False
        return self.io.holding()

    def settle(self, timeout=3.0, v=0.004, w=0.05):
        n = int(timeout / CTRL_DT)
        for i in range(n):
            lin, ang = self.io.wand_speed()
            if lin < v and ang < w and i > 4:
                return True
            if self.wait(CTRL_DT):
                return False
        return False

    # ------------------------------------------------------------------ grasping
    def finger(self):
        m = self.io.model
        if not hasattr(self, "_fj"):
            self._fj = m.jnt_qposadr[m.joint("finger_joint1").id]
        return float(self.io.data.qpos[self._fj])

    def grasp_candidates(self, spec, pw, Rw, *, approach=0.07, n_keep=6, require_sweep=False, deck_ok=False):
        """Collision-free grasp configurations for `spec` on the wand at (pw, Rw), each with a checked straight
        approach line from the pre-grasp pose `approach` back along the hand axis: [(margin, q_grasp, q_pre)]."""
        g = parse_grasp(spec, self.L)
        tp, tr = tcp_from_wand(pw, Rw, g)
        ap = tr.apply([0, 0, 1])
        out = []
        prefer = getattr(self, "_last_q", self.io.q())
        for seed in [prefer] + self.seeds:
            self.kin.set_gap(gap_for(spec))
            q, ok, _, _ = self.kin.ik(tp, tr, seed, iters=300)
            if not ok:
                continue
            mg = self.robot_margin(q, wand=(pw, Rw), deck_ok=deck_ok)
            if mg < 0.0005:
                continue
            swept = True                  # the fingers' sweep while closing: the pads clear of every static body
            for gap in (0.085, 0.06, 0.04, 0.025, 0.015):   # but the flat deck (the fingertips' arc dips 2-5 mm
                self.kin.set_gap(gap)                         # below the TCP mid-close; the deck pushes the hand up)
                for dist, kind, g1, g2 in self.kin.contacts(q, wand=(pw, Rw)):
                    nm = self.kin.geom_name(g1) + "|" + self.kin.geom_name(g2)
                    deck = ("wire_base::deck" in nm or "finish_support" in nm
                            or ("wire_base" in nm and "terminal__" not in nm))     # the deck slab and its top
                    if kind == "robot-static" and dist < 0.0003 and not deck:
                        swept = False; break
                if not swept:
                    break
            self.kin.set_gap(gap_for(spec))
            if not swept and require_sweep:
                continue
            if any(np.abs(q - o[1]).max() < 0.05 for o in out):
                continue
            self.kin.set_gap(0.085)
            back = self.cartesian_line(q, tp, tr, tp - approach * ap, tr, step=0.004, robot_clear=0.0005,
                                       robot_wand=(pw, Rw), deck_ok=deck_ok)
            if back is None:
                continue
            out.append((mg, q, back[-1]))
            if len(out) >= n_keep:
                break
        return out

    def grasp_at(self, spec, *, approach=0.07, pre_open=OPEN, close_s=1.0, timeout=3.5, pose=None, stage=None,
                 deck_ok=None, choose=None):
        """Grasp the (resting or hanging) wand with `spec` at its measured pose (or `pose`, e.g. the ideal hanging
        pose when the hung ring sits yawed on its sleeve: the closing pads square it up): a collision-free grasp
        configuration with a checked straight approach, the hand opened first, a slow close, then wait for the weld."""
        for attempt in range(3):          # re-plan if the (hung, swinging) wand moved while the hand came over
            pw, Rw = self.io.wand_pose() if pose is None else pose
            cands = self.grasp_candidates(spec, pw, Rw, approach=approach,
                                          deck_ok=(stage is not None) if deck_ok is None else deck_ok)
            if not cands:
                raise RuntimeError(f"no collision-free grasp configuration for {spec}")
            here = getattr(self, "_last_q", self.io.q())
            cands = sorted(cands, key=lambda c: np.abs(c[2] - here).max() - 2.0 * min(c[0], 0.01))
            self._grasp_no = getattr(self, "_grasp_no", 0) + (1 if attempt == 0 else 0)
            at = getattr(self, "noise_branch_at", None)       # 1-based grasp numbers that take another branch
            if (getattr(self, "noise_branch", False) and len(cands) > 1
                    and (self._grasp_no in at if at else getattr(self, "_branch_used", 0) < getattr(self, "noise_branch_n", 99))):
                self._branch_used = getattr(self, "_branch_used", 0) + 1
                # noise (user's step 4): another arm branch where one exists (wrist flip / elbow), not the nearest
                cands = sorted(cands, key=lambda c: -np.abs(c[1] - here).max())
                self.note(event="noise_branch", spec=spec, options=len(cands),
                          dq=round(float(np.abs(cands[0][1] - here).max()), 2))
            if choose is not None:      # plan-aware: the first configuration from which the next segment plans
                ok = [c for c in cands if choose(c[1], spec, pw, Rw)]
                if not ok:
                    raise RuntimeError(f"no grasp configuration for {spec} from which the segment plans")
                cands = ok
            mg, q_g, q_pre = cands[0]
            self.kin.set_gap(0.085)
            self.set_grip(pre_open, 0.4)
            self.wait_finger(lambda f: abs(f - pre_open) < 0.06, 3.0)
            self.free_move(q_pre, wand_pose=(pw, Rw))
            if pose is not None:
                break
            self.settle(timeout=6.0)
            pw2, Rw2 = self.io.wand_pose()
            if np.linalg.norm(pw2 - pw) < 0.003 and (Rw2 * Rw.inv()).magnitude() < np.deg2rad(5):
                break
            self.note(event="grasp_replan", spec=spec, moved_mm=round(1e3 * float(np.linalg.norm(pw2 - pw)), 1))
        g = parse_grasp(spec, self.L)
        tp, tr = tcp_from_wand(pw, Rw, g)
        self._last_q = q_pre
        if stage is not None:             # deck grasps: stop `stage[0]` m above, close to `stage[1]` in the air (the
            ap = tr.apply([0, 0, 1])      # pads sweep an arc and would jam on the deck from wide open), then go down
            self.line_to(tp - stage[0] * ap, tr, speed=0.03, robot_clear=0.0, robot_wand=(pw, Rw), deck_ok=True)
            self.set_grip(stage[1], 0.5)
            self.wait_finger(lambda f: abs(f - stage[1]) < 0.05, 3.0)
        self.line_to(tp, tr, speed=0.02, robot_clear=0.0, robot_wand=(pw, Rw), deck_ok=stage is not None)
        self.wait(0.3)
        ph, Rh = self.io.hand_pose()
        tcp_err = float(np.linalg.norm(ph + Rh.apply([0, 0, .1029]) - tp))
        if tcp_err > 0.004:
            self.note(event="grasp_tcp_short", spec=spec, err_mm=round(1e3 * tcp_err, 1))
        self.set_grip(CLOSE, close_s)
        ok = self.wait_holding(timeout)
        self.kin.set_gap(gap_for(spec))
        self.measure_grasp()
        self.note(event="grasp", spec=spec, holding=ok, margin_mm=round(1e3 * mg, 1), finger=round(self.finger(), 3),
                  g_p=np.round(self.g[0], 4).tolist())
        if not ok:
            raise RuntimeError(f"grasp {spec} not acquired")
        return ok

    def wait_finger(self, cond, timeout):
        n = int(timeout / CTRL_DT)
        for _ in range(n):
            if cond(self.finger()):
                return True
            if self.wait(CTRL_DT):
                return False
        return cond(self.finger())

    def release(self, retreat=0.07, open_s=0.6, open_to=0.25, speed=0.04):
        """Open (the weld lets go at once; the fingers take ~1-2 s), wait until the fingers have opened to `open_to`
        (native), then retreat along the hand axis."""
        f0 = self.finger()
        self.set_grip(OPEN, open_s)
        self.wait_finger(lambda f: f < min(open_to, f0 - 0.15), 4.0)
        q = getattr(self, "_last_q", self.io.q())
        p, r = self.kin.fk(q)
        ap = r.apply([0, 0, 1])
        self.kin.set_gap(0.085)
        for d in (-ap, np.array([0, 0, 1.]), -ap + np.array([0, 0, 1.]), np.array([0, 0, 1.]) - 0.5 * ap):
            d = d / np.linalg.norm(d)
            for dist in (retreat, 0.5 * retreat):
                try:
                    self.line_to(p + dist * d, r, speed=speed, robot_clear=-0.001)
                    self.note(event="released", finger=round(self.finger(), 3), retreat=np.round(dist * d, 3).tolist())
                    return
                except RuntimeError:
                    continue
        self.note(event="released", finger=round(self.finger(), 3), retreat=None)

    # ------------------------------------------------------------------ wand motion helpers (held)
    def move_wand(self, pw, Rw, *, speed=0.02):
        tp, tr = tcp_from_wand(pw, Rw, self.g)
        return self.line_to(tp, tr, speed=speed)

    def hang_pose(self, s):
        """The wand hanging on the sleeve at arc length s: ring normal along the (horizontalised) tangent, handle down,
        the ring's inner top edge 0.3 mm above the sleeve."""
        i = self.path.idx(s)
        t = self.path.t[i].copy(); p = self.path.p[i]
        t[2] = 0 if abs(t[2]) < 0.5 else t[2]
        t /= np.linalg.norm(t)
        down = np.array([0, 0, -1.]) - (np.array([0, 0, -1.]) @ t) * t
        down /= np.linalg.norm(down)
        c = p + down * (self.L.loop_radius_m - self.L.loop_tube_radius_m - self.sleeve_r - 0.0003)
        return c, R.from_matrix(np.column_stack([down, np.cross(t, down), t]))

    def centred_pose(self, s, Rw):
        """The ring centred on the wire at s with orientation Rw's handle direction."""
        i = self.path.idx(s)
        return self.path.p[i].copy(), Rw

    # ------------------------------------------------------------------ the stages
    def try_grasps(self, specs, **kw):
        """The first spec that grasps; a grasp not acquired is let go (open, back off) before the next one."""
        last = None
        for spec in dict.fromkeys(specs):
            try:
                self.grasp_at(spec, **kw)
                return spec
            except RuntimeError as exc:
                last = exc
                if self.io.holding():
                    raise
                if "not acquired" in str(exc):
                    self.release(retreat=0.05, open_s=1.0, speed=0.015)
                    self.settle(timeout=2.0)
        raise RuntimeError(f"no grasp of {len(specs)} specs: {last}")

    def pick(self):
        self.note(event="pick_start")
        kind, roll, frac, pitch, depth = (self.pick_grasp.split(":") + ["0.9", "-10", "-2"])[:5]
        # springref 2.62: the pads sweep an arc and reach lowest wide open, where they jam on the deck; the hand comes
        # down pre-closed to ~3 cm (DECK_PRE_CLOSE) so the pads stay off the deck, then closes on the 9.8 mm width
        # head c64294012: a weld lets go 0.2 s after both fingers leave the grip. With the TCP above the grip's centre
        # the closing pads slid off its upper edges (contact lost, weld released at lift); 2 mm below the centre both
        # pads stay on the grip's sides (finger 0.665, both in contact after a 2 cm lift)
        base = ":".join(self.pick_grasp.split(":")[:4] + ["2"])
        variants = [base] + [f"top:{r}:{f}:{p}:{dd}" for dd in ("2", "1", "3", "0") for f in ("0.9", "0.85", "0.8")
                                        for r, p in (("180", "-10"), ("0", "10"), ("180", "-20"), ("0", "20"),
                                                     ("180", "0"), ("0", "0"))]
        variants = [v for v in variants if v not in getattr(self, "_used_picks", set())]
        spec = self.try_grasps(variants, stage=DECK_STAGE)
        self._used_picks = getattr(self, "_used_picks", set()) | {spec}
        pw, Rw = self.io.wand_pose()
        self._deck_pose = (pw.copy(), Rw)
        # level the ring and centre it on the post, at the rail's first point (s = 0)
        p0 = self.path.p[0]
        h = Rw.apply([1, 0, 0]); h[2] = 0; h /= np.linalg.norm(h)
        Rt = R.from_matrix(np.column_stack([h, np.cross([0, 0, 1.], h), [0, 0, 1.]]))
        self.move_wand(np.array([p0[0], p0[1], pw[2] + 0.002]), Rt, speed=0.01)
        self.move_wand(p0 + [0, 0, 0.004], Rt, speed=0.01)
        self.note(event="lifted")

    def traverse(self, k, seg):
        s0 = seg.get("s0", self.cuts[k - 1] if k > 0 else 0.004)
        s0 = max(s0, 0.004)
        s1 = seg["s1"]
        pw, Rw = self.held_wand(getattr(self, "_last_q", self.io.q()))
        psi = self.path.psi_toward(s0, Rw.apply([1, 0, 0]))
        j0 = int(np.round(psi / (2 * np.pi / self.psi_n))) % self.psi_n
        # snap: the ring onto the planned frame at s0 (a few mm / deg)
        p_s, R_s = self.path.frame(s0, j0 * 2 * np.pi / self.psi_n)
        self.move_wand(p_s, R_s, speed=0.01)
        q_start = getattr(self, "_last_q", self.io.q())
        insp = float(getattr(self, "inspect_deg", 0.0))
        if insp > 0 and s0 > 0.02:
            # noise (user's step 4): a small inspection turn of the held wand about the wire's tangent (the handle
            # swings by +-insp around the ring, the ring stays centred on the wire); every pose checked (wand and robot);
            # a turn that does not check out is skipped, never forced
            done = []
            base = j0 * 2 * np.pi / self.psi_n
            for dpsi in (insp, -insp):
                for target in (dpsi, 0.0):
                    p_i, R_i = self.path.frame(s0, base + np.deg2rad(target))
                    tp_i, tr_i = tcp_from_wand(p_i, R_i, self.g)
                    try:
                        self.line_to(tp_i, tr_i, speed=0.006, wand=True, clear=0.002)
                    except RuntimeError:
                        break
                    if target != 0.0:
                        done.append(dpsi)
            self.note(event="inspection_turn", k=k, done_deg=done)
            q_start = getattr(self, "_last_q", self.io.q())
        end = seg.get("end", "park")
        jB = None
        if end == "park":
            jB = int(np.round(self.path.psi_toward(s1, [0, 0, -1]) / (2 * np.pi / self.psi_n))) % self.psi_n
        elif end == "drop":       # release with the handle already near the finish heading (the post keeps it)
            fh = endpoint_pose(self.route, 1)[1].apply([1, 0, 0])
            # drop_offset_deg: aim the release heading this far (about the vertical) from the finish heading, so the
            # wand lands inside the arm's graspable window about the post (the fall turns it; measured per tier)
            fh = R.from_euler("z", np.deg2rad(getattr(self, "drop_offset_deg", 0.0))).apply(fh)
            jB = int(np.round(self.path.psi_toward(s1, fh) / (2 * np.pi / self.psi_n))) % self.psi_n
        if end == "drop":                 # carry the ring down the insulated post as far as the hand can go, so the wand
            post_end = self.path.length - 0.05      # does not fall the whole post (it tumbled and leaned at 2.62)
            backs = getattr(self, "drop_backs", None) or (0.0, 0.04, 0.08)      # option: release higher up the post
            for s_try in [post_end - b for b in backs]:
                if s_try <= s1:
                    break
                r_try = explore(self.kin, self.path, self.g, s0=s0, s1=s_try, ds=self.ds, psi_n=self.psi_n,
                                start_qs=[(j0, 0, q_start)], keep=True, clear_robot=0.002, per_bin=4, dq_max=self.dq_max)
                if r_try["complete"]:
                    s1 = s_try
                    self.note(event="drop_lower", s1=round(s1, 3))
                    break
        res = explore(self.kin, self.path, self.g, s0=s0, s1=s1, ds=self.ds, psi_n=self.psi_n,
                      start_qs=[(j0, 0, q_start)], keep=True, clear_robot=0.002, per_bin=4,
                      dq_max=self.dq_max)
        pth = extract(res, end_psi=jB, end_lag=0, psi_tol=seg.get("psi_tol", 6 if end == "park" else 2)) \
            if res["complete"] else None
        choice = getattr(self, "drop_psi_choice", None)
        if end == "drop" and res["complete"] and choice is not None:
            # drop heading search: the release headings this grip can reach at the drop end (psi bins of the last
            # layer, lag 0), ordered by distance from the commanded heading; take the choice-th (modulo their number).
            # The release heading decides on which side of the finish post the wand comes to rest.
            n = len(res["psis"])
            reach = sorted({j for j, l, q, _ in res["layers"][-1] if l == 0},
                           key=lambda j: (min((j - jB) % n, (jB - j) % n), j))
            if reach:
                jC = reach[choice % len(reach)]
                pth = extract(res, end_psi=jC, end_lag=0, psi_tol=0)
                self.note(event="drop_heading_choice", choice=choice, reachable=reach, commanded=jB, chosen=jC)
        if pth is None and end == "drop" and res["complete"]:
            for tol in (4, 6, 12):
                pth = extract(res, end_psi=jB, end_lag=0, psi_tol=tol)
                if pth is not None:
                    self.note(event="drop_heading_relaxed", psi_tol=tol)
                    break
        if pth is None:
            raise RuntimeError(f"segment {k} not plannable from the measured state (reached {res['reached_s']}; "
                               f"rejections {dict(sorted(res.get('whys', {}).items(), key=lambda kv: -kv[1])[:4])})")
        self.note(event="segment_planned", k=k, s0=round(s0, 3), s1=round(s1, 3), nodes=len(pth))
        qs = self.densify(pth)
        jump = max(float(np.abs(b[0] - a[0]).max()) for a, b in zip(qs[:-1], qs[1:]))
        self.note(event="segment_path", k=k, max_node_dq=round(jump, 3))
        if self.track(qs[1:]):
            raise RuntimeError(f"segment {k}: the episode ended during the traverse")
        self.note(event="segment_done", k=k)
        if self.io.touched():
            raise RuntimeError(f"segment {k}: the ring touched the wire")

    def densify(self, pth, *, dq_step=0.025):
        """Joint targets between planned nodes: the held wand's pose interpolated on the wire (arc length, handle angle
        psi, lag) and solved by IK continuation, so the ring stays centred between nodes even where the plan turns the
        wrist fast (a linear joint blend between two nodes 0.15 rad apart can swing the ring off the wire)."""
        out = [(pth[0][3], 0.0)]
        q = np.asarray(pth[0][3], float)
        bad = 0
        for (sa, pa, la, qa), (sb, pb, lb, qb) in zip(pth[:-1], pth[1:]):
            n = max(1, int(np.ceil(np.abs(np.asarray(qb) - qa).max() / dq_step)))
            dp = (pb - pa + np.pi) % (2 * np.pi) - np.pi
            for t in np.arange(1, n + 1) / n:
                if t >= 1.0:
                    q2 = np.asarray(qb, float)
                else:
                    pw, Rw = self.path.frame(sa + (sb - sa) * t, pa + dp * t, la + (lb - la) * t)
                    tp, tr = tcp_from_wand(pw, Rw, self.g)
                    q2, ok, _, _ = self.kin.ik(tp, tr, q, iters=60)
                    if not ok or np.abs(q2 - q).max() > 2.5 * dq_step:
                        q2 = q + (np.asarray(qb) - q) / max(1, int(round((1 - t) * n)) + 1)
                        bad += 1
                out.append((q2, abs(sb - sa) / n))
                q = q2
        if bad:
            self.note(event="densify_fallback", n=bad)
        return out

    def track(self, qs, *, w_max=0.3, tol=0.006, max_wait=1.5):
        """Follow held-path nodes closely: each node at the ring speed (and at most w_max rad/s per joint), then hold
        the node until the arm's worst joint is within `tol` of it (the drives lag ~0.1 s; without the hold the lag
        cuts the wire's bends)."""
        for i, (q1, dsr) in enumerate(qs):
            per = max(CTRL_DT, dsr / self.v_ring)
            if self.move_joints([q1], w_max=w_max, speeds=[per]):
                return True
            n = 0
            while np.abs(self.io.q() - q1).max() > tol and n < max_wait / CTRL_DT:
                if self.send(q1):
                    return True
                n += 1
            if self.diag and i % 3 == 0:
                qm = self.io.q()
                pp, _ = self.held_wand(q1)
                pk, Rk = self.held_wand(qm)
                pm, Rm = self.io.wand_pose()
                s = self.path.s[int(np.argmin(np.linalg.norm(self.path.p - pm, axis=1)))]
                j = self.path.idx(s)
                off = pm - self.path.p[j]
                off_mm = 1e3 * np.linalg.norm(off - (off @ self.path.t[j]) * self.path.t[j])
                self.log(event="node", i=i, row=self.io.rows, s=round(float(s), 4), wait=n,
                         qerr=round(float(np.abs(qm - q1).max()), 4), track_mm=round(1e3 * float(np.linalg.norm(pk - pp)), 2),
                         model_mm=round(1e3 * float(np.linalg.norm(pm - pk)), 2),
                         model_deg=round(float(np.degrees((Rm * Rk.inv()).magnitude())), 2), off_mm=round(float(off_mm), 2),
                         tilt_deg=round(float(np.degrees(np.arccos(np.clip(abs(Rm.apply([0, 0, 1]) @ self.path.t[j]), -1, 1)))), 2))
        return False

    def park(self, k, seg):
        """Lower the ring onto the sleeve (its inner top edge 0.3 mm above the sleeve) with the wand as it arrives
        (the handle within ~60 deg of down), release, and let it swing to hanging and settle."""
        s1 = seg["s1"]
        i = self.path.idx(s1)
        pw, Rw = self.held_wand(getattr(self, "_last_q", self.io.q()))
        c = self.path.p[i] + np.array([0, 0, -1.0]) * (self.L.loop_radius_m - self.L.loop_tube_radius_m - self.sleeve_r
                                                         - 0.0003)
        self.move_wand(self.path.p[i], Rw, speed=0.01)
        self.move_wand(c, Rw, speed=0.005)
        self.release(open_s=1.5, speed=0.015)       # slow: the hung wand swings for many seconds on the sleeve
        ok = self.settle(timeout=20.0)
        pw2, Rw2 = self.io.wand_pose()
        self.note(event="parked", k=k, settled=ok, drift_mm=round(1e3 * float(np.linalg.norm(pw2 - c)), 2),
                  handle_down_deg=round(float(np.degrees(np.arccos(np.clip(-Rw2.apply([1, 0, 0])[2], -1, 1)))), 1))

    def regrasp(self, k, seg):
        self.settle(timeout=20.0)
        f = seg["grasp"].split(":")
        kind, roll = f[0], f[1]
        other = "0" if roll == "180" else "180"
        variants = [seg["grasp"]] + [":".join([kind, r, fr] + f[3:]) for r in (roll, other) for fr in ("0.55", "0.7", "0.8")]
        last = None
        s_rest = seg.get("s0", self.cuts[k - 1])
        ideal = self.hang_pose(s_rest)
        done = False
        for pose in (None, ideal):
            for spec in dict.fromkeys(variants):
                try:
                    self.grasp_at(spec, pre_open=0.35, pose=pose, choose=lambda q, sp, pw, Rw: self._plannable(q, sp, pw, Rw, seg))
                    done = True
                    break
                except RuntimeError as exc:
                    last = exc
                    if self.io.holding():
                        raise
                    if "not acquired" in str(exc):
                        self.release(retreat=0.04, open_s=1.0, speed=0.015)
                        self.settle(timeout=4.0)
            if done:
                if pose is not None:
                    self.note(event="regrasp_ideal_pose", k=k)
                break
        if not done:
            raise RuntimeError(f"regrasp {k}: {last}")
        # lift the ring off the sleeve: centre it on the wire (up by the hanging offset)
        s = seg.get("s0", self.cuts[k - 1])
        pw, Rw = self.held_wand(getattr(self, "_last_q", self.io.q()))
        self.move_wand(self.path.p[self.path.idx(s)], Rw, speed=0.005)
        self.note(event="lifted_off", k=k)

    def _plannable(self, q, spec, pw, Rw, seg):
        """Would the held segment `seg` plan from grasp configuration q (the nominal grip `spec` on the wand at (pw, Rw),
        lifted onto the wire at seg['s0'])? A quick explore with the segment's own end rule."""
        g = parse_grasp(spec, self.L)
        tp, tr = tcp_from_wand(pw, Rw, g)
        s0, s1, end = max(seg["s0"], 0.004), seg["s1"], seg.get("end", "park")
        p_s = self.path.p[self.path.idx(s0)]
        psi = self.path.psi_toward(s0, Rw.apply([1, 0, 0]))
        j0 = int(np.round(psi / (2 * np.pi / self.psi_n))) % self.psi_n
        # the lift-off: the same grip, the ring centred on the wire at s0
        q0, ok, _, _ = self.kin.ik(*tcp_from_wand(*self.path.frame(s0, j0 * 2 * np.pi / self.psi_n), g), q, iters=200)
        if not ok:
            return False
        res = explore(self.kin, self.path, g, s0=s0, s1=s1, ds=self.ds, psi_n=self.psi_n, start_qs=[(j0, 0, q0)],
                      keep=True, clear_robot=0.002, per_bin=4, dq_max=self.dq_max)
        if not res["complete"]:
            self.note(event="candidate_unplannable", spec=spec, reached=res["reached_s"],
                      why=dict(sorted(res.get("whys", {}).items(), key=lambda kv: -kv[1])[:2]))
            return False
        if end == "park":
            jB = int(np.round(self.path.psi_toward(s1, [0, 0, -1]) / (2 * np.pi / self.psi_n))) % self.psi_n
            return extract(res, end_psi=jB, end_lag=0, psi_tol=6) is not None
        return True

    def drop(self):
        # drop_open_s: how fast the fingers open at the release (the fall's start); drop_retreat: the hand's back-off
        self.release(retreat=getattr(self, "drop_retreat", 0.06), open_s=getattr(self, "drop_open_s", 0.6))
        ok = self.settle(timeout=4.0)
        pw, Rw = self.io.wand_pose()
        self.note(event="dropped", settled=ok, wand_z=round(float(pw[2]), 4))

    @staticmethod
    def _yaw(v):
        return float(np.arctan2(v[1], v[0]))

    def _rot_plan(self, spec, pw, Rw, post, dyaw, p_end, *, n=8, up=0.004, sweep=True):
        """A collision-free grasp configuration for `spec` on the resting wand and an IK chain that lifts it 4 mm,
        turns it by dyaw about the finish post's vertical axis (the ring stays around the post) and lowers it at
        p_end: -> (q_grasp, q_pre, [(p, R)]) or None."""
        g = parse_grasp(spec, self.L)
        poses = [(pw + [0, 0, up], Rw)]
        for k in range(1, n + 1):
            Rz = R.from_euler("z", dyaw * k / n)
            pk = post + Rz.apply(pw - post)
            pk = pk + (p_end - (post + R.from_euler("z", dyaw).apply(pw - post))) * (k / n) + [0, 0, up]
            poses.append((pk, Rz * Rw))
        poses.append((poses[-1][0] - [0, 0, up - 0.0005], poses[-1][1]))
        for mg, q_g, q_pre in self.grasp_candidates(spec, pw, Rw, require_sweep=sweep, deck_ok=True):
            self.kin.set_gap(gap_for(spec))
            q, ok = q_g, True
            for pk, Rk in poses:
                tp, tr = tcp_from_wand(pk, Rk, g)
                q2, conv, _, _ = self.kin.ik(tp, tr, q, iters=120)
                if not conv or np.abs(q2 - q).max() > 0.6:
                    ok = False; break
                good, _ = node_ok(self.kin, q2, (pk, Rk), clear_robot=-1.0, wand_soft=-0.001)
                good = good and self.robot_margin(q2, wand=(pk, Rk), deck_ok=True) >= 0.0005
                if not good:
                    ok = False; break
                q = q2
            if ok:
                return q_g, q_pre, poses
        return None

    def final_place(self, rounds=6, max_turn=np.deg2rad(70)):
        """Set the final pose on the deck in grasp-turn-release rounds about the finish post: each round grasps the
        resting wand from above (its upper face: 'top', or 'topb' when it lies flipped), turns it by at most
        max_turn towards the finish heading, and puts it down; the last round puts it on the target pose."""
        tp, tr = endpoint_pose(self.route, 1)
        target_p, target_r = tp + self.origin, tr
        post = target_p.copy()
        hy_t = self._yaw(target_r.apply([1, 0, 0]))
        bad = set()
        pushes = 0
        for rnd in range(rounds + 12):
            pw, Rw = self.io.wand_pose()
            flip = Rw.apply([0, 0, 1])[2] < 0
            d = (hy_t - self._yaw(Rw.apply([1, 0, 0])) + np.pi) % (2 * np.pi) - np.pi
            post = np.array([target_p[0], target_p[1], pw[2]])
            self.note(event="place_round", round=rnd, flip=bool(flip), dyaw_deg=round(float(np.degrees(d)), 1),
                      dist_mm=round(1e3 * float(np.linalg.norm(pw - target_p)), 2))
            if abs(d) < np.deg2rad(2.0) and np.linalg.norm(pw[:2] - target_p[:2]) < 0.0015:
                break
            kind = "topb" if flip else "top"
            specs = [f"{kind}:{r}:{f}:{p}:{dd}" for dd in (2, 1, 3, 0) for f in (0.9, 0.85, 0.8, 0.7) for r, p in
                     ((180, -10), (0, 10), (180, -20), (0, 20), (180, 0), (0, 0), (180, 10), (0, -10))]
            plan = None
            # quick test first: no grasp configuration at all on the landed wand -> straight to the push-turn
            reachable = [sp for sp in specs[::3] if sp not in bad
                         and self.grasp_candidates(sp, pw, Rw, require_sweep=False, deck_ok=True, n_keep=1)]
            if not reachable:
                specs = []
            for sweep, step in [(sw, st) for sw in (True, False) for st in
                                dict.fromkeys([d, np.clip(d, -max_turn, max_turn), np.clip(d, -0.7, 0.7), np.clip(d, -0.4, 0.4)])]:
                final = abs(step - d) < 1e-9
                p_end = target_p.copy() if final else post + R.from_euler("z", step).apply(pw - post)
                p_end[2] = target_p[2] if final else pw[2]
                for spec in specs:
                    if spec in bad:
                        continue
                    plan = self._rot_plan(spec, pw, Rw, post, step, p_end, sweep=sweep)
                    if plan is not None:
                        break
                if plan is not None:
                    break
            if plan is None and pushes < 6:
                # no grasp reaches the wand at this heading (the finish wire arches over that side): turn it about
                # the post by pushing its handle with the closed fingers, towards dyaw ~ +45 deg (graspable side)
                beta = (d - np.deg2rad(45) + np.pi) % (2 * np.pi) - np.pi
                pushes += 1
                pushed = False
                for b in (beta, beta - np.sign(beta) * 2 * np.pi):      # the short way round first
                    for lim in (60, 40, 25, 15):                         # shorter pushes plan where long ones do not
                        if self.push_turn(pw, Rw, post, float(np.clip(b, -np.deg2rad(lim), np.deg2rad(lim)))):
                            pushed = True; break
                    if pushed:
                        break
                if pushed:
                    continue
            if plan is None:
                raise RuntimeError(f"final place: no grasp/turn plan in round {rnd} (dyaw {np.degrees(d):.1f} deg)")
            q_g, q_pre, poses = plan
            self.kin.set_gap(0.085)
            self.set_grip(OPEN, 0.4)
            self.wait_finger(lambda f: f < 0.08, 3.0)
            self.free_move(q_pre, wand_pose=(pw, Rw))
            self._last_q = q_pre
            g = parse_grasp(spec, self.L)
            gp, gr = tcp_from_wand(pw, Rw, g)
            gap_ = gr.apply([0, 0, 1])
            self.line_to(gp - DECK_STAGE[0] * gap_, gr, speed=0.03, robot_clear=0.0, robot_wand=(pw, Rw), deck_ok=True)
            self.set_grip(DECK_STAGE[1], 0.5)
            self.wait_finger(lambda f: abs(f - DECK_STAGE[1]) < 0.05, 3.0)
            self.line_to(gp, gr, speed=0.02, robot_clear=0.0, robot_wand=(pw, Rw), deck_ok=True)
            self.wait(0.3)
            ph, Rh = self.io.hand_pose()
            err = float(np.linalg.norm(ph + Rh.apply([0, 0, .1029]) - gp))
            if err > 0.004:
                self.note(event="place_tcp_short", round=rnd, spec=spec, err_mm=round(1e3 * err, 1))
            self.set_grip(CLOSE, 1.0)
            if not self.wait_holding(3.5):
                self.note(event="place_grasp_missed", round=rnd, spec=spec, finger=round(self.finger(), 3))
                bad.add(spec)
                self.release(retreat=0.05, open_s=1.0, speed=0.015)
                self.settle(timeout=2.0)
                continue
            self.kin.set_gap(gap_for(spec))
            self.measure_grasp()
            self.note(event="place_grasp", round=rnd, spec=spec, turn_deg=round(float(np.degrees(step)), 1))
            for pk, Rk in poses:
                self.move_wand(pk, Rk, speed=0.008)
            self.release(retreat=0.08, open_s=0.8)
            self.settle(timeout=3.0)
        self.note(event="placed")

    def plan_push(self, pw, Rw, post, beta, *, frac=0.6, tilt=0.0, clear=0.004, half=0.012, pad_above_deck=0.0015):
        """Plan a closed-finger push that turns the resting wand about the finish post by `beta` (rad, + = counter-
        clockwise from above): the fingertips beside the handle at `frac` of the grip (1.0 = its end), 1.5 mm above
        the deck, the hand leaning `tilt` rad outwards (away from the post, under the finish wire's arch), sweeping an
        arc about the post. Kinematic model; robot clear of the wire / posts by `clear` (the deck and the wand allowed).
        -> ((joint path, pre-configuration, tcp_at), None) or (None, why)."""
        s = 1.0 if beta > 0 else -1.0
        psi = self._yaw(Rw.apply([1, 0, 0]))
        rp = self.L.loop_radius_m + self.L.stem_length_m + frac * self.L.grip_length_m
        delta = np.arcsin(min(0.9, (self.L.grip_radius_m + half + 0.004) / rp))
        # TCP height: the fingertips' lowest point pad_above_deck above the deck (they reach 9.7 mm past the TCP along
        # the hand axis and ~11 mm to the side across the closed fingers' width)
        z = pw[2] - self.L.grip_half_thickness_m + pad_above_deck + 0.0097 * np.cos(tilt) + 0.011 * np.sin(tilt)

        def tcp_at(a):
            p = np.array([post[0] + rp * np.cos(a), post[1] + rp * np.sin(a), z])
            return p, R.from_euler("z", a) * R.from_euler("y", tilt) * R.from_euler("x", np.pi)
        a0 = psi - s * delta
        angles = a0 + s * np.linspace(0, abs(beta), max(2, int(np.degrees(abs(beta)) / 2) + 1))
        self.kin.set_gap(0.0)
        why = {}
        for seed in [getattr(self, "_last_q", self.io.q())] + self.seeds:
            p, rv = tcp_at(angles[0])
            q, ok, _, _ = self.kin.ik(p, rv, seed, iters=300)
            if not ok:
                why["ik0"] = why.get("ik0", 0) + 1; continue
            bad = [self.kin.geom_name(g1) + "|" + self.kin.geom_name(g2)
                   for dist, kind, g1, g2 in self.kin.contacts(q, wand=(pw, Rw))
                   if kind == "robot-static" and dist < clear and not self.is_deck(g1, g2)]
            if bad:
                k = "start:" + bad[0].split("__")[-2] if "__" in bad[0] else "start:" + bad[0]
                why[k] = why.get(k, 0) + 1; continue
            qs, good = [q], True
            for a in angles[1:]:
                p, rv = tcp_at(a)
                q2, ok, _, _ = self.kin.ik(p, rv, qs[-1], iters=100)
                if not ok or np.abs(q2 - qs[-1]).max() > 0.3 or not self.robot_clear(q2, clear, deck_ok=True):
                    why["arc"] = why.get("arc", 0) + 1
                    good = False; break
                qs.append(q2)
            if not good:
                continue
            p, rv = tcp_at(angles[0])
            up = self.cartesian_line(q, p, rv, p + [0, 0, 0.08], rv, step=0.004, robot_clear=clear,
                                     robot_wand=(pw, Rw), deck_ok=True)
            if up is None:
                why["lift"] = why.get("lift", 0) + 1; continue
            return (qs, up[-1], tcp_at, angles), None
        return None, why

    def push_turn(self, pw, Rw, post, beta, **kw):
        """Turn the resting wand about the finish post by `beta` with a closed-finger push (plan_push), trying the
        grip's middle to its end and an outward lean up to 40 deg. -> True if executed."""
        plan, whys = None, {}
        for frac in (0.6, 0.85, 1.0):
            for tilt in (0.0, 0.35, 0.7, 1.05, 1.3, 1.5):
                plan, why = self.plan_push(pw, Rw, post, beta, frac=frac, tilt=tilt, **kw)
                if plan is not None:
                    break
                whys[f"{frac}/{tilt}"] = why
            if plan is not None:
                break
        if plan is None:
            self.note(event="push_unplannable", beta_deg=round(float(np.degrees(beta)), 1),
                      why=json.dumps(whys)[:600])
            return False
        qs, q_pre, tcp_at, angles = plan
        self.note(event="push_turn", beta_deg=round(float(np.degrees(beta)), 1), steps=len(qs), frac=frac, tilt=tilt)
        self.set_grip(CLOSE, 0.6)
        self.wait_finger(lambda f: f > 0.7, 3.0)
        self.free_move(q_pre, wand_pose=(pw, Rw))
        p, rv = tcp_at(angles[0])
        self.line_to(p, rv, speed=0.02, robot_clear=0.004, robot_wand=(pw, Rw), deck_ok=True)
        self.move_joints(qs[1:], w_max=0.15)
        self.wait(0.5)
        p, rv = self.kin.fk(getattr(self, "_last_q", self.io.q()))
        self.line_to(p + [0, 0, 0.08], rv, speed=0.03, robot_clear=None)
        if self.io.holding():
            self.release(retreat=0.03, open_s=1.0, speed=0.015)
        self.settle(timeout=3.0)
        return True

    def hold(self, max_s=8.0):
        n = int(max_s / CTRL_DT)
        q = getattr(self, "_last_q", self.io.q())
        for _ in range(n):
            if self.send(q):
                break
        self.note(event="hold_end")

    def run_graph(self, rests, graph, start_rest=0, start_k=0):
        """Walk the rest graph from the all-pairs analysis: rests = arc lengths [start, sleeve centres..., drop];
        graph = {i: [[j, grasp], ...]} the feasible held segments. From each rest, the furthest feasible next rest
        first; a segment that cannot be planned from the measured grip (or a grip that cannot be taken) falls back to
        the next option (after letting go, when hanging). start_rest > 0: resume with the wand parked (hanging, not
        held) on that rest, the next held segment numbered start_k (a checkpoint replayed up to a 'parked' event)."""
        last = len(rests) - 1
        g = {int(k): v for k, v in graph.items()}
        good = {last}                      # rests from which the drop is reachable in the graph
        for _ in range(last + 1):
            good |= {i for i, opts in g.items() if any(j in good for j, _ in opts)}
        graph = {i: [o for o in opts if o[0] in good] for i, opts in g.items()}
        if start_rest:
            i, k, holding_from_start = int(start_rest), int(start_k), False
            self.note(event="resume_parked", i=i, k=k)
        else:
            self.pick()
            i, k, holding_from_start = 0, 0, True
        while i != last:
            opts = sorted(graph.get(i, []), key=lambda o: -o[0])
            done = False
            for j, grasp in opts:
                seg = dict(s0=rests[i] if i > 0 else 0.004, s1=rests[j], grasp=grasp, end="drop" if j == last else "park")
                if not holding_from_start:
                    if self.io.holding():
                        self.release(retreat=0.05, open_s=1.0, speed=0.015); self.settle(timeout=4.0)
                    try:
                        self.regrasp(k, seg)
                    except RuntimeError as exc:
                        self.note(event="graph_regrasp_failed", i=i, j=j, grasp=grasp, why=str(exc)[:600])
                        continue
                try:
                    self.traverse(k, seg)
                except RuntimeError as exc:
                    if "not plannable" not in str(exc):
                        raise
                    self.note(event="graph_option_failed", i=i, j=j, grasp=grasp, why=str(exc)[:600])
                    if holding_from_start:
                        continue
                    continue
                if seg["end"] == "park":
                    self.park(k, seg)
                self.note(event="graph_step", i=i, j=j, grasp=grasp)
                i, k, holding_from_start, done = j, k + 1, False, True
                break
            if not done and i == 0 and len(getattr(self, "_used_picks", ())) < 6:
                # the start grip as measured cannot reach any rest: put the wand back on the deck and pick it again
                # with another variant (the grip pose in the hand varies by a few mm / deg)
                pw0, Rw0 = self._deck_pose
                self.snap_back = True
                self.move_wand(pw0 + [0, 0, 0.004], Rw0, speed=0.01)
                self.move_wand(pw0 + [0, 0, 0.0005], Rw0, speed=0.004)
                self.release(retreat=0.06)
                self.settle(timeout=3.0)
                self.note(event="repick", tried=sorted(self._used_picks))
                self.pick()
                continue
            if not done:
                raise RuntimeError(f"rest graph: no way on from rest {i}")
        self.drop()
        self.final_place()
        self.hold()

    def run(self):
        self.pick()
        for k, seg in enumerate(self.segments):
            if k > 0:
                self.regrasp(k, seg)
            self.traverse(k, seg)
            if seg.get("end", "park") == "park":
                self.park(k, seg)
        self.drop()
        self.final_place()
        self.hold()
