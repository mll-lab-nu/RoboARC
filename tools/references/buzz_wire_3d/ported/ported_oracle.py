"""A reference for a ported buzz_wire_3d tier (easy / medium / hard: no rest sleeves, the ring must stay on the wire
from the start post to the finish post under one grasp), recorded live in one fresh Session of the frozen instance.

The rd tier oracle's machinery (topogym.planning.buzz_wire_3d.redesign: kinematics on the Session's own model, the
single-grasp frontier search along the wire, staged deck grasps, the drop and the grasp-turn-release finish about the
finish post) with a design adapter for the ported layouts. Every row goes through the standard per-step joint limit
(topogym.planning.standard_rate.RateLimiter, spread mode), so the privileged reference is also a standard one.

Stages: [route rows [0, cut) of a released route] -> [pick with --pick specs] -> traverse (measured grasp) to the drop
point -> drop -> final place rounds -> hold until the episode ends (tail). Saved only on success.
usage: ported_oracle.py <instance.json> <out_dir> [--route r.npz --cut N] [--pick spec,...] [--drop-s a,b,..]"""
import faulthandler
faulthandler.enable()
import argparse, json, os, sys, time, traceback
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

ap = argparse.ArgumentParser()
ap.add_argument("instance"); ap.add_argument("out")
ap.add_argument("--route"); ap.add_argument("--cut", type=int, default=0)
ap.add_argument("--pick", default="top:180:0.8:0:2")
ap.add_argument("--drop-back", default="0.004,0.012,0.02,0.03,0.045,0.06", help="drop points, m of wire before its end")
ap.add_argument("--drop-offset", type=float, default=0.0)
ap.add_argument("--drop-open-s", type=float, default=0.6); ap.add_argument("--drop-retreat", type=float, default=0.06)
ap.add_argument("--lags", default="0", help="ring lag angles (deg) the traverse may use")
ap.add_argument("--psi-n", type=int, default=24); ap.add_argument("--ds", type=float, default=0.003)
ap.add_argument("--dq-max", type=float, default=0.35)
ap.add_argument("--v-ring", type=float, default=0.03); ap.add_argument("--w-max", type=float, default=0.6)
ap.add_argument("--rng", type=int, default=0); ap.add_argument("--tail", type=int, default=400)
ap.add_argument("--clear", type=float, default=0.002)
ap.add_argument("--resume", help="actions .npy: replay its first --resume-rows rows (the wand already dropped), then place");
ap.add_argument("--resume-rows", type=int); ap.add_argument("--no-place", action="store_true"); ap.add_argument("--place2", action="store_true"); ap.add_argument("--landing-sweep", default=None, help="ups_mm:turns_deg grid, e.g. 0,10,20:-60,-40,-20,0,20,40,60 (resume-drop: trial drops, rewound; then the best band landing is executed)"); ap.add_argument("--band", default="-4,21"); ap.add_argument("--resume-drop", action="store_true", help="resume: the wand is still held at the post's foot; drop it first"); ap.add_argument("--drop-soft", action="store_true"); ap.add_argument("--drop-turn", type=float, default=0.0, help="deg: turn the held wand about the wire before the drop"); ap.add_argument("--push-search", type=int, default=0, help="pushes chosen by trial (rewound), this many levels, aiming at a graspable heading"); ap.add_argument("--retreat", action="store_true", help="resume: only lift the hand clear of the resting wand, then hold"); ap.add_argument("--release", default=None, choices=[None, "centre", "level", "lower", "heldturn"], help="hand the held wand to buzz_finish's release instead of the drop"); ap.add_argument("--slide", type=float, default=None, help="after the finish: a centring push at the handle end, extra mm"); ap.add_argument("--push-first", type=float, default=0.0, help="closed-finger pushes about the post first, until the heading is within this many deg"); ap.add_argument("--adaptive", action="store_true", help="finish with buzz_finish.run_finish (the released 01/02 references' finish)"); ap.add_argument("--soft", action="store_true"); ap.add_argument("--soft-open", type=float, default=0.04)
ap.add_argument("--guide", action="store_true", help="after the cut, follow the route's own wand path (kinematic, the "
                "measured grasp), re-centred on the wire, up to --guide-stop m before the wire's end; then plan the rest")
ap.add_argument("--guide-stop", type=float, default=0.10)
ap.add_argument("--guide-lag-max", type=float, default=30.0)
ap.add_argument("--guide-every", type=int, default=1)
ap.add_argument("--retime", action="store_true", help="after the cut, play the route's own rows, subdivided in joint "
                "space (at most --cap rad per row, --tight-cap where the kinematic wand-wire clearance is under --tight-mm), "
                "until the ring is --guide-stop m before the wire's end; then plan the rest")
ap.add_argument("--cap", type=float, default=0.01); ap.add_argument("--tight-cap", type=float, default=0.004)
ap.add_argument("--tight-mm", type=float, default=3.0)
ap.add_argument("--ki", type=float, default=0.0, help="retimed route: integral joint-error compensation gain per row")
ap.add_argument("--ki-max", type=float, default=0.03)
ap.add_argument("--nudge", action="store_true", help="the route's wand path with the hand turned about the wire's tangent "
                "(through the ring centre) by a smooth per-node angle, chosen by DP so the closed gripper and the arm clear "
                "the wire and the deck (live finger state, a 4 mm distance margin on a model copy)")
ap.add_argument("--nudge-max", type=float, default=60.0); ap.add_argument("--nudge-step", type=float, default=5.0)
ap.add_argument("--shift-dx", type=float, default=0.0, help="take the route's traversal grasp this far further out along "
                "the handle (m): its approach rows after the deck release, translated along the lying wand's handle")
ap.add_argument("--robot-clear-mm", type=float, default=2.0)
ap.add_argument("--plan", help="nplan.py prep + plan: replay the prep's prefix rows, then the plan's joint path"); ap.add_argument("--node-mm", type=float, default=2.0)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
from rcb_topogym import Session
from topogym.planning.standard_rate import RateLimiter
from topogym.planning.buzz_wire_3d.redesign.io import SessionIO
from topogym.planning.buzz_wire_3d.redesign import oracle as O
from topogym.planning.buzz_wire_3d.redesign.feasibility import explore, extract
from topogym.planning.buzz_wire_3d.redesign.wandpath import tcp_from_wand
from topogym.assets.geometry.buzz_wire_3d import endpoint_pose
logf = open(out / "log.jsonl", "w")


def log(**kv):
    logf.write(json.dumps(kv, default=str) + "\n"); logf.flush()
    print(json.dumps(kv, default=str), flush=True)


class PortedDesign:
    """The rd oracle's design interface on a ported tier's own task object (no sleeves, the frozen scene's mount)."""
    sleeves = ()
    sleeve_radius_m = 0.0
    robot_mount = None

    def __init__(self, task):
        self.task = task

    def layout(self):
        return self.task.layout

    def wire(self):
        return self.task.route

    def origin(self):
        return np.asarray(self.task.origin, float)


class LimitedIO(SessionIO):
    def __init__(self, session, **kw):
        super().__init__(session, **kw)
        self.limiter = RateLimiter(session, mode="spread")

    def step(self, q, grip):
        row = np.r_[np.asarray(q, float), float(grip)].astype(np.float32)
        self.done = bool(self.limiter.step(row))
        self.rows += 1
        return self.done


class PortedOracle(O.TierOracle):
    def send(self, q, grip=None, n=1):
        """rd send + an integral joint-error offset (ki > 0): the drives' quasi-static lag would put the TCP mm off."""
        grip = self.grip if grip is None else grip
        self.grip = grip
        q = np.asarray(q, float)
        for _ in range(n):
            if self.io.step(q + self.ioff, grip):
                return True
            if self.ki:
                self.ioff = np.clip(self.ioff + self.ki * (q - self.io.q()), -self.ki_max, self.ki_max)
        return False

    def release_soft(self, part=None, lift=0.03, open_s=1.0):
        """Let go without kicking the wand: open only until the pads clear the grip (the weld lets go at the first
        opening command), lift the hand straight up, then open fully. Wide open, the 2.62 pads swing ~1 cm down and
        would strike the wand / deck right next to it."""
        if part is None:                     # just below where the pads stand on the grip: no squeeze, ~1 mm of travel
            part = max(0.0, self.finger() - getattr(self, "soft_open", 0.04))
        def snap(tag):
            pw_, Rw_ = self.io.wand_pose()
            import mujoco as _mj
            m_, d_ = self.io.model, self.io.data
            cs = set()
            for j_ in range(d_.ncon):
                c_ = d_.contact[j_]
                if m_.geom_bodyid[c_.geom1] == self.io.wand or m_.geom_bodyid[c_.geom2] == self.io.wand:
                    n1 = (_mj.mj_id2name(m_, _mj.mjtObj.mjOBJ_GEOM, c_.geom1) or "").split("__")[-2:][0]
                    n2 = (_mj.mj_id2name(m_, _mj.mjtObj.mjOBJ_GEOM, c_.geom2) or "").split("__")[-2:][0]
                    cs.add((n1, n2))
            self.note(event="rel_snap", tag=tag, wand=np.round(pw_, 4).tolist(), yaw=round(float(np.degrees(self._yaw(Rw_.apply([1, 0, 0])))), 2),
                      finger=round(self.finger(), 3), contacts=sorted(cs)[:6])
        snap("before")
        self.set_grip(part, open_s)
        snap("opened")
        self.wait(0.6)
        snap("waited")
        q = getattr(self, "_last_q", self.io.q())
        p, r = self.kin.fk(q)
        for d in (lift, 0.6 * lift, 0.3 * lift):
            try:
                self.line_to(p + np.array([0, 0, d]), r, speed=0.01, robot_clear=None)
                break
            except RuntimeError:
                continue
        snap("lifted")
        self.set_grip(O.OPEN, 0.6)
        self.wait(0.3)
        snap("open")
        self.note(event="released_soft", finger=round(self.finger(), 3))

    def push_to_heading(self, within_deg, max_pushes=5):
        """Closed-finger pushes about the finish post (rd push_turn) towards the finish heading, until within_deg."""
        tp_, tr_ = endpoint_pose(self.route, 1)
        target_p = tp_ + self.origin
        hy_t = self._yaw(tr_.apply([1, 0, 0]))
        for k in range(max_pushes):
            self.settle(timeout=3.0)
            pw, Rw = self.io.wand_pose()
            d = (hy_t - self._yaw(Rw.apply([1, 0, 0])) + np.pi) % (2 * np.pi) - np.pi
            self.note(event="push_round", k=k, dyaw_deg=round(float(np.degrees(d)), 1))
            if abs(d) < np.deg2rad(within_deg):
                return True
            post = np.array([target_p[0], target_p[1], pw[2]])
            done = False
            for lim in (25, 15, 10):
                beta = float(np.clip(d, -np.deg2rad(lim), np.deg2rad(lim)))
                for pad in (0.0015, 0.006, 0.012):
                    if self.push_turn(pw, Rw, post, beta, pad_above_deck=pad):
                        done = True
                        break
                if done:
                    break
            if not done:
                return False
        return False

    def push_slide(self, extra=0.0, pad=0.004, approach_gap=0.012, speed=0.01):
        """Centre the resting wand on the finish mount: a closed-finger push at the handle's end, straight along the
        handle towards the ring, by the ring's offset from the target projected on the handle (plus `extra`)."""
        tp_, tr_ = endpoint_pose(self.route, 1)
        target = tp_ + self.origin
        self.settle(timeout=3.0)
        pw, Rw = self.io.wand_pose()
        h = Rw.apply([1., 0, 0]); h[2] = 0; h /= np.linalg.norm(h)
        v = target - pw; v[2] = 0
        along = float(-(v @ h))                      # how far the wand must move towards its ring (along -h)
        self.note(event="slide_plan", offset_mm=round(1e3 * float(np.linalg.norm(v)), 2), along_mm=round(1e3 * along, 2))
        if abs(along) < 0.001:
            return False
        L = self.L
        a = float(np.arctan2(h[1], h[0]))
        rv = R.from_euler("z", a + np.pi / 2) * R.from_euler("x", np.pi)
        if along > 0:                                # push the handle's end towards the ring
            end = pw + h * (L.loop_radius_m + L.stem_length_m + L.grip_length_m)
            z = pw[2] - L.grip_half_thickness_m + pad + 0.0097
            p0 = np.array([end[0], end[1], z]) + h * approach_gap
            p1 = p0 - h * (approach_gap + along + extra)
        else:                                        # push the ring's far rim towards the handle
            rim = pw - h * (L.loop_radius_m + L.loop_tube_radius_m)
            z = pw[2] - L.loop_tube_radius_m + pad + 0.0097
            p0 = np.array([rim[0], rim[1], z]) - h * approach_gap
            p1 = p0 + h * (approach_gap - along + extra)
        self.kin.set_gap(0.0)
        q0 = self.ik_any(p0 + [0, 0, 0.06], rv, prefer=getattr(self, "_last_q", self.io.q()))
        if q0 is None:
            self.note(event="slide_unplannable", why="pre-pose IK"); return False
        self.set_grip(O.CLOSE, 0.6)
        self.wait_finger(lambda f: f > 0.7, 3.0)
        try:
            self.free_move(q0, wand_pose=(pw, Rw))
            self._last_q = q0
            self.line_to(p0, rv, speed=0.02, robot_clear=None)
            self.line_to(p1, rv, speed=speed, robot_clear=None)
            self.wait(0.5)
            self.line_to(p1 + [0, 0, 0.06], rv, speed=0.02, robot_clear=None)
        except RuntimeError as exc:
            self.note(event="slide_failed", why=str(exc)); return False
        self.settle(timeout=3.0)
        pw2, _ = self.io.wand_pose()
        self.note(event="slide_done", offset_mm=round(1e3 * float(np.linalg.norm((target - pw2)[:2])), 2))
        return True

    def retreat_clear(self, lift=0.08, speed=0.01):
        """Lift the hand straight up off the resting wand (slowly), then hold: the final settling needs the robot clear."""
        q = getattr(self, "_last_q", self.io.q())
        p, r = self.kin.fk(q)
        for d in (lift, 0.5 * lift, 0.25 * lift):
            try:
                self.line_to(p + np.array([0, 0, d]), r, speed=speed, robot_clear=None)
                break
            except RuntimeError:
                continue
        self.note(event="retreated")
        self.hold(max_s=4.0)

    def graspable_count(self, specs=None):
        pw, Rw = self.io.wand_pose()
        flip = Rw.apply([0, 0, 1])[2] < 0
        kinds = ["topb"] if flip else ["top"]
        specs = specs or [f"{k}:{r}:{f}:{p}:2" for k in kinds for r in (0, 180) for f in (0.55, 0.7, 0.8, 0.9)
                          for p in (-40, -20, 0, 20, 40)]
        return sum(1 for sp in specs if self.grasp_candidates(sp, pw, Rw, require_sweep=False, deck_ok=True, n_keep=1))

    def push_turn(self, pw, Rw, post, beta, **kw):
        """PUSH_WIDE=1: rd push_turn over a wider grid (the grip's stem side, frac 0.3, and the fingertips starting
        further from the handle, half 20 mm); otherwise rd's own."""
        if not os.environ.get("PUSH_WIDE"):
            return super().push_turn(pw, Rw, post, beta, **kw)
        plan = None
        for half in (0.012, 0.02):
            for frac in (0.3, 0.6, 0.85, 1.0):
                for tilt in (0.0, 0.35, 0.7, 1.05, 1.3):
                    plan, _ = self.plan_push(pw, Rw, post, beta, frac=frac, tilt=tilt, half=half, **kw)
                    if plan is not None:
                        break
                if plan is not None:
                    break
            if plan is not None:
                break
        if plan is None:
            self.note(event="push_unplannable", beta_deg=round(float(np.degrees(beta)), 1))
            return False
        qs, q_pre, tcp_at, angles = plan
        self.note(event="push_turn", beta_deg=round(float(np.degrees(beta)), 1), steps=len(qs), frac=frac, tilt=tilt, half=half)
        self.set_grip(O.CLOSE, 0.6)
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

    def push_search(self, levels=2):
        """Closed-finger pushes about the finish post chosen by trial: each candidate push (turn, pad height) is executed,
        the result scored (graspable from above, then the remaining heading error), and rewound; the best is executed
        again (deterministic), up to `levels` pushes or until the wand is graspable within 25 deg of the heading."""
        sys.path.insert(0, str(Path(O.__file__).resolve().parents[5] / "tools/references/buzz_wire_3d"))
        import buzz_finish as bf
        tp_, tr_ = endpoint_pose(self.route, 1)
        target_p = tp_ + self.origin
        hy_t = self._yaw(tr_.apply([1, 0, 0]))
        sess = self.io.s

        def dyaw():
            pw, Rw = self.io.wand_pose()
            return (hy_t - self._yaw(Rw.apply([1, 0, 0])) + np.pi) % (2 * np.pi) - np.pi
        for lvl in range(levels):
            self.settle(timeout=3.0)
            d0 = dyaw()
            g0 = self.graspable_count()
            self.note(event="push_search_level", level=lvl, dyaw_deg=round(float(np.degrees(d0)), 1), graspable=g0)
            if g0 and abs(d0) < np.deg2rad(25):
                return True
            pw, Rw = self.io.wand_pose()
            post = np.array([target_p[0], target_p[1], pw[2]])
            cp = bf.checkpoint(sess); last_q, grip = getattr(self, "_last_q", self.io.q()), self.grip
            sgn = 1.0 if d0 > 0 else -1.0
            results = []
            for mag in ((5, 8, 10, 15, 20) if os.environ.get("PUSH_WIDE") else (10, 15, 20, 25, 30, 40)):
                for pad in (0.0015, 0.006, 0.012):
                    beta = sgn * np.deg2rad(mag)
                    ok = self.push_turn(pw, Rw, post, beta, pad_above_deck=pad)
                    if ok and sess.episode_end is None:
                        self.settle(timeout=3.0)
                        d1 = dyaw(); g1 = self.graspable_count()
                        pw1, _ = self.io.wand_pose()
                        results.append((mag, pad, float(np.degrees(d1)), g1, float(np.linalg.norm((pw1 - target_p)[:2]))))
                        self.note(event="push_try", mag=mag, pad=pad, dyaw_deg=round(results[-1][2], 1), graspable=g1)
                    bf.rewind(sess, cp); self._last_q, self.grip = last_q, grip
            if not results:
                return False
            best = min(results, key=lambda r: (0 if (r[3] and abs(r[2]) < 25) else 1, abs(r[2])))
            self.note(event="push_choose", mag=best[0], pad=best[1], dyaw_deg=round(best[2], 1), graspable=best[3])
            self.push_turn(pw, Rw, post, sgn * np.deg2rad(best[0]), pad_above_deck=best[1])
        self.settle(timeout=3.0)
        return self.graspable_count() > 0

    def held_wand_move(self, s_ring, turn, step_s=0.002, step_turn=np.deg2rad(5)):
        """Move the held wand along the vertical end of the wire to arc length s_ring and turn it about the wire by
        `turn` (rad), by IK continuation with penetration checks (robot vs static except the deck, wand metal vs wire
        metal >= 0.3 mm). Returns the joint list or None."""
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from nudge_lib import Dist
        from topogym.planning.buzz_wire_3d.redesign.feasibility import classify
        if not hasattr(self, "_dist"):
            self._dist = Dist(self.kin, classify(self.kin))
        q = np.asarray(getattr(self, "_last_q", self.io.q()), float)
        self.kin.d.qpos[:] = self.io.data.qpos[:]
        pw, Rw = self.held_wand(q)
        s0 = float(self.path.s[int(np.argmin(np.linalg.norm(self.path.p - pw, axis=1)))])
        n_s = max(1, int(np.ceil(abs(s_ring - s0) / step_s))); n_t = max(1, int(np.ceil(abs(turn) / step_turn)))
        poses = []
        for i in range(1, n_s + 1):                   # along the wire, orientation kept
            sv = s0 + (s_ring - s0) * i / n_s
            poses.append((self.path.p[self.path.idx(sv)].copy(), Rw))
        p_top = poses[-1][0] if poses else pw
        t_ax = self.path.t[self.path.idx(s_ring)]
        for i in range(1, n_t + 1):
            poses.append((p_top, R.from_rotvec(t_ax * turn * i / n_t) * Rw))
        qs = []
        import mujoco
        for p_, R_ in poses:
            tp, tr = tcp_from_wand(p_, R_, self.g)
            q2, ok, _, _ = self.kin.ik(tp, tr, q, iters=100)
            if not ok or np.abs(q2 - q).max() > 0.3:
                return None
            self.kin.set_q(q2); self.kin.set_wand(p_, R_); mujoco.mj_kinematics(self.kin.m, self.kin.d)
            rs, pair, deck = self._dist.robot_static(dmax=0.003)
            if rs < 0 or deck < -0.0005 or self._dist.wand_wire() < 0.0003:
                return None
            qs.append(q2); q = q2
        return qs

    def landing_sweep(self, ups, turns, band):
        sys.path.insert(0, str(Path(O.__file__).resolve().parents[5] / "tools/references/buzz_wire_3d"))
        import buzz_finish as bf
        tp_, tr_ = endpoint_pose(self.route, 1)
        hy_t = self._yaw(tr_.apply([1, 0, 0]))
        sess = self.io.s
        cp = bf.checkpoint(sess); last_q, grip = getattr(self, "_last_q", self.io.q()), self.grip
        q_now = np.asarray(last_q, float)
        pw, _ = self.held_wand(q_now)
        s_now = float(self.path.s[int(np.argmin(np.linalg.norm(self.path.p - pw, axis=1)))])
        results = []
        for up in ups:
            for tu in turns:
                up_path = self.held_wand_move(s_now - up, np.deg2rad(tu)) if (up or tu) else []
                if up_path is None:
                    self.note(event="landing_try", up_mm=up * 1e3, turn=tu, plannable=False); continue
                down = []
                self._last_q = up_path[-1] if up_path else q_now
                if up:                                   # back down, as far as it plans
                    for back in (0.0, 0.002, 0.004, 0.008, 0.012, 0.02):
                        dp = self.held_wand_move(s_now - back, 0.0)
                        if dp is not None:
                            down = dp; break
                ok = True
                for q_ in up_path + down:
                    if self.send(q_):
                        ok = False; break
                    self._last_q = q_
                if ok and sess.episode_end is None:
                    for _ in range(int(0.8 / O.CTRL_DT)):
                        if self.send(self._last_q):
                            ok = False; break
                if ok and sess.episode_end is None:
                    self.drop()
                    self.settle(timeout=4.0)
                    m = self.io.metrics()
                    pw2, Rw2 = self.io.wand_pose()
                    d = float(np.degrees((hy_t - self._yaw(Rw2.apply([1, 0, 0])) + np.pi) % (2 * np.pi) - np.pi))
                    res = dict(up_mm=up * 1e3, turn=tu, dyaw=round(d, 1), dist_mm=None if m.get("distance_to_finish_m") is None else round(1e3 * m["distance_to_finish_m"], 2),
                               end=sess.episode_end)
                else:
                    res = dict(up_mm=up * 1e3, turn=tu, end=sess.episode_end or "ended")
                results.append(res)
                self.note(event="landing_try", **res)
                bf.rewind(sess, cp); self._last_q, self.grip = last_q, grip
        good = [r for r in results if r.get("dyaw") is not None and r.get("end") is None and band[0] <= r["dyaw"] <= band[1]]
        self.note(event="landing_sweep_done", tried=len(results), in_band=good)
        return good

    def final_place2(self, rounds=8, max_turn_deg=40.0):
        """The finish on the ported deck: grasp-turn-release rounds about the finish mount. Each round grasps the
        resting wand from its upper face (top / topb), lifts it 4 mm (the ring stays below the wire's end), turns it
        about the mount by at most max_turn_deg towards the finish heading while moving the ring centre towards the
        target in proportion, lowers and releases it slowly. Spec grid as the rd place probe (roll, fraction, pitch, depth)."""
        tp_, tr_ = endpoint_pose(self.route, 1)
        target_p = tp_ + self.origin
        hy_t = self._yaw(tr_.apply([1, 0, 0]))
        tried_bad = set()
        for rnd in range(rounds):
            self.settle(timeout=3.0)
            pw, Rw = self.io.wand_pose()
            flip = Rw.apply([0, 0, 1])[2] < 0
            d = (hy_t - self._yaw(Rw.apply([1, 0, 0])) + np.pi) % (2 * np.pi) - np.pi
            dist_xy = float(np.linalg.norm(pw[:2] - target_p[:2]))
            self.note(event="place2_round", round=rnd, flip=bool(flip), dyaw_deg=round(float(np.degrees(d)), 1),
                      dist_xy_mm=round(1e3 * dist_xy, 2))
            if abs(d) < np.deg2rad(3.0) and dist_xy < 0.0015:
                break
            post = np.array([target_p[0], target_p[1], pw[2]])
            step = float(np.clip(d, -np.deg2rad(max_turn_deg), np.deg2rad(max_turn_deg)))
            frac = 1.0 if abs(d) < 1e-6 else min(1.0, abs(step / d))
            pw_rot = post + R.from_euler("z", step).apply(pw - post)
            p_end = pw_rot + frac * (np.array([target_p[0], target_p[1], pw[2]]) - pw_rot)
            p_end[2] = pw[2]
            kinds = ["topb", "top"] if flip else ["top", "topb"]
            specs = [f"{k}:{r}:{f}:{pi}:{dd}" for k in kinds for r in (0, 180, 90, 270) for dd in (2, 0)
                     for f in (0.8, 0.7, 0.9, 0.55, 1.0) for pi in (-20, 20, -10, 10, 0, -30, 30, -40, 40)]
            good = getattr(self, "_place_good", [])
            specs = [sp for sp in good if sp in specs] + [sp for sp in specs if sp not in good]
            plan, used = None, None
            for st in (step, step / 2) if abs(step) > np.deg2rad(5) else (step,):
                pe = p_end if st == step else (pw + post + R.from_euler("z", st).apply(pw - post)) / 2 * 0 + (post + R.from_euler("z", st).apply(pw - post))
                for spec in specs:
                    if spec in tried_bad:
                        continue
                    plan = self._rot_plan(spec, pw, Rw, post, st, pe, sweep=False)
                    if plan is not None:
                        used = (spec, st); break
                if plan is not None:
                    break
            if plan is None:
                self.note(event="place2_no_plan", round=rnd)
                break
            spec, st = used
            self._place_good = [spec] + [sp for sp in getattr(self, "_place_good", []) if sp != spec]
            q_g, q_pre, poses = plan
            self.kin.set_gap(0.085)
            self.set_grip(O.OPEN, 0.4)
            self.wait_finger(lambda f: f < 0.08, 3.0)
            self.free_move(q_pre, wand_pose=(pw, Rw))
            self._last_q = q_pre
            g = O.parse_grasp(spec, self.L)
            gp, gr = tcp_from_wand(pw, Rw, g)
            gap_ = gr.apply([0, 0, 1])
            self.line_to(gp - O.DECK_STAGE[0] * gap_, gr, speed=0.03, robot_clear=0.0, robot_wand=(pw, Rw), deck_ok=True)
            self.set_grip(O.DECK_STAGE[1], 0.5)
            self.wait_finger(lambda f: abs(f - O.DECK_STAGE[1]) < 0.05, 3.0)
            self.line_to(gp, gr, speed=0.02, robot_clear=0.0, robot_wand=(pw, Rw), deck_ok=True)
            self.wait(0.3)
            self.set_grip(O.CLOSE, 1.0)
            if not self.wait_holding(3.5):
                self.note(event="place2_grasp_missed", round=rnd, spec=spec)
                tried_bad.add(spec)
                self.release(retreat=0.05, open_s=1.0, speed=0.015)
                continue
            self.kin.set_gap(O.gap_for(spec))
            self.measure_grasp()
            self.note(event="place2_grasp", round=rnd, spec=spec, turn_deg=round(float(np.degrees(st)), 1))
            for pk, Rk in poses:
                self.move_wand(pk, Rk, speed=0.006)
            self.wait(0.5)
            # feedback: the arm sags / the weld gives by a few mm and deg; correct the held wand's measured pose
            # towards the round's goal (ring centre in xy, heading) before letting go
            p_goal, R_goal = poses[-1]
            pc, Rc = p_goal.copy(), R_goal
            for it in range(5):
                if not self.io.holding():
                    self.note(event="place2_lost_hold", round=rnd); break
                pm, Rm = self.io.wand_pose()
                e_p = p_goal - pm; e_p[2] = 0.0
                e_y = (self._yaw(R_goal.apply([1, 0, 0])) - self._yaw(Rm.apply([1, 0, 0])) + np.pi) % (2 * np.pi) - np.pi
                self.note(event="place2_correct", round=rnd, it=it, err_mm=round(1e3 * float(np.linalg.norm(e_p)), 2),
                          err_deg=round(float(np.degrees(e_y)), 2))
                if np.linalg.norm(e_p) < 0.0008 and abs(e_y) < np.deg2rad(0.8):
                    break
                pc = pc + e_p
                Rc = R.from_euler("z", e_y) * Rc
                try:
                    self.move_wand(pc, Rc, speed=0.003)
                except RuntimeError as exc:
                    self.note(event="place2_correct_failed", why=str(exc)); break
                self.wait(0.4)
            if getattr(self, "soft", False):
                self.release_soft()
            else:
                self.release(retreat=0.06, open_s=1.5, speed=0.015)
            self.settle(timeout=4.0)
        pw, Rw = self.io.wand_pose()
        self.note(event="placed2", dist_mm=round(1e3 * float(np.linalg.norm(pw - target_p)), 2))

    def traverse_to_drop(self, s0, drop_backs, lags):
        pw, Rw = self.held_wand(getattr(self, "_last_q", self.io.q()))
        psi = self.path.psi_toward(s0, Rw.apply([1, 0, 0]))
        j0 = int(np.round(psi / (2 * np.pi / self.psi_n))) % self.psi_n
        p_s, R_s = self.path.frame(s0, j0 * 2 * np.pi / self.psi_n)
        try:
            self.move_wand(p_s, R_s, speed=0.01)
        except RuntimeError as exc:
            self.note(event="snap_skipped", why=str(exc))
        q_start = getattr(self, "_last_q", self.io.q())
        fh = endpoint_pose(self.route, 1)[1].apply([1, 0, 0])
        fh = R.from_euler("z", np.deg2rad(self.drop_offset_deg)).apply(fh)
        L = self.path.length
        lags_r = tuple(sorted(np.deg2rad(float(x)) for x in lags))
        l0 = int(np.argmin(np.abs(np.asarray(lags_r))))
        res = None
        for b in drop_backs:
            s1 = L - b
            if s1 <= s0 + 0.01:
                continue
            r_try = explore(self.kin, self.path, self.g, s0=s0, s1=s1, ds=self.ds, psi_n=self.psi_n,
                            start_qs=[(j0, l0, q_start)], keep=True, clear_robot=a.clear, per_bin=4,
                            dq_max=self.dq_max, lags=lags_r)
            self.note(event="drop_try", s1=round(s1, 4), complete=r_try["complete"], reached=r_try["reached_s"],
                      why=dict(sorted(r_try["whys"].items(), key=lambda kv: -kv[1])[:3]))
            if r_try["complete"]:
                res = r_try
                break
            if res is None or (r_try["reached_s"] or 0) > (res["reached_s"] or 0):
                res = r_try
        if res is None or not res["complete"]:
            raise RuntimeError(f"traverse not plannable (reached {None if res is None else res['reached_s']})")
        jB = int(np.round(self.path.psi_toward(s1, fh) / (2 * np.pi / self.psi_n))) % self.psi_n
        pth = None
        for tol in (2, 4, 6, 12):
            pth = extract(res, end_psi=jB, end_lag=l0, psi_tol=tol)
            if pth is None:
                pth = extract(res, end_psi=jB, end_lag=None, psi_tol=tol)
            if pth is not None:
                self.note(event="drop_heading", psi_tol=tol, s1=round(s1, 4))
                break
        if pth is None:
            raise RuntimeError("no path to the drop")
        qs = self.densify(pth)
        self.note(event="segment_planned", s0=round(s0, 4), s1=round(s1, 4), nodes=len(pth), rows_planned=len(qs))
        if self.track(qs[1:]):
            raise RuntimeError("the episode ended during the traverse")
        if self.io.touched():
            raise RuntimeError("the ring touched the wire")
        self.note(event="traverse_done")

    def retimed_route(self, route_rows, s_stop, cap, tight_cap, tight_mm):
        """The released route's rows, slower: each row's joint step subdivided (linear in joint space) to at most `cap`
        rad per control step, `tight_cap` where the kinematic wand-wire clearance (commanded joints, the measured grasp)
        is under tight_mm; stops once the commanded ring is s_stop m before the wire's end."""
        import mujoco
        sys.path.insert(0, str(Path(O.__file__).resolve().parents[5] / "tools/references/buzz_wire_3d"))
        import buzzlib
        from topogym.planning.buzz_wire_3d.redesign.wandpath import wand_from_tcp
        w = self.io.w
        wand_ids, wire_ids = buzzlib.role_geoms(w, "wand_metal"), buzzlib.role_geoms(w, "wire_metal")
        m = self.io.model

        class KW:
            pass
        kw = KW(); kw.model, kw.data = m, self.kin.d

        def kclear(q):
            tp, tr = self.kin.fk(q)
            wp, wr = wand_from_tcp(tp, tr, self.g)
            self.kin.set_wand(wp, wr); mujoco.mj_kinematics(m, self.kin.d)
            a1, b1, r1, _ = buzzlib.capsules(kw, wand_ids); a2, b2, r2, _ = buzzlib.capsules(kw, wire_ids)
            return float((buzzlib.segment_distance(a1, b1, a2, b2) - r1[:, None] - r2[None, :]).min()), wp
        q_prev = np.asarray(getattr(self, "_last_q", self.io.q()), float)
        s_prev, n_rows, n_tight, cmin = 0.0, 0, 0, 1.0
        for k, row in enumerate(route_rows):
            q1 = np.asarray(row[:7], float)
            c, wp = kclear(q1)
            cmin = min(cmin, c)
            lo, hi = max(0, self.path.idx(s_prev - 0.03)), min(len(self.path.s), self.path.idx(s_prev + 0.03) + 1)
            s_prev = float(self.path.s[lo + int(np.argmin(np.linalg.norm(self.path.p[lo:hi] - wp, axis=1)))])
            cp = tight_cap if c < 1e-3 * tight_mm else cap
            n_tight += c < 1e-3 * tight_mm
            n = max(1, int(np.ceil(np.abs(q1 - q_prev).max() / cp)))
            for j in range(1, n + 1):
                qd = q_prev + (q1 - q_prev) * j / n
                if self.send(qd, float(row[7])):
                    raise RuntimeError(f"the episode ended in the retimed route (route row {k})")
                n_rows += 1
            q_prev = q1
            self._last_q = q1
            if k % 200 == 0:
                self.note(event="retime", ioff_max=round(float(np.abs(self.ioff).max()), 4), k=k, s_kin=round(s_prev, 4), clear_mm=round(1e3 * c, 2))
            if s_prev >= self.path.length - s_stop:
                break
        self.note(event="retime_done", route_rows=k + 1, rows=n_rows, tight=int(n_tight), kin_clear_min_mm=round(1e3 * cmin, 2),
                  s_kin=round(s_prev, 4))
        return float(self.io.metrics().get("route_cursor_m") or s_prev)

    def nudged_route(self, route_rows, s_stop, *, dmax=60.0, dstep=5.0, rclear=0.002, node_mm=2.0, cap=0.01,
                     tight_cap=0.004):
        import mujoco
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from nudge_lib import Dist
        from topogym.planning.buzz_wire_3d.redesign.kin import PandaKin
        from topogym.planning.buzz_wire_3d.redesign.wandpath import wand_from_tcp
        from topogym.planning.buzz_wire_3d.redesign.feasibility import classify
        kin = PandaKin(self.io.model)
        kin.d.qpos[:] = self.io.data.qpos[:]                       # the live finger state (closed on the grip)
        dist = Dist(kin, classify(kin))
        path, g = self.path, self.g
        g_route = (g[0] - np.array([getattr(self, 'shift_dx', 0.0), 0, 0]), g[1])
        # route nodes (kinematic): wand pose from the commanded joints, a node every node_mm of ring travel or 3 deg
        nodes, last = [], None
        s_prev = 0.0
        for k, row in enumerate(route_rows):
            qr = np.asarray(row[:7], float)
            tp, tr = kin.fk(qr)
            wp, wr = wand_from_tcp(tp, tr, g_route)
            lo, hi = max(0, path.idx(s_prev - 0.03)), min(len(path.s), path.idx(s_prev + 0.03) + 1)
            i = lo + int(np.argmin(np.linalg.norm(path.p[lo:hi] - wp, axis=1)))
            s_prev = float(path.s[i])
            if last is not None and np.linalg.norm(wp - last[0]) < 1e-3 * node_mm and (wr * last[1].inv()).magnitude() < np.deg2rad(3):
                continue
            nodes.append((k, s_prev, qr, wp, wr, path.t[i].copy()))
            last = (wp, wr)
            if s_prev >= path.length - s_stop:
                break
        D = np.deg2rad(np.arange(-dmax, dmax + 1e-9, dstep))
        N, M = len(nodes), len(D)
        feas = np.zeros((N, M), bool); Q = np.zeros((N, M, 7)); margin = np.full((N, M), -1.0)
        t0 = time.time()
        for n_, (k, s, qr, wp, wr, tg) in enumerate(nodes):
            for j, dl in enumerate(D):
                rot = R.from_rotvec(tg * dl)
                w2 = (wp, rot * wr)
                tp2, tr2 = tcp_from_wand(wp, rot * wr, g)
                seed = Q[n_ - 1, j] if n_ and feas[n_ - 1, j] else qr
                q, ok, _, _ = kin.ik(tp2, tr2, seed, iters=80)
                if not ok:
                    q, ok, _, _ = kin.ik(tp2, tr2, qr, iters=150)
                if not ok:
                    continue
                kin.set_q(q); kin.set_wand(*w2); mujoco.mj_kinematics(kin.m, kin.d)
                rmin, pair, deck = dist.robot_static()
                if rmin < rclear or deck < -0.0005 or dist.wand_wire() < 0.0003:
                    continue
                feas[n_, j], Q[n_, j], margin[n_, j] = True, q, rmin
        self.note(event="nudge_table", nodes=N, feasible_frac=round(float(feas.mean()), 3),
                  zero_ok=round(float(feas[:, M // 2].mean()), 3), wall_s=round(time.time() - t0, 1))
        # DP: |step| <= 1 grid step per node, joint jump <= 0.3 rad; cost = sum |delta| (deg/10) - small margin bonus
        INF = 1e18
        cost = np.full((N, M), INF); back = np.full((N, M), -1, int)
        cost[0] = np.where(feas[0], np.abs(np.rad2deg(D)) / 10, INF)
        for n_ in range(1, N):
            for j in np.flatnonzero(feas[n_]):
                best, bj = INF, -1
                for jj in (j - 1, j, j + 1):
                    if 0 <= jj < M and cost[n_ - 1, jj] < INF and np.abs(Q[n_, j] - Q[n_ - 1, jj]).max() <= 0.3:
                        c = cost[n_ - 1, jj] + abs(np.rad2deg(D[j])) / 10 + (0.5 if jj != j else 0)
                        if c < best:
                            best, bj = c, jj
                if bj >= 0:
                    cost[n_, j], back[n_, j] = best, bj
            if not np.isfinite(cost[n_]).any() or (cost[n_] >= INF).all():
                self.note(event="nudge_dead_end", node=n_, s=round(nodes[n_][1], 4), route_row=nodes[n_][0],
                          feasible=int(feas[n_].sum()))
                raise RuntimeError(f"nudge: no smooth feasible chain at s={nodes[n_][1]:.3f}")
        j = int(np.argmin(cost[-1])); seq = []
        for n_ in range(N - 1, -1, -1):
            seq.append(j); j = back[n_, j] if n_ else j
        seq = seq[::-1]
        qs = [Q[n_, j] for n_, j in enumerate(seq)]
        mg = [margin[n_, j] for n_, j in enumerate(seq)]
        self.note(event="nudge_planned", nodes=N, s_end=round(nodes[-1][1], 4), max_delta_deg=float(np.rad2deg(np.abs(D[seq]).max())),
                  min_robot_clear_mm=round(1e3 * min(mg), 2))
        # execute: from the current command to node 0, then node to node, joint-linear, capped per row
        q_prev = np.asarray(getattr(self, "_last_q", self.io.q()), float)
        for n_, q1 in enumerate(qs):
            cp = tight_cap if mg[n_] < 0.004 else cap
            n = max(1, int(np.ceil(np.abs(q1 - q_prev).max() / cp)))
            for jj in range(1, n + 1):
                if self.send(q_prev + (q1 - q_prev) * jj / n):
                    raise RuntimeError(f"the episode ended in the nudged route (node {n_}, s {nodes[n_][1]:.3f})")
            q_prev = q1
            self._last_q = q1
            if n_ % 100 == 0:
                self.note(event="nudge_exec", node=n_, s=round(nodes[n_][1], 4), delta_deg=round(float(np.rad2deg(D[seq[n_]])), 1))
        self.note(event="nudge_done")
        return float(self.io.metrics().get("route_cursor_m") or nodes[-1][1])

    def guided_traverse(self, route_rows, s_stop, lag_max_deg=30.0, every=1):
        """Follow the released route's wand path: per route row the wand pose its joints give with the measured grasp
        (FK, kinematic), projected onto the wire (nearest arc length, continuity window), its handle angle and lag kept;
        IK of that centred pose with the measured grasp, seeded by the previous node; then densify + track (rd)."""
        from topogym.planning.buzz_wire_3d.redesign.wandpath import wand_from_tcp
        from topogym.planning.buzz_wire_3d.redesign.feasibility import node_ok
        path, g = self.path, self.g
        pw, Rw = self.held_wand(getattr(self, "_last_q", self.io.q()))
        s_prev = float(path.s[int(np.argmin(np.linalg.norm(path.p - pw, axis=1)))])
        q_prev = np.asarray(getattr(self, "_last_q", self.io.q()), float)
        nodes = [(s_prev, path.psi_toward(s_prev, Rw.apply([1, 0, 0])), 0.0, q_prev)]
        bad, ik_fail, last_psi = 0, 0, nodes[0][1]
        step = path.s[1] - path.s[0]
        for k, row in enumerate(route_rows[::every]):
            qr = np.asarray(row[:7], float)
            tp, tr = self.kin.fk(qr)
            wp, wr = wand_from_tcp(tp, tr, g)
            lo, hi = max(0, path.idx(s_prev - 0.03)), min(len(path.s), path.idx(s_prev + 0.03) + 1)
            i = lo + int(np.argmin(np.linalg.norm(path.p[lo:hi] - wp, axis=1)))
            s = float(path.s[i])
            psi = path.psi_toward(s, wr.apply([1, 0, 0]))
            n = wr.apply([0, 0, 1.])
            if n @ path.t[i] < 0:
                n = -n
            lag = float(np.arctan2(path.b[i] @ np.cross(path.t[i], n), path.t[i] @ n))
            lag = float(np.clip(lag, -np.deg2rad(lag_max_deg), np.deg2rad(lag_max_deg)))
            dpsi = abs((psi - last_psi + np.pi) % (2 * np.pi) - np.pi)
            if abs(s - nodes[-1][0]) < 0.003 and dpsi < np.deg2rad(4):
                continue
            fp, fr = path.frame(s, psi, lag)
            ttp, ttr = tcp_from_wand(fp, fr, g)
            q, ok, _, _ = self.kin.ik(ttp, ttr, q_prev, iters=100)
            if not ok or np.abs(q - q_prev).max() > 0.35:
                q, ok, _, _ = self.kin.ik(ttp, ttr, qr, iters=200)
            if not ok:
                ik_fail += 1
                continue
            good, why = node_ok(self.kin, q, (fp, fr), clear_robot=0.0, s_ring=s)
            if not good:
                bad += 1
                if bad < 20:
                    self.note(event="guide_node_unclear", s=round(s, 4), why=why)
            nodes.append((s, psi, lag, q))
            q_prev, s_prev, last_psi = q, s, psi
            if s >= path.length - s_stop:
                break
        self.note(event="guide_planned", nodes=len(nodes), s_end=round(nodes[-1][0], 4), unclear=bad, ik_fail=ik_fail)
        qs = self.densify(nodes)
        if self.track(qs[1:]):
            raise RuntimeError("the episode ended during the guided traverse")
        if self.io.touched():
            raise RuntimeError("the ring touched the wire (guided)")
        self.note(event="guide_done")
        return nodes[-1][0]



def finish_adaptive():
    """The ported very_easy / easier references' finish (tools/references/buzz_wire_3d/buzz_finish.py,
    run_finish: grasp-turn rounds from the measured pose, the staged slow final release), on the same Session."""
    sys.path.insert(0, str(Path(O.__file__).resolve().parents[5] / "tools/references/buzz_wire_3d"))
    import buzz_finish
    from topogym.planning.standard_rate import rate_limited_runner
    runner = rate_limited_runner(s)
    (out / "finish").mkdir(exist_ok=True)
    bias = float(os.environ.get("FINISH_BIAS_MM", "0")) * 1e-3
    if bias:                       # aim the final turn's ring centre this far along the finish heading (the release drift)
        from topogym.assets.geometry.buzz_wire_3d import endpoint_pose as _ep
        hx = _ep(s.env._task.route, 1)[1].apply([1., 0, 0]); hx[2] = 0; hx /= np.linalg.norm(hx)
        base = [(bias * hx[0], bias * hx[1])]
        buzz_finish.OFFSETS[:] = [(bx + ox, by + oy) for bx, by in base for ox, oy in ((0, 0), (.002, 0), (0, .002), (-.002, 0), (0, -.002))]
        log(event="finish_bias", bias_mm=bias * 1e3, offsets=buzz_finish.OFFSETS)
    rel = None if not a.release else dict(lower=buzz_finish.lower_and_release, level=buzz_finish.level_release,
                                          centre=buzz_finish.centre_release, heldturn=buzz_finish.held_turn_release)[a.release]
    fin = buzz_finish.run_finish(s.env, s, runner, out / "finish", lambda **kv: log(event="finish", **kv), release=rel,
                                 start_held=False, grasp_budget=int(os.environ.get("BUZZ_GRASP_BUDGET", "8")))
    log(event="finish_done", failure=fin)
    return fin


s = Session("buzz_wire_3d", instance=a.instance, strict=True)
task = s.env._task
design = PortedDesign(task)
io = LimitedIO(s)
t0 = time.time()
s.record_start()
orc = PortedOracle(io, design, [], pick_grasp=a.pick.split(",")[0], log=log, rng=a.rng, psi_n=a.psi_n, ds=a.ds,
                   dq_max=a.dq_max, v_ring=a.v_ring, w_max=a.w_max)
orc.drop_offset_deg = a.drop_offset
orc.ki, orc.ki_max, orc.ioff = a.ki, a.ki_max, np.zeros(7)
orc.shift_dx = a.shift_dx
orc.soft = a.soft
orc.soft_open = a.soft_open
orc.drop_open_s, orc.drop_retreat = a.drop_open_s, a.drop_retreat
failure = None
try:
    if a.resume:
        rr = np.load(a.resume)[:a.resume_rows]
        for row in rr:
            if s.step(row):
                raise RuntimeError("episode ended in the resumed rows")
        orc._last_q = np.asarray(rr[-1][:7], float); orc.grip = float(rr[-1][7])
        log(event="resumed", rows=len(rr), metrics={k: io.metrics().get(k) for k in ("traversal_complete", "distance_to_finish_m", "finish_angle_deg")})
        if a.resume_drop and a.landing_sweep:
            orc.measure_grasp()
            ups_s, turns_s = a.landing_sweep.split(":")
            band = [float(x) for x in a.band.split(",")]
            good = orc.landing_sweep([1e-3 * float(x) for x in ups_s.split(",")], [float(x) for x in turns_s.split(",")], band)
            if not good:
                raise SystemExit("no landing in the band")
            raise SystemExit("landing sweep only")
        if a.resume_drop:
            orc.measure_grasp()
            if a.drop_turn:
                pw_, Rw_ = io.wand_pose()
                i_ = orc.path.idx(float(io.metrics().get("route_cursor_m") or orc.path.length))
                Rt_ = R.from_rotvec(orc.path.t[i_] * np.deg2rad(a.drop_turn)) * Rw_
                try:
                    orc.move_wand(pw_, Rt_, speed=0.005)
                except RuntimeError as exc:
                    log(event="drop_turn_failed", why=str(exc))
            if a.drop_soft:
                orc.release_soft()
                orc.settle(timeout=4.0)
            else:
                orc.drop()
            pw_, Rw_ = io.wand_pose()
            log(event="landed", metrics={k: io.metrics().get(k) for k in ("distance_to_finish_m", "finish_angle_deg")},
                yaw=round(float(np.degrees(orc._yaw(Rw_.apply([1, 0, 0])))), 1))
        if a.retreat:
            orc.retreat_clear()
        elif a.push_search:
            orc.push_search(a.push_search)
        elif a.push_first:
            orc.push_to_heading(a.push_first)
        if a.retreat or a.no_place:
            pass
        elif a.slide is not None and not a.adaptive and not a.place2:
            orc.push_slide(extra=1e-3 * a.slide); orc.hold()
        elif a.adaptive:
            finish_adaptive()
            if a.slide is not None and not s.success() and s.episode_end is None:
                orc.push_slide(extra=1e-3 * a.slide); orc.hold()
        else:
            (orc.final_place2 if a.place2 else orc.final_place)()
            orc.hold()
    if a.resume:
        pass
    elif a.plan:
        P = np.load(Path(a.plan).parent / "prep.npz"); PL = np.load(a.plan)
        for row in P["prefix"]:
            if io.step(row[:7], row[7]):
                raise RuntimeError("episode ended in the prefix rows")
        orc._last_q = np.asarray(P["prefix"][-1][:7], float); orc.grip = float(P["prefix"][-1][7])
        orc.measure_grasp()
        log(event="prefix_replayed", rows=len(P["prefix"]), held=io.holding(), g=np.round(orc.g[0], 4).tolist(),
            g_prep=np.round(P["g_p"], 4).tolist())
        q_prev = orc._last_q
        for n_, q1 in enumerate(PL["qs"]):
            cp = a.tight_cap if PL["margin"][n_] < 0.004 else a.cap
            n = max(1, int(np.ceil(np.abs(q1 - q_prev).max() / cp)))
            for jj in range(1, n + 1):
                if orc.send(q_prev + (q1 - q_prev) * jj / n):
                    raise RuntimeError(f"the episode ended in the planned route (node {n_}, s {PL['s'][n_]:.3f})")
            q_prev = q1; orc._last_q = q1
            if n_ % 100 == 0:
                orc.note(event="plan_exec", node=n_, s=round(float(PL["s"][n_]), 4))
        for _ in range(int(2.5 / O.CTRL_DT)):                      # hold the last node until the arm is there
            if np.abs(io.q() - q_prev).max() < 0.002:
                break
            if orc.send(q_prev):
                raise RuntimeError("the episode ended holding the plan's last node")
        orc.note(event="plan_done")
        # the drives sag a few mm on the vertical post: step the held wand down along the wire, 1 mm at a time, until
        # the task records the traversal complete (its cursor within 1 cm of the wire's end)
        for k_ in range(12):
            if io.metrics().get("traversal_complete"):
                break
            q_now = getattr(orc, "_last_q", io.q())
            tp_, tr_ = orc.kin.fk(q_now)
            q_dn, ok_, _, _ = orc.kin.ik(tp_ + np.array([0, 0, -0.001]), tr_, q_now, iters=100)
            if not ok_:
                break
            for _ in range(4):
                if orc.send(q_dn):
                    raise RuntimeError("the episode ended stepping down the post")
            orc._last_q = q_dn
            orc.note(event="post_step", k=k_)
        s0 = float(io.metrics().get("route_cursor_m") or PL["s"][-1])
    elif a.route and a.cut:
        rows = np.load(a.route)["actions"].astype(np.float32)[:a.cut].copy()
        if a.shift_dx:
            rel = 600 + int(np.flatnonzero(rows[600:, 7] < 0.4)[0])          # the route's deck release
            v = None
            for i in range(len(rows)):
                if i == rel + 40:
                    pw_, Rw_ = io.wand_pose()
                    v = Rw_.apply([a.shift_dx, 0, 0])
                    q_seed = io.q()
                if v is not None:
                    f = min(1.0, (i - rel - 40) / 60)
                    tp_, tr_ = orc.kin.fk(rows[i, :7].astype(float))
                    q_, ok_, _, _ = orc.kin.ik(tp_ + f * v, tr_, q_seed, iters=200)
                    if not ok_:
                        raise RuntimeError(f"shifted grasp approach: IK failed at route row {i}")
                    rows[i, :7] = q_
                    q_seed = q_
                if io.step(rows[i, :7], rows[i, 7]):
                    raise RuntimeError("episode ended in the route rows")
        else:
          for row in rows:
            if io.step(row[:7], row[7]):
                raise RuntimeError("episode ended in the route rows")
        orc._last_q = np.asarray(rows[-1][:7], float)
        orc.grip = float(rows[-1][7])
        orc.measure_grasp()
        log(event="route_replayed", rows=len(rows), held=io.holding(), metrics={k: io.metrics().get(k) for k in
            ("route_cursor_m", "traversal_tracked", "traversal_complete")}, g=np.round(orc.g[0], 4).tolist())
        s0 = float(io.metrics().get("route_cursor_m") or 0.004)
        pw, Rw = io.wand_pose()
        if s0 < 0.01 and pw[2] < orc.path.p[0][2] and not (a.retime or a.nudge):       # held on the deck around the start post: level it, lift onto the wire
            p0 = orc.path.p[0]
            h = Rw.apply([1, 0, 0]); h[2] = 0; h /= np.linalg.norm(h)
            Rt = R.from_matrix(np.column_stack([h, np.cross([0, 0, 1.], h), [0, 0, 1.]]))
            orc.move_wand(np.array([p0[0], p0[1], pw[2] + 0.002]), Rt, speed=0.01)
            orc.move_wand(p0 + [0, 0, 0.004], Rt, speed=0.01)
            orc.note(event="lifted_from_route")
            s0 = 0.004
    else:
        spec = orc.try_grasps(a.pick.split(","), stage=O.DECK_STAGE)
        pw, Rw = io.wand_pose()
        p0 = orc.path.p[0]
        h = Rw.apply([1, 0, 0]); h[2] = 0; h /= np.linalg.norm(h)
        Rt = R.from_matrix(np.column_stack([h, np.cross([0, 0, 1.], h), [0, 0, 1.]]))
        orc.move_wand(np.array([p0[0], p0[1], pw[2] + 0.002]), Rt, speed=0.01)
        orc.move_wand(p0 + [0, 0, 0.004], Rt, speed=0.01)
        orc.note(event="lifted", spec=spec)
        s0 = 0.004
    if a.nudge:
        allrows = np.load(a.route)["actions"].astype(np.float32)
        s0 = orc.nudged_route(allrows[a.cut:], a.guide_stop, dmax=a.nudge_max, dstep=a.nudge_step,
                              rclear=1e-3 * a.robot_clear_mm, node_mm=a.node_mm, cap=a.cap, tight_cap=a.tight_cap)
    if a.retime:
        allrows = np.load(a.route)["actions"].astype(np.float32)
        s0 = orc.retimed_route(allrows[a.cut:], a.guide_stop, a.cap, a.tight_cap, a.tight_mm)
    if a.guide:
        allrows = np.load(a.route)["actions"].astype(np.float32)
        s0 = orc.guided_traverse(allrows[a.cut:], a.guide_stop, a.guide_lag_max, a.guide_every)
    if a.resume:
        pass
    elif io.metrics().get("traversal_complete") and s0 >= orc.path.length - 0.02:
        orc.note(event="traverse_skipped", why="traversal complete at the post's foot")
    else:
      orc.traverse_to_drop(max(s0, 0.004), [float(x) for x in a.drop_back.split(",")], a.lags.split(","))
    if not a.resume:
        orc.drop()
        if a.push_first:
            orc.push_to_heading(a.push_first)
        if a.adaptive:
            finish_adaptive()
        elif not a.no_place:
            (orc.final_place2 if a.place2 else orc.final_place)()
        if not a.adaptive:
            orc.hold()
except Exception as exc:
    failure = f"{type(exc).__name__}: {exc}"
    traceback.print_exc()
if failure is None and not s.success() and s.episode_end is None and not a.retreat:
    try:                                   # the final settling needs the robot clear of the resting wand
        orc.retreat_clear()
    except Exception as exc:
        log(event="retreat_failed", why=str(exc)[:200])
held = 0
while s.episode_end is None and held < a.tail:
    s.step(s.hold_action()); held += 1
acts = s.record_stop()
st = s.status()
lim = io.limiter.report()
arm = np.abs(np.diff(np.vstack([np.r_[s.initial_state()["joint_targets"] if False else acts[0]][None], acts]), axis=0))[:, :7].max()
res = dict(instance=a.instance, args=vars(a), rows=int(len(acts)), success=st["success"], episode_end=st["episode_end"],
           state_sha=st["state_sha"], wall_s=round(time.time() - t0, 1), failure=failure,
           metrics=task.result.metrics, limiter=lim, max_row_dq=float(arm))
np.save(out / "actions_all.npy", acts)
if st["success"]:
    s.save_trajectory(acts, out / "privileged.npz", meta=dict(
        source="ported buzz oracle (rd machinery, feedback, live; rate-limited to the standard joint step)",
        args=vars(a), wall_s=res["wall_s"]))
log(event="end", **res)
(out / "result.json").write_text(json.dumps(res, default=str, indent=1))
s.close()
