"""Reference (oracle) of a redesigned claws tier, generated live in a MuJoCo Session from measured states.

The oracle is a program of phases; each phase is planned from the state the Session actually reached (measured piece
poses, measured joints, the welds' actual hand-piece offsets), turned into joint rows (robot_plan.rows_from_waypoints),
and executed. The recorded rows are the reference: an open-loop replay from the same birth reproduces them.

  python oracle.py <tier> <out_dir> [--speed 0.25]

Phases of the "held" program (very_easy): acquire (hold rows: the welds form on the closed hands), disengage (the
right hand moves B along the designed relative path, the left hand holds A still), transport (B into the goal area),
settle (hold until the task's success).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from topogym.planning.interlocked_claws.redesign import paths                       # noqa: E402
from topogym.tasks.interlocked_claws.redesign.task import make_session            # noqa: E402
from topogym.planning.interlocked_claws.redesign.robot_plan import (              # noqa: E402
    ARMS, Kin, rows_from_waypoints, CLOSED, OPEN, solve_path)


COM = {}          # piece -> its composite COM in the design frame (the body frame sits there); set by Runner


def body_T(state, name):
    """A piece's pose in its DESIGN frame (the simulated body frame is at the piece's composite COM)."""
    b = state["bodies"][name]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(b["quaternion_wxyz"], float)[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = b["position_m"]
    if name in COM:
        T[:3, 3] = T[:3, 3] - T[:3, :3] @ COM[name]
    return T


def se3_log(T):
    """(rotation vector, v) with T = exp: rotation about the screw axis and the translation along it (twist)."""
    R, p = T[:3, :3], T[:3, 3]
    w = Rotation.from_matrix(R).as_rotvec()
    th = float(np.linalg.norm(w))
    if th < 1e-9:
        return w, p.copy()
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    V = np.eye(3) + (1 - np.cos(th)) / th * K + (th - np.sin(th)) / th * K @ K
    return w, np.linalg.solve(V, p)


def se3_exp(w, v):
    th = float(np.linalg.norm(w))
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(w).as_matrix()
    if th < 1e-9:
        T[:3, 3] = v
        return T
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    V = np.eye(3) + (1 - np.cos(th)) / th * K + (th - np.sin(th)) / th * K @ K
    T[:3, 3] = V @ v
    return T


def split_poses(T_A0, P0, poses, share_A=.5, fixed=None):
    """World poses of A and B along a relative path, the motion shared between the hands: with D = P P0^-1 (the
    relative motion so far, in A's start frame), A moves by D^(-share_A) and B by D^(1 - share_A) (powers of the same
    screw), so B stays at the relative pose P in A's frame."""
    TA, TB = [], []
    P0i = np.linalg.inv(P0)
    for P in poses:
        D = P @ P0i
        w, v = se3_log(D)
        N = se3_exp(-share_A * w, -share_A * v)
        A = T_A0 @ N
        if fixed is not None and share_A:      # the same motion of A, but turning about the world point `fixed`
            G = A @ np.linalg.inv(T_A0)
            A = A.copy()
            A[:3, 3] += np.asarray(fixed, float) - (G[:3, :3] @ np.asarray(fixed, float) + G[:3, 3])
        TA.append(A)
        TB.append(A @ P)
    return TA, TB


class Runner:
    def __init__(self, tier, out, speed=.25, share=.5, log=print, plan_only=False, feedback=True):
        self.tier, self.out, self.speed, self.share, self.log = tier, Path(out), speed, share, log
        self.feedback = feedback
        self.plan_only = plan_only
        self.out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        self.s = make_session(tier)
        self.build_s = time.time() - t0
        self.birth_sha = self.s.state_sha()
        self.kin = Kin(self.s.env._world.model)
        self.layout = list(self.s.action_layout)
        self.rows = []
        self.phases = []
        self.task = self.s.env._task
        from topogym.planning.interlocked_claws.redesign import tiers as _tiers   # the authoring design (moves)
        self.design = dict(_tiers.make(self.task.tier), **{k: v for k, v in self.task.design_record.items() if k == "settle"})
        COM.clear()
        COM.update({k: np.asarray(v, float) for k, v in getattr(self.task, "piece_com", {}).items()})
        self.kin.com = dict(COM)

    # ------------------------------------------------------------- measured state
    def state(self):
        return self.s.env._world.read_state()

    def q(self):
        qm = np.asarray(self.s.robot.positions(), float)
        cols = {n: i for i, n in enumerate(self.layout)}
        return {arm: np.array([qm[cols[f"{arm}/{j}"]] for j in
                               ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")])
                for arm in ARMS}

    def targets(self):
        """The current commanded targets as a waypoint (what the robot is holding now)."""
        t = np.asarray(self.s._targets, float)
        cols = {n: i for i, n in enumerate(self.layout)}
        w = {arm: np.array([t[cols[f"{arm}/{j}"]] for j in
                            ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")])
             for arm in ARMS}
        w["lg"], w["rg"] = t[cols["left_native_gripper"]], t[cols["right_native_gripper"]]
        return w

    # ------------------------------------------------------------- execution
    def execute(self, name, rows, stop_on_success=True):
        t0 = time.time()
        n0 = len(self.rows)
        success_row = None
        for r in rows:
            self.s.step(np.asarray(r, np.float32))
            self.rows.append(np.asarray(self.s._targets, np.float32).copy())
            if self.s.success() and success_row is None:
                success_row = len(self.rows)
                if stop_on_success:
                    break
            if self.s.status()["episode_end"]:
                break
        st = self.s.status()
        ph = dict(name=name, rows=len(self.rows) - n0, wall_s=round(time.time() - t0, 1), success=bool(st["success"]),
                  episode_end=st["episode_end"], held={a: self.s.object_in_hand(a) for a in ARMS})
        self.phases.append(ph)
        self.log(json.dumps(ph))
        return success_row

    def hold(self, name, n):
        w = self.targets()
        return self.execute(name, rows_from_waypoints([w, w], self.layout, min_steps=n))

    # ------------------------------------------------------------- planning helpers
    def follow(self, arm, T_tcp_list, seed, other=None, check=None, prior_w=.03):
        """IK along a list of TCP targets (continuity-seeded); -> (joint list, worst errors, collisions found)."""
        qs, worst, hits = [], [0., 0.], []
        q = np.asarray(seed, float)
        for k, T in enumerate(T_tcp_list):
            q, ep, er = self.kin.ik(arm, T, q, prior=q, w_prior=prior_w)
            worst = [max(worst[0], ep), max(worst[1], er)]
            qs.append(q.copy())
            if check is not None and k % 3 == 0:
                c = check(k, q)
                if c:
                    hits.append((k, c[:3]))
        return qs, worst, hits

    def save(self, extra=None):
        rows = np.asarray(self.rows, np.float32)
        st = self.s.status()
        meta = dict(source="redesign claws oracle (agent claws), generated live from measured states", tier=self.tier,
                    phases=self.phases, success=bool(st["success"]), episode_end=st["episode_end"],
                    final_state_sha=st["state_sha"], birth_sha=self.birth_sha, rows=len(rows), **(extra or {}))
        np.savez_compressed(self.out / "privileged.npz", actions=rows, action_layout=np.array(self.layout),
                            task="interlocked_claws", variation_id=self.s.config.get("variation_id"),
                            scene_seed=0, gripper_space="native", instance_sha=self.birth_sha,
                            meta=json.dumps(meta), trajectory_version=2)
        (self.out / "oracle.json").write_text(json.dumps(meta, indent=1, default=str))
        return meta


def program_held(run: Runner):
    """very_easy: the left hand holds A still; the right hand takes B out along the designed moves; B to the goal."""
    s, kin, d = run.s, run.kin, run.design
    run.hold("acquire", 30)
    st = run.state()
    T_A, T_B = body_T(st, "piece_1"), body_T(st, "piece_0")
    q = run.q()
    w0 = run.targets()
    T_tcpR = kin.tcp_frame("right", q["right"])
    grip_B = np.linalg.inv(T_B) @ T_tcpR                  # the weld's actual hand offset, measured
    T_tcpL = kin.tcp_frame("left", q["left"])
    grip_A = np.linalg.inv(T_A) @ T_tcpL
    rel0 = np.linalg.inv(T_A) @ T_B
    poses, starts, which = paths.dense(rel0, d["moves"], step_m=.002, step_deg=1.)
    TA, TB = split_poses(T_A, rel0, poses, share_A=run.share)
    held = (("left", "piece_1"), ("right", "piece_0"))
    qL, qR = [], []
    a, b = w0["left"].copy(), w0["right"].copy()
    worst, hits = [0., 0., 0., 0.], []
    for k in range(len(poses)):
        a, epa, era = kin.ik("left", TA[k] @ grip_A, a, prior=a)
        b, epb, erb = kin.ik("right", TB[k] @ grip_B, b, prior=b)
        worst = [max(worst[0], epa), max(worst[1], era), max(worst[2], epb), max(worst[3], erb)]
        qL.append(a.copy()); qR.append(b.copy())
        if k % 3 == 0:
            c = kin.contacts({"left": a, "right": b}, {"left": .0222, "right": .0222},
                             {"piece_0": TB[k], "piece_1": TA[k]}, held=held, margin=.001)
            if c:
                hits.append((k, c[:3]))
    run.log(json.dumps(dict(plan="disengage", poses=len(poses), share_A=run.share, ik_worst_L_R=worst,
                            collisions=hits[:5], n_collisions=len(hits),
                            wrist_range_L=[float(min(x[4] for x in qL)), float(max(x[4] for x in qL))],
                            wrist_range_R=[float(min(x[4] for x in qR)), float(max(x[4] for x in qR))])))
    if run.plan_only:
        return dict(success=False, episode_end="plan_only", rows=0, final_state_sha=None, birth_sha=run.birth_sha)
    wps = [w0] + [dict(left=qL[k], right=qR[k], lg=w0["lg"], rg=w0["rg"]) for k in range(0, len(poses), 2)]
    wps.append(dict(left=qL[-1], right=qR[-1], lg=w0["lg"], rg=w0["rg"]))
    run.execute("disengage", rows_from_waypoints(wps, run.layout, speed=run.speed))
    # transport: B's centre into the goal area (lift, carry, lower), orientation kept
    st = run.state()
    T_B = body_T(st, "piece_0")
    q = run.q()
    w1 = run.targets()
    T_tcpR = kin.tcp_frame("right", q["right"])
    grip_B = np.linalg.inv(T_B) @ T_tcpR
    region = run.task.goal_region
    corners = d["B"].corners()
    world = corners @ T_B[:3, :3].T + T_B[:3, 3]
    centre = world.mean(0)
    goal = np.array([region.center_xy_m[0], region.center_xy_m[1], region.table_height_m + .03
                     + (centre[2] - world[:, 2].min())])
    lift = centre + np.array([0., 0., .06])
    over = np.array([goal[0], goal[1], lift[2]])
    path = [centre + u * (lift - centre) for u in np.linspace(0, 1, 15)[1:]]
    path += [lift + u * (over - lift) for u in np.linspace(0, 1, 60)[1:]]
    path += [over + u * (goal - over) for u in np.linspace(0, 1, 30)[1:]]
    T_list = []
    for c in path:
        TB = T_B.copy(); TB[:3, 3] += c - centre
        T_list.append(TB @ grip_B)
    qs, worst, hits = run.follow("right", T_list, w1["right"])
    run.log(json.dumps(dict(plan="transport", ik_worst_m_rad=worst, goal=goal.tolist())))
    wps = [w1] + [dict(left=w1["left"], right=qq, lg=w1["lg"], rg=w1["rg"]) for qq in qs[::2]] + \
          [dict(left=w1["left"], right=qs[-1], lg=w1["lg"], rg=w1["rg"])]
    sr = run.execute("transport", rows_from_waypoints(wps, run.layout, speed=run.speed))
    if sr is None:
        run.hold("settle", 120)
    return run.save(dict(program="held"))


GRIP_OF = {"piece_0": None, "piece_1": None}


def piece_of(grip):
    return "piece_0" if grip.startswith("b_") else "piece_1"


def program_generic(run: Runner):
    """Any tier: both hands start on their grips; for each designed move the held pieces move along the relative path
    (the left hand's share of the motion per move: move["share"], else --share); a move with "hold" first changes
    grips (release, retreat over the lock, approach the new grip, close until the weld forms); at the end the hand
    holding B carries it into the goal area; then hold until success."""
    s, kin, d = run.s, run.kin, run.design
    if getattr(run.task, "table_start", False):    # hands empty at home: take the lock first, then the key
        run.hold("start", 10)
        holds = {}
        for arm in ("left", "right"):
            regrasp(run, arm, d["hands"][arm], holds, release=False)
            holds[arm] = d["hands"][arm]
        from topogym.planning.interlocked_claws.redesign.placement2 import LIFT_M
        lift(run, holds, LIFT_M)
    else:
        run.hold("acquire", 30)
        holds = dict(d["hands"])                   # arm -> grip name
    run.nominal = {}
    for i, mv in enumerate(d["moves"]):
        if i == 0 or mv.get("hold"):
            pass                                   # (planned after the hand change below)
        for arm, g in (mv.get("hold") or {}).items():
            if g is None:                          # the hand lets go and moves out of the way
                if arm in holds:
                    release(run, arm, holds)
                    holds.pop(arm)
            elif holds.get(arm) != g:
                regrasp(run, arm, g, holds)
                holds[arm] = g
        if i == 0 or mv.get("hold"):
            run.nominal = segment_nominal(run, i, holds)
        relmove(run, i, mv, holds)
    transport(run, holds)
    park_lock(run, holds)
    if not run.s.success():
        run.hold("settle", 160)
    return run.save(dict(program="generic", share=run.share))


def park_lock(run, holds, margin=.03):
    """The hand that holds the lock (A) carries it back to the system side of the goal separator if any part of it
    crossed (a pull-out shared between the hands moves the lock toward the goal side), 3 cm clear."""
    arms = [a for a, g in holds.items() if piece_of(g) == "piece_1"]
    if not arms:
        return
    arm = arms[0]
    kin, d = run.kin, run.design
    T, q, w0 = measured(run)
    region = run.task.goal_region
    n = np.asarray(region.separator_normal_xy, float)
    pts = d["A"].corners() @ T["piece_1"][:3, :3].T + T["piece_1"][:3, 3]
    over = float((pts[:, :2] @ n).max() - region.separator_offset_m + margin)
    if over <= 0:
        return
    T0 = kin.tcp_frame(arm, q[arm])
    T1 = T0.copy(); T1[:2, 3] -= over * n
    qs, rep = arm_path(run, arm, line(T0, T1, .004), w0[arm])
    if not rep.get("ok", False):              # no straight IK path: an exact end pose (the lock may also be raised
        best = None                           # and turned about z a little) and a joint-space move to it
        for dz in (0., .03, .06):
            for yaw in (0., 15., -15., 30., -30.):
                Rz = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
                TA1 = T["piece_1"].copy(); TA1[:3, :3] = Rz @ TA1[:3, :3]
                TA1[:3, 3] = T["piece_1"][:3, 3] + np.r_[-over * n, dz]
                ptsA = d["A"].corners() @ TA1[:3, :3].T + TA1[:3, 3]
                if float((ptsA[:, :2] @ n).max()) > region.separator_offset_m - margin / 2:
                    continue
                Tt = TA1 @ np.linalg.inv(T["piece_1"]) @ T0
                qe, ep, er = kin.ik_multi(arm, Tt, [w0[arm], q[arm]])
                if ep < 1.5e-3 and er < .02:
                    best = qe
                    break
            if best is not None:
                break
        if best is not None:
            k = max(2, int(np.ceil(np.abs(best - w0[arm]).max() / .02)))
            qs = [w0[arm] + u * (best - w0[arm]) for u in np.linspace(0, 1, k + 1)[1:]]
            rep = dict(rep, fallback="joint move to an exact end pose", ok=True)
    run.log(json.dumps(dict(plan=f"park_lock_{arm}", shift_m=round(over, 3), ik=rep), default=str))
    if run.plan_only:
        return
    wps = [w0] + [dict(w0, **{arm: qq}) for qq in qs[::2]] + [dict(w0, **{arm: qs[-1]})]
    run.execute("park_lock", rows_from_waypoints(wps, run.layout, speed=run.speed))
    if not run.s.success():                   # the arm sags under the lock: servo its TCP onto the planned end
        servo_to(run, arm, kin.tcp_frame(arm, qs[-1]), "lg" if arm == "left" else "rg")


def measured(run):
    st = run.state()
    return {p: body_T(st, p) for p in ("piece_0", "piece_1")}, run.q(), run.targets()


def relmove(run, i, mv, holds):
    kin, d = run.kin, run.design
    T, q, w0 = measured(run)
    T_A, T_B = T["piece_1"], T["piece_0"]
    offs = {arm: np.linalg.inv(T[piece_of(g)]) @ kin.tcp_frame(arm, q[arm]) for arm, g in holds.items()}
    rel0 = np.linalg.inv(T_A) @ T_B
    if run.feedback:                      # follow the designed relative path (errors do not accumulate), blended in
        from topogym.planning.interlocked_claws.redesign.geom import interp
        P = np.asarray(d["engaged"], float)
        for m_ in d["moves"][:i]:
            P = paths.apply(P, m_)
        poses, _, _ = paths.dense(P, [mv], step_m=.002, step_deg=1.)
        nb = min(8, len(poses) - 1)
        poses = [interp(rel0, Pk, min(1., k / nb)) if k < nb else Pk for k, Pk in enumerate(poses)]
        poses[0] = rel0
    else:
        poses, _, _ = paths.dense(rel0, [mv], step_m=.002, step_deg=1.)
    share = float(mv.get("share", run.share))
    # the left hand moves A only if it holds A; a hand on B moves with B
    a_held = any(piece_of(g) == "piece_1" for g in holds.values()) and not d.get("a_on_table")
    fixed = None
    fx = mv.get("fixed")                      # A turns about this world point: a hand's TCP, or their midpoint
    if fx:
        tcp = {a: kin.tcp_frame(a, q[a])[:3, 3] for a in ARMS}
        fixed = (tcp["left"] + tcp["right"]) / 2 if fx == "mid" else tcp[fx]
    TA, TB = split_poses(T_A, rel0, poses, share_A=share if a_held else 0., fixed=fixed)
    held = tuple((arm, piece_of(g)) for arm, g in holds.items())
    qs, worst, hits, reports = {}, {}, [], {}
    for arm in ARMS:
        if arm not in holds:
            qs[arm] = [w0[arm].copy()] * len(poses)
            continue
        Ts = [(TA[k] if piece_of(holds[arm]) == "piece_1" else TB[k]) @ offs[arm] for k in range(len(poses))]
        nom = (getattr(run, "nominal", None) or {}).get(i, {}).get(arm)
        if nom is not None and len(nom) == len(Ts):      # the segment's global plan: stay on its IK branch
            sol, ew = [], [0., 0.]
            for k, T_ in enumerate(Ts):
                qq, ep, er = kin.ik(arm, T_, nom[k], prior=nom[k], w_prior=.01)
                ew = [max(ew[0], ep), max(ew[1], er)]
                sol.append(qq)
            rep = dict(source="segment plan", ok=bool(ew[0] < 3e-3 and ew[1] < .04), worst=ew)
        else:
            sol, rep = solve_path(kin, arm, Ts, w0[arm])
        reports[arm] = rep
        if sol is None:                       # no branch-continuous solution: track (logged; the Session decides)
            sol, cur = [], w0[arm].copy()
            for T in Ts:
                cur, ep, er = kin.ik(arm, T, cur, prior=cur)
                sol.append(cur.copy())
        qs[arm] = sol
        worst[arm] = rep.get("worst")
    for k in range(len(poses)):
        if k % 3 == 0:
            c = kin.contacts({a: qs[a][k] for a in ARMS}, {a: (.0222 if a in holds else .037) for a in ARMS},
                             {"piece_0": TB[k], "piece_1": TA[k]}, held=held, margin=.001)
            if c:
                hits.append((k, c[:3]))
    run.log(json.dumps(dict(plan=f"move{i}:{mv.get('label')}", poses=len(poses), share_A=share, ik=reports,
                            collisions=hits[:4], n_collisions=len(hits),
                            wrist={a: [round(float(min(x[4] for x in qs[a])), 2), round(float(max(x[4] for x in qs[a])), 2)]
                                   for a in ARMS})))
    if run.plan_only:
        return
    if run.feedback:
        track(run, f"move{i}", poses, TA, TB, qs, holds, offs, T_A, T_B)
        return
    wps = [w0] + [dict(left=qs["left"][k], right=qs["right"][k], lg=w0["lg"], rg=w0["rg"])
                  for k in list(range(0, len(poses), 2)) + [len(poses) - 1]]
    run.execute(f"move{i}", rows_from_waypoints(wps, run.layout, speed=run.speed))


BIAS_MAX = .06


def track(run, name, poses, TA, TB, qs, holds, offs, T_A0, T_B0, chunk=4):
    """Closed-loop execution of a planned relative move: in chunks of `chunk` dense poses, (1) measure both pieces and
    the joints, (2) gravity / load bias per arm = commanded - measured joints (clamped), (3) the hand on A aims at A's
    planned pose, the hand on B at B's planned pose moved by A's current deviation from its plan (so B follows the
    relative path in A's actual frame), IK seeded on the nominal branch, (4) command target + bias. The rows are the
    reference (an open-loop replay reproduces them)."""
    kin = run.kin
    n0 = len(run.rows)
    t0 = time.time()
    k = 0
    worst_dev = 0.
    while k < len(poses) - 1:
        k1 = min(len(poses) - 1, k + chunk)
        T, q, w0 = measured(run)
        dev = T["piece_1"] @ np.linalg.inv(TA[k])             # A's actual pose vs its plan at this point (world)
        cmd = {a: w0[a] for a in ARMS}
        bias = {a: np.clip(cmd[a] - q[a], -BIAS_MAX, BIAS_MAX) for a in ARMS}
        rel_err = np.linalg.norm((np.linalg.inv(T["piece_1"]) @ T["piece_0"])[:3, 3]
                                 - (np.linalg.inv(TA[k]) @ TB[k])[:3, 3])
        worst_dev = max(worst_dev, float(rel_err))
        wps = [w0]
        for j in range(k + 1, k1 + 1):
            w = dict(lg=w0["lg"], rg=w0["rg"])
            for a in ARMS:
                if a not in holds:
                    w[a] = w0[a]
                    continue
                if piece_of(holds[a]) == "piece_1":
                    Tt = TA[j] @ offs[a]
                else:
                    Tt = dev @ TB[j] @ offs[a]
                qq, ep, er = kin.ik(a, Tt, qs[a][j], prior=qs[a][j], w_prior=.01)
                w[a] = qq + bias[a]
            wps.append(w)
        rows = rows_from_waypoints(wps, run.layout, speed=run.speed)
        for r in rows[1:]:
            run.s.step(np.asarray(r, np.float32))
            run.rows.append(np.asarray(run.s._targets, np.float32).copy())
            if run.s.status()["episode_end"]:
                break
        if run.s.status()["episode_end"]:
            break
        k = k1
    st = run.s.status()
    T, q, w0 = measured(run)
    rel_err = np.linalg.norm((np.linalg.inv(T["piece_1"]) @ T["piece_0"])[:3, 3] - (np.linalg.inv(TA[-1]) @ TB[-1])[:3, 3])
    ph = dict(name=name, rows=len(run.rows) - n0, wall_s=round(time.time() - t0, 1), success=bool(st["success"]),
              episode_end=st["episode_end"], held={a: run.s.object_in_hand(a) for a in ARMS},
              worst_rel_dev_mm=round(1000 * max(worst_dev, float(rel_err)), 1), end_rel_dev_mm=round(1000 * float(rel_err), 1))
    run.phases.append(ph)
    run.log(json.dumps(ph))


def segment_nominal(run, i0, holds):
    """A global joint plan of the moves from i0 to the next hand change, from the measured state: the designed relative
    poses (as relmove densifies them), A's share per move, every arm's targets in one list, one IK-branch DP per arm
    (joint steps <= 0.35 rad between anchors). -> {move: {arm: joints per pose}} ({} when the DP finds none)."""
    kin, d = run.kin, run.design
    T, q, w0 = measured(run)
    moves = d["moves"]
    j = i0 + 1
    while j < len(moves) and not moves[j].get("hold"):
        j += 1
    if any(moves[i].get("fixed") for i in range(i0, j)):
        return {}                                  # turning points move with the hands: plan move by move
    offs = {arm: np.linalg.inv(T[piece_of(g)]) @ kin.tcp_frame(arm, q[arm]) for arm, g in holds.items()}
    a_held = any(piece_of(g) == "piece_1" for g in holds.values()) and not d.get("a_on_table")
    P = np.asarray(d["engaged"], float)
    for m_ in moves[:i0]:
        P = paths.apply(P, m_)
    TAc = T["piece_1"]
    per = []
    for i in range(i0, j):
        poses, _, _ = paths.dense(P, [moves[i]], step_m=.002, step_deg=1.)
        sh = float(moves[i].get("share", run.share)) if a_held else 0.
        TA, TB = split_poses(TAc, P, poses, share_A=sh)
        per.append((i, TA, TB))
        P, TAc = poses[-1], TA[-1]
    out = {i: {} for i, _, _ in per}
    for arm, g in holds.items():
        Ts, lens = [], []
        for i, TA, TB in per:
            Ts += [(A_ if piece_of(g) == "piece_1" else B_) @ offs[arm] for A_, B_ in zip(TA, TB)]
            lens.append(len(TA))
        sol, rep = solve_path(kin, arm, Ts, w0[arm], anchor_every=5, n_random=10, max_step=.35)
        run.log(json.dumps(dict(plan=f"segment{i0}-{j - 1}:{arm}", **{k: v for k, v in rep.items() if k != "solutions"})))
        if sol is None:
            return {}
        a = 0
        for (i, TA, TB), L in zip(per, lens):
            out[i][arm] = sol[a:a + L]
            a += L
    return out


def arm_path(run, arm, T_list, seed):
    sol, rep = solve_path(run.kin, arm, T_list, seed)
    if sol is not None:
        return sol, rep
    out, q, worst = [], np.asarray(seed, float), [0., 0.]
    for T in T_list:
        q, ep, er = run.kin.ik(arm, T, q, prior=q)
        worst = [max(worst[0], ep), max(worst[1], er)]
        out.append(q.copy())
    rep["tracked_worst"] = worst
    return out, rep


def line(T0, T1, step=.004):
    n = max(2, int(np.ceil(np.linalg.norm(T1[:3, 3] - T0[:3, 3]) / step)))
    from topogym.planning.interlocked_claws.redesign.geom import interp
    return [interp(T0, T1, u) for u in np.linspace(0, 1, n + 1)[1:]]


def lift(run, holds, dz):
    """Both hands raise the assembly by dz (a common translation, the relative pose kept)."""
    kin = run.kin
    T, q, w0 = measured(run)
    sols = {}
    for arm in holds:
        T0 = kin.tcp_frame(arm, q[arm])
        T1 = T0.copy(); T1[2, 3] += dz
        sols[arm], _ = arm_path(run, arm, line(T0, T1, .003), w0[arm])
    n = max(len(v) for v in sols.values())
    wps = [w0]
    for k in range(n):
        w = dict(w0)
        for arm, qs in sols.items():
            w[arm] = qs[min(k, len(qs) - 1)]
        wps.append(w)
    if not run.plan_only:
        run.execute("lift", rows_from_waypoints(wps, run.layout, speed=run.speed * .6))


def servo_to(run, arm, T_target, gkey, rounds=3, settle=16):
    """Close the last millimetres before a grasp: the open hand's measured TCP vs the target; the joint target becomes
    IK(target) + (commanded - measured) (the load bias), a few rounds. The pads have ~4.5 mm per side on a grip block."""
    kin = run.kin
    errs = []
    for _ in range(rounds):
        T, q, w0 = measured(run)
        tcp = kin.tcp_frame(arm, q[arm])
        e = float(np.linalg.norm(tcp[:3, 3] - T_target[:3, 3]))
        errs.append(round(e * 1000, 2))
        if e < 1e-3:
            break
        bias = np.clip(w0[arm] - q[arm], -BIAS_MAX, BIAS_MAX)
        qt, ep, er = kin.ik(arm, T_target, q[arm], prior=q[arm], w_prior=.005)
        w1 = dict(w0); w1[arm] = qt + bias
        run.execute(f"servo_{arm}", rows_from_waypoints([w0, w1], run.layout, speed=run.speed * .5, min_steps=settle))
    T, q, w0 = measured(run)
    errs.append(round(float(np.linalg.norm(kin.tcp_frame(arm, q[arm])[:3, 3] - T_target[:3, 3])) * 1000, 2))
    run.log(json.dumps(dict(servo=arm, tcp_error_mm=errs)))


def release(run, arm, holds):
    """Open the hand (the weld lets go), back off 6 cm along the approach and rise 8 cm."""
    kin = run.kin
    gkey = "lg" if arm == "left" else "rg"
    w = run.targets(); w2 = dict(w); w2[gkey] = OPEN
    if run.plan_only:
        return
    run.execute(f"open_{arm}", rows_from_waypoints([w, w2], run.layout, min_steps=12))
    run.hold("settle_release", 16)
    T, q, w0 = measured(run)
    T0 = kin.tcp_frame(arm, q[arm])
    back = T0.copy(); back[:3, 3] -= .06 * T0[:3, 0]
    up = back.copy(); up[2, 3] += .08
    qs, worst = arm_path(run, arm, line(T0, back) + line(back, up), w0[arm])
    wps = [w0] + [dict(w0, **{arm: qq}) for qq in qs[::2]] + [dict(w0, **{arm: qs[-1]})]
    run.execute(f"retreat_{arm}", rows_from_waypoints(wps, run.layout, speed=run.speed))


def regrasp(run, arm, grip, holds, release=True):
    """Release the arm's piece, back off, go over the lock at a safe height, come down onto the new grip, close.
    release=False: the hand is empty already (the table start)."""
    kin, d = run.kin, run.design
    gkey = "lg" if arm == "left" else "rg"
    # 1. open (the weld lets go on the open command); the piece settles in the lock
    w = run.targets(); w2 = dict(w); w2[gkey] = OPEN
    if not run.plan_only and release:
        run.execute(f"open_{arm}", rows_from_waypoints([w, w2], run.layout, min_steps=12))
        run.hold("settle_release", 16)
    T, q, w0 = measured(run)
    piece = piece_of(grip)
    G = np.asarray((d["B"] if piece == "piece_0" else d["A"]).grips[grip], float)
    # the grip and its 180 deg twin about the approach (the same parallel grasp, pads swapped): the one whose IK is
    # exact and nearest the arm's current joints
    best = None
    for flip in (False, True):
        Tc = T[piece] @ G @ (np.diag([1., -1., -1., 1.]) if flip else np.eye(4))
        first = [np.asarray(d["birth_q"][arm], float)] if (not release and d.get("birth_q", {}).get(arm)) else []
        qc, ep, er = kin.ik_multi(arm, Tc, first + [w0[arm], np.array([0., -.5, .8, 0., .6, 0.]),
                                            np.array([0., -.3, .5, 0., 1.2, 0.]), np.array([.5, -.4, .8, 0., .8, 0.]),
                                            np.array([-.5, -.4, .8, 0., .8, 0.])])
        score = np.abs(qc - w0[arm]).max() + (10. if (ep > 1.5e-3 or er > .02) else 0.)
        if best is None or score < best[0]:
            best = (score, Tc, flip, ep, er)
    Tg = best[1]
    run.log(json.dumps(dict(regrasp_target=grip, flip=best[2], ik_err=[best[3], best[4]])))
    T0 = kin.tcp_frame(arm, q[arm])
    corners = np.concatenate([d["A"].corners() @ T["piece_1"][:3, :3].T + T["piece_1"][:3, 3],
                              d["B"].corners() @ T["piece_0"][:3, :3].T + T["piece_0"][:3, 3]])
    safe_z = corners[:, 2].max() + .08
    back = T0.copy(); back[:3, 3] -= .05 * T0[:3, 0]
    up = back.copy(); up[2, 3] = max(safe_z, back[2, 3])
    pre = Tg.copy(); pre[:3, 3] -= .07 * Tg[:3, 0]
    over = pre.copy(); over[2, 3] = max(safe_z, pre[2, 3])
    other = [a for a in ARMS if a != arm][0]
    held_other = tuple((a, piece_of(g)) for a, g in holds.items() if a != arm)

    def hits_of(qs_):
        out = []
        for k in range(0, len(qs_), 3):
            c = kin.contacts({arm: qs_[k], other: w0[other]}, {arm: OPEN, other: .0222},
                             {"piece_0": T["piece_0"], "piece_1": T["piece_1"]}, held=held_other, margin=.001)
            if c:
                out.append((k, c[:3]))
        return out
    # 1. cartesian: back off along the approach, rise to the safe height (orientation kept); the first variant whose
    # IK holds (placement2.LIFTS: the search checked the same list)
    from topogym.planning.interlocked_claws.redesign.placement2 import LIFTS
    for b_, u_ in LIFTS:
        back = T0.copy(); back[:3, 3] -= b_ * T0[:3, 0]
        up = back.copy(); up[2, 3] = max(safe_z, back[2, 3]) if u_ is None else back[2, 3] + u_
        q1, w1 = arm_path(run, arm, line(T0, back) + line(back, up), w0[arm])
        if w1.get("ok", False):
            break
    # 2. a joint-space hop to the point over the pre-grasp (new orientation), 3. down to the pre-grasp and in
    best = None
    from topogym.planning.interlocked_claws.redesign.placement2 import OVERS, over_z
    for extra in OVERS:
        ov = pre.copy(); ov[2, 3] = over_z(safe_z, pre[2, 3], extra)
        seeds = [q1[-1], w0[arm], np.array([0., -.5, .8, 0., .6, 0.]), np.array([0., -.3, .5, 0., 1.2, 0.]),
                 np.array([.5, -.4, .8, 0., .8, 0.]), np.array([-.5, -.4, .8, 0., .8, 0.])]
        q_ov, ep, er = kin.ik_multi(arm, ov, seeds)
        if ep > 1.5e-3 or er > .02:
            continue
        n = max(2, int(np.ceil(np.abs(q_ov - q1[-1]).max() / .02)))
        hop = [q1[-1] + u * (q_ov - q1[-1]) for u in np.linspace(0, 1, n + 1)[1:]]
        q3, w3 = arm_path(run, arm, line(ov, pre, .004) + line(pre, Tg, .002), q_ov)
        cand = list(q1) + hop + list(q3)
        h = hits_of(cand)
        score = len(h) + (100 if not w3.get("ok", True) and w3.get("tracked_worst", [0])[0] > 3e-3 else 0)
        if best is None or score < best[0]:
            best = (score, cand, dict(lift=w1, approach=w3, extra=extra), h)
        if score == 0:
            break
    if best is None:                              # no IK over the pre-grasp: the old cartesian path
        q3, w3 = arm_path(run, arm, line(up, over, .006) + line(over, pre) + line(pre, Tg, .002), q1[-1])
        best = (None, list(q1) + list(q3), dict(lift=w1, approach=w3, extra=None), hits_of(list(q1) + list(q3)))
    _, qs, worst, hits = best
    run.log(json.dumps(dict(plan=f"regrasp_{arm}_{grip}", waypoints=len(qs), ik_worst=worst, collisions=hits[:4],
                            n_collisions=len(hits), safe_z=round(float(safe_z), 3)), default=str))
    if run.plan_only:
        return
    wps = [w0] + [dict(w0, **{arm: qq, gkey: OPEN}) for qq in qs[::2]] + [dict(w0, **{arm: qs[-1], gkey: OPEN})]
    run.execute(f"reach_{arm}_{grip}", rows_from_waypoints(wps, run.layout, speed=run.speed))
    servo_to(run, arm, Tg, gkey)
    w = run.targets(); w2 = dict(w); w2[gkey] = CLOSED
    run.execute(f"close_{arm}", rows_from_waypoints([w, w2], run.layout, min_steps=20))
    run.hold("acquire_" + grip, 20)
    run.log(json.dumps(dict(regrasp=grip, held=run.s.object_in_hand(arm))))


def _old_regrasp_tail():
    pass


def transport(run, holds):
    """The hand holding B carries it (lift above the lock, over, down) so that B's box centre is at the goal area's
    centre, 4 cm above the table; the other hand keeps A where it is."""
    kin, d = run.kin, run.design
    T, q, w0 = measured(run)
    arm = [a for a, g in holds.items() if piece_of(g) == "piece_0"][0]
    T_B = T["piece_0"]
    off = np.linalg.inv(T_B) @ kin.tcp_frame(arm, q[arm])
    region = run.task.goal_region
    wc = d["B"].corners() @ T_B[:3, :3].T + T_B[:3, 3]
    lo, hi = wc.min(0), wc.max(0)
    centre = (lo + hi) / 2
    a_top = (d["A"].corners() @ T["piece_1"][:3, :3].T + T["piece_1"][:3, 3])[:, 2].max()
    # the nearest centre that puts B's whole box 3 cm inside the area (the area's centre if B is too long for it)
    c, half = np.asarray(region.center_xy_m, float), np.asarray(region.half_size_m, float)
    ext = (hi - lo)[:2] / 2
    room = np.maximum(half - ext - .03, 0.)
    gxy = c + np.clip(centre[:2] - c, -room, room)
    goal = np.array([gxy[0], gxy[1], region.table_height_m + .04 + (centre[2] - lo[2])])
    from topogym.planning.interlocked_claws.redesign.geom import PairChecker
    pair = PairChecker(d["A"].boxes, d["B"].boxes)
    T_Ai = np.linalg.inv(T["piece_1"])
    other = [a for a in ARMS if a != arm][0]

    def check(qs_):
        """min B-A distance along a joint path (B carried by the hand) and robot contacts."""
        dmin, hits = np.inf, []
        for k in range(0, len(qs_), 3):
            TB_ = kin.tcp_frame(arm, qs_[k]) @ np.linalg.inv(off)
            dmin = min(dmin, pair.distance(T_Ai @ TB_))
            c = kin.contacts({arm: qs_[k], other: w0[other]}, {arm: .0222, other: .0222},
                             {"piece_0": TB_, "piece_1": T["piece_1"]}, held=((arm, "piece_0"),), margin=.001)
            if c:
                hits.append((k, c[:2]))
        return dmin, hits
    best = None
    for extra in (0., .05, .10, .15):
        lift = centre.copy(); lift[2] = max(centre[2] + .05, a_top + .04 + (centre[2] - lo[2])) + extra
        TB1 = T_B.copy(); TB1[:3, 3] += lift - centre
        lift_T = []
        for c_ in [centre + u * (lift - centre) for u in np.linspace(0, 1, 16)[1:]]:
            X = T_B.copy(); X[:3, 3] += c_ - centre
            lift_T.append(X @ off)
        q1, w1 = arm_path(run, arm, lift_T, w0[arm])
        cands_g = []
        for yaw in (0., 30., -30., 60., -60., 90., -90.):
            Rz = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
            Rg = Rz @ T_B[:3, :3]
            wcg = d["B"].corners() @ Rg.T
            lo_g, hi_g = wcg.min(0), wcg.max(0)
            ext_g = (hi_g - lo_g)[:2] / 2
            room_g = np.maximum(half - ext_g - .03, 0.)
            cg = (lo_g + hi_g) / 2                       # box centre offset from B's origin
            for fx in np.linspace(-1, 1, 5):
                for fy in np.linspace(-1, 1, 5):
                    gxy_ = c + np.array([fx, fy]) * room_g
                    cands_g.append((np.linalg.norm(gxy_ - centre[:2]) + abs(yaw) / 300., yaw, Rg, gxy_, lo_g, cg))
        cands_g.sort(key=lambda t: t[0])
        tried = 0
        for _, yaw, Rg, gxy_, lo_g, cg in cands_g:
            if tried >= 12:
                break
            TBg = np.eye(4); TBg[:3, :3] = Rg
            TBg[:3, 3] = np.r_[gxy_, region.table_height_m + .04 - lo_g[2]] - np.r_[cg[:2], 0.]
            TBo = TBg.copy(); TBo[2, 3] = max(TBg[2, 3] + .05, TB1[2, 3])     # over the goal at the lift height
            q_ov, ep, er = kin.ik_multi(arm, TBo @ off, [q1[-1], w0[arm], np.array([0., -.5, .8, 0., .6, 0.]),
                                                         np.array([0., -.3, .5, 0., 1.2, 0.])])
            if ep > 1.5e-3 or er > .02:
                continue
            tried += 1
            n = max(2, int(np.ceil(np.abs(q_ov - q1[-1]).max() / .02)))
            hop = [q1[-1] + u * (q_ov - q1[-1]) for u in np.linspace(0, 1, n + 1)[1:]]
            q3, w3 = arm_path(run, arm, line(TBo @ off, TBg @ off, .004), q_ov)
            if not w3.get("ok", True) and w3.get("tracked_worst", [0])[0] > 3e-3:
                continue
            cand = list(q1) + hop + list(q3)
            dmin, hits = check(cand)
            score = len(hits) + (50 if dmin < .004 else 0) + abs(yaw) / 1000.
            if best is None or score < best[0]:
                best = (score, cand, dict(extra=extra, yaw=yaw, min_B_A_m=round(float(dmin), 4), hits=hits[:3],
                                          goal=np.round(TBg[:3, 3], 3).tolist()))
            if score < 1:
                break
        if best is not None and best[0] < 1:
            break
    if best is None:                              # the straight cartesian carry (tracked), as the first oracles did
        pts = [centre + u * (lift - centre) for u in np.linspace(0, 1, 16)[1:]]
        over_ = np.array([goal[0], goal[1], lift[2]])
        pts += [lift + u * (over_ - lift) for u in np.linspace(0, 1, 70)[1:]]
        pts += [over_ + u * (goal - over_) for u in np.linspace(0, 1, 30)[1:]]
        T_list = []
        for c_ in pts:
            X = T_B.copy(); X[:3, 3] += c_ - centre
            T_list.append(X @ off)
        qs_, rep_ = arm_path(run, arm, T_list, w0[arm])
        best = (None, qs_, dict(fallback="cartesian", ik=rep_, goal=np.round(goal, 3).tolist()))
    qs = best[1]
    run.log(json.dumps(dict(plan=f"transport_{arm}", **best[2], region_half=region.half_size_m), default=str))
    if run.plan_only:
        return
    wps = [w0] + [dict(w0, **{arm: qq}) for qq in qs[::2]] + [dict(w0, **{arm: qs[-1]})]
    run.execute("transport", rows_from_waypoints(wps, run.layout, speed=run.speed))


PROGRAMS = {t: program_generic for t in ("very_easy", "easier", "easy", "medium", "hard")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tier")
    ap.add_argument("out")
    ap.add_argument("--speed", type=float, default=.25)
    ap.add_argument("--share", type=float, default=.5, help="the left hand's share of the relative motion")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--open-loop", action="store_true", help="no chunked feedback on the moves")
    a = ap.parse_args(argv)
    run = Runner(a.tier, a.out, speed=a.speed, share=a.share, plan_only=a.plan_only, feedback=not a.open_loop)
    meta = PROGRAMS[a.tier](run)
    print(json.dumps({k: meta.get(k) for k in ("success", "episode_end", "rows", "final_state_sha", "birth_sha")}))


if __name__ == "__main__":
    main()
