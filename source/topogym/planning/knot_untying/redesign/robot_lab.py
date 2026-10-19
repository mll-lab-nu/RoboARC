"""Robot backend of the redesign's untying oracle: the object-level lab's interface (mjlab.Lab, as untie.Untie uses it)
on the real ALOHA in a MuJoCo rcb_topogym.Session, so the decision logic of untie.Untie is shared by the lab and the
robot (knot-robot, 2026-10-10/11).

What a lab hand is, on the robot:
  - a hand = one ALOHA arm. Its position (hand_pos) is the commanded pads' contact point (the TCP + R CONTACT_IN_TCP_M,
    the family's vertical-pinch convention, topogym/planning/knot_untying/kinematics.py), i.e. the rope centre while
    it pinches; its orientation is a vertical pinch (TCP x down, jaws across the rope), kept while it moves.
  - move(): every listed hand on a straight Cartesian min-jerk path at <= speed (the lab's timing rule), one Session row
    per control step: damped least-squares IK of the TCP on a private MjData of the Session's own model (never the
    live data), seeded with the previous row. An IK failure stops that hand there (the family Runner stops on an IK
    failure) and flags it (reach_stopped).
  - grasp(): the jaws close on the rope where the hand is; the World binds contact_weld_v1 (both fingers on a rope
    link, 20 ms) or not: the grasp is read from the GraspManager's binding, never assumed. A lost binding (the weld
    slipped, ~8 N on a rope link) shows as hands[h]["grasp"] None and an event "slip".
  - the hand's force window = the binding weld's slip window (the MuJoCo grasp backend's last 20 ms of translational
    weld force, read only); max_force its largest mean.
  - rows: every row goes through the standard-rate limiter (topogym/planning/standard_rate.py, "spread") into
    Session.step (joint clip, pi rad/s slew, table guard); the Session records the accepted rows. Nothing here writes
    the live MjData / MjModel: replaying the recorded rows in a fresh process gives the same states.
Ropes: rods[name] views of the World's rope chains (material s from 0 at the chain's first node, node positions, link
centres, link speeds), the lab's crossings / pinch clearance / goal metrics on them; fixtures from the lab scene
(task-local, as scene_start.json declares them).
"""
from __future__ import annotations

import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .mjlab import TABLE_TOP, to_local, to_world

CONTACT_IN_TCP_M = np.array([.01381, 0., -.00055])      # pads' contact point in the TCP frame (family kinematics.py)
# gripper widths (pad gap, m): approach, pinch. The pinch commands the fully closed jaws: the finger drive then presses
# the 8 mm rope with its largest force (~2 N with the jaws commanded shut, knot-inst's weld_release.json), and the hold
# along the rope is mostly the pads' friction (the weld itself lets the link drift 25 mm before its pose rule drops it)
OPEN_W, CLOSE_W = .030, .0
RELEASE_W = .016                                         # a release opens 4 mm clear of the rope on each side
FINGER_Q_ZERO_GAP = .0083     # the finger joints' value at a zero pad gap (planner model, measured: pad gap = 2 (q - this);
                              # the model's qpos0 = 0 has the finger meshes 15.6 mm through each other)
HOLD_W = .008                 # the pad gap while the jaws hold the 8 mm rope
HOME_Q = np.array([0., -.96, 1.16, 0., -.3, 0.])
WRIST_UNSAFE = (np.deg2rad(116.), np.deg2rad(150.))     # wrist_angle above 116 deg with |wrist_rotate| above 150 deg
BIAS_LIMIT_M = .015                                      # cap of the learned per-arm contact-point bias
BLOCKED_M = .008                                         # a resting arm further than this off its target is blocked


class Solved(Exception):
    """The Session ended the episode in success (later rows would not change anything)."""


class Budget(Exception):
    """The wall-clock budget is used up."""


class EpisodeEnded(RuntimeError):
    pass


def minjerk(u):
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def vertical_pinch_R(tangent_xy, flip=1):
    """TCP rotation of a vertical pinch: x down, jaws (y) across the rope tangent (family vertical_pinch)."""
    down = np.array([0., 0., -1.])
    t = np.asarray(tangent_xy, float)[:2]
    t = t / max(np.linalg.norm(t), 1e-12)
    jaw = flip * np.array([-t[1], t[0], 0.])
    return np.column_stack([down, jaw, np.cross(down, jaw)])


# ------------------------------------------------------------------------------------------------ kinematics
class ArmKinematics:
    """Stateless TCP forward / inverse kinematics of the ALOHA arms on a private MjData of the Session's model."""

    def __init__(self, session):
        import mujoco
        self.mj = mujoco
        w = session.env._world
        robot = session.env._robot
        self.m, self.reduced = self._robot_model(w), True
        if self.m is None:                      # fallback: the Session's whole model (rope links included)
            self.m, self.reduced = w.model, False
        self.d = mujoco.MjData(self.m)
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))
        self.arm = {}
        for side, g in robot.groups.items():
            jids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in g.arm_joints]
            if min(jids) < 0:
                raise ValueError(f"model lacks the {side} arm joints")
            sl = session.robot.arm_slice[side]
            q = np.asarray(g.tcp_orientation_in_hand_wxyz, float)
            self.arm[side] = dict(
                qadr=np.array([int(self.m.jnt_qposadr[j]) for j in jids]),
                dadr=np.array([int(self.m.jnt_dofadr[j]) for j in jids]),
                body=mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, g.hand_link),
                tcp_p=np.asarray(g.tcp_position_in_hand_m, float),
                R_th=Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix(),
                lo=np.asarray(session.robot.lo[sl], float) + 1e-6, hi=np.asarray(session.robot.hi[sl], float) - 1e-6,
                names=[n.split("/")[-1] for n in g.arm_joints])

    @staticmethod
    def _robot_model(world):
        """The Session's own scene without its rope chains (robot, table, fixtures: the same bodies, joints and geoms):
        the planner's FK / IK / distance queries run ~10x faster on it than on the 300-link chains. None if it fails."""
        try:
            import xml.etree.ElementTree as ET
            import mujoco
            from .robot_ik import _resolve_meshes
            scene_dir = Path(world.scene_dir)
            root = ET.parse(scene_dir / "scene.xml").getroot()
            _resolve_meshes(root, scene_dir)
            prefixes = tuple(f"{name}::" for name in world._ropes)
            wb = root.find("worldbody")
            for body in list(wb.findall("body")):
                if (body.get("name") or "").startswith(prefixes):
                    wb.remove(body)
            for tag in ("equality", "contact"):
                sec = root.find(tag)
                if sec is None:
                    continue
                for el in list(sec):
                    if any(str(v).startswith(prefixes) for v in el.attrib.values()):
                        sec.remove(el)
            m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
            for name in ("left/gripper_link", "right/gripper_link"):
                if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
                    return None
            return m
        except Exception as e:  # noqa: BLE001
            print("ArmKinematics: reduced model failed, using the full one:", repr(e)[:200], flush=True)
            return None

    def fk(self, side, q6):
        A = self.arm[side]
        self.d.qpos[A["qadr"]] = q6
        self.mj.mj_kinematics(self.m, self.d)
        Rh = self.d.xmat[A["body"]].reshape(3, 3).copy()
        return self.d.xpos[A["body"]] + Rh @ A["tcp_p"], Rh @ A["R_th"]

    def contact(self, side, q6):
        p, R = self.fk(side, q6)
        return p + R @ CONTACT_IN_TCP_M, R

    def solve(self, side, p_tcp, R_tcp, seed, iters=80, tol_p=2e-4, tol_r=2e-3, w_rot=.2):
        """DLS IK of the TCP to (p_tcp, R_tcp) from `seed`, joint limits enforced. -> (q6, pos err m, rot err rad)."""
        A = self.arm[side]
        q = np.clip(np.asarray(seed, float), A["lo"], A["hi"])
        best = None
        for _ in range(iters):
            p, R = self.fk(side, q)
            ep = np.asarray(p_tcp) - p
            er = Rotation.from_matrix(np.asarray(R_tcp) @ R.T).as_rotvec()
            e_p, e_r = float(np.linalg.norm(ep)), float(np.linalg.norm(er))
            if best is None or e_p + w_rot * e_r < best[0]:
                best = (e_p + w_rot * e_r, q.copy(), e_p, e_r)
            if e_p < tol_p and e_r < tol_r:
                break
            self.mj.mj_comPos(self.m, self.d)
            self.mj.mj_jac(self.m, self.d, self.jacp, self.jacr, p, A["body"])
            J = np.vstack([self.jacp[:, A["dadr"]], w_rot * self.jacr[:, A["dadr"]]])
            e = np.r_[ep, w_rot * er]
            lam = .02 if e_p > .01 else .004
            dq = J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(6), e)
            step = float(np.max(np.abs(dq)))
            if step > .25:
                dq *= .25 / step
            q = np.clip(q + dq, A["lo"], A["hi"])
        _, q, e_p, e_r = best
        return q, e_p, e_r

    @staticmethod
    def wrist_unsafe(q6):
        return bool(q6[4] > WRIST_UNSAFE[0] and abs(q6[5]) > WRIST_UNSAFE[1])


# ------------------------------------------------------------------------------------------------ rope views
class ChainView:
    """rods[name] for untie.Untie: a World rope chain with material s from 0 at its first node."""

    def __init__(self, chain):
        self.chain = chain
        self.bodies = np.asarray(chain.bodies)
        self.lengths = np.asarray(chain.lengths, float)
        self.s_nodes = np.asarray(chain.s_nodes, float) - float(chain.s_nodes[0])
        self.radius = float(chain.record["rope"]["radius_m"])
        self.n = len(self.bodies)
        self.dof = np.asarray(chain.dof)

    def nodes(self, d):
        return self.chain.node_positions(d)

    def centres(self, d):
        P = np.asarray(d.xpos)[self.bodies]
        X = np.asarray(d.xmat)[self.bodies].reshape(-1, 3, 3)[:, :, 0]
        return P + X * (self.lengths[:, None] / 2)

    def speeds(self, d):
        """Link origin speeds (m/s): a free joint's linear velocity, world frame."""
        v = np.asarray(d.qvel)
        return np.linalg.norm(np.stack([v[self.dof + k] for k in range(3)], axis=1), axis=1)


def _fixture_distance(item, p_local):
    """Distance from a task-local point to a scene fixture's surface (box / cylinder / capsule)."""
    if item["kind"] == "capsule":
        a, b = np.asarray(item["fromto"][:3], float), np.asarray(item["fromto"][3:], float)
        t = np.clip(np.dot(p_local - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0, 1)
        return float(np.linalg.norm(p_local - (a + t * (b - a))) - float(item["radius"]))
    c = np.asarray(item.get("pos", (0, 0, 0)), float)
    yaw = np.deg2rad(float(item.get("yaw_deg", 0.)))
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.]])
    q = R.T @ (np.asarray(p_local, float) - c)
    if item["kind"] == "box":
        out = np.abs(q) - np.asarray(item["size"], float) / 2
        return float(np.linalg.norm(np.maximum(out, 0)) + min(out.max(), 0))
    if item["kind"] == "cylinder":
        dr = math.hypot(q[0], q[1]) - float(item["radius"])
        dz = abs(q[2]) - float(item["height"]) / 2
        return float(math.hypot(max(dr, 0), max(dz, 0)) + min(max(dr, dz), 0))
    raise ValueError(item["kind"])


# ------------------------------------------------------------------------------------------------ the backend
class RobotLab:
    """The lab interface (hands, rods, point/tangent/length/crossings/pinch_clearance_at, move/grasp/release/wait/settle,
    t, d, events, max_force) on the ALOHA in a recording Session."""

    def __init__(self, session, scene, *, pinch_clearance_m=.010, hand_limit_n=8., rate_limit=True, budget_s=None,
                 trace_every=8, log=print, trace_path=None, approach_width=OPEN_W, close_width=CLOSE_W, hash_every=400):
        from topogym.planning.standard_rate import RateLimiter
        self.s = session
        self.w = session.env._world
        self.m, self.d = self.w.model, self.w.data
        self.robot = session.env._robot
        self.task = session.env._task
        self.scene = scene or {}
        self.ik = ArmKinematics(session)
        self.rods = {name: ChainView(c) for name, c in self.w._ropes.items()}
        self.dt = 1. / float(session.control_hz)
        self.P = dict(dt=self.dt)
        self.pinch_clearance = float(pinch_clearance_m)
        self.close_width, self.approach_width = float(close_width), float(approach_width)
        self.sides = tuple(self.robot.groups)
        self.hands = {h: dict(grasp=None, window=[0.], limit=float(hand_limit_n)) for h in self.sides}
        self.max_force = {h: 0. for h in self.sides}
        self.events, self.trace = [], []
        self.limiter = RateLimiter(session) if rate_limit else None
        self.budget_s, self.wall0 = budget_s, time.time()
        self.trace_every, self.log = int(trace_every), log
        self.trace_path = None if trace_path is None else Path(trace_path)
        self.rows = 0
        self.hash_every = int(hash_every or 0)
        self.hashes = []              # (row, Session.state_sha()) every hash_every rows: a replay checks them on the way
        self.reach_stopped = {h: None for h in self.sides}
        self.ik_worst = {h: 0. for h in self.sides}
        # real_fingers (the runner's --real-fingers): the fixture distances use the fingers' real opening (a planned pad
        # gap inside plan_width(), else the live fingers); otherwise the planner model keeps qpos0's over-closed fingers
        self.real_fingers = False
        self._plan_width = {}
        self._binding_seen = {h: None for h in self.sides}
        self._releasing = {h: False for h in self.sides}
        self.slips = {h: 0 for h in self.sides}
        # the PD-tracked arm leaves a few mm of contact-point error at rest: converge() learns a per-arm bias that every
        # Cartesian target then carries (the family Runner's TCP bias), capped at BIAS_LIMIT_M
        self.bias = {h: np.zeros(3) for h in self.sides}
        row = session.hold_action().astype(np.float64)
        self.q = {h: row[session.robot.arm_slice[h]].copy() for h in self.sides}
        self.g = {h: float(row[session.robot.grip_index[h]]) for h in self.sides}
        self.R = {}
        self.cmd = {}
        for h in self.sides:
            c, R = self.ik.contact(h, self.q[h])
            self.cmd[h], self.R[h] = c, R
        self.fixtures = [f for f in self.scene.get("fixtures", []) if f.get("collide", True)]
        for h in self.sides:          # a resumed run may start with a hand holding the rope: read the World's binding
            self.hands[h]["grasp"] = self._bound_station(h)
        self.tip_below = self.fingertips_below_contact()
        self._tip_below_contact = max(self.tip_below.values())
        self.gap_geoms = self._gap_geoms()
        self.fixture_geoms = self._fixture_geoms()

    def _fixture_geoms(self):
        """Collision geoms of the scene's fixtures (posts, curb, clamp staple): task bodies named <fixture>::<link>, in the
        planner's kinematics model (self.ik.m)."""
        import mujoco
        m = self.ik.m
        names = [f["name"] for f in self.scene.get("fixtures", [])]
        out = []
        for g in range(m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
            if any(body == n or body.startswith(n + "::") for n in names):
                out.append(g)
        return out

    def fixture_gap(self, h, q_h, distmax=.05):
        """Smallest distance (m, capped at distmax) between arm h at q_h (forearm, wrist, gripper geoms) and the scene's
        fixtures (a 16 cm post next to a pinch can meet the gripper's 15 cm finger bar or its camera mount)."""
        import mujoco
        if not self.fixture_geoms:
            return distmax
        m, d = self.ik.m, self.ik.d
        d.qpos[self.ik.arm[h]["qadr"]] = q_h
        self._pose_fingers(h)
        mujoco.mj_kinematics(m, d)
        best = distmax
        for g1 in self.gap_geoms[h]:
            for g2 in self.fixture_geoms:
                best = min(best, float(mujoco.mj_geomDistance(m, d, g1, g2, distmax, None)))
        return best

    def fixture_nearest(self, h, q_h, distmax=.05):
        """The nearest fixture to arm h at q_h: (distance m, the arm's nearest point, the fixture's nearest point), world;
        None if none within distmax."""
        import mujoco
        if not self.fixture_geoms:
            return None
        m, d = self.ik.m, self.ik.d
        d.qpos[self.ik.arm[h]["qadr"]] = q_h
        self._pose_fingers(h)
        mujoco.mj_kinematics(m, d)
        best, fromto = None, np.zeros(6)
        for g1 in self.gap_geoms[h]:
            for g2 in self.fixture_geoms:
                dist = float(mujoco.mj_geomDistance(m, d, g1, g2, distmax, fromto))
                if dist < distmax and (best is None or dist < best[0]):
                    best = (dist, fromto[:3].copy(), fromto[3:].copy())
        return best

    def _finger_adr(self, h):
        """(planner-model qpos addresses, live qpos addresses) of hand h's two finger joints."""
        cache = getattr(self, "_fadr", None)
        if cache is None:
            import mujoco
            cache = self._fadr = {}
            for side in self.sides:
                pa, la = [], []
                for n in ("left_finger", "right_finger"):
                    j = mujoco.mj_name2id(self.ik.m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}/{n}")
                    jl = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}/{n}")
                    if j >= 0 and jl >= 0:
                        pa.append(int(self.ik.m.jnt_qposadr[j]))
                        la.append(int(self.m.jnt_qposadr[jl]))
                cache[side] = (pa, la)
        return cache[h]

    def _pose_fingers(self, h, width=None):
        """With real_fingers: hand h's finger joints in the planner's private data at the pad gap `width` (m; None: the
        width plan_width() set for h, else the live fingers). Without it they stay at the model's qpos0, past closed,
        i.e. 12 mm (holding the rope) to 23 mm (opened to 30 mm) narrower per side than the real fingers: an opened
        gripper planned 4 mm from the curb came down on its top (medium_loose inst8, 10-11)."""
        if not self.real_fingers:
            return
        pa, la = self._finger_adr(h)
        if not pa:
            return
        w = width if width is not None else self._plan_width.get(h)
        q = float(np.mean(self.d.qpos[la])) if w is None else FINGER_Q_ZERO_GAP + .5 * float(w)
        self.ik.d.qpos[pa] = float(np.clip(q, 0., .041))

    def _model_fingers(self, h):
        """With real_fingers: hand h's finger joints back at the model's qpos0 in the private data (the arm-to-arm gaps
        keep the model GAP_M was tuned on; a fixture check may have opened them)."""
        if not self.real_fingers:
            return
        pa, _ = self._finger_adr(h)
        if pa:
            self.ik.d.qpos[pa] = self.ik.m.qpos0[pa]

    @contextlib.contextmanager
    def plan_width(self, **widths):
        """Inside: the fixture distances of the hands named take their fingers at these pad gaps (m)."""
        saved = dict(self._plan_width)
        self._plan_width.update({h: float(w) for h, w in widths.items()})
        try:
            yield
        finally:
            self._plan_width = saved

    def _gap_geoms(self):
        """{hand: collision geoms of its lower forearm, wrist and gripper subtree}: what two arms working close
        together can touch (the gripper's 15 cm finger bar and its camera mount are the wide parts); planner model ids."""
        import mujoco
        m = self.ik.m
        out = {}
        for h in self.sides:
            root = self.ik.arm[h]["body"]
            sub = {root}
            grown = True
            while grown:
                grown = False
                for b in range(m.nbody):
                    if b not in sub and int(m.body_parentid[b]) in sub:
                        sub.add(b)
                        grown = True
            # the whole moving arm: an arm reaching across the table passes its upper arm and forearm over the other
            # arm's side (kr_ve_d4: the left wrist came down onto the right forearm, 16 cm above its station)
            for nm in ("shoulder_link", "upper_arm_link", "upper_forearm_link", "lower_forearm_link", "wrist_link"):
                b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{h}/{nm}")
                if b >= 0:
                    sub.add(b)
            out[h] = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in sub
                      and (m.geom_contype[g] or m.geom_conaffinity[g])]
        return out

    def q_actual(self, h):
        """Arm h's measured joint positions (the live Session data, read by joint name)."""
        if not hasattr(self, "_qadr_live"):
            import mujoco
            self._qadr_live = {}
            for side in self.sides:
                adr = []
                for a in self.ik.arm[side]["qadr"]:
                    j = next(j for j in range(self.ik.m.njnt) if int(self.ik.m.jnt_qposadr[j]) == int(a))
                    name = mujoco.mj_id2name(self.ik.m, mujoco.mjtObj.mjOBJ_JOINT, j)
                    adr.append(int(self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)]))
                self._qadr_live[side] = np.array(adr)
        return np.asarray(self.d.qpos[self._qadr_live[h]], float).copy()

    def gripper_gap(self, h, q_h, q_o=None, distmax=.05):
        """Smallest distance (m, capped at distmax) between arm h at joints q_h and the other arm at q_o (default: the
        nearer of its commanded and its measured joints -- a press pin rests above its command on the rope, a PD arm
        lags: easy_v8, 10-11): forearm, wrist and gripper collision geoms, exact (mj_geomDistance on a private
        MjData)."""
        import mujoco
        o = "left" if h == "right" else "right"
        m, d = self.ik.m, self.ik.d
        others = [self.q[o], self.q_actual(o)] if q_o is None else [q_o]
        best = distmax
        for q_other in others:
            d.qpos[self.ik.arm[h]["qadr"]] = q_h
            d.qpos[self.ik.arm[o]["qadr"]] = q_other
            self._model_fingers(h)
            self._model_fingers(o)
            mujoco.mj_kinematics(m, d)
            for g1 in self.gap_geoms[h]:
                for g2 in self.gap_geoms[o]:
                    best = min(best, float(mujoco.mj_geomDistance(m, d, g1, g2, distmax, None)))
        return best

    def fingertips_below_contact(self):
        """{hand: how far the finger collision geometry reaches below the pads' contact point in a vertical pinch}; also
        sets self.gripper_radius {hand: the largest horizontal distance of the hand's gripper subtree (collision geoms,
        up to 15 cm above the contact point) from the vertical through the contact point}: two vertical grippers whose
        stations are closer than the sum collide."""
        import mujoco
        m, d = self.ik.m, self.ik.d
        out, self.gripper_radius = {}, {}
        for k, h in enumerate(self.sides):
            c0 = to_world(np.array([-.2 if h == "left" else .2, 0., .06]))
            q, _, _ = self.ik.solve(h, c0 - vertical_pinch_R([1., 0.]) @ CONTACT_IN_TCP_M, vertical_pinch_R([1., 0.]),
                                    self.q[h], iters=300)
            contact, _ = self.ik.contact(h, q)       # leaves the private data at q (mj_kinematics)
            root = self.ik.arm[h]["body"]
            sub = {root}
            grown = True
            while grown:
                grown = False
                for b in range(m.nbody):
                    if b not in sub and int(m.body_parentid[b]) in sub:
                        sub.add(b)
                        grown = True
            low, rad = np.inf, 0.
            for gid in range(m.ngeom):
                if int(m.geom_bodyid[gid]) not in sub or not (m.geom_contype[gid] or m.geom_conaffinity[gid]):
                    continue
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gid])) or ""
                if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = int(m.geom_dataid[gid])
                    v = np.asarray(m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]], float)
                    vw = np.asarray(d.geom_xpos[gid]) + v @ np.asarray(d.geom_xmat[gid]).reshape(3, 3).T
                else:
                    rb = float(m.geom_rbound[gid])
                    vw = np.asarray(d.geom_xpos[gid])[None] + rb * np.array([[0, 0, -1.], [0, 0, 1], [1, 0, 0], [-1, 0, 0],
                                                                             [0, 1, 0], [0, -1, 0]])
                if "finger" in name:
                    low = min(low, float(vw[:, 2].min()))
                near = vw[vw[:, 2] < contact[2] + .15]
                if len(near):
                    rad = max(rad, float(np.linalg.norm(near[:, :2] - contact[:2], axis=1).max()))
            out[h] = float(contact[2] - low)
            self.gripper_radius[h] = rad
        return out

    # -------------------------------------------------------------- time / state
    @property
    def t(self):
        return float(self.s.sim_time_s)

    def say(self, *a):
        if self.log:
            self.log(f"[{self.t:7.2f}s r{self.s.n_steps:5d} {time.time() - self.wall0:6.0f}w]", *a)

    # -------------------------------------------------------------- readout (the lab's)
    def centerline(self, rope):
        return self.rods[rope].nodes(self.d)

    def centerlines_local(self):
        return {k: to_local(r.nodes(self.d)) for k, r in self.rods.items()}

    def link_at(self, rope, s):
        rod = self.rods[rope]
        centres = .5 * (rod.s_nodes[:-1] + rod.s_nodes[1:])
        return int(np.argmin(np.abs(centres - s)))

    def point(self, rope, s):
        rod = self.rods[rope]
        nodes = rod.nodes(self.d)
        return np.array([np.interp(s, rod.s_nodes, nodes[:, k]) for k in range(3)])

    def tangent(self, rope, s, h=.006):
        L = self.rods[rope].s_nodes[-1]
        t = self.point(rope, min(L, s + h)) - self.point(rope, max(0., s - h))
        return t / np.linalg.norm(t)

    def length(self, rope):
        return float(self.rods[rope].s_nodes[-1])

    def pinch_clearance_at(self, rope, s, exclude_m=.03):
        p = self.point(rope, s)
        best = np.inf
        for name, rod in self.rods.items():
            c = rod.centres(self.d)
            sc = .5 * (rod.s_nodes[:-1] + rod.s_nodes[1:])
            keep = np.ones(len(c), bool) if name != rope else np.abs(sc - s) > exclude_m
            if keep.any():
                best = min(best, float(np.min(np.linalg.norm(c[keep] - p, axis=1))) - rod.radius)
        pl = to_local(p)
        for f in self.fixtures:
            best = min(best, _fixture_distance(f, pl))
        return best

    def crossings(self):
        from topogym.tasks.knot_untying.redesign.geometry import projected_crossings
        return projected_crossings([self.rods[k].nodes(self.d) for k in self.rods])

    def goal_metrics(self, region=None):
        from .mjlab import Lab
        return Lab.goal_metrics(self, region)

    def quiet(self, limit=.001):
        return max(float(r.speeds(self.d).max()) for r in self.rods.values()) <= limit

    def settle(self, max_s=3., limit=.001, hold=.5, check_every=.05):
        t0, quiet_since = self.t, None
        while self.t - t0 < max_s:
            self.wait(check_every)
            if self.quiet(limit):
                quiet_since = self.t if quiet_since is None else quiet_since
                if self.t - quiet_since >= hold - 1e-9:
                    return True
            else:
                quiet_since = None
        return False

    # -------------------------------------------------------------- hands
    def hand_pos(self, h):
        """The hand's commanded position: the commanded pads' contact point (world)."""
        return self.cmd[h].copy()

    def hand_actual(self, h):
        p, q = self.s.ee_pose(h)
        R = Rotation.from_quat(np.asarray(q, float)[[1, 2, 3, 0]]).as_matrix()
        return np.asarray(p, float) + R @ CONTACT_IN_TCP_M

    def binding(self, h):
        gm = self.w._grasp_manager
        return None if gm is None else gm.hands[h]["binding"]

    def _weld_window(self, h):
        gm = self.w._grasp_manager
        b = self.binding(h)
        if gm is None or b is None:
            return [0.]
        win = gm.backend._window.get(int(b["weld"]))
        return list(win) if win else [0.]

    def floor_z(self, h, R=None):
        """Lowest commanded contact-point z for hand h (world): the fingertips 1 mm above the table for a vertical
        hand (the table guard clamps lower commands anyway)."""
        return TABLE_TOP + getattr(self, "_tip_below_contact", .0) + .001

    # -------------------------------------------------------------- rows
    def _row(self):
        row = self.s.hold_action().astype(np.float64)
        for h in self.sides:
            row[self.s.robot.arm_slice[h]] = self.q[h]
            row[self.s.robot.grip_index[h]] = self.g[h]
        return row

    def _step(self, label=""):
        s = self.s
        if s.episode_end == "success":
            raise Solved()
        if s.episode_end is not None:
            raise EpisodeEnded(f"episode ended: {s.episode_end}")
        if self.budget_s is not None and time.time() - self.wall0 > self.budget_s:
            raise Budget(f"wall-clock budget {self.budget_s / 60:.0f} min used up")
        row = self._row()
        t0 = time.time()
        if self.limiter is not None:
            self.limiter.step(row)
        else:
            s.step(row.astype(np.float32))
        self.phys_s = getattr(self, "phys_s", 0.) + time.time() - t0     # wall time inside Session.step (physics)
        self.rows += 1
        self._after_step(label)
        if s.episode_end == "success":
            raise Solved()
        if s.episode_end is not None:
            raise EpisodeEnded(f"episode ended: {s.episode_end}")

    def _after_step(self, label):
        gm = self.w._grasp_manager
        for h in self.sides:
            b = None if gm is None else gm.hands[h]["binding"]
            hd = self.hands[h]
            if hd["grasp"] is not None and hd["grasp"].get("press"):
                hd["window"] = [0.]           # a press pin: no weld, nothing to slip
                continue
            if b is not None:
                win = self._weld_window(h)
                hd["window"] = win
                mean = float(np.mean(win))
                self.max_force[h] = max(self.max_force[h], mean)
                hd["force_peak"] = max(hd.get("force_peak", 0.), mean)
            else:
                last = float(np.mean(hd["window"]))       # the weld's 20 ms mean force on the row before it let go
                hd["window"] = [0.]
                if hd["grasp"] is not None and not self._releasing[h]:
                    reason = gm.hands[h].get("rejection") if gm is not None else None
                    lost = next((e for e in reversed(gm.events) if e.get("hand") == h and e.get("event") == "lost"),
                                None) if gm is not None else None
                    self.events.append(dict(t=round(self.t, 4), row=int(self.s.n_steps), kind="slip", hand=h,
                                            reason=reason, force_last=round(last, 3),
                                            force_peak=round(hd.get("force_peak", 0.), 3),
                                            lost_physics_step=None if lost is None else lost.get("physics_step"),
                                            tcp_err_mm=round(1000 * float(np.linalg.norm(self.hand_actual(h) - self.cmd[h]
                                                                                          - self.bias[h])), 2),
                                            **hd["grasp"]))
                    self.slips[h] += 1
                    self.say(f"{h}: grasp lost ({reason}): 20 ms weld force {last:.2f} N on the row before, peak "
                             f"{hd.get('force_peak', 0.):.2f} N, held station s={hd['grasp']['s']:.3f}")
                    hd["grasp"] = None
        if self.hash_every and self.s.n_steps % self.hash_every == 0:
            self.hashes.append((int(self.s.n_steps), self.s.state_sha()))
        if self.trace_every and self.s.n_steps % self.trace_every == 0:
            rec = dict(t=round(self.t, 4), row=int(self.s.n_steps), wall=round(time.time() - self.wall0, 1),
                       phys=round(getattr(self, "phys_s", 0.), 1), label=label, crossings=len(self.crossings()),
                       force={h: round(float(np.mean(self.hands[h]["window"])), 3) for h in self.sides},
                       held={h: (None if self.hands[h]["grasp"] is None else round(self.hands[h]["grasp"]["s"], 4))
                             for h in self.sides},
                       width={h: round(self.g[h], 4) for h in self.sides})
            try:
                rec["tcp_err_mm"] = {h: round(1000 * float(np.linalg.norm(self.hand_actual(h) - self.cmd[h]
                                                                          - self.bias[h])), 2) for h in self.sides}
                blocked = {h: self.arm_contacts(h) for h in self.sides if rec["tcp_err_mm"][h] > 8.}
                if any(blocked.values()):
                    rec["arm_contacts"] = blocked
            except Exception:  # noqa: BLE001
                pass
            self.trace.append(rec)
            if self.trace_path is not None:
                with self.trace_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")

    def arm_contacts(self, h, top=4):
        """The largest contacts between arm h's bodies and anything that is not that arm (diagnostics): [(own body,
        other body, normal force N, depth mm)]."""
        import mujoco
        m, d = self.m, self.d
        out = []
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
            n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2) or ""
            mine1, mine2 = n1.startswith(h + "/"), n2.startswith(h + "/")
            if mine1 == mine2:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            own, other = (n1, n2) if mine1 else (n2, n1)
            out.append((own.split("/", 1)[1], other, round(float(f6[0]), 2), round(-1000 * float(c.dist), 2)))
        out.sort(key=lambda x: -abs(x[2]))
        return out[:top]

    def wait(self, seconds, label="wait"):
        for _ in range(max(1, int(round(seconds / self.dt)))):
            self._step(label)

    # -------------------------------------------------------------- motion
    def set_orientation(self, h, R):
        """The orientation hand h takes on its next move (slerped along it)."""
        self._R_goal = getattr(self, "_R_goal", {})
        self._R_goal[h] = np.asarray(R, float)

    def solve_contact(self, h, c, R, seed=None, bias=True):
        """Joint targets putting h's pads' contact point at c (+ the arm's learned bias) with TCP rotation R.
        -> (q, pos err, rot err)."""
        p_tcp = np.asarray(c, float) + (self.bias[h] if bias else 0.) - np.asarray(R) @ CONTACT_IN_TCP_M
        return self.ik.solve(h, p_tcp, R, self.q[h] if seed is None else seed)

    def converge(self, h, *, tol=.0015, settle_speed=.004, max_s=.4, rounds=1, label="converge"):
        """Hold h still until its measured contact point settles (or max_s); if it rests more than `tol` (in plan) from
        the commanded one, add the plan error to the arm's bias and re-command (`rounds` times). Only x / y are
        learned: the height error differs between 5 cm up and at the table (where the fingertips may rest on it), and
        the pads span the rope's height anyway. An error over BLOCKED_M is learned too unless the arm touches something
        (the left arm folded near its base rests 10-13 mm off with no contact: hard v7, 10-11). -> the final error
        vector (m)."""
        target = self.cmd[h].copy()
        prev = self.hand_actual(h)
        err = prev - target
        for r in range(rounds + 1):
            for _ in range(max(1, int(round(max_s * self.converge_scale / self.dt)))):
                self._step(label)
                now = self.hand_actual(h)
                speed = float(np.linalg.norm(now - prev)) / self.dt
                prev = now
                err = now - target
                if speed < settle_speed:
                    break
            if float(np.linalg.norm(err[:2])) < tol or r == rounds:
                break
            if float(np.linalg.norm(err)) > BLOCKED_M and self.arm_contacts(h, top=1):
                break                                        # the arm is blocked (a contact), not offset: learn nothing
            b = self.bias[h] - err * np.array([1., 1., 0.])
            n = float(np.linalg.norm(b))
            self.bias[h] = b * min(1., BIAS_LIMIT_M / max(n, 1e-12))
            q, ep, er = self.solve_contact(h, target, self.R[h])
            if ep < .002 and er < .05 and not self.ik.wrist_unsafe(q):
                self.q[h] = q
        return err

    move_min_time, rot_speed, close_s = .2, 1.2, .3  # (10-14, opt-in --turbo: shorter; defaults as before)
    converge_scale = 1.                              # (10-14, opt-in --converge-scale: converge's max_s x this)

    def move(self, moves, speed=.05, min_time=None, label="move", tol=.002, tol_rot=None):
        """moves: {hand: contact point target (world)}; all hands together on min-jerk straight paths at <= speed (the
        lab's rule); a pending set_orientation is slerped along the path. A hand whose IK fails stops there
        (reach_stopped[h] = the target it could not reach). Returns the duration."""
        starts = {h: self.cmd[h].copy() for h in moves}
        goals = {h: np.asarray(moves[h], float).copy() for h in moves}
        for h in goals:
            goals[h][2] = max(goals[h][2], self.floor_z(h))
        Rgoal = getattr(self, "_R_goal", {})
        slerps = {}
        for h in moves:
            if h in Rgoal:
                slerps[h] = Slerp([0., 1.], Rotation.from_matrix(np.stack([self.R[h], Rgoal.pop(h)])))
        dist = max(float(np.linalg.norm(goals[h] - starts[h])) for h in moves)
        ang = max([float((slerps[h]([1.])[0] * Rotation.from_matrix(self.R[h]).inv()).magnitude()) for h in slerps] + [0.])
        T = max(self.move_min_time if min_time is None else min_time, 1.875 * dist / speed, 1.875 * ang / self.rot_speed)
        n = max(1, int(math.ceil(T / self.dt - 1e-9)))
        stopped = {}
        for h in moves:
            self.reach_stopped[h] = None
        for i in range(1, n + 1):
            u = minjerk(i / n)
            for h in moves:
                if h in stopped:
                    continue
                c = starts[h] + (goals[h] - starts[h]) * u
                R = slerps[h]([u])[0].as_matrix() if h in slerps else self.R[h]
                q, ep, er = self.solve_contact(h, c, R)
                self.ik_worst[h] = max(self.ik_worst[h], ep)
                if ep > tol or er > (tol_rot if tol_rot is not None else (.05 if tol <= .002 else .15)) \
                        or self.ik.wrist_unsafe(q):
                    stopped[h] = dict(target=goals[h].round(4).tolist(), at=self.cmd[h].round(4).tolist(),
                                      ik_err_mm=round(ep * 1000, 2), rot_err=round(er, 4),
                                      wrist_unsafe=self.ik.wrist_unsafe(q))
                    self.reach_stopped[h] = stopped[h]
                    self.events.append(dict(t=round(self.t, 4), row=int(self.s.n_steps), kind="reach_stop", hand=h,
                                            label=label, **stopped[h]))
                    self.say(f"{h}: reach stop ({label}) {stopped[h]}")
                    continue
                self.q[h], self.cmd[h], self.R[h] = q, c, R
            self._step(label)
        return n * self.dt

    def track(self, targets, label="track", tol=.002):
        """One row: {hand: contact point} at the hands' current orientations (a continuous motion's next sample).
        -> {hand: True if reached by IK, False if that hand stopped (reach_stopped set)}."""
        ok = {}
        for h, c in targets.items():
            c = np.asarray(c, float).copy()
            c[2] = max(c[2], self.floor_z(h))
            q, ep, er = self.solve_contact(h, c, self.R[h])
            self.ik_worst[h] = max(self.ik_worst[h], ep)
            if ep > tol or er > .05 or self.ik.wrist_unsafe(q):
                self.reach_stopped[h] = dict(target=c.round(4).tolist(), at=self.cmd[h].round(4).tolist(),
                                             ik_err_mm=round(ep * 1000, 2), rot_err=round(er, 4),
                                             wrist_unsafe=self.ik.wrist_unsafe(q))
                self.events.append(dict(t=round(self.t, 4), row=int(self.s.n_steps), kind="reach_stop", hand=h,
                                        label=label, **self.reach_stopped[h]))
                self.say(f"{h}: reach stop ({label}) {self.reach_stopped[h]}")
                ok[h] = False
                continue
            self.reach_stopped[h] = None
            self.q[h], self.cmd[h] = q, c
            ok[h] = True
        self._step(label)
        return ok

    def path(self, h, waypoints, speed=.05):
        for wpt in waypoints:
            self.move({h: np.asarray(wpt, float)}, speed)

    def joint_move(self, targets, seconds, label="joint_move"):
        """{hand: q6}: a min-jerk joint-space move (free motion away from the rope)."""
        starts = {h: self.q[h].copy() for h in targets}
        n = max(1, int(math.ceil(seconds / self.dt)))
        for i in range(1, n + 1):
            u = minjerk(i / n)
            for h, q in targets.items():
                self.q[h] = starts[h] + (np.asarray(q, float) - starts[h]) * u
                c, self.R[h] = self.ik.contact(h, self.q[h])
                self.cmd[h] = c - self.bias[h]          # the commanded point as solve_contact means it (bias added)
            self._step(label)

    def set_width(self, widths, seconds=.15, label="gripper"):
        """{hand: width}: ramp the gripper commands over `seconds` (at least one row)."""
        starts = {h: self.g[h] for h in widths}
        n = max(1, int(round(seconds / self.dt)))
        for i in range(1, n + 1):
            for h, wv in widths.items():
                self.g[h] = starts[h] + (float(wv) - starts[h]) * (i / n)
            self._step(label)

    # -------------------------------------------------------------- grasp
    def _bound_station(self, h):
        b = self.binding(h)
        if b is None:
            return None
        rope, link = b["target"][0], int(b["target"][1])
        rod = self.rods[rope]
        return dict(rope=rope, s=float(.5 * (rod.s_nodes[link] + rod.s_nodes[link + 1])), link=link)

    def grasp(self, h, rope, s, *, check_pinch=True, lift=0., wait_s=.25):
        """Close h's jaws where it is (its pads straddling the rope at s); True when the World bound a weld to `rope`."""
        clearance = self.pinch_clearance_at(rope, s)
        rec = dict(t=round(self.t, 4), row=int(self.s.n_steps), kind="grasp", hand=h, rope=rope, s=float(s),
                   pinch_clearance_m=round(float(clearance), 4), point_world=self.point(rope, s).round(5).tolist(),
                   tangent_world=self.tangent(rope, s).round(4).tolist())
        if check_pinch and clearance < self.pinch_clearance:
            rec["refused"] = "no room for a parallel gripper at this strand"
            self.events.append(rec)
            return False
        self.set_width({h: self.close_width}, self.close_s, label="close")
        n = max(1, int(round(wait_s / self.dt)))
        got = None
        for _ in range(n):
            got = self._bound_station(h)
            if got is not None and got["rope"] == rope:
                break
            self._step("squeeze")
        got = self._bound_station(h)
        if got is None or got["rope"] != rope:
            gm = self.w._grasp_manager
            rec["refused"] = f"no weld bound ({None if gm is None else gm.hands[h].get('rejection')})"
            self.events.append(rec)
            return False
        self.hands[h]["grasp"] = got
        self.hands[h]["force_peak"] = 0.
        rec.update(bound_s=round(got["s"], 4), bound_link=got["link"], ds_m=round(got["s"] - float(s), 4))
        self.events.append(rec)
        return True

    def press(self, h, rope, s, depth=.005, speed=.05):
        """A press pin: h's closed fingertips (already above the station) come down `depth` into the strand's top and
        hold it on the table by friction (no weld). -> True (the pin is marked held as a press)."""
        rod = self.rods[rope]
        p = self.point(rope, s)
        c = self.cmd[h].copy()
        c[:2] = p[:2]
        c[2] = p[2] + rod.radius - depth + self._tip_below_contact
        self.move({h: c}, speed, label="press", tol=.004)
        if self.reach_stopped[h]:            # the fingers did not get down there: no pin
            return False
        self.wait(.1, label="press")
        k = self.link_at(rope, s)
        self.hands[h]["grasp"] = dict(rope=rope, s=float(s), link=int(k), press=True)
        self.events.append(dict(t=round(self.t, 4), row=int(self.s.n_steps), kind="press", hand=h, rope=rope,
                                s=float(s), depth_m=depth, contacts=self.arm_contacts(h)))
        return True

    def release(self, h, reason="release", width=None, seconds=.1):
        """Open h's jaws (the GraspManager releases the weld on an opening command); a press pin is just let go."""
        held = self.hands[h]["grasp"]
        if held is not None and held.get("press"):
            self.events.append(dict(t=round(self.t, 4), row=int(self.s.n_steps), kind="unpress", hand=h, **held))
            self.hands[h]["grasp"] = None
            return
        self._releasing[h] = True
        try:
            self.set_width({h: RELEASE_W if width is None else width}, seconds, label="open")
            for _ in range(4):
                if self.binding(h) is None:
                    break
                self._step("open")
        finally:
            self._releasing[h] = False
        if held is not None:
            self.events.append(dict(t=round(self.t, 4), row=int(self.s.n_steps), kind=reason, hand=h, **held))
        self.hands[h]["grasp"] = None

    def place_hand(self, h, pos, quat):        # the lab teleports; a robot cannot (replay_old is lab-only)
        raise NotImplementedError("place_hand: robot hands do not teleport")
