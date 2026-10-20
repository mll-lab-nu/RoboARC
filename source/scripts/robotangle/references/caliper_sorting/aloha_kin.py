"""ALOHA two-arm kinematics for the bed-bolt design study (adapted from runs/cast_trinity_design_20261011/scripts/arms.py).

Frame = robot mount frame (project field env.robot_mount.pos): table top z = 0, x from the left arm base toward the right arm
base, y away from the observation cameras, z up.  Left base (-0.469, -0.019, 0), right base (+0.469, -0.019, 0) turned half a
circle about z.  World frame = mount frame + (-0.1356786, 0, 0.9196429).

TCP = site '<side>/gripper': x = approach direction, y = finger closing direction, z = the side the wrist camera sticks out to.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot

# port: the repository's ALOHA model (TOPOGYM_ASSET_ROOT, else assets/ under the working directory, the repository)
ALOHA = os.environ.get('ALOHA_XML') or os.path.join(os.environ.get('TOPOGYM_ASSET_ROOT', 'assets'), 'robots', 'aloha', 'aloha.xml')
ARM_J = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']
SIDE = {'L': 'left', 'R': 'right'}
MARGIN = 0.15            # joint distance kept from every limit (rad), project IK check
Q_FINGER_CLOSED = 0.0079  # finger joint value at which the finger pad reaches the TCP plane (hand.py, Trinity)
Q_FINGER_HOME = 0.0084
# observation cameras (configs/camera_configs/aloha_observation.yaml) moved into the mount frame
CAMS = {
    'overhead': dict(pos=(0.0, -0.303794, 1.0052400), lookat=(0.0, 0.118522, 0.0987915), fovy=58.008335),
    'worms_eye': dict(pos=(0.0, -0.377167, 0.0116055), lookat=(0.0, 0.618305, 0.1066658), fovy=58.008335),
}


def look_at_quat(pos, lookat, up=(0, 0, 1)):
    """MuJoCo camera looks along its -z axis with +y up."""
    f = np.asarray(lookat, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.c_[r, u, -f]
    q = Rot.from_matrix(R).as_quat()
    return [q[3], q[0], q[1], q[2]]


def rotvec(Rm):
    """Rotation matrix -> rotation vector (fast path; falls back to scipy near a half turn)."""
    c = (Rm[0, 0] + Rm[1, 1] + Rm[2, 2] - 1) / 2
    v = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    if c > 0.999999:
        return 0.5 * v
    if c < -0.99:
        return Rot.from_matrix(Rm).as_rotvec()
    ang = np.arccos(c)
    return ang / (2 * np.sin(ang)) * v


def frame(x, z_hint):
    """Rotation with first column x (approach) and third column as close to z_hint as possible."""
    x = np.asarray(x, float) / np.linalg.norm(x)
    z = np.asarray(z_hint, float) - np.dot(z_hint, x) * x
    z /= np.linalg.norm(z)
    return np.c_[x, np.cross(z, x), z]


class Robot:
    """Kinematic model: two arms + whatever static geoms `build(spec)` adds to the world body."""

    def __init__(self, build=None, w=960, h=720):
        spec = mujoco.MjSpec.from_file(ALOHA)
        for s in ('left', 'right'):
            spec.body(s + '/base_link').pos[2] = 0.0
        spec.worldbody.add_geom(name='table', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.75, 0.70, 0.01], pos=[0, 0, -0.01],
                                rgba=[0.78, 0.80, 0.82, 1], group=0)
        spec.add_texture(name='sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[0.97, 0.97, 0.98], rgb2=[0.80, 0.84, 0.88], width=256, height=256)
        spec.visual.headlight.ambient = [0.42, 0.42, 0.42]
        spec.visual.headlight.diffuse = [0.50, 0.50, 0.50]
        spec.visual.global_.offwidth = w
        spec.visual.global_.offheight = h
        for name, c in CAMS.items():
            spec.worldbody.add_camera(name=name, pos=list(c['pos']), quat=look_at_quat(c['pos'], c['lookat']), fovy=c['fovy'])
        if build is not None:
            build(spec)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        m = self.m
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.qadr = {s: np.array([m.jnt_qposadr[jid('%s/%s' % (SIDE[s], j))] for j in ARM_J]) for s in 'LR'}
        self.dadr = {s: np.array([m.jnt_dofadr[jid('%s/%s' % (SIDE[s], j))] for j in ARM_J]) for s in 'LR'}
        self.lim = {s: np.array([m.jnt_range[jid('%s/%s' % (SIDE[s], j))] for j in ARM_J]) for s in 'LR'}
        self.site = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, SIDE[s] + '/gripper') for s in 'LR'}
        self.fadr = {s: [m.jnt_qposadr[jid('%s/%s' % (SIDE[s], j))] for j in ('left_finger', 'right_finger')] for s in 'LR'}
        mujoco.mj_resetDataKeyframe(m, self.d, 0)
        self.q_home = {s: self.d.qpos[self.qadr[s]].copy() for s in 'LR'}
        self._jp = np.zeros((3, m.nv))
        self._jr = np.zeros((3, m.nv))
        # collision geoms of each arm (group 3 meshes), split into hand (gripper + fingers + wrist) and the rest
        self.arm_geoms = {'L': [], 'R': []}
        self.hand_geoms = {'L': [], 'R': []}
        self.finger_geoms = {'L': [], 'R': []}
        for g in range(m.ngeom):
            bn = m.body(m.geom_bodyid[g]).name
            if m.geom_group[g] != 3 or '/' not in bn or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            s = 'L' if bn.startswith('left') else 'R'
            if 'base_link' in bn or 'shoulder_link' in bn:
                continue
            self.arm_geoms[s].append(g)
            if 'gripper' in bn or 'finger' in bn or 'wrist' in bn:
                self.hand_geoms[s].append(g)
            if 'finger' in bn:
                self.finger_geoms[s].append(g)

    # ---------- kinematics
    def set_q(self, s, q, fingers=None):
        self.d.qpos[self.qadr[s]] = q
        if fingers is not None:
            for a in self.fadr[s]:
                self.d.qpos[a] = fingers

    def fk(self, s, q):
        self.d.qpos[self.qadr[s]] = q
        mujoco.mj_kinematics(self.m, self.d)
        T = np.eye(4)
        T[:3, :3] = self.d.site_xmat[self.site[s]].reshape(3, 3)
        T[:3, 3] = self.d.site_xpos[self.site[s]]
        return T

    def ik(self, s, T, q0, iters=150, margin=MARGIN):
        lo, hi = self.lim[s][:, 0] + margin, self.lim[s][:, 1] - margin
        q = np.clip(np.asarray(q0, float), lo, hi)
        da, sid = self.dadr[s], self.site[s]
        ep = er = np.ones(3)
        for _ in range(iters):
            self.d.qpos[self.qadr[s]] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            ep = T[:3, 3] - self.d.site_xpos[sid]
            er = rotvec(T[:3, :3] @ self.d.site_xmat[sid].reshape(3, 3).T)
            if np.linalg.norm(ep) < 2e-5 and np.linalg.norm(er) < 2e-4:
                break
            mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, sid)
            J = np.r_[self._jp[:, da], self._jr[:, da]]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(6), np.r_[ep, er])
            q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)
        return q, float(np.linalg.norm(ep)), float(np.linalg.norm(er))

    def ik_near(self, s, T, q_ref, q0=None, iters=40, margin=MARGIN, pos_tol=0.001, rot_tol=0.02, mu=1.0):
        """IK that stays near q_ref: minimises |pose error|^2 (position in units of pos_tol, rotation in units of rot_tol)
        + mu |q - q_ref|^2.  Near a wrist singularity (wrist_angle ~ 0, forearm_roll and wrist_rotate on one axis) a
        small rotation about the lost axis would need a large swing of both joints; this keeps the joints and leaves
        that small rotation error instead.  Starts from q0 (default q_ref)."""
        lo, hi = self.lim[s][:, 0] + margin, self.lim[s][:, 1] - margin
        q_ref = np.asarray(q_ref, float)
        q = np.clip(np.asarray(q_ref if q0 is None else q0, float), lo, hi)
        da, sid = self.dadr[s], self.site[s]
        w = np.r_[np.full(3, 1 / pos_tol), np.full(3, 1 / rot_tol)]
        for _ in range(iters):
            self.d.qpos[self.qadr[s]] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            e = np.r_[T[:3, 3] - self.d.site_xpos[sid], rotvec(T[:3, :3] @ self.d.site_xmat[sid].reshape(3, 3).T)]
            mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, sid)
            J = w[:, None] * np.r_[self._jp[:, da], self._jr[:, da]]
            dq = np.linalg.solve(J.T @ J + (mu + 1e-6) * np.eye(6), J.T @ (w * e) + mu * (q_ref - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)
            if np.abs(dq).max() < 1e-6:
                break
        self.d.qpos[self.qadr[s]] = q
        mujoco.mj_kinematics(self.m, self.d)
        ep = T[:3, 3] - self.d.site_xpos[sid]
        er = rotvec(T[:3, :3] @ self.d.site_xmat[sid].reshape(3, 3).T)
        return q, float(np.linalg.norm(ep)), float(np.linalg.norm(er))

    def margin(self, s, q):
        return float(np.minimum(q - self.lim[s][:, 0], self.lim[s][:, 1] - q).min())

    def ik_best(self, s, T, rng, n=16, q_prefer=None):
        """IK from home + n random seeds; returns (q, joint margin) of the solution with the largest joint margin, or None."""
        lo, hi = self.lim[s][:, 0] + MARGIN, self.lim[s][:, 1] - MARGIN
        seeds = [self.q_home[s]] + ([q_prefer] if q_prefer is not None else []) + [rng.uniform(lo, hi) for _ in range(n)]
        best = None
        for q0 in seeds:
            q, ep, er = self.ik(s, T, q0)
            if ep < 1e-3 and er < 1e-2:
                mg = self.margin(s, q)
                if best is None or mg > best[1]:
                    best = (q, mg)
        return best

    def ik_all(self, s, T, rng, n=40):
        """IK from home + n random seeds; distinct solutions (joint-space distance > 0.3 rad), each with its joint margin."""
        lo, hi = self.lim[s][:, 0] + MARGIN, self.lim[s][:, 1] - MARGIN
        sols = []
        for q0 in [self.q_home[s]] + [rng.uniform(lo, hi) for _ in range(n)]:
            q, ep, er = self.ik(s, T, q0, iters=200)
            if ep < 1e-3 and er < 1e-2 and all(np.abs(q - o).max() > 0.3 for o, _ in sols):
                sols.append((q, self.margin(s, q)))
        return sols

    def track(self, s, q0, A, B, n=None):
        """Follow the straight line + slerp from TCP pose A (where the arm is at q0) to B with warm-started IK;
        returns (q at B, worst position error, worst rotation error)."""
        from scipy.spatial.transform import Slerp
        n = n or max(4, int(np.ceil(max(np.linalg.norm(B[:3, 3] - A[:3, 3]) / 0.005,
                                         np.linalg.norm(rotvec(B[:3, :3] @ A[:3, :3].T)) / 0.03))))
        sl = Slerp([0, 1], Rot.from_matrix([A[:3, :3], B[:3, :3]]))
        q, wp, wr = np.asarray(q0, float), 0.0, 0.0
        for i in range(1, n + 1):
            u = i / n
            T = np.eye(4)
            T[:3, 3] = (1 - u) * A[:3, 3] + u * B[:3, 3]
            T[:3, :3] = sl(u).as_matrix()
            q, ep, er = self.ik(s, T, q)
            wp, wr = max(wp, ep), max(wr, er)
        return q, wp, wr

    # ---------- clearance (MuJoCo convex distance between collision geoms)
    def clearance(self, geoms_a, geoms_b, distmax=0.05):
        """Smallest signed distance between any geom of a and any geom of b (negative = penetration), capped at distmax."""
        best = distmax
        ft = np.zeros(6)
        for a in geoms_a:
            for b in geoms_b:
                dd = mujoco.mj_geomDistance(self.m, self.d, a, b, distmax, ft)
                if dd < best:
                    best = dd
        return float(best)
