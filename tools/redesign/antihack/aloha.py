"""ALOHA constants and a stateless TCP forward kinematics on a MuJoCo model (no Session, no physics).

The model is a frozen instance scene (`instance_assets/mujoco/`, loaded with topogym's own `load_scene`) or any MJCF
that carries the ALOHA's bodies `left/gripper_link`, `right/gripper_link` and joints `left/waist`, ... The TCP is the
robot description's tool point: `gripper_link` origin + R (0.13, 0, -0.003) (topogym/robot/aloha/description.py).

    from antihack.aloha import AlohaFK
    fk = AlohaFK.from_instance("/path/instance.json")         # or AlohaFK.from_scene_dir(dir) / AlohaFK(model)
    P, R = fk.tcp_path(rows, layout)                           # (T, 2, 3) TCP positions, (T, 2, 3, 3) rotations
    pose = fk.tcp({"left": q6, "right": q6})                   # {"left": (p, R), "right": (p, R)}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ARMS = ("left", "right")
ARM_JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
HAND_LINK = "gripper_link"
TCP_IN_HAND = np.array([0.13, 0.0, -0.003])
GRIPPER_OPEN_NATIVE, GRIPPER_CLOSED_NATIVE = 0.037, 0.002      # finger displacement along the rail (m)
JOINT_SPEED_LIMIT_RAD_S = float(np.pi)
TORQUE_CAPS_NM = (12.9, 25.8, 25.8, 12.9, 12.9, 4.8)
ARM_JOINT_LIMITS = ((-3.14158, 3.14158), (-1.85005, 1.25664), (-1.76278, 1.6057), (-3.14158, 3.14158),
                    (-1.8675, 2.23402), (-3.14158, 3.14158))
HOME = (0.0, -0.96, 1.16, 0.0, -0.3, 0.0)
# control periods of the three families on MuJoCo (rcb_topogym.TASKS): rope_threading 40 ms; knot / blanket 25 ms
CONTROL_DT_S = {"rope_threading": 0.04, "knot_untying": 0.025, "blanket_flatten": 0.025}


def layout_columns(layout) -> dict:
    """{arm: (arm column indices (6), gripper column index, gripper space "native" | "width")} of an action layout."""
    layout = [str(x) for x in layout]
    out = {}
    for arm in ARMS:
        cols = [layout.index(f"{arm}/{j}") for j in ARM_JOINTS]
        if f"{arm}_native_gripper" in layout:
            g, space = layout.index(f"{arm}_native_gripper"), "native"
        elif f"{arm}_gripper" in layout:
            g, space = layout.index(f"{arm}_gripper"), "width"
        else:
            g, space = None, None
        out[arm] = (cols, g, space)
    return out


def default_layout(space: str = "native") -> list:
    """The ALOHA action layout: per arm its six joints, then its gripper column."""
    g = "_native_gripper" if space == "native" else "_gripper"
    return [x for a in ARMS for x in ([f"{a}/{j}" for j in ARM_JOINTS] + [f"{a}{g}"])]


def scene_dir_of_instance(instance_json) -> Path:
    inst = json.loads(Path(instance_json).read_text())
    scene = (inst.get("simulator") or {}).get("mujoco_scene")
    if not scene:
        raise ValueError(f"{instance_json}: not a MuJoCo instance (simulator.mujoco_scene missing)")
    p = Path(scene)
    return p if p.is_absolute() else Path(instance_json).resolve().parent / p


def load_model(scene_dir, verify: bool = False):
    """(MjModel, manifest) of a frozen / built scene directory, through topogym's own loader (asset tokens resolved)."""
    from topogym.world.mujoco.scene import load_scene
    return load_scene(scene_dir, verify=verify)


class AlohaFK:
    """Stateless forward kinematics of both ALOHA arms on a private MjData of `model` (the scene's other bodies stay
    at qpos0). Thread-unsafe, cheap (one mj_kinematics per query)."""

    def __init__(self, model):
        import mujoco
        self.mj = mujoco
        self.m = model
        self.d = mujoco.MjData(model)
        self.qadr, self.hand, self.finger_qadr = {}, {}, {}
        for arm in ARMS:
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}/{j}") for j in ARM_JOINTS]
            if min(ids) < 0:
                raise ValueError(f"model lacks the ALOHA {arm} arm joints")
            self.qadr[arm] = np.array([int(model.jnt_qposadr[i]) for i in ids])
            self.hand[arm] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}/{HAND_LINK}")
            fq = []
            for f in ("left_finger", "right_finger"):
                j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}/{f}")
                if j >= 0:
                    fq.append(int(model.jnt_qposadr[j]))
            self.finger_qadr[arm] = np.array(fq, dtype=int)
        self.qpos0 = np.asarray(model.qpos0).copy()

    @classmethod
    def from_scene_dir(cls, scene_dir):
        model, _ = load_model(scene_dir)
        return cls(model)

    @classmethod
    def from_instance(cls, instance_json):
        return cls.from_scene_dir(scene_dir_of_instance(instance_json))

    def _set(self, q_arms: dict, fingers: dict | None = None):
        d = self.d
        d.qpos[:] = self.qpos0
        for arm, q in q_arms.items():
            d.qpos[self.qadr[arm]] = np.asarray(q, dtype=np.float64)[:6]
        for arm, f in (fingers or {}).items():
            if len(self.finger_qadr[arm]):
                d.qpos[self.finger_qadr[arm]] = float(f)
        self.mj.mj_kinematics(self.m, d)

    def tcp(self, q_arms: dict) -> dict:
        """{arm: (position (3), rotation (3, 3))} of the TCP for the given arm joint vectors."""
        self._set(q_arms)
        out = {}
        for arm in q_arms:
            p = np.asarray(self.d.xpos[self.hand[arm]]).copy()
            R = np.asarray(self.d.xmat[self.hand[arm]]).reshape(3, 3).copy()
            out[arm] = (p + R @ TCP_IN_HAND, R)
        return out

    def body_positions(self, q_arms: dict, names) -> dict:
        """World positions of named bodies (e.g. finger links) at the given arm joints."""
        self._set(q_arms)
        out = {}
        for n in names:
            b = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_BODY, n)
            out[n] = np.asarray(self.d.xpos[b]).copy()
        return out

    def tcp_path(self, rows, layout) -> tuple[np.ndarray, np.ndarray]:
        """TCP positions (T, 2, 3) and rotations (T, 2, 3, 3) of both arms (order ARMS) along action rows."""
        cols = layout_columns(layout)
        rows = np.asarray(rows, dtype=np.float64)
        P = np.zeros((len(rows), 2, 3))
        R = np.zeros((len(rows), 2, 3, 3))
        for t, row in enumerate(rows):
            res = self.tcp({arm: row[cols[arm][0]] for arm in ARMS})
            for k, arm in enumerate(ARMS):
                P[t, k], R[t, k] = res[arm]
        return P, R

    def ik(self, arm: str, pos, R=None, seed=None, iters: int = 200, tol_p: float = 1e-4, tol_r: float = 1e-3,
           prior=None, prior_weight: float = 1e-2):
        """Bounded least-squares IK (scipy trf, as rcb_topogym._Robot.ik) of one arm's TCP: position, and rotation if R
        is given (rotation residual weighted 0.2 m/rad), from `seed` (default HOME), within the joint limits. `prior`
        (q6) resolves the redundancy near the wrist singularity (forearm roll vs wrist rotate) toward a preferred
        configuration with a small weight. -> (q6, position error m, rotation error rad)."""
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        lo = np.array([l[0] for l in ARM_JOINT_LIMITS]); hi = np.array([l[1] for l in ARM_JOINT_LIMITS])
        q0 = np.clip(np.array(HOME if seed is None else seed, dtype=np.float64), lo + 1e-9, hi - 1e-9)
        pos = np.asarray(pos, dtype=np.float64)
        Rt = None if R is None else np.asarray(R, dtype=np.float64)
        pr = None if prior is None else np.asarray(prior, dtype=np.float64)

        def main(x):
            p, Rx = self.tcp({arm: x})[arm]
            e = p - pos
            if Rt is not None:
                e = np.r_[e, 0.2 * Rotation.from_matrix(Rt.T @ Rx).as_rotvec()]
            return e

        def resid(x):
            e = main(x)
            return e if pr is None else np.r_[e, prior_weight * (x - pr)]
        r = least_squares(resid, q0, bounds=(lo, hi), max_nfev=int(iters), diff_step=1e-6, xtol=1e-10, ftol=1e-10,
                          gtol=1e-10, method="trf")
        e = main(r.x)
        return r.x, float(np.linalg.norm(e[:3])), (float(np.linalg.norm(e[3:6]) / 0.2) if Rt is not None else 0.0)


def gripper_closedness(values, space: str) -> np.ndarray:
    """0 = fully open .. 1 = fully closed, from a gripper column (native finger displacement, or width = pad gap)."""
    v = np.asarray(values, dtype=np.float64)
    if space == "native":
        return np.clip((GRIPPER_OPEN_NATIVE - v) / (GRIPPER_OPEN_NATIVE - GRIPPER_CLOSED_NATIVE), 0.0, 1.0)
    # width space: the pad gap, about 0 closed .. ~0.07 open (2 x travel minus the pads); normalise by the open gap
    open_w = 2 * (GRIPPER_OPEN_NATIVE - GRIPPER_CLOSED_NATIVE)
    return np.clip(1.0 - v / open_w, 0.0, 1.0)
