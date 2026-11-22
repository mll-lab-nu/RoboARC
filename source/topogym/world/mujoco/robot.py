"""The joint-position robot on a MuJoCo scene: topogym.robot.joint_position.JointPositionRobot unchanged (validation,
targets, observation, checkpoints), driving the scene's PD actuators through the robot EntityView. Only reset_home
differs: Genesis' IK scratch preparation does not exist here."""
from __future__ import annotations

import numpy as np

from topogym.robot.joint_position import JointPositionRobot

# The real robots' arm joint speed limits (rad/s), per robot and arm joint. The task tooling's command path
# (rcb_topogym.Session.step) limits how fast the commanded arm joint targets move to these speeds (a target slew limit:
# clamp, never fail) and reports it as a robot-limit notice. Torque limits are the drives' force range, from the robot
# description (Panda: 87 N m on joints 1-4, 12 N m on joints 5-7, the datasheet values Menagerie's panda.xml uses).
# Franka Emika Panda datasheet: 2.175 rad/s on joints 1-4, 2.61 rad/s on joints 5-7. ALOHA: the Interbotix ViperX arm
# URDF's velocity limit, pi rad/s on all six arm joints.
JOINT_SPEED_LIMITS_RAD_S = {
    "panda_robotiq": (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61),
    "aloha": (np.pi,) * 6,
}
# The real robots' arm joint torque limits (N m), per robot and arm joint, on MuJoCo only: the task tooling's torque caps
# (rcb_topogym.ARM_TORQUE_CAPS_NM, both engines) become the smaller of the two (clamp, never fail). ALOHA: its ViperX
# 300 S servos at their 14.8 V stall torque, Dynamixel XM540-W270 12.9 N m (doubled on the dual-servo shoulder and
# elbow), XM430-W350 4.8 N m on the wrist rotate. The model's own drives reach 35 / 144 / 59 / 22 / 35 / 35 N m, and the
# Session's caps let an ALOHA hand push about 3x a real one (2026-10-09: 520 vs 180 N along the reach); the claws
# references need at most 6.5 N m on any joint. Genesis' ALOHA keeps the Session caps (its rope pinch needs them).
JOINT_TORQUE_LIMITS_NM = {
    "aloha": (12.9, 25.8, 25.8, 12.9, 12.9, 4.8),
}
# The real robots' Cartesian speed limits (m/s, rad/s) of the end-effector frame, per robot, on MuJoCo only: the task
# tooling's command path also shortens each arm's commanded joint step so its TCP (the description's grasp centre) moves
# at most this fast (clamp, never fail). Franka Emika Robot (Panda), Franka Control Interface documentation, "Control
# Interface Specification and Robot Limits", Limits for Franka Emika Robot (FER), Cartesian-space limits
# (https://frankarobotics.github.io/docs/robot_specifications.html): translation 1.7 m/s, rotation 2.5 rad/s, limits on
# the commanded end-effector pose O_T_EE. The joint-space limits above alone let a Robotiq pad reach 3.3 m/s (the
# validator's B2). ALOHA: no published Cartesian limit.
CARTESIAN_SPEED_LIMITS = {
    "panda_robotiq": (1.7, 2.5),
}
# The robots whose commanded targets may lead the measured arm only by what their drives can use (the task tooling's
# command path, rcb_topogym.Session._lead_bound; clamp, never fail). The Panda: its stiff position drives (kp 2000-4500,
# kv / kp = 0.1 s) store the offset of a blocked or lagging arm and release it as a catch-up burst well past the
# commanded speed (2026-10-10, buzz B3 and the GPU search's G5-G7: 2.5-4.8 m/s at the TCP behind a 1.7 m/s command,
# through the 1.47 mm wire). The ALOHA's servos are too weak to store such a burst (its fastest catch-up, the wrist
# rotate at 4.0 rad/s, stays below the XM430-W350's no-load speed): its command path is unchanged.
COMMAND_LEAD_BOUND = ("panda_robotiq",)
# The robots' internal linkage bodies whose collision geoms touch without friction, per robot: the scene gives them
# condim 1 and contact priority 1, so their own frictionless contact wins against any other geom. The Robotiq 2F-85's
# spring links and drivers are bars of its four-bar, not gripping surfaces. With Genesis' armature (0.1 kg m^2) on the
# four-bar's passive joints, the friction rows of such a link's contact were regularised by an approximate constraint
# diagonal far from the true one, and returned energy (2026-10-10, buzz G8 / G9: a spring link pressed on the static
# wire base, +0.67 / +3.68 J; G10: a driver on the wire, +3.27 J; BREAKDOWN). The followers carry the pads and the
# couplers hold what is gripped: both keep their friction. Without the armature
# the grasp changes (9 of the 13 Panda references fail); these frictionless links change no reference.
FRICTIONLESS_LINKS = {
    "panda_robotiq": tuple(f"r85_{side}_{part}" for side in ("left", "right") for part in ("spring_link", "driver")),
}
# The robots' fingertip pads, per robot: gripper -> finger link -> its pad geoms. The menagerie ALOHA's pads are three
# r 0.6 mm spheres per finger (aloha.xml's sphere_collision), their centres 0.21 mm inside the finger's convex hull, so
# they stand 0.39 mm proud of it. MujocoWorld.build gives each gripper's pads and finger hulls their own collision bits
# (separate_finger_pads): a pad meets the other finger's pads, and everything else exactly as before, but never the other
# finger's hull; hull meets hull. An empty close rests pad on pad (0.17 mm), 0.21 mm short of any pad-hull contact, so
# no reference changes. Pressed closed by an outside load (blanket BL1: an arm folded onto its own base, 50-190 N on a
# finger's slide), the fingers drove a pad past its centre into the other finger's hull, where that contact took a side
# face's normal almost perpendicular to the slides (A_ii 0.003-0.007 against 3.3); on the exact constraint diagonal its
# regulariser shrank with A_ii, and three such rows locked the gripper at 15-41 kN (docs/rcb/MUJOCO_CORE.md).
# Robot body pairs that never collide, per robot, added to every loaded scene (MujocoWorld.build -> scene.load_scene,
# frozen scenes included). The ALOHA's lower forearm and its own gripper base are two joints apart (wrist angle, wrist
# rotate). Their collision hulls (the gripper's wrist mount and the forearm) overlap only past wrist_angle 116.5 deg with
# wrist_rotate within about 15 deg of 180 deg: up to 11.9 mm at the ViperX's 123 deg and 18.4 mm at the model's 128 deg
# limit. With only those two joints between them, such a contact's normal can lie almost perpendicular to every motion
# they have; the exact diagonal then gives it almost no regulariser (validator CL8t: 2.2-2.4 kN on 0.1 mm, A_ii
# 0.005-0.008, one substep). No reference comes within 15 mm of it (every ALOHA -i01 reference's commanded joints), so
# the exclusion changes none; a joint-range guard instead would cut valid motions.
SELF_CONTACT_EXCLUDES = {
    "aloha": tuple((f"{arm}/lower_forearm_link", f"{arm}/gripper_base") for arm in ("left", "right")),
}
# The robots' joint spring references (rad), per robot, set on every loaded model (MujocoWorld.build, frozen scenes
# included; the scene files are unchanged). The Robotiq 2F-85's spring links keep the MJCF's own springref 2.62
# (2026-10-11). The scenes had imported them about 0 (as Genesis does: it reads a joint's stiffness, not its springref),
# where the 0.05 N m/rad springs no longer hold the followers back: the fingertips curl inward, the pads close in a V and
# touch only at their tips, and a grasp weld then carried buzz's wand with no finger on it for 41-65 % of its weld time
# (fiber's sheath: the distal pad only). At 2.62 the pads close parallel and both touch the held body for the whole
# weld (docs/rcb/MUJOCO_CORE.md, "The Robotiq's spring links").
SPRING_REFS = {
    "panda_robotiq": {"r85_left_spring_link_joint": 2.62, "r85_right_spring_link_joint": 2.62},
}


def set_spring_refs(m, robot_id: str) -> int:
    """The robot's SPRING_REFS on a compiled model (model.qpos_spring). Returns the number of joints set."""
    import mujoco
    refs = SPRING_REFS.get(robot_id, {})
    for name, value in refs.items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise RuntimeError(f"spring references: no joint {name}")
        m.qpos_spring[int(m.jnt_qposadr[j])] = value
    return len(refs)


FINGER_PADS = {
    "aloha": {arm: {f"{arm}/{side}_finger_link": tuple(f"{arm}/{side}_g{k}" for k in range(3)) for side in ("left", "right")}
              for arm in ("left", "right")},
}


def separate_finger_pads(m, robot_id: str) -> dict | None:
    """Fresh collision bits for each gripper of the robot (FINGER_PADS): one for its pads, one for its finger hulls (the
    finger links' other colliding geoms). Contype is that bit; conaffinity is it, plus every contype bit of the geoms and
    flexes the finger geoms collided with, plus the other grippers' bits. Then a pad meets the other finger's pads, a hull
    the other finger's hull, and everything else as its contype / conaffinity said; only pad <-> other-finger hull is off.
    Checked against the original bits for every finger geom x every geom and flex: refused (RuntimeError) unless exactly
    those pairs change. Sets the finger bodies' body_contype / body_conaffinity to match. Returns {gripper: (pad bit,
    hull bit)}, or None when the robot has no pads."""
    grippers = FINGER_PADS.get(robot_id)
    if not grippers:
        return None
    import mujoco
    obj = mujoco.mjtObj
    ct0, ca0 = m.geom_contype.astype(np.int64), m.geom_conaffinity.astype(np.int64)
    fct, fca = m.flex_contype.astype(np.int64), m.flex_conaffinity.astype(np.int64)
    used = int(np.bitwise_or.reduce(np.concatenate([ct0, ca0, fct, fca, [0]])))
    free = [1 << k for k in range(31) if not used >> k & 1]
    if len(free) < 2 * len(grippers):
        raise RuntimeError(f"finger pads: {len(free)} free collision bits for {len(grippers)} grippers")
    finger = np.full(m.ngeom, -1)          # gripper index of a finger geom
    is_pad = np.zeros(m.ngeom, bool)
    body_of = np.asarray(m.geom_bodyid)
    plan, bodies = {}, {}
    for k, (gripper, fingers) in enumerate(sorted(grippers.items())):
        plan[gripper] = (free[2 * k], free[2 * k + 1])
        for body, pads in fingers.items():
            b = mujoco.mj_name2id(m, obj.mjOBJ_BODY, body)
            if b < 0:
                raise RuntimeError(f"finger pads: no body {body}")
            bodies[b] = k
            for name in pads:
                g = mujoco.mj_name2id(m, obj.mjOBJ_GEOM, name)
                if g < 0 or int(body_of[g]) != b:
                    raise RuntimeError(f"finger pads: no pad geom {name} on {body}")
                is_pad[g] = True
    for g in range(m.ngeom):
        if int(body_of[g]) in bodies and (ct0[g] or ca0[g]):
            finger[g] = bodies[int(body_of[g])]
    F = np.flatnonzero(finger >= 0)
    collide = lambda c1, a1, c2, a2: ((c1 & a2) | (c2 & a1)) != 0        # noqa: E731
    others = finger < 0
    E = 0
    for f in F:
        E |= int(np.bitwise_or.reduce(np.where(others & collide(ct0[f], ca0[f], ct0, ca0), ct0, 0)))
        E |= int(np.bitwise_or.reduce(np.where(collide(ct0[f], ca0[f], fct, fca), fct, 0))) if m.nflex else 0
    allbits = 0
    for p, h in plan.values():
        allbits |= p | h
    ct, ca = ct0.copy(), ca0.copy()
    for f in F:
        p, h = plan[sorted(grippers)[finger[f]]]
        ct[f] = p if is_pad[f] else h
        ca[f] = ct[f] | E | (allbits & ~(p | h))
    for f in F:                             # the check: only pad <-> other-finger hull of the same gripper changes
        old, new = collide(ct0[f], ca0[f], ct0, ca0), collide(ct[f], ca[f], ct, ca)
        same_body = body_of == body_of[f]
        meant = (finger == finger[f]) & ~same_body & (is_pad != is_pad[f])
        if np.any((old != new) & ~meant & ~same_body) or np.any(new & meant):
            raise RuntimeError(f"finger pads: the bit plan would change other collisions of geom {f}")
        if m.nflex and np.any(collide(ct0[f], ca0[f], fct, fca) != collide(ct[f], ca[f], fct, fca)):
            raise RuntimeError(f"finger pads: the bit plan would change flex collisions of geom {f}")
    m.geom_contype[F] = ct[F]
    m.geom_conaffinity[F] = ca[F]
    for b in bodies:
        own = np.flatnonzero(body_of == b)
        m.body_contype[b] = int(np.bitwise_or.reduce(np.asarray(m.geom_contype[own], dtype=np.int64)))
        m.body_conaffinity[b] = int(np.bitwise_or.reduce(np.asarray(m.geom_conaffinity[own], dtype=np.int64)))
    return plan


class MujocoRobot(JointPositionRobot):
    def __init__(self, entity, config):
        super().__init__(entity, config)
        limits = JOINT_SPEED_LIMITS_RAD_S.get(self.description.robot_id)
        # arm -> per arm joint speed limit (rad/s); None: no slew limit
        self.joint_speed_limits = None if limits is None else {
            name: np.asarray(limits, dtype=np.float64) for name in self.groups}
        if limits is not None and any(len(limits) != len(g.arm_joints) for g in self.groups.values()):
            raise ValueError(f"{self.description.robot_id}: joint speed limits do not match the arm joints")
        cartesian = CARTESIAN_SPEED_LIMITS.get(self.description.robot_id)
        # arm -> (TCP linear m/s, angular rad/s) speed limit; None: no Cartesian limit
        self.cartesian_speed_limits = None if cartesian is None else {name: tuple(map(float, cartesian)) for name in self.groups}
        # True: the command path bounds the commanded targets' lead over the measured arm
        self.command_lead_bound = self.description.robot_id in COMMAND_LEAD_BOUND
        torques = JOINT_TORQUE_LIMITS_NM.get(self.description.robot_id)
        # arm -> per arm joint torque limit (N m); None: the Session's caps alone
        self.joint_torque_limits = None if torques is None else {
            name: np.asarray(torques, dtype=np.float64) for name in self.groups}
        if torques is not None and any(len(torques) != len(g.arm_joints) for g in self.groups.values()):
            raise ValueError(f"{self.description.robot_id}: joint torque limits do not match the arm joints")

    def reset_home(self):
        for name, g in self.groups.items():
            self.entity.set_dofs_position(np.asarray(g.home), self._indices[name], zero_velocity=True)
            n = len(g.arm_joints)
            self._targets[name] = dict(joint_position_rad=list(g.home[:n]), **{g.gripper_field: float(g.home[n])})
        self.apply_action({})


def make_robot(entity, config):
    return MujocoRobot(entity, config)
