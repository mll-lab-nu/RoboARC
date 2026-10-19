"""Entity, link, joint and geometry views of a MuJoCo scene, with the attribute and method names of the Genesis entity
API subset that the rest of TopoGym and the task tooling use (docs/rcb/MUJOCO_CORE.md lists it). Indices:

- link ``idx``: the MuJoCo body id (unique in the scene, as Genesis' global link index); ``parent_idx`` -1 at an entity root;
- joint ``qs_idx_local`` / ``dofs_idx_local``: positions in the entity's own q / dof vectors (get_qpos, get_dofs_velocity).

Poses are body (authored link) frames, wxyz; velocities are world-frame, the linear one at the body origin (Genesis'
get_links_vel default, "link_origin"). The world's endpoint kinematics are refreshed after every tick, so these views read
the state at the end of the last tick.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def _mj():
    import mujoco
    return mujoco


class GeomView:
    """One collision (or visual) geom: vertices in its own frame (init_verts / init_faces), its pose in the link frame
    (init_pos / init_quat, wxyz) and its world vertices now (get_verts)."""

    def __init__(self, world, gid: int):
        from .scene import _primitive_mesh
        mj = _mj()
        self.world, self.idx = world, int(gid)
        m = world.model
        kind = int(m.geom_type[gid])
        if kind == mj.mjtGeom.mjGEOM_MESH:
            mesh = int(m.geom_dataid[gid])
            a, n = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
            fa, fn = int(m.mesh_faceadr[mesh]), int(m.mesh_facenum[mesh])
            self.init_verts = np.array(m.mesh_vert[a:a + n], dtype=np.float64)
            self.init_faces = np.array(m.mesh_face[fa:fa + fn], dtype=np.int64)
            self.kind = "mesh"
        elif kind == mj.mjtGeom.mjGEOM_PLANE:            # the ground: no vertices
            self.kind = "plane"
            self.init_verts, self.init_faces = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
        else:
            names = {mj.mjtGeom.mjGEOM_BOX: "box", mj.mjtGeom.mjGEOM_SPHERE: "sphere",
                     mj.mjtGeom.mjGEOM_CYLINDER: "cylinder", mj.mjtGeom.mjGEOM_CAPSULE: "capsule"}
            if kind not in names:
                raise NotImplementedError(f"geom type {kind} has no vertex view")
            self.kind = names[kind]
            self.init_verts, self.init_faces = _primitive_mesh(self.kind, np.array(m.geom_size[gid]))
        self.init_pos = np.array(m.geom_pos[gid], dtype=np.float64)
        self.init_quat = np.array(m.geom_quat[gid], dtype=np.float64)
        self.size = np.array(m.geom_size[gid], dtype=np.float64)

    @property
    def mesh(self):
        """Genesis' geom.mesh.trimesh: this geom's triangles in its own frame (planning tools' collision checks)."""
        import trimesh
        return SimpleNamespace(trimesh=trimesh.Trimesh(self.init_verts, self.init_faces, process=False))

    # Genesis' geom type / data / convexity, as the planning tools read them: a mesh
    # collides as its convex hull here; primitive data are Genesis' full extents (box sides, cylinder / capsule length).
    @property
    def type(self):
        return SimpleNamespace(name="PLANE" if self.kind == "plane" else self.kind.upper())

    @property
    def data(self):
        size = self.size
        return {"box": 2 * size[:3], "sphere": size[:1], "cylinder": np.array([size[0], 2 * size[1]]),
                "capsule": np.array([size[0], 2 * size[1]]), "plane": np.array([0., 0., 1.]),
                "mesh": np.zeros(0)}[self.kind]

    is_convex = True

    def world_pose(self):
        """(position, scipy Rotation) of this geom in the world now."""
        from scipy.spatial.transform import Rotation
        d = self.world.data
        return np.array(d.geom_xpos[self.idx]), Rotation.from_matrix(np.asarray(d.geom_xmat[self.idx]).reshape(3, 3))

    def get_verts(self) -> np.ndarray:
        d = self.world.data
        return self.init_verts @ np.asarray(d.geom_xmat[self.idx]).reshape(3, 3).T + d.geom_xpos[self.idx]

    def world_aabb(self) -> np.ndarray:
        """[[min], [max]] of this geom in the world: exact for meshes, boxes and spheres; conservative otherwise."""
        d = self.world.data
        if self.kind == "sphere":
            c = np.asarray(d.geom_xpos[self.idx])
            return np.stack([c - self.size[0], c + self.size[0]])
        if self.kind == "mesh" or self.kind == "box":
            v = self.get_verts()
            return np.stack([v.min(axis=0), v.max(axis=0)])
        R = np.asarray(d.geom_xmat[self.idx]).reshape(3, 3)
        half = np.array([self.size[0], self.size[0], self.size[1] + (self.size[0] if self.kind == "capsule" else 0.)])
        extent = np.abs(R) @ half
        c = np.asarray(d.geom_xpos[self.idx])
        return np.stack([c - extent, c + extent])


class LinkView:
    def __init__(self, world, entity, name: str, body: int, parent_idx: int, idx_local: int):
        self.world, self.entity, self.name = world, entity, name
        self.idx, self.parent_idx, self.idx_local = int(body), int(parent_idx), int(idx_local)
        m = world.model
        mj = _mj()
        own = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == self.idx]
        collide = [g for g in own if int(m.geom_contype[g]) or int(m.geom_conaffinity[g])]
        self._collision_ids = collide
        self._visual_ids = [g for g in own if g not in collide and int(m.geom_type[g]) == mj.mjtGeom.mjGEOM_MESH]
        self._geoms = self._vgeoms = None
        # Fixed to the world: no joint on the path from this body to the world body.
        b, fixed = self.idx, True
        while b > 0:
            if int(m.body_jntnum[b]):
                fixed = False
                break
            b = int(m.body_parentid[b])
        self.is_fixed = fixed

    @property
    def geoms(self) -> list:
        if self._geoms is None:
            self._geoms = [GeomView(self.world, g) for g in self._collision_ids]
        return self._geoms

    @property
    def vgeoms(self) -> list:
        if self._vgeoms is None:
            self._vgeoms = [GeomView(self.world, g) for g in self._visual_ids]
        return self._vgeoms

    def get_pos(self) -> np.ndarray:
        return np.array(self.world.data.xpos[self.idx])

    def get_quat(self) -> np.ndarray:
        return np.array(self.world.data.xquat[self.idx])

    def get_vel(self) -> np.ndarray:
        return self.world.body_velocities()[1][self.idx].copy()

    def get_ang(self) -> np.ndarray:
        return self.world.body_velocities()[0][self.idx].copy()


class JointView:
    def __init__(self, name: str, mj_id: int, qs: list, dofs: list, kind: str):
        self.name, self.mj_id, self.qs_idx_local, self.dofs_idx_local, self.type = name, int(mj_id), qs, dofs, kind


class EntityView:
    """One scene entity (the robot, the table, a task body): its links, joints and the Genesis-named accessors."""

    def __init__(self, world, name: str, record: dict, *, fixed: bool):
        mj = _mj()
        m = world.model
        self.world, self.name, self.record = world, name, record
        self.morph = SimpleNamespace(fixed=bool(fixed))
        self.material = SimpleNamespace(coup_type=None)
        self.links, self._link_by_name = [], {}
        bodies = {}
        for i, link in enumerate(record["links"]):
            body = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, link["body"])
            if body < 0:
                raise ValueError(f"scene lacks body {link['body']} of {name}")
            bodies[link["name"]] = body
        for i, link in enumerate(record["links"]):
            parent = bodies[link["parent"]] if link.get("parent") is not None else -1
            view = LinkView(world, self, link["name"], bodies[link["name"]], parent, i)
            self.links.append(view)
            self._link_by_name[link["name"]] = view
        self.base_link = self.links[0]
        self.base_link_idx = self.base_link.idx
        # joints: the free root joint first (7 q / 6 dofs), then the scalar joints in the scene's order
        self.joints, self._joint_by_name = [], {}
        self._qadr, self._dadr = [], []
        free = record.get("free_joint")
        specs = ([dict(name="root", mj=free)] if free else []) + list(record["joints"])
        for spec in specs:
            jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, spec["mj"])
            if jid < 0:
                raise ValueError(f"scene lacks joint {spec['mj']} of {name}")
            kind = int(m.jnt_type[jid])
            nq, nv = {mj.mjtJoint.mjJNT_FREE: (7, 6), mj.mjtJoint.mjJNT_BALL: (4, 3)}.get(kind, (1, 1))
            qa, da = int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])
            qs = list(range(len(self._qadr), len(self._qadr) + nq))
            dofs = list(range(len(self._dadr), len(self._dadr) + nv))
            self._qadr += list(range(qa, qa + nq))
            self._dadr += list(range(da, da + nv))
            label = {mj.mjtJoint.mjJNT_FREE: "free", mj.mjtJoint.mjJNT_BALL: "ball",
                     mj.mjtJoint.mjJNT_SLIDE: "prismatic", mj.mjtJoint.mjJNT_HINGE: "revolute"}[kind]
            view = JointView(spec["name"], jid, qs, dofs, label)
            self.joints.append(view)
            self._joint_by_name[spec["name"]] = view
        self._qadr = np.asarray(self._qadr, dtype=np.int64)
        self._dadr = np.asarray(self._dadr, dtype=np.int64)
        self.n_qs, self.n_dofs = len(self._qadr), len(self._dadr)
        self.q_start = int(self._qadr[0]) if self.n_qs else 0
        self.dof_start = int(self._dadr[0]) if self.n_dofs else 0
        # actuators per entity-local dof (the robot's drives)
        self._actuator_of_dof = {}
        dof_local = {int(a): i for i, a in enumerate(self._dadr)}
        for a in range(m.nu):
            if int(m.actuator_trntype[a]) != mj.mjtTrn.mjTRN_JOINT:
                continue
            j = int(m.actuator_trnid[a, 0])
            local = dof_local.get(int(m.jnt_dofadr[j]))
            if local is not None:
                self._actuator_of_dof[local] = a
        self._fk = None

    # -- lookups
    def get_link(self, name: str) -> LinkView:
        try:
            return self._link_by_name[name]
        except KeyError:
            raise KeyError(f"{self.name} has no link {name!r}") from None

    def get_joint(self, name: str) -> JointView:
        try:
            return self._joint_by_name[name]
        except KeyError:
            raise KeyError(f"{self.name} has no joint {name!r}") from None

    # -- state
    def get_qpos(self) -> np.ndarray:
        return np.array(self.world.data.qpos[self._qadr], dtype=np.float64)

    def get_dofs_velocity(self) -> np.ndarray:
        return np.array(self.world.data.qvel[self._dadr], dtype=np.float64)

    def get_dofs_control_force(self) -> np.ndarray:
        """The force each drive applied in the tick's last physics substep (its PD output after the force clamp), per
        entity dof; 0 for dofs without a drive."""
        return np.array(self.world.qfrc_actuator()[self._dadr], dtype=np.float64)

    def get_pos(self) -> np.ndarray:
        return self.base_link.get_pos()

    def get_quat(self) -> np.ndarray:
        return self.base_link.get_quat()

    def set_dofs_position(self, values, dofs_idx_local=None, zero_velocity=True):
        """Scalar joints only (as the controller and the world's initial joint positions use it)."""
        dofs = range(self.n_dofs) if dofs_idx_local is None else [int(i) for i in np.asarray(dofs_idx_local).reshape(-1)]
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        d, q_of_dof = self.world.data, self._q_of_dof()
        for value, i in zip(values, dofs, strict=True):
            d.qpos[self._qadr[q_of_dof[i]]] = value
            if zero_velocity:
                d.qvel[self._dadr[i]] = 0.
        self.world._state_changed()

    def _q_of_dof(self) -> dict:
        out = {}
        for j in self.joints:
            if len(j.qs_idx_local) == 1:
                out[j.dofs_idx_local[0]] = j.qs_idx_local[0]
        return out

    # -- drives
    def _actuators(self, dofs) -> list:
        dofs = [int(i) for i in np.asarray(dofs).reshape(-1)]
        missing = [i for i in dofs if i not in self._actuator_of_dof]
        if missing:
            raise ValueError(f"{self.name}: dofs {missing} have no drive in the scene")
        return [self._actuator_of_dof[i] for i in dofs]

    def set_dofs_kp(self, kp, dofs_idx_local):
        m = self.world.model
        for a, k in zip(self._actuators(dofs_idx_local), np.asarray(kp, dtype=np.float64).reshape(-1), strict=True):
            m.actuator_gainprm[a, 0] = k
            m.actuator_biasprm[a, 1] = -k

    def set_dofs_kv(self, kv, dofs_idx_local):
        m = self.world.model
        for a, k in zip(self._actuators(dofs_idx_local), np.asarray(kv, dtype=np.float64).reshape(-1), strict=True):
            m.actuator_biasprm[a, 2] = -k

    def set_dofs_force_range(self, lower, upper, dofs_idx_local):
        m = self.world.model
        lower, upper = np.asarray(lower, dtype=np.float64).reshape(-1), np.asarray(upper, dtype=np.float64).reshape(-1)
        for a, lo, hi in zip(self._actuators(dofs_idx_local), lower, upper, strict=True):
            m.actuator_forcerange[a] = (lo, hi)
            m.actuator_forcelimited[a] = 1

    def get_dofs_force_range(self, dofs_idx_local):
        m = self.world.model
        a = self._actuators(dofs_idx_local)
        return np.array(m.actuator_forcerange[a, 0]), np.array(m.actuator_forcerange[a, 1])

    def control_dofs_position(self, targets, dofs_idx_local):
        d = self.world.data
        for a, t in zip(self._actuators(dofs_idx_local), np.asarray(targets, dtype=np.float64).reshape(-1), strict=True):
            d.ctrl[a] = t

    # -- geometry
    def get_AABB(self) -> np.ndarray:
        boxes = [g.world_aabb() for link in self.links for g in link.geoms]
        if not boxes:
            raise ValueError(f"{self.name} has no collision geometry")
        boxes = np.asarray(boxes)
        return np.stack([boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)])

    def get_vAABB(self) -> np.ndarray:
        boxes = [g.world_aabb() for link in self.links for g in link.vgeoms] or \
                [g.world_aabb() for link in self.links for g in link.geoms]
        if not boxes:
            raise ValueError(f"{self.name} has no geometry")
        boxes = np.asarray(boxes)
        return np.stack([boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)])

    # -- stateless kinematics
    def chain_fk(self):
        """A callable q_local -> {link name: 4x4 world pose}: forward kinematics of this entity on a private MjData (the
        live simulation is never touched). Joints of other entities stay at the model's reference configuration."""
        if self._fk is None:
            self._fk = _ChainFK(self)
        return self._fk

    def forward_kinematics(self, qpos):
        """Genesis' entity.forward_kinematics: (positions (n_links, 3), quaternions wxyz (n_links, 4)) of this entity's
        links (idx_local order) for its q vector, on the private MjData of chain_fk."""
        fk = self.chain_fk()
        fk(qpos)
        d = fk.data
        bodies = [link.idx for link in self.links]
        return np.array(d.xpos[bodies]), np.array(d.xquat[bodies])

    def inverse_kinematics(self, link, pos=None, quat=None, init_qpos=None, max_samples=50, max_solver_iters=20,
                           damping=.01, pos_tol=5e-4, rot_tol=5e-3, max_step_size=.5, dofs_idx_local=None,
                           respect_joint_limit=True, **options):
        """Genesis 1.2.1's entity.inverse_kinematics for one link, as its kernel computes it
        (genesis/engine/solvers/rigid/abd/inverse_kinematics.py), on the private MjData of chain_fk (the live simulation
        is never touched): from init_qpos, at most max_solver_iters damped least-squares steps over the given dofs
        (default all) until the world-frame position error <= pos_tol and the rotation vector of target * current^-1 <=
        rot_tol; the Jacobian at the link origin (MuJoCo's mj_jacBody: world frame), step J^T (J J^T + damping^2 I)^-1 e,
        each dof's step clamped to +-max_step_size, joint ranges enforced after every step; damping, tolerances and step
        size as float32 (the kernel's argument types). Deterministic: one sample only (Genesis restarts from random joint
        values when the first sample fails and max_samples > 1; its first sample is this one). Returns the entity's q
        vector; callers check the residual (topogym.solvers.genesis_kinematics.GenesisIKSolver.solve_checked)."""
        if any(v is not None and k in ("local_point", "envs_idx") for k, v in options.items()) or options.get("return_error") \
                or not all(options.get("pos_mask", (1, 1, 1))) or not all(options.get("rot_mask", (1, 1, 1))):
            raise NotImplementedError(f"MuJoCo inverse_kinematics: options {sorted(options)} are not ported")
        mj = _mj()
        f32 = lambda x: float(np.float32(x))       # noqa: E731
        damping, pos_tol, rot_tol, max_step_size = f32(damping), f32(pos_tol), f32(rot_tol), f32(max_step_size)
        m, fk = self.world.model, self.chain_fk()
        d, b = fk.data, int(link.idx)
        q = np.array(self.get_qpos() if init_qpos is None else init_qpos, dtype=np.float64).reshape(-1)
        dofs = list(range(self.n_dofs)) if dofs_idx_local is None else [int(i) for i in np.asarray(dofs_idx_local).reshape(-1)]
        q_of_dof = self._q_of_dof()
        qi = np.array([q_of_dof[i] for i in dofs])
        cols = self._dadr[dofs]
        joint = np.array([int(m.dof_jntid[c]) for c in cols])
        lo, hi = np.array(m.jnt_range[joint, 0]), np.array(m.jnt_range[joint, 1])
        limited = np.array(m.jnt_limited[joint], dtype=bool) & bool(respect_joint_limit)
        target_p = None if pos is None else np.asarray(pos, dtype=np.float64).reshape(3)
        target_q = None if quat is None else np.asarray(quat, dtype=np.float64).reshape(4)
        jacp, jacr, inv, err_q = np.zeros((3, m.nv)), np.zeros((3, m.nv)), np.zeros(4), np.zeros(4)
        for _ in range(int(max_solver_iters)):
            fk(q)
            e_p = np.zeros(3) if target_p is None else target_p - d.xpos[b]
            e_r = np.zeros(3)
            if target_q is not None:
                mj.mju_negQuat(inv, d.xquat[b])
                mj.mju_mulQuat(err_q, target_q, inv)
                err_q /= np.linalg.norm(err_q)
                s = float(np.linalg.norm(err_q[1:]))
                e_r = err_q[1:] * (np.sign(err_q[0]) or 1.) * (2. * np.arctan2(s, abs(err_q[0])) / s) if s > 1e-15 else err_q[1:].copy()
            if np.linalg.norm(e_p) <= pos_tol and np.linalg.norm(e_r) <= rot_tol:
                break
            mj.mj_comPos(m, d)
            mj.mj_jacBody(m, d, jacp, jacr, b)
            J = np.vstack([jacp[:, cols] * (target_p is not None), jacr[:, cols] * (target_q is not None)])
            step = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), np.r_[e_p, e_r])
            q[qi] += np.clip(step, -max_step_size, max_step_size)
            q[qi] = np.where(limited, np.clip(q[qi], lo, hi), q[qi])
        return q


def collision_pairs(world, geoms) -> set:
    """The (sorted) pairs of these geom ids that MuJoCo's collision filter admits: different bodies that are not one
    weld-parent of the other (filterparent), not excluded, and contype / conaffinity compatible; like Genesis' collider
    _valid_collision_pairs (for the planning tools) also without the pairs that overlap in the model's neutral
    configuration qpos0 (Genesis filters those out of a robot's self-collision: e.g. the two ALOHA fingers at 0)."""
    mj = _mj()
    m = world.model
    neutral = mj.MjData(m)
    mj.mj_kinematics(m, neutral)
    excluded = {int(x) for x in np.asarray(m.exclude_signature).reshape(-1)}
    weld, parent = np.asarray(m.body_weldid), np.asarray(m.body_parentid)
    out = set()
    geoms = sorted(int(g) for g in geoms)
    for i, a in enumerate(geoms):
        for b in geoms[i + 1:]:
            b1, b2 = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
            w1, w2 = int(weld[b1]), int(weld[b2])
            if w1 == w2:
                continue
            p1, p2 = int(weld[parent[w1]]), int(weld[parent[w2]])
            if w1 and w2 and (p1 == w2 or p2 == w1):  # filterparent spares pairs with a world-welded body
                continue
            lo, hi = min(b1, b2), max(b1, b2)
            if (lo << 16) + hi in excluded:            # MuJoCo's exclude_signature: (body1 << 16) + body2
                continue
            if not ((int(m.geom_contype[a]) & int(m.geom_conaffinity[b])) or (int(m.geom_contype[b]) & int(m.geom_conaffinity[a]))):
                continue
            if mj.mj_geomDistance(m, neutral, a, b, 1e-4, None) < 1e-5:     # touching in qpos0 (within 10 um)
                continue
            out.add((a, b))
    return out


class _ChainFK:
    def __init__(self, entity: EntityView):
        mj = _mj()
        self.entity = entity
        self.model = entity.world.model
        self.data = mj.MjData(self.model)
        self.names = [(l.name, l.idx) for l in entity.links]

    def __call__(self, q_local) -> dict:
        mj = _mj()
        q = np.asarray(q_local, dtype=np.float64).reshape(-1)
        d = self.data
        d.qpos[:] = self.model.qpos0
        d.qpos[self.entity._qadr[:len(q)]] = q
        mj.mj_kinematics(self.model, d)
        out = {}
        for name, b in self.names:
            T = np.eye(4)
            T[:3, :3] = np.asarray(d.xmat[b]).reshape(3, 3)
            T[:3, 3] = d.xpos[b]
            out[name] = T
        return out
