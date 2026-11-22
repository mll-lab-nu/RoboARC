"""Per-substep contact evidence for tasks that declare contact roles (EntitySpec.contact_roles): GenesisWorld's
RigidContactCollector (topogym/world/physics/rigid_contacts.py) on MuJoCo, with the same schema-1 summary.

Every MuJoCo step of a tick (the World's solver substeps) is sampled after its collision and constraint solve. A
contact is touching at distance <= 0 (Genesis: penetration >= 0). Roles: the task's declared role of an authored
collision geom (resolved by name through the scene manifest), "robot" for every robot collision geom (robot_part:
"gripper" in the subtree of a group's grasp.contact_root_link, else "arm"), "other" for the rest.

Listed events: every touching contact with a robot geom on one side and a role-tagged task geom on the other, and every
touching contact between two different declared roles (buzz: wand metal on wire metal). Contacts within one role (the
strands of a bundle touching each other) and contacts without a declared role are not listed: no task reads them, and
listing them all would cost a dictionary per contact per substep. Coverage flags keep Genesis' meaning.
"""
from __future__ import annotations

import numpy as np

ROBOT = "robot"


class SubstepContactCollector:
    def __init__(self, world, roles: dict, *, geometry_complete: bool, robot_parts: dict, robot_geometry_complete: bool):
        self.world = world
        m = world.model
        names = sorted({r for r in roles.values()} | {ROBOT, "other"})
        self.names = names
        code = {n: i for i, n in enumerate(names)}
        self.role = np.full(m.ngeom, code["other"], np.int64)
        for g, r in roles.items():
            self.role[int(g)] = code[r]
        self.roles = {int(g): r for g, r in roles.items()}     # {geom id: declared role}, as Genesis' collector (planning)
        self.part = np.zeros(m.ngeom, np.int64)          # 0 none, 1 gripper, 2 arm
        for g, p in robot_parts.items():
            self.role[int(g)] = code[ROBOT]
            self.part[int(g)] = 1 if p == "gripper" else 2
        self.robot_code, self.other_code = code[ROBOT], code["other"]
        self.declared = np.zeros(len(names), bool)      # role code -> a task-declared role
        for n, i in code.items():
            self.declared[i] = n not in (ROBOT, "other")
        self.geometry_complete = bool(geometry_complete)
        self.robot_geometry_complete = bool(robot_geometry_complete)
        self.expected = int(world.substeps)
        self.summary = self._empty(0)
        self._f6 = np.zeros(6)

    def _empty(self, tick):
        return dict(schema_version=1, physics_step=int(tick), source="mujoco_substep_contacts",
                    substeps_complete=False, geometry_roles_complete=self.geometry_complete,
                    robot_geometry_complete=self.robot_geometry_complete, expected_substeps=self.expected,
                    sampled_substeps=0, events=[], ipc_contact_coverage=False, continuous_time_coverage=False,
                    contact_snapshots=None)

    def begin(self, tick):
        self.summary = self._empty(tick)

    def sample(self, substep):
        """After one mj_step: its contacts (the collision and constraint solve of that step)."""
        self._sample(substep)
        self.summary["sampled_substeps"] += 1

    def _sample(self, substep):
        import mujoco
        m, d = self.world.model, self.world.data
        n = int(d.ncon)
        if not n:
            return
        c = d.contact
        geom = np.asarray(c.geom[:n])
        dist = np.asarray(c.dist[:n])
        ra, rb = self.role[geom[:, 0]], self.role[geom[:, 1]]
        task_a, task_b = self.declared[ra], self.declared[rb]
        listed = (np.asarray(c.exclude[:n]) == 0) & (dist <= 0) & (
            ((ra == self.robot_code) & task_b) | ((rb == self.robot_code) & task_a) | (task_a & task_b & (ra != rb)))
        parts = {0: None, 1: "gripper", 2: "arm"}
        for i in np.flatnonzero(listed):
            a, b = int(geom[i, 0]), int(geom[i, 1])
            mujoco.mj_contactForce(m, d, int(i), self._f6)
            self.summary["events"].append(dict(
                geom_a=a, geom_b=b, role_a=self.names[ra[i]], role_b=self.names[rb[i]],
                robot_part_a=parts[int(self.part[a])], robot_part_b=parts[int(self.part[b])],
                touching=True, substep=int(substep), penetration_m=float(-dist[i]),
                force_n=float(np.linalg.norm(self._f6[:3]))))

    def end(self):
        self.summary["substeps_complete"] = self.summary["sampled_substeps"] == self.expected

    def birth(self):
        """The initialized pose, without integration (the caller ran a forward pass)."""
        self.summary = self._empty(0)
        self._sample(-1)
        self.summary.update(substeps_complete=True, expected_substeps=0, sampled_substeps=0,
                            sampling_scope="initial_pose")

    def read(self):
        # A summary holds scalars and a list of flat event dicts: copying those two levels is a full copy.
        return dict(self.summary, events=[dict(e) for e in self.summary["events"]])

    def restore(self, value):
        self.summary = dict(value, events=[dict(e) for e in value["events"]])


def resolve(world, specs, manifest, description):
    """(roles {geom id: role}, geometry_complete, robot_parts {geom id: part}, robot_complete) for the specs with roles."""
    import mujoco
    m = world.model
    roles, complete = {}, True
    for spec in specs:
        if not spec.contact_roles:
            continue
        record = manifest["entities"][spec.name]
        names = record.get("geoms", {})
        for authored, role in spec.contact_roles.items():
            mj_name = names.get(authored)
            gid = -1 if mj_name is None else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, mj_name)
            if gid < 0:
                complete = False
                continue
            roles[gid] = role
        # every colliding geom of the entity must carry a declared role
        bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link["body"]) for link in record["links"]}
        own = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies
               and (int(m.geom_contype[g]) or int(m.geom_conaffinity[g]))]
        complete &= all(g in roles for g in own)
    robot = world.entities.get("robot")
    parts, robot_complete = {}, False
    if robot is not None and description is not None and description.groups:
        gripper = set()
        robot_complete = True
        for group in description.groups:
            grasp = group.grasp
            root = None
            if grasp is not None:
                try:
                    root = robot.get_link(grasp.contact_root_link)
                except KeyError:
                    root = None
            if root is None:
                robot_complete = False
                continue
            subtree = {root.idx}
            for b in range(root.idx + 1, m.nbody):
                if int(m.body_parentid[b]) in subtree:
                    subtree.add(b)
            for finger in grasp.finger_links:
                try:
                    link = robot.get_link(finger)
                except KeyError:
                    robot_complete = False
                    continue
                robot_complete &= link.idx in subtree and bool(link.geoms)
            gripper |= subtree
        for link in robot.links:
            for g in link._collision_ids:
                parts[int(g)] = "gripper" if link.idx in gripper else "arm"
        robot_complete &= bool(parts) and bool(gripper)
    return roles, bool(complete and roles), parts, robot_complete
