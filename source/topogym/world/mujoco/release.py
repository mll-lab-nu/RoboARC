"""Release-confirmation evidence on MuJoCo: GenesisWorld's ReleaseGeometry (topogym/world/physics/release_geometry.py,
schema 2) from the MuJoCo scene, for tasks with a final settling spec (configure_final_settling).

Read-only: every read uses the current geom poses. Per collision geom of the robot and of each rigid settling target: a
conservative world box, the geom's local bounding box (model.geom_aabb: a mesh's vertices, a primitive's extent) turned by
|R|. Per robot geom: the convex cover planes of a mesh's vertices (topogym.evaluation.terminal.clearance), none for a
primitive, in the world. A capped collision hull (maxhullvert) lies inside its mesh's vertices, so both stay conservative.
Rope (FEM) targets with the exact robot contact policy (knot_untying's terminal v5) read schema 3: also every robot geom's
native solid and pose, and the ropes' tetrahedra over their skinned vertices (rope_chain.py).
"""
from __future__ import annotations

import numpy as np

from topogym.evaluation.terminal.clearance import convex_cover_planes


class MujocoReleaseGeometry:
    def __init__(self, world, rigid_targets, *, exact_robot_contact=False, ropes=None):
        import mujoco
        ropes = dict(ropes or {})
        if exact_robot_contact and not ropes:
            raise NotImplementedError("MuJoCo core: the exact robot release contact policy needs FEM targets (ropes)")
        self.exact = bool(exact_robot_contact)
        # FEM targets on MuJoCo are rope chains (rope_chain.py): their tetrahedra (the frozen volume's) over the skinned
        # vertices of state["fem"]
        self.fem_elements = {name: np.asarray(rope.elements, dtype=np.int64) for name, rope in ropes.items()}
        m = world.model
        self.world, self.targets = world, tuple(rigid_targets)
        self.robot_present = "robot" in world.entities
        names = (("robot",) if self.robot_present else ()) + self.targets
        ids, self.rows, self.complete = [], {}, True
        for name in names:
            own = [int(g) for link in world.entities[name].links for g in link._collision_ids]
            self.complete &= bool(own)
            self.rows[name] = np.arange(len(ids), len(ids) + len(own))
            ids += own
        self.ids = np.asarray(ids, dtype=np.int64)
        aabb = np.asarray(m.geom_aabb, dtype=np.float64)[self.ids].reshape(-1, 6)
        self.centers, self.halves = aabb[:, :3].copy(), aabb[:, 3:].copy()
        if not np.isfinite(aabb).all() or np.any(self.halves < 0):
            raise ValueError("Invalid release collision geometry")
        groups = {}
        for row in self.rows.get("robot", ()):
            g = int(self.ids[row])
            planes = np.empty((0, 4))
            if int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = int(m.geom_dataid[g])
                a, n = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
                planes = convex_cover_planes(np.asarray(m.mesh_vert[a:a + n], dtype=np.float64))
            groups.setdefault(len(planes), []).append((int(row), planes))
        # robot rows grouped by plane count: one stacked product per group (as ReleaseGeometry._batches)
        self.groups = [(np.array([r for r, _ in items]), np.array([p for _, p in items]).reshape(len(items), k, 4))
                       for k, items in groups.items()]
        # the exact policy (schema 3, ReleaseGeometry's robot_solids): every robot collision geom as its native solid,
        # in the geom frame (a mesh: its vertices, which contain the capped hull it collides as; a sphere: its radius)
        self.robot_solids = []
        if self.exact:
            from .views import GeomView
            from .scene import _primitive_mesh
            links = {link.idx: link for link in world.entities["robot"].links}
            for row in self.rows.get("robot", ()):
                g = int(self.ids[row])
                kind = int(m.geom_type[g])
                link = links[int(m.geom_bodyid[g])]
                if kind == mujoco.mjtGeom.mjGEOM_MESH:
                    view = GeomView(world, g)
                    native, data, v, f = "MESH", np.zeros(0), view.init_verts, view.init_faces
                elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                    v, f = _primitive_mesh("sphere", m.geom_size[g])
                    native, data = "SPHERE", np.array([float(m.geom_size[g][0])])
                elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                    v, f = _primitive_mesh("box", m.geom_size[g])
                    native, data = "BOX", 2 * np.asarray(m.geom_size[g], dtype=np.float64)
                else:
                    raise NotImplementedError(f"exact release contact: robot geom type {kind} is not ported")
                solid = dict(link=link.name, geom_index=g, fixed_to_world=bool(link.is_fixed), native_type=native,
                             native_convex=True, native_data=np.asarray(data, dtype=np.float64).copy(),
                             vertices_m=np.asarray(v, dtype=np.float64).copy(), faces=np.asarray(f, dtype=np.int64).copy())
                for key in ("native_data", "vertices_m", "faces"):
                    solid[key].setflags(write=False)
                self.robot_solids.append(solid)

    def read(self):
        d = self.world.data
        p = np.asarray(d.geom_xpos, dtype=np.float64)[self.ids].reshape(-1, 3)
        r = np.asarray(d.geom_xmat, dtype=np.float64)[self.ids].reshape(-1, 3, 3)
        centers = p + (r @ self.centers[:, :, None])[:, :, 0]
        halves = (np.abs(r) @ self.halves[:, :, None])[:, :, 0]
        bounds = np.stack([centers - halves, centers + halves], axis=1)
        planes = {}
        for rows, local in self.groups:
            normals = local[:, :, :3] @ np.swapaxes(r[rows], 1, 2)
            offsets = local[:, :, 3] - (normals @ p[rows][:, :, None])[:, :, 0]
            for j, i in enumerate(rows):
                planes[int(i)] = np.concatenate([normals[j], offsets[j][:, None]], axis=1)
        robot = self.rows.get("robot", np.zeros(0, np.int64))
        out = dict(schema_version=3 if self.exact else 2, physics_step=self.world.physics_step, complete=bool(self.complete),
                   robot_present=self.robot_present, robot_bounds_m=bounds[robot].reshape(-1, 2, 3),
                   target_bounds_m={name: bounds[self.rows[name]].reshape(-1, 2, 3) for name in self.targets},
                   robot_convex_planes_m=[planes[int(i)] for i in robot], fem_elements=dict(self.fem_elements))
        if self.exact:
            out.update(robot_solids=self.robot_solids,
                       robot_solid_poses=[dict(position_m=p[int(i)].copy(), rotation=r[int(i)].copy()) for i in robot])
        return out
