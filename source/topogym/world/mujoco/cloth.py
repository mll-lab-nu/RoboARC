"""PBD cloth on MuJoCo: a "pbd_cloth" declaration (blanket_flatten's blanket) becomes a 2D flex over point-mass vertex
bodies, each carrying one contact sphere (profile "cloth").

The model (2026-10-10, blanket agent; evidence in tmp/2026-10-09_robotangle_mujoco/blanket):
  - vertices: one body per flex vertex `<entity>::v<k>` at its flat rest point (the authored sheet as the task declares it),
    three slide joints (x, y, z), a point mass (the cloth's rho * area, shared equally, as PBD shares it per particle) and
    one sphere geom: the cloth's contact with the table and with its own layers, as PBD's particles collide. Spheres touch
    ordinary geoms; MuJoCo 3.6-3.15 cap a flex's own contacts at 50 per geom and body (a cloth on one table box falls
    through) and its self-collision costs 130-500 ms per step, so neither is used for that.
  - the flex: over the vertex bodies (dim 2), elastic2d membrane + bending (implicit under the `discrete` integrator), and
    the robot's contact surface (contype / conaffinity bits: the flex meets robot geoms only).
  - which mesh: "visual" (default) = the authored grid itself, whose nodes are PBD particles (the PBD mesh is that grid
    remeshed: every edge split once); "particles" = the PBD particle mesh (Genesis' own remesh, reproduced with pymeshlab,
    the same vertices and faces): exact for every particle, ~6x the cost.
  - contact bits: table and ground contype 2 / conaffinity 3 (robot 1 + cloth spheres 2); spheres 4 / 6 (table, each
    other); flex 8 / 1 (the robot). A sphere net closes its holes when 2 r exceeds the grid's quad diagonal (12 mm grid:
    r > 4.25 mm; at 4 mm a corner flap passed through the sheet).
  - the birth: the frozen PBD particles (SimulatorConfig.mujoco_birth: the released instance whose soft_state holds them)
    give the flex vertices their birth positions (the grid nodes are particles: exact); the scene keeps them
    (cloth_<entity>.npz); the world puts them in after the task's bind (restore_declared_initial_state), at rest. A
    gathered PBD state is not at rest on the flex (it snaps out at up to 3.3 m/s): its instances are born from a settled
    source instead (made by make-instance itself, simulator override mujoco_birth_settle: rcb_topogym.py), whose
    evidence the cloth record keeps.
  - the particle view: every PBD particle is a fixed barycentric combination of its flex triangle's vertices (exact on the
    flat rest; the grid nodes exactly the vertices), so the task's predicates read state["particles"][entity] unchanged.
    The velocity is the end-of-tick displacement over the tick, as GenesisWorld reports it since tooling 19.
  - the assisted grasp (pbd_local_patch_v1): a pool of inactive connect equalities per hand, re-targeted at acquisition to
    the patch's vertices (MujocoClothGraspBackend, grasp_cloth.py).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CLOTH_FORMAT = "robotangle-mujoco-cloth/1"
SEP = "::"
TABLE_BITS = dict(contype="2", conaffinity="3")
SPHERE_BITS = dict(contype="4", conaffinity="6")
FLEX_BITS = dict(contype="8", conaffinity="1")

# Profile "cloth" keys and their defaults.
CLOTH_DEFAULTS = dict(
    mesh="visual",                # the flex mesh: "visual" (the authored grid) or "particles" (the PBD particle mesh)
    particle_size_m=.008,         # PBD's particle size (the task's pbd.particle_size): the particle mesh's edge target
    sphere_radius_m=.0045,        # every vertex's contact sphere
    flex_radius_m=.004,           # the flex elements' radius (the robot's contact surface)
    young=3e7, poisson=.3, thickness=1.1e-4, elastic2d="both", elastic_damping=0.,
    vertex_damping=0.,            # the vertex slides' damping (N s / m)
    self_friction=.15,            # sphere - sphere (PBD's own particle friction)
    friction=.5,                  # sphere - table and flex - robot (the task's table_friction static mu)
    solref=[.005, 1.], solimp=[.9, .99, .001, .5, 2.],
    pins=24,                      # connect equalities per hand (the assisted patch grasp)
    pin=dict(solref=[.005, 1.], solimp=[.95, .99, .001, .5, 2.]),
)


def settings_of(profile) -> dict:
    return {**CLOTH_DEFAULTS, **dict(profile.get("cloth") or {})}


# ----------------------------------------------------------------------------------------------- geometry
def particle_mesh(vertices, faces, particle_size):
    """The PBD particle mesh of a cloth mesh, as Genesis builds it (PBDTetEntity.sample: pymeshlab's isotropic explicit
    remeshing to edge length particle_size, wrapped in trimesh.Trimesh): its vertices are the particles, in Genesis'
    order. Checked against Genesis' own particles on the released blanket instances: same order, 1.7e-16 m."""
    import pymeshlab
    import trimesh
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(vertices, dtype=np.float64),
                               face_matrix=np.asarray(faces, dtype=np.int32)))
    ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PureValue(float(particle_size)))
    m = ms.current_mesh()
    tm = trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix())
    return np.asarray(tm.vertices, dtype=np.float64), np.asarray(tm.faces, dtype=np.int64)


def orient_up(rest, faces):
    """Counter-clockwise faces seen from +z on the flat rest sheet (a 2D flex's element order)."""
    a, b, c = (rest[faces[:, k], :2] for k in range(3))
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    out = faces.copy()
    out[cross < 0] = out[cross < 0][:, [0, 2, 1]]
    return out


def barycentric_binding(points, vertices, faces):
    """Each point as a barycentric combination of the triangle of (vertices, faces) closest to it: (corners (n, 3),
    weights (n, 3), distance (n,))."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    closest, distance, tri = trimesh.proximity.closest_point(mesh, points)
    corners = faces[tri]
    weights = trimesh.triangles.points_to_barycentric(vertices[corners], closest)
    return corners.astype(np.int64), np.asarray(weights, dtype=np.float64), np.asarray(distance)


def birth_from_instance(path, name: str) -> dict:
    """A cloth's frozen birth from a released (Genesis) instance: its soft_state particles and the PBD archive it names.
    A settled birth source (make-instance with mujoco_birth_settle: the same instance with the particles of the cloth
    relaxed at rest on MuJoCo) also carries its settle evidence (`mujoco_cloth_settle`), kept in the cloth record."""
    inst = json.loads(Path(path).read_text())
    soft = (inst.get("soft_state") or {}).get(name)
    if soft is None:
        raise ValueError(f"the birth instance {path} has no soft state of {name!r}")
    params = (inst.get("simulator") or {}).get("task_parameters") or {}
    return dict(particles=np.asarray(soft["position_m"], dtype=np.float64), source=str(path),
                source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                archive_sha256=params.get("initial_state_sha256"), state_sha=inst.get("state_sha"),
                settle=(inst.get("mujoco_cloth_settle") or {}).get(name),
                rest_angle=(inst.get("vbd_cloth_rest_angle") or {}).get(name))


# ----------------------------------------------------------------------------------------------- scene building
def add_cloth(builder, spec, birth) -> None:
    """Write cloth `spec` into the SceneBuilder (vertex bodies, flex, pins, contact bits) and its data file
    cloth_<name>.npz; register the manifest record (kind "pbd_cloth")."""
    import xml.etree.ElementTree as ET
    import trimesh
    from scipy.spatial.transform import Rotation
    from .scene import numbers
    if birth is None:
        raise ValueError(f"{spec.name}: building a MuJoCo cloth needs its frozen birth (SimulatorConfig.mujoco_birth)")
    s = settings_of(builder.profile)
    mesh = trimesh.load(spec.asset_path, force="mesh", process=False)
    R = Rotation.from_quat(np.asarray(spec.quaternion_wxyz, dtype=float)[[1, 2, 3, 0]]).as_matrix()
    V = np.asarray(mesh.vertices, dtype=np.float64) @ R.T + np.asarray(spec.position_m, dtype=float)
    F = np.asarray(mesh.faces, dtype=np.int64)
    P, PF = particle_mesh(V, F, s["particle_size_m"])
    born = np.asarray(birth["particles"], dtype=np.float64)
    if born.shape != P.shape:
        raise ValueError(f"{spec.name}: the birth has {born.shape} particles, the PBD particle mesh {P.shape}")
    if s["mesh"] == "visual":
        rest, faces = V, F
        from scipy.spatial import cKDTree
        dist, node = cKDTree(P).query(V)
        if dist.max() > 1e-9:
            raise ValueError(f"{spec.name}: the cloth grid's nodes are not PBD particles ({dist.max():.3g} m)")
    elif s["mesh"] == "particles":
        rest, faces, node = P, PF, np.arange(len(P))
    else:
        raise ValueError(f"unknown cloth mesh {s['mesh']!r}")
    faces = orient_up(rest, faces)
    corners, weights, gap = barycentric_binding(P, rest, faces)
    if gap.max() > 1e-9:
        raise ValueError(f"{spec.name}: a PBD particle lies off the flat cloth ({gap.max():.3g} m)")
    n = len(rest)
    material = dict(spec.material)
    area = float(trimesh.Trimesh(vertices=V, faces=F, process=False).area)
    mass = float(material["rho"]) * area
    m_vertex = mass / n
    r = float(s["sphere_radius_m"])
    # the tabletop meets the cloth spheres (and the robot): collision bit 2 besides the robot's 1
    for body in builder.world.findall("body"):
        if body.get("name") in ("table", "ground"):
            for g in body.findall("geom"):
                g.attrib.update(TABLE_BITS)
    cls = ET.SubElement(builder.default, "default", **{"class": f"{spec.name}_cloth"})
    ET.SubElement(cls, "joint", type="slide", damping=numbers([s["vertex_damping"]]), armature="0")
    ET.SubElement(cls, "geom", type="sphere", size=numbers([r]), condim="3", mass="0",
                  friction=numbers([s["self_friction"], .005, .0001]), solref=numbers(s["solref"]),
                  solimp=numbers(s["solimp"]), **SPHERE_BITS)
    names = []
    for k in range(n):
        name = f"{spec.name}{SEP}v{k}"
        body = ET.SubElement(builder.world, "body", name=name, pos=numbers(rest[k]))
        ET.SubElement(body, "inertial", pos="0 0 0", mass=numbers([m_vertex]), diaginertia="1e-9 1e-9 1e-9")
        for axis, label in ((("1 0 0"), "x"), (("0 1 0"), "y"), (("0 0 1"), "z")):
            ET.SubElement(body, "joint", name=f"{name}{label}", axis=axis, **{"class": f"{spec.name}_cloth"})
        ET.SubElement(body, "geom", name=f"{name}{SEP}s", **{"class": f"{spec.name}_cloth"})
        names.append(name)
    deformable = builder.root.find("deformable")
    if deformable is None:
        deformable = ET.SubElement(builder.root, "deformable")
    flex = ET.SubElement(deformable, "flex", name=spec.name, dim="2", radius=numbers([s["flex_radius_m"]]),
                         body=" ".join(names), element=" ".join(str(int(i)) for i in faces.reshape(-1)))
    ET.SubElement(flex, "elasticity", young=numbers([s["young"]]), poisson=numbers([s["poisson"]]),
                  thickness=numbers([s["thickness"]]), elastic2d=s["elastic2d"], damping=numbers([s["elastic_damping"]]))
    ET.SubElement(flex, "contact", selfcollide="none", condim="3", friction=numbers([s["friction"], .005, .0001]),
                  solref=numbers(s["solref"]), solimp=numbers(s["solimp"]), **FLEX_BITS)
    pins = {}
    for g in builder.description.groups:
        pins[g.name] = []
        for k in range(int(s["pins"])):
            pin = f"grasp/{g.name}/{spec.name}/pin{k}"
            ET.SubElement(builder.equality, "connect", name=pin, body1=g.hand_link, body2=names[0], anchor="0 0 0",
                          active="false", solref=numbers(s["pin"]["solref"]), solimp=numbers(s["pin"]["solimp"]))
            pins[g.name].append(pin)
    data = f"cloth_{spec.name}.npz"
    np.savez(builder.out / data, rest=rest, faces=faces, birth=born[node], node=node, particle_rest=P,
             particle_faces=PF, corners=corners, weights=weights)
    builder.entities[spec.name] = dict(
        kind="pbd_cloth", root=names[0], links=[], joints=[], free_joint=None, collision=True,
        cloth=dict(format=CLOTH_FORMAT, file=data, mesh=s["mesh"], vertices=n, particles=int(len(P)), mass_kg=mass,
                   vertex_mass_kg=m_vertex, sphere_radius_m=r, settings=json.loads(json.dumps(s)), pins=pins,
                   asset_sha256=hashlib.sha256(Path(spec.asset_path).read_bytes()).hexdigest(),
                   birth=dict(source_sha256=birth.get("source_sha256"), archive_sha256=birth.get("archive_sha256"),
                              state_sha=birth.get("state_sha"), settle=birth.get("settle"),
                              particles_sha256=hashlib.sha256(born.tobytes()).hexdigest(),
                              off_surface_max_m=float(np.linalg.norm(
                                  np.einsum("nk,nki->ni", weights, born[node][corners]) - born, axis=1).max()))))


# ----------------------------------------------------------------------------------------------- runtime
class Cloth:
    """One cloth of a built MuJoCo world: its vertex dofs, the particle view, the birth."""

    def __init__(self, world, name: str, record: dict, scene_dir):
        import mujoco
        m = world.model
        self.world, self.name, self.record = world, name, record
        info = record["cloth"]
        with np.load(Path(scene_dir) / info["file"], allow_pickle=False) as data:
            self.rest = data["rest"].copy()
            self.faces = data["faces"].copy()
            self.birth = data["birth"].copy()
            self.corners, self.weights = data["corners"].copy(), data["weights"].copy()
            self.particle_rest = data["particle_rest"].copy()
            self.node_particles = data["node"].copy()   # the PBD particle each flex vertex is
        self.flex = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, name)
        if self.flex < 0:
            raise ValueError(f"scene lacks the flex of cloth {name}")
        self.vadr, self.vnum = int(m.flex_vertadr[self.flex]), int(m.flex_vertnum[self.flex])
        self.bodies = np.asarray(m.flex_vertbodyid[self.vadr:self.vadr + self.vnum]).copy()
        j0 = np.asarray(m.body_jntadr)[self.bodies]
        if np.any(np.asarray(m.body_jntnum)[self.bodies] != 3):
            raise ValueError(f"{name}: every cloth vertex needs three slides")
        self.qadr = np.asarray(m.jnt_qposadr)[j0[:, None] + np.arange(3)[None]]
        self.dadr = np.asarray(m.jnt_dofadr)[j0[:, None] + np.arange(3)[None]]
        self.body_set = frozenset(int(b) for b in self.bodies)
        self.tick_start = None          # flex vertex positions at the start of the last tick (velocity readout)

    def vertices(self, d) -> np.ndarray:
        """The flex vertices now: their bodies' origins (fresh after mj_kinematics; flexvert_xpos is refreshed by mj_flex)."""
        return np.asarray(d.xpos)[self.bodies].copy()

    def particles_of(self, vertices) -> np.ndarray:
        return np.einsum("nk,nki->ni", self.weights, np.asarray(vertices)[self.corners])

    def apply_birth(self, d) -> None:
        """Every vertex at its frozen birth position, at rest (the slides' coordinates are offsets from the flat rest)."""
        d.qpos[self.qadr.reshape(-1)] = (self.birth - self.rest).reshape(-1)
        d.qvel[self.dadr.reshape(-1)] = 0.

    def particle_state(self, d, dt) -> dict:
        x = self.vertices(d)
        p = self.particles_of(x)
        start = x if self.tick_start is None else self.tick_start
        v = (p - self.particles_of(start)) / float(dt)
        return dict(position_m=p, velocity_m_s=v)


class ClothView:
    """A cloth as the tooling sees it (the subset of Genesis' PBD entity API its task, grasp and evaluation use): name,
    n_particles, get_particles_pos (the PBD particles, now), grasp_material_coordinates (bind_material_map sets it), no
    links (the vertices are not articulated links: the world's rigid readout skips the cloth)."""

    engine = "mujoco"

    def __init__(self, world, cloth: Cloth):
        from types import SimpleNamespace
        self.world, self.cloth, self.name = world, cloth, cloth.name
        self.links, self.joints = [], []
        self.n_particles = int(len(cloth.particle_rest))
        self.morph = SimpleNamespace(fixed=False)
        self.material = SimpleNamespace(coup_type=None)

    def get_particles_pos(self) -> np.ndarray:
        return self.cloth.particles_of(self.cloth.vertices(self.world.data))
