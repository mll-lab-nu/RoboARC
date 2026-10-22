"""The MuJoCo scene of one task instance: MJCF built from the same declarations as the Genesis scene.

Inputs are the ones GenesisWorld.from_tabletop / add_entities receive: the tabletop scene configuration (table, robot
mount, gravity, timing), the robot's MJCF (topogym.assets.robot.robot_asset: the same file Genesis loads), the robot
description (controlled joints, PD gains, force limits), the task's EntitySpecs (after finalize_spec), its physics
options and the episode's grasp assistance. A per-family profile (profiles.py) holds what MuJoCo needs beyond them:
contact softness, the collision representation of non-convex meshes (CoACD hulls), solver settings.

The output is a directory: scene.xml, meshes/ (task collision and visual meshes, CoACD hulls) and scene.json (the
manifest: entity -> MuJoCo bodies / joints, the robot's actuators, the grasp welds, the hull audit, sha256 of every file).
make-instance freezes this directory with the instance (instance_assets/mujoco/), so every later session loads the
same bytes: CPU-generated geometry (URDF primitive sizes, CoACD hulls) is never regenerated at run time.

Robot meshes are referenced, not copied, through a token resolved at load: ROBOT_ASSETS (the image's
TOPOGYM_ASSET_ROOT) or GENESIS_ASSETS (a path in the genesis package's asset layout). The Panda's GENESIS_ASSETS paths
(xml/franka_emika_panda/) resolve to its vendored copy in the robot asset root (robots/franka_emika_panda/: MuJoCo
Menagerie's franka_emika_panda, the same bytes as genesis-world 1.2.1's) when that copy is whole (vendored_source), else
all of them to the installed genesis package; the manifest load_scene returns says which (robot_asset_sources). A frozen
scene holds no absolute path, so it loads wherever the image installs its packages.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

SCENE_FORMAT = "robotangle-mujoco-scene/1"
ROBOT_ASSETS = "@robot-assets@/"          # resolved at load to topogym.assets.robot.asset_root()
GENESIS_ASSETS = "@genesis-assets@/"      # a path in the genesis package's asset layout (_genesis_asset)
GENESIS_VENDORED = {"xml/franka_emika_panda/": "robots/franka_emika_panda/"}   # genesis asset dir -> its copy in the robot root
SEP = "::"                                # MuJoCo name of a task entity's link / joint: "<entity>::<link>"


def _genesis_assets():
    """The installed genesis package's assets directory (robot MJCFs and meshes), or None."""
    from importlib.util import find_spec
    spec = find_spec("genesis")
    if spec is None or spec.origin is None:
        return None
    return (Path(spec.origin).parent / "assets").resolve()


_VENDORED: dict = {}     # (robot root, GENESIS_VENDORED prefix) -> (files source.json lists, or None; why not)


def vendored_source(prefix: str, robot_root: Path, needed=()) -> str:
    """Where a scene's files under a GENESIS_VENDORED prefix come from, decided for all of them at once: "vendored" when
    the vendored copy's source.json lists every needed file and every file it lists is present with its sha256 (checked
    once per process), else "genesis: <why>" (the installed genesis package, for all of them)."""
    key = (str(robot_root), prefix)
    if key not in _VENDORED:
        folder = robot_root / GENESIS_VENDORED[prefix]
        if not (folder / "source.json").is_file():
            _VENDORED[key] = (None, "no vendored copy")
        else:
            files = json.loads((folder / "source.json").read_text())["sha256"]
            bad = [f for f, digest in files.items() if not (folder / f).is_file() or sha256_file(folder / f) != digest]
            _VENDORED[key] = (None, f"{len(bad)} of {len(files)} vendored files differ from source.json") if bad else (set(files), "")
    files, why = _VENDORED[key]
    missing = [n for n in needed if files is not None and n not in files]
    if files is None or missing:
        return "genesis: " + (why or f"{len(missing)} needed files not vendored")
    return "vendored"


def _genesis_asset(rel: str, robot_root: Path, sources: dict) -> Path:
    """The file of a GENESIS_ASSETS path: in the vendored copy when sources (vendored_source of its prefix) says so,
    else in the installed genesis package."""
    for prefix, vendored in GENESIS_VENDORED.items():
        if rel.startswith(prefix) and sources.get(prefix) == "vendored":
            return robot_root / vendored / rel[len(prefix):]
    if _genesis_assets() is None:
        raise FileNotFoundError(f"the scene needs the robot asset {rel}: not vendored, and genesis is not installed")
    return _genesis_assets() / rel


def numbers(values) -> str:
    return " ".join(format(float(x), ".17g") for x in np.asarray(values, dtype=float).reshape(-1))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def quat_wxyz(matrix) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(matrix, dtype=float)).as_quat()[[3, 0, 1, 2]]


def _child(root, tag):
    found = root.find(tag)
    return ET.SubElement(root, tag) if found is None else found


def _urdf_origin(node) -> np.ndarray:
    T = np.eye(4)
    origin = None if node is None else node.find("origin")
    if origin is not None:
        T[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        T[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return T


def mujoco_quat_z_to(q, axis):
    """q (wxyz) <- the shortest rotation of +z onto the unit vector axis (MuJoCo's own fromto frame)."""
    import mujoco
    mujoco.mju_quatZ2Vec(q, np.asarray(axis, dtype=np.float64))


def _pose_attrs(T) -> dict:
    return {"pos": numbers(T[:3, 3]), "quat": numbers(quat_wxyz(T[:3, :3]))}


def _urdf_capsules(pieces) -> dict:
    """{cylinder index: (sphere index, sphere index)} of one URDF link's collisions: a cylinder with a sphere of its radius
    centred on each end cap (within 1 nm) is exactly a capsule. Each sphere joins at most one cylinder."""
    from scipy.spatial import cKDTree
    spheres = [(j, float(list(p.find("geometry"))[0].get("radius")), _urdf_origin(p)[:3, 3]) for j, p in enumerate(pieces)
               if list(p.find("geometry"))[0].tag == "sphere"]
    if not spheres:
        return {}
    tree, used, out = cKDTree(np.array([c for _, _, c in spheres])), set(), {}
    for i, piece in enumerate(pieces):
        shape = list(piece.find("geometry"))[0]
        if shape.tag != "cylinder":
            continue
        T, r, h = _urdf_origin(piece), float(shape.get("radius")), float(shape.get("length")) / 2
        pair = []
        for end in (T[:3, 3] - h * T[:3, 2], T[:3, 3] + h * T[:3, 2]):
            k = next((k for k in sorted(tree.query_ball_point(end, 1e-9)) if k not in used and k not in pair
                      and abs(spheres[k][1] - r) <= 1e-12), None)
            if k is not None:
                pair.append(k)
        if len(pair) == 2:
            used.update(pair)
            out[i] = tuple(spheres[k][0] for k in pair)
    return out


class _Meshes:
    """Mesh files written next to scene.xml (meshes/), named by content so repeated geometry is stored once."""

    def __init__(self, root: Path, asset: ET.Element):
        self.root, self.asset, self.names = root, asset, {}
        (root / "meshes").mkdir(parents=True, exist_ok=True)

    def add(self, vertices, faces, stem: str) -> str:
        import trimesh
        mesh = trimesh.Trimesh(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64), process=False)
        data = mesh.export(file_type="obj", digits=17, include_normals=False, include_texture=False)
        data = data.encode() if isinstance(data, str) else data
        key = hashlib.sha256(data).hexdigest()[:16]
        if key in self.names:
            return self.names[key]
        name = f"{stem}_{key}"
        (self.root / "meshes" / f"{name}.obj").write_bytes(data)
        ET.SubElement(self.asset, "mesh", name=name, file=f"meshes/{name}.obj")
        self.names[key] = name
        return name


def _coacd_hulls(vertices, faces, settings: dict, cache: Path | None):
    """Convex decomposition of a non-convex collision mesh (CoACD), cached by the mesh bytes and settings."""
    v, f = np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)
    key = hashlib.sha256(v.tobytes() + f.tobytes() + json.dumps(settings, sort_keys=True).encode()).hexdigest()
    if cache is not None and (cache / f"{key}.npz").is_file():
        with np.load(cache / f"{key}.npz") as data:
            count = int(data["count"])
            return [(data[f"v{i}"], data[f"f{i}"]) for i in range(count)], key, True
    import coacd
    parts = coacd.run_coacd(coacd.Mesh(v, f), **settings)
    parts = [(np.asarray(pv, dtype=np.float64), np.asarray(pf, dtype=np.int64)) for pv, pf in parts]
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        tmp = cache / f".{key}.{os.getpid()}.npz"
        np.savez(tmp, count=len(parts), **{f"v{i}": p[0] for i, p in enumerate(parts)},
                 **{f"f{i}": p[1] for i, p in enumerate(parts)})
        tmp.replace(cache / f"{key}.npz")
    return parts, key, False


def _winding_numbers(mesh, points):
    """The generalized winding number of each point about a closed triangle mesh: 1 inside its solid, 0 outside (the sum
    of the triangles' solid angles, van Oosterom & Strackee, over 4 pi); coincident faces of opposite normals cancel."""
    tri = np.asarray(mesh.triangles, dtype=np.float64)
    out = np.empty(len(points))
    for k in range(0, len(points), 16):
        a, b, c = (tri[None, :, i] - points[k:k + 16, None] for i in range(3))
        la, lb, lc = (np.linalg.norm(x, axis=2) for x in (a, b, c))
        num = np.einsum("pti,pti->pt", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("pti,pti->pt", a, b) * lc + np.einsum("pti,pti->pt", b, c) * la
               + np.einsum("pti,pti->pt", c, a) * lb)
        out[k:k + 16] = np.arctan2(num, den).sum(axis=1) / (2 * np.pi)
    return out


def _family_parts(path: str, vertices, faces):
    """A family's exact convex decomposition of a non-convex collision mesh (profile "split": entity -> "module:function"
    returning [(vertices, faces)]), checked against the declared mesh: the parts' volumes add up to the mesh's (1e-6, or a
    10 nm layer over its surface), no part vertex lies outside the mesh and no mesh vertex or face centre outside every
    part (1e-7 m). MuJoCo then collides the declared solid itself, openings included, instead of its convex hull or an
    approximate decomposition."""
    import importlib
    import trimesh
    from scipy.spatial import ConvexHull
    module, name = path.split(":")
    parts = getattr(importlib.import_module(module), name)(vertices, faces)
    mesh = trimesh.Trimesh(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64), process=False)
    hulls = [ConvexHull(np.asarray(v, dtype=np.float64)) for v, _ in parts]
    volume = float(sum(h.volume for h in hulls))
    points = np.vstack([np.asarray(v, dtype=np.float64) for v, _ in parts])
    signed = trimesh.proximity.signed_distance(mesh, points)
    # signed_distance signs a point by its nearest face's normal, or by ray parity. Beside an internal wall (coincident
    # faces of opposite normals inside the solid, which a union of parts touching along faces can keep) both can call an
    # interior point outside: claws_rotor's v12f mesh, part vertices put up to 3.5 mm outside, their winding number 1. A
    # point it puts more than 1e-7 m outside stays outside only if its winding number says so (below 1/2).
    flagged = np.flatnonzero(signed < -1e-7)
    if len(flagged):
        signed[flagged[_winding_numbers(mesh, points[flagged]) > .5]] = 0.
    outside = float(max(0., -signed.min()))
    probe = np.r_[mesh.vertices, mesh.triangles_center]
    depth = np.full(len(probe), np.inf)        # how far each probe lies outside the nearest part (<= 0: inside one)
    for h in hulls:
        depth = np.minimum(depth, np.max(probe @ h.equations[:, :3].T + h.equations[:, 3], axis=1))
    check = dict(volume_rel_error=float(abs(volume - mesh.volume) / mesh.volume), max_part_outside_m=outside,
                 max_uncovered_m=float(max(0., depth.max())))
    # Volume: within 1e-6 of the mesh's, or within a 10 nm layer over its surface. Mesh files carry 8 significant digits,
    # so a polygonal torus' planar quads are warped by ~1e-8 m and the hulls of its segments bulge by nanometres (orbits
    # easier's 90 mm ring: 1.08e-6 of its volume, a 4 nm layer).
    if (abs(volume - mesh.volume) > max(1e-6 * mesh.volume, 1e-8 * mesh.area) or outside > 1e-7
            or check["max_uncovered_m"] > 1e-7):
        raise ValueError(f"{path}: not an exact decomposition of the declared mesh: {check}")
    return parts, check


def _primitive_mesh(kind: str, size) -> tuple[np.ndarray, np.ndarray]:
    """A small triangle mesh of a primitive (geom frame), for geometry consumers that want vertices (the tooling's
    table guard and gripper opening)."""
    import trimesh
    if kind == "box":
        m = trimesh.creation.box(extents=2 * np.asarray(size[:3], dtype=float))
    elif kind == "sphere":
        m = trimesh.creation.icosphere(subdivisions=1, radius=float(size[0]))
    elif kind == "cylinder":
        m = trimesh.creation.cylinder(radius=float(size[0]), height=2 * float(size[1]), sections=16)
    elif kind == "capsule":
        m = trimesh.creation.capsule(radius=float(size[0]), height=2 * float(size[1]), count=[8, 8])
    else:
        raise ValueError(f"no primitive mesh for {kind}")
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


class SceneBuilder:
    """One MJCF scene; build() writes it. Every number that leaves Python is formatted with 17 significant digits."""

    def __init__(self, out_dir, *, env_cfg, robot_cfg, description, profile: dict, hull_cache: Path | None = None,
                 fem_birth: dict | None = None):
        self.out = Path(out_dir)
        self.fem_birth = dict(fem_birth or {})   # soft entity -> its frozen birth vertices (ropes: rope_chain.py)
        self.out.mkdir(parents=True, exist_ok=True)
        self.env_cfg, self.robot_cfg, self.description, self.profile = env_cfg, robot_cfg, description, dict(profile)
        self.hull_cache = hull_cache
        self.root = ET.Element("mujoco", model="robotangle")
        ET.SubElement(self.root, "compiler", angle="radian", autolimits="true", balanceinertia="false",
                      inertiafromgeom="auto", boundmass="0", boundinertia="0")
        dt = float(env_cfg.sim.dt) / int(env_cfg.sim.substeps)
        solver = self.profile["solver"]
        collision = self.profile.get("collision", {})
        option = ET.SubElement(self.root, "option", timestep=numbers([dt]), gravity=numbers(env_cfg.sim.gravity),
                      integrator=solver["integrator"], solver=solver["solver"], iterations=str(int(solver["iterations"])),
                      ls_iterations=str(int(solver.get("ls_iterations", 50))), tolerance=numbers([solver.get("tolerance", 1e-8)]),
                      cone=solver["cone"], impratio=numbers([solver["impratio"]]),
                      **({"ccd_iterations": str(int(collision["ccd_iterations"])),
                          "ccd_tolerance": numbers([collision["ccd_tolerance"]])} if collision else {}))
        if collision.get("multiccd") is False:    # profile collision.multiccd False: multi-point convex collision off
            ET.SubElement(option, "flag", multiccd="disable")
        ET.SubElement(self.root, "size", memory=str(self.profile.get("memory", "256M")))
        self.default = ET.SubElement(self.root, "default")
        self.asset = ET.SubElement(self.root, "asset")
        self.world = ET.SubElement(self.root, "worldbody")
        self.contact = ET.SubElement(self.root, "contact")
        self.equality = ET.SubElement(self.root, "equality")
        self.actuator = ET.SubElement(self.root, "actuator")
        self.meshes = _Meshes(self.out, self.asset)
        self.entities: dict[str, dict] = {}
        self.welds: list[dict] = []
        self.hulls: list[dict] = []
        self.collision_identities: list[str] = []      # MJCF entities with their own collision bits (mjcf_entity.py)

    # -- shared contact parameters
    def _contact_attrs(self, role: str, friction=None) -> dict:
        """solref / solimp / condim / friction of a collision geom. role: 'robot', 'task', 'table'."""
        c = self.profile["contact"]
        if role == "robot":       # a family may give the robot's own geoms other contact parameters (profile robot.contact)
            c = {**c, **self.profile["robot"].get("contact", {})}
        attrs = {"solref": numbers(c["solref"]), "solimp": numbers(c["solimp"])}
        if role != "robot":       # the robot keeps its MJCF's own condim / friction (ALOHA: condim 6, mu 1)
            mu = float(c["table_friction"] if role == "table" else (friction if friction is not None else c["friction"]))
            attrs["friction"] = numbers([mu, c["torsional_friction"], c["rolling_friction"]])
            attrs["condim"] = str(int(c["condim"]))
        if c.get("margin"):
            attrs["margin"] = numbers([c["margin"]])
        return attrs

    # -- the shared tabletop
    def add_tabletop(self):
        world = self.env_cfg.world
        if bool(world.plane.enabled):
            body = ET.SubElement(self.world, "body", name="ground")
            ET.SubElement(body, "geom", name="ground", type="plane", size="0 0 1", **self._contact_attrs("table"))
            self.entities["ground"] = dict(kind="static", root="ground", links=[dict(name="ground", body="ground", parent=None)],
                                           joints=[], free_joint=None, collision=True)
        t = world.table
        body = ET.SubElement(self.world, "body", name="table", pos=numbers(t.pos))
        ET.SubElement(body, "geom", name="table", type="box", size=numbers(np.asarray(t.size, dtype=float) / 2),
                      rgba=numbers([*t.color, 1.]), **self._contact_attrs("table"))
        self.entities["table"] = dict(kind="static", root="table", links=[dict(name="table", body="table", parent=None)],
                                      joints=[], free_joint=None, collision=True)

    # -- the robot: its own MJCF (the file Genesis loads), placed at the mount
    def add_robot(self, robot_xml: Path, *, home: dict):
        src = ET.parse(robot_xml).getroot()
        compiler = src.find("compiler")
        meshdir = Path(compiler.get("meshdir", "")) if compiler is not None else Path()
        if not meshdir.is_absolute():
            meshdir = Path(robot_xml).parent / meshdir
        from topogym.assets.robot import asset_root
        roots = [(ROBOT_ASSETS, asset_root(self.robot_cfg.robot.get("asset_root")).resolve())]
        if _genesis_assets() is not None:
            roots.append((GENESIS_ASSETS, _genesis_assets()))
        for node in src.findall("asset/mesh"):
            path = (meshdir / node.get("file")).resolve()
            token = next(((t, root) for t, root in roots if path.is_relative_to(root)), None)
            if token is None:
                raise ValueError(f"robot mesh {path} lies outside the robot asset roots: a frozen scene cannot name it")
            node.set("file", token[0] + path.relative_to(token[1]).as_posix())
            node.set("name", node.get("name") or Path(node.get("file")).stem)
        for tag, target in (("default", self.default), ("asset", self.asset), ("contact", self.contact),
                            ("equality", self.equality)):
            section = src.find(tag)
            if section is not None:
                target.extend(copy.deepcopy(list(section)))
        coupling = self.profile["robot"].get("coupling")
        if coupling:    # profile robot.coupling: the robot MJCF's joint couplings (the ALOHA's follower finger)
            for eq in self.equality.findall("joint"):
                eq.set("solref", numbers(coupling["solref"]))
                eq.set("solimp", numbers(coupling["solimp"]))
        own_contact = self._genesis_robot_import()
        mount = self.env_cfg.robot_mount
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("xyz", np.asarray(mount.euler, dtype=float), degrees=True).as_matrix()
        T[:3, 3] = np.asarray(mount.pos, dtype=float)
        base = ET.SubElement(self.world, "body", name="robot", **_pose_attrs(T))
        for node in src.find("worldbody"):
            if node.tag == "body":
                base.append(copy.deepcopy(node))
        links, joints, hull_meshes = [dict(name="robot", body="robot", parent=None)], [], set()
        armature = self.profile["robot"].get("armature")
        self_contact = self.profile["robot"].get("self_contact")
        self_contact_pads = bool(self.profile["robot"].get("self_contact_pads"))
        from .robot import FRICTIONLESS_LINKS
        frictionless = set(FRICTIONLESS_LINKS.get(self.description.robot_id, ()))
        for parent in base.iter("body"):
            for body in parent.findall("body"):
                body.set("gravcomp", "1")       # Genesis robot material: gravity_compensation=1.0
                links.append(dict(name=body.get("name"), body=body.get("name"), parent=parent.get("name")))
                for j in body.findall("joint"):
                    if armature is not None:
                        j.set("armature", numbers([armature]))
                    # The drive's force limit is the actuator's forcerange (Genesis: the dof force range), not a joint clamp.
                    j.set("actuatorfrclimited", "false")
                    joints.append(dict(name=j.get("name"), mj=j.get("name")))
                for g in body.findall("geom"):
                    collides = g.get("class") != "visual" and not (g.get("contype") == "0" and g.get("conaffinity") == "0")
                    if collides and g.get("mesh"):
                        hull_meshes.add(g.get("mesh"))
                    own = g.get("class") in own_contact     # the MJCF's own contact parameters (Genesis keeps them too)
                    if collides and not own:
                        for k, v in self._contact_attrs("robot").items():
                            g.set(k, v)
                    if collides and self_contact and (self_contact_pads or not own) and body.get("name") not in frictionless:
                        # robot-robot contacts get the robot's own class; with solmix 0 a contact with any other geom
                        # takes that geom's solref / solimp unchanged (MuJoCo mixes by solmix; friction stays the
                        # maximum of the two). own_contact geoms (the grasp pads) keep their class and solmix 1 unless
                        # robot.self_contact_pads: their grasps of task bodies keep the mean of both classes.
                        g.set("solref", numbers(self_contact["solref"]))
                        g.set("solimp", numbers(self_contact["solimp"]))
                        g.set("solmix", "0")
                    if collides and body.get("name") in frictionless:
                        # an internal linkage part (robot.py FRICTIONLESS_LINKS): its contacts carry no friction, and
                        # with priority 1 its condim 1 wins over any other geom's (MuJoCo would take the larger). Its
                        # solref / solimp win too, so it keeps the robot's task-contact class, never robot.self_contact's
                        # soft one (fiber: a spring link pinched into the cable sank 19 mm on 5 ms / .95)
                        g.set("condim", "1")
                        g.set("friction", "0 0 0")
                        g.set("priority", "1")
        # Small collision hulls (profile robot.maxhullvert): MuJoCo collides a mesh as its convex hull; the cap keeps a
        # subset of the full hull's vertices (inside it). The meshes stay whole for anything else (visual geoms).
        limit, skip = self.profile["robot"].get("maxhullvert"), tuple(self.profile["robot"].get("maxhullvert_skip", ()))
        if limit:
            for node in self.asset.findall("mesh"):
                if node.get("name") in hull_meshes and not node.get("name").startswith(skip):
                    node.set("maxhullvert", str(int(limit)))
        # Genesis drives every controlled dof with its own PD (kp, kv, force range): one general actuator per dof.
        actuators = []
        for g in self.description.groups:
            n = len(g.arm_joints)
            for i, joint in enumerate(list(g.arm_joints) + list(g.actuated_fingers)):
                lo, hi = (g.arm_limits_rad[i] if i < n else g.gripper_limits)
                kp, kv, force = float(g.kp[i]), float(g.kv[i]), float(g.force[i])
                name = f"{g.name}/drive/{joint.split('/')[-1]}"
                ET.SubElement(self.actuator, "general", name=name, joint=joint, gaintype="fixed", gainprm=numbers([kp]),
                              biastype="affine", biasprm=numbers([0., -kp, -kv]), ctrllimited="true",
                              ctrlrange=numbers([lo, hi]), forcelimited="true", forcerange=numbers([-force, force]))
                actuators.append(dict(name=name, group=g.name, joint=joint, kp=kp, kv=kv, force=force))
        self.entities["robot"] = dict(kind="robot", root="robot", links=links, joints=joints, free_joint=None,
                                      collision=True, home=home)
        self.robot = dict(mount_body="robot", actuators=actuators, source_xml=Path(robot_xml).name,
                          source_sha256=sha256_file(robot_xml))

    def _genesis_robot_import(self) -> set:
        """The robot as Genesis 1.2.1 imports its MJCF, where the profile asks for it (profile["robot"]):
          own_contact     robot geoms whose default class sets solref keep their own solref / solimp (Genesis reads the
                          MJCF's per-geom contact parameters; the Robotiq pads: 4 ms, .95 .99 .001)
          no_priority     contact priority dropped (Genesis averages both geoms' parameters)
          pad_collision   the Robotiq pad collision boxes end at the visual pad face (the Menagerie boxes stood 1.4 mm
                          proud of the visual pad backing; the grasp face at -6.6 mm is unchanged; L3 2026-10-09)
        Returns the classes whose contact parameters stay the MJCF's own."""
        r = self.profile["robot"]
        keep = set()
        for d in self.default.iter("default"):
            g = d.find("geom")
            if g is not None and r.get("own_contact") and g.get("solref") is not None:
                keep.add(d.get("class"))
            if g is not None and r.get("no_priority") and g.get("priority") is not None:
                del g.attrib["priority"]
            if g is not None and r.get("pad_collision") and d.get("class") in ("r85_pad_box1", "r85_pad_box2"):
                pos, size = np.fromstring(g.get("pos"), sep=" "), np.fromstring(g.get("size"), sep=" ")
                pos[1], size[1] = -.0033, .0033
                g.set("pos", numbers(pos))
                g.set("size", numbers(size))
        return keep

    # -- task entities
    def add_entity(self, spec):
        if spec.physics_model == "fem_elastic" and self.profile.get("rope") is not None:
            from .rope_chain import add_rope_chain     # a FEM rope becomes a rigid capsule chain (profile "rope")
            add_rope_chain(self, spec, self.fem_birth.get(spec.name))
            return
        if spec.physics_model == "pbd_cloth" and self.profile.get("cloth") is not None:
            from .cloth_vbd import is_vbd, add_cloth_vbd
            if is_vbd(self.profile):        # a VBD cloth (Newton) steps next to MuJoCo: nothing enters the MJCF
                add_cloth_vbd(self, spec, self.fem_birth.get(spec.name))
                return
            from .cloth import add_cloth    # a PBD cloth becomes a 2D flex over contact-sphere vertices (profile "cloth")
            add_cloth(self, spec, self.fem_birth.get(spec.name))
            return
        if spec.physics_model != "rigid":
            raise NotImplementedError(f"MuJoCo core: physics model {spec.physics_model} of {spec.name} is not ported")
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(np.asarray(spec.quaternion_wxyz, dtype=float)[[1, 2, 3, 0]]).as_matrix()
        T[:3, 3] = np.asarray(spec.position_m, dtype=float)
        options = dict(spec.morph_options)
        material = dict(spec.material)
        friction = material.get("friction")
        collide = options.get("collision", True)
        if spec.morph_type == "urdf":
            self._add_urdf(spec, T, friction, collide)
            return
        if spec.morph_type == "mjcf":
            from .mjcf_entity import merge_mjcf_entity      # the authored articulated tree (fiber_bundling)
            merge_mjcf_entity(self, spec)
            return
        body = ET.SubElement(self.world, "body", name=spec.name, **_pose_attrs(T))
        if not spec.fixed:
            ET.SubElement(body, "freejoint", name=f"{spec.name}{SEP}root")
        rho = material.get("rho")
        attrs = self._contact_attrs("task", friction) if collide else dict(contype="0", conaffinity="0", group="2")
        if rho is not None:
            attrs["density"] = numbers([rho])
        kind = spec.morph_type
        if kind == "box":
            ET.SubElement(body, "geom", type="box", size=numbers(np.asarray(options["size"], dtype=float) / 2), **attrs)
        elif kind == "sphere":
            ET.SubElement(body, "geom", type="sphere", size=numbers([options["radius"]]), **attrs)
        elif kind == "cylinder":
            ET.SubElement(body, "geom", type="cylinder", size=numbers([options["radius"], options["height"] / 2]), **attrs)
        elif kind == "mesh":
            import trimesh
            mesh = trimesh.load(spec.asset_path, force="mesh", process=False)
            scale = np.broadcast_to(np.asarray(options.get("scale", 1.), dtype=float), (3,))
            v = np.asarray(mesh.vertices, dtype=np.float64) * scale
            self._mesh_geoms(body, spec.name, v, np.asarray(mesh.faces), attrs, collide,
                             split=self.profile.get("split", {}).get(spec.name))
        else:
            raise ValueError(f"Unknown entity morph: {kind}")
        if not collide and spec.fixed is False:
            raise ValueError(f"{spec.name}: a movable entity needs collision geometry")
        self.entities[spec.name] = dict(kind="static" if spec.fixed else "rigid", root=spec.name,
                                        links=[dict(name=spec.name, body=spec.name, parent=None)], joints=[],
                                        free_joint=None if spec.fixed else f"{spec.name}{SEP}root", collision=bool(collide))

    def _mesh_geoms(self, body, stem, vertices, faces, attrs, collide, split=None):
        """A mesh geom; with collision, a non-convex mesh becomes the family's exact convex parts (profile "split") or
        CoACD hulls when the profile asks for it."""
        if collide and split:
            if "name" in attrs:
                raise NotImplementedError(f"{attrs['name']}: contact roles on split parts are not ported")
            parts, check = _family_parts(split, vertices, faces)
            for name in [self.meshes.add(pv, pf, f"{stem}_part") for pv, pf in parts]:
                ET.SubElement(body, "geom", type="mesh", mesh=name, **attrs)
            self.hulls.append(dict(geom=stem, parts=len(parts), split=split, check=check,
                                   source_vertices=int(len(vertices)), source_faces=int(len(faces))))
            return
        hulls = self.profile.get("coacd")
        if collide and hulls:
            if "name" in attrs:
                raise NotImplementedError(f"{attrs['name']}: contact roles on CoACD hulls are not ported")
            parts, key, cached = _coacd_hulls(vertices, faces, hulls, self.hull_cache)
            names = [self.meshes.add(pv, pf, f"{stem}_hull") for pv, pf in parts]
            for name in names:
                ET.SubElement(body, "geom", type="mesh", mesh=name, **attrs)
            self.hulls.append(dict(geom=stem, hulls=len(parts), settings=hulls, key=key, cached=cached,
                                   source_vertices=int(len(vertices)), source_faces=int(len(faces))))
            return
        name = self.meshes.add(vertices, faces, stem)
        ET.SubElement(body, "geom", type="mesh", mesh=name, **attrs)

    def _add_urdf(self, spec, T_root, friction, collide):
        import trimesh
        path = Path(spec.asset_path)
        root = ET.parse(path).getroot()
        links = {l.get("name"): l for l in root.findall("link")}
        joints = root.findall("joint")
        children = {j.find("child").get("link") for j in joints}
        roots = [n for n in links if n not in children]
        if len(roots) != 1:
            raise ValueError(f"{spec.name}: a URDF needs exactly one root link")
        by_parent = {}
        for j in joints:
            by_parent.setdefault(j.find("parent").get("link"), []).append(j)
        rho = dict(spec.material).get("rho")
        record = dict(kind="static" if spec.fixed else "rigid", root=f"{spec.name}{SEP}{roots[0]}", links=[], joints=[],
                      free_joint=None, collision=bool(collide))
        # Contact roles (EntitySpec.contact_roles) name authored collisions: those geoms carry the authored name
        # ("<entity>::<collision>"), and record["geoms"] maps it, as mjcf_entity does for an MJCF entity.
        names = {} if spec.contact_roles else None

        def add_link(parent_el, link_name, T, parent_link, joint):
            body = ET.SubElement(parent_el, "body", name=f"{spec.name}{SEP}{link_name}", **_pose_attrs(T))
            record["links"].append(dict(name=link_name, body=body.get("name"), parent=parent_link))
            if joint is None and parent_link is None and not spec.fixed:
                ET.SubElement(body, "freejoint", name=f"{spec.name}{SEP}root")
                record["free_joint"] = f"{spec.name}{SEP}root"
            elif joint is not None and joint.get("type") != "fixed":
                kind = joint.get("type")
                mj_kind = {"revolute": "hinge", "continuous": "hinge", "prismatic": "slide"}.get(kind)
                if mj_kind is None:
                    raise NotImplementedError(f"{spec.name}: URDF joint type {kind}")
                axis = joint.find("axis")
                attrs = dict(name=f"{spec.name}{SEP}{joint.get('name')}", type=mj_kind,
                             axis=axis.get("xyz", "1 0 0") if axis is not None else "1 0 0")
                limit = joint.find("limit")
                if kind != "continuous" and limit is not None and limit.get("lower") is not None:
                    attrs.update(limited="true", range=numbers([float(limit.get("lower")), float(limit.get("upper"))]))
                dynamics = joint.find("dynamics")
                if dynamics is not None:
                    attrs.update(damping=dynamics.get("damping", "0"), frictionloss=dynamics.get("friction", "0"))
                ET.SubElement(body, "joint", **attrs)
                record["joints"].append(dict(name=joint.get("name"), mj=attrs["name"]))
            link = links[link_name]
            inertial = link.find("inertial")
            if inertial is not None:
                O = _urdf_origin(inertial)
                I = inertial.find("inertia")
                m = np.array([[float(I.get("ixx")), float(I.get("ixy", 0)), float(I.get("ixz", 0))],
                              [float(I.get("ixy", 0)), float(I.get("iyy")), float(I.get("iyz", 0))],
                              [float(I.get("ixz", 0)), float(I.get("iyz", 0)), float(I.get("izz"))]])
                m = O[:3, :3] @ m @ O[:3, :3].T     # in the link frame, as fullinertia wants it
                ET.SubElement(body, "inertial", pos=numbers(O[:3, 3]), mass=inertial.find("mass").get("value"),
                              fullinertia=numbers([m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[0, 2], m[1, 2]]))
            attrs = self._contact_attrs("task", friction) if collide else dict(contype="0", conaffinity="0", group="2")
            if inertial is None and rho is not None:
                attrs = {**attrs, "density": numbers([rho])}
            else:
                attrs = {**attrs, "mass": "0"}
            pieces = link.findall("collision")
            # profile "capsules": a cylinder and the two spheres on its end caps become one capsule geom (the same solid)
            capsules = _urdf_capsules(pieces) if self.profile.get("capsules") and collide else {}
            merged = {j: i for i, pair in capsules.items() for j in pair}
            partition = self.profile.get("prism_partition")     # True, or its options ({"fatten_m": ...}, prisms.py)
            prisms = [] if collide and partition else None
            for index, piece in enumerate(pieces):
                G = _urdf_origin(piece)
                shape = list(piece.find("geometry"))[0]
                named = {}
                if names is not None and piece.get("name"):
                    if piece.get("name") in names:
                        raise ValueError(f"{spec.name}: collision name {piece.get('name')} is not unique")
                    owner = pieces[merged[index]].get("name") if index in merged else piece.get("name")
                    names[piece.get("name")] = named["name"] = f"{spec.name}{SEP}{owner}"
                if index in merged:
                    continue
                if shape.tag == "box":
                    ET.SubElement(body, "geom", type="box", size=numbers(np.fromstring(shape.get("size"), sep=" ") / 2),
                                  **attrs, **named, **_pose_attrs(G))
                elif shape.tag == "cylinder":
                    ET.SubElement(body, "geom", type="capsule" if index in capsules else "cylinder",
                                  size=numbers([float(shape.get("radius")), float(shape.get("length")) / 2]), **attrs, **named, **_pose_attrs(G))
                elif shape.tag == "sphere":
                    ET.SubElement(body, "geom", type="sphere", size=numbers([float(shape.get("radius"))]), **attrs, **named, **_pose_attrs(G))
                elif shape.tag == "mesh":
                    file = Path(shape.get("filename"))
                    file = file if file.is_absolute() else path.parent / file
                    mesh = trimesh.load(file, force="mesh", process=False)
                    v = np.asarray(mesh.vertices, dtype=np.float64) * np.fromstring(shape.get("scale", "1 1 1"), sep=" ")
                    v = v @ G[:3, :3].T + G[:3, 3]           # into the link frame: hull files then need no pose
                    if prisms is not None:
                        if named:
                            raise NotImplementedError(f"{named['name']}: contact roles on prism-partitioned pieces are not ported")
                        prisms.append(v)
                        continue
                    self._mesh_geoms(body, f"{spec.name}_{link_name}_{index}", v, np.asarray(mesh.faces), {**attrs, **named},
                                     collide, split=self.profile.get("split", {}).get(spec.name))
                else:
                    raise ValueError(f"{spec.name}: unsupported URDF collision shape {shape.tag}")
            if prisms:          # profile prism_partition: the link's slab prisms as fat exact convex prisms (prisms.py)
                from .prisms import convex_partition
                parts, audit = convex_partition(prisms, **(partition if isinstance(partition, dict) else {}))
                for v, f in parts:
                    ET.SubElement(body, "geom", type="mesh", mesh=self.meshes.add(v, f, f"{spec.name}_{link_name}_prism"),
                                  **attrs)
                self.hulls.append(dict(geom=f"{spec.name}_{link_name}", settings="prism_partition", **audit))
            for j in by_parent.get(link_name, []):
                add_link(body, j.find("child").get("link"), _urdf_origin(j), link_name, j)

        add_link(self.world, roots[0], T_root, None, None)
        # The entity's own joint equalities (morph_options "joint_equalities": [{name, joint1, optional joint2, polycoef
        # (5 values), optional active (default true), solref, solimp}]), names and joints prefixed as the joints are. An
        # MJCF entity authors them in its file (mjcf_entity.py); a URDF entity, whose glTF visuals Nyx draws with their
        # textures, declares them here (caliper_sorting's slider lock, switched by the task's mechanism).
        for eq in dict(spec.morph_options or {}).get("joint_equalities", ()):
            attrs = dict(name=f"{spec.name}{SEP}{eq['name']}", joint1=f"{spec.name}{SEP}{eq['joint1']}")
            if eq.get("joint2") is not None:
                attrs["joint2"] = f"{spec.name}{SEP}{eq['joint2']}"
            attrs.update(polycoef=numbers(eq["polycoef"]), active="true" if eq.get("active", True) else "false")
            attrs.update({k: numbers(eq[k]) for k in ("solref", "solimp") if k in eq})
            ET.SubElement(self.equality, "joint", **attrs)
            record.setdefault("equalities", []).append(attrs["name"])
        if names is not None:
            record["geoms"] = names
        if spec.initial_joint_positions:
            record["initial_joint_positions"] = {k: float(v) for k, v in spec.initial_joint_positions.items()}
        self.entities[spec.name] = record

    # -- permanent attachments (tasks.base.AttachmentSpec): a rope end welded to a rigid task entity
    def add_attachments(self, attachments):
        """Each attachment welds the rope chain's links whose material span starts within band_length_m of the endpoint
        to the rigid entity's root body, for good: body1 the rigid body, body2 the link, the weld point at the link's
        origin, the relative pose MuJoCo's qpos0 one (the birth), at the chain's own node-connect impedance (nearly hard;
        AttachmentSpec.stiffness_n_m is IPC's band stiffness and is not used). The rigid entity is static (a clamp) or
        movable. Only rope chains attach; the rope record lists the welded links (scene.json).
        Assumes the birth's link poses are the scene's qpos0 (as a frozen birth's are): a birth that moved the welded
        links off qpos0 would have them snap back toward their qpos0 pose on the first step."""
        for a in attachments:
            rope, rigid = self.entities.get(a.soft_entity), self.entities.get(a.rigid_entity)
            if rope is None or rope.get("kind") != "rope_chain":
                raise NotImplementedError(f"attachment {a.name}: only rope chains attach on MuJoCo ({a.soft_entity})")
            if rigid is None or rigid.get("kind") not in ("static", "rigid"):
                raise ValueError(f"attachment {a.name}: {a.rigid_entity} is not a rigid task entity")
            info = rope["rope"]
            n, link = int(info["links"]), float(info["link_m"])
            count = max(1, min(n, int(math.ceil(float(a.band_length_m) / link - 1e-9))))
            joint = info["settings"]["joint"]
            welded = []
            for k in (range(count) if a.endpoint == "start" else range(n - count, n)):
                body = rope["links"][k]["body"]
                ET.SubElement(self.equality, "weld", name=f"attach/{a.name}/{k:03d}", body1=rigid["root"], body2=body,
                              solref=numbers(joint["solref"]), solimp=numbers(joint["solimp"]))
                welded.append(body)
            info.setdefault("attachments", []).append(dict(name=a.name, rigid=a.rigid_entity, endpoint=a.endpoint,
                                                           band_length_m=float(a.band_length_m), links=welded))

    # -- assisted grasp: one inactive weld per (hand, target link), anchored at acquisition
    def add_grasp_welds(self, grasp_assist):
        if grasp_assist is None or grasp_assist.model == "none":
            return
        if grasp_assist.model == "pbd_local_patch_v1" and self.profile.get("cloth") is not None:
            return          # a cloth's grasp pins are written with the cloth (cloth.add_cloth)
        if grasp_assist.model != "contact_weld_v1":
            raise NotImplementedError(f"MuJoCo core: grasp model {grasp_assist.model} is not ported")
        w = self.profile["weld"]
        for g in self.description.groups:
            for target in grasp_assist.targets:
                record = self.entities.get(target.entity)
                if record is None:
                    raise ValueError(f"grasp target {target.entity} is not an entity")
                link = next((l for l in record["links"] if l["name"] == target.link), None)
                if link is None:
                    raise ValueError(f"grasp target link {target.link} not in {target.entity}")
                name = f"grasp/{g.name}/{target.entity}/{target.link}"
                ET.SubElement(self.equality, "weld", name=name, body1=g.hand_link, body2=link["body"], active="false",
                              solref=numbers(w["solref"]), solimp=numbers(w["solimp"]))
                self.welds.append(dict(name=name, hand=g.name, hand_body=g.hand_link, target=[target.entity, target.link],
                                       body=link["body"]))

    def _static_contacts(self):
        """profile static_contact {soft, keep}: the contacts of static bodies (no joint between them and the world: the
        table, the ground, task links fixed to the world). MuJoCo mixes two geoms' solref / solimp by their solmix: a
        solmix-0 geom takes the other's, two solmix-0 geoms their mean, 1e9 outweighs 1.
          soft (a class): a static body not kept carries it with solmix 0. A robot geom with robot.self_contact (solmix
                0) touching it then gets the class; a grasp pad (own_contact, solmix 1) keeps its own class and a task
                body its own parameters, so every task contact keeps the family's precision.
          keep {entity: reason}: static bodies that keep their own (task) parameters and win every contact with them
                (solmix KEEP_SOLMIX = 2^60: against a solmix-1 geom MuJoCo's weight 2^60 / (2^60 + 1) rounds to exactly
                1, so a task body with the same parameters, rope_threading's rope on its rings, keeps them bit for bit;
                1e9 left an ulp-level mix): thin static geometry, where a softer contact would let a robot geom pass its
                axis. A key is an fnmatch pattern on entity names (an exact name matches itself); each must match an
                entity.
          soft None: every static body keeps its parameters and wins."""
        sc = self.profile.get("static_contact")
        if not sc:
            return
        soft, keep = sc.get("soft"), dict(sc.get("keep", {}))
        if soft and not self.profile["robot"].get("self_contact"):
            raise ValueError("profile static_contact.soft needs robot.self_contact (both sides of a robot-static contact)")
        from fnmatch import fnmatchcase
        unknown = sorted(k for k in keep if not any(fnmatchcase(e, k) for e in self.entities))
        if unknown:
            raise ValueError(f"static_contact keeps unknown entities {unknown}")

        def visit(body, jointed):
            jointed = jointed or body.find("joint") is not None or body.find("freejoint") is not None
            if not jointed:
                kept = bool(soft) and any(fnmatchcase(body.get("name").split(SEP)[0], k) for k in keep)
                wins = not soft or kept
                for g in body.findall("geom"):
                    if g.get("contype") == "0" and g.get("conaffinity") == "0":
                        continue
                    if wins:
                        g.set("solmix", numbers([KEEP_SOLMIX if kept else 1e9]))
                    else:
                        g.set("solref", numbers(soft["solref"]))
                        g.set("solimp", numbers(soft["solimp"]))
                        g.set("solmix", "0")
            for child in body.findall("body"):
                visit(child, jointed)
        for body in self.world.findall("body"):
            if body.get("name") != "robot":           # the robot's own geoms: robot.self_contact
                visit(body, False)

    def _joint_rules(self):
        """Profile joints: [{match: an fnmatch pattern on the MuJoCo joint name, MJCF joint attributes...}], applied in
        order. A rule that matches no joint is an error."""
        from fnmatch import fnmatchcase
        for rule in self.profile.get("joints", []):
            attrs = {k: v for k, v in rule.items() if k != "match"}
            hits = [j for j in self.world.iter("joint") if fnmatchcase(j.get("name", ""), rule["match"])]
            if not hits:
                raise ValueError(f"profile joint rule {rule['match']!r} matches no joint")
            for j in hits:
                for k, v in attrs.items():
                    j.set(k, numbers(np.atleast_1d(np.asarray(v, dtype=float))))

    PAIR_ATTRS = ("condim", "friction", "solref", "solimp", "margin", "gap")      # each pairs entry sets all of them

    def _pairs(self) -> list | None:
        """Profile pairs: [{bodies: [[fnmatch patterns on body names], [patterns]], condim, friction (5 values), solref,
        solimp, margin, gap, optional solreffriction, optional geom_types, optional reason}]. Every geom of a task body the first list
        matches meets every geom of a task body the second list matches through an explicit MuJoCo <pair> with the
        entry's parameters (no mixing), and one <exclude> per body pair removes the automatic contacts the pairs
        replace. Only contacts MuJoCo would make anyway become pairs: a body pair its collision filter skips (one weld
        body, weld parent and child) or excludes already, and a geom pair whose contype / conaffinity (default classes
        resolved) never collide, stay as they are. A matched body's unnamed geoms are named "<body>#<index among its
        geoms>" (a <pair> names its geoms). A pattern matching no task body, a body in both lists, or an entry without
        any pair is an error. geom_types: [[MuJoCo geom types] or None, [...] or None], one per selector: only a matched
        body's geoms of these types (own type, else its class chain's, else MuJoCo's sphere) get pairs; the body pair's
        <exclude> still drops all automatic contacts between the two bodies, so a geom left out does not collide with the
        other side at all. Returns the audit for scene.json (None without the field)."""
        entries = self.profile.get("pairs")
        if not entries:
            return None
        from fnmatch import fnmatchcase
        parent = {child: el for el in self.root.iter() for child in el}
        classes = {d.get("class", "main"): d for d in self.default.iter("default")}
        robot = next((b for b in self.world.findall("body") if b.get("name") == "robot"), None)
        robot_bodies = set() if robot is None else {b.get("name") for b in robot.iter("body")}
        bodies = [b for b in self.world.iter("body") if b.get("name") not in robot_bodies]
        geom_names = {g.get("name") for g in self.root.iter("geom") if g.get("name")}
        excluded = {frozenset((e.get("body1"), e.get("body2"))) for e in self.contact.findall("exclude")}

        def bits(g):          # (contype, conaffinity): the geom's own value, else its class chain's, else MuJoCo's 1
            cls, el = g.get("class"), parent[g]
            while cls is None and el is not None:
                cls, el = el.get("childclass"), parent.get(el)
            out = []
            for key in ("contype", "conaffinity"):
                value, d = g.get(key), classes[cls or "main"]
                while value is None and d is not None:
                    gd = d.find("geom")
                    value = None if gd is None else gd.get(key)
                    d = parent.get(d) if d is not self.default else None
                out.append(1 if value is None else int(value))
            return out

        def gtype(g):         # the geom's type: its own, else its class chain's, else MuJoCo's default sphere
            value, cls, el = g.get("type"), g.get("class"), parent[g]
            while cls is None and el is not None:
                cls, el = el.get("childclass"), parent.get(el)
            d = classes[cls or "main"]
            while value is None and d is not None:
                gd = d.find("geom")
                value = None if gd is None else gd.get("type")
                d = parent.get(d) if d is not self.default else None
            return value or "sphere"

        def weld(b):          # the body b moves with: itself if it has a joint, else its parent's (None: the world)
            while b is not None and b.tag == "body" and b.find("joint") is None and b.find("freejoint") is None:
                b = parent.get(b)
            return b if b is not None and b.tag == "body" else None

        def skipped(b1, b2):  # MuJoCo's body-pair filter: one weld body; filterparent (unless one is welded to the world)
            w1, w2 = weld(b1), weld(b2)
            return w1 is w2 or (w1 is not None and w2 is not None
                                and (weld(parent[w2]) is w1 or weld(parent[w1]) is w2))

        audit = []
        for k, entry in enumerate(entries):
            unknown = sorted(set(entry) - {"bodies", "reason", "solreffriction", "geom_types", *self.PAIR_ATTRS})
            missing = [a for a in self.PAIR_ATTRS if a not in entry]
            if unknown or missing or len(entry["bodies"]) != 2:
                raise ValueError(f"profile pairs[{k}]: two body selectors and {self.PAIR_ATTRS} (unknown {unknown}, "
                                 f"missing {missing})")
            sides = []
            for patterns in entry["bodies"]:
                for p in patterns:
                    if not any(fnmatchcase(b.get("name"), p) for b in bodies):
                        raise ValueError(f"profile pairs[{k}]: pattern {p!r} matches no task body")
                sides.append([b for b in bodies if any(fnmatchcase(b.get("name"), p) for p in patterns)])
            both = sorted({b.get("name") for b in sides[0]} & {b.get("name") for b in sides[1]})
            if both:
                raise ValueError(f"profile pairs[{k}]: bodies {both} in both selectors")
            attrs = {a: str(int(entry[a])) if a == "condim" else numbers(np.atleast_1d(np.asarray(entry[a], dtype=float)))
                     for a in (*self.PAIR_ATTRS, "solreffriction") if a in entry}
            types = entry.get("geom_types") or [None, None]
            if len(types) != 2 or any(t is not None and (not t or not all(isinstance(x, str) for x in t)) for t in types):
                raise ValueError(f"profile pairs[{k}]: geom_types is [[types] or None, [types] or None]")
            record = dict(bodies=[[b.get("name") for b in side] for side in sides], pairs=0, body_pairs=0,
                          filtered_body_pairs=0, incompatible_geom_pairs=0)
            if entry.get("geom_types"):
                record["geom_types"] = types
            take = lambda g, side: types[side] is None or gtype(g) in types[side]
            for b1 in sides[0]:
                for b2 in sides[1]:
                    if skipped(b1, b2) or frozenset((b1.get("name"), b2.get("name"))) in excluded:
                        record["filtered_body_pairs"] += 1
                        continue
                    n = 0
                    for g1 in (g for g in b1.findall("geom") if take(g, 0)):
                        for g2 in (g for g in b2.findall("geom") if take(g, 1)):
                            (t1, a1), (t2, a2) = bits(g1), bits(g2)
                            if not (t1 & a2 or t2 & a1):
                                record["incompatible_geom_pairs"] += 1
                                continue
                            for b, g in ((b1, g1), (b2, g2)):
                                if not g.get("name"):
                                    name = f"{b.get('name')}#{b.findall('geom').index(g)}"
                                    if name in geom_names:
                                        raise ValueError(f"profile pairs[{k}]: geom name {name} is taken")
                                    g.set("name", name)
                                    geom_names.add(name)
                            ET.SubElement(self.contact, "pair", geom1=g1.get("name"), geom2=g2.get("name"), **attrs)
                            n += 1
                    if n:
                        ET.SubElement(self.contact, "exclude", body1=b1.get("name"), body2=b2.get("name"))
                        excluded.add(frozenset((b1.get("name"), b2.get("name"))))
                        record["pairs"] += n
                        record["body_pairs"] += 1
            if not record["pairs"]:
                raise ValueError(f"profile pairs[{k}]: no geom pair would collide")
            audit.append(record)
        return audit

    PRISM_FACETS = 32

    def _cylinder_prisms(self) -> list:
        """Every cylinder geom outside the robot that a robot geom can collide with (collision bits, default classes
        resolved) becomes a convex prism of PRISM_FACETS facets with its vertices on the radius (vertex k at angle
        2 pi k / PRISM_FACETS from the geom's local x axis, the axis its local z; a fromto cylinder takes the shortest
        rotation of z onto its axis), the same pose, and the cylinder's mass (density x pi r^2 x length) unless the geom
        states its own. MuJoCo has no analytic cylinder test against a convex hull: its GJK / EPA returned, near a flat
        cap's rim, the right depth with a sign-flipped normal and the row pushed nothing (knot easy_v8 w1_e10: a finger
        11 mm into a post; FB3: fiber's sheath against a ring segment); a prism collides as a polytope. Built scenes only:
        frozen scenes keep their cylinders. Returns the audit [{geom, body, r_m, half_length_m}]."""
        parent = {child: el for el in self.root.iter() for child in el}
        classes = {d.get("class", "main"): d for d in self.default.iter("default")}

        def attr(g, key):
            cls, el = g.get("class"), parent[g]
            while cls is None and el is not None:
                cls, el = el.get("childclass"), parent.get(el)
            value, d = g.get(key), classes[cls or "main"]
            while value is None and d is not None:
                gd = d.find("geom")
                value = None if gd is None else gd.get(key)
                d = parent.get(d) if d is not self.default else None
            return value

        def bits(g):
            return int(attr(g, "contype") or 1), int(attr(g, "conaffinity") or 1)

        robot = next((b for b in self.world.findall("body") if b.get("name") == "robot"), None)
        robot_geoms = set() if robot is None else set(robot.iter("geom"))
        robot_bits = {bits(g) for g in robot_geoms}
        n = self.PRISM_FACETS
        ang = 2 * np.pi * np.arange(n) / n
        faces = [[k, (k + 1) % n, n + k] for k in range(n)] + [[(k + 1) % n, n + (k + 1) % n, n + k] for k in range(n)]
        faces += [[0, k + 1, k] for k in range(1, n - 1)] + [[n, n + k, n + k + 1] for k in range(1, n - 1)]
        audit = []
        for g in list(self.world.iter("geom")):
            if g in robot_geoms or (attr(g, "type") or "sphere") != "cylinder":
                continue
            ct, ca = bits(g)
            if not any((ct & ra) or (rt & ca) for rt, ra in robot_bits):
                continue
            size = np.fromstring(attr(g, "size") or "", sep=" ")
            r = float(size[0])
            if g.get("fromto") is not None:
                p0, p1 = np.split(np.fromstring(g.get("fromto"), sep=" "), 2)
                pos, axis, h = (p0 + p1) / 2, p1 - p0, float(np.linalg.norm(p1 - p0)) / 2
                q = np.zeros(4)
                mujoco_quat_z_to(q, axis / np.linalg.norm(axis))
            else:
                if any(g.get(k) is not None for k in ("euler", "xyaxes")):
                    raise NotImplementedError(f"cylinder prism: {g.get('name')} states euler / xyaxes")
                h = float(size[1])
                pos = np.fromstring(g.get("pos") or "0 0 0", sep=" ")
                q = np.fromstring(g.get("quat") or "1 0 0 0", sep=" ")
                if g.get("axisangle") is not None:
                    aa = np.fromstring(g.get("axisangle"), sep=" ")
                    q = np.r_[np.cos(aa[3] / 2), np.sin(aa[3] / 2) * aa[:3] / np.linalg.norm(aa[:3])]
                elif g.get("zaxis") is not None:
                    q = np.zeros(4)
                    z = np.fromstring(g.get("zaxis"), sep=" ")
                    mujoco_quat_z_to(q, z / np.linalg.norm(z))
            v = np.array([[r * np.cos(t), r * np.sin(t), z] for z in (-h, h) for t in ang])
            if g.get("mass") is None and attr(g, "mass") is None:
                g.set("mass", numbers([float(attr(g, "density") or 1000.) * np.pi * r * r * 2 * h]))
            for k in ("fromto", "size", "axisangle", "zaxis", "quat", "pos"):
                g.attrib.pop(k, None)
            name = g.get("name") or f"cyl{len(audit)}"
            g.set("type", "mesh")
            g.set("mesh", self.meshes.add(v, faces, f"{name.replace(SEP, '_')}_prism{n}"))
            g.set("pos", numbers(pos))
            g.set("quat", numbers(q))
            audit.append(dict(geom=g.get("name"), body=parent[g].get("name"), r_m=r, half_length_m=h))
        return audit

    def build(self) -> dict:
        prisms = self._cylinder_prisms()
        self._static_contacts()
        self._joint_rules()
        pairs = self._pairs()
        ET.indent(self.root)
        xml = self.out / "scene.xml"
        xml.write_text(ET.tostring(self.root, encoding="unicode"))
        files = {p.relative_to(self.out).as_posix(): sha256_file(p) for p in sorted(self.out.rglob("*"))
                 if p.is_file() and p.name != "scene.json"}
        import mujoco
        manifest = dict(format=SCENE_FORMAT, xml="scene.xml", mujoco_version=mujoco.__version__, profile=self.profile,
                        entities=self.entities, robot=self.robot, welds=self.welds, hulls=self.hulls, files=files)
        if pairs is not None:
            manifest["pairs"] = pairs
        if prisms:
            manifest["cylinder_prisms"] = prisms
        (self.out / "scene.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
        return manifest


def default_hull_cache() -> Path:
    """CoACD decompositions are cached by input (mesh bytes and settings), next to the robot asset cache."""
    base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return base / "topogym" / "mujoco-hulls"


def build_scene(out_dir, *, env_cfg, robot_cfg, description, specs, grasp_assist, profile, home, hull_cache=None,
                fem_birth=None, attachments=()) -> dict:
    """Write the instance's MuJoCo scene into out_dir (scene.xml, meshes/, scene.json); return the manifest."""
    from topogym.assets.robot import robot_asset
    hull_cache = default_hull_cache() if hull_cache is None else Path(hull_cache)
    builder = SceneBuilder(out_dir, env_cfg=env_cfg, robot_cfg=robot_cfg, description=description, profile=profile,
                           hull_cache=hull_cache, fem_birth=fem_birth)
    builder.add_tabletop()
    builder.add_robot(Path(robot_asset(robot_cfg)), home=home)
    for spec in specs:
        builder.add_entity(spec)
    builder.add_attachments(attachments)
    builder.add_grasp_welds(grasp_assist)
    return builder.build()


KEEP_SOLMIX = 2. ** 60       # static_contact.keep's weight: exactly 1 against any solmix <= 1 (see _static_contacts)


def migrate_contacts(m, profile) -> dict:
    """A frozen scene's contact classes as the profile builds them now (MujocoWorld.build, after load): geoms carrying
    profiles.SOFT_CONTACT_V1 take SOFT_CONTACT (constant impedance, 2026-10-11), and the static bodies of the profile's
    static_contact.keep patterns that a scene built softer (solmix 0) take the task's contact parameters and win
    (solmix KEEP_SOLMIX), as _static_contacts sets them. A scene built on the current profile is unchanged. Returns
    the counts."""
    from fnmatch import fnmatchcase
    from .profiles import SOFT_CONTACT, SOFT_CONTACT_V1
    out = dict(soft=0, keep=0)
    old_ref, old_imp = np.asarray(SOFT_CONTACT_V1["solref"]), np.asarray(SOFT_CONTACT_V1["solimp"])
    for g in range(m.ngeom):
        if np.allclose(m.geom_solref[g], old_ref) and np.allclose(m.geom_solimp[g], old_imp):
            m.geom_solimp[g] = SOFT_CONTACT["solimp"]
            out["soft"] += 1
    sc = profile.get("static_contact") or {}
    keep = sc.get("keep") or {}
    if keep and sc.get("soft"):
        import mujoco
        c = profile["contact"]
        for b in range(1, m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if int(m.body_weldid[b]) != 0 or not any(fnmatchcase(name.split(SEP)[0], k) for k in keep):
                continue
            for g in range(int(m.body_geomadr[b]), int(m.body_geomadr[b]) + int(m.body_geomnum[b])):
                if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_solmix[g] == 0:
                    m.geom_solref[g], m.geom_solimp[g], m.geom_solmix[g] = c["solref"], c["solimp"], KEEP_SOLMIX
                    out["keep"] += 1
    return out


def load_scene(directory, *, robot_asset_root=None, verify=True, excludes=()):
    """(MjModel, manifest) of a scene directory (built or frozen). verify: every file must match its sha256. excludes:
    (body1, body2) pairs whose contacts the compiled model leaves out (MuJoCo <exclude>; the robot's
    SELF_CONTACT_EXCLUDES), added in memory to every scene that has both bodies; the scene's files are unchanged."""
    import mujoco
    from topogym.assets.robot import asset_root
    directory = Path(directory)
    manifest = json.loads((directory / "scene.json").read_text())
    if manifest.get("format") != SCENE_FORMAT:
        raise ValueError(f"not a {SCENE_FORMAT} scene: {directory}")
    if verify:
        bad = [name for name, digest in manifest["files"].items() if sha256_file(directory / name) != digest]
        if bad:
            raise ValueError(f"MuJoCo scene files differ from their manifest: {bad[:4]}")
    root = ET.parse(directory / manifest["xml"]).getroot()
    robot_root = asset_root(robot_asset_root).resolve()
    needed = [n.get("file")[len(GENESIS_ASSETS):] for n in root.iter("mesh") if (n.get("file") or "").startswith(GENESIS_ASSETS)]
    sources = {prefix: vendored_source(prefix, robot_root, [r[len(prefix):] for r in needed if r.startswith(prefix)])
               for prefix in GENESIS_VENDORED if any(r.startswith(prefix) for r in needed)}
    for node in root.iter("mesh"):
        file = node.get("file")
        if file is None:
            continue
        if file.startswith(ROBOT_ASSETS):
            node.set("file", str(robot_root / file[len(ROBOT_ASSETS):]))
        elif file.startswith(GENESIS_ASSETS):
            node.set("file", str(_genesis_asset(file[len(GENESIS_ASSETS):], robot_root, sources)))
        elif not Path(file).is_absolute():
            node.set("file", str((directory / file).resolve()))
    bodies = {b.get("name") for b in root.iter("body")}
    pairs = [(a, b) for a, b in excludes if a in bodies and b in bodies]
    if pairs:
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        for a, b in pairs:
            ET.SubElement(contact, "exclude", body1=a, body2=b)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    manifest["robot_asset_sources"] = sources      # in memory only (scene.json is unchanged): the Session records it
    return model, manifest
