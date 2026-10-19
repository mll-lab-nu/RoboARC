"""Merge a task's authored MJCF entity (EntitySpec.morph_type == "mjcf") into the MuJoCo scene.

fiber_bundling declares one (bundle.xml, the fixed terminal, slack lead, sheath and fine strands as one articulated
tree), and so do slotted_post_ring's and orbits' rigid chains (cord, ring, beads, discs, handle; several free roots, the
#108 fixture in the worldbody, file excludes, an inline disc mesh, orbits' <connect> tie). The file already is MJCF, so
the tree is kept exactly: every body, inertia, joint field, collision primitive and authored contact parameter (unless
the profile's mjcf / mjcf_equality replace the softness). Genesis loads the same file. What changes on the way
in (see merge_mjcf_entity for the collision-bit remap and the profile's birth_overlap_exclude):

  - names get the entity prefix ("<entity>::<name>", core.scene.SEP) for bodies, joints, geoms and sites;
  - the file's <default> becomes the class "<entity>" (nested classes "<entity>/<class>"), applied as childclass;
  - top-level bodies are placed at the EntitySpec pose;
  - geoms that neither collide nor carry mass (visual copies, material meshes, copper filaments) are dropped: rendering
    is Nyx's (render bridge), and MuJoCo would only spend narrowphase work on them;
  - every parent-child body pair is excluded from contact. That is Genesis' default adjacency rule; MuJoCo's own
    parent filter skips pairs whose parent is welded to the world (the fixed terminal -> first lead link), which
    produced a 24 mm birth contact in L12's first import.

Returns the entity record (core's scene.json "entities" schema) plus the authored-name -> MuJoCo-name map of the
colliding geoms, so contact roles declared on authored names (EntitySpec.contact_roles) resolve exactly.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

SEP = "::"
NAMED = ("body", "joint", "geom", "site", "camera", "light")


def _numbers(values) -> str:
    return " ".join(format(float(x), ".17g") for x in np.asarray(values, dtype=float).reshape(-1))


def _pose(element) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.fromstring(element.get("pos", "0 0 0"), sep=" ")
    if element.get("quat") is not None:
        q = np.fromstring(element.get("quat"), sep=" ")
        T[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]] / np.linalg.norm(q)).as_matrix()
    elif any(element.get(k) is not None for k in ("euler", "axisangle", "xyaxes", "zaxis")):
        raise NotImplementedError("mjcf entity: body orientation must be a quaternion")
    return T


def _collides(attrs: dict) -> bool:
    return bool(int(attrs.get("contype", "1")) or int(attrs.get("conaffinity", "1")))


# Genesis' collision-mask semantics (genesis 1.2.1 rigid_entity.py `_is_local_collision_mask = isinstance(morph, MJCF)`,
# collider.py `con_skip = (same_entity | ~has_local_mask) & (con_match == 0)`): an MJCF entity's contype / conaffinity act
# only inside it, and across entities every pair collides (the ALOHA is an MJCF entity too). MuJoCo applies the bits
# globally. An entity whose colliding geoms use bits other than 1 / 1 is remapped so MuJoCo reproduces that:
#   contype = (ct << COLLISION_SHIFT) | id,  conaffinity = (ca << COLLISION_SHIFT) | 1 | (IDENTITY_BITS & ~id)
# with id one bit of IDENTITY_BITS per remapped entity. Inside the entity only the authored bits match; across two
# remapped entities the identity bit matches; against the 1 / 1 world (robot, table, URDF fixtures) bit 0 matches.
COLLISION_SHIFT = 16
IDENTITY_BITS = ((1 << COLLISION_SHIFT) - 1) & ~1


def _class_geoms(top_default) -> dict:
    """{authored default class (None: the file's top level): geom attributes}, a nested class inheriting its parent's."""
    out = {None: {}}

    def walk(node, name, inherited):
        geom = node.find("geom")
        out[name] = {**inherited, **(geom.attrib if geom is not None else {})}
        for sub in node.findall("default"):
            walk(sub, sub.get("class"), out[name])
    if top_default is not None:
        walk(top_default, None, {})
    return out


def _source_geoms(element, active):
    """(geom, its default class) below a worldbody or body of the authored file."""
    for child in element:
        if child.tag == "geom":
            yield child, child.get("class") or active
        elif child.tag == "body":
            yield from _source_geoms(child, child.get("childclass") or active)


def _bits(geom, klass, class_geoms) -> tuple[int, int]:
    attrs = {**class_geoms[klass], **{k: v for k, v in geom.attrib.items() if k != "class"}}
    return int(attrs.get("contype", "1")), int(attrs.get("conaffinity", "1"))


def merge_mjcf_entity(builder, spec, *, drop_visual=True) -> dict:
    """Add spec's MJCF tree to builder (core's SceneBuilder: .world, .default, .asset, .contact, .meshes, .entities).

    The profile may set contact attributes on every geom of an entity (profile mjcf: {entity: {contact: {attribute:
    value}}}): on its classes and on the geoms that set them themselves."""
    path = Path(spec.asset_path)
    src = ET.parse(path).getroot()
    if src.tag != "mujoco":
        raise ValueError(f"{spec.name}: not an MJCF file")
    compiler = src.find("compiler")
    if compiler is not None and compiler.get("angle", "degree") != "radian":
        raise NotImplementedError(f"{spec.name}: the MJCF entity must use radians")
    if src.find("include") is not None or src.find("tendon") is not None:
        raise NotImplementedError(f"{spec.name}: includes and tendons are not merged")
    name = spec.name
    cls = name
    options = builder.profile.get("mjcf", {}).get(name, {})
    contact = {k: _numbers(np.atleast_1d(np.asarray(v, dtype=float))) for k, v in options.get("contact", {}).items()}

    # defaults: the file's top-level <default> -> class "<entity>", nested classes "<entity>/<class>"
    top_default = src.find("default")
    default = ET.SubElement(builder.default, "default", {"class": cls})
    geom_defaults = {}
    if top_default is not None:
        for child in top_default:
            if child.tag == "default":
                nested = copy.deepcopy(child)
                for d in nested.iter("default"):
                    d.set("class", f"{cls}/{d.get('class')}")
                default.append(nested)
            else:
                default.append(copy.deepcopy(child))
                if child.tag == "geom":
                    geom_defaults = dict(child.attrib)

    for g in default.iter("geom"):          # the entity's classes: the profile's contact attributes
        for k, v in contact.items():
            g.set(k, v)
        if g is default.find("geom"):
            geom_defaults.update(contact)
    if contact and default.find("geom") is None:
        ET.SubElement(default, "geom", contact)

    # assets: only meshes that a kept geom uses (colliding meshes), re-registered through the builder
    meshes = {m.get("name") or Path(m.get("file")).stem: m for m in src.findall("asset/mesh")}
    mesh_names = {}

    def mesh_for(key):
        if key not in mesh_names:
            import trimesh
            node = meshes[key]
            if node.get("file") is not None:
                mesh = trimesh.load(path.parent / node.get("file"), force="mesh", process=False)
            elif node.get("vertex") is not None and node.get("face") is None:
                # An inline mesh of vertices only (#108's disc prism): MuJoCo collides a mesh as its convex hull, and
                # builds a vertex-only mesh as that hull, so the hull is the same solid.
                mesh = trimesh.convex.convex_hull(np.fromstring(node.get("vertex"), sep=" ").reshape(-1, 3))
            else:
                raise NotImplementedError(f"{name}: mesh {key} is neither a file nor inline vertices")
            scale = np.fromstring(node.get("scale", "1 1 1"), sep=" ")
            mesh_names[key] = builder.meshes.add(np.asarray(mesh.vertices) * scale, mesh.faces, f"{name}_{key}")
        return mesh_names[key]

    T_spec = np.eye(4)
    T_spec[:3, :3] = Rotation.from_quat(np.asarray(spec.quaternion_wxyz, dtype=float)[[1, 2, 3, 0]]).as_matrix()
    T_spec[:3, 3] = np.asarray(spec.position_m, dtype=float)
    record = dict(kind="articulated", root=None, links=[], joints=[], free_joint=None, collision=True,
                  source=path.name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), geoms={},
                  dropped_geoms=0)
    excludes = []

    def convert(body, parent_name, active):
        authored = body.get("name")
        if not authored:
            raise ValueError(f"{name}: every body of an MJCF entity needs a name")
        body.set("name", f"{name}{SEP}{authored}")
        record["links"].append(dict(name=authored, body=body.get("name"), parent=parent_name))
        if parent_name is not None:
            excludes.append((f"{name}{SEP}{parent_name}", body.get("name")))
        active = body.get("childclass") or active
        if body.get("childclass") is not None:
            body.set("childclass", f"{cls}/{body.get('childclass')}")
        for child in list(body):
            if child.tag == "body":
                continue                     # converted below, with its authored name
            if child.tag == "geom":
                ct, ca = _bits(child, child.get("class") or active, class_geoms)
            if child.get("class") is not None:
                child.set("class", f"{cls}/{child.get('class')}")
            if child.tag in ("joint", "freejoint"):
                if (child.tag == "freejoint" or child.get("type") == "free") and parent_name is not None:
                    raise NotImplementedError(f"{name}: a free joint below the top level of an MJCF entity")
                if not child.get("name"):
                    raise ValueError(f"{name}: every joint of an MJCF entity needs a name")
                # A free joint (a free root: slotted's ring, beads and middle chain link, orbits' handle) is recorded
                # like any other joint; EntityView maps it as 7 q / 6 dofs.
                record["joints"].append(dict(name=child.get("name"), mj=f"{name}{SEP}{child.get('name')}"))
                child.set("name", f"{name}{SEP}{child.get('name')}")
            elif child.tag == "geom":
                attrs = {**geom_defaults, **child.attrib}
                if drop_visual and not _collides(attrs) and float(attrs.get("mass", "1")) == 0.:
                    body.remove(child)
                    record["dropped_geoms"] += 1
                    continue
                for k, v in contact.items():
                    if child.get(k) is not None:
                        child.set(k, v)
                if identity and (ct or ca):
                    child.set("contype", str((ct << COLLISION_SHIFT) | identity))
                    child.set("conaffinity", str((ca << COLLISION_SHIFT) | 1 | (IDENTITY_BITS & ~identity)))
                if child.get("mesh") is not None:
                    child.set("mesh", mesh_for(child.get("mesh")))
                if child.get("material") is not None:
                    del child.attrib["material"]
                if child.get("name"):
                    record["geoms"][child.get("name")] = f"{name}{SEP}{child.get('name')}"
                    child.set("name", f"{name}{SEP}{child.get('name')}")
            elif child.tag in NAMED and child.get("name"):
                child.set("name", f"{name}{SEP}{child.get('name')}")
            elif child.tag not in ("inertial",):
                raise NotImplementedError(f"{name}: <{child.tag}> in an MJCF entity body")
        for child in body.findall("body"):
            convert(child, authored, active)

    worldbody = src.find("worldbody")
    roots = worldbody.findall("body")
    loose = [child for child in worldbody if child.tag != "body"]
    if any(child.tag != "geom" for child in loose):
        raise NotImplementedError(f"{name}: worldbody must hold only bodies and geoms")
    if loose:
        # Geoms placed in the worldbody (#108's base, rails, sill and bridge): Genesis keeps the file's world body as
        # the entity's fixed base link "world"; here a static body "<entity>::world" at the entity pose holds them,
        # first in the link record. It is not the parent of the free roots, so its geoms meet every movable body (as
        # different roots do in Genesis) and, being static, never the table (static pairs are skipped in both).
        static = ET.Element("body", name="world")
        static.extend(copy.deepcopy(loose))
        roots = [static] + roots
    if not roots:
        raise NotImplementedError(f"{name}: worldbody holds no body")
    # Collision bits (COLLISION_SHIFT above): an entity whose colliding geoms all use 1 / 1 keeps them (fiber).
    class_geoms = _class_geoms(top_default)
    bits = {_bits(g, klass, class_geoms) for g, klass in _source_geoms(worldbody, None)} - {(0, 0)}
    identity = 0
    if bits - {(1, 1)}:
        if max(max(b) for b in bits) >= 1 << (31 - COLLISION_SHIFT):
            raise NotImplementedError(f"{name}: authored collision bits {sorted(bits)} do not fit the private range")
        remapped = builder.collision_identities
        if len(remapped) >= COLLISION_SHIFT - 1:
            raise NotImplementedError("more MJCF entities with their own collision bits than identity bits")
        identity = 1 << (len(remapped) + 1)
        remapped.append(name)
        record["collision_bits"] = dict(identity=identity, shift=COLLISION_SHIFT, authored=sorted(map(list, bits)))
    for root in roots:
        root = copy.deepcopy(root)
        local = _pose(root)
        T = T_spec @ local
        root.set("pos", _numbers(T[:3, 3]))
        root.set("quat", _numbers(Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]))
        authored_class = root.get("childclass")
        convert(root, None, authored_class)  # prefixes an authored childclass
        if authored_class is None:
            root.set("childclass", cls)
        builder.world.append(root)
        if record["root"] is None:
            record["root"] = root.get("name")
            if not np.array_equal(local, np.eye(4)):
                # The first root's authored pose in the entity frame (a free root off the entity origin): the frozen
                # scene's declaration check composes it with the entity pose.
                record["root_pose"] = dict(pos=local[:3, 3].tolist(),
                                           quat=Rotation.from_matrix(local[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist())
    for a, b in excludes:
        ET.SubElement(builder.contact, "exclude", body1=a, body2=b)
    record["adjacent_exclusions"] = len(excludes)
    # The file's own <contact><exclude> pairs (slotted: each disc and its 3 nearest links), prefixed; a pair already
    # excluded as parent and child is not repeated.
    authored = 0
    for child in (src.find("contact") if src.find("contact") is not None else ()):
        if child.tag != "exclude":
            raise NotImplementedError(f"{name}: <contact><{child.tag}> in an MJCF entity")
        pair = (f"{name}{SEP}{child.get('body1')}", f"{name}{SEP}{child.get('body2')}")
        if pair in excludes or pair[::-1] in excludes:
            continue
        ET.SubElement(builder.contact, "exclude", body1=pair[0], body2=pair[1])
        excludes.append(pair)
        authored += 1
    if authored:
        record["authored_exclusions"] = authored
    # Profile birth_overlap_exclude (opt-in): Genesis drops same-root geom pairs that already overlap in the authored pose
    # (collider.py, with self collision on: the neutral-overlap check). orbits' two tie links sit in different branches
    # of one tree with coincident end spheres; unexcluded, MuJoCo pushed the tie 3.6 mm apart. The authored file is
    # compiled alone and run to mj_forward at qpos0; every same-root, non-parent body pair with a contact of negative
    # distance is excluded.
    if builder.profile.get("birth_overlap_exclude"):
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        found = set()
        for g1, g2, dist in zip(data.contact.geom1, data.contact.geom2, data.contact.dist):
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            if (dist < 0 and b1 != b2 and model.body_rootid[b1] == model.body_rootid[b2]
                    and b2 != model.body_parentid[b1] and b1 != model.body_parentid[b2]):
                found.add(tuple(sorted((model.body(b1).name, model.body(b2).name))))
        overlaps = []
        for a, b in sorted(found):
            pair = (f"{name}{SEP}{a}", f"{name}{SEP}{b}")
            if pair in excludes or pair[::-1] in excludes:
                continue
            ET.SubElement(builder.contact, "exclude", body1=pair[0], body2=pair[1])
            excludes.append(pair)
            overlaps.append([a, b])
        record["birth_overlap_exclusions"] = overlaps
    # <equality><connect> (orbits' middle tie: site tip_a to site tip_b), prefixed, under the entity's default class. The
    # profile's mjcf_equality {solref, solimp} replaces the authored softness (the authored 5 ms tie is Genesis' soft
    # constraint).
    override = builder.profile.get("mjcf_equality")
    for child in (src.find("equality") if src.find("equality") is not None else ()):
        if child.tag != "connect":
            raise NotImplementedError(f"{name}: <equality><{child.tag}> in an MJCF entity")
        eq = copy.deepcopy(child)
        for key in ("name", "site1", "site2", "body1", "body2"):
            if eq.get(key) is not None:
                eq.set(key, f"{name}{SEP}{eq.get(key)}")
        eq.set("class", f"{cls}/{eq.get('class')}" if eq.get("class") is not None else cls)
        if override:
            eq.set("solref", _numbers(override["solref"]))
            eq.set("solimp", _numbers(override["solimp"]))
        builder.equality.append(eq)
        record.setdefault("equalities", []).append(eq.get("name"))
    builder.entities[name] = record
    return record
