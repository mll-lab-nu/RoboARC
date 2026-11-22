"""Compound-box claw pieces for the redesigned interlocked_claws tiers (step 3, 2026-10-10, agent "claws").

Every piece is a union of oriented boxes: exact MuJoCo primitives (no hulls, no slivers), the same representation as
the released tiers (topogym.assets.geometry.interlocked_claws.FamilyModel). A piece may have several links joined by
fixed joints, all with the piece frame as their frame: the root link carries the body, each extra link one grip block,
so that every grip block is its own grasp target (contact_weld_v1 declares one region per link, and its TCP-to-centre
gate is 6 cm, so one region cannot cover two grips far apart).

Units: metres; frames: 4x4 homogeneous matrices; boxes: {"size": [x, y, z], "pose": 4x4 list} in the piece frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

DENSITY = 600.                      # kg/m3, the family's (wood-like) density
COLORS = ((.74, .40, .24, 1.), (.20, .53, .61, 1.))   # brown target, teal other (the family's colours)
GRIP_BLOCK = (.030, .028, .024)     # the family's grip block: depth along the approach, pad gap, height


def pose(position=(0., 0., 0.), rotation=None):
    T = np.eye(4)
    T[:3, 3] = position
    if rotation is not None:
        T[:3, :3] = rotation.as_matrix() if isinstance(rotation, Rotation) else np.asarray(rotation, float)
    return T


def rot(axes, degrees):
    return Rotation.from_euler(axes, degrees, degrees=True)


def box(center, size, rotation=None):
    return dict(size=[float(s) for s in size], pose=pose(center, rotation).tolist())


def bar(p0, p1, section=.012, up=(0., 0., 1.), extend=None):
    """A square bar from p0 to p1 (both ends extended by `extend`, default half the section, so bars meet at joints)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    x = d / L
    up = np.asarray(up, float)
    if abs(x @ up) > .95:
        up = np.array([1., 0., 0.]) if abs(x[0]) < .9 else np.array([0., 1., 0.])
    y = np.cross(up, x); y /= np.linalg.norm(y)
    z = np.cross(x, y)
    e = section / 2 if extend is None else extend
    return box((p0 + p1) / 2, (L + 2 * e, section, section), np.stack([x, y, z], 1))


def polyline(points, section=.012):
    return [bar(a, b, section) for a, b in zip(points[:-1], points[1:])]


def window(center, normal, up, inner_w, inner_h, section=.012, depth=None):
    """A rectangular frame (4 bars) around an aperture inner_w (along up x normal) by inner_h (along up), in the
    plane through `center` with the given normal; frame bars `section` wide (in-plane) and `depth` deep (along the
    normal, default = section)."""
    n = np.asarray(normal, float); n /= np.linalg.norm(n)
    u = np.asarray(up, float); u -= (u @ n) * n; u /= np.linalg.norm(u)
    w = np.cross(u, n)
    c = np.asarray(center, float)
    depth = section if depth is None else depth
    R = np.stack([n, w, u], 1)        # box axes: x = normal (depth), y = width direction, z = up
    hw, hh = inner_w / 2 + section / 2, inner_h / 2 + section / 2
    return [box(c + hh * u, (depth, inner_w + 2 * section, section), R),
            box(c - hh * u, (depth, inner_w + 2 * section, section), R),
            box(c + hw * w, (depth, section, inner_h), R),
            box(c - hw * w, (depth, section, inner_h), R)]


def grip(tcp_frame):
    """The family's grip block centred on a TCP frame (gripper x = approach, y = pad gap direction, z = height)."""
    T = np.asarray(tcp_frame, float)
    return dict(size=list(GRIP_BLOCK), pose=T.tolist())


def tcp_frame(position, approach, closing, up_hint=(0., 0., 1.)):
    """A TCP frame as the ALOHA's (topogym reference FK): x along the approach (into the part), y along the pad-closing
    direction, z = x cross y; the closing sign is chosen so that z points along up_hint (the gripper's camera side up),
    when the hint is not along the approach."""
    a = np.asarray(approach, float); a /= np.linalg.norm(a)
    c = np.asarray(closing, float); c -= (c @ a) * a; c /= np.linalg.norm(c)
    z = np.cross(a, c)
    if up_hint is not None and abs(float(np.asarray(up_hint) @ a)) < .99 and float(z @ np.asarray(up_hint)) < 0:
        c, z = -c, -z
    return pose(position, np.stack([a, c, z], 1))


def box_mesh(item):
    import trimesh
    return trimesh.creation.box(extents=item["size"], transform=np.asarray(item["pose"], float))


def union_mesh(boxes):
    import trimesh
    mesh = trimesh.boolean.union([box_mesh(b) for b in boxes], engine="manifold")
    return mesh


def box_corners(item):
    T = np.asarray(item["pose"], float)
    s = np.asarray(item["size"], float) / 2
    c = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float) * s
    return c @ T[:3, :3].T + T[:3, 3]


@dataclass
class Piece:
    """A rigid piece: `body` boxes (the root link) and named grip links (one grip block each, a grasp target).
    grips: {link name: TCP frame (4x4, piece frame) at which a hand holds that grip}."""
    name: str
    body: list
    grips: dict = field(default_factory=dict)
    color: tuple = COLORS[0]

    @property
    def boxes(self):
        return list(self.body) + [grip(T) for T in self.grips.values()]

    def corners(self):
        return np.concatenate([box_corners(b) for b in self.boxes])

    def com(self):
        """The composite centre of mass of the whole piece (body + grip blocks, uniform density), piece frame."""
        return np.asarray(union_mesh(self.boxes).center_mass, float)

    def shifted(self, origin):
        """The same piece with its frame moved to `origin` (piece frame): every box and grip frame translated by
        -origin."""
        o = np.asarray(origin, float)
        def mv(b):
            T = np.asarray(b["pose"], float).copy(); T[:3, 3] -= o
            return dict(size=list(b["size"]), pose=T.tolist())
        grips = {}
        for g, T in self.grips.items():
            T = np.asarray(T, float).copy(); T[:3, 3] -= o
            grips[g] = T
        return Piece(self.name, [mv(b) for b in self.body], grips, self.color)

    def save(self, folder: Path, density=DENSITY) -> dict:
        """<folder>/<name>.urdf (+ one OBJ per link): the root link holds the body boxes, each grip a link of its own
        joined by a fixed joint at the identity (every link frame = the piece frame). Inertia from each link's own
        solid union."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        root = ET.Element("robot", name=self.name)
        links = [(self.name, list(self.body))] + [(g, [grip(T)]) for g, T in self.grips.items()]
        report = dict(name=self.name, links=[], mass_kg=0.)
        for link_name, boxes in links:
            mesh = union_mesh(boxes)
            mesh.density = density
            if not mesh.is_watertight or mesh.volume <= 0:
                raise ValueError(f"{self.name}/{link_name}: not one closed solid")
            obj = f"{link_name}.obj"
            mesh.export(folder / obj, digits=12, include_texture=False)
            link = ET.SubElement(root, "link", name=link_name)
            props = mesh.mass_properties
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "origin", xyz=" ".join(map(str, props.center_mass)), rpy="0 0 0")
            ET.SubElement(inertial, "mass", value=str(props.mass))
            ET.SubElement(inertial, "inertia", **{f"i{a}{b}": str(props.inertia[j, k]) for a, b, j, k in
                          (("x", "x", 0, 0), ("y", "y", 1, 1), ("z", "z", 2, 2),
                           ("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2))})
            visual = ET.SubElement(link, "visual")
            ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename=obj)
            ET.SubElement(ET.SubElement(visual, "material", name=f"{self.name}_{link_name}"), "color",
                          rgba=" ".join(map(str, self.color)))
            for item in boxes:
                T = np.asarray(item["pose"], float)
                col = ET.SubElement(link, "collision")
                ET.SubElement(col, "origin", xyz=" ".join(map(repr, T[:3, 3].tolist())),
                              rpy=" ".join(map(repr, Rotation.from_matrix(T[:3, :3]).as_euler("xyz").tolist())))
                ET.SubElement(ET.SubElement(col, "geometry"), "box", size=" ".join(map(repr, item["size"])))
            report["links"].append(dict(name=link_name, boxes=len(boxes), mass_kg=float(props.mass),
                                        watertight=bool(mesh.is_watertight)))
            report["mass_kg"] += float(props.mass)
        for g in self.grips:
            j = ET.SubElement(root, "joint", name=f"{g}_fixed", type="fixed")
            ET.SubElement(j, "parent", link=self.name)
            ET.SubElement(j, "child", link=g)
            ET.SubElement(j, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.indent(root)
        ET.ElementTree(root).write(folder / f"{self.name}.urdf", encoding="unicode")
        whole = union_mesh(self.boxes)
        report.update(watertight=bool(whole.is_watertight), components=len(whole.split()),
                      bounds=whole.bounds.tolist(), grips={g: np.asarray(T).tolist() for g, T in self.grips.items()})
        if report["components"] != 1:
            raise ValueError(f"{self.name}: the piece must be one connected solid ({report['components']} parts)")
        return report


def dump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
