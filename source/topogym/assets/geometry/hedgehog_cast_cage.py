"""Hanayama Cast Cage at four times its real size (hedgehog_in_cage_cast_cage): the cage's and the star's exact convex
pieces from the packaged design, and a push rod.

The pieces are the design's own (runs/hedgehog_cast_cage_20261010/faithful: ccf_model.py, window set v2 and the design
star), exported as RESOURCE by that directory's scripts/export_topogym.py, which also states the frames: the cage frame
has the barrel axis as z, its origin at the cavity centre and window W0 facing +x; the star frame has its origin at the
star centre. Every piece is convex, so each collides exactly as its own hull.
"""
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from topogym.assets.library import resource_path
from topogym.utils.geometry.rigid_geometry import vector_text

RESOURCE = "variations/hedgehog_in_cage/cast_cage_v1/pieces.npz"

# The push rod (metres), in its link frame: the origin at the grip point, the handle's centre; +x along the axis toward
# the tip. A square handle (flat faces: it cannot roll) and a round shaft from the
# handle's front face to the hemispherical tip, ROD_TIP_M from the grip point (30 mm of handle + 200 mm of shaft).
ROD_HANDLE_M = (.060, .016, .016)
ROD_RADIUS_M = .005
ROD_TIP_M = .230
ROD_MASS_KG = .05


def load_design():
    """The packaged design: "cage" (convex trimesh pieces, cage frame), "star" ({part name: convex trimesh}, star
    frame) and the other arrays of RESOURCE (dimensions in metres, spike axes, window centres)."""
    with np.load(resource_path(RESOURCE)) as data:
        arrays = {name: data[name] for name in data.files}

    def pieces(prefix):
        vertices, faces = arrays.pop(f"{prefix}_vertices"), arrays.pop(f"{prefix}_faces")
        nv = np.cumsum(np.r_[0, arrays.pop(f"{prefix}_vertex_counts")])
        nf = np.cumsum(np.r_[0, arrays.pop(f"{prefix}_face_counts")])
        return [trimesh.Trimesh(vertices[a:b], faces[c:d], process=False)
                for a, b, c, d in zip(nv[:-1], nv[1:], nf[:-1], nf[1:])]

    cage = pieces("cage")
    star = dict(zip(arrays.pop("star_names").tolist(), pieces("star")))
    return dict(arrays, cage=cage, star=star)


def _rod_segments():
    """(handle front face x, tip sphere centre x): the shaft's cylinder between them, a sphere of its radius on each end."""
    return ROD_HANDLE_M[0] / 2, ROD_TIP_M - ROD_RADIUS_M


def rod_mesh():
    """The rod's solid as one triangle mesh in the link frame: the handle box and the shaft capsule (its root cap lies
    inside the handle)."""
    x0, x1 = _rod_segments()
    shaft = trimesh.creation.capsule(height=x1 - x0, radius=ROD_RADIUS_M, count=[24, 24])
    turn = trimesh.transformations.rotation_matrix(np.pi / 2, [0., 1., 0.])     # the capsule's z axis onto +x
    turn[:3, 3] = [(x0 + x1) / 2, 0., 0.]
    shaft.apply_transform(turn)
    return trimesh.util.concatenate([trimesh.creation.box(ROD_HANDLE_M), shaft])


def rod_inertial():
    """(mass, centre of mass, diagonal inertia about it) of the rod: ROD_MASS_KG spread uniformly over the handle, the
    shaft's cylinder and the tip's hemisphere, which are disjoint pieces of the one solid. Exact (no tessellation)."""
    a, b, c = ROD_HANDLE_M
    r = ROD_RADIUS_M
    x0, x1 = _rod_segments()
    length = x1 - x0
    # (volume, centre x, per-unit-mass inertia about the piece's own centre: xx, yy, zz)
    parts = [(a * b * c, 0., (b * b + c * c) / 12, (a * a + c * c) / 12, (a * a + b * b) / 12),
             (np.pi * r * r * length, (x0 + x1) / 2, r * r / 2, (3 * r * r + length ** 2) / 12, (3 * r * r + length ** 2) / 12),
             (2 * np.pi * r ** 3 / 3, x1 + 3 * r / 8, 2 * r * r / 5, 83 * r * r / 320, 83 * r * r / 320)]
    volume = sum(p[0] for p in parts)
    masses = [ROD_MASS_KG * p[0] / volume for p in parts]
    com = sum(m * p[1] for m, p in zip(masses, parts)) / ROD_MASS_KG
    inertia = np.zeros(3)
    for m, (_, x, ixx, iyy, izz) in zip(masses, parts):
        inertia += m * np.array([ixx, iyy + (x - com) ** 2, izz + (x - com) ** 2])
    return ROD_MASS_KG, np.array([com, 0., 0.]), inertia


def rod_rest():
    """The rod lying on a flat table: its centre of mass is in front of the handle, so it rests on the handle's front
    bottom edge and on the tip, tipped (tip down) about its own y axis. Returns (that angle in rad, the grip point's
    height above the table in m). Both supports touch: x_e sin t + h cos t = x_t sin t + r, with the edge at x_e (the
    handle's front face), h the handle's half height and the tip's sphere centre at x_t."""
    x_e, x_t = _rod_segments()
    h = ROD_HANDLE_M[2] / 2
    lever = x_t - x_e
    angle = np.arctan2(h, lever) - np.arcsin(ROD_RADIUS_M / np.hypot(lever, h))
    return float(angle), float(x_e * np.sin(angle) + h * np.cos(angle))


def write_rod(folder, color=(.70, .74, .78, 1.)):
    """push_rod.urdf and its visual push_rod.glb in folder. Collision: the handle as a box and the shaft as a cylinder
    with a sphere of its radius on each end cap, the solid of one capsule (MuJoCo's profile option `capsules` collides it
    as exactly that capsule). Returns the rod's report."""
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals
    visual = rod_mesh()
    visual.visual = TextureVisuals(uv=np.zeros((len(visual.vertices), 2)),
                                   material=PBRMaterial(baseColorFactor=color, metallicFactor=.9, roughnessFactor=.3))
    visual.export(folder / "push_rod.glb", include_normals=True)
    mass, com, inertia = rod_inertial()
    root = ET.Element("robot", name="push_rod")
    link = ET.SubElement(root, "link", name="push_rod")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=vector_text(com))
    ET.SubElement(inertial, "mass", value=str(mass))
    ET.SubElement(inertial, "inertia", ixx=str(inertia[0]), iyy=str(inertia[1]), izz=str(inertia[2]),
                  ixy="0", ixz="0", iyz="0")
    node = ET.SubElement(link, "visual")
    ET.SubElement(ET.SubElement(node, "geometry"), "mesh", filename="push_rod.glb")
    ET.SubElement(ET.SubElement(node, "material", name="metal"), "color", rgba=vector_text(color))
    x0, x1 = _rod_segments()
    for kind, size, xyz, rpy in (("box", dict(size=vector_text(ROD_HANDLE_M)), (0., 0., 0.), (0., 0., 0.)),
                                 ("cylinder", dict(radius=str(ROD_RADIUS_M), length=str(x1 - x0)), ((x0 + x1) / 2, 0., 0.),
                                  (0., np.pi / 2, 0.)),
                                 ("sphere", dict(radius=str(ROD_RADIUS_M)), (x0, 0., 0.), (0., 0., 0.)),
                                 ("sphere", dict(radius=str(ROD_RADIUS_M)), (x1, 0., 0.), (0., 0., 0.))):
        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", xyz=vector_text(xyz), rpy=vector_text(rpy))
        ET.SubElement(ET.SubElement(collision, "geometry"), kind, **size)
    ET.indent(root)
    ET.ElementTree(root).write(folder / "push_rod.urdf", encoding="unicode", xml_declaration=True)
    return dict(mass_kg=mass, centre_of_mass_m=com.tolist(), inertia_kg_m2=inertia.tolist(), grip_point_m=[0., 0., 0.],
                tip_m=[ROD_TIP_M, 0., 0.], handle_m=list(ROD_HANDLE_M), shaft_radius_m=ROD_RADIUS_M,
                collision="handle box; shaft cylinder with end-cap spheres (one capsule)")
