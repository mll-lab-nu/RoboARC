"""Hanayama Cast Rotor as the two rigid pieces of an interlocked_claws setting: two identical flat pieces, each threaded
through the other.

The puzzle (Hanayama Cast Rotor, level 6, by Kyoo Wong): each piece is a hub with three spokes; each spoke ends in a
T-shaped head whose two horns run along the outer rim, with a slit between the tips of neighbouring horns and a window
between neighbouring spokes. One spoke has a groove along its middle, one is thin throughout, and the two tips beside one
slit carry a notch on one face.

Units and frames: metres. Piece frame: origin at the piece's centre, the piece's mid-plane at z = 0. The real piece's
outer radius is 28 mm, the task's 100 mm.

The shape (model v12f, the design run's exact_piece.py): the outline traced from the official top view, triangulated,
each triangle extruded as a prism symmetric about the mid-plane by the thickness rule (the notches remove the upper
half), strictly coplanar neighbours merged; then re-cut by topogym.world.mujoco.prisms.fatten and four slivers under
0.45 mm removed. PARTS_FILE holds the resulting 4809 convex pieces (the thinnest 0.459 mm), UNION_FILE the mesh of their
union (watertight; its volume equals the pieces' sum to 3e-15).

Collision: each piece collides as these exact convex pieces (profile "split": rotor_parts), so the windows, slits, groove
and notches stay open in MuJoCo."""
from pathlib import Path
from types import SimpleNamespace
import json
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

# the family's colours (the target brown, the other teal), density (600 kg/m^3) and ALOHA kinematics
from topogym.assets.geometry.interlocked_claws import COLORS, DENSITY, load_reference, retarget_qpos, tcp_frames

RESOURCES = Path(__file__).resolve().parents[1] / "resources" / "claws_rotor"
UNION_FILE = "rotor_union_v12f.npz"     # one piece's mesh: vertices (n, 3) and faces (m, 3), metres, piece frame
PARTS_FILE = "rotor_parts_v12f.npz"     # its exact convex pieces: n and v00000 ... (each a vertex set), metres, piece frame


def union_mesh():
    """One piece as stored (trimesh, metres, piece frame); both pieces are this solid."""
    with np.load(RESOURCES / UNION_FILE) as data:
        mesh = trimesh.Trimesh(np.asarray(data["vertices"], np.float64), np.asarray(data["faces"], np.int64),
                               process=False)
    if not mesh.is_volume:
        raise ValueError("claws_rotor: the piece mesh is not one closed solid")
    return mesh


def rotor_parts(vertices=None, faces=None):
    """The piece's exact convex pieces (metres, piece frame), for MuJoCo (profile "split"), which collides a mesh as its
    convex hull: the hull would close the windows and slits. Each piece is returned as its own convex hull's vertices
    and faces (453 of the stored vertex sets also hold points that are not hull vertices). (vertices, faces) is the
    declared collision mesh, which the caller checks the pieces against (topogym.world.mujoco.scene); it is not needed
    to build them."""
    with np.load(RESOURCES / PARTS_FILE) as data:
        stored = [np.asarray(data[f"v{i:05d}"], np.float64) for i in range(int(data["n"]))]
    out = []
    for V in stored:
        hull = trimesh.convex.convex_hull(V)
        out.append((np.asarray(hull.vertices, np.float64), np.asarray(hull.faces, np.int64)))
    return out


# The opening (the family's pregrasp birth): the pieces at the official start, between the arms with the gold piece's
# centre 28 cm above the table; poses in the ALOHA model frame, retargeted by the family's retarget_qpos.
OPENING = dict(
    gold_pose=[[-0.4954531140875881, -0.8295263481753494, 0.25770574193017226, 0.032394791247575924],
               [0.8053310083921622, -0.549847623731053, -0.22160676343328847, 0.012687659881754285],
               [0.3255275390239276, 0.09774266397118178, 0.9404670079151325, 0.2978222631218364], [0., 0., 0., 1.]],
    silver_pose=[[0.44027727888520096, 0.87411264400495, -0.20514142265305255, -0.0006145549889976648],
                 [0.06505584239641411, -0.25893427851423023, -0.9637016015242518, 0.04172200764019932],
                 [-0.8955019012082196, 0.4109502707153146, -0.1708690139594469, 0.3434813936774791], [0., 0., 0., 1.]],
    # the TCP frame (aloha.xml site <side>/gripper: x approach, y the fingers' closing direction) in the held piece:
    # the left hand holds the gold piece, the right hand the silver one
    grasp_left=[[0.8090169943749473, 0.0, -0.5877852522924732, -0.0899047793412919],
                [-0.5877852522924732, 0.0, -0.8090169943749473, 0.009449373079395897],
                [0.0, 1.0, 0.0, 0.0], [0., 0., 0., 1.]],
    grasp_right=[[-0.5877852522924731, 0.0, -0.8090169943749475, 0.05284189418109334],
                 [-0.8090169943749475, 0.0, 0.5877852522924731, 0.07273062779430778],
                 [0.0, 1.0, 0.0, 0.0], [0., 0., 0., 1.]],
    seed_q=[-0.4308362565033188, 0.9568412361958631, -0.8107657530903529, 1.7383589259509868, 1.9334131137510657,
            0.20063823162332017, 0., 0., -0.05543653922904425, -0.41701133584691, 0.9609076720826302,
            0.17357185565650451, -0.7479473719277375, -0.32628134494887784, 0., 0.],
    # finger joints of both hands with both pads on the piece's faces: the pieces are 9.0 mm thick at both grasps, and the
    # finger rails stand 2 q - 15.7 mm apart
    finger_q=(.01235, .01235),
)


def _write_urdf(path, name, mesh, color):
    root = ET.Element("robot", name=name)
    link = ET.SubElement(root, "link", name=name)
    props = mesh.mass_properties
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=" ".join(map(repr, map(float, props.center_mass))), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=repr(float(props.mass)))
    ET.SubElement(inertial, "inertia", **{f"i{a}{b}": repr(float(props.inertia[j, k])) for a, b, j, k in
        (("x", "x", 0, 0), ("y", "y", 1, 1), ("z", "z", 2, 2), ("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2))})
    for kind in ("visual", "collision"):
        node = ET.SubElement(link, kind)
        ET.SubElement(ET.SubElement(node, "geometry"), "mesh", filename=f"{name}.obj")
        if kind == "visual":
            ET.SubElement(ET.SubElement(node, "material", name=name), "color", rgba=" ".join(map(str, color)))
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode")


def build_claws_rotor(folder):
    """Write both pieces (OBJ + URDF, metres) into folder/family and return (assembly, report, initial 16 ALOHA joints).
    The silver piece comes first: it is the target piece."""
    folder = Path(folder)
    root = folder / "family"
    root.mkdir(parents=True, exist_ok=False)
    poses = dict(piece_0=np.asarray(OPENING["silver_pose"], float), piece_1=np.asarray(OPENING["gold_pose"], float))
    grasp = dict(piece_1=np.asarray(OPENING["grasp_left"], float), piece_0=np.asarray(OPENING["grasp_right"], float))
    targets = [poses["piece_1"] @ grasp["piece_1"], poses["piece_0"] @ grasp["piece_0"]]
    reference = load_reference()
    seed = np.asarray(OPENING["seed_q"], float)
    q = retarget_qpos(reference, seed, targets)
    q[[6, 7]], q[[14, 15]] = OPENING["finger_q"]
    bodies, parts = [], []
    for name, color in zip(("piece_0", "piece_1"), COLORS):
        mesh = union_mesh()
        mesh.density = DENSITY
        mesh.export(root / f"{name}.obj", digits=12, include_texture=False)
        _write_urdf(root / f"{name}.urdf", name, mesh, color)
        bodies.append(SimpleNamespace(name=name, mesh=mesh, pose=poses[name], color=color, asset_path=root / f"{name}.urdf"))
        parts.append(dict(name=name, mass_kg=float(mesh.mass), volume_m3=float(mesh.volume),
                          bounds_m=mesh.bounds.tolist(), triangles=int(len(mesh.faces)), watertight=bool(mesh.is_watertight)))

    def tcp_error(qpos):
        return max(float(np.linalg.norm(a[:3, 3] - b[:3, 3])) for a, b in zip(tcp_frames(reference, qpos), targets))
    report = dict(family="hanayama_cast_rotor", version="rotor_v12f_r100mm", outer_radius_m=.1, density_kg_m3=DENSITY,
                  union_file=UNION_FILE, parts_file=PARTS_FILE, parts=parts,
                  initial_poses={k: v.tolist() for k, v in poses.items()}, initial_robot_qpos=q.tolist(),
                  tcp_position_error_m=tcp_error(q), seed_tcp_position_error_m=tcp_error(seed),
                  retarget_max_joint_change_rad=float(np.abs(q - seed)[[*range(6), *range(8, 14)]].max()),
                  collision="each piece's exact convex decomposition (profile split: claws_rotor.rotor_parts)",
                  robot_geometry_modified=False, initialization="both pieces held in their start configuration")
    (folder / "geometry_report.json").write_text(json.dumps(report, indent=2) + "\n")
    scene = trimesh.Scene()
    for body in bodies:
        scene.add_geometry(body.mesh, node_name=body.name, transform=body.pose)
    scene.export(folder / "assembly.glb")
    return SimpleNamespace(bodies=bodies, reference={"density_kg_m3": DENSITY}), report, q
