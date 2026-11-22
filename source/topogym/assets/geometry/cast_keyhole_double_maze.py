"""The double-maze setting of cast_keyhole (design dm4b, 2026-10-12: design dm4 of 2026-10-11 with its three round holes
as 24-gons around the circles): two 9 mm plates, each with a slot grid of five columns, threaded through each other at
right angles. The silver plate's left edge runs in the gold plate's grid and the
gold plate's right arm runs in the silver plate's grid; both grids share one depth axis (how far the plates are pushed
into each other).

Shapes come from the catalog resource (rings in mm, plate frame: x across the plate, y along it, bottom edge y = -105 mm,
z through it). Collision: the family's horizontal slabs (cast_keyhole.polygon_slabs / extrude_slabs: exact convex
trapezoid prisms, re-partitioned by the profile's prism_partition), so every slot, door and bore stays open. The round
holes are 24-gons (flat at top and bottom, only material removed): drawn as 192-gons their slabs left needle pieces down to
0.09 mm that the partition could not re-cut (dm4: 100 pieces under 0.5 mm; dm4b: none, thinnest 0.59 mm).

Opening (the family's in-hand birth for this setting): both plates in the hands, interlocked. Plate poses and grasp frames
(the ALOHA TCP in each plate's frame) are stored in the ALOHA model frame ("source robot-mount coordinates", as
interlocked_claws); the arm joints are solved for them with the family reference model (interlocked_claws.retarget_qpos)."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from topogym.assets.geometry.cast_keyhole import (KeyholeBody, extrude_slabs, load_template, mesh_clearance,
                                                  polygon_slabs, write_body)

RESOURCE = "variations/cast_keyhole/double_maze/geometry.json"
GRASP_RADIUS_M = .03


def build_double_maze(folder, resource=RESOURCE):
    """Write both plates (URDF + collision prisms + GLB, metres) into folder and return
    (assembly, report, initial 16 ALOHA joints). The silver plate is the target piece."""
    from topogym.assets.variation_resources import read_json
    from topogym.assets.geometry.interlocked_claws import load_reference, retarget_qpos
    record = read_json(resource)
    template = load_template()
    unit, thickness = float(record["scale"]) / 1000., float(record["thickness_mm"])
    opening = record["opening"]
    poses = dict(gold=np.asarray(opening["gold_plate_pose"], float), silver=np.asarray(opening["silver_plate_pose"], float))
    grasps = dict(gold=np.asarray(opening["grasp_left"], float), silver=np.asarray(opening["grasp_right"], float))
    folder = Path(folder)
    bodies = []
    for name in ("gold", "silver"):
        rings = [np.asarray(r, float) for r in record["rings_mm"][name]]
        if not all(np.isfinite(r).all() for r in rings):
            raise ValueError("Non-finite outline")
        mesh, cells = extrude_slabs(polygon_slabs(rings), thickness)
        for part in [mesh, *cells]:
            part.apply_scale(unit)
        color = tuple(template["colors"][f"keyhole_{name}"])
        mesh.visual = TextureVisuals(uv=np.zeros((len(mesh.vertices), 2)), material=PBRMaterial(
            baseColorFactor=color, metallicFactor=.8, roughnessFactor=.3))
        body = KeyholeBody(f"keyhole_{name}", mesh, cells, color, record["rings_mm"][name])
        body.pose, body.grasp = poses[name], grasps[name]
        bodies.append(body)
    q = retarget_qpos(load_reference(), np.asarray(opening["seed_q"], float),
                      [poses["gold"] @ grasps["gold"], poses["silver"] @ grasps["silver"]])
    q[[6, 7, 14, 15]] = float(opening["finger_q"])
    pieces = []
    for body, expected in zip(bodies, (2, -2)):     # gold: one open fork; silver: the maze and the finger hole
        mesh = body.visual
        pieces.append(dict(name=body.name, watertight=bool(mesh.is_watertight),
                           winding_consistent=bool(mesh.is_winding_consistent), components=len(mesh.split()),
                           euler_number=int(mesh.euler_number), expected_euler_number=expected,
                           volume_m3=float(mesh.volume), collision_prisms=len(body.collision),
                           all_prisms_convex=all(c.is_convex for c in body.collision),
                           collision_volume_error_m3=float(abs(sum(c.volume for c in body.collision) - mesh.volume))))
    placed = []
    for body in bodies:
        m = body.visual.copy()
        m.apply_transform(body.pose)
        placed.append(m)
    gap = mesh_clearance(*placed)
    passed = gap > 1e-8 and all(p["watertight"] and p["winding_consistent"] and p["components"] == 1
                                and p["euler_number"] == p["expected_euler_number"] and p["volume_m3"] > 0
                                and p["all_prisms_convex"] and p["collision_volume_error_m3"] < p["volume_m3"] * 1e-8
                                for p in pieces)
    report = dict(template_id=record["id"], scale=float(record["scale"]), body_count=2, design=record["design"],
                  geometry_status="double_maze_dm4b", pieces=pieces, minimum_initial_surface_clearance_m=gap,
                  no_initial_collision=gap > 1e-8, passed=bool(passed), initial_robot_qpos=q.tolist(),
                  initial_plate_poses_robot_mount={b.name: b.pose.tolist() for b in bodies},
                  initialization="both plates held in the hands, interlocked",
                  robot_geometry_modified=False, source_resource=resource)
    if not passed:
        raise ValueError(f"Invalid double-maze Keyhole geometry: {report}")
    folder.mkdir(parents=True, exist_ok=True)
    report["bodies"] = [write_body(folder / b.name, b, float(record["density_kg_m3"])) for b in bodies]
    (folder / "geometry_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    trimesh.Scene(placed).export(folder / "assembly.glb")
    assembly = SimpleNamespace(bodies=bodies, unit=unit, template=template, record=record)
    return assembly, report, q
