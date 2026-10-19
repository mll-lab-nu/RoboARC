"""A planning model for a redesign tier: the robot (and table) of a frozen buzz scene, with the task bodies replaced by
the tier's wire (metal capsules, rest sleeves, endpoint mounts), deck (boxes), obstacles and a free wand. Planning only
(kinematics, collision, clearance, previews); the Session builds its own scene from the task's declared assets."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

from topogym.tasks.buzz_wire_3d.redesign.design import Design, sleeve_segments, DECK_TOP

TASK_BODIES = ("wire_base::base", "wire::wire", "wand::wand")


def _f(v):
    return " ".join(f"{float(x):.9g}" for x in np.ravel(v))


def robot_xml(frozen_scene_dir, asset_root):
    scene_dir = Path(frozen_scene_dir)
    xml = (scene_dir / "scene.xml").read_text()
    xml = xml.replace("@genesis-assets@/", str(asset_root) + "/").replace("@robot-assets@/", str(asset_root) + "/")
    xml = re.sub(r'file="(meshes/[^"]+)"', lambda m: f'file="{scene_dir / m.group(1)}"', xml)
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    for b in list(world):
        if b.tag == "body" and b.get("name") in TASK_BODIES:
            world.remove(b)
    eq = root.find("equality")
    if eq is not None:
        for e in list(eq):
            if (e.get("name") or "").startswith("grasp/"):
                eq.remove(e)
    asset = root.find("asset")
    if asset is not None:
        for a in list(asset):
            if a.tag == "mesh" and (a.get("name") or "").startswith(("wire_base_", "wire_", "wand_")):
                asset.remove(a)
    return root


STATIC_CONTACT = dict(solref="0.00125 2", solimp="0.9 0.99 0.001 0.5 2", friction="1 0.005 0.0001", condim="3")
WAND_CONTACT = dict(solref="0.00125 2", solimp="0.9 0.99 0.001 0.5 2", friction="0.3 0.005 0.0001", condim="3")


def add_design(root, design: Design, *, wand_pose=None, margin=0.0, rgba_metal="0.8 0.45 0.2 1",
               rgba_sleeve="0.15 0.15 0.15 1", rgba_wood="0.8 0.65 0.45 1", rgba_obst="0.55 0.35 0.2 1"):
    """Static geoms take the buzz profile's task contact (1.25 ms, dampratio 2, .9 / .99, friction 1), the wand its own
    (friction .3), as in the frozen buzz scenes; the grasp weld as the profile's (1.25 ms, .99 / .999), inactive."""
    world = root.find("worldbody")
    wire = design.wire()
    L = design.layout()
    o = design.origin()
    common = dict(contype="1", conaffinity="1", margin=f"{margin}", gap="0", **STATIC_CONTACT)
    wb = ET.SubElement(world, "body", name="wire::wire", pos=_f(o))
    rail = wire.rail
    for i, (a, b) in enumerate(zip(rail[:-1], rail[1:])):
        ET.SubElement(wb, "geom", name=f"wire::metal_{i}", type="capsule", fromto=_f(np.r_[a, b]),
                      size=f"{L.wire_radius_m}", rgba=rgba_metal, **common)
    for i, (a, b) in enumerate(sleeve_segments(rail, design.sleeves)):
        ET.SubElement(wb, "geom", name=f"wire::sleeve_{i}", type="capsule", fromto=_f(np.r_[a, b]),
                      size=f"{design.sleeve_radius_m}", rgba=rgba_sleeve, **common)
    for k, (a, _, _, part) in enumerate([c for c in wire.fixed_capsules() if c[3] == "stop"]):
        ET.SubElement(wb, "geom", name=f"wire::mount_{k}", type="box", pos=_f(a), size=_f(np.asarray(L.mount_size_m) / 2),
                      rgba=rgba_wood, **common)
    bb = ET.SubElement(world, "body", name="wire_base::base", pos=_f(o))
    x, y, t = L.base_size_m
    ET.SubElement(bb, "geom", name="wire_base::deck", type="box", pos=_f([0, 0, L.base_elevation_m + t / 2]),
                  size=_f([x / 2, y / 2, t / 2]), rgba=rgba_wood, **common)
    for sx in (-1, 1):
        for sy in (-1, 1):
            ET.SubElement(bb, "geom", name=f"wire_base::foot_{sx}_{sy}", type="box",
                          pos=_f([sx * (x / 2 - .044), sy * (y / 2 - .044), L.base_elevation_m / 2]),
                          size=_f([.0225, .0225, L.base_elevation_m / 2]), rgba=rgba_wood, **common)
    for k, ob in enumerate(design.obstacles):
        quat = R.from_euler("xyz", ob.get("euler_deg", (0, 0, 0)), degrees=True).as_quat()[[3, 0, 1, 2]]
        bo = ET.SubElement(world, "body", name=f"obst::{k}", pos=_f(o + np.asarray(ob["pos"])), quat=_f(quat))
        if ob["kind"] == "box":
            ET.SubElement(bo, "geom", name=f"obst::{k}_g", type="box", size=_f(np.asarray(ob["size"]) / 2),
                          rgba=rgba_obst, **common)
        elif ob["kind"] == "cylinder":
            ET.SubElement(bo, "geom", name=f"obst::{k}_g", type="cylinder", size=_f([ob["radius"], ob["height"] / 2]),
                          rgba=rgba_obst, **common)
        elif ob["kind"] == "capsule":
            ET.SubElement(bo, "geom", name=f"obst::{k}_g", type="capsule", fromto=_f(np.r_[ob["a"], ob["b"]]) if "a" in ob else None,
                          size=f"{ob['radius']}", rgba=rgba_obst, **common)
    # the wand: loop capsules + stem capsule (metal), the grip box (wood), a free body
    wp = wand_pose or (o + np.array([0, 0, .5]), R.identity())
    wbody = ET.SubElement(world, "body", name="wand::wand", pos=_f(wp[0]), quat=_f(wp[1].as_quat()[[3, 0, 1, 2]]))
    ET.SubElement(wbody, "freejoint", name="wand::root")
    ET.SubElement(wbody, "inertial", pos="0.008 0 0", mass="0.0097", diaginertia="1e-6 4e-6 4.6e-6")
    for k, (a, b, r, part) in enumerate(L.tool_capsules()):
        if part == "grip":
            ET.SubElement(wbody, "geom", name="wand::grip", type="box", pos=_f((a + b) / 2),
                          size=_f([L.grip_length_m / 2, L.grip_radius_m, L.grip_half_thickness_m]),
                          rgba="0.85 0.75 0.55 1", contype="1", conaffinity="1", **WAND_CONTACT)
        else:
            ET.SubElement(wbody, "geom", name=f"wand::{part}_{k}", type="capsule", fromto=_f(np.r_[a, b]),
                          size=f"{r}", rgba="0.75 0.75 0.78 1", contype="1", conaffinity="1", **WAND_CONTACT)
    eq = root.find("equality")
    if eq is None:
        eq = ET.SubElement(root, "equality")
    ET.SubElement(eq, "weld", name="grasp/arm/wand/wand", body1="hand", body2="wand::wand", active="false",
                  solref="0.00125 1", solimp="0.99 0.999 0.001 0.5 2")
    return root


def set_mount(root, mount):
    """Move the robot body to a table-top mount (x, y, yaw_deg); z (the table surface) stays."""
    body = next(b for b in root.find("worldbody") if b.tag == "body" and b.get("name") == "robot")
    z = float(body.get("pos").split()[2])
    body.set("pos", _f([mount[0], mount[1], z]))
    body.set("quat", _f(R.from_euler("z", float(mount[2]), degrees=True).as_quat()[[3, 0, 1, 2]]))
    return root


def planning_model(frozen_scene_dir, asset_root, design: Design, *, diagexact=True, mount=None, **kw):
    """mount: (x, y, yaw_deg) overrides the design's robot_mount (None: the design's, else the frozen scene's)."""
    root = robot_xml(frozen_scene_dir, asset_root)
    mount = design.robot_mount if mount is None else mount
    if mount is not None:
        set_mount(root, mount)
    add_design(root, design, **kw)
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if diagexact:
        m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_DIAGEXACT
    return m
