"""Robot-level kinematic check of a redesigned tier's oracle on the real ALOHA meshes (the cloud host: needs the robot assets).

Takes a frozen rope_threading MuJoCo scene (any instance: the same robot, mount, risers and table), removes its rope and
rings, adds the tier's fixtures (rings as their exact convex tube segments, stands, bulkhead, posts, shelf), and replays
the oracle's joint waypoints (oracle.timeline, densified) kinematically: at every sample MuJoCo's own collision detection
reports robot-fixture, robot-table and arm-arm contacts with penetration. No physics is stepped.

usage: python -m topogym.planning.rope_threading.redesign.mesh_check <tier> <frozen scene dir> <out.json>
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from topogym.planning.rope_threading.redesign import scenes as S
from . import oracle as O
from . import aloha_kin as K
from topogym.planning.rope_threading.redesign.legs import legs_for as _legs_for

ARM_JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")


def _n(v):
    return " ".join(f"{float(x):.9g}" for x in np.ravel(v))


def build(design, scene_dir):
    """MjModel of the frozen scene's robot + table with the tier's fixtures (world frame)."""
    import mujoco
    from topogym.world.mujoco.scene import ROBOT_ASSETS, GENESIS_ASSETS
    from topogym.assets.robot import asset_root
    from topogym.assets.geometry.rope_threading import ring_mesh, ring_hulls
    scene_dir = Path(scene_dir)
    man = json.loads((scene_dir / "scene.json").read_text())
    root = ET.parse(scene_dir / man["xml"]).getroot()
    world = root.find("worldbody")
    drop = lambda name: name and (name.startswith("rope::") or name.split("_")[0] in ("R1", "R2", "R3"))
    for b in list(world.findall("body")):
        if drop(b.get("name")):
            world.remove(b)
    for sec in ("equality", "contact"):
        el = root.find(sec)
        if el is None:
            continue
        for c in list(el):
            refs = [c.get(k) or "" for k in ("body1", "body2", "geom1", "geom2", "name")]
            if any(r.startswith("rope::") or r.startswith("grasp/") or r.split("_")[0] in ("R1", "R2", "R3") for r in refs):
                el.remove(c)
    asset = root.find("asset")
    for mnode in list(asset.findall("mesh")):
        nm = mnode.get("name") or ""
        if nm.split("_")[0] in ("R1", "R2", "R3"):
            asset.remove(mnode)
    robot_root = asset_root(None).resolve()
    for node in asset.iter("mesh"):
        f = node.get("file")
        if f and f.startswith(ROBOT_ASSETS):
            node.set("file", str(robot_root / f[len(ROBOT_ASSETS):]))
        elif f and not Path(f).is_absolute() and not f.startswith(GENESIS_ASSETS):
            node.set("file", str((scene_dir / f).resolve()))
    off = K.ROOT_WORLD + np.array([0, 0, K.TABLE_Z_WORLD])
    fx = ET.SubElement(world, "body", name="rd_fixtures", pos=_n(off))
    attrs = dict(contype="1", conaffinity="1", margin="0")
    for ring in design.rings:
        mesh = ring_mesh(dict(ring.as_layout(), segments=64), 24)
        for i, (v, f) in enumerate(ring_hulls(np.asarray(mesh.vertices), np.asarray(mesh.faces))):
            nm = f"rd_{ring.name}_p{i:02d}"
            ET.SubElement(asset, "mesh", name=nm, vertex=_n(v), face=" ".join(str(int(x)) for x in np.ravel(f)))
            ET.SubElement(fx, "geom", name=f"rd::{ring.name}::{i:02d}", type="mesh", mesh=nm, **attrs)
        st = ring.stand()
        if st is not None:
            ET.SubElement(fx, "geom", name=f"rd::{ring.name}_stand", type="capsule", fromto=_n(np.r_[st["a"], st["b"]]),
                          size=_n([st["radius"]]), **attrs)
    for o in design.obstacles:
        if isinstance(o, S.Box):
            q = O.K.Rotation.from_matrix(o.R()).as_quat()[[3, 0, 1, 2]]
            ET.SubElement(fx, "geom", name=f"rd::{o.name}", type="box", pos=_n(o.center), size=_n(o.half), quat=_n(q), **attrs)
        else:
            ET.SubElement(fx, "geom", name=f"rd::{o.name}", type="capsule", fromto=_n(np.r_[o.a, o.b]), size=_n([o.radius]),
                          **attrs)
    xml = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml)


def run(tier, scene_dir, out, every_s=0.08, timeline_json=None):
    import mujoco
    d_ = S.build(tier)
    m = build(d_, scene_dir)
    d = mujoco.MjData(m)
    if timeline_json:
        tl = json.loads(Path(timeline_json).read_text())
        plan, extra = tl["plan"], dict(joints=tl["joints"], spans=tl["spans"])
    else:
        phases = O.Planner(d_).plan()
        plan, extra = O.timeline(d_, phases)
    J = {arm: np.asarray(extra["joints"][arm]) for arm in ("left", "right")}
    grip = {arm: [] for arm in J}
    for e in plan["events"]:
        grip[e["arm"]].append((e["t"], e["event"]))
    T = max(J["left"][-1, 0], J["right"][-1, 0])
    qadr = {arm: [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}/{j}")] for j in ARM_JOINTS]
            for arm in J}
    fadr = {arm: [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}/{f}")]
                  for f in ("left_finger", "right_finger")] for arm in J}
    body_name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    geom_name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
    worst = {}
    samples = 0
    ts = np.unique(np.r_[np.arange(0, T, every_s), J["left"][:, 0], J["right"][:, 0]])
    for t in ts:
        d.qpos[:] = m.qpos0
        for arm in J:
            q = np.array([np.interp(t, J[arm][:, 0], J[arm][:, 1 + k]) for k in range(6)])
            for a, v in zip(qadr[arm], q):
                d.qpos[a] = v
            closed = False
            for (te, ev) in grip[arm]:
                if te <= t:
                    closed = ev == "grasp"
            for a in fadr[arm]:
                d.qpos[a] = 0.006 if closed else 0.030
        mujoco.mj_forward(m, d)
        samples += 1
        for c in d.contact[: d.ncon]:
            if c.dist >= -1e-4:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = body_name(m.geom_bodyid[g1]), body_name(m.geom_bodyid[g2])
            n1, n2 = geom_name(g1), geom_name(g2)
            robot1, robot2 = b1.startswith(("left/", "right/")), b2.startswith(("left/", "right/"))
            if not (robot1 or robot2):
                continue
            other = n2 if robot1 else n1
            rb = b1 if robot1 else b2
            if (robot1 and robot2) and b1.split("/")[0] == b2.split("/")[0]:
                kind = "self"
            elif robot1 and robot2:
                kind = "arm_arm"
            elif other.startswith("rd::"):
                kind = "fixture"
            elif other == "table" or (b1 == "table" or b2 == "table"):
                kind = "table"
            else:
                kind = "other"
            key = (kind, rb, other if kind != "arm_arm" else (b2 if robot1 else b1))
            dep = -float(c.dist)
            if key not in worst or dep > worst[key]["depth_m"]:
                ph = [sp for sp in extra["spans"] if sp["t0"] - 1e-9 <= t <= sp["t1"] + 1e-9]
                worst[key] = dict(kind=kind, robot_body=rb, other=key[2], depth_m=round(dep, 4), t=round(float(t), 2),
                                  phase=[f"{sp['hand']}: {sp['name']}" for sp in ph])
    rows = sorted(worst.values(), key=lambda r: -r["depth_m"])
    summary = {}
    for r in rows:
        k = r["kind"]
        summary[k] = max(summary.get(k, 0.0), r["depth_m"])
    rep = dict(tier=tier, samples=samples, duration_s=round(float(T), 2), max_depth_by_kind_m=summary,
               contacts=rows[:40], note="kinematic replay of the planned joint waypoints on the real ALOHA meshes "
                                       "(robot collision hulls as the port builds them); fingers 30 mm open / 6 mm closed")
    Path(out).write_text(json.dumps(rep, indent=1))
    return rep


def forced_mesh(tier, scene_dir, out, n_dirs=48):
    """The forced hand-over / regrasp claims on the real meshes: per leg, for the designated receiver and the other
    arm, the fraction of gripper orientations (ladder.orientations) whose IK pose at the receive pinch has no
    penetrating contact of that arm with a fixture, the table or the other arm (at home). For grommets also the far
    side sweep: the highest point behind the wall (x 6 cm past it, at the grommet's y, z up to above the wall top) the
    feeding arm reaches contact-free, from any orientation."""
    import mujoco
    from .ladder import orientations
    d_ = S.build(tier)
    m = build(d_, scene_dir)
    d = mujoco.MjData(m)
    off = K.ROOT_WORLD + np.array([0, 0, K.TABLE_Z_WORLD])
    qadr = {arm: [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}/{j}")] for j in ARM_JOINTS]
            for arm in ("left", "right")}
    fadr = {arm: [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}/{f}")]
                  for f in ("left_finger", "right_finger")] for arm in ("left", "right")}
    body_name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""

    def free(arm, q):
        d.qpos[:] = m.qpos0
        for a in ("left", "right"):
            for k, adr in enumerate(qadr[a]):
                d.qpos[adr] = (q if a == arm else K.HOME)[k]
            for adr in fadr[a]:
                d.qpos[adr] = 0.006
        mujoco.mj_forward(m, d)
        for c in d.contact[: d.ncon]:
            if c.dist >= -1e-4:
                continue
            b1, b2 = body_name(m.geom_bodyid[c.geom1]), body_name(m.geom_bodyid[c.geom2])
            mine = (b1.startswith(arm + "/"), b2.startswith(arm + "/"))
            if not any(mine):
                continue
            if all(mine):
                continue                        # the arm's own fingers / links (not part of the claim)
            return False
        return True

    def stats(arm, point, dv):
        ok = tried = 0
        for Rm in orientations(dv, n_dirs):
            tried += 1
            q, pe, re = K.ik(arm, point, rotation=Rm)
            if pe > 0.003 or re > 0.05:
                continue
            ok += free(arm, q)
        return dict(feasible=ok, tried=tried, fraction=round(ok / tried, 3))

    rows = []
    for leg in _legs_for(d_):
        ring = [r for r in d_.rings if r.name == leg["ring"]][0]
        dv = S._unit(ring.travel)
        pinch = ring.center + (S.POKE_M - S.RECEIVE_STATION_M) * dv
        other = "left" if leg["receiver"] == "right" else "right"
        row = dict(ring=ring.name, receiver=leg["receiver"], designated=stats(leg["receiver"], pinch, dv),
                   other_arm=other, other=stats(other, pinch, dv))
        if ring.mount == "bulkhead":
            sweep = []
            for z in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
                pt = np.array([pinch[0], ring.center[1], z])
                st = stats(leg["feeder"], pt, dv)
                sweep.append(dict(z=z, feasible=st["feasible"]))
            row["feeder_far_side_sweep"] = sweep
        rows.append(row)
        print(tier, json.dumps(row), flush=True)
    rep = dict(tier=tier, legs=rows, note="real ALOHA collision meshes (the port's hulls); a pose counts when that arm "
               "has no penetrating contact with a fixture, the table or the other arm (other arm at home)")
    Path(out).write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    if sys.argv[1] == "--forced":
        forced_mesh(sys.argv[2], sys.argv[3], sys.argv[4])
        raise SystemExit(0)
    rep = run(sys.argv[1], sys.argv[2], sys.argv[3], timeline_json=sys.argv[4] if len(sys.argv) > 4 else None)
    print(json.dumps({k: v for k, v in rep.items() if k != "contacts"}), json.dumps(rep["contacts"][:8]))
