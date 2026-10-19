"""ALOHA kinematic feasibility for the redesigned knot_untying scenes (MuJoCo only, no Session, no physics stepping).

The robot model is taken from a frozen knot_untying MuJoCo instance scene (the ALOHA exactly as the family's port builds
it: two ViperX 300 arms facing each other along x, bases at robot + (-+0.469, -0.019, 0)); the rope bodies are dropped and
the redesigned scene's static fixtures (anchor clamps, clutter posts / blocks) are added as static geoms, so a pinch pose
can be checked for joint limits, reach, table clearance and arm-vs-clutter collision.

Grasps are the family's vertical pinches (topogym/planning/knot_untying/kinematics.py: vertical_pinch): the TCP x axis
points down, the jaws close across the rope tangent, and the pads' contact point (CONTACT_IN_TCP_M below the TCP) sits on
the rope centre line. IK: damped least squares on the 6 arm joints, joint limits enforced, several seeds; accepted at
0.5 mm / 0.01 rad (the family's KnotIK tolerances).
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

CONTACT_IN_TCP_M = np.array([.01381, 0., -.00055])     # pad contact point in the TCP frame (family kinematics.py)
TCP_IN_HAND_M = np.array([.13, 0., -.003])              # ALOHA TCP in gripper_link (robot/aloha/description.py)
ARM_JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
HOME = np.array([0., -.96, 1.16, 0., -.3, 0.])
LIMITS = np.array([[-3.14158, 3.14158], [-1.85005, 1.25664], [-1.76278, 1.6057], [-3.14158, 3.14158],
                   [-1.8675, 2.23402], [-3.14158, 3.14158]])
TABLE_TOP_M = 0.9196429


def vertical_pinch(point_m, tangent_xy, flip=1, tilt_deg=0.):
    """TCP position and rotation whose pad contact sits on point_m, jaws across the rope (family convention)."""
    down = np.array([0., 0., -1.])
    t = np.asarray(tangent_xy, float)[:2]
    jaw = flip * np.array([-t[1], t[0], 0.]) / np.linalg.norm(t)
    turn = Rotation.from_matrix(np.column_stack([down, jaw, np.cross(down, jaw)]))
    if tilt_deg:
        turn = turn * Rotation.from_euler('y', tilt_deg, degrees=True)
    return np.asarray(point_m, float) - turn.apply(CONTACT_IN_TCP_M), turn


def _resolve_meshes(root, scene_dir: Path):
    from topogym.world.mujoco import scene as sc
    from topogym.assets.robot import asset_root
    for node in root.iter("mesh"):
        f = node.get("file")
        if f and f.startswith(sc.ROBOT_ASSETS):
            node.set("file", str(asset_root(None).resolve() / f[len(sc.ROBOT_ASSETS):]))
        elif f and f.startswith(sc.GENESIS_ASSETS):
            node.set("file", str(sc._genesis_assets() / f[len(sc.GENESIS_ASSETS):]))
        elif f and not Path(f).is_absolute():
            node.set("file", str((scene_dir / f).resolve()))


def static_geom_xml(parent, item: dict, name: str):
    """A static fixture as MJCF geom(s): item = dict(kind=box|cylinder|capsule, pos (world, m), size, yaw_deg)."""
    yaw = float(item.get("yaw_deg", 0.))
    q = Rotation.from_euler('z', yaw, degrees=True).as_quat()[[3, 0, 1, 2]]
    kind = item["kind"]
    attrs = dict(name=name, pos=" ".join(f"{v:.9g}" for v in item["pos"]), quat=" ".join(f"{v:.9g}" for v in q),
                 rgba=item.get("rgba", ".45 .45 .5 1"))
    if kind == "box":
        attrs.update(type="box", size=" ".join(f"{v / 2:.9g}" for v in item["size"]))
    elif kind == "cylinder":
        attrs.update(type="cylinder", size=f"{item['radius']:.9g} {item['height'] / 2:.9g}")
    elif kind == "capsule":
        attrs.update(type="capsule", fromto=" ".join(f"{v:.9g}" for v in item["fromto"]), size=f"{item['radius']:.9g}")
        attrs.pop("pos"); attrs.pop("quat")
    else:
        raise ValueError(f"unknown fixture kind {kind}")
    return ET.SubElement(parent, "geom", **attrs)


def robot_model(scene_dir, fixtures=()):
    """MjModel of the ALOHA + table (+ fixtures) from a frozen knot_untying instance scene; rope bodies removed."""
    import mujoco
    scene_dir = Path(scene_dir)
    root = ET.parse(scene_dir / "scene.xml").getroot()
    _resolve_meshes(root, scene_dir)
    world = root.find("worldbody")
    for body in list(world.findall("body")):
        if (body.get("name") or "").startswith("rope::"):
            world.remove(body)
    for tag in ("equality", "contact"):
        sec = root.find(tag)
        if sec is None:
            continue
        for el in list(sec):
            text = " ".join(str(v) for v in el.attrib.values())
            if "rope::" in text:
                sec.remove(el)
    fix = ET.SubElement(world, "body", name="fixtures")
    for k, item in enumerate(fixtures):
        static_geom_xml(fix, item, f"fixture{k:02d}")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class AlohaIK:
    def __init__(self, model):
        import mujoco
        self.m = model
        self.d = mujoco.MjData(model)
        self.qadr, self.dadr, self.hand = {}, {}, {}
        for side in ("left", "right"):
            jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}/{j}") for j in ARM_JOINTS]
            if min(jid) < 0:
                raise ValueError(f"model lacks {side} arm joints")
            self.qadr[side] = np.array([model.jnt_qposadr[j] for j in jid])
            self.dadr[side] = np.array([model.jnt_dofadr[j] for j in jid])
            self.hand[side] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}/gripper_link")
        self.q = self.d.qpos.copy()
        for side in ("left", "right"):
            self.q[self.qadr[side]] = HOME
        self.robot_geoms = self._robot_geoms()
        self.static_names = {}

    def _robot_geoms(self):
        import mujoco
        m = self.m
        out = {}
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if name.startswith("left/") or name.startswith("right/"):
                out[g] = name
        return out

    def tcp(self, q, side):
        import mujoco
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)          # mj_jac reads cdof
        R = self.d.xmat[self.hand[side]].reshape(3, 3)
        return self.d.xpos[self.hand[side]] + R @ TCP_IN_HAND_M, R

    def solve(self, side, target_p, target_R, seed=None, iters=300, tol_p=5e-4, tol_r=1e-2):
        """DLS IK of one arm's TCP to (target_p, target_R: Rotation of the TCP frame). Returns q (full) or raises."""
        import mujoco
        seeds = [self.q.copy() if seed is None else np.asarray(seed, float).copy()]
        for delta in ([.0, .0, .0, .0, .0, .0], [.4, -.2, .3, .5, -.3, .6], [-.4, .2, -.2, -.5, .3, -.6],
                      [0., .3, -.4, 0., .5, 0.], [1.0, 0., 0., 0., 0., 0.], [-1.0, 0., 0., 0., 0., 0.]):
            s = self.q.copy()
            s[self.qadr[side]] = np.clip(HOME + np.array(delta), LIMITS[:, 0], LIMITS[:, 1])
            seeds.append(s)
        Rt = target_R.as_matrix()
        best = None
        jacp, jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for q0 in seeds:
            q = q0.copy()
            for it in range(iters):
                p, R = self.tcp(q, side)
                ep = np.asarray(target_p) - p
                er = Rotation.from_matrix(Rt @ R.T).as_rotvec()
                if np.linalg.norm(ep) < tol_p and np.linalg.norm(er) < tol_r:
                    return q, dict(iterations=it, position_error_m=float(np.linalg.norm(ep)),
                                   orientation_error_rad=float(np.linalg.norm(er)))
                mujoco.mj_jac(self.m, self.d, jacp, jacr, p, self.hand[side])
                J = np.vstack([jacp[:, self.dadr[side]], jacr[:, self.dadr[side]]])
                e = np.r_[ep, er]
                lam = .02 if np.linalg.norm(ep) > .01 else .005
                dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
                step = np.max(np.abs(dq))
                if step > .2:
                    dq *= .2 / step
                qa = q[self.qadr[side]] + dq
                q[self.qadr[side]] = np.clip(qa, LIMITS[:, 0], LIMITS[:, 1])
            score = np.linalg.norm(ep) + .05 * np.linalg.norm(er)
            if best is None or score < best[0]:
                best = (score, float(np.linalg.norm(ep)), float(np.linalg.norm(er)))
        raise ValueError(f"IK failed for {side}: best position error {best[1]*1000:.1f} mm, orientation {best[2]:.3f} rad")

    def collisions(self, q, sides=("left", "right"), ignore_table_for_fingers=False):
        """Robot geoms (of the listed arms) in contact with the table / fixtures / the other arm at q (distance < 0)."""
        import mujoco
        self.d.qpos[:] = q
        mujoco.mj_forward(self.m, self.d)
        hits = []
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            r1, r2 = self.robot_geoms.get(g1), self.robot_geoms.get(g2)
            if r1 is None and r2 is None:
                continue
            if r1 is not None and r2 is not None:
                if r1.split("/")[0] == r2.split("/")[0]:
                    continue              # same-arm pairs are the MJCF's business (excluded / adjacent)
            mine = [r for r in (r1, r2) if r is not None and r.split("/")[0] in sides]
            if not mine:
                continue
            other = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g2 if r1 is not None else g1) or str(g2)
            if c.dist < 0:
                hits.append(dict(robot=mine[0], other=other, depth_mm=round(-1000 * float(c.dist), 2)))
        return hits

    def finger_table_gap(self, q, side):
        """Lowest point of the side's finger / gripper geoms' bounding spheres above the table (approximate, m)."""
        import mujoco
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        low = np.inf
        for g, name in self.robot_geoms.items():
            if not name.startswith(side + "/") or ("finger" not in name and "gripper" not in name):
                continue
            low = min(low, float(self.d.geom_xpos[g][2] - self.m.geom_rbound[g]))
        return low - TABLE_TOP_M


def pinch_feasible(ik: AlohaIK, side, point_m, tangent_xy, *, pre_m=.05, flips=(1, -1), check_collision=True):
    """Can `side` pinch the rope at point_m (rope centre, world) with jaws across tangent_xy, with a pre-grasp 5 cm
    above? Returns a dict (ok, flip, q, errors, collisions)."""
    errors = []
    for flip in flips:
        tcp, turn = vertical_pinch(point_m, tangent_xy, flip)
        try:
            qpre, _ = ik.solve(side, tcp + [0, 0, pre_m], turn)
            q, info = ik.solve(side, tcp, turn, seed=qpre)
        except ValueError as e:
            errors.append(str(e))
            continue
        hits = ik.collisions(q, (side,)) if check_collision else []
        hits = [h for h in hits if not (h["other"] == "table" and h["depth_mm"] < .5)]
        return dict(ok=not hits, flip=flip, q=q[ik.qadr[side]].tolist(), info=info, collisions=hits)
    return dict(ok=False, errors=errors)


def reach_map(scene_dir, out_json, xs, ys, z_rel=.004, yaws_deg=(0, 45, 90, 135)):
    """For each table point and arm: which jaw yaws admit a vertical pinch (+ 5 cm pre-grasp)."""
    model = robot_model(scene_dir)
    ik = AlohaIK(model)
    rows = []
    for x in xs:
        for y in ys:
            p = np.array([x, y, TABLE_TOP_M + z_rel])
            row = dict(x=float(x), y=float(y))
            for side in ("left", "right"):
                ok = []
                for yaw in yaws_deg:
                    t = [np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))]
                    r = pinch_feasible(ik, side, p, t, check_collision=False)
                    if r["ok"]:
                        ok.append(yaw)
                row[side] = ok
            rows.append(row)
    Path(out_json).write_text(json.dumps(dict(scene=str(scene_dir), z_rel_m=z_rel, yaws_deg=list(yaws_deg), rows=rows)))
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("out")
    ap.add_argument("--step", type=float, default=.05)
    a = ap.parse_args()
    xs = np.arange(-.85, .60 + 1e-9, a.step)
    ys = np.arange(-.45, .45 + 1e-9, a.step)
    reach_map(a.scene_dir, a.out, xs, ys)
