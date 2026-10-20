"""Preview renders of a redesign tier (MuJoCo's own renderer on the planning model; EGL): the start state (robot at
home, wand on the start post), the goal state (wand resting at the finish pose), and the oracle path drawn (the TCP
path in blue, the ring path in orange; grasp / release points as dots). Images, not physics.
usage: preview.py <tier> <out dir> [--path path.npz (tcp_p, wand_p, held)] [--frozen dir --assets dir]"""
import argparse, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R
from topogym.tasks.buzz_wire_3d.redesign import tiers
from topogym.planning.buzz_wire_3d.redesign.scene import planning_model
from topogym.planning.buzz_wire_3d.redesign.kin import PandaKin, HOME
from topogym.assets.geometry.buzz_wire_3d import endpoint_pose

VIEWS = {  # azimuth, elevation, distance, lookat offset
    "front": (-120, -28, 0.95, (0.0, 0.0, 0.18)),
    "side": (-30, -22, 0.95, (0.0, 0.0, 0.18)),
    "top": (-90, -89, 0.95, (0.0, 0.0, 0.15)),
    "robot": (147, -32, 1.05, (-0.05, 0.05, 0.20)),
}


def add_path(scene, pts, rgba, width=0.0018):
    pts = np.asarray(pts, float)
    for a, b in zip(pts[:-1], pts[1:]):
        if scene.ngeom >= scene.maxgeom - 1 or np.linalg.norm(b - a) < 1e-6:
            continue
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                            np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        scene.ngeom += 1


def add_dots(scene, pts, rgba, r=0.006):
    for p in np.asarray(pts, float):
        if scene.ngeom >= scene.maxgeom:
            break
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), p, np.eye(3).ravel(),
                            np.asarray(rgba, np.float32))
        scene.ngeom += 1


def render(model, data, center, view, *, w=900, h=640, paths=(), dots=()):
    az, el, dist, off = VIEWS[view]
    r = mujoco.Renderer(model, h, w)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = np.asarray(center) + off
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = opt.geomgroup[1] = opt.geomgroup[2] = 1
    r.update_scene(data, cam, opt)
    for pts, rgba in paths:
        add_path(r.scene, pts, rgba)
    for pts, rgba in dots:
        add_dots(r.scene, pts, rgba)
    img = r.render().copy()
    r.close()
    return img


def label(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 26], fill=(255, 255, 255))
    d.text((8, 6), text, fill=(0, 0, 0))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tier"); ap.add_argument("out")
    ap.add_argument("--path", default=None)
    ap.add_argument("--frozen", default=str(Path(__file__).resolve().parents[5] / "assets/v1/instance_assets/mujoco"))
    ap.add_argument("--assets", default=str(Path(__file__).resolve().parents[5] / "assets/v1"))
    ap.add_argument("--views", default="front,side,top,robot")
    ap.add_argument("--robot-alpha", type=float, default=0.18)
    a = ap.parse_args()
    d = tiers.design(a.tier)
    w = d.wire(); o = d.origin()
    p0, r0 = endpoint_pose(w, 0); p1, r1 = endpoint_pose(w, 1)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m = planning_model(a.frozen, a.assets, d)
    m.vis.global_.offwidth, m.vis.global_.offheight = 1600, 1200
    m.vis.quality.shadowsize = 4096
    k = PandaKin(m)
    data = k.d
    # the robot drawn faint (it would hide the wire from most views); the wire, sleeves, deck, posts as they are
    for g in range(m.ngeom):
        if k.robot_body[m.geom_bodyid[g]]:
            m.geom_matid[g] = -1
            m.geom_rgba[g] = (0.75, 0.75, 0.78, a.robot_alpha)
    paths, dots = [], []
    if a.path:
        z = np.load(a.path)
        if "tcp_p" in z.files:
            paths.append((z["tcp_p"], (0.15, 0.35, 0.95, 0.9)))
        if "wand_p" in z.files:
            paths.append((z["wand_p"], (0.95, 0.5, 0.1, 0.9)))
        if "grasp_p" in z.files:
            dots.append((z["grasp_p"], (0.1, 0.8, 0.2, 1)))
        if "release_p" in z.files:
            dots.append((z["release_p"], (0.9, 0.1, 0.1, 1)))
    center = o + [0, 0, 0]
    sheets = []
    for state, (pp, rr) in (("start", (p0, r0)), ("goal", (p1, r1))):
        k.set_q(HOME); k.set_gripper(0.0); k.set_wand(o + pp, rr)
        mujoco.mj_forward(m, data)
        row = []
        for v in a.views.split(","):
            img = render(m, data, center, v, paths=paths if state == "start" else (), dots=dots if state == "start" else ())
            im = label(img, f"{d.tier} ({d.name})  {state}  view {v}" + ("  [path: TCP blue, ring orange]" if state == "start" and paths else ""))
            im.save(out / f"{d.tier}_{state}_{v}.png")
            row.append(im)
        sheets.append(row)
    W, H = sheets[0][0].width, sheets[0][0].height
    nv = len(sheets[0])
    sheet = Image.new("RGB", (W * nv // 2, H * len(sheets) // 2), (255, 255, 255))
    for i, row in enumerate(sheets):
        for j, im in enumerate(row):
            sheet.paste(im.resize((W // 2, H // 2)), (j * W // 2, i * H // 2))
    sheet.save(out / f"{d.tier}_sheet.png")
    print("wrote", out)


if __name__ == "__main__":
    main()
