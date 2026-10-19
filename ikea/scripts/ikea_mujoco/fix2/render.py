#!/usr/bin/env python3
"""Render saved states (dump_states.py) of one IKEA scene offscreen (MUJOCO_GL=egl), close-ups on given geoms.

usage: MUJOCO_GL=egl python render.py --task T --scene-dir D --states S.npz --geoms a,b --out DIR [--tag base]
       [--dist 0.12] [--az 45,135] [--el -20] [--size 960x720]
Per state row and geom, from each azimuth: "vis" the visual model (groups 0-2) from --far; "col" an x-ray of the
collision model (groups 0 and 3) cut down to the focus geom (opaque red), its contact partners (40 % opaque) and the
small geoms (bounding radius <= --small) within --near of it (20 %); the rest hidden. So a dowel
inside its hole or a stand on its support is visible. Files: <tag>_r<row>_<geom>_<az>_<vis|col>.png.
"""
import argparse, os, sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
import rcb_ikea  # noqa: E402

ap = argparse.ArgumentParser()
for k in ("--task", "--scene-dir", "--states", "--geoms", "--out"):
    ap.add_argument(k, required=True)
ap.add_argument("--tag", default="x")
ap.add_argument("--dist", type=float, default=0.12)
ap.add_argument("--far", type=float, default=0.4)
ap.add_argument("--near", type=float, default=0.02)
ap.add_argument("--small", type=float, default=0.03)
ap.add_argument("--az", default="45,135")
ap.add_argument("--el", type=float, default=-20)
ap.add_argument("--size", default="960x720")
a = ap.parse_args()
W, H = (int(x) for x in a.size.split("x"))
s = rcb_ikea.Session(a.task, a.scene_dir)
m, d = s.model, s.data
m.vis.global_.offwidth, m.vis.global_.offheight = max(W, m.vis.global_.offwidth), max(H, m.vis.global_.offheight)
z = np.load(a.states)
rgba0 = m.geom_rgba.copy()
mat0 = m.geom_matid.copy()


def xray_rgba(g):
    """focus geom g opaque red, its contact partners 40 %, small geoms within --near 20 %, the rest hidden."""
    x = rgba0.copy()
    near = np.linalg.norm(d.geom_xpos - d.geom_xpos[g], axis=1) - m.geom_rbound - m.geom_rbound[g] < a.near
    x[:, 3] = np.where(near & (m.geom_rbound <= a.small), 0.2, 0.)
    c = d.contact[:d.ncon]
    partners = np.r_[c.geom2[c.geom1 == g], c.geom1[c.geom2 == g]].astype(int)
    x[partners, 3] = 0.4
    x[g] = [0.9, 0.1, 0.1, 1.0]
    return x
spec = mujoco.mjtState.mjSTATE_INTEGRATION
r = mujoco.Renderer(m, H, W)
os.makedirs(a.out, exist_ok=True)
for row, st in zip(z["rows"], z["state"]):
    mujoco.mj_setState(m, d, st, spec)
    mujoco.mj_forward(m, d)
    for gname in a.geoms.split(","):
        g = m.geom(gname).id
        for az in (float(x) for x in a.az.split(",")):
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = d.geom_xpos[g]
            cam.azimuth, cam.elevation = az, a.el
            for mode in ("vis", "col"):
                cam.distance = a.far if mode == "vis" else a.dist
                m.geom_rgba[:] = rgba0 if mode == "vis" else xray_rgba(g)
                m.geom_matid[:] = mat0 if mode == "vis" else -1
                opt = mujoco.MjvOption()
                opt.geomgroup[:] = [1, 1, 1, 0, 0, 0] if mode == "vis" else [1, 0, 0, 1, 0, 0]
                opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
                r.update_scene(d, camera=cam, scene_option=opt)
                img = r.render()
                fn = os.path.join(a.out, f"{a.tag}_r{int(row):06d}_{gname}_{int(az)}_{mode}.png")
                try:
                    from PIL import Image
                    Image.fromarray(img).save(fn)
                except ImportError:
                    import imageio.v3 as iio
                    iio.imwrite(fn, img)
print("rendered", len(z["rows"]), "states")
