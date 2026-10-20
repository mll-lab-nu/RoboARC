"""Demo video of a recorded run (record.py npz: per-row arm joints, wand pose): MuJoCo's renderer on the planning model
(robot + the tier's wire / sleeves / deck / posts + the wand), two views side by side, a caption with the row, the
grip state and the route cursor. Images only, no physics. usage: video.py <tier> <rec.npz> <out.mp4> [--every 4]"""
import argparse, os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R
from topogym.tasks.buzz_wire_3d.redesign import tiers
from topogym.planning.buzz_wire_3d.redesign.scene import planning_model
from topogym.planning.buzz_wire_3d.redesign.kin import PandaKin

ap = argparse.ArgumentParser()
ap.add_argument("tier"); ap.add_argument("rec"); ap.add_argument("out")
ap.add_argument("--frozen", required=True); ap.add_argument("--assets", required=True)
ap.add_argument("--every", type=int, default=4); ap.add_argument("--fps", type=int, default=40)
ap.add_argument("--w", type=int, default=640); ap.add_argument("--h", type=int, default=480)
a = ap.parse_args()
d = tiers.design(a.tier)
m = planning_model(a.frozen, a.assets, d)
m.vis.global_.offwidth, m.vis.global_.offheight = 2 * a.w, a.h
k = PandaKin(m)
z = np.load(a.rec)
q, wp, wq, held = z["q"], z["wand_p"], z["wand_q"], z["held"]
grip = z["grip_cmd"]
fj = [m.jnt_qposadr[m.joint(n).id] for n in ("finger_joint1", "finger_joint2")]
r = mujoco.Renderer(m, a.h, a.w)
centre = d.origin() + [0, 0, 0.2]
views = [(-125, -25, 0.95), (150, -30, 1.0)]
opt = mujoco.MjvOption()
opt.geomgroup[:] = 0; opt.geomgroup[0] = opt.geomgroup[1] = opt.geomgroup[2] = 1
wr = imageio.get_writer(a.out, fps=a.fps // a.every * 2 if a.every > 1 else a.fps, codec="libx264", quality=7,
                        macro_block_size=8)
for i in range(0, len(q), a.every):
    k.set_q(q[i])
    k.d.qpos[fj] = np.clip(grip[i], 0, 0.8)
    k.set_wand(wp[i], R.from_quat(wq[i][[1, 2, 3, 0]]))
    mujoco.mj_kinematics(m, k.d)
    frames = []
    for az, el, dist in views:
        cam = mujoco.MjvCamera(); cam.lookat[:] = centre; cam.azimuth, cam.elevation, cam.distance = az, el, dist
        r.update_scene(k.d, cam, opt)
        frames.append(r.render().copy())
    im = Image.fromarray(np.concatenate(frames, axis=1))
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, im.width, 22], fill=(255, 255, 255))
    dr.text((8, 5), f"buzz_wire_3d {a.tier}   row {i}   t {i * 0.025:6.1f} s   {'HOLDING' if held[i] else 'free'}", fill=(0, 0, 0))
    wr.append_data(np.asarray(im))
wr.close()
print("wrote", a.out)
