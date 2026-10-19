"""python check.py <mjcf>: load the standalone two-geom model and run mj_forward (MuJoCo 3.14.0: SIGSEGV, rc -11)."""
import sys, mujoco
m = mujoco.MjModel.from_xml_path(sys.argv[1]); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
print("ncon", d.ncon, "mujoco", mujoco.__version__)
