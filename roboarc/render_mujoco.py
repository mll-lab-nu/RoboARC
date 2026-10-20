"""The quick-look renderer: MuJoCo's own OpenGL raster of the physics model at the task's camera rig. NOT the graded
renderer: the graded images (privileged tooling, standard mode, verifier videos) are Nyx's. Shapes are the physics
model's geoms (visual and collision groups 0-2), colours MuJoCo's; a cloth whose solver steps outside MuJoCo (blanket on
the GPU VBD) shows its MuJoCo stand-in. Rendering reads a scratch copy of the joint positions and never touches the
physics data.

The OpenGL context is EGL (headless GPUs) unless MUJOCO_GL names osmesa (CPU, needs libOSMesa) or glfw / glx (a
display), as the tooling's raster depth chooses it."""
from __future__ import annotations

import math
import os

import numpy as np


def gl_context(width: int, height: int):
    gl = os.environ.get("MUJOCO_GL", "").lower().strip()
    if gl == "osmesa":
        from mujoco.osmesa import GLContext
    elif gl in ("glfw", "glx"):
        from mujoco.glfw import GLContext
    else:
        from mujoco.egl import GLContext
    return GLContext(width, height)


class Raster:
    """RGB and z-depth of a MuJoCo model at given camera poses (cam2world in OpenGL axes + vertical field of view)."""

    def __init__(self, model, width: int, height: int):
        import copy
        import mujoco
        self.m = copy.deepcopy(model)                 # a private copy: its offscreen size is changed, never the physics'
        self.m.vis.global_.offwidth = max(int(self.m.vis.global_.offwidth), int(width))
        self.m.vis.global_.offheight = max(int(self.m.vis.global_.offheight), int(height))
        # the physics scenes carry no lights for a viewer: a brighter headlight than MuJoCo's default
        self.m.vis.headlight.ambient[:] = 0.45
        self.m.vis.headlight.diffuse[:] = 0.6
        self.m.vis.headlight.specular[:] = 0.1
        self.d = mujoco.MjData(self.m)
        self.W, self.H = int(width), int(height)
        self._gl = gl_context(self.W, self.H)
        self._gl.make_current()
        self._ctx = mujoco.MjrContext(self.m, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self._ctx)
        self._scene = mujoco.MjvScene(self.m, maxgeom=max(10000, 2 * self.m.ngeom))
        self._opt = mujoco.MjvOption()
        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.near = float(self.m.vis.map.znear * self.m.stat.extent)
        self.far = float(self.m.vis.map.zfar * self.m.stat.extent)

    def update(self, data) -> None:
        """Copy the live state's joint positions (and mocap poses) into the scratch data and run its kinematics."""
        import mujoco
        self.d.qpos[:] = data.qpos
        if self.m.nmocap:
            self.d.mocap_pos[:] = data.mocap_pos
            self.d.mocap_quat[:] = data.mocap_quat
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)
        if self.m.nflex:
            mujoco.mj_flex(self.m, self.d)

    def render(self, cam2world_gl, fovy_deg: float, size, depth: bool = False, clip=None):
        """-> (RGB (H, W, 3) uint8, z-depth (H, W) float32 m or None) for one camera; `clip` = the camera's (near, far)
        in m (default: the model's), depth outside it reads 0 (no reading)."""
        import mujoco
        w, h = int(size[0]), int(size[1])
        near, far = (float(clip[0]), float(clip[1])) if clip else (self.near, self.far)
        self._gl.make_current()
        mujoco.mjv_updateScene(self.m, self.d, self._opt, None, self._cam, mujoco.mjtCatBit.mjCAT_ALL.value, self._scene)
        T = np.asarray(cam2world_gl, dtype=np.float64)
        top = near * math.tan(math.radians(fovy_deg) / 2.0)
        for gl in self._scene.camera:
            gl.pos[:] = T[:3, 3]
            gl.forward[:] = -T[:3, 2]
            gl.up[:] = T[:3, 1]
            gl.frustum_near, gl.frustum_far = near, far
            gl.frustum_top, gl.frustum_bottom, gl.frustum_center = top, -top, 0.0
            gl.orthographic = 0
        rect = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjr_render(rect, self._scene, self._ctx)
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        z = np.empty((h, w), dtype=np.float32) if depth else None
        mujoco.mjr_readPixels(rgb, z, rect, self._ctx)
        rgb = np.flipud(rgb).copy()
        if z is not None:
            # OpenGL depth buffer [0, 1] -> metres along the optical axis
            zn, zf = near, far
            z = np.flipud(zn * zf / (zf - z.astype(np.float64) * (zf - zn))).astype(np.float32)
            z[~np.isfinite(z) | (z >= 0.999 * zf) | (z < zn)] = 0.0       # nothing hit within the range: no reading
        return rgb, z

    def close(self) -> None:
        if self._ctx is not None:
            self._ctx.free()
            self._ctx = None
        if self._gl is not None:
            self._gl.free()
            self._gl = None


def fovy_from_intrinsics(K, height: int) -> float:
    fy = float(np.asarray(K, dtype=np.float64)[1][1])
    return math.degrees(2.0 * math.atan(0.5 * float(height) / fy))
