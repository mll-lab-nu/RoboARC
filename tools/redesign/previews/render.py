"""CPU-only previews of a MuJoCo model + state: offscreen EGL on software Mesa (llvmpipe), never the GPU; matplotlib
3D fallback. Fixed views: front (operator side, -y), side (-x), top, aloha (the ALOHA's cam_high); PNG <= 800 px wide
and a 2 x 2 contact sheet.

Python:
    from previews.render import setup_cpu_gl; setup_cpu_gl()        # before anything imports mujoco (the CLI does it)
    from previews.render import render_state, save_views, load_instance_state
    model, data, info = load_instance_state("/path/instance.json")  # frozen scene at its birth, robot at home
    imgs = render_state(model, data)                                 # {view: HxWx3 uint8}
    save_views(imgs, out_dir, prefix="medium", title="rope_threading medium v0")  # <prefix>_<view>.png + _sheet.png

CLI (on the cloud host, env sourced; a few seconds per state):
    python -m previews.render instance <instance.json> <out_dir> [--prefix P] [--views front,side,top,aloha]
    python -m previews.render record <instance.json> <recording.npz> <out_dir> [--samples 0,-1] [--prefix P]
    python -m previews.render scene <scene_dir | model.xml> <out_dir> [--prefix P]
    python -m previews.render selftest <out_dir>          # renders a tiny model; prints the GL renderer string
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

MESA_EGL_VENDOR = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
DEFAULT_LOOKAT = (-0.12, 0.0, 0.95)
VIEWS = {
    "front": dict(azimuth=90.0, elevation=-30.0, distance=1.0),
    "side": dict(azimuth=0.0, elevation=-45.0, distance=1.1),       # from -x, over the left arm
    "back": dict(azimuth=-90.0, elevation=-20.0, distance=1.0),     # from +y (behind the scene): heights in profile
    "top": dict(azimuth=90.0, elevation=-89.0, distance=1.15),
    # the ALOHA's cam_high (topogym/config/profiles/camera_configs/aloha_observation.yaml)
    "aloha": dict(pos=(-0.1356786, -0.42, 1.68), lookat=(-0.1, 0.0, 0.97), fovy=54.0),
}
_GL_INFO: dict = {}


def gpu_allowed() -> bool:
    """GPU renders only when the job opted in through launch.sh (RD_GPU=1, main's 04:52 rule: EGL on the NVIDIA
    device for previews / frames); otherwise software EGL (Mesa llvmpipe), never the GPU."""
    return os.environ.get("RD_GPU") == "1"


def setup_cpu_gl() -> None:
    """Software EGL: Mesa's vendor library forced (the NVIDIA EGL vendor is never loaded), no CUDA device. With
    RD_GPU=1 (an opted-in launch.sh job) EGL is left to pick the NVIDIA device instead."""
    os.environ["MUJOCO_GL"] = "egl"
    if gpu_allowed():
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        return
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = MESA_EGL_VENDOR
    os.environ["EGL_PLATFORM"] = "surfaceless"
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ["GALLIUM_DRIVER"] = "llvmpipe"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def select_software_egl_device() -> int:
    """Pick Mesa's software EGL device (EGL_MESA_device_software, llvmpipe) and pin MuJoCo to it
    (MUJOCO_EGL_DEVICE_ID). MuJoCo's default would try device 0 first, which on the cloud host is the NVIDIA card's DRM node:
    Mesa then opens /dev/dri/renderD128 (seen 2026-10-10: "pci id for fd 3: 10de:2bb4"). Enumeration itself opens no
    device node. Raises if there is no software device (never falls back to a hardware one)."""
    import ctypes
    from mujoco.egl import egl_ext as EGL
    devices = EGL.eglQueryDevicesEXT()
    query = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)(
        EGL.EGL.eglGetProcAddress("eglQueryDeviceStringEXT"))
    for i, dev in enumerate(devices):
        ext = (query(ctypes.cast(dev, ctypes.c_void_p), 0x3055) or b"").decode()   # EGL_EXTENSIONS
        if "EGL_MESA_device_software" in ext:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(i)
            _GL_INFO["egl_device"] = dict(index=i, extensions=ext, n_devices=len(devices))
            return i
    raise RuntimeError("no Mesa software EGL device; refusing to render on a hardware device")


def _renderer(model, width, height):
    import mujoco
    gpu = gpu_allowed()
    if not gpu and (os.environ.get("MUJOCO_GL") != "egl"
                    or os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES") != MESA_EGL_VENDOR):
        raise RuntimeError("call previews.render.setup_cpu_gl() before importing mujoco (CPU-only rendering)")
    if "egl_device" not in _GL_INFO and not gpu:
        select_software_egl_device()
    # mujoco picked its GL backend when it was imported (GLFW when MUJOCO_GL was unset, e.g. after a Session imported
    # it): point both the classic renderer's context module and the compatibility wrapper at the EGL context
    from mujoco.egl import GLContext as _EGL
    for mod in ("mujoco.rendering.classic.gl_context", "mujoco.gl_context"):
        try:
            import importlib
            setattr(importlib.import_module(mod), "GLContext", _EGL)
        except Exception:  # noqa: BLE001
            pass
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))
    r = mujoco.Renderer(model, height=int(height), width=int(width))
    if "renderer" not in _GL_INFO:
        try:
            from OpenGL import GL
            _GL_INFO["renderer"] = GL.glGetString(GL.GL_RENDERER).decode()
            _GL_INFO["vendor"] = GL.glGetString(GL.GL_VENDOR).decode()
        except Exception as exc:  # noqa: BLE001
            _GL_INFO["renderer"] = f"unknown ({exc!r})"
        if not gpu and ("nvidia" in _GL_INFO.get("renderer", "").lower() or "nvidia" in _GL_INFO.get("vendor", "").lower()):
            r.close()
            raise RuntimeError(f"refusing to render on the GPU without RD_GPU=1: {_GL_INFO}")
    return r


def gl_info() -> dict:
    return dict(_GL_INFO)


def _camera(model, view: dict, lookat, distance_scale: float = 1.0):
    import mujoco
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if "pos" in view:
        pos, at = np.asarray(view["pos"], float), np.asarray(view["lookat"], float)
        v = at - pos
        dist = float(np.linalg.norm(v))
        cam.lookat[:] = at
        cam.distance = dist
        cam.azimuth = float(np.degrees(np.arctan2(v[1], v[0])))
        cam.elevation = float(np.degrees(np.arcsin(v[2] / dist)))
    else:
        cam.lookat[:] = np.asarray(lookat, float)
        cam.distance = float(view["distance"]) * distance_scale
        cam.azimuth = float(view["azimuth"])
        cam.elevation = float(view["elevation"])
    return cam


PALETTE = [(0.95, 0.45, 0.10, 1), (0.15, 0.55, 0.95, 1), (0.20, 0.75, 0.30, 1), (0.85, 0.20, 0.55, 1),
           (0.95, 0.80, 0.15, 1), (0.55, 0.35, 0.85, 1), (0.40, 0.80, 0.80, 1)]


def highlight(model, manifest: dict, colors: dict | None = None) -> dict:
    """Preview colours (in place on `model`'s rgba; the physics is unaffected): every task entity (not the robot, table,
    ground) in its own palette colour (ropes / cloths first), a brighter headlight. Returns {entity: rgba}."""
    import mujoco
    model.vis.headlight.ambient[:] = (0.45, 0.45, 0.45)
    model.vis.headlight.diffuse[:] = (0.65, 0.65, 0.65)
    used, k = {}, 0
    ents = (manifest or {}).get("entities") or {}
    order = sorted(ents, key=lambda n: (ents[n].get("kind") not in ("rope_chain", "pbd_cloth"), n))
    for name in order:
        rec = ents[name]
        if name in ("robot", "table", "ground"):
            continue
        rgba = (colors or {}).get(name) or PALETTE[k % len(PALETTE)]
        k += 1
        used[name] = rgba
        if rec.get("kind") == "pbd_cloth":
            f = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name)
            if f >= 0:
                model.flex_rgba[f] = rgba
            continue
        bodies = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, l["body"]) for l in rec.get("links", [])}
        bodies.discard(-1)
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) in bodies:
                model.geom_rgba[g] = rgba
                model.geom_matid[g] = -1
    return used


def render_state(model, data, *, views=("front", "side", "top", "aloha"), width: int = 800, height: int = 450,
                 lookat=None, distance_scale: float = 1.0, hide_groups=()) -> dict:
    """{view: (height, width, 3) uint8} of `data` (kinematics refreshed here; physics untouched)."""
    import mujoco
    width = min(int(width), 800)
    mujoco.mj_forward(model, data)       # positions -> bodies, geoms, flex vertices (qpos / qvel unchanged)
    opt = mujoco.MjvOption()
    for g in hide_groups:
        opt.geomgroup[int(g)] = 0
    out = {}
    r = _renderer(model, width, height)
    try:
        for name in views:
            view = VIEWS[name] if isinstance(name, str) else name
            fovy0 = float(model.vis.global_.fovy)
            if "fovy" in view:
                model.vis.global_.fovy = float(view["fovy"])
            cam = _camera(model, view, lookat if lookat is not None else DEFAULT_LOOKAT, distance_scale)
            r.update_scene(data, camera=cam, scene_option=opt)
            out[name if isinstance(name, str) else f"view{len(out)}"] = r.render().copy()
            model.vis.global_.fovy = fovy0
    finally:
        r.close()
    return out


def contact_sheet(images: dict, title: str | None = None, cols: int = 2, max_width: int = 800) -> np.ndarray:
    from PIL import Image, ImageDraw
    names = list(images)
    tiles = []
    for n in names:
        im = Image.fromarray(images[n])
        ImageDraw.Draw(im).text((8, 6), n, fill=(255, 255, 255))
        ImageDraw.Draw(im).text((9, 7), n, fill=(0, 0, 0))
        tiles.append(im)
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    top = 22 if title else 0
    sheet = Image.new("RGB", (cols * w, rows * h + top), (255, 255, 255))
    if title:
        ImageDraw.Draw(sheet).text((8, 5), title, fill=(0, 0, 0))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, top + (i // cols) * h))
    if sheet.size[0] > max_width:
        s = max_width / sheet.size[0]
        sheet = sheet.resize((max_width, int(sheet.size[1] * s)), Image.LANCZOS)
    return np.asarray(sheet)


def save_views(images: dict, out_dir, prefix: str, title: str | None = None, sheet: bool = True) -> list:
    """<out_dir>/<prefix>_<view>.png (<= 800 px wide) and <prefix>_sheet.png. Returns the paths."""
    from PIL import Image
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, img in images.items():
        im = Image.fromarray(img)
        if im.size[0] > 800:
            im = im.resize((800, int(im.size[1] * 800 / im.size[0])), Image.LANCZOS)
        p = out / f"{prefix}_{name}.png"
        im.save(p, optimize=True)
        paths.append(str(p))
    if sheet and len(images) > 1:
        p = out / f"{prefix}_sheet.png"
        Image.fromarray(contact_sheet(images, title)).save(p, optimize=True)
        paths.append(str(p))
    return paths


# ------------------------------------------------------------------------------------------------ states
def _scene_dir(instance_json) -> Path:
    inst = json.loads(Path(instance_json).read_text())
    scene = (inst.get("simulator") or {}).get("mujoco_scene")
    if not scene:
        raise ValueError(f"{instance_json} has no simulator.mujoco_scene")
    p = Path(scene)
    return p if p.is_absolute() else Path(instance_json).resolve().parent / p


def load_scene_model(scene_dir):
    """(model, manifest, scene_dir) of a frozen / built scene dir, or (model, {}, None) of a plain MJCF file."""
    import mujoco
    p = Path(scene_dir)
    if p.is_file() and p.suffix == ".xml":
        return mujoco.MjModel.from_xml_path(str(p)), {}, None
    from topogym.world.mujoco.scene import load_scene
    model, manifest = load_scene(p, verify=False)
    return model, manifest, p


def set_robot(model, data, q: dict) -> None:
    """q: {"left": [6 arm (+ 1 or 2 finger)], "right": [...]} (finger = native finger displacement)."""
    import mujoco
    names = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
    for arm, vals in q.items():
        vals = list(map(float, vals))
        for j, v in zip(names, vals[:6]):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}/{j}")
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = v
        if len(vals) > 6:
            for k, f in enumerate(("left_finger", "right_finger")):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}/{f}")
                if jid >= 0:
                    data.qpos[model.jnt_qposadr[jid]] = vals[6 + min(k, len(vals) - 7)]


def set_birth(model, data, manifest: dict, scene_dir) -> None:
    """The scene's birth: cloth slides at birth - rest (flex vertices), robot at its home; rope chains are authored at
    their birth pose (qpos0)."""
    import mujoco
    data.qpos[:] = model.qpos0
    for name, rec in (manifest.get("entities") or {}).items():
        if rec.get("kind") == "pbd_cloth" and scene_dir is not None:
            with np.load(Path(scene_dir) / rec["cloth"]["file"], allow_pickle=False) as z:
                birth, rest = z["birth"], z["rest"]
            set_cloth_vertices(model, data, name, birth, rest)
        if name == "robot" and rec.get("home"):
            set_robot(model, data, rec["home"])
    mujoco.mj_kinematics(model, data)


def set_cloth_vertices(model, data, name, verts, rest) -> None:
    import mujoco
    f = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name)
    vadr, vnum = int(model.flex_vertadr[f]), int(model.flex_vertnum[f])
    bodies = np.asarray(model.flex_vertbodyid[vadr:vadr + vnum])
    j0 = np.asarray(model.body_jntadr)[bodies]
    qadr = np.asarray(model.jnt_qposadr)[j0[:, None] + np.arange(3)[None]]
    data.qpos[qadr.reshape(-1)] = (np.asarray(verts, float) - np.asarray(rest, float)).reshape(-1)


def set_rope_links(model, data, manifest: dict, name: str, pos, quat) -> None:
    import mujoco
    links = manifest["entities"][name]["links"]
    for i, l in enumerate(links):
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, l["body"])
        adr = int(model.jnt_qposadr[model.body_jntadr[b]])
        data.qpos[adr:adr + 3] = pos[i]
        data.qpos[adr + 3:adr + 7] = quat[i]


def load_instance_state(instance_json):
    """(model, data, info) of a frozen instance at its birth (robot at home)."""
    import mujoco
    model, manifest, sd = load_scene_model(_scene_dir(instance_json))
    data = mujoco.MjData(model)
    set_birth(model, data, manifest, sd)
    return model, data, dict(manifest=manifest, scene_dir=str(sd))


def apply_recording(model, data, manifest: dict, scene_dir, rec_npz, sample: int) -> int:
    """Set `data` to sample `sample` (index into sample_rows; negative from the end) of an antihack.record npz.
    Returns the row of that sample."""
    import mujoco
    z = np.load(rec_npz, allow_pickle=False)
    rows = z["sample_rows"]
    i = sample % len(rows)
    row = int(rows[i])
    set_birth(model, data, manifest, scene_dir)
    if row > 0:
        q = z["q_measured"][row - 1]
        layout = [str(x) for x in z["action_layout"]]
        arms = {}
        for arm in ("left", "right"):
            vals = [q[layout.index(f"{arm}/{j}")] for j in ("waist", "shoulder", "elbow", "forearm_roll",
                                                           "wrist_angle", "wrist_rotate")]
            if f"{arm}_native_gripper" in layout:
                vals.append(q[layout.index(f"{arm}_native_gripper")])
            elif f"{arm}_gripper" in layout:
                vals.append(float(np.clip(q[layout.index(f"{arm}_gripper")] / 2 + 0.002, 0.002, 0.037)))
            arms[arm] = vals
        set_robot(model, data, arms)
    for key in z.files:
        if key.startswith("obj:rope/") and key.endswith("/pos"):
            name = key.split("/")[1]
            set_rope_links(model, data, manifest, name, z[key][i], z[f"obj:rope/{name}/quat"][i])
        if key.startswith("obj:cloth/") and key.endswith("/verts") and scene_dir is not None:
            name = key.split("/")[1]
            rec = manifest["entities"][name]
            with np.load(Path(scene_dir) / rec["cloth"]["file"], allow_pickle=False) as c:
                rest = c["rest"]
            set_cloth_vertices(model, data, name, z[key][i], rest)
        if key.startswith("obj:rigid/") and key.endswith("/pose"):
            name = key.split("/")[1]
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, manifest["entities"][name]["links"][0]["body"])
            if b >= 0 and model.body_jntadr[b] >= 0:
                adr = int(model.jnt_qposadr[model.body_jntadr[b]])
                data.qpos[adr:adr + 7] = z[key][i]
    mujoco.mj_kinematics(model, data)
    return row


def object_lookat(model, data, manifest: dict) -> tuple:
    """The centre of the task objects (rope links / cloth vertices / task bodies), for framing."""
    import mujoco
    pts = []
    for name, rec in (manifest.get("entities") or {}).items():
        if name in ("robot", "table", "ground"):
            continue
        if rec.get("kind") == "pbd_cloth":
            f = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name)
            vadr, vnum = int(model.flex_vertadr[f]), int(model.flex_vertnum[f])
            pts.append(np.asarray(data.xpos)[np.asarray(model.flex_vertbodyid[vadr:vadr + vnum])])
            continue
        for l in rec.get("links", []):
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, l["body"])
            if b >= 0:
                pts.append(np.asarray(data.xpos[b])[None])
    if not pts:
        return DEFAULT_LOOKAT
    P = np.concatenate(pts)
    c = 0.5 * (P.min(axis=0) + P.max(axis=0))
    return tuple(float(x) for x in c)


def filmstrip(model, manifest: dict, scene_dir, rec_npz, out_png, *, frames: int = 8, view: str = "aloha",
              cols: int = 4, tile=(400, 225), title: str | None = None, lookat=None) -> str:
    """One view at `frames` evenly spaced samples of an antihack.record recording, tiled (cols per row, <= 800 px
    wide), each tile labelled with its row and time."""
    import mujoco
    from PIL import Image, ImageDraw
    z = np.load(rec_npz, allow_pickle=False)
    n = len(z["sample_rows"])
    idx = sorted(set(np.linspace(0, n - 1, frames).round().astype(int).tolist()))
    dt = None
    js = Path(str(rec_npz)[:-4] + ".json")
    if js.is_file():
        dt = json.loads(js.read_text()).get("control_dt_s")
    d = mujoco.MjData(model)
    tiles = []
    for i in idx:
        row = apply_recording(model, d, manifest, scene_dir, rec_npz, i)
        look = lookat if lookat is not None else object_lookat(model, d, manifest)
        img = render_state(model, d, views=(view,), width=tile[0], height=tile[1], lookat=look)[view]
        im = Image.fromarray(img)
        label = f"row {row}" + (f"  t {row * dt:.1f} s" if dt else "")
        ImageDraw.Draw(im).text((6, 4), label, fill=(0, 0, 0))
        tiles.append(im)
    rows_n = (len(tiles) + cols - 1) // cols
    top = 18 if title else 0
    sheet = Image.new("RGB", (cols * tile[0], rows_n * tile[1] + top), (255, 255, 255))
    if title:
        ImageDraw.Draw(sheet).text((6, 3), title, fill=(0, 0, 0))
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * tile[0], top + (k // cols) * tile[1]))
    if sheet.size[0] > 800:
        sheet = sheet.resize((800, int(sheet.size[1] * 800 / sheet.size[0])), Image.LANCZOS)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png, optimize=True)
    return str(out_png)


# ------------------------------------------------------------------------------------------------ matplotlib fallback
def matplotlib_points(point_sets: dict, out_png, title: str = "", views=(("front", 20, -90), ("top", 89, -90))):
    """A no-GL fallback: 3D scatter / polyline of named point sets {label: (N, 3)}; one panel per view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 4), dpi=100)
    for k, (vname, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, len(views), k + 1, projection="3d")
        for label, P in point_sets.items():
            P = np.asarray(P)
            if len(P) > 400:
                ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=1, label=label)
            else:
                ax.plot(P[:, 0], P[:, 1], P[:, 2], lw=1.5, label=label)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(vname, fontsize=8)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        try:
            ax.set_box_aspect((1, 1, 0.5))
        except Exception:  # noqa: BLE001
            pass
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return str(out_png)


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    import argparse
    setup_cpu_gl()
    ap = argparse.ArgumentParser(description="CPU-only MuJoCo previews")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("instance"); a1.add_argument("instance"); a1.add_argument("out")
    a2 = sub.add_parser("record"); a2.add_argument("instance"); a2.add_argument("rec"); a2.add_argument("out")
    a2.add_argument("--samples", default="0,-1")
    a3 = sub.add_parser("scene"); a3.add_argument("scene"); a3.add_argument("out")
    a4 = sub.add_parser("selftest"); a4.add_argument("out")
    a5 = sub.add_parser("filmstrip"); a5.add_argument("instance"); a5.add_argument("rec"); a5.add_argument("out_png")
    a5.add_argument("--frames", type=int, default=8); a5.add_argument("--view", default="aloha")
    for p in (a1, a2, a3, a4, a5):
        p.add_argument("--prefix", default=None)
        p.add_argument("--views", default="front,side,top,aloha")
        p.add_argument("--width", type=int, default=800)
        p.add_argument("--height", type=int, default=450)
        p.add_argument("--title", default=None)
        p.add_argument("--frame-objects", action="store_true", help="centre the free views on the task objects")
        p.add_argument("--no-highlight", action="store_true", help="keep the scene's own colours")
    a = ap.parse_args(argv)
    import mujoco
    views = tuple(v for v in a.views.split(",") if v)
    written = []
    if a.cmd == "selftest":
        m = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><light pos='0 0 3'/><geom type='plane' size='1 1 .1'/>"
            "<body pos='0 0 .2'><freejoint/><geom type='capsule' size='.02 .1' rgba='1 .3 .2 1'/></body>"
            "</worldbody></mujoco>")
        d = mujoco.MjData(m)
        imgs = render_state(m, d, views=views, width=a.width, height=a.height, lookat=(0, 0, 0.1))
        written = save_views(imgs, a.out, a.prefix or "selftest", title="previews selftest")
    elif a.cmd == "instance":
        m, d, info = load_instance_state(a.instance)
        if not a.no_highlight:
            highlight(m, info["manifest"])
        look = object_lookat(m, d, info["manifest"]) if a.frame_objects else None
        imgs = render_state(m, d, views=views, width=a.width, height=a.height, lookat=look)
        written = save_views(imgs, a.out, a.prefix or Path(a.instance).parent.name, title=a.title or str(a.instance))
    elif a.cmd == "record":
        m, manifest, sd = load_scene_model(_scene_dir(a.instance))
        if not a.no_highlight:
            highlight(m, manifest)
        d = mujoco.MjData(m)
        for s in [int(x) for x in a.samples.split(",") if x]:
            row = apply_recording(m, d, manifest, sd, a.rec, s)
            look = object_lookat(m, d, manifest) if a.frame_objects else None
            imgs = render_state(m, d, views=views, width=a.width, height=a.height, lookat=look)
            pre = f"{a.prefix or Path(a.rec).stem}_row{row:05d}"
            written += save_views(imgs, a.out, pre, title=(a.title or Path(a.rec).stem) + f" row {row}")
    elif a.cmd == "filmstrip":
        m, manifest, sd = load_scene_model(_scene_dir(a.instance))
        if not a.no_highlight:
            highlight(m, manifest)
        written = [filmstrip(m, manifest, sd, a.rec, a.out_png, frames=a.frames, view=a.view,
                             title=a.title or Path(a.rec).stem)]
    elif a.cmd == "scene":
        m, manifest, sd = load_scene_model(a.scene)
        if not a.no_highlight:
            highlight(m, manifest)
        d = mujoco.MjData(m)
        set_birth(m, d, manifest, sd)
        look = object_lookat(m, d, manifest) if a.frame_objects else None
        imgs = render_state(m, d, views=views, width=a.width, height=a.height, lookat=look)
        written = save_views(imgs, a.out, a.prefix or Path(a.scene).stem, title=a.title or str(a.scene))
    print(json.dumps(dict(written=written, gl=gl_info())))


if __name__ == "__main__":
    main()
