#!/usr/bin/env python
"""Render-bridge parity acceptance on the Genesis 1.2.1 + Nyx runtime (needs CUDA and RT cores; not a CPU check).

reference  today's pipeline: rcb_topogym.Session (Genesis physics) replays a trajectory; at the selected control steps
           it records the rigid qpos, every camera's RGB (env.sensor_frame, the observation path), depth, calibration
           and the buffers the Nyx pass read (visual-geometry poses, camera poses).
bridge     a fresh process builds render_bridge.build(instance) (never stepped) and a MuJoCo model (--mjcf, or a
           kinematic mirror of the host's moving entities), turns each recorded state into MuJoCo qpos, runs
           mj_kinematics, pushes mj_data through bridge.update and renders the same sequence; compares with every --ref.
noise      two reference runs against each other: the renderer's own run-to-run noise (the RGB envelope).

Exact checks: qpos round trip, Nyx inputs, depth and calibration bit for bit; MuJoCo-vs-host forward kinematics.
RGB: Nyx's path tracer is not bit-reproducible between runs (temporal history, GPU nondeterminism), so bridge-vs-
reference RGB is judged against reference-vs-reference; --render-mode Forward (test only, both stages) is nearly exact.

  run.sh BASE scripts/robotangle/test_render_bridge.py reference --traj T --out R1 [--steps 0-200,210-99999:10]
  run.sh BASE scripts/robotangle/test_render_bridge.py bridge --ref R1 --ref R2 --out B1 [--mjcf X --alias piece_0=claw_0]
  run.sh BASE scripts/robotangle/test_render_bridge.py noise --ref R1 --ref R2 --out N
  run.sh BASE scripts/robotangle/test_render_bridge.py ghost --traj T --out G [--ref G_reference: bridge side]
  python scripts/robotangle/test_render_bridge.py selftest --out S     (CPU only: numpy + mujoco)
(INSTANCE in the environment names the frozen instance.json.) Every output directory must be new.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "shared"))


def sha(x) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


def parse_steps(spec: str, limit: int) -> list[int]:
    out = set()
    for part in spec.split(","):
        rng, _, every = part.partition(":")
        lo, _, hi = rng.partition("-")
        lo, hi = int(lo), int(hi or lo)
        out.update(range(lo, min(hi, limit) + 1, int(every or 1)))
    return sorted(out)


def force_nyx_options(render_mode=None):
    """Test only: every Nyx camera of this process uses `render_mode` (both stages get the same flag)."""
    if not render_mode:
        return
    import genesis.engine.scene as gscene
    from gs_nyx import nyx_py_renderer as npr
    original = gscene.Scene.add_sensor

    def add_sensor(self, options, *args, **kwargs):
        if type(options).__name__ == "NyxCameraOptions":
            options.render_mode = getattr(npr.ERenderMode, render_mode)
        return original(self, options, *args, **kwargs)
    gscene.Scene.add_sensor = add_sensor


def plain(x):
    """Frozen observation metadata (mappings, tuples, arrays) as plain JSON values."""
    if hasattr(x, "items"):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def rgb_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    m = d.max(-1)
    mse = float((d.astype(np.float64) ** 2).mean())
    return dict(changed_px=int((m > 0).sum()), max_abs=int(m.max()), mean_abs=float(d.mean()),
                psnr_db=float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse)))


def nyx_scene_signature(world) -> dict:
    """The exported Nyx scene file (instances, materials, textures, lights, HDRI, camera settings) and the meshes the
    exporter wrote, with their per-process random parts (UUIDs, file and folder names, the episode's temporary
    directory) normalized. Equal signatures = the same static scene."""
    import re
    sensor = next(c.sensor for c in world.cameras.values() if getattr(c, "sensor", None) is not None)
    folder = Path("__nyx_cache__") / sensor._shared_metadata.scene_description_export_path
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())
             if p.name != "nyx_scene.json"}
    text = (folder / "nyx_scene.json").read_text()
    for name, digest in files.items():
        text = text.replace(name, digest)
    text = re.sub(r"topogym-episode-[A-Za-z0-9_]+", "topogym-episode-<tmp>", text.replace(folder.name, "<export>"))
    ids = {}

    def norm(x, key=""):
        if isinstance(x, dict):
            return {k: norm(v, k) for k, v in x.items()}
        if isinstance(x, list):
            return [norm(v, key) for v in x]
        if "uuid" in key.lower():
            return ids.setdefault(json.dumps(x), len(ids))
        return x
    data = norm(json.loads(text))
    return dict(scene_sha=hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16],
                exported_meshes=len(files), uuids=len(ids), bytes=len(text),
                mesh_sha=hashlib.sha256("".join(sorted(files.values())).encode()).hexdigest()[:16])


class Recorder:
    """One run directory: frames.npz (all RGB), depth.npz (every 10th state), rows.json (hashes, timings)."""

    def __init__(self, out: Path, meta: dict):
        self.out, self.meta, self.rows, self.rgb, self.depth = out, meta, [], {}, {}

    def add(self, i, step, qpos, rgb, depth, cam_meta, nyx, timing):
        row = dict(i=i, step=int(step), qpos_sha=sha(qpos), rgb_sha={c: sha(v) for c, v in rgb.items()},
                   depth_sha={c: sha(v) for c, v in depth.items()}, nyx_sha={k: sha(v) for k, v in nyx.items()},
                   meta_sha=hashlib.sha256(json.dumps(cam_meta, sort_keys=True).encode()).hexdigest()[:16], **timing)
        self.rows.append(row)
        for c, v in rgb.items():
            self.rgb[f"{i}/{c}"] = v
        if i % 10 == 0:
            for c, v in depth.items():
                self.depth[f"{i}/{c}"] = v
        return row

    def save(self, qpos_all, steps):
        np.savez_compressed(self.out / "frames.npz", **self.rgb)
        np.savez_compressed(self.out / "depth.npz", **self.depth)
        np.savez_compressed(self.out / "states.npz", steps=np.asarray(steps), qpos=np.stack(qpos_all))
        (self.out / "rows.json").write_text(json.dumps(dict(meta=self.meta, rows=self.rows), indent=1))


def capture_reference(env):
    """The 0.2.22 observation path: env.sensor_frame() (Nyx RGB, one shared pass), then its depth."""
    import render_bridge
    t = time.perf_counter()
    frame = env.sensor_frame()
    t_rgb = time.perf_counter() - t
    nyx = render_bridge.nyx_inputs(env._world)
    t = time.perf_counter()
    fd = env.sensor_frame(depth=True)
    t_depth = time.perf_counter() - t
    rgb = {c: np.array(v, copy=True) for c, v in frame.rgb.items()}
    depth = {c: np.array(v, dtype=np.float32, copy=True) for c, v in fd.depth.items()}
    return rgb, depth, plain(frame.camera_metadata), nyx, dict(t_rgb=t_rgb, t_depth=t_depth)


def stage_reference(a):
    import torch  # noqa: F401
    force_nyx_options(a.render_mode)
    from rcb_topogym import Session, load_trajectory
    from topogym.utils.arrays import to_numpy
    path = os.environ["INSTANCE"]
    inst = json.load(open(path))
    t0 = time.perf_counter()
    s = Session(inst["task"], instance=path, strict=True)
    startup = time.perf_counter() - t0
    acts, ticks = load_trajectory(a.traj, s.task, s.action_dim, trusted=True, layout=s.action_layout)
    steps = parse_steps(a.steps, len(acts))
    rs = s.env._world.scene.sim.rigid_solver
    rec = Recorder(a.out, dict(stage="reference", task=inst["task"], instance=path, startup_s=startup,
                               render_mode=a.render_mode or "default", trajectory=str(a.traj), rows=len(acts)))
    s.reset()
    k, qs, done_steps, step_s = 0, [], [], []
    for i, target in enumerate(steps):
        done = False
        while s.n_steps < target and k < len(acts):
            t = time.perf_counter()
            done = s.step(acts[k], _physics_steps=None if ticks is None else int(ticks[k]))
            step_s.append(time.perf_counter() - t)
            k += 1
            if done:
                break
        qpos = np.asarray(to_numpy(rs.get_qpos()), dtype=np.float64).reshape(-1).copy()
        rgb, depth, meta, nyx, timing = capture_reference(s.env)
        rec.add(i, s.n_steps, qpos, rgb, depth, meta, nyx, timing)
        time.sleep(a.pace)
        if i == 0:
            rec.meta["nyx_scene"] = nyx_scene_signature(s.env._world)
        qs.append(qpos)
        done_steps.append(s.n_steps)
        if done or s.n_steps < target:
            break
    rec.meta.update(physics_step_s_mean=float(np.mean(step_s)) if step_s else None, frames=len(qs))
    rec.save(qs, done_steps)
    s.close()
    print(json.dumps(dict(stage="reference", frames=len(qs), startup_s=startup,
                          t_rgb_median=float(np.median([r["t_rgb"] for r in rec.rows])),
                          t_depth_median=float(np.median([r["t_depth"] for r in rec.rows])))), flush=True)


def mirror_model(bridge):
    """Test fixture: a kinematic MuJoCo copy of the host's moving entities (bodies `<entity>/<link>`, joints
    `<entity>/<joint>`, the same local frames, axes, anchors and reference angles; nominal inertia, no geoms)."""
    import mujoco
    from genesis.utils.misc import qd_to_torch
    import render_bridge
    qpos0 = qd_to_torch(bridge.solver._rigid_global_info.qpos0).cpu().numpy().reshape(-1)
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    kinds = {1: mujoco.mjtJoint.mjJNT_HINGE, 2: mujoco.mjtJoint.mjJNT_SLIDE, 3: mujoco.mjtJoint.mjJNT_BALL}
    for ename, entity in render_bridge._rigid_entities(bridge.env):
        bodies = {}
        for link in entity.links:
            parent = bodies.get(link.parent_idx, spec.worldbody)
            body = parent.add_body(name=f"{ename}/{link.name}", pos=np.asarray(link.pos, dtype=float),
                                   quat=np.asarray(link.quat, dtype=float))
            body.explicitinertial = True
            body.mass = 1.0
            body.inertia = [1e-3, 1e-3, 1e-3]
            for joint in link.joints:
                kind, name = int(joint.type), f"{ename}/{joint.name}"
                if kind == 4:
                    body.add_freejoint(name=name)
                elif kind in kinds:
                    q = entity.q_start + int(joint.qs_idx_local[0])
                    axis = np.asarray(joint.dofs_motion_vel if kind == 2 else joint.dofs_motion_ang, dtype=float)[0]
                    body.add_joint(name=name, type=kinds[kind], pos=np.asarray(joint.pos, dtype=float),
                                   **({} if kind == 3 else dict(axis=axis, ref=float(qpos0[q]))))
            bodies[link.idx] = body
    return spec.compile()


def load_mjcf(path):
    """An MJCF file, or a frozen MuJoCo-core scene directory (scene.json + scene.xml; robot meshes behind the
    @robot-assets@/ token, resolved like topogym.world.mujoco.scene.load_scene does)."""
    import xml.etree.ElementTree as ET
    import mujoco
    from topogym.assets.robot import asset_root
    path = Path(path)
    if path.is_dir():
        path = path / json.loads((path / "scene.json").read_text())["xml"]
    root = ET.parse(path).getroot()
    for node in root.iter("mesh"):
        file = node.get("file")
        if file and file.startswith("@robot-assets@/"):
            node.set("file", str(asset_root() / file[len("@robot-assets@/"):]))
        elif file and not Path(file).is_absolute():
            node.set("file", str((path.parent / file).resolve()))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def stage_bridge(a):
    import mujoco
    import torch  # noqa: F401
    force_nyx_options(a.render_mode)
    import render_bridge
    path = os.environ["INSTANCE"]
    aliases = dict(x.split("=", 1) for x in a.alias)
    t0 = time.perf_counter()
    bridge = render_bridge.build(path, aliases=aliases, keep=a.keep)
    build_s = time.perf_counter() - t0
    model = load_mjcf(a.mjcf) if a.mjcf else mirror_model(bridge)
    data = mujoco.MjData(model)
    jmap = bridge.bind(model)
    refs = [Path(r) for r in a.ref]
    ref_rows = [json.loads((r / "rows.json").read_text())["rows"] for r in refs]
    ref_frames = [np.load(r / "frames.npz") for r in refs]
    states = np.load(refs[0] / "states.npz")
    rec = Recorder(a.out, dict(stage="bridge", instance=path, build_s=build_s, render_mode=a.render_mode or "default",
                               mujoco=mujoco.__version__, model=a.mjcf or "mirror", joint_map=jmap.report(),
                               asset_differences=bridge.asset_differences, refs=[str(r) for r in refs]))
    report = dict(frames=0, qpos_exact=0, fk_max_pos_m=0.0, fk_max_rot_rad=0.0, fk_links=None,
                  nyx_identical=[0] * len(refs), depth_identical=[0] * len(refs), meta_identical=[0] * len(refs),
                  rgb={}, mismatches=[])
    qs = []
    for i, step in enumerate(states["steps"]):
        q_ref = states["qpos"][i]
        data.qpos[:] = model.qpos0
        data.qpos[jmap.src] = q_ref[jmap.dst]
        mujoco.mj_kinematics(model, data)
        t = time.perf_counter()
        bridge.update(model, data)
        t_update = time.perf_counter() - t
        t = time.perf_counter()
        rgb = bridge.render()
        t_rgb = time.perf_counter() - t
        nyx = bridge.nyx_inputs()
        t = time.perf_counter()
        depth = bridge.depth()
        t_depth = time.perf_counter() - t
        meta = plain(bridge.camera_metadata())
        q_host = bridge.qpos()
        fk = bridge.check_fk(model, data)
        row = rec.add(i, step, q_host, rgb, depth, meta, nyx,
                      dict(t_update=t_update, t_rgb=t_rgb, t_depth=t_depth))
        if i == 0:
            rec.meta["nyx_scene"] = nyx_scene_signature(bridge.world)
        qs.append(q_host)
        report["frames"] += 1
        report["qpos_exact"] += int(np.array_equal(q_host, q_ref))
        report["fk_max_pos_m"] = max(report["fk_max_pos_m"], fk["max_position_error_m"])
        report["fk_max_rot_rad"] = max(report["fk_max_rot_rad"], fk["max_rotation_error_rad"])
        report["fk_links"] = fk["links_compared"]
        for j, rows in enumerate(ref_rows):
            ref = rows[i]
            if ref["step"] != int(step):
                raise RuntimeError("reference runs record different steps")
            same_nyx = ref["nyx_sha"] == row["nyx_sha"]
            same_depth = ref["depth_sha"] == row["depth_sha"]
            same_meta = ref["meta_sha"] == row["meta_sha"]
            report["nyx_identical"][j] += int(same_nyx)
            report["depth_identical"][j] += int(same_depth)
            report["meta_identical"][j] += int(same_meta)
            if not (same_nyx and same_depth and same_meta) and len(report["mismatches"]) < 20:
                report["mismatches"].append(dict(ref=j, i=i, step=int(step), nyx=same_nyx, depth=same_depth,
                                                 meta=same_meta))
            for c, v in rgb.items():
                m = rgb_metrics(ref_frames[j][f"{i}/{c}"], v)
                report["rgb"].setdefault(f"ref{j}/{c}", []).append(dict(i=i, step=int(step), **m))
    ref_scenes = [json.loads((r / "rows.json").read_text())["meta"].get("nyx_scene") for r in refs]
    report["nyx_scene_identical"] = [None if x is None else x == rec.meta["nyx_scene"] for x in ref_scenes]
    rec.meta.update(report={k: v for k, v in report.items() if k != "rgb"}, timing=bridge.timing)
    rec.save(qs, states["steps"])
    summary = summarize(report["rgb"])
    out = dict(stage="bridge", build_s=build_s, frames=report["frames"], qpos_exact=report["qpos_exact"],
               fk_max_pos_m=report["fk_max_pos_m"], fk_max_rot_rad=report["fk_max_rot_rad"],
               fk_links=report["fk_links"], nyx_identical=report["nyx_identical"],
               depth_identical=report["depth_identical"], meta_identical=report["meta_identical"],
               nyx_scene=rec.meta["nyx_scene"], nyx_scene_identical=report["nyx_scene_identical"],
               mismatches=report["mismatches"][:5], joint_map=jmap.report(),
               t_update_median=float(np.median([r["t_update"] for r in rec.rows])),
               t_rgb_median=float(np.median([r["t_rgb"] for r in rec.rows])),
               t_depth_median=float(np.median([r["t_depth"] for r in rec.rows])), rgb=summary)
    (a.out / "parity.json").write_text(json.dumps(dict(summary=out, per_frame=report["rgb"]), indent=1))
    print(json.dumps(out), flush=True)
    bridge.close()


def summarize(per_pair: dict) -> dict:
    out = {}
    for key, rows in per_pair.items():
        cp = np.array([r["changed_px"] for r in rows])
        ma = np.array([r["mean_abs"] for r in rows])
        mx = np.array([r["max_abs"] for r in rows])
        out[key] = dict(frames=len(rows), identical=int((cp == 0).sum()), changed_px_median=float(np.median(cp)),
                        changed_px_p95=float(np.percentile(cp, 95)), changed_px_max=int(cp.max()),
                        mean_abs_median=float(np.median(ma)), mean_abs_max=float(ma.max()), max_abs_max=int(mx.max()))
    return out


def stage_noise(a):
    refs = [Path(r) for r in a.ref]
    rows = [json.loads((r / "rows.json").read_text())["rows"] for r in refs]
    frames = [np.load(r / "frames.npz") for r in refs]
    per = {}
    exact = dict(nyx=0, depth=0, meta=0, qpos=0, frames=len(rows[0]))
    for i, (r0, r1) in enumerate(zip(rows[0], rows[1])):
        exact["nyx"] += int(r0["nyx_sha"] == r1["nyx_sha"])
        exact["depth"] += int(r0["depth_sha"] == r1["depth_sha"])
        exact["meta"] += int(r0["meta_sha"] == r1["meta_sha"])
        exact["qpos"] += int(r0["qpos_sha"] == r1["qpos_sha"])
        for c in r0["rgb_sha"]:
            per.setdefault(c, []).append(dict(i=i, step=r0["step"], **rgb_metrics(frames[0][f"{i}/{c}"],
                                                                                  frames[1][f"{i}/{c}"])))
    out = dict(stage="noise", refs=[str(r) for r in refs], exact=exact, rgb=summarize(per))
    (a.out / "noise.json").write_text(json.dumps(dict(summary=out, per_frame=per), indent=1))
    print(json.dumps(out), flush=True)


def stage_ghost(a):
    """A large state jump. A late state X (--jump-from control steps into the trajectory) is rendered 3 times, then the
    birth state B 6 times in a row (every render a fresh Nyx pass). reference: today's pipeline, the jump is
    Session.reset(). bridge (--ref: that reference run): X and B pushed through the host. Ghosting = the first frames
    after the jump differ from the settled one inside the region where X and B differ."""
    import torch  # noqa: F401
    force_nyx_options(a.render_mode)
    frames, t0 = {}, time.perf_counter()
    if a.ref:
        import mujoco
        import render_bridge
        bridge = render_bridge.build(os.environ["INSTANCE"])
        model = load_mjcf(a.mjcf) if a.mjcf else mirror_model(bridge)
        data = mujoco.MjData(model)
        jmap = bridge.bind(model)
        ref = np.load(Path(a.ref[0]) / "states.npz")

        def show(q, tag, n):
            data.qpos[:] = model.qpos0
            data.qpos[jmap.src] = q[jmap.dst]
            mujoco.mj_kinematics(model, data)
            bridge.update(model, data)
            for j in range(n):
                for c, v in bridge.render(force=True).items():
                    frames[f"{tag}{j}/{c}"] = v
        show(ref["qpos"][0], "X", 3)
        show(ref["qpos"][1], "B", 6)
        qx, qb = ref["qpos"][0], ref["qpos"][1]
        bridge.close()
    else:
        from rcb_topogym import Session, load_trajectory
        from topogym.utils.arrays import to_numpy
        path = os.environ["INSTANCE"]
        s = Session(json.load(open(path))["task"], instance=path, strict=True)
        acts, ticks = load_trajectory(a.traj, s.task, s.action_dim, trusted=True, layout=s.action_layout)
        world = s.env._world
        rs = world.scene.sim.rigid_solver

        def show(tag, n):
            for j in range(n):
                s.env._sensors.invalidate()
                for camera in world.cameras.values():
                    camera.invalidate_frame()
                for c, v in s.env.sensor_frame().rgb.items():
                    frames[f"{tag}{j}/{c}"] = np.array(v, copy=True)
            return np.asarray(to_numpy(rs.get_qpos()), dtype=np.float64).reshape(-1).copy()
        s.reset()
        for k in range(min(a.jump_from, len(acts))):
            if s.step(acts[k], _physics_steps=None if ticks is None else int(ticks[k])):
                break
        qx = show("X", 3)
        s.reset()
        qb = show("B", 6)
        s.close()
    np.savez_compressed(a.out / "frames.npz", **frames)
    np.savez_compressed(a.out / "states.npz", qpos=np.stack([qx, qb]))
    cams = sorted({k.split("/")[1] for k in frames})
    report = {}
    for c in cams:
        settled = frames[f"B5/{c}"].astype(np.int16)
        moved = np.abs(frames[f"X2/{c}"].astype(np.int16) - settled).max(-1) > 30
        rows = []
        for j in range(5):
            d = np.abs(frames[f"B{j}/{c}"].astype(np.int16) - settled).max(-1)
            rows.append(dict(frame=j, changed_px=int((d > 0).sum()), in_moved_region_mean=float(d[moved].mean()),
                             outside_mean=float(d[~moved].mean()), in_moved_region_over_20=int((d[moved] > 20).sum())))
        report[c] = dict(moved_region_px=int(moved.sum()), first_frames_vs_settled=rows)
    (a.out / "ghost.json").write_text(json.dumps(dict(stage="ghost", bridge=bool(a.ref), jump_from=a.jump_from,
                                                      wall_s=time.perf_counter() - t0, cameras=report), indent=1))
    print(json.dumps(dict(stage="ghost", bridge=bool(a.ref), cameras={
        c: [(r["changed_px"], round(r["in_moved_region_mean"], 3), r["in_moved_region_over_20"])
            for r in v["first_frames_vs_settled"]] for c, v in report.items()})), flush=True)


def skin_checks() -> dict:
    """Deformable skins (numpy + trimesh): exact at birth, rigid motions carried rigidly, a bent chain stays closed."""
    import render_bridge
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(0)
    checks = {}
    links, length, radius = 6, 0.01, 0.003                       # a straight tube along x, one link per 10 mm
    s = rng.uniform(0, links * length, 2000)
    angle = rng.uniform(0, 2 * np.pi, 2000)
    rest = np.column_stack([s, radius * np.cos(angle), radius * np.sin(angle)])
    pos = np.column_stack([np.arange(links) * length, np.zeros(links), np.zeros(links)])
    rot = np.tile(np.eye(3).reshape(1, 9), (links, 1))
    skin = render_bridge.ChainSkin(range(links), rest, s, np.arange(links + 1) * length, pos, rot.reshape(-1, 3, 3))
    checks["chain exact at birth"] = bool(np.abs(skin(pos, rot) - rest).max() < 1e-15)
    R = Rotation.from_rotvec([0.3, -0.5, 0.9])
    moved = skin(R.apply(pos) + [0.1, 0.2, 0.3], (R.as_matrix() @ rot.reshape(-1, 3, 3)).reshape(-1, 9))
    checks["chain rigid motion"] = bool(np.abs(moved - (R.apply(rest) + [0.1, 0.2, 0.3])).max() < 1e-12)
    bend = Rotation.from_rotvec([0, 0, np.radians(30)])          # links 3.. turn 30 deg about the joint at x = 30 mm
    pos2, rot2 = pos.copy(), rot.reshape(-1, 3, 3).copy()
    pos2[3:] = bend.apply(pos[3:] - pos[3]) + pos[3]
    rot2[3:] = bend.as_matrix() @ rot2[3:]
    skinned = skin(pos2, rot2.reshape(-1, 9))
    near = np.abs(s - 0.03) < 1e-4                               # material within 0.1 mm of the joint
    centre = skinned[near].mean(axis=0)
    checks["chain closed at a bent joint"] = bool(np.linalg.norm(skinned[near] - centre, axis=1).max() < 1.5 * radius)
    grid = np.stack(np.meshgrid(np.linspace(0, .2, 9), np.linspace(0, .1, 5), indexing="ij"), -1).reshape(-1, 2)
    verts = np.column_stack([grid, np.zeros(len(grid))])
    faces = np.array([[i * 5 + j, i * 5 + j + 5, i * 5 + j + 1] for i in range(8) for j in range(4)]
                     + [[i * 5 + j + 1, i * 5 + j + 5, i * 5 + j + 6] for i in range(8) for j in range(4)])
    on_vertices = render_bridge.SurfaceSkin(verts, verts, faces)
    checks["surface: points = vertices, exact"] = bool(np.abs(on_vertices(verts) - verts).max() < 1e-15)
    points = np.column_stack([rng.uniform(0, .2, 500), rng.uniform(0, .1, 500), rng.uniform(-.002, .002, 500)])
    coarse = render_bridge.SurfaceSkin(points, verts, faces)
    checks["surface: off-surface points exact at birth"] = bool(np.abs(coarse(verts) - points).max() < 1e-12)
    checks["surface: rigid motion"] = bool(np.abs(coarse(R.apply(verts)) - R.apply(points)).max() < 1e-12)
    return checks


def stage_selftest(a):
    """CPU only (numpy, mujoco, scipy, trimesh; no Genesis): the joint-matching rules and their refusals on a stand-in
    host, and the deformable skins."""
    from types import SimpleNamespace as NS
    import mujoco
    import render_bridge

    solver = object()

    def entity(q_start, links, joints):
        ls = {name: NS(name=name, idx=idx, parent_idx=parent) for name, idx, parent in links}
        js = [NS(name=n, type=t, qs_idx_local=list(range(q0, q0 + nq)), link=ls[link]) for n, t, q0, nq, link in joints]
        base = min(ls.values(), key=lambda l: l.idx)
        return NS(_solver=solver, q_start=q_start, n_qs=sum(len(j.qs_idx_local) for j in js), joints=js,
                  links=list(ls.values()), base_link_idx=base.idx)

    host = dict(
        robot=entity(0, [("world", 0, -1), ("left/shoulder_link", 1, 0), ("left/finger_link", 2, 1)],
                     [("left/waist", 1, 0, 1, "left/shoulder_link"), ("left/left_finger", 2, 1, 1, "left/finger_link")]),
        piece_0=entity(2, [("piece_0", 3, -1)], [("root_joint", 4, 0, 7, "piece_0")]),
        piece_1=entity(9, [("piece_1", 4, -1)], [("root_joint", 4, 0, 7, "piece_1")]),
        fiber_bundle=entity(16, [("fixed_terminal", 5, -1), ("lead_00", 6, 5)], [("lead_bend_00", 3, 0, 4, "lead_00")]),
        table=NS(_solver=solver, n_qs=0))
    env = NS(_world=NS(entities=host, scene=NS(sim=NS(rigid_solver=solver))))

    def model(waist=mujoco.mjtJoint.mjJNT_HINGE, bundle=True):
        spec = mujoco.MjSpec()
        w = spec.worldbody
        b = w.add_body(name="left/shoulder_link")
        b.add_joint(name="left/waist", type=waist)
        b.add_geom(size=[.01, 0, 0])
        b = b.add_body(name="left/finger_link")
        b.add_joint(name="robot::left/left_finger", type=mujoco.mjtJoint.mjJNT_SLIDE)
        b.add_geom(size=[.01, 0, 0])
        b = w.add_body(name="claw_0")          # unnamed free joint, entity renamed: needs the alias piece_0=claw_0
        b.add_freejoint()
        b.add_geom(size=[.01, 0, 0])
        b = w.add_body(name="piece_1::piece_1")    # the MuJoCo core's naming: <entity>::<link>, free joint <entity>::root
        b.add_freejoint(name="piece_1::root")
        b.add_geom(size=[.01, 0, 0])
        if bundle:
            b = w.add_body(name="fixed_terminal").add_body(name="lead_00")
            b.add_joint(type=mujoco.mjtJoint.mjJNT_BALL)   # unnamed: matched through its body
            b.add_geom(size=[.01, 0, 0])
        return spec.compile()

    checks = {}
    m = model()
    jm = render_bridge.map_joints(env, m, aliases={"piece_0": "claw_0"})
    rules = {g: r for g, _, r in jm.pairs}
    checks["rules"] = rules == {"robot/left/waist": "name", "robot/left/left_finger": "prefixed name",
                                "piece_0/root_joint": "body", "piece_1/root_joint": "body",
                                "fiber_bundle/lead_bend_00": "body"}
    d = mujoco.MjData(m)
    d.qpos[:] = np.arange(m.nq) + 0.5
    host_q = np.zeros(20)
    host_q[jm.dst] = d.qpos[jm.src]
    checks["coordinates"] = (jm.dst.size == 20 and sorted(jm.dst.tolist()) == list(range(20))
                             and host_q[2:9].tolist() == d.qpos[2:9].tolist())
    for name, kwargs, words in (("missing alias", dict(), "piece_0/root_joint"),
                                ("type mismatch", dict(model=model(waist=mujoco.mjtJoint.mjJNT_SLIDE)), "type or size"),
                                ("no bundle", dict(model=model(bundle=False)), "fiber_bundle/lead_bend_00")):
        try:
            render_bridge.map_joints(env, kwargs.get("model", m))
            checks[name] = False
        except render_bridge.MappingError as exc:
            checks[name] = words in str(exc)
    kept = render_bridge.map_joints(env, model(bundle=False), aliases={"piece_0": "claw_0"},
                                    keep=["fiber_bundle/lead_bend_00"])
    checks["keep"] = kept.kept == ("fiber_bundle/lead_bend_00",) and kept.dst.size == 16
    checks.update(skin_checks())
    (a.out / "selftest.json").write_text(json.dumps(checks, indent=1))
    print(json.dumps(dict(stage="selftest", passed=all(checks.values()), checks=checks)), flush=True)
    if not all(checks.values()):
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["reference", "bridge", "noise", "ghost", "selftest"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--traj", type=Path)
    ap.add_argument("--steps", default="0-200,210-99999:10")
    ap.add_argument("--ref", action="append", default=[])
    ap.add_argument("--mjcf")
    ap.add_argument("--alias", action="append", default=[], help="genesis_entity=mujoco_prefix")
    ap.add_argument("--keep", action="append", default=[], help="entity/joint kept at its frozen value")
    ap.add_argument("--render-mode", default=None, help="test only: Nyx ERenderMode for every camera (both stages)")
    ap.add_argument("--jump-from", type=int, default=400, help="ghost: control steps into the trajectory before the jump")
    ap.add_argument("--pace", type=float, default=0.0, help="reference: extra seconds between captures (frame-timing probe)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    dict(reference=stage_reference, bridge=stage_bridge, noise=stage_noise, ghost=stage_ghost,
         selftest=stage_selftest)[a.stage](a)


if __name__ == "__main__":
    main()
