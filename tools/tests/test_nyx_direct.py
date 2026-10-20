#!/usr/bin/env python
"""nyx_direct acceptance: Nyx without Genesis against today's host path (render_bridge) on the same MuJoCo states.
Needs the image's Python with CUDA and RT cores (the host stage also needs genesis-world and gs-nyx-plugin).

host     today's path: a MuJoCo Session replays a reference; at N states along it the render host the Session
         builds (rcb_topogym._render_bridge: Genesis host + gs-nyx-plugin; a rope instance on its released Genesis
         host, its rope chains bound at birth) renders the rig. Records the exported Nyx scene folder, the reference
         table, every native NyxRenderer call, the static deformable data, and per state: MuJoCo qpos, the Nyx input
         buffers, RGB, depth, timings.
direct   nyx_direct (from_world with the instance) on the same states in a fresh process (genesis is never
         imported; checked): compares the static scene (instances, materials, cameras, lights, environment;
         generated assets by the instance's asset identity), the reference table, the static deformable data, the
         input buffers bit for bit, RGB against the host and against the host's own run-to-run spread (two host
         recordings), MuJoCo raster depth against Genesis' raster depth, timings.
spread   RGB of every pair of runs by pair type (host-host, direct-host, direct-direct).
bench    per-frame cost of one rig render (state in, RGB of every camera on the host) through --path host|direct,
         on a reference's states (--traj) or a host recording's (--host: its qpos, no replay).
ab       the per-frame cost of both paths in one process, alternating frame by frame on a host recording's states
         (the same GPU contention for both); the median ratio direct / host.
selftest CPU only: the camera arithmetic replica against torch, pose helpers against MuJoCo, URDF visual parsing.

  python scripts/robotangle/test_nyx_direct.py host --instance I --traj T --out H1 [--states 8]
  python scripts/robotangle/test_nyx_direct.py direct --instance I --traj T --host H1 --host H2 --out D
  python scripts/robotangle/test_nyx_direct.py spread --host H1 --host H2 --direct D1 --direct D2 [--out S.json]
  python scripts/robotangle/test_nyx_direct.py bench --path direct --instance I --host H1 --out B.json
  python scripts/robotangle/test_nyx_direct.py ab --instance I --host H1 --out AB.json
  python scripts/robotangle/test_nyx_direct.py selftest
Every output directory must be new.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "shared"))


def _session(instance):
    from rcb_topogym import Session
    inst = json.load(open(instance))
    s = Session(inst["task"], instance=instance, strict=True)
    s.env._world.renderer_factory = None
    return inst, s


def _host(instance, world):
    """Today's render host of the instance, as the Session builds it (rcb_topogym._render_bridge: a rope instance
    renders on its released Genesis host, its rope chains bound at birth).

    A cloth instance needs its released Genesis instance in instance_assets/render_host/ (its own MuJoCo configuration
    does not restore the PBD archive), and two test-only changes (RENDER_BRIDGE.md, Acceptance: blanket):
    - the archive's backend identity (its GPU profile) is not checked: the host never steps, the restore only seeds
      the PBD solver's tables, and every particle is overwritten each frame;
    - the host's particles are the MuJoCo cloth's particle view (Cloth.particles_of), as nyx_direct's, instead of
      render_bridge.bind_flex's skin of the released particles, which keeps a settled birth's displacement."""
    import rcb_topogym
    cloths = getattr(world, "_cloths", None) or {}
    if not cloths:
        return rcb_topogym._render_bridge(str(Path(instance).resolve()), world)
    import topogym.world.physics.pbd_initial_state as pbd_state
    check = pbd_state.require_compatible
    pbd_state.require_compatible = lambda *args, **kwargs: None
    try:
        bridge = rcb_topogym._render_bridge(str(Path(instance).resolve()), world)
    finally:
        pbd_state.require_compatible = check
    for name, cloth in cloths.items():
        entity = bridge.world.entities[name]
        bridge._deformables[name] = (entity, "pbd", lambda data, cloth=cloth: cloth.particles_of(cloth.vertices(data)))
    return bridge


def _recorded_states(s, host):
    """Yield (step, model, data) at a host recording's states without replaying: its MuJoCo qpos, forward kinematics
    and flex vertex positions (all the renderers read)."""
    import mujoco
    h = np.load(Path(host) / "frames.npz")
    s.reset()
    world = s.env._world
    for step, qpos in zip(h["step"], h["qpos"]):
        world.data.qpos[:] = qpos
        mujoco.mj_kinematics(world.model, world.data)
        if world.model.nflex:       # a cloth's flex vertex positions (mj_flex), which bind_flex reads
            mujoco.mj_flex(world.model, world.data)
        yield int(step), world.model, world.data


def _states(s, traj, count=None, steps=None):
    """Yield (step, model, data) at `steps`, or at `count` control steps spread over the reference (birth and end
    included)."""
    from rcb_topogym import load_trajectory
    acts, ticks = load_trajectory(traj, s.task, s.action_dim, trusted=True, layout=s.action_layout)
    rows = len(acts)
    picks = steps if steps is not None else sorted({0, rows} | {int(round(rows * f)) for f in np.linspace(0, 1, count)})
    s.reset()
    world, k = s.env._world, 0
    for step in picks:
        while k < step:
            s.step(acts[k], _physics_steps=None if ticks is None else int(ticks[k]))
            k += 1
        yield step, world.model, world.data


def rgb_metrics(a, b) -> dict:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    m = d.max(-1)
    return dict(mean_abs=round(float(d.mean()), 4), px_gt8=int((m > 8).sum()), max=int(m.max()))


def depth_metrics(mine, ref) -> dict:
    both = (mine > 0) & (ref > 0)
    d = np.abs(mine[both].astype(np.float64) - ref[both])
    return dict(valid_both=int(both.sum()), only_mine=int(((mine > 0) & ~(ref > 0)).sum()),
                only_host=int((~(mine > 0) & (ref > 0)).sum()), p50_mm=round(1e3 * float(np.median(d)), 4),
                p99_mm=round(1e3 * float(np.percentile(d, 99)), 3), lt_1mm=round(float((d < 1e-3).mean()), 5),
                lt_5mm=round(float((d < 5e-3).mean()), 5))


# ------------------------------------------------------------------------------------------- host (today's path)
def stage_host(a):
    import gs_nyx.nyx_py_renderer as npr
    calls, table = [], {}
    real = npr.NyxRenderer

    class Recorder:
        """Forwards every native call (the plugin aliases npr.NyxRenderer at import, so it is swapped first)."""

        def __init__(self):
            object.__setattr__(self, "_real", real())

        def __getattr__(self, name):
            attr = getattr(self._real, name)
            if not callable(attr):
                return attr

            def call(*args):
                if args and isinstance(args[0], npr.BridgeUpdateDesc):
                    d = args[0]
                    table.update(numCameras=d.numCameras, numRigidGeom=d.numRigidGeom,
                                 numDeformVertices=d.numDeformVertices, numDeformTriangles=d.numDeformTriangles,
                                 numDeformObjects=d.numDeformObjects,
                                 references=[[int(r.targetUUID), r.nodeName] for r in
                                             (d.get_reference(i) for i in range(d.reference_size()))])
                calls.append([name] + [x if isinstance(x, (int, str)) else type(x).__name__ for x in args])
                return attr(*args)
            return call

    npr.NyxRenderer = Recorder
    a.out.mkdir(parents=True)
    _, s = _session(a.instance)
    t0 = time.perf_counter()
    b = _host(a.instance, s.env._world)
    build_s = time.perf_counter() - t0
    store = {}
    for i, (step, model, data) in enumerate(_states(s, a.traj, a.states)):
        t = time.perf_counter()
        b.update(model, data)
        rgb = b.render()
        render_s = time.perf_counter() - t
        if i == 0:      # the renderer (and its scene export) exists after the first RGB request
            meta = next(c.sensor._shared_metadata for c in b.world.cameras.values() if getattr(c, "sensor", None))
            from gs_nyx_plugin.nyx_camera_shared_metadata import _nyx_make_complete_path
            shutil.copytree(_nyx_make_complete_path(meta.scene_description_export_path), a.out / "scene")
            if meta.renderer._num_deform_verts:      # static deformable data, uploaded once
                np.savez(a.out / "deform_static.npz", uvs=meta.renderer._deformable_uvs_cuda.cpu().numpy(),
                         indices=meta.renderer._deformable_indices_cuda.cpu().numpy())
        rec = dict(step=step, qpos=np.array(data.qpos), render_s=render_s,
                   **{f"in_{k}": v for k, v in b.nyx_inputs().items()},
                   **{f"rgb_{k}": v for k, v in rgb.items()}, **{f"depth_{k}": v for k, v in b.depth().items()})
        for k, v in rec.items():
            store.setdefault(k, []).append(v)
    np.savez_compressed(a.out / "frames.npz", **{k: np.stack([np.asarray(x) for x in v]) for k, v in store.items()})
    (a.out / "host.json").write_text(json.dumps(dict(build_s=round(build_s, 2), calls=calls, table=table)))
    print(json.dumps(dict(out=str(a.out), build_s=round(build_s, 1), states=[int(x) for x in store["step"]],
                          native_calls=len(calls))), flush=True)
    b.close()
    s.close()


# ------------------------------------------------------------------------------------------- direct (no Genesis)
def _static_differences(mine, ref, mine_dir, ref_dir, identity, asset_root) -> list:
    """Field-by-field difference of two nyx_scene.json documents: UUIDs as a pattern, mesh URIs by content (meshes
    exported into the scene folders by their bytes, generated assets by the instance's asset identity)."""
    def content(uri, folder):
        p = Path(folder) / Path(uri).name
        if p.is_file():
            return "obj:" + hashlib.sha256(p.read_bytes()).hexdigest()
        p = Path(uri)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if asset_root is not None and p.is_relative_to(asset_root):
            return "asset:" + p.relative_to(asset_root).as_posix()
        return "path:" + str(p)

    def material(mat):
        # a texture that is a generated asset (each Session writes its own copy): by name and frozen identity
        if not isinstance(mat, dict):
            return mat
        return {k: f"asset:{Path(v).name}:{identity[Path(v).name]}"
                if k.endswith("_texture") and isinstance(v, str) and Path(v).name in identity else v for k, v in mat.items()}

    out, uuids = [], {}
    if len(mine["instance_array"]) != len(ref["instance_array"]):
        out.append(("instances", len(mine["instance_array"]), len(ref["instance_array"])))
    for i, (m, r) in enumerate(zip(mine["instance_array"], ref["instance_array"])):
        for k in sorted(set(m) | set(r)):
            if k == "uuid":
                if uuids.setdefault(r[k], m[k]) != m[k]:
                    out.append((i, k, "uuid pattern"))
            elif k == "uri":
                cm, cr = content(m[k], mine_dir), content(r[k], ref_dir)
                if cm.startswith("asset:"):       # a generated asset: the host's bytes are the frozen identity
                    rel = cm[len("asset:"):]
                    data = (asset_root / rel).read_bytes().replace(str(asset_root).encode(), b"<instance-assets>")
                    if hashlib.sha256(data).hexdigest() != identity.get(rel) or Path(r[k]).name != Path(rel).name:
                        out.append((i, k, m[k], r[k], "asset identity differs"))
                elif cm != cr:
                    out.append((i, k, m[k], r[k]))
            elif k == "material_override" and material(m.get(k)) == material(r.get(k)):
                continue
            elif m.get(k) != r.get(k):
                out.append((i, k, m.get(k), r.get(k)))
    for key in ("version", "asset_root_path", "camera_array", "light_array", "environment_map_array",
                "animation_array", "light_field_array"):
        if mine.get(key) != ref.get(key):
            out.append((key, mine.get(key), ref.get(key)))
    return out


def stage_direct(a):
    import nyx_direct
    a.out.mkdir(parents=True)
    inst, s = _session(a.instance)
    world = s.env._world
    t0 = time.perf_counter()
    nyx = nyx_direct.NyxDirect.from_world(world, instance=a.instance)
    build_s = time.perf_counter() - t0
    shutil.copytree(nyx.scene_path.parent, a.out / "scene")
    host = a.host[0]
    ref = json.loads((host / "scene" / "nyx_scene.json").read_text())
    mine = json.loads((a.out / "scene" / "nyx_scene.json").read_text())
    root = getattr(getattr(s.env, "_asset_build", None), "root", None)
    static = _static_differences(mine, ref, a.out / "scene", host / "scene", inst.get("asset_identity") or {},
                                 None if root is None else Path(root).resolve())
    table = json.loads((host / "host.json").read_text())["table"]
    pattern = {}
    refs_ok = (table["numRigidGeom"] == len(nyx.slots) and table["numCameras"] == len(nyx.cams)
               and table.get("numDeformObjects", 0) == len(nyx.deformables)
               and table.get("numDeformVertices", 0) == sum(d["n_vertices"] for d in nyx.deformables)
               and table.get("numDeformTriangles", 0) == sum(d["n_triangles"] for d in nyx.deformables)
               and [n for _, n in nyx.references] == [n for _, n in table["references"]]
               and all(pattern.setdefault(r[0], m[0]) == m[0] for m, r in zip(nyx.references, table["references"])))
    deform_static = None
    if nyx.deformables:
        ref_static = np.load(host / "deform_static.npz")
        uvs = np.empty((sum(d["n_vertices"] for d in nyx.deformables), 2), dtype=np.float32)
        tri = np.empty(3 * sum(d["n_triangles"] for d in nyx.deformables), dtype=np.int32)
        nyx.cuda.download(nyx._dev_uvs, uvs)
        nyx.cuda.download(nyx._dev_tri, tri)
        deform_static = dict(uvs_equal=bool(np.array_equal(uvs, ref_static["uvs"])),
                             indices_equal=bool(np.array_equal(tri, ref_static["indices"])))
    hosts = [np.load(h / "frames.npz") for h in a.host]
    frames, store = [], {}
    for j, (step, model, data) in enumerate(_states(s, a.traj, steps=[int(x) for x in hosts[0]["step"]])):
        h = hosts[0]
        if step != int(h["step"][j]) or not np.array_equal(np.asarray(data.qpos), h["qpos"][j]):
            raise RuntimeError(f"state {step}: not the host's recorded state")
        t = time.perf_counter()
        nyx.update(model, data)
        rgb = nyx.render()
        render_s = time.perf_counter() - t
        depth = nyx.depth()
        inp = nyx.inputs()
        row = dict(step=step, render_ms=round(1e3 * render_s, 2), host_render_ms=round(1e3 * float(h["render_s"][j]), 2))
        for k, x in inp.items():
            same = (x.view(np.uint32) == h[f"in_{k}"][j].view(np.uint32)).all(axis=-1)
            row[k] = [int(same.sum()), int(same.size), float(np.abs(x.astype(np.float64) - h[f"in_{k}"][j]).max())]
        for n in nyx.names:
            row[f"rgb_{n}"] = rgb_metrics(rgb[n], h[f"rgb_{n}"][j])
            if len(hosts) > 1:
                row[f"rgb_{n}_host_spread"] = rgb_metrics(hosts[1][f"rgb_{n}"][j], h[f"rgb_{n}"][j])
            row[f"depth_{n}"] = depth_metrics(depth[n], h[f"depth_{n}"][j])
        frames.append(row)
        print(json.dumps(row), flush=True)
        for k, v in dict(step=step, **{f"rgb_{n}": x for n, x in rgb.items()}, **{f"depth_{n}": x for n, x in depth.items()},
                         **{f"in_{k}": x for k, x in inp.items()}).items():
            store.setdefault(k, []).append(v)
    np.savez_compressed(a.out / "frames.npz", **{k: np.stack([np.asarray(x) for x in v]) for k, v in store.items()})
    buffers = {k: [sum(f[k][0] for f in frames), sum(f[k][1] for f in frames)] for k in inp}
    report = dict(build_s=round(build_s, 2), export_s=round(nyx.export_s, 2), camera_device=nyx.camera_device,
                  static_differences=static, static_identical=not static, reference_table_identical=refs_ok,
                  deform_static=deform_static, buffers_bitwise=buffers, genesis_imported="genesis" in sys.modules,
                  frames=frames)
    (a.out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, default=str), flush=True)
    nyx.close()
    s.close()


# ------------------------------------------------------------------------------------------- spread
def stage_spread(a):
    """RGB of every pair of runs (host / direct), by pair type: Nyx's own run-to-run spread is the yardstick."""
    runs = [("host", np.load(d / "frames.npz")) for d in a.host] + [("direct", np.load(d / "frames.npz")) for d in a.direct]
    cams = sorted(k[4:] for k in runs[0][1].files if k.startswith("rgb_"))
    table = {}
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            kind = "-".join(sorted((runs[i][0], runs[j][0])))
            for c in cams:
                x, y = runs[i][1][f"rgb_{c}"].astype(np.int16), runs[j][1][f"rgb_{c}"].astype(np.int16)
                d = np.abs(x - y)
                per_frame = d.reshape(len(d), -1).mean(axis=1)
                px = (d.max(-1) > 8).reshape(len(d), -1).sum(axis=1)
                t = table.setdefault(c, {}).setdefault(kind, dict(mean_abs=[], px_gt8_max=[], identical_frames=[]))
                t["mean_abs"].append(round(float(per_frame.mean()), 5))
                t["px_gt8_max"].append(int(px.max()))
                t["identical_frames"].append(int((per_frame == 0).sum()))
    print(json.dumps(table, indent=1), flush=True)
    if a.out:
        a.out.write_text(json.dumps(table, indent=1))


# ------------------------------------------------------------------------------------------- bench
def stage_bench(a):
    _, s = _session(a.instance)
    world = s.env._world
    t0 = time.perf_counter()
    if a.path == "host":
        r = _host(a.instance, world)
    else:
        import nyx_direct
        r = nyx_direct.NyxDirect.from_world(world, instance=a.instance)
    build_s = time.perf_counter() - t0
    frames, warm = [], False
    for step, model, data in (_recorded_states(s, a.host) if a.host else _states(s, a.traj, a.states)):
        if not warm:                       # first frame: pipeline warm-up, not timed
            r.update(model, data)
            r.render()
            warm = True
        for _ in range(a.repeat):
            t1 = time.perf_counter()
            r.update(model, data)
            r.render()
            frames.append(1e3 * (time.perf_counter() - t1))
    out = dict(path=a.path, build_s=round(build_s, 2), frames=len(frames), ms_median=round(float(np.median(frames)), 2),
               ms_p10=round(float(np.percentile(frames, 10)), 2), ms_p90=round(float(np.percentile(frames, 90)), 2),
               genesis_imported="genesis" in sys.modules, start=time.time() - sum(frames) / 1e3, end=time.time())
    a.out.write_text(json.dumps(out))
    print(json.dumps(out), flush=True)
    r.close()
    s.close()


def stage_ab(a):
    """Today's host and nyx_direct built in one process, alternating frame by frame on a host recording's states: both
    paths see the same GPU contention (separate processes on a shared GPU land in different contention regimes)."""
    import mujoco
    import nyx_direct
    _, s = _session(a.instance)
    s.reset()
    world = s.env._world
    model, data = world.model, world.data
    t0 = time.perf_counter()
    paths = dict(host=_host(a.instance, world))
    host_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    paths["direct"] = nyx_direct.NyxDirect.from_world(world, instance=a.instance)
    direct_build = time.perf_counter() - t0
    rec = np.load(a.host / "frames.npz")
    for r in paths.values():                   # pipeline warm-up, not timed
        data.qpos[:] = rec["qpos"][0]
        mujoco.mj_kinematics(model, data)
        if model.nflex:
            mujoco.mj_flex(model, data)
        r.update(model, data)
        r.render()
    times = dict(host=[], direct=[])
    for rnd in range(a.rounds):
        for j, qpos in enumerate(rec["qpos"]):
            data.qpos[:] = qpos
            mujoco.mj_kinematics(model, data)
            if model.nflex:
                mujoco.mj_flex(model, data)
            order = ("host", "direct") if (rnd + j) % 2 == 0 else ("direct", "host")
            for _ in range(a.repeat):
                for p in order:
                    t1 = time.perf_counter()
                    paths[p].update(model, data)
                    paths[p].render()
                    times[p].append(1e3 * (time.perf_counter() - t1))
    ratio = np.array(times["direct"]) / np.array(times["host"])
    out = dict(host_build_s=round(host_build, 2), direct_build_s=round(direct_build, 2), frames=len(ratio),
               **{p: dict(ms_median=round(float(np.median(v)), 2), ms_p10=round(float(np.percentile(v, 10)), 2),
                          ms_p90=round(float(np.percentile(v, 90)), 2)) for p, v in times.items()},
               direct_over_host=dict(median=round(float(np.median(ratio)), 3), p10=round(float(np.percentile(ratio, 10)), 3),
                                     p90=round(float(np.percentile(ratio, 90)), 3)))
    a.out.write_text(json.dumps(dict(**out, times=times)))
    print(json.dumps(out), flush=True)
    paths["host"].close()
    paths["direct"].close()      # gs_nyx then reports the plugin's assets as leaked at shutdown (two renderers, one SDK)
    s.close()


# ------------------------------------------------------------------------------------------- selftest
def stage_selftest(a):
    import mujoco
    import nyx_direct as nd
    rng = np.random.default_rng(0)
    checks = {}
    # pose helpers against MuJoCo's own quaternion arithmetic
    worst = 0.0
    for _ in range(200):
        q1, q2 = rng.normal(size=4), rng.normal(size=4)
        q1, q2 = q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2)
        ref = np.empty(4)
        mujoco.mju_mulQuat(ref, q1, q2)
        v, rv = rng.normal(size=3), np.empty(3)
        mujoco.mju_rotVecQuat(rv, v, q1)
        worst = max(worst, float(np.abs(nd.quat_mul(q1, q2) - ref).max()), float(np.abs(nd.quat_rotate(q1, v) - rv).max()))
    checks["quaternions_vs_mujoco_max_abs"] = worst
    # camera arithmetic: the numpy replica against torch's float32 kernels (the reference wherever torch exists)
    torch = nd._torch()
    if torch is not None:
        agree = 0
        for _ in range(300):
            p, l, u = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3), rng.normal(size=3)
            _, qt = nd.camera_quat_f32(p, l, u, device="cpu")
            nd._TORCH, saved = False, nd._TORCH
            try:
                _, qn = nd.camera_quat_f32(p, l, u, device="cpu")
            finally:
                nd._TORCH = saved
            agree += int((qt.view(np.uint32) == qn.view(np.uint32)).all())
        checks["camera_numpy_replica_bitwise_of_300"] = agree
    # URDF visual parsing (origin pose, mesh scale)
    import tempfile
    import trimesh
    with tempfile.TemporaryDirectory() as tmp:
        trimesh.creation.box(extents=(1, 2, 3)).export(Path(tmp) / "m.obj")
        (Path(tmp) / "r.urdf").write_text('<robot name="r"><link name="a"><visual><origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/>'
                                          '<geometry><mesh filename="m.obj" scale="2 2 2"/></geometry></visual></link></robot>')
        (link, pos, quat, shape), = nd.urdf_visuals(Path(tmp) / "r.urdf")
        checks["urdf_visual"] = dict(link=link, pos=pos.tolist(), quat=np.round(quat, 12).tolist(),
                                     mesh_extent=np.ptp(shape[1], axis=0).tolist())
    ok = (checks["quaternions_vs_mujoco_max_abs"] < 1e-12 and checks["urdf_visual"]["mesh_extent"] == [2.0, 4.0, 6.0]
          and abs(checks["urdf_visual"]["quat"][0] - 0.7071067811865476) < 1e-12)
    print(json.dumps(dict(ok=bool(ok), **checks)), flush=True)
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    h = sub.add_parser("host")
    h.add_argument("--instance", required=True)
    h.add_argument("--traj", required=True)
    h.add_argument("--out", type=Path, required=True)
    h.add_argument("--states", type=int, default=8)
    d = sub.add_parser("direct")
    d.add_argument("--instance", required=True)
    d.add_argument("--traj", required=True)
    d.add_argument("--host", type=Path, action="append", required=True)
    d.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("bench")
    b.add_argument("--path", choices=("host", "direct"), required=True)
    b.add_argument("--instance", required=True)
    b.add_argument("--traj", help="replay this reference to --states states")
    b.add_argument("--host", type=Path, help="or render a host recording's states (no replay)")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--states", type=int, default=8)
    b.add_argument("--repeat", type=int, default=5)
    ab = sub.add_parser("ab")
    ab.add_argument("--instance", required=True)
    ab.add_argument("--host", type=Path, required=True)
    ab.add_argument("--out", type=Path, required=True)
    ab.add_argument("--rounds", type=int, default=3)
    ab.add_argument("--repeat", type=int, default=5)
    sp = sub.add_parser("spread")
    sp.add_argument("--host", type=Path, action="append", required=True)
    sp.add_argument("--direct", type=Path, action="append", default=[])
    sp.add_argument("--out", type=Path)
    sub.add_parser("selftest")
    a = ap.parse_args()
    dict(host=stage_host, direct=stage_direct, spread=stage_spread, bench=stage_bench, ab=stage_ab,
         selftest=stage_selftest)[a.stage](a)


if __name__ == "__main__":
    main()
