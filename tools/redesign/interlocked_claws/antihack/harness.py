"""Old-oracle replay harness: a strict reference's rows on a NEW scene, as-is and retargeted (the old TCP path moved
rigidly with the object's start pose, re-solved by IK), plus a geometric check and an export of the old TCP program
for object-level labs (virtual hands).

Session level (a new scene that builds as a MuJoCo Session: a frozen instance, or a family builder function):
  python -m antihack.harness session --old-instance OLD.json --old-rows OLD_rows.npz [--old-record OLD_rec.npz]
         (--new-instance NEW.json | --builder pkg.module:function [--builder-args '{"k": v}'])
         [--variants as_is,retargeted,nop] [--retarget auto | none | transform.json] [--hold 50] [--every 10]
         --family F --out DIR
  -> DIR/<variant>.npz/.json (antihack.record) + DIR/harness.json: per variant success, first success row, closest
     approach to the goal (family-aware), collisions (robot-obstacle rows, deepest), grasp events, IK retarget errors;
     its "old_oracle_on_new_scene" block goes straight into antihack.report.build_report(old_on_new=...).
     Run it through launch.sh: each variant replays every row (~0.2-0.7 s per row).
Geometric (no physics; labelled as an argument, not a replay):
  python -m antihack.harness geometric --old-record OLD_rec.npz --new-instance NEW.json --family F
         [--transform auto|none|file] --out check.json
  -> the old oracle's FINAL object state, moved by the transform, against the new scene's goal geometry (rope_threading:
     ring passages / order of the new rings; knot: crossings; cloth: layers / coverage proxy), and the old TCP path
     (retargeted) against the new scene's obstacles (robot geoms vs static obstacles, swept row by row).
Export for object-level labs:
  python -m antihack.harness export --old-instance OLD.json --old-rows OLD_rows.npz [--old-record OLD_rec.npz]
         [--transform file.json] --out program.json
  -> {"dt", "arms": {arm: [[t, x, y, z, qw, qx, qy, qz, closedness, holding], ...]}, "grasp_events": [...]}

Transforms: {"R": 3x3, "t": [3]} applied as p' = R p + t (world frame); "auto" = a planar (yaw + translation, plus the
mean height change) fit of the object's start state old -> new (rope nodes resampled by arc length; cloth particles).
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np

from .aloha import ARMS, CONTROL_DT_S, JOINT_SPEED_LIMIT_RAD_S, AlohaFK, gripper_closedness, layout_columns


# ------------------------------------------------------------------------------------------------ transforms
def resample_polyline(P, n):
    P = np.asarray(P, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, P[:, k]) for k in range(3)], axis=1)


def planar_fit(src, dst) -> dict:
    """Yaw + translation (xy) and the mean height change mapping src points onto dst points (matched by index)."""
    A, B = np.asarray(src, float), np.asarray(dst, float)
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    H = (A[:, :2] - ca[:2]).T @ (B[:, :2] - cb[:2])
    U, S, Vt = np.linalg.svd(H)
    R2 = Vt.T @ np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    R = np.eye(3); R[:2, :2] = R2
    t = cb - R @ ca
    res = B - (A @ R.T + t)
    return dict(R=R.tolist(), t=t.tolist(), yaw_deg=float(np.degrees(np.arctan2(R2[1, 0], R2[0, 0]))),
                translation_m=float(np.linalg.norm(t + R @ ca - ca)), residual_rms_m=float(np.sqrt((res ** 2).sum(1).mean())))


def identity() -> dict:
    return dict(R=np.eye(3).tolist(), t=[0.0, 0.0, 0.0], yaw_deg=0.0, translation_m=0.0, residual_rms_m=None)


def apply(T, P):
    return np.asarray(P, float) @ np.asarray(T["R"]).T + np.asarray(T["t"])


def object_points(state_or_rec: dict, which: int = 0):
    """Start (which=0) / final (-1) object points of a recording dict: rope nodes, else the cloth's flex vertices."""
    for k in sorted(state_or_rec):
        if k.startswith("obj:rope/") and k.endswith("/nodes"):
            return "rope", np.asarray(state_or_rec[k][which])
    for k in sorted(state_or_rec):
        if k.startswith("obj:cloth/") and k.endswith("/verts"):
            return "cloth", np.asarray(state_or_rec[k][which])
    return None, None


def auto_transform(old_pts, new_pts, kind) -> dict:
    if kind == "rope":
        n = max(len(old_pts), len(new_pts))
        return planar_fit(resample_polyline(old_pts, n), resample_polyline(new_pts, n))
    if len(old_pts) != len(new_pts):
        raise ValueError("cloth point sets differ in size; pass an explicit transform")
    return planar_fit(old_pts, new_pts)


# ------------------------------------------------------------------------------------------------ retargeting
def _bbox_distance(P, lo, hi):
    d = np.maximum(np.maximum(lo - P, 0.0), P - hi)
    return np.linalg.norm(d, axis=-1)


def retarget_rows(rows, layout, fk: AlohaFK, T: dict, *, dt: float, object_points=None, blend=(0.15, 0.35),
                  tol_m: float = 0.005, iters: int = 400) -> tuple:
    """The rows' TCP poses moved by T, re-solved per row by bounded least-squares IK (seeded by the previous row, the
    original row's joints as a weak redundancy prior); grippers unchanged.

    With `object_points` (the OLD object's start points) the move is blended by the TCP's distance to their bounding
    box: fully moved within blend[0] (m), unchanged beyond blend[1], linear between. So the manipulation (approach,
    grasp, transport) moves with the object while idle / home poses far from it stay where they were (a rigid move of
    a parked arm's home pose would only swing its waist). Without it every row moves rigidly."""
    from scipy.spatial.transform import Rotation
    rows = np.asarray(rows, float)
    cols = layout_columns(layout)
    R = np.asarray(T["R"], float); t = np.asarray(T["t"], float)
    if np.allclose(R, np.eye(3), atol=1e-9) and np.linalg.norm(t) < 1e-9:
        return rows.astype(np.float32), dict(transform=T, identity=True, ik_pos_err_max_m=0.0, rows=int(len(rows)),
                                             rows_over_tol=0, rows_over_speed_limit=0)
    rv = Rotation.from_matrix(R).as_rotvec()
    P, Rm = fk.tcp_path(rows, layout)
    if object_points is not None:
        O = np.asarray(object_points, float)
        lo, hi = O.min(axis=0), O.max(axis=0)
        W = np.clip((blend[1] - _bbox_distance(P, lo, hi)) / (blend[1] - blend[0]), 0.0, 1.0)
    else:
        W = np.ones(P.shape[:2])
    out = rows.copy()
    errs = np.zeros((len(rows), 2)); rerr = np.zeros((len(rows), 2))
    for k, arm in enumerate(ARMS):
        prev = None
        for i in range(len(rows)):
            w = float(W[i, k])
            if w <= 0.0:
                prev = None
                continue
            Rw = Rotation.from_rotvec(w * rv).as_matrix()
            target = P[i, k] + w * ((R @ P[i, k] + t) - P[i, k])
            # two seeds: the original row's joints (the move is small) and the previous solution (continuity);
            # keep the better fit, the continuous one on a tie
            # seeds: the original row's joints first (the move is small, so this stays on the original IK branch;
            # seeding by the previous solution let a solve drift onto the flipped forearm-roll / wrist branch), then
            # the previous solution if the first fit is poor; the better fit wins, continuity breaks ties
            orig = rows[i, cols[arm][0]]
            best = None
            for seed in ([orig] if prev is None else [orig, prev]):
                q, e, er = fk.ik(arm, target, Rw @ Rm[i, k], seed=seed, iters=iters, prior=orig)
                jump = 0.0 if prev is None else float(np.abs(q - prev).max())
                score = e + 0.05 * er + 0.02 * jump
                if best is None or score < best[0] - 1e-6:
                    best = (score, q, e, er)
                if e <= 0.5 * tol_m and er <= 0.02:
                    break                       # the original-branch solution fits: keep it
            _, q, e, er = best
            out[i, cols[arm][0]] = q
            errs[i, k], rerr[i, k] = e, er
            prev = q
    dq = np.abs(np.diff(np.stack([out[:, cols[a][0]] for a in ARMS], axis=1), axis=0))
    cap = JOINT_SPEED_LIMIT_RAD_S * dt
    info = dict(transform=T, blend_m=list(blend) if object_points is not None else None,
                rows_moved=int((W > 0).any(axis=1).sum()), rows_fully_moved=int((W >= 1).any(axis=1).sum()),
                ik_pos_err_max_m=float(errs.max()), ik_pos_err_mean_m=float(errs[W > 0].mean()) if (W > 0).any() else 0.0,
                ik_rot_err_max_rad=float(rerr.max()), rows_over_tol=int((errs > tol_m).any(axis=1).sum()),
                rows=int(len(rows)), joint_step_max_rad=float(dq.max()) if len(dq) else 0.0,
                rows_over_speed_limit=int((dq > cap + 1e-9).any(axis=(1, 2)).sum()) if len(dq) else 0,
                note="rows over the speed limit are shortened by the Session's own clamp (clamp, never fail)")
    return out.astype(np.float32), info


# ------------------------------------------------------------------------------------------------ closest approach
def closest_approach(rec_npz, family: str, rings=None) -> dict:
    """How near a run came to the goal: the task's own predicate metrics over the samples plus object geometry."""
    from .objects import projected_crossings, ring_passages, layers
    z = np.load(rec_npz, allow_pickle=False)
    meta = json.loads(Path(str(rec_npz)[:-4] + ".json").read_text())
    samples = meta.get("metrics_samples") or []
    out = dict(success=meta.get("success"), first_success_row=meta.get("first_success_row"),
               max_progress=meta.get("max_progress"))
    if family == "rope_threading":
        exp = meta.get("expected_crossings") or []
        best, best_row = 0, None
        for row, m in samples:
            seq = m.get("crossing_sequence") or []
            k = 0
            while k < min(len(seq), len(exp)) and seq[k] == exp[k]:
                k += 1
            if k > best:
                best, best_row = k, row
        out.update(expected_crossings=exp, best_correct_prefix=best, best_row=best_row,
                   final_crossing_sequence=(samples[-1][1].get("crossing_sequence") if samples else None))
        if rings:
            key = next((k for k in z.files if k.startswith("obj:rope/") and k.endswith("/nodes")), None)
            if key:
                counts = [ring_passages(N, rings)["total"] for N in z[key]]
                out.update(max_ring_passages=int(max(counts)), final_ring_passages=int(counts[-1]))
    elif family == "knot_untying":
        key = next((k for k in z.files if k.startswith("obj:rope/") and k.endswith("/nodes")), None)
        if key:
            others = [k for k in z.files if k.startswith("obj:rope/") and k.endswith("/nodes") and k != key]
            cr = [projected_crossings([z[key][i]] + [z[o][i] for o in others])["count"] for i in range(len(z[key]))]
            out.update(crossings_start=int(cr[0]), crossings_min=int(min(cr)), crossings_final=int(cr[-1]))
        clear = [bool(m.get("planar_clear")) for _, m in samples if "planar_clear" in m]
        out.update(planar_clear_ever=any(clear) if clear else None)
    elif family == "blanket_flatten":
        cov = [(row, m.get("coverage_ratio")) for row, m in samples if m.get("coverage_ratio") is not None]
        if cov:
            r, c = max(cov, key=lambda x: x[1])
            out.update(max_coverage_ratio=float(c), max_coverage_row=int(r), final_coverage_ratio=float(cov[-1][1]))
    return out


def run_summary(rec_json: dict) -> dict:
    keys = ("rows_run", "success", "first_success_row", "episode_end", "final_state_sha", "collisions", "limited_rows",
            "max_progress", "grasp_events", "wall_s", "review", "forcing", "instability")
    return {k: rec_json.get(k) for k in keys}


# ------------------------------------------------------------------------------------------------ session runs
def make_session(args):
    if args.new_instance:
        from rcb_topogym import Session
        inst = json.loads(Path(args.new_instance).read_text())
        return Session(inst["task"], instance=args.new_instance, strict=not args.no_strict)
    mod, fn = args.builder.split(":")
    spec = args.builder_args or "{}"
    if spec.startswith("@"):                    # --builder-args @file.json
        spec = Path(spec[1:]).read_text()
    return getattr(importlib.import_module(mod), fn)(**json.loads(spec))


def cmd_session(a):
    from . import record
    from .aloha import load_model, scene_dir_of_instance
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    z = np.load(a.old_rows, allow_pickle=False)
    rows = z["actions"].astype(np.float32)
    if a.max_rows:
        rows = rows[:a.max_rows]
    layout = [str(x) for x in z["action_layout"]]
    dt = CONTROL_DT_S.get(a.family, 0.04)
    model, _ = load_model(scene_dir_of_instance(a.old_instance))
    fk = AlohaFK(model)
    t0 = time.time()
    s = make_session(a)
    if list(s.action_layout) != layout:
        raise SystemExit(f"layout mismatch: old {layout} vs new {s.action_layout}")
    # the new scene's start state, for the transform
    s.reset()
    new_state = record.Recorder(s).objects()
    report = dict(old_instance=a.old_instance, old_rows=a.old_rows, new_instance=a.new_instance, builder=a.builder,
                  family=a.family, build_s=round(time.time() - t0, 1), variants={})
    T = identity()
    if "retargeted" in a.variants:
        if a.retarget == "auto":
            if not a.old_record:
                raise SystemExit("--retarget auto needs --old-record (the old start state)")
            old = dict(np.load(a.old_record, allow_pickle=False))
            kind, old_pts = object_points(old, 0)
            _, new_pts = object_points({("obj:" + k): np.asarray(v)[None] for k, v in new_state.items()}, 0)
            T = auto_transform(old_pts, new_pts, kind)
        elif a.retarget != "none":
            T = json.loads(Path(a.retarget).read_text())
    rings = None
    if a.family == "rope_threading":
        try:
            from .scene_geom import rings_from_model, task_rings
            w = s.env._world
            rings = task_rings(s) or rings_from_model(w.model, w.data, w.manifest)
        except Exception as exc:  # noqa: BLE001
            report["rings_error"] = repr(exc)
    for v in a.variants.split(","):
        if v == "as_is":
            vr, info = rows, dict(transform=identity())
        elif v == "retargeted":
            opts = None
            if a.old_record:
                _, opts = object_points(dict(np.load(a.old_record, allow_pickle=False)), 0)
            vr, info = retarget_rows(rows, layout, fk, T, dt=dt, object_points=opts)
            np.savez_compressed(out / "retargeted_rows.npz", actions=vr, action_layout=np.array(layout))
        elif v == "nop":
            vr, info = np.repeat(s.hold_action()[None].astype(np.float32), len(rows), axis=0), dict(note="hold start")
        else:
            raise SystemExit(f"unknown variant {v}")
        js = record.run(s, vr, out / f"{v}.npz", every=a.every, hold=a.hold, label=v,
                        extra=dict(variant=v, retarget=info))
        report["variants"][v] = dict(run=run_summary(js), retarget=info,
                                     closest_approach=closest_approach(out / f"{v}.npz", a.family, rings))
        (out / "harness.json").write_text(json.dumps(report, indent=1, default=str))
    real = {k: r for k, r in report["variants"].items() if k != "nop"}
    report["old_oracle_on_new_scene"] = dict(
        method="mujoco_replay", label="MuJoCo replay of the old rows (as-is / retargeted) on the new scene",
        success=any(r["run"]["success"] for r in real.values()),
        variants={k: r["run"]["success"] for k, r in report["variants"].items()},
        closest_approach={k: r["closest_approach"] for k, r in real.items()},
        evidence=str(out / "harness.json"))
    (out / "harness.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report["old_oracle_on_new_scene"]))
    s.close()


# ------------------------------------------------------------------------------------------------ geometric check
def sweep_robot(model, rows, layout, obstacle_bodies: set, robot_bodies: set, every: int = 1) -> dict:
    """Kinematic sweep: the robot at each row's commanded joints in `model` (other bodies at qpos0), contacts between
    robot geoms and obstacle bodies (MuJoCo's own collision detection; no dynamics)."""
    import mujoco
    d = mujoco.MjData(model)
    cols = layout_columns(layout)
    qadr = {arm: [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}/{j}")])
                  for j in ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")] for arm in ARMS}
    hit_rows, deepest, pairs = [], 0.0, {}
    for i in range(0, len(rows), every):
        d.qpos[:] = model.qpos0
        for arm in ARMS:
            d.qpos[qadr[arm]] = rows[i, cols[arm][0]]
        mujoco.mj_fwdPosition(model, d)
        hit = False
        for c in range(int(d.ncon)):
            con = d.contact[c]
            b1, b2 = int(model.geom_bodyid[con.geom1]), int(model.geom_bodyid[con.geom2])
            if (b1 in robot_bodies and b2 in obstacle_bodies) or (b2 in robot_bodies and b1 in obstacle_bodies):
                hit = True
                deepest = max(deepest, -float(con.dist))
                ob = b2 if b2 in obstacle_bodies else b1
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, ob)
                pairs[nm] = pairs.get(nm, 0) + 1
        if hit:
            hit_rows.append(i)
    return dict(rows_checked=len(range(0, len(rows), every)), rows_in_collision=len(hit_rows),
                first_collision_row=hit_rows[0] if hit_rows else None, deepest_m=deepest, obstacles_hit=pairs)


def cmd_geometric(a):
    import mujoco
    from .aloha import load_model, scene_dir_of_instance
    from .objects import cloth_pose_change, projected_crossings, ring_passages
    old = dict(np.load(a.old_record, allow_pickle=False))
    model, manifest = load_model(scene_dir_of_instance(a.new_instance))
    data = mujoco.MjData(model)
    from previews.render import set_birth
    set_birth(model, data, manifest, scene_dir_of_instance(a.new_instance))
    kind, old0 = object_points(old, 0)
    _, old1 = object_points(old, -1)
    # the new start state from the scene (rope chains at qpos0 = birth; cloth slides at birth)
    new0 = None
    for name, rec in manifest["entities"].items():
        if rec.get("kind") == "rope_chain":
            bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, l["body"]) for l in rec["links"]]
            P = np.asarray(data.xpos)[bodies]
            end = P[-1] + np.asarray(data.xmat[bodies[-1]]).reshape(3, 3)[:, 0] * \
                (np.linalg.norm(P[-1] - P[-2]))
            new0 = np.vstack([P, end]); break
        if rec.get("kind") == "pbd_cloth":
            from .scene_geom import cloth_mesh
            cm = cloth_mesh(manifest, scene_dir_of_instance(a.new_instance), name)
            new0 = cm["birth"]; break
    if a.transform == "auto" and new0 is not None:
        T = auto_transform(old0, new0, kind)
    elif a.transform in ("none", "auto"):
        T = identity()
    else:
        T = json.loads(Path(a.transform).read_text())
    moved = apply(T, old1)
    out = dict(method="geometric_argument", transform=T, kind=kind,
               note="the old oracle's final object state moved with the start-pose transform, checked against the new "
                    "scene's goal geometry; an argument, not a replay")
    if kind == "rope":
        if a.family == "rope_threading":
            from .scene_geom import rings_from_model
            rings = rings_from_model(model, data, manifest)
            rp = ring_passages(moved, rings)
            out.update(new_rings=[r["name"] for r in rings], passages=rp,
                       all_rings_passed=all(v["passages"] >= 1 for v in rp["rings"].values()))
        out.update(crossings_final_moved=projected_crossings([moved])["count"])
    elif kind == "cloth" and new0 is not None:
        out.update(final_vs_new_start=cloth_pose_change(new0, moved))
    # the old TCP path (moved) against the new scene's obstacles: robot geoms at the rows' joints
    if a.old_rows:
        z = np.load(a.old_rows, allow_pickle=False)
        rows = z["actions"].astype(np.float64); layout = [str(x) for x in z["action_layout"]]
        fk = AlohaFK(model)
        dt = CONTROL_DT_S.get(a.family, 0.04)
        if abs(T.get("yaw_deg") or 0.0) > 1e-6 or np.linalg.norm(T["t"]) > 1e-6:
            rows, info = retarget_rows(rows, layout, fk, T, dt=dt, object_points=old0)
            out["retarget"] = info
        robot = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, l["body"]) for l in manifest["entities"]["robot"]["links"]}
        obst = set()
        for name, rec in manifest["entities"].items():
            if name in ("robot", "table", "ground") or rec.get("kind") in ("rope_chain", "pbd_cloth") or \
                    name.endswith("_base_riser"):
                continue
            if rec.get("free_joint"):
                continue
            obst |= {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, l["body"]) for l in rec.get("links", [])}
        out["sweep"] = sweep_robot(model, rows, layout, obst, robot, every=a.every)
    Path(a.out).write_text(json.dumps(out, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    print(json.dumps({k: v for k, v in out.items() if k not in ("passages",)}, default=str)[:1500])


# ------------------------------------------------------------------------------------------------ export
def tcp_program(rows, layout, fk, dt, record=None, T=None) -> dict:
    from scipy.spatial.transform import Rotation
    from .trajectory import holding_from_record
    rows = np.asarray(rows, float)
    cols = layout_columns(layout)
    P, R = fk.tcp_path(rows, layout)
    if T is not None:
        P = P @ np.asarray(T["R"]).T + np.asarray(T["t"])
        R = np.einsum("ij,tkjl->tkil", np.asarray(T["R"]), R)
    closed = np.stack([gripper_closedness(rows[:, cols[a][1]], cols[a][2]) for a in ARMS], axis=1)
    hold = None
    if record is not None:
        hold, _ = holding_from_record(record, closed[:len(record.get("targets", rows))])
    out = dict(dt=dt, transform=T, arms={})
    for k, arm in enumerate(ARMS):
        q = Rotation.from_matrix(R[:, k]).as_quat()[:, [3, 0, 1, 2]]
        h = hold[:, k] if hold is not None else (closed[:, k] > 0.5)
        n = min(len(P), len(h))
        out["arms"][arm] = [[round(i * dt, 4), *map(lambda x: round(float(x), 5), P[i, k]),
                             *map(lambda x: round(float(x), 6), q[i]), round(float(closed[i, k]), 3), int(h[i])]
                            for i in range(n)]
    out["columns"] = ["t", "x", "y", "z", "qw", "qx", "qy", "qz", "closedness", "holding"]
    return out


def cmd_export(a):
    from .aloha import load_model, scene_dir_of_instance
    z = np.load(a.old_rows, allow_pickle=False)
    rows = z["actions"].astype(np.float64); layout = [str(x) for x in z["action_layout"]]
    model, _ = load_model(scene_dir_of_instance(a.old_instance))
    fk = AlohaFK(model)
    rec = dict(np.load(a.old_record, allow_pickle=False)) if a.old_record else None
    T = json.loads(Path(a.transform).read_text()) if a.transform else None
    prog = tcp_program(rows, layout, fk, CONTROL_DT_S.get(a.family, 0.04), rec, T)
    if rec is not None and Path(str(a.old_record)[:-4] + ".json").is_file():
        prog["grasp_events_session"] = json.loads(Path(str(a.old_record)[:-4] + ".json").read_text()).get("grasp_events")
    Path(a.out).write_text(json.dumps(prog))
    print(json.dumps(dict(out=a.out, rows=len(rows))))


def main(argv=None):
    ap = argparse.ArgumentParser(description="old-oracle replay harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("session")
    s1.add_argument("--old-instance", required=True); s1.add_argument("--old-rows", required=True)
    s1.add_argument("--old-record", default=None)
    s1.add_argument("--new-instance", default=None); s1.add_argument("--builder", default=None)
    s1.add_argument("--builder-args", default=None); s1.add_argument("--no-strict", action="store_true")
    s1.add_argument("--variants", default="as_is,retargeted"); s1.add_argument("--retarget", default="auto")
    s1.add_argument("--hold", type=int, default=50); s1.add_argument("--every", type=int, default=10)
    s1.add_argument("--family", required=True); s1.add_argument("--out", required=True)
    s1.add_argument("--max-rows", type=int, default=0)
    s2 = sub.add_parser("geometric")
    s2.add_argument("--old-record", required=True); s2.add_argument("--old-rows", default=None)
    s2.add_argument("--new-instance", required=True); s2.add_argument("--family", required=True)
    s2.add_argument("--transform", default="auto"); s2.add_argument("--every", type=int, default=5)
    s2.add_argument("--out", required=True)
    s3 = sub.add_parser("export")
    s3.add_argument("--old-instance", required=True); s3.add_argument("--old-rows", required=True)
    s3.add_argument("--old-record", default=None); s3.add_argument("--transform", default=None)
    s3.add_argument("--family", default="rope_threading"); s3.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"session": cmd_session, "geometric": cmd_geometric, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
