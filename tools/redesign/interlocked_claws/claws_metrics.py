"""Anti-hack metrics of a claws recording (antihack.record npz): trajectory complexity (antihack.trajectory, FK of the
commanded rows on the scene's own ALOHA) and the rigid pieces' pose change, absolute and relative.

Relative motion = the brown piece (piece_0) in the frame of the other piece (piece_1): what the separation itself
needed, independent of how the two arms shared it.
  - relative translation / rotation start -> end; the relative path (sum of sampled steps) in m and deg;
  - rotation axes used: the sampled relative rotation increments (in the teal piece's frame) clustered by direction
    (axes within 30 deg, either sign, are one axis); an axis counts when its cumulative rotation is >= 15 deg;
  - motion moves: maximal runs of samples in which the relative pose changes (> 1 mm or > 1 deg per sample), split where
    the dominant kind (translation / rotation) or the rotation axis cluster changes.

usage: python claws_metrics.py <rec.npz> --variation V [--tier T] [--out metrics.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

PIECES = ("piece_0", "piece_1")      # brown target, teal other
CONTROL_DT_S = 0.025


def _pose(arr):
    """(N, 7) [x y z qw qx qy qz] -> positions (N, 3), Rotation."""
    a = np.asarray(arr, float)
    return a[:, :3], Rotation.from_quat(a[:, [4, 5, 6, 3]])


def piece_names(rec):
    """(target, other): piece_0 / piece_1, or the released hard's claw_0 / claw_1 (sorted names: target first)."""
    names = sorted(k.split("/")[1] for k in rec if str(k).startswith("obj:rigid/") and str(k).endswith("/pose"))
    return (names[0], names[1]) if len(names) == 2 else PIECES


def relative_series(rec: dict):
    tgt, oth = piece_names(rec)
    pb, rb = _pose(rec[f"obj:rigid/{tgt}/pose"])
    pt, rt = _pose(rec[f"obj:rigid/{oth}/pose"])
    rel_p = rt.inv().apply(pb - pt)
    rel_r = rt.inv() * rb
    return rel_p, rel_r


def axis_clusters(rel_r: Rotation, min_step_deg=0.5, cluster_deg=30.0, count_deg=15.0):
    """Cluster the per-sample relative rotation increments (expressed in the teal frame) by axis direction."""
    inc = (rel_r[:-1].inv() * rel_r[1:])          # increment in the moving (brown) frame
    # express the increment axis in the teal frame: R_rel_k * axis
    rv = inc.as_rotvec()
    ang = np.linalg.norm(rv, axis=1)
    axes_body = rv / np.maximum(ang[:, None], 1e-12)
    axes = rel_r[:-1].apply(axes_body)
    clusters = []      # [axis, cumulative deg]
    labels = np.full(len(ang), -1)
    cosc = np.cos(np.deg2rad(cluster_deg))
    for i in np.flatnonzero(np.rad2deg(ang) >= min_step_deg):
        a = axes[i]
        for k, c in enumerate(clusters):
            if abs(float(a @ c[0])) >= cosc:
                c[1] += float(np.rad2deg(ang[i]))
                labels[i] = k
                break
        else:
            clusters.append([a.copy(), float(np.rad2deg(ang[i]))])
            labels[i] = len(clusters) - 1
    used = [dict(axis_in_teal_frame=np.round(c[0], 3).tolist(), cumulative_deg=round(c[1], 1))
            for c in clusters if c[1] >= count_deg]
    return used, labels, ang


def moves(rel_p, rel_r, labels, ang, step_m=0.001, step_deg=1.0):
    d = np.linalg.norm(np.diff(rel_p, axis=0), axis=1)
    deg = np.rad2deg(ang)
    out, cur = [], None
    for i in range(len(d)):
        moving = d[i] > step_m or deg[i] > step_deg
        if not moving:
            cur = None
            continue
        kind = "rotation" if deg[i] > step_deg and deg[i] * 0.001 >= d[i] * 0.5 else "translation"
        key = (kind, int(labels[i]) if kind == "rotation" else -1)
        if cur is None or cur["key"] != key:
            cur = dict(key=key, kind=kind, start=i, end=i + 1, path_m=0.0, rot_deg=0.0)
            out.append(cur)
        cur["end"] = i + 1
        cur["path_m"] += float(d[i])
        cur["rot_deg"] += float(deg[i])
    keep = [m for m in out if m["path_m"] >= 0.005 or m["rot_deg"] >= 5.0]
    return [dict(kind=m["kind"], axis_cluster=m["key"][1], sample_start=m["start"], sample_end=m["end"],
                 path_m=round(m["path_m"], 4), rot_deg=round(m["rot_deg"], 1)) for m in keep]


def object_metrics(rec: dict) -> dict:
    out = {}
    for name in piece_names(rec):
        p, r = _pose(rec[f"obj:rigid/{name}/pose"])
        steps = np.linalg.norm(np.diff(p, axis=0), axis=1)
        rsteps = (r[:-1].inv() * r[1:]).magnitude()
        out[name] = dict(translation_m=round(float(np.linalg.norm(p[-1] - p[0])), 4),
                         rotation_deg=round(float(np.rad2deg((r[0].inv() * r[-1]).magnitude())), 1),
                         path_m=round(float(steps.sum()), 4), rotation_path_deg=round(float(np.rad2deg(rsteps.sum())), 1),
                         start=np.round(p[0], 4).tolist(), end=np.round(p[-1], 4).tolist())
    rel_p, rel_r = relative_series(rec)
    used, labels, ang = axis_clusters(rel_r)
    mv = moves(rel_p, rel_r, labels, ang)
    out["relative"] = dict(
        frame="brown piece_0 in the teal piece_1 frame",
        translation_m=round(float(np.linalg.norm(rel_p[-1] - rel_p[0])), 4),
        rotation_deg=round(float(np.rad2deg((rel_r[0].inv() * rel_r[-1]).magnitude())), 1),
        path_m=round(float(np.linalg.norm(np.diff(rel_p, axis=0), axis=1).sum()), 4),
        rotation_path_deg=round(float(np.rad2deg(ang.sum())), 1),
        rotation_axes=used, rotation_axes_count=len(used),
        moves=mv, move_count=len(mv),
        rotation_moves=sum(1 for m in mv if m["kind"] == "rotation"),
        translation_moves=sum(1 for m in mv if m["kind"] == "translation"))
    return out


def load_record(path):
    z = np.load(path, allow_pickle=False)
    rec = {k: z[k] for k in z.files}
    js = Path(str(path)[:-4] + ".json")
    meta = json.loads(js.read_text()) if js.is_file() else {}
    rec["first_success_row"] = meta.get("first_success_row")
    rec["_meta"] = meta
    return rec


def fk_for_variation(variation, seed=0):
    """An AlohaFK on the claws scene's own model (built through a Session; the robot is the same in every tier)."""
    from antihack.aloha import AlohaFK
    from rcb_topogym import Session
    from topogym.tasks.interlocked_claws.redesign.task import VARIATIONS, register
    for tier, vid in VARIATIONS.items():          # the redesign tiers are registered in-process
        if vid == variation:
            register(tier)
    s = Session("interlocked_claws", variation_id=variation, seed=seed, gripper_space="native", strict=False)
    fk = AlohaFK(s.env._world.model)
    return fk, s


def metrics(rec_path, variation, fk=None):
    from antihack.trajectory import metrics_from_rows, summary
    rec = load_record(rec_path)
    rows = np.asarray(rec["targets"], float)
    layout = [str(x) for x in rec["action_layout"]]
    if fk is None:
        fk, s = fk_for_variation(variation)
    traj = metrics_from_rows(rows, layout, CONTROL_DT_S, fk, record=rec)
    return dict(trajectory=summary(traj), object=object_metrics(rec),
                trajectory_detail=dict(grasp_events=traj["grasp_events"], phases=traj["phases"],
                                       waypoint_rows={k: v["waypoint_rows"] for k, v in traj["per_arm"].items()}),
                recording=dict(success=rec["_meta"].get("success"), first_success_row=rec["first_success_row"],
                               final_state_sha=rec["_meta"].get("final_state_sha"), rows=int(len(rows))))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("--variation", required=True)
    ap.add_argument("--tier", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    m = metrics(a.rec, a.variation)
    m.update(tier=a.tier, variation=a.variation, recording_file=str(a.rec))
    txt = json.dumps(m, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    if a.out:
        Path(a.out).write_text(txt)
    s = m["trajectory"]
    o = m["object"]["relative"]
    print(json.dumps(dict(rows=s["rows"], duration_s=s["duration_s"], phases_total=s["phases_total"],
                          motion_phases=s["motion_phases"], waypoints=s["waypoints_total"], tcp_path_m=s["tcp_path_m_total"],
                          joint_path_rad=s["joint_path_rad_total"], grasps=s["grasp_count"], regrasps=s["regrasp_count"],
                          handovers=s["handover_count"], rel_translation_m=o["translation_m"], rel_rotation_deg=o["rotation_deg"],
                          rel_path_m=o["path_m"], rel_rotation_path_deg=o["rotation_path_deg"],
                          rel_axes=o["rotation_axes_count"], rel_moves=o["move_count"])))


if __name__ == "__main__":
    main()
