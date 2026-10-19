"""antihack_report.json: one schema for all families (old oracle vs new task, verdict). numpy-free.

    from antihack.report import build_report, write_report
    rep = build_report(family="rope_threading", tier="medium", strict_ref="rope_threading/hard",
                       old=json.load(open("evidence/strict_refs/metrics_rope_threading_hard.json")),
                       new_trajectory=<antihack.trajectory.metrics_from_plan / metrics_from_rows result>,
                       new_object=<antihack.objects.rope_pose_change / cloth_pose_change result>,
                       old_on_new=dict(method="mujoco_replay" | "retargeted_replay" | "geometric_argument",
                                       success=False, closest_approach={...}, evidence="the cloud host path / file"),
                       breakers=[dict(name="straight_line", method="kinematic", success=False, detail={...})],
                       design="DESIGN_rope_threading.md#medium", notes="...")
    write_report(rep, "evidence/rope_threading/medium")       # antihack_report.json + antihack.md

Schema (robotangle-redesign-antihack/1):
  family, tier, scene, generated, strict_reference {ref, metrics_file},
  old_oracle {trajectory (summary), object (pose change)}, new_task {trajectory, object, source},
  comparison {metric: {old, new, delta, ratio, new_ge_old, clearly_more}},
  old_oracle_on_new_scene {method, success, closest_approach, evidence, label: "MuJoCo replay" | "argument"},
  breakers [{name, method, success, detail}],
  verdict {needs_more, required_ge: {metric: bool}, clearly_more_in: [...], old_oracle_fails, breakers_fail,
           pass, pass_strict, reasons: [...]}
Verdict rule (COMMON.md): the new task needs more = at least as many phases, waypoints and as much TCP path as the old
oracle, and clearly more in at least one metric (a count +1: grasps, hand-overs, regrasps, phases; or a length x1.25:
TCP / joint path, duration, object displacement / rotation); and the old oracle fails on the new scene. pass_strict also
needs at least one breaker run and every breaker to fail. `evidence_level`: "replay" (robot-level MuJoCo replay of the old
rows on the new scene), "object_replay" (object-level MuJoCo replay with virtual hands) or "argument" (geometric /
kinematic argument, labelled).
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

SCHEMA = "robotangle-redesign-antihack/1"

# (name, path in the trajectory / object block, kind)
TRAJ = [("motion_phases", "motion_phases", "count"), ("phases_total", "phases_total", "count"),
        ("waypoints_total", "waypoints_total", "count"), ("tcp_path_m_total", "tcp_path_m_total", "length"),
        ("joint_path_rad_total", "joint_path_rad_total", "length"), ("duration_s", "duration_s", "length"),
        ("grasp_count", "grasp_count", "count"), ("regrasp_count", "regrasp_count", "count"),
        ("handover_count", "handover_count", "count"), ("arms_used", "arms_used", "count"),
        ("bimanual_overlap_s", "bimanual_overlap_s", "length")]
ROPE_OBJ = [("centroid_translation_m", "centroid_translation_m", "length"),
            ("chord_rotation_deg", "chord_rotation_deg", "length"),
            ("link_rotation_mean_deg", "link_rotation_mean_deg", "length"),
            ("link_axis_rotation_mean_deg", "link_axis_rotation_mean_deg", "length"),
            ("link_displacement_max_m", "link_displacement_max_m", "length"),
            ("link_displacement_mean_m", "link_displacement_mean_m", "length"),
            ("crossings_start", "crossings_start", "count"), ("ring_passages_goal", "ring_passages_goal", "count")]
CLOTH_OBJ = [("procrustes_translation_m", "procrustes_translation_m", "length"),
             ("procrustes_rotation_deg", "procrustes_rotation_deg", "length"),
             ("displacement_mean_m", "displacement_mean_m", "length"),
             ("displacement_max_m", "displacement_max_m", "length"),
             ("layered_area_fraction_start", "layered_area_fraction_start", "length"),
             ("fold_layers_start", "fold_layers_start", "count"),
             ("flipped_area_fraction_start", "flipped_area_fraction_start", "length")]
REQUIRED_GE = ("phases_total", "waypoints_total", "tcp_path_m_total")
CLEAR_COUNT = 1          # a count metric is clearly more at +1 or more
CLEAR_RATIO = 1.25       # a length metric is clearly more at x1.25 or more


def _cmp(old, new, kind):
    if old is None or new is None:
        return dict(old=old, new=new, delta=None, ratio=None, new_ge_old=None, clearly_more=None)
    delta = new - old
    ratio = (new / old) if old else (None if new == 0 else float("inf"))
    if kind == "count":
        clear = delta >= CLEAR_COUNT
    else:
        clear = (old > 0 and new >= CLEAR_RATIO * old) or (old == 0 and new > 0)
    return dict(old=old, new=new, delta=round(delta, 6) if isinstance(delta, float) else delta,
                ratio=None if ratio is None else (round(ratio, 4) if ratio != float("inf") else "inf"),
                new_ge_old=bool(new >= old), clearly_more=bool(clear))


def compare(old_traj: dict, new_traj: dict, old_obj: dict | None, new_obj: dict | None) -> dict:
    out = {}
    for name, key, kind in TRAJ:
        out[name] = _cmp(old_traj.get(key), new_traj.get(key), kind)
    objs = ROPE_OBJ if (old_obj or new_obj or {}).get("links") is not None or "chord_rotation_deg" in (new_obj or {}) \
        else CLOTH_OBJ
    for name, key, kind in objs:
        out["object." + name] = _cmp((old_obj or {}).get(key), (new_obj or {}).get(key), kind)
    return out


def verdict(comparison: dict, old_on_new: dict | None, breakers: list | None) -> dict:
    req = {k: comparison.get(k, {}).get("new_ge_old") for k in REQUIRED_GE}
    clear = [k for k, v in comparison.items() if v.get("clearly_more")]
    needs_more = all(v is True for v in req.values()) and bool(clear)
    fails = None if not old_on_new or old_on_new.get("success") is None else (not bool(old_on_new["success"]))
    br = [b for b in (breakers or []) if b.get("success") is not None]
    breakers_fail = None if not br else all(not b["success"] for b in br)
    reasons = []
    for k, v in req.items():
        if v is not True:
            reasons.append(f"{k}: new {comparison.get(k, {}).get('new')} < old {comparison.get(k, {}).get('old')}"
                           if v is False else f"{k}: not available")
    if not clear:
        reasons.append("not clearly more in any metric (count +1 or length x1.25)")
    if fails is None:
        reasons.append("old oracle on the new scene: not tested")
    elif not fails:
        reasons.append("the old oracle SOLVES the new scene")
    for b in br:
        if b["success"]:
            reasons.append(f"breaker {b.get('name')} ({b.get('method')}) solves the new scene")
    method = (old_on_new or {}).get("method")
    ml = str(method or "").lower()
    level = ("replay" if method in ("mujoco_replay", "retargeted_replay", "session_replay") else
             "object_replay" if any(w in ml for w in ("object-level", "object level", "object_level", "virtual hand",
                                                       "lab")) else
             "argument" if method else None)
    if not br:
        reasons.append("no breaker run yet")
    out = dict(needs_more=needs_more, required_ge=req, clearly_more_in=clear, old_oracle_fails=fails,
               old_oracle_method=method, evidence_level=level, breakers_fail=breakers_fail, reasons=reasons)
    out["pass"] = bool(needs_more and fails is True)
    # strict: also at least one breaker run and every breaker fails
    out["pass_strict"] = bool(needs_more and fails is True and breakers_fail is True)
    return out


def _summ(t: dict | None) -> dict:
    if not t:
        return {}
    from .trajectory import SUMMARY_KEYS
    out = {k: t.get(k) for k in SUMMARY_KEYS if k in t}
    if "per_arm" in t:
        out["per_arm"] = {a: {k: v for k, v in r.items() if k != "waypoint_rows"} for a, r in t["per_arm"].items()}
    return out


def build_report(*, family: str, tier: str, strict_ref: str, old: dict, new_trajectory: dict, new_object: dict | None,
                 old_on_new: dict | None = None, breakers: list | None = None, scene: str = "", design: str = "",
                 new_source: str = "", notes: str = "", metrics_file: str = "") -> dict:
    old_traj = old.get("trajectory") or {}
    old_obj = old.get("object")
    comp = compare(old_traj, new_trajectory, old_obj, new_object)
    v = verdict(comp, old_on_new, breakers)
    return dict(schema=SCHEMA, family=family, tier=tier, scene=scene, design=design,
                generated=_dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                strict_reference=dict(ref=strict_ref, metrics_file=metrics_file,
                                      status=(old.get("strict_reference") or {}).get("status")),
                old_oracle=dict(trajectory=_summ(old_traj), object=old_obj),
                new_task=dict(trajectory=_summ(new_trajectory), object=new_object, source=new_source),
                comparison=comp, old_oracle_on_new_scene=old_on_new or dict(method=None, success=None),
                breakers=breakers or [], verdict=v, notes=notes)


def _fmt(x):
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def write_report(rep: dict, out_dir) -> tuple:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pj = out / "antihack_report.json"
    pj.write_text(json.dumps(rep, indent=1, default=str))
    v = rep["verdict"]
    L = [f"# Anti-hack check: {rep['family']} / {rep['tier']}", "",
         f"- strict reference: `{rep['strict_reference']['ref']}` ({rep['strict_reference'].get('status')})",
         f"- new task source: {rep['new_task'].get('source', '')}; scene: {rep.get('scene', '')}",
         f"- **verdict: {'PASS' if v['pass'] else 'NOT YET'}** (needs more: {v['needs_more']}; old oracle fails: "
         f"{v['old_oracle_fails']}; breakers fail: {v['breakers_fail']}; strict pass: {v['pass_strict']})",
         f"- clearly more in: {', '.join(v['clearly_more_in']) or '-'}",
         *(f"- open: {r}" for r in v["reasons"]), "",
         "| metric | old oracle | new task | new >= old | clearly more |", "|---|---|---|---|---|"]
    for k, c in rep["comparison"].items():
        L.append(f"| {k} | {_fmt(c['old'])} | {_fmt(c['new'])} | {c['new_ge_old']} | {c['clearly_more']} |")
    o = rep["old_oracle_on_new_scene"]
    L += ["", f"Old oracle on the new scene: method `{o.get('method')}`, success {o.get('success')}; "
              f"closest approach: `{json.dumps(o.get('closest_approach'))[:300]}`; evidence: {o.get('evidence', '')}", ""]
    if rep["breakers"]:
        L += ["| breaker | method | success | detail |", "|---|---|---|---|"]
        for b in rep["breakers"]:
            L.append(f"| {b.get('name')} | {b.get('method')} | {b.get('success')} | "
                     f"{json.dumps(b.get('detail'))[:160]} |")
    if rep.get("notes"):
        L += ["", rep["notes"]]
    pm = out / "antihack.md"
    pm.write_text("\n".join(L) + "\n")
    return str(pj), str(pm)
