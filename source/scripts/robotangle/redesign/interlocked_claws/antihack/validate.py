"""Validate every family's antihack_report.json against the shared definitions (json only; runs on the laptop).

  python3 -m antihack.validate <session root D> [--out-dir D/review/tooling]
  -> antihack_validation.{md,json}: per family x tier: report present, schema, the strict reference used vs the registry,
     old-oracle numbers vs the official metrics file (stale / different reference), the verdict recomputed from the
     report's own numbers vs the stored one, the evidence level (MuJoCo replay vs argument), breakers run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .report import SCHEMA, compare, verdict

FAMILIES = ("rope_threading", "knot_untying", "blanket_flatten")
TIERS = ("very_easy", "easier", "easy", "medium", "hard")
CHECK_KEYS = ("phases_total", "motion_phases", "waypoints_total", "tcp_path_m_total", "grasp_count", "handover_count",
              "regrasp_count", "duration_s")


def expected_ref(registry: dict, family: str, tier: str) -> list:
    """The strict reference(s) the registry names for a tier: its own when a MuJoCo oracle succeeded, else the named
    closest successful tier(s)."""
    rec = ((registry.get("families") or {}).get(family) or {}).get(tier) or {}
    if rec.get("status") == "strict_reference_mujoco":
        return [f"{family}/{tier}"]
    return [c["ref"] for c in rec.get("closest", []) if c.get("ref")]


def check(D: Path, family: str, tier: str, registry: dict) -> dict:
    p = D / "evidence" / family / tier / "antihack_report.json"
    out = dict(family=family, tier=tier, report=str(p.relative_to(D)), present=p.is_file(), issues=[], notes=[])
    if not p.is_file():
        out["issues"].append("no antihack_report.json")
        return out
    try:
        rep = json.loads(p.read_text())
    except ValueError as exc:
        out["issues"].append(f"unreadable: {exc}")
        return out
    if rep.get("schema") != SCHEMA:
        out["issues"].append(f"schema {rep.get('schema')!r} != {SCHEMA}")
    for k in ("strict_reference", "old_oracle", "new_task", "comparison", "old_oracle_on_new_scene", "verdict"):
        if k not in rep:
            out["issues"].append(f"missing {k}")
    ref = (rep.get("strict_reference") or {}).get("ref")
    for f_ in FAMILIES:                      # "metrics_<family>_<tier>.json" names the same reference as "<family>/<tier>"
        for t_ in TIERS:
            if ref == f"metrics_{f_}_{t_}.json":
                ref = f"{f_}/{t_}"
    out["strict_ref"] = ref
    want = expected_ref(registry, family, tier)
    out["registry_ref"] = want
    mf = None
    if ref and "/" in ref:
        rf, rt = ref.split("/", 1)
        mf = D / "evidence" / "strict_refs" / f"metrics_{rf}_{rt}.json"
        if want and ref not in want:
            out["issues"].append(f"strict reference {ref} differs from the registry's {want}")
    elif ref and ref.startswith(f"metrics_{family}_{tier}__"):
        # a closest reference of this tier (registry: closest list) named by its metrics file
        mf = D / "evidence" / "strict_refs" / ref
        out["notes"].append(f"compared against a closest reference ({ref}); the registry's strict reference is {want}")
    elif ref:
        mf = D / "evidence" / "strict_refs" / ref if ref.endswith(".json") else None
        if want:
            out["issues"].append(f"strict reference {ref} differs from the registry's {want}")
    # old numbers vs the official metrics file
    if mf is not None:
        if mf.is_file():
            off = json.loads(mf.read_text())["trajectory"]
            got = (rep.get("old_oracle") or {}).get("trajectory") or {}
            diff = {k: (got.get(k), off.get(k)) for k in CHECK_KEYS
                    if got.get(k) is not None and off.get(k) is not None and abs(float(got[k]) - float(off[k])) > 1e-6}
            if diff:
                out["issues"].append(f"old-oracle numbers differ from {mf.name}: {diff}")
        else:
            out["notes"].append(f"no official metrics file {mf.name}")
    # verdict recomputed from the report's own numbers
    old_t = (rep.get("old_oracle") or {}).get("trajectory") or {}
    new_t = (rep.get("new_task") or {}).get("trajectory") or {}
    comp = compare(old_t, new_t, (rep.get("old_oracle") or {}).get("object"), (rep.get("new_task") or {}).get("object"))
    v = verdict(comp, rep.get("old_oracle_on_new_scene"), rep.get("breakers"))
    sv = rep.get("verdict") or {}
    for k in ("needs_more", "pass"):
        if sv.get(k) is not None and sv.get(k) != v.get(k):
            out["issues"].append(f"stored verdict {k}={sv.get(k)} but recomputed {v.get(k)}")
    if sv.get("pass_strict") and not v.get("pass_strict"):
        out["issues"].append("claims pass_strict without a breaker run where every breaker fails (review R5: "
                             "pass_strict needs >= 1 breaker run, all failing)")
    out.update(needs_more=v["needs_more"], old_oracle_fails=v["old_oracle_fails"], evidence_level=v["evidence_level"],
               old_oracle_method=v["old_oracle_method"], breakers_run=len([b for b in rep.get("breakers") or []
                                                                         if b.get("success") is not None]),
               breakers_fail=v["breakers_fail"], recomputed_pass=v["pass"], recomputed_pass_strict=v["pass_strict"],
               clearly_more_in=v["clearly_more_in"],
               key_numbers={k: (old_t.get(k), new_t.get(k)) for k in ("phases_total", "waypoints_total",
                                                                       "tcp_path_m_total", "grasp_count", "handover_count")})
    if v["evidence_level"] == "argument":
        out["notes"].append("old oracle judged by an argument, not a MuJoCo replay")
    elif v["evidence_level"] == "object_replay":
        out["notes"].append("old oracle replayed at object level (virtual hands), not on the robot")
    if not out["breakers_run"]:
        out["notes"].append("no breaker results in the report")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    D = Path(a.root)
    registry = json.loads((D / "evidence/strict_refs/registry.json").read_text())
    rows = [check(D, f, t, registry) for f in FAMILIES for t in TIERS]
    out_dir = Path(a.out_dir or D / "review" / "tooling")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "antihack_validation.json").write_text(json.dumps(rows, indent=1, default=str))
    L = ["# Anti-hack report validation (tooling lane, `python3 -m antihack.validate`)", "",
         "Recomputed from each report's own numbers with the shared definitions; old-oracle numbers checked against",
         "`evidence/strict_refs/metrics_<ref>.json`; strict reference checked against the registry.", "",
         "| family / tier | ref | needs more | old oracle fails (level) | breakers run / fail | pass / strict | issues | notes |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r["present"]:
            L.append(f"| {r['family']} / {r['tier']} | | | | | | no report | |")
            continue
        L.append(f"| {r['family']} / {r['tier']} | {r.get('strict_ref')} | {r.get('needs_more')} | "
                 f"{r.get('old_oracle_fails')} ({r.get('evidence_level')}) | {r.get('breakers_run')} / {r.get('breakers_fail')} | "
                 f"{r.get('recomputed_pass')} / {r.get('recomputed_pass_strict')} | {'; '.join(r['issues']) or '-'} | "
                 f"{'; '.join(r['notes']) or '-'} |")
    (out_dir / "antihack_validation.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
