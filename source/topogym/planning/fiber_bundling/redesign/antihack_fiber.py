"""The anti-hack report of a fiber_bundling redesign tier with the redesign tooling's report builder
(antihack.report: schema robotangle-redesign-antihack/1): the strict reference's trajectory metrics (metrics.py on
the tier's -i01 oracle), the new oracle's (metrics.py on its rows), the old oracle replayed on the new scene
(oldreplay.py) and the breakers. A breaker result is an oldreplay.py result, an oracle run's report.json (a breaker
run of oracle.py), or one case of a route_check.py result (name=route_check.json#case). numpy-free.

usage: antihack_fiber.py <tier> <strict_metrics.json> <new_metrics.json> <old_on_new.json> <out_dir>
       [--strict-ref fiber_bundling/<tier>] [--breaker name=result.json[#case] ...] [--source TEXT] [--scene TEXT]
"""
import argparse
import json
from pathlib import Path

from antihack.report import build_report, write_report


def fails(res):
    """A replayed stream fails the tier when the task did not succeed."""
    return not bool(res.get("success"))


def closest(res):
    return {k: res.get(k) for k in ("episode_end", "strands_complete", "strands", "front_sheath_clear", "passage_reached",
                                    "lay", "routes", "rows", "result")}


def breaker_result(path):
    """-> (method, oldreplay-style result) for an oldreplay.py result, an oracle report.json or a route_check case."""
    path, _, case = path.partition("#")
    res = json.loads(Path(path).read_text())
    if case:                                           # route_check.py: the route terms of one sampled centre line
        terms = res["cases"][case]
        terms = terms["terms"] if isinstance(terms, dict) else terms
        return "object_route_terms", dict(success=all(r["routed"] for r in terms), routes=terms)
    if "final_metrics" in res or "result" in res and "pickup" in res:     # an oracle.py run
        fm = res.get("final_metrics") or {}
        return "mujoco_session", dict(success=res.get("result") == "success",
                                      episode_end=(res.get("status") or {}).get("episode_end"), rows=res.get("rows"),
                                      result=res.get("result"), **fm)
    return "mujoco_replay", res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tier"); ap.add_argument("strict"); ap.add_argument("new"); ap.add_argument("old_on_new")
    ap.add_argument("out")
    ap.add_argument("--strict-ref"); ap.add_argument("--breaker", action="append", default=[])
    ap.add_argument("--source", default=""); ap.add_argument("--scene", default="")
    a = ap.parse_args(argv)
    old = json.loads(Path(a.strict).read_text())
    new = json.loads(Path(a.new).read_text())
    oon = json.loads(Path(a.old_on_new).read_text())
    breakers = []
    for spec in a.breaker:
        name, path = spec.split("=", 1)
        method, res = breaker_result(path)
        breakers.append(dict(name=name, method=method, success=not fails(res), detail=closest(res)))
    rep = build_report(family="fiber_bundling", tier=a.tier, strict_ref=a.strict_ref or f"fiber_bundling/{a.tier}",
                       old=dict(trajectory=old["trajectory"]), new_trajectory=new["trajectory"], new_object=None,
                       old_on_new=dict(method="mujoco_replay", success=not fails(oon), closest_approach=closest(oon),
                                       evidence=a.old_on_new),
                       breakers=breakers, scene=a.scene, design=f"docs/simulator/redesign/fiber_bundling.md#{a.tier}",
                       new_source=a.source, metrics_file=a.strict)
    print(write_report(rep, a.out))


if __name__ == "__main__":
    main()
