"""Plan one v1 tier (rope agent): the lay-and-pick relay (relay_plan.RelayPlanner), realised by oracle.realize (IK,
capsule checks, vias / RRT), saved as <out>/timeline_<tier>.json (plan format + spans with steps + the check report).

usage: python plan_tier.py <tier> <out_dir>
"""
import json
import sys
import time
from pathlib import Path

from topogym.planning.rope_threading.redesign import scenes as S
from topogym.planning.rope_threading.redesign import oracle as O
from topogym.planning.rope_threading.redesign import relay_plan as RP

tier, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
d = S.build(tier)
phases = RP.RelayPlanner(d).plan()
print(f"{tier}: {len(phases)} phases planned in {time.time() - t0:.1f} s", flush=True)
plan, extra = O.timeline(d, phases)
print(f"{tier}: realised in {time.time() - t0:.1f} s; failures {len(extra['failures'])}", flush=True)
for f in extra["failures"][:30]:
    print("   INFEASIBLE", json.dumps(f, default=str), flush=True)
rep = dict(tier=tier, phases=len(phases), waypoints=len(extra["steps"]),
           vias=sum(1 for s_ in extra["steps"] if s_["kind"] == "via"), duration_s=extra["duration_s"],
           joint_path_rad=extra["joint_path_rad"], failures=extra["failures"], feasible=not extra["failures"],
           wall_s=round(time.time() - t0, 1))
(out / f"timeline_{tier}.json").write_text(json.dumps(dict(tier=tier, plan=plan, joints=extra["joints"], spans=extra["spans"],
                                                           failures=extra["failures"], duration_s=extra["duration_s"],
                                                           joint_path_rad=extra["joint_path_rad"], check=rep),
                                                      default=float))
print(json.dumps({k: v for k, v in rep.items() if k != "failures"}, default=float), flush=True)
