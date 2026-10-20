"""Run a redesign tier's oracle live in a fresh MuJoCo Session (the cloud host; through launch.sh): the tier's segment plan
(grasp per segment, park / drop), feedback-planned from the measured state; then hold rows until the episode ends.
The rows are saved only when the episode succeeds. usage: run_oracle.py <tier> --out <dir> [--instance <instance.json>]"""
import faulthandler
faulthandler.enable()
import argparse, json, sys, time, traceback
from pathlib import Path
import numpy as np
import rdreg
import oracle_plans

ap = argparse.ArgumentParser()
ap.add_argument("tier"); ap.add_argument("--out", required=True); ap.add_argument("--instance", default=None)
ap.add_argument("--plan", default=None, help="JSON segment plan overriding the tier's default")
ap.add_argument("--tail", type=int, default=200)
ap.add_argument("--replay", default=None, help="actions .npy to replay first (debugging a later stage)")
ap.add_argument("--replay-rows", type=int, default=None)
ap.add_argument("--from-stage", default=None, choices=[None, "drop", "final", "segment", "parked"])
ap.add_argument("--rest", default=None, help="i,k: with --from-stage parked, the replay ends right after the wand was "
                "parked on rest i (a 'parked' event); the rest graph resumes there with held segment k")
ap.add_argument("--segment", default=None, help="k,i,j,grasp: with --from-stage segment, the held segment to run "
                "after the replay (the wand held at rest i after its regrasp), then the drop and the final place")
ap.add_argument("--drop-offset", type=float, default=None,
                help="release heading offset from the finish heading (deg); default: the plan's drop_offset_deg, else 0")
ap.add_argument("--drop-open-s", type=float, default=0.6, help="finger opening time at the release (s)")
ap.add_argument("--drop-retreat", type=float, default=0.06, help="hand back-off after the release (m)")
ap.add_argument("--noise-branch", action="store_true", help="noise: prefer another arm branch at every grasp")
ap.add_argument("--noise-branch-n", type=int, default=99, help="noise: apply the other branch at the first N grasps only")
ap.add_argument("--noise-branch-at", default=None, help="noise: 1-based grasp numbers that take another branch, e.g. 1,4")
ap.add_argument("--drop-psi-choice", type=int, default=None,
                help="drop heading search: release at the k-th reachable heading (by distance from the commanded one)")
ap.add_argument("--drop-backs", default=None, help="comma list: release this far (m of wire) above the post's foot, "
                "first that plans (default 0,0.04,0.08)")
ap.add_argument("--inspect-deg", type=float, default=0.0, help="noise: inspection turn about the tangent per segment")
ap.add_argument("--mount", default=None, help="x,y,yaw_deg: a candidate robot mount (sets BUZZ_MOUNT_OVERRIDE)")
ap.add_argument("--springref", type=float, default=None, help="Robotiq springref (default: env BUZZ_SPRINGREF)")
ap.add_argument("--rng", type=int, default=0, help="the oracle's IK seed set (another arm-branch variant)")
a = ap.parse_args()
import os
if a.mount:
    os.environ["BUZZ_MOUNT_OVERRIDE"] = f"{a.tier}:{a.mount}"
if a.springref is not None:
    os.environ["BUZZ_SPRINGREF"] = str(a.springref)
vids = rdreg.register()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
from rcb_topogym import Session
from topogym.tasks.buzz_wire_3d.redesign import tiers
from topogym.planning.buzz_wire_3d.redesign.io import SessionIO
from topogym.planning.buzz_wire_3d.redesign.oracle import TierOracle
logf = open(out / "log.jsonl", "w")


def log(**kv):
    logf.write(json.dumps(kv, default=str) + "\n"); logf.flush()
    print(json.dumps(kv, default=str), flush=True)


t0 = time.time()
if a.instance:
    s = Session("buzz_wire_3d", instance=a.instance, strict=True)
else:
    s = Session("buzz_wire_3d", variation_id=vids[a.tier], seed=0, gripper_space="native", strict=False)
design = tiers.design(a.tier)
plan = json.loads(Path(a.plan).read_text()) if a.plan else oracle_plans.ORACLE_PLANS[a.tier]
if "graph_plan" in plan:
    plan = plan["graph_plan"]
io = SessionIO(s)
s.record_start()
orc = TierOracle(io, design, plan.get("segments", []), pick_grasp=plan.get("pick", "top:180:0.8:0:-2"),
                 place_grasp=plan.get("place", "top:180:0.8:0:-2"), log=log, rng=a.rng)
if a.drop_offset is None:
    a.drop_offset = float(plan.get("drop_offset_deg", 0.0)) if isinstance(plan, dict) else 0.0
orc.drop_offset_deg = a.drop_offset
orc.drop_open_s, orc.drop_retreat = a.drop_open_s, a.drop_retreat
orc.drop_psi_choice = a.drop_psi_choice
orc.drop_backs = None if not a.drop_backs else tuple(float(x) for x in a.drop_backs.split(","))
orc.noise_branch, orc.inspect_deg, orc.noise_branch_n = a.noise_branch, a.inspect_deg, a.noise_branch_n
orc.noise_branch_at = None if not a.noise_branch_at else {int(x) for x in a.noise_branch_at.split(",")}
failure = None
try:
    if a.replay:
        rows = np.load(a.replay)[:a.replay_rows]
        for row in rows:
            io.step(row[:7], row[7])
        orc._last_q = np.asarray(rows[-1][:7], float)
        orc.grip = float(rows[-1][7])
        log(event="replayed", rows=len(rows))
        if a.from_stage == "parked":
            i, k = (int(x) for x in a.rest.split(","))
            orc.run_graph(plan["rests"], plan["graph"], start_rest=i, start_k=k)
        elif a.from_stage == "segment":
            k, i, j, grasp = a.segment.split(",")
            rests = plan["rests"]
            last = len(rests) - 1
            seg = dict(s0=rests[int(i)], s1=rests[int(j)], grasp=grasp, end="drop" if int(j) == last else "park")
            orc.measure_grasp()
            orc.traverse(int(k), seg)
            if seg["end"] == "park":
                raise RuntimeError("--from-stage segment runs the drop segment only")
            orc.drop()
        elif a.from_stage == "drop":
            orc.drop()
        if a.from_stage != "parked":
            orc.final_place()
            orc.hold()
    elif "graph" in plan:
        orc.run_graph(plan["rests"], plan["graph"])
    else:
        orc.run()
except Exception as exc:
    failure = f"{type(exc).__name__}: {exc}"
    traceback.print_exc()
held = 0
while s.episode_end is None and held < a.tail:
    s.step(s.hold_action()); held += 1
acts = s.record_stop()
st = s.status()
res = dict(tier=a.tier, mount=design.robot_mount, rng=a.rng, drop_offset=a.drop_offset, drop_open_s=a.drop_open_s, drop_retreat=a.drop_retreat, noise_branch=a.noise_branch, noise_branch_n=a.noise_branch_n, noise_branch_at=a.noise_branch_at, inspect_deg=a.inspect_deg, springref=io.springref, rows=int(len(acts)), success=st["success"], episode_end=st["episode_end"], state_sha=st["state_sha"],
           wall_s=round(time.time() - t0, 1), failure=failure, metrics=s.env._task.result.metrics, events=len(orc.events))
np.save(out / "actions_all.npy", acts)
if st["success"]:
    s.save_trajectory(acts, out / "privileged.npz", meta=dict(source="buzz redesign oracle (feedback, live)", tier=a.tier,
                                                              robotiq_springref=io.springref, robot_mount=design.robot_mount,
                                                              plan=plan, wall_s=res["wall_s"]))
log(event="end", **res)
(out / "result.json").write_text(json.dumps(res, default=str, indent=1))
s.close()
