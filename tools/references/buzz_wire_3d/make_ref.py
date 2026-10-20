"""A buzz_wire_3d MuJoCo reference, recorded live in one fresh Session (the buzz agent's tool; the core agent re-planned
the -i01 references with it on the exact constraint diagonal, 2026-10-10): the released Genesis route's rows [0, cut),
then the family's own feedback finish stages planned from the measured MuJoCo state (topogym.planning.buzz_wire_3d: the
simplified oracle's finish, recipes.easy_finish_stages, or buzz_finish.run_finish's adaptive rounds), every committed row
within the standard per-step joint limit (standard_rate.rate_limited_runner), then hold rows until the episode ends:
also after a finish that reports a failure, since the wand may already rest within tolerance and need only its
stability window (buzz hard). An episode that ends without success is never saved.

Run it with the RoboTangle image's Python, as the stress tools (tools/stress/_env.py: this checkout's
topogym and tooling first on sys.path, the task containers' numpy / OpenBLAS SIMD masking, one thread per library).

usage: make_ref.py <instance.json> <route.npz> --cut N --finish {simplified,easy,adaptive,...} --out ref.npz [--tail 400]
       [--log x] [--output-dir d]
buzz_finish.py reads its knobs from the environment (BUZZ_REACH_DEG, BUZZ_ROUNDS, BUZZ_GRASP_BUDGET, BUZZ_PARTIAL, ...).
The -i01 references (docs/simulator/families/buzz_wire_3d.md): the route tools/references/oracles_genesis/buzz-wire-3d-<tier>-i00/
privileged.npz, --finish adaptive, --cut 7000 (very_easy), 7850 (easier); on the Robotiq's springref 2.62 with the
finish's staged grasps and slow releases (buzz_finish.STAGED, the default)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "stress"))
import _env  # noqa: F401,E402  (before numpy: SIMD masking, threads, this checkout first on sys.path)
import argparse, hashlib, json, os, time  # noqa: E402
import numpy as np  # noqa: E402
sys.path.insert(0, str(HERE))
from rcb_topogym import Session  # noqa: E402
import buzzlib  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("instance"); ap.add_argument("route")
ap.add_argument("--cut", type=int, required=True); ap.add_argument("--finish", required=True, choices=("simplified", "easy", "adaptive", "lower+adaptive", "level+adaptive", "centre+adaptive", "held+adaptive", "heldturn+adaptive"))
ap.add_argument("--out"); ap.add_argument("--tail", type=int, default=400); ap.add_argument("--log")
ap.add_argument("--output-dir", default=None)
ap.add_argument("--route-grasps", default=None, help="Genesis trace (gtrace_buzz.py) of the released rows: its regrasps' "
                "hand-on-wand poses and arm joints become the first grasp candidates")
ap.add_argument("--grasp-rows", default="", help="comma list of trace rows (default: every regrasp after the cut)")
ap.add_argument("--grasp-back", default="0,100,200,300,400", help="rows before each grasp row to sample")
ap.add_argument("--grid-grasps", default=None, help="probe_grid2.py output: its collision-free pre-grasp goals (best "
                "clearance first) become the first grasp candidates")
ap.add_argument("--grid-top", type=int, default=12); ap.add_argument("--grid-filter", default="")
ap.add_argument("--seed-rows", default="", help="trace rows whose arm joints seed every grid grasp's IK goal")
a = ap.parse_args()
t0 = time.time()
s = Session("buzz_wire_3d", instance=a.instance, strict=True)
env, w = s.env, s.env._world
tier = env._task.variation_id.replace("buzz_wire_3d_", "")
out_dir = Path(a.output_dir or f"buzz-finish-{tier}-{int(time.time())}")
out_dir.mkdir(parents=True, exist_ok=True)
log = open(a.log or out_dir / "log.jsonl", "w")


def note(**kv):
    m = env._task.result.metrics
    rec = dict(n=s.n_steps, t=round(s.sim_time_s, 3), wall=round(time.time() - t0, 1), end=s.episode_end,
               complete=m.get("traversal_complete"), dist_mm=None if m.get("distance_to_finish_m") is None else round(1e3 * m["distance_to_finish_m"], 3),
               angle=None if m.get("finish_angle_deg") is None else round(m["finish_angle_deg"], 3), supported=m.get("supported"),
               released=m.get("released"), quiet=m.get("quiet"), stable=m.get("stable_time_s"),
               clear_mm=round(1e3 * buzzlib.clearance(w)[0], 3), **kv)
    log.write(json.dumps(rec, default=str) + "\n"); log.flush()
    print(json.dumps(rec, default=str), flush=True)
    return rec


s.record_start()
rows = np.load(a.route)["actions"].astype(np.float32)[:a.cut]
for i, row in enumerate(rows):
    if s.step(row):
        break
    if i % 1000 == 0:
        note(phase="route")
note(phase="route_end")
from topogym.planning.standard_rate import rate_limited_runner
from topogym.planning.buzz_wire_3d.simplified_policy import SimplifiedBuzzOracle, make_stage_policy
runner = rate_limited_runner(s)
if a.finish == "simplified":
    oracle = SimplifiedBuzzOracle(env, out_dir, variant=tier, regrasp=True, finish_mode="low_slide_regrasp",
                                  terminal_contact_close=False)
    stages = oracle.stages(start="released", part="solve")
failure = None
if a.finish in ("simplified", "easy"):
    try:
        if a.finish == "easy":
            from topogym.planning.buzz_wire_3d.recipes import easy_finish_stages
            stages = ((f"finish_recipe_{k}", p) for k, p in enumerate(easy_finish_stages(env, out_dir)) if k > 0)
        for name, parameters in stages:
            tp = time.time()
            note(phase="stage_start", stage=name, part=parameters.get("part"))
            result = runner.run_policy(request_id=name, policy=make_stage_policy(env, parameters),
                                       submission_hash=s.instance_sha, state_fields=("world",))
            note(phase="stage_end", stage=name, reason=result["reason"], stage_wall=round(time.time() - tp, 1))
            if s.success() or s.episode_end is not None:
                break
            if result["reason"] != "policy_done":
                failure = f"stage {name}: {result['reason']}"
                break
    except Exception as exc:
        import traceback
        failure = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
else:
    import buzz_finish
    release = dict(lower=buzz_finish.lower_and_release, level=buzz_finish.level_release,
                   centre=buzz_finish.centre_release, heldturn=buzz_finish.held_turn_release).get(a.finish.split("+")[0])
    extra = []
    if a.route_grasps:
        from scipy.spatial.transform import Rotation as _R
        recs = {}
        for line in open(a.route_grasps):
            if line.startswith("{\"i\""):
                r_ = json.loads(line); recs[int(r_["i"])] = r_
        idx = sorted(recs)
        rows_ = [int(x) for x in a.grasp_rows.split(",") if x] or [i for k, i in enumerate(idx[1:], 1)
                 if recs[i]["grasp"] is not None and recs[idx[k - 1]]["grasp"] is None and i > a.cut]
        for b in rows_:
            for back in [int(x) for x in a.grasp_back.split(",") if x]:
                j = b - back
                if j not in recs:
                    continue
                r_ = recs[j]
                W = _R.from_quat(np.asarray(r_["wand_q"])[[1, 2, 3, 0]]); T = _R.from_quat(np.asarray(r_["tcp_q"])[[1, 2, 3, 0]])
                extra.append((f"g{b}-{back}", (W.inv().apply(np.asarray(r_["tcp_p"]) - np.asarray(r_["wand_p"])), W.inv() * T,
                                                np.asarray(r_["q"]))))
        note(phase="plan", route_grasps=[l for l, _ in extra])
    if a.seed_rows and a.route_grasps:
        recs_ = {}
        for line in open(a.route_grasps):
            if line.startswith("{\"i\""):
                r_ = json.loads(line); recs_[int(r_["i"])] = r_
        buzz_finish.GOAL_SEEDS[:] = [np.asarray(recs_[int(x)]["q"]) for x in a.seed_rows.split(",") if int(x) in recs_]
        note(phase="plan", goal_seeds=a.seed_rows)
    if a.grid_grasps:
        goals = [json.loads(l) for l in open(a.grid_grasps) if l.startswith("{\"fraction\"")]
        for f in [x for x in a.grid_filter.split(",") if x]:
            k_, v_ = f.split("=")
            goals = [g_ for g_ in goals if abs(float(g_[k_]) - float(v_)) < 1e-9]
        goals.sort(key=lambda g_: -g_["clear_mm"])
        for g_ in goals[:a.grid_top]:
            extra.append((f"grid2-f{g_['fraction']}-t{g_['tilt']}-y{g_['yaw']}-r{g_['roll']}-h{g_['height']}-a{g_['approach']}",
                          dict(yaw=g_["yaw"], tilt=g_["tilt"], roll=g_["roll"], fraction=g_["fraction"], height=g_["height"],
                               approach=g_["approach"], q_goal=g_["q"])))
        note(phase="plan", grid_grasps=[l for l, _ in extra])
    try:
        failure = buzz_finish.run_finish(env, s, runner, out_dir, lambda **kv: note(**kv), release=release,
                                         start_held=a.finish.startswith("held"), extra_grasps=extra,
                                         grasp_budget=int(os.environ.get("BUZZ_GRASP_BUDGET", "8")))
    except Exception as exc:
        import traceback
        failure = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
held = 0
while s.episode_end is None and held < a.tail:      # also after a finish failure (see above)
    s.step(s.hold_action())
    held += 1
note(phase="end", failure=failure, held=held)
acts = s.record_stop()
st = s.status()
res = dict(tier=tier, rows=int(len(acts)), cut=a.cut, success=st["success"], episode_end=st["episode_end"],
           physics_steps=st["physics_steps"], wall_s=round(time.time() - t0, 1), state_sha=st["state_sha"],
           review=st["review"], forcing=st["forcing"], failure=failure, rate=runner.rate_limiter.report())
if a.out and st["success"]:
    meta = dict(source="mujoco reference (buzz agent): released Genesis route rows [0, cut) + the family's feedback finish "
                       "stages planned live on MuJoCo + hold rows", route=a.route,
                route_sha256=hashlib.sha256(Path(a.route).read_bytes()).hexdigest(), cut=a.cut, finish=a.finish,
                wall_s=res["wall_s"], success=st["success"], episode_end=st["episode_end"])
    s.save_trajectory(acts, a.out, meta=meta)
    res.update(out=a.out, sha256=hashlib.sha256(Path(a.out).read_bytes()).hexdigest())
print(json.dumps(res, default=str), flush=True)
s.close()
