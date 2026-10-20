"""Robot-level breakers of a redesigned claws tier, each in a fresh Session (the task's own verdict decides):
  straight:<dir>   the hand on B pulls it 0.25 m in a straight line (dir: shaft = out along B's shaft, up, xp, xm, yp,
                   ym = world axes), the left hand holds A (closed-loop tracking as the oracle), then the oracle's
                   transport into the goal area and a settle;
  one_arm          the left hand lets go of A at once; the oracle's right-arm rows run as recorded, left arm frozen open;
  rows:<npz>       any row file replayed as is (the old -i01 oracle on the new scene).
python work/session_breakers.py <tier> <variant> <out_dir> [--oracle rows.npz]"""
import json, sys
from pathlib import Path
import numpy as np
import os; sys.path.insert(0, os.environ.get("CL_ORACLE_DIR", os.path.dirname(os.path.abspath(__file__))))
from oracle import Runner, measured, track, transport, piece_of, split_poses
from topogym.planning.interlocked_claws.redesign import paths
from topogym.planning.interlocked_claws.redesign.robot_plan import ARMS, solve_path, rows_from_waypoints, OPEN

tier, variant, out = sys.argv[1], sys.argv[2], sys.argv[3]
oracle_rows = sys.argv[sys.argv.index("--oracle") + 1] if "--oracle" in sys.argv else None
run = Runner(tier, out, speed=.2, share=0.)
res = dict(tier=tier, variant=variant, birth=run.birth_sha)
if variant.startswith("straight:"):
    run.hold("acquire", 30)
    holds = dict(run.design["hands"])
    T, q, w0 = measured(run)
    T_A, T_B = T["piece_1"], T["piece_0"]
    key = variant.split(":")[1]
    if key == "shaft":                         # out along B's shaft: from the claw toward the grip end
        v = T_B[:3, 0]
    else:
        v = dict(up=[0, 0, 1], xp=[1, 0, 0], xm=[-1, 0, 0], yp=[0, 1, 0], ym=[0, -1, 0])[key]
    v = .25 * np.asarray(v, float) / np.linalg.norm(v)
    mv = dict(kind="slide", vector=(T_A[:3, :3].T @ v).tolist())
    offs = {arm: np.linalg.inv(T[piece_of(g)]) @ run.kin.tcp_frame(arm, q[arm]) for arm, g in holds.items()}
    rel0 = np.linalg.inv(T_A) @ T_B
    poses, _, _ = paths.dense(rel0, [mv], step_m=.002)
    TA, TB = split_poses(T_A, rel0, poses, share_A=0.)
    qs = {}
    for arm in ARMS:
        Ts = [(TA[k] if piece_of(holds[arm]) == "piece_1" else TB[k]) @ offs[arm] for k in range(len(poses))]
        sol, rep = solve_path(run.kin, arm, Ts, w0[arm])
        if sol is None:
            sol, cur = [], w0[arm].copy()
            for Tt in Ts:
                cur, _, _ = run.kin.ik(arm, Tt, cur, prior=cur); sol.append(cur.copy())
        qs[arm] = sol
    track(run, "pull", poses, TA, TB, qs, holds, offs, T_A, T_B)
    if not run.s.status()["episode_end"] and run.s.object_in_hand("right"):
        transport(run, holds)
    if not run.s.success() and not run.s.status()["episode_end"]:
        run.hold("settle", 120)
elif variant == "one_arm":
    z = np.load(oracle_rows, allow_pickle=False)
    rows = np.asarray(z["actions"], np.float64)
    cols = {n: i for i, n in enumerate(run.layout)}
    lc = [cols[f"left/{j}"] for j in ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")]
    rows[:, lc] = rows[0, lc]
    g = cols["left_native_gripper"]
    rows[:, g] = OPEN
    run.execute("one_arm", rows, stop_on_success=True)
    run.hold("settle", 120)
elif variant.startswith("rows:"):
    z = np.load(variant[5:], allow_pickle=False)
    if list(z["action_layout"]) != run.layout:
        raise SystemExit("layout differs")
    run.execute("rows", np.asarray(z["actions"]), stop_on_success=True)
    run.hold("settle", 120)
st = run.s.status()
T, q, w0 = measured(run)
task = run.task
ok, fail, metrics = task.evaluate_goal_region(run.state(), task.goal_region)
res.update(success=st["success"], episode_end=st["episode_end"], rows=len(run.rows), goal=metrics,
           held={a: run.s.object_in_hand(a) for a in ARMS}, phases=run.phases,
           B_moved_rel_m=None)
print("BREAKER", json.dumps(res, default=str))
Path(out).mkdir(parents=True, exist_ok=True)
(Path(out) / "breaker.json").write_text(json.dumps(res, indent=1, default=str))
