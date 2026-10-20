"""The two assisted-grasp welds' constraint forces over a cast_keyhole reference (production Session; fiber's
weldforce.py per weld): per MuJoCo step and active weld, the norm of its 3 translational rows (N); per weld the maximum,
percentiles and the slip rule's reading (grasp.WELD_SLIP_FORCE_N["aloha"] on the mean over grasp.WELD_SLIP_WINDOW_S of
substeps) with the control rows where it peaks; the episode end and the final state hash. With --phases (the oracle's
keyhole_monitor.jsonl: control step -> phase), the largest 20 ms mean per phase.

Run it with the RoboTangle image's Python, as the stress tools (scripts/robotangle/stress/_env.py).

usage: weldforce.py <instance.json> <privileged.npz> <out.json> [--phases <keyhole_monitor.jsonl>]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stress"))
import _env  # noqa: F401,E402  (before numpy: SIMD masking, threads, this checkout first on sys.path)
import json, time  # noqa: E402
import numpy as np  # noqa: E402
import mujoco as mj  # noqa: E402
from rcb_topogym import Session, load_trajectory  # noqa: E402
from topogym.world.mujoco.grasp import WELD_SLIP_FORCE_N, WELD_SLIP_WINDOW_S  # noqa: E402

inst, traj, out = sys.argv[1:4]
phases = {}
if "--phases" in sys.argv:
    for line in open(sys.argv[sys.argv.index("--phases") + 1]):
        r = json.loads(line)
        phases[int(r["state"]["control_step"])] = r["phase"]
s = Session(json.loads(Path(inst).read_text())["task"], instance=inst, strict=True)
w = s.env._world
m, d = w.model, w.data
welds = [e for e in range(m.neq) if int(m.eq_type[e]) == int(mj.mjtEq.mjEQ_WELD)]
EQ = int(mj.mjtConstraint.mjCNSTR_EQUALITY)
F = {e: [] for e in welds}
cur = [0]
orig = mj.mj_step


def step(model, data):
    orig(model, data)
    if model is not m:
        return
    n = int(d.nefc)
    typ, eid, f = np.asarray(d.efc_type[:n]), np.asarray(d.efc_id[:n]), np.asarray(d.efc_force[:n])
    for e in welds:
        if not d.eq_active[e]:
            continue
        rows = np.flatnonzero((typ == EQ) & (eid == e))
        if len(rows) >= 6:
            F[e].append((cur[0], float(np.linalg.norm(f[rows[:3]]))))


mj.mj_step = step
acts, ticks = load_trajectory(traj, s.task, s.action_dim, trusted=True, layout=s.action_layout)
t0 = time.time()
for i, row in enumerate(acts):
    cur[0] = i + 1
    if s.step(row, _physics_steps=None if ticks is None else int(ticks[i])):
        break
st = s.status()
dt = float(m.opt.timestep)
win, slip = max(1, round(WELD_SLIP_WINDOW_S / dt)), WELD_SLIP_FORCE_N.get("aloha")
res = dict(instance=inst, traj=traj, rows=cur[0], end=st["episode_end"], success=st["success"], state_sha=st["state_sha"],
           slip_force_N=slip, slip_window_substeps=win, wall_s=None, welds={})
for e, rec in F.items():
    if not rec:
        continue
    rw, f = np.array([r for r, _ in rec]), np.array([v for _, v in rec])
    mean = np.convolve(f, np.ones(win) / win, "valid") if len(f) >= win else f.mean()[None]
    mrow = rw[win - 1:] if len(f) >= win else rw[-1:]
    top = []
    for k in np.argsort(-mean):
        if all(abs(int(mrow[k]) - t["row"]) > 200 for t in top):
            top.append(dict(row=int(mrow[k]), mean_N=round(float(mean[k]), 2), phase=phases.get(int(mrow[k]))))
        if len(top) == 5:
            break
    per_phase = {}
    for v, r in zip(mean, mrow):
        p = phases.get(int(r))
        if p is not None and v > per_phase.get(p, 0.):
            per_phase[p] = round(float(v), 2)
    res["welds"][mj.mj_id2name(m, mj.mjtObj.mjOBJ_EQUALITY, e) or str(e)] = dict(
        held_s=round(len(f) * dt, 2), max_N=round(float(f.max()), 2), p99_N=round(float(np.percentile(f, 99)), 2),
        median_N=round(float(np.median(f)), 2), window_mean_max_N=round(float(mean.max()), 2), window_peaks=top,
        window_mean_max_per_phase_N=per_phase)
res["wall_s"] = round(time.time() - t0, 1)
Path(out).write_text(json.dumps(res, indent=1))
print(json.dumps(res))
