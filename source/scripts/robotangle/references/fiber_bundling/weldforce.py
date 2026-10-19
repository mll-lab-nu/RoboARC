"""The assisted-grasp weld's constraint force over a stream (production Session): per MuJoCo step, for every active grasp
weld, the norm of its 3 translational rows (N) and of its 3 rotational rows (N m, MuJoCo's weld torque rows); the
maximum, percentiles, the slip rule's reading (grasp.WELD_SLIP_FORCE_N on the mean over grasp.WELD_SLIP_WINDOW_S of
substeps) with the control rows where it peaks, and the time spent above given levels. Also the episode end, the
final state hash and each strand's final largest radius from the sheath axis (mm).

Run it with the RoboTangle image's Python, as the stress tools (scripts/robotangle/stress/_env.py).

usage: weldforce.py <instance.json> <actions.npz> <out.json>"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stress"))
import _env  # noqa: F401,E402  (before numpy: SIMD masking, threads, this checkout first on sys.path)
import json, time  # noqa: E402
import numpy as np  # noqa: E402
import mujoco as mj  # noqa: E402
from rcb_topogym import Session  # noqa: E402

inst, actions, out = sys.argv[1:4]
s = Session(json.loads(Path(inst).read_text())["task"], instance=inst, strict=True)
w = s.env._world
m, d = w.model, w.data
welds = [e for e in range(m.neq) if int(m.eq_type[e]) == int(mj.mjtEq.mjEQ_WELD)]
EQ = int(mj.mjtConstraint.mjCNSTR_EQUALITY)
F, T, RW = [], [], []
cur = [0]
orig = mj.mj_step


def step(model, data):
    orig(model, data)
    if model is not m:
        return
    n = int(d.nefc)
    if not n:
        return
    typ, eid, f = np.asarray(d.efc_type[:n]), np.asarray(d.efc_id[:n]), np.asarray(d.efc_force[:n])
    for e in welds:
        if not d.eq_active[e]:
            continue
        rows = np.flatnonzero((typ == EQ) & (eid == e))
        if len(rows) >= 6:
            F.append(float(np.linalg.norm(f[rows[:3]])))
            T.append(float(np.linalg.norm(f[rows[3:6]])))
            RW.append(cur[0])


mj.mj_step = step
rows = np.load(actions)["actions"].astype(np.float32)
t0 = time.time()
n = 0
for row in rows:
    n += 1
    cur[0] = n
    if s.step(row):
        break
st = s.status()
from topogym.planning.fiber_bundling.keyframes import strand_measures
final_radius = [round(1e3 * x["max_radius_m"], 2) for x in strand_measures(w.read_state(), s.env._task.layout)]
F, T = np.array(F), np.array(T)


def window_peaks(F, RW, win):
    if len(F) < win:
        return []
    m = np.convolve(F, np.ones(win) / win, "valid")
    rw = np.asarray(RW)[win - 1:]
    best = {}
    for v, r in zip(m, rw):
        b = int(r) // 200
        if v > best.get(b, (0., 0))[0]:
            best[b] = (float(v), int(r))
    return [dict(row=r, mean_N=round(v, 2)) for v, r in sorted(best.values(), reverse=True)[:5]]
dt = float(m.opt.timestep)
try:
    from topogym.world.mujoco.grasp import WELD_SLIP_FORCE_N, WELD_SLIP_WINDOW_S
    win, slip = max(1, round(WELD_SLIP_WINDOW_S / dt)), WELD_SLIP_FORCE_N.get("panda_robotiq")
except ImportError:
    win, slip = 32, None
res = dict(instance=inst, rows=n, end=st["episode_end"], success=st["success"], state_sha=st["state_sha"],
           final_radius_mm=final_radius, weld_steps=int(len(F)),
           held_s=round(len(F) * dt, 2), wall_s=round(time.time() - t0, 1))
if len(F):
    res.update(force_N=dict(max=round(float(F.max()), 2), p99=round(float(np.percentile(F, 99)), 2),
                            p90=round(float(np.percentile(F, 90)), 2), median=round(float(np.median(F)), 2)),
               torque_Nm=dict(max=round(float(T.max()), 3), p99=round(float(np.percentile(T, 99)), 3)),
               # the slip rule's reading (grasp.WELD_SLIP_FORCE_N on the mean over grasp.WELD_SLIP_WINDOW_S of substeps)
               slip_window_mean_N_max=round(float(np.convolve(F, np.ones(win) / win, "valid").max()) if len(F) >= win else float(F.mean()), 2),
               slip_window_substeps=win, slip_force_N=slip,
               # control rows where the 20 ms mean is highest (1 per 200-row block, the top 5)
               window_peaks=window_peaks(F, RW, win),
               time_above_s={str(k): round(float((F > k).sum()) * dt, 3) for k in (20, 40, 60, 100, 200)})
Path(out).write_text(json.dumps(res, indent=1))
print(json.dumps(res))
