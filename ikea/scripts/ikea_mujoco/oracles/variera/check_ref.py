#!/usr/bin/env python3
"""Fresh-process check of the variera reference: rows over the joint-speed limits (row to row, the first row from the
keyframe ctrl), then the grader's path (rcb_ikea.Session: open-loop rows through the clamp, final_check()), wall time.
usage: check_ref.py REF.npz OUT.json   (rcb_ikea on PYTHONPATH or RCB_SHARED, the scene on IKEA_SCENES)"""
import json, os, sys, time
sys.path.insert(0, os.environ.get("RCB_SHARED", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared")))
import numpy as np
import rcb_ikea as R

ref, out = sys.argv[1:3]
t0 = time.time()
s = R.Session("variera"); init = s.state_sha()
c = R.load_trajectory(ref, s.nu, s.timestep)
prev = np.vstack([s.data.ctrl.copy()[None], c[:-1]]); lim = s.step_limit; fin = np.isfinite(lim)
d = np.abs(np.clip(c, s._clo, s._chi) - prev)
over = int((d[:, fin] > lim[fin] + R.SPEED_TOL).any(1).sum())
t1 = time.time(); s.run(c); t_run = time.time() - t1
fc = s.final_check()
res = dict(ref=ref, ref_sha256=R.file_sha(ref), rows=len(c), rows_over_limit=over, max_ratio=float((d[:, fin] / lim[fin]).max()),
           slewed_ticks=s.slewed_ticks, initial_state_sha=init, success=fc["success"], state_sha=fc["state_sha"],
           settled_sha=fc["settled_sha"], end={k: fc["end"][k] for k in ("success", "goal", "released", "at_rest", "breakdown", "terminal")},
           settled={k: fc["settled"][k] for k in ("success", "goal", "released", "at_rest", "breakdown", "terminal")},
           report=fc["end"]["report"], wall_run_s=round(t_run, 1), wall_total_s=round(time.time() - t0, 1), host=os.uname().nodename)
try:
    res["cpu"] = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
except Exception:
    pass
json.dump(res, open(out, "w"), default=R._plain, indent=1)
print(json.dumps({k: res[k] for k in ("success", "rows", "rows_over_limit", "slewed_ticks", "state_sha", "settled_sha", "wall_run_s", "host")}, default=R._plain))
