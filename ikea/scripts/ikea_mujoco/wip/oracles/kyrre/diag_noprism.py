#!/usr/bin/env python3
"""DIAGNOSTIC ONLY (not the benchmark physics): replay a reference with the cylinder->prism rule switched off (the
scene's ikea_runtime.cylinder_prisms replaced by a no-op), to see whether a failure comes from the prism rule.
usage: diag_noprism.py TASK REF.npz OUT.json [--every N] [--prisms]   (--prisms: keep the rule, the control run)"""
import json, os, sys, time
sys.path.insert(0, os.environ["RCB_SHARED"])
import numpy as np
import rcb_ikea as R
task, ref, out = sys.argv[1:4]
every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 20000
scene_dir = os.path.join(R.SCENES_ROOT, task)
sys.path.insert(0, scene_dir)
import ikea_runtime
if "--prisms" not in sys.argv:
    ikea_runtime.cylinder_prisms = lambda spec: []
s = R.Session(task)
print("prisms", len(s.S.cylinder_prisms), flush=True)
ctrl = R.load_trajectory(ref, s.nu, s.timestep)
weld_keys = list(s.S.mech.eq.keys()) if hasattr(s.S.mech, "eq") else []
log = []; t0 = time.time()
for i, row in enumerate(ctrl):
    s.step(row)
    if (i + 1) % every == 0:
        w = [k for k in weld_keys if s.data.eq_active[s.S.mech.eq[k]]]
        log.append(dict(row=i + 1, wall=round(time.time() - t0, 1), welds=len(w), last=w[-1] if w else None))
        print(json.dumps(log[-1]), flush=True)
fc = s.final_check()
res = dict(task=task, prisms=len(s.S.cylinder_prisms), success=fc["success"], state_sha=fc["state_sha"],
           end={k: fc["end"][k] for k in ("goal", "released", "at_rest", "breakdown", "report")}, log=log,
           wall=round(time.time() - t0, 1))
json.dump(res, open(out, "w"), default=R._plain, indent=1)
print(json.dumps({k: res[k] for k in ("success", "prisms", "state_sha", "wall")}))
