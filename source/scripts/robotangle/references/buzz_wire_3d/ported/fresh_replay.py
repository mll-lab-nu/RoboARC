"""Fresh-process full replay of a reference, the graders' way (Session.replay(grading=True)).
usage: fresh_replay.py <instance.json> <privileged.npz> [--bypass-binding]   -> one JSON line"""
import json, sys, time
inst, ref = sys.argv[1], sys.argv[2]
if "--bypass-binding" in sys.argv:
    import topogym.simulator.frozen_binding as fb
    import rcb_topogym as R
    fb.check_inputs = lambda instance_path, tooling_path, lock: json.loads(open(instance_path).read())
    R.check_runtime = lambda lock: dict(lock.get("runtime") or {})
    R._check_render_bridge = lambda lock: None
import numpy as np
import rcb_topogym as R
t0 = time.time()
task = json.load(open(inst))["task"]
s = R.Session(task, instance=inst, strict=True)
acts, ticks = R.load_trajectory(ref, task, s.action_dim, trusted=True, layout=s.action_layout)
t1 = time.time()
st = s.replay(acts, physics_steps=ticks, grading=True)
home = np.asarray(s.initial_state()["joint_targets"] if False else acts[0], float)
d = np.abs(np.diff(np.asarray(acts, float), axis=0))[:, :7].max()
print(json.dumps(dict(instance=inst, ref=ref, rows=int(len(acts)), instance_sha=s.instance_sha, instance_exact=s.instance_exact,
                      assets_exact=s.assets_exact, success=st["success"], state_sha=st["state_sha"],
                      episode_end=st.get("episode_end"), grading=st.get("grading"), build_s=round(t1 - t0, 1),
                      replay_s=round(time.time() - t1, 1), max_row_dq=float(d)), default=str), flush=True)
