"""orbits: replay a recorded reference on its frozen instance (fresh process, the recorded ticks) and measure, after every
control step, the middle tie's gap (OrbitsBelowEasyTask.cord_points: link m-1's tip to link m's root) and every held
contact weld's drift (the hand-target translation and rotation error against the pose captured at acquisition, as the
grasp manager checks it against max_translation_error_m). Reports the maxima, where they occur, and the final hash.
usage: orbits_measure.py <instance.json> <privileged.npz> <out.json>"""
import json
import sys

import numpy as np

sys.path.insert(0, "scripts/robotangle/shared")
from rcb_topogym import Session, load_trajectory  # noqa: E402
from topogym.utils.grasp import relative_error  # noqa: E402

inst, traj, out = sys.argv[1:4]
s = Session(json.load(open(inst))["task"], instance=inst, strict=True)
s.env._world.renderer_factory = None
acts, ticks = load_trajectory(traj, s.task, s.action_dim, trusted=True, layout=s.action_layout)
task, world = s.env._task, s.env._world
manager = world._grasp_manager
tie_max, tie_at, drift_max, drift_at, angle_max = 0., None, 0., None, 0.
series = []
for i, a in enumerate(acts):
    s.step(a, _physics_steps=None if ticks is None else int(ticks[i]))
    _, tie = task.cord_points(world.read_state())
    if tie > tie_max:
        tie_max, tie_at = tie, i
    drift = 0.
    for hand, state in manager.hands.items():
        b = state.get("binding")
        if b is None:
            continue
        t, ang = relative_error(b["relative_pose"], manager.backend.relative_pose(hand, b["target"]))
        drift = max(drift, float(t))
        angle_max = max(angle_max, float(ang))
        if t > drift_max:
            drift_max, drift_at = float(t), (i, hand, list(b["target"]))
    series.append((round(1000 * tie, 3), round(1000 * drift, 3)))
    if s.episode_end is not None:
        break
st = s.status()
report = dict(instance=inst, traj=traj, rows=len(acts), executed=i + 1, success=st.get("success"),
              episode_end=st.get("episode_end"), state_sha=st.get("state_sha"),
              tie_max_mm=round(1000 * tie_max, 3), tie_max_row=tie_at, tie_limit_mm=1000 * task.TIE_LIMIT_M,
              weld_drift_max_mm=round(1000 * drift_max, 3), weld_drift_max_at=drift_at,
              weld_rotation_max_rad=round(angle_max, 4),
              weld_limit_mm=1000 * float(s.config.get("grasp_assist", {}).get("max_translation_error_m", float("nan"))),
              tie_over_2mm_rows=int(sum(t > 2. for t, _ in series)), drift_over_10mm_rows=int(sum(d > 10. for _, d in series)))
open(out, "w").write(json.dumps(dict(report, series_mm=series), indent=0))
print(json.dumps(report))
s.close()
