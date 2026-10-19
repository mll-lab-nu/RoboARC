"""The premade knot rd instance's checks inside the release image (one process each, by argv[1]; with make_premade.py):
  birth  <instance.json>                 the Session built strict, its state hash right after load (birth_probe.py's)
  nop    <instance.json> <rows>          the reset targets held for <rows> control steps: no success, no instability
  speed  <instance.json> <rows.npz>      every row's joint step within the robot's joint speed limits (Session._slew),
                                         the Cartesian ones if the robot has any, and every arm joint step within the
                                         standard service's per-step limit (rcb_topogym.JOINT_STEP_LIMIT_RAD)
(the replay itself is rcb_topogym.py replay --strict)"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/tooling")
from rcb_topogym import Session  # noqa: E402

what, inst = sys.argv[1], sys.argv[2]
s = Session("knot_untying", instance=inst, strict=True)
if what == "birth":
    out = dict(state_sha=s.state_sha(), frozen_state_sha=json.load(open(inst)).get("state_sha"),
               instance_exact=bool(s.instance_exact), assets_exact=bool(getattr(s, "assets_exact", False)),
               binding=bool(s.runtime_binding), NPY=os.environ.get("NPY_DISABLE_CPU_FEATURES"),
               OPENBLAS=os.environ.get("OPENBLAS_CORETYPE"))
elif what == "nop":
    n = int(sys.argv[3])
    hold = np.asarray(s._targets, dtype=np.float32).copy()
    end = None
    for i in range(n):
        if s.step(hold):
            end = i
            break
    st = s.status()
    out = dict(rows=int(s.n_steps), ended_at=end, success=st.get("success"), episode_end=st.get("episode_end"),
               instability=st.get("instability"), review=st.get("review"), forcing=st.get("forcing"),
               state_sha=st.get("state_sha"))
elif what == "speed":
    z = np.load(sys.argv[3], allow_pickle=False)
    acts = np.asarray(z["actions"], np.float64)
    r = s.robot
    speeds = getattr(s.env._robot, "joint_speed_limits", None) or {}
    cart = getattr(s.env._robot, "cartesian_speed_limits", None) or {}
    prev = np.asarray(s._targets, np.float64)[None]
    steps = np.diff(np.vstack([prev, acts]), axis=0)
    rep = {}
    for arm, sp in speeds.items():
        sl = r.arm_slice[arm]
        cap = np.asarray(sp, np.float64) / s.control_hz
        ratio = np.abs(steps[:, sl]) / cap
        over = np.abs(steps[:, sl]) > cap + 1e-6            # Session._slew's test (float32 rounding never re-triggers)
        rep[arm] = dict(max_ratio=float(ratio.max()), rows_over=int(over.any(axis=1).sum()),
                        worst_row=int(ratio.max(axis=1).argmax()), cap_rad_per_step=cap.round(6).tolist())
    import rcb_topogym
    lim = float(rcb_topogym.JOINT_STEP_LIMIT_RAD)
    arm_cols = np.concatenate([np.arange(len(acts[0]))[r.arm_slice[a]] for a in speeds]) if speeds else np.arange(0)
    std = float(np.abs(steps[:, arm_cols]).max()) if len(arm_cols) else 0.
    out = dict(rows=int(len(acts)), control_hz=s.control_hz, joint=rep, cartesian_limits=bool(cart),
               standard_step_limit_rad=lim, max_arm_step_rad=round(std, 6),
               ok=all(v["rows_over"] == 0 for v in rep.values()) and std <= lim)
print(what.upper() + " " + json.dumps(out, default=str), flush=True)
os._exit(0)
