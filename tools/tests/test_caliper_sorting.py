#!/usr/bin/env python
"""caliper_sorting's terminal rules (sleeves_in_labelled_bins_v2, 2026-10-19) through a Session:

  1. a sleeve set down still in a bin that is not its class's ends the episode in failure once it has stood there for
     the confirmation time (0.5 s), and not before;
  2. the same sleeve set down in its own class's bin ends nothing (the other three are still in the rack: no success);
  3. the birth targets held (a nop) end nothing.

The sleeve is put in place by writing its free joint in the simulation (a test of the rules only, never of a route).

  python tools/tests/test_caliper_sorting.py --instance <instance.json>
Run it where make-instance runs (the release image, the SIMD masking, rendering off). Exit 0 when every check
holds; prints one JSON line."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "roboarc" / "shared"))


def put(s, name, bin_index):
    """Sleeve `name` standing on the floor of bin `bin_index` (its axis along the mount's y, as the reference sets them
    down), at rest."""
    import mujoco
    from scipy.spatial.transform import Rotation
    from topogym.assets.geometry import caliper_sorting as G
    w, task = s.env._world, s.env._task
    m, d = w.model, w.data
    k = task.sleeves.index(name)
    ro = task.scenario["sleeves"][k]["od_mm"] / 2e3
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{name}::root")
    qa, va = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
    centre = np.asarray(task.bin_centres[bin_index], float) + task.mount_rotation.apply([0., 0., G.BINS["wall"] * 1e-3 + ro + 2e-4])
    quat = (task.mount_rotation * Rotation.from_quat(np.asarray(G.SLEEVE_QUAT_WXYZ)[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]
    d.qpos[qa:qa + 3], d.qpos[qa + 3:qa + 7] = centre, quat
    d.qvel[va:va + 6] = 0.
    mujoco.mj_forward(m, d)


def run(s, name, bin_index, seconds):
    """From the birth (reset): the sleeve put in the bin, then the birth targets held for `seconds`."""
    s.reset()
    put(s, name, bin_index)
    ended_at, steps = None, int(round(seconds * s.control_hz))
    for i in range(steps):
        if s.step(s.hold_action()):
            ended_at = (i + 1) / s.control_hz
            break
    st = s.status()
    return dict(ended_at_s=ended_at, episode_end=st.get("episode_end"), success=bool(st.get("success")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    a = ap.parse_args()
    from rcb_topogym import Session
    s = Session("caliper_sorting", instance=a.instance, strict=True)
    task = s.env._task
    own = task.scenario["bins"].index(task.scenario["sleeves"][0]["cls"])
    wrong = next(j for j in range(len(task.scenario["bins"])) if j != own)
    out = {}
    w = run(s, "sleeve_0", wrong, 2.0)
    out["wrong_bin"] = w
    out["wrong_bin_fails"] = bool(w["ended_at_s"] is not None and not w["success"]
                                  and "not its own" in str(w["episode_end"]))
    out["wrong_bin_not_before_confirmation"] = bool(w["ended_at_s"] is not None and w["ended_at_s"] >= 0.5)
    r = run(s, "sleeve_0", own, 2.0)
    out["own_bin"] = r
    out["own_bin_ends_nothing"] = bool(r["ended_at_s"] is None and not r["success"])
    s.reset()
    for _ in range(80):
        s.step(s.hold_action())
    st = s.status()
    out["nop_ends_nothing"] = bool(not st.get("success") and st.get("episode_end") in (None, ""))
    s.close()
    ok = out["wrong_bin_fails"] and out["wrong_bin_not_before_confirmation"] and out["own_bin_ends_nothing"] and out["nop_ends_nothing"]
    print(json.dumps(dict(ok=ok, own_bin_index=own, wrong_bin_index=wrong, **out)), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
