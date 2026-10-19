#!/usr/bin/env python3
"""uppsta's reference from the author's demo, brought within the joint-speed limits without changing its timing.

The demo holds each joint target for one source tick of 2.5 ms (8 physics steps of 0.3125 ms); our rows are 4 physics
steps (1.25 ms), so the demo file repeats every tick's row twice. A tick that moves a joint by up to 2 x (limit x 1.25 ms)
is within the robot's limit over its 2.5 ms, but its first row asks for twice the per-row limit. Methods (arm joints only;
the gripper commands are the demo's, bit for bit; every row keeps its time):
  interp: the first row of each tick goes halfway from the previous tick's target to this tick's target, the second row
          is the tick's target (what the tick means at the 1.25 ms row); from the keyframe ctrl for the first tick.
  clamp:  exactly the clamp rcb_ikea.Session.applied() applies to the demo (std_reference.clamp): the reference replays
          bit for bit as the demo does under the clamp.
Run inside the IKEA MuJoCo image (needs /opt/ikea/scenes/uppsta or --scene-dir):
  python build_ref.py --demo trajectory.npz --method interp --out uppsta.npz [--rcb-ikea DIR] [--std DIR]
Prints the step statistics (rows over the limit + 1e-9 from the keyframe ctrl: must be 0)."""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def tick_interp(ctrl: np.ndarray, ctrl0: np.ndarray, arm: list) -> np.ndarray:
    x = np.asarray(ctrl, dtype=np.float64)
    assert len(x) % 2 == 0 and np.array_equal(x[0::2], x[1::2]), "not a 2x-repeated tick file"
    T = x[0::2, arm]
    prev = np.vstack([np.asarray(ctrl0, dtype=np.float64)[arm][None], T[:-1]])
    out = x.copy()
    out[0::2, arm] = prev + 0.5 * (T - prev)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=("interp", "clamp"), default="interp")
    ap.add_argument("--scene-dir", default=None)
    ap.add_argument("--rcb-ikea", default=os.path.join(HERE, "..", "..", "shared"))
    ap.add_argument("--std", default=os.path.join(HERE, "..", "..", "standard"))
    a = ap.parse_args()
    sys.path[:0] = [os.path.abspath(a.rcb_ikea), os.path.abspath(a.std)]
    import rcb_ikea as R
    import std_reference as SR
    s = R.Session("uppsta", a.scene_dir)
    ctrl0 = s.data.ctrl.copy()
    grips = SR.grip_columns(s.actuators)
    arm = [j for j in range(s.nu) if j not in grips]
    limit = np.asarray(s.step_limit, dtype=np.float64)[arm]
    c = R.load_trajectory(a.demo, s.nu, s.timestep)
    if a.method == "interp":
        out = tick_interp(c, ctrl0, arm)
        info = {"method": "interp", "rows_changed": int((out != c).any(1).sum())}
    else:
        out, info = SR.clamp(c, ctrl0, grips, limit)
    dev = np.abs(out[:, arm] - c[:, arm]).max(1)
    info.update(max_abs_dev_rad=float(dev.max()), grip_rows_identical=bool(np.array_equal(out[:, grips], c[:, grips])))
    s.save_trajectory(out, a.out)
    print(json.dumps({"task": "uppsta", **info, **SR.step_stats(out, ctrl0, grips, limit),
                      "demo": SR.step_stats(c, ctrl0, grips, limit), "out": a.out,
                      "sha256": R.file_sha(a.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
