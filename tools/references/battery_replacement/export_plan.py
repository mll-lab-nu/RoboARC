"""Write the battery_replacement reference's plan (plan.npz, plan.json) from the design pipeline's kinematic oracle
(runs/controller_battery_design_20261014/scripts/d3_oracle.py: out/demo/oracle_<tag>.pkl, 20 ms frames), resampled to the
40 Hz control steps the oracle runs at (joints interpolated, the other fields from the frame at or before the step).

plan.npz: Q (n, 12) both arms' planned joints (left 6, right 6); F (n, 2) the planned finger values (native, m);
STAGE (n,) index into plan.json's stages; HELD (n, 2) the part each hand holds in the plan (index into parts, -1 none);
P (n, 6, 7) every part's planned pose in the scene frame (position m, quaternion wxyz); TPLAN (n,) the plan's time (s).
plan.json: stages, parts, place (x0 m, y0 m, yaw deg: the scene frame in the ALOHA mount frame), polarity, source.
usage: export_plan.py <oracle pkl> <out dir>"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

PARTS = ("cover", "old_A", "old_B", "new_1", "new_2", "stick")
DT_PLAN, DT = .02, .025


def main(src, out):
    D = pickle.load(open(src, "rb"))
    fr = D["frames"]
    stages = []
    index = {}
    rows = dict(Q=[], F=[], STAGE=[], HELD=[], P=[], TPLAN=[])
    t = 0.
    while t <= fr[-1]["t"] + 1e-9:
        x = t / DT_PLAN
        i = min(int(x), len(fr) - 1)
        j = min(i + 1, len(fr) - 1)
        a = x - i
        q = np.r_[(1 - a) * fr[i]["qL"] + a * fr[j]["qL"], (1 - a) * fr[i]["qR"] + a * fr[j]["qR"]]
        st = fr[i]["stage"]
        if st not in index:
            index[st] = len(stages)
            stages.append(st)
        rows["Q"].append(q)
        rows["F"].append([fr[i]["fL"], fr[i]["fR"]])
        rows["STAGE"].append(index[st])
        rows["HELD"].append([PARTS.index(fr[i]["held"][h]) if fr[i]["held"][h] else -1 for h in "LR"])
        rows["P"].append([np.r_[fr[i]["parts"][p][:3, 3], R.from_matrix(fr[i]["parts"][p][:3, :3]).as_quat()[[3, 0, 1, 2]]]
                          for p in PARTS])
        rows["TPLAN"].append(fr[i]["t"])
        t += DT
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "plan.npz", Q=np.array(rows["Q"]), F=np.array(rows["F"]), STAGE=np.array(rows["STAGE"], np.int32),
                        HELD=np.array(rows["HELD"], np.int32), P=np.array(rows["P"]), TPLAN=np.array(rows["TPLAN"]))
    (out / "plan.json").write_text(json.dumps(dict(stages=stages, parts=PARTS, place=list(D["place"]), polarity=D["polarity"],
                                                   source=str(Path(src).name), frames=len(fr), steps=len(rows["Q"])), indent=1) + "\n")
    print(f"{len(rows['Q'])} control steps, {len(stages)} stages -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
