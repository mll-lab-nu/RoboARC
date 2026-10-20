#!/usr/bin/env python
"""clock_assembling's task mechanism through a Session (topogym/tasks/clock_assembling/mechanism.py):

  1. the first rows of the -i01 reference (the barrel seated by row 1100) end on a state hash; reset()
     restores the birth (the mechanism's state too: nothing seated, the gates open) and the same rows end on the same hash,
     with the same rules' state;
  2. the birth targets held (a nop) seat nothing and do not succeed.

  python scripts/robotangle/test_clock_assembling.py --instance <instance.json> \
      [--reference scripts/robotangle/oracles/clock-assembling-01-i01/privileged.npz] [--rows 1100]
Run it where make-instance runs (the RoboTangle MuJoCo image, the SIMD masking, rendering off). Exit 0 when every check
holds; prints one JSON line."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "shared"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--reference", default="scripts/robotangle/oracles/clock-assembling-01-i01/privileged.npz")
    ap.add_argument("--rows", type=int, default=1100)
    a = ap.parse_args()
    from rcb_topogym import Session, load_trajectory
    s = Session("clock_assembling", instance=a.instance, strict=True)
    birth = s.state_sha()
    rows, _ = load_trajectory(a.reference, "clock_assembling", s.action_dim, trusted=True, layout=s.action_layout)
    rows = rows[:a.rows]
    read = lambda: dict(s.env._world.read_state()["mechanism"])      # noqa: E731
    out, ok = {}, True
    runs = []
    for k in range(2):
        for row in rows:
            s.step(row)
        runs.append((s.state_sha(), read()))
        s.reset()
        after = read()
        mech = s.env._world._mechanism
        out[f"reset_{k}_birth"] = s.state_sha() == birth
        out[f"reset_{k}_nothing_seated"] = not any(v for n, v in after.items() if n.endswith(("_seated", "_meshed", "_gate_closed", "_tight")))
        out[f"reset_{k}_click_open"] = bool(not after["fixed"] and mech.m.jnt_range[mech.j_click, 0] == mech.click_range0)
    out["same_hash"] = runs[0][0] == runs[1][0]
    out["same_rules"] = runs[0][1] == runs[1][1]
    out["seated_after_rows"] = [n for n, v in runs[0][1].items() if n.endswith("_seated") and v]
    for _ in range(len(rows)):
        s.step(s.hold_action())
    nop = read()
    out["nop_seats_nothing"] = not any(v for n, v in nop.items() if n.endswith("_seated"))
    out["nop_no_success"] = not s.status()["success"]
    ok = all(v for k, v in out.items() if k != "seated_after_rows") and len(out["seated_after_rows"]) >= 1
    print(json.dumps(dict(ok=ok, rows=len(rows), hash=runs[0][0], **out)), flush=True)
    s.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
