"""Replay a claws row stream in a MuJoCo Session built from (variation, seed) and record it with antihack.record.

The released -i01 references were recorded on instances made from (variation, seed 0, native gripper); a Session built
the same way starts from the same birth (its state hash is checked against the file's instance_sha), so no frozen
instance is needed for the strict references.

usage (locally: source tools/lenv.sh; elsewhere: the redesign env):
  python strict_replay.py --rows <privileged.npz> --out <rec.npz> [--variation V] [--every 10] [--hold 0]
  python strict_replay.py --rows nop:400 --variation interlocked_claws_easy --out <rec.npz>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variation", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--hold", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    a = ap.parse_args(argv)
    from antihack import record
    from rcb_topogym import Session
    variation, expect = a.variation, None
    if not a.rows.startswith("nop:"):
        z = np.load(a.rows, allow_pickle=False)
        variation = variation or str(z["variation_id"])
        expect = str(z["instance_sha"]) if "instance_sha" in z.files else None
    from topogym.tasks.interlocked_claws.redesign.task import VARIATIONS, register
    for tier, vid in VARIATIONS.items():          # the redesign tiers are registered in-process
        if vid == variation:
            register(tier)
    t0 = time.time()
    s = Session("interlocked_claws", variation_id=variation, seed=a.seed, gripper_space="native", strict=False)
    build_s = time.time() - t0
    birth = s.state_sha()
    if expect and birth != expect:
        raise SystemExit(f"birth {birth} != the rows' instance_sha {expect}")
    rows, _ = record.load_rows(a.rows, s)
    if a.max_rows:
        rows = rows[:a.max_rows]
    out = record.run(s, rows, a.out, every=a.every, hold=a.hold, label=Path(a.rows).parent.name,
                     extra=dict(rows_file=str(a.rows), variation=variation, birth_sha=birth, build_s=round(build_s, 1)))
    print(json.dumps({k: v for k, v in out.items() if k not in ("metrics_samples", "grasp_events")}, default=str)[:3000])
    s.close()


if __name__ == "__main__":
    main()
