"""Shrink a failing action stream to a short reproducer that still fails the same way: in-process episodes from the
frozen birth. Confirm the result afterwards in two fresh processes with replay.py (same verdict, same state hash).

usage: shrink.py --instance <instance.json> --actions <cand.npz> --match <substring of a fail reason> --out <min.npz>
                 [--max-tests 150] [--experiment-opt ...]
Steps, each kept only if the stream still fails with --match:
  1. the shortest prefix (binary search) followed by 40 controls holding its last row
  2. segments (runs of equal rows) removed, largest first, then halves of the rest (ddmin-like)
  3. each arm's columns (and each gripper) replaced by the birth targets for the whole stream
  4. each segment shortened to half (repeatedly)
Writes <min.npz> (actions) and <min>.json (the steps, test count, the final verdict)."""
import _env  # noqa: F401  (before numpy: SIMD masking, threads, sys.path)
import argparse, hashlib, json, os
from pathlib import Path

import numpy as np

from search import Search   # noqa: E402

HOLD = 40


def segments(rows):
    out, start = [], 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or not np.array_equal(rows[i], rows[start]):
            out.append((start, i))
            start = i
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--actions", required=True)
    ap.add_argument("--match", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tests", type=int, default=150)
    ap.add_argument("--experiment-opt", default=os.environ.get("STRESS_EXPERIMENT_OPT"))
    a = ap.parse_args()
    S = Search(a.instance, 0, experiment=a.experiment_opt)
    rows = np.load(a.actions)["actions"].astype(np.float64)
    tests, log = [0], []

    def fails(cand):
        if tests[0] >= a.max_tests:
            return False
        tests[0] += 1
        cand = np.asarray(cand, dtype=np.float64)
        rec, _, _ = S.run(list(cand) + [cand[-1]] * HOLD)
        ok = any(a.match in f for f in rec["fails"])
        log.append(dict(test=tests[0], rows=int(len(cand)), fails=rec["fails"], ok=ok))
        return ok

    if not fails(rows):
        raise SystemExit(f"the input does not fail with {a.match!r}: {log[-1]}")
    # 1. shortest prefix
    lo, hi = 1, len(rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if fails(rows[:mid]):
            hi = mid
        else:
            lo = mid + 1
    rows = rows[:hi]
    steps = [dict(step="prefix", rows=int(len(rows)), tests=tests[0])]
    # 2. remove segments, largest first
    changed = True
    while changed and tests[0] < a.max_tests:
        changed = False
        segs = sorted(segments(rows), key=lambda s: s[0] - s[1])
        for s0, s1 in segs:
            if len(segments(rows)) <= 1:
                break
            cand = np.delete(rows, np.s_[s0:s1], axis=0)
            if len(cand) and fails(cand):
                rows, changed = cand, True
                break
    steps.append(dict(step="segments", rows=int(len(rows)), segments=len(segments(rows)), tests=tests[0]))
    # 3. columns back to the birth targets (per arm, per gripper)
    groups = [(f"{arm} arm", list(range(S.arm_slice[arm].start, S.arm_slice[arm].stop))) for arm in S.arms] + \
             [(f"{arm} gripper", [S.grip[arm]]) for arm in S.arms]
    for name, cols in groups:
        cand = rows.copy()
        cand[:, cols] = S.hold0[cols]
        if not np.array_equal(cand, rows) and fails(cand):
            rows = cand
            steps.append(dict(step=f"birth targets: {name}", tests=tests[0]))
    # 4. shorten segments
    changed = True
    while changed and tests[0] < a.max_tests:
        changed = False
        for s0, s1 in segments(rows):
            if s1 - s0 >= 2:
                cand = np.delete(rows, np.s_[s0:s0 + (s1 - s0) // 2], axis=0)
                if fails(cand):
                    rows, changed = cand, True
                    break
    steps.append(dict(step="shorten", rows=int(len(rows)), segments=len(segments(rows)), tests=tests[0]))
    final = np.asarray(list(rows) + [rows[-1]] * HOLD, dtype=np.float32)
    rec, _, summ = S.run(list(final))
    np.savez(a.out, actions=final)
    meta = dict(source=a.actions, instance=a.instance, match=a.match, rows=int(len(final)), core_rows=int(len(rows)),
                hold_rows=HOLD, sha256=hashlib.sha256(Path(a.out).read_bytes()).hexdigest(), tests=tests[0], steps=steps,
                final=rec, log=log[-30:])
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps(dict(out=a.out, rows=len(final), core_rows=len(rows), tests=tests[0], fails=rec["fails"],
                          sha256=meta["sha256"])))


if __name__ == "__main__":
    main()
