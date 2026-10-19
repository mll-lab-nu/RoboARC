"""Replay an action stream (an old -i01 oracle, a breaker) open loop on a fiber scene in the production Session and
report how close it got: success, the episode end, how many strands passed, the front, the lay and the routes (the
redesign task's own terms). Run it through run_registered.py for a redesign instance.

usage: oldreplay.py <instance.json> <actions.npz> <out.json> [--hold N]   (N rows of the last row after the stream)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from rcb_topogym import Session

ap = argparse.ArgumentParser()
ap.add_argument("instance"); ap.add_argument("actions"); ap.add_argument("out")
ap.add_argument("--hold", type=int, default=0)
a = ap.parse_args()
s = Session("fiber_bundling", instance=a.instance, strict=True)
rows = np.asarray(np.load(a.actions)["actions"], np.float32)
rows = np.concatenate([rows, np.repeat(rows[-1:], a.hold, axis=0)])
n = 0
for row in rows:
    n += 1
    if s.step(row):
        break
st = s.status()
m = dict(getattr(s.env._task.result, "metrics", None) or {})
done = m.get("strands_complete") or []
res = dict(instance=a.instance, actions=a.actions, rows=n, success=bool(st["success"]), episode_end=st["episode_end"],
           state_sha=st["state_sha"], strands_complete=int(sum(bool(x) for x in done)), strands=len(done),
           front_sheath_clear=m.get("front_sheath_clear"), maximum_bend_deg=m.get("maximum_bend_deg"),
           passage_reached=m.get("passage_reached"), lay=m.get("lay"), routes=m.get("routes"))
Path(a.out).write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, default=str))
