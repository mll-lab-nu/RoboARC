"""Fresh-process replay of a robot-oracle run's rows (knot-robot): the Session of <run>/session_spec.json (the frozen
instance, or the dev rebuild), the rows of <run>/privileged.npz stepped open-loop through Session.step (privileged
mode: joint clip, slew, table guard, as recorded), with the recorded tick ledger; then the final state hash and
success, against the live run's (<run>/oracle.json).

  python replay_check.py <run dir> [--out <run>/replay_<n>.json] [--hold N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hold", type=int, default=0, help="hold rows after the stream (if the episode has not ended)")
    ap.add_argument("--partial", action="store_true", help="a run still going: replay trajectory.partial.npz and check "
                    "the intermediate state hashes the live run logged so far (oracle.json, partial)")
    a = ap.parse_args(argv)
    run = Path(a.run)
    from run_robot_oracle import session_from_spec
    spec = json.loads((run / "session_spec.json").read_text())
    live = json.loads((run / "oracle.json").read_text()) if (run / "oracle.json").is_file() else {}
    npz = run / ("trajectory.partial.npz" if a.partial else "privileged.npz")
    z = np.load(npz, allow_pickle=False)
    rows = np.asarray(z["actions"], np.float32)
    ticks = np.asarray(z["physics_steps_per_action"]) if "physics_steps_per_action" in z.files else None
    t0 = time.time()
    s = session_from_spec(spec)
    build_s = time.time() - t0
    if list(z["action_layout"]) != list(s.action_layout):
        raise SystemExit("action layout differs")
    want = {int(r): h for r, h in (live.get("state_hashes") or [])}
    got = {}
    t1 = time.time()
    ended_at = None
    for i, row in enumerate(rows):
        ended = s.step(row, _physics_steps=None if ticks is None else int(ticks[i]))
        if s.n_steps in want:
            got[s.n_steps] = s.state_sha()
        if ended:
            ended_at = i
            break
    held = 0
    if ended_at is None:
        for _ in range(a.hold):
            held += 1
            if s.step(s.hold_action()):
                break
    wall = time.time() - t1
    st = s.status()
    lst = live.get("status") or {}
    res = dict(run=str(run), spec=spec, rows_file=npz.name, partial=bool(a.partial), rows=int(len(rows)), executed=int(s.n_steps), ended_at_row=ended_at, held=held,
               build_s=round(build_s, 1), wall_s=round(wall, 1), ms_per_row=round(1000 * wall / max(1, s.n_steps), 1),
               rows_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(), instance_sha=s.instance_sha,
               success=st["success"], episode_end=st["episode_end"], state_sha=st["state_sha"],
               live_state_sha=lst.get("state_sha"), live_success=lst.get("success"),
               same_state=st["state_sha"] == lst.get("state_sha"), review=st.get("review"), forcing=st.get("forcing"),
               intermediate=dict(checked=len(got), equal=sum(got[r] == want[r] for r in got),
                                 first_mismatch=next(([r, want[r], got[r]] for r in sorted(got) if got[r] != want[r]), None)))
    s.close()
    out = Path(a.out) if a.out else run / f"replay_{int(time.time())}.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, default=str))
    os.replace(tmp, out)
    print("REPLAY", json.dumps({k: res[k] for k in ("executed", "ended_at_row", "success", "episode_end", "state_sha",
                                                     "live_state_sha", "same_state", "intermediate", "wall_s")}), flush=True)


if __name__ == "__main__":
    main()
