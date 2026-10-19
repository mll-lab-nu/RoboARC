"""Fresh-process replay of rows on a redesign tier: python work/replay_check.py <tier> <rows.npz | nop:N> [--hold N]
Prints the birth sha, success, the first success row, episode end, final state sha, review / forcing events."""
import json, sys, time
import numpy as np
from topogym.tasks.interlocked_claws.redesign.task import make_session
tier, src = sys.argv[1], sys.argv[2]
hold = int(sys.argv[sys.argv.index("--hold") + 1]) if "--hold" in sys.argv else 0
t0 = time.time()
s = make_session(tier)
birth = s.state_sha()
if src.startswith("nop:"):
    rows = np.repeat(s.hold_action()[None], int(src[4:]), 0)
    expect = None
else:
    z = np.load(src, allow_pickle=False)
    rows, expect = z["actions"], str(z["instance_sha"]) if "instance_sha" in z.files else None
    if list(z["action_layout"]) != list(s.action_layout):
        raise SystemExit("action layout differs")
first = None
for k, r in enumerate(rows, 1):
    s.step(np.asarray(r, np.float32))
    if first is None and s.success():
        first = k
    if s.status()["episode_end"] and s.status()["episode_end"] != "success":
        break
for _ in range(hold):
    s.step(s.hold_action())
st = s.status()
print("RESULT", json.dumps(dict(tier=tier, rows_file=src, birth=birth, expected_birth=expect, rows=len(rows), steps=k,
                                success=st["success"], first_success_row=first, episode_end=st["episode_end"],
                                final_sha=st["state_sha"], review=st["review"], forcing=st["forcing"],
                                wall_s=round(time.time() - t0, 1)), default=str))
