"""Re-record a reference on the current physics (any MuJoCo family): execute its rows, then N hold rows of its last row,
in a fresh Session until the episode ends; record every executed row and its tick ledger and save them as the new
privileged reference trajectory (make_reference.py's recording, for any instance).

Run it with the RoboTangle image's Python, as the stress tools (scripts/robotangle/stress/_env.py: this checkout's
topogym and tooling first on sys.path, the task containers' numpy / OpenBLAS SIMD masking, one thread per library).

usage: rerecord.py <instance.json> <reference.npz> <out.npz> [--tail N] [--note TEXT]
The interlocked_claws -i01 references (docs/rcb/families/interlocked_claws.md): very_easy and easier from the released
oracle scripts/robotangle/oracles/interlocked-claws-<tier>-i00/privileged.npz directly; easy, medium and hard from that
oracle with its wrist dither notched out (notch_ref.py --length <v5 rows>)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stress"))
import _env  # noqa: F401,E402  (before numpy: SIMD masking, threads, this checkout first on sys.path)
import argparse, hashlib, json, os, time  # noqa: E402
import numpy as np  # noqa: E402
from rcb_topogym import Session  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("instance"); ap.add_argument("reference"); ap.add_argument("out")
ap.add_argument("--tail", type=int, default=400); ap.add_argument("--note", default="")
a = ap.parse_args()
s = Session(json.loads(Path(a.instance).read_text())["task"], instance=a.instance, strict=True)
birth = s.status()["state_sha"]
ref = np.load(a.reference)
rows = ref["actions"].astype(np.float32)
if "action_layout" in ref.files:
    assert ref["action_layout"].tolist() == s.action_layout
n_ref = len(rows)
rows = np.concatenate([rows, np.repeat(rows[-1:], a.tail, axis=0)])
t0 = time.time()
s.record_start()
for row in rows:
    if s.step(row):
        break
acts = s.record_stop()
st = s.status()
meta = dict(source="mujoco reference re-recorded on the exact constraint diagonal (core agent)", route=str(a.reference),
            route_sha256=hashlib.sha256(Path(a.reference).read_bytes()).hexdigest(), route_rows=n_ref, tail_rows=a.tail,
            note=a.note, wall_s=round(time.time() - t0, 1), success=st["success"], episode_end=st["episode_end"])
s.save_trajectory(acts, a.out, meta=meta)
print(json.dumps(dict(instance=a.instance, birth_sha=birth, rows=int(len(acts)), route_rows=n_ref, success=st["success"],
                      episode_end=st["episode_end"], review=st["review"], forcing=st["forcing"], state_sha=st["state_sha"],
                      out=a.out, sha256=hashlib.sha256(Path(a.out).read_bytes()).hexdigest())))
s.close()
