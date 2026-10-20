"""Export what the runtime needs of each redesigned claws tier (the pieces, their start pose, the birth hands) from the
tier factories (topogym/planning/interlocked_claws/redesign/tiers.py) into
topogym/assets/resources/variations/interlocked_claws/rd_<tier>.json. Values keep their exact float bits and their
Python / numpy scalar kind (the pieces' URDF writes repr() of box sizes). No moves, no route summary, no layouts.

usage: python export_runtime.py [--check]   (--check: compare with the files in the package instead of writing)"""
import json
import os
import sys
from pathlib import Path

import numpy as np

from topogym.planning.interlocked_claws.redesign import tiers

OUT = Path(os.environ.get("CLAWS_EXPORT_DIR") or
           Path(__file__).resolve().parents[3] / "source/topogym/assets/resources/variations/interlocked_claws")


def scalar(x):
    if isinstance(x, np.floating):
        return {"np": float(x)}
    if isinstance(x, (float, int)) and not isinstance(x, bool):
        return x
    raise TypeError(f"unexpected scalar {type(x).__name__}")


def matrix(T):
    return np.asarray(T, float).tolist()


def piece(p):
    body = []
    for b in p.body:
        if set(b) != {"size", "pose"}:
            raise ValueError(f"{p.name}: box keys {sorted(b)}")
        pose = b["pose"]
        body.append(dict(size=[scalar(v) for v in b["size"]], pose=matrix(pose),
                         pose_kind="array" if isinstance(pose, np.ndarray) else "list"))
    return dict(name=p.name, body=body, grips={g: matrix(T) for g, T in p.grips.items()},
                grip_kind={g: ("array" if isinstance(T, np.ndarray) else "list") for g, T in p.grips.items()},
                color=[scalar(v) for v in p.color])


def record(tier):
    d = tiers.make(tier)
    return dict(schema="claws_rd_runtime/1", tier=tier, name=f"rd_{tier}", start=d.get("start"),
                hands=dict(d["hands"]), birth_flip=dict(d.get("birth_flip") or {}),
                birth_q=d.get("birth_q"), clearance=scalar(d["clearance"]),
                A_in_assembly=matrix(d["A_in_assembly"]), engaged=matrix(d["engaged"]),
                A=piece(d["A"]), B=piece(d["B"]))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    bad = 0
    for tier in ("very_easy", "easier", "easy", "medium", "hard"):
        text = json.dumps(record(tier), indent=1) + "\n"
        path = OUT / f"rd_{tier}.json"
        if "--check" in sys.argv:
            same = path.is_file() and path.read_text() == text
            bad += not same
            print(tier, "same" if same else "DIFFERS")
        else:
            path.write_text(text)
            print("wrote", path)
    sys.exit(1 if bad else 0)
