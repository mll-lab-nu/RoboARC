"""Export what the runtime needs of each redesigned rope_threading tier (rings in the entity frame, fixtures, required
signed order, start centre line, rope length) from the authoring scene module
(topogym/planning/rope_threading/redesign/scenes.py) into
topogym/assets/resources/variations/rope_threading/rd_<tier>.json. Exact float bits; numpy scalars are refused (the
runtime reads Python floats). No hand plans, design notes, station constants or goal arc lengths.

usage: python export_runtime.py [--check]"""
import json
import os
import sys
from pathlib import Path

import numpy as np

from topogym.planning.rope_threading.redesign import scenes as S

OUT = Path(os.environ.get("ROPE_EXPORT_DIR") or
           Path(__file__).resolve().parents[4] / "topogym/assets/resources/variations/rope_threading")


def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        raise TypeError(f"numpy scalar {x!r}")
    return x


def record(tier):
    d = S.build(tier)
    rings = [{k: v for k, v in r.as_layout(world=True).items() if k != "goal_arc_length_m"}
             for r in sorted(d.rings, key=lambda r: r.s_goal)]
    obstacles = []
    for o in d.obstacles:
        if isinstance(o, S.Box):
            obstacles.append(dict(kind="box", name=o.name, center=o.center, half=o.half, R=o.R(), rgba=list(o.rgba)))
        else:
            obstacles.append(dict(kind="capsule", name=o.name, a=o.a, b=o.b, radius=o.radius, rgba=list(o.rgba)))
    assert S.ROPE_RADIUS_M == 0.004 and np.array_equal(S.TASK_ROOT_WORLD, [-0.1356786, 0.0, 0.0])
    return plain(dict(schema="rope_threading_rd_runtime/1", tier=tier, rings=rings,
                      required_crossings=[list(x) for x in d.required_crossings()], start=d.start,
                      rope_length=float(d.rope_length) if isinstance(d.rope_length, np.generic) else d.rope_length,
                      obstacles=obstacles))


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
