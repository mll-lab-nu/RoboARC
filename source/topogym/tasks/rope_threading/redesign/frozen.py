"""The redesigned rope_threading tiers as the runtime builds them: each tier's rings (in the entity frame), fixtures,
required signed order, start centre line and rope length, read from frozen data
(topogym/assets/resources/variations/rope_threading/rd_<tier>.json, exported from the authoring scene module by
scripts/robotangle/redesign/rope_threading/export_runtime.py: the same values, bit for bit)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROPE_RADIUS_M = 0.004
TASK_ROOT_WORLD = np.array([-0.1356786, 0.0, 0.0])


def load(tier):
    from topogym.assets.library import resource_path
    rec = json.loads(Path(resource_path(f"variations/rope_threading/rd_{tier}.json")).read_text())
    if rec.get("schema") != "rope_threading_rd_runtime/1" or rec.get("tier") != tier:
        raise ValueError(f"rope_threading rd {tier}: not a runtime design record")
    obstacles = []
    for o in rec["obstacles"]:
        if o["kind"] == "box":
            obstacles.append(SimpleNamespace(kind="box", name=o["name"], center=np.asarray(o["center"], float),
                                             half=np.asarray(o["half"], float), R=np.asarray(o["R"], float),
                                             rgba=tuple(o["rgba"])))
        else:
            obstacles.append(SimpleNamespace(kind="capsule", name=o["name"], a=np.asarray(o["a"], float),
                                             b=np.asarray(o["b"], float), radius=o["radius"], rgba=tuple(o["rgba"])))
    return SimpleNamespace(tier=tier, rings=rec["rings"], required_crossings=[tuple(x) for x in rec["required_crossings"]],
                           start=np.asarray(rec["start"], float), rope_length=rec["rope_length"], obstacles=obstacles)
