"""In-process registration of the buzz_wire_3d redesign tiers (no registry edit): one variation record per tier, its
task class built by topogym.tasks.buzz_wire_3d.redesign.task.tier_class. Import before building a Session."""
from __future__ import annotations

import sys
import types

from topogym.tasks.buzz_wire_3d.redesign import task as T
from topogym.tasks.buzz_wire_3d.redesign.tiers import TIERS

MODULE = "topogym.tasks.buzz_wire_3d.redesign.registered"
_mod = sys.modules.get(MODULE) or types.ModuleType(MODULE)
sys.modules[MODULE] = _mod


def register(tiers=None):
    from topogym.config import variation_records
    out = {}
    for tier, (vid, factory) in TIERS.items():
        if tiers and tier not in tiers:
            continue
        cls = T.tier_class(vid, tier, factory)
        cls.__module__ = MODULE
        setattr(_mod, cls.__name__, cls)
        rec = dict(task_id="buzz_wire_3d", variation_id=vid, difficulty=tier, family_id="redesign_2026_10",
                   model_revision=cls.model_revision, implementation=f"{MODULE}:{cls.__name__}", aliases=[],
                   status="integrated", reason=None, validation="redesign_dev", pool="redesign")
        if not any(r["task_id"] == "buzz_wire_3d" and r["variation_id"] == vid for r in variation_records.RECORDS):
            variation_records.RECORDS = (*variation_records.RECORDS, rec)
        out[tier] = vid
    return out
