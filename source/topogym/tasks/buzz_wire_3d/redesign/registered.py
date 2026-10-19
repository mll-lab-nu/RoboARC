"""The registered buzz_wire_3d redesign tier classes (agent "buzz", 2026-10-10/11): the implementation module of their
variation records (topogym/config/variation_records.py). One class per tier, built by task.tier_class from its design
factory in tiers.TIERS, as scripts/robotangle/redesign/buzz_wire_3d/rdreg.py built them in-process (into a module of
this name) for the frozen instances and the acceptance rounds: same class names, same module, same values.
"""
from __future__ import annotations

from .task import tier_class
from .tiers import TIERS

CLASSES = {}
for _tier, (_vid, _factory) in TIERS.items():
    _cls = tier_class(_vid, _tier, _factory)
    _cls.__module__ = __name__
    globals()[_cls.__name__] = _cls
    CLASSES[_tier] = _cls
del _tier, _vid, _factory, _cls

