"""Product registration of the redesigned buzz_wire_3d tiers (agent "buzz", 2026-10-10/11): the variation records
topogym/config/variation_records.py appends, as scripts/robotangle/redesign/buzz_wire_3d/rdreg.py appended them in-process
for the frozen instances and the acceptance rounds. Literal values (no import of the tier classes, whose module builds
every design): the classes are task.tier_class's (BuzzWireRedesign_<tier>, model_revision redesign_v0) in registered.py.
The family's MuJoCo profile serves them (no entry of their own).
"""
from __future__ import annotations

TIERS = ("very_easy", "easier", "easy", "medium", "hard")
IMPLEMENTATION = "topogym.tasks.buzz_wire_3d.redesign.registered:BuzzWireRedesign_{tier}"


def variation_id(tier: str) -> str:
    return f"buzz_wire_3d_rd_{tier}"


def variation_records() -> list:
    """The variation records (topogym.config.variation_records format) of the five tiers."""
    return [dict(task_id="buzz_wire_3d", variation_id=variation_id(t), difficulty=t, family_id="redesign_2026_10",
                 model_revision="redesign_v0", implementation=IMPLEMENTATION.format(tier=t), aliases=[],
                 status="integrated", reason=None, validation="redesign_dev", pool="redesign")
            for t in TIERS]
