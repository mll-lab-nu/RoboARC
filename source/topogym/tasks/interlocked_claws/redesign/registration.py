"""Product registration of the redesigned interlocked_claws tiers (keyhole gates; agent "claws", 2026-10-10/11): the
variation records topogym/config/variation_records.py appends, as task.register() appended them in-process (the
overlay scripts/robotangle/redesign/interlocked_claws/overlay/sitecustomize.py) for the frozen instances and the
acceptance rounds. Literal values (no import of the task module): the class names are task._make's, family_id and
model_revision RedesignClawsTask's. The family's MuJoCo profile serves them (no entry of their own).
"""
from __future__ import annotations

TIERS = ("very_easy", "easier", "easy", "medium", "hard")
IMPLEMENTATION = "topogym.tasks.interlocked_claws.redesign.task:RedesignClaws{name}"


def variation_id(tier: str) -> str:
    return f"interlocked_claws_rd_{tier}"


def variation_records() -> list:
    """The variation records (topogym.config.variation_records format) of the five tiers."""
    return [dict(task_id="interlocked_claws", variation_id=variation_id(t), difficulty=t,
                 family_id="redesign_keyhole_gates", model_revision="rd_v1",
                 implementation=IMPLEMENTATION.format(name="".join(w.capitalize() for w in t.split("_"))),
                 aliases=[], status="integrated", reason=None, validation="redesign_dev", pool="redesign")
            for t in TIERS]
