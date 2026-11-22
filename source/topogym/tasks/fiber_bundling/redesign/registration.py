"""Product registration of the redesigned fiber_bundling tiers (formboard; agent "fiber", 2026-10-10/11): the variation
records topogym/config/variation_records.py appends, as the fiber lane's build.py appended them in-process
(session_kit.register_variation) for the frozen instances and the acceptance rounds. Literal values (no import of the
task module): the classes are tasks._tier_class's (FiberBundlingRedesign<Tier>Task, family_id redesign_<tier>,
model_revision fiber_redesign_v0). Their assisted grasp is common.recommended_grasp_assist's (the released sheath grasp,
which build.py passed as a make-instance override). The family's MuJoCo profile serves them (no entry of their own).
"""
from __future__ import annotations

TIERS = ("very_easy", "easier", "easy", "medium", "hard")
IMPLEMENTATION = "topogym.tasks.fiber_bundling.redesign.tasks:FiberBundlingRedesign{name}Task"


def variation_id(tier: str) -> str:
    return f"fiber_bundling_rd_{tier}"


def variation_records() -> list:
    """The variation records (topogym.config.variation_records format) of the five tiers."""
    return [dict(task_id="fiber_bundling", variation_id=variation_id(t), difficulty=t, family_id=f"redesign_{t}",
                 model_revision="fiber_redesign_v0",
                 implementation=IMPLEMENTATION.format(name="".join(w.capitalize() for w in t.split("_"))),
                 aliases=[], status="integrated", reason=None, validation="redesign_dev", pool="redesign")
            for t in TIERS]
