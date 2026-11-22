"""Product registration of the redesigned knot_untying tiers (variation records), without importing the task module.

The five tiers (very_easy, easier, easy, medium, hard) are one task class, task_rd.KnotUntyingRedesignTask, with the
scene in task_parameters["redesign"]; the per-tier classes are generated in task_rd. Their MuJoCo profile is the family's
(profiles.profile_for falls back to ("knot_untying", None), knot_chain_v7). task_rd.register() (the in-process
registration the redesign scripts call) finds these rows and adds nothing.
"""
from __future__ import annotations

TIERS = ("very_easy", "easier", "easy", "medium", "hard")
FAMILY_ID = "knot_untying_redesign"
MODEL_REVISION = "knot_untying_rd_v1_mujoco_chain"      # task_rd.KnotUntyingRedesignTask.model_revision
IMPLEMENTATION = "topogym.tasks.knot_untying.redesign.task_rd:{cls}"


def variation_id(tier: str) -> str:
    return f"knot_untying_rd_{tier}"


def class_name(tier: str) -> str:
    return "KnotUntyingRd" + "".join(w.capitalize() for w in tier.split("_")) + "Task"


def variation_records() -> list:
    """The variation records (topogym.config.variation_records format) of the five tiers."""
    return [dict(task_id="knot_untying", variation_id=variation_id(t), difficulty=t, family_id=FAMILY_ID,
                 model_revision=MODEL_REVISION, implementation=IMPLEMENTATION.format(cls=class_name(t)), aliases=[],
                 status="integrated", reason=None, validation="redesign_dev", pool="redesign") for t in TIERS]
