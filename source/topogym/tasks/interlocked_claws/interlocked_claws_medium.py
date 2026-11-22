"""interlocked_claws: orthogonal_gate_balanced_v2. Difficulty is a design target, not a qualification."""

from .common import InterlockedClawsTask


class InterlockedClawsMediumTask(InterlockedClawsTask):
    # pinned report files exempt from the frozen-identity comparison (rcb_topogym Session: not compared)
    frozen_identity_exempt = ("geometry_report.json",)
    variation_id = 'interlocked_claws_medium'
    family_id = 'acute_windows_opposed'
    model_revision = 'opposed_pregrasp_wide2_v1'
    difficulty = 'medium'
    # Preserve each piece's acquired pose relative to its pregrasped hand.
    pregrasp_weld = True

    geometry_family = 'acute_windows_opposed'
    # Both gates +2 mm per side.
    gate_extra_m = (.002, .002)
