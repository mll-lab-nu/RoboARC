"""blanket_fold medium (ladder B, lab tier b_medium): flatten easier4's start, the half fold, then one top-layer dog-ear."""
from .common import BlanketFoldTask, tier_defaults


class BlanketFoldMediumTask(BlanketFoldTask):
    variation_id = "blanket_fold_medium"
    difficulty = "medium"
    defaults = tier_defaults("b_medium")
