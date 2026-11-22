"""blanket_fold hard (ladder B, lab tier b_hard): flatten hard4's start (fence), the half fold, then one top-layer dog-ear."""
from .common import BlanketFoldTask, tier_defaults


class BlanketFoldHardTask(BlanketFoldTask):
    variation_id = "blanket_fold_hard"
    difficulty = "hard"
    defaults = tier_defaults("b_hard")
