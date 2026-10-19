"""blanket_fold easy (ladder B, lab tier b_easy): flatten easier4's start, then the half fold."""
from .common import BlanketFoldTask, tier_defaults


class BlanketFoldEasyTask(BlanketFoldTask):
    variation_id = "blanket_fold_easy"
    difficulty = "easy"
    defaults = tier_defaults("b_easy")
