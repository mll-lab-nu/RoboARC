"""blanket_fold very_easy (ladder B, lab tier b_very_easy): the half fold of the flat sheet."""
from .common import BlanketFoldTask, tier_defaults


class BlanketFoldVeryEasyTask(BlanketFoldTask):
    variation_id = "blanket_fold_very_easy"
    difficulty = "very_easy"
    defaults = tier_defaults("b_very_easy")
