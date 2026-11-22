"""blanket_fold easier (ladder B, lab tier b_easier): the half fold, then the top layer's far-left corner dog-eared onto a named point."""
from .common import BlanketFoldTask, tier_defaults


class BlanketFoldEasierTask(BlanketFoldTask):
    variation_id = "blanket_fold_easier"
    difficulty = "easier"
    defaults = tier_defaults("b_easier")
