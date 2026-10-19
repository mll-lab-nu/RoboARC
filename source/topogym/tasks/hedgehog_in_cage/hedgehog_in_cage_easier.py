"""hedgehog_in_cage: below_easy, easier. 108 mm windows."""

from topogym.tasks.base import merge_parameters
from .below_easy import HedgehogBelowEasyTask, cage


class HedgehogInCageEasierTask(HedgehogBelowEasyTask):
    variant_parameters = {"cage": cage(108.)}
    defaults = merge_parameters(HedgehogBelowEasyTask.defaults, variant_parameters)
    variation_id = 'hedgehog_in_cage_easier'
    difficulty = 'easier'
