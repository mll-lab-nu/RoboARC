"""hedgehog_in_cage: classic_3star. Difficulty is a design target, not a qualification."""

from topogym.tasks.base import merge_parameters
from .common import HedgehogInCageTask


class HedgehogInCageEasyTask(HedgehogInCageTask):
    variation_id = 'hedgehog_in_cage_easy'
    family_id = 'classic_3star'
    model_revision = 'simulator_v1'
    difficulty = 'easy'
    fixed_parameters = ('scale',)
    # The cage's and the ball's finishes at appearance seeds 1 and 2 (common.FINISHES); the defaults name the released
    # dark and light wood.
    defaults = merge_parameters(HedgehogInCageTask.defaults, dict(cage_finish="dark_wood", ball_finish="light_wood"))
    hash_when_changed = (*HedgehogInCageTask.hash_when_changed, "cage_finish", "ball_finish")
    finish_bodies = {"cage_finish": "cage", "ball_finish": "hedgehog"}
