"""clock_assembling's first setting: the barrel, the center, third and fourth wheels, one train bridge and two screws
(the design's v3 demo, runs/watch_assembly_design_20261014/README_v3.md)."""
from .common import ClockAssemblingTask


class ClockAssemblingGearTrainTask(ClockAssemblingTask):
    variation_id = "clock_assembling_gear_train"
    family_id = "clock_gear_train"
    model_revision = "clock_gear_train_v2"
    difficulty = None
