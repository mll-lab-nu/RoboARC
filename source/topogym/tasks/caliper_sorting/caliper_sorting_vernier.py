"""The design's setting: four sleeves of two classes, the long vernier in the caliper's own unit, the slider's locking
screw (CaliperSortingTask as it is)."""
from .common import CaliperSortingTask


class CaliperSortingVernierTask(CaliperSortingTask):
    variation_id = "caliper_sorting_vernier"
