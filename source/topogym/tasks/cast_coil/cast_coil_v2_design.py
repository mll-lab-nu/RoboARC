"""cast_coil v2 design: any two pieces of the v2 schema through the task parameter ``design`` (a pieces record: cells,
pins, grasp cells, block centre), so a candidate from the design search can be built before it becomes a tier. Defaults to the v2 very_easy pieces. The record enters the instance identity (hash_when_changed is
empty), so every design is its own instance.
"""
from .cast_coil_v2_very_easy import CastCoilV2VeryEasyTask


class CastCoilV2DesignTask(CastCoilV2VeryEasyTask):
    variation_id = "cast_coil_v2_design"
    family_id = "lattice_design"
    difficulty = None
    defaults = dict(CastCoilV2VeryEasyTask.defaults, design=None)

    def solution(self):
        return self.config["design"] or super().solution()
