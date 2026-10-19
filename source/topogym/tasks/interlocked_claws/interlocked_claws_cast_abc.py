"""Hanayama Cast ABC: free the C ring from the AB plate (topogym.assets.geometry.cast_abc). Both pieces start in the
hands; the goal is the family's whole-piece separation, held quiet."""
from .common import InterlockedClawsTask


class InterlockedClawsCastABCTask(InterlockedClawsTask):
    # pinned report files exempt from the frozen-identity comparison (rcb_topogym Session: not compared)
    frozen_identity_exempt = ("geometry_report.json",)
    variation_id = "interlocked_claws_cast_abc"
    family_id = "hanayama_cast_abc"
    model_revision = "cast_abc_v4_w10mm"
    difficulty = None
    pregrasp_weld = True
    placement_description = "the entire C ring"
    separation_summary = "Completely separate the C ring from the AB plate and briefly hold both still."

    def build_assets(self, folder):
        from topogym.assets.geometry.cast_abc import build_cast_abc
        assembly, report, self.initial_robot_q = build_cast_abc(folder)
        return assembly, report
