"""Hanayama Cast Rotor: separate two identical interlocked pieces (topogym.assets.geometry.claws_rotor). Both pieces start
in the hands; the goal is the family's whole-piece separation, held quiet."""
from .common import InterlockedClawsTask


class InterlockedClawsRotorTask(InterlockedClawsTask):
    # pinned report files exempt from the frozen-identity comparison (rcb_topogym Session: not compared)
    frozen_identity_exempt = ("geometry_report.json",)
    variation_id = "interlocked_claws_rotor"
    family_id = "hanayama_cast_rotor"
    model_revision = "rotor_v12f_r100mm"
    difficulty = "very_hard"
    pregrasp_weld = True
    # Each piece's grasp region covers the piece (pregrasp_targets), and a hand may acquire with its TCP up to 15 cm from
    # the region's centre.
    grasp_max_tcp_distance_m = .15
    placement_description = "the entire brown piece"
    separation_summary = "Completely separate the two interlocked pieces and briefly hold both still."

    def build_assets(self, folder):
        from topogym.assets.geometry.claws_rotor import build_claws_rotor
        assembly, report, self.initial_robot_q = build_claws_rotor(folder)
        return assembly, report

    def pregrasp_targets(self, assembly, entities, transformed):
        """The family's targets (each hand on a different piece, the one nearest its TCP), each centred on its piece's
        centre with a radius of 11.5 cm: the whole piece (its farthest point, a horn tip, is 111.3 mm from the centre)."""
        return tuple(dict(t, center_m=[0., 0., 0.], radius_m=.115)
                     for t in super().pregrasp_targets(assembly, entities, transformed))
