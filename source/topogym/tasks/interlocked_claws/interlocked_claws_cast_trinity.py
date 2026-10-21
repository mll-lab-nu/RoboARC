"""Hanayama Cast Trinity: separate three interlocked pieces (topogym.assets.geometry.cast_trinity). Two pieces start in
the hands, the third hangs on them; the goal is every pair separated by a plane, all three held quiet."""
from .common import InterlockedClawsTask


class InterlockedClawsCastTrinityTask(InterlockedClawsTask):
    variation_id = "interlocked_claws_cast_trinity"
    family_id = "hanayama_cast_trinity"
    model_revision = "cast_trinity_v12"
    terminal_revision = "trinity_pairwise_separation_v1"
    difficulty = None
    pregrasp_weld = True
    placement_description = "the entire gold piece (T)"
    separation_summary = "Completely separate all three pieces from one another and briefly hold them still."
    # Appearance only: the colour of the third piece, B (None: other_rgb, as for H), so that the three pieces differ.
    defaults = dict(InterlockedClawsTask.defaults, third_rgb=None)
    hash_when_changed = (*InterlockedClawsTask.hash_when_changed, "third_rgb")
    # The reference takes all three pieces (T again at its head, H and B where they hang): every piece is a grasp target
    # of both hands, centred on its bounding box with a radius of 9 cm (the farthest point, a corner of B, is 85.8 mm from
    # its box centre), and a hand may acquire with its TCP up to 12 cm from that centre.
    grasp_max_tcp_distance_m = .12

    @property
    def separation_rules(self):
        """The public goal's rules (goal_description.describe_goal): every two of the three pieces, not "both pieces"."""
        gap = f"{float(self.config.get('minimum_separation_m', .01)) * 1000:.6g} mm"
        return [f"Leave at least {gap} between every two of the three pieces, with a plane separating all of the two "
                "pieces' geometry; a gap inside an interlock is insufficient.",
                "Keep all three pieces' linear speed at most 1 mm/s and angular speed at most 0.02 rad/s. "
                "Holding pieces is allowed; release and transport into the green region are not required."]

    def build_assets(self, folder):
        from topogym.assets.geometry.cast_trinity import build_cast_trinity
        assembly, report, self.initial_robot_q = build_cast_trinity(folder)
        return assembly, report

    def other_color(self, index):
        """B (index 2; T, H, B in build order) takes third_rgb when it is set."""
        if index == 2 and self.config["third_rgb"] is not None:
            return self.config["third_rgb"]
        return super().other_color(index)

    def pregrasp_targets(self, assembly, entities, transformed):
        """All three pieces, each as a whole (the family's targets are the two opening pieces near the opening grasps)."""
        import xml.etree.ElementTree as ET
        paths = {e.name: e.asset_path for e in entities}
        return tuple(dict(entity=b.name, link=ET.parse(paths[b.name]).getroot().find("link").get("name"),
                          center_m=((b.mesh.vertices.min(0) + b.mesh.vertices.max(0)) / 2).tolist(), radius_m=.09)
                     for b in assembly.bodies)

    def evaluate_goal_region(self, state, region):
        from .trinity_terminal import pairwise_separation_candidate
        return pairwise_separation_candidate(state, self.placement_vertices,
                                             minimum_separation_m=self.config["minimum_separation_m"])

    def terminal_metadata(self):
        return super().terminal_metadata() | dict(
            goal_kind="pairwise_whole_piece_separation",
            separation_certificate="for every pair of pieces, a plane separating every visible and collision vertex")
