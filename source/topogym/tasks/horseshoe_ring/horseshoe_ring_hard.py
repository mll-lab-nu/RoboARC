"""waist_ellipse."""
from .common import HorseshoeRingTask


class HorseshoeRingHardTask(HorseshoeRingTask):
    variation_id = 'horseshoe_ring_hard'
    family_id = 'waist_ellipse'
    model_revision = 'ring_release_clearance_candidate_v8'
    difficulty = 'hard'
    template_resource = 'variations/horseshoe_ring/waist_ellipse.json'
    # Naturally settled ring-clearance candidate.
    layout_resource = 'variations/horseshoe_ring/waist_ellipse_settled_poses.json'
