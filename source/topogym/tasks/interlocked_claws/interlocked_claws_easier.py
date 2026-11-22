"""XYZ cross head and a square frame with a lateral handle extension."""
from .simplified import SimplifiedClawsTask


class InterlockedClawsEasierTask(SimplifiedClawsTask):
    variation_id = "interlocked_claws_easier"
    family_id = "cross_square_offset"
    model_revision = "xyz_cross_offset_frame_candidate_v1"
    difficulty = "easier"
    # The pieces by shape (their colours change with the appearance seed): the cross is piece_0, the frame piece_1.
    placement_target_description = "the entire cross piece"
    placement_remainder_description = "the frame piece"
    placement_scene_note = ("The cross piece is the straight bar ending in a plus-shaped cross; the frame piece is the "
                            "square frame with a bent side handle.")
