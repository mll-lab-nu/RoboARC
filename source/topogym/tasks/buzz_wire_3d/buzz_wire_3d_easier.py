"""Approved design-2 layout; qualification recorded separately."""

from .simplified import BuzzWireSimplifiedTask


class BuzzWire3DEasierTask(BuzzWireSimplifiedTask):
    variation_id = "buzz_wire_3d_easier"
    family_id = "buzz_design_2"
    model_revision = "design_2_deck_rest_v2"
    difficulty = "easier"
    layout_resource = "variations/buzz_wire_3d/design_2_deck_rest_v2/layout.json"
