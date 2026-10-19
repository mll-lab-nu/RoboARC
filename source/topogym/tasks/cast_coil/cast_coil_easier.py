"""cast_coil: design 2 (cubic baffle pieces)."""
from .simplified import SimplifiedCoilTask


class CastCoilEasierTask(SimplifiedCoilTask):
    variation_id = "cast_coil_easier"
    family_id = "coil_design_2"
    model_revision = "cubic_baffle_v2_aloha_v1"
    difficulty = None
    geometry_resource = "variations/cast_coil/easier/geometry.json"
