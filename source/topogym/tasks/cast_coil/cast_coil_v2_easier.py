"""cast_coil v2 easier: the lattice design s43 of the v2 design search, the tier after v2 very_easy (tentative, 2026-10-10).

Two pieces interleaved layer by layer in a 4 x 8 x 5-cell block of 35 mm bars, joined by round pins through the other
piece's layers. The packaged record (assets/resources/variations/cast_coil/lattice_s43.json) holds the pieces
only. Everything else, the
start on the table, the grasp regions, the goal and the MuJoCo profile, is v2 very_easy's.
"""
from .cast_coil_v2_very_easy import CastCoilV2VeryEasyTask


class CastCoilV2EasierTask(CastCoilV2VeryEasyTask):
    variation_id = "cast_coil_v2_easier"
    family_id = "lattice_s43"
    model_revision = "lattice_s43_v1"
    difficulty = "easier"
    geometry_resource = "variations/cast_coil/lattice_s43.json"
