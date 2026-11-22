"""cast_coil v2 easy: the lattice design s121 of the v2 design search, the tier after v2 easier (2026-10-12).

s43 (v2 easier) grown in an 8 x 8 x 6-cell search block: two pieces interleaved layer by layer in 8 x 8 x 5 cells of 35 mm
bars, joined by round pins through the other piece's layers. The packaged record
(assets/resources/variations/cast_coil/lattice_s121.json) holds the pieces only. Everything else, the start on the
table, the grasp regions, the goal and the MuJoCo profile, is v2 very_easy's.
"""
from .cast_coil_v2_very_easy import CastCoilV2VeryEasyTask


class CastCoilV2EasyTask(CastCoilV2VeryEasyTask):
    variation_id = "cast_coil_v2_easy"
    family_id = "lattice_s121"
    model_revision = "lattice_s121_v1"
    difficulty = "easy"
    geometry_resource = "variations/cast_coil/lattice_s121.json"
