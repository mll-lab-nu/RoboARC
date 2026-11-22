"""cast_keyhole: design 1. Difficulty is a design target, not a qualification."""

from .common import CastKeyholeTask


class CastKeyholeEasyTask(CastKeyholeTask):
    variation_id = 'cast_keyhole_easy'
    family_id = 'keyhole_design_1'
    model_revision = 'seed_families_v2'
    difficulty = 'easy'
    fixed_parameters = ('scale',)

    geometry_resource = 'variations/cast_keyhole/F1/geometry.json'
