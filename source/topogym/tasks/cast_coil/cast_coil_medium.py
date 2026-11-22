"""cast_coil: design 4. Difficulty is a design target, not a qualification."""

from .common import CastCoilTask


class CastCoilMediumTask(CastCoilTask):
    variation_id = 'cast_coil_medium'
    family_id = 'coil_design_4'
    model_revision = 'seed_families_v1_9mm'
    difficulty = 'medium'
    fixed_parameters = ('scale', 'objective')

    template_resource = 'variations/cast_coil/design_4.json'
