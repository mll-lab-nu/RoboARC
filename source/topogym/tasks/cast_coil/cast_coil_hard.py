"""cast_coil: design 5. Difficulty is a design target, not a qualification."""

from .common import CastCoilTask


class CastCoilHardTask(CastCoilTask):
    variation_id = 'cast_coil_hard'
    family_id = 'coil_design_5'
    model_revision = 'seed_families_v1_9mm'
    difficulty = 'hard'
    fixed_parameters = ('scale', 'objective')

    template_resource = 'variations/cast_coil/design_5.json'
