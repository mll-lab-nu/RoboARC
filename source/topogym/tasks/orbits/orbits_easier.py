"""One fixed ring and two posts: the smallest complete recursive Orbits fixture."""
from topogym.tasks.base import merge_parameters
from .below_easy import OrbitsBelowEasyTask


class OrbitsEasierTask(OrbitsBelowEasyTask):
    variation_id = 'orbits_easier'
    family_id = 'single_ring_two_posts'
    difficulty = 'easier'
    variant_parameters = merge_parameters(OrbitsBelowEasyTask.variant_parameters, {
        'cord_profile': 'easier_ring_hole_v2',
        'geometry': {'post_spacing_m': .165/.85, 'ring_radius_m': .11/.85+.008, 'base_length_m': .36/.85,
                     'base_width_m': .26/.85, 'bight_radius_m': .07/.85},
    })
    defaults = merge_parameters(OrbitsBelowEasyTask.defaults, variant_parameters)
