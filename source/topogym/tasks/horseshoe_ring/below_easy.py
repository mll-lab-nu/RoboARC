"""Below-easy Horseshoe: the easy tier's shoes and chains with a larger ring, still around both chains.

The ring bore sets the difficulty: a round 144 mm bore (very_easy), a 146 x 136 mm ellipse (easier);
the easy tier's enlarged ring is 142.6 x 135.9 mm. Knob: ``ring``. Each setting starts from its own
gravity-settled layout. These are design targets, not robot qualifications.
"""
from copy import deepcopy

from topogym.tasks.base import merge_parameters
from .horseshoe_ring_easy import HorseshoeRingEasyTask


def ring(long_mm, short_mm):
    """Ring bore (inner long and short axes, mm); the wire stays the easy ring's."""
    return {"bore_mm": [float(long_mm), float(short_mm)]}


class HorseshoeBelowEasyTask(HorseshoeRingEasyTask):
    """Shared construction; very_easy / easier set the ring bore and their settled layout."""

    model_revision = "below_easy_v1"
    family_id = "below_easy_ring_bore"
    defaults = merge_parameters(HorseshoeRingEasyTask.defaults, {"ring": ring(144., 144.)})
    fixed_parameters = ("scale", "ring")
    layout_resource = None

    def template(self):
        template = deepcopy(super().template())
        wire, scale = template["target_ring"]["wire_radius"], float(self.config["scale"])
        template["template_id"] += "+below_easy"
        template["target_ring"]["centerline_radii"] = [b / (2 * scale) + wire for b in self.config["ring"]["bore_mm"]]
        return template
