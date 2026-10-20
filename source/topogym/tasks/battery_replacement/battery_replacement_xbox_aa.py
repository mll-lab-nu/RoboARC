"""battery_replacement on an Xbox-style controller with two AA cells (design v2 + demo v3b's pry stick with a foot)."""
from .common import BatteryReplacementTask


class BatteryReplacementXboxAaTask(BatteryReplacementTask):
    variation_id = "battery_replacement_xbox_aa"
    family_id = "xbox_aa"
    model_revision = "xbox_aa_v3b"
    difficulty = None
