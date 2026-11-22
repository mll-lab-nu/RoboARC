"""bed_bolt with the cross board (the design's v2): the bay in front of the corner block is 16 cm deep, so the key's long
arm cannot lie along the bolt; the short arm goes into the socket and the long arm stands up as a lever."""
from .common import BedBoltTask


class BedBoltCrossbarTask(BedBoltTask):
    variation_id = "bed_bolt_crossbar"
    family_id = "bed_corner_bolt"
    model_revision = "bed_bolt_v2_crossbar"
    difficulty = None
